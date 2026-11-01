"""Crop per-tooth patches from full-jaw scans and attach landmark annotations.

Rewritten from the original two loose functions into a runnable CLI with QC, because
the landmark->tooth assignment is the weakest link in the data pipeline and it fails
*silently* upstream of everything that matters:

  * 3DTeethLand annotations carry NO tooth id. A landmark is attached to a tooth by
    nearest-vertex distance. On crowded anterior teeth - exactly the contacts the
    Bolton ratio depends on - a `Mesial` point sits on the boundary between two
    patches and can be assigned to the neighbour. That silently corrupts MD widths.
    We keep the nearest-vertex rule (so patches stay comparable with the reference
    pipeline) but record the runner-up margin and flag ambiguous assignments.

  * The release ships landmarks nested (upper/<SCAN>/<SCAN>_upper__kpt.json); the
    original script only looked for them flat. Both layouts are accepted here.

Layout in:   <mesh_root>/<arch>/<SCAN>/<SCAN>_<arch>.obj + .json   (segmentation)
             <landmark_root>/.../<SCAN>_<arch>__kpt.json            (landmarks)
Layout out:  <patch_root>/<SCAN>_<arch>/<t_idx>.off / .json
             with t_idx 1..16 == FDI 11..18,21..28 (upper) / 31..38,41..48 (lower)

Usage:
    python scripts/seg_to_single.py --mesh-root data/3DTeethLand/teeth3ds_root \
        --landmark-root data/3DTeethLand/landmarks \
        --patch-root data/3DTeethLand/patches \
        --split-root data/3DTeethLand/Split_exp --split-file training_upper.txt \
        --anterior-only --jobs 6
"""

import glob
import json
import os
import sys
import traceback

import numpy as np
import trimesh
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

TAR_LABELS_UPPER = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28]
TAR_LABELS_LOWER = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]

# FDI of the 12 anterior teeth, and the t_idx each maps to
ANTERIOR_UPPER = [11, 12, 13, 21, 22, 23]
ANTERIOR_LOWER = [31, 32, 33, 41, 42, 43]

FIXED_CLASSES = ('Mesial', 'Distal', 'InnerPoint', 'OuterPoint', 'FacialPoint')
CUSP = 'Cusp'


def trans_labels(labels, category):
    """FDI labels -> per-arch tooth index 1..16 (in place)."""
    table = TAR_LABELS_UPPER if category == 'upper' else TAR_LABELS_LOWER
    for i, l in enumerate(table):
        labels[labels == l] = i + 1
    return labels


def t_idx_of(fdi, category):
    table = TAR_LABELS_UPPER if category == 'upper' else TAR_LABELS_LOWER
    return table.index(fdi) + 1


def find_arch_files(mesh_root, scan, arch):
    """Locate one arch's mesh + segmentation labels, whatever the directory naming.

    The release nests them as `<arch>/<SCAN>/<SCAN>_<arch>.obj`, where `scan` carries the
    arch suffix but the *directory* does not (`lower/O52P1SZT/O52P1SZT_lower.obj`), so the
    directory must not be rebuilt from `scan`.
    """
    for obj in (os.path.join(mesh_root, arch, scan, f'{scan}.obj'),) + tuple(
            glob.glob(os.path.join(mesh_root, arch, '*', f'{scan}.obj'))):
        if os.path.exists(obj):
            seg = obj[:-4] + '.json'
            # the release ships 1900 meshes but only 1800 label files; a mesh without
            # labels is unusable here, and must be reported as such, not crash later
            return obj, (seg if os.path.exists(seg) else None)
    return None, None


def find_landmark_file(landmark_root, scan, arch):
    """Accept both the flat and the nested release layouts.

    `scan` already carries the arch (`013TXGFK_upper`), which is also how TL-DETR names
    the annotation file: `<scan>__kpt.json`. The *directory* does not carry the arch
    (`landmarks/upper/013TXGFK/013TXGFK_upper__kpt.json`), so it must not be rebuilt
    from `scan`.
    """
    bare = scan.rsplit('_', 1)[0]
    for c in (os.path.join(landmark_root, f'{scan}__kpt.json'),
              os.path.join(landmark_root, arch, bare, f'{scan}__kpt.json'),
              os.path.join(landmark_root, arch, scan, f'{scan}__kpt.json'),
              os.path.join(landmark_root, scan, f'{scan}__kpt.json'),
              os.path.join(landmark_root, f'{scan}_{arch}__kpt.json')):
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(landmark_root, '**', f'{scan}__kpt.json'), recursive=True)
    return sorted(hits)[0] if hits else None


def crop_teeth(mesh, labels, category, patch_dir, anterior_only=False):
    """Write <t_idx>.off per tooth. Returns {t_idx: mesh} for the QC/assign step."""
    vs, fs = mesh.vertices, mesh.faces
    fv_labels = labels[fs]
    keep_fdi = (ANTERIOR_UPPER if category == 'upper' else ANTERIOR_LOWER) if anterior_only \
        else (TAR_LABELS_UPPER if category == 'upper' else TAR_LABELS_LOWER)

    patches = {}
    for fdi in keep_fdi:
        l = t_idx_of(fdi, category)          # labels were remapped to 1..16 already
        v_idx = np.argwhere(labels == l).ravel()
        if v_idx.size == 0:
            continue
        f_idx = np.where(np.all(fv_labels == l, axis=1))[0]
        if f_idx.size == 0:
            continue

        vl = vs[v_idx]
        fl = fs[f_idx]
        mapping = {int(old): new for new, old in enumerate(v_idx)}
        try:
            new_faces = np.array([[mapping[int(v)] for v in f] for f in fl], dtype=np.int64)
        except KeyError:
            # a face with a vertex outside v_idx - cannot happen with np.all(), skip defensively
            continue

        patch = trimesh.Trimesh(vertices=vl, faces=new_faces, process=False)
        t_idx = t_idx_of(fdi, category)
        patch.export(os.path.join(patch_dir, f'{t_idx}.off'))
        patches[t_idx] = (patch, fdi)
    return patches


def assign_landmarks(patches, landmarks, category, tolerance=0.5, margin=0.05):
    """Attach each annotated landmark to the nearest tooth patch.

    Returns {t_idx: [landmark dicts]} plus QC stats.
    """
    if not patches:
        return {}, {'unassigned': len(landmarks), 'ambiguous': 0, 'far': 0}
    t_idxs = sorted(patches)
    trees = {t: cKDTree(np.asarray(patches[t][0].vertices, dtype=np.float64)) for t in t_idxs}

    coords = np.array([lm['coord'] for lm in landmarks], dtype=np.float64)
    dists = np.stack([trees[t].query(coords, k=1)[0] for t in t_idxs], axis=1)  # (L, T)

    out = {t: [] for t in t_idxs}
    stats = {'unassigned': 0, 'ambiguous': 0, 'far': 0, 'far_detail': []}
    for i, lm in enumerate(landmarks):
        order = np.argsort(dists[i])
        best, second = order[0], (order[1] if len(order) > 1 else None)
        d_best = dists[i, best]
        d_second = dists[i, second] if second is not None else np.inf
        if d_best > tolerance:
            stats['unassigned'] += 1
            stats['far'] += 1
            stats['far_detail'].append({'class': lm['class'], 'dist_mm': round(float(d_best), 3)})
            continue
        if d_second - d_best < margin:
            stats['ambiguous'] += 1
        out[t_idxs[best]].append({'class': lm['class'],
                                  'coord': [float(c) for c in lm['coord']],
                                  'dist': float(d_best), 'margin': float(d_second - d_best)})
    return out, stats


def dedupe_by_class(cands):
    """Keep one landmark per *fixed* class, and **every** cusp.

    Fixed classes (Mesial, Distal, Inner, Outer, Facial) are one-per-tooth by definition, and
    landmarks carry no tooth id: on a crowded proximal contact two points of the SAME class can
    both end up nearest to one patch — the tooth's own and its neighbour's (`FINDINGS.md` §11).
    The dataset loader keeps whichever comes last in the file, i.e. the annotation JSON's order,
    which means nothing; keep the better-assigned one instead, the larger lead over the
    runner-up tooth.

    `Cusp` is different and must not be collapsed. The paper's cusp channel is a *variable*
    number of points on the occlusal table — one buccal cusp on a first premolar, MB + DB on a
    molar, and up to four on a second molar. Keeping one per tooth taught the cusp head to emit
    exactly one cusp for every tooth, which cannot build a posterior FACC at all
    (`FINDINGS.md` §24g); the loss, the matcher and `g_cheats` were already written for a
    variable set (`utils/loss.py` matches per sample), so the only thing that was wrong was
    this function.

    Cusps are ordered by assignment confidence and then by coordinate: the JSON has to be
    reproducible, but the annotation carries no cusp identity to preserve, and Hungarian
    matching is order-invariant anyway.

    Returns (kept, dropped): `kept` in a fixed class order (so the JSON is stable) with the
    cusps last, `dropped` for the QC record — never silently discarded.
    """
    best, dropped, cusps = {}, [], []
    for c in cands:
        if c['class'] == CUSP:
            cusps.append(c)
            continue
        cur = best.get(c['class'])
        if cur is None:
            best[c['class']] = c
            continue
        # ties on margin -> the closer point wins
        keep_new = (c['margin'], -c['dist']) > (cur['margin'], -cur['dist'])
        loser = cur if keep_new else c
        best[c['class']] = c if keep_new else cur
        dropped.append({'class': loser['class'], 'coord': loser['coord'],
                        'dist_mm': round(loser['dist'], 3),
                        'margin_mm': round(loser['margin'], 3)})

    order = {c: i for i, c in enumerate(FIXED_CLASSES + (CUSP,))}
    kept = [{'class': c, 'coord': best[c]['coord']}
            for c in sorted(best, key=lambda k: order.get(k, len(order)))]
    cusps.sort(key=lambda c: (round(c['dist'], 6), round(c['margin'], 6),
                              tuple(round(v, 6) for v in c['coord'])))
    kept += [{'class': CUSP, 'coord': c['coord']} for c in cusps]
    return kept, dropped


def process_scan(scan, mesh_root, landmark_root, patch_root, anterior_only=False,
                 tolerance=0.5, margin=0.05):
    """Crop + assign + write JSONs for one arch. Returns a QC record."""
    arch = scan.split('_')[1]
    mesh_file, seg_file = find_arch_files(mesh_root, scan, arch)
    if mesh_file is None:
        return {'scan': scan, 'status': 'missing_mesh'}
    if seg_file is None:
        return {'scan': scan, 'status': 'missing_segmentation'}

    lm_file = find_landmark_file(landmark_root, scan, arch)
    if lm_file is None:
        return {'scan': scan, 'status': 'missing_landmarks'}

    try:
        mesh = trimesh.load(mesh_file, process=False)
        with open(seg_file) as f:
            gt = json.load(f)
        labels = np.asarray(gt['labels'], dtype=np.int64)
        labels = trans_labels(labels, arch)

        patch_dir = os.path.join(patch_root, scan)
        os.makedirs(patch_dir, exist_ok=True)
        patches = crop_teeth(mesh, labels, arch, patch_dir, anterior_only)
        if not patches:
            return {'scan': scan, 'status': 'no_patches'}

        with open(lm_file) as f:
            anno = json.load(f)
        landmarks = anno['objects'] if isinstance(anno, dict) else anno
        assigned, stats = assign_landmarks(patches, landmarks, arch, tolerance, margin)

        # --- Bolton-specific QC: all 12 anteriors need Mesial AND Distal -----------
        want = ANTERIOR_UPPER if arch == 'upper' else ANTERIOR_LOWER
        missing = {}
        duplicated = {}
        for fdi in want:
            t_idx = t_idx_of(fdi, arch)
            classes = [lm['class'] for lm in assigned.get(t_idx, [])]
            miss = [c for c in ('Mesial', 'Distal') if c not in classes]
            if miss:
                missing[fdi] = miss
            # A landmark with no tooth id can land on the WRONG tooth: on a crowded
            # contact the neighbour's Mesial point is sometimes nearest to this patch too.
            # Counted here, and resolved deterministically just below
            # (FINDINGS.md section 11). Cusp is excluded: a molar legitimately has several,
            # and they are all kept (see `dedupe_by_class`).
            dups = sorted({c for c in classes if c != CUSP and classes.count(c) > 1})
            if dups:
                duplicated[fdi] = dups

        # one landmark per class per tooth; the losers are recorded, not dropped silently
        deduped, discards = {}, []
        for t_idx, cands in assigned.items():
            kept, dropped = dedupe_by_class(cands)
            deduped[t_idx] = kept
            if dropped:
                discards.append({'t_idx': t_idx, 'dropped': dropped})
        assigned = deduped

        # write per-tooth landmark json
        n_written = 0
        for t_idx, lms in assigned.items():
            if not lms:
                continue
            with open(os.path.join(patch_dir, f'{t_idx}.json'), 'w') as f:
                json.dump(lms, f, indent=1)
            n_written += 1

        return {
            'scan': scan, 'status': 'ok',
            'n_patches': len(patches), 'n_json': n_written,
            'n_landmarks': len(landmarks),
            'unassigned': stats['unassigned'], 'ambiguous': stats['ambiguous'],
            'duplicated': duplicated, 'deduped': discards,
            'anterior_complete': not missing, 'missing': missing,
        }
    except Exception as exc:                                  # noqa: BLE001
        return {'scan': scan, 'status': 'error',
                'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()}


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--mesh-root', required=True)
    p.add_argument('--landmark-root', required=True)
    p.add_argument('--patch-root', required=True)
    p.add_argument('--split-root', default=None)
    p.add_argument('--split-file', action='append', default=None,
                   help='one or more split files of scan names (may be repeated)')
    p.add_argument('--scans', nargs='*', default=None, help='explicit scan list instead')
    p.add_argument('--anterior-only', action='store_true',
                   help='only write the 12 anterior teeth (much faster, all Bolton needs)')
    p.add_argument('--tolerance', type=float, default=0.5, help='max landmark->patch distance (mm)')
    p.add_argument('--margin', type=float, default=0.05,
                   help='min lead over the runner-up tooth before a landmark counts as ambiguous')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--qc-out', default=None, help='write per-scan QC json here')
    a = p.parse_args(argv)

    scans = a.scans
    if scans is None:
        if not (a.split_root and a.split_file):
            print('[error] give --scans, or --split-root with --split-file', file=sys.stderr)
            return 2
        scans = []
        for sf in a.split_file:
            with open(os.path.join(a.split_root, sf)) as f:
                scans += [ln.strip() for ln in f if ln.strip()]
    os.makedirs(a.patch_root, exist_ok=True)

    print(f'[seg] {len(scans)} arches, anterior_only={a.anterior_only}, jobs={a.jobs}')
    recs = Parallel(n_jobs=a.jobs, backend='loky', verbose=1)(
        delayed(process_scan)(s, a.mesh_root, a.landmark_root, a.patch_root,
                              a.anterior_only, a.tolerance, a.margin)
        for s in scans)

    ok = [r for r in recs if r['status'] == 'ok']
    print(f'[seg] ok={len(ok)}/{len(recs)}')
    complete = [r for r in ok if r['anterior_complete']]
    print(f'[seg] anterior-complete arches: {len(complete)}/{len(ok)}')
    amb = sum(r['ambiguous'] for r in ok)
    unass = sum(r['unassigned'] for r in ok)
    print(f'[seg] landmarks: ambiguous={amb} unassigned={unass}')
    for r in recs:
        if r['status'] not in ('ok',):
            print(f"[seg] {r['status']}: {r['scan']} {r.get('error','')}", file=sys.stderr)
    for r in ok[:5]:
        if not r['anterior_complete']:
            print(f"[seg] incomplete anterior in {r['scan']}: {r['missing']}")

    if a.qc_out:
        with open(a.qc_out, 'w') as f:
            json.dump(recs, f, indent=1)
        print(f'[seg] QC written to {a.qc_out}')

    # per-scan completeness list, handy for filtering the training set
    with open(os.path.join(a.patch_root, '_anterior_complete.txt'), 'w') as f:
        f.write('\n'.join(r['scan'] for r in complete) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
