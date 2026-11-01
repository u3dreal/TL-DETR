"""Geodesic distance maps for tooth-patch landmarks  (macOS / MPS port).

Replaces the original sklearn.manifold.Isomap implementation, which was the single
biggest bottleneck in preprocessing.

Why the original was unusable
-----------------------------
    isomap = Isomap(n_components=2, n_neighbors=5)
    isomap.fit_transform(points)          # <- spectral embedding, O(n^3)
    return isomap.dist_matrix_            # <- we only ever used this

* `dist_matrix_` is a dense n x n float64 array: 8000 faces -> 512 MB *per tooth*.
* `fit_transform` additionally runs an eigendecomposition to build a 2-D embedding
  that the caller then throws away.
* 3000 teeth x (hundreds of MB + O(n^3)) = days of wall clock.

Three changes
-------------
1. **kNN graph -> csgraph.dijkstra.** Same intrinsic-distance model as Isomap
   (k nearest triangle centres, Euclidean edge weights), but shortest paths instead
   of a spectral embedding. Symmetrised with `csr.maximum`, matching sklearn's
   undirected handling; note that `csr + csr.T` (as in the brief's sketch) *sums*
   duplicate edges and inflates distances, so it is avoided here.
2. **Only the rows we need.** The caller wants `distance_matrix[closest_idx]`, i.e.
   n_landmarks rows out of n, and n_landmarks is 5-8. Pass `indices=` to dijkstra and
   the dense n x n matrix is never materialised: 512 MB -> ~64 kB for n=8000.
   This is why no decimation is needed (and decimation would be *wrong* here anyway:
   the dataset indexes face geodesics by face id, so the .off mesh and the _f.npy
   must describe the same face list).
3. **joblib over patches.** Preprocessing is embarrassingly parallel.

Usage (same layout TL-DETR expects):
    python scripts/geodesic_distance.py \\
        --split-root data/3DTeethLand/Split_exp \\
        --patch-root data/3DTeethLand/patches \\
        --jobs 6
"""

import json
import os
import sys
import traceback

import numpy as np
import trimesh
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

# a landmark further than this from any triangle centre is almost certainly a
# mis-annotation (or a units mismatch); reported, not silently accepted
SURFACE_WARN_MM = 1.5


# --------------------------------------------------------------------------------------
# graph construction
# --------------------------------------------------------------------------------------
def knn_graph(points, k=5):
    """Symmetric kNN graph over `points` with Euclidean edge weights.

    Mirrors sklearn's Isomap graph: `kneighbors_graph(mode='distance')` followed by
    symmetrisation. Duplicate (i,j)/(j,i) entries carry the same weight, so a
    maximum-combine is the exact symmetrisation; summing them would double the length
    of shared edges (the brief's snippet does exactly that).
    """
    n = len(points)
    if n < 2:
        return csr_matrix((n, n), dtype=np.float64), 0

    k = min(k, n - 1)
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=k + 1)          # column 0 is the point itself
    if np.ndim(dist) == 1:                           # n == 1 edge case
        dist, idx = dist[:, None], idx[:, None]

    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    vals = dist[:, 1:].reshape(-1)

    # drop zero-length edges (duplicate triangle centres) so sparsity stays valid
    keep = vals > 0
    g = csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, n))
    g = g.maximum(g.T)
    return g, k


def mesh_graph(mesh):
    """Edge-adjacency graph on triangle centres (true intrinsic surface distances).

    More accurate than kNN over centres and always connected, but not what the
    reference implementation trained on - opt in with --graph mesh.
    """
    cs = np.asarray(mesh.triangles_center, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    n = len(cs)
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)                 # unique face-adjacency pairs
    edges = edges[edges[:, 0] != edges[:, 1]]        # drop degenerate/duplicate index
    w = np.linalg.norm(cs[edges[:, 0]] - cs[edges[:, 1]], axis=1)
    rows, cols = edges[:, 0], edges[:, 1]
    g = csr_matrix((np.r_[w, w], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n))
    return g, cs


def _graph_for(points, k, mode, mesh=None):
    if mode == 'mesh' and mesh is not None:
        return mesh_graph(mesh)
    g, _ = knn_graph(points, k)
    return g, None


def _ensure_connected(graph, points, k, mode, mesh, max_k=25):
    """Escalate k until the kNN graph is a single component.

    A disconnected graph yields inf geodesics, hence all-zero heatmaps, hence a
    sample that supervises nothing. Fail loudly instead of training on holes.
    """
    if mode == 'mesh':
        ncomp, _ = connected_components(graph, directed=False)
        return graph, ncomp == 1
    ncomp, _ = connected_components(graph, directed=False)
    while ncomp > 1 and k < max_k:
        k = min(k + 5, max_k)
        g, _ = knn_graph(points, k)
        ncomp_new, _ = connected_components(g, directed=False)
        if ncomp_new >= ncomp:            # no progress; stop escalating
            ncomp = ncomp_new
            break
        graph, ncomp = g, ncomp_new
    return graph, ncomp == 1


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def geodesic_distance_rows(points, sources, k=5, mode='knn', mesh=None):
    """Distances from each source point to every point. Shape (len(sources), n).

    Drop-in for `geodesic_distance_matrix(points)[sources]` without the n x n array.
    """
    points = np.asarray(points, dtype=np.float64)
    graph, _ = _graph_for(points, k, mode, mesh)
    graph, ok = _ensure_connected(graph, points, k, mode, mesh)
    rows = dijkstra(graph, directed=False, indices=np.asarray(sources, dtype=np.int64))
    rows = np.atleast_2d(rows)
    if not ok:
        print(f"[warn] geodesic graph still disconnected after k escalation; "
              f"{(~np.isfinite(rows)).any(axis=0).sum()} points unreachable "
              f"(their heatmap will be 0)", file=sys.stderr)
    return rows


def geodesic_distance_matrix(points, k=5):
    """Full dense matrix. Kept for API compatibility - prefer geodesic_distance_rows.

    n=8000 costs 512 MB; only use this for inspection on small meshes.
    """
    points = np.asarray(points, dtype=np.float64)
    graph, _ = knn_graph(points, k)
    graph, _ = _ensure_connected(graph, points, k, 'knn', None)
    from scipy.sparse.csgraph import shortest_path
    return shortest_path(graph, method='auto', directed=False)


# --------------------------------------------------------------------------------------
# per-patch driver
# --------------------------------------------------------------------------------------
def nearest_face_index(landmark_coords, centers):
    """Index of the triangle centre closest to each landmark (same rule as original)."""
    d = np.linalg.norm(landmark_coords[:, None, :] - centers[None, :, :], axis=2)
    idx = np.argmin(d, axis=1)
    return idx, d[np.arange(len(idx)), idx]


def write_patch_geodesics(off_file, json_file, out_file, k=5, mode='knn', overwrite=True,
                          warn_mm=SURFACE_WARN_MM):
    """Compute and save <t_idx>_f.npy for one tooth patch.

    Returns a small dict of QC info so callers can aggregate it.
    """
    if os.path.exists(out_file) and not overwrite:
        return {'status': 'skipped'}

    try:
        mesh = trimesh.load(off_file, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)

        with open(json_file) as f:
            landmarks = json.load(f)
        if not landmarks:
            return {'status': 'no_landmarks', 'file': off_file}

        coords = np.array([lm['coord'] for lm in landmarks], dtype=np.float64)
        if coords.size == 0:
            return {'status': 'no_landmarks', 'file': off_file}

        src, src_dist = nearest_face_index(coords, centers)
        rows = geodesic_distance_rows(centers, src, k=k, mode=mode, mesh=mesh)

        # numpy saves .npy atomically enough here, but write to tmp to avoid
        # leaving a truncated file if the process dies mid-write
        np.save(out_file + '.tmp.npy', rows.astype(np.float64))
        os.replace(out_file + '.tmp.npy', out_file)

        return {
            'status': 'ok',
            'file': off_file,
            'n_faces': len(centers),
            'n_landmarks': len(landmarks),
            'max_src_dist_mm': float(src_dist.max()),
            'far_landmarks': int((src_dist > warn_mm).sum()),
            'unreachable': int((~np.isfinite(rows)).sum()),
        }
    except Exception as exc:                     # noqa: BLE001 - report, never abort the batch
        return {'status': 'error', 'file': off_file,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()}


def iter_patches(patch_root, split_root=None, split_file=None):
    """Yield (off, json) pairs; tolerant of either layout."""
    if split_root and split_file:
        with open(os.path.join(split_root, split_file)) as f:
            scans = [ln.strip() for ln in f if ln.strip()]
    elif split_root:
        scans = sorted(os.listdir(split_root))
    else:
        scans = sorted(os.listdir(patch_root))

    for scan in scans:
        scan_dir = os.path.join(patch_root, scan)
        if not os.path.isdir(scan_dir):
            continue
        for name in sorted(os.listdir(scan_dir)):
            if not name.endswith('.off'):
                continue
            stem = os.path.splitext(name)[0]
            off_file = os.path.join(scan_dir, stem + '.off')
            js_file = os.path.join(scan_dir, stem + '.json')
            if os.path.exists(js_file):
                yield off_file, js_file


def run(patch_root, split_root=None, split_file=None, jobs=4, k=5, mode='knn',
        overwrite=True, limit=None):
    tasks = list(iter_patches(patch_root, split_root, split_file))
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        print(f"[error] no patches found under {patch_root}", file=sys.stderr)
        return []

    print(f"[geo] {len(tasks)} patches, k={k}, graph={mode}, jobs={jobs}")

    def _one(off, js):
        out = os.path.splitext(off)[0] + '_f.npy'
        return write_patch_geodesics(off, js, out, k=k, mode=mode, overwrite=overwrite)

    results = Parallel(n_jobs=jobs, backend='loky', verbose=1)(
        delayed(_one)(off, js) for off, js in tasks
    )

    ok = [r for r in results if r['status'] == 'ok']
    bad = [r for r in results if r['status'] == 'error']
    far = [r for r in ok if r['far_landmarks']]
    unreachable = [r for r in ok if r['unreachable']]

    print(f"[geo] done: ok={len(ok)} errors={len(bad)} "
          f"skipped={sum(r['status']=='skipped' for r in results)}")
    if far:
        print(f"[geo] {len(far)} patch(es) have a landmark >{SURFACE_WARN_MM} mm from the "
              f"surface - check annotation/units, e.g. {far[0]['file']}")
    if unreachable:
        print(f"[geo] {len(unreachable)} patch(es) have unreachable faces")
    for r in bad[:10]:
        print(f"[geo] ERROR {r['file']}: {r['error']}", file=sys.stderr)
    return results


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--patch-root', required=True, help='root containing <scan>/<t_idx>.off')
    p.add_argument('--split-root', default=None, help='root containing train/val/test .txt')
    p.add_argument('--split-file', default=None, help='e.g. train.txt')
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument('--k', type=int, default=5, help='kNN neighbours (original Isomap used 5)')
    p.add_argument('--graph', choices=['knn', 'mesh'], default='knn',
                   help='knn matches the reference implementation; mesh uses true '
                        'surface edges (more accurate, not what the weights expect)')
    p.add_argument('--no-overwrite', action='store_true')
    p.add_argument('--limit', type=int, default=None)
    a = p.parse_args(argv)
    res = run(a.patch_root, a.split_root, a.split_file, jobs=a.jobs, k=a.k,
              mode=a.graph, overwrite=not a.no_overwrite, limit=a.limit)
    return 0 if not [r for r in res if r['status'] == 'error'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
