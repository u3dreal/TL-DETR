import numpy as np
import torch


def calc_features(vs, fs, fn):
    nf = fs.shape[0]
    fea = torch.empty((15, nf), dtype=torch.float32).to(vs.device)

    vs_in_ts = vs[fs]

    fea[:3, :] = vs_in_ts.mean(1).T[None]  # centers , 3
    fea[3:6, :] = fn.T[None]  # normal, 3
    fea[6:15, :] = (vs_in_ts - vs_in_ts.mean(1, keepdim=True)).reshape((nf, -1)).T[None]
    return fea


def patch_features(mesh):
    """Deterministic (15, F) feature matrix for a WHOLE patch, float32.

    Same definition as `calc_features`, but numpy and over every face, because this is
    what `tools/prepare_dataset.py --steps features` caches to `<t_idx>_feat.npy`.

    Caching the *full* patch (not a draw of `num_points` faces) is deliberate: the random
    face subsample in `sample_indices` is part of the augmentation, so freezing it would
    mean training forever on one draw of 2 048 of ~8 000 faces per tooth - which also
    fixes the decoded landmark resolution for the whole run (Agents.md section 7.2).
    """
    vs = np.asarray(mesh.vertices, dtype=np.float64)
    vs = vs - vs.mean(0)                      # identical to TeethLandDataset.__getitem__
    fs = np.asarray(mesh.faces, dtype=np.int64)
    fn = np.asarray(mesh.face_normals, dtype=np.float64)

    v = vs[fs]                                # (F, 3, 3)
    c = v.mean(1)                             # (F, 3) face centroids
    fea = np.empty((15, len(fs)), dtype=np.float32)
    fea[:3, :] = c.T
    fea[3:6, :] = fn.T
    fea[6:15, :] = (v - c[:, None]).reshape(len(fs), -1).T
    return fea


def geodesic_heatmaps(geodesic_matrix, args):
    ys = []
    for i, dist in enumerate(geodesic_matrix):
        y = np.exp(- dist ** 2 / (2 * args.landmark_std ** 2))
        ys.append(y)
    return np.asarray(ys).T


def sample_indices(n_faces, num_points):
    """Face indices for one sample: a random `num_points` draw, with replacement if short.

    Kept in one place because two code paths use it - the mesh loader and the cached-feature
    loader - and they must draw the same way. Short patches are padded by DUPLICATING real
    faces (they carry their real heatmaps); the original code did the same, which is a loss
    weighting issue, not corrupted supervision (FINDINGS.md section 9).
    """
    if n_faces < num_points:
        n = n_faces
        return np.concatenate([np.arange(n),
                               np.random.choice(n_faces, num_points - n, replace=True)])
    return np.random.permutation(n_faces)[:num_points]


def sample(fs, fn, hs, num_points):  # 15xN MxN
    idx = sample_indices(len(fs), num_points)
    return fs[idx], fn[idx], hs[idx].T


def sample_patch(feats, heatmaps, num_points):
    """Cached full-patch arrays -> one sample: (features (15,N), heatmaps (M,N)).

    The features and the heatmaps MUST use the same face draw - two independent draws would
    silently attach every landmark heatmap to the wrong face, which trains cleanly and is
    wrong at inference. That is why this returns both instead of exposing the indices.
    """
    idx = sample_indices(feats.shape[1], num_points)
    return np.ascontiguousarray(feats[:, idx]), heatmaps[idx].T
