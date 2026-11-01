import os
import numpy as np
import random
import json
import torch
import trimesh
from torch.utils.data import Dataset
from data.common import calc_features, geodesic_heatmaps, sample, sample_patch
from utils.landmarks import CLASS_TO_CHANNEL, CUSP_CLASS, FIXED_LANDMARK_CLASSES


class TeethLandDataset(Dataset):
    def __init__(self, args, split_file: str, train: bool):
        self.args = args
        self.files = []
        self.num_points = args.num_points
        self.augmentation = args.augmentation if train else False
        # `--steps features` caches <t_idx>_feat.npy next to the patch. Reading the cache is
        # optional: a patch without one still loads from the .off, so a half-finished
        # preprocessing run trains, just slower. Set use_feature_cache=False to force the
        # mesh path (the tests use that to prove the two agree).
        self.use_feature_cache = getattr(args, 'use_feature_cache', True)

        with open(os.path.join(args.split_root, split_file)) as f:
            for line in f:
                filename = line.strip()
                if not filename:
                    continue
                teeth_root = os.path.join(self.args.patch_root, filename)

                t_files = sorted(os.listdir(teeth_root))
                for f in t_files:
                    if f.endswith('.off'):
                        t_idx = os.path.splitext(f)[0]
                        off_file = os.path.join(teeth_root, t_idx + '.off')
                        js_file = os.path.join(teeth_root, t_idx + '.json')
                        gd_file = os.path.join(teeth_root, t_idx + '_f.npy')
                        if os.path.exists(off_file) and os.path.exists(js_file) and os.path.exists(gd_file):
                            self.files.append((off_file, js_file, gd_file, t_idx))
        if train:
            # only shuffle the training view; val/test order must be stable so that
            # limit_val_batches and error reports stay comparable across runs
            random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        off_file, js_file, gd_file, t_idx = self.files[idx]
        landmarks = json.load(open(js_file))
        geo_dists = np.load(gd_file)
        heatmaps = geodesic_heatmaps(geo_dists, self.args)

        # Fast path: `tools/prepare_dataset.py --steps features` caches the deterministic
        # 15 x F feature matrix for the whole patch, so a step no longer parses a ~1 MB
        # ASCII .off and recomputes face normals (10-80 ms/sample measured on real
        # patches - the reason worker processes were needed at all). The face subsample is
        # still drawn at random here, so sampling remains an augmentation.
        feat_file = os.path.splitext(off_file)[0] + '_feat.npy'
        if self.use_feature_cache and os.path.exists(feat_file):
            # features and heatmaps come from ONE draw, never two
            feat_np, heatmaps = sample_patch(np.load(feat_file), heatmaps, self.num_points)
            features = torch.from_numpy(feat_np)
        else:
            mesh = trimesh.load(off_file)
            vs = mesh.vertices
            vs = vs - vs.mean(0)
            fs, fn = mesh.faces, mesh.face_normals
            fs, fn, heatmaps = sample(fs, fn, heatmaps, self.num_points)
            vs = torch.tensor(vs, dtype=torch.float32)
            fs = torch.tensor(fs, dtype=torch.long)
            fn = torch.tensor(fn, dtype=torch.float32)
            features = calc_features(vs, fs, fn)  # (15, nf)

        cusp_heatmaps = []
        fixed_heatmaps = np.zeros([len(FIXED_LANDMARK_CLASSES), heatmaps.shape[1]], dtype=np.float32)
        mask = np.zeros([len(FIXED_LANDMARK_CLASSES), self.num_points])
        for i, lm in enumerate(landmarks):
            cls = lm['class']
            if cls == CUSP_CLASS:
                cusp_heatmaps.append(heatmaps[i])
                continue
            ch = CLASS_TO_CHANNEL.get(cls)
            if ch is None:
                # unknown landmark class: ignore loudly rather than silently
                # shifting the channel order (which would swap mesial/distal)
                print(f"[warn] {js_file}: unknown landmark class {cls!r}, ignored")
                continue
            fixed_heatmaps[ch] = heatmaps[i]
            mask[ch, :] = 1
        cusp_heatmaps = np.array(cusp_heatmaps)

        t_idx_int = int(t_idx) if str(t_idx).lstrip('-').isdigit() else -1

        return features, t_idx_int, \
            torch.tensor(mask, dtype=torch.float32), \
            torch.tensor(fixed_heatmaps, dtype=torch.float32), \
            torch.tensor(cusp_heatmaps, dtype=torch.float32)


if __name__ == '__main__':
    class Args(object):
        def __init__(self):
            self.split_root = 'F:/dataset/3DTeethLand/Split_exp'
            self.patch_root = 'F:/dataset/3DTeethLand/patches'
            self.augmentation = True
            self.normalize_pc = True
            self.landmark_std = 0.5
            self.num_points = 5000

    data = TeethLandDataset(Args(), 'train.txt', True)
    i = 0
    for feats, t_idx, mask, oh,  ch in data:
        print(t_idx)
        print(mask.shape)
        print(feats.shape)
        print(oh.shape)
        print(ch.shape)
        i += 1
    print(i)



