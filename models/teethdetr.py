import numpy as np
import torch
import torch.nn as nn
from .dgcnn import Backbone, SharedMLP1d, EdgeConv, knn
from utils.landmarks import FIXED_LANDMARK_LABELS, LABEL_CUSP


class QueryDecoder(nn.Module):
    def __init__(self, args):
        super(QueryDecoder, self).__init__()

        self.k = args.k

        self.smlp1 = SharedMLP1d([args.emb_dims+args.n_edgeconvs_backbone*64, 256, 64], args.norm)
        self.conv = EdgeConv([64*2, 128], args.k, args.norm)

    def forward(self, x, p):
        x = self.smlp1(x)
        x = self.conv(x, knn(p, self.k))

        return x


class TeethDETR(nn.Module):
    def __init__(self, args):
        super(TeethDETR, self).__init__()

        self.query_num = args.query_num
        self.dynamic = args.dynamic
        self.backbone = Backbone(args)

        self.query_decoder = QueryDecoder(args)

        self.fheats_out = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                        nn.Dropout(args.dropout),
                                        nn.Conv1d(64, 5, kernel_size=1))

        self.cusp_decoder = nn.Sequential(SharedMLP1d([128, 64], args.norm),
                                          nn.Dropout(args.dropout),
                                          nn.Conv1d(64, args.query_num, kernel_size=1))
        self.cprob_out = nn.Sequential(SharedMLP1d([args.num_points, 1024], args.norm),
                                       SharedMLP1d([1024, 256], args.norm),
                                       SharedMLP1d([256, 64], args.norm),
                                       nn.Dropout(args.dropout),
                                       nn.Conv1d(64, 2, kernel_size=1), )  # 2 x 50

        self.cheats_out = nn.Sequential(SharedMLP1d([args.query_num, args.query_num], args.norm),
                                        nn.Dropout(args.dropout),
                                        nn.Conv1d(args.query_num, args.query_num, kernel_size=1))

    def _heads(self, x):
        """Shared trunk -> (fheats (B,5,N), cprobs (B,Q,2), cheats (B,Q,N))."""
        xyz = x[:, :3, :].contiguous()  # B,3,N

        feat = self.backbone(x)
        feat = self.query_decoder(feat, feat if self.dynamic else xyz)  # B,128,N

        fheats = self.fheats_out(feat)

        cheat_feat = self.cusp_decoder(feat)  # B,M,N
        cprobs = self.cprob_out(cheat_feat.permute(0, 2, 1))  # B,2,M
        cheats = self.cheats_out(cheat_feat)  # B,N,M

        return fheats, cprobs.permute(0, 2, 1), cheats

    def forward(self, x):
        return self._heads(x)

    def inference(self, x):
        """Decode landmark coordinates for a batch.

        Returns a list of (points (P,3) float32, labels (P,) int64), one entry per
        sample. P = 5 fixed landmarks + however many cusps the query head accepted.

        Batch-safe: the previous implementation used bare .squeeze(), which
        collapsed the batch dimension and silently returned garbage for B > 1.
        """
        with torch.no_grad():
            fheats, cprobs, cheats = self._heads(x)

            xyz_np = x[:, :3, :].permute(0, 2, 1).detach().float().cpu().numpy()  # B,N,3
            fh_np = fheats.detach().float().cpu().numpy()                        # B,5,N
            cheat_np = cheats.detach().float().cpu().numpy()                     # B,Q,N
            prob_np = cprobs.detach().float().cpu().numpy()                      # B,Q,2

        batch_size = xyz_np.shape[0]
        out = []
        for b in range(batch_size):
            # fixed landmarks: one argmax per channel
            f_idx = np.argmax(fh_np[b], axis=1)              # (5,)
            f_xyz = xyz_np[b][f_idx]                         # (5,3)

            # cusps: query i is "real" when class-1 probability wins
            # NOTE: cprobs is (B, Q, 2) - the class axis is last. The original code
            # did `argmax(axis=0)` on this, which takes the argmax across the *query*
            # axis and is meaningless; it happened not to crash, so it went unnoticed.
            c_idx = np.argmax(cheat_np[b], axis=1)           # (Q,)
            keep = np.argmax(prob_np[b], axis=-1) == 1       # (Q,)
            c_xyz = xyz_np[b][c_idx][keep]

            p_xyz = np.concatenate([f_xyz, c_xyz], axis=0).astype(np.float32)
            p_label = np.concatenate([FIXED_LANDMARK_LABELS,
                                      np.full(len(c_xyz), LABEL_CUSP, dtype=np.int64)])
            out.append((p_xyz, p_label))
        return out
