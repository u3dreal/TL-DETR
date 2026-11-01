"""Single source of truth for landmark identity.

The landmark ORDER is load-bearing and was previously implicit in two places:
  * data/st_data.py  concatenates heatmap channels in this order
  * models/teethdetr.py::inference emitted the hardcoded label array [1,2,3,4,5]
Any mismatch silently swaps mesial/distal and destroys the Bolton ratio
(MD width would be measured between the wrong pair of points). Import these
constants instead of re-typing the numbers.

Label scheme (as used by the 3DTeethLand annotations / TL-DETR):
    1 Mesial, 2 Distal, 3 InnerPoint, 4 OuterPoint, 5 FacialPoint, 6 Cusp
"""

import numpy as np
import torch

# heatmap channel order and label id, in the order st_data.py concatenates them
FIXED_LANDMARK_CLASSES = ("Mesial", "Distal", "InnerPoint", "OuterPoint", "FacialPoint")

LABEL_MESIAL = 1
LABEL_DISTAL = 2
LABEL_INNER = 3
LABEL_OUTER = 4
LABEL_FACIAL = 5
LABEL_CUSP = 6

FIXED_LANDMARK_LABELS = np.array(
    [LABEL_MESIAL, LABEL_DISTAL, LABEL_INNER, LABEL_OUTER, LABEL_FACIAL], dtype=np.int64
)

# channel indices
MESIAL_CHANNEL = 0
DISTAL_CHANNEL = 1
INNER_CHANNEL = 2
OUTER_CHANNEL = 3
FACIAL_CHANNEL = 4

CUSP_CLASS = "Cusp"

CLASS_TO_CHANNEL = {name: i for i, name in enumerate(FIXED_LANDMARK_CLASSES)}

# The two channels the Bolton ratio is built from.
MD_CHANNELS = (MESIAL_CHANNEL, DISTAL_CHANNEL)


def soft_argmax(heat, xyz, tau: float = 0.1, topk: int = 64):
    """Sub-mesh-resolution landmark decode.

    A hard argmax can only ever return a sampled triangle centre, so its precision
    is bounded by the sampling spacing (~0.3-0.6 mm on a 10 mm tooth), which is
    2-4x the 0.15-0.25 mm per-tooth width budget needed for +-1% Bolton.
    This returns a weighted centroid of the top-k peaks instead.

    Args:
        heat: (..., N) score per point - any real dtype, must share a device with xyz
        xyz:  (..., N, 3) point coordinates
        tau:  temperature; smaller = closer to hard argmax
        topk: number of peaks to blend (clamped to N)

    Returns:
        (..., 3) decoded points, same dtype/device as xyz.

    NOT yet wired into TeethDETR.inference - Phase 4 of the plan.
    """
    n = heat.shape[-1]
    k = min(int(topk), n)
    v, i = heat.topk(k, dim=-1)                       # (..., k)
    w = torch.softmax(v / tau, dim=-1)                # (..., k)
    gather = i.unsqueeze(-1).expand(*i.shape, xyz.shape[-1])
    pts = xyz.gather(-2, gather)                      # (..., k, 3)
    return (pts * w.unsqueeze(-1)).sum(-2)            # (..., 3)
