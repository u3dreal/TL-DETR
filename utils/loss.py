import torch
from torch import nn
from .matcher import HungarianMatcher


class Criterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.matcher = HungarianMatcher(cost_prob=100, cost_heat=1)
        self.loss_h = nn.MSELoss()
        self.loss_c = nn.CrossEntropyLoss()

    def forward(self, g_heats, probs, heats):
        # A tooth with no cusp GT must never reach this function: `pl_model._shared_step` routes
        # it to `nch_loss`, and here it would otherwise either raise from inside `torch.cdist`
        # (a 1-D tensor after flatten) or return NaN through an empty matching. Both read like
        # a torch problem rather than a contract violation, so say what is wrong.
        if g_heats.numel() == 0:
            raise ValueError(
                'Criterion needs at least one cusp target per sample; a tooth without cusps '
                'belongs on the nch_loss path (pl_model._shared_step)')
        indices = self.matcher(probs, heats, g_heats)

        loss = 0
        for i, idx in enumerate(indices):  # for batch_size
            gheat = g_heats[i, idx[1], :].squeeze()

            gprob = torch.zeros(probs.shape[1], dtype=torch.long).to(probs.device)
            gprob[idx[0]] = 1

            pheat = heats[i, idx[0], :]
            loss_h = self.loss_h(pheat.squeeze(), gheat.squeeze())  # 0.08
            loss_c = self.loss_c(probs.squeeze(), gprob)  # 0.65

            loss = loss + loss_c + loss_h * 10

        return loss