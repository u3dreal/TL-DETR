import argparse
import types

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from models.teethdetr import TeethDETR
from data.st_data import TeethLandDataset
from utils.loss import Criterion

# torch >= 2.6 defaults torch.load(weights_only=True), which refuses to unpickle the
# argparse.Namespace we stash in hparams -> load_from_checkpoint fails with
# "Unsupported global: GLOBAL types.SimpleNamespace". Allowlist the two container types
# instead of disabling weights_only wholesale.
try:
    torch.serialization.add_safe_globals([argparse.Namespace, types.SimpleNamespace])
except AttributeError:                                    # torch < 2.6
    pass


class LitModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.save_hyperparameters()
        self.net = TeethDETR(args)
        self.fh_loss = nn.MSELoss()
        self.nch_loss = nn.CrossEntropyLoss()
        self.ch_loss = Criterion(args)

    def forward(self, x):
        return self.net.forward(x)

    def infer(self, x):
        """Returns a list of (points, labels) - one entry per batch element."""
        return self.net.inference(x)

    def _shared_step(self, batch):
        """Returns (loss, batch_size, fh_loss, ch_loss).

        The two terms are returned so they can be logged apart. `loss` is
        `fh_loss * 10 + ch_loss`, so a single `val_loss` says almost nothing about the cusp
        head: the five fixed landmarks outvote it 10:1, and the cusp channel is the part whose
        supervision changed (one point per tooth before, every cusp now - `FINDINGS.md` §24g).
        ModelCheckpoint still selects on `val_loss`; the split is what tells a reader whether
        the cusps were still improving when the landmarks had already plateaued.
        """
        x, t_idx, mask, g_fheats, g_cheats = batch

        p_fheats, cprobs, p_cheats = self(x)
        fh_loss = self.fh_loss(p_fheats * mask, g_fheats * mask)  # ~0.18
        if g_cheats.shape[1] == 0:
            # no cusp GT for this tooth: there is nothing to predict, push "no object" class.
            # device MUST come from a live tensor - never hardcode cuda/mps.
            target = torch.zeros([cprobs.shape[0], cprobs.shape[1]],
                                 dtype=torch.long, device=cprobs.device)
            ch_loss = self.nch_loss(cprobs.squeeze(), target.squeeze())  # ~0.62
            loss = fh_loss * 10 + ch_loss * 2
        else:
            ch_loss = self.ch_loss(g_cheats, cprobs, p_cheats)
            loss = fh_loss * 10 + ch_loss
        return loss, x.size(0), fh_loss.detach(), ch_loss.detach()

    def training_step(self, batch, _):
        loss, bs, fh, ch = self._shared_step(batch)
        self.log('loss', loss, batch_size=bs)
        self.log('fh_loss', fh, batch_size=bs)
        self.log('ch_loss', ch, batch_size=bs)
        self.log('lr', self.optimizers().param_groups[0]['lr'])
        return loss

    def validation_step(self, batch, _):
        loss, bs, fh, ch = self._shared_step(batch)
        self.log('val_loss', loss, prog_bar=True, batch_size=bs)
        self.log('val_fh_loss', fh, batch_size=bs)
        self.log('val_ch_loss', ch, batch_size=bs)

    def test_step(self, batch, _):
        loss, bs, fh, ch = self._shared_step(batch)
        self.log('test_loss', loss, prog_bar=True, batch_size=bs)
        self.log('test_fh_loss', fh, batch_size=bs)
        self.log('test_ch_loss', ch, batch_size=bs)

    def configure_optimizers(self):
        args = self.hparams.args
        optimizer = torch.optim.Adam(self.net.parameters(), args.lr_max, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, float(args.lr_max),
                                                        pct_start=args.pct_start, div_factor=float(args.div_factor),
                                                        final_div_factor=float(args.final_div_factor),
                                                        epochs=args.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()))
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def _loader(self, split_file, train):
        args = self.hparams.args
        # pin_memory is meaningless (and warns) on MPS/CPU
        return DataLoader(TeethLandDataset(args, split_file, train),
                          batch_size=args.batch_size,
                          shuffle=train,
                          num_workers=args.train_workers if train else getattr(args, 'val_workers', 0),
                          pin_memory=False)

    def train_dataloader(self):
        return self._loader(self.hparams.args.train_file, True)

    def val_dataloader(self):
        return self._loader(self.hparams.args.val_file, False)

    def test_dataloader(self):
        return self._loader(self.hparams.args.test_file, False)


class LitModelInference(LitModel):
    def forward(self, x):
        # self.net(x) returns a (fheats, cprobs, cheats) tuple - the old
        # torch.argmax(self.net(x), dim=2) could never have worked. Use the
        # decoded points instead.
        return self.net.inference(x)
