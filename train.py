import os
import sys

import yaml
import argparse
import torch
import pytorch_lightning as pl

from pl_model import LitModel

from pytorch_lightning.callbacks import ModelCheckpoint


def select_accelerator():
    """Pick the best available backend on this machine.

    Note: this is launcher-level policy only. Nothing under models/ or utils/
    may hardcode a device - they derive it from their input tensors.
    """
    if torch.cuda.is_available():
        return 'gpu'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_parser():
    parser = argparse.ArgumentParser(description='Tooth Landmark Detection')
    parser.add_argument("--config", type=str, default="config/tooth_landmark.yaml", help="path to config file")
    # overrides (handy for smoke tests / short runs)
    parser.add_argument("--accelerator", type=str, default='auto',
                        choices=['auto', 'mps', 'cpu', 'gpu'], help="training backend")
    parser.add_argument("--max-epochs", type=int, default=None, help="override TRAIN.max_epochs")
    parser.add_argument("--limit-train-batches", type=int, default=None,
                        help="override: only run N training batches per epoch (smoke tests)")
    parser.add_argument("--limit-val-batches", type=int, default=None,
                        help="override: only run N validation batches per epoch")
    parser.add_argument("--debug", action='store_true', default=None,
                        help="short run: 1 epoch, 10 train batches, no checkpointing")
    parser.add_argument("--skip-test", action='store_true', help="do not run trainer.test() at the end")
    parser.add_argument("--fast-dev-run", action='store_true',
                        help="Lightning fast_dev_run: 1 batch train + 1 batch val, no checkpoint")
    parser.add_argument("--split-root", type=str, default=None, help="override DATA.split_root")
    parser.add_argument("--patch-root", type=str, default=None, help="override DATA.patch_root")
    parser.add_argument("--num-points", type=int, default=None, help="override DATA.num_points")
    parser.add_argument("--experiment", type=str, default=None,
                        help="override GENERAL.experiment (names the runs/<experiment> log dir)")
    parser.add_argument("--train-workers", type=int, default=None,
                        help="override TRAIN.train_workers")

    args_cfg = parser.parse_args()

    # Anything the user typed on the command line must win over the config file,
    # otherwise cfg keys like TRAIN.debug silently overwrite --debug.
    explicit = {}
    argv = sys.argv[1:]
    for action in parser._actions:
        if any(opt in argv or any(a.split('=')[0] == opt for a in argv)
               for opt in action.option_strings):
            explicit[action.dest] = getattr(args_cfg, action.dest)

    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        if not isinstance(config[key], dict):
            continue
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    for k, v in explicit.items():
        setattr(args_cfg, k, v)

    if args_cfg.train_workers is None:
        args_cfg.train_workers = 0
    return args_cfg


def build_logger(experiment):
    """TensorBoard if available, CSV otherwise - never crash a run over logging."""
    try:
        from pytorch_lightning.loggers import TensorBoardLogger
        return TensorBoardLogger("runs", experiment)
    except (ImportError, ModuleNotFoundError):
        from pytorch_lightning.loggers import CSVLogger
        print("[warn] tensorboard not installed, using CSVLogger")
        return CSVLogger("runs", experiment)


def build_trainer_kwargs(args):
    """Translate config + CLI overrides into pl.Trainer kwargs."""
    kwargs = dict(
        logger=build_logger(args.experiment),
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
    )

    # Nested keys the config cannot express: only enable checkpointing when the
    # val loop actually runs, otherwise every epoch "improves" a bogus val_loss.
    if args.limit_val_batches == 0:
        kwargs['enable_checkpointing'] = False
    else:
        kwargs['callbacks'] = [
            ModelCheckpoint(monitor='val_loss', save_top_k=5, save_last=True, mode='min')
        ]

    if args.limit_train_batches is not None:
        kwargs['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        kwargs['limit_val_batches'] = args.limit_val_batches
    if args.fast_dev_run:
        kwargs['fast_dev_run'] = True

    return kwargs


if __name__ == "__main__":
    args = get_parser()

    if args.accelerator == 'auto':
        args.accelerator = select_accelerator()

    # torch.cdist(..., p=1) in utils/matcher.py has no MPS kernel; PYTORCH_ENABLE_MPS_FALLBACK
    # lets that single op drop to CPU instead of raising. Everything hot-path is MPS-native.
    if args.accelerator == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
        print("[warn] PYTORCH_ENABLE_MPS_FALLBACK is not set to 1. "
              "utils/matcher.py uses torch.cdist(..., p=1), which may raise on MPS.")

    if args.debug:
        args.max_epochs = 1
        args.limit_train_batches = 10
        args.limit_val_batches = 2 if args.limit_val_batches is None else args.limit_val_batches
        print("[debug] 1 epoch, 10 train batches, 2 val batches")

    pl.seed_everything(args.seed)

    model = LitModel(args)
    if args.load_from_checkpoint:
        model = LitModel.load_from_checkpoint(args.load_from_checkpoint, map_location='cpu')
        # the checkpoint carries the args it was trained with; keep runtime paths fresh
        model.hparams.args.split_root = args.split_root
        model.hparams.args.patch_root = args.patch_root

    trainer = pl.Trainer(**build_trainer_kwargs(args))
    trainer.fit(model)

    if not args.skip_test and not args.fast_dev_run:
        ckpt = 'best'
        best = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ''
        if not best:
            print("[skip] no best checkpoint was written, not running test()")
        else:
            results = trainer.test(model, ckpt_path=ckpt, verbose=False)
            print(results)
