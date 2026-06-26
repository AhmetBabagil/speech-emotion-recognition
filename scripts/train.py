"""Train / evaluate a model from a YAML config, with CLI overrides.

Examples
--------
    # Classical MFCC baseline on CREMA-D:
    python scripts/train.py --config configs/baseline_cremad.yaml --baseline

    # CNN on log-mel, CREMA-D (speaker-independent):
    python scripts/train.py --config configs/cnn_cremad.yaml

    # wav2vec2 transfer learning (needs `pip install -e .[transfer]` + GPU):
    python scripts/train.py --config configs/wav2vec2_cremad.yaml

    # Full within + cross-corpus matrix for the CNN:
    python scripts/train.py --config configs/cnn_cremad.yaml --cross-corpus

    # Override anything on the command line:
    python scripts/train.py --config configs/cnn_cremad.yaml --epochs 5 --experiment quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.train import train_torch, train_baseline  # noqa: E402
from ser.cross_corpus import run_cross_corpus  # noqa: E402
from ser.utils import get_logger  # noqa: E402

log = get_logger("train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline", action="store_true", help="Use the classical MFCC baseline.")
    ap.add_argument("--baseline-kind", default="svm", choices=["svm", "logreg", "rf"])
    ap.add_argument("--cross-corpus", action="store_true",
                    help="Run the within+cross corpus matrix instead of a single run.")
    # common overrides
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--model", default=None, choices=["cnn", "wav2vec2"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--eval-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--split", default=None, choices=["speaker", "meld_official", "random"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.experiment:    cfg.experiment = args.experiment
    if args.model:         cfg.model.name = args.model
    if args.epochs:        cfg.train.epochs = args.epochs
    if args.batch_size:    cfg.train.batch_size = args.batch_size
    if args.manifest:      cfg.data.manifest = args.manifest
    if args.train_corpora: cfg.data.train_corpora = tuple(args.train_corpora)
    if args.eval_corpora:  cfg.data.eval_corpora = tuple(args.eval_corpora)
    if args.split:         cfg.data.split = args.split

    if args.cross_corpus:
        run_cross_corpus(cfg, use_baseline=args.baseline, baseline_kind=args.baseline_kind)
    elif args.baseline:
        train_baseline(cfg, kind=args.baseline_kind)
    else:
        train_torch(cfg)


if __name__ == "__main__":
    main()
