"""Unsupervised clustering + semi-supervised (reduced-label) analysis.

Implements the proposal's promised component (feature clustering + reduced-label
training) on the MFCC-statistics features. Fast on CPU (reuses cached features).

Examples
--------
    # Full analysis on CREMA-D (clustering + label-efficiency + self-training):
    python scripts/semisupervised.py --config configs/baseline_cremad.yaml

    # On MELD instead:
    python scripts/semisupervised.py --config configs/baseline_cremad.yaml \
        --experiment semisup_meld --train-corpora meld --eval-corpora meld
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.semisupervised import run_all, cluster_analysis, label_efficiency, self_training  # noqa: E402
from ser.utils import ensure_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline_cremad.yaml")
    ap.add_argument("--experiment", default="semisupervised_cremad")
    ap.add_argument("--only", choices=["cluster", "label-eff", "self-train"], default=None,
                    help="Run only one analysis instead of all three.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--eval-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.experiment = args.experiment
    if args.manifest:      cfg.data.manifest = args.manifest
    if args.train_corpora: cfg.data.train_corpora = tuple(args.train_corpora)
    if args.eval_corpora:  cfg.data.eval_corpora = tuple(args.eval_corpora)

    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    if args.only == "cluster":
        cluster_analysis(cfg, out_dir)
    elif args.only == "label-eff":
        label_efficiency(cfg, out_dir)
    elif args.only == "self-train":
        self_training(cfg, out_dir)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
