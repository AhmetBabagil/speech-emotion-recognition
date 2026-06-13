"""Run the full experiment suite end-to-end and aggregate results.

Assumes data is already downloaded and the manifest is built. Runs:
  * MFCC baseline (CREMA-D, MELD)
  * CNN within-corpus (CREMA-D, MELD)
  * CNN cross-corpus matrix (CREMA-D <-> MELD)
then writes outputs/results.csv.

    python scripts/run_all.py                 # everything
    python scripts/run_all.py --skip-baseline # CNN only
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

log = get_logger("run_all")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    def base_cfg(name):
        cfg = Config.from_yaml(f"configs/{name}.yaml")
        cfg.data.manifest = args.manifest
        if args.epochs:
            cfg.train.epochs = args.epochs
        return cfg

    if not args.skip_baseline:
        log.info("### Baseline: CREMA-D ###")
        train_baseline(base_cfg("baseline_cremad"), kind="svm")
        c = base_cfg("baseline_cremad")
        c.experiment = "baseline_meld"
        c.data.train_corpora = ("meld",); c.data.eval_corpora = ("meld",); c.data.split = "meld_official"
        log.info("### Baseline: MELD ###")
        train_baseline(c, kind="svm")

    log.info("### CNN within: CREMA-D ###")
    train_torch(base_cfg("cnn_cremad"))
    log.info("### CNN within: MELD ###")
    train_torch(base_cfg("cnn_meld"))

    if not args.skip_cross:
        log.info("### CNN cross-corpus matrix ###")
        cc = base_cfg("cnn_cremad")
        cc.experiment = "cnn"
        run_cross_corpus(cc)

    log.info("Aggregating results ...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name("aggregate_results.py"))])


if __name__ == "__main__":
    main()
