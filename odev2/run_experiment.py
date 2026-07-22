"""Run Project Assignment 2 experiments.

Examples:
    python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --grid-mode report
    python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --quick --corpora cremad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev2.model_pipeline import MODEL_SPECS, run_all  # noqa: E402


def main() -> None:
    """CLI argumanlarini okuyup Odev 2 deney yoneticisi `run_all` fonksiyonunu cagirir."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--out-root", default="odev2/outputs")
    ap.add_argument("--knn-out-root", default="odev1/outputs")
    ap.add_argument("--corpora", nargs="+", default=["cremad", "meld"], choices=["cremad", "meld"])
    ap.add_argument("--models", nargs="+", choices=[s.name for s in MODEL_SPECS])
    ap.add_argument("--grid-mode", choices=["quick", "report", "full"], default="report")
    ap.add_argument("--quick", action="store_true", help="Alias for --grid-mode quick.")
    args = ap.parse_args()

    run_all(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        out_root=args.out_root,
        corpora=tuple(args.corpora),
        quick=args.quick,
        grid_mode=args.grid_mode,
        model_keys=tuple(args.models) if args.models else None,
        knn_out_root=args.knn_out_root,
    )


if __name__ == "__main__":
    main()
