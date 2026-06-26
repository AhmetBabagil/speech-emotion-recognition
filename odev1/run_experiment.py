"""Ödev 1 KNN deneyini iki veri setinde de çalıştır.

Önkoşul: önce `python odev1/extract.py` ile wav2vec2 öznitelikleri çıkarılmalı.

    python odev1/run_experiment.py
    python odev1/run_experiment.py --corpora cremad      # tek veri seti
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev1.knn_pipeline import run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--out-root", default="odev1/outputs")
    ap.add_argument("--corpora", nargs="+", default=["cremad", "meld"], choices=["cremad", "meld"])
    args = ap.parse_args()
    run_all(args.manifest, args.cache_dir, args.out_root, tuple(args.corpora))


if __name__ == "__main__":
    main()
