"""Extract + cache wav2vec2 features for every clip in the manifest (Ödev 1).

    python odev1/extract.py
    python odev1/extract.py --manifest data/processed/manifest.csv

This is the one expensive step (frozen wav2vec2 forward on ~19.5k clips). It is
resumable — already-cached clips are skipped — and the KNN stage reads only the
cached numpy vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev1.features_w2v import extract_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    extract_all(args.manifest, cache_dir=args.cache_dir, model_name=args.model,
                max_seconds=args.max_seconds, shard=args.shard, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
