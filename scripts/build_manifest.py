"""Build the unified CREMA-D + MELD manifest CSV.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --out data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.data.build_manifest import build_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cremad-dir", default="data/raw/cremad/AudioWAV")
    ap.add_argument("--meld-audio-root", default="data/raw/meld/audio")
    ap.add_argument("--meld-csv-dir", default=None,
                    help="Auto-detected under data/raw/meld if omitted.")
    ap.add_argument("--out", default="data/processed/manifest.csv")
    args = ap.parse_args()

    build_manifest(
        cremad_dir=args.cremad_dir,
        meld_csv_dir=args.meld_csv_dir,
        meld_audio_root=args.meld_audio_root,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
