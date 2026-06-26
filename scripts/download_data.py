"""Download CREMA-D and/or MELD and extract MELD audio.

Examples
--------
    # Everything (CREMA-D + full MELD, ~10 GB download + ffmpeg extraction):
    python scripts/download_data.py --datasets cremad meld

    # CREMA-D only (fast, ~1-2 GB):
    python scripts/download_data.py --datasets cremad

    # MELD with a small per-split audio cap (for a quick test):
    python scripts/download_data.py --datasets meld --meld-limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.data.download_cremad import download_cremad  # noqa: E402
from ser.data.download_meld import download_meld, extract_meld_audio  # noqa: E402
from ser.utils import get_logger  # noqa: E402

log = get_logger("download")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["cremad", "meld"],
                    choices=["cremad", "meld"])
    ap.add_argument("--data-root", default="data/raw")
    ap.add_argument("--cremad-method", default="hf", choices=["hf", "lfs"],
                    help="'hf' = Hugging Face mirror (reliable); 'lfs' = GitHub git-lfs.")
    ap.add_argument("--meld-limit", type=int, default=None,
                    help="Cap audio extraction to N utterances per split (testing).")
    ap.add_argument("--meld-keep-archive", action="store_true",
                    help="Keep the 10 GB MELD.Raw.tar.gz after extraction.")
    args = ap.parse_args()
    root = Path(args.data_root)

    if "cremad" in args.datasets:
        log.info("=== CREMA-D ===")
        download_cremad(str(root / "cremad"), method=args.cremad_method)

    if "meld" in args.datasets:
        log.info("=== MELD ===")
        csv_dir = download_meld(str(root / "meld"), keep_archive=args.meld_keep_archive)
        extract_meld_audio(csv_dir, str(root / "meld"), limit=args.meld_limit)

    log.info("Done. Next: python scripts/build_manifest.py")


if __name__ == "__main__":
    main()
