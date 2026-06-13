"""Download MELD raw data and extract per-utterance audio with ffmpeg.

Pipeline:
  1. Download MELD.Raw.tar.gz (~10 GB, resumable).
  2. Extract the outer archive, then the nested train/dev/test tarballs into
     ``<dest>/video/<split>/``.
  3. For every utterance whose emotion is in the common six, transcode its .mp4
     to 16 kHz mono WAV at ``<dest>/audio/<split>/diaX_uttY.wav`` using ffmpeg.

Only the common-six utterances are transcoded (``surprise`` is skipped), which
saves time and disk. Corrupt/silent clips are logged and skipped.
"""

from __future__ import annotations

import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

from ..constants import MELD_LABEL_TO_CANONICAL
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

MELD_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download(url: str, out_path: Path) -> Path:
    from tqdm import tqdm

    out_path = Path(out_path)
    resume_from = out_path.stat().st_size if out_path.exists() else 0
    req = urllib.request.Request(url)
    if resume_from:
        req.add_header("Range", f"bytes={resume_from}-")
        log.info("Resuming MELD download from %.2f GB", resume_from / 1e9)

    with urllib.request.urlopen(req, timeout=60) as resp:
        # If the server ignored Range it returns 200 -> restart from scratch.
        if resume_from and resp.status == 200:
            log.warning("Server ignored resume; restarting download.")
            resume_from = 0
        total = int(resp.headers.get("Content-Length", 0)) + resume_from
        mode = "ab" if resume_from else "wb"
        with open(out_path, mode) as f, tqdm(
            total=total or None, initial=resume_from, unit="B", unit_scale=True,
            desc="MELD.Raw.tar.gz",
        ) as bar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    return out_path


def _remote_size(url: str) -> int | None:
    """Content-Length of the remote file, or None if unknown."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception as e:
        log.warning("Could not HEAD %s: %s", url, e)
        return None


def _ensure_complete_download(url: str, out_path: Path, max_retries: int = 10) -> None:
    """Download with resume + retries until the file matches the remote size.

    A dropped connection leaves a truncated file; we resume (HTTP Range) until the
    on-disk size reaches Content-Length, so extraction never runs on a partial tar.
    """
    expected = _remote_size(url)
    for attempt in range(1, max_retries + 1):
        have = out_path.stat().st_size if out_path.exists() else 0
        if expected and have >= expected:
            log.info("Archive complete: %.2f GB.", have / 1e9)
            return
        if expected:
            log.info("Have %.2f / %.2f GB -- download attempt %d/%d",
                     have / 1e9, expected / 1e9, attempt, max_retries)
        try:
            _download(url, out_path)
        except Exception as e:  # network blip -> retry/resume
            log.warning("Download attempt %d failed: %s", attempt, e)
        if expected is None:
            return  # cannot verify size; assume the single pass sufficed
    have = out_path.stat().st_size if out_path.exists() else 0
    if expected and have < expected:
        raise RuntimeError(
            f"MELD download still incomplete after {max_retries} attempts "
            f"({have}/{expected} bytes). Re-run to resume."
        )


def _safe_extract(tar_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with tarfile.open(tar_path, "r:*") as tar:
        members = tar.getmembers()
        base = out_dir.resolve()
        for m in members:
            target = (out_dir / m.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe path in tar: {m.name}")
        tar.extractall(out_dir)


def _classify_split(name: str) -> str | None:
    n = name.lower()
    if "train" in n:
        return "train"
    if "dev" in n:
        return "dev"
    if "test" in n:
        return "test"
    return None


def download_meld(dest: str = "data/raw/meld", keep_archive: bool = True) -> Path:
    """Download + extract MELD raw. Returns the directory containing the CSVs."""
    dest = ensure_dir(dest)
    archive = dest / "MELD.Raw.tar.gz"

    csv_dir = _find_csv_dir(dest)
    if csv_dir is None:
        # Download to completion (resumes a truncated file) BEFORE extracting.
        _ensure_complete_download(MELD_URL, archive)
        log.info("Extracting outer archive ...")
        _safe_extract(archive, dest)
        csv_dir = _find_csv_dir(dest)
    if csv_dir is None:
        raise RuntimeError(f"Could not find *_sent_emo.csv under {dest} after extraction.")

    # Extract nested per-split video tarballs into video/<split>/
    for nested in csv_dir.glob("*.tar.gz"):
        split = _classify_split(nested.name)
        if split is None:
            continue
        target = dest / "video" / split
        if any(target.rglob("*.mp4")):
            continue
        log.info("Extracting %s -> %s", nested.name, target)
        _safe_extract(nested, target)

    if keep_archive is False and archive.exists():
        archive.unlink()
    log.info("MELD raw ready. CSV dir: %s", csv_dir)
    return csv_dir


def _find_csv_dir(root: Path) -> Path | None:
    for csv in root.rglob("train_sent_emo.csv"):
        return csv.parent
    return None


# --------------------------------------------------------------------------- #
# Audio extraction
# --------------------------------------------------------------------------- #
def _build_mp4_index(video_split_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for mp4 in video_split_dir.rglob("*.mp4"):
        index[mp4.stem] = mp4
    return index


def extract_meld_audio(
    csv_dir: str | Path,
    dest: str = "data/raw/meld",
    ffmpeg_bin: str = "ffmpeg",
    limit: int | None = None,
) -> Path:
    """Transcode common-six MELD utterances to 16 kHz mono WAV.

    Returns the audio root dir (``<dest>/audio``) with split subfolders.
    """
    from tqdm import tqdm

    csv_dir = Path(csv_dir)
    dest = Path(dest)
    audio_root = ensure_dir(dest / "audio")

    for split, csv_name in SPLIT_CSV.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            log.warning("Missing %s -- skipping %s split", csv_name, split)
            continue
        df = pd.read_csv(csv_path)
        video_dir = dest / "video" / split
        mp4_index = _build_mp4_index(video_dir)
        out_dir = ensure_dir(audio_root / split)

        rows = df.itertuples(index=False)
        done = 0
        n_ok = n_skip = n_fail = 0
        for row in tqdm(rows, total=len(df), desc=f"MELD {split} audio"):
            r = row._asdict()
            emotion = str(r["Emotion"]).strip().lower()
            if MELD_LABEL_TO_CANONICAL.get(emotion) is None:
                n_skip += 1
                continue  # surprise / unknown
            key = f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}"
            mp4 = mp4_index.get(key)
            out_wav = out_dir / f"{key}.wav"
            if out_wav.exists():
                n_ok += 1
                continue
            if mp4 is None:
                n_fail += 1
                continue
            ok = _ffmpeg_to_wav(ffmpeg_bin, mp4, out_wav)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            done += 1
            if limit is not None and done >= limit:
                break
        log.info("MELD %s: ok=%d skipped=%d failed/missing=%d", split, n_ok, n_skip, n_fail)
    return audio_root


def _ffmpeg_to_wav(ffmpeg_bin: str, mp4: Path, out_wav: Path) -> bool:
    cmd = [ffmpeg_bin, "-y", "-loglevel", "error", "-i", str(mp4),
           "-vn", "-ac", "1", "-ar", "16000", str(out_wav)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return out_wav.exists() and out_wav.stat().st_size > 0
    except subprocess.CalledProcessError as e:
        log.debug("ffmpeg failed for %s: %s", mp4.name, e.stderr.decode(errors="ignore")[:200])
        if out_wav.exists():
            out_wav.unlink()
        return False


if __name__ == "__main__":
    csv_dir = download_meld()
    extract_meld_audio(csv_dir)
