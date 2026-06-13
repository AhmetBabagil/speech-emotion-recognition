"""Download the CREMA-D audio (7,442 WAVs, original filenames preserved).

Two methods:

* ``"hf"`` (default, reliable) — pull the WAVs from the Hugging Face mirror
  ``AbstractTTS/CREMA-D``. Each row carries a ``file`` field with the ORIGINAL
  filename (e.g. ``1001_DFA_ANG_XX.wav``, encoding actor id + sentence + emotion)
  and the raw WAV bytes, which we write to ``<dest>/AudioWAV/``. No audio decoding
  is needed, so this works cleanly on Windows. Requires ``datasets``
  (``pip install datasets``).

* ``"lfs"`` (alternative) — the canonical GitHub repo via a git-lfs sparse
  checkout of ``AudioWAV``. Correct, but GitHub's LFS bandwidth for this repo is
  frequently throttled to a crawl, so it is not the default.

The original filename convention is preserved either way, so speaker-independent
splitting (which keys on the actor id) works.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"
HF_REPO = "AbstractTTS/CREMA-D"
EXPECTED_MIN_WAVS = 7000


def _count_wavs(audiowav: Path) -> int:
    return len(list(audiowav.glob("*.wav"))) if audiowav.is_dir() else 0


def download_cremad(dest: str = "data/raw/cremad", method: str = "hf") -> Path:
    """Ensure CREMA-D AudioWAV exists under ``dest``. Returns the AudioWAV path."""
    dest = ensure_dir(dest)
    audiowav = dest / "AudioWAV"
    if _count_wavs(audiowav) >= EXPECTED_MIN_WAVS:
        log.info("CREMA-D already present (%d wavs) at %s", _count_wavs(audiowav), audiowav)
        return audiowav

    if method == "hf":
        return _download_via_hf(audiowav)
    if method == "lfs":
        return _download_via_gitlfs(dest, audiowav)
    raise ValueError(f"Unknown method={method!r} (expected 'hf' or 'lfs')")


def _download_via_hf(audiowav: Path) -> Path:
    try:
        from datasets import load_dataset, Audio
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The 'hf' download method needs the `datasets` package. "
            "Install it (`pip install datasets`) or use method='lfs'."
        ) from e
    from tqdm import tqdm

    ensure_dir(audiowav)
    log.info("Downloading CREMA-D from Hugging Face mirror '%s' ...", HF_REPO)
    # decode=False -> we get the raw WAV bytes directly (no torchcodec needed).
    ds = load_dataset(HF_REPO, split="train").cast_column("audio", Audio(decode=False))
    n = 0
    for row in tqdm(ds, total=len(ds), desc="CREMA-D wavs"):
        audio = row["audio"]
        data = audio.get("bytes")
        name = row.get("file") or audio.get("path")
        if data is None or not name:
            continue
        if not str(name).lower().endswith(".wav"):
            name = f"{name}.wav"
        (audiowav / Path(name).name).write_bytes(data)
        n += 1
    if n < EXPECTED_MIN_WAVS:
        raise RuntimeError(f"Only wrote {n} CREMA-D wavs (expected >= {EXPECTED_MIN_WAVS}).")
    log.info("CREMA-D ready: %d wavs at %s", n, audiowav)
    return audiowav


def _run(cmd: list[str], env: dict | None = None) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def _download_via_gitlfs(dest: Path, audiowav: Path) -> Path:
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not (dest / ".git").exists():
        _run(["git", "clone", "--no-checkout", "--filter=blob:none", REPO_URL, str(dest)], env=env)
    _run(["git", "-C", str(dest), "sparse-checkout", "init", "--cone"], env=env)
    _run(["git", "-C", str(dest), "sparse-checkout", "set", "AudioWAV"], env=env)
    branch = "master"
    try:
        out = subprocess.run(
            ["git", "-C", str(dest), "rev-parse", "--abbrev-ref", "origin/HEAD"],
            capture_output=True, text=True, check=True, env=env,
        ).stdout.strip()
        if "/" in out:
            branch = out.split("/", 1)[1]
    except subprocess.CalledProcessError:
        pass
    _run(["git", "-C", str(dest), "checkout", branch], env=env)
    _run(["git", "-C", str(dest), "lfs", "pull", "--include", "AudioWAV/**"])
    n = _count_wavs(audiowav)
    if n < EXPECTED_MIN_WAVS:
        raise RuntimeError(
            f"Expected >= {EXPECTED_MIN_WAVS} CREMA-D wavs but found {n}. "
            "GitHub LFS may be throttled -- try the default method='hf' instead."
        )
    log.info("CREMA-D ready: %d wavs at %s", n, audiowav)
    return audiowav


if __name__ == "__main__":
    download_cremad()
