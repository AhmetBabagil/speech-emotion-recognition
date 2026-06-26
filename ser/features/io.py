"""Audio I/O: load any WAV to a mono float32 array at the target sample rate,
and fix it to a fixed number of samples (pad or crop)."""

from __future__ import annotations

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load ``path`` as mono float32 resampled to ``target_sr``.

    Uses librosa (which wraps soundfile/audioread) so it works uniformly on
    Windows/Linux for the WAVs produced by CREMA-D and by our MELD ffmpeg
    extraction. Returns a 1-D float32 array in roughly [-1, 1].

    A corrupt/unreadable file does not crash a whole run: we log a warning and
    return a short silent clip (it is then padded to the fixed length upstream).
    """
    import librosa

    try:
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        log.warning("Failed to load audio %s (%s) -- using silence.", path, e)
        return np.zeros(int(target_sr), dtype=np.float32)
    wav = np.asarray(wav, dtype=np.float32)
    # Guard against all-zero / NaN clips.
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav, copy=False)
    return wav


def fix_length(
    wav: np.ndarray,
    num_samples: int,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pad (with zeros) or crop ``wav`` to exactly ``num_samples`` samples.

    On training we randomly crop longer clips for augmentation; otherwise we
    take a centered crop so evaluation is deterministic.
    """
    n = wav.shape[0]
    if n == num_samples:
        return wav
    if n < num_samples:
        pad = num_samples - n
        # pad symmetrically-ish (front pad random on train would also work)
        return np.pad(wav, (0, pad), mode="constant")
    # n > num_samples -> crop
    if random_crop:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, n - num_samples + 1))
    else:
        start = (n - num_samples) // 2
    return wav[start : start + num_samples]
