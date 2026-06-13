"""Log-mel spectrogram features for the CNN model."""

from __future__ import annotations

import numpy as np


def fixed_num_frames(num_samples: int, hop_length: int) -> int:
    """Number of mel frames produced by librosa for ``num_samples`` with
    ``center=True`` (the default). Used to make every spectrogram the same
    width so they batch cleanly."""
    return num_samples // hop_length + 1


def log_mel_spectrogram(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    """Return a [n_mels, T] float32 log-mel spectrogram (decibel scale).

    ``feature_cfg`` is a :class:`ser.config.FeatureConfig`.
    """
    import librosa

    fmax = feature_cfg.fmax
    if fmax is None or fmax > sample_rate / 2:
        fmax = sample_rate / 2
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        win_length=feature_cfg.win_length,
        n_mels=feature_cfg.n_mels,
        fmin=feature_cfg.fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def fix_frames(spec: np.ndarray, num_frames: int, *, random_crop: bool = False,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """Pad/crop a [n_mels, T] spectrogram to exactly ``num_frames`` along time."""
    t = spec.shape[1]
    if t == num_frames:
        return spec
    if t < num_frames:
        pad = num_frames - t
        return np.pad(spec, ((0, 0), (0, pad)), mode="constant",
                      constant_values=spec.min())
    if random_crop:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, t - num_frames + 1))
    else:
        start = (t - num_frames) // 2
    return spec[:, start : start + num_frames]


def standardize(spec: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-instance zero-mean/unit-variance normalization."""
    mu = spec.mean()
    sd = spec.std()
    return (spec - mu) / (sd + eps)


def spec_augment(spec: np.ndarray, freq_mask: int, time_mask: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Lightweight SpecAugment: one frequency band + one time band masked to 0."""
    rng = rng or np.random.default_rng()
    spec = spec.copy()
    n_mels, t = spec.shape
    if freq_mask > 0 and n_mels > freq_mask:
        f = int(rng.integers(0, freq_mask + 1))
        if f > 0:
            f0 = int(rng.integers(0, n_mels - f + 1))
            spec[f0 : f0 + f, :] = 0.0
    if time_mask > 0 and t > time_mask:
        m = int(rng.integers(0, time_mask + 1))
        if m > 0:
            t0 = int(rng.integers(0, t - m + 1))
            spec[:, t0 : t0 + m] = 0.0
    return spec
