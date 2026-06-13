"""MFCC features for the classical baseline.

Two views:
  * ``mfcc_sequence`` -> [n_mfcc, T] sequence (not used by the baseline, handy for
    inspection / a possible RNN).
  * ``mfcc_statistics`` -> fixed-length vector of summary statistics over time
    (mean/std of MFCC + delta + delta-delta), which is the standard, strong
    feature set for a frame-pooled SVM/MLP baseline.
"""

from __future__ import annotations

import numpy as np


def mfcc_sequence(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    import librosa

    return librosa.feature.mfcc(
        y=wav,
        sr=sample_rate,
        n_mfcc=feature_cfg.n_mfcc,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        win_length=feature_cfg.win_length,
    ).astype(np.float32)


def mfcc_statistics(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    """Return a 1-D feature vector: mean+std of [MFCC, dMFCC, ddMFCC] over time.

    Length = n_mfcc * 3 (mfcc, delta, delta2) * 2 (mean, std).
    """
    import librosa

    mfcc = mfcc_sequence(wav, feature_cfg, sample_rate)  # [n_mfcc, T]
    if mfcc.shape[1] < 2:
        # too short for deltas -> pad time dimension
        mfcc = np.pad(mfcc, ((0, 0), (0, 2 - mfcc.shape[1])), mode="edge")
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)  # [3*n_mfcc, T]
    feats = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
    return feats.astype(np.float32)
