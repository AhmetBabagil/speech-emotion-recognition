"""PyTorch dataset + caching for spectrogram/waveform models, plus the MFCC
feature-matrix builder used by the classical baseline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import NUM_CLASSES
from ..features import io as audio_io
from ..features.melspec import (
    log_mel_spectrogram,
    fix_frames,
    fixed_num_frames,
    standardize,
    spec_augment,
)
from ..features.mfcc import mfcc_statistics
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)


def _feature_hash(cfg, kind: str) -> str:
    """Short stable hash of the feature params that affect cached arrays."""
    f = cfg.feature
    key = (
        kind,
        cfg.audio.sample_rate,
        f.n_fft,
        f.hop_length,
        f.win_length,
        f.n_mels,
        f.fmin,
        f.fmax,
        f.n_mfcc,
    )
    return hashlib.md5(repr(key).encode()).hexdigest()[:10]


def _cache_path(cache_dir: Path, corpus: str, audio_path: str, h: str) -> Path:
    # Include the parent folder name in the key. MELD's dia{D}_utt{U} ids restart
    # per split, so dia0_utt0 exists as DIFFERENT clips under audio/train, audio/dev
    # and audio/test; keying on the stem alone would collide and load the wrong
    # cached spectrogram. (CREMA-D filenames are globally unique; parent = AudioWAV.)
    p = Path(audio_path)
    return cache_dir / corpus / f"{p.parent.name}_{p.stem}__{h}.npy"


def _load_cached(path: Path):
    try:
        return np.load(path)
    except Exception:
        return None


def _save_cached(path: Path, arr: np.ndarray) -> None:
    try:
        ensure_dir(path.parent)
        tmp = path.with_suffix(".npy.tmp")
        np.save(tmp, arr)
        tmp.replace(path)
    except Exception as e:  # caching is best-effort
        log.debug("cache write failed for %s: %s", path, e)


class SERDataset:
    """torch.utils.data.Dataset over a manifest DataFrame.

    mode="logmel"  -> returns (FloatTensor[1, n_mels, T], label) for the CNN.
    mode="waveform"-> returns (FloatTensor[num_samples], label) for wav2vec2.
    """

    def __init__(self, df: pd.DataFrame, cfg, *, mode: str = "logmel", train: bool = False):
        import torch  # local import so non-torch tools (manifest) don't need it

        self.torch = torch
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.train = train
        # Reproducible augmentation: a per-sample RNG is seeded from
        # (global seed, epoch, index). This is deterministic for a given run
        # (full reproducibility) yet varies across epochs (real augmentation),
        # and is worker-safe because it carries no shared mutable state.
        self.seed = int(cfg.train.seed)
        self.epoch = 0
        self.num_samples = cfg.audio.num_samples
        self.num_frames = fixed_num_frames(self.num_samples, cfg.feature.hop_length)
        self.use_cache = cfg.data.cache_features and mode == "logmel"
        self.cache_dir = Path(cfg.data.cache_dir)
        self._hash = _feature_hash(cfg, "logmel")

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation across epochs while staying reproducible."""
        self.epoch = int(epoch)

    def _full_logmel(self, row) -> np.ndarray:
        cache_p = _cache_path(self.cache_dir, row["corpus"], row["path"], self._hash)
        if self.use_cache:
            cached = _load_cached(cache_p)
            if cached is not None:
                return cached
        wav = audio_io.load_audio(row["path"], self.cfg.audio.sample_rate)
        spec = log_mel_spectrogram(wav, self.cfg.feature, self.cfg.audio.sample_rate)
        if self.use_cache:
            _save_cached(cache_p, spec)
        return spec

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label_idx"])
        # Seeded only for training (eval is deterministic: center crop, no masking).
        rng = np.random.default_rng([self.seed, self.epoch, idx]) if self.train else None

        if self.mode == "waveform":
            wav = audio_io.load_audio(row["path"], self.cfg.audio.sample_rate)
            wav = audio_io.fix_length(wav, self.num_samples, random_crop=self.train, rng=rng)
            # zero-mean / unit-var (what wav2vec2 expects)
            wav = (wav - wav.mean()) / (wav.std() + 1e-5)
            return self.torch.from_numpy(wav.astype(np.float32)), label

        # mode == "logmel"
        spec = self._full_logmel(row)
        spec = fix_frames(spec, self.num_frames, random_crop=self.train, rng=rng)
        spec = standardize(spec)
        if self.train and self.cfg.feature.augment:
            spec = spec_augment(spec, self.cfg.feature.freq_mask,
                                self.cfg.feature.time_mask, rng=rng)
        tensor = self.torch.from_numpy(np.ascontiguousarray(spec))[None, :, :]
        return tensor, label


def class_weights(df: pd.DataFrame, scheme: str = "balanced"):
    """Return a length-NUM_CLASSES weight tensor for CrossEntropyLoss.

    "balanced": n_total / (n_classes * count_c)   (sklearn-style)
    "inverse" : 1 / count_c (normalized to mean 1)
    "none"    : all ones
    """
    import torch

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in df["label_idx"].astype(int):
        counts[c] += 1
    counts = np.maximum(counts, 1.0)
    if scheme == "none":
        w = np.ones(NUM_CLASSES)
    elif scheme == "inverse":
        w = 1.0 / counts
        w = w / w.mean()
    else:  # balanced
        w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32)


def mfcc_feature_matrix(df: pd.DataFrame, cfg, *, show_progress: bool = True):
    """Compute the [N, D] MFCC-statistics matrix and label vector for ``df``.

    Used by the classical (sklearn) baseline. Cached per-file as .npy.
    """
    from tqdm import tqdm

    cache_dir = Path(cfg.data.cache_dir)
    h = _feature_hash(cfg, "mfccstat")
    X, y = [], []
    skipped = 0
    it = df.itertuples(index=False)
    if show_progress:
        it = tqdm(it, total=len(df), desc="MFCC features")
    for row in it:
        row = row._asdict()
        feat = None
        cache_p = _cache_path(cache_dir, row["corpus"], row["path"], h)
        if cfg.data.cache_features:
            feat = _load_cached(cache_p)
        if feat is None:
            try:
                wav = audio_io.load_audio(row["path"], cfg.audio.sample_rate)
                feat = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)
            except Exception as e:
                log.warning("MFCC extraction failed for %s: %s", row["path"], e)
                skipped += 1
                continue
            if cfg.data.cache_features:
                _save_cached(cache_p, feat)
        X.append(feat)
        y.append(int(row["label_idx"]))
    if skipped:
        log.warning("Skipped %d unreadable file(s) during MFCC extraction.", skipped)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
