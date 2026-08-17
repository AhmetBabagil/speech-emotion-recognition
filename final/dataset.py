'''Feature caching, train-only standardization and in-memory datasets.

Both methods produce a fixed-shape float32 array per clip, so one cache and
one loader cover the CNN images and the RNN series alike.
'''

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def feature_cache_path(
    audio_path: str | Path,
    cache_dir: str | Path,
    fingerprint: str,
) -> Path:
    '''Stable cache location unique to the source file and feature settings.'''

    audio_path = Path(audio_path)
    stat = audio_path.stat()
    identity = (
        f'{os.path.normcase(str(audio_path.resolve()))}\0{stat.st_size}\0{stat.st_mtime_ns}'
    )
    digest = hashlib.sha1(identity.encode('utf-8')).hexdigest()[:16]
    return Path(cache_dir) / fingerprint / f'{audio_path.stem[:48]}_{digest}.npy'


def load_or_extract(
    audio_path: str | Path,
    cache_dir: str | Path,
    fingerprint: str,
    extract: Callable[[str | Path], np.ndarray],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    '''Return a valid cached array or atomically compute and store one.'''

    cache_path = feature_cache_path(audio_path, cache_dir, fingerprint)
    if cache_path.is_file():
        try:
            cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)
            if cached.shape == expected_shape and np.isfinite(cached).all():
                return cached
        except (OSError, ValueError):
            pass  # partial/stale cache entry is replaced below

    array = extract(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp.npy')
    np.save(temp_path, array, allow_pickle=False)
    os.replace(temp_path, cache_path)
    return array


def _progress(iterator: Iterable, total: int, description: str, enabled: bool):
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc=description, unit='ses')
    except ImportError:
        return iterator


def load_feature_tensor(
    records,
    cache_dir: str | Path,
    fingerprint: str,
    extract: Callable[[str | Path], np.ndarray],
    expected_shape: tuple[int, ...],
    *,
    workers: int = 1,
    show_progress: bool = True,
    description: str = 'Öznitelikler',
) -> tuple[np.ndarray, np.ndarray]:
    '''Extract/cache every record of a fold into one [N, ...] tensor.'''

    records = records.reset_index(drop=True)
    paths = records['path'].astype(str).tolist()
    labels = records['label_idx'].to_numpy(dtype=np.int64, copy=True)
    if not paths:
        raise ValueError('Cannot build features from an empty fold.')

    def load_one(path: str) -> np.ndarray:
        return load_or_extract(path, cache_dir, fingerprint, extract, expected_shape)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            arrays = list(
                _progress(executor.map(load_one, paths), len(paths), description, show_progress)
            )
    else:
        arrays = list(_progress(map(load_one, paths), len(paths), description, show_progress))

    features = np.stack(arrays).astype(np.float32, copy=False)
    if features.shape != (len(records), *expected_shape):
        raise ValueError(f'Unexpected feature tensor shape: {features.shape}.')
    return features, labels


@dataclass(frozen=True)
class Standardizer:
    '''Z-score parameters fitted only on the training fold.

    For [N, mels, T] mel images the statistics are per mel bin; for
    [N, T, D] interval series they are per feature dimension. Both reduce
    over the sample and time axes, i.e. everything except ``feature_axis``.
    '''

    mean: np.ndarray
    scale: np.ndarray
    feature_axis: int

    @classmethod
    def fit(cls, features: np.ndarray, feature_axis: int, epsilon: float = 1e-6) -> 'Standardizer':
        features = np.asarray(features)
        if features.ndim != 3 or len(features) == 0:
            raise ValueError(f'Expected a non-empty 3-D tensor, got {features.shape}.')
        if not np.isfinite(features).all():
            raise ValueError('Training features must be finite.')
        axes = tuple(axis for axis in range(features.ndim) if axis != feature_axis)
        mean = features.mean(axis=axes, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=axes, dtype=np.float64).astype(np.float32)
        scale = np.where(scale < epsilon, 1.0, scale)
        return cls(mean=mean, scale=scale, feature_axis=feature_axis)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        shape = [1] * features.ndim
        shape[self.feature_axis] = self.mean.shape[0]
        mean = self.mean.reshape(shape)
        scale = self.scale.reshape(shape)
        return np.ascontiguousarray((features - mean) / scale, dtype=np.float32)


class ArrayDataset(Dataset):
    '''In-memory tensors reused across hyperparameter trials.'''

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(
                f'Invalid feature/label shapes: {features.shape}, {labels.shape}.'
            )
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
