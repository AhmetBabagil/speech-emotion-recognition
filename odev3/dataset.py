'''Feature caching, train-only normalization and PyTorch datasets.'''

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from odev3.features_melspec import DEFAULT_CONFIG, MelSpecConfig, extract_melspec


def feature_cache_path(
    audio_path: str | Path,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> Path:
    '''Return a stable cache path unique to the file and feature settings.'''

    audio_path = Path(audio_path)
    source_stat = audio_path.stat()
    normalized = os.path.normcase(str(audio_path.resolve()))
    source_identity = (
        f'{normalized}\0{source_stat.st_size}\0{source_stat.st_mtime_ns}'
    )
    digest = hashlib.sha1(source_identity.encode('utf-8')).hexdigest()[:16]
    safe_stem = audio_path.stem[:48]
    return Path(cache_dir) / config.fingerprint / f'{safe_stem}_{digest}.npy'


def _valid_vector(vector: np.ndarray, config: MelSpecConfig) -> bool:
    return (
        vector.shape == (config.vector_size,)
        and np.issubdtype(vector.dtype, np.number)
        and bool(np.isfinite(vector).all())
    )


def load_or_extract(
    audio_path: str | Path,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    '''Load a valid cache entry or atomically compute and store one.'''

    cache_path = feature_cache_path(audio_path, cache_dir, config)
    if cache_path.is_file():
        try:
            cached = np.asarray(np.load(cache_path, allow_pickle=False), dtype=np.float32)
            if _valid_vector(cached, config):
                return cached
        except (OSError, ValueError):
            # A partial/stale cache is replaced below.
            pass

    vector = extract_melspec(audio_path, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(
        f'.{os.getpid()}.{threading.get_ident()}.tmp.npy'
    )
    np.save(temp_path, vector, allow_pickle=False)
    os.replace(temp_path, cache_path)
    return vector


@dataclass(frozen=True)
class FeatureStandardizer:
    '''Per-dimension z-score parameters fitted only on the training fold.'''

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, epsilon: float = 1e-6) -> 'FeatureStandardizer':
        features = np.asarray(features)
        if features.ndim != 2 or len(features) == 0:
            raise ValueError(f'Expected a non-empty 2-D matrix, got {features.shape}.')
        if not np.isfinite(features).all():
            raise ValueError('Training features must contain only finite values.')
        mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = features.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale < epsilon] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f'Feature size {features.shape[-1]} does not match scaler '
                f'size {self.mean.shape[0]}.'
            )
        if not np.isfinite(features).all():
            raise ValueError('Features to transform must contain only finite values.')
        return np.ascontiguousarray((features - self.mean) / self.scale, dtype=np.float32)


class EmotionDataset(Dataset):
    '''Lazy dataset retained for notebooks and small interactive experiments.'''

    def __init__(
        self,
        records,
        cache_dir: str | Path,
        config: MelSpecConfig = DEFAULT_CONFIG,
        standardizer: FeatureStandardizer | None = None,
    ) -> None:
        self.records = records.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.standardizer = standardizer

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records.iloc[index]
        vector = load_or_extract(row['path'], self.cache_dir, self.config)
        if self.standardizer is not None:
            vector = self.standardizer.transform(vector)
        return (
            torch.from_numpy(np.asarray(vector, dtype=np.float32)),
            torch.tensor(int(row['label_idx']), dtype=torch.long),
        )


class ArrayDataset(Dataset):
    '''In-memory tensors used during repeated hyperparameter trials.'''

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
            raise ValueError(
                f'Invalid feature/label shapes: {features.shape}, {labels.shape}.'
            )
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def _progress(iterator: Iterable, total: int, description: str, enabled: bool):
    if not enabled:
        return iterator
    try:
        from tqdm import tqdm

        return tqdm(iterator, total=total, desc=description, unit='ses')
    except ImportError:
        return iterator


def load_feature_matrix(
    records,
    cache_dir: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
    *,
    workers: int = 1,
    show_progress: bool = True,
    description: str = 'Mel özellikleri',
) -> tuple[np.ndarray, np.ndarray]:
    '''Load/cache all records once so trials do not repeat disk extraction.'''

    records = records.reset_index(drop=True)
    paths = records['path'].astype(str).tolist()
    labels = records['label_idx'].to_numpy(dtype=np.int64, copy=True)
    if len(paths) == 0:
        raise ValueError('Cannot build a feature matrix from an empty fold.')

    def load_one(path: str) -> np.ndarray:
        return load_or_extract(path, cache_dir, config)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            vectors = list(
                _progress(
                    executor.map(load_one, paths),
                    len(paths),
                    description,
                    show_progress,
                )
            )
    else:
        vectors = list(
            _progress(map(load_one, paths), len(paths), description, show_progress)
        )

    features = np.stack(vectors).astype(np.float32, copy=False)
    if features.shape != (len(records), config.vector_size):
        raise ValueError(f'Unexpected feature matrix shape: {features.shape}.')
    return features, labels
