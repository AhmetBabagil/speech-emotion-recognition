"""Experiment configuration: nested dataclasses loaded from / dumped to YAML.

A single ``Config`` object is threaded through data loading, feature extraction,
model building, training and evaluation so every stage agrees on the audio
parameters (sample rate, clip length, mel bins, ...).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .constants import SAMPLE_RATE


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    # Fixed clip length (seconds) used for batching spectrograms/waveforms.
    # Shorter clips are zero-padded, longer clips are (randomly on train) cropped.
    clip_seconds: float = 4.0

    @property
    def num_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_seconds))


@dataclass
class FeatureConfig:
    # Shared STFT params (used by both MFCC and mel-spectrogram).
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float | None = 8000.0  # <= sample_rate / 2
    # MFCC (baseline) params.
    n_mfcc: int = 40
    # SpecAugment-style masking (applied on the log-mel during training only).
    freq_mask: int = 8
    time_mask: int = 16
    augment: bool = True


@dataclass
class ModelConfig:
    # one of: "cnn", "wav2vec2"  (the MFCC baseline is sklearn and ignores this)
    name: str = "cnn"
    dropout: float = 0.3
    # CNN-specific
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256)
    # wav2vec2-specific
    pretrained_name: str = "facebook/wav2vec2-base"
    freeze_feature_encoder: bool = True


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    # "none" | "inverse" | "balanced" -- class weighting for CrossEntropyLoss.
    class_weighting: str = "balanced"
    label_smoothing: float = 0.05
    early_stop_patience: int = 8
    num_workers: int = 0  # Windows-safe default; raise on Linux/GPU box
    grad_clip: float = 5.0
    seed: int = 42
    # validation/monitor metric: "macro_f1" or "accuracy"
    monitor: str = "macro_f1"
    amp: bool = True  # automatic mixed precision (ignored on CPU)


@dataclass
class DataConfig:
    manifest: str = "data/processed/manifest.csv"
    # Which corpora to include for this run, e.g. ["cremad"], ["meld"], or both.
    train_corpora: tuple[str, ...] = ("cremad",)
    eval_corpora: tuple[str, ...] = ("cremad",)
    # Split strategy (within-corpus only): "speaker" (speaker-independent),
    # "meld_official" (MELD's official dialogue-based folds, NOT speaker-independent),
    # or "random". In cross-corpus mode this is ignored: the training corpus is
    # always speaker-independently split into train/val and the eval corpus is the
    # entire test set.
    split: str = "speaker"
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    cache_features: bool = True
    cache_dir: str = "data/cache"


@dataclass
class Config:
    experiment: str = "default"
    output_dir: str = "outputs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # ---- (de)serialization ----------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d or {})
        sub = {
            "audio": AudioConfig,
            "feature": FeatureConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "data": DataConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, klass in sub.items():
            section = d.pop(key, {}) or {}
            kwargs[key] = _build_dataclass(klass, section)
        # remaining top-level scalars (experiment, output_dir); ignore unknown
        # keys so a stray annotation in a YAML file doesn't crash loading.
        valid = {f.name for f in dataclasses.fields(cls)}
        for k, v in d.items():
            if k in valid:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build_dataclass(klass, section: dict[str, Any]):
    """Build a dataclass, coercing list->tuple for tuple-typed fields and
    ignoring unknown keys (with a clear error for typos would be nicer, but we
    keep it permissive so configs can carry extra annotations)."""
    valid = {f.name for f in dataclasses.fields(klass)}
    tuple_fields = {
        f.name for f in dataclasses.fields(klass)
        if "tuple" in str(f.type).lower()
    }
    kwargs = {}
    for k, v in (section or {}).items():
        if k not in valid:
            continue
        if k in tuple_fields and isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return klass(**kwargs)
