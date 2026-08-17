'''Shape/determinism checks for the final-assignment feature extractors.'''

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from final.dataset import Standardizer, load_or_extract  # noqa: E402
from final.features import (  # noqa: E402
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
    interval_starts,
)


@pytest.fixture()
def wav_path(tmp_path: Path) -> Path:
    '''A 1.5 s synthetic tone so tests never depend on corpus files.'''

    sr = 16_000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / 'tone.wav'
    sf.write(path, audio, sr)
    return path


def test_mel_image_shape_and_determinism(wav_path: Path) -> None:
    config = MelImageConfig(n_mels=32, n_frames=48)
    first = extract_mel_image(wav_path, config)
    second = extract_mel_image(wav_path, config)
    assert first.shape == (32, 48)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_interval_starts_cover_short_and_long_audio() -> None:
    config = IntervalConfig(n_intervals=8, interval_ms=300)
    window = config.interval_samples
    short = interval_starts(window // 2, config)  # shorter than one interval
    long = interval_starts(window * 20, config)
    assert len(short) == len(long) == 8
    assert short.min() >= 0 and short.max() == 0
    assert long.min() == 0 and long.max() == window * 20 - window


def test_interval_series_shape(wav_path: Path) -> None:
    config = IntervalConfig(n_intervals=6, interval_ms=250)
    series = extract_interval_series(wav_path, config)
    assert series.shape == config.shape == (6, config.feature_dim)
    assert np.isfinite(series).all()
    # Identical intervals of a stationary tone must give near-identical rows.
    assert np.std(series[1:-1], axis=0).max() < 1.0


def test_cache_roundtrip(wav_path: Path, tmp_path: Path) -> None:
    config = IntervalConfig(n_intervals=4, interval_ms=200)
    cache_dir = tmp_path / 'cache'
    calls = {'n': 0}

    def counting_extract(path):
        calls['n'] += 1
        return extract_interval_series(path, config)

    first = load_or_extract(wav_path, cache_dir, config.fingerprint,
                            counting_extract, config.shape)
    second = load_or_extract(wav_path, cache_dir, config.fingerprint,
                             counting_extract, config.shape)
    assert calls['n'] == 1
    np.testing.assert_array_equal(first, second)


def test_standardizer_axes() -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(3.0, 2.0, size=(50, 12, 7)).astype(np.float32)
    scaler = Standardizer.fit(series, feature_axis=2)
    transformed = scaler.transform(series)
    means = transformed.mean(axis=(0, 1))
    stds = transformed.std(axis=(0, 1))
    assert np.abs(means).max() < 1e-4
    assert np.abs(stds - 1.0).max() < 1e-3

    images = rng.normal(0.0, 5.0, size=(50, 16, 20)).astype(np.float32)
    scaler = Standardizer.fit(images, feature_axis=1)
    transformed = scaler.transform(images)
    assert np.abs(transformed.mean(axis=(0, 2))).max() < 1e-4
