'''Tests for the mel-spectrogram feature layer used by the MLP.'''

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from odev3.features_melspec import (
    DEFAULT_CONFIG,
    MelSpecConfig,
    extract_melspec,
    fix_frames,
)


def _write_tone(path: Path, *, frequency: float = 220.0) -> None:
    sample_rate = DEFAULT_CONFIG.sample_rate
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = 0.25 * np.sin(2.0 * np.pi * frequency * time)
    sf.write(path, audio, sample_rate)


def test_default_vector_meets_assignment_minimum() -> None:
    assert DEFAULT_CONFIG.vector_size == 4096
    assert DEFAULT_CONFIG.vector_size >= 4000


def test_config_fingerprint_changes_with_feature_parameters() -> None:
    changed = MelSpecConfig(hop_length=256)

    assert DEFAULT_CONFIG.fingerprint != changed.fingerprint
    assert DEFAULT_CONFIG.fingerprint == MelSpecConfig().fingerprint


@pytest.mark.parametrize(
    'config',
    [
        MelSpecConfig(sample_rate=0),
        MelSpecConfig(n_frames=-1),
        MelSpecConfig(fmin=100.0, fmax=50.0),
        MelSpecConfig(top_db=0.0),
    ],
)
def test_invalid_configs_are_rejected(config: MelSpecConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()


def test_fix_frames_right_pads_short_input_with_floor() -> None:
    mel = np.array([[2.0, 3.0], [-4.0, 5.0]], dtype=np.float32)

    fixed = fix_frames(mel, n_frames=4)

    assert fixed.shape == (2, 4)
    np.testing.assert_array_equal(fixed[:, :2], mel)
    np.testing.assert_array_equal(fixed[:, 2:], np.full((2, 2), -4.0))


def test_fix_frames_centre_crops_long_input() -> None:
    mel = np.arange(18, dtype=np.float32).reshape(2, 9)

    fixed = fix_frames(mel, n_frames=5)

    np.testing.assert_array_equal(fixed, mel[:, 2:7])


def test_extract_melspec_returns_finite_float32_vector(tmp_path: Path) -> None:
    audio_path = tmp_path / 'tone.wav'
    _write_tone(audio_path)

    vector = extract_melspec(audio_path)

    assert vector.shape == (4096,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    assert float(vector.max()) <= 1e-4
    assert float(vector.min()) >= -DEFAULT_CONFIG.top_db - 1e-4


def test_extract_melspec_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_melspec(tmp_path / 'missing.wav')
