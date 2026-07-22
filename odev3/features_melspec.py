'''Fixed-size log-mel features for the PyTorch MLP.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class MelSpecConfig:
    '''All parameters that affect a cached feature vector.'''

    sample_rate: int = 16_000
    n_mels: int = 64
    n_frames: int = 64
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 20.0
    fmax: float = 8_000.0
    top_db: float = 80.0
    frame_strategy: str = 'crop_pad'

    def validate(self) -> None:
        '''Reject settings that cannot produce a meaningful mel vector.'''

        integer_fields = {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'n_frames': self.n_frames,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
        }
        invalid = {name: value for name, value in integer_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f'Mel parameters must be positive; invalid={invalid}.')
        if self.fmin < 0.0 or self.fmax <= self.fmin:
            raise ValueError(
                f'Expected 0 <= fmin < fmax, got fmin={self.fmin}, '
                f'fmax={self.fmax}.'
            )
        if self.top_db <= 0.0:
            raise ValueError(f'top_db must be positive, got {self.top_db}.')
        if self.frame_strategy not in {'crop_pad', 'resize'}:
            raise ValueError(
                'frame_strategy must be crop_pad or resize, got '
                f'{self.frame_strategy!r}.'
            )

    @property
    def vector_size(self) -> int:
        return self.n_mels * self.n_frames

    @property
    def fingerprint(self) -> str:
        '''Short stable id used to keep incompatible caches separate.'''

        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


DEFAULT_CONFIG = MelSpecConfig()

# Backward-compatible constants used in notebooks and tests.
SAMPLE_RATE = DEFAULT_CONFIG.sample_rate
N_MELS = DEFAULT_CONFIG.n_mels
N_FRAMES = DEFAULT_CONFIG.n_frames
N_FFT = DEFAULT_CONFIG.n_fft
HOP_LENGTH = DEFAULT_CONFIG.hop_length
VECTOR_SIZE = DEFAULT_CONFIG.vector_size


def fix_frames(
    mel_db: np.ndarray,
    n_frames: int = N_FRAMES,
    *,
    strategy: str = 'crop_pad',
) -> np.ndarray:
    '''Convert a log-mel matrix to exactly n_frames time columns.'''

    mel_db = np.asarray(mel_db, dtype=np.float32)
    if mel_db.ndim != 2 or mel_db.shape[0] == 0:
        raise ValueError(f'Expected a non-empty 2-D mel matrix, got {mel_db.shape}.')
    if n_frames <= 0:
        raise ValueError(f'n_frames must be positive, got {n_frames}.')
    if strategy not in {'crop_pad', 'resize'}:
        raise ValueError(f'Unknown frame strategy: {strategy!r}.')

    current_frames = mel_db.shape[1]
    if current_frames == 0:
        return np.full((mel_db.shape[0], n_frames), -80.0, dtype=np.float32)
    if strategy == 'resize' and current_frames != n_frames:
        if current_frames == 1:
            return np.repeat(mel_db, n_frames, axis=1).astype(
                np.float32, copy=False
            )
        source_positions = np.linspace(0.0, 1.0, current_frames)
        target_positions = np.linspace(0.0, 1.0, n_frames)
        resized = np.empty((mel_db.shape[0], n_frames), dtype=np.float32)
        for mel_index, row in enumerate(mel_db):
            resized[mel_index] = np.interp(target_positions, source_positions, row)
        return resized
    if current_frames < n_frames:
        floor = float(np.min(mel_db))
        return np.pad(
            mel_db,
            ((0, 0), (0, n_frames - current_frames)),
            mode='constant',
            constant_values=floor,
        ).astype(np.float32, copy=False)
    if current_frames > n_frames:
        start = (current_frames - n_frames) // 2
        return mel_db[:, start : start + n_frames]
    return mel_db


def extract_melspec(
    audio_path: str | Path,
    config: MelSpecConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    '''Convert one audio file to a finite float32 log-mel vector.'''

    config.validate()
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f'Audio file does not exist: {audio_path}')

    audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f'Audio file is empty: {audio_path}')
    if not np.isfinite(audio).all():
        raise ValueError(f'Audio file contains non-finite samples: {audio_path}')

    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=min(config.fmax, config.sample_rate / 2),
        power=2.0,
    )
    # A positive reference handles silent recordings without divide-by-zero.
    reference = max(float(np.max(mel_power)), float(np.finfo(np.float32).tiny))
    mel_db = librosa.power_to_db(mel_power, ref=reference, top_db=config.top_db)
    mel_fixed = fix_frames(
        mel_db,
        config.n_frames,
        strategy=config.frame_strategy,
    )
    vector = mel_fixed.reshape(config.vector_size).astype(np.float32, copy=False)

    if vector.shape != (config.vector_size,) or not np.isfinite(vector).all():
        raise ValueError(
            f'Invalid mel vector for {audio_path}: shape={vector.shape}, '
            f'finite={bool(np.isfinite(vector).all())}'
        )
    return vector
