'''Feature extraction for both final-assignment methods.

Method 1 uses a 2-D log-mel spectrogram image per clip (input to the CNN).
Method 2 splits the waveform into a fixed number of intervals and summarises
each interval with classical acoustic statistics; the resulting [T, D] series
feeds the LSTM/GRU. Interval count and width are method hyperparameters.

librosa is only used for signal-level feature extraction, which the assignment
explicitly allows; no pretrained model is involved anywhere.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000


def _fingerprint(payload: dict) -> str:
    '''Short stable id so incompatible caches never collide.'''

    text = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def _load_audio(audio_path: str | Path, sample_rate: int) -> np.ndarray:
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f'Audio file does not exist: {audio_path}')
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f'Audio file is empty: {audio_path}')
    if not np.isfinite(audio).all():
        raise ValueError(f'Audio file contains non-finite samples: {audio_path}')
    return audio


# --- Method 1: log-mel spectrogram image ---------------------------------------


@dataclass(frozen=True)
class MelImageConfig:
    '''Parameters of the 2-D log-mel input for the CNN.'''

    sample_rate: int = SAMPLE_RATE
    n_mels: int = 64
    n_frames: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    fmin: float = 20.0
    fmax: float = 8_000.0
    top_db: float = 80.0

    def validate(self) -> None:
        if min(self.sample_rate, self.n_mels, self.n_frames, self.n_fft, self.hop_length) <= 0:
            raise ValueError(f'Mel parameters must be positive: {self}.')
        if self.fmin < 0.0 or self.fmax <= self.fmin or self.top_db <= 0.0:
            raise ValueError(f'Invalid frequency/dB settings: {self}.')

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_mels, self.n_frames)

    @property
    def fingerprint(self) -> str:
        self.validate()
        return 'mel_' + _fingerprint(asdict(self))


def fix_frames(mel_db: np.ndarray, n_frames: int) -> np.ndarray:
    '''Center-crop or edge-pad a log-mel matrix to exactly n_frames columns.'''

    current = mel_db.shape[1]
    if current == 0:
        return np.full((mel_db.shape[0], n_frames), -80.0, dtype=np.float32)
    if current < n_frames:
        floor = float(np.min(mel_db))
        return np.pad(
            mel_db,
            ((0, 0), (0, n_frames - current)),
            mode='constant',
            constant_values=floor,
        ).astype(np.float32, copy=False)
    if current > n_frames:
        start = (current - n_frames) // 2
        return mel_db[:, start : start + n_frames]
    return mel_db.astype(np.float32, copy=False)


def extract_mel_image(
    audio_path: str | Path,
    config: MelImageConfig,
) -> np.ndarray:
    '''Convert one clip to a [n_mels, n_frames] float32 log-mel matrix.'''

    config.validate()
    audio = _load_audio(audio_path, config.sample_rate)
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
    # A positive reference keeps silent clips finite instead of -inf.
    reference = max(float(np.max(mel_power)), float(np.finfo(np.float32).tiny))
    mel_db = librosa.power_to_db(mel_power, ref=reference, top_db=config.top_db)
    image = fix_frames(mel_db, config.n_frames)
    if image.shape != config.shape or not np.isfinite(image).all():
        raise ValueError(f'Invalid mel image for {audio_path}: shape={image.shape}.')
    return image


# --- Method 2: interval feature series ------------------------------------------


@dataclass(frozen=True)
class IntervalConfig:
    '''Interval layout and per-interval settings for the RNN feature series.

    n_intervals and interval_ms are the hyperparameters the assignment asks
    for. Interval start positions are spread evenly over the clip, so short
    clips produce overlapping intervals and long clips leave gaps between
    them; either way every clip yields exactly n_intervals rows.
    '''

    sample_rate: int = SAMPLE_RATE
    n_intervals: int = 24
    interval_ms: int = 300
    n_mfcc: int = 13
    n_fft: int = 512
    hop_length: int = 160

    def validate(self) -> None:
        if min(self.sample_rate, self.n_intervals, self.interval_ms,
               self.n_mfcc, self.n_fft, self.hop_length) <= 0:
            raise ValueError(f'Interval parameters must be positive: {self}.')

    @property
    def interval_samples(self) -> int:
        return int(round(self.sample_rate * self.interval_ms / 1000.0))

    @property
    def feature_dim(self) -> int:
        # MFCC mean/std + delta-MFCC mean + 5 scalar statistics per interval.
        return 3 * self.n_mfcc + 5

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_intervals, self.feature_dim)

    @property
    def fingerprint(self) -> str:
        self.validate()
        return 'seq_' + _fingerprint(asdict(self))


def interval_starts(total_samples: int, config: IntervalConfig) -> np.ndarray:
    '''Evenly spaced start indices of the n_intervals windows.'''

    span = max(total_samples - config.interval_samples, 0)
    if config.n_intervals == 1:
        return np.array([span // 2], dtype=np.int64)
    return np.linspace(0, span, config.n_intervals).round().astype(np.int64)


def _interval_features(segment: np.ndarray, config: IntervalConfig) -> np.ndarray:
    '''Summarise one interval into a fixed-length statistics vector.'''

    if len(segment) < config.n_fft:
        segment = np.pad(segment, (0, config.n_fft - len(segment)))
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    if mfcc.shape[1] >= 2:
        delta = librosa.feature.delta(mfcc, width=min(9, mfcc.shape[1] // 2 * 2 + 1))
    else:
        delta = np.zeros_like(mfcc)
    stft = np.abs(
        librosa.stft(segment, n_fft=config.n_fft, hop_length=config.hop_length)
    )
    rms = librosa.feature.rms(S=stft, frame_length=config.n_fft)[0]
    zcr = librosa.feature.zero_crossing_rate(
        segment, frame_length=config.n_fft, hop_length=config.hop_length
    )[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=config.sample_rate)[0]
    rolloff = librosa.feature.spectral_rolloff(S=stft, sr=config.sample_rate)[0]

    parts = [
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        [
            float(np.log1p(rms.mean())),
            float(rms.std()),
            float(zcr.mean()),
            float(centroid.mean() / (config.sample_rate / 2)),
            float(rolloff.mean() / (config.sample_rate / 2)),
        ],
    ]
    vector = np.concatenate([np.asarray(p, dtype=np.float32).ravel() for p in parts])
    if vector.shape != (config.feature_dim,):
        raise ValueError(f'Unexpected interval vector shape: {vector.shape}.')
    return vector


def extract_interval_series(
    audio_path: str | Path,
    config: IntervalConfig,
) -> np.ndarray:
    '''Convert one clip to a [n_intervals, feature_dim] float32 series.'''

    config.validate()
    audio = _load_audio(audio_path, config.sample_rate)
    window = config.interval_samples
    rows = []
    for start in interval_starts(len(audio), config):
        segment = audio[start : start + window]
        if len(segment) < window:
            segment = np.pad(segment, (0, window - len(segment)))
        rows.append(_interval_features(segment, config))
    series = np.stack(rows).astype(np.float32, copy=False)
    if series.shape != config.shape or not np.isfinite(series).all():
        raise ValueError(
            f'Invalid interval series for {audio_path}: shape={series.shape}.'
        )
    return series
