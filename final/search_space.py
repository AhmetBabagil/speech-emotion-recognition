'''Deterministic hyperparameter candidates for both final methods.

Every candidate pairs a feature configuration with a model configuration, so
feature-level hyperparameters (mel resolution for Method 1; interval count
and width for Method 2, as the assignment requires) are searched on the
validation fold exactly like model-level ones.
'''

from __future__ import annotations

from dataclasses import replace

from final.features import IntervalConfig, MelImageConfig
from final.models import CNNConfig, OptimSettings, RNNConfig

GRID_MODES = ('quick', 'report', 'full')

CNNCandidate = tuple[MelImageConfig, CNNConfig]
RNNCandidate = tuple[IntervalConfig, RNNConfig]


def _unique(candidates):
    result, seen = [], set()
    for feature_cfg, model_cfg in candidates:
        feature_cfg.validate()
        model_cfg.validate()
        key = (feature_cfg, model_cfg)
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


# --- Method 1: mel image + CNN ---------------------------------------------------

CNN_BASE_FEATURES = MelImageConfig()
CNN_BASE_MODEL = CNNConfig()


def cnn_space(mode: str) -> list[CNNCandidate]:
    if mode == 'quick':
        small = MelImageConfig(n_mels=32, n_frames=48)
        return _unique([
            (small, CNNConfig(channels=(8, 16), dropout=0.0,
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, CNNConfig(channels=(16, 32), dropout=0.3,
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':
        base_f, base_m = CNN_BASE_FEATURES, CNN_BASE_MODEL
        base_o = base_m.optim
        return _unique([
            (base_f, base_m),
            # one-factor model/optimization variations
            (base_f, replace(base_m, channels=(16, 32, 64))),
            (base_f, replace(base_m, channels=(32, 64, 128, 256))),
            (base_f, replace(base_m, dropout=0.1)),
            (base_f, replace(base_m, dropout=0.5)),
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-3))),
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=1e-4))),
            (base_f, replace(base_m, optim=replace(base_o, batch_size=64))),
            (base_f, replace(base_m, optim=replace(base_o, weight_decay=0.0))),
            # one-factor feature variations
            (replace(base_f, n_frames=96), base_m),
            (replace(base_f, n_mels=80), base_m),
        ])
    if mode == 'full':
        candidates = []
        for channels in ((16, 32, 64), (32, 64, 128), (32, 64, 128, 256)):
            for dropout in (0.1, 0.3, 0.5):
                for lr in (1e-3, 3e-4):
                    candidates.append((
                        CNN_BASE_FEATURES,
                        CNNConfig(channels=channels, dropout=dropout,
                                  optim=replace(CNN_BASE_MODEL.optim, learning_rate=lr)),
                    ))
        return _unique(candidates)
    raise ValueError(f'Unknown grid mode {mode!r}; expected one of {GRID_MODES}.')


def cnn_refinement(winner: CNNCandidate, exclude=()) -> list[CNNCandidate]:
    '''Local second-stage search around the validation winner.'''

    feature_cfg, model_cfg = winner
    optim = model_cfg.optim
    scaled_down = tuple(max(8, c // 2) for c in model_cfg.channels)
    scaled_up = tuple(min(512, c * 2) for c in model_cfg.channels)
    candidates = _unique([
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=min(3e-3, optim.learning_rate * 2)))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=max(16, optim.batch_size // 2)))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, batch_size=min(128, optim.batch_size * 2)))),
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),
        (feature_cfg, replace(model_cfg, channels=scaled_down)),
        (feature_cfg, replace(model_cfg, channels=scaled_up)),
    ])
    excluded = set(exclude) | {winner}
    return [c for c in candidates if c not in excluded]


# --- Method 2: interval series + LSTM/GRU ---------------------------------------

RNN_BASE_FEATURES = IntervalConfig()
RNN_BASE_MODEL = RNNConfig()


def rnn_space(mode: str) -> list[RNNCandidate]:
    if mode == 'quick':
        small = IntervalConfig(n_intervals=8, interval_ms=200)
        return _unique([
            (small, RNNConfig(rnn_type='gru', hidden_size=32, num_layers=1, bidirectional=False,
                              dropout=0.0, pooling='mean',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
            (small, RNNConfig(rnn_type='lstm', hidden_size=32, num_layers=1, bidirectional=True,
                              dropout=0.0, pooling='last',
                              optim=OptimSettings(batch_size=64, learning_rate=1e-3, patience=2))),
        ])
    if mode == 'report':
        base_f, base_m = RNN_BASE_FEATURES, RNN_BASE_MODEL
        base_o = base_m.optim
        return _unique([
            (base_f, base_m),
            # interval layout is a first-class hyperparameter (assignment req.)
            (replace(base_f, n_intervals=16), base_m),
            (replace(base_f, n_intervals=32), base_m),
            (replace(base_f, interval_ms=200), base_m),
            (replace(base_f, interval_ms=400), base_m),
            # one-factor model/optimization variations
            (base_f, replace(base_m, rnn_type='lstm')),
            (base_f, replace(base_m, hidden_size=96)),
            (base_f, replace(base_m, hidden_size=256)),
            (base_f, replace(base_m, num_layers=1)),
            (base_f, replace(base_m, bidirectional=False)),
            (base_f, replace(base_m, pooling='last')),
            (base_f, replace(base_m, pooling='max')),
            (base_f, replace(base_m, dropout=0.1)),
            (base_f, replace(base_m, dropout=0.5)),
            (base_f, replace(base_m, optim=replace(base_o, learning_rate=3e-4))),
            (base_f, replace(base_m, optim=replace(base_o, batch_size=32))),
        ])
    if mode == 'full':
        candidates = []
        for n_intervals in (16, 24, 32):
            for interval_ms in (200, 300, 400):
                feature_cfg = IntervalConfig(n_intervals=n_intervals, interval_ms=interval_ms)
                for rnn_type in ('gru', 'lstm'):
                    for hidden in (96, 192):
                        candidates.append((
                            feature_cfg,
                            replace(RNN_BASE_MODEL, rnn_type=rnn_type, hidden_size=hidden),
                        ))
        return _unique(candidates)
    raise ValueError(f'Unknown grid mode {mode!r}; expected one of {GRID_MODES}.')


def rnn_refinement(winner: RNNCandidate, exclude=()) -> list[RNNCandidate]:
    '''Local second-stage search around the validation winner.'''

    feature_cfg, model_cfg = winner
    optim = model_cfg.optim
    other_type = 'lstm' if model_cfg.rnn_type == 'gru' else 'gru'
    candidates = _unique([
        (replace(feature_cfg, n_intervals=max(8, feature_cfg.n_intervals - 8)), model_cfg),
        (replace(feature_cfg, n_intervals=min(48, feature_cfg.n_intervals + 8)), model_cfg),
        (replace(feature_cfg, interval_ms=max(100, feature_cfg.interval_ms - 100)), model_cfg),
        (replace(feature_cfg, interval_ms=min(600, feature_cfg.interval_ms + 100)), model_cfg),
        (feature_cfg, replace(model_cfg, rnn_type=other_type)),
        (feature_cfg, replace(model_cfg, hidden_size=max(32, model_cfg.hidden_size // 2))),
        (feature_cfg, replace(model_cfg, hidden_size=min(512, model_cfg.hidden_size * 2))),
        (feature_cfg, replace(model_cfg, dropout=round(max(0.0, model_cfg.dropout - 0.1), 2))),
        (feature_cfg, replace(model_cfg, dropout=round(min(0.6, model_cfg.dropout + 0.1), 2))),
        (feature_cfg, replace(model_cfg, optim=replace(optim, learning_rate=optim.learning_rate / 2))),
    ])
    excluded = set(exclude) | {winner}
    return [c for c in candidates if c not in excluded]
