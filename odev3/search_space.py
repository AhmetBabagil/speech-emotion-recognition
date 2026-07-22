'''Deterministic hyperparameter candidates for quick, report and full runs.'''

from __future__ import annotations

from dataclasses import replace
from itertools import product

from odev3.model import MLPConfig


GRID_MODES = ('quick', 'report', 'full')


def _unique(configs: list[MLPConfig]) -> list[MLPConfig]:
    result: list[MLPConfig] = []
    seen: set[tuple] = set()
    for config in configs:
        config.validate()
        key = (
            config.batch_size,
            config.learning_rate,
            config.patience,
            config.hidden_dims,
            config.activation,
            config.batch_norm,
            config.dropout,
            config.weight_decay,
        )
        if key not in seen:
            seen.add(key)
            result.append(config)
    return result


def _report_candidates() -> list[MLPConfig]:
    '''One-factor controls around a strong baseline for interpretable effects.'''

    base = MLPConfig()
    return _unique(
        [
            base,
            replace(base, batch_size=32),
            replace(base, batch_size=128),
            replace(base, learning_rate=1e-3),
            replace(base, learning_rate=1e-4),
            replace(base, patience=5),
            replace(base, patience=12),
            replace(base, hidden_dims=(256,)),
            replace(base, hidden_dims=(512, 256, 128)),
            replace(base, hidden_dims=(1024, 512, 256)),
            replace(base, activation='gelu'),
            replace(base, activation='tanh'),
            replace(base, batch_norm=False),
            replace(base, dropout=0.0),
            replace(base, dropout=0.5),
            replace(base, weight_decay=0.0),
        ]
    )


def _full_candidates() -> list[MLPConfig]:
    configs = []
    for values in product(
        (32, 64),
        (1e-3, 3e-4),
        (6, 10),
        ((256,), (512, 256)),
        ('relu', 'gelu'),
        (False, True),
        (0.2, 0.5),
    ):
        configs.append(
            MLPConfig(
                batch_size=values[0],
                learning_rate=values[1],
                patience=values[2],
                hidden_dims=values[3],
                activation=values[4],
                batch_norm=values[5],
                dropout=values[6],
                weight_decay=1e-4,
            )
        )
    return _unique(configs)


def _scaled_hidden_dims(hidden_dims: tuple[int, ...], factor: float) -> tuple[int, ...]:
    '''Scale widths while retaining practical multiples of 32.'''

    return tuple(
        max(32, int(round((size * factor) / 32.0)) * 32)
        for size in hidden_dims
    )


def refinement_space(
    winner: MLPConfig,
    *,
    exclude: list[MLPConfig] | tuple[MLPConfig, ...] = (),
) -> list[MLPConfig]:
    '''Build a local second-stage search around the validation winner.'''

    winner.validate()
    alternative_activations = [
        name for name in ('relu', 'gelu', 'tanh') if name != winner.activation
    ]
    candidates = _unique(
        [
            replace(winner, learning_rate=max(1e-5, winner.learning_rate / 2.0)),
            replace(winner, learning_rate=min(5e-3, winner.learning_rate * 2.0)),
            replace(winner, batch_size=max(16, winner.batch_size // 2)),
            replace(winner, batch_size=min(256, winner.batch_size * 2)),
            replace(winner, patience=max(3, winner.patience - 2)),
            replace(winner, patience=min(20, winner.patience + 2)),
            replace(winner, dropout=round(max(0.0, winner.dropout - 0.1), 2)),
            replace(winner, dropout=round(min(0.7, winner.dropout + 0.1), 2)),
            replace(
                winner,
                hidden_dims=_scaled_hidden_dims(winner.hidden_dims, 0.5),
            ),
            replace(
                winner,
                hidden_dims=_scaled_hidden_dims(winner.hidden_dims, 1.5),
            ),
            replace(winner, batch_norm=not winner.batch_norm),
            replace(winner, activation=alternative_activations[0]),
            replace(winner, activation=alternative_activations[1]),
            replace(
                winner,
                weight_decay=0.0 if winner.weight_decay > 0.0 else 1e-4,
            ),
        ]
    )
    excluded = set(exclude)
    return [config for config in candidates if config != winner and config not in excluded]


def search_space(mode: str) -> list[MLPConfig]:
    '''Return candidates without touching validation or test data.'''

    if mode == 'quick':
        return [
            MLPConfig(
                batch_size=64,
                learning_rate=1e-3,
                patience=2,
                hidden_dims=(64,),
                activation='relu',
                batch_norm=False,
                dropout=0.0,
                weight_decay=0.0,
            ),
            MLPConfig(
                batch_size=64,
                learning_rate=3e-4,
                patience=3,
                hidden_dims=(128, 64),
                activation='gelu',
                batch_norm=True,
                dropout=0.3,
                weight_decay=1e-4,
            ),
        ]
    if mode == 'report':
        return _report_candidates()
    if mode == 'full':
        return _full_candidates()
    raise ValueError(f'Unknown grid mode {mode!r}; expected one of {GRID_MODES}.')
