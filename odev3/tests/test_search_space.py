'''Tests that the validation search covers the assignment requirements.'''

import pytest

from odev3.model import MLPConfig
from odev3.search_space import refinement_space, search_space


def test_report_search_varies_every_required_hyperparameter() -> None:
    candidates = search_space('report')

    varied_values = {
        'batch_size': {config.batch_size for config in candidates},
        'learning_rate': {config.learning_rate for config in candidates},
        'patience': {config.patience for config in candidates},
        'hidden_dims': {config.hidden_dims for config in candidates},
        'activation': {config.activation for config in candidates},
        'batch_norm': {config.batch_norm for config in candidates},
        'dropout': {config.dropout for config in candidates},
    }

    assert all(len(values) >= 2 for values in varied_values.values())
    assert {len(config.hidden_dims) for config in candidates} >= {1, 2, 3}


@pytest.mark.parametrize('mode', ['quick', 'report', 'full'])
def test_search_candidates_are_valid_and_unique(mode: str) -> None:
    candidates = search_space(mode)
    serialized = [str(config.to_dict()) for config in candidates]

    assert candidates
    assert len(serialized) == len(set(serialized))
    for config in candidates:
        config.validate()


def test_search_mode_sizes_make_runtime_explicit() -> None:
    assert len(search_space('quick')) == 2
    assert len(search_space('report')) == 16
    assert len(search_space('full')) == 128


def test_refinement_builds_new_valid_neighbors_around_validation_winner() -> None:
    winner = MLPConfig(
        batch_size=32,
        learning_rate=1e-3,
        patience=5,
        hidden_dims=(512, 256, 128),
        activation='gelu',
        batch_norm=True,
        dropout=0.3,
        weight_decay=1e-4,
    )
    screening = search_space('report')

    refinements = refinement_space(winner, exclude=screening)

    assert 8 <= len(refinements) <= 14
    assert len(refinements) == len(set(refinements))
    assert not set(refinements) & set(screening)
    assert any(config.learning_rate < winner.learning_rate for config in refinements)
    assert any(config.learning_rate > winner.learning_rate for config in refinements)
    assert any(config.hidden_dims != winner.hidden_dims for config in refinements)
    assert any(config.dropout != winner.dropout for config in refinements)
    for config in refinements:
        config.validate()
