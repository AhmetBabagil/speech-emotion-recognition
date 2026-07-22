"""Tests for descriptive summaries of completed validation searches."""

import json
from pathlib import Path

import pandas as pd
import pytest

from odev3 import hyperparameter_effects as effects


def _validation_frame() -> pd.DataFrame:
    common = {
        'batch_size': 64,
        'patience': 8,
        'hidden_dims': '512-256',
        'activation': 'relu',
        'batch_norm': True,
        'dropout': 0.3,
        'weight_decay': 0.0001,
    }
    return pd.DataFrame(
        [
            {
                **common,
                'trial': 1,
                'search_stage': 'screening',
                'config_id': 'a',
                'learning_rate': 0.001,
                'val_macro_f1': 0.4,
            },
            {
                **common,
                'trial': 2,
                'search_stage': 'refinement',
                'config_id': 'b',
                'learning_rate': 0.001,
                'val_macro_f1': 0.5,
            },
            {
                **common,
                'trial': 3,
                'search_stage': 'refinement',
                'config_id': 'c',
                'learning_rate': 0.0001,
                'activation': 'gelu',
                'batch_norm': False,
                'val_macro_f1': 0.3,
            },
            {
                **common,
                'trial': 4,
                'search_stage': 'stability',
                'config_id': 'b',
                'learning_rate': 0.001,
                'val_macro_f1': 0.9,
            },
        ]
    )


def test_validation_search_rows_excludes_repeated_stability_seed() -> None:
    filtered = effects.validation_search_rows(_validation_frame())

    assert len(filtered) == 3
    assert set(filtered['search_stage']) == {'screening', 'refinement'}
    assert filtered['config_id'].is_unique


def test_parameter_effects_report_group_counts_and_descriptive_spread() -> None:
    rows = effects.parameter_effect_rows(_validation_frame())
    learning_rate = [row for row in rows if row['parameter'] == 'learning_rate']

    assert len(learning_rate) == 2
    best_group = next(row for row in learning_rate if row['value'] == '0.001')
    assert best_group['runs'] == 2
    assert best_group['val_macro_f1_mean'] == pytest.approx(0.45)
    assert best_group['val_macro_f1_std'] == pytest.approx(0.05)

    overview = effects.parameter_effect_overview(rows)
    learning_rate_overview = next(row for row in overview if row['parameter'] == 'learning_rate')
    assert learning_rate_overview['best_value'] == '0.001'
    assert learning_rate_overview['worst_value'] == '0.0001'
    assert learning_rate_overview['mean_macro_f1_spread'] == pytest.approx(0.15)
    batch_norm = next(row for row in overview if row['parameter'] == 'batch_norm')
    assert set(batch_norm['tested_values'].split(' | ')) == {'false', 'true'}


def test_validation_search_rows_rejects_missing_or_duplicate_configs() -> None:
    missing = _validation_frame().drop(columns=['dropout'])
    with pytest.raises(ValueError, match='missing columns'):
        effects.validation_search_rows(missing)

    duplicated = _validation_frame()
    duplicated.loc[2, 'config_id'] = 'a'
    with pytest.raises(ValueError, match='must be unique'):
        effects.validation_search_rows(duplicated)


def test_analysis_writes_per_dataset_csv_and_json_artifacts(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / 'outputs'
    for corpus in ('cremad', 'meld'):
        corpus_dir = output_root / corpus
        corpus_dir.mkdir(parents=True)
        _validation_frame().to_csv(corpus_dir / 'validation_results.csv', index=False)
    analysis_root = tmp_path / 'analysis'

    summary = effects.analyze_hyperparameter_effects(
        output_root=output_root,
        analysis_root=analysis_root,
    )

    assert summary['included_search_stages'] == ['screening', 'refinement']
    assert 'not a balanced full-factorial' in summary['important_limitation']
    for corpus in ('cremad', 'meld'):
        result = summary['per_dataset'][corpus]
        assert result['configuration_trials'] == 3
        assert result['excluded_stability_trials'] == 1
        assert len(result['overview']) == len(effects.PARAMETERS)
        assert (analysis_root / corpus / 'parameter_effects.csv').is_file()
        assert (analysis_root / corpus / 'parameter_effect_overview.csv').is_file()
    saved = json.loads((analysis_root / 'summary.json').read_text(encoding='utf-8'))
    assert saved == summary
