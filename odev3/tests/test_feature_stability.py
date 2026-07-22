'''Tests for validation-only multi-seed Mel representation confirmation.'''

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from odev3 import feature_stability as stability
from ser.constants import CANONICAL_EMOTIONS


def _fold(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'path': f'{prefix}-{index}.wav',
                'speaker': f'{prefix}-speaker-{index}',
                'emotion': emotion,
                'label_idx': index,
            }
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        ]
    )


def _metrics(score: float) -> dict:
    return {
        'accuracy': score,
        'balanced_accuracy': score,
        'macro_f1': score,
        'weighted_f1': score,
        'per_class': {},
        'confusion_matrix': np.eye(6, dtype=int).tolist(),
    }


def _result_row(
    corpus: str,
    candidate: str,
    seed: int,
    macro_f1: float,
) -> dict:
    return {
        'corpus': corpus,
        'candidate': candidate,
        'feature_fingerprint': candidate,
        'frame_strategy': 'resize',
        'n_mels': 64,
        'n_frames': 64,
        'vector_size': 4096,
        'seed': seed,
        'val_macro_f1': macro_f1,
        'val_accuracy': macro_f1 + 0.01,
        'val_balanced_accuracy': macro_f1 + 0.005,
        'val_loss': 2.0 - macro_f1,
    }


def test_confirmation_candidates_cover_main_and_challenger() -> None:
    cremad = stability.confirmation_candidates('cremad')
    meld = stability.confirmation_candidates('meld')

    assert stability.CONFIRMATION_SEEDS == (42, 143, 244)
    assert [name for name, _ in cremad] == [
        'main_64x64_resize',
        'challenger_96x64_crop_pad',
    ]
    assert [name for name, _ in meld] == [
        'main_64x64_crop_pad',
        'challenger_80x64_resize',
    ]
    assert all(
        config.vector_size >= 4000
        for _, config in (*cremad, *meld)
    )
    with pytest.raises(ValueError, match='Unsupported corpus'):
        stability.confirmation_candidates('unknown')


def test_completed_keys_require_protocol_model_and_epoch_match() -> None:
    reference = stability.reference_model('cremad')
    changed_model = replace(reference, dropout=0.45)
    base = {
        'feature_fingerprint': 'feature-a',
        'seed': 42,
        'max_epochs': 60,
        'protocol_version': stability.PROTOCOL_VERSION,
        'model_config': json.dumps(reference.to_dict()),
    }
    rows = [
        base,
        {**base, 'feature_fingerprint': 'old', 'protocol_version': 0},
        {**base, 'feature_fingerprint': 'short', 'max_epochs': 30},
        {
            **base,
            'feature_fingerprint': 'changed-model',
            'model_config': json.dumps(changed_model.to_dict()),
        },
    ]

    completed = stability._completed_keys(rows, 60, reference)

    assert completed == {('feature-a', 42)}


def test_aggregate_stability_rows_ranks_mean_before_best_single_seed() -> None:
    rows = [
        _result_row('cremad', 'stable', 42, 0.48),
        _result_row('cremad', 'stable', 143, 0.47),
        _result_row('cremad', 'stable', 244, 0.46),
        _result_row('cremad', 'volatile', 42, 0.55),
        _result_row('cremad', 'volatile', 143, 0.40),
        _result_row('cremad', 'volatile', 244, 0.41),
    ]

    aggregates = stability.aggregate_stability_rows(rows)

    assert [row['candidate'] for row in aggregates] == ['stable', 'volatile']
    assert [row['rank'] for row in aggregates] == [1, 2]
    assert aggregates[0]['runs'] == 3
    assert aggregates[0]['seeds'] == '42,143,244'
    assert aggregates[0]['val_macro_f1_mean'] == pytest.approx(0.47)
    assert aggregates[0]['val_macro_f1_std'] < aggregates[1]['val_macro_f1_std']


def test_paired_candidate_comparison_counts_seed_level_wins() -> None:
    rows = [
        _result_row('cremad', 'main_feature', 42, 0.46),
        _result_row('cremad', 'main_feature', 143, 0.48),
        _result_row('cremad', 'main_feature', 244, 0.45),
        _result_row('cremad', 'challenger_feature', 42, 0.47),
        _result_row('cremad', 'challenger_feature', 143, 0.44),
        _result_row('cremad', 'challenger_feature', 244, 0.45),
    ]

    comparison = stability.paired_candidate_comparison(rows)

    assert comparison['main_wins'] == 1
    assert comparison['challenger_wins'] == 1
    assert comparison['ties'] == 1
    assert comparison['mean_main_minus_challenger'] == pytest.approx(0.01)
    assert [row['seed'] for row in comparison['paired_runs']] == [42, 143, 244]

    with pytest.raises(ValueError, match='identical seed sets'):
        stability.paired_candidate_comparison(rows[:-1])


def test_stability_run_never_loads_test_and_resumes_completed_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    loaded_descriptions: list[str] = []
    trained_seeds: list[int] = []
    monkeypatch.setattr(
        stability,
        '_splits_for',
        lambda corpus, manifest: (_fold('train'), _fold('validation'), _fold('test')),
    )

    def fake_load(records, cache_dir, config, **kwargs):
        loaded_descriptions.append(kwargs['description'])
        labels = records['label_idx'].to_numpy(dtype=np.int64)
        features = np.zeros((len(records), config.vector_size), dtype=np.float32)
        features[:, :6] = np.eye(6, dtype=np.float32)
        return features, labels

    def fake_train(*args, **kwargs):
        seed = int(kwargs['seed'])
        trained_seeds.append(seed)
        score = 0.2 + seed / 1000.0
        return SimpleNamespace(
            history=[{'epoch': 1, 'val_macro_f1': score}],
            best_epoch=1,
            epochs_trained=1,
            stopped_early=False,
            validation_loss=1.0 - score,
            validation_metrics=_metrics(score),
        )

    monkeypatch.setattr(stability, 'load_feature_matrix', fake_load)
    monkeypatch.setattr(stability, 'train_with_early_stopping', fake_train)
    kwargs = {
        'manifest_path': tmp_path / 'manifest.csv',
        'cache_root': tmp_path / 'cache',
        'output_root': tmp_path / 'outputs',
        'max_epochs': 2,
        'seeds': (1, 2, 3),
        'device_name': 'cpu',
    }

    result = stability.run_corpus_stability('cremad', **kwargs)

    assert trained_seeds == [1, 2, 3, 1, 2, 3]
    assert len(result['runs']) == 6
    assert len(result['aggregates']) == 2
    assert result['comparison']['ties'] == 3
    assert all('test' not in description for description in loaded_descriptions)
    assert all(row['test_features_loaded'] is False for row in result['runs'])
    assert all(
        row['class_weighting'] == 'inverse-frequency CrossEntropyLoss'
        for row in result['runs']
    )
    corpus_dir = tmp_path / 'outputs' / 'cremad'
    assert (corpus_dir / 'feature_stability_runs.csv').is_file()
    assert (corpus_dir / 'feature_stability.csv').is_file()
    comparison_path = corpus_dir / 'feature_stability_comparison.json'
    assert json.loads(comparison_path.read_text(encoding='utf-8')) == result[
        'comparison'
    ]
    plot_path = corpus_dir / 'feature_stability.png'
    assert plot_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert plot_path.stat().st_size > 1_000

    trained_seeds.clear()
    loaded_descriptions.clear()
    resumed = stability.run_corpus_stability('cremad', **kwargs)

    assert trained_seeds == []
    assert loaded_descriptions == []
    assert len(resumed['runs']) == 6


@pytest.mark.parametrize(
    'kwargs',
    [
        {'max_epochs': 0},
        {'seeds': ()},
        {'seeds': (42, 42)},
        {'seeds': (-1,)},
        {'feature_workers': 0},
        {'loader_workers': -1},
    ],
)
def test_stability_run_rejects_invalid_settings(
    tmp_path: Path,
    kwargs: dict,
) -> None:
    settings = {
        'manifest_path': tmp_path / 'manifest.csv',
        'cache_root': tmp_path / 'cache',
        'output_root': tmp_path / 'outputs',
        'max_epochs': 2,
        'device_name': 'cpu',
        **kwargs,
    }
    with pytest.raises(ValueError):
        stability.run_corpus_stability('cremad', **settings)
