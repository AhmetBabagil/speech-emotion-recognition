'''Tests for split integrity and Assignment 3 pipeline artifacts.'''

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from odev3 import pipeline as pipeline_module
from odev3.features_melspec import MelSpecConfig
from odev3.model import MLP, MLPConfig
from odev3.pipeline import _split_summary, _stratified_limit
from ser.constants import CANONICAL_EMOTIONS


def _fold(speakers: list[str]) -> pd.DataFrame:
    rows = []
    for index, speaker in enumerate(speakers):
        rows.append(
            {
                'path': f'{speaker}.wav',
                'speaker': speaker,
                'emotion': f'emotion-{index}',
                'label_idx': index,
            }
        )
    return pd.DataFrame(rows)


def test_split_summary_reports_disjoint_speakers() -> None:
    summary = _split_summary(
        _fold(['train-a', 'train-b']),
        _fold(['validation-a']),
        _fold(['test-a']),
    )

    assert summary['train']['records'] == 2
    assert summary['train']['speakers'] == 2
    assert summary['speaker_overlap'] == {
        'train_validation': [],
        'train_test': [],
        'validation_test': [],
    }


def test_split_summary_rejects_speaker_leakage() -> None:
    with pytest.raises(ValueError, match='Speaker leakage'):
        _split_summary(
            _fold(['shared']),
            _fold(['shared']),
            _fold(['test-only']),
        )


def test_diagnostic_limit_is_stratified_and_deterministic() -> None:
    frame = pd.DataFrame(
        {
            'row_id': range(60),
            'label_idx': [label for label in range(6) for _ in range(10)],
        }
    )

    first = _stratified_limit(frame, limit=18, seed=42)
    second = _stratified_limit(frame, limit=18, seed=42)

    assert len(first) == 18
    assert set(first['label_idx']) == set(range(6))
    assert first['row_id'].tolist() == second['row_id'].tolist()


def _six_class_fold(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                'path': f'{prefix}-{label}.wav',
                'speaker': f'{prefix}-speaker-{label}',
                'emotion': emotion,
                'label_idx': label,
            }
            for label, emotion in enumerate(CANONICAL_EMOTIONS)
        ]
    )


def _metrics(score: float) -> dict:
    return {
        'accuracy': score,
        'balanced_accuracy': score,
        'macro_f1': score,
        'weighted_f1': score,
        'per_class': {
            emotion: {
                'precision': score,
                'recall': score,
                'f1': score,
                'support': 1,
            }
            for emotion in CANONICAL_EMOTIONS
        },
        'confusion_matrix': np.eye(6, dtype=int).tolist(),
    }


def test_test_fold_is_untouched_until_validation_search_finishes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    events: list[str] = []
    feature_config = MelSpecConfig(n_mels=2, n_frames=3)
    candidates = [
        MLPConfig(hidden_dims=(4,), batch_norm=False, dropout=0.0),
        MLPConfig(hidden_dims=(8,), batch_norm=False, dropout=0.0),
    ]

    monkeypatch.setattr(
        pipeline_module,
        '_splits_for',
        lambda corpus, manifest: (
            _six_class_fold('train'),
            _six_class_fold('validation'),
            _six_class_fold('test'),
        ),
    )
    monkeypatch.setattr(pipeline_module, 'search_space', lambda mode: candidates)

    def fake_load(records, cache_dir, config, **kwargs):
        description = kwargs['description']
        events.append(f'load:{description}')
        labels = records['label_idx'].to_numpy(dtype=np.int64)
        features = np.eye(config.vector_size, dtype=np.float32)[labels]
        return features, labels

    def fake_train(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        config,
        **kwargs,
    ):
        events.append('train')
        assert not any(event.endswith('test') for event in events)
        score = 0.4 if config.hidden_dims == (4,) else 0.6
        history = [
            {
                'epoch': 1,
                'train_loss': 1.0,
                'train_accuracy': score,
                'train_balanced_accuracy': score,
                'train_macro_f1': score,
                'train_weighted_f1': score,
                'val_loss': 1.0 - score,
                'val_accuracy': score,
                'val_balanced_accuracy': score,
                'val_macro_f1': score,
                'val_weighted_f1': score,
                'improved': True,
            }
        ]
        return SimpleNamespace(
            model=MLP(config= config, input_dim=6, num_classes=6),
            history=history,
            best_epoch=1,
            epochs_trained=1,
            stopped_early=False,
            validation_loss=1.0 - score,
            validation_metrics=_metrics(score),
        )

    def fake_evaluate(model, features, labels, **kwargs):
        events.append('evaluate:test')
        return 0.2, _metrics(0.8), np.eye(6, dtype=np.float32)[labels]

    monkeypatch.setattr(pipeline_module, 'load_feature_matrix', fake_load)
    monkeypatch.setattr(pipeline_module, 'train_with_early_stopping', fake_train)
    monkeypatch.setattr(pipeline_module, 'evaluate_arrays', fake_evaluate)
    monkeypatch.setattr(pipeline_module, '_plot_history', lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline_module, '_plot_confusion', lambda *args, **kwargs: None)

    result = pipeline_module.run_corpus(
        'cremad',
        manifest_path=tmp_path / 'manifest.csv',
        cache_root=tmp_path / 'cache',
        output_root=tmp_path / 'outputs',
        grid_mode='quick',
        max_epochs=2,
        device=torch.device('cpu'),
        feature_config=feature_config,
    )

    assert events == [
        'load:cremad eğitim',
        'load:cremad geçerleme',
        'train',
        'train',
        'load:cremad test',
        'evaluate:test',
    ]
    assert result['best_trial'] == 2
    assert result['test']['macro_f1'] == 0.8
    checkpoint_path = tmp_path / 'outputs' / 'cremad' / 'best_model.pt'
    assert checkpoint_path.is_file()

    loaded_model, standardizer, checkpoint = pipeline_module.load_saved_model(
        checkpoint_path
    )
    logits = loaded_model(torch.zeros(2, 6))

    assert logits.shape == (2, 6)
    assert standardizer.mean.shape == (6,)
    assert checkpoint['corpus'] == 'cremad'


def test_model_comparison_preserves_prior_rows_and_adds_mlp(tmp_path: Path) -> None:
    columns = [
        'corpus',
        'model',
        'feature_dim',
        'pca_dim',
        'params',
        'test_accuracy',
        'test_balanced_accuracy',
        'test_macro_f1',
        'test_weighted_f1',
    ]
    prior_path = tmp_path / 'prior.csv'
    pd.DataFrame(
        [
            [
                'cremad',
                'KNN (Ödev 1)',
                2304,
                256,
                '{"K": 15}',
                0.48,
                0.47,
                0.46,
                0.45,
            ]
        ],
        columns=columns,
    ).to_csv(prior_path, index=False)
    results = {
        'cremad': {
            'test': _metrics(0.62),
            'feature': {'vector_size': 4096},
            'best_config': MLPConfig().to_dict(),
        }
    }

    comparison = pipeline_module._write_model_comparison(
        results,
        prior_path,
        tmp_path / 'outputs',
    )

    assert comparison['model'].tolist() == ['MLP (Ödev 3)', 'KNN (Ödev 1)']
    assert comparison.iloc[0]['test_macro_f1'] == 0.62
    assert (tmp_path / 'outputs' / 'model_comparison.csv').is_file()


def test_training_history_plot_is_written(tmp_path: Path) -> None:
    history = [
        {
            'epoch': 1,
            'train_loss': 1.2,
            'val_loss': 1.4,
            'train_macro_f1': 0.35,
            'val_macro_f1': 0.30,
        },
        {
            'epoch': 2,
            'train_loss': 0.9,
            'val_loss': 1.1,
            'train_macro_f1': 0.55,
            'val_macro_f1': 0.48,
        },
    ]
    output_path = tmp_path / 'history.png'

    pipeline_module._plot_history(history, output_path, 'Test history')

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_confusion_matrix_plot_is_written(tmp_path: Path) -> None:
    matrix = np.eye(6, dtype=int).tolist()
    output_path = tmp_path / 'confusion.png'

    pipeline_module._plot_confusion(matrix, output_path, 'Test confusion')

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
