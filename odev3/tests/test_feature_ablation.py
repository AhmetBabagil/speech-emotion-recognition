'''Tests for validation-only Mel feature ablations.'''

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from odev3 import feature_ablation as ablation
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


def test_feature_candidates_cover_time_and_frequency_resolutions() -> None:
    candidates = ablation.feature_candidates()

    assert [(c.n_mels, c.n_frames, c.frame_strategy) for c in candidates] == [
        (64, 64, 'crop_pad'),
        (64, 64, 'resize'),
        (64, 96, 'crop_pad'),
        (64, 96, 'resize'),
        (80, 64, 'crop_pad'),
        (80, 64, 'resize'),
        (96, 64, 'crop_pad'),
        (96, 64, 'resize'),
    ]
    assert all(candidate.vector_size >= 4000 for candidate in candidates)
    assert len({candidate.fingerprint for candidate in candidates}) == len(candidates)


def test_completed_keys_require_matching_model_seed_and_epoch_budget() -> None:
    reference = ablation.REFERENCE_MODELS['cremad']
    changed_model = replace(reference, dropout=0.45)
    rows = [
        {
            'feature_fingerprint': 'matching',
            'seed': ablation.SEED,
            'max_epochs': 60,
            'model_config': json.dumps(reference.to_dict(), indent=2),
        },
        {
            'feature_fingerprint': 'changed-model',
            'seed': ablation.SEED,
            'max_epochs': 60,
            'model_config': json.dumps(changed_model.to_dict()),
        },
        {
            'feature_fingerprint': 'changed-seed',
            'seed': ablation.SEED + 1,
            'max_epochs': 60,
            'model_config': json.dumps(reference.to_dict()),
        },
        {
            'feature_fingerprint': 'changed-budget',
            'seed': ablation.SEED,
            'max_epochs': 30,
            'model_config': json.dumps(reference.to_dict()),
        },
        {
            'feature_fingerprint': 'malformed',
            'seed': ablation.SEED,
            'max_epochs': 60,
            'model_config': '{invalid json',
        },
    ]

    completed = ablation._completed_keys(rows, 60, reference)

    assert completed == {'matching'}


def test_ablation_never_loads_test_features(
    tmp_path: Path,
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded_descriptions: list[str] = []
    monkeypatch.setattr(
        ablation,
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
        return SimpleNamespace(
            history=[{'epoch': 1, 'val_macro_f1': 0.25}],
            best_epoch=1,
            epochs_trained=1,
            stopped_early=False,
            validation_loss=1.0,
            validation_metrics=_metrics(0.25),
        )

    monkeypatch.setattr(ablation, 'load_feature_matrix', fake_load)
    monkeypatch.setattr(ablation, 'train_with_early_stopping', fake_train)

    rows = ablation.run_corpus_ablation(
        'cremad',
        manifest_path=tmp_path / 'manifest.csv',
        cache_root=tmp_path / 'cache',
        output_root=tmp_path / 'outputs',
        max_epochs=2,
        device_name='cpu',
    )

    candidate_count = len(ablation.feature_candidates())
    assert len(rows) == candidate_count
    assert len(loaded_descriptions) == candidate_count * 2
    assert all('test' not in description for description in loaded_descriptions)
    assert all(row['test_features_loaded'] in (False, np.False_) for row in rows)
    assert (tmp_path / 'outputs' / 'cremad' / 'feature_ablation.csv').is_file()
    output = capsys.readouterr().out
    assert 'trial 5/8 (80 mels, 64 frames, crop_pad)' in output
    assert 'trial 8/8 (96 mels, 64 frames, resize)' in output
