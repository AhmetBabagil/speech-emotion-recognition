'''Tests for validation-only Mel feature ablations.'''

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


def test_feature_candidates_cover_crop_resize_and_two_resolutions() -> None:
    candidates = ablation.feature_candidates()

    assert [(c.frame_strategy, c.n_frames) for c in candidates] == [
        ('crop_pad', 64),
        ('resize', 64),
        ('crop_pad', 96),
        ('resize', 96),
    ]
    assert all(candidate.vector_size >= 4000 for candidate in candidates)


def test_ablation_never_loads_test_features(tmp_path: Path, monkeypatch) -> None:
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

    assert len(rows) == 4
    assert len(loaded_descriptions) == 8
    assert all('test' not in description for description in loaded_descriptions)
    assert all(row['test_features_loaded'] in (False, np.False_) for row in rows)
    assert (tmp_path / 'outputs' / 'cremad' / 'feature_ablation.csv').is_file()
