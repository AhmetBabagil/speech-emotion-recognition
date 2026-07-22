'''Validation-only comparison of fixed-size Mel time representations.'''

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.features_melspec import MelSpecConfig
from odev3.model import MLPConfig
from odev3.pipeline import SEED, _splits_for
from odev3.training import inverse_frequency_weights, train_with_early_stopping
from ser.constants import NUM_CLASSES
from ser.utils import ensure_dir, get_device


REFERENCE_MODELS = {
    'cremad': MLPConfig(
        batch_size=64,
        learning_rate=3e-4,
        patience=12,
        hidden_dims=(768, 384),
        activation='relu',
        batch_norm=True,
        dropout=0.3,
        weight_decay=1e-4,
    ),
    'meld': MLPConfig(
        batch_size=64,
        learning_rate=3e-4,
        patience=8,
        hidden_dims=(512, 256),
        activation='relu',
        batch_norm=True,
        dropout=0.5,
        weight_decay=1e-4,
    ),
}


def feature_candidates() -> tuple[MelSpecConfig, ...]:
    '''Hold frequency resolution fixed while comparing time handling/resolution.'''

    return (
        MelSpecConfig(n_frames=64, frame_strategy='crop_pad'),
        MelSpecConfig(n_frames=64, frame_strategy='resize'),
        MelSpecConfig(n_frames=96, frame_strategy='crop_pad'),
        MelSpecConfig(n_frames=96, frame_strategy='resize'),
    )


def _completed_keys(rows: list[dict[str, Any]], max_epochs: int) -> set[str]:
    return {
        str(row['feature_fingerprint'])
        for row in rows
        if int(row['seed']) == SEED and int(row['max_epochs']) == max_epochs
    }


def _load_partial(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient='records')


def run_corpus_ablation(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    max_epochs: int,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> list[dict[str, Any]]:
    '''Train four matched candidates using only train and validation folds.'''

    if corpus not in REFERENCE_MODELS:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')

    device = get_device(device_name)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    partial_path = corpus_dir / 'feature_ablation.partial.csv'
    rows = _load_partial(partial_path)
    completed = _completed_keys(rows, max_epochs)
    train_frame, validation_frame, _held_out_test_frame = _splits_for(
        corpus,
        manifest_path,
    )
    model_config = REFERENCE_MODELS[corpus]

    for trial, feature_config in enumerate(feature_candidates(), start=1):
        if feature_config.fingerprint in completed:
            print(
                f'{corpus}: feature trial {trial}/4 already complete '
                f'({feature_config.frame_strategy}, {feature_config.n_frames} frames)'
            )
            continue

        print(
            f'{corpus}: feature trial {trial}/4 '
            f'({feature_config.frame_strategy}, {feature_config.n_frames} frames)'
        )
        started = time.perf_counter()
        cache_dir = Path(cache_root) / corpus
        train_features, train_labels = load_feature_matrix(
            train_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} ablation train {trial}',
        )
        validation_features, validation_labels = load_feature_matrix(
            validation_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} ablation validation {trial}',
        )
        standardizer = FeatureStandardizer.fit(train_features)
        train_features = standardizer.transform(train_features)
        validation_features = standardizer.transform(validation_features)
        class_weights = inverse_frequency_weights(train_labels, NUM_CLASSES)
        outcome = train_with_early_stopping(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            model_config,
            input_dim=feature_config.vector_size,
            num_classes=NUM_CLASSES,
            device=device,
            max_epochs=max_epochs,
            seed=SEED,
            num_workers=loader_workers,
            amp=device.type == 'cuda',
        )
        elapsed = time.perf_counter() - started
        metrics = outcome.validation_metrics
        row = {
            'corpus': corpus,
            'trial': trial,
            'seed': SEED,
            'max_epochs': max_epochs,
            'feature_fingerprint': feature_config.fingerprint,
            'frame_strategy': feature_config.frame_strategy,
            'n_mels': feature_config.n_mels,
            'n_frames': feature_config.n_frames,
            'vector_size': feature_config.vector_size,
            'model_config': json.dumps(model_config.to_dict(), sort_keys=True),
            'best_epoch': outcome.best_epoch,
            'epochs_trained': outcome.epochs_trained,
            'stopped_early': outcome.stopped_early,
            'val_loss': outcome.validation_loss,
            'val_accuracy': metrics['accuracy'],
            'val_balanced_accuracy': metrics['balanced_accuracy'],
            'val_macro_f1': metrics['macro_f1'],
            'val_weighted_f1': metrics['weighted_f1'],
            'elapsed_seconds': elapsed,
            'test_features_loaded': False,
        }
        rows.append(row)
        completed.add(feature_config.fingerprint)
        pd.DataFrame(outcome.history).to_csv(
            history_dir / f'trial_{trial:02d}_{feature_config.fingerprint}.csv',
            index=False,
        )
        pd.DataFrame(rows).to_csv(partial_path, index=False)
        print(
            f'{corpus}: trial {trial} validation macro-F1='
            f'{metrics["macro_f1"]:.4f}'
        )

        del train_features, validation_features, outcome

    ranked = pd.DataFrame(rows).sort_values(
        ['val_macro_f1', 'val_balanced_accuracy', 'val_loss'],
        ascending=[False, False, True],
    )
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked.to_csv(corpus_dir / 'feature_ablation.csv', index=False)
    return ranked.to_dict(orient='records')


def run_feature_ablation(
    *,
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/feature_ablation',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    results = {
        corpus: run_corpus_ablation(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            max_epochs=max_epochs,
            device_name=device_name,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
        )
        for corpus in corpora
    }
    summary = {
        'protocol': 'validation only; held-out test audio is not loaded',
        'seed': SEED,
        'max_epochs': max_epochs,
        'reference_models': {
            corpus: config.to_dict() for corpus, config in REFERENCE_MODELS.items()
        },
        'feature_candidates': [asdict(config) for config in feature_candidates()],
        'results': results,
    }
    output_root = ensure_dir(output_root)
    (Path(output_root) / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--corpora',
        nargs='+',
        default=['cremad', 'meld'],
        choices=sorted(REFERENCE_MODELS),
    )
    parser.add_argument('--manifest', default='odev1/manifest_subset.csv')
    parser.add_argument('--cache-root', default='data/cache/odev3_melspec')
    parser.add_argument('--out-root', default='odev3/feature_ablation')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=0)
    args = parser.parse_args()
    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('Worker counts must be feature>=1 and loader>=0.')
    run_feature_ablation(
        corpora=tuple(args.corpora),
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        output_root=args.out_root,
        max_epochs=args.max_epochs,
        device_name=args.device,
        feature_workers=args.feature_workers,
        loader_workers=args.loader_workers,
    )


if __name__ == '__main__':
    main()
