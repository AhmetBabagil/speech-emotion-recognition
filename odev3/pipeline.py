'''End-to-end validation search and held-out test evaluation for Assignment 3.'''

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.features_melspec import DEFAULT_CONFIG, MelSpecConfig
from odev3.model import MLP, MLPConfig, count_parameters
from odev3.search_space import search_space
from odev3.training import (
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.config import Config
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES
from ser.data import prepare_splits
from ser.utils import ensure_dir, get_device, get_logger, set_seed


log = get_logger('odev3.mlp')
SEED = 42


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def _write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding='utf-8',
    )


def _splits_for(corpus: str, manifest_path: str | Path):
    '''Reuse the same seed-42 speaker-independent folds as Assignments 1 and 2.'''

    manifest = pd.read_csv(manifest_path)
    cfg = Config()
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = 'speaker'
    cfg.data.val_fraction = 0.15
    cfg.data.test_fraction = 0.15
    return prepare_splits(manifest, cfg.data, seed=SEED)


def _stratified_limit(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    '''Deterministically shrink a fold for diagnostics while retaining all classes.'''

    if limit is None or limit >= len(frame):
        return frame.copy()
    if limit < NUM_CLASSES:
        raise ValueError(f'Fold limit must be at least {NUM_CLASSES}, got {limit}.')

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    per_class = max(1, limit // NUM_CLASSES)
    for label in range(NUM_CLASSES):
        candidates = frame.index[frame['label_idx'] == label].to_numpy()
        if len(candidates) == 0:
            raise ValueError(f'Fold has no examples for label {label}.')
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())

    remaining = limit - len(selected)
    if remaining > 0:
        pool = frame.index[~frame.index.isin(selected)].to_numpy()
        take = min(remaining, len(pool))
        selected.extend(rng.choice(pool, size=take, replace=False).tolist())
    rng.shuffle(selected)
    return frame.loc[selected].reset_index(drop=True)


def _fold_details(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame['label_idx'].value_counts().to_dict()
    return {
        'records': int(len(frame)),
        'speakers': int(frame['speaker'].astype(str).nunique()),
        'class_counts': {
            emotion: int(counts.get(index, 0))
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
    }


def _split_summary(train, validation, test) -> dict[str, Any]:
    speaker_sets = {
        'train': set(train['speaker'].astype(str)),
        'validation': set(validation['speaker'].astype(str)),
        'test': set(test['speaker'].astype(str)),
    }
    overlaps = {
        'train_validation': sorted(speaker_sets['train'] & speaker_sets['validation']),
        'train_test': sorted(speaker_sets['train'] & speaker_sets['test']),
        'validation_test': sorted(speaker_sets['validation'] & speaker_sets['test']),
    }
    if any(overlaps.values()):
        raise ValueError(f'Speaker leakage detected: {overlaps}.')
    return {
        'protocol': 'speaker-independent 70/15/15, seed=42',
        'train': _fold_details(train),
        'validation': _fold_details(validation),
        'test': _fold_details(test),
        'speaker_overlap': overlaps,
    }


def _plot_confusion(matrix: list[list[int]], path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    displayed = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix),
        where=row_sums != 0,
    )
    fig, axis = plt.subplots(figsize=(7.0, 5.8))
    image = axis.imshow(displayed, cmap='Blues', vmin=0.0, vmax=1.0)
    axis.set_xticks(range(NUM_CLASSES), CANONICAL_EMOTIONS, rotation=45, ha='right')
    axis.set_yticks(range(NUM_CLASSES), CANONICAL_EMOTIONS)
    axis.set_xlabel('Tahmin edilen sınıf')
    axis.set_ylabel('Gerçek sınıf')
    axis.set_title(title)
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            value = displayed[row, column]
            axis.text(
                column,
                row,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color='white' if value > 0.5 else 'black',
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_history(history: list[dict[str, Any]], path: str | Path, title: str) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].plot(epochs, [row['train_loss'] for row in history], label='Eğitim')
    axes[0].plot(epochs, [row['val_loss'] for row in history], label='Geçerleme')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Ağırlıklı çapraz entropi')
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row['train_macro_f1'] for row in history],
        label='Eğitim',
    )
    axes[1].plot(
        epochs,
        [row['val_macro_f1'] for row in history],
        label='Geçerleme',
    )
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Macro-F1')
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _validation_row(
    trial_id: int,
    config: MLPConfig,
    outcome,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = outcome.validation_metrics
    return {
        'trial': trial_id,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'patience': config.patience,
        'hidden_dims': '-'.join(str(value) for value in config.hidden_dims),
        'hidden_layers': len(config.hidden_dims),
        'activation': config.activation,
        'batch_norm': config.batch_norm,
        'dropout': config.dropout,
        'weight_decay': config.weight_decay,
        'parameters': count_parameters(outcome.model),
        'best_epoch': outcome.best_epoch,
        'epochs_trained': outcome.epochs_trained,
        'stopped_early': outcome.stopped_early,
        'val_loss': outcome.validation_loss,
        'val_accuracy': metrics['accuracy'],
        'val_balanced_accuracy': metrics['balanced_accuracy'],
        'val_macro_f1': metrics['macro_f1'],
        'val_weighted_f1': metrics['weighted_f1'],
        'elapsed_seconds': elapsed_seconds,
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row['val_macro_f1']),
        float(row['val_balanced_accuracy']),
        -float(row['val_loss']),
    )


def run_corpus(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    grid_mode: str,
    max_epochs: int,
    device: torch.device,
    feature_config: MelSpecConfig = DEFAULT_CONFIG,
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    '''Search on validation data, then evaluate the selected model once on test.'''

    set_seed(SEED)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    cache_dir = Path(cache_root) / corpus
    train_frame, validation_frame, test_frame = _splits_for(corpus, manifest_path)
    train_frame = _stratified_limit(train_frame, limit_per_split, SEED)
    validation_frame = _stratified_limit(validation_frame, limit_per_split, SEED + 1)
    test_frame = _stratified_limit(test_frame, limit_per_split, SEED + 2)
    split_summary = _split_summary(train_frame, validation_frame, test_frame)
    _write_json(corpus_dir / 'split_summary.json', split_summary)
    log.info(
        '[%s] train=%d validation=%d test=%d | mode=%s',
        corpus,
        len(train_frame),
        len(validation_frame),
        len(test_frame),
        grid_mode,
    )

    # Test features and labels are intentionally not loaded until the complete
    # validation search has selected one candidate.
    train_features, train_labels = load_feature_matrix(
        train_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} eğitim',
    )
    validation_features, validation_labels = load_feature_matrix(
        validation_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} geçerleme',
    )
    standardizer = FeatureStandardizer.fit(train_features)
    train_features = standardizer.transform(train_features)
    validation_features = standardizer.transform(validation_features)
    class_weights = inverse_frequency_weights(train_labels, NUM_CLASSES)

    candidates = search_space(grid_mode)
    rows: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    for trial_id, candidate in enumerate(candidates, start=1):
        log.info(
            '[%s] trial %d/%d | %s',
            corpus,
            trial_id,
            len(candidates),
            candidate.to_dict(),
        )
        started = time.perf_counter()
        outcome = train_with_early_stopping(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            candidate,
            input_dim=feature_config.vector_size,
            num_classes=NUM_CLASSES,
            device=device,
            max_epochs=max_epochs,
            seed=SEED,
            num_workers=loader_workers,
            amp=amp,
        )
        elapsed = time.perf_counter() - started
        row = _validation_row(trial_id, candidate, outcome, elapsed)
        rows.append(row)
        pd.DataFrame(outcome.history).to_csv(
            history_dir / f'trial_{trial_id:03d}.csv', index=False
        )
        pd.DataFrame(rows).to_csv(corpus_dir / 'validation_results.partial.csv', index=False)
        log.info(
            '[%s] trial %d | best epoch=%d, val macro-F1=%.4f',
            corpus,
            trial_id,
            outcome.best_epoch,
            row['val_macro_f1'],
        )

        if best_bundle is None or _selection_key(row) > _selection_key(best_bundle['row']):
            best_bundle = {
                'row': row.copy(),
                'config': candidate,
                'history': list(outcome.history),
                'validation_metrics': outcome.validation_metrics,
                'state_dict': {
                    name: value.detach().cpu().clone()
                    for name, value in outcome.model.state_dict().items()
                },
            }
        del outcome
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if best_bundle is None:
        raise RuntimeError(f'No successful validation trial for {corpus}.')

    ranked = pd.DataFrame(rows).sort_values(
        ['val_macro_f1', 'val_balanced_accuracy', 'val_loss'],
        ascending=[False, False, True],
    )
    ranked.insert(0, 'rank', np.arange(1, len(ranked) + 1))
    ranked['selected'] = ranked['trial'] == int(best_bundle['row']['trial'])
    ranked.to_csv(corpus_dir / 'validation_results.csv', index=False)

    best_config: MLPConfig = best_bundle['config']
    model = MLP(feature_config.vector_size, NUM_CLASSES, best_config)
    model.load_state_dict(best_bundle['state_dict'])
    model.to(device)

    # This is the first point at which held-out test records are loaded or used.
    test_features, test_labels = load_feature_matrix(
        test_frame,
        cache_dir,
        feature_config,
        workers=feature_workers,
        description=f'{corpus} test',
    )
    test_features = standardizer.transform(test_features)
    test_loss, test_metrics, probabilities = evaluate_arrays(
        model,
        test_features,
        test_labels,
        class_weights=class_weights,
        device=device,
        batch_size=max(best_config.batch_size, 256),
        num_workers=loader_workers,
    )
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    prediction_frame = test_frame[
        ['path', 'speaker', 'emotion', 'label_idx']
    ].reset_index(drop=True).copy()
    prediction_frame['predicted_idx'] = predictions
    prediction_frame['predicted_emotion'] = [
        CANONICAL_EMOTIONS[int(value)] for value in predictions
    ]
    prediction_frame['confidence'] = confidences
    for index, emotion in enumerate(CANONICAL_EMOTIONS):
        prediction_frame[f'prob_{emotion}'] = probabilities[:, index]
    prediction_frame.to_csv(corpus_dir / 'test_predictions.csv', index=False)

    checkpoint = {
        'schema_version': 1,
        'assignment': 3,
        'corpus': corpus,
        'model_type': 'PyTorch MLP from scratch',
        'input_dim': feature_config.vector_size,
        'num_classes': NUM_CLASSES,
        'class_names': list(CANONICAL_EMOTIONS),
        'feature_config': asdict(feature_config),
        'model_config': best_config.to_dict(),
        'model_state_dict': best_bundle['state_dict'],
        'standardizer_mean': torch.from_numpy(standardizer.mean.copy()),
        'standardizer_scale': torch.from_numpy(standardizer.scale.copy()),
        'class_weights': class_weights.clone(),
        'seed': SEED,
        'selection_metric': 'validation macro-F1',
        'best_epoch': int(best_bundle['row']['best_epoch']),
    }
    torch.save(checkpoint, corpus_dir / 'best_model.pt')

    history_path = corpus_dir / 'best_training_history.csv'
    pd.DataFrame(best_bundle['history']).to_csv(history_path, index=False)
    _plot_history(
        best_bundle['history'],
        corpus_dir / 'best_training_history.png',
        f'{corpus.upper()} - en iyi MLP eğitim süreci',
    )
    _plot_confusion(
        test_metrics['confusion_matrix'],
        corpus_dir / 'test_confusion_matrix.png',
        f'{corpus.upper()} - MLP test karmaşıklık matrisi',
    )

    result = {
        'corpus': corpus,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'diagnostic_limit_per_split': limit_per_split,
        'seed': SEED,
        'selection_metric': 'validation macro-F1',
        'num_trials': len(candidates),
        'feature': {
            **asdict(feature_config),
            'vector_size': feature_config.vector_size,
            'method': 'librosa.feature.melspectrogram -> log dB -> 64x64 -> flatten',
            'pca': False,
        },
        'preprocessing': {
            'normalization': 'per-dimension z-score',
            'normalizer_fit': 'training fold only',
        },
        'split': split_summary,
        'training_class_weights': {
            emotion: float(class_weights[index])
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
        'best_trial': int(best_bundle['row']['trial']),
        'best_config': best_config.to_dict(),
        'model_parameters': count_parameters(model),
        'best_epoch': int(best_bundle['row']['best_epoch']),
        'epochs_trained': int(best_bundle['row']['epochs_trained']),
        'stopped_early': bool(best_bundle['row']['stopped_early']),
        'validation_loss': float(best_bundle['row']['val_loss']),
        'validation': best_bundle['validation_metrics'],
        'test_loss': float(test_loss),
        'test': test_metrics,
        'artifacts': {
            'model': str(corpus_dir / 'best_model.pt'),
            'validation_results': str(corpus_dir / 'validation_results.csv'),
            'history': str(history_path),
            'confusion_matrix': str(corpus_dir / 'test_confusion_matrix.png'),
            'predictions': str(corpus_dir / 'test_predictions.csv'),
        },
    }
    _write_json(corpus_dir / 'result.json', result)
    log.info(
        '[%s] TEST | accuracy=%.4f balanced=%.4f macro-F1=%.4f',
        corpus,
        test_metrics['accuracy'],
        test_metrics['balanced_accuracy'],
        test_metrics['macro_f1'],
    )
    return result


def _write_model_comparison(
    results: dict[str, dict[str, Any]],
    prior_results_path: str | Path,
    output_root: str | Path,
) -> pd.DataFrame:
    output_root = ensure_dir(output_root)
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
    prior_path = Path(prior_results_path)
    if prior_path.is_file():
        comparison = pd.read_csv(prior_path)
    else:
        comparison = pd.DataFrame(columns=columns)

    new_rows = []
    for corpus, result in results.items():
        test = result['test']
        new_rows.append(
            {
                'corpus': corpus,
                'model': 'MLP (Ödev 3)',
                'feature_dim': result['feature']['vector_size'],
                'pca_dim': 'none',
                'params': json.dumps(
                    result['best_config'], ensure_ascii=False, sort_keys=True
                ),
                'test_accuracy': test['accuracy'],
                'test_balanced_accuracy': test['balanced_accuracy'],
                'test_macro_f1': test['macro_f1'],
                'test_weighted_f1': test['weighted_f1'],
            }
        )
    comparison = pd.concat([comparison[columns], pd.DataFrame(new_rows)], ignore_index=True)
    comparison = comparison.sort_values(
        ['corpus', 'test_macro_f1'], ascending=[True, False]
    ).reset_index(drop=True)
    comparison.to_csv(output_root / 'model_comparison.csv', index=False)
    return comparison


def run_all(
    *,
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/outputs',
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    grid_mode: str = 'report',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    limit_per_split: int | None = None,
    prior_results_path: str | Path = 'odev2/outputs/test_comparison_with_knn.csv',
) -> dict[str, dict[str, Any]]:
    '''Run independent searches for every requested dataset.'''

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}.')
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    output_root = ensure_dir(output_root)
    device = get_device(device_name)
    feature_config = DEFAULT_CONFIG

    results: dict[str, dict[str, Any]] = {}
    for corpus in corpora:
        results[corpus] = run_corpus(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            grid_mode=grid_mode,
            max_epochs=max_epochs,
            device=device,
            feature_config=feature_config,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
            amp=amp,
            limit_per_split=limit_per_split,
        )

    comparison = _write_model_comparison(results, prior_results_path, output_root)
    summary = {
        'assignment': 3,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'max_epochs': max_epochs,
        'diagnostic_limit_per_split': limit_per_split,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'feature_config': {
            **asdict(feature_config),
            'vector_size': feature_config.vector_size,
        },
        'per_dataset': results,
        'comparison_rows': comparison.to_dict(orient='records'),
    }
    _write_json(Path(output_root) / 'summary.json', summary)
    return results


def load_saved_model(
    checkpoint_path: str | Path,
    device_name: str = 'cpu',
) -> tuple[MLP, FeatureStandardizer, dict[str, Any]]:
    '''Load a delivered model and its training-only normalization parameters.'''

    device = get_device(device_name)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    config = MLPConfig.from_dict(checkpoint['model_config'])
    model = MLP(checkpoint['input_dim'], checkpoint['num_classes'], config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    standardizer = FeatureStandardizer(
        mean=checkpoint['standardizer_mean'].cpu().numpy(),
        scale=checkpoint['standardizer_scale'].cpu().numpy(),
    )
    return model, standardizer, checkpoint
