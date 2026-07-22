'''End-to-end validation search and held-out test evaluation for Assignment 3.'''

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from odev3.dataset import FeatureStandardizer, load_feature_matrix
from odev3.features_melspec import DEFAULT_CONFIG, MelSpecConfig
from odev3.model import MLP, MLPConfig, count_parameters
from odev3.search_space import refinement_space, search_space
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


def _manifest_identity(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    identity: dict[str, Any] = {'path': str(manifest_path.resolve())}
    if manifest_path.is_file():
        stat = manifest_path.stat()
        identity.update({'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return identity


def _search_signature(
    *,
    corpus: str,
    manifest_path: str | Path,
    grid_mode: str,
    max_epochs: int,
    feature_config: MelSpecConfig,
    limit_per_split: int | None,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    return {
        'corpus': corpus,
        'manifest': _manifest_identity(manifest_path),
        'grid_mode': grid_mode,
        'max_epochs': max_epochs,
        'feature_config': asdict(feature_config),
        'limit_per_split': limit_per_split,
        'device': str(device),
        'amp': bool(amp and device.type == 'cuda'),
        'seed': SEED,
    }


def _config_json(config: MLPConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, separators=(',', ':'))


def _config_id(config: MLPConfig) -> str:
    return hashlib.sha1(_config_json(config).encode('utf-8')).hexdigest()[:12]


def _trial_key(stage: str, config: MLPConfig, trial_seed: int) -> str:
    config_json = _config_json(config)
    return f'{stage}|{trial_seed}|{config_json}'


def _serialize_best_bundle(bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if bundle is None:
        return None
    return {
        'row': bundle['row'],
        'config': bundle['config'].to_dict(),
        'history': bundle['history'],
        'validation_metrics': bundle['validation_metrics'],
        'state_dict': bundle['state_dict'],
    }


def _restore_best_bundle(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        'row': payload['row'],
        'config': MLPConfig.from_dict(payload['config']),
        'history': payload['history'],
        'validation_metrics': payload['validation_metrics'],
        'state_dict': payload['state_dict'],
    }


def _save_search_state(
    path: Path,
    *,
    signature: dict[str, Any],
    rows: list[dict[str, Any]],
    completed_keys: set[str],
    best_bundle: dict[str, Any] | None,
    refinement_base: MLPConfig | None,
    stability_base: MLPConfig | None,
) -> None:
    payload = {
        'schema_version': 2,
        'signature': signature,
        'rows': rows,
        'completed_keys': sorted(completed_keys),
        'best_bundle': _serialize_best_bundle(best_bundle),
        'refinement_base': refinement_base.to_dict() if refinement_base else None,
        'stability_base': stability_base.to_dict() if stability_base else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalized_search_signature(signature: dict[str, Any]) -> dict[str, Any]:
    '''Fill defaults added after older, behaviorally equivalent search states.'''

    normalized = json.loads(json.dumps(signature, default=_json_default))
    normalized.setdefault('feature_config', {}).setdefault(
        'frame_strategy',
        'crop_pad',
    )
    return normalized


def _load_search_state(
    path: Path,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location='cpu')
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
        log.warning('Ignoring unreadable search state %s: %s', path, error)
        return None
    saved_signature = payload.get('signature', {})
    signatures_match = _normalized_search_signature(
        saved_signature
    ) == _normalized_search_signature(signature)
    if payload.get('schema_version') != 2 or not signatures_match:
        log.warning('Ignoring incompatible search state: %s', path)
        return None
    return {
        'rows': list(payload.get('rows', [])),
        'completed_keys': set(payload.get('completed_keys', [])),
        'best_bundle': _restore_best_bundle(payload.get('best_bundle')),
        'refinement_base': (
            MLPConfig.from_dict(payload['refinement_base'])
            if payload.get('refinement_base')
            else None
        ),
        'stability_base': (
            MLPConfig.from_dict(payload['stability_base'])
            if payload.get('stability_base')
            else None
        ),
    }


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
    search_stage: str,
    trial_seed: int,
    config: MLPConfig,
    outcome,
    elapsed_seconds: float,
) -> dict[str, Any]:
    metrics = outcome.validation_metrics
    return {
        'trial': trial_id,
        'search_stage': search_stage,
        'seed': trial_seed,
        'config_id': _config_id(config),
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


def _feature_method(config: MelSpecConfig) -> str:
    return (
        'librosa.feature.melspectrogram -> log dB -> '
        f'{config.n_mels}x{config.n_frames} -> '
        f'{config.frame_strategy} -> flatten'
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
    resume: bool = True,
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

    screening_candidates = search_space(grid_mode)
    state_path = corpus_dir / 'search_state.pt'
    signature = _search_signature(
        corpus=corpus,
        manifest_path=manifest_path,
        grid_mode=grid_mode,
        max_epochs=max_epochs,
        feature_config=feature_config,
        limit_per_split=limit_per_split,
        device=device,
        amp=amp,
    )
    saved_state = _load_search_state(state_path, signature) if resume else None
    if saved_state:
        rows: list[dict[str, Any]] = saved_state['rows']
        completed_keys: set[str] = saved_state['completed_keys']
        best_bundle: dict[str, Any] | None = saved_state['best_bundle']
        refinement_base: MLPConfig | None = saved_state['refinement_base']
        stability_base: MLPConfig | None = saved_state['stability_base']
        log.info('[%s] resuming %d completed validation trials', corpus, len(rows))
    else:
        rows = []
        completed_keys = set()
        best_bundle = None
        refinement_base = None
        stability_base = None
    stage_counts: dict[str, int] = {}

    def run_stage(
        stage: str,
        candidates: list[MLPConfig],
        trial_seed: int = SEED,
    ) -> None:
        nonlocal best_bundle
        stage_counts[stage] = stage_counts.get(stage, 0) + len(candidates)
        for stage_index, candidate in enumerate(candidates, start=1):
            key = _trial_key(stage, candidate, trial_seed)
            if key in completed_keys:
                log.info(
                    '[%s] skipping completed %s trial %d/%d',
                    corpus,
                    stage,
                    stage_index,
                    len(candidates),
                )
                continue
            trial_id = len(rows) + 1
            log.info(
                '[%s] %s trial %d/%d | global trial %d | %s',
                corpus,
                stage,
                stage_index,
                len(candidates),
                trial_id,
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
                seed=trial_seed,
                num_workers=loader_workers,
                amp=amp,
            )
            elapsed = time.perf_counter() - started
            row = _validation_row(
                trial_id,
                stage,
                trial_seed,
                candidate,
                outcome,
                elapsed,
            )
            rows.append(row)
            pd.DataFrame(outcome.history).to_csv(
                history_dir / f'trial_{trial_id:03d}.csv', index=False
            )
            pd.DataFrame(rows).to_csv(
                corpus_dir / 'validation_results.partial.csv', index=False
            )
            log.info(
                '[%s] trial %d | best epoch=%d, val macro-F1=%.4f',
                corpus,
                trial_id,
                outcome.best_epoch,
                row['val_macro_f1'],
            )

            if best_bundle is None or _selection_key(row) > _selection_key(
                best_bundle['row']
            ):
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
            completed_keys.add(key)
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
            del outcome
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    initial_stage = 'screening' if grid_mode == 'report' else grid_mode
    run_stage(initial_stage, screening_candidates)

    refinement_candidates: list[MLPConfig] = []
    if grid_mode == 'report':
        if best_bundle is None:
            raise RuntimeError(f'Screening produced no valid trial for {corpus}.')
        if refinement_base is None:
            refinement_base = best_bundle['config']
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
        refinement_candidates = refinement_space(
            refinement_base,
            exclude=screening_candidates,
        )
        run_stage('refinement', refinement_candidates)
        if best_bundle is None:
            raise RuntimeError(f'Refinement produced no valid trial for {corpus}.')
        if stability_base is None:
            stability_base = best_bundle['config']
            _save_search_state(
                state_path,
                signature=signature,
                rows=rows,
                completed_keys=completed_keys,
                best_bundle=best_bundle,
                refinement_base=refinement_base,
                stability_base=stability_base,
            )
        run_stage('stability', [stability_base], trial_seed=SEED + 101)
        run_stage('stability', [stability_base], trial_seed=SEED + 202)

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
        'seed': int(best_bundle['row']['seed']),
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

    stability_summary = None
    if grid_mode == 'report' and stability_base is not None:
        stability_id = _config_id(stability_base)
        stability_rows = [row for row in rows if row['config_id'] == stability_id]
        stability_summary = {
            'config_id': stability_id,
            'config': stability_base.to_dict(),
            'seeds': [int(row['seed']) for row in stability_rows],
            'runs': len(stability_rows),
            'val_macro_f1_mean': float(
                np.mean([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_std': float(
                np.std([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_min': float(
                np.min([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_macro_f1_max': float(
                np.max([row['val_macro_f1'] for row in stability_rows])
            ),
            'val_balanced_accuracy_mean': float(
                np.mean([row['val_balanced_accuracy'] for row in stability_rows])
            ),
            'val_accuracy_mean': float(
                np.mean([row['val_accuracy'] for row in stability_rows])
            ),
        }

    result = {
        'corpus': corpus,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'diagnostic_limit_per_split': limit_per_split,
        'seed': SEED,
        'selection_metric': 'validation macro-F1',
        'search_protocol': (
            'screening -> local refinement -> multi-seed stability'
            if grid_mode == 'report'
            else grid_mode
        ),
        'stability_seeds': [SEED, SEED + 101, SEED + 202]
        if grid_mode == 'report'
        else [SEED],
        'num_trials': len(rows),
        'search_stages': stage_counts,
        'stability': stability_summary,
        'feature': {
            **asdict(feature_config),
            'vector_size': feature_config.vector_size,
            'method': _feature_method(feature_config),
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
        'selected_seed': int(best_bundle['row']['seed']),
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
            'search_state': str(state_path),
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
    resume: bool = True,
    feature_config: MelSpecConfig = DEFAULT_CONFIG,
    feature_configs: dict[str, MelSpecConfig] | None = None,
) -> dict[str, dict[str, Any]]:
    '''Run independent searches for every requested dataset.'''

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Manifest not found: {manifest_path}.')
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    output_root = ensure_dir(output_root)
    device = get_device(device_name)
    feature_config.validate()
    selected_feature_configs = {
        corpus: (feature_configs or {}).get(corpus, feature_config)
        for corpus in corpora
    }
    for corpus, selected_config in selected_feature_configs.items():
        selected_config.validate()

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
            feature_config=selected_feature_configs[corpus],
            feature_workers=feature_workers,
            loader_workers=loader_workers,
            amp=amp,
            limit_per_split=limit_per_split,
            resume=resume,
        )

    comparison = _write_model_comparison(results, prior_results_path, output_root)
    summary = {
        'assignment': 3,
        'manifest': str(manifest_path),
        'grid_mode': grid_mode,
        'max_epochs': max_epochs,
        'diagnostic_limit_per_split': limit_per_split,
        'resume_enabled': resume,
        'device': str(device),
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'feature_config_by_dataset': {
            corpus: {
                **asdict(selected_config),
                'vector_size': selected_config.vector_size,
            }
            for corpus, selected_config in selected_feature_configs.items()
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
