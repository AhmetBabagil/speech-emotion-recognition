'''End-to-end final-assignment pipeline for one corpus.

For each method: candidates are searched on the validation fold, a local
refinement round runs around the winner, and only the final winner touches
the test fold once. All artefacts (search logs, winner configs, histories,
metrics, confusion matrices, model weights) land under the output root.
'''

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from final.dataset import Standardizer, load_feature_tensor
from final.features import (
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import MelCNN, RNNConfig, SeqRNN, count_parameters
from final.search_space import (
    cnn_refinement,
    cnn_space,
    rnn_refinement,
    rnn_space,
)
from final.training import (
    TrainingOutcome,
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.data.splits import prepare_splits
from ser.constants import NUM_CLASSES
from ser.evaluate import report as evaluate_report
from ser.utils import ensure_dir, get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class SplitSettings:
    '''Just enough of ser.config's data section for prepare_splits.'''

    train_corpora: tuple[str, ...]
    eval_corpora: tuple[str, ...]
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    split: str = 'speaker'


def _limit_stratified(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    '''Diagnostic-only per-fold cap that keeps the class ratio intact.'''

    if limit >= len(df):
        return df
    parts = []
    rng_seed = seed
    for _, group in df.groupby('label_idx'):
        take = max(1, int(round(limit * len(group) / len(df))))
        parts.append(group.sample(n=min(take, len(group)), random_state=rng_seed))
        rng_seed += 1
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _feature_folds(
    folds: dict[str, pd.DataFrame],
    feature_cfg,
    extract_fn: Callable,
    cache_root: str | Path,
    *,
    workers: int,
    cache: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    '''Extract (or reuse) the feature tensors of the given folds.'''

    result = {}
    for name, records in folds.items():
        key = (feature_cfg.fingerprint, name)
        if key not in cache:
            cache[key] = load_feature_tensor(
                records,
                cache_root,
                feature_cfg.fingerprint,
                lambda path: extract_fn(path, feature_cfg),
                feature_cfg.shape,
                workers=workers,
                description=f'{feature_cfg.fingerprint} {name}',
            )
        result[name] = cache[key]
    return result


def _candidate_row(feature_cfg, model_cfg, outcome: TrainingOutcome, seconds: float,
                   stage: str, n_params: int) -> dict[str, Any]:
    row = {
        'stage': stage,
        'feature_fingerprint': feature_cfg.fingerprint,
        'feature_config': json.dumps(feature_cfg.__dict__, sort_keys=True),
        'model_config': json.dumps(model_cfg.to_dict(), sort_keys=True),
        'parameters': n_params,
        'best_epoch': outcome.best_epoch,
        'epochs_trained': outcome.epochs_trained,
        'stopped_early': outcome.stopped_early,
        'val_loss': outcome.validation_loss,
        'val_accuracy': outcome.validation_metrics['accuracy'],
        'val_balanced_accuracy': outcome.validation_metrics['balanced_accuracy'],
        'val_macro_f1': outcome.validation_metrics['macro_f1'],
        'val_weighted_f1': outcome.validation_metrics['weighted_f1'],
        'seconds': round(seconds, 1),
    }
    return row


def _plot_history(history: list[dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, [h['train_loss'] for h in history], label='train')
    ax1.plot(epochs, [h['val_loss'] for h in history], label='validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Weighted CE loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, [h['train_macro_f1'] for h in history], label='train')
    ax2.plot(epochs, [h['val_macro_f1'] for h in history], label='validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_method(
    method: str,
    folds: dict[str, pd.DataFrame],
    *,
    cache_root: Path,
    out_dir: Path,
    grid_mode: str,
    max_epochs: int,
    device: torch.device,
    feature_workers: int,
    loader_workers: int,
    amp: bool,
    refine: bool,
    seed: int,
) -> dict[str, Any]:
    '''Search, refine and test one method; return its summary dict.'''

    if method == 'cnn':
        candidates = cnn_space(grid_mode)
        extract_fn = extract_mel_image
        feature_axis = 1  # [N, mels, T] -> per mel bin
        refinement_fn = cnn_refinement
    elif method == 'rnn':
        candidates = rnn_space(grid_mode)
        extract_fn = extract_interval_series
        feature_axis = 2  # [N, T, D] -> per feature dimension
        refinement_fn = rnn_refinement
    else:
        raise ValueError(f'Unknown method {method!r}.')

    def build_model(feature_cfg, model_cfg):
        if method == 'cnn':
            return MelCNN(NUM_CLASSES, model_cfg)
        return SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)

    ensure_dir(out_dir)
    feature_cache: dict = {}
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    def run_stage(stage: str, stage_candidates) -> None:
        nonlocal best
        for index, (feature_cfg, model_cfg) in enumerate(stage_candidates, start=1):
            tensors = _feature_folds(
                {'train': folds['train'], 'val': folds['val']},
                feature_cfg,
                extract_fn,
                cache_root,
                workers=feature_workers,
                cache=feature_cache,
            )
            train_x, train_y = tensors['train']
            val_x, val_y = tensors['val']
            standardizer = Standardizer.fit(train_x, feature_axis)
            model = build_model(feature_cfg, model_cfg)
            n_params = count_parameters(model)
            started = time.perf_counter()
            outcome = train_with_early_stopping(
                model,
                standardizer.transform(train_x),
                train_y,
                standardizer.transform(val_x),
                val_y,
                model_cfg.optim,
                num_classes=NUM_CLASSES,
                device=device,
                max_epochs=max_epochs,
                seed=seed,
                num_workers=loader_workers,
                amp=amp,
            )
            seconds = time.perf_counter() - started
            row = _candidate_row(feature_cfg, model_cfg, outcome, seconds, stage, n_params)
            rows.append(row)
            log.info(
                '[%s/%s %d/%d] val macro-F1=%.4f acc=%.4f (epoch %d/%d, %.0fs)',
                method, stage, index, len(stage_candidates),
                row['val_macro_f1'], row['val_accuracy'],
                outcome.best_epoch, outcome.epochs_trained, seconds,
            )
            if best is None or row['val_macro_f1'] > best['row']['val_macro_f1']:
                best = {
                    'row': row,
                    'feature_cfg': feature_cfg,
                    'model_cfg': model_cfg,
                    'standardizer': standardizer,
                    'outcome': outcome,
                }

    run_stage('search', candidates)
    if refine and grid_mode != 'quick' and best is not None:
        seen = set(candidates) | {(best['feature_cfg'], best['model_cfg'])}
        refinement = [
            c for c in refinement_fn((best['feature_cfg'], best['model_cfg']))
            if c not in seen
        ]
        if refinement:
            run_stage('refine', refinement)

    if best is None:
        raise RuntimeError(f'No successful candidate for method {method}.')

    search_log = pd.DataFrame(rows)
    search_log.to_csv(out_dir / 'search_log.csv', index=False)

    feature_cfg = best['feature_cfg']
    model_cfg = best['model_cfg']
    standardizer = best['standardizer']
    outcome: TrainingOutcome = best['outcome']

    pd.DataFrame(outcome.history).to_csv(out_dir / 'winner_history.csv', index=False)
    _plot_history(outcome.history, out_dir / 'winner_learning_curve.png',
                  f'{method.upper()} winner learning curve')
    with open(out_dir / 'winner.json', 'w', encoding='utf-8') as handle:
        json.dump(
            {
                'method': method,
                'feature_config': feature_cfg.__dict__,
                'model_config': model_cfg.to_dict(),
                'parameters': best['row']['parameters'],
                'best_epoch': outcome.best_epoch,
                'val_metrics': outcome.validation_metrics,
            },
            handle,
            indent=2,
        )
    torch.save(
        {
            'state_dict': outcome.model.state_dict(),
            'feature_config': feature_cfg.__dict__,
            'model_config': model_cfg.to_dict(),
            'standardizer_mean': standardizer.mean,
            'standardizer_scale': standardizer.scale,
            'feature_axis': standardizer.feature_axis,
        },
        out_dir / 'winner_model.pt',
    )

    # The test fold is touched exactly once, by the final winner.
    test_x, test_y = _feature_folds(
        {'test': folds['test']},
        feature_cfg,
        extract_fn,
        cache_root,
        workers=feature_workers,
        cache=feature_cache,
    )['test']
    class_weights = inverse_frequency_weights(
        folds['train']['label_idx'].to_numpy(), NUM_CLASSES
    )
    test_loss, _, test_prob = evaluate_arrays(
        outcome.model,
        standardizer.transform(test_x),
        test_y,
        class_weights=class_weights,
        device=device,
        num_workers=loader_workers,
    )
    test_pred = test_prob.argmax(axis=1)
    test_metrics = evaluate_report(
        test_y, test_pred, out_dir, prefix='test',
        title=f'{method.upper()} test confusion matrix',
    )
    val_metrics = outcome.validation_metrics
    log.info('[%s] TEST acc=%.4f macro-F1=%.4f (val macro-F1=%.4f)',
             method, test_metrics['accuracy'], test_metrics['macro_f1'],
             val_metrics['macro_f1'])
    return {
        'method': method,
        'winner_feature': feature_cfg.__dict__,
        'winner_model': model_cfg.to_dict(),
        'val': val_metrics,
        'test': test_metrics,
        'test_loss': test_loss,
        'search_rows': len(rows),
    }


def run_all(
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    *,
    corpus: str = 'cremad',
    methods: tuple[str, ...] = ('cnn', 'rnn'),
    grid_mode: str = 'report',
    max_epochs: int = 60,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
    amp: bool = True,
    refine: bool = True,
    limit_per_split: int | None = None,
    prior_results_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    settings = SplitSettings(train_corpora=(corpus,), eval_corpora=(corpus,))
    train_df, val_df, test_df = prepare_splits(manifest, settings, seed=seed)
    if limit_per_split is not None:
        train_df = _limit_stratified(train_df, limit_per_split, seed)
        val_df = _limit_stratified(val_df, limit_per_split, seed + 1)
        test_df = _limit_stratified(test_df, limit_per_split, seed + 2)
        log.warning('Diagnostic limit active: train=%d val=%d test=%d',
                    len(train_df), len(val_df), len(test_df))
    folds = {'train': train_df, 'val': val_df, 'test': test_df}

    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    log.info('Corpus=%s device=%s grid=%s | train=%d val=%d test=%d',
             corpus, device, grid_mode, len(train_df), len(val_df), len(test_df))

    output_root = ensure_dir(Path(output_root) / corpus)
    cache_root = Path(cache_root)
    results: dict[str, dict[str, Any]] = {}
    for method in methods:
        results[method] = run_method(
            method,
            folds,
            cache_root=cache_root,
            out_dir=Path(output_root) / method,
            grid_mode=grid_mode,
            max_epochs=max_epochs,
            device=device,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
            amp=amp,
            refine=refine,
            seed=seed,
        )

    comparison = _comparison_table(results, prior_results_path)
    comparison.to_csv(Path(output_root) / 'method_comparison.csv', index=False)
    with open(Path(output_root) / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2, default=str)
    return results


def _comparison_table(
    results: dict[str, dict[str, Any]],
    prior_results_path: str | Path | None,
) -> pd.DataFrame:
    rows = []
    names = {'cnn': 'Yöntem 1: Mel + CNN', 'rnn': 'Yöntem 2: Aralık + LSTM/GRU'}
    for method, result in results.items():
        rows.append({
            'model': names.get(method, method),
            'source': 'final',
            'val_macro_f1': result['val']['macro_f1'],
            'test_accuracy': result['test']['accuracy'],
            'test_balanced_accuracy': result['test']['balanced_accuracy'],
            'test_macro_f1': result['test']['macro_f1'],
            'test_weighted_f1': result['test']['weighted_f1'],
        })
    table = pd.DataFrame(rows)
    if prior_results_path and Path(prior_results_path).is_file():
        try:
            prior = pd.read_csv(prior_results_path)
            prior['source'] = Path(prior_results_path).parent.name
            table = pd.concat([table, prior], ignore_index=True)
        except (OSError, ValueError, pd.errors.ParserError) as error:
            log.warning('Prior results could not be merged: %s', error)
    return table
