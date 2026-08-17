'''Improvement stage: augmentation / attention variants of each winner.

Reads the winners produced by run_experiment.py, retrains them with the
literature-motivated improvements (SpecAugment masking for the CNN; attention
pooling and feature noise for the RNN), compares on the validation fold and
evaluates the best improved variant on the test fold.

Examples:
    python final/improve.py
    python final/improve.py --methods rnn --out-root final/outputs
'''

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from final.augment import FeatureNoise, SpecMask  # noqa: E402
from final.dataset import Standardizer  # noqa: E402
from final.features import (  # noqa: E402
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
)
from final.models import (  # noqa: E402
    CNNConfig,
    MelCNN,
    OptimSettings,
    RNNConfig,
    SeqRNN,
)
from final.pipeline import SplitSettings, _feature_folds, _limit_stratified  # noqa: E402
from final.training import (  # noqa: E402
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)
from ser.constants import NUM_CLASSES  # noqa: E402
from ser.data.splits import prepare_splits  # noqa: E402
from ser.evaluate import report as evaluate_report  # noqa: E402
from ser.utils import ensure_dir, get_logger  # noqa: E402

log = get_logger(__name__)


def _model_config_from_dict(method: str, values: dict[str, Any]):
    values = dict(values)
    optim = OptimSettings(**values.pop('optim'))
    if method == 'cnn':
        return CNNConfig(channels=tuple(values['channels']),
                         dropout=values['dropout'], optim=optim)
    return RNNConfig(
        rnn_type=values['rnn_type'],
        hidden_size=values['hidden_size'],
        num_layers=values['num_layers'],
        bidirectional=values['bidirectional'],
        dropout=values['dropout'],
        pooling=values['pooling'],
        optim=optim,
    )


def _variants(method: str, model_cfg) -> list[dict[str, Any]]:
    '''Named improvement candidates; transform=None means model-only change.'''

    if method == 'cnn':
        return [
            {'name': 'specaugment', 'model_cfg': model_cfg,
             'transform': SpecMask()},
            {'name': 'specaugment_light', 'model_cfg': model_cfg,
             'transform': SpecMask(freq_masks=1, freq_width=4,
                                   time_masks=1, time_width=8)},
        ]
    variants = [
        {'name': 'feature_noise', 'model_cfg': model_cfg,
         'transform': FeatureNoise(std=0.1)},
    ]
    if model_cfg.pooling != 'attn':
        attn_cfg = replace(model_cfg, pooling='attn')
        variants.insert(0, {'name': 'attention_pooling', 'model_cfg': attn_cfg,
                            'transform': None})
        variants.append({'name': 'attention_plus_noise', 'model_cfg': attn_cfg,
                         'transform': FeatureNoise(std=0.1)})
    return variants


def improve_method(
    method: str,
    folds: dict[str, pd.DataFrame],
    *,
    method_dir: Path,
    cache_root: Path,
    device: torch.device,
    max_epochs: int,
    feature_workers: int,
    loader_workers: int,
    amp: bool,
    seed: int,
) -> dict[str, Any]:
    winner_path = method_dir / 'winner.json'
    if not winner_path.is_file():
        raise FileNotFoundError(
            f'{winner_path} not found; run final/run_experiment.py first.'
        )
    with open(winner_path, encoding='utf-8') as handle:
        winner = json.load(handle)

    if method == 'cnn':
        feature_cfg = MelImageConfig(**winner['feature_config'])
        extract_fn = extract_mel_image
        feature_axis = 1
    else:
        feature_cfg = IntervalConfig(**winner['feature_config'])
        extract_fn = extract_interval_series
        feature_axis = 2
    base_model_cfg = _model_config_from_dict(method, winner['model_config'])
    base_val_f1 = float(winner['val_metrics']['macro_f1'])

    feature_cache: dict = {}
    tensors = _feature_folds(
        folds, feature_cfg, extract_fn, cache_root,
        workers=feature_workers, cache=feature_cache,
    )
    train_x, train_y = tensors['train']
    val_x, val_y = tensors['val']
    test_x, test_y = tensors['test']
    standardizer = Standardizer.fit(train_x, feature_axis)
    train_std = standardizer.transform(train_x)
    val_std = standardizer.transform(val_x)

    def build_model(model_cfg):
        if method == 'cnn':
            return MelCNN(NUM_CLASSES, model_cfg)
        return SeqRNN(feature_cfg.feature_dim, NUM_CLASSES, model_cfg)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for variant in _variants(method, base_model_cfg):
        started = time.perf_counter()
        outcome = train_with_early_stopping(
            build_model(variant['model_cfg']),
            train_std,
            train_y,
            val_std,
            val_y,
            variant['model_cfg'].optim,
            num_classes=NUM_CLASSES,
            device=device,
            max_epochs=max_epochs,
            seed=seed,
            num_workers=loader_workers,
            amp=amp,
            train_transform=variant['transform'],
        )
        seconds = time.perf_counter() - started
        row = {
            'variant': variant['name'],
            'val_accuracy': outcome.validation_metrics['accuracy'],
            'val_balanced_accuracy': outcome.validation_metrics['balanced_accuracy'],
            'val_macro_f1': outcome.validation_metrics['macro_f1'],
            'val_weighted_f1': outcome.validation_metrics['weighted_f1'],
            'delta_vs_winner': outcome.validation_metrics['macro_f1'] - base_val_f1,
            'best_epoch': outcome.best_epoch,
            'epochs_trained': outcome.epochs_trained,
            'seconds': round(seconds, 1),
        }
        rows.append(row)
        log.info('[%s/improve %s] val macro-F1=%.4f (taban %.4f, fark %+.4f)',
                 method, variant['name'], row['val_macro_f1'], base_val_f1,
                 row['delta_vs_winner'])
        if best is None or row['val_macro_f1'] > best['row']['val_macro_f1']:
            best = {'row': row, 'variant': variant, 'outcome': outcome}

    improvements = pd.DataFrame(rows)
    improvements.insert(0, 'method', method)
    improvements.to_csv(method_dir / 'improvements.csv', index=False)

    summary: dict[str, Any] = {
        'method': method,
        'baseline_val_macro_f1': base_val_f1,
        'variants': rows,
        'improved_on_validation': bool(best and best['row']['delta_vs_winner'] > 0),
    }
    if best and best['row']['delta_vs_winner'] > 0:
        class_weights = inverse_frequency_weights(
            folds['train']['label_idx'].to_numpy(), NUM_CLASSES
        )
        _, _, test_prob = evaluate_arrays(
            best['outcome'].model,
            standardizer.transform(test_x),
            test_y,
            class_weights=class_weights,
            device=device,
            num_workers=loader_workers,
        )
        test_metrics = evaluate_report(
            test_y, test_prob.argmax(axis=1), method_dir, prefix='test_improved',
            title=f'{method.upper()} improved test confusion matrix',
        )
        torch.save(
            {
                'state_dict': best['outcome'].model.state_dict(),
                'variant': best['variant']['name'],
                'feature_config': feature_cfg.__dict__,
                'model_config': best['variant']['model_cfg'].to_dict(),
                'standardizer_mean': standardizer.mean,
                'standardizer_scale': standardizer.scale,
                'feature_axis': standardizer.feature_axis,
            },
            method_dir / 'improved_model.pt',
        )
        summary['best_variant'] = best['variant']['name']
        summary['test_improved'] = test_metrics
        log.info('[%s] improved TEST acc=%.4f macro-F1=%.4f',
                 method, test_metrics['accuracy'], test_metrics['macro_f1'])
    else:
        log.info('[%s] no variant beat the winner on validation; '
                 'test left untouched.', method)

    with open(method_dir / 'improvements_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--out-root', default='final/outputs')
    parser.add_argument('--corpus', default='cremad', choices=['cremad', 'meld'])
    parser.add_argument(
        '--methods', nargs='+', default=['cnn', 'rnn'], choices=['cnn', 'rnn']
    )
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=4)
    parser.add_argument('--loader-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument(
        '--limit-per-split',
        type=int,
        help='Diagnostic only: stratified row limit for every fold.',
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    settings = SplitSettings(train_corpora=(args.corpus,), eval_corpora=(args.corpus,))
    train_df, val_df, test_df = prepare_splits(manifest, settings, seed=args.seed)
    if args.limit_per_split is not None:
        train_df = _limit_stratified(train_df, args.limit_per_split, args.seed)
        val_df = _limit_stratified(val_df, args.limit_per_split, args.seed + 1)
        test_df = _limit_stratified(test_df, args.limit_per_split, args.seed + 2)
        log.warning('Diagnostic limit active: train=%d val=%d test=%d',
                    len(train_df), len(val_df), len(test_df))
    folds = {'train': train_df, 'val': val_df, 'test': test_df}

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    corpus_dir = ensure_dir(Path(args.out_root) / args.corpus)
    for method in args.methods:
        improve_method(
            method,
            folds,
            method_dir=Path(corpus_dir) / method,
            cache_root=Path(args.cache_root),
            device=device,
            max_epochs=args.max_epochs,
            feature_workers=args.feature_workers,
            loader_workers=args.loader_workers,
            amp=not args.no_amp,
            seed=args.seed,
        )


if __name__ == '__main__':
    main()
