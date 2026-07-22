'''Validation-only multi-seed confirmation of selected Mel representations.'''

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
from odev3.feature_ablation import REFERENCE_MODELS
from odev3.features_melspec import MelSpecConfig
from odev3.model import MLPConfig
from odev3.pipeline import _splits_for
from odev3.training import train_with_early_stopping
from ser.constants import NUM_CLASSES
from ser.utils import ensure_dir, get_device


PROTOCOL_VERSION = 1
CONFIRMATION_SEEDS = (42, 143, 244)
CORPUS_DISPLAY_NAMES = {'cremad': 'CREMA-D', 'meld': 'MELD'}


def confirmation_candidates(
    corpus: str,
) -> tuple[tuple[str, MelSpecConfig], ...]:
    '''Return the main representation and its strongest single-seed challenger.'''

    candidates = {
        'cremad': (
            ('main_64x64_resize', MelSpecConfig(frame_strategy='resize')),
            (
                'challenger_96x64_crop_pad',
                MelSpecConfig(n_mels=96, frame_strategy='crop_pad'),
            ),
        ),
        'meld': (
            ('main_64x64_crop_pad', MelSpecConfig(frame_strategy='crop_pad')),
            (
                'challenger_80x64_resize',
                MelSpecConfig(n_mels=80, frame_strategy='resize'),
            ),
        ),
    }
    if corpus not in candidates:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    return candidates[corpus]


def _normalized_model_config(payload: Any) -> str | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _completed_keys(
    rows: list[dict[str, Any]],
    max_epochs: int,
    model_config: MLPConfig,
) -> set[tuple[str, int]]:
    expected_model = _normalized_model_config(model_config.to_dict())
    return {
        (str(row['feature_fingerprint']), int(row['seed']))
        for row in rows
        if int(row.get('protocol_version', 0)) == PROTOCOL_VERSION
        and int(row['max_epochs']) == max_epochs
        and _normalized_model_config(row.get('model_config')) == expected_model
    }


def _load_partial(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient='records')


def _matching_rows(
    rows: list[dict[str, Any]],
    *,
    max_epochs: int,
    model_config: MLPConfig,
    candidates: tuple[tuple[str, MelSpecConfig], ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    expected_model = _normalized_model_config(model_config.to_dict())
    candidate_keys = {
        (name, config.fingerprint) for name, config in candidates
    }
    expected_seeds = set(seeds)
    return [
        row
        for row in rows
        if int(row.get('protocol_version', 0)) == PROTOCOL_VERSION
        and int(row['max_epochs']) == max_epochs
        and _normalized_model_config(row.get('model_config')) == expected_model
        and (str(row['candidate']), str(row['feature_fingerprint']))
        in candidate_keys
        and int(row['seed']) in expected_seeds
    ]


def aggregate_stability_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    '''Rank feature candidates by mean validation macro-F1 across seeds.'''

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row['corpus'],
            row['candidate'],
            row['feature_fingerprint'],
            row['frame_strategy'],
            int(row['n_mels']),
            int(row['n_frames']),
            int(row['vector_size']),
        )
        groups.setdefault(key, []).append(row)

    aggregates = []
    for key, candidate_rows in groups.items():
        macro_f1 = np.asarray(
            [row['val_macro_f1'] for row in candidate_rows],
            dtype=np.float64,
        )
        accuracy = np.asarray(
            [row['val_accuracy'] for row in candidate_rows],
            dtype=np.float64,
        )
        balanced_accuracy = np.asarray(
            [row['val_balanced_accuracy'] for row in candidate_rows],
            dtype=np.float64,
        )
        losses = np.asarray(
            [row['val_loss'] for row in candidate_rows],
            dtype=np.float64,
        )
        aggregates.append(
            {
                'corpus': key[0],
                'candidate': key[1],
                'feature_fingerprint': key[2],
                'frame_strategy': key[3],
                'n_mels': key[4],
                'n_frames': key[5],
                'vector_size': key[6],
                'runs': len(candidate_rows),
                'seeds': ','.join(
                    str(seed)
                    for seed in sorted({int(row['seed']) for row in candidate_rows})
                ),
                'val_macro_f1_mean': float(macro_f1.mean()),
                'val_macro_f1_std': float(macro_f1.std(ddof=0)),
                'val_macro_f1_min': float(macro_f1.min()),
                'val_macro_f1_max': float(macro_f1.max()),
                'val_accuracy_mean': float(accuracy.mean()),
                'val_accuracy_std': float(accuracy.std(ddof=0)),
                'val_balanced_accuracy_mean': float(balanced_accuracy.mean()),
                'val_balanced_accuracy_std': float(
                    balanced_accuracy.std(ddof=0)
                ),
                'val_loss_mean': float(losses.mean()),
            }
        )

    aggregates.sort(
        key=lambda row: (
            str(row['corpus']),
            -float(row['val_macro_f1_mean']),
            float(row['val_macro_f1_std']),
            float(row['val_loss_mean']),
        )
    )
    ranks: dict[str, int] = {}
    for row in aggregates:
        corpus = str(row['corpus'])
        ranks[corpus] = ranks.get(corpus, 0) + 1
        row['rank'] = ranks[corpus]
    return aggregates


def paired_candidate_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    '''Compare main and challenger scores on the exact same seed set.'''

    candidates = {str(row['candidate']) for row in rows}
    main_candidates = {
        candidate for candidate in candidates if candidate.startswith('main_')
    }
    challenger_candidates = {
        candidate
        for candidate in candidates
        if candidate.startswith('challenger_')
    }
    if len(main_candidates) != 1 or len(challenger_candidates) != 1:
        raise ValueError('rows must contain one main and one challenger candidate.')
    main_candidate = main_candidates.pop()
    challenger_candidate = challenger_candidates.pop()

    def scores_for(candidate: str) -> dict[int, float]:
        candidate_rows = [row for row in rows if row['candidate'] == candidate]
        scores = {
            int(row['seed']): float(row['val_macro_f1'])
            for row in candidate_rows
        }
        if len(scores) != len(candidate_rows):
            raise ValueError(f'duplicate seed rows found for {candidate}.')
        return scores

    main_scores = scores_for(main_candidate)
    challenger_scores = scores_for(challenger_candidate)
    if set(main_scores) != set(challenger_scores):
        raise ValueError('main and challenger must use identical seed sets.')

    paired_rows = []
    main_wins = 0
    challenger_wins = 0
    ties = 0
    for seed in sorted(main_scores):
        difference = main_scores[seed] - challenger_scores[seed]
        if np.isclose(difference, 0.0, rtol=0.0, atol=1e-12):
            winner = 'tie'
            ties += 1
        elif difference > 0.0:
            winner = 'main'
            main_wins += 1
        else:
            winner = 'challenger'
            challenger_wins += 1
        paired_rows.append(
            {
                'seed': seed,
                'main_val_macro_f1': main_scores[seed],
                'challenger_val_macro_f1': challenger_scores[seed],
                'main_minus_challenger': difference,
                'winner': winner,
            }
        )
    return {
        'main_candidate': main_candidate,
        'challenger_candidate': challenger_candidate,
        'seeds': sorted(main_scores),
        'paired_runs': paired_rows,
        'main_wins': main_wins,
        'challenger_wins': challenger_wins,
        'ties': ties,
        'mean_main_minus_challenger': float(
            np.mean([row['main_minus_challenger'] for row in paired_rows])
        ),
    }


def reference_model(corpus: str) -> MLPConfig:
    if corpus not in REFERENCE_MODELS:
        raise ValueError(f'Unsupported corpus: {corpus!r}.')
    return REFERENCE_MODELS[corpus]


def _plot_stability(
    rows: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    path: str | Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ordered = sorted(aggregates, key=lambda row: int(row['rank']))
    positions = np.arange(len(ordered), dtype=np.float64)
    means = [float(row['val_macro_f1_mean']) for row in ordered]
    errors = [float(row['val_macro_f1_std']) for row in ordered]
    labels = [
        (
            f'{int(row["n_mels"])}×{int(row["n_frames"])}\n'
            f'{str(row["frame_strategy"]).replace("_", "-")}'
        )
        for row in ordered
    ]
    colors = ['#2878b5', '#e07a1f']
    fig, axis = plt.subplots(figsize=(7.2, 4.6))

    for index, (position, aggregate) in enumerate(
        zip(positions, ordered, strict=True)
    ):
        axis.errorbar(
            position,
            means[index],
            yerr=errors[index],
            fmt='D',
            markersize=9,
            color=colors[index],
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=8,
            capthick=2.2,
            zorder=2,
            label='Üç-seed ortalaması ± std' if index == 0 else None,
        )
        candidate_rows = [
            row for row in rows if row['candidate'] == aggregate['candidate']
        ]
        candidate_rows.sort(key=lambda row: int(row['seed']))
        offsets = np.linspace(-0.12, 0.12, len(candidate_rows))
        scores = [float(row['val_macro_f1']) for row in candidate_rows]
        axis.scatter(
            position + offsets,
            scores,
            color='#202020',
            s=34,
            zorder=4,
            label='Tekil seed sonucu' if index == 0 else None,
        )
        axis.text(
            position,
            means[index] + errors[index] + 0.006,
            f'{means[index]:.4f}',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    all_scores = [float(row['val_macro_f1']) for row in rows]
    lower = max(0.0, min(all_scores) - 0.035)
    upper = min(1.0, max(all_scores) + 0.045)
    axis.set_ylim(lower, upper)
    axis.set_xticks(positions, labels)
    axis.set_ylabel('Validation macro-F1')
    axis.set_title(title)
    axis.grid(axis='y', alpha=0.25)
    axis.legend(loc='lower left')
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _persist_corpus_rows(
    rows: list[dict[str, Any]],
    corpus_dir: Path,
) -> list[dict[str, Any]]:
    ordered = pd.DataFrame(rows).sort_values(['candidate', 'seed'])
    ordered.to_csv(corpus_dir / 'feature_stability.partial.csv', index=False)
    ordered.to_csv(corpus_dir / 'feature_stability_runs.csv', index=False)
    aggregates = aggregate_stability_rows(rows)
    pd.DataFrame(aggregates).to_csv(
        corpus_dir / 'feature_stability.csv',
        index=False,
    )
    corpus = str(rows[0]['corpus'])
    _plot_stability(
        rows,
        aggregates,
        corpus_dir / 'feature_stability.png',
        f'{CORPUS_DISPLAY_NAMES.get(corpus, corpus.upper())} Mel temsil kararlılığı',
    )
    return aggregates


def run_corpus_stability(
    corpus: str,
    *,
    manifest_path: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    max_epochs: int,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, Any]:
    '''Train two fixed feature candidates across seeds without loading test audio.'''

    model_config = reference_model(corpus)
    if max_epochs <= 0:
        raise ValueError(f'max_epochs must be positive, got {max_epochs}.')
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('seeds must contain unique non-negative integers.')
    if feature_workers < 1 or loader_workers < 0:
        raise ValueError('Worker counts must be feature>=1 and loader>=0.')

    device = get_device(device_name)
    corpus_dir = ensure_dir(Path(output_root) / corpus)
    history_dir = ensure_dir(corpus_dir / 'histories')
    partial_path = corpus_dir / 'feature_stability.partial.csv'
    candidates = confirmation_candidates(corpus)
    rows = _matching_rows(
        _load_partial(partial_path),
        max_epochs=max_epochs,
        model_config=model_config,
        candidates=candidates,
        seeds=seeds,
    )
    completed = _completed_keys(rows, max_epochs, model_config)
    train_frame, validation_frame, _held_out_test_frame = _splits_for(
        corpus,
        manifest_path,
    )
    total_trials = len(candidates) * len(seeds)
    trial = 0

    for candidate_name, feature_config in candidates:
        if all(
            (feature_config.fingerprint, seed) in completed
            for seed in seeds
        ):
            for seed in seeds:
                trial += 1
                print(
                    f'{corpus}: stability trial {trial}/{total_trials} already '
                    f'complete ({candidate_name}, seed={seed})'
                )
            continue

        cache_dir = Path(cache_root) / corpus
        train_features, train_labels = load_feature_matrix(
            train_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} stability train {candidate_name}',
        )
        validation_features, validation_labels = load_feature_matrix(
            validation_frame,
            cache_dir,
            feature_config,
            workers=feature_workers,
            description=f'{corpus} stability validation {candidate_name}',
        )
        standardizer = FeatureStandardizer.fit(train_features)
        train_features = standardizer.transform(train_features)
        validation_features = standardizer.transform(validation_features)

        for seed in seeds:
            trial += 1
            key = (feature_config.fingerprint, seed)
            if key in completed:
                print(
                    f'{corpus}: stability trial {trial}/{total_trials} already '
                    f'complete ({candidate_name}, seed={seed})'
                )
                continue

            print(
                f'{corpus}: stability trial {trial}/{total_trials} '
                f'({candidate_name}, seed={seed})'
            )
            started = time.perf_counter()
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
                seed=seed,
                num_workers=loader_workers,
                amp=device.type == 'cuda',
            )
            elapsed = time.perf_counter() - started
            metrics = outcome.validation_metrics
            row = {
                'protocol_version': PROTOCOL_VERSION,
                'corpus': corpus,
                'candidate': candidate_name,
                'feature_fingerprint': feature_config.fingerprint,
                'frame_strategy': feature_config.frame_strategy,
                'n_mels': feature_config.n_mels,
                'n_frames': feature_config.n_frames,
                'vector_size': feature_config.vector_size,
                'seed': seed,
                'max_epochs': max_epochs,
                'model_config': json.dumps(
                    model_config.to_dict(),
                    sort_keys=True,
                ),
                'class_weighting': 'inverse-frequency CrossEntropyLoss',
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
            completed.add(key)
            pd.DataFrame(outcome.history).to_csv(
                history_dir
                / f'{candidate_name}_seed_{seed}_{feature_config.fingerprint}.csv',
                index=False,
            )
            _persist_corpus_rows(rows, corpus_dir)
            print(
                f'{corpus}: {candidate_name} seed={seed} validation '
                f'macro-F1={metrics["macro_f1"]:.4f}'
            )
            del outcome

        del train_features, validation_features

    aggregates = _persist_corpus_rows(rows, corpus_dir)
    comparison = paired_candidate_comparison(rows)
    (corpus_dir / 'feature_stability_comparison.json').write_text(
        json.dumps(
            comparison,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding='utf-8',
    )
    return {
        'runs': rows,
        'aggregates': aggregates,
        'comparison': comparison,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__}.')


def run_feature_stability(
    *,
    corpora: tuple[str, ...] = ('cremad', 'meld'),
    manifest_path: str | Path = 'odev1/manifest_subset.csv',
    cache_root: str | Path = 'data/cache/odev3_melspec',
    output_root: str | Path = 'odev3/feature_stability',
    max_epochs: int = 60,
    seeds: tuple[int, ...] = CONFIRMATION_SEEDS,
    device_name: str = 'auto',
    feature_workers: int = 1,
    loader_workers: int = 0,
) -> dict[str, dict[str, Any]]:
    results = {
        corpus: run_corpus_stability(
            corpus,
            manifest_path=manifest_path,
            cache_root=cache_root,
            output_root=output_root,
            max_epochs=max_epochs,
            seeds=seeds,
            device_name=device_name,
            feature_workers=feature_workers,
            loader_workers=loader_workers,
        )
        for corpus in corpora
    }
    summary = {
        'protocol_version': PROTOCOL_VERSION,
        'protocol': (
            'validation only; two fixed Mel candidates per corpus; '
            'held-out test audio is not loaded'
        ),
        'selection_rule': 'highest mean validation macro-F1 across fixed seeds',
        'class_weighting': 'inverse-frequency CrossEntropyLoss',
        'test_features_loaded': False,
        'seeds': list(seeds),
        'max_epochs': max_epochs,
        'reference_models': {
            corpus: reference_model(corpus).to_dict() for corpus in corpora
        },
        'candidates': {
            corpus: [
                {'name': name, 'config': asdict(config)}
                for name, config in confirmation_candidates(corpus)
            ]
            for corpus in corpora
        },
        'per_dataset': results,
    }
    output_root = ensure_dir(output_root)
    (Path(output_root) / 'summary.json').write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
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
    parser.add_argument('--out-root', default='odev3/feature_stability')
    parser.add_argument('--max-epochs', type=int, default=60)
    parser.add_argument('--seeds', nargs='+', type=int, default=CONFIRMATION_SEEDS)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=0)
    args = parser.parse_args()
    try:
        run_feature_stability(
            corpora=tuple(args.corpora),
            manifest_path=args.manifest,
            cache_root=args.cache_root,
            output_root=args.out_root,
            max_epochs=args.max_epochs,
            seeds=tuple(args.seeds),
            device_name=args.device,
            feature_workers=args.feature_workers,
            loader_workers=args.loader_workers,
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
