'''Run the final-assignment experiments (both methods) on one corpus.

Examples:
    python final/run_experiment.py --grid-mode report --feature-workers 8
    python final/run_experiment.py --grid-mode quick --limit-per-split 60
    python final/run_experiment.py --methods rnn --grid-mode report
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final.pipeline import run_all  # noqa: E402
from final.search_space import GRID_MODES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--out-root')
    parser.add_argument('--corpus', default='cremad', choices=['cremad', 'meld'])
    parser.add_argument(
        '--methods', nargs='+', default=['cnn', 'rnn'], choices=['cnn', 'rnn']
    )
    parser.add_argument('--grid-mode', choices=GRID_MODES, default='report')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=4)
    parser.add_argument('--loader-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-refine', action='store_true',
                        help='Skip the local refinement round around the winner.')
    parser.add_argument(
        '--limit-per-split',
        type=int,
        help='Diagnostic only: stratified row limit for every fold.',
    )
    parser.add_argument('--prior-results')
    args = parser.parse_args()

    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('Worker counts must be feature>=1 and loader>=0.')
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = 6 if args.grid_mode == 'quick' else 60

    diagnostic = args.limit_per_split is not None or args.grid_mode == 'quick'
    if args.out_root:
        output_root = args.out_root
    elif diagnostic:
        output_root = 'final/smoke_outputs'
    else:
        output_root = 'final/outputs'

    results = run_all(
        args.manifest,
        args.cache_root,
        output_root,
        corpus=args.corpus,
        methods=tuple(args.methods),
        grid_mode=args.grid_mode,
        max_epochs=max_epochs,
        device_name=args.device,
        feature_workers=args.feature_workers,
        loader_workers=args.loader_workers,
        amp=not args.no_amp,
        refine=not args.no_refine,
        limit_per_split=args.limit_per_split,
        prior_results_path=args.prior_results,
        seed=args.seed,
    )
    for method, result in results.items():
        test = result['test']
        print(
            f'{method}: test accuracy={test["accuracy"]:.4f}, '
            f'macro-F1={test["macro_f1"]:.4f}'
        )


if __name__ == '__main__':
    main()
