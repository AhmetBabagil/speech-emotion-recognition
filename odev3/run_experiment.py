'''Run Assignment 3 MLP validation and test experiments.

Examples:
    python odev3/run_experiment.py --grid-mode report
    python odev3/run_experiment.py --grid-mode quick --limit-per-split 48
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev3.pipeline import run_all  # noqa: E402
from odev3.search_space import GRID_MODES, search_space  # noqa: E402


def _result_line(corpus: str, result: dict) -> str:
    test_metrics = result['test']
    return (
        f'{corpus}: test accuracy={test_metrics["accuracy"]:.4f}, '
        f'macro-F1={test_metrics["macro_f1"]:.4f}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--manifest', default='odev1/manifest_subset.csv')
    parser.add_argument('--cache-root', default='data/cache/odev3_melspec')
    parser.add_argument('--out-root')
    parser.add_argument(
        '--corpora',
        nargs='+',
        default=['cremad', 'meld'],
        choices=['cremad', 'meld'],
    )
    parser.add_argument('--grid-mode', choices=GRID_MODES, default='report')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--feature-workers', type=int, default=1)
    parser.add_argument('--loader-workers', type=int, default=0)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument(
        '--limit-per-split',
        type=int,
        help='Diagnostic only: stratified row limit for every fold.',
    )
    parser.add_argument(
        '--prior-results',
        default='odev2/outputs/test_comparison_with_knn.csv',
    )
    parser.add_argument('--skip-report', action='store_true')
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore a compatible search_state.pt and rerun every trial.',
    )
    args = parser.parse_args()

    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('Worker counts must be feature>=1 and loader>=0.')
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = 8 if args.grid_mode == 'quick' else 60

    diagnostic = args.limit_per_split is not None or args.grid_mode == 'quick'
    if args.out_root:
        output_root = args.out_root
    elif diagnostic:
        output_root = 'odev3/smoke_outputs'
    else:
        output_root = 'odev3/outputs'

    print(
        f'Mod={args.grid_mode} | aday={len(search_space(args.grid_mode))} '
        f'| epoch üst sınırı={max_epochs} | çıktı={output_root}'
    )
    results = run_all(
        manifest_path=args.manifest,
        cache_root=args.cache_root,
        output_root=output_root,
        corpora=tuple(args.corpora),
        grid_mode=args.grid_mode,
        max_epochs=max_epochs,
        device_name=args.device,
        feature_workers=args.feature_workers,
        loader_workers=args.loader_workers,
        amp=not args.no_amp,
        limit_per_split=args.limit_per_split,
        prior_results_path=args.prior_results,
        resume=not args.no_resume,
    )

    if not args.skip_report:
        from odev3.build_report import build_report

        build_report(output_root=output_root)
    for corpus, result in results.items():
        print(_result_line(corpus, result))


if __name__ == '__main__':
    main()
