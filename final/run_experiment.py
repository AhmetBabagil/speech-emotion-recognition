'''Final deneylerini (iki yöntem) tek corpus üzerinde koşturan komut satırı aracı.

Örnekler:
    python final/run_experiment.py --grid-mode report --feature-workers 8
    python final/run_experiment.py --grid-mode quick --limit-per-split 60
    python final/run_experiment.py --methods rnn --grid-mode report

Asıl iş final/pipeline.py'dedir; bu dosya yalnızca argümanları toplayıp
run_all'a iletir.
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Depo kökünü import yoluna ekle ki "final.*" ve "ser.*" paketleri
# betik nereden çalıştırılırsa çalıştırılsın bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from final.pipeline import run_all  # noqa: E402
from final.search_space import GRID_MODES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Veri ve çıktı yolları.
    parser.add_argument('--manifest', default='data/processed/manifest.csv')
    parser.add_argument('--cache-root', default='data/cache/final')
    parser.add_argument('--out-root')
    parser.add_argument('--corpus', default='cremad', choices=['cremad', 'meld'])
    # Hangi yöntemler koşulacak (varsayılan: ikisi de).
    parser.add_argument(
        '--methods', nargs='+', default=['cnn', 'rnn'], choices=['cnn', 'rnn']
    )
    parser.add_argument('--grid-mode', choices=GRID_MODES, default='report')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    # Paralellik: öznitelik çıkarımı iş parçacığı + DataLoader işçileri.
    parser.add_argument('--feature-workers', type=int, default=4)
    parser.add_argument('--loader-workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-refine', action='store_true',
                        help='Kazanan çevresindeki yerel iyileştirme turunu atla.')
    parser.add_argument(
        '--limit-per-split',
        type=int,
        help='YALNIZ TANI: her katmanı oransal olarak bu satır sayısına indir.',
    )
    parser.add_argument('--prior-results')
    args = parser.parse_args()

    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('İşçi sayıları: feature>=1 ve loader>=0 olmalı.')
    # quick modda epoch üst sınırı düşük tutulur (duman testi hızlı bitsin).
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = 6 if args.grid_mode == 'quick' else 60

    # Tanı koşuları gerçek çıktıların üzerine yazmasın diye ayrı klasöre gider.
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
    # Koşu sonunda kısa özet bas.
    for method, result in results.items():
        test = result['test']
        print(
            f'{method}: test accuracy={test["accuracy"]:.4f}, '
            f'macro-F1={test["macro_f1"]:.4f}'
        )


if __name__ == '__main__':
    main()
