# Final deneylerini (iki yöntem) tek corpus üzerinde koşturan komut satırı aracı.
#
# Örnekler:
# python final/run_experiment.py --grid-mode report --feature-workers 8
# python final/run_experiment.py --grid-mode quick --limit-per-split 60
# python final/run_experiment.py --methods rnn --grid-mode report
#
# Asıl iş final/pipeline.py'dedir; bu dosya yalnızca argümanları toplayıp run_all'a iletir.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

import argparse  # komut satırı argümanları
from pathlib import Path  # dosya yolları
import sys  # import yolu

# Depo kökünü import yoluna ekle ki "final.*" ve "ser.*" paketleri
# betik nereden çalıştırılırsa çalıştırılsın bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # proje kökünü import yoluna ekle

from final.pipeline import run_all  # noqa: E402  # asıl deney hattı
from final.search_space import GRID_MODES  # noqa: E402  # geçerli arama modları


def main() -> None:
    parser = argparse.ArgumentParser(  # argüman ayrıştırıcı
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Veri ve çıktı yolları.
    parser.add_argument('--manifest', default='data/processed/manifest.csv')  # manifest yolu
    parser.add_argument('--cache-root', default='data/cache/final')  # önbellek kökü
    parser.add_argument('--out-root')  # çıktı kökü (opsiyonel)
    parser.add_argument('--corpus', default='cremad', choices=['cremad', 'meld'])  # veri seti

    # Hangi yöntemler koşulacak (varsayılan: ikisi de).
    parser.add_argument(  # çalıştırılacak yöntemler
        '--methods', nargs='+', default=['cnn', 'rnn'], choices=['cnn', 'rnn']
    )
    parser.add_argument('--grid-mode', choices=GRID_MODES, default='report')  # arama modu
    parser.add_argument('--max-epochs', type=int)  # en fazla epoch (opsiyonel)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')  # cihaz
    # Paralellik: öznitelik çıkarımı iş parçacığı + DataLoader işçileri.
    parser.add_argument('--feature-workers', type=int, default=4)  # öznitelik işçisi
    parser.add_argument('--loader-workers', type=int, default=0)  # veri yükleme işçisi
    parser.add_argument('--seed', type=int, default=42)  # tohum
    parser.add_argument('--no-amp', action='store_true')  # karışık hassasiyeti kapat
    parser.add_argument('--no-refine', action='store_true',  # iyileştirme turunu atla
                        help='Kazanan çevresindeki yerel iyileştirme turunu atla.')
    parser.add_argument(  # tanı amaçlı katman küçültme
        '--limit-per-split',
        type=int,
        help='YALNIZ TANI: her katmanı oransal olarak bu satır sayısına indir.',
    )
    parser.add_argument('--prior-results')  # önceki sonuç CSV'si (opsiyonel)
    args = parser.parse_args()  # argümanları oku

    if args.feature_workers < 1 or args.loader_workers < 0:  # işçi sayıları geçerli mi
        parser.error('İşçi sayıları: feature>=1 ve loader>=0 olmalı.')  # değilse hata
    # quick modda epoch üst sınırı düşük tutulur (duman testi hızlı bitsin).
    max_epochs = args.max_epochs  # kullanıcının verdiği epoch
    if max_epochs is None:  # verilmediyse
        max_epochs = 6 if args.grid_mode == 'quick' else 60  # quick=6, aksi 60

    # Tanı koşuları gerçek çıktıların üzerine yazmasın diye ayrı klasöre gider.
    diagnostic = args.limit_per_split is not None or args.grid_mode == 'quick'  # tanı koşusu mu
    if args.out_root:  # çıktı kökü elle verildiyse
        output_root = args.out_root  # onu kullan
    elif diagnostic:  # tanı koşusuysa
        output_root = 'final/smoke_outputs'  # ayrı (duman) klasör
    else:  # gerçek koşu
        output_root = 'final/outputs'  # asıl çıktı klasörü

    results = run_all(  # deney hattını koştur
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
        amp=not args.no_amp,  # --no-amp verilmediyse AMP açık
        refine=not args.no_refine,  # --no-refine verilmediyse iyileştirme açık
        limit_per_split=args.limit_per_split,
        prior_results_path=args.prior_results,
        seed=args.seed,
    )
    # Koşu sonunda kısa özet bas.
    for method, result in results.items():  # her yöntem için
        test = result['test']  # test metrikleri
        print(  # test sonucunu yazdır
            f'{method}: test accuracy={test["accuracy"]:.4f}, '
            f'macro-F1={test["macro_f1"]:.4f}'
        )


if __name__ == '__main__':  # doğrudan çalıştırılırsa
    main()  # ana fonksiyon
