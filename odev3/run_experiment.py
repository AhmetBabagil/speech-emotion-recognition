'''Ödev 3 MLP doğrulama ve test deneylerini çalıştıran komut satırı girişi.

Bu dosya projenin "başlat düğmesi"dir: komut satırı argümanlarını okur,
öznitelik ayarlarını kurar, ardından asıl işi yapan ``odev3.pipeline.run_all``
fonksiyonuna devreder. İş bitince (istenirse) raporu üretir ve her veri
kümesi için tek satırlık test özetini yazdırır.

Examples:
    python odev3/run_experiment.py --grid-mode report
    python odev3/run_experiment.py --grid-mode quick --limit-per-split 48
'''

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Depo kökünü import yoluna ekle: böylece bu dosya "python odev3/run_experiment.py"
# şeklinde doğrudan çalıştırıldığında da 'odev3' ve 'ser' paketleri bulunur.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev3.pipeline import run_all  # noqa: E402
from odev3.features_melspec import MelSpecConfig  # noqa: E402
from odev3.search_space import GRID_MODES, search_space  # noqa: E402


# Ödev şartı: öznitelik vektörü en az 4000 boyutlu olmalı. Bu sabit, yanlış
# parametrelerle (örn. çok az kare) şartı ihlal eden bir vektör üretilmesini
# daha komut satırında engeller.
MIN_ASSIGNMENT_VECTOR_SIZE = 4_000


def build_feature_config(n_frames: int, frame_strategy: str) -> MelSpecConfig:
    # Verilen kare sayısı ve stratejiyle bir MelSpecConfig kurar, doğrular ve
    # ödevin minimum vektör boyutu şartını denetler. Örn. n_frames=62 iken
    # 64*62=3968 < 4000 olur ve burada hata alınır.
    config = MelSpecConfig(
        n_frames=n_frames,
        frame_strategy=frame_strategy,
    )
    config.validate()
    if config.vector_size < MIN_ASSIGNMENT_VECTOR_SIZE:
        raise ValueError(
            'Mel feature vector must contain at least '
            f'{MIN_ASSIGNMENT_VECTOR_SIZE} values; got {config.vector_size}.'
        )
    return config


def build_feature_configs(
    corpora: tuple[str, ...],
    *,
    default_frames: int,
    default_strategy: str,
    frame_overrides: dict[str, int | None],
    strategy_overrides: dict[str, str | None],
) -> dict[str, MelSpecConfig]:
    # Her veri kümesi (corpus) için ayrı öznitelik ayarı üretir.
    # Mantık: kullanıcı o kümeye özel bir değer verdiyse (override) onu,
    # vermediyse genel varsayılanı kullan. "or" hilesi burada işe yarar çünkü
    # override None olduğunda ifade varsayılana düşer.
    return {
        corpus: build_feature_config(
            frame_overrides.get(corpus) or default_frames,
            strategy_overrides.get(corpus) or default_strategy,
        )
        for corpus in corpora
    }


def _result_line(corpus: str, result: dict) -> str:
    # Deney bitiminde konsola basılan tek satırlık özet: test doğruluğu ve
    # macro-F1. Biçim testlerde de doğrulandığı için sabittir.
    test_metrics = result['test']
    return (
        f'{corpus}: test accuracy={test_metrics["accuracy"]:.4f}, '
        f'macro-F1={test_metrics["macro_f1"]:.4f}'
    )


def main() -> None:
    # ---------------------------------------------------------------
    # 1) Komut satırı arayüzü: tüm ayarlar bayraklarla değiştirilebilir,
    #    ama varsayılanlar ödevin standart akışını çalıştırır.
    # ---------------------------------------------------------------
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
    parser.add_argument('--mel-frames', type=int, default=64)
    parser.add_argument(
        '--frame-strategy',
        choices=['crop_pad', 'resize'],
        default='crop_pad',
    )
    # Her korpus için ayrı mel-frames / frame-strategy bayrakları üret
    # (--cremad-mel-frames, --meld-frame-strategy gibi). Döngüyle üretmek,
    # iki kümenin bayraklarının birbirinden sapmasını engeller.
    for corpus in ('cremad', 'meld'):
        parser.add_argument(f'--{corpus}-mel-frames', type=int)
        parser.add_argument(
            f'--{corpus}-frame-strategy',
            choices=['crop_pad', 'resize'],
        )
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

    # ---------------------------------------------------------------
    # 2) Argüman doğrulama ve türetilmiş ayarlar.
    # ---------------------------------------------------------------
    if args.feature_workers < 1 or args.loader_workers < 0:
        parser.error('Worker counts must be feature>=1 and loader>=0.')
    try:
        feature_configs = build_feature_configs(
            tuple(args.corpora),
            default_frames=args.mel_frames,
            default_strategy=args.frame_strategy,
            frame_overrides={
                'cremad': args.cremad_mel_frames,
                'meld': args.meld_mel_frames,
            },
            strategy_overrides={
                'cremad': args.cremad_frame_strategy,
                'meld': args.meld_frame_strategy,
            },
        )
    except ValueError as error:
        # parser.error hatayı kullanıcıya CLI diliyle gösterir ve programı
        # kullanım bilgisiyle birlikte sonlandırır.
        parser.error(str(error))
    # Epoch üst sınırı verilmemişse moda göre seç: quick modda kısa (8),
    # gerçek deneylerde uzun (60, erken durdurma zaten gereksiz epoch'ları keser).
    max_epochs = args.max_epochs
    if max_epochs is None:
        max_epochs = 8 if args.grid_mode == 'quick' else 60

    # "Tanı" (diagnostic) çalıştırmalar — quick mod veya satır limiti — gerçek
    # deney çıktılarının üzerine yazmasın diye ayrı bir klasöre gider.
    diagnostic = args.limit_per_split is not None or args.grid_mode == 'quick'
    if args.out_root:
        output_root = args.out_root
    elif diagnostic:
        output_root = 'odev3/smoke_outputs'
    else:
        output_root = 'odev3/outputs'

    # ---------------------------------------------------------------
    # 3) Deneyi çalıştır: tüm ağır iş pipeline.run_all içindedir.
    # ---------------------------------------------------------------
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
        feature_configs=feature_configs,
    )

    # ---------------------------------------------------------------
    # 4) Rapor üretimi ve konsol özeti. build_report importu burada, fonksiyon
    #    içinde yapılır ki --skip-report verildiğinde o ağır modül hiç yüklenmesin.
    # ---------------------------------------------------------------
    if not args.skip_report:
        from odev3.build_report import build_report

        build_report(output_root=output_root)
    for corpus, result in results.items():
        print(_result_line(corpus, result))


if __name__ == '__main__':
    main()
