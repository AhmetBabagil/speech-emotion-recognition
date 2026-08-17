"""Ödev 2 (Project Assignment 2) deneylerini komut satırından çalıştırır.

Kullanım örnekleri:

    python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --grid-mode report
    python odev2/run_experiment.py --manifest odev1/manifest_subset.csv --quick --corpora cremad

Önkoşul: Ödev 1'in öznitelik önbelleği hazır olmalıdır (python odev1/extract.py);
bu betik Wav2Vec2 çalıştırmaz, yalnızca cache'lenmiş vektörleri okur.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü modül arama yoluna ekle: dosya doğrudan çalıştırıldığında da
# `odev2.model_pipeline` paketi bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deneyin tamamı model_pipeline.run_all içinde; burası yalnızca CLI katmanı.
# MODEL_SPECS, --models seçeneğinin geçerli değerlerini üretmek için gerekli.
from odev2.model_pipeline import MODEL_SPECS, run_all  # noqa: E402


def main() -> None:
    """Komut satırı argümanlarını okuyup Ödev 2 deney yöneticisi `run_all` fonksiyonunu çağırır.

    Argümanların anlamları:
      * ``--manifest``     : klip listesi ve etiketleri içeren CSV,
      * ``--cache-dir``    : Ödev 1'de üretilen .npy öznitelik önbelleği,
      * ``--out-root``     : Ödev 2 çıktılarının yazılacağı kök klasör,
      * ``--knn-out-root`` : karşılaştırma tablosuna eklenecek KNN sonuçlarının yeri,
      * ``--corpora``      : çalışacak veri setleri (cremad, meld veya ikisi),
      * ``--models``       : yalnızca belirli model ailelerini koşmak için (boşsa hepsi),
      * ``--grid-mode``    : arama genişliği (quick / report / full),
      * ``--quick``        : --grid-mode quick için kısayol bayrağı.
    """
    # description=__doc__: modülün açıklaması --help çıktısında aynen görünür.
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--out-root", default="odev2/outputs")
    ap.add_argument("--knn-out-root", default="odev1/outputs")
    ap.add_argument("--corpora", nargs="+", default=["cremad", "meld"], choices=["cremad", "meld"])
    # choices, MODEL_SPECS'ten türetilir: yeni model eklenirse CLI otomatik öğrenir.
    ap.add_argument("--models", nargs="+", choices=[s.name for s in MODEL_SPECS])
    ap.add_argument("--grid-mode", choices=["quick", "report", "full"], default="report")
    ap.add_argument("--quick", action="store_true", help="Alias for --grid-mode quick.")
    args = ap.parse_args()

    # argparse listeleri tuple'a çevrilir (run_all imzası tuple bekler);
    # --models verilmediyse None geçilir ve tüm modeller çalışır.
    run_all(
        manifest=args.manifest,
        cache_dir=args.cache_dir,
        out_root=args.out_root,
        corpora=tuple(args.corpora),
        quick=args.quick,
        grid_mode=args.grid_mode,
        model_keys=tuple(args.models) if args.models else None,
        knn_out_root=args.knn_out_root,
    )


if __name__ == "__main__":
    main()
