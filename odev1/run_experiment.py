"""Ödev 1 KNN deneyini iki veri setinde de çalıştırır (deneyin giriş noktası).

Önkoşul: önce `python odev1/extract.py` ile wav2vec2 öznitelikleri çıkarılmış
olmalıdır; bu betik ses dosyalarını değil, o adımın ürettiği cache'i kullanır.

Kullanım örnekleri:

    python odev1/run_experiment.py
    python odev1/run_experiment.py --corpora cremad      # yalnızca tek veri seti
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü modül arama yoluna ekle: dosya doğrudan çalıştırıldığında da
# `odev1.knn_pipeline` paketi bulunabilsin.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Deneyin tamamı knn_pipeline.run_all içinde; burası yalnızca CLI katmanı.
from odev1.knn_pipeline import run_all  # noqa: E402


def main():
    """Komut satırı seçeneklerini okuyup Ödev 1 KNN deneylerini `run_all` ile başlatır.

    Argümanların anlamları:
      * ``--manifest``  : klip listesi ve etiketleri içeren CSV,
      * ``--cache-dir`` : extract.py'nin ürettiği .npy öznitelik önbelleği,
      * ``--out-root``  : sonuç JSON/CSV/PNG dosyalarının yazılacağı kök klasör,
      * ``--corpora``   : hangi veri setleri çalışacak (cremad, meld veya ikisi).
    """
    # description=__doc__: modül açıklaması --help çıktısı olarak da kullanılır.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--out-root", default="odev1/outputs")
    # nargs="+" birden çok değer alabilmeyi sağlar; choices yazım hatasını daha
    # deney başlamadan yakalar.
    ap.add_argument("--corpora", nargs="+", default=["cremad", "meld"], choices=["cremad", "meld"])
    args = ap.parse_args()
    run_all(args.manifest, args.cache_dir, args.out_root, tuple(args.corpora))


if __name__ == "__main__":
    main()
