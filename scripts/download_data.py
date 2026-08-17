"""CREMA-D ve/veya MELD veri kümelerini indirir; MELD kliplerinin sesini ayıklar.

İki korpus farklı biçimlerde gelir, o yüzden iki ayrı indirme yolu vardır:

* CREMA-D: oyuncuların stüdyoda seslendirdiği kısa WAV dosyaları — indirilir
  ve doğrudan kullanılabilir.
* MELD: "Friends" dizisinden video klipleri, dev bir tar.gz arşivi olarak
  gelir; indirme sonrası her replik (utterance) ffmpeg ile videodan ayrı bir
  ses dosyasına çıkarılmak zorundadır (ekstra adım bundan dolayı).

Örnekler
--------
    # Her şey (CREMA-D + tam MELD, ~10 GB indirme + ffmpeg ile ses ayıklama):
    python scripts/download_data.py --datasets cremad meld

    # Yalnızca CREMA-D (hızlı, ~1-2 GB):
    python scripts/download_data.py --datasets cremad

    # MELD'i split başına küçük bir ses limitiyle indir (hızlı deneme için):
    python scripts/download_data.py --datasets meld --meld-limit 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü import arama yoluna ekle; `ser` paketi böylece betiğin
# çalıştırıldığı klasörden bağımsız olarak bulunur.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Asıl indirme mantığı ser/data altında yaşar; bu betik yalnızca komut satırı
# arayüzü sağlar. noqa: E402 = "import en üstte değil" uyarısını bilinçli sustur.
from ser.data.download_cremad import download_cremad  # noqa: E402
from ser.data.download_meld import download_meld, extract_meld_audio  # noqa: E402
from ser.utils import get_logger  # noqa: E402

# print yerine proje genelindeki ortak logger: zaman damgalı, tutarlı biçimli.
log = get_logger("download")


def main():
    """Argümanları çözümler ve seçilen veri kümelerini sırasıyla indirir."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # nargs="+" birden fazla değere izin verir ("--datasets cremad meld");
    # choices yazım hatalarını daha argüman aşamasında yakalar.
    ap.add_argument("--datasets", nargs="+", default=["cremad", "meld"],
                    choices=["cremad", "meld"])
    ap.add_argument("--data-root", default="data/raw")
    # CREMA-D için iki kaynak var: Hugging Face aynası (varsayılan, güvenilir)
    # ve orijinal GitHub deposu (git-lfs gerektirir, daha kırılgan).
    ap.add_argument("--cremad-method", default="hf", choices=["hf", "lfs"],
                    help="'hf' = Hugging Face mirror (reliable); 'lfs' = GitHub git-lfs.")
    # MELD'in tamamını ayıklamak saatler sürebilir; --meld-limit ile her
    # split'ten yalnızca N kayıt ayıklayarak boru hattı hızla test edilebilir.
    ap.add_argument("--meld-limit", type=int, default=None,
                    help="Cap audio extraction to N utterances per split (testing).")
    # Varsayılan davranış arşivi silmektir (10 GB yer kaplar); bu bayrak
    # yeniden ayıklama ihtimaline karşı arşivi saklamayı seçer.
    ap.add_argument("--meld-keep-archive", action="store_true",
                    help="Keep the 10 GB MELD.Raw.tar.gz after extraction.")
    args = ap.parse_args()
    root = Path(args.data_root)

    # Her veri kümesi yalnızca istenmişse indirilir; ikisi bağımsız adımlardır.
    if "cremad" in args.datasets:
        log.info("=== CREMA-D ===")
        download_cremad(str(root / "cremad"), method=args.cremad_method)

    if "meld" in args.datasets:
        log.info("=== MELD ===")
        # download_meld arşivi indirir/açar ve etiket CSV'lerinin klasörünü
        # döndürür; extract_meld_audio o CSV'lerdeki her replik için videodan
        # ses dosyası üretir (bu adım ffmpeg'e ihtiyaç duyar).
        csv_dir = download_meld(str(root / "meld"), keep_archive=args.meld_keep_archive)
        extract_meld_audio(csv_dir, str(root / "meld"), limit=args.meld_limit)

    # Kullanıcıya boru hattındaki bir sonraki adımı hatırlat.
    log.info("Done. Next: python scripts/build_manifest.py")


if __name__ == "__main__":
    main()
