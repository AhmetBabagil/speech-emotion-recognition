"""CREMA-D + MELD kayıtlarını tek bir birleşik manifest CSV'sinde toplar.

Manifest, projenin "veri sözleşmesi"dir: her satırı tek bir ses kaydını
tanımlar (dosya yolu, hangi korpus, konuşmacı kimliği, kanonik duygu
etiketi...). Eğitim kodu ham veri klasörlerinin karmaşık düzenini hiç bilmek
zorunda kalmaz; yalnızca bu CSV'yi okur. Böylece iki çok farklı kaynaktan
gelen veri (stüdyo kayıtları CREMA-D ile dizi diyalogları MELD) aynı ortak
formata indirgenir ve tüm deneyler tek bir dosyadan beslenir.

    python scripts/build_manifest.py
    python scripts/build_manifest.py --out data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü import arama yoluna ekle: betik hangi klasörden çalıştırılırsa
# çalıştırılsın `ser` paketi bulunabilsin. (parents[1] = scripts/'in bir üstü.)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import'un sys.path ayarından SONRA gelmesi zorunlu (yoksa `ser` bulunamaz);
# `noqa: E402` linter'ın "import en üstte olmalı" uyarısını bilinçli susturur.
from ser.data.build_manifest import build_manifest  # noqa: E402


def main():
    """Komut satırı argümanlarını okuyup asıl işi `ser` paketine devreder."""
    # description=__doc__: yukarıdaki modül açıklaması --help çıktısında
    # aynen görünür; RawDescriptionHelpFormatter satır sonlarını korur.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Varsayılan yollar, download_data.py betiğinin oluşturduğu klasör
    # düzeniyle birebir aynıdır; veriyi başka bir yere indirdiyseniz bu
    # argümanlarla doğru konumu gösterin.
    ap.add_argument("--cremad-dir", default="data/raw/cremad/AudioWAV")
    ap.add_argument("--meld-audio-root", default="data/raw/meld/audio")
    ap.add_argument("--meld-csv-dir", default=None,
                    help="Auto-detected under data/raw/meld if omitted.")
    ap.add_argument("--out", default="data/processed/manifest.csv")
    args = ap.parse_args()

    # Asıl mantık ser/data/build_manifest.py içinde yaşar; bu betik yalnızca
    # ince bir komut satırı sarmalayıcısıdır. Bu ayrım sayesinde aynı fonksiyon
    # testlerden ya da başka betiklerden de çağrılabilir (kod tekrarı yok).
    build_manifest(
        cremad_dir=args.cremad_dir,
        meld_csv_dir=args.meld_csv_dir,
        meld_audio_root=args.meld_audio_root,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
