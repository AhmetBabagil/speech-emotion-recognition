"""Manifest'teki her klip için Wav2Vec2 özniteliklerini çıkarır ve önbelleğe yazar (Ödev 1).

Kullanım örnekleri:

    python odev1/extract.py
    python odev1/extract.py --manifest data/processed/manifest.csv

Bu betik, projenin tek pahalı adımıdır: yaklaşık 19.5 bin klip, donmuş (ağırlıkları
güncellenmeyen) Wav2Vec2 modelinden ileri yönde (forward) geçirilir. İşlem
kesintiye uğrarsa kaldığı yerden devam edebilir — önbellekte (cache) zaten var
olan klipler atlanır. Sonraki KNN aşaması ses dosyalarına hiç dokunmaz; yalnızca
burada üretilen numpy vektörlerini okur. Böylece hiperparametre denemeleri
saniyeler içinde tekrarlanabilir.
"""

# `annotations` içe aktarımı, tip ipuçlarının çalışma anında değil metin olarak
# değerlendirilmesini sağlar (ileriye dönük uyumluluk için standart bir kalıp).
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kök dizinini modül arama yoluna ekliyoruz. Böylece bu dosya
# `python odev1/extract.py` şeklinde doğrudan çalıştırıldığında da
# `odev1.features_w2v` gibi paket içi içe aktarımlar sorunsuz bulunur.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Asıl işi yapan fonksiyon features_w2v modülünde; bu dosya yalnızca ince bir
# komut satırı (CLI) sarmalayıcısıdır. (# noqa: E402 → import'un dosyanın en
# üstünde olmamasına dair stil uyarısını susturur; sys.path ayarı önce gelmeli.)
from odev1.features_w2v import extract_all  # noqa: E402


def main():
    """Komut satırı argümanlarını okuyup kaldığı yerden devam edebilen Wav2Vec2 önbellekleme işini başlatır.

    Argümanların anlamları:
      * ``--manifest``    : tüm kliplerin yolunu ve etiketini listeleyen CSV dosyası,
      * ``--cache-dir``   : üretilen .npy vektörlerinin kaydedileceği klasör,
      * ``--model``       : kullanılacak HuggingFace Wav2Vec2 modeli,
      * ``--max-seconds`` : bir klipten işlenecek en fazla saniye (uzun klipler ortadan kırpılır),
      * ``--shard`` / ``--num-shards`` : işi birden çok sürece bölmek için; her süreç
        eksik kliplerin yalnızca kendine düşen dilimini işler (paralellik).
    """
    # description=__doc__ sayesinde `--help` çıktısında yukarıdaki modül
    # açıklaması aynen görünür; RawDescriptionHelpFormatter satır sonlarını korur.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    ap.add_argument("--cache-dir", default="odev1/cache/w2v")
    ap.add_argument("--model", default="facebook/wav2vec2-base")
    ap.add_argument("--max-seconds", type=float, default=6.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()
    # Tüm ağır iş extract_all içinde: manifesti okur, eksik klipleri bulur,
    # her biri için havuzlanmış Wav2Vec2 vektörünü üretip diske kaydeder.
    extract_all(args.manifest, cache_dir=args.cache_dir, model_name=args.model,
                max_seconds=args.max_seconds, shard=args.shard, num_shards=args.num_shards)


if __name__ == "__main__":
    main()
