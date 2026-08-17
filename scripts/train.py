"""YAML config'inden model eğitir / değerlendirir; komut satırından ayar ezilebilir.

Bu betik projenin ana giriş kapısıdır. Tasarım deseni: "config dosyası + CLI
override". Bir deneyin TÜM ayarları (model, veri, özellikler, protokol) bir
YAML dosyasında sabit durur — bu, deneyin aylar sonra bile aynen tekrarlanmasını
sağlar. Sık değişen ayarlar (epoch, batch size, deney adı...) ise komut
satırından geçici olarak ezilebilir — bu da hızlı denemeyi pratik kılar.
Böylece tekrarlanabilirlik ile esneklik aynı anda elde edilir.

Örnekler
--------
    # CREMA-D üzerinde klasik MFCC baseline:
    python scripts/train.py --config configs/baseline_cremad.yaml --baseline

    # Log-mel üzerinde CNN, CREMA-D (konuşmacı-bağımsız):
    python scripts/train.py --config configs/cnn_cremad.yaml

    # wav2vec2 transfer öğrenme (`pip install -e .[transfer]` + GPU gerekir):
    python scripts/train.py --config configs/wav2vec2_cremad.yaml

    # CNN için korpus içi + korpuslar arası tam matris:
    python scripts/train.py --config configs/cnn_cremad.yaml --cross-corpus

    # Komut satırından herhangi bir ayarı ez:
    python scripts/train.py --config configs/cnn_cremad.yaml --epochs 5 --experiment quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü import arama yoluna ekle (ser paketi için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Eğitim/değerlendirme mantığının tamamı ser paketindedir; bu betik yalnızca
# argümanları toplayıp doğru fonksiyona yönlendirir (ince CLI katmanı).
from ser.config import Config  # noqa: E402
from ser.train import train_torch, train_baseline  # noqa: E402
from ser.cross_corpus import run_cross_corpus  # noqa: E402
from ser.utils import get_logger  # noqa: E402

log = get_logger("train")


def main():
    """Argümanları çözümler, config'i kurar ve uygun eğitim modunu başlatır."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    # Mod seçici bayraklar: --baseline scikit-learn modelini, --cross-corpus
    # tam korpus matrisini seçer; ikisi de verilmezse PyTorch modeli eğitilir.
    ap.add_argument("--baseline", action="store_true", help="Use the classical MFCC baseline.")
    ap.add_argument("--baseline-kind", default="svm", choices=["svm", "logreg", "rf"])
    ap.add_argument("--cross-corpus", action="store_true",
                    help="Run the within+cross corpus matrix instead of a single run.")
    # Sık kullanılan geçersiz kılmalar: default=None bilinçli — None demek
    # "kullanıcı bu ayarı vermedi, YAML'daki değere dokunma" demek.
    # choices listeleri yazım hatalarını argüman aşamasında yakalar.
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--model", default=None, choices=["cnn", "wav2vec2"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--eval-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--split", default=None, choices=["speaker", "meld_official", "random"])
    args = ap.parse_args()

    # Önce YAML okunur; ardından komut satırında GERÇEKTEN verilen (None
    # olmayan) her değer YAML'daki karşılığının üstüne yazılır. Tek satırlık
    # if'lerin hizalı dizilişi bilinçli: sekizinin de aynı kalıp olduğu bir
    # bakışta görülsün. Korpus listeleri tuple'a çevrilir (değişmez tip).
    cfg = Config.from_yaml(args.config)
    if args.experiment:    cfg.experiment = args.experiment
    if args.model:         cfg.model.name = args.model
    if args.epochs:        cfg.train.epochs = args.epochs
    if args.batch_size:    cfg.train.batch_size = args.batch_size
    if args.manifest:      cfg.data.manifest = args.manifest
    if args.train_corpora: cfg.data.train_corpora = tuple(args.train_corpora)
    if args.eval_corpora:  cfg.data.eval_corpora = tuple(args.eval_corpora)
    if args.split:         cfg.data.split = args.split

    # Üç çalışma modundan birini seç:
    # 1) --cross-corpus: korpus içi + korpuslar arası tüm kombinasyon matrisi;
    # 2) --baseline: klasik scikit-learn modeli (türü --baseline-kind ile);
    # 3) varsayılan: config'te adı geçen PyTorch modeli (cnn / wav2vec2).
    if args.cross_corpus:
        run_cross_corpus(cfg, use_baseline=args.baseline, baseline_kind=args.baseline_kind)
    elif args.baseline:
        train_baseline(cfg, kind=args.baseline_kind)
    else:
        train_torch(cfg)


if __name__ == "__main__":
    main()
