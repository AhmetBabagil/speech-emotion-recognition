"""Tüm deney takımını uçtan uca çalıştırır ve sonuçları tek tabloda toplar.

Verinin indirilmiş ve manifest'in kurulmuş olduğunu varsayar (yani önce
download_data.py + build_manifest.py çalıştırılmış olmalı). Sırasıyla:
  * MFCC baseline (CREMA-D, MELD)
  * Korpus içi CNN (CREMA-D, MELD)
  * Korpuslar arası CNN matrisi (CREMA-D <-> MELD)
koşar, en sonda outputs/results.csv dosyasını üretir. Amaç: raporun bütün
sayılarını TEK komutla, aynı sırayla ve tekrarlanabilir şekilde üretebilmek.

    python scripts/run_all.py                 # her şey
    python scripts/run_all.py --skip-baseline # yalnızca CNN
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü import arama yoluna ekle (ser paketi için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.train import train_torch, train_baseline  # noqa: E402
from ser.cross_corpus import run_cross_corpus  # noqa: E402
from ser.utils import get_logger  # noqa: E402

log = get_logger("run_all")


def main():
    """Deneyleri (baseline -> CNN -> çapraz korpus) sırayla koşturur."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/processed/manifest.csv")
    # --skip-* bayrakları uzun süren bölümleri atlamaya yarar (ör. yalnızca
    # CNN'leri yeniden koşmak istediğinizde). --epochs tüm deneylerin epoch
    # sayısını birden ezmek için pratik bir kısayoldur (hızlı prova koşusu).
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-cross", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    def base_cfg(name):
        """configs/<name>.yaml'ı yükleyip ortak CLI ayarlarını üstüne uygular.

        Her deneyde aynı iki dokunuş gerekiyor (manifest yolu + isteğe bağlı
        epoch); bu küçük yardımcı fonksiyon o tekrarları tek yerde toplar.
        """
        cfg = Config.from_yaml(f"configs/{name}.yaml")
        cfg.data.manifest = args.manifest
        if args.epochs:
            cfg.train.epochs = args.epochs
        return cfg

    if not args.skip_baseline:
        log.info("### Baseline: CREMA-D ###")
        train_baseline(base_cfg("baseline_cremad"), kind="svm")
        # MELD baseline'ı için ayrı bir YAML dosyası tutmuyoruz: CREMA-D
        # config'ini yükleyip yalnızca deney adını ve korpus seçimini
        # değiştirmek yetiyor (tek fark bu; diğer tüm ayarlar ortak).
        c = base_cfg("baseline_cremad")
        c.experiment = "baseline_meld"
        # MELD'de de konuşmacı-bağımsız protokol kullanılıyor (CNN ile aynı
        # protokol); böylece baseline ve CNN'in MELD sayıları doğrudan
        # karşılaştırılabilir olur — protokol farkı sonucu çarpıtmaz.
        c.data.train_corpora = ("meld",); c.data.eval_corpora = ("meld",)
        log.info("### Baseline: MELD ###")
        train_baseline(c, kind="svm")

    # Korpus içi CNN deneyleri: her korpusun kendi config dosyası var.
    log.info("### CNN within: CREMA-D ###")
    train_torch(base_cfg("cnn_cremad"))
    log.info("### CNN within: MELD ###")
    train_torch(base_cfg("cnn_meld"))

    if not args.skip_cross:
        # Çapraz korpus matrisi: bir korpusta eğitip DİĞERİNDE test etmek,
        # modelin kayıt koşullarını mı yoksa gerçekten duyguyu mu öğrendiğini
        # gösterir (genelleme testi). Hem baseline hem CNN için koşuyoruz ki
        # "derin model çaprazda daha mı dayanıklı?" sorusu cevaplanabilsin.
        log.info("### Baseline cross-corpus matrix ###")
        bc = base_cfg("baseline_cremad")
        bc.experiment = "baseline"
        run_cross_corpus(bc, use_baseline=True, baseline_kind="logreg")
        log.info("### CNN cross-corpus matrix ###")
        cc = base_cfg("cnn_cremad")
        cc.experiment = "cnn"
        run_cross_corpus(cc)

    # Son adım: kardeş betik aggregate_results.py'yi ayrı bir süreç olarak
    # çağırıp tüm test_summary.json'ları results.csv/md'ye topla.
    # `sys.executable` = şu an çalışan Python'un kendisi (doğru sanal ortam
    # garantisi); `with_name` = bu dosyayla aynı klasördeki kardeş dosyanın yolu.
    log.info("Aggregating results ...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name("aggregate_results.py"))])


if __name__ == "__main__":
    main()
