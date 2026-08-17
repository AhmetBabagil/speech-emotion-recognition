"""Gözetimsiz kümeleme + yarı-gözetimli (azaltılmış etiket) analiz betiği.

Proje önerisinde söz verilen bileşeni hayata geçirir. MFCC-istatistik
özellikleri üzerinde üç ayrı analiz koşar:

1. Kümeleme (cluster): etiketlere hiç bakmadan özellik uzayında doğal
   gruplar var mı, bu gruplar duygularla örtüşüyor mu?
2. Etiket verimliliği (label-eff): etiketlerin yalnızca %x'i ile eğitilirse
   başarı nasıl değişir? (etiketlemenin pahalı olduğu gerçek dünyaya dair soru)
3. Self-training (self-train): model, etiketsiz veride kendinden emin olduğu
   tahminleri "sahte etiket" olarak eğitim setine katarsa kazanç sağlar mı?

Önbelleğe alınmış özellikleri yeniden kullandığı için CPU'da bile hızlıdır.

Örnekler
--------
    # CREMA-D üzerinde tam analiz (kümeleme + etiket verimliliği + self-training):
    python scripts/semisupervised.py --config configs/baseline_cremad.yaml

    # Aynısı MELD üzerinde:
    python scripts/semisupervised.py --config configs/baseline_cremad.yaml \
        --experiment semisup_meld --train-corpora meld --eval-corpora meld
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Proje kökünü import arama yoluna ekle (ser paketi için).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Analizlerin asıl gerçekleştirimi ser/semisupervised.py içindedir; buradan
# yalnızca seçilip çağrılırlar (betik = ince komut satırı katmanı).
from ser.config import Config  # noqa: E402
from ser.semisupervised import run_all, cluster_analysis, label_efficiency, self_training  # noqa: E402
from ser.utils import ensure_dir  # noqa: E402


def main():
    """Config'i yükler, CLI ile ezer ve istenen analiz(ler)i çalıştırır."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Baseline config'i başlangıç noktası olarak kullanılır: aynı özellik
    # ayarları ve aynı split protokolü sayesinde buradaki sonuçlar gözetimli
    # deneylerle adil biçimde karşılaştırılabilir.
    ap.add_argument("--config", default="configs/baseline_cremad.yaml")
    ap.add_argument("--experiment", default="semisupervised_cremad")
    # --only ile üç analizden yalnızca biri koşulabilir (hata ayıklarken ya da
    # tek bir grafiği yeniden üretirken zaman kazandırır); verilmezse üçü de koşar.
    ap.add_argument("--only", choices=["cluster", "label-eff", "self-train"], default=None,
                    help="Run only one analysis instead of all three.")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    ap.add_argument("--eval-corpora", nargs="+", default=None, choices=["cremad", "meld"])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.experiment = args.experiment
    # Komut satırında GERÇEKTEN verilen (None olmayan) değerler YAML'daki
    # ayarları geçersiz kılar; verilmeyenlere dokunulmaz. Tek satırlık if'ler
    # bilerek hizalı: üçünün de aynı kalıp olduğu bir bakışta görülsün.
    if args.manifest:      cfg.data.manifest = args.manifest
    if args.train_corpora: cfg.data.train_corpora = tuple(args.train_corpora)
    if args.eval_corpora:  cfg.data.eval_corpora = tuple(args.eval_corpora)

    # Çıktı klasörü outputs/<deney_adı>; yoksa oluşturulur. --only verildiyse
    # ilgili tek analiz bu klasöre yazar; verilmediyse run_all üçünü birden
    # kendi düzeniyle koşturur (out_dir'i kendi içinde türetir).
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    if args.only == "cluster":
        cluster_analysis(cfg, out_dir)
    elif args.only == "label-eff":
        label_efficiency(cfg, out_dir)
    elif args.only == "self-train":
        self_training(cfg, out_dir)
    else:
        run_all(cfg)


if __name__ == "__main__":
    main()
