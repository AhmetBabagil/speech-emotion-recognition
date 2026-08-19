# Korpus-içi ve korpuslar-arası deney matrisini çalıştırır ve özetler.
#
# Seçilen bir model için dört ayar (2x2 matris) denenir:
# within_cremad        CREMA-D ile eğit -> CREMA-D'de test  (konuşmacı-bağımsız)
# within_meld          MELD ile eğit    -> MELD'de test     (konuşmacı-bağımsız)
# cross_cremad_to_meld CREMA-D ile eğit -> MELD'de test     (alan kayması / domain shift)
# cross_meld_to_cremad MELD ile eğit    -> CREMA-D'de test  (alan kayması / domain shift)
#
# Neden bu matris? Köşegendeki (within) skorlar modelin "kendi evinde" ne kadar iyi olduğunu, köşegen dışındaki (cross) skorlar ise başka kayıt koşullarına / konuşmacılara ne kadar GENELLEDİĞİNİ gösterir. Aradaki düşüş, alan kaymasının bedelidir ve proje önerisindeki cross-corpus genelleme analizinin özüdür.
#
# Çıktılar: ``outputs/<exp>_crosscorpus/summary.csv`` + bir makro-F1 ısı haritası.

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import CORPUS_CREMAD, CORPUS_MELD
from .utils import get_logger, ensure_dir

log = get_logger(__name__)

# Her satır bir deney ayarı: (isim, eğitim korpusları, değerlendirme korpusları).
# Tuple'lar tek elemanlı bile olsa tuple'dır — DataConfig alanlarının tipiyle uyumlu.
SETTINGS = [
    ("within_cremad",        (CORPUS_CREMAD,), (CORPUS_CREMAD,)),
    ("within_meld",          (CORPUS_MELD,),   (CORPUS_MELD,)),
    ("cross_cremad_to_meld", (CORPUS_CREMAD,), (CORPUS_MELD,)),
    ("cross_meld_to_cremad", (CORPUS_MELD,),   (CORPUS_CREMAD,)),
]


def run_cross_corpus(cfg: Config, use_baseline: bool = False, baseline_kind: str = "svm") -> pd.DataFrame:
    # Dört ayarın hepsini sırayla eğitip test eder, sonuç tablosunu döndürür.
    #
    # ``use_baseline=True`` verilirse derin model yerine klasik MFCC taban modeli (sklearn) kullanılır — aynı matris ucuz ve hızlı şekilde CPU'da koşulabilir.
    # İçeride import: ser.train'in ağır bağımlılıkları (torch) yalnızca
    # gerçekten deney koşulacağında yüklensin; ayrıca döngüsel import riski azalır.
    from .train import train_torch, train_baseline

    base_exp = cfg.experiment
    out_root = ensure_dir(Path(cfg.output_dir) / f"{base_exp}_crosscorpus")
    rows = []
    for name, train_corpora, eval_corpora in SETTINGS:
        # deepcopy şart: cfg'yi yerinde değiştirseydik bir ayarın değişikliği
        # sonraki ayarlara sızardı. Her ayar kendi bağımsız kopyasını alır.
        c = copy.deepcopy(cfg)
        c.experiment = f"{base_exp}_{name}"   # her ayar kendi çıktı klasörüne yazar
        c.data.train_corpora = train_corpora
        c.data.eval_corpora = eval_corpora
        log.info("=== Cross-corpus setting: %s (train=%s eval=%s) ===",
                 name, train_corpora, eval_corpora)
        try:
            if use_baseline:
                m = train_baseline(c, kind=baseline_kind)
            else:
                m = train_torch(c)
        except Exception as e:  # tek bir ayarın çökmesi tüm matrisi öldürmesin
            log.exception("Setting %s failed: %s", name, e)
            continue
        # Her ayarın özet metriklerini tabloya ekle; "+" ile birleştirme,
        # ileride çok-korpuslu eğitim (örn. "cremad+meld") desteklenirse de okunur.
        rows.append({
            "setting": name,
            "train": "+".join(train_corpora),
            "eval": "+".join(eval_corpora),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        })

    # Özet: hem CSV (tablo işlemek için) hem JSON (programatik okuma için).
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False)
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _plot_matrix(df, out_root / "macro_f1_matrix.png")
    log.info("Cross-corpus summary:\n%s", df.to_string(index=False))
    return df


def _plot_matrix(df: pd.DataFrame, out_path: Path) -> None:
    # 2x2 makro-F1 matrisini ısı haritası olarak çizer.
    #
    # Satırlar eğitim korpusu, sütunlar test korpusudur: köşegen = within, köşegen dışı = cross. Tek bakışta genelleme kaybı görülür.
    if df.empty:
        return  # hiçbir ayar başarılı olmadıysa çizecek bir şey yok
    import matplotlib
    matplotlib.use("Agg")  # GUI'siz (dosyaya) çizim arka ucu
    import matplotlib.pyplot as plt
    import seaborn as sns

    corpora = [CORPUS_CREMAD, CORPUS_MELD]
    # NaN ile başlat: başarısız/eksik ayarların hücresi boş görünsün.
    mat = np.full((2, 2), np.nan)
    for _, r in df.iterrows():
        # Yalnızca tek-korpuslu satırları matrise yerleştir ("a+b" gibi
        # birleşik eğitimler 2x2 gösterime sığmaz).
        if r["train"] in corpora and r["eval"] in corpora:
            mat[corpora.index(r["train"]), corpora.index(r["eval"])] = r["macro_f1"]
    plt.figure(figsize=(5, 4))
    # vmin/vmax=0..1: renk skalası koşudan koşuya değişmesin, karşılaştırılabilir olsun.
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=[f"test:{c}" for c in corpora],
                yticklabels=[f"train:{c}" for c in corpora], vmin=0, vmax=1)
    plt.title("Macro-F1: within (diagonal) vs cross-corpus (off-diagonal)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
