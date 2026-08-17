"""Deney çıktısından rapor tablolarını (markdown) ve bulguları üretir.

    python odev1/build_report.py

`odev1/outputs/...` altındaki sonuçları okuyup:
  - her veri seti için F×P ızgarasının en iyi-K özet tablosunu,
  - test karşılaştırma tablosunu,
  - en iyi yapılandırmaları ve hiperparametre etkisi bulgularını
markdown olarak `odev1/RAPOR_tablolar.md` dosyasına yazar (Doc'a yapıştırmaya hazır).

Bu betik hiçbir model çalıştırmaz; yalnızca `run_experiment.py`'nin ürettiği
JSON/CSV dosyalarını okuyup rapor formatına çevirir. Deney ve raporlama işinin
ayrılması sayesinde tablolar, deneyi yeniden koşturmadan istenildiği kadar
yeniden üretilebilir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Proje kökünü arama yoluna ekle (diğer odev1 betikleriyle aynı kalıp).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- Sabitler ---------------------------------------------------------------
OUT = Path("odev1/outputs")                       # deney çıktılarının okunacağı klasör
CORPORA = ["cremad", "meld"]                      # raporlanacak veri setleri
CORP_TR = {"cremad": "CREMA-D", "meld": "MELD"}   # klasör adı → raporda görünen ad


def _md_table(df: pd.DataFrame) -> str:
    """Bir DataFrame'i markdown tablo metnine çevirir.

    Markdown tablo biçimi: ilk satır başlıklar, ikinci satır `---` ayraçları,
    sonrası veri satırları. pandas'ın kendi to_markdown'u yerine elle kurulması,
    ek bağımlılık (tabulate) gerektirmemesi içindir.
    """
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _fp_best_k(grid: pd.DataFrame) -> pd.DataFrame:
    """Her (F, P) çifti için en iyi validation makro-F1'i veren satırı (yani en iyi K'yı) seçer.

    Ham ızgarada her (F, P) için 8 ayrı K satırı vardır; raporda hepsini basmak
    tabloyu şişirir. Bunun yerine her (F, P) hücresi, kendi en iyi K'sı ile
    özetlenir: `groupby(...).idxmax()` her grupta en yüksek makro-F1'li satırın
    indeksini bulur, `loc` ile o satırlar çekilir. Sütunlar rapor diline
    çevrilir ve başa sıra numarası ("Deney") eklenir.
    """
    idx = grid.groupby(["feature_dim", "pca_dim"])["val_macro_f1"].idxmax()
    best = grid.loc[idx].sort_values(["feature_dim", "val_macro_f1"], ascending=[True, False])
    best = best.rename(columns={"feature_dim": "F", "pca_dim": "P", "K": "K",
                                "val_accuracy": "Doğruluk", "val_macro_f1": "Makro-F1"})
    best.insert(0, "Deney", range(1, len(best) + 1))
    return best[["Deney", "F", "P", "K", "Doğruluk", "Makro-F1"]]


def main():
    """Çıktı dosyalarını okur, markdown rapor bölümlerini kurar ve diske yazar.

    Sıra: (1) her corpus için geçerleme özeti, (2) test karşılaştırma tablosu,
    (3) otomatik bulgular listesi. Eksik çıktısı olan corpus atlanır ki rapor,
    deneylerin yalnızca bir kısmı bitmişken de üretilebilsin.
    """
    lines = []    # raporun satırları burada birikir; en sonda tek seferde yazılır
    results = {}  # corpus → result.json içeriği (test tablosu için saklanır)

    # ---- 1) Her veri setinin geçerleme (validation) özeti -------------------
    for c in CORPORA:
        rj = OUT / c / "result.json"
        vg = OUT / c / "validation_grid.csv"
        if not rj.exists() or not vg.exists():
            print(f"[skip] {c}: outputs yok ({rj})")
            continue
        results[c] = json.loads(rj.read_text(encoding="utf-8"))
        grid = pd.read_csv(vg)
        lines.append(f"### {CORP_TR[c]} — geçerleme (her F×P için en iyi K)\n")
        lines.append(_md_table(_fp_best_k(grid)))
        lines.append("")

    # ---- 2) Test karşılaştırma tablosu (her corpus'un en iyi modeli) --------
    if results:
        comp_rows = []
        for c, r in results.items():
            bc, t = r["best_config"], r["test"]
            comp_rows.append({"Veri seti": CORP_TR[c], "F": bc["feature_dim"], "P": bc["pca_dim"],
                              "K": bc["K"], "Doğruluk": t["accuracy"],
                              "Dengeli doğr.": t["balanced_accuracy"], "Makro-F1": t["macro_f1"]})
        lines.append("### Test karşılaştırma tablosu (her veri setinin en iyisi)\n")
        lines.append(_md_table(pd.DataFrame(comp_rows)))
        lines.append("")

        # ---- 3) Otomatik bulgular: rakamlar cümleye dökülür -----------------
        lines.append("### Bulgular (otomatik özet)\n")
        for c, r in results.items():
            bc, t = r["best_config"], r["test"]
            lines.append(f"- **{CORP_TR[c]}**: en iyi F={bc['feature_dim']}, P={bc['pca_dim']}, "
                         f"K={bc['K']} → test doğruluk {t['accuracy']:.3f}, makro-F1 {t['macro_f1']:.3f}.")
        # Genel en iyi: iki corpus arasında test makro-F1'i yüksek olan seçilir.
        overall = max(results.items(), key=lambda kv: kv[1]["test"]["macro_f1"])
        lines.append(f"- **Genel en iyi**: {CORP_TR[overall[0]]} "
                     f"(makro-F1 {overall[1]['test']['macro_f1']:.3f}). "
                     f"Karmaşıklık matrisi: `odev1/outputs/overall_best_confusion.png`.")

    text = "\n".join(lines)
    out = Path("odev1/RAPOR_tablolar.md")
    out.write_text(text, encoding="utf-8")
    # Konsol için güvenli onay mesajı (Windows cp1254 konsolunda unicode
    # karakterler hata verebildiğinden bilerek diakritiksiz yazılmıştır).
    print(f"[yazildi] {out} ({len(lines)} satir)")


if __name__ == "__main__":
    main()
