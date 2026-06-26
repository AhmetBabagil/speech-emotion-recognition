"""Deney çıktısından rapor tablolarını (markdown) ve bulguları üretir.

    python odev1/build_report.py

`odev1/outputs/...` altındaki sonuçları okuyup:
  - her veri seti için F×P ızgarasının en iyi-K özet tablosunu,
  - test karşılaştırma tablosunu,
  - en iyi yapılandırmaları ve hiperparametre etkisi bulgularını
markdown olarak `odev1/RAPOR_tablolar.md` dosyasına yazar (Doc'a yapıştırmaya hazır).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("odev1/outputs")
CORPORA = ["cremad", "meld"]
CORP_TR = {"cremad": "CREMA-D", "meld": "MELD"}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _fp_best_k(grid: pd.DataFrame) -> pd.DataFrame:
    """For each (F, P) pick the row with the best validation macro-F1 (best K)."""
    idx = grid.groupby(["feature_dim", "pca_dim"])["val_macro_f1"].idxmax()
    best = grid.loc[idx].sort_values(["feature_dim", "val_macro_f1"], ascending=[True, False])
    best = best.rename(columns={"feature_dim": "F", "pca_dim": "P", "K": "K",
                                "val_accuracy": "Doğruluk", "val_macro_f1": "Makro-F1"})
    best.insert(0, "Deney", range(1, len(best) + 1))
    return best[["Deney", "F", "P", "K", "Doğruluk", "Makro-F1"]]


def main():
    lines = []
    results = {}
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

    # test comparison
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

        # findings
        lines.append("### Bulgular (otomatik özet)\n")
        for c, r in results.items():
            bc, t = r["best_config"], r["test"]
            lines.append(f"- **{CORP_TR[c]}**: en iyi F={bc['feature_dim']}, P={bc['pca_dim']}, "
                         f"K={bc['K']} → test doğruluk {t['accuracy']:.3f}, makro-F1 {t['macro_f1']:.3f}.")
        overall = max(results.items(), key=lambda kv: kv[1]["test"]["macro_f1"])
        lines.append(f"- **Genel en iyi**: {CORP_TR[overall[0]]} "
                     f"(makro-F1 {overall[1]['test']['macro_f1']:.3f}). "
                     f"Karmaşıklık matrisi: `odev1/outputs/overall_best_confusion.png`.")

    text = "\n".join(lines)
    out = Path("odev1/RAPOR_tablolar.md")
    out.write_text(text, encoding="utf-8")
    # Console-safe confirmation (avoid Windows cp1254 errors on unicode chars).
    print(f"[yazildi] {out} ({len(lines)} satir)")


if __name__ == "__main__":
    main()
