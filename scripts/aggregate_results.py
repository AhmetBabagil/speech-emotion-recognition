"""Tüm outputs/<deney>/test_summary.json dosyalarını tek bir sonuç tablosunda toplar.

Neden böyle bir betik var? Projedeki her deney (baseline, CNN, cross-corpus...)
kendi klasörüne küçük bir ``test_summary.json`` bırakır. Rapor yazarken bu
dağınık JSON dosyalarını elle tek tek açmak hem yorucu hem de hataya açıktır.
Bu betik hepsini otomatik gezip iki dosya üretir:

* ``outputs/results.csv`` -> Excel/pandas ile açılabilen ham tablo
* ``outputs/results.md``  -> rapora doğrudan yapıştırılabilen Markdown tablo

Kullanım:

    python scripts/aggregate_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Proje kökünü (scripts/ klasörünün bir üstünü) Python'un import arama yoluna
# ekliyoruz. Böylece betik hangi klasörden çalıştırılırsa çalıştırılsın
# projedeki paketler sorunsuz bulunur. `__file__` bu dosyanın yolu;
# `parents[0]` içinde bulunduğu scripts/ klasörü, `parents[1]` ise proje kökü.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import'un dosyanın en üstünde olmaması (sys.path satırından sonra gelmesi)
# bilinçli bir tercih; `noqa: E402` yorumu linter'ın bu stil uyarısını susturur.
import pandas as pd  # noqa: E402


def main():
    """Tüm deney özetlerini bulur, tek tabloya çevirir ve diske yazar."""
    # Tek argüman: --outputs ile farklı bir çıktı klasörü gösterilebilir.
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    args = ap.parse_args()
    root = Path(args.outputs)

    # outputs/*/test_summary.json desenine uyan her dosyayı gez. `sorted`
    # sayesinde tablo her çalıştırmada aynı sırayla üretilir (deterministik
    # çıktı, iki çalıştırmanın diff'ini karşılaştırmayı kolaylaştırır).
    rows = []
    for summary in sorted(root.glob("*/test_summary.json")):
        with open(summary, encoding="utf-8") as f:
            d = json.load(f)
        # Her deneyden rapor için önemli 4 metriği çekiyoruz. `d.get(..., nan)`
        # tercihi bilinçli: eski/yarım kalmış bir deneyde alan eksikse betik
        # çökmesin, tabloya NaN düşsün. round(..., 4) yalnızca okunabilirlik
        # için; metriğin kendisini değiştirmez. Deney adı = klasör adı.
        rows.append({
            "experiment": summary.parent.name,
            "accuracy": round(d.get("accuracy", float("nan")), 4),
            "balanced_accuracy": round(d.get("balanced_accuracy", float("nan")), 4),
            "macro_f1": round(d.get("macro_f1", float("nan")), 4),
            "weighted_f1": round(d.get("weighted_f1", float("nan")), 4),
        })

    # Hiç özet bulunamadıysa boş dosya yazmak yanıltıcı olurdu; bunun yerine
    # kullanıcıyı bilgilendirip çıkıyoruz (muhtemelen henüz deney koşulmadı).
    if not rows:
        print(f"No test_summary.json found under {root}/. Run some experiments first.")
        return

    # Satırları DataFrame'e çevirip deney adına göre sırala: baseline_* ve
    # cnn_* deneyleri tabloda alfabetik olarak gruplanmış görünür.
    df = pd.DataFrame(rows).sort_values("experiment")
    df.to_csv(root / "results.csv", index=False)
    # Markdown tablo, pandas'ın `to_markdown`ına dayanır; o da arka planda
    # `tabulate` paketini ister. Paket kurulu değilse programı düşürmek yerine
    # düz metin tabloya geri düşüyoruz (zarif geri çekilme / graceful fallback).
    try:
        md = df.to_markdown(index=False)  # `tabulate` paketi gerektirir
    except ImportError:
        md = df.to_string(index=False)
    (root / "results.md").write_text(md, encoding="utf-8")
    # Tabloyu ekrana da basıyoruz ki dosya açmadan sonuçlara göz atılabilsin.
    print(df.to_string(index=False))
    print(f"\nWrote {root/'results.csv'} and {root/'results.md'}")


if __name__ == "__main__":
    main()
