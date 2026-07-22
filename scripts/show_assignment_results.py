"""Print the committed Assignment 1/2 experiment evidence without changing it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ser.config import Config  # noqa: E402
from ser.data import prepare_splits  # noqa: E402

MANIFEST = ROOT / "odev1" / "manifest_subset.csv"
ODEV1_OUT = ROOT / "odev1" / "outputs"
ODEV2_OUT = ROOT / "odev2" / "outputs"
CORPORA = ("cremad", "meld")


def heading(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{title}\n{bar}")


def show_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("(veri yok)")
        return
    with pd.option_context(
        "display.max_rows",
        100,
        "display.max_columns",
        30,
        "display.width",
        220,
        "display.max_colwidth",
        70,
    ):
        print(frame.to_string(index=False))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Beklenen cikti bulunamadi: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def split_for(df: pd.DataFrame, corpus: str):
    cfg = Config()
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = "speaker"
    return prepare_splits(df, cfg.data, seed=42)


def show_data_statistics() -> None:
    heading("VERI VE SPLIT ISTATISTIKLERI")
    df = pd.read_csv(MANIFEST)

    overview = (
        df.groupby("corpus")
        .agg(kayit=("path", "size"), konusmaci=("speaker", "nunique"))
        .reset_index()
    )
    show_table(overview)

    heading("SINIF DAGILIMI")
    distribution = pd.crosstab(df["corpus"], df["emotion"]).reset_index()
    show_table(distribution)

    heading("SPEAKER-INDEPENDENT TRAIN / VALIDATION / TEST")
    rows = []
    for corpus in CORPORA:
        train_df, val_df, test_df = split_for(df, corpus)
        folds = {"train": train_df, "validation": val_df, "test": test_df}
        for name, part in folds.items():
            rows.append(
                {
                    "corpus": corpus,
                    "fold": name,
                    "kayit": len(part),
                    "konusmaci": part["speaker"].nunique(),
                }
            )

        speaker_sets = {name: set(part["speaker"].astype(str)) for name, part in folds.items()}
        overlaps = {
            "train-validation": len(speaker_sets["train"] & speaker_sets["validation"]),
            "train-test": len(speaker_sets["train"] & speaker_sets["test"]),
            "validation-test": len(speaker_sets["validation"] & speaker_sets["test"]),
        }
        print(f"{corpus} konusmaci kesisimleri: {overlaps}")
    show_table(pd.DataFrame(rows))


def show_assignment_1() -> None:
    heading("ODEV 1 - KNN FINAL TEST SONUCLARI")
    result_rows = []
    for corpus in CORPORA:
        result = read_json(ODEV1_OUT / corpus / "result.json")
        best = result["best_config"]
        test = result["test"]
        result_rows.append(
            {
                "corpus": corpus,
                "F": best["feature_dim"],
                "PCA": best["pca_dim"],
                "K": best["K"],
                "val_macro_f1": best["val_macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
            }
        )
    show_table(pd.DataFrame(result_rows))

    heading("ODEV 1 - VALIDATION GRID KAPSAMI VE EN IYI 5")
    for corpus in CORPORA:
        path = ODEV1_OUT / corpus / "validation_grid.csv"
        grid = pd.read_csv(path).sort_values("val_macro_f1", ascending=False)
        print(f"\n{corpus}: {len(grid)} kombinasyon | {path.relative_to(ROOT)}")
        show_table(grid.head(5))


def result_params(result: dict) -> str:
    return json.dumps(result["best_config"]["params"], sort_keys=True, ensure_ascii=True)


def show_assignment_2() -> None:
    heading("ODEV 2 - MODEL FINAL TEST SONUCLARI (KNN DAHIL)")
    comparison_path = ODEV2_OUT / "test_comparison_with_knn.csv"
    comparison = pd.read_csv(comparison_path).sort_values(
        ["corpus", "test_macro_f1"], ascending=[True, False]
    )
    columns = [
        "corpus",
        "model",
        "feature_dim",
        "pca_dim",
        "params",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    show_table(comparison[columns])

    heading("ODEV 2 - VALIDATION GRID KAPSAMI VE SECILEN AYAR")
    coverage_rows = []
    for corpus in CORPORA:
        for grid_path in sorted((ODEV2_OUT / corpus).glob("*_validation_grid.csv")):
            model_key = grid_path.name.removesuffix("_validation_grid.csv")
            result = read_json(ODEV2_OUT / corpus / f"{model_key}_result.json")
            grid = pd.read_csv(grid_path)
            coverage_rows.append(
                {
                    "corpus": corpus,
                    "model": result["model"],
                    "kombinasyon": len(grid),
                    "feature_dim": result["best_config"]["feature_dim"],
                    "PCA": result["best_config"]["pca_dim"],
                    "parametreler": result_params(result),
                    "val_macro_f1": result["best_config"]["val_macro_f1"],
                    "test_macro_f1": result["test"]["macro_f1"],
                }
            )
    show_table(pd.DataFrame(coverage_rows))

    heading("ODEV 2 - EN IYI MODELLERIN SINIF BAZLI TEST F1 DEGERLERI")
    per_class_rows = []
    for corpus in CORPORA:
        candidates = []
        for path in (ODEV2_OUT / corpus).glob("*_result.json"):
            result = read_json(path)
            candidates.append(result)
        best = max(candidates, key=lambda item: item["test"]["macro_f1"])
        for emotion, metrics in best["test_per_class"].items():
            per_class_rows.append(
                {
                    "corpus": corpus,
                    "model": best["model"],
                    "emotion": emotion,
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1"], 4),
                    "support": metrics["support"],
                }
            )
    show_table(pd.DataFrame(per_class_rows))

    heading("KARMASIKLIK MATRISI DOSYALARI")
    for corpus in CORPORA:
        for path in sorted((ODEV2_OUT / corpus).glob("*_confusion_matrix.png")):
            print(path.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section",
        choices=("all", "data", "odev1", "odev2"),
        default="all",
        help="Gosterilecek bolum (varsayilan: all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.section in ("all", "data"):
        show_data_statistics()
    if args.section in ("all", "odev1"):
        show_assignment_1()
    if args.section in ("all", "odev2"):
        show_assignment_2()

    heading("NOT")
    print("Bu komut mevcut resmi CSV/JSON ciktilarini okur; deney dosyalarini degistirmez.")
    print("pytest kod testidir; ML test metrikleri yukaridaki test fold sonuclaridir.")


if __name__ == "__main__":
    main()
