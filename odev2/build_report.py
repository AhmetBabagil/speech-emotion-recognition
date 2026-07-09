"""Build markdown tables for the Assignment 2 Google Doc.

Run after `python odev2/run_experiment.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path("odev2/outputs")
REPORT = Path("odev2/RAPOR_tablolar.md")
CORP_TR = {"cremad": "CREMA-D", "meld": "MELD"}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _short_params(params_text: str) -> str:
    try:
        params = json.loads(params_text)
    except Exception:
        return params_text
    return ", ".join(f"{k}={v}" for k, v in params.items())


def _top_validation_table(grid_path: Path, limit: int = 12) -> pd.DataFrame:
    grid = pd.read_csv(grid_path)
    top = grid.sort_values(
        ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"], ascending=False
    ).head(limit)
    return pd.DataFrame(
        {
            "Sira": range(1, len(top) + 1),
            "F": top["feature_dim"],
            "PCA": top["pca_dim"],
            "Hiperparametreler": top["params"].map(_short_params),
            "Val dogr.": top["val_accuracy"],
            "Val dengeli dogr.": top["val_balanced_accuracy"],
            "Val makro-F1": top["val_macro_f1"],
        }
    )


def _comparison_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame(
        {
            "Veri seti": df["corpus"].map(lambda c: CORP_TR.get(c, c)),
            "Model": df["model"],
            "F": df["feature_dim"],
            "PCA": df["pca_dim"],
            "Hiperparametreler": df["params"].map(_short_params),
            "Test dogr.": df["test_accuracy"],
            "Test dengeli dogr.": df["test_balanced_accuracy"],
            "Test makro-F1": df["test_macro_f1"],
            "Test agirlikli-F1": df["test_weighted_f1"],
        }
    )

def _result_rows() -> list[dict]:
    rows = []
    for corpus_dir in sorted([p for p in OUT.iterdir() if p.is_dir()]):
        corpus = corpus_dir.name
        for result_path in sorted(corpus_dir.glob("*_result.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            bc = result["best_config"]
            test = result["test"]
            rows.append(
                {
                    "corpus": corpus,
                    "model": result["model"],
                    "feature_dim": bc["feature_dim"],
                    "pca_dim": bc["pca_dim"],
                    "params": json.dumps(bc["params"], ensure_ascii=False, sort_keys=True),
                    "test_accuracy": test["accuracy"],
                    "test_balanced_accuracy": test["balanced_accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "test_weighted_f1": test["weighted_f1"],
                }
            )
    return rows


def _knn_rows(corpora: list[str]) -> list[dict]:
    rows = []
    for corpus in corpora:
        result_path = Path("odev1/outputs") / corpus / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        bc = result["best_config"]
        test = result["test"]
        rows.append(
            {
                "corpus": corpus,
                "model": "KNN (Odev 1)",
                "feature_dim": bc["feature_dim"],
                "pca_dim": bc["pca_dim"],
                "params": json.dumps({"K": bc["K"]}, ensure_ascii=False, sort_keys=True),
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
            }
        )
    return rows


def _comparison_table_from_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["corpus", "test_macro_f1"], ascending=[True, False])
    return pd.DataFrame(
        {
            "Veri seti": df["corpus"].map(lambda c: CORP_TR.get(c, c)),
            "Model": df["model"],
            "F": df["feature_dim"],
            "PCA": df["pca_dim"],
            "Hiperparametreler": df["params"].map(_short_params),
            "Test dogr.": df["test_accuracy"],
            "Test dengeli dogr.": df["test_balanced_accuracy"],
            "Test makro-F1": df["test_macro_f1"],
            "Test agirlikli-F1": df["test_weighted_f1"],
        }
    )


def _coverage_table() -> pd.DataFrame:
    rows = []
    for corpus_dir in sorted([p for p in OUT.iterdir() if p.is_dir()]):
        corpus = corpus_dir.name
        for grid_path in sorted(corpus_dir.glob("*_validation_grid.csv")):
            model_key = grid_path.name.replace("_validation_grid.csv", "")
            result_path = corpus_dir / f"{model_key}_result.json"
            model_name = model_key
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                model_name = result["model"]
            grid = pd.read_csv(grid_path)
            rows.append(
                {
                    "Veri seti": CORP_TR.get(corpus, corpus),
                    "Model": model_name,
                    "Kombinasyon": len(grid),
                    "F secenekleri": ", ".join(str(x) for x in sorted(grid["feature_dim"].unique())),
                    "PCA secenekleri": ", ".join(str(x) for x in grid["pca_dim"].drop_duplicates()),
                    "Hiperparametre kombinasyonu": grid["params"].nunique(),
                }
            )
    return pd.DataFrame(rows)

def main() -> None:
    if not OUT.exists():
        raise SystemExit("odev2/outputs bulunamadi. Once python odev2/run_experiment.py calistirin.")

    lines: list[str] = []
    lines.append("# Project Assignment 2 - Rapor tablolari\n")
    lines.append(
        "Not: Asagidaki validasyon tablolarinda her veri seti-model ikilisi icin "
        "en yuksek validasyon makro-F1 degerine sahip ilk kombinasyonlar verilmistir. "
        "Calistirilan grid CSV dosyalari `odev2/outputs/<veri_seti>/` altindadir.\n"
    )

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        grid_mode = summary.get("grid_mode") or ("quick" if summary.get("quick") else "full")
        mode_names = {
            "quick": "hizli kontrol gridi",
            "report": "rapor/teslim gridi",
            "full": "tam grid",
        }
        mode = mode_names.get(grid_mode, str(grid_mode))
        lines.append(
            f"Deney bilgisi: manifest `{summary.get('manifest')}`, "
            f"feature cache `{summary.get('cache_dir')}`, mod `{mode}`.\n"
        )

    coverage = _coverage_table()
    if not coverage.empty:
        lines.append("## Deney kapsami\n")
        lines.append(_md_table(coverage))
        lines.append("")

    for corpus_dir in sorted([p for p in OUT.iterdir() if p.is_dir()]):
        corpus = corpus_dir.name
        for grid_path in sorted(corpus_dir.glob("*_validation_grid.csv")):
            model_key = grid_path.name.replace("_validation_grid.csv", "")
            result_path = corpus_dir / f"{model_key}_result.json"
            if not result_path.exists():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            lines.append(f"## {CORP_TR.get(corpus, corpus)} - {result['model']} validasyon\n")
            lines.append(_md_table(_top_validation_table(grid_path)))
            lines.append("")

    result_rows = _result_rows()
    if result_rows:
        pd.DataFrame(result_rows).sort_values(
            ["corpus", "test_macro_f1"], ascending=[True, False]
        ).to_csv(OUT / "model_comparison.csv", index=False)

        lines.append("## 2. asama modelleri test karsilastirmasi\n")
        lines.append(_md_table(_comparison_table_from_rows(result_rows)))
        lines.append("")

        corpora = sorted({row["corpus"] for row in result_rows})
        with_knn_rows = result_rows + _knn_rows(corpora)
        pd.DataFrame(with_knn_rows).sort_values(
            ["corpus", "test_macro_f1"], ascending=[True, False]
        ).to_csv(OUT / "test_comparison_with_knn.csv", index=False)

        lines.append("## KNN dahil genel test karsilastirmasi\n")
        lines.append(_md_table(_comparison_table_from_rows(with_knn_rows)))
        lines.append("")

        df = pd.DataFrame(with_knn_rows)
        lines.append("## Otomatik bulgular\n")
        for corpus, part in df.groupby("corpus"):
            best = part.sort_values("test_macro_f1", ascending=False).iloc[0]
            lines.append(
                f"- {CORP_TR.get(corpus, corpus)} icin en iyi model {best['model']} "
                f"(test makro-F1={best['test_macro_f1']})."
            )
        overall = df.sort_values("test_macro_f1", ascending=False).iloc[0]
        lines.append(
            f"- Genel en iyi sonuc {CORP_TR.get(overall['corpus'], overall['corpus'])} "
            f"veri setinde {overall['model']} ile elde edildi "
            f"(test makro-F1={overall['test_macro_f1']})."
        )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[yazildi] {REPORT}")


if __name__ == "__main__":
    main()
