"""Assignment 2 classical ML pipeline.

For each corpus and model:
  - choose the Wav2Vec2 pooled feature size,
  - repeat the search with and without PCA,
  - optimize the required model hyperparameters on validation data,
  - refit the best configuration on train+validation,
  - evaluate once on test data.

The ML workflow uses numpy, pandas and scikit-learn. Wav2Vec2 is not run here;
this file only reads the cached vectors produced in Assignment 1. Report-mode outputs are kept under odev2/outputs for submission review.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from odev1.evaluation import compute_metrics, plot_confusion  # noqa: E402
from odev1.features_w2v import load_pooled  # noqa: E402
from ser.config import Config  # noqa: E402
from ser.constants import CORPUS_CREMAD, CORPUS_MELD  # noqa: E402
from ser.data import prepare_splits  # noqa: E402
from ser.utils import ensure_dir, get_logger, set_seed  # noqa: E402

log = get_logger("odev2.models")

SEED = 42

POOLS = ["mean", "mean_std", "mean_std_max"]
POOL_DIM = {"mean": 768, "mean_std": 1536, "mean_std_max": 2304}
PCA_DIMS = [0, 32, 64, 128, 256, 512]

QUICK_POOLS = ["mean"]
QUICK_PCA_DIMS = [0, 64]
REPORT_POOLS = POOLS
REPORT_PCA_DIMS = [0, 64, 128, 256]
GRID_MODES = ("quick", "report", "full")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    param_grid: dict[str, list[Any]]
    report_grid: dict[str, list[Any]]
    quick_grid: dict[str, list[Any]]
    make_estimator: Callable[[dict[str, Any]], Any]
    fit_with_sample_weight: bool = False


def _grid_product(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _grid_settings(spec: ModelSpec, grid_mode: str) -> tuple[list[str], list[int], list[dict[str, Any]]]:
    if grid_mode == "quick":
        return QUICK_POOLS, QUICK_PCA_DIMS, _grid_product(spec.quick_grid)
    if grid_mode == "report":
        pca_dims = [0, 64] if spec.name == "gradient_boosting" else REPORT_PCA_DIMS
        return REPORT_POOLS, pca_dims, _grid_product(spec.report_grid)
    if grid_mode == "full":
        return POOLS, PCA_DIMS, _grid_product(spec.param_grid)
    raise ValueError(f"Unknown grid mode: {grid_mode}. Expected one of {GRID_MODES}.")


def _json_default(value: Any) -> str:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def _params_to_text(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True, default=_json_default)


def _make_decision_tree(params: dict[str, Any]) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=SEED, class_weight="balanced", **params)


def _make_random_forest(params: dict[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=SEED,
        class_weight="balanced",
        # n_jobs=1 avoids Windows permission issues in restricted environments.
        n_jobs=1,
        **params,
    )


def _make_gradient_boosting(params: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=SEED,
        max_iter=30,
        max_bins=32,
        early_stopping=True,
        **params,
    )


MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="decision_tree",
        display_name="Karar Agaci",
        param_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 4, 8, 12, 16, 24],
            "min_samples_split": [2, 5, 10, 20],
        },
        report_grid={
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 8, 16],
            "min_samples_split": [2, 10],
        },
        quick_grid={
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 8],
            "min_samples_split": [2, 10],
        },
        make_estimator=_make_decision_tree,
    ),
    ModelSpec(
        name="random_forest",
        display_name="Rastgele Orman",
        param_grid={
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 8, 16, 24],
            "max_features": ["sqrt", "log2", 0.5],
        },
        report_grid={
            "n_estimators": [100],
            "max_depth": [None, 16, 24],
            "max_features": ["sqrt", "log2"],
        },
        quick_grid={
            "n_estimators": [100],
            "max_depth": [None, 16],
            "max_features": ["sqrt"],
        },
        make_estimator=_make_random_forest,
    ),
    ModelSpec(
        name="gradient_boosting",
        display_name="Gradient Boosting",
        param_grid={
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_depth": [1, 2, 3, 5],
        },
        report_grid={
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 3],
        },
        quick_grid={
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 3],
        },
        make_estimator=_make_gradient_boosting,
        fit_with_sample_weight=True,
    ),
]


def _splits_for(corpus: str, manifest: str):
    cfg = Config()
    cfg.data.manifest = manifest
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = "speaker"
    cfg.train.seed = SEED
    df = pd.read_csv(manifest)
    return prepare_splits(df, cfg.data, SEED)


def _fit_pca(Xtr_scaled: np.ndarray, requested_dim: int) -> PCA | None:
    if requested_dim == 0:
        return None
    n_eff = min(requested_dim, Xtr_scaled.shape[1], Xtr_scaled.shape[0])
    return PCA(n_components=n_eff, random_state=SEED).fit(Xtr_scaled)


def _transform_with_pca(
    Xtr: np.ndarray,
    Xva: np.ndarray,
    requested_dim: int,
) -> tuple[np.ndarray, np.ndarray, int | str]:
    pca = _fit_pca(Xtr, requested_dim)
    if pca is None:
        return Xtr, Xva, "none"
    return pca.transform(Xtr), pca.transform(Xva), int(pca.n_components_)


def _fit_estimator(spec: ModelSpec, params: dict[str, Any], X: np.ndarray, y: np.ndarray):
    estimator = spec.make_estimator(params)
    if spec.fit_with_sample_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)
    return estimator


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": round(metrics["accuracy"], 4),
        f"{prefix}_balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        f"{prefix}_macro_f1": round(metrics["macro_f1"], 4),
        f"{prefix}_weighted_f1": round(metrics["weighted_f1"], 4),
    }


def run_model_for_dataset(
    corpus: str,
    spec: ModelSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cache_dir: str,
    out_dir: Path,
    grid_mode: str = "report",
) -> dict[str, Any]:
    pools, pca_dims, params_grid = _grid_settings(spec, grid_mode)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for pool in pools:
        Xtr, ytr = load_pooled(train_df, pool, cache_dir=cache_dir)
        Xva, yva = load_pooled(val_df, pool, cache_dir=cache_dir)
        if len(Xtr) == 0 or len(Xva) == 0:
            log.warning("[%s/%s/%s] missing train or validation features", corpus, spec.name, pool)
            continue

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xva_s = scaler.transform(Xva)

        for requested_pca in pca_dims:
            Xtr_p, Xva_p, effective_pca = _transform_with_pca(Xtr_s, Xva_s, requested_pca)

            for params in params_grid:
                estimator = _fit_estimator(spec, params, Xtr_p, ytr)
                metrics = compute_metrics(yva, estimator.predict(Xva_p))
                row = {
                    "corpus": corpus,
                    "model": spec.display_name,
                    "model_key": spec.name,
                    "feature": pool,
                    "feature_dim": POOL_DIM[pool],
                    "pca_requested": "none" if requested_pca == 0 else requested_pca,
                    "pca_dim": effective_pca,
                    "params": _params_to_text(params),
                    **_metric_row("val", metrics),
                }
                rows.append(row)

                score = (
                    metrics["macro_f1"],
                    metrics["balanced_accuracy"],
                    metrics["accuracy"],
                )
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "pool": pool,
                        "requested_pca": requested_pca,
                        "effective_pca": effective_pca,
                        "params": params,
                        "val_metrics": metrics,
                    }

    if best is None:
        raise RuntimeError(
            f"No valid validation result for corpus={corpus}, model={spec.name}. "
            "Run odev1/extract.py first or pass a manifest covered by the feature cache."
        )

    grid = pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"], ascending=False
    )
    grid_path = out_dir / f"{spec.name}_validation_grid.csv"
    grid.to_csv(grid_path, index=False)

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    Xfit, yfit = load_pooled(fit_df, best["pool"], cache_dir=cache_dir)
    Xte, yte = load_pooled(test_df, best["pool"], cache_dir=cache_dir)
    if len(Xfit) == 0 or len(Xte) == 0:
        raise RuntimeError(
            f"No valid train+val or test features for corpus={corpus}, model={spec.name}."
        )

    scaler = StandardScaler().fit(Xfit)
    Xfit_s = scaler.transform(Xfit)
    Xte_s = scaler.transform(Xte)
    pca = _fit_pca(Xfit_s, best["requested_pca"])
    if pca is None:
        Xfit_p, Xte_p = Xfit_s, Xte_s
        test_pca_dim: int | str = "none"
    else:
        Xfit_p, Xte_p = pca.transform(Xfit_s), pca.transform(Xte_s)
        test_pca_dim = int(pca.n_components_)

    estimator = _fit_estimator(spec, best["params"], Xfit_p, yfit)
    test_metrics = compute_metrics(yte, estimator.predict(Xte_p))

    result = {
        "corpus": corpus,
        "model": spec.display_name,
        "model_key": spec.name,
        "best_config": {
            "feature": best["pool"],
            "feature_dim": POOL_DIM[best["pool"]],
            "pca_dim": test_pca_dim,
            "params": best["params"],
            "val_macro_f1": round(best["val_metrics"]["macro_f1"], 4),
            "val_balanced_accuracy": round(best["val_metrics"]["balanced_accuracy"], 4),
        },
        "test": {
            k: round(test_metrics[k], 4)
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
        },
        "test_per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "validation_grid": str(grid_path),
    }

    result_path = out_dir / f"{spec.name}_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_confusion(
        test_metrics["confusion_matrix"],
        out_dir / f"{spec.name}_confusion_matrix.png",
        title=f"{corpus} - {spec.display_name} (test)",
    )

    log.info(
        "[%s] %s best feat=%s(%d) pca=%s params=%s | val macroF1=%.3f -> test macroF1=%.3f",
        corpus,
        spec.display_name,
        best["pool"],
        POOL_DIM[best["pool"]],
        test_pca_dim,
        _params_to_text(best["params"]),
        best["val_metrics"]["macro_f1"],
        test_metrics["macro_f1"],
    )
    return result


def run_dataset(
    corpus: str,
    manifest: str,
    cache_dir: str,
    out_root: str,
    grid_mode: str = "report",
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    set_seed(SEED)
    out_dir = ensure_dir(Path(out_root) / corpus)
    train_df, val_df, test_df = _splits_for(corpus, manifest)
    log.info("[%s] train=%d val=%d test=%d", corpus, len(train_df), len(val_df), len(test_df))

    if model_keys is None:
        specs = MODEL_SPECS
    else:
        known = {s.name for s in MODEL_SPECS}
        unknown = sorted(set(model_keys) - known)
        if unknown:
            raise ValueError(f"Unknown model keys: {unknown}. Expected one of {sorted(known)}.")
        specs = [s for s in MODEL_SPECS if s.name in set(model_keys)]

    results: dict[str, Any] = {}
    for spec in specs:
        results[spec.name] = run_model_for_dataset(
            corpus=corpus,
            spec=spec,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            cache_dir=cache_dir,
            out_dir=out_dir,
            grid_mode=grid_mode,
        )
    return results


def _comparison_rows(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus, corpus_results in results.items():
        for result in corpus_results.values():
            bc = result["best_config"]
            test = result["test"]
            rows.append(
                {
                    "corpus": corpus,
                    "model": result["model"],
                    "feature": bc["feature"],
                    "feature_dim": bc["feature_dim"],
                    "pca_dim": bc["pca_dim"],
                    "params": _params_to_text(bc["params"]),
                    "val_macro_f1": bc["val_macro_f1"],
                    "val_balanced_accuracy": bc["val_balanced_accuracy"],
                    "test_accuracy": test["accuracy"],
                    "test_balanced_accuracy": test["balanced_accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "test_weighted_f1": test["weighted_f1"],
                }
            )
    return rows


def _load_knn_rows(corpora: tuple[str, ...], knn_out_root: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus in corpora:
        path = Path(knn_out_root) / corpus / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        bc = result["best_config"]
        test = result["test"]
        rows.append(
            {
                "corpus": corpus,
                "model": "KNN (Odev 1)",
                "feature": bc.get("feature"),
                "feature_dim": bc.get("feature_dim"),
                "pca_dim": bc.get("pca_dim"),
                "params": _params_to_text({"K": bc.get("K")}),
                "val_macro_f1": bc.get("val_macro_f1"),
                "val_balanced_accuracy": "",
                "test_accuracy": test["accuracy"],
                "test_balanced_accuracy": test["balanced_accuracy"],
                "test_macro_f1": test["macro_f1"],
                "test_weighted_f1": test["weighted_f1"],
            }
        )
    return rows


def run_all(
    manifest: str = "data/processed/manifest.csv",
    cache_dir: str = "odev1/cache/w2v",
    out_root: str = "odev2/outputs",
    corpora: tuple[str, ...] = (CORPUS_CREMAD, CORPUS_MELD),
    quick: bool = False,
    grid_mode: str = "report",
    model_keys: tuple[str, ...] | None = None,
    knn_out_root: str = "odev1/outputs",
) -> dict[str, Any]:
    if quick:
        grid_mode = "quick"
    if grid_mode not in GRID_MODES:
        raise ValueError(f"Unknown grid mode: {grid_mode}. Expected one of {GRID_MODES}.")
    out_root_path = ensure_dir(out_root)
    results: dict[str, dict[str, Any]] = {}

    for corpus in corpora:
        results[corpus] = run_dataset(
            corpus, manifest, cache_dir, out_root, grid_mode=grid_mode, model_keys=model_keys
        )

    comparison = pd.DataFrame(_comparison_rows(results))
    comparison = comparison.sort_values(["corpus", "test_macro_f1"], ascending=[True, False])
    comparison.to_csv(out_root_path / "model_comparison.csv", index=False)

    with_knn_rows = _comparison_rows(results) + _load_knn_rows(corpora, knn_out_root)
    if with_knn_rows:
        with_knn = pd.DataFrame(with_knn_rows)
        with_knn = with_knn.sort_values(["corpus", "test_macro_f1"], ascending=[True, False])
        with_knn.to_csv(out_root_path / "test_comparison_with_knn.csv", index=False)

    summary = {
        "manifest": manifest,
        "cache_dir": cache_dir,
        "quick": grid_mode == "quick",
        "grid_mode": grid_mode,
        "models": list(model_keys) if model_keys is not None else [s.name for s in MODEL_SPECS],
        "per_dataset": results,
    }
    (out_root_path / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote comparison tables under %s", out_root_path)
    return summary
