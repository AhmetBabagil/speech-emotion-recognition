"""KNN hiperparametre çalışması — Ödev 1 (yalnızca numpy/pandas/scikit-learn).

Her veri seti için ayrı ayrı, üç hiperparametre taranır:
  1. Öznitelik vektör boyutu  : mean(768) / mean_std(1536) / mean_std_max(2304)
  2. PCA çıktı boyutu         : yok / 32 / 64 / 128 / 256
  3. KNN komşu sayısı K       : 1 / 3 / 5 / 7 / 11 / 15 / 21 / 31

Akış: StandardScaler (train'e uydurulur) → PCA (train'e uydurulur) → KNN.
Geçerleme (validation) kümesi en iyi (öznitelik, PCA, K) üçlüsünü seçer; en iyi
yapılandırma train+val üzerine yeniden uydurulup **test** kümesinde ölçülür.

Ölçütler: doğruluk, dengeli doğruluk, makro-F1, ağırlıklı-F1 ve karmaşıklık
matrisi. Çıktılar ``odev1/outputs/<corpus>/`` altına yazılır.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.data import prepare_splits  # noqa: E402
from ser.constants import CORPUS_CREMAD, CORPUS_MELD  # noqa: E402
from ser.utils import get_logger, ensure_dir, set_seed  # noqa: E402
from odev1.features_w2v import load_pooled  # noqa: E402
from odev1.evaluation import compute_metrics, plot_confusion  # noqa: E402

log = get_logger("odev1.knn")

POOLS = ["mean", "mean_std", "mean_std_max"]          # feature-size hyperparameter
POOL_DIM = {"mean": 768, "mean_std": 1536, "mean_std_max": 2304}
PCA_DIMS = [0, 32, 64, 128, 256]                       # 0 = PCA yok
KS = [1, 3, 5, 7, 11, 15, 21, 31]                      # KNN K
SEED = 42


def _splits_for(corpus: str, manifest: str):
    cfg = Config()
    cfg.data.manifest = manifest
    cfg.data.train_corpora = (corpus,)
    cfg.data.eval_corpora = (corpus,)
    cfg.data.split = "speaker"          # speaker-independent for both corpora
    cfg.train.seed = SEED
    df = pd.read_csv(manifest)
    return prepare_splits(df, cfg.data, SEED)


def _fit_pca(Xtr_scaled, n):
    """PCA fit on train. n=0 -> identity (no PCA). Clamp to valid range."""
    if n == 0:
        return None
    n_eff = min(n, Xtr_scaled.shape[1], Xtr_scaled.shape[0])
    pca = PCA(n_components=n_eff, random_state=SEED).fit(Xtr_scaled)
    return pca


def run_dataset(corpus: str, manifest: str, cache_dir: str, out_root: str) -> dict:
    set_seed(SEED)
    out_dir = ensure_dir(Path(out_root) / corpus)
    train_df, val_df, test_df = _splits_for(corpus, manifest)
    log.info("[%s] train=%d val=%d test=%d", corpus, len(train_df), len(val_df), len(test_df))

    grid_rows = []
    best = None  # (val_macro_f1, dict)

    for pool in POOLS:
        # Load features once per size (numpy only; sliced from the cached vectors).
        Xtr, ytr = load_pooled(train_df, pool, cache_dir=cache_dir)
        Xva, yva = load_pooled(val_df, pool, cache_dir=cache_dir)
        if len(Xtr) == 0 or len(Xva) == 0:
            log.warning("[%s/%s] missing features — did extraction finish?", corpus, pool)
            continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

        for pdim in PCA_DIMS:
            pca = _fit_pca(Xtr_s, pdim)
            Xtr_p = Xtr_s if pca is None else pca.transform(Xtr_s)
            Xva_p = Xva_s if pca is None else pca.transform(Xva_s)
            eff_dim = Xtr_p.shape[1]
            for k in KS:
                if k > len(Xtr_p):
                    continue
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(Xtr_p, ytr)
                m = compute_metrics(yva, knn.predict(Xva_p))
                row = {"feature": pool, "feature_dim": POOL_DIM[pool],
                       "pca_dim": ("none" if pdim == 0 else eff_dim), "K": k,
                       "val_accuracy": round(m["accuracy"], 4),
                       "val_macro_f1": round(m["macro_f1"], 4)}
                grid_rows.append(row)
                if best is None or m["macro_f1"] > best[0]:
                    best = (m["macro_f1"], {"feature": pool, "pca_dim": pdim, "K": k})

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(out_dir / "validation_grid.csv", index=False)

    # ---- refit best on train+val, evaluate on test ----------------------------
    bf = best[1]
    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    Xfit, yfit = load_pooled(fit_df, bf["feature"], cache_dir=cache_dir)
    Xte, yte = load_pooled(test_df, bf["feature"], cache_dir=cache_dir)
    scaler = StandardScaler().fit(Xfit)
    Xfit_s, Xte_s = scaler.transform(Xfit), scaler.transform(Xte)
    pca = _fit_pca(Xfit_s, bf["pca_dim"])
    Xfit_p = Xfit_s if pca is None else pca.transform(Xfit_s)
    Xte_p = Xte_s if pca is None else pca.transform(Xte_s)
    knn = KNeighborsClassifier(n_neighbors=bf["K"]).fit(Xfit_p, yfit)
    test_m = compute_metrics(yte, knn.predict(Xte_p))

    best_cfg = {"feature": bf["feature"], "feature_dim": POOL_DIM[bf["feature"]],
                "pca_dim": ("none" if bf["pca_dim"] == 0 else
                            (Xfit_p.shape[1] if pca is None else pca.n_components_)),
                "K": bf["K"], "val_macro_f1": round(best[0], 4)}
    result = {"corpus": corpus, "best_config": best_cfg,
              "test": {kk: round(test_m[kk], 4) for kk in
                       ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")},
              "test_per_class": test_m["per_class"],
              "confusion_matrix": test_m["confusion_matrix"]}
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    plot_confusion(test_m["confusion_matrix"], out_dir / "confusion_matrix.png",
                   title=f"{corpus} — en iyi KNN (test)")
    log.info("[%s] BEST feat=%s(%d) pca=%s K=%d | val_macroF1=%.3f -> "
             "test acc=%.3f macroF1=%.3f",
             corpus, best_cfg["feature"], best_cfg["feature_dim"], best_cfg["pca_dim"],
             best_cfg["K"], best_cfg["val_macro_f1"], test_m["accuracy"], test_m["macro_f1"])
    return result


def run_all(manifest: str = "data/processed/manifest.csv",
            cache_dir: str = "odev1/cache/w2v",
            out_root: str = "odev1/outputs",
            corpora=(CORPUS_CREMAD, CORPUS_MELD)) -> dict:
    results = {}
    for corpus in corpora:
        results[corpus] = run_dataset(corpus, manifest, cache_dir, out_root)

    # ---- test comparison table + overall best confusion matrix ----------------
    out_root = ensure_dir(out_root)
    comp = pd.DataFrame([{
        "Veri seti": c, "Öznitelik": r["best_config"]["feature"],
        "Boyut": r["best_config"]["feature_dim"], "PCA": r["best_config"]["pca_dim"],
        "K": r["best_config"]["K"], "Doğruluk": r["test"]["accuracy"],
        "Dengeli doğr.": r["test"]["balanced_accuracy"], "Makro-F1": r["test"]["macro_f1"],
    } for c, r in results.items()])
    comp.to_csv(Path(out_root) / "test_comparison.csv", index=False)

    overall = max(results.values(), key=lambda r: r["test"]["macro_f1"])
    plot_confusion(overall["confusion_matrix"], Path(out_root) / "overall_best_confusion.png",
                   title=f"Genel en iyi: {overall['corpus']} (test)")
    with open(Path(out_root) / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"per_dataset": results, "overall_best": overall["corpus"]},
                  f, indent=2, ensure_ascii=False)
    log.info("Comparison:\n%s", comp.to_string(index=False))
    log.info("Overall best dataset: %s (test macroF1=%.3f)",
             overall["corpus"], overall["test"]["macro_f1"])
    return results
