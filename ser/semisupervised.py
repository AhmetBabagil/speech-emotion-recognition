"""Unsupervised (feature clustering) and semi-supervised (reduced-label) analysis.

This implements the component promised in the proposal:
"azaltılmış etiketle eğitim ve öznitelik kümeleme üzerinden yarı-denetimli/
denetimsiz bir bileşen" — investigating, via feature clustering and reduced-label
training, how much emotion structure lives in the acoustics and how many labels
the task actually needs.

All three analyses use the MFCC-statistics features (the same 240-dim vectors as
the classical baseline), so they reuse the cached features and run fast on CPU.

  1. cluster_analysis  — K-Means (K = #emotions) on standardized MFCC features
     (UNSUPERVISED, no labels used to fit). Cluster quality vs the true emotions
     is measured with Adjusted Rand Index (ARI) and Normalized Mutual Information
     (NMI), both permutation-invariant, plus a Hungarian-matched cluster→emotion
     accuracy / macro-F1 (an interpretable "label-free classification" score).

  2. label_efficiency  — train the baseline on a fraction of the labels
     (1% … 100%) and report test macro-F1 (the learning curve). Shows how the
     task degrades as labels become scarce.

  3. self_training     — SEMI-SUPERVISED pseudo-labeling: from a small labeled
     seed, iteratively add high-confidence pseudo-labels from the unlabeled pool
     and retrain. Compared against a supervised-only model at the SAME label
     budget, to see whether unlabeled data helps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import CANONICAL_EMOTIONS, NUM_CLASSES
from .data import mfcc_feature_matrix, prepare_splits
from .evaluate import compute_metrics, save_confusion_matrix
from .models import build_baseline
from .utils import get_logger, set_seed, ensure_dir

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Shared feature loading
# --------------------------------------------------------------------------- #
def _load_features(cfg: Config):
    """Return (X_train, y_train, X_test, y_test) MFCC-statistics matrices.

    train+val are merged into the labeled pool; test is held out. Uses the same
    speaker-independent split as the rest of the project.
    """
    df = pd.read_csv(cfg.data.manifest)
    train_df, val_df, test_df = prepare_splits(df, cfg.data, cfg.train.seed)
    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    log.info("Extracting MFCC features (pool=%d, test=%d) ...", len(pool_df), len(test_df))
    X_tr, y_tr = mfcc_feature_matrix(pool_df, cfg)
    X_te, y_te = mfcc_feature_matrix(test_df, cfg)
    return X_tr, y_tr, X_te, y_te


def _stratified_subset(y: np.ndarray, fraction: float, rng, min_per_class: int = 1):
    """Indices of a class-stratified subset (>= min_per_class per present class)."""
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        n = int(round(fraction * len(c_idx)))
        n = max(min_per_class, min(n, len(c_idx)))
        idx.extend(rng.choice(c_idx, size=n, replace=False).tolist())
    return np.array(sorted(idx))


# --------------------------------------------------------------------------- #
# 1. Unsupervised clustering
# --------------------------------------------------------------------------- #
def cluster_analysis(cfg: Config, out_dir: Path, n_clusters: int | None = None) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from scipy.optimize import linear_sum_assignment

    out_dir = ensure_dir(out_dir)
    k = n_clusters or NUM_CLASSES
    X_tr, y_tr, X_te, y_te = _load_features(cfg)

    scaler = StandardScaler().fit(X_tr)  # fit on train only (no test leakage)
    Xtr, Xte = scaler.transform(X_tr), scaler.transform(X_te)

    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.train.seed).fit(Xtr)
    c_tr, c_te = km.labels_, km.predict(Xte)

    # Map cluster id -> emotion id via Hungarian on the TRAIN co-occurrence matrix
    # (maximize agreement), then apply that fixed map to the test clusters.
    co = np.zeros((k, NUM_CLASSES), dtype=np.int64)
    for cl, t in zip(c_tr, y_tr):
        co[cl, t] += 1
    rows, cols = linear_sum_assignment(-co)
    cluster_to_label = {int(r): int(c) for r, c in zip(rows, cols)}
    # clusters with no assigned label (k>classes) fall back to their argmax emotion
    for cl in range(k):
        cluster_to_label.setdefault(cl, int(co[cl].argmax()))
    y_pred = np.array([cluster_to_label[int(cl)] for cl in c_te])

    metrics = compute_metrics(y_te, y_pred)
    ari = float(adjusted_rand_score(y_te, c_te))
    nmi = float(normalized_mutual_info_score(y_te, c_te))

    result = {
        "n_clusters": k,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "hungarian_accuracy": metrics["accuracy"],
        "hungarian_macro_f1": metrics["macro_f1"],
        "cluster_to_emotion": {int(c): CANONICAL_EMOTIONS[lbl] for c, lbl in cluster_to_label.items()},
    }
    with open(out_dir / "cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    save_confusion_matrix(
        metrics["confusion_matrix"], out_dir / "cluster_confusion.png",
        title="Unsupervised K-Means (Hungarian-matched)", normalize=True,
    )
    log.info("[cluster] ARI=%.3f NMI=%.3f hungarian_acc=%.3f macroF1=%.3f (chance acc=%.3f)",
             ari, nmi, metrics["accuracy"], metrics["macro_f1"], 1.0 / NUM_CLASSES)
    return result


# --------------------------------------------------------------------------- #
# 2. Reduced-label learning curve
# --------------------------------------------------------------------------- #
DEFAULT_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)


def label_efficiency(cfg: Config, out_dir: Path, fractions=DEFAULT_FRACTIONS,
                     kind: str = "logreg") -> list[dict]:
    out_dir = ensure_dir(out_dir)
    X_tr, y_tr, X_te, y_te = _load_features(cfg)
    rng = np.random.default_rng(cfg.train.seed)

    rows = []
    for f in fractions:
        idx = _stratified_subset(y_tr, f, rng)
        pipe = build_baseline(kind)
        pipe.fit(X_tr[idx], y_tr[idx])
        m = compute_metrics(y_te, pipe.predict(X_te))
        rows.append({"label_fraction": float(f), "n_labeled": int(len(idx)),
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
        log.info("[label-eff] frac=%.2f n=%d acc=%.3f macroF1=%.3f",
                 f, len(idx), m["accuracy"], m["macro_f1"])

    pd.DataFrame(rows).to_csv(out_dir / "label_efficiency.csv", index=False)
    _plot_label_curve(rows, out_dir / "label_efficiency.png")
    return rows


def _plot_label_curve(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fr = [r["label_fraction"] * 100 for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(fr, f1, "o-", label="macro-F1")
    plt.plot(fr, acc, "s--", label="accuracy")
    plt.axhline(1.0 / NUM_CLASSES, color="gray", ls=":", label="chance")
    plt.xscale("log")
    plt.xlabel("Labeled fraction of training set (%, log scale)")
    plt.ylabel("Test score")
    plt.title("Label efficiency (reduced-label training)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# 3. Semi-supervised self-training
# --------------------------------------------------------------------------- #
def self_training(cfg: Config, out_dir: Path, label_fraction: float = 0.10,
                  threshold: float = 0.80, iterations: int = 10) -> dict:
    out_dir = ensure_dir(out_dir)
    X_tr, y_tr, X_te, y_te = _load_features(cfg)
    rng = np.random.default_rng(cfg.train.seed)

    seed_idx = _stratified_subset(y_tr, label_fraction, rng)
    labeled = np.zeros(len(y_tr), dtype=bool)
    labeled[seed_idx] = True

    # Supervised-only reference at this label budget.
    sup = build_baseline("logreg").fit(X_tr[seed_idx], y_tr[seed_idx])
    sup_m = compute_metrics(y_te, sup.predict(X_te))

    # Self-training: grow the labeled set with high-confidence pseudo-labels.
    X_lab, y_lab = X_tr[seed_idx].copy(), y_tr[seed_idx].copy()
    pool = ~labeled
    n_pseudo = 0
    for _ in range(iterations):
        clf = build_baseline("logreg").fit(X_lab, y_lab)
        if not pool.any():
            break
        proba = clf.predict_proba(X_tr[pool])
        conf, pred = proba.max(1), proba.argmax(1)
        take = conf >= threshold
        if not take.any():
            break
        pool_idx = np.where(pool)[0]
        add = pool_idx[take]
        X_lab = np.vstack([X_lab, X_tr[add]])
        y_lab = np.concatenate([y_lab, pred[take]])
        pool[add] = False
        n_pseudo += int(take.sum())

    final = build_baseline("logreg").fit(X_lab, y_lab)
    semi_m = compute_metrics(y_te, final.predict(X_te))

    result = {
        "label_fraction": label_fraction,
        "n_seed_labels": int(len(seed_idx)),
        "n_pseudo_labels_added": n_pseudo,
        "confidence_threshold": threshold,
        "supervised_only": {"accuracy": sup_m["accuracy"], "macro_f1": sup_m["macro_f1"]},
        "self_training": {"accuracy": semi_m["accuracy"], "macro_f1": semi_m["macro_f1"]},
        "macro_f1_delta": semi_m["macro_f1"] - sup_m["macro_f1"],
    }
    with open(out_dir / "self_training.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("[self-train] seed=%d(%.0f%%) +%d pseudo | supervised macroF1=%.3f -> "
             "self-train macroF1=%.3f (Δ=%+.3f)",
             len(seed_idx), label_fraction * 100, n_pseudo,
             sup_m["macro_f1"], semi_m["macro_f1"], result["macro_f1_delta"])
    return result


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_all(cfg: Config) -> dict:
    set_seed(cfg.train.seed)
    out_dir = ensure_dir(Path(cfg.output_dir) / cfg.experiment)
    cfg.save(out_dir / "config.yaml")
    log.info("=== Unsupervised clustering ===")
    clu = cluster_analysis(cfg, out_dir)
    log.info("=== Reduced-label learning curve ===")
    eff = label_efficiency(cfg, out_dir)
    log.info("=== Semi-supervised self-training ===")
    semi = self_training(cfg, out_dir)
    summary = {"clustering": clu, "label_efficiency": eff, "self_training": semi}
    with open(out_dir / "semisupervised_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
