"""Metrics + confusion-matrix plot for Ödev 1 (numpy/pandas/sklearn + matplotlib).

Kept independent of the rest of the repo so the KNN stage uses only the allowed
libraries (scikit-learn for metrics, matplotlib for plotting — no seaborn, no
deep-learning libraries).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # label space only  # noqa: E402


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score,
        precision_recall_fscore_support, confusion_matrix,
    )
    labels = list(range(NUM_CLASSES))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {CANONICAL_EMOTIONS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                              "f1": float(f1[i]), "support": int(sup[i])}
                      for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_confusion(cm, out_path, title="Karmaşıklık matrisi", normalize=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        rs = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, rs, out=np.zeros_like(cm), where=rs != 0)
        fmt = ".2f"
    else:
        disp, fmt = cm, ".0f"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CANONICAL_EMOTIONS, rotation=45, ha="right")
    ax.set_yticklabels(CANONICAL_EMOTIONS)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title(title)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, format(disp[i, j], fmt), ha="center", va="center",
                    color="white" if disp[i, j] > (0.5 if normalize else disp.max() / 2) else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
