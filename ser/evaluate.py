"""Metrics, confusion-matrix plotting and a reusable torch evaluation loop.

All metrics are computed over the canonical six-class label space so that
within-corpus and cross-corpus numbers are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import CANONICAL_EMOTIONS, NUM_CLASSES
from .utils import get_logger, ensure_dir

log = get_logger(__name__)


def compute_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
        balanced_accuracy_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError(f"Cannot compute metrics on empty arrays "
                         f"(y_true={len(y_true)}, y_pred={len(y_pred)}).")
    labels = list(range(NUM_CLASSES))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            CANONICAL_EMOTIONS[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i in labels
        },
        "confusion_matrix": cm.tolist(),
    }


def save_confusion_matrix(cm, out_path, *, title: str = "Confusion matrix", normalize: bool = True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_disp = cm
        fmt = ".0f"
    ensure_dir(Path(out_path).parent)
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm_disp, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CANONICAL_EMOTIONS, yticklabels=CANONICAL_EMOTIONS,
                vmin=0, vmax=1 if normalize else None, cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def report(y_true, y_pred, out_dir, prefix: str = "test", title: str | None = None) -> dict:
    """Compute metrics, write metrics.json + confusion_matrix.png, log a summary."""
    out_dir = ensure_dir(out_dir)
    metrics = compute_metrics(y_true, y_pred)
    with open(Path(out_dir) / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_matrix(
        metrics["confusion_matrix"],
        Path(out_dir) / f"{prefix}_confusion_matrix.png",
        title=title or f"{prefix} confusion matrix",
    )
    log.info("[%s] acc=%.4f  bal_acc=%.4f  macro_f1=%.4f  weighted_f1=%.4f",
             prefix, metrics["accuracy"], metrics["balanced_accuracy"],
             metrics["macro_f1"], metrics["weighted_f1"])
    return metrics


def evaluate_torch(model, loader, device):
    """Run ``model`` over ``loader`` -> (y_true, y_pred, y_prob) numpy arrays."""
    import torch

    model.eval()
    ys, preds, probs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)
            preds.append(prob.argmax(1).cpu().numpy())
            probs.append(prob.cpu().numpy())
            ys.append(np.asarray(yb))
    return (np.concatenate(ys), np.concatenate(preds), np.concatenate(probs))
