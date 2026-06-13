"""Run the within-corpus and cross-corpus experiment matrix and summarize it.

Four settings for a chosen model:
    within_cremad        train CREMA-D  -> test CREMA-D   (speaker-independent)
    within_meld          train MELD     -> test MELD      (speaker-independent)
    cross_cremad_to_meld train CREMA-D  -> test MELD      (domain shift)
    cross_meld_to_cremad train MELD     -> test CREMA-D   (domain shift)

Produces ``outputs/<exp>_crosscorpus/summary.csv`` and a macro-F1 heatmap, the
core deliverable of the proposal's cross-corpus generalization analysis.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .constants import CORPUS_CREMAD, CORPUS_MELD
from .utils import get_logger, ensure_dir

log = get_logger(__name__)

SETTINGS = [
    ("within_cremad",        (CORPUS_CREMAD,), (CORPUS_CREMAD,)),
    ("within_meld",          (CORPUS_MELD,),   (CORPUS_MELD,)),
    ("cross_cremad_to_meld", (CORPUS_CREMAD,), (CORPUS_MELD,)),
    ("cross_meld_to_cremad", (CORPUS_MELD,),   (CORPUS_CREMAD,)),
]


def run_cross_corpus(cfg: Config, use_baseline: bool = False, baseline_kind: str = "svm") -> pd.DataFrame:
    from .train import train_torch, train_baseline

    base_exp = cfg.experiment
    out_root = ensure_dir(Path(cfg.output_dir) / f"{base_exp}_crosscorpus")
    rows = []
    for name, train_corpora, eval_corpora in SETTINGS:
        c = copy.deepcopy(cfg)
        c.experiment = f"{base_exp}_{name}"
        c.data.train_corpora = train_corpora
        c.data.eval_corpora = eval_corpora
        log.info("=== Cross-corpus setting: %s (train=%s eval=%s) ===",
                 name, train_corpora, eval_corpora)
        try:
            if use_baseline:
                m = train_baseline(c, kind=baseline_kind)
            else:
                m = train_torch(c)
        except Exception as e:  # one setting failing shouldn't kill the matrix
            log.exception("Setting %s failed: %s", name, e)
            continue
        rows.append({
            "setting": name,
            "train": "+".join(train_corpora),
            "eval": "+".join(eval_corpora),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_root / "summary.csv", index=False)
    with open(out_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _plot_matrix(df, out_root / "macro_f1_matrix.png")
    log.info("Cross-corpus summary:\n%s", df.to_string(index=False))
    return df


def _plot_matrix(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    corpora = [CORPUS_CREMAD, CORPUS_MELD]
    mat = np.full((2, 2), np.nan)
    for _, r in df.iterrows():
        if r["train"] in corpora and r["eval"] in corpora:
            mat[corpora.index(r["train"]), corpora.index(r["eval"])] = r["macro_f1"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="viridis",
                xticklabels=[f"test:{c}" for c in corpora],
                yticklabels=[f"train:{c}" for c in corpora], vmin=0, vmax=1)
    plt.title("Macro-F1: within (diagonal) vs cross-corpus (off-diagonal)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
