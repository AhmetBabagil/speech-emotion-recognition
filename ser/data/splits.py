"""Train/val/test splitting strategies.

Two regimes, selected automatically from the config's train/eval corpora:

* WITHIN-CORPUS  (train_corpora == eval_corpora): the chosen corpus is split
  3-way. With ``split="speaker"`` the split is speaker-independent (no speaker
  appears in more than one fold) -- the correct protocol for SER. With
  ``split="meld_official"`` MELD's own train/dev/test folds are used.

* CROSS-CORPUS  (train_corpora != eval_corpora): the training corpus is split
  speaker-independently into train/val, and the *entire* eval corpus becomes the
  test set. This measures generalization across recording conditions/speakers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..constants import NUM_CLASSES
from ..utils import get_logger

log = get_logger(__name__)


def _valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["label_idx"].between(0, NUM_CLASSES - 1)].copy()
    df["speaker"] = df["speaker"].astype(str)
    return df


def _check_nonempty(train_df, val_df, test_df):
    """Fail fast with a clear message if any fold ended up empty."""
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        if len(part) == 0:
            raise ValueError(
                f"Split produced an empty {name} fold "
                f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}). "
                "Likely too few speakers for the requested fractions, or a "
                "missing corpus/split column."
            )
    return train_df, val_df, test_df


def _speaker_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    """Partition ``df`` into len(fractions) folds by whole speakers.

    fractions sum to 1; speakers are shuffled deterministically and sliced.
    """
    speakers = sorted(df["speaker"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    if n < len(fractions):
        raise ValueError(
            f"Need at least {len(fractions)} distinct speakers for a "
            f"speaker-independent split, but only found {n}."
        )
    bounds = np.cumsum([int(round(f * n)) for f in fractions])
    bounds[-1] = n  # absorb rounding into last fold
    folds, start = [], 0
    for end in bounds:
        fold_speakers = set(speakers[start:end])
        folds.append(df[df["speaker"].isin(fold_speakers)].copy())
        start = end
    return folds


def _random_partition(df: pd.DataFrame, fractions: list[float], seed: int) -> list[pd.DataFrame]:
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n = len(df)
    bounds = np.cumsum([int(round(f * n)) for f in fractions])
    bounds[-1] = n
    folds, start = [], 0
    for end in bounds:
        folds.append(df.iloc[idx[start:end]].copy())
        start = end
    return folds


def prepare_splits(manifest: pd.DataFrame, data_cfg, seed: int = 42):
    """Return (train_df, val_df, test_df) according to ``data_cfg``."""
    df = _valid_rows(manifest)
    train_corpora = set(data_cfg.train_corpora)
    eval_corpora = set(data_cfg.eval_corpora)

    train_pool = df[df["corpus"].isin(train_corpora)].copy()
    eval_pool = df[df["corpus"].isin(eval_corpora)].copy()
    if len(train_pool) == 0:
        raise ValueError(f"No rows for train_corpora={train_corpora}. "
                         f"Available: {sorted(df['corpus'].unique())}")

    cross = train_corpora != eval_corpora
    vf, tf = data_cfg.val_fraction, data_cfg.test_fraction

    if cross:
        if len(eval_pool) == 0:
            raise ValueError(f"No rows for eval_corpora={eval_corpora}. "
                             f"Available: {sorted(df['corpus'].unique())}")
        train_df, val_df = _speaker_partition(train_pool, [1 - vf, vf], seed)
        test_df = eval_pool
        log.info("CROSS-CORPUS: train=%s eval=%s | train=%d val=%d test=%d",
                 sorted(train_corpora), sorted(eval_corpora),
                 len(train_df), len(val_df), len(test_df))
        return _check_nonempty(train_df, val_df, test_df)

    # within-corpus
    if data_cfg.split == "meld_official" and "split" in train_pool.columns:
        # NOTE: MELD's official folds are DIALOGUE-based, not speaker-based, so the
        # same TV character can appear in train/dev/test. This is the standard MELD
        # benchmark protocol (comparable to the literature) but is NOT speaker-
        # independent. Use split="speaker" for a speaker-independent MELD evaluation.
        sp = train_pool["split"].astype(str)
        train_df = train_pool[sp == "train"].copy()
        val_df = train_pool[sp == "dev"].copy()
        test_df = train_pool[sp == "test"].copy()
        if len(val_df) == 0 or len(test_df) == 0:
            log.warning("meld_official split incomplete; falling back to speaker split")
        else:
            log.info("MELD-OFFICIAL (dialogue-based, not speaker-independent) | "
                     "train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
            return _check_nonempty(train_df, val_df, test_df)

    fractions = [1 - vf - tf, vf, tf]
    if data_cfg.split == "random":
        train_df, val_df, test_df = _random_partition(train_pool, fractions, seed)
        proto = "RANDOM"
    else:
        train_df, val_df, test_df = _speaker_partition(train_pool, fractions, seed)
        proto = "SPEAKER-INDEPENDENT"
    log.info("%s split | train=%d val=%d test=%d", proto,
             len(train_df), len(val_df), len(test_df))
    return _check_nonempty(train_df, val_df, test_df)
