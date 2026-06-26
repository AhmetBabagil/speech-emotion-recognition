"""Canonical label space and corpus-specific label mappings.

This module is the single source of truth for the emotion classes used across the
whole project. Both corpora are mapped onto the SAME six canonical labels so that
within-corpus and cross-corpus evaluation share one index space.

Common six emotions (intersection of CREMA-D and MELD):
    angry, disgust, fear, happy, neutral, sad

MELD additionally contains ``surprise``; CREMA-D does not, so ``surprise`` is
mapped to ``None`` and those utterances are dropped from the unified manifest.
"""

from __future__ import annotations

# --- Canonical label space ------------------------------------------------------
# Sorted, fixed order. The index of an emotion in this list IS its class id.
CANONICAL_EMOTIONS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

EMOTION_TO_IDX: dict[str, int] = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}
IDX_TO_EMOTION: dict[int, str] = {i: e for e, i in EMOTION_TO_IDX.items()}
NUM_CLASSES: int = len(CANONICAL_EMOTIONS)

# --- CREMA-D --------------------------------------------------------------------
# Filename pattern: <ActorID>_<Sentence>_<Emotion>_<Level>.wav  e.g. 1001_DFA_ANG_XX.wav
# The third underscore-separated token is the emotion code.
CREMAD_CODE_TO_CANONICAL: dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# --- MELD -----------------------------------------------------------------------
# The "Emotion" column in *_sent_emo.csv uses these lowercase strings.
# ``surprise`` is outside the common six -> None (dropped).
MELD_LABEL_TO_CANONICAL: dict[str, str | None] = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": None,
}

# --- Audio defaults -------------------------------------------------------------
# Target sample rate for ALL audio after loading/resampling. 16 kHz is standard for
# speech and is what wav2vec2-style models expect.
SAMPLE_RATE: int = 16_000

# Corpus identifiers used in the manifest "corpus" column.
CORPUS_CREMAD = "cremad"
CORPUS_MELD = "meld"


def cremad_code_to_idx(code: str) -> int | None:
    """Map a CREMA-D 3-letter emotion code to a canonical class id (or None)."""
    canon = CREMAD_CODE_TO_CANONICAL.get(code.upper())
    return EMOTION_TO_IDX[canon] if canon is not None else None


def meld_label_to_idx(label: str) -> int | None:
    """Map a MELD emotion string to a canonical class id (or None if not in common 6)."""
    canon = MELD_LABEL_TO_CANONICAL.get(label.strip().lower())
    return EMOTION_TO_IDX[canon] if canon is not None else None
