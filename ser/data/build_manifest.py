"""Build the unified manifest CSV from CREMA-D and MELD.

Output columns (one row per usable utterance):
    path        absolute/relative path to a WAV file
    corpus      'cremad' | 'meld'
    speaker     CREMA-D actor id  /  MELD speaker name (for speaker-independent splits)
    split       '' for CREMA-D, 'train'/'dev'/'test' for MELD (official folds)
    orig_label  the dataset's own label (emotion code/string)
    emotion     canonical label (angry/disgust/fear/happy/neutral/sad)
    label_idx   canonical class id 0..5

Only the common six emotions are kept; MELD 'surprise' rows are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..constants import (
    CREMAD_CODE_TO_CANONICAL,
    EMOTION_TO_IDX,
    MELD_LABEL_TO_CANONICAL,
    CORPUS_CREMAD,
    CORPUS_MELD,
)
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)

SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


def cremad_rows(audiowav_dir: str | Path) -> list[dict]:
    audiowav_dir = Path(audiowav_dir)
    rows = []
    for wav in sorted(audiowav_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        actor, _sentence, code = parts[0], parts[1], parts[2]
        canon = CREMAD_CODE_TO_CANONICAL.get(code.upper())
        if canon is None:
            continue
        rows.append({
            "path": str(wav),
            "corpus": CORPUS_CREMAD,
            "speaker": actor,
            "split": "",
            "orig_label": code.upper(),
            "emotion": canon,
            "label_idx": EMOTION_TO_IDX[canon],
        })
    log.info("CREMA-D: %d usable rows from %s", len(rows), audiowav_dir)
    return rows


def meld_rows(csv_dir: str | Path, audio_root: str | Path) -> list[dict]:
    csv_dir = Path(csv_dir)
    audio_root = Path(audio_root)
    rows = []
    for split, csv_name in SPLIT_CSV.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            log.warning("MELD: missing %s, skipping %s", csv_name, split)
            continue
        df = pd.read_csv(csv_path)
        kept = 0
        for r in df.itertuples(index=False):
            d = r._asdict()
            emotion = str(d["Emotion"]).strip().lower()
            canon = MELD_LABEL_TO_CANONICAL.get(emotion)
            if canon is None:
                continue
            key = f"dia{int(d['Dialogue_ID'])}_utt{int(d['Utterance_ID'])}"
            wav = audio_root / split / f"{key}.wav"
            if not wav.exists():
                continue  # audio not extracted (corrupt clip or not yet run)
            rows.append({
                "path": str(wav),
                "corpus": CORPUS_MELD,
                "speaker": str(d["Speaker"]).strip(),
                "split": split,
                "orig_label": emotion,
                "emotion": canon,
                "label_idx": EMOTION_TO_IDX[canon],
            })
            kept += 1
        log.info("MELD %s: %d usable rows", split, kept)
    return rows


def build_manifest(
    cremad_dir: str | Path | None = "data/raw/cremad/AudioWAV",
    meld_csv_dir: str | Path | None = None,
    meld_audio_root: str | Path | None = "data/raw/meld/audio",
    out_path: str | Path = "data/processed/manifest.csv",
) -> pd.DataFrame:
    rows: list[dict] = []

    if cremad_dir and Path(cremad_dir).is_dir():
        rows += cremad_rows(cremad_dir)
    else:
        log.warning("CREMA-D dir not found: %s", cremad_dir)

    # Auto-locate MELD csv dir if not given.
    if meld_csv_dir is None:
        guess = Path("data/raw/meld")
        found = list(guess.rglob("train_sent_emo.csv"))
        meld_csv_dir = found[0].parent if found else None
    if meld_csv_dir and meld_audio_root and Path(meld_audio_root).is_dir():
        rows += meld_rows(meld_csv_dir, meld_audio_root)
    else:
        log.warning("MELD audio/CSV not found (csv_dir=%s audio=%s)", meld_csv_dir, meld_audio_root)

    if not rows:
        raise RuntimeError("No data found for either corpus. Run the download scripts first.")

    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    log.info("Manifest written: %s (%d rows)", out_path, len(df))
    log.info("Class distribution:\n%s", df.groupby(["corpus", "emotion"]).size())
    return df
