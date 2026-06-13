"""End-to-end sanity check on SYNTHETIC audio (no downloads needed).

Generates a few seconds of distinguishable audio per (emotion, speaker), writes a
manifest, then runs the classical baseline and a 2-epoch CNN through the real code
paths. Verifies that data -> features -> train -> evaluate -> report all work and
that artifacts are produced. Not a measure of real accuracy.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, EMOTION_TO_IDX, CORPUS_CREMAD  # noqa: E402
from ser.utils import get_logger, ensure_dir  # noqa: E402

log = get_logger("smoke")

SR = 16000
DUR = 2.0
N_SPEAKERS = 6
REPS = 3
ROOT = Path("data/smoke")


def _synth_clip(emotion_idx: int, speaker_idx: int, rep: int) -> np.ndarray:
    """Make a clip whose spectral content depends on the emotion (so a model can
    actually separate classes), plus speaker-dependent timbre and noise."""
    rng = np.random.default_rng(1000 * emotion_idx + 10 * speaker_idx + rep)
    t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)
    base = 180.0 + 90.0 * emotion_idx          # emotion -> pitch
    formant = 500.0 + 120.0 * speaker_idx        # speaker -> timbre
    sig = (
        0.6 * np.sin(2 * np.pi * base * t)
        + 0.3 * np.sin(2 * np.pi * formant * t)
        + 0.1 * np.sin(2 * np.pi * (2 * base) * t)
    )
    # emotion-dependent amplitude envelope (tempo/energy cue)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * (1 + emotion_idx) * t / DUR)
    sig = sig * env + 0.02 * rng.standard_normal(t.shape)
    return (sig / np.max(np.abs(sig) + 1e-8)).astype(np.float32)


def build_synthetic_dataset() -> Path:
    import soundfile as sf

    audio_dir = ensure_dir(ROOT / "audio")
    rows = []
    for e_idx, emotion in enumerate(CANONICAL_EMOTIONS):
        for spk in range(N_SPEAKERS):
            for rep in range(REPS):
                clip = _synth_clip(e_idx, spk, rep)
                fname = f"spk{spk:02d}_{emotion}_{rep}.wav"
                path = audio_dir / fname
                sf.write(path, clip, SR)
                rows.append({
                    "path": str(path), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk:02d}",
                    "split": "", "orig_label": emotion, "emotion": emotion,
                    "label_idx": EMOTION_TO_IDX[emotion],
                })
    manifest = ROOT / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    log.info("Synthetic dataset: %d clips -> %s", len(rows), manifest)
    return manifest


def main():
    manifest = build_synthetic_dataset()

    cfg = Config.from_yaml("configs/smoke.yaml")
    cfg.data.manifest = str(manifest)
    cfg.data.cache_dir = str(ROOT / "cache")
    cfg.output_dir = str(ROOT / "outputs")

    # 1) classical baseline
    from ser.train import train_baseline, train_torch
    log.info("--- baseline (logreg) ---")
    cfg.experiment = "smoke_baseline"
    bm = train_baseline(cfg, kind="logreg")

    # 2) CNN
    log.info("--- cnn (2 epochs) ---")
    cfg.experiment = "smoke_cnn"
    cm = train_torch(cfg)

    print("\n=== SMOKE TEST OK ===")
    print(f"baseline: acc={bm['accuracy']:.3f}  macroF1={bm['macro_f1']:.3f}")
    print(f"cnn     : acc={cm['accuracy']:.3f}  macroF1={cm['macro_f1']:.3f}")
    print("Artifacts under", ROOT / "outputs")


if __name__ == "__main__":
    main()
