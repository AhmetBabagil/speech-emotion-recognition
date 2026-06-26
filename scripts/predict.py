"""Predict the emotion of a single audio file with a trained model.

This is the proposal's input->output in action: a WAV in, a predicted emotion
plus the full probability distribution out (acoustics only).

Examples
--------
    # With a trained CNN / wav2vec2 checkpoint:
    python scripts/predict.py --checkpoint outputs/cnn_cremad/best.pt --audio some.wav

    # With the classical baseline:
    python scripts/predict.py --baseline outputs/baseline_cremad/baseline.joblib --audio some.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config  # noqa: E402
from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES  # noqa: E402
from ser.features.io import load_audio, fix_length  # noqa: E402
from ser.features.melspec import (  # noqa: E402
    log_mel_spectrogram, fixed_num_frames, fix_frames, standardize,
)
from ser.features.mfcc import mfcc_statistics  # noqa: E402


def _print_distribution(probs: np.ndarray) -> None:
    order = np.argsort(probs)[::-1]
    print(f"\nPredicted emotion: {CANONICAL_EMOTIONS[order[0]]}  "
          f"(p={probs[order[0]]:.3f})\n")
    print("Full distribution:")
    for i in order:
        bar = "#" * int(round(probs[i] * 30))
        print(f"  {CANONICAL_EMOTIONS[i]:<8} {probs[i]:6.3f} {bar}")


def predict_torch(checkpoint: str, audio: str) -> np.ndarray:
    import torch
    from ser.models import build_model

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ckpt["config"])
    model = build_model(cfg, NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.eval()

    wav = load_audio(audio, cfg.audio.sample_rate)
    if cfg.model.name.lower() == "wav2vec2":
        wav = fix_length(wav, cfg.audio.num_samples)
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)
        x = torch.from_numpy(wav.astype(np.float32))[None, :]
    else:
        spec = log_mel_spectrogram(wav, cfg.feature, cfg.audio.sample_rate)
        nf = fixed_num_frames(cfg.audio.num_samples, cfg.feature.hop_length)
        spec = standardize(fix_frames(spec, nf))
        x = torch.from_numpy(np.ascontiguousarray(spec))[None, None, :, :]

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].numpy()
    return probs


def predict_baseline(model_path: str, audio: str) -> np.ndarray:
    import joblib

    pipe = joblib.load(model_path)
    # Use the exact feature params the model was trained with (saved alongside it),
    # mirroring predict_torch. Fall back to defaults only if no config is present.
    cfg_path = Path(model_path).parent / "config.yaml"
    cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()
    wav = load_audio(audio, cfg.audio.sample_rate)
    feat = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)[None, :]
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(feat)[0]
    # SVC without probability -> one-hot the predicted label
    pred = int(pipe.predict(feat)[0])
    probs = np.zeros(NUM_CLASSES, dtype=np.float32)
    probs[pred] = 1.0
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="Path to a WAV file.")
    ap.add_argument("--checkpoint", default=None, help="Trained torch checkpoint (best.pt).")
    ap.add_argument("--baseline", default=None, help="Trained baseline (baseline.joblib).")
    args = ap.parse_args()

    if not Path(args.audio).exists():
        raise SystemExit(f"Audio file not found: {args.audio}")
    if args.checkpoint:
        probs = predict_torch(args.checkpoint, args.audio)
    elif args.baseline:
        probs = predict_baseline(args.baseline, args.audio)
    else:
        raise SystemExit("Provide --checkpoint <best.pt> or --baseline <baseline.joblib>.")

    _print_distribution(probs)


if __name__ == "__main__":
    main()
