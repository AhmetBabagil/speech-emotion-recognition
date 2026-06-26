"""Lightweight unit tests for the SER pipeline (synthetic audio, no downloads).

Run with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.config import Config
from ser.constants import (
    CANONICAL_EMOTIONS, NUM_CLASSES, EMOTION_TO_IDX,
    cremad_code_to_idx, meld_label_to_idx, CORPUS_CREMAD, CORPUS_MELD,
)
from ser.features.io import fix_length
from ser.features.melspec import log_mel_spectrogram, fixed_num_frames, fix_frames
from ser.features.mfcc import mfcc_statistics
from ser.data.splits import prepare_splits


def test_label_maps():
    assert NUM_CLASSES == 6
    assert cremad_code_to_idx("ANG") == EMOTION_TO_IDX["angry"]
    assert cremad_code_to_idx("hap") == EMOTION_TO_IDX["happy"]
    assert meld_label_to_idx("joy") == EMOTION_TO_IDX["happy"]
    assert meld_label_to_idx("sadness") == EMOTION_TO_IDX["sad"]
    assert meld_label_to_idx("surprise") is None  # outside common 6


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.feature.fmax = None  # exercise the None case
    p = tmp_path / "c.yaml"
    cfg.save(p)
    cfg2 = Config.from_yaml(p)
    assert cfg2.audio.sample_rate == cfg.audio.sample_rate
    assert cfg2.feature.fmax is None
    assert cfg2.model.cnn_channels == cfg.model.cnn_channels
    assert isinstance(cfg2.model.cnn_channels, tuple)  # list -> tuple coercion


def test_config_ignores_unknown_keys():
    # Unknown keys (top-level and in a subsection) must not crash loading.
    cfg = Config.from_dict({"model": {"name": "cnn", "bogus": 1}, "totally_unknown": 2})
    assert cfg.model.name == "cnn"


def test_fix_length():
    wav = np.ones(1000, dtype=np.float32)
    assert fix_length(wav, 500).shape[0] == 500          # crop
    assert fix_length(wav, 2000).shape[0] == 2000        # pad


def _tone(freq=220.0, sr=16000, dur=2.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_feature_shapes():
    cfg = Config()
    wav = _tone()
    mel = log_mel_spectrogram(wav, cfg.feature, cfg.audio.sample_rate)
    assert mel.shape[0] == cfg.feature.n_mels
    nf = fixed_num_frames(cfg.audio.num_samples, cfg.feature.hop_length)
    fixed = fix_frames(mel, nf)
    assert fixed.shape == (cfg.feature.n_mels, nf)
    stats = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)
    assert stats.ndim == 1 and stats.shape[0] == cfg.feature.n_mfcc * 3 * 2


def _synthetic_manifest(tmp_path, n_speakers=6):
    import soundfile as sf
    rows = []
    for e_idx, emo in enumerate(CANONICAL_EMOTIONS):
        for spk in range(n_speakers):
            wav = _tone(freq=150 + 50 * e_idx)
            p = tmp_path / f"spk{spk}_{emo}.wav"
            sf.write(p, wav, 16000)
            rows.append({"path": str(p), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk}",
                         "split": "", "orig_label": emo, "emotion": emo,
                         "label_idx": EMOTION_TO_IDX[emo]})
    return pd.DataFrame(rows)


def test_speaker_independent_split(tmp_path):
    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    tr, va, te = prepare_splits(df, cfg.data, seed=0)
    # disjoint speakers across folds
    s_tr, s_va, s_te = set(tr.speaker), set(va.speaker), set(te.speaker)
    assert s_tr.isdisjoint(s_va)
    assert s_tr.isdisjoint(s_te)
    assert s_va.isdisjoint(s_te)
    assert len(tr) + len(va) + len(te) == len(df)


def test_cross_corpus_split(tmp_path):
    df = _synthetic_manifest(tmp_path)
    df2 = df.copy()
    df2["corpus"] = CORPUS_MELD
    df2["speaker"] = "m_" + df2["speaker"]
    full = pd.concat([df, df2], ignore_index=True)
    cfg = Config()
    cfg.data.train_corpora = (CORPUS_CREMAD,)
    cfg.data.eval_corpora = (CORPUS_MELD,)
    tr, va, te = prepare_splits(full, cfg.data, seed=0)
    assert set(tr.corpus) == {CORPUS_CREMAD}
    assert set(te.corpus) == {CORPUS_MELD}


def test_dataset_and_cnn_forward(tmp_path):
    pytest.importorskip("torch")
    from ser.data.dataset import SERDataset
    from ser.models import build_model

    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    cfg.data.cache_features = False
    ds = SERDataset(df, cfg, mode="logmel", train=True)
    x, y = ds[0]
    assert x.shape[0] == 1 and x.shape[1] == cfg.feature.n_mels
    model = build_model(cfg, NUM_CLASSES)
    out = model(x.unsqueeze(0))
    assert out.shape == (1, NUM_CLASSES)


def test_augmentation_is_reproducible(tmp_path):
    torch = pytest.importorskip("torch")
    from ser.data.dataset import SERDataset

    df = _synthetic_manifest(tmp_path)
    cfg = Config()
    cfg.data.cache_features = False
    # Same seed + epoch + index -> identical augmented sample (reproducible).
    ds1 = SERDataset(df, cfg, mode="logmel", train=True); ds1.set_epoch(3)
    ds2 = SERDataset(df, cfg, mode="logmel", train=True); ds2.set_epoch(3)
    x1, _ = ds1[5]
    x2, _ = ds2[5]
    assert torch.allclose(x1, x2)


def test_class_weights_balanced():
    pytest.importorskip("torch")
    from ser.data.dataset import class_weights
    # class 0 is over-represented -> should get a lower weight than the rare classes
    df = pd.DataFrame({"label_idx": [0, 0, 0, 0, 1, 2, 3, 4, 5]})
    w = class_weights(df, "balanced")
    assert w.shape[0] == NUM_CLASSES
    assert w[0] < w[1]


def test_cremad_manifest_parsing(tmp_path):
    import soundfile as sf
    from ser.data.build_manifest import cremad_rows

    audiowav = tmp_path / "AudioWAV"
    audiowav.mkdir()
    for n in ["1001_DFA_ANG_XX.wav", "1002_IEO_HAP_HI.wav", "1091_TIE_SAD_LO.wav"]:
        sf.write(audiowav / n, _tone(), 16000)
    rows = cremad_rows(audiowav)
    assert len(rows) == 3
    assert {r["emotion"] for r in rows} == {"angry", "happy", "sad"}
    assert {r["speaker"] for r in rows} == {"1001", "1002", "1091"}  # actor id preserved


def test_stratified_subset():
    from ser.semisupervised import _stratified_subset
    y = np.array([0] * 100 + [1] * 100 + [2] * 100)
    rng = np.random.default_rng(0)
    idx = _stratified_subset(y, 0.1, rng)
    # ~10 per class, every class represented (>= min_per_class)
    counts = np.bincount(y[idx], minlength=3)
    assert (counts >= 1).all()
    assert 25 <= len(idx) <= 40
    # tiny fraction still keeps >= 1 per class
    idx2 = _stratified_subset(y, 0.0, rng)
    assert (np.bincount(y[idx2], minlength=3) >= 1).all()


def test_semisupervised_runs(tmp_path):
    pytest.importorskip("sklearn")
    import soundfile as sf
    from ser.semisupervised import cluster_analysis, label_efficiency

    # synthetic manifest with separable per-emotion tones, several speakers
    rows = []
    for e_idx, emo in enumerate(CANONICAL_EMOTIONS):
        for spk in range(8):
            wav = _tone(freq=140 + 60 * e_idx)
            p = tmp_path / f"spk{spk}_{emo}.wav"
            sf.write(p, wav, 16000)
            rows.append({"path": str(p), "corpus": CORPUS_CREMAD, "speaker": f"spk{spk}",
                         "split": "", "orig_label": emo, "emotion": emo,
                         "label_idx": EMOTION_TO_IDX[emo]})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    cfg = Config()
    cfg.data.manifest = str(manifest)
    cfg.data.cache_features = False
    cfg.output_dir = str(tmp_path / "out")

    clu = cluster_analysis(cfg, tmp_path / "out")
    assert clu["n_clusters"] == NUM_CLASSES
    assert -1.0 <= clu["adjusted_rand_index"] <= 1.0
    assert 0.0 <= clu["normalized_mutual_info"] <= 1.0001

    eff = label_efficiency(cfg, tmp_path / "out", fractions=(0.5, 1.0))
    assert len(eff) == 2
    assert all(0.0 <= r["macro_f1"] <= 1.0 for r in eff)
