# Speech Emotion Recognition (Konuşmadan Duygu Tanıma)

Cross-corpus speech emotion recognition for **YAP 470 / BİL 570** (Group 23).
Classify a short speech clip into one emotion from its **acoustic/prosodic**
properties only — no transcription, no text, no language model.

Two corpora are unified onto the **common six emotions**
`angry · disgust · fear · happy · neutral · sad`, enabling both within-corpus and
**cross-corpus (domain-shift) generalization** analysis:

| Corpus | Type | #Clips (common 6) | Speakers | Condition |
|--------|------|-------------------|----------|-----------|
| **CREMA-D** | acted, studio | ~7,442 (all 6) | 91 actors | controlled |
| **MELD** | TV dialogues | ~10k of 13,708 (drops `surprise`) | many | in-the-wild |

> **Note on `surprise`:** MELD has 7 emotions; CREMA-D has 6. We train/evaluate on
> the 6-class intersection, so MELD's `surprise` utterances are dropped.

## What's implemented

- **Data pipeline** — scripted download of CREMA-D (git-lfs sparse pull of audio)
  and MELD (download + `ffmpeg` audio extraction), unified into one manifest.
- **Features** — log-mel spectrograms (deep models) and MFCC summary statistics
  (classical baseline).
- **Three models**
  1. `baseline` — MFCC stats → StandardScaler → SVM/LogReg/RandomForest (sklearn).
  2. `cnn` — 2-D CNN over log-mel spectrograms (PyTorch).
  3. `wav2vec2` — transfer learning, fine-tuning a pretrained `wav2vec2-base` head.
- **Protocols** — **speaker-independent** within-corpus splits, MELD official
  folds, and the full **cross-corpus matrix** (train A → test B).
- **Metrics** — accuracy, balanced accuracy, macro-F1, weighted-F1, per-class
  precision/recall/F1, and confusion matrices (saved as JSON + PNG).

## Project structure

```
ser/
  constants.py        canonical 6-class label space + corpus label maps
  config.py           dataclass config (YAML)
  utils.py            device/seed/logging helpers
  features/           audio I/O, log-mel, MFCC
  data/               download_cremad, download_meld, build_manifest, splits, dataset
  models/             baseline (sklearn), cnn, wav2vec2
  train.py            training loops (deep + baseline)
  evaluate.py         metrics + confusion matrix
  cross_corpus.py     within/cross-corpus experiment matrix
configs/              cnn_cremad, cnn_meld, baseline_cremad, wav2vec2_cremad, smoke
scripts/              download_data, build_manifest, train, smoke_test
tests/                test_pipeline.py (synthetic end-to-end)
```

## Setup

Python ≥ 3.10. Install PyTorch **first** (the right wheel depends on your machine),
then the rest.

### A) GPU machine — RTX 5080 (Blackwell, sm_120) ← real training runs here
The RTX 50-series needs a **CUDA 12.8** PyTorch build:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .            # core deps from pyproject
pip install -e .[transfer]  # adds transformers, for the wav2vec2 model
```
If `torch.cuda.is_available()` is `False` on the 5080, you have a CPU/old-CUDA
wheel — reinstall from the **cu128** index above (nightly `…/nightly/cu128` is the
fallback).

### B) CPU dev box (e.g. this machine — no NVIDIA GPU)
```bash
pip install torch torchaudio   # CPU build from PyPI
pip install -e .
```
Everything runs on CPU; use `configs/smoke.yaml` / small subsets for quick checks.

`ffmpeg` must be on PATH (needed only for MELD audio extraction).

## 1. Get the data

CREMA-D downloads by default from the Hugging Face mirror `AbstractTTS/CREMA-D`
(needs `pip install datasets` — also covered by `pip install -e .[data]`). This is
used instead of the GitHub repo because GitHub's git-LFS bandwidth for CREMA-D is
frequently throttled to a standstill. The mirror keeps the **original filenames**
(`1001_DFA_ANG_XX.wav`), so the actor id needed for speaker-independent splits is
preserved. Pass `--cremad-method lfs` to use the canonical GitHub git-lfs route
instead.

```bash
# CREMA-D only (~520 MB via Hugging Face, fast):
python scripts/download_data.py --datasets cremad

# CREMA-D + MELD (MELD.Raw.tar.gz is ~10 GB + ffmpeg extraction — do this on a
# machine with disk/time to spare):
python scripts/download_data.py --datasets cremad meld
```

## 2. Build the unified manifest

```bash
python scripts/build_manifest.py
# -> data/processed/manifest.csv  (path, corpus, speaker, split, emotion, label_idx)
```

## 3. Train & evaluate

```bash
# Classical baseline (CREMA-D, speaker-independent):
python scripts/train.py --config configs/baseline_cremad.yaml --baseline

# CNN on log-mel (CREMA-D):
python scripts/train.py --config configs/cnn_cremad.yaml

# CNN on MELD (official folds):
python scripts/train.py --config configs/cnn_meld.yaml

# wav2vec2 transfer learning (GPU + transformers):
python scripts/train.py --config configs/wav2vec2_cremad.yaml

# Full within + cross-corpus matrix (the proposal's headline result):
python scripts/train.py --config configs/cnn_cremad.yaml --cross-corpus
```

Outputs land in `outputs/<experiment>/`: `config.yaml`, `test_metrics.json`,
`test_confusion_matrix.png`, and `test_summary.json` (the last is what
`scripts/aggregate_results.py` reads); plus `best.pt` and `history.json` for the
deep models, or `baseline.joblib` for the baseline. The cross-corpus run
additionally writes `outputs/<exp>_crosscorpus/summary.csv` and
`macro_f1_matrix.png`.

CLI overrides (no need to edit YAML): `--epochs`, `--batch-size`, `--model`,
`--experiment`, `--train-corpora`, `--eval-corpora`, `--split`.

## Quick sanity check (no real data needed)

```bash
python scripts/smoke_test.py        # synthesizes tiny audio, runs baseline + CNN
# or:  pytest -q
```

## Evaluation protocol notes

- **Speaker-independent** splits: no speaker appears in two folds — the correct,
  harder protocol for SER (otherwise the model memorizes voices, not emotion).
- **Cross-corpus** is the hard case: train on CREMA-D (clean/acted), test on MELD
  (noisy/natural), and vice-versa. A large within→cross drop is expected and is
  itself the finding the proposal targets.
- **Class imbalance** (MELD is neutral-heavy) is handled with balanced class
  weights; we report **macro-F1** and **balanced accuracy** alongside accuracy.

## Hardware note

The proposal targets an RTX 5080 (16 GB). Deep training (CNN, wav2vec2) should run
there with `amp: true`. The classical baseline and the smoke test run fine on CPU.
