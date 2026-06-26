# Running wav2vec2 transfer learning on the GPU (RTX 5080)

This guide covers the third model — **wav2vec2 transfer learning** — which is the
one experiment meant for the GPU machine. It fine-tunes a pretrained
`facebook/wav2vec2-base` speech model with a small classification head on top, on
the raw 16 kHz waveform (no spectrogram). The implementation is
`ser/models/wav2vec2.py` and the config is `configs/wav2vec2_cremad.yaml`.

> **Why GPU only?** wav2vec2-base is a ~95 M-parameter transformer over raw audio.
> A forward+backward pass on a batch of 4 s clips takes ~10–30 s **per batch** on a
> CPU, i.e. *hours per epoch* — impractical. On an RTX 5080 it is seconds per
> epoch-fraction. The MFCC-SVM baseline and the log-mel CNN run fine on CPU; only
> this model needs CUDA.

---

## 1. Prerequisites

- An NVIDIA **RTX 5080** (Blackwell, compute capability **sm_120**), 16 GB VRAM.
- A recent NVIDIA driver (the one bundled with CUDA 12.8+ support). Verify the GPU
  is visible:
  ```bash
  nvidia-smi
  ```
  You should see the RTX 5080 and a driver version. (If `nvidia-smi` isn't found,
  install/repair the NVIDIA driver first.)
- Python ≥ 3.10 and the project cloned:
  ```bash
  git clone https://github.com/AhmetBabagil/speech-emotion-recognition.git
  cd speech-emotion-recognition
  ```

## 2. Install PyTorch for Blackwell (CUDA 12.8 / cu128)

The RTX 50-series needs a **cu128** PyTorch build. A default `pip install torch`
gives a CPU or older-CUDA wheel that **cannot** run on sm_120, so install from the
cu128 index explicitly:

```bash
# (recommended) a fresh virtual environment first
python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then **verify CUDA actually works on the 5080**:

```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

Expected: a `...+cu128` version, `cuda: True`, and `NVIDIA GeForce RTX 5080`.
If `cuda: False` or you get an `sm_120 is not compatible` warning, you have the
wrong wheel — reinstall from the cu128 index above. If the stable cu128 build still
doesn't recognise sm_120, use the nightly cu128 index as a fallback:
`pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`.

## 3. Install the project + the transfer-learning extra

```bash
pip install -e .            # core deps (numpy, librosa, scikit-learn, ...)
pip install -e .[transfer]  # adds `transformers` -- REQUIRED for wav2vec2
```

`transformers` is what loads `Wav2Vec2Model`. Without it the model raises a clear
`ImportError`.

## 4. Get the data + manifest (same as the rest of the project)

```bash
python scripts/download_data.py --datasets cremad meld   # CREMA-D + MELD
python scripts/build_manifest.py                          # -> data/processed/manifest.csv
```

CREMA-D alone is enough to start (`--datasets cremad`); MELD is only needed for the
within-MELD and cross-corpus wav2vec2 runs.

## 5. Train wav2vec2

```bash
# Within CREMA-D (speaker-independent):
python scripts/train.py --config configs/wav2vec2_cremad.yaml

# Within MELD (speaker-independent) -- override the corpora:
python scripts/train.py --config configs/wav2vec2_cremad.yaml \
    --experiment wav2vec2_meld --train-corpora meld --eval-corpora meld

# Full within + cross-corpus matrix (trains 4 wav2vec2 models):
python scripts/train.py --config configs/wav2vec2_cremad.yaml --cross-corpus --experiment wav2vec2
```

Outputs land in `outputs/<experiment>/` exactly like the CNN: `best.pt`,
`history.json`, `test_metrics.json`, `test_confusion_matrix.png`, `test_summary.json`.
Aggregate everything (baseline + CNN + wav2vec2) into one table with:

```bash
python scripts/aggregate_results.py    # -> outputs/results.csv / results.md
```

**First run** downloads `facebook/wav2vec2-base` (~360 MB) into the Hugging Face
cache; subsequent runs reuse it.

## 6. Config & VRAM tuning (`configs/wav2vec2_cremad.yaml`)

The defaults are sized for a 16 GB card:

| Field | Default | Meaning / tuning |
|---|---|---|
| `model.pretrained_name` | `facebook/wav2vec2-base` | Backbone. `...-large` is stronger but ~3× the VRAM. |
| `model.freeze_feature_encoder` | `true` | Freezes the conv feature encoder (recommended for small SER datasets — fewer params, less VRAM, less overfit). Set `false` to fine-tune everything (more VRAM). |
| `train.batch_size` | `8` | **Main VRAM lever.** Drop to 4 (or 2) if you hit OOM. |
| `audio.clip_seconds` | `4.0` | Clip length. Shorter (e.g. 3.0) → less VRAM/compute. |
| `train.amp` | `true` | Mixed precision (fp16) — roughly halves activation memory on CUDA; auto-disabled on CPU. |
| `train.lr` | `1e-4` | Good default for a fine-tuned transformer + head. |
| `train.num_workers` | `4` | Data-loading workers. On **Windows**, set `0` if you see worker/pickling errors. |
| `train.epochs` | `20` | Early stopping (patience 5) usually stops earlier. |

> There is no gradient accumulation in the trainer, so **`batch_size` is the way to
> trade VRAM for throughput**. If you must use a tiny batch, also lower `lr` a little.

Per-instance waveform normalization (zero-mean/unit-variance) is done in the dataset
(`mode="waveform"`), matching what wav2vec2 expects, and waveform mode is **not
cached** (`cache_features: false`) — caching only applies to log-mel.

## 7. Expected behaviour

- VRAM: with the defaults (base backbone, frozen feature encoder, batch 8, AMP) the
  run fits comfortably under 16 GB.
- Speed: vastly faster than CPU — typically a few minutes per epoch for CREMA-D on
  the 5080 (vs. hours on CPU).
- Result: wav2vec2 transfer learning is generally expected to **match or beat** the
  log-mel CNN (CREMA-D CNN macro-F1 ≈ 0.557) within-corpus; cross-corpus it usually
  still drops sharply (the domain-shift finding holds — see `docs/RAPOR.md`).

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `torch.cuda.is_available()` is `False` | CPU/old-CUDA wheel installed. Reinstall from the **cu128** index (§2). |
| `sm_120 is not compatible with the current PyTorch installation` | Same — you need a cu128 build (stable, else nightly cu128). |
| `ImportError: wav2vec2 model requires transformers` | Run `pip install -e .[transfer]`. |
| `CUDA out of memory` | Lower `train.batch_size` (8→4→2), and/or `audio.clip_seconds` (4→3), keep `amp: true`, keep `freeze_feature_encoder: true`. |
| Hangs / pickling error on Windows with workers | Set `train.num_workers: 0`. |
| Slow first run | It's downloading `facebook/wav2vec2-base` (~360 MB) once. |
| `ffmpeg not found` (only for MELD) | Install ffmpeg and put it on PATH. |

## 9. Going further (optional)

- **Unfreeze the feature encoder:** `freeze_feature_encoder: false` — can help on
  larger data, costs more VRAM.
- **Bigger backbones:** set `model.pretrained_name` to `facebook/wav2vec2-large-960h`
  or a HuBERT checkpoint (`facebook/hubert-base-ls960`) — the code path is identical
  (any `Wav2Vec2Model`-compatible checkpoint); reduce `batch_size` accordingly.
- **Full wav2vec2 cross-corpus matrix** for the report:
  `python scripts/train.py --config configs/wav2vec2_cremad.yaml --cross-corpus --experiment wav2vec2`,
  then `python scripts/aggregate_results.py` and drop the numbers into
  `docs/RAPOR.md` §5 next to the CNN/LogReg columns.
