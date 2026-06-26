"""Wav2Vec2 frozen-embedding feature extraction (Ödev 1).

Ödev gereği ses girdileri **Wav2Vec2** ile vektöre dönüştürülür (donmuş, ince ayar
YOK — yalnızca öznitelik çıkarıcı). Her klip için son gizli katman (last hidden
state) zaman ekseninde havuzlanır ve ``[mean | std | max]`` = 3·H boyutlu bir vektör
diske kaydedilir (cache). Öznitelik vektör **boyutu bir hiperparametredir** ve bu
cache'i dilimleyerek elde edilir:

  * ``mean``         → H      (768)
  * ``mean_std``     → 2·H    (1536)
  * ``mean_std_max`` → 3·H    (2304)

torch / transformers YALNIZCA bu dosyada (öznitelik çıkarımı) kullanılır; KNN
modelleme tarafı (knn_pipeline.py) yalnızca numpy/pandas/scikit-learn kullanır.
Çıkarılan vektörler cache'lendiğinden sonraki çalışmalar hızlıdır ve kesinti
durumunda kaldığı yerden devam eder.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.features.io import load_audio  # data reading (librosa) — allowed  # noqa: E402
from ser.utils import get_logger, ensure_dir  # noqa: E402

log = get_logger("odev1.w2v")

POOL_MULT = {"mean": 1, "mean_std": 2, "mean_std_max": 3}
DEFAULT_MODEL = "facebook/wav2vec2-base"


def _meta_hash(model_name: str, sr: int, max_seconds: float) -> str:
    return hashlib.md5(f"{model_name}|{sr}|{max_seconds}".encode()).hexdigest()[:10]


def _cache_path(cache_dir: Path, corpus: str, audio_path: str, h: str) -> Path:
    p = Path(audio_path)
    # parent folder + stem: MELD dia{D}_utt{U} ids restart per split, so include it
    return Path(cache_dir) / corpus / f"{p.parent.name}_{p.stem}__{h}.npy"


class W2VExtractor:
    """Lazy wav2vec2 forward → pooled [3H] vector. Built only when needed."""

    def __init__(self, model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                 max_seconds: float = 6.0):
        import os
        import torch
        from transformers import Wav2Vec2Model

        # Limit threads per process so several shards can run in parallel without
        # oversubscribing the CPU (set TORCH_THREADS in the environment).
        nt = int(os.environ.get("TORCH_THREADS", "0"))
        if nt > 0:
            torch.set_num_threads(nt)
        self.torch = torch
        self.sr = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()
        self.H = int(self.model.config.hidden_size)
        log.info("Loaded %s (hidden=%d) on %s", model_name, self.H, self.device)

    def pool(self, wav: np.ndarray) -> np.ndarray:
        torch = self.torch
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[0] > self.max_samples:  # center-crop very long clips
            s = (wav.shape[0] - self.max_samples) // 2
            wav = wav[s:s + self.max_samples]
        if wav.shape[0] < 400:  # wav2vec2 needs a minimum length
            wav = np.pad(wav, (0, 400 - wav.shape[0]))
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)  # per-utterance normalization
        x = torch.from_numpy(wav)[None, :].to(self.device)
        with torch.no_grad():
            h = self.model(x).last_hidden_state[0]  # [T, H]
        mean = h.mean(0)
        std = h.std(0)
        mx = h.max(0).values
        return torch.cat([mean, std, mx]).cpu().numpy().astype(np.float32)  # [3H]


def extract_all(manifest_csv: str, cache_dir: str = "odev1/cache/w2v",
                model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                max_seconds: float = 6.0, shard: int = 0, num_shards: int = 1) -> int:
    """Extract+cache the pooled wav2vec2 vector for every clip in the manifest.

    Resumable: clips already cached are skipped. For parallelism, run several
    processes with the same num_shards and different shard (0..num_shards-1);
    each handles a disjoint slice of the still-missing clips. Returns manifest size.
    """
    from tqdm import tqdm

    df = pd.read_csv(manifest_csv)
    h = _meta_hash(model_name, sample_rate, max_seconds)
    cache_dir = Path(cache_dir)

    todo = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        cp = _cache_path(cache_dir, r["corpus"], r["path"], h)
        if not cp.exists():
            todo.append((r["corpus"], r["path"], cp))
    if num_shards > 1:
        todo = todo[shard::num_shards]
    log.info("Manifest=%d, to extract this shard=%d (shard %d/%d)",
             len(df), len(todo), shard, num_shards)

    if todo:
        ext = W2VExtractor(model_name, sample_rate, max_seconds)
        for corpus, path, cp in tqdm(todo, desc="wav2vec2 features"):
            try:
                wav = load_audio(path, sample_rate)
                vec = ext.pool(wav)
            except Exception as e:
                log.warning("extract failed %s: %s", path, e)
                continue
            ensure_dir(cp.parent)
            # tmp must end in .npy, else np.save appends .npy and the rename target
            # would not exist. Atomic via replace().
            tmp = cp.with_name(cp.stem + ".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(cp)
    log.info("Done. cached vectors live under %s", cache_dir)
    return len(df)


def load_pooled(df: pd.DataFrame, pool: str, cache_dir: str = "odev1/cache/w2v",
                model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                max_seconds: float = 6.0):
    """Load the [N, mult*H] feature matrix + labels for ``df`` at the given pool.

    Reads cached [3H] vectors and slices to the requested size. numpy only — no
    torch — so the KNN stage stays within the allowed libraries.
    """
    h = _meta_hash(model_name, sample_rate, max_seconds)
    cache_dir = Path(cache_dir)
    mult = POOL_MULT[pool]
    X, y = [], []
    missing = 0
    for row in df.itertuples(index=False):
        r = row._asdict()
        cp = _cache_path(cache_dir, r["corpus"], r["path"], h)
        if not cp.exists():
            missing += 1
            continue
        v = np.load(cp)
        Hd = v.shape[0] // 3
        X.append(v[:mult * Hd])
        y.append(int(r["label_idx"]))
    if missing:
        log.warning("%d clips missing from w2v cache (run extract_all first).", missing)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
