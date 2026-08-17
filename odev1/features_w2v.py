"""Wav2Vec2 ile donmuş-gömme (frozen embedding) öznitelik çıkarımı (Ödev 1).

Ödev gereği ses girdileri **Wav2Vec2** ile vektöre dönüştürülür (donmuş, ince ayar
YOK — model yalnızca öznitelik çıkarıcı olarak kullanılır). Wav2Vec2 her klip için
zaman ekseninde ilerleyen bir gizli durum dizisi ``[T, H]`` üretir; T klipten klibe
değişir ama KNN sabit boyutlu girdi ister. Bu yüzden son gizli katman (last hidden
state) zaman ekseninde havuzlanır (pooling): her gizli boyutun ortalaması,
standart sapması ve maksimumu alınıp yan yana eklenir ve ``[mean | std | max]``
= 3·H boyutlu tek bir vektör diske kaydedilir (cache).

Öznitelik vektör **boyutu bir hiperparametredir** ve ayrı ayrı çıkarım yapmak
yerine bu 3·H'lik cache dilimlenerek elde edilir (ucuz ve pratik):

  * ``mean``         → H      (768)   — yalnızca ortalama bloğu
  * ``mean_std``     → 2·H    (1536)  — ortalama + standart sapma
  * ``mean_std_max`` → 3·H    (2304)  — ortalama + std + maksimum

torch / transformers YALNIZCA bu dosyada (öznitelik çıkarımı) kullanılır; KNN
modelleme tarafı (knn_pipeline.py) yalnızca numpy/pandas/scikit-learn kullanır.
Çıkarılan vektörler cache'lendiğinden sonraki çalışmalar hızlıdır ve kesinti
durumunda kaldığı yerden devam eder (var olan .npy dosyaları yeniden üretilmez).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Doğrudan `python odev1/...` ile çalıştırıldığında proje kökündeki `ser`
# paketinin bulunabilmesi için kök dizini arama yoluna ekliyoruz.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ser.features.io import load_audio  # veri okuma (librosa) — ödev kuralına göre izinli  # noqa: E402
from ser.utils import get_logger, ensure_dir  # noqa: E402

log = get_logger("odev1.w2v")

# Havuzlama adı → cache'ten alınacak H'lik blok sayısı (768'in katları).
POOL_MULT = {"mean": 1, "mean_std": 2, "mean_std_max": 3}
DEFAULT_MODEL = "facebook/wav2vec2-base"


def _meta_hash(model_name: str, sr: int, max_seconds: float) -> str:
    """Cache dosyalarını model, örnekleme hızı ve klip süresi ayarlarına bağlayan kısa imza üretir.

    Neden gerekli? Cache dosya adının sonuna bu imza (hash) eklenir. Ayarlardan
    biri değişirse (örn. farklı model veya farklı max_seconds) imza da değişir ve
    farklı cache dosyaları kullanılır; böylece birbiriyle uyumsuz öznitelik
    vektörlerinin aynı deneyde yanlışlıkla karışması engellenir.
    md5'in ilk 10 karakteri dosya adını kısa tutmak için yeterlidir.
    """
    return hashlib.md5(f"{model_name}|{sr}|{max_seconds}".encode()).hexdigest()[:10]


def _cache_path(cache_dir: Path, corpus: str, audio_path: str, h: str) -> Path:
    """Bir ses kaydının Wav2Vec2 `.npy` cache yolunu deterministik biçimde kurar.

    Aynı girdiler her zaman aynı yolu üretir; çıkarım ve okuma tarafı bu sayede
    birbirini dosya adı üzerinden bulur.
    """
    p = Path(audio_path)
    # Üst klasör adı + dosya adı birlikte kullanılır: MELD'de dia{D}_utt{U}
    # kimlikleri her split'te (train/dev/test) baştan başladığı için yalnızca
    # dosya adı çakışabilir; klasör adını eklemek adları benzersiz yapar.
    return Path(cache_dir) / corpus / f"{p.parent.name}_{p.stem}__{h}.npy"


class W2VExtractor:
    """Tembel (lazy) Wav2Vec2 ileri geçişi → havuzlanmış [3H] vektör.

    "Tembel" çünkü model ancak gerçekten çıkarım gerektiğinde kurulur; her şey
    cache'te hazırsa torch/transformers hiç yüklenmez ve zaman kaybedilmez.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                 max_seconds: float = 6.0):
        """Donmuş Wav2Vec2 modelini bir kez yükler ve CPU/GPU cihazını seçer.

        torch ve transformers bilerek fonksiyon içinde import edilir: bu modülü
        import eden KNN tarafı böylece bu ağır kütüphaneleri hiç yüklemek
        zorunda kalmaz (ödev kuralı ile de uyumlu).
        """
        import os
        import torch
        from transformers import Wav2Vec2Model

        # Süreç başına iş parçacığı (thread) sayısını sınırla: birden çok shard
        # paralel çalışırken her biri tüm çekirdekleri kapmaya çalışırsa CPU
        # aşırı yüklenir. Ortam değişkeni TORCH_THREADS ile ayarlanır.
        nt = int(os.environ.get("TORCH_THREADS", "0"))
        if nt > 0:
            torch.set_num_threads(nt)
        self.torch = torch
        self.sr = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        # GPU varsa kullan; yoksa CPU'da da çalışır (daha yavaş ama sorunsuz).
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # .eval() modeli çıkarım moduna alır (dropout vb. kapanır);
        # ince ayar yapılmadığı için eğitim moduna hiç geçilmez.
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device).eval()
        self.H = int(self.model.config.hidden_size)
        log.info("Loaded %s (hidden=%d) on %s", model_name, self.H, self.device)

    def pool(self, wav: np.ndarray) -> np.ndarray:
        """Tek ses dalgasını sabit uzunluklu `[mean, std, max]` vektörüne çevirir.

        Adım adım: (1) ses gerekirse kırpılır/uzatılır, (2) normalize edilir,
        (3) Wav2Vec2'den geçirilir, (4) zaman ekseninde üç istatistikle
        havuzlanıp tek vektörde birleştirilir.

        `facebook/wav2vec2-base` modelinin gizli boyutu (hidden size) 768 olduğu
        için çıktı boyutu 3 * 768 = 2304'tür. `torch.no_grad()` gradyan hesabını
        kapatır: ağırlık güncellenmez, bellek ve zaman tasarrufu sağlanır.
        """
        torch = self.torch
        wav = np.asarray(wav, dtype=np.float32)
        if wav.shape[0] > self.max_samples:  # çok uzun klipleri ortadan kırp (duygu genelde ortada yoğundur)
            s = (wav.shape[0] - self.max_samples) // 2
            wav = wav[s:s + self.max_samples]
        if wav.shape[0] < 400:  # wav2vec2'nin evrişim katmanları için gereken en küçük örnek sayısı
            wav = np.pad(wav, (0, 400 - wav.shape[0]))
        # Kayıt başına normalizasyon: ortalamayı çıkar, standart sapmaya böl.
        # (+1e-5 payda sıfır olmasın diye eklenir.) Ses seviyesi farklarını giderir.
        wav = (wav - wav.mean()) / (wav.std() + 1e-5)
        # [None, :] başa batch boyutu ekler: model [batch, örnek] bekler.
        x = torch.from_numpy(wav)[None, :].to(self.device)
        with torch.no_grad():
            h = self.model(x).last_hidden_state[0]  # [T, H]: T zaman adımı, H gizli boyut
        # Zaman ekseni (0. eksen) boyunca üç özet istatistik:
        mean = h.mean(0)        # her boyutun ortalaması — genel seviye
        std = h.std(0)          # değişkenlik — duygunun dalgalanması
        mx = h.max(0).values    # tepe değer — kısa ama güçlü vurgular
        return torch.cat([mean, std, mx]).cpu().numpy().astype(np.float32)  # [3H]


def extract_all(manifest_csv: str, cache_dir: str = "odev1/cache/w2v",
                model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                max_seconds: float = 6.0, shard: int = 0, num_shards: int = 1) -> int:
    """Manifest'teki her klip için havuzlanmış Wav2Vec2 vektörünü çıkarır ve cache'ler.

    Kaldığı yerden devam edebilir (resumable): cache'te zaten olan klipler
    atlanır. Paralellik için aynı num_shards ve farklı shard (0..num_shards-1)
    değerleriyle birden çok süreç çalıştırılabilir; her süreç henüz eksik olan
    kliplerin ayrık (çakışmayan) bir dilimini işler. Manifest boyutunu döndürür.

    Özet akış: manifesti okur, cache'i olmayan sesleri bulur, `W2VExtractor.pool`
    ile 2304 boyutlu vektör üretir ve atomik biçimde `.npy` dosyasına kaydeder.
    "Atomik" olması yarım yazılmış dosya kalmasını engeller (aşağıya bakınız).
    """
    from tqdm import tqdm

    df = pd.read_csv(manifest_csv)
    # Ayar imzası: cache dosya adlarına gömülür (bkz. _meta_hash).
    h = _meta_hash(model_name, sample_rate, max_seconds)
    cache_dir = Path(cache_dir)

    # ---- 1) Yapılacak iş listesini çıkar: yalnızca cache'te OLMAYAN klipler ----
    todo = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        cp = _cache_path(cache_dir, r["corpus"], r["path"], h)
        if not cp.exists():
            todo.append((r["corpus"], r["path"], cp))
    # Shard'lama: liste üzerinde `shard::num_shards` dilimlemesi her sürece
    # farklı elemanlar verir (0. süreç 0,N,2N...; 1. süreç 1,N+1,... gibi).
    if num_shards > 1:
        todo = todo[shard::num_shards]
    log.info("Manifest=%d, to extract this shard=%d (shard %d/%d)",
             len(df), len(todo), shard, num_shards)

    # ---- 2) Model yalnızca gerçekten iş varsa kurulur (tembel yükleme) --------
    if todo:
        ext = W2VExtractor(model_name, sample_rate, max_seconds)
        for corpus, path, cp in tqdm(todo, desc="wav2vec2 features"):
            try:
                wav = load_audio(path, sample_rate)
                vec = ext.pool(wav)
            except Exception as e:
                # Tek bozuk dosya tüm işi durdurmasın: logla ve devam et.
                log.warning("extract failed %s: %s", path, e)
                continue
            ensure_dir(cp.parent)
            # Atomik yazma: önce geçici (.tmp.npy) dosyaya kaydet, sonra tek
            # hamlede asıl ada taşı (replace). Süreç tam yazma sırasında ölürse
            # yarım dosya asıl adla var olmaz; iş tekrar başlatıldığında o klip
            # eksik görünür ve yeniden üretilir. Geçici ad .npy ile bitmek
            # zorunda, çünkü np.save uzantı yoksa kendisi .npy ekler ve o zaman
            # taşınacak hedef dosya adı bulunamazdı.
            tmp = cp.with_name(cp.stem + ".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(cp)
    log.info("Done. cached vectors live under %s", cache_dir)
    return len(df)


def load_pooled(df: pd.DataFrame, pool: str, cache_dir: str = "odev1/cache/w2v",
                model_name: str = DEFAULT_MODEL, sample_rate: int = 16000,
                max_seconds: float = 6.0):
    """``df`` içindeki klipler için [N, mult*H] öznitelik matrisi + etiketleri yükler.

    Cache'lenmiş [3H] vektörlerini okur ve istenen havuz boyutuna göre dilimler.
    Yalnızca numpy kullanır — torch yok — böylece KNN aşaması ödevde izin verilen
    kütüphanelerin dışına çıkmaz.

    Dilimleme mantığı: vektör [mean | std | max] sırasıyla kaydedildiği için
    `mean` ilk 768, `mean_std` ilk 1536, `mean_std_max` tüm 2304 boyutu alır.
    Çıktı `(X, y)` çiftidir: X öznitelik matrisi, y tam sayı etiket dizisi.
    Cache'te bulunamayan satırlar atlanır ve sayısı uyarı olarak loglanır.
    """
    h = _meta_hash(model_name, sample_rate, max_seconds)
    cache_dir = Path(cache_dir)
    mult = POOL_MULT[pool]  # kaç H'lik blok alınacak (1, 2 veya 3)
    X, y = [], []
    missing = 0
    for row in df.itertuples(index=False):
        r = row._asdict()
        cp = _cache_path(cache_dir, r["corpus"], r["path"], h)
        if not cp.exists():
            missing += 1
            continue
        v = np.load(cp)
        # H'yi dosyadan türet (v uzunluğu 3H): modele göre sabit yazmaktan güvenli.
        Hd = v.shape[0] // 3
        X.append(v[:mult * Hd])
        y.append(int(r["label_idx"]))
    if missing:
        log.warning("%d clips missing from w2v cache (run extract_all first).", missing)
    # float32/int64: sklearn için yeterli hassasiyet, yarı yarıya bellek tasarrufu.
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
