"""Spektrogram/dalga-formu modelleri için PyTorch dataset'i + disk önbelleği,
ve klasik taban modelinin kullandığı MFCC öznitelik-matrisi üreticisi.

Önbellek neden var? Log-mel spektrogram çıkarmak (STFT + mel filtre bankası)
her örnek için pahalıdır ve her epoch'ta aynı dosya için aynı sonucu üretir.
Bu yüzden "tam" spektrogram bir kez hesaplanıp .npy olarak diske yazılır;
sonraki erişimler yalnızca dosya okur. Kırpma/maskeleme gibi RASTGELE işlemler
ise önbelleğe girmez — onlar her epoch değişmelidir ve ucuzdur.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..constants import NUM_CLASSES
from ..features import io as audio_io
from ..features.melspec import (
    log_mel_spectrogram,
    fix_frames,
    fixed_num_frames,
    standardize,
    spec_augment,
)
from ..features.mfcc import mfcc_statistics
from ..utils import get_logger, ensure_dir

log = get_logger(__name__)


def _feature_hash(cfg, kind: str) -> str:
    """Önbelleklenen dizileri etkileyen öznitelik parametrelerinin kısa, kararlı hash'i.

    Amaç: config'te örneğin n_mels 64'ten 80'e çıkarsa hash değişir, dolayısıyla
    dosya adları değişir ve ESKİ önbellek yanlışlıkla okunmaz — bayat önbellek,
    fark edilmesi çok güç hatalara yol açardı. MD5 burada güvenlik için değil,
    yalnızca kısa ve kararlı bir anahtar üretmek için kullanılır.
    """
    f = cfg.feature
    key = (
        kind,                    # "logmel" ve "mfccstat" önbellekleri ayrışsın
        cfg.audio.sample_rate,
        f.n_fft,
        f.hop_length,
        f.win_length,
        f.n_mels,
        f.fmin,
        f.fmax,
        f.n_mfcc,
    )
    return hashlib.md5(repr(key).encode()).hexdigest()[:10]


def _cache_path(cache_dir: Path, corpus: str, audio_path: str, h: str) -> Path:
    """Bir ses dosyasının önbellek (.npy) yolunu üretir.

    Anahtara üst klasör adı da eklenir. Nedeni: MELD'in dia{D}_utt{U} kimlikleri
    her split'te BAŞTAN başlar; yani dia0_utt0 hem audio/train, hem audio/dev,
    hem audio/test altında FARKLI klipler olarak vardır. Yalnızca dosya gövdesi
    (stem) ile anahtarlasaydık bu üçü çakışır ve yanlış split'in spektrogramı
    yüklenirdi. (CREMA-D adları zaten global benzersizdir; üst klasör = AudioWAV.)
    """
    p = Path(audio_path)
    return cache_dir / corpus / f"{p.parent.name}_{p.stem}__{h}.npy"


def _load_cached(path: Path):
    """Önbellekten diziyi okumayı dener; dosya yoksa/bozuksa sessizce None döner.

    Bozuk bir .npy (örn. yarıda kesilmiş yazma) koşuyu düşürmemeli: None dönünce
    öznitelik yeniden hesaplanır ve önbellek tazelenir.
    """
    try:
        return np.load(path)
    except Exception:
        return None


def _save_cached(path: Path, arr: np.ndarray) -> None:
    """Diziyi önbelleğe atomik biçimde yazar (önce .tmp, sonra yeniden adlandır).

    Neden iki aşama? Yazma ortasında süreç ölürse yarım dosya kalır; .tmp'ye
    yazıp sonra replace etmek, önbellekte ya TAM dosya ya HİÇ dosya olmasını
    garantiler. Önbellek "elden geldiğince" (best-effort) bir hızlandırmadır:
    yazma başarısız olursa yalnızca debug logu düşülür, eğitim devam eder.
    """
    try:
        ensure_dir(path.parent)
        tmp = path.with_suffix(".npy.tmp")
        np.save(tmp, arr)
        tmp.replace(path)
    except Exception as e:  # önbellekleme best-effort'tur; hatası ölümcül değildir
        log.debug("cache write failed for %s: %s", path, e)


class SERDataset:
    """Manifest DataFrame'i üzerinde bir torch.utils.data.Dataset.

    İki mod, iki model ailesine karşılık gelir:
      mode="logmel"   -> (FloatTensor[1, n_mels, T], label) döndürür — CNN için
                         (baştaki 1, tek "görüntü kanalı" demektir).
      mode="waveform" -> (FloatTensor[num_samples], label) döndürür — wav2vec2 için
                         (o model ham dalga formunu kendi içinde işler).

    Not: torch.utils.data.Dataset'ten miras almadan sadece __len__/__getitem__
    tanımlamak yeterlidir; DataLoader "duck typing" ile bu ikisini kullanır.
    """

    def __init__(self, df: pd.DataFrame, cfg, *, mode: str = "logmel", train: bool = False):
        # torch importu yerelde: torch kurulmamış ortamlarda da modülün geri
        # kalanı (örn. manifest araçları) import edilebilsin.
        import torch

        self.torch = torch
        # reset_index: bölme sonrası DataFrame'in indeksleri delik deşiktir;
        # iloc ile 0..N-1 aralığında güvenle erişebilmek için sıfırlanır.
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.mode = mode
        self.train = train
        # Tekrarlanabilir augmentasyon tasarımı: her örnek için RNG,
        # (global seed, epoch, örnek indeksi) üçlüsünden tohumlanır. Böylece
        #   * aynı koşu tekrarında aynı augmentasyonlar üretilir (tam
        #     tekrarlanabilirlik),
        #   * epoch değiştikçe augmentasyon değişir (gerçek çeşitlilik),
        #   * paylaşılan değişken durum olmadığından DataLoader işçi
        #     süreçleriyle de güvenlidir (her __getitem__ kendi RNG'sini kurar).
        self.seed = int(cfg.train.seed)
        self.epoch = 0
        self.num_samples = cfg.audio.num_samples
        # Sabit klibin spektrogramda kaç zaman karesine denk geldiği önceden
        # hesaplanır: her örnek bu genişliğe kırpılır/doldurulur ki batch'lensin.
        self.num_frames = fixed_num_frames(self.num_samples, cfg.feature.hop_length)
        # Önbellek yalnızca logmel modunda kullanılır: dalga formu zaten "ham"
        # olduğu için önbelleklemenin kazandıracağı hesap yok denecek kadar azdır.
        self.use_cache = cfg.data.cache_features and mode == "logmel"
        self.cache_dir = Path(cfg.data.cache_dir)
        self._hash = _feature_hash(cfg, "logmel")

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Epoch numarasını günceller: augmentasyon epoch'lar arasında değişir
        ama aynı (seed, epoch, idx) üçlüsü hep aynı sonucu üretir (tekrarlanabilir)."""
        self.epoch = int(epoch)

    def _full_logmel(self, row) -> np.ndarray:
        """Bir kaydın TAM (kırpılmamış) log-mel spektrogramını getirir.

        Sıra: önce önbelleğe bak; yoksa sesi yükle, spektrogramı hesapla,
        önbelleğe yaz. Kırpma/normalizasyon burada YAPILMAZ — onlar örnek
        bazında ve (eğitimde) rastgele olduğundan __getitem__'e aittir.
        """
        cache_p = _cache_path(self.cache_dir, row["corpus"], row["path"], self._hash)
        if self.use_cache:
            cached = _load_cached(cache_p)
            if cached is not None:
                return cached
        wav = audio_io.load_audio(row["path"], self.cfg.audio.sample_rate)
        spec = log_mel_spectrogram(wav, self.cfg.feature, self.cfg.audio.sample_rate)
        if self.use_cache:
            _save_cached(cache_p, spec)
        return spec

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["label_idx"])
        # RNG yalnızca eğitimde kurulur (değerlendirme deterministiktir:
        # merkez kırpma, maskeleme yok). Tohum listesi [seed, epoch, idx]:
        # aynı örnek farklı epoch'ta farklı, aynı koşu tekrarında aynı davranır.
        rng = np.random.default_rng([self.seed, self.epoch, idx]) if self.train else None

        if self.mode == "waveform":
            wav = audio_io.load_audio(row["path"], self.cfg.audio.sample_rate)
            # Sabit uzunluğa getir (eğitimde rastgele, değerlendirmede merkez kırpma).
            wav = audio_io.fix_length(wav, self.num_samples, random_crop=self.train, rng=rng)
            # Sıfır-ortalama / birim-varyans normalizasyonu: wav2vec2 ön
            # eğitiminde girişini böyle görmüştür; 1e-5 sessiz kliplerde
            # 0'a bölünmeyi engeller.
            wav = (wav - wav.mean()) / (wav.std() + 1e-5)
            return self.torch.from_numpy(wav.astype(np.float32)), label

        # mode == "logmel" — CNN yolu:
        spec = self._full_logmel(row)                                        # 1) tam spektrogram (önbellekli)
        spec = fix_frames(spec, self.num_frames, random_crop=self.train, rng=rng)  # 2) sabit genişliğe kırp/doldur
        spec = standardize(spec)                                             # 3) örnek-içi normalizasyon
        if self.train and self.cfg.feature.augment:
            # 4) SpecAugment yalnızca eğitimde: normalizasyondan SONRA uygulanır
            #    ki maskelenen bölgeler tam 0 (yani ortalama) değerinde kalsın.
            spec = spec_augment(spec, self.cfg.feature.freq_mask,
                                self.cfg.feature.time_mask, rng=rng)
        # ascontiguousarray: kırpma "view" üretmiş olabilir; torch'a bitişik
        # bellek ver. [None, :, :] başa kanal boyutu ekler: [1, n_mels, T].
        tensor = self.torch.from_numpy(np.ascontiguousarray(spec))[None, :, :]
        return tensor, label


def class_weights(df: pd.DataFrame, scheme: str = "balanced"):
    """CrossEntropyLoss için NUM_CLASSES uzunluğunda ağırlık tensörü döndürür.

    Amaç sınıf dengesizliğini telafi etmek: nadir sınıfın hatası daha pahalı
    olur, model "hep çoğunluk sınıfını söyle" kolaycılığına kaçamaz.

    "balanced": n_toplam / (n_sınıf * sayı_c)   (sklearn'ün formülü;
                dengeli veride tüm ağırlıklar 1'e yakın çıkar)
    "inverse" : 1 / sayı_c (ortalaması 1 olacak şekilde normalize edilir;
                "balanced"tan daha sert bir düzeltme)
    "none"    : hepsi 1 (ağırlıksız)
    """
    import torch

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in df["label_idx"].astype(int):
        counts[c] += 1
    # Hiç örneği olmayan sınıf için 0'a bölmeyi önle (sayıyı 1 varsay).
    counts = np.maximum(counts, 1.0)
    if scheme == "none":
        w = np.ones(NUM_CLASSES)
    elif scheme == "inverse":
        w = 1.0 / counts
        w = w / w.mean()   # ortalama 1: kaybın genel ölçeği değişmesin
    else:  # balanced
        w = counts.sum() / (NUM_CLASSES * counts)
    return torch.tensor(w, dtype=torch.float32)


def mfcc_feature_matrix(df: pd.DataFrame, cfg, *, show_progress: bool = True):
    """``df`` için [N, D] MFCC-istatistik matrisi ve etiket vektörü hesaplar.

    Klasik (sklearn) taban modeli tarafından kullanılır. Her dosyanın öznitelik
    vektörü ayrı bir .npy olarak önbelleklenir: aynı manifest üzerinde ikinci
    koşu (ör. farklı sınıflandırıcı denemek) saniyeler sürer.

    Okunamayan dosyalar koşuyu düşürmez: uyarı loglanır ve satır atlanır —
    binlerce dosyalık bir işte tek bozuk WAV her şeyi durdurmasın.
    """
    from tqdm import tqdm

    cache_dir = Path(cfg.data.cache_dir)
    # Hash "mfccstat" türüyle üretilir: logmel önbelleğiyle asla karışmaz.
    h = _feature_hash(cfg, "mfccstat")
    X, y = [], []
    skipped = 0
    it = df.itertuples(index=False)
    if show_progress:
        it = tqdm(it, total=len(df), desc="MFCC features")
    for row in it:
        row = row._asdict()
        feat = None
        cache_p = _cache_path(cache_dir, row["corpus"], row["path"], h)
        if cfg.data.cache_features:
            feat = _load_cached(cache_p)   # önce önbelleği dene
        if feat is None:
            try:
                wav = audio_io.load_audio(row["path"], cfg.audio.sample_rate)
                feat = mfcc_statistics(wav, cfg.feature, cfg.audio.sample_rate)
            except Exception as e:
                log.warning("MFCC extraction failed for %s: %s", row["path"], e)
                skipped += 1
                continue
            if cfg.data.cache_features:
                _save_cached(cache_p, feat)
        X.append(feat)
        y.append(int(row["label_idx"]))
    if skipped:
        log.warning("Skipped %d unreadable file(s) during MFCC extraction.", skipped)
    # float32/int64: sklearn ve torch'un beklediği standart tipler.
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
