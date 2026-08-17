"""CNN modeli için log-mel spektrogram öznitelikleri.

Kısa kuram: Ham dalga formu, duygu gibi "tınısal" bilgiyi doğrudan göstermez.
STFT ile ses küçük pencerelere bölünüp frekans uzayına taşınır; mel filtre
bankası bu frekans eksenini insan kulağının duyarlılığına göre (alçak
frekanslarda ince, yüksekte kaba) yeniden ölçekler; logaritma (dB) ise insan
gürlük algısına uyan sıkıştırmayı yapar. Sonuç, CNN'in "görüntü gibi"
işleyebileceği 2 boyutlu bir [n_mels, T] matristir.
"""

from __future__ import annotations

import numpy as np


def fixed_num_frames(num_samples: int, hop_length: int) -> int:
    """``num_samples`` örnek için librosa'nın üreteceği mel karesi sayısı
    (``center=True`` varsayılanıyla).

    Formülün nedeni: center=True iken sinyal iki uçtan yansıtılarak doldurulur
    ve kare sayısı floor(N / hop) + 1 olur. Bu sayı önceden bilinirse her
    spektrogram aynı genişliğe kırpılıp/doldurulup sorunsuz batch'lenebilir.
    """
    return num_samples // hop_length + 1


def log_mel_spectrogram(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    """[n_mels, T] boyutlu float32 log-mel spektrogram döndürür (desibel ölçeği).

    ``feature_cfg`` bir :class:`ser.config.FeatureConfig` nesnesidir; STFT/mel
    parametreleri oradan gelir (tek doğruluk kaynağı ilkesi).
    """
    import librosa

    # fmax güvenlik kontrolü: Nyquist frekansının (sample_rate/2) üzerindeki
    # değerler fiziksel olarak anlamsızdır; None ise de Nyquist kullanılır.
    fmax = feature_cfg.fmax
    if fmax is None or fmax > sample_rate / 2:
        fmax = sample_rate / 2
    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sample_rate,
        n_fft=feature_cfg.n_fft,          # STFT pencere başına FFT boyutu
        hop_length=feature_cfg.hop_length, # pencerenin her adımda kayma miktarı
        win_length=feature_cfg.win_length, # pencereleme fonksiyonunun uzunluğu
        n_mels=feature_cfg.n_mels,         # mel bandı sayısı (çıktının yüksekliği)
        fmin=feature_cfg.fmin,
        fmax=fmax,
        power=2.0,                         # güç spektrumu (genliğin karesi)
    )
    # ref=np.max: dB'ler klibin en güçlü noktasına göre ifade edilir (tepe = 0 dB,
    # gerisi negatif). Bu, kayıt seviyesi farklarına karşı kaba bir normalizasyondur.
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def fix_frames(spec: np.ndarray, num_frames: int, *, random_crop: bool = False,
               rng: np.random.Generator | None = None) -> np.ndarray:
    """[n_mels, T] spektrogramı zaman ekseninde tam ``num_frames``'e getirir.

    fix_length'in spektrogram karşılığıdır: eğitimde rastgele, değerlendirmede
    merkez kırpma; kısaysa doldurma.
    """
    t = spec.shape[1]
    if t == num_frames:
        return spec
    if t < num_frames:
        pad = num_frames - t
        # Doldurma değeri olarak 0 değil spektrogramın MİNİMUMU kullanılır:
        # dB ölçeğinde 0 "tepe gürlük" demektir; minimum ise "sessizlik"e karşılık
        # gelir. Sıfırla doldursaydık klip sonuna sahte bir gürültü duvarı eklerdik.
        return np.pad(spec, ((0, 0), (0, pad)), mode="constant",
                      constant_values=spec.min())
    if random_crop:
        rng = rng or np.random.default_rng()
        start = int(rng.integers(0, t - num_frames + 1))
    else:
        start = (t - num_frames) // 2  # merkez kırpma: deterministik değerlendirme
    return spec[:, start : start + num_frames]


def standardize(spec: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Örnek-içi sıfır-ortalama/birim-varyans normalizasyonu.

    Her spektrogram KENDİ ortalama ve std'siyle normalize edilir (global veri
    istatistikleriyle değil). Bu, kayıt seviyesi/mikrofon farklarını örnek
    bazında siler ve global istatistik tutma derdinden kurtarır. ``eps``,
    tamamen sabit (std=0) girdilerde 0'a bölünmeyi önler.
    """
    mu = spec.mean()
    sd = spec.std()
    return (spec - mu) / (sd + eps)


def spec_augment(spec: np.ndarray, freq_mask: int, time_mask: int,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Hafif SpecAugment: bir frekans bandı + bir zaman dilimi 0'a maskelenir.

    Fikir (Park ve ark., 2019): spektrogramın rastgele bir şeridini silmek,
    modeli tek bir frekans bandına ya da tek bir ana ezberlenmiş anına bağımlı
    olmaktan caydırır — dropout'un girdi-uzayındaki karşılığı gibi düşünülebilir.
    Maske değeri 0'dır; standardize sonrası 0 zaten ortalamaya denk gelir.
    Orijinal makaledeki çoklu maske yerine tek maske kullanılır (küçük veri
    kümelerinde aşırı bozmamak için "hafif" versiyon).
    """
    rng = rng or np.random.default_rng()
    spec = spec.copy()  # çağıranın (muhtemelen önbellekteki) dizisini bozma
    n_mels, t = spec.shape
    if freq_mask > 0 and n_mels > freq_mask:
        # Maske genişliği 0..freq_mask arasında rastgele; 0 çıkarsa maske yok
        # (augmentasyonun şiddeti de örnekten örneğe değişir).
        f = int(rng.integers(0, freq_mask + 1))
        if f > 0:
            f0 = int(rng.integers(0, n_mels - f + 1))  # bandın başlangıç satırı
            spec[f0 : f0 + f, :] = 0.0
    if time_mask > 0 and t > time_mask:
        m = int(rng.integers(0, time_mask + 1))
        if m > 0:
            t0 = int(rng.integers(0, t - m + 1))       # dilimin başlangıç sütunu
            spec[:, t0 : t0 + m] = 0.0
    return spec
