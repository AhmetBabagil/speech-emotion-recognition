"""Ses G/Ç katmanı: herhangi bir WAV'ı hedef örnekleme frekansında mono float32
diziye yükler ve sabit örnek sayısına getirir (doldur ya da kırp).

Bu iki işlem ayrı bir modüldedir çünkü hem CNN yolu (spektrogramdan önce) hem
wav2vec2 yolu (doğrudan dalga formu) aynı yükleme/uzunluk mantığını paylaşır.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger

log = get_logger(__name__)


def load_audio(path: str, target_sr: int) -> np.ndarray:
    """``path``'i mono float32 olarak yükler ve ``target_sr``'a yeniden örnekler.

    librosa kullanılır (arkada soundfile/audioread'i sarar); böylece CREMA-D'nin
    WAV'ları ve bizim MELD ffmpeg çıkarımımızın ürettiği WAV'lar Windows/Linux
    fark etmeksizin aynı şekilde açılır. Dönen değer kabaca [-1, 1] aralığında
    1 boyutlu float32 dizidir.

    Hata felsefesi: bozuk/okunamayan TEK dosya koca bir eğitim koşusunu
    çökertmemeli. Bu yüzden hata durumunda uyarı loglanır ve kısa bir SESSİZ
    klip döndürülür (yukarıda sabit uzunluğa zaten doldurulacaktır). Binlerce
    dosyalık gerçek veride bu dayanıklılık şarttır.
    """
    import librosa

    try:
        # sr=target_sr: librosa gerekiyorsa yeniden örnekler; mono=True kanalları
        # ortalayarak teke indirir. Böylece tüm borunun girdisi tek tiptir.
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        log.warning("Failed to load audio %s (%s) -- using silence.", path, e)
        return np.zeros(int(target_sr), dtype=np.float32)  # 1 saniyelik sessizlik
    wav = np.asarray(wav, dtype=np.float32)
    # NaN/sonsuz değerlere karşı koruma: bozuk örnekler 0'a çevrilir; NaN'lar
    # spektrograma, oradan kayba yayılıp tüm eğitimi NaN yapabilirdi.
    if not np.isfinite(wav).all():
        wav = np.nan_to_num(wav, copy=False)
    return wav


def fix_length(
    wav: np.ndarray,
    num_samples: int,
    *,
    random_crop: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """``wav``'ı tam olarak ``num_samples`` örneğe getirir (sıfırla doldur ya da kırp).

    Neden sabit uzunluk? Tensörleri batch'lemek için tüm örneklerin aynı boyutta
    olması gerekir. Eğitimde uzun klipler RASTGELE konumdan kırpılır — her
    epoch'ta klibin farklı bir bölümü görülür, bu bedava bir veri çoğaltmadır.
    Değerlendirmede ise MERKEZDEN kırpılır ki sonuç deterministik olsun.
    """
    n = wav.shape[0]
    if n == num_samples:
        return wav  # zaten istenen uzunlukta: dokunma
    if n < num_samples:
        pad = num_samples - n
        # Sona sıfır ekle (eğitimde başa rastgele pad de denenebilirdi ama
        # basitlik tercih edildi; konuşma zaten klibin başındadır).
        return np.pad(wav, (0, pad), mode="constant")
    # n > num_samples -> kırpma gerekli
    if random_crop:
        # rng verilmemişse tohumsuz bir RNG kur (tekrarlanabilirlik istenirse
        # çağıran taraf tohumlanmış rng geçirir — SERDataset bunu yapar).
        rng = rng or np.random.default_rng()
        # +1: başlangıç son geçerli konumu da (n - num_samples) alabilsin.
        start = int(rng.integers(0, n - num_samples + 1))
    else:
        start = (n - num_samples) // 2  # merkez kırpma: deterministik
    return wav[start : start + num_samples]
