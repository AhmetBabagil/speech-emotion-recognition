"""Klasik taban modeli için MFCC öznitelikleri.

MFCC (Mel-Frequency Cepstral Coefficients) nedir? Mel spektrumunun logaritması
üzerine bir DCT (kosinüs dönüşümü) uygulanarak elde edilen, spektral zarfı az
sayıda katsayıyla özetleyen klasik konuşma öznitelikleridir. Derin öğrenme
öncesi dönemin standart temsilidir; bu projede "derin model gerçekten
gerekli mi?" sorusunun referans noktasıdır.

İki görünüm sunulur:
  * ``mfcc_sequence``   -> [n_mfcc, T] zaman serisi (taban modeli kullanmaz;
    inceleme ya da olası bir RNN denemesi için elde tutulur).
  * ``mfcc_statistics`` -> zaman üzerinden özet istatistiklerden oluşan SABİT
    uzunluklu vektör (MFCC + delta + delta-delta'nın ortalama/std'si).
    Kare-havuzlamalı (frame-pooled) SVM/MLP taban modelleri için standart ve
    güçlü öznitelik kümesi budur: sklearn sabit boyutlu girdi ister, kayıtlar
    ise değişken uzunluktadır — zaman ekseni istatistiklerle "eritilir".
"""

from __future__ import annotations

import numpy as np


def mfcc_sequence(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    """Ham dalga formundan [n_mfcc, T] MFCC dizisi çıkarır.

    STFT parametreleri (n_fft/hop/win) FeatureConfig'ten gelir — mel
    spektrogramla aynı pencereleme kullanılır ki iki temsil karşılaştırılabilir
    olsun. float32: bellek yarı yarıya, sklearn/torch için fazlasıyla yeterli.
    """
    import librosa

    return librosa.feature.mfcc(
        y=wav,
        sr=sample_rate,
        n_mfcc=feature_cfg.n_mfcc,
        n_fft=feature_cfg.n_fft,
        hop_length=feature_cfg.hop_length,
        win_length=feature_cfg.win_length,
    ).astype(np.float32)


def mfcc_statistics(wav: np.ndarray, feature_cfg, sample_rate: int) -> np.ndarray:
    """1 boyutlu öznitelik vektörü döndürür: [MFCC, ΔMFCC, ΔΔMFCC]'nin zaman
    üzerinden ortalaması + standart sapması.

    Delta'lar neden var? MFCC bir anın spektral şeklini verir; delta (birinci
    türev) bunun nasıl DEĞİŞTİĞİNİ, delta-delta (ikinci türev) değişimin
    ivmesini yakalar. Duygu tam da bu dinamikte saklıdır: kızgın konuşmada
    enerji/perde hızlı dalgalanır, üzgün konuşmada durağandır.

    Uzunluk = n_mfcc * 3 (mfcc, delta, delta2) * 2 (ortalama, std).
    Varsayılan n_mfcc=40 ile 40*3*2 = 240 boyut.
    """
    import librosa

    mfcc = mfcc_sequence(wav, feature_cfg, sample_rate)  # [n_mfcc, T]
    if mfcc.shape[1] < 2:
        # Delta hesabı en az 2 kare ister; aşırı kısa klipte zaman ekseni
        # "edge" moduyla (son değeri kopyalayarak) doldurulur — çökme yerine
        # dejenere ama geçerli bir öznitelik üretilir.
        mfcc = np.pad(mfcc, ((0, 0), (0, 2 - mfcc.shape[1])), mode="edge")
    delta = librosa.feature.delta(mfcc)             # 1. türev: değişim hızı
    delta2 = librosa.feature.delta(mfcc, order=2)   # 2. türev: değişim ivmesi
    stacked = np.concatenate([mfcc, delta, delta2], axis=0)  # [3*n_mfcc, T]
    # Zaman ekseni (axis=1) istatistiklerle özetlenir: değişken T sabit boyuta iner.
    feats = np.concatenate([stacked.mean(axis=1), stacked.std(axis=1)])
    return feats.astype(np.float32)
