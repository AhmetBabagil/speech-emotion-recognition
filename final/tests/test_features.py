# Final öznitelik çıkarıcıları için boyut/determinizm testleri.
#
# Testler corpus dosyalarına bağımlı değildir: geçici dizinde sentetik bir ton (sinüs) üretilir ve her şey onun üzerinde doğrulanır. Böylece testler her makinede, veri indirilmeden çalışır.
#
# Kapsam iki yöntemi de içerir: Yöntem 1 mel görüntüsü (CNN) + Yöntem 2 aralık serisi (RNN), artı önbellek ve z-skor eksen mantığı. Sonradan eklenenler: ödevin zorunlu tuttuğu aralık hiperparametreleri (n_intervals, interval_ms) ve öznitelik boyutunun ablasyon bayraklarına göre hesabı.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import numpy as np  # diziler + rastgele sinyal
import pytest  # test çatısı
import soundfile as sf  # geçici WAV yazma

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.dataset import Standardizer, load_or_extract  # noqa: E402  # normalizasyon + önbellek
from final.features import (  # noqa: E402  # öznitelik çıkarıcılar
    IntervalConfig,
    MelImageConfig,
    extract_interval_series,
    extract_mel_image,
    interval_starts,
)


@pytest.fixture()
def wav_path(tmp_path: Path) -> Path:  # 1,5 saniyelik sentetik 220 Hz ton — testlerin sabit girdisi.

    sr = 16_000  # örnekleme hızı
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)  # zaman ekseni (1,5 sn)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # 220 Hz sinüs ton
    path = tmp_path / 'tone.wav'  # geçici dosya yolu
    sf.write(path, audio, sr)  # WAV olarak yaz
    return path  # ton dosyasının yolu


def test_mel_image_shape_and_determinism(wav_path: Path) -> None:  # Mel görüntüsü: doğru boyut, sonlu değerler ve iki çağrıda aynı sonuç.

    config = MelImageConfig(n_mels=32, n_frames=48)  # küçük mel ayarı
    first = extract_mel_image(wav_path, config)  # birinci çıkarım
    second = extract_mel_image(wav_path, config)  # ikinci çıkarım
    assert first.shape == (32, 48)  # doğru boyut
    assert first.dtype == np.float32  # doğru tip
    assert np.isfinite(first).all()  # bozuk değer yok
    # Determinizm: aynı dosya + aynı ayar = bit-bit aynı çıktı.
    np.testing.assert_array_equal(first, second)  # iki çağrı özdeş olmalı


def test_interval_starts_cover_short_and_long_audio() -> None:  # Aralık yerleşimi hem kısa hem uzun kayıtta doğru çalışmalı.

    config = IntervalConfig(n_intervals=8, interval_ms=300)  # 8 aralık ayarı
    window = config.interval_samples  # bir aralığın örnek uzunluğu
    short = interval_starts(window // 2, config)   # tek pencereden bile kısa kayıt
    long = interval_starts(window * 20, config)    # pencerenin 20 katı kayıt
    # Her iki durumda da tam 8 başlangıç üretilmeli (sabit uzunlukta seri).
    assert len(short) == len(long) == 8  # ikisi de 8 başlangıç
    # Kısa kayıtta tüm pencereler 0'dan başlar (hepsi üst üste biner).
    assert short.min() >= 0 and short.max() == 0  # hepsi 0'da
    # Uzun kayıtta ilk pencere başta, son pencere tam sonda biter.
    assert long.min() == 0 and long.max() == window * 20 - window  # baştan sona yayılmış


def test_interval_series_shape(wav_path: Path) -> None:  # Aralık serisi: doğru boyut + durağan sinyalde satırlar birbirine yakın.

    config = IntervalConfig(n_intervals=6, interval_ms=250)  # 6 aralık ayarı
    series = extract_interval_series(wav_path, config)  # seriyi çıkar
    assert series.shape == config.shape == (6, config.feature_dim)  # doğru boyut
    assert np.isfinite(series).all()  # bozuk değer yok
    # Sabit bir tonun ortadaki aralıkları neredeyse özdeş olmalı
    # (kenar aralıklar dolgu nedeniyle farklılaşabilir).
    assert np.std(series[1:-1], axis=0).max() < 1.0  # orta aralıklar birbirine yakın


def test_cache_roundtrip(wav_path: Path, tmp_path: Path) -> None:  # Önbellek: ilk çağrı hesaplar, ikinci çağrı diskten okur (tekrar hesaplamaz).

    config = IntervalConfig(n_intervals=4, interval_ms=200)  # 4 aralık ayarı
    cache_dir = tmp_path / 'cache'  # geçici önbellek klasörü
    calls = {'n': 0}  # gerçek hesaplama sayacı

    def counting_extract(path):  # kaç kez hesaplandığını sayan sarmalayıcı
        calls['n'] += 1   # kaç kez gerçekten hesaplandığını say
        return extract_interval_series(path, config)  # gerçek çıkarım

    first = load_or_extract(wav_path, cache_dir, config.fingerprint,  # ilk çağrı (hesaplar + yazar)
                            counting_extract, config.shape)
    second = load_or_extract(wav_path, cache_dir, config.fingerprint,  # ikinci çağrı (okur)
                             counting_extract, config.shape)
    assert calls['n'] == 1   # ikinci çağrı önbellekten gelmiş olmalı
    np.testing.assert_array_equal(first, second)  # iki sonuç özdeş olmalı


def test_standardizer_axes() -> None:  # Z-skor doğru eksende uygulanmalı: dönüşüm sonrası ortalama ~0, std ~1.

    rng = np.random.default_rng(0)  # rastgele üreteç
    # RNN durumu: [N, T, D] -> öznitelik boyutu (axis=2) başına istatistik.
    series = rng.normal(3.0, 2.0, size=(50, 12, 7)).astype(np.float32)  # ort 3, std 2 sentetik seri
    scaler = Standardizer.fit(series, feature_axis=2)  # öznitelik ekseninde öğren
    transformed = scaler.transform(series)  # normalize et
    means = transformed.mean(axis=(0, 1))  # öznitelik başına ortalama
    stds = transformed.std(axis=(0, 1))  # öznitelik başına std
    assert np.abs(means).max() < 1e-4  # ortalama ~0 olmalı
    assert np.abs(stds - 1.0).max() < 1e-3  # std ~1 olmalı

    # CNN durumu: [N, mels, T] -> mel bandı (axis=1) başına istatistik.
    images = rng.normal(0.0, 5.0, size=(50, 16, 20)).astype(np.float32)  # sentetik mel yığını
    scaler = Standardizer.fit(images, feature_axis=1)  # mel ekseninde öğren
    transformed = scaler.transform(images)  # normalize et
    assert np.abs(transformed.mean(axis=(0, 2))).max() < 1e-4  # mel başına ortalama ~0


# --- Sonradan eklenen testler: aralık hiperparametreleri + öznitelik boyutu ---

def test_feature_dim_reflects_ablation_flags() -> None:  # Öznitelik boyutu, açık bayrakların toplamı olmalı (kanıta dayalı set).
    taban = IntervalConfig(use_jitter=False, use_contrast=False)  # yalnız taban öznitelikler
    assert taban.feature_dim == 44  # 3x13 MFCC + 5 skaler = 44
    assert IntervalConfig().feature_dim == 44 + 2 + 7  # varsayılan: +2 jitter, +7 kontrast = 53
    # Her bayrak, belgelenen miktarda boyut ekler:
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_pitch=True).feature_dim == 44 + 3  # +3 pitch
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_delta2=True).feature_dim == 44 + 13  # +13 delta-delta
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_bandwidth=True).feature_dim == 44 + 2  # +2 bant/düzlük


def test_shape_and_interval_samples_from_hyperparameters() -> None:  # İki hiperparametre çıktı boyutunu ve pencere uzunluğunu belirler.
    assert IntervalConfig(n_intervals=32).shape == (32, 53)  # aralık sayısı -> satır sayısı
    assert IntervalConfig(n_intervals=16).shape == (16, 53)  # 16 aralık -> 16 satır
    assert IntervalConfig(interval_ms=300).interval_samples == 4800  # 300 ms * 16000 / 1000
    assert IntervalConfig(interval_ms=200).interval_samples == 3200  # 200 ms * 16000 / 1000


def test_extract_series_fixed_shape_via_audio_arg() -> None:  # audio= yolu: kayıt kısa da uzun da olsa çıktı hep (n_intervals, feature_dim).
    rng = np.random.default_rng(0)  # sabit tohum
    cfg = IntervalConfig(n_intervals=8)  # gerçek üretim özniteliği (jitter+kontrast)
    kisa = rng.standard_normal(16000 // 5).astype(np.float32)  # 0.2 sn (kısa)
    uzun = rng.standard_normal(16000 * 3).astype(np.float32)  # 3 sn (uzun)
    s_kisa = extract_interval_series('sentetik_kisa.wav', cfg, audio=kisa)  # dosyasız çıkar
    s_uzun = extract_interval_series('sentetik_uzun.wav', cfg, audio=uzun)  # dosyasız çıkar
    assert s_kisa.shape == (8, 53)  # kısa -> sabit boyut
    assert s_uzun.shape == (8, 53)  # uzun -> AYNI sabit boyut (RNN hep sabit seri görür)
    assert np.isfinite(s_kisa).all() and np.isfinite(s_uzun).all()  # bozuk değer yok
