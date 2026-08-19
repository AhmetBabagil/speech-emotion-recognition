# Aralık öznitelikleri (Yöntem 2) için birim testleri.
#
# Odak: ödevin zorunlu tuttuğu iki hiperparametre — aralık SAYISI (n_intervals) ve aralık GENİŞLİĞİ (interval_ms) — ve bunların "RNN her zaman sabit uzunlukta seri görür" garantisini nasıl sağladığı. Ayrıca öznitelik boyutunun ablasyon bayraklarına göre doğru hesaplandığını doğrular.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import numpy as np  # sentetik ses + diziler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.features import (  # noqa: E402  # test edilen bileşenler
    IntervalConfig, extract_interval_series, interval_starts,
)


def test_feature_dim_reflects_ablation_flags() -> None:  # Öznitelik boyutu, açık bayrakların toplamı olmalı (kanıta dayalı set).
    hepsi_kapali = IntervalConfig(use_jitter=False, use_contrast=False)  # taban set
    assert hepsi_kapali.feature_dim == 44  # 3x13 MFCC + 5 skaler = 44
    varsayilan = IntervalConfig()  # üretim ayarı: jitter + kontrast açık
    assert varsayilan.feature_dim == 44 + 2 + 7  # +2 jitter/shimmer, +7 kontrast = 53
    # Her bayrak, belgelenen miktarda boyut ekler:
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_pitch=True).feature_dim == 44 + 3  # +3 pitch
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_delta2=True).feature_dim == 44 + 13  # +13 delta-delta
    assert IntervalConfig(use_jitter=False, use_contrast=False, use_bandwidth=True).feature_dim == 44 + 2  # +2 bant/düzlük


def test_shape_uses_n_intervals_hyperparameter() -> None:  # shape = (n_intervals, feature_dim): aralık sayısı satır sayısını belirler.
    assert IntervalConfig(n_intervals=32).shape == (32, 53)  # 32 aralık -> 32 satır
    assert IntervalConfig(n_intervals=16).shape == (16, 53)  # 16 aralık -> 16 satır


def test_interval_samples_from_ms() -> None:  # Aralık genişliği ms'den örnek sayısına doğru çevrilmeli (16 kHz).
    assert IntervalConfig(interval_ms=300).interval_samples == 4800  # 300 ms * 16000 / 1000
    assert IntervalConfig(interval_ms=200).interval_samples == 3200  # 200 ms * 16000 / 1000


def test_interval_starts_count_sorted_and_in_bounds() -> None:  # Başlangıçlar: tam n_intervals adet, artan sırada ve sınırlar içinde.
    cfg = IntervalConfig(n_intervals=10, interval_ms=300)  # 10 aralık
    total = 16000 * 3  # 3 saniyelik (uzun) kayıt
    starts = interval_starts(total, cfg)  # başlangıç indeksleri
    assert len(starts) == 10  # tam olarak n_intervals kadar pencere
    assert list(starts) == sorted(starts)  # artan sırada (soldan sağa)
    span = total - cfg.interval_samples  # son pencerenin taşmayacağı en geç başlangıç
    assert starts.min() >= 0 and starts.max() <= span  # hepsi [0, span] içinde -> pencere kayıt dışına taşmaz


def test_short_recording_still_yields_n_intervals() -> None:  # Kısa kayıtta pencereler örtüşür ama satır sayısı yine n_intervals kalır.
    cfg = IntervalConfig(n_intervals=8, interval_ms=300)  # pencere 4800 örnek
    total = 1600  # 0.1 sn: pencereden (4800) kısa -> örtüşme
    starts = interval_starts(total, cfg)  # başlangıçlar
    assert len(starts) == 8  # yine 8 pencere (sabit uzunluk korunur)
    assert np.all(starts == 0)  # span=0 -> hepsi 0'dan başlar (tam örtüşme)


def test_extract_series_is_fixed_shape_regardless_of_length() -> None:  # ASIL GARANTİ: kayıt kısa da olsa uzun da olsa çıktı hep (n_intervals, feature_dim).
    rng = np.random.default_rng(0)  # sabit tohum
    cfg = IntervalConfig(n_intervals=8)  # küçük ama gerçek üretim özniteliği (jitter+kontrast)
    kisa = rng.standard_normal(16000 // 5).astype(np.float32)  # 0.2 sn gürültü (kısa kayıt)
    uzun = rng.standard_normal(16000 * 3).astype(np.float32)  # 3 sn gürültü (uzun kayıt)
    s_kisa = extract_interval_series('sentetik_kisa.wav', cfg, audio=kisa)  # dosyasız çıkar (audio= verildi)
    s_uzun = extract_interval_series('sentetik_uzun.wav', cfg, audio=uzun)  # dosyasız çıkar
    assert s_kisa.shape == (8, 53)  # kısa kayıt -> sabit boyut
    assert s_uzun.shape == (8, 53)  # uzun kayıt -> AYNI sabit boyut (RNN hep sabit seri görür)
    assert np.isfinite(s_kisa).all() and np.isfinite(s_uzun).all()  # bozuk (NaN/inf) değer yok
