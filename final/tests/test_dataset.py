# Standardizer (z-skor) için birim testleri.
#
# En kritik iddia: ortalama/std YALNIZCA eğitim katmanından öğrenilir; geçerleme ve test aynı (eğitim) parametreleriyle dönüştürülür. Sızıntılı bir uygulama val/test'e kendi istatistiğini uygular ve bu testler onu yakalar.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import numpy as np  # sentetik tensörler
import pytest  # test çatısı + hata yakalama

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.dataset import Standardizer  # noqa: E402  # z-skor standartlaştırıcı


def test_fit_standardizes_training_to_zero_mean_unit_std() -> None:  # Eğitimden fit edilip eğitim dönüştürülünce: ~0 ortalama, ~1 std.
    rng = np.random.default_rng(0)  # sabit tohum
    train = rng.normal(0.0, 1.0, (50, 6, 4)).astype(np.float32)  # [N, T, D] eğitim serisi
    std = Standardizer.fit(train, feature_axis=2)  # öznitelik (D) ekseninde öğren
    z = std.transform(train)  # eğitimi dönüştür
    per_feature_mean = z.mean(axis=(0, 1))  # her öznitelik için ortalama
    per_feature_std = z.std(axis=(0, 1))  # her öznitelik için std
    assert np.abs(per_feature_mean).max() < 0.2  # ortalamalar ~0
    assert np.abs(per_feature_std - 1.0).max() < 0.2  # std'ler ~1


def test_val_is_transformed_with_train_stats_not_its_own() -> None:  # SIZINTI TESTİ: kaydırılmış val, eğitim istatistiğiyle dönüşür (kendi 0 ortalamasına GELMEZ).
    rng = np.random.default_rng(1)  # sabit tohum
    train = rng.normal(0.0, 1.0, (50, 6, 4)).astype(np.float32)  # eğitim: ortalama ~0
    val = (rng.normal(0.0, 1.0, (30, 6, 4)) + 5.0).astype(np.float32)  # geçerleme: ortalama ~5 (bilerek kaydık)
    std = Standardizer.fit(train, feature_axis=2)  # YALNIZ eğitimden öğren
    z_val = std.transform(val)  # geçerlemeyi eğitim parametreleriyle dönüştür
    val_mean = z_val.mean(axis=(0, 1))  # dönüşmüş geçerleme ortalaması
    # Eğitim ortalaması ~0, std ~1 olduğundan (5 - 0)/1 ≈ 5 beklenir.
    assert np.all(val_mean > 3.0)  # ~0'a GELMEDİ -> kendi istatistiğiyle normalize edilmedi
    assert np.all(np.abs(val_mean - 5.0) < 1.0)  # eğitim parametreleriyle ~5'e oturdu (sızıntı yok kanıtı)


def test_fit_uses_only_the_data_it_is_given() -> None:  # fit(eğitim) ile fit(eğitim+val) farklı parametre verir -> fit yalnız argümanını görür.
    rng = np.random.default_rng(2)  # sabit tohum
    train = rng.normal(0.0, 1.0, (50, 6, 4)).astype(np.float32)  # eğitim ~0
    val = (rng.normal(0.0, 1.0, (30, 6, 4)) + 5.0).astype(np.float32)  # val ~5
    s_train = Standardizer.fit(train, feature_axis=2)  # yalnız eğitim
    s_both = Standardizer.fit(np.concatenate([train, val], axis=0), feature_axis=2)  # eğitim+val birlikte
    # Eğer sadece eğitimle aynı çıksaydı fit veriyi umursamıyor olurdu; belirgin farklı olmalı.
    assert np.abs(s_both.mean.mean() - s_train.mean.mean()) > 1.0  # birleşik ortalama yukarı kaydı


def test_fit_rejects_nan_and_wrong_ndim() -> None:  # Bozuk/eksik girdi sessizce geçmemeli.
    good = np.zeros((10, 6, 4), dtype=np.float32)  # geçerli 3B tensör
    bad = good.copy()  # kopya
    bad[0, 0, 0] = np.nan  # bir NaN yerleştir
    with pytest.raises(ValueError):  # NaN reddedilmeli
        Standardizer.fit(bad, feature_axis=2)
    with pytest.raises(ValueError):  # 2B girdi reddedilmeli (3B beklenir)
        Standardizer.fit(np.zeros((10, 4), dtype=np.float32), feature_axis=1)


def test_standardizer_is_frozen() -> None:  # Öğrenilen parametreler sonradan değiştirilememeli (frozen dataclass).
    std = Standardizer.fit(np.ones((10, 6, 4), dtype=np.float32) * 2.0, feature_axis=2)  # bir standartlaştırıcı öğren
    with pytest.raises(Exception):  # kilitli -> atama hata vermeli
        std.mean = np.zeros(4, dtype=np.float32)  # parametreyi ezmeye çalış
