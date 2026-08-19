# training.py yardımcıları için birim testleri: ağırlıklı kayıp formülü + erken durdurma.
#
# Ödevin iki zorunlu şartını (sınıf-ağırlıklı kayıp, erken durdurma) doğrudan test eder. Küçük sentetik veri kullanır; corpus indirmeye gerek yoktur ve testler saniyeler içinde biter.

from __future__ import annotations  # tip ipuçlarını esnek yazmak için

from pathlib import Path  # dosya yolları
import sys  # import yolu

import numpy as np  # sentetik veri
import pytest  # test çatısı
import torch  # eğitim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # proje kökünü import yoluna ekle

from final.models import OptimSettings, RNNConfig, SeqRNN  # noqa: E402  # model + ayarlar
from final.training import inverse_frequency_weights, train_with_early_stopping  # noqa: E402  # ağırlık + eğitim


def test_inverse_frequency_weights_formula() -> None:  # Ağırlık formülü w = n/(K*n_k) doğru mu: az örnekli sınıf büyük ağırlık alır.
    labels = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])  # sınıf 0:4, 1:2, 2:6 (K=3, n=12)
    w = inverse_frequency_weights(labels, 3)  # ağırlıkları hesapla
    assert w[1] > w[0] > w[2]  # az olan (1) en büyük, çok olan (2) en küçük ağırlık
    assert abs(w[1].item() - 12 / (3 * 2)) < 1e-5  # 12/(3*2) = 2.0
    assert abs(w[0].item() - 12 / (3 * 4)) < 1e-5  # 12/(3*4) = 1.0
    assert abs(w[2].item() - 12 / (3 * 6)) < 1e-5  # 12/(3*6) ≈ 0.667


def test_inverse_frequency_rejects_missing_class() -> None:  # Bir sınıf hiç yoksa hata vermeli (sessiz bozulma olmasın).
    with pytest.raises(ValueError):  # ValueError beklenir
        inverse_frequency_weights(np.array([0, 0, 1, 1]), 3)  # sınıf 2 hiç yok -> hata


def test_early_stopping_returns_best_model() -> None:  # Erken durdurma: küçük veride geçerli sonuç + en iyi epoch döndürmeli.
    rng = np.random.default_rng(0)  # sabit tohumlu üreteç
    n, T, D, K = 60, 5, 8, 3  # örnek/zaman/öznitelik/sınıf
    # Ayrılabilir sahte veri: her sınıfın ortalaması label kadar kayık (öğrenilebilir).
    y = rng.integers(0, K, size=n)  # eğitim etiketleri
    X = (rng.normal(0, 1, (n, T, D)) + y[:, None, None]).astype(np.float32)  # eğitim öznitelikleri
    vy = rng.integers(0, K, size=30)  # geçerleme etiketleri
    vX = (rng.normal(0, 1, (30, T, D)) + vy[:, None, None]).astype(np.float32)  # geçerleme öznitelikleri
    cfg = RNNConfig(hidden_size=16, num_layers=1,  # küçük model (hızlı)
                    optim=OptimSettings(batch_size=16, patience=3))
    model = SeqRNN(D, K, cfg)  # taze model
    out = train_with_early_stopping(model, X, y, vX, vy, cfg.optim,  # CPU'da kısa eğitim
                                    num_classes=K, device=torch.device('cpu'),
                                    max_epochs=5, seed=0, amp=False)
    assert 1 <= out.best_epoch <= out.epochs_trained  # geçerli bir en iyi epoch seçilmeli
    assert 'macro_f1' in out.validation_metrics  # geçerleme metrikleri döndürülmeli
    assert len(out.history) == out.epochs_trained  # her epoch öğrenme eğrisine kaydedilmeli
