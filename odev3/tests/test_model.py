'''Yapılandırılabilir, sıfırdan yazılmış PyTorch MLP'nin testleri.

Bu dosya model.py'nin sözleşmesini doğrular: doğru çıktı şekli, parametre
sayısı, konfigürasyonun kayıpsız gidiş-dönüşü (round-trip), tüm aktivasyon
seçeneklerinin çalışması, mimari anahtarlarının gerçekten katman
ekleyip/çıkarması ve geçersiz girdilerin reddedilmesi.
'''

import pytest
import torch
from torch import nn

from odev3.model import MLP, MLPConfig, count_parameters


def test_mlp_forward_shape_and_parameter_count() -> None:
    '''İleri geçiş (batch, sınıf) şeklinde çıktı vermeli; parametre sayısı elle doğrulanmalı.

    Beklenen parametre sayısı katman katman elle hesaplanır: her Linear için
    (giriş x çıkış ağırlık) + (çıkış bias). Bu, count_parameters'ın ve katman
    kurulumunun aynı anda doğrulanmasını sağlar.
    '''

    config = MLPConfig(
        hidden_dims=(8, 4),
        batch_norm=False,
        dropout=0.0,
    )
    model = MLP(input_dim=16, num_classes=6, config=config)

    logits = model(torch.zeros(3, 16))

    assert logits.shape == (3, 6)
    # 16->8, 8->4, 4->6 katmanlarının ağırlık+bias toplamı.
    expected_parameters = (16 * 8 + 8) + (8 * 4 + 4) + (4 * 6 + 6)
    assert count_parameters(model) == expected_parameters


def test_model_config_round_trip_preserves_hyperparameters() -> None:
    '''to_dict -> from_dict dönüşümü konfigürasyonu birebir korumalı.

    Bu özellik kritik: arama durumu ve checkpoint'ler konfigürasyonu JSON
    olarak saklar. Dönüşümde tek bir alan bile bozulsaydı, diskten geri
    yüklenen deneyler farklı bir modelle devam ederdi.
    '''

    original = MLPConfig(
        batch_size=32,
        learning_rate=1e-3,
        patience=12,
        hidden_dims=(256, 128, 64),
        activation='gelu',
        batch_norm=False,
        dropout=0.5,
        weight_decay=0.0,
    )

    restored = MLPConfig.from_dict(original.to_dict())

    assert restored == original


@pytest.mark.parametrize('activation', ['relu', 'gelu', 'tanh', 'leaky_relu'])
def test_supported_activations_produce_finite_logits(activation: str) -> None:
    '''Desteklenen dört aktivasyonun tamamı sonlu (NaN/inf olmayan) logit üretmeli.'''

    config = MLPConfig(
        hidden_dims=(12,),
        activation=activation,
        batch_norm=False,
        dropout=0.0,
    )
    model = MLP(input_dim=10, num_classes=6, config=config)

    logits = model(torch.randn(4, 10))

    assert torch.isfinite(logits).all()


def test_batch_normalization_and_dropout_switches_change_architecture() -> None:
    '''batch_norm/dropout anahtarları katman dizilimini GERÇEKTEN değiştirmeli.

    Anahtar açıkken ilgili katmanlar modülde bulunmalı, kapalıyken hiç
    eklenmemeli — "ayar var ama etkisiz" durumunu yakalayan test.
    '''

    regularized = MLP(
        10,
        6,
        MLPConfig(hidden_dims=(12,), batch_norm=True, dropout=0.4),
    )
    plain = MLP(
        10,
        6,
        MLPConfig(hidden_dims=(12,), batch_norm=False, dropout=0.0),
    )

    assert any(isinstance(layer, nn.BatchNorm1d) for layer in regularized.modules())
    assert any(isinstance(layer, nn.Dropout) for layer in regularized.modules())
    assert not any(isinstance(layer, nn.BatchNorm1d) for layer in plain.modules())
    assert not any(isinstance(layer, nn.Dropout) for layer in plain.modules())


@pytest.mark.parametrize(
    'config',
    [
        MLPConfig(learning_rate=float('nan')),
        MLPConfig(learning_rate=float('inf')),
        MLPConfig(weight_decay=float('nan')),
    ],
)
def test_config_rejects_non_finite_optimization_values(config: MLPConfig) -> None:
    '''NaN/sonsuz öğrenme oranı ve weight decay değerleri doğrulamada reddedilmeli.

    NaN karşılaştırmaları Python'da sessizce False döndüğü için bu değerler
    özel isfinite kontrolü olmadan sızabilirdi; test bu korumayı kilitler.
    '''

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    'features',
    [torch.zeros(16), torch.zeros(2, 15)],
)
def test_forward_rejects_wrong_input_shape(features: torch.Tensor) -> None:
    '''Yanlış şekilli girdi (1-D vektör ya da yanlış boyut) açık hata mesajıyla reddedilmeli.'''

    model = MLP(
        input_dim=16,
        num_classes=6,
        config=MLPConfig(hidden_dims=(8,), batch_norm=False),
    )

    with pytest.raises(ValueError, match='Expected input shape'):
        model(features)
