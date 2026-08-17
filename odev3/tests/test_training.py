'''Ağırlıklı kayıp, değerlendirme ve erken durdurmalı MLP eğitiminin testleri.

training.py'nin üç sözleşmesi doğrulanır: sınıf ağırlığı formülünün doğruluğu,
erken durdurmanın çalışıp EN İYİ epoch'un geri yüklenmesi ve aynı tohumla
eğitimin bit-bit tekrarlanabilir olması. Testler yapay, kolay ayrılabilir
bir veri kümesi kullanır — böylece CPU'da saniyeler içinde biterler.
'''

import numpy as np
import pytest
import torch

from odev3.model import MLPConfig
from odev3.training import (
    evaluate_arrays,
    inverse_frequency_weights,
    train_with_early_stopping,
)


def _separable_six_class_data(
    *, samples_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    # Yapay ama neredeyse mükemmel ayrılabilir 6 sınıflı veri üretir:
    # her örnek, kendi sınıfının one-hot vektörünün 4 katı + küçük Gauss
    # gürültüsüdür. Basit bir MLP bunu birkaç epoch'ta ~%100 öğrenir; bu da
    # erken durdurma ve geri yükleme davranışını hızlıca test etmeyi sağlar.
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(6, dtype=np.int64), samples_per_class)
    features = np.eye(6, dtype=np.float32)[labels] * 4.0
    features += rng.normal(0.0, 0.05, size=features.shape).astype(np.float32)
    return features, labels


def test_inverse_frequency_weights_match_assignment_formula() -> None:
    '''Ağırlıklar tam olarak n / (K * n_k) formülüne uymalı.

    6 örnek, 3 sınıf, sayımlar (3, 2, 1): beklenen ağırlıklar elle
    hesaplanıp karşılaştırılır. Ayrıca sıralama kontrolü: en nadir sınıf
    en büyük ağırlığı almalı.
    '''

    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)

    weights = inverse_frequency_weights(labels, num_classes=3)

    expected = torch.tensor([6 / (3 * 3), 6 / (3 * 2), 6 / (3 * 1)])
    torch.testing.assert_close(weights, expected)
    assert weights[0] < weights[1] < weights[2]


def test_class_weights_require_every_training_class() -> None:
    '''Eğitim verisinde hiç görülmeyen sınıf (burada 2) açık hatayla reddedilmeli.

    Sıfır sayım hem formülde sıfıra bölme demektir hem de o sınıfın
    öğrenilmesinin imkansız olduğunun işaretidir.
    '''

    labels = np.array([0, 0, 1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match='Every class'):
        inverse_frequency_weights(labels, num_classes=3)


def test_training_uses_early_stopping_and_restores_best_model() -> None:
    '''Erken durdurma devreye girmeli ve dönen model EN İYİ epoch'un modeli olmalı.

    Kolay veri + düşük patience: model hızla öğrenir, iyileşme durur ve
    eğitim 30 epoch dolmadan kesilir. Ardından dönen model yeniden
    değerlendirilir; skoru, eğitim sırasında kaydedilen "en iyi" skora
    BİREBİR eşitse geri yükleme doğru çalışıyor demektir (son epoch değil,
    en iyi epoch döndürülmüş). Olasılıkların satır toplamının 1 olması da
    softmax çıkışının tutarlılığını doğrular.
    '''

    train_features, train_labels = _separable_six_class_data(
        samples_per_class=10,
        seed=1,
    )
    validation_features, validation_labels = _separable_six_class_data(
        samples_per_class=4,
        seed=2,
    )
    config = MLPConfig(
        batch_size=12,
        learning_rate=0.02,
        patience=3,
        hidden_dims=(12,),
        activation='relu',
        batch_norm=False,
        dropout=0.0,
        weight_decay=0.0,
    )

    outcome = train_with_early_stopping(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        config,
        input_dim=6,
        num_classes=6,
        device=torch.device('cpu'),
        max_epochs=30,
        seed=7,
        amp=False,
    )
    _, restored_metrics, probabilities = evaluate_arrays(
        outcome.model,
        validation_features,
        validation_labels,
        class_weights=inverse_frequency_weights(train_labels, 6),
        device=torch.device('cpu'),
    )

    assert outcome.stopped_early
    assert outcome.epochs_trained < 30
    assert 1 <= outcome.best_epoch <= outcome.epochs_trained
    assert outcome.validation_metrics['macro_f1'] >= 0.95
    # Geri yüklenen modelin skoru, kaydedilen en iyi skora eşit olmalı.
    assert restored_metrics['macro_f1'] == outcome.validation_metrics['macro_f1']
    assert probabilities.shape == (len(validation_labels), 6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    # history satırları eğitim metriklerini de içermeli (öğrenme eğrileri için).
    assert {
        'train_accuracy',
        'train_balanced_accuracy',
        'train_macro_f1',
        'train_weighted_f1',
    }.issubset(outcome.history[0])


def test_training_is_reproducible_for_same_seed() -> None:
    '''Aynı tohum ve aynı argümanlarla iki eğitim, aynı geçmişi ve aynı ağırlıkları üretmeli.

    Determinizm garantisinin doğrudan testi: history sözlükleri eşit olmalı
    ve state_dict'teki her tensör sayısal olarak birebir örtüşmeli. Bu
    garanti olmadan "devam etme" (resume) ve kararlılık analizleri anlamsız
    olurdu.
    '''

    train_features, train_labels = _separable_six_class_data(
        samples_per_class=3,
        seed=10,
    )
    validation_features, validation_labels = _separable_six_class_data(
        samples_per_class=2,
        seed=11,
    )
    config = MLPConfig(
        batch_size=6,
        learning_rate=1e-3,
        patience=2,
        hidden_dims=(8,),
        batch_norm=False,
        dropout=0.0,
        weight_decay=0.0,
    )
    arguments = dict(
        train_features=train_features,
        train_labels=train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        config=config,
        input_dim=6,
        num_classes=6,
        device=torch.device('cpu'),
        max_epochs=2,
        seed=123,
        amp=False,
    )

    first = train_with_early_stopping(**arguments)
    second = train_with_early_stopping(**arguments)

    assert first.history == second.history
    for name, first_value in first.model.state_dict().items():
        torch.testing.assert_close(first_value, second.model.state_dict()[name])
