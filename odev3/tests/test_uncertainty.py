'''Ayrılmış (held-out) metrik belirsizlik tahminlerinin testleri.

uncertainty.py'deki bootstrap zincirinin her halkası ayrı sınanır: etiket
doğrulama, tabakalı yeniden örnekleme, yüzdelik sınırlar ve uçtan uca
bootstrap sonuçlarının hem matematiksel tutarlılığı hem tekrarlanabilirliği.
'''

import numpy as np
import pytest

from odev3.uncertainty import (
    bootstrap_metric_intervals,
    percentile_bounds,
    stratified_resample_indices,
    validate_label_arrays,
)


def test_validate_label_arrays_returns_aligned_int64_views() -> None:
    '''Farklı tamsayı tiplerindeki (int32/int16) girdiler int64'e çevrilmeli, değerler korunmalı.'''

    y_true = np.array([0, 1, 2, 2], dtype=np.int32)
    y_pred = np.array([0, 2, 2, 1], dtype=np.int16)

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)

    assert true_array.dtype == np.int64
    assert predicted_array.dtype == np.int64
    np.testing.assert_array_equal(true_array, y_true)
    np.testing.assert_array_equal(predicted_array, y_pred)


@pytest.mark.parametrize(
    ('y_true', 'y_pred'),
    [
        # Sırasıyla: boş diziler, uzunluk uyumsuzluğu, 2 boyutlu dizi,
        # float etiket, negatif etiket, sınıf aralığı dışında etiket (>=6).
        (np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
        (np.array([0, 1]), np.array([0])),
        (np.array([[0, 1]]), np.array([0, 1])),
        (np.array([0.0, 1.0]), np.array([0, 1])),
        (np.array([0, -1]), np.array([0, 1])),
        (np.array([0, 6]), np.array([0, 1])),
    ],
)
def test_validate_label_arrays_rejects_invalid_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    '''Geçersiz etiket kombinasyonlarının tamamı ValueError ile reddedilmeli.'''

    with pytest.raises(ValueError):
        validate_label_arrays(y_true, y_pred)


def test_stratified_resample_preserves_each_class_size() -> None:
    '''Tabakalı yeniden örnekleme her sınıfın örnek SAYISINI aynen korumalı.

    Yöntemin tanımlayıcı özelliği budur: iadeli örnekleme sınıf İÇİNDE
    yapılır, sınıflar arası oran asla değişmez. bincount karşılaştırması
    bunu doğrudan doğrular.
    '''

    labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)

    indices = stratified_resample_indices(
        labels,
        np.random.default_rng(42),
    )
    sampled_labels = labels[indices]

    assert len(indices) == len(labels)
    np.testing.assert_array_equal(
        np.bincount(sampled_labels, minlength=3),
        np.bincount(labels, minlength=3),
    )


def test_percentile_bounds_use_equal_probability_tails() -> None:
    '''%80 güven için alt/üst sınırlar 10. ve 90. yüzdelik olmalı (eşit kuyruk).

    0..100 dizisinde yüzdelikler kendisiyle çakışır, bu yüzden beklenen
    değerler tam olarak 10.0 ve 90.0'dır — formülün en okunur doğrulaması.
    '''

    samples = np.arange(101, dtype=np.float64)

    lower, upper = percentile_bounds(samples, confidence=0.80)

    assert lower == pytest.approx(10.0)
    assert upper == pytest.approx(90.0)


def test_bootstrap_intervals_are_exact_for_perfect_predictions() -> None:
    '''Kusursuz tahminlerde tüm aralıklar [1.0, 1.0] olmalı.

    y_pred == y_true iken hangi alt küme örneklenirse örneklensin her metrik
    1.0 çıkar; dolayısıyla bootstrap dağılımı tek noktaya çöker. Bu, hem
    aralık mantığının hem metrik hesabının uç durum sağlamasıdır.
    '''

    y_true = np.repeat(np.arange(6, dtype=np.int64), 4)

    result = bootstrap_metric_intervals(
        y_true,
        y_true.copy(),
        iterations=100,
        confidence=0.90,
        seed=7,
    )

    assert result['sample_size'] == 24
    assert set(result['class_counts'].values()) == {4}
    for interval in result['metrics'].values():
        assert interval == {
            'estimate': 1.0,
            'lower': 1.0,
            'upper': 1.0,
        }


def test_bootstrap_intervals_are_reproducible_for_same_seed() -> None:
    '''Aynı tohumla iki çağrı BİREBİR aynı sonucu vermeli (tekrarlanabilirlik).

    Ayrıca kasıtlı hatalar eklendiği için macro-F1 alt sınırının 1.0'dan
    küçük olması beklenir — aralık gerçekten "açılmış" olmalı.
    '''

    y_true = np.repeat(np.arange(6, dtype=np.int64), 5)
    y_pred = y_true.copy()
    # Beş örneğin tahminini bilerek boz: metrikler 1.0'ın altına insin.
    y_pred[[1, 7, 13, 19, 25]] = np.array([1, 2, 3, 4, 5])

    first = bootstrap_metric_intervals(y_true, y_pred, iterations=100, seed=123)
    second = bootstrap_metric_intervals(y_true, y_pred, iterations=100, seed=123)

    assert first == second
    assert first['metrics']['macro_f1']['lower'] < 1.0
    assert first['metrics']['macro_f1']['upper'] <= 1.0


def test_bootstrap_rejects_incomplete_classes_and_too_few_iterations() -> None:
    '''Eksik sınıf (5/6) ve yetersiz iterasyon (<100) ayrı ayrı reddedilmeli.

    Eksik sınıf, tabakalı örneklemeyi ve macro metrikleri tanımsız bırakır;
    az iterasyon ise yüzdelikleri güvenilmez yapar. İkisi de erken hatadır.
    '''

    five_class_labels = np.repeat(np.arange(5, dtype=np.int64), 3)
    six_class_labels = np.repeat(np.arange(6, dtype=np.int64), 3)

    with pytest.raises(ValueError, match='Every canonical class'):
        bootstrap_metric_intervals(
            five_class_labels,
            five_class_labels,
            iterations=100,
        )
    with pytest.raises(ValueError, match='at least 100'):
        bootstrap_metric_intervals(
            six_class_labels,
            six_class_labels,
            iterations=99,
        )
