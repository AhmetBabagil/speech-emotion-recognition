'''Çok sınıflı olasılık kalibrasyonu metriklerinin testleri.

calibration.py'nin her fonksiyonu iki yönden sınanır:
- **Matematik**: NLL, Brier ve ECE değerleri elle hesaplanan küçük örneklerle
  karşılaştırılır; formül hatası anında yakalanır.
- **Sözleşme**: Sıcaklık ölçekleme argmax'ı korumalı, geçersiz girdiler
  (yanlış boyut, NaN, olasılık olmayan değerler, bool sıcaklık) her zaman
  ValueError ile reddedilmeli, raporlar JSON'a kayıpsız çevrilebilmeli.
'''

import json

import numpy as np
import pytest

from odev3.calibration import (
    classification_calibration_report,
    confidence_bin_statistics,
    expected_calibration_error,
    fit_temperature,
    multiclass_brier_score,
    multiclass_negative_log_likelihood,
    temperature_scale_probabilities,
    validate_probability_inputs,
    validate_probability_matrix,
)


def test_validate_probability_matrix_does_not_require_labels() -> None:
    '''Matris doğrulaması etiketten bağımsız çalışmalı ve float64'e yükseltmeli.

    Birim matris (her satır tek sınıfa 1.0 verir) geçerli bir olasılık
    matrisidir; doğrulama onu değiştirmeden, yalnızca tip yükselterek
    döndürmelidir.
    '''

    probabilities = np.eye(6, dtype=np.float32)

    probability_array = validate_probability_matrix(probabilities)

    assert probability_array.dtype == np.float64
    np.testing.assert_allclose(probability_array, probabilities)


def test_temperature_scaling_controls_confidence_and_preserves_argmax() -> None:
    '''T=1 kimlik, T>1 yumuşatma, T<1 keskinleştirme olmalı; argmax hep korunmalı.

    Sıcaklık ölçeklemenin tanımlayıcı üç özelliği aynı testte: satır
    toplamları 1 kalmalı, tahmin edilen sınıf değişmemeli, en yüksek
    olasılık T=2'de düşmeli ve T=0.5'te yükselmeli.
    '''

    probabilities = np.array(
        [
            [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
            [0.05, 0.65, 0.1, 0.08, 0.07, 0.05],
        ]
    )

    unchanged = temperature_scale_probabilities(probabilities, 1.0)
    softened = temperature_scale_probabilities(probabilities, 2.0)
    sharpened = temperature_scale_probabilities(probabilities, 0.5)

    np.testing.assert_array_equal(unchanged, probabilities)
    np.testing.assert_allclose(softened.sum(axis=1), 1.0)
    np.testing.assert_allclose(sharpened.sum(axis=1), 1.0)
    np.testing.assert_array_equal(
        softened.argmax(axis=1),
        probabilities.argmax(axis=1),
    )
    np.testing.assert_array_equal(
        sharpened.argmax(axis=1),
        probabilities.argmax(axis=1),
    )
    assert np.all(softened.max(axis=1) < probabilities.max(axis=1))
    assert np.all(sharpened.max(axis=1) > probabilities.max(axis=1))


@pytest.mark.parametrize('temperature', [0.0, -1.0, np.nan, np.inf, True, 'bad'])
def test_temperature_scaling_rejects_invalid_temperature(
    temperature: object,
) -> None:
    '''Sıfır, negatif, NaN, sonsuz, bool ve metin sıcaklıkların tümü reddedilmeli.

    True özellikle sinsi bir durum: float(True)=1.0 sessizce geçebilirdi;
    kod bool'u bilinçli olarak ayrıca kontrol eder.
    '''

    with pytest.raises(ValueError, match='positive finite'):
        temperature_scale_probabilities(
            np.eye(6),
            temperature,  # type: ignore[arg-type]
        )


def test_fit_temperature_reduces_nll_for_overconfident_predictions() -> None:
    '''Aşırı özgüvenli (%95 emin ama yarısı yanlış) modelde T>1 öğrenilmeli ve NLL düşmeli.

    Kurgu: tahminlerin yarısı kasten yanlışlanır ama olasılıklar hep 0.95
    verir — klasik aşırı özgüven. Doğru davranış: arama 1'den büyük bir
    sıcaklık bulur, NLL iyileşir, iyileşme bağımsız NLL hesabıyla tutarlıdır
    ve ortalama güven düşer. Sonuç sözlüğü ayrıca JSON'a kayıpsız çevrilebilmeli.
    '''

    predicted = np.arange(24) % 6
    labels = predicted.copy()
    # Her ikinci örneğin gerçek etiketini kaydır: modelin yarısı yanlış olsun.
    labels[1::2] = (labels[1::2] + 1) % 6
    probabilities = np.full((len(labels), 6), 0.01)
    probabilities[np.arange(len(labels)), predicted] = 0.95

    fit = fit_temperature(probabilities, labels)
    scaled = temperature_scale_probabilities(
        probabilities,
        fit['temperature'],
    )

    assert fit['temperature'] > 1.0
    assert fit['validation_nll_after'] < fit['validation_nll_before']
    assert fit['validation_nll_after'] == pytest.approx(
        multiclass_negative_log_likelihood(scaled, labels)
    )
    assert scaled.max(axis=1).mean() < probabilities.max(axis=1).mean()
    assert json.loads(json.dumps(fit)) == fit


def test_fit_temperature_prefers_one_when_all_candidates_tie() -> None:
    '''Tekdüze (uniform) olasılıklarda tüm sıcaklıklar aynı NLL'i verir; T=1 seçilmeli.

    Beraberlik bozma kuralının testi: eşit kayıplarda 1.0'a en yakın aday
    kazanır ("en az müdahale" ilkesi) ve iyileşme sıfır raporlanır.
    '''

    probabilities = np.full((12, 6), 1.0 / 6.0)
    labels = np.tile(np.arange(6), 2)

    fit = fit_temperature(probabilities, labels)

    assert fit['temperature'] == pytest.approx(1.0)
    assert fit['validation_nll_improvement'] == pytest.approx(0.0)


@pytest.mark.parametrize(
    'kwargs',
    [
        # Sırasıyla: minimum=0 (pozitif değil), minimum=1 (1'i dışlar),
        # maximum=1 (1'i dışlar), sonsuz maksimum, çok az kaba adım,
        # bool adım sayısı.
        {'minimum': 0.0},
        {'minimum': 1.0},
        {'maximum': 1.0},
        {'maximum': np.inf},
        {'coarse_steps': 20},
        {'fine_steps': True},
    ],
)
def test_fit_temperature_rejects_invalid_search_settings(
    kwargs: dict[str, object],
) -> None:
    '''Geçersiz arama ayarları (aralık 1'i içermiyor, adım sayısı yetersiz/bool) reddedilmeli.'''

    with pytest.raises(ValueError):
        fit_temperature(
            np.eye(6),
            np.arange(6),
            **kwargs,  # type: ignore[arg-type]
        )


def test_validate_probability_inputs_returns_aligned_arrays() -> None:
    '''Olasılık+etiket doğrulaması tipleri normalize etmeli (float64, int64), değerleri korumalı.'''

    probabilities = np.eye(6, dtype=np.float32)
    labels = np.arange(6, dtype=np.int32)

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )

    assert probability_array.dtype == np.float64
    assert label_array.dtype == np.int64
    np.testing.assert_allclose(probability_array, probabilities)
    np.testing.assert_array_equal(label_array, labels)


@pytest.mark.parametrize(
    ('probabilities', 'labels'),
    [
        # Sırasıyla: boş matris, 5 sütun (6 gerekli), uzunluk uyumsuzluğu,
        # 2-D etiket, float etiket, NaN olasılık, negatif olasılık,
        # satır toplamı 1 olmayan matris, aralık dışı etiket (6).
        (np.array([]), np.array([], dtype=np.int64)),
        (np.ones((2, 5)) / 5.0, np.array([0, 1])),
        (np.ones((2, 6)) / 6.0, np.array([0])),
        (np.ones((2, 6)) / 6.0, np.array([[0, 1]])),
        (np.ones((2, 6)) / 6.0, np.array([0.0, 1.0])),
        (np.full((2, 6), np.nan), np.array([0, 1])),
        (
            np.array(
                [
                    [-0.1, 0.3, 0.2, 0.2, 0.2, 0.2],
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
                ]
            ),
            np.array([0, 1]),
        ),
        (np.ones((2, 6)) / 5.0, np.array([0, 1])),
        (np.ones((2, 6)) / 6.0, np.array([0, 6])),
    ],
)
def test_validate_probability_inputs_rejects_invalid_data(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> None:
    '''"Olasılık gibi görünen ama olmayan" her girdi biçimi ValueError ile reddedilmeli.'''

    with pytest.raises(ValueError):
        validate_probability_inputs(probabilities, labels)


def test_negative_log_likelihood_matches_manual_mean() -> None:
    '''NLL, elle hesaplanan -(log p1 + log p2)/2 değerine eşit olmalı.

    Doğru sınıf olasılıkları 0.8 ve 0.5; formül başka hiçbir şeyi hesaba
    katmamalı (yalnızca doğru sınıfın olasılığı önemlidir).
    '''

    probabilities = np.array(
        [
            [0.8, 0.04, 0.04, 0.04, 0.04, 0.04],
            [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    labels = np.array([0, 1])

    nll = multiclass_negative_log_likelihood(probabilities, labels)

    assert nll == pytest.approx(-(np.log(0.8) + np.log(0.5)) / 2.0)


def test_negative_log_likelihood_handles_zero_true_probability() -> None:
    '''Doğru sınıfa tam 0 olasılık verilse bile NLL sonlu kalmalı (taban kırpması).

    Kırpma olmasaydı log(0) = -inf olur ve tüm ortalama sonsuza giderdi.
    Sonuç sonlu ama ÇOK büyük (>20) olmalı — hata cezasız kalmıyor, sadece
    sayısal olarak yönetiliyor.
    '''

    probabilities = np.array(
        [[0.0, 0.2, 0.2, 0.2, 0.2, 0.2]]
    )

    nll = multiclass_negative_log_likelihood(
        probabilities,
        np.array([0]),
    )

    assert np.isfinite(nll)
    assert nll > 20.0


def test_brier_score_is_zero_for_perfect_probabilities() -> None:
    '''Mükemmel one-hot tahminlerde Brier skoru tam 0 olmalı (alt sınır).'''

    probabilities = np.eye(6, dtype=np.float64)
    labels = np.arange(6, dtype=np.int64)

    score = multiclass_brier_score(probabilities, labels)

    assert score == pytest.approx(0.0)


def test_brier_score_matches_uniform_six_class_baseline() -> None:
    '''Tekdüze 1/6 tahminlerde Brier skoru analitik değere (5/6) eşit olmalı.

    El hesabı: doğru sınıf için (1/6 - 1)^2 + beş yanlış sınıf için (1/6)^2
    = 25/36 + 5/36 = 30/36 = 5/6. "Hiçbir şey bilmeyen" modelin taban çizgisi.
    '''

    probabilities = np.full((6, 6), 1.0 / 6.0)
    labels = np.arange(6, dtype=np.int64)

    score = multiclass_brier_score(probabilities, labels)

    assert score == pytest.approx(5.0 / 6.0)


def test_confidence_bins_account_for_boundaries_and_empty_bins() -> None:
    '''Dilim ataması sınır değerlerini tutarlı işlemeli; boş dilimler None kalmalı.

    5 dilim, 4 örnek: güvenler 0.2, 0.4, 0.8 ve 1.0. Beklenen sayımlar
    [0, 1, 1, 0, 2] — 0.2 ve 0.4 sınır değerleri sol dilime (side='right'
    kuralı), 0.8 ve 1.0 son dilime düşer. Boş dilimin ortalama güveni None
    olmalı (0.0 ile karışmamalı).
    '''

    probabilities = np.array(
        [
            [0.2, 0.16, 0.16, 0.16, 0.16, 0.16],
            [0.4, 0.12, 0.12, 0.12, 0.12, 0.12],
            [0.04, 0.04, 0.8, 0.04, 0.04, 0.04],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 2, 3])

    statistics = confidence_bin_statistics(
        probabilities,
        labels,
        bins=5,
    )

    assert [entry['count'] for entry in statistics] == [0, 1, 1, 0, 2]
    assert sum(entry['fraction'] for entry in statistics) == pytest.approx(1.0)
    assert statistics[0]['mean_confidence'] is None
    assert statistics[1]['accuracy'] == pytest.approx(1.0)
    assert statistics[2]['accuracy'] == pytest.approx(0.0)
    assert statistics[4]['mean_confidence'] == pytest.approx(0.9)


@pytest.mark.parametrize('bins', [1, 2.5, True])
def test_confidence_bins_reject_invalid_bin_counts(bins: object) -> None:
    '''Dilim sayısı en az 2 olan gerçek bir tamsayı olmalı; 1, 2.5 ve True reddedilmeli.'''

    with pytest.raises(ValueError):
        confidence_bin_statistics(
            np.eye(6),
            np.arange(6),
            bins=bins,  # type: ignore[arg-type]
        )


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    '''Güven=doğruluk=1.0 olan mükemmel tahminlerde ECE tam 0 olmalı.'''

    error = expected_calibration_error(
        np.eye(6),
        np.arange(6),
        bins=5,
    )

    assert error == pytest.approx(0.0)


def test_expected_calibration_error_matches_weighted_bin_gaps() -> None:
    '''ECE, dolu dilimlerin oran-ağırlıklı |doğruluk - güven| toplamına eşit olmalı.

    El hesabı (önceki dilim testiyle aynı veri): dilim2 -> 0.25*|1-0.2|=0.2,
    dilim3 -> 0.25*|0-0.4|=0.1, dilim5 -> 0.5*|1-0.9|=0.05; toplam 0.35.
    '''

    probabilities = np.array(
        [
            [0.2, 0.16, 0.16, 0.16, 0.16, 0.16],
            [0.4, 0.12, 0.12, 0.12, 0.12, 0.12],
            [0.04, 0.04, 0.8, 0.04, 0.04, 0.04],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    error = expected_calibration_error(
        probabilities,
        np.array([0, 1, 2, 3]),
        bins=5,
    )

    assert error == pytest.approx(0.35)


def test_calibration_report_is_zero_error_for_perfect_predictions() -> None:
    '''Mükemmel tahminlerde raporun TÜM hata metrikleri 0, doğruluk ve güven 1 olmalı.'''

    report = classification_calibration_report(
        np.eye(6),
        np.arange(6),
        bins=5,
    )

    assert report['sample_size'] == 6
    assert report['num_classes'] == 6
    assert report['num_bins'] == 5
    assert report['negative_log_likelihood'] == pytest.approx(0.0)
    assert report['multiclass_brier_score'] == pytest.approx(0.0)
    assert report['expected_calibration_error'] == pytest.approx(0.0)
    assert report['maximum_calibration_error'] == pytest.approx(0.0)
    assert report['accuracy'] == pytest.approx(1.0)
    assert report['mean_confidence'] == pytest.approx(1.0)
    assert report['confidence_minus_accuracy'] == pytest.approx(0.0)


def test_calibration_report_matches_metrics_and_is_json_serializable() -> None:
    '''Çatı rapor, tek tek metrik fonksiyonlarıyla tutarlı ve JSON'a kayıpsız çevrilebilir olmalı.

    İç tutarlılık testi: raporun içindeki NLL/Brier/ECE, aynı veriye ayrı
    ayrı uygulanan fonksiyonların sonuçlarına eşit olmalı. JSON gidiş-dönüşü
    de raporun diske sorunsuz yazılabileceğini garanti eder.
    '''

    probabilities = np.array(
        [
            [0.6, 0.08, 0.08, 0.08, 0.08, 0.08],
            [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.1, 0.1, 0.2],
        ]
    )
    labels = np.array([0, 2, 2])

    report = classification_calibration_report(
        probabilities,
        labels,
        bins=5,
    )

    assert report['negative_log_likelihood'] == pytest.approx(
        multiclass_negative_log_likelihood(probabilities, labels)
    )
    assert report['multiclass_brier_score'] == pytest.approx(
        multiclass_brier_score(probabilities, labels)
    )
    assert report['expected_calibration_error'] == pytest.approx(
        expected_calibration_error(probabilities, labels, bins=5)
    )
    assert report['accuracy'] == pytest.approx(2.0 / 3.0)
    assert json.loads(json.dumps(report)) == report
