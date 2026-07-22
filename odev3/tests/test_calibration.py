'''Tests for multiclass probability calibration metrics.'''

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
    probabilities = np.eye(6, dtype=np.float32)

    probability_array = validate_probability_matrix(probabilities)

    assert probability_array.dtype == np.float64
    np.testing.assert_allclose(probability_array, probabilities)


def test_temperature_scaling_controls_confidence_and_preserves_argmax() -> None:
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
    with pytest.raises(ValueError, match='positive finite'):
        temperature_scale_probabilities(
            np.eye(6),
            temperature,  # type: ignore[arg-type]
        )


def test_fit_temperature_reduces_nll_for_overconfident_predictions() -> None:
    predicted = np.arange(24) % 6
    labels = predicted.copy()
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
    probabilities = np.full((12, 6), 1.0 / 6.0)
    labels = np.tile(np.arange(6), 2)

    fit = fit_temperature(probabilities, labels)

    assert fit['temperature'] == pytest.approx(1.0)
    assert fit['validation_nll_improvement'] == pytest.approx(0.0)


@pytest.mark.parametrize(
    'kwargs',
    [
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
    with pytest.raises(ValueError):
        fit_temperature(
            np.eye(6),
            np.arange(6),
            **kwargs,  # type: ignore[arg-type]
        )


def test_validate_probability_inputs_returns_aligned_arrays() -> None:
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
    with pytest.raises(ValueError):
        validate_probability_inputs(probabilities, labels)


def test_negative_log_likelihood_matches_manual_mean() -> None:
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
    probabilities = np.eye(6, dtype=np.float64)
    labels = np.arange(6, dtype=np.int64)

    score = multiclass_brier_score(probabilities, labels)

    assert score == pytest.approx(0.0)


def test_brier_score_matches_uniform_six_class_baseline() -> None:
    probabilities = np.full((6, 6), 1.0 / 6.0)
    labels = np.arange(6, dtype=np.int64)

    score = multiclass_brier_score(probabilities, labels)

    assert score == pytest.approx(5.0 / 6.0)


def test_confidence_bins_account_for_boundaries_and_empty_bins() -> None:
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
    with pytest.raises(ValueError):
        confidence_bin_statistics(
            np.eye(6),
            np.arange(6),
            bins=bins,  # type: ignore[arg-type]
        )


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    error = expected_calibration_error(
        np.eye(6),
        np.arange(6),
        bins=5,
    )

    assert error == pytest.approx(0.0)


def test_expected_calibration_error_matches_weighted_bin_gaps() -> None:
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
