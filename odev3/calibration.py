'''Probability calibration metrics for held-out multiclass predictions.'''

from __future__ import annotations

from typing import Any

import numpy as np

from ser.constants import NUM_CLASSES


ROW_SUM_TOLERANCE = 1e-6
PROBABILITY_FLOOR = 1e-12


def validate_probability_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Return aligned float64 probabilities and integer class labels.'''

    probability_array = np.asarray(probabilities)
    label_array = np.asarray(labels)

    if probability_array.ndim != 2:
        raise ValueError('probabilities must be a two-dimensional matrix.')
    if probability_array.shape[1] != NUM_CLASSES:
        raise ValueError(
            f'probabilities must contain exactly {NUM_CLASSES} class columns.'
        )
    if probability_array.shape[0] == 0:
        raise ValueError('probabilities and labels cannot be empty.')
    if not np.issubdtype(probability_array.dtype, np.number):
        raise ValueError('probabilities must contain numeric values.')
    if label_array.ndim != 1:
        raise ValueError('labels must be one-dimensional.')
    if len(label_array) != len(probability_array):
        raise ValueError('probabilities and labels must have equal sample counts.')
    if not np.issubdtype(label_array.dtype, np.integer):
        raise ValueError('labels must contain integer class indices.')

    probability_array = probability_array.astype(np.float64, copy=False)
    label_array = label_array.astype(np.int64, copy=False)
    if not np.all(np.isfinite(probability_array)):
        raise ValueError('probabilities must contain only finite values.')
    if np.any(probability_array < 0.0) or np.any(probability_array > 1.0):
        raise ValueError('probabilities must remain between 0 and 1.')
    if np.any(label_array < 0) or np.any(label_array >= NUM_CLASSES):
        raise ValueError(f'labels must remain between 0 and {NUM_CLASSES - 1}.')

    row_sums = probability_array.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=ROW_SUM_TOLERANCE, rtol=0.0):
        raise ValueError('each probability row must sum to 1.')
    return probability_array, label_array


def multiclass_negative_log_likelihood(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    '''Return mean negative log-probability assigned to the true class.'''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    row_indices = np.arange(len(label_array))
    true_probabilities = probability_array[row_indices, label_array]
    safe_probabilities = np.clip(
        true_probabilities,
        PROBABILITY_FLOOR,
        1.0,
    )
    return float(-np.log(safe_probabilities).mean())


def multiclass_brier_score(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> float:
    '''Return mean summed squared error against one-hot class targets.'''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    one_hot_targets = np.eye(NUM_CLASSES, dtype=np.float64)[label_array]
    squared_errors = np.square(probability_array - one_hot_targets)
    return float(squared_errors.sum(axis=1).mean())


def confidence_bin_statistics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    '''Summarize confidence and accuracy inside equal-width bins.'''

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise ValueError('bins must be an integer.')
    if bins < 2:
        raise ValueError('bins must be at least 2.')

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    predictions = probability_array.argmax(axis=1)
    confidences = probability_array.max(axis=1)
    correct = predictions == label_array
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    assignments = np.searchsorted(
        edges[1:-1],
        confidences,
        side='right',
    )

    statistics: list[dict[str, Any]] = []
    sample_size = len(label_array)
    for bin_index in range(int(bins)):
        mask = assignments == bin_index
        count = int(mask.sum())
        summary: dict[str, Any] = {
            'bin': bin_index + 1,
            'lower': float(edges[bin_index]),
            'upper': float(edges[bin_index + 1]),
            'count': count,
            'fraction': float(count / sample_size),
            'mean_confidence': None,
            'accuracy': None,
            'absolute_gap': None,
        }
        if count:
            mean_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
            summary.update(
                {
                    'mean_confidence': mean_confidence,
                    'accuracy': accuracy,
                    'absolute_gap': abs(accuracy - mean_confidence),
                }
            )
        statistics.append(summary)
    return statistics


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    '''Return confidence-bin-weighted absolute calibration error.'''

    statistics = confidence_bin_statistics(
        probabilities,
        labels,
        bins=bins,
    )
    weighted_gaps = (
        entry['fraction'] * entry['absolute_gap']
        for entry in statistics
        if entry['count'] > 0
    )
    return float(sum(weighted_gaps))


def classification_calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, Any]:
    '''Return a JSON-ready held-out multiclass calibration summary.'''

    probability_array, label_array = validate_probability_inputs(
        probabilities,
        labels,
    )
    statistics = confidence_bin_statistics(
        probability_array,
        label_array,
        bins=bins,
    )
    nonempty_gaps = [
        float(entry['absolute_gap'])
        for entry in statistics
        if entry['count'] > 0
    ]
    predictions = probability_array.argmax(axis=1)
    confidences = probability_array.max(axis=1)
    accuracy = float((predictions == label_array).mean())
    mean_confidence = float(confidences.mean())

    return {
        'method': 'top-label equal-width confidence calibration',
        'sample_size': int(len(label_array)),
        'num_classes': NUM_CLASSES,
        'num_bins': int(bins),
        'negative_log_likelihood': multiclass_negative_log_likelihood(
            probability_array,
            label_array,
        ),
        'multiclass_brier_score': multiclass_brier_score(
            probability_array,
            label_array,
        ),
        'expected_calibration_error': expected_calibration_error(
            probability_array,
            label_array,
            bins=bins,
        ),
        'maximum_calibration_error': max(nonempty_gaps),
        'accuracy': accuracy,
        'mean_confidence': mean_confidence,
        'confidence_minus_accuracy': mean_confidence - accuracy,
        'bins': statistics,
    }
