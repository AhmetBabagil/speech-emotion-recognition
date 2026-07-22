'''Uncertainty estimates for held-out classification metrics.'''

from __future__ import annotations

from typing import Any

import numpy as np

from ser.constants import CANONICAL_EMOTIONS, NUM_CLASSES
from ser.evaluate import compute_metrics


CORE_METRICS = (
    'accuracy',
    'balanced_accuracy',
    'macro_f1',
    'weighted_f1',
)


def percentile_bounds(
    samples: np.ndarray,
    confidence: float,
) -> tuple[float, float]:
    '''Return equal-tailed percentile bounds for bootstrap samples.'''

    sample_array = np.asarray(samples, dtype=np.float64)
    if sample_array.ndim != 1 or len(sample_array) == 0:
        raise ValueError('samples must be a non-empty one-dimensional array.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be strictly between 0 and 1.')
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(sample_array, [tail, 1.0 - tail])
    return float(lower), float(upper)


def validate_label_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''Return aligned one-dimensional integer label arrays.'''

    true_array = np.asarray(y_true)
    predicted_array = np.asarray(y_pred)
    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError('y_true and y_pred must be one-dimensional.')
    if len(true_array) == 0 or len(true_array) != len(predicted_array):
        raise ValueError(
            'y_true and y_pred must be non-empty and have equal lengths.'
        )
    if not np.issubdtype(true_array.dtype, np.integer):
        raise ValueError('y_true must contain integer class labels.')
    if not np.issubdtype(predicted_array.dtype, np.integer):
        raise ValueError('y_pred must contain integer class labels.')
    if np.any(true_array < 0) or np.any(predicted_array < 0):
        raise ValueError('Class labels cannot be negative.')
    if np.any(true_array >= NUM_CLASSES) or np.any(predicted_array >= NUM_CLASSES):
        raise ValueError(f'Class labels must be smaller than {NUM_CLASSES}.')
    return (
        true_array.astype(np.int64, copy=False),
        predicted_array.astype(np.int64, copy=False),
    )


def stratified_resample_indices(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    '''Sample with replacement inside each observed class.'''

    label_array = np.asarray(labels)
    if label_array.ndim != 1 or len(label_array) == 0:
        raise ValueError('labels must be a non-empty one-dimensional array.')

    sampled_parts: list[np.ndarray] = []
    for label in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == label)
        sampled_parts.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    return np.concatenate(sampled_parts).astype(np.int64, copy=False)


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    '''Estimate held-out metric uncertainty with a stratified bootstrap.'''

    true_array, predicted_array = validate_label_arrays(y_true, y_pred)
    if iterations < 100:
        raise ValueError('iterations must be at least 100.')
    if not 0.0 < confidence < 1.0:
        raise ValueError('confidence must be strictly between 0 and 1.')

    class_counts = np.bincount(true_array, minlength=NUM_CLASSES)
    if np.any(class_counts == 0):
        raise ValueError(
            'Every canonical class must occur in y_true; '
            f'counts={class_counts.tolist()}.'
        )

    observed = compute_metrics(true_array, predicted_array)
    distributions = {
        metric: np.empty(iterations, dtype=np.float64)
        for metric in CORE_METRICS
    }
    rng = np.random.default_rng(seed)
    for iteration in range(iterations):
        indices = stratified_resample_indices(true_array, rng)
        sampled = compute_metrics(
            true_array[indices],
            predicted_array[indices],
        )
        for metric in CORE_METRICS:
            distributions[metric][iteration] = float(sampled[metric])

    intervals: dict[str, dict[str, float]] = {}
    for metric in CORE_METRICS:
        lower, upper = percentile_bounds(distributions[metric], confidence)
        intervals[metric] = {
            'estimate': float(observed[metric]),
            'lower': lower,
            'upper': upper,
        }

    return {
        'method': 'class-stratified percentile bootstrap',
        'iterations': int(iterations),
        'confidence': float(confidence),
        'seed': int(seed),
        'sample_size': int(len(true_array)),
        'class_counts': {
            emotion: int(class_counts[index])
            for index, emotion in enumerate(CANONICAL_EMOTIONS)
        },
        'metrics': intervals,
    }
