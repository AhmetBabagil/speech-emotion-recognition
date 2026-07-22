'''Tests for the Assignment 3 command-line interface.'''

import pytest

from odev3.run_experiment import (
    _result_line,
    build_feature_config,
    build_feature_configs,
)


def test_result_line_formats_test_metrics() -> None:
    result = {'test': {'accuracy': 0.61234, 'macro_f1': 0.54321}}

    line = _result_line('cremad', result)

    assert line == 'cremad: test accuracy=0.6123, macro-F1=0.5432'


def test_build_feature_config_supports_full_utterance_resizing() -> None:
    config = build_feature_config(96, 'resize')

    assert config.n_frames == 96
    assert config.frame_strategy == 'resize'
    assert config.vector_size == 6144


def test_build_feature_config_enforces_assignment_minimum() -> None:
    with pytest.raises(ValueError, match='at least 4000'):
        build_feature_config(62, 'crop_pad')


def test_build_feature_configs_applies_dataset_specific_overrides() -> None:
    configs = build_feature_configs(
        ('cremad', 'meld'),
        default_frames=64,
        default_strategy='crop_pad',
        frame_overrides={'cremad': None, 'meld': 96},
        strategy_overrides={'cremad': 'resize', 'meld': None},
    )

    assert configs['cremad'].frame_strategy == 'resize'
    assert configs['cremad'].n_frames == 64
    assert configs['meld'].frame_strategy == 'crop_pad'
    assert configs['meld'].n_frames == 96
