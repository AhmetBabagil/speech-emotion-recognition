'''Tests for the Assignment 3 command-line interface.'''

from odev3.run_experiment import _result_line


def test_result_line_formats_test_metrics() -> None:
    result = {'test': {'accuracy': 0.61234, 'macro_f1': 0.54321}}

    line = _result_line('cremad', result)

    assert line == 'cremad: test accuracy=0.6123, macro-F1=0.5432'
