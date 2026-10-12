"""Focused checks for pair-centered response mapping summaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/summarize_fixed_response_mapping.py'
spec = importlib.util.spec_from_file_location('fixed_response_mapping_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_response_nre_ignores_large_pair_common_bias():
    target = np.array([[-1.0], [1.0]])
    pairs = np.array([[0, 1]])
    groups = np.array(['source'])

    perfect = summary.aggregate(summary.statistics(target, target + 100.0, pairs), groups)
    assert perfect['NRE'] == pytest.approx(0.0)
    assert perfect['swapped_NRE'] == pytest.approx(4.0)
    assert perfect['centered_target_error_ratio'] == pytest.approx(10000.0)

    zero = summary.aggregate(summary.statistics(target, np.zeros_like(target), pairs), groups)
    assert zero['NRE'] == pytest.approx(1.0)


def test_mirror_duplicates_within_each_source_leave_bootstrap_unchanged():
    stats = {key: np.array(values, dtype=float) for key, values in {
        'B': [2.0, 4.0, 8.0],
        'R': [0.2, 1.0, 6.0],
        'S': [5.0, 8.0, 10.0],
        'E': [1.0, 2.0, 3.0],
    }.items()}
    groups = np.array(['first', 'first', 'second'])
    original = summary.aggregate(stats, groups)
    doubled = summary.aggregate({key: np.repeat(values, 2) for key, values in stats.items()},
                                np.repeat(groups, 2))

    assert doubled['source_groups'] == original['source_groups'] == 2
    assert doubled['queries'] == 2 * original['queries']
    for key in ('NRE', 'swapped_NRE', 'centered_target_error_ratio'):
        assert doubled[key] == pytest.approx(original[key])
    assert doubled['NRE_ci95'] == pytest.approx(original['NRE_ci95'])
