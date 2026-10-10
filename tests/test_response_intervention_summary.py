"""Checks for source-paired, energy-pooled intervention summaries."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_response_intervention.py'
spec = importlib.util.spec_from_file_location('response_summary', SCRIPT)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def row(query, source, error, energy, mse):
    return {'query_id': query, 'source_group': source, 'response_mse': error,
            'target_response_energy': energy, 'endpoint_mse': mse, 'common_mse': mse - error}


def test_identical_arms_have_zero_paired_delta_and_interval():
    rows = [row('a', 's1', 1, 2, 3), row('b', 's2', 2, 4, 5)]
    got = summary.paired_delta(rows, rows, repetitions=80)
    assert got['weighted_minus_native'] == 0
    assert got['ci95'] == [0, 0]
    assert got['weighted_minus_native_endpoint_mse_over_initial'] == 0
    assert got['endpoint_mse_difference_ci95'] == [0, 0]


def test_repeated_source_groups_are_resampled_together_and_ratios_are_pooled():
    native = [row('a', 's1', 1, 1, 2), row('b', 's1', 1, 9, 4), row('c', 's2', 2, 2, 6)]
    weighted = [row('a', 's1', 0, 1, 1), row('b', 's1', 1, 9, 3), row('c', 's2', 1, 2, 5)]
    assert summary.pooled(native)['response_nre'] == pytest.approx(4 / 12)
    assert summary.pooled(native)['endpoint_mse'] == pytest.approx(4)
    drawn = summary.resampled(native, ['s1', 's2'], [0, 0])
    assert [r['query_id'] for r in drawn] == ['a', 'b', 'a', 'b']
    assert summary.pooled(drawn)['response_nre'] == pytest.approx(2 / 10)
    got = summary.paired_delta(native, weighted, native, repetitions=80)
    assert got['weighted_minus_native'] == pytest.approx(-2 / 12)
    assert got['source_groups'] == 2
    assert got['weighted_minus_native_endpoint_mse_over_initial'] == pytest.approx(-1 / 4)
    assert got['endpoint_mse_difference_ci95'][0] <= got['weighted_minus_native_endpoint_mse_over_initial'] <= got['endpoint_mse_difference_ci95'][1]
