"""Algebra and pairing tests for the read-only native policy diagnostic."""
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import diagnose_native_policy_contrast as diagnostic  # noqa: E402


def test_pair_decomposition_and_common_bias():
    y = np.array([[0., 0.], [2., 0.], [10., 0.], [12., 0.]])
    pairs = np.array([[0, 1], [2, 3]])
    perfect = diagnostic.pair_statistics(y, y, y[[1, 0, 3, 2]], pairs)
    np.testing.assert_allclose(perfect['R'], 0)
    np.testing.assert_allclose(perfect['E'], 0)
    assert diagnostic.aggregate(perfect)['NRE'] == 0
    biased = diagnostic.pair_statistics(y, y + [5., 0.], y + [5., 0.], pairs)
    np.testing.assert_allclose(biased['R'], 0)
    np.testing.assert_allclose(biased['E'], biased['C'])
    assert diagnostic.aggregate(biased)['E_over_B'] > 1
    zero_response = np.repeat(y[pairs].mean(1), 2, axis=0)
    zero = diagnostic.pair_statistics(y, zero_response, zero_response, pairs)
    np.testing.assert_allclose(zero['R'], zero['B'])
    assert diagnostic.aggregate(zero)['NRE'] == 1
    assert diagnostic.aggregate(zero)['main_choice'] == 0  # strict tie is not a correct choice


def test_swap_respects_query_identity_not_row_order():
    data = dict(query_ids=np.array(['q2', 'q1', 'q2', 'q1']),
                labels=np.array([1, 0, 0, 1]),
                source_groups=np.array(['s2', 's1', 's2', 's1']),
                history_pixels=np.array([[[2], [9]], [[1], [8]], [[3], [9]], [[4], [8]]]),
                rawactions=np.array([[2], [1], [2], [1]]),
                action_blocks=np.array([[2], [1], [2], [1]]))
    pairs, swap = diagnostic.matched_pairs(data)
    np.testing.assert_array_equal(swap, [2, 3, 0, 1])
    np.testing.assert_array_equal(pairs, [[1, 3], [2, 0]])
    assert np.array_equal(swap[swap], np.arange(4))


def test_nonmatched_current_or_action_is_rejected():
    data = dict(query_ids=np.array(['q', 'q']), labels=np.array([0, 1]),
                source_groups=np.array(['s', 's']),
                history_pixels=np.array([[[1], [8]], [[2], [8]]]),
                rawactions=np.array([[1], [1]]), action_blocks=np.array([[1], [1]]))
    import pytest
    for key, index, value in [('history_pixels', (1, -1, 0), 9),
                              ('rawactions', (1, 0), 2), ('action_blocks', (1, 0), 2)]:
        broken = {k: v.copy() for k, v in data.items()}
        broken[key][index] = value
        with pytest.raises(AssertionError):
            diagnostic.matched_pairs(broken)
