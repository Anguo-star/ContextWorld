import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_conditional_map import pair_diff, metric, select_alpha, kernel  # noqa: E402


def test_pair_difference_uses_matched_labels_and_groups():
    data = {'labels': np.array([1, 0, 1, 0]), 'query_ids': np.array(
        ['b', 'a', 'a', 'b']), 'source_groups': np.array(['g2', 'g1', 'g1', 'g2'])}
    diff, q, g = pair_diff(np.array([[5.], [2.], [7.], [3.]]), data)
    assert q.tolist() == ['a', 'b'] and g.tolist() == ['g1', 'g2']
    np.testing.assert_array_equal(diff, [[5.], [2.]])


def test_no_intercept_map_and_metric():
    x = np.array([[1., 0.], [0., 1.], [1., 1.]])
    y = 2*x
    coef = kernel(x, y, 1e-9)
    np.testing.assert_allclose(x@coef, y, atol=1e-8)
    score = metric(x@coef, y)
    assert score['nre'] < 1e-15 and abs(score['gain']-1) < 1e-8


def test_grouped_cv_perfect_linear_map():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 3))
    y = x@rng.normal(size=(3, 2))
    groups = np.repeat(np.arange(15), 2)
    alpha, scores = select_alpha(x, y, groups)
    assert alpha == 1e-6 and scores[str(alpha)] < 1e-10
