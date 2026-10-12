"""Behavioral checks for frozen response mapping, without reading experiment panels."""
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'diagnose_fixed_response_mapping.py'
spec = importlib.util.spec_from_file_location('fixed_response_mapping', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_interleaved_pairs_center_and_group_folds():
    ids = np.array(['q2', 'q1', 'q2', 'q1', 'q3', 'q4', 'q3', 'q4'])
    labels = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    groups = np.array(['mirror_b', 'mirror_a', 'mirror_b', 'mirror_a', 'mirror_c', 'mirror_d', 'mirror_c', 'mirror_d'])
    rows = m.pair_rows(ids, labels, groups)
    assert np.array_equal(labels[rows], np.tile([0, 1], (4, 1)))
    future = np.arange(8.0)[:, None]
    y = m.centered_targets(future, rows)
    np.testing.assert_allclose(y[rows].sum(axis=1), 0)
    assert np.allclose(y[rows[:, 0]], -y[rows[:, 1]])
    row_groups = np.empty(8, dtype=groups.dtype)
    row_groups[rows] = groups[rows[:, 0], None]
    from sklearn.model_selection import GroupKFold
    for train, held in GroupKFold(4).split(y, groups=row_groups):
        assert not (set(row_groups[train]) & set(row_groups[held]))


def test_current_duplicate_inputs_cancel_even_with_distinct_targets():
    rng = np.random.default_rng(4)
    n = 12
    history = rng.normal(size=(n, 3, 4))
    actions = rng.normal(size=(n, 15, 2))
    history[1::2, -1] = history[::2, -1]
    actions[1::2] = actions[::2]
    target = rng.normal(size=(n, 3))
    target[1::2] = -target[::2]
    x = m.features(history, actions, current=True)
    assert np.array_equal(x[::2], x[1::2])
    pt, _, _ = m.fit_predict(x, x, target, 'rbf', .01)
    np.testing.assert_allclose(pt[::2], pt[1::2], atol=1e-11)
    rows = np.arange(n).reshape(-1, 2)
    centered = m.centered_targets(target, rows)
    assert abs(m.metrics(centered, pt, rows)['nre']-1) < 1e-10


def test_exact_linear_and_nonlinear_response_and_train_only_selection():
    rng = np.random.default_rng(8)
    n = 24
    base = rng.normal(size=(n//2, 2))
    x = np.empty((n, 2))
    x[::2], x[1::2] = base, -base
    y = (2*x[:, :1] - x[:, 1:2])
    groups = np.repeat(np.arange(6), 4)
    choice, scores, folds = m.cv_select(x, y, groups, np.repeat(np.arange(n//2), 2), 'linear', alphas=[1e-6, 1e-2])
    assert choice['nre'] < 1e-7
    assert len(folds) == 4 and len(scores) == 2
    for train, held in folds:
        assert not (set(groups[train]) & set(groups[held]))
    train, dev, _ = m.fit_predict(x, x[:5], y, 'linear', 1e-8)
    np.testing.assert_allclose(train, y, atol=1e-6)
    np.testing.assert_allclose(dev, y[:5], atol=1e-6)
    # RBF interpolates a nonlinear conditional response without an optimizer.
    nonlinear = np.sin(3*x[:, :1]) + x[:, 1:2]**2
    fitted, _, _ = m.fit_predict(x, x[:3], nonlinear, 'rbf', 1e-8)
    np.testing.assert_allclose(fitted, nonlinear, atol=2e-3)


def test_pair_response_error_ignores_common_prediction_offset():
    target = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    prediction = target + np.array([[10.0], [10.0], [-7.0], [-7.0]])
    rows = np.array([[0, 1], [2, 3]])
    error, energy = m.pair_response_error(target, prediction, rows)
    assert error == 0.0 and energy == 20.0
    assert np.square(prediction-target).sum() > 0


def test_split_local_damping_groups_and_exact_duplicate_guard():
    original = np.array(['group-23', 'group-24'])
    train_groups = m.source_identities('motion_damping', 'training', original)
    dev_groups = m.source_identities('motion_damping', 'development', original)
    assert not (set(train_groups) & set(dev_groups))
    np.testing.assert_array_equal(m.source_identities('contact_friction', 'training', original), original)
    h = np.zeros((2, 3, 4), dtype=np.float32)
    a = np.zeros((2, 15, 2), dtype=np.float32)
    h[1] = 1
    train = {'input_fingerprints': m.input_fingerprints(h, a)}
    other = {'input_fingerprints': m.input_fingerprints(h.copy(), a.copy())}
    import pytest
    with pytest.raises(ValueError, match='exact history\\+action inputs reused'):
        m.reject_cross_split_duplicate_inputs(train, other)
    other['input_fingerprints'] = m.input_fingerprints(h+2, a)
    m.reject_cross_split_duplicate_inputs(train, other)
