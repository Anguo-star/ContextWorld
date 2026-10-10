"""Fixed native readout math, source resampling, and matched-control contracts."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


readout = module('task_history_readout', 'scripts/diagnose_task_history_readout.py')


def test_dual_ridge_matches_sklearn_binary_multiclass_and_continuous():
    pytest.importorskip('sklearn')
    readout.verify_dual()


def test_bootstrap_resamples_source_groups_and_recomputes_class_balance():
    # One source has many rows from class 0; a row-weighted accuracy bootstrap
    # would give a different interval from balanced accuracy.
    labels = np.array([0] * 9 + [1] + [2] + [0, 1, 2, 3, 4, 5])
    prediction = np.array([0] * 9 + [0] + [2] + [1, 1, 2, 3, 4, 5])
    groups = np.array(['a'] * 11 + ['b'] * 6)
    classes = np.arange(6)
    actual = readout.bootstrap_balanced(labels, prediction, groups, classes)
    keys, inverse = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(readout.SEED).integers(len(keys), size=(2000, len(keys)))
    expected = []
    for draw in draws:
        indices = np.concatenate([np.flatnonzero(inverse == group) for group in draw])
        present = np.unique(labels[indices])
        expected.append(np.mean([(prediction[indices][labels[indices] == cls] == cls).mean() for cls in present]))
    assert actual['interval_95'] == pytest.approx(np.quantile(expected, [.025, .975]))
    assert actual['resamples_missing_at_least_one_class'] > 0
    assert not np.allclose(actual['interval_95'], [np.mean(labels == prediction)] * 2)


def test_natural_unmatched_rows_have_no_current_only_ceiling():
    data = {'query_ids': np.array(['a', 'b', 'c']), 'labels': np.array([0, 1, 0]),
        'history_pixels': np.zeros((3, 3, 2, 2, 3), dtype=np.uint8),
        'rawactions': np.zeros((3, 15, 2), dtype=np.float32)}
    result = readout.controls(data, {'matched_group_key': 'query_ids', 'current_action_matched': False})
    assert result['repeated_groups'] == 0
    assert result['current_only_deterministic_balanced_accuracy_ceiling'] is None


def test_door_builder_requires_same_physical_label_in_both_splits():
    pytest.importorskip('lance')
    builder = module('task_panel_builder', 'scripts/build_task_mechanism_panels.py')
    valid = [dict(mode='blocked', label=0, physical=0), dict(mode='passable', label=1, physical=1),
        dict(mode='rule_blocked', label=0, physical=0), dict(mode='rule_passable', label=1, physical=1)]
    builder.assert_door_label_mapping(valid)
    with pytest.raises(AssertionError):
        builder.assert_door_label_mapping([dict(mode='rule_blocked', label=1, physical=1)])
