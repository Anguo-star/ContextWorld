"""Behavioral controls for the candidate-selection regret diagnostic."""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from summarize_speed_action_selection import selection_metrics


def test_correct_history_oracle_and_wrong_history_control():
    physical = np.array([[[0., 4.], [4., 0.]]])
    r = selection_metrics(physical, physical, physical)
    assert r['correct_regret'] == pytest.approx([0.])
    assert r['wrong_regret'] == pytest.approx([4.])
    assert r['history_benefit'] == pytest.approx([4.])
    assert r['history_blind_lower_bound'] == pytest.approx([2.])


def test_history_blind_ties_cannot_fabricate_context_benefit():
    physical = np.array([[[0., 3., 8.], [5., 0., 5.], [8., 3., 0.]]])
    predicted = np.zeros_like(physical)
    r = selection_metrics(physical, predicted, physical)
    assert r['history_benefit'] == pytest.approx([0.])
    assert np.all(r['correct_regret'] >= r['history_blind_lower_bound'])
    assert r['encoded_true_regret'] == pytest.approx([0.])


def test_all_wrong_histories_are_equal_weight_and_labels_do_not_matter():
    physical = np.array([[[0., 1., 7.], [2., 0., 6.], [3., 4., 0.]]])
    r = selection_metrics(physical, physical, physical)
    assert r['wrong_regret'] == pytest.approx([(1+7+2+6+3+4)/6])
    perm = [2, 0, 1]
    reordered = selection_metrics(physical[:, perm], physical[:, perm], physical[:, perm])
    for key in r:
        assert r[key] == pytest.approx(reordered[key])
