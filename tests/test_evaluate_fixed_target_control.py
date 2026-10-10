"""Small arithmetic and paired-history controls for the fixed T0 evaluator."""
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluate_fixed_target_control.py'
spec = importlib.util.spec_from_file_location('fixed_target', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_common_offset_changes_full_error_but_not_response():
    target = np.array([[0., 0.], [1., 0.]], dtype=np.float32)
    predicted = target + np.array([4., -2.], dtype=np.float32)
    assert mod.ratio(predicted[1:] - predicted[:1], target[1:] - target[:1]) == 0
    assert mod.terms(predicted, target)[0].mean() == 20


def test_history_swap_is_real_prediction_not_pair_permutation():
    data = {'labels': np.array([0, 1]), 'query_ids': np.array(['q', 'q']),
            'source_groups': np.array(['g', 'g']), 'action_blocks': np.zeros((2, 3, 5, 2))}
    target = np.array([[0.], [2.]], dtype=np.float32)
    old = {'target': target, 'prediction': target.copy(), 'swapped': target[::-1].copy()}
    new = {'target': target.copy(), 'prediction': target.copy(), 'swapped': target.copy()}
    result = mod.evaluate_split(data, old, new)
    assert result['old']['history_benefit'] == 4
    assert result['new']['history_benefit'] == 0
    assert result['delta_response_nre']['delta'] == 0


def test_identity_bootstrap_and_ignore_history_response():
    data = {'labels': np.array([0, 1, 0, 1]), 'query_ids': np.array(['a', 'a', 'b', 'b']),
            'source_groups': np.array(['g1', 'g1', 'g2', 'g2']), 'action_blocks': np.zeros((4, 3, 5, 2))}
    target = np.array([[0.], [2.], [1.], [3.]], dtype=np.float32)
    correct = {'target': target, 'prediction': target.copy(), 'swapped': target[[1, 0, 3, 2]].copy()}
    identity = mod.evaluate_split(data, correct, correct)
    assert identity['old']['response_nre'] == 0
    for key in ('delta_response_nre', 'delta_full_error', 'delta_full_error_over_B', 'delta_history_benefit_over_B'):
        assert identity[key]['delta'] == 0
        assert identity[key]['ci95'] == [0, 0]
    ignored = {'target': target, 'prediction': np.array([[1.], [1.], [2.], [2.]], dtype=np.float32),
               'swapped': np.array([[1.], [1.], [2.], [2.]], dtype=np.float32)}
    result = mod.evaluate_split(data, correct, ignored)
    assert result['new']['response_nre'] == 1
