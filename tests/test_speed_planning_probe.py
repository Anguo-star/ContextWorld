"""Response recovery and physical candidate regret measure different errors."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_speed_planning_probe import aggregate, decision_opportunity, measurements


def measure(tmp_path, prediction):
    # Each speed needs a different response; goal distance also varies with
    # candidate. Perfect latent predictions must recover the physical ranking.
    target = np.broadcast_to(np.array([[1., 2.], [2., 4.], [3., 6.]])[:, :, None, None], (3, 2, 5, 1)).copy()
    pred = prediction(target)
    path = tmp_path/'result.npz'
    np.savez(path, prediction=pred, target=target, initial=np.zeros((3, 1)),
             goal=np.array([4.]), physical_cost=np.abs(target[..., 0]-4.))
    return measurements(path)


def test_perfect_prediction_recovers_response_and_candidate_optimum(tmp_path):
    rows = measure(tmp_path, lambda z: z.copy())
    result = aggregate(rows)
    assert result['response_score'] == pytest.approx(100)
    assert result['prediction_error_ratio'] == 0
    assert result['regret'] == 0
    assert result['encoded_regret'] == 0
    assert result['history_benefit'] > 0


def test_history_invariant_prediction_has_zero_response_score(tmp_path):
    result = aggregate(measure(tmp_path, lambda z: np.repeat(z[1:2], 3, axis=0)))
    assert result['response_score'] == pytest.approx(0)
    assert result['history_benefit'] == pytest.approx(0)


def test_common_prediction_bias_survives_response_cancellation(tmp_path):
    result = aggregate(measure(tmp_path, lambda z: z+10))
    assert result['response_score'] == pytest.approx(100)
    assert result['prediction_error_ratio'] > 1
    assert result['regret'] > 0
    assert result['encoded_regret'] == 0


def test_different_futures_need_not_require_different_actions():
    cost = np.array([[1., 2.], [3., 4.], [5., 6.]])[:, :, None]
    states = np.repeat(cost[..., None], 2, axis=-1)
    row = decision_opportunity(cost, states)[0]
    assert row['mean_pair_future_separation'] > 0
    assert row['context_value'] == pytest.approx(0)
    assert row['common_candidate_within_one_px']


def test_speed_dependent_optimal_actions_have_positive_context_value():
    cost = np.array([[0., 10.], [5., 0.], [7., 1.]])[:, :, None]
    row = decision_opportunity(cost, np.repeat(cost[..., None], 2, axis=-1))[0]
    assert row['context_value'] == pytest.approx(10/3)
    assert not row['common_candidate_within_one_px']
