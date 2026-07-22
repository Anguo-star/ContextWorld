from __future__ import annotations

import numpy as np
import pytest

from contextworld.evaluation.planner_mechanism import (
    fixed_candidate_bank,
    simulate_tworoom_candidates,
    spearman,
    topk_overlap,
)


def test_fixed_candidate_bank_is_deterministic_and_seeded() -> None:
    first = fixed_candidate_bank(eval_seed=42, evaluation_index=3, query_index=2)
    second = fixed_candidate_bank(eval_seed=42, evaluation_index=3, query_index=2)
    other = fixed_candidate_bank(eval_seed=43, evaluation_index=3, query_index=2)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert np.count_nonzero(first[0]) == 0


def test_exact_candidates_use_assumed_speed_and_report_success_step() -> None:
    actions = np.zeros((2, 25, 2), dtype=np.float32)
    actions[0, :, 0] = 1.0
    result = simulate_tworoom_candidates(
        query_state=np.asarray([30.0, 100.0]),
        goal_state=np.asarray([80.0, 100.0]),
        raw_actions=actions,
        speed=5.0,
        door_position=49.0,
    )
    assert result["success"][0]
    assert result["steps_to_success"][0] == 7
    assert not result["success"][1]
    assert result["final_distances"][0] < result["final_distances"][1]


def test_rank_metrics() -> None:
    a = np.asarray([1.0, 2.0, 3.0])
    b = np.asarray([1.1, 2.1, 3.1])
    c = np.asarray([3.0, 2.0, 1.0])
    assert spearman(a, b) == 1.0
    assert spearman(a, c) == -1.0
    assert topk_overlap(np.arange(40), np.arange(40), k=30) == 1.0


def test_spearman_uses_average_ranks_for_many_ties_and_is_order_invariant() -> None:
    endpoint = np.asarray([0.0] * 49 + [1.0] * 40 + [2.0] * 11)
    predicted = np.asarray([0.0] * 49 + [2.0] * 40 + [1.0] * 11)
    first = spearman(predicted, endpoint)
    permutation = np.random.default_rng(17).permutation(len(endpoint))
    second = spearman(predicted[permutation], endpoint[permutation])
    assert first == pytest.approx(second, abs=1e-15)
    assert spearman(np.zeros(300), np.arange(300.0)) == 0.0
