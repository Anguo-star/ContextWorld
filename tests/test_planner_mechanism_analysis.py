from __future__ import annotations

import pytest

from contextworld.evaluation.planner_mechanism_analysis import (
    paired_binary_comparison,
    paired_continuous_comparison,
    summarize_closed_loop_condition,
)


def test_paired_binary_comparison_preserves_pair_direction() -> None:
    result = paired_binary_comparison(
        [True, True, False, True],
        [False, True, True, False],
        left_label="fast",
        right_label="slow",
    )
    assert result["fast"]["successes"] == 3
    assert result["slow"]["successes"] == 2
    assert result["fast_minus_slow_success_rate_points"] == 25.0
    assert result["fast_only_successes"] == 2
    assert result["slow_only_successes"] == 1
    assert result["paired_exact_sign_test"]["discordant_pairs"] == 3


def test_paired_continuous_comparison_reports_lower_side() -> None:
    result = paired_continuous_comparison(
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        left_label="correct",
        right_label="fast",
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )
    assert result["correct_minus_fast"]["mean"] == -1.0
    assert result["correct_lower_pairs"] == 3
    assert result["fast_lower_pairs"] == 0
    assert result["correct_minus_fast"][
        "evaluation_bootstrap_95_ci"
    ] == [-1.0, -1.0]


def _row(
    *,
    evaluation_id: str,
    success: bool,
    final_distance: float,
) -> dict:
    return {
        "eval_seed": 42,
        "evaluation_id": evaluation_id,
        "success": success,
        "final_distance": final_distance,
        "trajectory": {
            "goal_distances": [10.0, final_distance],
            "raw_steps_executed": 1,
            "steps_to_success": 1 if success else None,
            "path_efficiency_success_only": 0.8 if success else None,
            "normalized_distance_auc": 0.75,
            "distance_auc_raw_step_mean": 7.5,
            "path_length": 5.0,
            "progress_per_path_length": (10.0 - final_distance) / 5.0,
        },
    }


def test_closed_loop_summary_keeps_success_and_progress_separate() -> None:
    rows = {
        (42, "a"): _row(
            evaluation_id="a", success=True, final_distance=5.0
        ),
        (42, "b"): _row(
            evaluation_id="b", success=False, final_distance=9.0
        ),
    }
    result = summarize_closed_loop_condition(rows)
    assert result["success_rate_percent"] == 50.0
    assert result["normalized_progress"]["mean"] == pytest.approx(0.3)
    assert result["steps_to_success_success_only"]["mean"] == 1.0
