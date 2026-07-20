from __future__ import annotations

from scripts.analyze_tworoom_speed_isolated_v2 import (
    _ability_comparison,
    _contrast,
    _exact_sign_test,
    _holm_adjust,
)


def test_exact_sign_test_handles_ties_outside_trial_count() -> None:
    assert _exact_sign_test(3, 0) == 0.25
    assert _exact_sign_test(0, 0) == 1.0


def test_paired_contrast_uses_other_minus_same_direction() -> None:
    records = [
        {
            "eval_seed": seed,
            "static_query_id": f"q{index}",
            "conditions": {
                "same": {"value": 1.0},
                "other": {"value": 2.0 + 0.1 * index},
            },
        }
        for seed in (42, 43)
        for index in range(3)
    ]
    summary = _contrast(
        records,
        same_condition="same",
        other_condition="other",
        value=lambda record, condition: record["conditions"][condition][
            "value"
        ],
        bootstrap_seed=7,
        bootstrap_resamples=100,
    )

    assert summary["other_minus_same_mean"] > 0.0
    assert summary["positive_eval_seeds"] == 2
    assert summary["passed_directional_stability"]


def test_ability_noninferiority_comparison_is_paired() -> None:
    reference = []
    candidate = []
    for index in range(10):
        common = {
            "eval_seed": 42,
            "evaluation_id": f"e{index}",
            "initial_state": [1.0, 2.0],
            "goal_state": [3.0, 4.0],
            "room_relation": "same_room",
            "source_kind": "original_h5",
            "source_path": "source.h5",
            "episode": index,
            "start_step": 0,
        }
        reference.append(
            {
                **common,
                "success": index != 0,
                "final_distance": 10.0,
            }
        )
        candidate.append(
            {
                **common,
                "success": True,
                "final_distance": 9.0,
            }
        )
    result = _ability_comparison(
        candidate,
        reference,
        bootstrap_seed=3,
        bootstrap_resamples=100,
    )

    assert result["candidate_minus_reference_success_rate_points"] == 10.0
    assert result["candidate_minus_reference_mean_final_distance_px"] == -1.0
    assert result["passed"]


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    rows = [
        {"cluster_sign_test_two_sided_p": value}
        for value in (0.01, 0.03, 0.02)
    ]
    _holm_adjust(rows, alpha=0.05)
    adjusted = sorted(
        (
            row["cluster_sign_test_two_sided_p"],
            row["holm_adjusted_p"],
        )
        for row in rows
    )

    assert [value for _, value in adjusted] == [0.03, 0.04, 0.04]
    assert all(row["holm_passed"] for row in rows)
