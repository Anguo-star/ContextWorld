from __future__ import annotations

import numpy as np

from scripts.analyze_tworoom_speed_isolated_v2 import (
    _ability_comparison,
    _contrast,
    _exact_sign_test,
    _holm_adjust,
    _physical_row_summary,
    _recover_speed_from_context,
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


def test_context_speed_recovery_uses_dense_action_block() -> None:
    states = np.asarray([[10.0, 20.0], [12.0, 22.0]])
    actions = np.asarray(
        [
            [
                [0.1, -0.1],
                [0.2, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
                [0.0, -0.1],
            ],
            [
                [-0.1, 0.2],
                [0.0, 0.1],
                [0.2, -0.1],
                [0.1, 0.0],
                [0.1, 0.1],
            ],
        ]
    )
    speed = 4.8
    next_states = states + speed * actions.sum(axis=1)

    estimates, residuals = _recover_speed_from_context(
        states=states,
        next_states=next_states,
        actions=actions,
    )

    np.testing.assert_allclose(estimates, speed)
    np.testing.assert_allclose(residuals, 0.0, atol=1.0e-12)


def test_physical_summary_reports_speed_and_displacement_response() -> None:
    records = []
    condition_values = {
        "low": {
            "history_speed": 3.0,
            "history_relation": "slower",
            "inferred_speed": 4.0,
            "predicted_displacement": [1.0, 0.0],
            "error": 1.0,
        },
        "same": {
            "history_speed": 5.0,
            "history_relation": "same",
            "inferred_speed": 5.0,
            "predicted_displacement": [2.0, 0.0],
            "error": 0.0,
        },
        "high": {
            "history_speed": 7.0,
            "history_relation": "faster",
            "inferred_speed": 6.0,
            "predicted_displacement": [3.0, 0.0],
            "error": 1.0,
        },
    }
    for seed in (42, 43):
        conditions = {}
        for name, row in condition_values.items():
            by_horizon = {}
            for horizon in (1, 2, 3, 5, 10):
                by_horizon[str(horizon)] = {
                    "inferred_speed": row["inferred_speed"],
                    "position_error_px": row["error"],
                    "displacement_magnitude_error_px": row["error"],
                    "displacement_direction_error_deg": 0.0,
                    "latent_mse_to_true_query_future": row["error"],
                    "latent_mse_to_nearest_oracle": 0.0,
                    "predicted_displacement": row[
                        "predicted_displacement"
                    ],
                    "true_displacement": [2.0, 0.0],
                    "inferred_minus_query_speed": (
                        row["inferred_speed"] - 5.0
                    ),
                    "inferred_minus_history_speed": (
                        row["inferred_speed"] - row["history_speed"]
                    ),
                }
            conditions[name] = {
                "history_speed": row["history_speed"],
                "history_relation": row["history_relation"],
                "by_horizon": by_horizon,
            }
        records.append(
            {
                "eval_seed": seed,
                "static_query_id": "shared-query",
                "action_probe": {"family": "varying_magnitude"},
                "conditions": conditions,
            }
        )

    summary = _physical_row_summary(
        records,
        bootstrap_seed=17,
        bootstrap_resamples=100,
    )

    one_block = summary["history_speed_response"]["1"]
    assert one_block["high_minus_low_inferred_speed"] == 2.0
    assert one_block["inferred_speed_response_gain"] == 0.5
    assert one_block["high_minus_low_predicted_displacement_px"] == 2.0
    assert summary["condition_means"]["same"]["by_horizon"]["1"][
        "predicted_displacement_magnitude_px"
    ] == 2.0
    assert summary["gates"]["passed"]

    for record in records:
        record["conditions"]["same"]["by_horizon"]["1"][
            "displacement_magnitude_error_px"
        ] = 2.0
    failed = _physical_row_summary(
        records,
        bootstrap_seed=18,
        bootstrap_resamples=100,
    )
    assert not failed["gates"]["one_block_same_speed_lowest"]
    assert not failed["gates"]["passed"]
