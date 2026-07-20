from __future__ import annotations

import math
from pathlib import Path

import pytest

from contextworld.evaluation.context_direction_analysis import (
    direction_result_path,
    directional_decisions,
)
from contextworld.evaluation.icl_sensitive import (
    SensitiveGeometry,
    build_speed_icl_sensitive_catalog,
    generate_same_room_geometries,
)
from contextworld.evaluation.icl_sensitive_analysis import (
    exact_paired_sign_test,
    select_distance_bins,
)


def test_sensitive_geometry_banks_are_deterministic_and_disjoint() -> None:
    kwargs = {
        "distances": [48, 72, 112],
        "variants_per_distance": 3,
    }
    calibration = generate_same_room_geometries(
        **kwargs, geometry_seed=2026071801
    )
    repeated = generate_same_room_geometries(
        **kwargs, geometry_seed=2026071801
    )
    heldout = generate_same_room_geometries(
        **kwargs, geometry_seed=2026071802
    )

    assert calibration == repeated
    assert len(calibration) == 9
    assert {
        (*row.reset_state, *row.goal_state) for row in calibration
    }.isdisjoint(
        {(*row.reset_state, *row.goal_state) for row in heldout}
    )
    for row in calibration + heldout:
        observed = math.dist(row.reset_state, row.goal_state)
        assert math.isclose(
            observed,
            float(row.distance_bin),
            rel_tol=0.0,
            abs_tol=1e-5,
        )
        assert 125.0 <= row.reset_state[0] <= 209.0
        assert 15.0 <= row.reset_state[1] <= 209.0
        assert 125.0 <= row.goal_state[0] <= 209.0
        assert 15.0 <= row.goal_state[1] <= 209.0


def test_distance_selection_uses_rank_then_adjacent_neighbor() -> None:
    summaries = [
        {
            "distance_bin": 64,
            "correct_minus_wrong_success_rate_points": 8.0,
            "pooled_correct_wrong_success_rate_percent": 50.0,
            "eligible": True,
        },
        {
            "distance_bin": 72,
            "correct_minus_wrong_success_rate_points": 7.0,
            "pooled_correct_wrong_success_rate_percent": 45.0,
            "eligible": True,
        },
        {
            "distance_bin": 96,
            "correct_minus_wrong_success_rate_points": 20.0,
            "pooled_correct_wrong_success_rate_percent": 50.0,
            "eligible": False,
        },
        {
            "distance_bin": 104,
            "correct_minus_wrong_success_rate_points": 6.0,
            "pooled_correct_wrong_success_rate_percent": 50.0,
            "eligible": True,
        },
    ]

    assert select_distance_bins(
        summaries, spacing=8, maximum_bins=2
    ) == [64, 72]


def test_exact_sign_test_is_two_sided_and_handles_no_discordance() -> None:
    assert exact_paired_sign_test(0, 0) == {
        "discordant_pairs": 0,
        "two_sided_p_value": 1.0,
    }
    result = exact_paired_sign_test(6, 0)
    assert result["discordant_pairs"] == 6
    assert result["two_sided_p_value"] == 0.03125


def test_sensitive_catalog_geometry_override_fails_closed_on_count(
    tmp_path: Path,
) -> None:
    geometry = SensitiveGeometry(
        template_id="d072_g00",
        distance_bin=72,
        geometry_variant=0,
        reset_state=(150.0, 100.0),
        goal_state=(150.0, 172.0),
        context_direction=(1.0, 0.0),
        query_action=(0.0, 0.35),
    )
    with pytest.raises(ValueError, match="Geometry count"):
        build_speed_icl_sensitive_catalog(
            repo_root=tmp_path,
            output_catalog=tmp_path / "catalog.json",
            payload_root=tmp_path / "payloads",
            split="test",
            distances=[72],
            variants_per_distance=2,
            geometry_seed=1,
            catalog_seed=2,
            stable_worldmodel_commit="test",
            speeds=(5.0, 5.1),
            geometries_override=[geometry],
        )


def test_direction_result_paths_and_frozen_decisions() -> None:
    assert direction_result_path(
        Path("/tmp/results"), direction="wrong_slow", seed=42
    ) == Path("/tmp/results/wrong_slow_n50_s42.json")
    with pytest.raises(ValueError, match="Unknown direction"):
        direction_result_path(
            Path("/tmp/results"), direction="wrong", seed=42
        )

    def summary(
        effect: float,
        first_only: int,
        second_only: int,
        p_value: float,
        *,
        effect_key: str,
        first_key: str,
        second_key: str,
    ) -> dict[str, object]:
        return {
            effect_key: effect,
            first_key: first_only,
            second_key: second_only,
            "paired_sign_test": {"two_sided_p_value": p_value},
        }

    config = {
        "decisions_frozen_before_execution": {
            "correctness_aligned_planning_icl": {
                "minimum_effect_each_pp": 5.0,
                "paired_exact_sign_test_p_max_each": 0.05,
            },
            "higher_speed_prompt_bias_confirmation": {
                "minimum_effect_pp": 5.0,
                "paired_exact_sign_test_p_max": 0.05,
            },
        }
    }
    decisions = directional_decisions(
        correct_vs_slow=summary(
            6.0,
            20,
            2,
            0.001,
            effect_key=(
                "correct_minus_wrong_slow_success_rate_points"
            ),
            first_key="correct_only_successes",
            second_key="wrong_slow_only_successes",
        ),
        correct_vs_fast=summary(
            7.0,
            21,
            1,
            0.001,
            effect_key=(
                "correct_minus_wrong_fast_success_rate_points"
            ),
            first_key="correct_only_successes",
            second_key="wrong_fast_only_successes",
        ),
        fast_vs_slow=summary(
            6.0,
            18,
            2,
            0.001,
            effect_key=(
                "wrong_fast_minus_wrong_slow_success_rate_points"
            ),
            first_key="wrong_fast_only_successes",
            second_key="wrong_slow_only_successes",
        ),
        config=config,
    )
    assert decisions["correctness_aligned_planning_icl"]["established"]
    assert decisions["higher_speed_prompt_bias"]["confirmed"]
    assert decisions["classification"] == (
        "correctness_alignment_and_higher_speed_bias_both_present"
    )
