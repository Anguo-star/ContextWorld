from __future__ import annotations

import numpy as np
import pytest

from scripts.analyze_tworoom_original_ability import (
    _mean_ci,
    _paired_planning_comparison,
)


def _record(index: int, success: bool, distance: float) -> dict:
    return {
        "evaluation_id": f"e{index}",
        "eval_seed": 42,
        "evaluation_index": index,
        "source_kind": "original_h5",
        "source_path": "artifacts/source.h5",
        "episode": index,
        "start_step": 0,
        "goal_offset": 25,
        "cem_group_seed": 42,
        "stratum": "original_future25",
        "room_relation": "cross_room",
        "initial_state": [10.0, 10.0],
        "goal_state": [180.0, 180.0],
        "success": success,
        "final_distance": distance,
    }


def test_mean_ci_is_exact_for_constant_delta() -> None:
    result = _mean_ci(
        np.full(20, -0.02),
        rng=np.random.default_rng(7),
        resamples=100,
        confidence=0.95,
    )
    assert result["point"] == pytest.approx(-0.02)
    assert result["ci_lower"] == pytest.approx(-0.02)
    assert result["ci_upper"] == pytest.approx(-0.02)


def test_paired_noninferiority_passes_identical_models() -> None:
    reference = {
        f"e{index}": _record(index, index % 2 == 0, 3.0 + index)
        for index in range(20)
    }
    candidate = {key: dict(value) for key, value in reference.items()}
    result = _paired_planning_comparison(
        reference,
        candidate,
        bootstrap_seed=3072,
        resamples=200,
        confidence=0.95,
        success_margin=-0.05,
        distance_margin=5.0,
    )
    assert result["passed"]
    assert all(result["gates"].values())


def test_paired_noninferiority_detects_success_and_stratum_collapse() -> None:
    reference = {
        f"e{index}": _record(index, True, 2.0) for index in range(20)
    }
    candidate = {
        key: {**value, "success": False, "final_distance": 20.0}
        for key, value in reference.items()
    }
    result = _paired_planning_comparison(
        reference,
        candidate,
        bootstrap_seed=3072,
        resamples=200,
        confidence=0.95,
        success_margin=-0.05,
        distance_margin=5.0,
    )
    assert not result["passed"]
    assert not result["gates"]["success_rate_non_inferior"]
    assert not result["gates"]["final_distance_non_inferior"]
    assert not result["gates"]["no_solvable_stratum_collapse"]
