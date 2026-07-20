from __future__ import annotations

from typing import Any

from contextworld.evaluation.context_model_attribution_analysis import (
    _cross_model_effect,
    _gate,
)


def _fast_slow_summary(
    *,
    effect: float,
    p_value: float,
    fast_only: int,
    slow_only: int,
) -> dict[str, Any]:
    return {
        "wrong_fast_minus_wrong_slow_success_rate_points": effect,
        "paired_sign_test": {"two_sided_p_value": p_value},
        "wrong_fast_only_successes": fast_only,
        "wrong_slow_only_successes": slow_only,
    }


def test_stable_fast_over_slow_gate_uses_all_frozen_requirements() -> None:
    spec = {
        "minimum_effect_pp": 5.0,
        "paired_exact_sign_test_p_max": 0.05,
        "require_fast_only_greater_than_slow_only": True,
    }
    passed = _gate(
        _fast_slow_summary(
            effect=6.0,
            p_value=0.01,
            fast_only=12,
            slow_only=2,
        ),
        spec,
    )
    assert passed["passed"]

    insufficient_effect = _gate(
        _fast_slow_summary(
            effect=4.99,
            p_value=0.01,
            fast_only=12,
            slow_only=2,
        ),
        spec,
    )
    assert not insufficient_effect["passed"]


def _record(
    *,
    seed: int,
    evaluation_id: str,
    condition: str,
    success: bool,
) -> dict[str, Any]:
    return {
        "eval_seed": seed,
        "evaluation_id": evaluation_id,
        "query_id": "query-1",
        "evaluation_index": 0,
        "repeat_index": 0,
        "speed": 5.0,
        "template_id": "d080_g00",
        "cem_seed": 123,
        "cem_rng_state_sha256_before": "same",
        "goal_state": [10.0, 20.0],
        "condition": condition,
        "success": success,
    }


def test_cross_model_effect_is_paired_difference_in_context_effect() -> None:
    key = (42, "eval-1")
    target = {
        "wrong_slow": {
            key: _record(
                seed=42,
                evaluation_id="eval-1",
                condition="wrong",
                success=False,
            )
        },
        "wrong_fast": {
            key: _record(
                seed=42,
                evaluation_id="eval-1",
                condition="wrong",
                success=True,
            )
        },
    }
    control = {
        "wrong_slow": {
            key: _record(
                seed=42,
                evaluation_id="eval-1",
                condition="wrong",
                success=False,
            )
        },
        "wrong_fast": {
            key: _record(
                seed=42,
                evaluation_id="eval-1",
                condition="wrong",
                success=False,
            )
        },
    }
    result = _cross_model_effect(
        target=target,
        control=control,
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )
    assert result["paired_evaluations"] == 1
    assert result["target_minus_control_context_effect_points"] == 100.0
    assert result["evaluation_bootstrap_95_ci_points"] == [
        100.0,
        100.0,
    ]
