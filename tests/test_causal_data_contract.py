from __future__ import annotations

import pytest

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)


def _valid(**overrides):
    values = {
        "component_id": "example",
        "evidence_scope": "all published pairs",
        "continuous_environment_trajectory": True,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
        "maximum_query_state_gap": 1e-8,
        "query_state_tolerance": 1e-6,
        "query_pixels_exact": True,
        "query_actions_exact": True,
        "history_effect_present": True,
        "true_future_effect_present": True,
        "x0_policy": "shared_visible_start",
        "x0_static_leakage_check_passed": True,
    }
    values.update(overrides)
    return audit_causal_data_contract(**values)


def test_valid_continuous_pair_passes() -> None:
    assert _valid()["passed"] is True


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("state_installations_after_x0", 1, "no_state_installation_after_x0"),
        ("query_simulator_recreated", True, "query_simulator_not_recreated"),
        ("maximum_query_state_gap", 2e-6, "query_full_state_within_tolerance"),
        ("query_pixels_exact", False, "query_pixels_exact"),
        ("query_actions_exact", False, "query_actions_exact"),
        ("history_effect_present", False, "history_reveals_hidden_rule"),
        (
            "true_future_effect_present",
            False,
            "real_future_depends_on_hidden_rule",
        ),
        (
            "x0_static_leakage_check_passed",
            False,
            "x0_does_not_reveal_hidden_label",
        ),
    ],
)
def test_each_causal_failure_is_a_hard_gate(
    field: str, value, failed_check: str
) -> None:
    result = _valid(**{field: value})
    assert result["passed"] is False
    assert result["checks"][failed_check] is False


def test_solver_cache_must_be_checked_when_required() -> None:
    with pytest.raises(ValueError):
        _valid(solver_cache_check_required=True)
    assert _valid(
        solver_cache_check_required=True,
        solver_cache_check_passed=True,
    )["passed"]


def test_different_x0_is_allowed_only_with_leakage_control() -> None:
    assert _valid(
        x0_policy="balanced_visible_start",
        x0_static_leakage_check_passed=True,
    )["passed"]
    assert not _valid(
        x0_policy="balanced_visible_start",
        x0_static_leakage_check_passed=False,
    )["passed"]
