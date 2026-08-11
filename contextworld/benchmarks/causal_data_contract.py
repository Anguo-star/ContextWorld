from __future__ import annotations

import math
from typing import Any, Iterable


X0_POLICIES = (
    "shared_visible_start",
    "balanced_visible_start",
)


def audit_causal_data_contract(
    *,
    component_id: str,
    evidence_scope: str,
    continuous_environment_trajectory: bool,
    state_installations_after_x0: int,
    query_simulator_recreated: bool,
    maximum_query_state_gap: float,
    query_state_tolerance: float,
    query_pixels_exact: bool,
    query_actions_exact: bool,
    history_effect_present: bool,
    true_future_effect_present: bool,
    x0_policy: str,
    x0_static_leakage_check_passed: bool,
    solver_cache_check_required: bool = False,
    solver_cache_check_passed: bool | None = None,
    evidence: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the common causal-data gate used by every ICL component.

    A paired ICL sample is valid only when both histories are genuine
    simulator trajectories, naturally reach the same query condition, and
    lead to different real futures.  The two trajectories do not have to
    share x0.  When they do not, their visible x0 distribution must not reveal
    the hidden label.
    """

    if x0_policy not in X0_POLICIES:
        raise ValueError(
            f"x0_policy must be one of {X0_POLICIES}, got {x0_policy!r}"
        )
    installations = int(state_installations_after_x0)
    if installations < 0:
        raise ValueError("state_installations_after_x0 cannot be negative")
    gap = float(maximum_query_state_gap)
    tolerance = float(query_state_tolerance)
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError("maximum_query_state_gap must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("query_state_tolerance must be finite and nonnegative")
    if solver_cache_check_required and solver_cache_check_passed is None:
        raise ValueError(
            "solver_cache_check_passed is required when the simulator has "
            "a hidden solver cache"
        )

    checks = {
        "continuous_environment_trajectory": bool(
            continuous_environment_trajectory
        ),
        "no_state_installation_after_x0": installations == 0,
        "query_simulator_not_recreated": not bool(query_simulator_recreated),
        "query_full_state_within_tolerance": gap <= tolerance,
        "query_pixels_exact": bool(query_pixels_exact),
        "query_actions_exact": bool(query_actions_exact),
        "history_reveals_hidden_rule": bool(history_effect_present),
        "real_future_depends_on_hidden_rule": bool(
            true_future_effect_present
        ),
        "x0_does_not_reveal_hidden_label": bool(
            x0_static_leakage_check_passed
        ),
        "solver_cache_does_not_change_real_future": (
            bool(solver_cache_check_passed)
            if solver_cache_check_required
            else True
        ),
    }
    return {
        "schema_version": 1,
        "component_id": str(component_id),
        "evidence_scope": str(evidence_scope),
        "x0_policy": x0_policy,
        "measurements": {
            "state_installations_after_x0": installations,
            "query_simulator_recreated": bool(query_simulator_recreated),
            "maximum_query_state_gap": gap,
            "query_state_tolerance": tolerance,
            "solver_cache_check_required": bool(
                solver_cache_check_required
            ),
        },
        "checks": checks,
        "evidence": [str(value) for value in evidence],
        "passed": all(checks.values()),
    }


__all__ = ["X0_POLICIES", "audit_causal_data_contract"]
