"""Versioned, fail-closed reference-gate decisions.

The individual benchmark scorers predate one shared admission contract.  This
module is deliberately small and data-oriented: it consumes their retained
result envelopes (or the decision-free Development metrics envelopes) and
emits a separate ``reference_decision`` receipt.  Existing ``gate``,
``decision`` and score fields are never rewritten by this module.

The receipt has one important property: a headline score is never a substitute
for a gate.  A Test result for speed, Action Delay, or Door must carry the
additive gate-completion block; a missing block is a rejection.  The six
already-gated components must carry their original gate evidence, including
latent-response and bootstrap evidence where the release requires it.
"""

from __future__ import annotations

import hashlib
import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from contextworld.paths import repository_root


REFERENCE_DECISION_SCHEMA_VERSION = 1
REFERENCE_CONTRACT_ID = "contextworld_reference_decision_v2"
REFERENCE_CONTRACT_VERSION = REFERENCE_CONTRACT_ID

COMPONENTS: tuple[str, ...] = (
    "speed",
    "action_delay",
    "door",
    "action_strength",
    "contact_friction",
    "cube_gripper_carry",
    "motion_damping",
    "portal_exit",
    "robot_arm_mass",
)

COMPONENT_ALIASES: dict[str, str] = {
    "cube_grasp_rule": "cube_gripper_carry",
    "reacher_arm_mass": "robot_arm_mass",
}

GATE_COMPLETION_COMPONENTS = frozenset({"speed", "action_delay", "door"})

_RELEASE_CONFIGS: dict[str, str] = {
    "speed": "configs/benchmark/tworoom_speed_icl_release_v1.yaml",
    "action_delay": "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml",
    "door": "configs/benchmark/tworoom_door_icl_release_v1.yaml",
    "action_strength": "configs/benchmark/pusht_action_strength_icl_release_v1.yaml",
    "contact_friction": "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml",
    "cube_gripper_carry": "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
    "motion_damping": "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
    "portal_exit": "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml",
    "robot_arm_mass": "configs/benchmark/reacher_arm_mass_icl_release_v1.yaml",
}

_GATE_COMPLETION_CONFIG = (
    "configs/benchmark/contextworld_test_gate_completion_v1.yaml"
)

# The Door release is a fixed six-seed x two-direction matrix.  Keep these
# members explicit in the independent decision reader: accepting whatever
# rule/cell keys happen to be present would allow an incomplete summary to
# pass while still carrying plausible aggregate values.
_DOOR_TRUE_RULES: tuple[str, ...] = ("passable", "blocked")
_DOOR_EVAL_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46, 47)
_DOOR_DIRECTIONS: tuple[str, ...] = ("left_to_right", "right_to_left")
_DOOR_REQUIRED_CELLS: tuple[str, ...] = tuple(
    f"s{seed}/{direction}"
    for seed in _DOOR_EVAL_SEEDS
    for direction in _DOOR_DIRECTIONS
)
_DOOR_REQUIRED_BOOTSTRAP_METRICS: tuple[str, ...] = (
    "passable/same_vs_other_rule_history",
    "blocked/same_vs_other_rule_history",
    "passable/matching_history_two_target_margin",
    "blocked/matching_history_two_target_margin",
)


def canonical_component(component: str) -> str:
    """Normalize historical scorer names to the nine freeze component IDs."""

    value = str(component)
    return COMPONENT_ALIASES.get(value, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(path: Path | str, *, root: Path | None = None) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = (root or repository_root()) / value
    return value.expanduser().resolve()


def reference_contract_identity(
    *,
    repo_root: Path | None = None,
    source_paths: Mapping[str, Path | str] | None = None,
) -> dict[str, Any]:
    """Return the immutable identity inputs used by a reference decision.

    The returned object is suitable for embedding in a freeze receipt.  It
    includes the contract ID, every release threshold source, the additive
    gate-completion source, and hashes of the decision implementation files.
    ``source_paths`` is an optional extension for callers that want to bind
    their own scorer/adapter sources as well.
    """

    root = (repo_root or repository_root()).resolve()
    release_sources: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        relative = _RELEASE_CONFIGS[component]
        path = _repo_path(relative, root=root)
        release_sources[component] = {
            "path": relative,
            "sha256": _sha256(path),
        }
    completion_path = _repo_path(_GATE_COMPLETION_CONFIG, root=root)
    implementation_paths = {
        "reference_decision": "contextworld/benchmarks/reference_decision.py",
        "gate_completion": "contextworld/benchmarks/test_gate_completion.py",
        "speed_scorer": "contextworld/benchmarks/speed_icl_score.py",
        "action_delay_scorer": "contextworld/benchmarks/action_delay_icl_score.py",
        "door_scorer": "contextworld/benchmarks/door_icl_score.py",
        "paired_latent_response": "contextworld/benchmarks/paired_latent_response.py",
        "causal_data_contract": "contextworld/benchmarks/causal_data_contract.py",
        "action_strength_scorer": "contextworld/benchmarks/action_strength_icl_score.py",
        "contact_friction_scorer": "contextworld/benchmarks/contact_friction_icl_score.py",
        "cube_gripper_carry_scorer": "contextworld/benchmarks/cube_grasp_rule_v4r1_icl_score.py",
        "motion_damping_scorer": "contextworld/benchmarks/motion_damping_icl_score.py",
        "portal_exit_scorer": "contextworld/benchmarks/portal_exit_icl_score.py",
        "robot_arm_mass_scorer": "contextworld/benchmarks/reacher_arm_mass_icl_score.py",
    }
    if source_paths:
        implementation_paths.update(
            {str(name): str(path) for name, path in source_paths.items()}
        )
    implementation_sources = {
        name: {
            "path": str(path),
            "sha256": _sha256(_repo_path(path, root=root)),
        }
        for name, path in sorted(implementation_paths.items())
    }
    sources = {
        **release_sources,
        "gate_completion_config": {
            "path": _GATE_COMPLETION_CONFIG,
            "sha256": _sha256(completion_path),
        },
        **implementation_sources,
    }
    return {
        "contract_id": REFERENCE_CONTRACT_ID,
        "contract_version": REFERENCE_CONTRACT_VERSION,
        "schema_version": REFERENCE_DECISION_SCHEMA_VERSION,
        "components": list(COMPONENTS),
        "gate_completion_components": sorted(GATE_COMPLETION_COMPONENTS),
        "release_configs": release_sources,
        "gate_completion_config": {
            "path": _GATE_COMPLETION_CONFIG,
            "sha256": _sha256(completion_path),
        },
        "implementation_sources": implementation_sources,
        # ``sources`` and ``source_hashes`` are intentionally redundant
        # convenience views for freeze builders that only need a flat source
        # identity without knowing the threshold/config grouping.
        "sources": sources,
        "source_hashes": {
            name: str(value["sha256"]) for name, value in sources.items()
        },
    }


def _load_yaml(path: Path | str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Release config must be a mapping: {path}")
    return payload


def _release_for(
    component: str,
    *,
    release: Mapping[str, Any] | None,
    release_config: Path | str | None,
    repo_root: Path | None,
) -> dict[str, Any] | None:
    component = canonical_component(component)
    if release is not None:
        return dict(release)
    path = release_config or _RELEASE_CONFIGS.get(component)
    if path is None:
        return None
    return _load_yaml(_repo_path(path, root=repo_root))


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _bool_gate_block(value: Any, *, name: str) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate an already-computed gate block without trusting its summary."""

    reasons: list[str] = []
    if not isinstance(value, Mapping):
        return False, {"present": False}, [f"missing_{name}"]
    checks = value.get("checks")
    passed = value.get("passed")
    if not isinstance(checks, Mapping) or not checks:
        reasons.append(f"{name}_checks_missing")
        checks_out: dict[str, Any] = {}
    else:
        checks_out = {str(key): bool(item) for key, item in checks.items()}
        if any(not isinstance(item, bool) for item in checks.values()):
            reasons.append(f"{name}_checks_not_boolean")
    if not isinstance(passed, bool):
        reasons.append(f"{name}_passed_missing")
    elif checks_out and passed != all(checks_out.values()):
        reasons.append(f"{name}_passed_inconsistent")
    ok = not reasons and bool(passed)
    return ok, {
        "present": True,
        "checks": checks_out,
        "passed": bool(passed) if isinstance(passed, bool) else None,
    }, reasons


def _latent_response_checks(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[dict[str, bool], list[str]]:
    response = metrics.get("latent_response")
    if not isinstance(response, Mapping):
        return {
            "target_latent_separation": False,
            "response_gain": False,
            "normalized_response_error": False,
        }, ["latent_response_missing"]
    separation = response.get("target_latent_separation")
    separation_passed = False
    if isinstance(separation, Mapping):
        # Development envelopes intentionally remove ``passed``.  The paired
        # response kernel's actual criterion is finite, non-identical target
        # pairs; reconstruct it from retained numeric evidence whenever it is
        # available.  This also prevents a stale Test boolean from replacing
        # the latent evidence.
        zero_pairs = separation.get("zero_separation_pair_count")
        minimum_mse = separation.get("minimum_target_response_mse")
        if zero_pairs is not None or minimum_mse is not None:
            separation_passed = bool(
                zero_pairs == 0
                and _finite_number(minimum_mse)
                and float(minimum_mse) > 0.0
            )
        else:
            separation_passed = separation.get("passed") is True
    gain = response.get("response_gain")
    error = response.get("normalized_response_error")
    gain_ok = _finite_number(gain) and float(gain) >= float(
        thresholds.get("response_gain_minimum", math.inf)
    )
    error_ok = _finite_number(error) and float(error) < float(
        thresholds.get("normalized_response_error_strict_maximum", -math.inf)
    )
    checks = {
        "target_latent_separation": bool(
            thresholds.get("target_latent_separation_required") is True
            and separation_passed
        ),
        "response_gain": bool(gain_ok),
        "normalized_response_error": bool(error_ok),
    }
    reasons = [f"gate_completion_{name}" for name, ok in checks.items() if not ok]
    return checks, reasons


def gate_completion_decision(
    block: Mapping[str, Any] | None,
    *,
    component: str,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the additive gate decision from metrics and bootstrap values.

    ``block`` may be a full Test gate-completion result or a decision-free
    Development ``gate_completion_inputs`` envelope.  Stored ``passed`` and
    ``checks`` fields are treated as audit evidence only; the returned
    decision is recomputed from the metrics and the frozen thresholds.
    """

    if component not in GATE_COMPLETION_COMPONENTS:
        return {
            "required": False,
            "present": False,
            "passed": True,
            "checks": {},
            "reason_codes": [],
        }
    if not isinstance(block, Mapping):
        return {
            "required": True,
            "present": False,
            "passed": False,
            "checks": {},
            "reason_codes": ["gate_completion_missing"],
        }
    if config is None:
        config_path = _repo_path(_GATE_COMPLETION_CONFIG)
        config = _load_yaml(config_path)
    section = config.get(component)
    if not isinstance(section, Mapping):
        return {
            "required": True,
            "present": False,
            "passed": False,
            "checks": {},
            "reason_codes": ["gate_completion_config_missing"],
        }
    gates = section.get("gates")
    bootstrap = section.get("bootstrap")
    metrics = block.get("metrics")
    uncertainty = block.get("uncertainty")
    if not isinstance(gates, Mapping) or not isinstance(bootstrap, Mapping):
        return {
            "required": True,
            "present": True,
            "passed": False,
            "checks": {},
            "reason_codes": ["gate_completion_thresholds_missing"],
        }
    reasons: list[str] = []
    if not isinstance(metrics, Mapping):
        metrics = {}
        reasons.append("gate_completion_metrics_missing")
    if not isinstance(uncertainty, Mapping):
        uncertainty = {}
        reasons.append("gate_completion_uncertainty_missing")
    checks: dict[str, bool] = {}
    scalar_metric_thresholds = {
        "correct_history_rate": "correct_history_rate_minimum",
        "context_switch_rate": "context_switch_rate_minimum",
        "worst_reference_speed_correct_history_rate": (
            "worst_reference_speed_correct_history_rate_minimum"
        ),
        "worst_rule_correct_target_choice_rate": (
            "worst_rule_correct_target_choice_rate_minimum"
        ),
    }
    for metric_name, threshold_name in scalar_metric_thresholds.items():
        if threshold_name not in gates:
            continue
        observed = metrics.get(metric_name)
        checks[metric_name] = bool(
            _finite_number(observed)
            and float(observed) >= float(gates[threshold_name])
        )
        if not checks[metric_name]:
            reasons.append(f"gate_completion_{metric_name}")
    response_checks, response_reasons = _latent_response_checks(metrics, gates)
    checks.update(response_checks)
    reasons.extend(response_reasons)
    lower_bounds = uncertainty.get("lower_bounds")
    if not isinstance(lower_bounds, Mapping):
        lower_bounds = {}
    lower_minimum = bootstrap.get("lower_bound_minimum")
    if not isinstance(lower_minimum, Mapping):
        lower_minimum = {}
    for metric_name, minimum in lower_minimum.items():
        observed = lower_bounds.get(metric_name)
        check_name = f"{metric_name}_bootstrap_lower_bound"
        checks[check_name] = bool(
            _finite_number(observed) and float(observed) >= float(minimum)
        )
        if not checks[check_name]:
            reasons.append(f"gate_completion_{check_name}")
    # A full Test block must expose its own checked result.  A Development
    # block intentionally does not, which is why ``stored_passed`` is only an
    # audit field and never required here.
    stored_passed = block.get("passed")
    if stored_passed is not None and not isinstance(stored_passed, bool):
        reasons.append("gate_completion_passed_not_boolean")
    passed = bool(checks) and all(checks.values()) and not reasons
    if isinstance(stored_passed, bool) and stored_passed != passed:
        reasons.append("gate_completion_passed_inconsistent")
        passed = False
    return {
        "required": True,
        "present": True,
        "passed": passed,
        "checks": checks,
        "stored_passed": stored_passed,
        "reason_codes": sorted(set(reasons)),
        "threshold_source": block.get("threshold_source"),
    }


def _hidden_future_gate(
    metrics: Mapping[str, Any] | None,
    release: Mapping[str, Any] | None,
    stored_gate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the six standard hidden-future gates from all evidence."""

    if not isinstance(metrics, Mapping):
        return {
            "present": isinstance(stored_gate, Mapping),
            "passed": False,
            "checks": {},
            "reason_codes": ["original_metrics_missing"],
        }
    scoring = release.get("scoring", {}) if isinstance(release, Mapping) else {}
    hidden = scoring.get("hidden_future_prediction", {})
    thresholds = hidden.get("gates", {}) if isinstance(hidden, Mapping) else {}
    if not isinstance(thresholds, Mapping) or not thresholds:
        return {
            "present": isinstance(stored_gate, Mapping),
            "passed": False,
            "checks": {},
            "reason_codes": ["original_thresholds_missing"],
        }
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    for threshold_name, minimum in thresholds.items():
        if threshold_name in {
            "target_latent_separation_required",
            "response_gain_minimum",
            "normalized_response_error_strict_maximum",
            "bootstrap_lower_bound_minimum",
        }:
            continue
        if not threshold_name.endswith("_minimum"):
            continue
        metric_name = threshold_name[: -len("_minimum")]
        observed = metrics.get(metric_name)
        check_name = metric_name
        checks[check_name] = bool(
            _finite_number(observed) and float(observed) >= float(minimum)
        )
        if not checks[check_name]:
            reasons.append(f"original_{check_name}")
    response_checks, response_reasons = _latent_response_checks(metrics, thresholds)
    checks.update(response_checks)
    reasons.extend(f"original_{value}" for value in response_reasons)

    uncertainty = metrics.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        uncertainty = {}
    lower_bounds = uncertainty.get("lower_bounds")
    if not isinstance(lower_bounds, Mapping):
        lower_bounds = metrics.get("paired_bootstrap_95_lower_bound", {})
    if not isinstance(lower_bounds, Mapping):
        lower_bounds = {}
    lower_minimum: Mapping[str, Any] = {}
    release_uncertainty = hidden.get("uncertainty", {})
    if isinstance(release_uncertainty, Mapping) and isinstance(
        release_uncertainty.get("lower_bound_minimum"), Mapping
    ):
        lower_minimum = release_uncertainty["lower_bound_minimum"]
    if isinstance(thresholds.get("bootstrap_lower_bound_minimum"), Mapping):
        lower_minimum = {
            **dict(lower_minimum),
            **dict(thresholds["bootstrap_lower_bound_minimum"]),
        }
    for metric_name, minimum in lower_minimum.items():
        observed = lower_bounds.get(metric_name)
        check_name = f"{metric_name}_bootstrap_lower_bound"
        checks[check_name] = bool(
            _finite_number(observed) and float(observed) >= float(minimum)
        )
        if not checks[check_name]:
            reasons.append(f"original_{check_name}")
    if not checks:
        reasons.append("original_checks_missing")

    # An explicitly stored gate must agree with the recomputation when it is
    # available.  This catches stale or hand-edited result envelopes while
    # allowing decision-free Development inputs to omit the old field.
    if isinstance(stored_gate, Mapping):
        stored_passed = stored_gate.get("passed")
        if stored_passed is not None and not isinstance(stored_passed, bool):
            reasons.append("original_passed_not_boolean")
        if isinstance(stored_passed, bool) and stored_passed != all(checks.values()):
            reasons.append("original_passed_inconsistent")
    passed = bool(checks) and all(checks.values()) and not reasons
    return {
        "present": isinstance(stored_gate, Mapping) or bool(metrics),
        "passed": passed,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
    }


def _finite_measure(value: Any) -> bool:
    """Return whether ``value`` is a real finite metric value.

    ``float(True)`` is finite, but a boolean is never valid evidence for a
    Door numeric criterion.  Keep this stricter helper local to the Door
    contract so historical non-Door readers retain their existing behavior.
    """

    return not isinstance(value, bool) and _finite_number(value)


def _door_legacy_diagnostics(
    stored_decision: Mapping[str, Any] | None,
    *,
    numeric_passed: bool,
) -> dict[str, Any]:
    """Expose old summary flags as audit information only.

    The old result decision is deliberately never used to make the new gate
    pass.  Keeping its internal consistency and agreement with the numeric
    result in the receipt makes stale envelopes visible without allowing them
    to override retained metrics.
    """

    if not isinstance(stored_decision, Mapping):
        return {"present": False}
    stored_passed = stored_decision.get("passed")
    stored_checks = stored_decision.get("checks")
    checks_are_boolean = isinstance(stored_checks, Mapping) and bool(
        stored_checks
    ) and all(isinstance(value, bool) for value in stored_checks.values())
    checks_passed = (
        bool(checks_are_boolean and all(stored_checks.values()))
        if isinstance(stored_checks, Mapping)
        else None
    )
    return {
        "present": True,
        "stored_passed": stored_passed
        if isinstance(stored_passed, bool)
        else None,
        "stored_checks": (
            {str(key): value for key, value in stored_checks.items()}
            if isinstance(stored_checks, Mapping)
            else None
        ),
        "checks_are_boolean": bool(checks_are_boolean),
        "checks_passed": checks_passed,
        "stored_self_consistent": bool(
            isinstance(stored_passed, bool)
            and checks_passed is not None
            and stored_passed == checks_passed
        ),
        "agrees_with_numeric": bool(
            isinstance(stored_passed, bool)
            and stored_passed == numeric_passed
        ),
    }


def _door_numeric_original_gate(
    metrics: Mapping[str, Any],
    release: Mapping[str, Any] | None,
    *,
    stored_decision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the original Door rule-switch gate from numeric evidence.

    This is shared by Public Test summaries and decision-free Development
    metrics.  It mirrors the ``informative_history_rule_switch_v2`` contract
    in ``hidden_passage_validation`` while requiring the complete fixed
    matrix, so aggregate flags or a partial set of cells cannot substitute for
    the original evidence.
    """

    if not isinstance(metrics, Mapping) or not metrics:
        return {
            "present": False,
            "passed": False,
            "checks": {},
            "reason_codes": ["original_metrics_missing"],
            "legacy_diagnostics": _door_legacy_diagnostics(
                stored_decision, numeric_passed=False
            ),
        }

    scoring = release.get("scoring", {}) if isinstance(release, Mapping) else {}
    gates = scoring.get("gates", {}) if isinstance(scoring, Mapping) else {}
    if not isinstance(gates, Mapping):
        gates = {}
    min_accuracy = gates.get(
        "minimum_same_history_two_target_accuracy_exclusive"
    )
    min_win = gates.get(
        "minimum_matching_vs_opposite_history_win_rate_exclusive"
    )
    min_separation = gates.get("minimum_target_pair_latent_mse_exclusive")
    bootstrap_config = gates.get("paired_bootstrap")
    if not isinstance(bootstrap_config, Mapping):
        bootstrap_config = {}
    bootstrap_lower_minimum = bootstrap_config.get(
        "minimum_lower_bound_exclusive"
    )

    reasons: list[str] = []
    for name, value in (
        ("accuracy_threshold", min_accuracy),
        ("history_win_threshold", min_win),
        ("separation_threshold", min_separation),
        ("bootstrap_lower_threshold", bootstrap_lower_minimum),
    ):
        if not _finite_measure(value):
            reasons.append(f"original_{name}_missing_or_nonfinite")

    checks: dict[str, bool] = {}
    rule_evidence: dict[str, Any] = {}
    rules = metrics.get("by_true_rule")
    expected_rules = set(_DOOR_TRUE_RULES)
    if not isinstance(rules, Mapping):
        reasons.append("original_rule_metrics_missing")
        rules = {}
    elif {str(key) for key in rules} != expected_rules:
        observed_rule_names = {str(key) for key in rules}
        missing = sorted(expected_rules - observed_rule_names)
        extra = sorted(observed_rule_names - expected_rules)
        if missing:
            reasons.append("original_rule_missing")
        if extra:
            reasons.append("original_rule_unexpected")

    overall_advantage: dict[str, bool] = {}
    overall_accuracy: dict[str, bool] = {}
    overall_history_win: dict[str, bool] = {}
    cell_advantage: dict[str, bool] = {}
    cell_accuracy: dict[str, bool] = {}
    for rule in _DOOR_TRUE_RULES:
        row = rules.get(rule)
        if not isinstance(row, Mapping):
            reasons.append(f"original_rule_{rule}_metrics_missing")
            rule_evidence[rule] = {"present": False}
            overall_advantage[rule] = False
            overall_accuracy[rule] = False
            overall_history_win[rule] = False
            for cell_name in _DOOR_REQUIRED_CELLS:
                cell_advantage[f"{rule}/{cell_name}"] = False
                cell_accuracy[f"{rule}/{cell_name}"] = False
            continue

        overall = row.get("overall")
        cells = row.get("by_eval_seed_and_direction")
        if not isinstance(overall, Mapping):
            reasons.append(f"original_rule_{rule}_overall_missing")
            overall = {}
        if not isinstance(cells, Mapping):
            reasons.append(f"original_rule_{rule}_cells_missing")
            cells = {}
        elif {str(key) for key in cells} != set(_DOOR_REQUIRED_CELLS):
            observed_cell_names = {str(key) for key in cells}
            missing_cells = set(_DOOR_REQUIRED_CELLS) - observed_cell_names
            extra_cells = observed_cell_names - set(_DOOR_REQUIRED_CELLS)
            if missing_cells:
                reasons.append(f"original_rule_{rule}_cell_missing")
            if extra_cells:
                reasons.append(f"original_rule_{rule}_cell_unexpected")

        overall_adv = overall.get("paired_advantage")
        overall_adv = (
            overall_adv.get("same_vs_other_rule_history")
            if isinstance(overall_adv, Mapping)
            else None
        )
        overall_accuracy_value = overall.get("same_history_two_target_accuracy")
        overall_win_value = overall.get(
            "matching_vs_opposite_history_win_rate"
        )
        # The history advantage criterion is independent of the latent-MSE
        # separation threshold and retains the original strict ``> 0`` rule.
        overall_advantage[rule] = bool(
            _finite_measure(overall_adv) and float(overall_adv) > 0.0
        )
        overall_accuracy[rule] = bool(
            _finite_measure(overall_accuracy_value)
            and _finite_measure(min_accuracy)
            and float(overall_accuracy_value) > float(min_accuracy)
        )
        overall_history_win[rule] = bool(
            _finite_measure(overall_win_value)
            and _finite_measure(min_win)
            and float(overall_win_value) > float(min_win)
        )
        if not overall_advantage[rule]:
            reasons.append(f"original_rule_{rule}_paired_advantage")
        if not overall_accuracy[rule]:
            reasons.append(f"original_rule_{rule}_target_accuracy")
        if not overall_history_win[rule]:
            reasons.append(f"original_rule_{rule}_history_win")

        evidence_cells: dict[str, Any] = {}
        for cell_name in _DOOR_REQUIRED_CELLS:
            cell = cells.get(cell_name)
            key = f"{rule}/{cell_name}"
            if not isinstance(cell, Mapping):
                reasons.append(f"original_rule_{rule}_{cell_name}_missing")
                cell_advantage[key] = False
                cell_accuracy[key] = False
                evidence_cells[cell_name] = {"present": False}
                continue
            cell_adv = cell.get("paired_advantage")
            cell_adv = (
                cell_adv.get("same_vs_other_rule_history")
                if isinstance(cell_adv, Mapping)
                else None
            )
            cell_accuracy_value = cell.get("same_history_two_target_accuracy")
            cell_advantage[key] = bool(
                _finite_measure(cell_adv) and float(cell_adv) > 0.0
            )
            cell_accuracy[key] = bool(
                _finite_measure(cell_accuracy_value)
                and _finite_measure(min_accuracy)
                and float(cell_accuracy_value) > float(min_accuracy)
            )
            if not cell_advantage[key]:
                reasons.append(
                    f"original_rule_{rule}_{cell_name}_paired_advantage"
                )
            if not cell_accuracy[key]:
                reasons.append(
                    f"original_rule_{rule}_{cell_name}_target_accuracy"
                )
            evidence_cells[cell_name] = {
                "paired_advantage": cell_advantage[key],
                "target_accuracy": cell_accuracy[key],
            }
        rule_evidence[rule] = {
            "present": True,
            "overall": {
                "paired_advantage": overall_advantage[rule],
                "target_accuracy": overall_accuracy[rule],
                "history_win": overall_history_win[rule],
            },
            "cells": evidence_cells,
        }

    checks[
        "matching_history_beats_opposite_history_for_each_true_rule"
    ] = bool(overall_advantage) and all(overall_advantage.values())
    checks[
        "matching_history_beats_opposite_history_in_every_seed_direction_cell"
    ] = bool(cell_advantage) and all(cell_advantage.values())
    checks[
        "matching_history_target_accuracy_above_threshold_for_each_rule"
    ] = bool(overall_accuracy) and all(overall_accuracy.values())
    checks[
        "matching_history_target_accuracy_above_threshold_in_every_seed_direction_cell"
    ] = bool(cell_accuracy) and all(cell_accuracy.values())
    checks[
        "matching_history_beats_opposite_history_on_majority_queries_for_each_rule"
    ] = bool(overall_history_win) and all(overall_history_win.values())

    bootstrap = metrics.get("paired_static_query_bootstrap")
    bootstrap_metrics = (
        bootstrap.get("metrics") if isinstance(bootstrap, Mapping) else None
    )
    bootstrap_checks: dict[str, bool] = {}
    if not isinstance(bootstrap, Mapping):
        reasons.append("original_bootstrap_missing")
    if not isinstance(bootstrap_metrics, Mapping):
        reasons.append("original_bootstrap_metrics_missing")
        bootstrap_metrics = {}
    for metric_name in _DOOR_REQUIRED_BOOTSTRAP_METRICS:
        item = bootstrap_metrics.get(metric_name)
        lower = item.get("lower") if isinstance(item, Mapping) else None
        bootstrap_checks[metric_name] = bool(
            _finite_measure(lower)
            and _finite_measure(bootstrap_lower_minimum)
            and float(lower) > float(bootstrap_lower_minimum)
        )
        if not bootstrap_checks[metric_name]:
            reasons.append(
                "original_bootstrap_"
                + metric_name.replace("/", "_").replace("-", "_")
            )
    checks["required_bootstrap_lower_bounds_above_threshold"] = bool(
        bootstrap_checks and all(bootstrap_checks.values())
    )

    separation = metrics.get("target_latent_separation")
    observed_separation = (
        separation.get("minimum_mse")
        if isinstance(separation, Mapping)
        else None
    )
    separation_check = bool(
        _finite_measure(observed_separation)
        and _finite_measure(min_separation)
        and float(observed_separation) > float(min_separation)
    )
    if not separation_check:
        reasons.append("original_target_latent_separation")
    checks["target_latents_are_separated_for_every_query"] = separation_check

    passed = bool(checks) and all(checks.values()) and not reasons
    return {
        "present": True,
        "passed": passed,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
        "numeric_evidence": {
            "required_rules": list(_DOOR_TRUE_RULES),
            "required_cells": list(_DOOR_REQUIRED_CELLS),
            "rules": rule_evidence,
            "bootstrap": bootstrap_checks,
            "target_latent_separation": separation_check,
        },
        "legacy_diagnostics": _door_legacy_diagnostics(
            stored_decision, numeric_passed=passed
        ),
    }


def _door_original_gate(
    result: Mapping[str, Any], release: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute the Test Door gate from its retained numeric summary."""

    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return _door_numeric_original_gate(
            {}, release, stored_decision=None
        )
    decision = summary.get("decision")
    return _door_numeric_original_gate(
        summary,
        release,
        stored_decision=decision if isinstance(decision, Mapping) else None,
    )


def _door_development_original_gate(
    metrics: Mapping[str, Any], release: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Recompute the Development Door gate from decision-free metrics."""

    return _door_numeric_original_gate(metrics, release)


def _speed_single_decision(
    result: Mapping[str, Any],
    *,
    split: str,
    gate_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tracks = result.get("tracks")
    if not isinstance(tracks, Mapping):
        metrics = result.get("metrics")
        if isinstance(metrics, Mapping):
            tracks = metrics.get("tracks")
    if not isinstance(tracks, Mapping):
        return _rejected("speed_tracks_missing")
    required_tracks = (
        "seen_for_multi",
        "unseen_interpolation",
        "extrapolation_low",
        "extrapolation_high",
    )
    per_track: dict[str, Any] = {}
    reasons: list[str] = []
    for track in required_tracks:
        value = tracks.get(track)
        if not isinstance(value, Mapping):
            per_track[track] = _rejected("track_missing")
            reasons.append(f"track_{track}_missing")
            continue
        horizons = value.get("horizons")
        h1 = horizons.get("1") if isinstance(horizons, Mapping) else None
        if not isinstance(h1, Mapping):
            per_track[track] = _rejected("horizon_1_missing")
            reasons.append(f"track_{track}_horizon_1_missing")
            continue
        if "formal_within_checkpoint_pass" in h1:
            original = bool(h1.get("formal_within_checkpoint_pass") is True)
        else:
            # Decision-free Development metrics retain the diagnostic boolean
            # and per-reference-speed evidence, while deliberately stripping
            # the old pass field.
            by_speed = h1.get("by_reference_speed")
            original = bool(
                h1.get("strict_each_alternative_diagnostic") is True
                and isinstance(by_speed, Mapping)
                and all(
                    row.get("all_eval_seed_directions_positive") is True
                    and _finite_number(row.get("matching_history_advantage"))
                    and float(row["matching_history_advantage"]) > 0.0
                    for row in by_speed.values()
                    if isinstance(row, Mapping)
                )
                and isinstance(by_speed, Mapping)
                and bool(by_speed)
            )
        original_block = {
            "present": (
                "formal_within_checkpoint_pass" in h1
                or "strict_each_alternative_diagnostic" in h1
            ),
            "passed": original,
            "checks": {"formal_within_checkpoint_pass": original},
            "reason_codes": [] if original else ["original_track_gate_failed"],
        }
        completion = gate_completion_decision(
            _completion_block(value),
            component="speed",
            config=gate_config,
        )
        passed = original_block["passed"] and completion["passed"]
        if not passed:
            reasons.append(f"track_{track}_rejected")
        per_track[track] = {
            "original_gate": original_block,
            "gate_completion": completion,
            "passed": passed,
        }
    primary = tracks.get("unseen_interpolation", {})
    primary_h1 = (
        primary.get("horizons", {}).get("1", {})
        if isinstance(primary, Mapping)
        else {}
    )
    return {
        "component": "speed",
        "split": split,
        "required_tracks": list(required_tracks),
        "primary_score_track": "unseen_interpolation",
        "primary_score_horizon": 1,
        "primary_score": primary_h1.get(
            "reference_speed_balanced_strict_query_win_rate_vs_every_other"
        ),
        "tracks": per_track,
        "all_tracks_passed": not reasons and all(
            value.get("passed") is True for value in per_track.values()
        ),
        "passed": not reasons and all(
            value.get("passed") is True for value in per_track.values()
        ),
        "reason_codes": sorted(set(reasons)),
    }


def _completion_block(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find a Test block or the decision-free Development input block."""

    direct = result.get("gate_completion") or result.get("gate_completion_inputs")
    if isinstance(direct, Mapping):
        return direct
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        nested = metrics.get("gate_completion_inputs") or metrics.get(
            "gate_completion"
        )
        if isinstance(nested, Mapping):
            return nested
    return None


def _generic_single_decision(
    component: str,
    result: Mapping[str, Any],
    *,
    split: str,
    release: Mapping[str, Any] | None,
    gate_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if component == "door":
        if isinstance(result.get("summary"), Mapping):
            original = _door_original_gate(result, release)
        else:
            metrics = result.get("metrics")
            original = (
                _door_development_original_gate(metrics, release)
                if isinstance(metrics, Mapping)
                else _rejected("original_metrics_missing")
            )
    else:
        metrics = result.get("metrics")
        if component == "action_delay" and not isinstance(metrics, Mapping):
            metrics = result.get("core_h1")
        stored_gate = result.get("gate")
        original = (
            {
                "present": False,
                "passed": True,
                "checks": {},
                "reason_codes": [],
            }
            if component == "action_delay"
            else _hidden_future_gate(metrics, release, stored_gate)
        )
        # Action Delay's formal gate is a legacy physical-group gate.  It is
        # not one of hidden-future's six-gate fields, so preserve it as an
        # additional required original gate while independently validating all
        # of its core metrics and bootstrap lower bound.
        if component == "action_delay":
            legacy = _legacy_action_delay_gate(result, release)
            original = _combine_gate_blocks(original, legacy, "legacy_primary_gate")
    completion = gate_completion_decision(
        _completion_block(result),
        component=component,
        config=gate_config,
    )
    passed = bool(original["passed"] and completion["passed"])
    reasons = list(original.get("reason_codes", [])) + list(
        completion.get("reason_codes", [])
    )
    return {
        "component": component,
        "split": split,
        "original_gate": original,
        "gate_completion": completion,
        "passed": passed,
        "reason_codes": sorted(set(reasons)),
    }


def _legacy_action_delay_gate(
    result: Mapping[str, Any], release: Mapping[str, Any] | None
) -> dict[str, Any]:
    metrics = result.get("core_h1")
    if not isinstance(metrics, Mapping):
        metrics = result.get("metrics")
    if not isinstance(metrics, Mapping):
        return _rejected("legacy_primary_metrics_missing")
    scoring = release.get("scoring", {}) if isinstance(release, Mapping) else {}
    thresholds = scoring.get("primary_gate", {}) if isinstance(scoring, Mapping) else {}
    checks: dict[str, bool] = {}
    mapping = {
        "physical_group_macro_accuracy": "physical_group_macro_accuracy_minimum",
        "minimum_physical_group_accuracy": "minimum_physical_group_accuracy",
    }
    for metric, threshold in mapping.items():
        checks[metric] = bool(
            _finite_number(metrics.get(metric))
            and isinstance(thresholds, Mapping)
            and threshold in thresholds
            and float(metrics[metric]) >= float(thresholds[threshold])
        )
    bootstrap = metrics.get("paired_query_bootstrap_95_percent_interval")
    lower = bootstrap.get("lower") if isinstance(bootstrap, Mapping) else None
    threshold = thresholds.get("paired_query_bootstrap_95_percent_lower_bound_minimum") if isinstance(thresholds, Mapping) else None
    checks["bootstrap_lower_bound"] = bool(
        _finite_number(lower) and _finite_number(threshold)
        and float(lower) >= float(threshold)
    )
    reasons = [f"legacy_{name}" for name, ok in checks.items() if not ok]
    stored_gate = result.get("gate")
    if isinstance(stored_gate, Mapping):
        stored_passed = stored_gate.get("passed")
        if not isinstance(stored_passed, bool):
            reasons.append("legacy_passed_missing")
        elif stored_passed != all(checks.values()):
            reasons.append("legacy_passed_inconsistent")
    return {
        "present": bool(metrics),
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "reason_codes": reasons,
    }


def _combine_gate_blocks(
    left: Mapping[str, Any], right: Mapping[str, Any], label: str
) -> dict[str, Any]:
    checks = {
        **dict(left.get("checks", {})),
        **{f"{label}.{key}": value for key, value in right.get("checks", {}).items()},
    }
    reasons = list(left.get("reason_codes", [])) + list(right.get("reason_codes", []))
    passed = bool(left.get("passed") and right.get("passed"))
    return {
        "present": bool(left.get("present") and right.get("present")),
        "passed": passed,
        "checks": checks,
        "reason_codes": sorted(set(reasons)),
    }


def _rejected(reason: str) -> dict[str, Any]:
    return {
        "passed": False,
        "reason_codes": [reason],
    }


def reference_decision_for_result(
    component: str,
    result: Mapping[str, Any],
    *,
    split: str = "test",
    release: Mapping[str, Any] | None = None,
    release_config: Path | str | None = None,
    gate_completion_config: Mapping[str, Any] | None = None,
    threshold_config: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Build a versioned decision receipt for one component result.

    ``result`` can be a Public Test scorer result or a Development bundle
    metrics envelope.  Development envelopes should use
    ``gate_completion_inputs`` and omit all decision fields; this function
    returns the independent receipt without mutating that envelope.
    """

    component = canonical_component(component)
    if component not in COMPONENTS:
        raise ValueError(f"Unknown reference component: {component}")
    if not isinstance(result, Mapping):
        raise TypeError("Reference result must be a mapping")
    # External Development/Test runners wrap the scorer payload under
    # ``result``.  Accepting that envelope here keeps the freeze builder from
    # having to duplicate schema knowledge.
    nested_result = result.get("result")
    if isinstance(nested_result, Mapping) and not any(
        key in result for key in ("metrics", "tracks", "summary", "gate_completion")
    ):
        result = nested_result
    normalized_split = str(split).lower()
    if normalized_split in {"public_test", "public", "test"}:
        normalized_split = "test"
    elif normalized_split in {"dev", "development"}:
        normalized_split = "development"
    else:
        raise ValueError(f"Unsupported reference split: {split}")
    release_payload = _release_for(
        component,
        release=release,
        release_config=release_config,
        repo_root=repo_root,
    )
    if gate_completion_config is None:
        gate_completion_config = threshold_config
    if component == "speed":
        decision = _speed_single_decision(
            result,
            split=normalized_split,
            gate_config=gate_completion_config,
        )
    else:
        decision = _generic_single_decision(
            component,
            result,
            split=normalized_split,
            release=release_payload,
            gate_config=gate_completion_config,
        )
    identity = reference_contract_identity(repo_root=repo_root)
    return {
        "schema_version": REFERENCE_DECISION_SCHEMA_VERSION,
        "contract_id": REFERENCE_CONTRACT_ID,
        "contract_version": REFERENCE_CONTRACT_VERSION,
        "component": component,
        "split": normalized_split,
        "threshold_sources": identity,
        **decision,
    }


def aggregate_method_decision(
    component: str,
    results: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    split: str = "test",
    required_checkpoints: int = 3,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Apply one component's contract independently to each checkpoint.

    Method aggregation must preserve per-seed evidence.  This helper returns
    a method receipt without pooling latent metrics or replacing any scorer's
    historical aggregate fields.
    """

    component = canonical_component(component)
    if component not in COMPONENTS:
        raise ValueError(f"Unknown reference component: {component}")
    if int(required_checkpoints) != 3:
        raise ValueError(
            "aggregate_method_decision requires exactly three checkpoints"
        )
    rows = list(results)
    receipts = [
        reference_decision_for_result(
            component,
            row,
            split=split,
            repo_root=repo_root,
        )
        for row in rows
    ]
    formal = len(rows) == int(required_checkpoints)
    identity_checks: dict[str, bool] = {
        "required_checkpoint_count": formal,
        "training_seeds_present": False,
        "training_seeds_distinct": False,
        "checkpoint_hashes_present": False,
        "checkpoint_hashes_distinct": False,
        "training_recipe_consistent": False,
        "adapter_family_consistent": False,
    }
    reasons: list[str] = []
    # The public report API normally passes the inner result mapping, while
    # freeze and audit callers may retain the outer ``{"result": ...}``
    # envelope.  Identity must be read from the same payload that the single
    # decision function evaluates in either form.
    payloads: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        nested = row.get("result")
        if isinstance(nested, Mapping) and not any(
            key in row for key in ("metrics", "tracks", "summary", "gate_completion")
        ):
            payloads.append(nested)
        else:
            payloads.append(row)
    models = [row.get("model") for row in payloads]
    model_rows = [model for model in models if isinstance(model, Mapping)]
    seeds = [model.get("training_seed") for model in model_rows]
    identity_checks["training_seeds_present"] = bool(
        len(seeds) == len(rows)
        and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
    )
    identity_checks["training_seeds_distinct"] = bool(
        identity_checks["training_seeds_present"] and len(set(seeds)) == len(seeds)
    )
    checkpoint_hashes: list[str] = []
    adapter_families: list[str] = []
    recipes: list[str] = []
    for model in model_rows:
        adapter = model.get("adapter")
        adapter = adapter if isinstance(adapter, Mapping) else {}
        checkpoint = adapter.get("checkpoint_sha256", model.get("checkpoint_sha256"))
        if isinstance(checkpoint, str) and len(checkpoint) == 64 and all(
            character in "0123456789abcdef" for character in checkpoint.lower()
        ):
            checkpoint_hashes.append(checkpoint.lower())
        adapter_id = (
            adapter.get("adapter_family")
            or adapter.get("adapter_id")
            or adapter.get("adapter_class")
            or model.get("adapter_family")
        )
        if isinstance(adapter_id, str) and adapter_id:
            adapter_families.append(adapter_id)
        recipe = model.get("training_recipe")
        if isinstance(recipe, str) and recipe:
            recipes.append(recipe)
    identity_checks["checkpoint_hashes_present"] = len(checkpoint_hashes) == len(rows)
    identity_checks["checkpoint_hashes_distinct"] = bool(
        identity_checks["checkpoint_hashes_present"]
        and len(set(checkpoint_hashes)) == len(checkpoint_hashes)
    )
    identity_checks["training_recipe_consistent"] = bool(
        len(recipes) == len(rows) and len(set(recipes)) == 1
    )
    identity_checks["adapter_family_consistent"] = bool(
        len(adapter_families) == len(rows) and len(set(adapter_families)) == 1
    )
    if not formal:
        reasons.append("wrong_checkpoint_count")
    for name, ok in identity_checks.items():
        if not ok:
            reasons.append(f"{name}_failed")
    identity_passed = bool(formal and all(identity_checks.values()))
    passed = bool(identity_passed and receipts and all(row["passed"] for row in receipts))
    if identity_passed and not all(row["passed"] for row in receipts):
        reasons.append("checkpoint_gate_failed")
    return {
        "schema_version": REFERENCE_DECISION_SCHEMA_VERSION,
        "contract_id": REFERENCE_CONTRACT_ID,
        "contract_version": REFERENCE_CONTRACT_VERSION,
        "component": str(component),
        "split": str(split).lower(),
        "required_checkpoints": int(required_checkpoints),
        "checkpoint_count": len(rows),
        "checkpoints": receipts,
        "identity": {
            **identity_checks,
            "training_seeds": seeds,
            "checkpoint_sha256": checkpoint_hashes,
            "training_recipes": recipes,
            "adapter_families": adapter_families,
        },
        "passed": passed,
        "status": "passed" if passed else "rejected",
        "reason_codes": sorted(set(reasons)),
        "threshold_sources": reference_contract_identity(repo_root=repo_root),
    }


__all__ = [
    "COMPONENTS",
    "COMPONENT_ALIASES",
    "GATE_COMPLETION_COMPONENTS",
    "REFERENCE_CONTRACT_ID",
    "REFERENCE_CONTRACT_VERSION",
    "REFERENCE_DECISION_SCHEMA_VERSION",
    "aggregate_method_decision",
    "canonical_component",
    "gate_completion_decision",
    "main",
    "reference_contract_identity",
    "reference_decision_for_result",
]


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m contextworld.benchmarks.reference_decision",
        description="Build a versioned ContextWorld reference-decision receipt.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single", help="evaluate one Development or Public Test result"
    )
    single.add_argument("--component", required=True, choices=COMPONENTS)
    single.add_argument("--input", type=Path, required=True)
    single.add_argument("--split", default="test", choices=("development", "test"))
    single.add_argument("--repo-root", type=Path, default=None)
    single.add_argument("--output", type=Path, default=None)

    method = subparsers.add_parser(
        "method", help="aggregate three independent checkpoint results"
    )
    method.add_argument("--component", required=True, choices=COMPONENTS)
    method.add_argument(
        "--input",
        dest="inputs",
        type=Path,
        action="append",
        required=True,
        help="one result JSON; repeat exactly three times",
    )
    method.add_argument("--split", default="test", choices=("development", "test"))
    method.add_argument("--repo-root", type=Path, default=None)
    method.add_argument("--output", type=Path, default=None)

    contract = subparsers.add_parser(
        "contract", help="emit the contract and source hash identity"
    )
    contract.add_argument("--repo-root", type=Path, default=None)
    contract.add_argument("--output", type=Path, default=None)
    return parser


def _cli_read_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, (Mapping, list)):
        raise ValueError(f"JSON input must be an object or list: {path}")
    return value


def _cli_write_json(value: Mapping[str, Any], output: Path | None) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(serialized, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for one receipt, a three-checkpoint method, or identity."""

    args = _cli_parser().parse_args(argv)
    try:
        if args.command == "contract":
            receipt = reference_contract_identity(repo_root=args.repo_root)
        elif args.command == "single":
            payload = _cli_read_json(args.input)
            if not isinstance(payload, Mapping):
                raise ValueError("single input must contain one JSON object")
            receipt = reference_decision_for_result(
                args.component,
                payload,
                split=args.split,
                repo_root=args.repo_root,
            )
        else:
            rows: list[Mapping[str, Any]] = []
            for path in args.inputs:
                payload = _cli_read_json(path)
                if isinstance(payload, list):
                    rows.extend(
                        item for item in payload if isinstance(item, Mapping)
                    )
                    if len(rows) == 0 or any(
                        not isinstance(item, Mapping) for item in payload
                    ):
                        raise ValueError(
                            f"method list input must contain only JSON objects: {path}"
                        )
                elif isinstance(payload, Mapping):
                    rows.append(payload)
                else:  # pragma: no cover - _cli_read_json guards this
                    raise ValueError(f"method input must be an object: {path}")
            receipt = aggregate_method_decision(
                args.component,
                rows,
                split=args.split,
                repo_root=args.repo_root,
            )
        _cli_write_json(receipt, args.output)
        return 0
    except (OSError, TypeError, ValueError, KeyError) as exc:
        print(f"reference-decision: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through CLI tests
    raise SystemExit(main())
