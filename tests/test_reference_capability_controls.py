"""Small scientific controls for the frozen reference capability gates.

These controls are deliberately synthetic and deterministic.  They check that
the released score kernels distinguish a real history-conditioned response
from a history-blind, reversed, or barely nonzero response.  They are useful
regression controls, not an estimate of a population false-positive rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from contextworld.benchmarks.action_delay_icl_score import (
    action_delay_gate_completion_metrics,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.benchmarks.speed_icl_score import (
    speed_gate_completion_metrics,
)
from contextworld.benchmarks.test_gate_completion import (
    load_test_gate_completion_config,
)


ROOT = Path(__file__).resolve().parents[1]
MODES = ("oracle", "no_history", "reversed", "tiny")
SIX_RELEASES = {
    "reacher_arm_mass": "configs/benchmark/reacher_arm_mass_icl_release_v1.yaml",
    "contact_friction": "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml",
    "motion_damping": "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
    "action_strength": "configs/benchmark/pusht_action_strength_icl_release_v1.yaml",
    "portal_exit": "configs/benchmark/tworoom_portal_exit_icl_release_v1.yaml",
    "cube_grasp_rule": "configs/benchmark/cube_gripper_carry_h3_v4r1_icl_release_v1.yaml",
}


def _paired_arrays(mode: str, *, pairs: int = 4) -> tuple[np.ndarray, ...]:
    if mode not in MODES:
        raise ValueError(mode)
    first = np.asarray(
        [[1.0 + i, -2.0 + 0.5 * i, 3.0, 0.25 * i] for i in range(pairs)],
        dtype=np.float64,
    )
    response = np.asarray(
        [[0.5 + 0.1 * i, -0.25, 0.75, 1.0] for i in range(pairs)],
        dtype=np.float64,
    )
    second = first + response
    if mode == "oracle":
        predicted_first, predicted_second = first.copy(), second.copy()
    elif mode == "no_history":
        predicted_first = first.copy()
        predicted_second = first.copy()
    elif mode == "reversed":
        predicted_first = first.copy()
        predicted_second = first - response
    else:
        predicted_first = first.copy()
        predicted_second = first + 0.01 * response
    return predicted_first, predicted_second, first, second


def _six_gate_thresholds() -> dict[str, dict[str, Any]]:
    output = {}
    for name, relative in SIX_RELEASES.items():
        payload = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        output[name] = payload["scoring"]["hidden_future_prediction"]["gates"]
    return output


def _speed_inputs(mode: str) -> tuple[dict[str, dict[float, dict[str, Any]]], list[dict[str, Any]]]:
    speeds = (1.0, 2.0, 3.0)
    conditions = {speed: f"history_{int(speed)}" for speed in speeds}
    groups: dict[str, dict[float, dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    for family in range(2):
        static_id = f"speed-family-{family}"
        targets = {
            speed: np.asarray(
                [1.0 + family, speed, 5.0 + family * speed],
                dtype=np.float64,
            )
            for speed in speeds
        }
        members: dict[float, dict[str, Any]] = {}
        for speed in speeds:
            predictions: dict[str, np.ndarray] = {}
            for condition_speed, condition in conditions.items():
                if mode == "oracle":
                    prediction = targets[condition_speed]
                elif mode == "no_history":
                    prediction = targets[speed]
                elif mode == "reversed":
                    prediction = 2.0 * targets[speed] - targets[condition_speed]
                else:
                    prediction = targets[speed] + 0.01 * (
                        targets[condition_speed] - targets[speed]
                    )
                predictions[condition] = prediction
                records.append(
                    {
                        "query_id": f"{static_id}@{speed:g}",
                        "reference_speed": speed,
                        "condition": condition,
                        "matching_condition": conditions[speed],
                        "latent_mse_by_horizon": {
                            "1": float(np.mean((prediction - targets[speed]) ** 2)),
                        },
                    }
                )
            members[speed] = {
                "query_id": f"{static_id}@{speed:g}",
                "eval_seed": 42 + family,
                "matching_condition": conditions[speed],
                "condition_speeds": {
                    condition: condition_speed
                    for condition_speed, condition in conditions.items()
                },
                "condition_predictions": predictions,
                "target": targets[speed],
            }
        groups[static_id] = members
    return groups, records


def _speed_control(mode: str, config: dict[str, Any]) -> dict[str, Any]:
    groups, records = _speed_inputs(mode)
    result = speed_gate_completion_metrics(
        groups=groups,
        records=records,
        config=config,
        full_protocol=True,
    )
    metrics = result["metrics"]
    return {
        "passed": bool(result["passed"]),
        "checks": {name: bool(value) for name, value in result["checks"].items()},
        "correct_history_rate": float(metrics["correct_history_rate"]),
        "context_switch_rate": float(metrics["context_switch_rate"]),
        "response_gain": float(metrics["latent_response"]["response_gain"]),
        "normalized_response_error": float(
            metrics["latent_response"]["normalized_response_error"]
        ),
    }


def _action_inputs(mode: str, *, queries: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoded = np.asarray(
        [[[min(delay, 5.0)] for delay in range(11)] for _ in range(queries)],
        dtype=np.float64,
    )
    if mode == "oracle":
        predicted = encoded.copy()
        wins = np.ones((queries, 11), dtype=bool)
    elif mode == "no_history":
        predicted = np.zeros_like(encoded)
        wins = np.zeros((queries, 11), dtype=bool)
    elif mode == "reversed":
        predicted = -encoded
        wins = np.zeros((queries, 11), dtype=bool)
    elif mode == "tiny":
        predicted = 0.01 * encoded
        wins = np.ones((queries, 11), dtype=bool)
    else:
        raise ValueError(mode)
    return predicted, encoded, wins


def _action_control(mode: str, config: dict[str, Any]) -> dict[str, Any]:
    predicted, encoded, wins = _action_inputs(mode)
    result = action_delay_gate_completion_metrics(
        predicted_h1=predicted,
        encoded_h1=encoded,
        query_ids=[f"action-query-{i}" for i in range(len(predicted))],
        history_strict_wins=wins,
        config=config,
    )
    metrics = result["metrics"]
    return {
        "passed": bool(result["passed"]),
        "checks": {name: bool(value) for name, value in result["checks"].items()},
        "correct_history_rate": float(metrics["correct_history_rate"]),
        "context_switch_rate": float(metrics["context_switch_rate"]),
        "response_gain": float(metrics["latent_response"]["response_gain"]),
        "normalized_response_error": float(
            metrics["latent_response"]["normalized_response_error"]
        ),
        "delay_pairs": list(metrics["delay_pairs"]),
    }


def _geometry_metrics(mode: str) -> dict[str, Any]:
    predicted_first, predicted_second, target_first, target_second = _paired_arrays(mode)
    metrics, _ = paired_latent_response_metrics(
        pair_ids=[f"pair-{i}" for i in range(len(target_first))],
        predicted_first=predicted_first,
        predicted_second=predicted_second,
        target_first=target_first,
        target_second=target_second,
    )
    response = metrics
    return {
        "response_gain": float(response["response_gain"]),
        "response_alignment": float(response["response_alignment"]),
        "normalized_response_error": float(response["normalized_response_error"]),
        "calibrated_response_success_rate": float(
            response["calibrated_response_success_rate"]
        ),
        "target_latent_separation": response["target_latent_separation"],
    }


def collect_control_evidence() -> dict[str, Any]:
    """Return compact JSON-safe evidence used by the optional verifier script."""

    config = load_test_gate_completion_config()
    paired = {}
    thresholds = _six_gate_thresholds()
    for mode in MODES:
        predicted_first, predicted_second, target_first, target_second = _paired_arrays(mode)
        metrics, _ = paired_latent_response_metrics(
            pair_ids=[f"pair-{i}" for i in range(len(target_first))],
            predicted_first=predicted_first,
            predicted_second=predicted_second,
            target_first=target_first,
            target_second=target_second,
        )
        response_envelope = {"latent_response": metrics}
        paired[mode] = {
            "response_gain": float(metrics["response_gain"]),
            "normalized_response_error": float(metrics["normalized_response_error"]),
            "checks_by_task": {
                name: {
                    key: bool(value)
                    for key, value in paired_latent_response_gate_checks(
                        response_envelope, thresholds=task_thresholds
                    ).items()
                }
                for name, task_thresholds in thresholds.items()
            },
        }
    return {
        "schema_version": 1,
        "scope": {
            "synthetic": True,
            "population_false_positive_rate_estimated": False,
            "common_offset_scope": (
                "response-difference geometry only; absolute prediction MSE and "
                "target-choice metrics are outside this invariance control"
            ),
            "no_history_control_scope": (
                "speed no_history uses a per-query target anchor held constant "
                "across histories; it is a privileged negative control for "
                "history use, not a realizable query-only predictor"
            ),
        },
        "shared_paired_latent": paired,
        "speed": {mode: _speed_control(mode, config) for mode in MODES},
        "action_delay": {mode: _action_control(mode, config) for mode in MODES},
    }


def test_speed_controls_discriminate_oracle_no_history_reversed_and_tiny() -> None:
    config = load_test_gate_completion_config()
    controls = {mode: _speed_control(mode, config) for mode in MODES}
    assert controls["oracle"]["passed"] is True
    assert controls["no_history"]["passed"] is False
    assert controls["reversed"]["passed"] is False
    assert controls["tiny"]["passed"] is False
    assert controls["tiny"]["correct_history_rate"] == pytest.approx(1.0)
    assert controls["tiny"]["context_switch_rate"] == pytest.approx(1.0)
    assert controls["tiny"]["response_gain"] == pytest.approx(0.01)
    assert controls["tiny"]["normalized_response_error"] == pytest.approx(
        0.9801
    )
    assert controls["tiny"]["checks"]["response_gain"] is False


def test_action_delay_controls_discriminate_oracle_no_history_reversed_and_tiny() -> None:
    config = load_test_gate_completion_config()
    controls = {mode: _action_control(mode, config) for mode in MODES}
    assert controls["oracle"]["passed"] is True
    assert controls["no_history"]["passed"] is False
    assert controls["reversed"]["passed"] is False
    assert controls["tiny"]["passed"] is False
    assert controls["tiny"]["context_switch_rate"] == pytest.approx(1.0)
    assert controls["tiny"]["response_gain"] == pytest.approx(0.01)
    assert controls["tiny"]["normalized_response_error"] == pytest.approx(
        0.9801
    )
    assert controls["tiny"]["checks"]["response_gain"] is False
    assert not any(
        pair in controls["oracle"]["delay_pairs"]
        for pair in ("5>6", "5>10", "9>10")
    )


@pytest.mark.parametrize("task", sorted(SIX_RELEASES))
def test_shared_six_paired_tasks_reject_shortcuts(task: str) -> None:
    thresholds = _six_gate_thresholds()[task]
    oracle = _geometry_metrics("oracle")
    no_history = _geometry_metrics("no_history")
    reversed_response = _geometry_metrics("reversed")
    tiny = _geometry_metrics("tiny")
    for metrics in (oracle, no_history, reversed_response, tiny):
        envelope = {"latent_response": metrics}
        checks = paired_latent_response_gate_checks(
            envelope, thresholds=thresholds
        )
        if metrics is oracle:
            assert all(checks.values())
        else:
            assert checks["target_latent_separation"] is True
            assert checks["response_gain"] is False
            assert checks["normalized_response_error"] is (
                metrics["normalized_response_error"] < 1.0
            )


def test_response_geometry_preserves_scale_and_common_offset_scope() -> None:
    predicted_first, predicted_second, target_first, target_second = _paired_arrays(
        "oracle"
    )
    baseline, _ = paired_latent_response_metrics(
        pair_ids=[f"pair-{i}" for i in range(len(target_first))],
        predicted_first=predicted_first,
        predicted_second=predicted_second,
        target_first=target_first,
        target_second=target_second,
    )
    for scale in (1.0e-6, 1.0e6):
        scaled, _ = paired_latent_response_metrics(
            pair_ids=[f"pair-{i}" for i in range(len(target_first))],
            predicted_first=predicted_first * scale,
            predicted_second=predicted_second * scale,
            target_first=target_first * scale,
            target_second=target_second * scale,
        )
        for name in (
            "response_gain",
            "response_alignment",
            "normalized_response_error",
            "calibrated_response_success_rate",
        ):
            assert scaled[name] == pytest.approx(baseline[name], rel=1e-10)

    offset = np.full(predicted_first.shape[-1], 100.0)
    shifted, _ = paired_latent_response_metrics(
        pair_ids=[f"pair-{i}" for i in range(len(target_first))],
        predicted_first=predicted_first + offset,
        predicted_second=predicted_second + offset,
        target_first=target_first,
        target_second=target_second,
    )
    assert shifted["response_gain"] == pytest.approx(baseline["response_gain"])
    assert shifted["normalized_response_error"] == pytest.approx(
        baseline["normalized_response_error"]
    )
    assert np.mean((predicted_first + offset - target_first) ** 2) >= 10_000.0


def test_control_evidence_is_json_safe_and_explicitly_not_fpr_calibration() -> None:
    import json

    evidence = collect_control_evidence()
    encoded = json.dumps(evidence, sort_keys=True)
    assert "population_false_positive_rate_estimated" in encoded
    assert evidence["scope"]["population_false_positive_rate_estimated"] is False
    assert all(
        all(evidence["shared_paired_latent"]["oracle"]["checks_by_task"][task].values())
        for task in SIX_RELEASES
    )
    assert all(
        not evidence["shared_paired_latent"]["tiny"]["checks_by_task"][task]["response_gain"]
        for task in SIX_RELEASES
    )
