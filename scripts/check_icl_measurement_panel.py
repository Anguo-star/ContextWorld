#!/usr/bin/env python3
"""Model-free validity checks for the nine-task ICL measurement panels.

This module deliberately does not import a world model, a simulator, or a
checkpoint.  It reads only the already materialized Development panels and
checks the parts of the measurement contract that can be checked from those
panels:

* the query image and executed context actions are shared by conditions;
* at least one historical image differs between conditions;
* every retained candidate future is physically mapped and its condition
  separability is reported at every stored horizon (zero-action rows stay in
  both the numerator and denominator);
* source-group counts and the Action Delay pending-queue caveat are explicit;
* the condition-mean reference and an ideal oracle obey their algebraic
  definitions.

The checks support the intended history-evidence protocol.  They do not claim
that a changed historical pixel is by itself sufficient for identifiability.
The command is CPU-only and never reads Training or Test payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_TASKS = (
    "speed",
    "action_strength",
    "robot_arm_mass",
    "action_delay",
    "contact_friction",
    "motion_damping",
    "cube_gripper_carry",
    "door",
    "portal_exit",
)
HORIZONS = (5, 10, 15, 20, 25)
# This is in the squared units of each task's physical coordinate readout.
# It is intentionally stated in the output so numerical near-zero values are
# not silently called physically distinct.
SEPARABILITY_TOLERANCE_SQUARED = 1.0e-8


TASK_ALIASES = {
    "speed": "tworoom",
    "door": "tworoom",
    "portal_exit": "tworoom",
    "action_delay": "tworoom",
    "action_strength": "pusht",
    "contact_friction": "pusht",
    "motion_damping": "pusht",
    "robot_arm_mass": "mass",
    "cube_gripper_carry": "cube",
}


# This table is deliberately documentary.  The current 2,436-scene panel is
# checked below without fitting a decoder or opening any other split.  These
# entries record the kind of evidence already described by the repository's
# task protocols, so the report can distinguish that prior evidence from the
# structural checks performed here.
HISTORY_EVIDENCE = {
    "speed": {
        "prior_evidence_level": "direct_observable_inversion_and_development_response",
        "references": (
            {
                "path": "scripts/analyze_tworoom_speed_isolated_v2.py",
                "role": "fixed context-state speed recovery and query/input audit",
            },
            {
                "path": "scripts/diagnose_speed_rollout_refresh.py",
                "role": "model-visible RGB velocity readout diagnostic",
            },
            {
                "path": "docs/protocols/TwoRoom_SpeedFull_Data_Protocol.md",
                "role": "prior Development E1 history-separation record",
            },
        ),
        "supports": (
            "A prior Speed protocol contains a fixed observable readout and a "
            "Development history-separation result."
        ),
    },
    "action_strength": {
        "prior_evidence_level": "development_model_history_response_without_dedicated_decoder",
        "references": (
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Training-to-Development history/switch/response table",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "matched query and physical-response construction contract",
            },
        ),
        "supports": (
            "Prior Development model response metrics exist, while the cited "
            "protocol does not provide a current-panel RGB-only decoder result."
        ),
    },
    "robot_arm_mass": {
        "prior_evidence_level": "development_model_history_response_without_dedicated_decoder",
        "references": (
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Training-to-Development history/switch/response table",
            },
            {
                "path": "scripts/audit_reacher_arm_mass_h3.py",
                "role": "query-state/history/future causal construction audit",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "matched query and split-isolation contract",
            },
        ),
        "supports": (
            "Prior Development model response metrics and causal construction "
            "audits exist; they are not a decoder trained for this panel."
        ),
    },
    "action_delay": {
        "prior_evidence_level": "development_model_history_response_and_analytic_physical_grouping",
        "references": (
            {
                "path": "scripts/analyze_tworoom_action_delay_h3.py",
                "role": "history-selection and physical-response grouping analysis",
            },
            {
                "path": "scripts/analyze_tworoom_action_delay_h3_multistep.py",
                "role": "horizon-wise Development history response analysis",
            },
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Development history-selection and delay-group limits",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "History=7 causal-chain and split-isolation contract",
            },
        ),
        "supports": (
            "Prior delay analyses document Development response and the analytic "
            "one-step 5--10 physical equivalence; they do not prove exact recovery "
            "of every queue state from the current panel."
        ),
    },
    "contact_friction": {
        "prior_evidence_level": "training_fit_development_rgb_history_decoder_protocol",
        "references": (
            {
                "path": "scripts/audit_pusht_contact_friction_rgb_identifiability.py",
                "role": "RGB-only Training-fit / Development-evaluated decoder audit",
            },
            {
                "path": "scripts/audit_pusht_contact_friction_history_identifiability.py",
                "role": "visible-history versus current-frame audit",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "RGB identifiability and matched-response contract",
            },
        ),
        "supports": (
            "The repository specifies a held-out RGB/history observability audit "
            "for this task; this checker does not rerun or import its numbers."
        ),
    },
    "motion_damping": {
        "prior_evidence_level": "training_fit_development_rgb_history_decoder_protocol",
        "references": (
            {
                "path": "scripts/audit_pusht_motion_damping_history_identifiability.py",
                "role": "fixed RGB block-motion-ratio Training/Development audit",
            },
            {
                "path": "scripts/diagnose_pusht_motion_damping_training_signal.py",
                "role": "visible-history audit upper bound and Development diagnostics",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "matched-response and no-static-start-shortcut contract",
            },
        ),
        "supports": (
            "The repository specifies a held-out RGB history readout and a "
            "Development diagnostic; this checker does not rerun or import its numbers."
        ),
    },
    "cube_gripper_carry": {
        "prior_evidence_level": "frozen_rgb_history_probe_and_development_response",
        "references": (
            {
                "path": "docs/protocols/Cube_Gripper_Carry_History3_Development_v4_Protocol.md",
                "role": "pre-registered Training-fit / Development RGB-history probe",
            },
            {
                "path": "docs/protocols/Cube_Gripper_Carry_History3_v4r1_Pre_Public_Handoff.md",
                "role": "v4r1 Development history and response record",
            },
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Development history/switch/response table",
            },
        ),
        "supports": (
            "A frozen RGB-history probe and prior Development response record are "
            "documented, with their own data-readiness scope."
        ),
    },
    "door": {
        "prior_evidence_level": "analytic_history_probe_and_development_model_response",
        "references": (
            {
                "path": "docs/protocols/TwoRoom_History3_Hidden_Passage_Feasibility_v1.md",
                "role": "continuous probe trajectory and history/query separation contract",
            },
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Development history/switch/response table",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "matched counterfactual and static-start leakage controls",
            },
        ),
        "supports": (
            "The door protocol analytically constructs a collision/through-door "
            "history probe and prior Development response is recorded."
        ),
    },
    "portal_exit": {
        "prior_evidence_level": "analytic_trajectory_probe_and_development_model_response",
        "references": (
            {
                "path": "scripts/audit_tworoom_portal_exit_h3.py",
                "role": "query equality, future-gap, and deterministic replay audit",
            },
            {
                "path": "docs/ContextWorld_ICL_Benchmark.md",
                "role": "Development history/switch/response table",
            },
            {
                "path": "docs/Data_Generation.md",
                "role": "shared query and split-isolation contract",
            },
        ),
        "supports": (
            "The portal protocol has a causal trajectory/replay audit and prior "
            "Development response metrics; it is not a decoder result for this panel."
        ),
    },
}

# The wording for the other tasks intentionally contains "without dedicated
# decoder" as a limitation, so this list is explicit rather than inferred by
# substring matching.
DIRECT_HISTORY_DECODER_TASKS = frozenset(
    {"speed", "contact_friction", "motion_damping", "cube_gripper_carry"}
)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _json_safe(value: Any) -> Any:
    """Convert NumPy values while refusing non-finite JSON numbers."""

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_scalar(value: Any) -> str:
    value = _json_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _manifest_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("scenes", "pairs", "queries"):
        entries = manifest.get(key)
        if entries is not None:
            if not isinstance(entries, list) or not entries:
                raise ValueError(f"manifest {key!r} must be a non-empty list")
            return [dict(entry) for entry in entries]
    raise ValueError("panel manifest has no scenes/pairs/queries list")


def canonical_task(task: str) -> str:
    key = str(task).strip().lower().replace("-", "_")
    if key not in TASK_ALIASES:
        raise ValueError(f"unsupported task {task!r}")
    return TASK_ALIASES[key]


def physical_coordinate_spec(task: str) -> dict[str, Any]:
    """Return the fixed model-free physical readout contract."""

    family = canonical_task(task)
    if family == "tworoom":
        return {
            "family": family,
            "names": ["agent_x_px", "agent_y_px"],
            "units": "px",
            "mapping": "TwoRoom future_states[..., :2] (agent xy)",
        }
    if family == "pusht":
        return {
            "family": family,
            "names": [
                "pusher_x_px",
                "pusher_y_px",
                "block_x_px",
                "block_y_px",
                "block_angle_sin_px_equivalent",
                "block_angle_cos_px_equivalent",
            ],
            "units": "px-equivalent",
            "mapping": (
                "7-vector: state[:4] + 40*sin/cos(state[4]); "
                "12-vector: state[0:2] + state[6:8] + "
                "40*sin/cos(state[10])"
            ),
        }
    if family == "mass":
        return {
            "family": family,
            "names": ["finger_x_mm", "finger_y_mm"],
            "units": "mm",
            "mapping": "finger_positions * 1000",
        }
    return {
        "family": family,
        "names": [
            "effector_x_mm",
            "effector_y_mm",
            "effector_z_mm",
            "cube_x_mm",
            "cube_y_mm",
            "cube_z_mm",
        ],
        "units": "mm",
        "mapping": "future_states[..., [0,1,6]] + future_states[..., 2:5], then *1000",
    }


def physical_targets_from_panel(
    panel: Mapping[str, Any], task: str
) -> np.ndarray:
    """Extract condition-first physical future coordinates ``[K,C,T,D]``."""

    family = canonical_task(task)
    if family == "mass":
        if "finger_positions" not in panel:
            raise KeyError("robot_arm_mass panel is missing finger_positions")
        values = np.asarray(panel["finger_positions"], dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != 2:
            raise ValueError(f"finger_positions has invalid shape {values.shape}")
        targets = values * 1000.0
    else:
        if "future_states" not in panel:
            raise KeyError("panel is missing future_states")
        states = np.asarray(panel["future_states"], dtype=np.float64)
        if states.ndim != 4:
            raise ValueError(f"future_states must be [K,C,T,D], got {states.shape}")
        if family == "tworoom":
            if states.shape[-1] < 2:
                raise ValueError(f"TwoRoom future_states width is {states.shape[-1]}")
            targets = states[..., :2]
        elif family == "cube":
            if states.shape[-1] < 7:
                raise ValueError(f"Cube future_states width is {states.shape[-1]}")
            targets = np.concatenate(
                [states[..., [0, 1, 6]], states[..., 2:5]], axis=-1
            ) * 1000.0
        elif states.shape[-1] == 7:
            targets = np.concatenate(
                [
                    states[..., :4],
                    (40.0 * np.sin(states[..., 4]))[..., None],
                    (40.0 * np.cos(states[..., 4]))[..., None],
                ],
                axis=-1,
            )
        elif states.shape[-1] >= 12:
            positions = np.concatenate(
                [states[..., 0:2], states[..., 6:8]], axis=-1
            )
            angle = states[..., 10]
            targets = np.concatenate(
                [
                    positions,
                    (40.0 * np.sin(angle))[..., None],
                    (40.0 * np.cos(angle))[..., None],
                ],
                axis=-1,
            )
        else:
            raise ValueError(
                "PushT future_states must have width 7 or at least 12, "
                f"got {states.shape[-1]}"
            )
    if not np.all(np.isfinite(targets)):
        raise ValueError("physical future coordinates contain non-finite values")
    return np.ascontiguousarray(targets, dtype=np.float64)


def _source_group(
    scene_id: str,
    manifest_entry: Mapping[str, Any],
    panel: Mapping[str, Any],
) -> tuple[str, str]:
    """Use audited grouping metadata, never a condition label or score."""

    for key in (
        "bootstrap_cluster",
        "source_group",
        "source_sha256",
        "source_id",
        "pair_id",
    ):
        if key in manifest_entry:
            text = _decode_scalar(manifest_entry[key])
            if text and text.lower() not in {"none", "nan"}:
                return f"{key}:{text}", key
    for key in ("source_group", "bootstrap_cluster", "source_sha256", "source_id", "pair_id"):
        if key in panel:
            text = _decode_scalar(panel[key])
            if text and text.lower() not in {"none", "nan"}:
                return f"{key}:{text}", key
    # Conservative fallback: no unrelated scenes are joined when source
    # provenance was not materialized in an older panel.
    return f"scene:{scene_id}", "scene_id_fallback"


def _same_condition_rows(array: np.ndarray, conditions: int) -> bool:
    if array.ndim == 0 or array.shape[0] != conditions:
        raise ValueError(
            f"expected condition axis of length {conditions}, got {array.shape}"
        )
    return bool(np.array_equal(array, np.repeat(array[:1], conditions, axis=0)))


def _pairwise_energy(targets: np.ndarray) -> np.ndarray:
    """Return unordered condition-pair squared distances as ``[P,C,T]``."""

    condition_count = int(targets.shape[0])
    pairs = [
        np.sum((targets[i] - targets[j]) ** 2, axis=-1, dtype=np.float64)
        for i, j in itertools.combinations(range(condition_count), 2)
    ]
    if not pairs:
        return np.empty((0, targets.shape[1], targets.shape[2]), dtype=np.float64)
    return np.stack(pairs, axis=0)


def physical_separability(
    targets: np.ndarray,
    *,
    tolerance_squared: float = SEPARABILITY_TOLERANCE_SQUARED,
) -> dict[str, Any]:
    """Summarize all condition/candidate future rows without filtering."""

    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 4:
        raise ValueError(f"targets must be [K,C,T,D], got {targets.shape}")
    k, candidates, times, dimensions = map(int, targets.shape)
    if k < 2:
        raise ValueError("physical separability requires at least two conditions")
    pair_energy = _pairwise_energy(targets)
    pair_count = int(pair_energy.shape[0] * candidates)
    energy_sum = np.sum(pair_energy, axis=(0, 1), dtype=np.float64)
    if pair_count:
        energy_mean = energy_sum / float(pair_count)
        separable = np.sum(pair_energy > float(tolerance_squared), axis=(0, 1))
        zero = np.sum(pair_energy <= float(tolerance_squared), axis=(0, 1))
        ratio = separable / float(pair_count)
    else:
        energy_mean = np.zeros(times, dtype=np.float64)
        separable = np.zeros(times, dtype=np.int64)
        zero = np.zeros(times, dtype=np.int64)
        ratio = np.zeros(times, dtype=np.float64)

    # The condition-mean oracle is the exact blind reference used by the
    # complete-prediction score.  It is a separate energy from pairwise
    # distance and is useful for comparing tasks with different K.
    condition_mean = np.mean(targets, axis=0, dtype=np.float64)
    condition_mean_energy = np.sum(
        (targets - condition_mean[None]) ** 2,
        axis=(0, 1, 3),
        dtype=np.float64,
    )
    return {
        "condition_count": k,
        "candidate_count": candidates,
        "horizon_count": times,
        "physical_dimension": dimensions,
        "pair_count_per_horizon": pair_count,
        "separable_pair_count_by_horizon": separable.astype(np.int64).tolist(),
        "zero_or_below_tolerance_pair_count_by_horizon": zero.astype(np.int64).tolist(),
        "separable_pair_fraction_by_horizon": ratio.tolist(),
        "pairwise_squared_distance_sum_by_horizon": energy_sum.tolist(),
        "pairwise_squared_distance_mean_by_horizon": energy_mean.tolist(),
        "condition_mean_reference_energy_by_horizon": condition_mean_energy.tolist(),
        "tolerance_squared": float(tolerance_squared),
        "zero_rows_retained": True,
        "candidate_condition_horizon_rows_retained": int(k * candidates * times),
    }


def _score(error: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(reference, dtype=np.float64), np.nan)
    positive = np.asarray(reference, dtype=np.float64) > 0.0
    result[positive] = 100.0 * (
        1.0 - np.asarray(error, dtype=np.float64)[positive] / np.asarray(reference, dtype=np.float64)[positive]
    )
    return result


def cross_condition_errors(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched error, all-other error, and condition-mean reference."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            "prediction and target must share [K,C,T,D] shape; "
            f"got {prediction.shape} and {target.shape}"
        )
    k = prediction.shape[0]
    distances = np.sum(
        (prediction[:, None] - target[None]) ** 2,
        axis=-1,
        dtype=np.float64,
    )
    matched = np.sum(
        distances[np.arange(k), np.arange(k)], axis=(0, 1), dtype=np.float64
    )
    if k > 1:
        mask = ~np.eye(k, dtype=bool)
        # Average over all ordered mismatched condition pairs and candidates,
        # then put it on the same per-scene sum scale as ``matched``.
        wrong = np.mean(distances[mask], axis=(0, 1), dtype=np.float64) * float(
            k * distances.shape[2]
        )
    else:
        wrong = np.full(target.shape[2], np.nan, dtype=np.float64)
    mean_target = np.mean(target, axis=0, dtype=np.float64)
    reference = np.sum(
        (target - mean_target[None]) ** 2,
        axis=(0, 1, 3),
        dtype=np.float64,
    )
    return matched, wrong, reference


def algebraic_self_test() -> dict[str, Any]:
    """Check ideal-oracle and condition-mean blind-reference identities."""

    target = np.asarray(
        [
            [[[0.0], [0.0]]],
            [[[2.0], [4.0]]],
        ],
        dtype=np.float64,
    )
    reference = np.asarray([2.0, 8.0], dtype=np.float64)

    # Ideal oracle: prediction under each history is that history's target.
    ideal_matched, ideal_wrong, ideal_reference = cross_condition_errors(target, target)
    ideal_score = _score(ideal_matched, ideal_reference)

    # Blind condition-mean prediction: every history receives the same mean.
    mean_target = np.mean(target, axis=0, dtype=np.float64)
    blind_prediction = np.repeat(mean_target[None], target.shape[0], axis=0)
    blind_matched, blind_wrong, blind_reference = cross_condition_errors(
        blind_prediction, target
    )
    blind_score = _score(blind_matched, blind_reference)
    blind_advantage = blind_wrong - blind_matched

    checks = {
        "ideal_oracle_error_zero": bool(np.array_equal(ideal_matched, np.zeros(2))),
        "ideal_oracle_score_100": bool(np.array_equal(ideal_score, np.full(2, 100.0))),
        "condition_mean_reference_matches_blind_error": bool(
            np.allclose(blind_matched, reference, rtol=0.0, atol=1.0e-12)
            and np.allclose(blind_reference, reference, rtol=0.0, atol=1.0e-12)
        ),
        "condition_mean_blind_score_zero": bool(
            np.array_equal(blind_score, np.zeros(2))
        ),
        "condition_mean_blind_history_advantage_zero": bool(
            np.array_equal(blind_advantage, np.zeros(2))
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"algebraic controls failed: {checks}")
    return {
        "passed": True,
        "horizons": list(HORIZONS[:2]),
        "reference_energy": reference.tolist(),
        "ideal_oracle": {
            "matched_error": ideal_matched.tolist(),
            "wrong_history_error": ideal_wrong.tolist(),
            "score": ideal_score.tolist(),
            "history_advantage": (ideal_wrong - ideal_matched).tolist(),
        },
        "condition_mean_blind": {
            "matched_error": blind_matched.tolist(),
            "wrong_history_error": blind_wrong.tolist(),
            "score": blind_score.tolist(),
            "history_advantage": blind_advantage.tolist(),
        },
        "checks": checks,
        "definitions": {
            "score": "100 * (1 - matched_error / condition_mean_reference_energy)",
            "history_advantage": "mean all-other-history error - matched-history error",
        },
    }


def _delay_queue_audit(
    scene_count: int,
    queue_value_equal: int,
    queue_length_equal: int,
    active_queue_different: int,
    lengths: list[int],
) -> dict[str, Any]:
    return {
        "present": True,
        "scene_count": int(scene_count),
        "padded_queue_value_equal_scenes": int(queue_value_equal),
        "queue_length_equal_scenes": int(queue_length_equal),
        "active_queue_different_scenes": int(active_queue_different),
        "lengths_observed": sorted(set(int(x) for x in lengths)),
        "interpretation": (
            "A shared query xy/current image is not a complete Markov state here: "
            "the pending action queue is hidden simulator state, and delay plus "
            "dynamics jointly determine its evolution."
        ),
        "do_not_equate_shared_xy_with_full_markov_state": True,
    }


def _empty_task_accumulator(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": spec["task"],
        "family": spec["family"],
        "physical_coordinate_spec": dict(spec["physical_coordinate_spec"]),
        "scene_count": 0,
        "source_groups": set(),
        "source_group_fields": set(),
        "current_image_equal_scenes": 0,
        "context_actions_equal_scenes": 0,
        "history_diff_scenes": 0,
        "history_frame_diff_scene_counts": None,
        "history_frame_total_condition_comparisons": None,
        "hash_mismatch_scenes": [],
        "conditions_seen": set(),
        "candidate_counts_seen": set(),
        "horizons_seen": set(),
        "physical_energy_sum": None,
        "physical_pair_energy_sum": None,
        "physical_pair_count": None,
        "physical_separable_pair_count": None,
        "physical_zero_pair_count": None,
        "scene_separability_ratios": [],
        "zero_action_candidate_count": 0,
        "candidate_count_total": 0,
        "delay_queue_value_equal": 0,
        "delay_queue_length_equal": 0,
        "delay_active_queue_different": 0,
        "delay_lengths": [],
    }


def _finalize_task_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    scene_count = int(acc["scene_count"])
    horizons = sorted(acc["horizons_seen"])
    times = len(horizons)
    energy = np.asarray(acc["physical_energy_sum"], dtype=np.float64)
    pair_energy = np.asarray(acc["physical_pair_energy_sum"], dtype=np.float64)
    pair_count = np.asarray(acc["physical_pair_count"], dtype=np.int64)
    separable = np.asarray(acc["physical_separable_pair_count"], dtype=np.int64)
    zero = np.asarray(acc["physical_zero_pair_count"], dtype=np.int64)
    ratio = np.divide(
        separable,
        pair_count,
        out=np.zeros(times, dtype=np.float64),
        where=pair_count > 0,
    )
    frame_counts = np.asarray(
        acc["history_frame_diff_scene_counts"], dtype=np.int64
    )
    frame_totals = np.asarray(
        acc["history_frame_total_condition_comparisons"], dtype=np.int64
    )
    result: dict[str, Any] = {
        "task": acc["task"],
        "family": acc["family"],
        "physical_coordinate_spec": acc["physical_coordinate_spec"],
        "scene_count": scene_count,
        "source_group_count": len(acc["source_groups"]),
        "source_group_fields": sorted(acc["source_group_fields"]),
        "conditions_seen": sorted(acc["conditions_seen"], key=str),
        "candidate_counts_seen": sorted(int(x) for x in acc["candidate_counts_seen"]),
        "horizons": horizons,
        "current_image": {
            "bitwise_equal_scenes": int(acc["current_image_equal_scenes"]),
            "all_scenes_equal": int(acc["current_image_equal_scenes"]) == scene_count,
        },
        "context_actions": {
            "bitwise_equal_scenes": int(acc["context_actions_equal_scenes"]),
            "all_scenes_equal": int(acc["context_actions_equal_scenes"]) == scene_count,
        },
        "historical_pixels": {
            "condition_difference_scenes": int(acc["history_diff_scenes"]),
            "all_scenes_have_a_historical_difference": int(acc["history_diff_scenes"]) == scene_count,
            "frame_indices_excluding_current": list(range(len(frame_counts))),
            "frame_difference_scene_counts": frame_counts.tolist(),
            "frame_total_condition_comparisons": frame_totals.tolist(),
        },
        "physical_separability": {
            "horizons": horizons,
            "separable_pair_fraction_by_horizon": ratio.tolist(),
            "separable_pair_count_by_horizon": separable.tolist(),
            "zero_or_below_tolerance_pair_count_by_horizon": zero.tolist(),
            "pair_count_per_horizon": pair_count.tolist(),
            "pairwise_squared_distance_sum_by_horizon": pair_energy.tolist(),
            "pairwise_squared_distance_mean_by_horizon": np.divide(
                pair_energy,
                pair_count,
                out=np.zeros(times, dtype=np.float64),
                where=pair_count > 0,
            ).tolist(),
            "condition_mean_reference_energy_by_horizon": energy.tolist(),
            "tolerance_squared": SEPARABILITY_TOLERANCE_SQUARED,
            "future_source": (
                "all stored true future_states/finger_positions for every "
                "candidate and horizon; no candidate or zero-action row filtered"
            ),
            "zero_rows_retained": True,
            "scene_equal_weight_mean_fraction_by_horizon": np.mean(
                np.asarray(acc["scene_separability_ratios"], dtype=np.float64), axis=0
            ).tolist(),
        },
        "zero_action": {
            "candidate_rows_detected": int(acc["zero_action_candidate_count"]),
            "candidate_rows_total": int(acc["candidate_count_total"]),
            "retained_in_separability_denominator": True,
        },
        "integrity": {
            "manifest_hash_mismatch_scenes": list(acc["hash_mismatch_scenes"]),
            "passed": not acc["hash_mismatch_scenes"],
        },
    }
    if acc["task"] == "action_delay":
        result["pending_queue"] = _delay_queue_audit(
            scene_count,
            acc["delay_queue_value_equal"],
            acc["delay_queue_length_equal"],
            acc["delay_active_queue_different"],
            acc["delay_lengths"],
        )
    return _json_safe(result)


def check_panel(panel_root: str | Path) -> dict[str, Any]:
    """Validate one task panel and return a JSON-safe summary."""

    panel_root = Path(panel_root)
    manifest_path = panel_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest.get("evaluation_split", manifest.get("protocol", {}).get("evaluation_split"))
    if str(split).lower() != "development":
        raise ValueError(
            f"{panel_root}: validator is Development-only, got evaluation_split={split!r}"
        )
    test_accessed = manifest.get(
        "test_payload_accessed", manifest.get("protocol", {}).get("test_payload_accessed", False)
    )
    if bool(test_accessed):
        raise ValueError(f"{panel_root}: manifest reports Test payload access")
    manifest_task = manifest.get("task", manifest.get("protocol", {}).get("task"))
    task = str(manifest_task or panel_root.name)
    if task not in EXPECTED_TASKS:
        raise ValueError(f"unexpected panel task {task!r} at {panel_root}")
    spec = {
        "task": task,
        "family": canonical_task(task),
        "physical_coordinate_spec": physical_coordinate_spec(task),
    }
    acc = _empty_task_accumulator(spec)
    entries = _manifest_entries(manifest)

    for entry in entries:
        if "path" not in entry:
            raise ValueError(f"scene entry has no path in {manifest_path}")
        scene_path = panel_root / str(entry["path"])
        scene_id = str(
            entry.get("scene_id", entry.get("pair_id", entry.get("query_id", scene_path.stem)))
        )
        expected_hash = entry.get("sha256")
        if expected_hash is not None and file_sha256(scene_path) != str(expected_hash):
            acc["hash_mismatch_scenes"].append(scene_id)
        # Index only small arrays.  In particular, do not decompress future
        # RGB arrays: this validity check is model-free and physical-state
        # based.  ``allow_pickle=False`` is part of the no-code-execution rule.
        with np.load(scene_path, allow_pickle=False) as archive:
            panel = {name: archive[name] for name in archive.files if name in {
                "history_pixels",
                "context_actions",
                "future_states",
                "finger_positions",
                "conditions",
                "physical_steps",
                "candidate_actions",
                "aux_pending_actions_at_query",
                "aux_pending_action_lengths",
                "pair_id",
            }}
        history = np.asarray(panel["history_pixels"])
        context_actions = np.asarray(panel["context_actions"])
        if history.ndim != 5 or history.shape[-3:] != (224, 224, 3):
            raise ValueError(f"{scene_id}: invalid history_pixels shape {history.shape}")
        if history.dtype != np.uint8:
            raise ValueError(f"{scene_id}: history_pixels must be uint8")
        conditions = int(history.shape[0])
        if conditions < 2:
            raise ValueError(f"{scene_id}: fewer than two condition rows")
        current_equal = _same_condition_rows(history[:, -1], conditions)
        context_equal = _same_condition_rows(context_actions, conditions)
        historical = history[:, :-1]
        frame_diff = np.any(
            historical != historical[:1], axis=tuple(range(2, historical.ndim))
        )
        history_diff = bool(np.any(frame_diff))
        if current_equal:
            acc["current_image_equal_scenes"] += 1
        if context_equal:
            acc["context_actions_equal_scenes"] += 1
        if history_diff:
            acc["history_diff_scenes"] += 1
        if acc["history_frame_diff_scene_counts"] is None:
            acc["history_frame_diff_scene_counts"] = np.zeros(
                historical.shape[1], dtype=np.int64
            )
            acc["history_frame_total_condition_comparisons"] = np.zeros(
                historical.shape[1], dtype=np.int64
            )
        acc["history_frame_diff_scene_counts"] += np.any(frame_diff, axis=0).astype(np.int64)
        # ``frame_diff`` compares every row to the first condition; the first
        # row is the reference, so there are K-1 actual comparisons.
        acc["history_frame_total_condition_comparisons"] += conditions - 1

        targets = physical_targets_from_panel(panel, task)
        if targets.shape[:3] != (
            conditions,
            int(np.asarray(panel["future_states"]).shape[1])
            if "future_states" in panel
            else int(np.asarray(panel["candidate_actions"]).shape[0]),
            int(np.asarray(panel["future_states"]).shape[2])
            if "future_states" in panel
            else int(np.asarray(panel["physical_steps"]).size),
        ):
            raise ValueError(f"{scene_id}: physical target axis mismatch {targets.shape}")
        _, candidates, times, _ = targets.shape
        steps = np.asarray(panel["physical_steps"], dtype=np.int64).reshape(-1)
        if steps.size != times or not np.all(np.diff(steps) > 0):
            raise ValueError(f"{scene_id}: invalid physical_steps {steps.tolist()}")
        if tuple(int(x) for x in steps) != HORIZONS:
            raise ValueError(f"{scene_id}: expected horizons {HORIZONS}, got {steps.tolist()}")
        if "candidate_actions" in panel:
            actions = np.asarray(panel["candidate_actions"], dtype=np.float64)
            if actions.ndim < 3 or actions.shape[0] != candidates or actions.shape[1] != times:
                raise ValueError(f"{scene_id}: candidate_actions shape {actions.shape}")
            zero_actions = np.all(np.abs(actions) <= 1.0e-12, axis=tuple(range(1, actions.ndim)))
            acc["zero_action_candidate_count"] += int(np.sum(zero_actions))
            acc["candidate_count_total"] += candidates
        sep = physical_separability(targets)
        if acc["physical_energy_sum"] is None:
            acc["physical_energy_sum"] = np.zeros(times, dtype=np.float64)
            acc["physical_pair_energy_sum"] = np.zeros(times, dtype=np.float64)
            acc["physical_pair_count"] = np.zeros(times, dtype=np.int64)
            acc["physical_separable_pair_count"] = np.zeros(times, dtype=np.int64)
            acc["physical_zero_pair_count"] = np.zeros(times, dtype=np.int64)
        acc["physical_energy_sum"] += np.asarray(
            sep["condition_mean_reference_energy_by_horizon"], dtype=np.float64
        )
        acc["physical_pair_energy_sum"] += np.asarray(
            sep["pairwise_squared_distance_sum_by_horizon"], dtype=np.float64
        )
        for key in (
            "physical_pair_count",
            "physical_separable_pair_count",
            "physical_zero_pair_count",
        ):
            source = {
                "physical_pair_count": np.full(
                    times, int(sep["pair_count_per_horizon"]), dtype=np.int64
                ),
                "physical_separable_pair_count": np.asarray(
                    sep["separable_pair_count_by_horizon"], dtype=np.int64
                ),
                "physical_zero_pair_count": np.asarray(
                    sep["zero_or_below_tolerance_pair_count_by_horizon"], dtype=np.int64
                ),
            }[key]
            acc[key] += source
        acc["scene_separability_ratios"].append(
            np.asarray(sep["separable_pair_fraction_by_horizon"], dtype=np.float64)
        )

        acc["scene_count"] += 1
        group, field = _source_group(scene_id, entry, panel)
        acc["source_groups"].add(group)
        acc["source_group_fields"].add(field)
        acc["conditions_seen"].update(_decode_scalar(x) for x in np.asarray(panel["conditions"]).reshape(-1))
        acc["candidate_counts_seen"].add(candidates)
        acc["horizons_seen"].update(int(x) for x in steps)

        if task == "action_delay":
            queue = np.asarray(panel["aux_pending_actions_at_query"])
            lengths = np.asarray(panel["aux_pending_action_lengths"], dtype=np.int64).reshape(-1)
            queue_equal = _same_condition_rows(queue, conditions)
            length_equal = _same_condition_rows(lengths, conditions)
            active = np.full(queue.shape, np.nan, dtype=np.float64)
            for condition, length in enumerate(lengths):
                if length < 0 or length > queue.shape[1]:
                    raise ValueError(f"{scene_id}: invalid pending queue length {length}")
                active[condition, : int(length)] = queue[condition, : int(length)]
            active_different = not np.array_equal(active, np.repeat(active[:1], conditions, axis=0), equal_nan=True)
            acc["delay_queue_value_equal"] += int(queue_equal)
            acc["delay_queue_length_equal"] += int(length_equal)
            acc["delay_active_queue_different"] += int(active_different)
            acc["delay_lengths"].extend(int(x) for x in lengths)

    if acc["scene_count"] != len(entries):
        raise AssertionError("scene count accounting failed")
    return _finalize_task_accumulator(acc)


def documentation_audit(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Confirm the base protocol treats history as evidence, not labels."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
    benchmark_doc = root / "docs" / "ContextWorld_ICL_Benchmark.md"
    generation_doc = root / "docs" / "Data_Generation.md"
    texts: dict[str, str] = {}
    for path in (benchmark_doc, generation_doc):
        if path.exists():
            texts[str(path)] = path.read_text(encoding="utf-8")
    joined = "\n".join(texts.values())
    checks = {
        "model_receives_images_and_actions": "模型只能接收图像和动作" in joined,
        "shared_current_query_and_different_history_response": (
            "共享当前状态" in joined and "历史中展示的物理响应不同" in joined
        ),
        "history_must_have_measurable_physical_response": (
            (
                "历史必须产生可测的物理响应" in joined
                or (
                    "历史可辨识性" in joined
                    and "隐藏规律在历史中产生可测响应" in joined
                )
            )
        ),
        "source_query_is_a_sampling_unit": "源查询" in joined or "source query" in joined.lower(),
        "development_scope_and_test_separation": "Development" in joined and "Test" in joined,
    }
    return {
        "documents_checked": sorted(texts),
        "checks": checks,
        "passed": bool(texts) and all(checks.values()),
        "interpretation": (
            "Source IDs and condition labels are used here only for provenance and "
            "scoring metadata. The intended model-visible evidence is image/action "
            "history; this document check cannot rule out an unintended pixel-level "
            "identifier, and historical pixel difference alone is not sufficient "
            "identifiability."
        ),
    }


def history_sufficiency_audit(
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Separate current-panel construction checks from prior evidence.

    This is intentionally a document/provenance audit.  It does not open a
    Training/Test table, fit a classifier, or treat a source/condition ID as a
    model-visible answer.  The current panel therefore remains
    ``construction_guarantee_only`` for every task, even when an earlier task
    protocol documents a held-out history decoder or Development response.
    """

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    direct_decoder_tasks = set(DIRECT_HISTORY_DECODER_TASKS)
    development_response_tasks = set(EXPECTED_TASKS)
    task_rows: dict[str, Any] = {}
    all_references_present = True
    for task in EXPECTED_TASKS:
        row = HISTORY_EVIDENCE[task]
        references = []
        for reference in row["references"]:
            relative = str(reference["path"])
            present = (root / relative).is_file()
            all_references_present &= present
            references.append(
                {
                    "path": relative,
                    "present": bool(present),
                    "role": str(reference["role"]),
                }
            )
        task_rows[task] = {
            "current_panel_evidence_level": "construction_guarantee_only",
            "current_panel_decoder_run": False,
            "current_panel_history_pixel_difference_is_sufficient": False,
            "prior_evidence_level": str(row["prior_evidence_level"]),
            "prior_evidence_is_documented": True,
            "prior_evidence_applies_to_current_panel": False,
            "prior_evidence_supports": str(row["supports"]),
            "references": references,
            "source_id_or_condition_label_used_as_history_evidence": False,
        }

    return {
        "scope": (
            "Repository protocol/source-document audit only; no Training or "
            "Test payload is read and no new decoder, model, or data is run."
        ),
        "current_panel_claim": (
            "For all nine tasks, matched query/context checks plus historical "
            "pixel differences are construction evidence only. They do not "
            "establish sufficient or generalized history identifiability."
        ),
        "prior_direct_history_decoder_tasks": sorted(direct_decoder_tasks),
        "prior_development_response_tasks": sorted(development_response_tasks),
        "current_panel_construction_only_tasks": list(EXPECTED_TASKS),
        "tasks": task_rows,
        "checks": {
            "all_nine_current_panels_construction_only": all(
                row["current_panel_evidence_level"] == "construction_guarantee_only"
                for row in task_rows.values()
            ),
            "history_pixel_difference_not_called_sufficient": all(
                not row["current_panel_history_pixel_difference_is_sufficient"]
                for row in task_rows.values()
            ),
            "source_ids_and_condition_labels_not_history_evidence": all(
                not row["source_id_or_condition_label_used_as_history_evidence"]
                for row in task_rows.values()
            ),
            "all_referenced_protocol_or_script_files_present": bool(
                all_references_present
            ),
            "no_new_decoder_or_model_run": True,
        },
        "interpretation": (
            "Prior decoder or Training-to-Development response records are "
            "useful external evidence about those earlier protocols. They are "
            "reported separately and cannot promote the present 2,436-scene "
            "panel beyond construction validity; source IDs, pair IDs, and "
            "condition labels remain provenance/scoring metadata only."
        ),
    }


def validate_panels(
    panels_root: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete nine-task model-free validation."""

    panels_root = Path(panels_root)
    if not panels_root.is_dir():
        raise FileNotFoundError(panels_root)
    task_dirs = {path.name: path for path in panels_root.iterdir() if path.is_dir()}
    missing = [task for task in EXPECTED_TASKS if task not in task_dirs]
    if missing:
        raise FileNotFoundError(f"missing expected task panel(s): {missing}")
    tasks = {task: check_panel(task_dirs[task]) for task in EXPECTED_TASKS}
    total_scenes = sum(int(value["scene_count"]) for value in tasks.values())
    return _json_safe(
        {
            "schema": "contextworld.icl_measurement_panel_validation.v1",
            "scope": {
                "panels_root": str(panels_root),
                "tasks": list(EXPECTED_TASKS),
                "task_count": len(EXPECTED_TASKS),
                "scene_count": total_scenes,
                "expected_scene_count": 2436,
                "evaluation_split": "development",
                "no_training": True,
                "test_payload_accessed": False,
                "model_free": True,
                "cpu_only": True,
            },
            "tasks": tasks,
            "global_checks": {
                "all_current_images_equal": all(
                    value["current_image"]["all_scenes_equal"] for value in tasks.values()
                ),
                "all_context_actions_equal": all(
                    value["context_actions"]["all_scenes_equal"] for value in tasks.values()
                ),
                "all_scenes_have_historical_pixel_difference": all(
                    value["historical_pixels"]["all_scenes_have_a_historical_difference"]
                    for value in tasks.values()
                ),
                "all_manifest_hashes_match": all(
                    value["integrity"]["passed"] for value in tasks.values()
                ),
            },
            "algebraic_self_test": algebraic_self_test(),
            "documentation": documentation_audit(repo_root),
            "history_sufficiency_evidence": history_sufficiency_audit(repo_root),
            "scientific_interpretation": {
                "supported": (
                    "The frozen Development panels implement a shared-present, "
                    "shared-context-actions history contrast and retain condition "
                    "separability measurements at each horizon."
                ),
                "not_supported": (
                    "Historical pixel differences do not by themselves prove that "
                    "the hidden condition is sufficiently identifiable or that a "
                    "world model will recover it or generalize. The physical readout "
                    "is a fixed model-free validity diagnostic, not a model capability "
                    "score; prior protocol/model evidence is listed separately and "
                    "does not upgrade the current panel's construction-only claim."
                ),
            },
        }
    )


def write_validation(
    output: str | Path,
    panels_root: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    result = validate_panels(panels_root, repo_root=repo_root)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing the base protocol documents",
    )
    args = parser.parse_args(argv)
    result = write_validation(args.output, args.panels_root, repo_root=args.repo_root)
    print(
        f"validated {result['scope']['scene_count']} scenes across "
        f"{result['scope']['task_count']} tasks; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
