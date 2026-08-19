"""Derived, descriptive summary for the frozen original-baseline matrix.

The source preregistration freezes eight original-environment checkpoints and
eighteen capability/family cells.  This module does not score models or alter
the formal ContextWorld scoreboard.  It verifies the immutable inputs and
reduces the heterogeneous result schemas to one machine-readable summary.

Two narrowly registered recoveries are part of the evidence chain:

* Action Delay uses an explicit H7-to-H3 tail projection after the originally
  registered native-H7 attempt failed before producing any predictions.
* Numerical rescore recoveries preserve successful raw model receipts while
  reproducing the evaluator's original aggregation dtype exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextworld.benchmarks.original_baseline_matrix import (
    DEFAULT_ORIGINAL_BASELINE_FREEZE,
    DEFAULT_ORIGINAL_BASELINE_PREREG,
    audit_original_baseline_prereg,
    load_original_baseline_prereg,
)
from contextworld.paths import repository_root


DEFAULT_MATRIX_SUMMARY = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/matrix_summary.json"
)
ACTION_DELAY_RECOVERY_PREREG = Path(
    "configs/benchmark/"
    "contextworld_action_delay_original_baseline_recovery_prereg_v1.yaml"
)
ACTION_DELAY_RECOVERY_FREEZE = Path(
    "configs/benchmark/"
    "contextworld_action_delay_original_baseline_recovery_freeze_v1.json"
)
ACTION_DELAY_RECOVERY_RELEASE = Path(
    "configs/benchmark/tworoom_action_delay_original_baseline_recovery_v1.yaml"
)
ACTION_STRENGTH_LEWM_RESCORE_RECOVERY = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/rescore_recovery/"
    "contextworld-action-strength/lewm_float32_rescore_recovery_v1.json"
)
PORTAL_RESCORE_RECOVERY_ROOT = Path(
    "artifacts/evaluation/original_baseline_matrix_v1/rescore_recovery/"
    "contextworld-portal-exit"
)

CAPABILITY_ORDER = (
    "contextworld-speed",
    "contextworld-door",
    "contextworld-action-delay",
    "contextworld-action-strength",
    "contextworld-contact-friction",
    "contextworld-motion-damping",
    "contextworld-reacher-arm-mass",
    "contextworld-portal-exit",
    "contextworld-cube-gripper-carry",
)
FAMILY_ORDER = ("lewm", "pldm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, root: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        logical = resolved.relative_to(root).as_posix()
    except ValueError:
        logical = resolved.as_posix()
    return {
        "path": logical,
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _checkpoint_sha(payload: Mapping[str, Any]) -> str:
    model = _mapping(payload.get("model"), label="model")
    adapter = model.get("adapter")
    if isinstance(adapter, Mapping):
        value = adapter.get("checkpoint_sha256")
    else:
        checkpoint = model.get("checkpoint")
        if isinstance(checkpoint, Mapping):
            value = checkpoint.get("sha256")
        else:
            value = model.get("checkpoint_sha256")
    result = str(value or "")
    if len(result) != 64:
        raise ValueError("Result does not bind a checkpoint SHA-256")
    return result


def _release_sha(payload: Mapping[str, Any]) -> str:
    release_config = payload.get("release_config")
    if isinstance(release_config, Mapping):
        value = release_config.get("sha256")
    else:
        release = payload.get("release")
        if isinstance(release, Mapping):
            value = release.get(
                "release_config_sha256",
                release.get("release_config_sha256_at_evaluation"),
            )
        else:
            contract = payload.get("contract")
            value = (
                contract.get("release_config_sha256")
                if isinstance(contract, Mapping)
                else None
            )
    result = str(value or "")
    if len(result) != 64:
        raise ValueError("Result does not bind a release-config SHA-256")
    return result


def _state_unchanged(payload: Mapping[str, Any]) -> bool | None:
    model = _mapping(payload.get("model"), label="model")
    before = model.get("state_sha256_before")
    after = model.get("state_sha256_after")
    if before is None and after is None:
        return None
    if not isinstance(before, str) or not isinstance(after, str):
        raise ValueError("Model-state identity is incomplete")
    return before == after


def _record_count(capability_id: str, payload: Mapping[str, Any]) -> int:
    if capability_id == "contextworld-speed":
        tracks = _mapping(payload.get("tracks"), label="speed tracks")
        count = 0
        for track in tracks.values():
            rows = _mapping(track, label="speed track").get("records")
            if not isinstance(rows, list):
                raise ValueError("Speed result must retain per-track records")
            count += len(rows)
        return count
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"{capability_id} result must retain records")
    return len(rows)


def _speed_metric(payload: Mapping[str, Any]) -> dict[str, Any]:
    tracks = _mapping(payload.get("tracks"), label="speed tracks")
    core: dict[str, Any] = {}
    all_core_passes: list[bool] = []
    for track_name in ("seen_for_multi", "unseen_interpolation"):
        track = _mapping(tracks.get(track_name), label=track_name)
        horizons = _mapping(track.get("horizons"), label=f"{track_name}.horizons")
        compact: dict[str, Any] = {}
        for horizon in ("1", "2", "3", "5"):
            row = _mapping(horizons.get(horizon), label=f"{track_name}.h{horizon}")
            passed = bool(row.get("formal_within_checkpoint_pass"))
            all_core_passes.append(passed)
            compact[horizon] = {
                "matching_to_other_loss_ratio": float(
                    row["reference_speed_balanced_matching_to_other_loss_ratio"]
                ),
                "query_win_rate_vs_other_mean": float(
                    row[
                        "reference_speed_balanced_query_win_rate_vs_other_mean"
                    ]
                ),
                "strict_query_win_rate_vs_every_other": float(
                    row[
                        "reference_speed_balanced_strict_query_win_rate_vs_every_other"
                    ]
                ),
                "within_checkpoint_passed": passed,
            }
        core[track_name] = compact
    primary = core["unseen_interpolation"]["1"]
    return {
        "name": "unseen_interpolation_h1_strict_query_win_rate_vs_every_other",
        "value": primary["strict_query_win_rate_vs_every_other"],
        "reader_metrics": core,
        "gate": {
            "passed": all(all_core_passes),
            "scope": "all_horizons_in_two_core_tracks_within_one_checkpoint",
        },
    }


def _door_metric(payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = _mapping(
        _mapping(payload.get("summary"), label="door summary").get("overall"),
        label="door overall",
    )
    return {
        "name": "same_history_two_target_accuracy",
        "value": float(overall["same_history_two_target_accuracy"]),
        "reader_metrics": {
            "matching_vs_opposite_history_win_rate": float(
                overall["matching_vs_opposite_history_win_rate"]
            ),
            "strict_win_rate": float(overall["strict_win_rate"]),
        },
        "gate": {
            "passed": bool(payload.get("formal_checkpoint_passed")),
            "scope": "single_checkpoint_door_gate",
        },
    }


def _action_delay_metric(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = _mapping(payload.get("core_h1"), label="Action Delay core_h1")
    interval = _mapping(
        core.get("paired_query_bootstrap_95_percent_interval"),
        label="Action Delay bootstrap interval",
    )
    gate = _mapping(payload.get("gate"), label="Action Delay gate")
    return {
        "name": "physical_group_macro_accuracy",
        "value": float(core["physical_group_macro_accuracy"]),
        "reader_metrics": {
            "minimum_physical_group_accuracy": float(
                core["minimum_physical_group_accuracy"]
            ),
            "paired_query_bootstrap_95_percent_lower_bound": float(
                interval["lower"]
            ),
        },
        "gate": {
            "passed": bool(gate.get("passed")),
            "scope": "single_checkpoint_action_delay_gate",
        },
    }


def _paired_metric(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(payload.get("metrics"), label="paired-task metrics")
    gate = _mapping(payload.get("gate"), label="paired-task gate")
    reader = {
        key: metrics[key]
        for key in (
            "correct_history_rate",
            "context_switch_rate",
            "rule_switch_rate",
            "joint_icl_pair_success_rate",
        )
        if key in metrics
    }
    return {
        "name": "correct_future_rate",
        "value": float(metrics["correct_future_rate"]),
        "reader_metrics": reader,
        "gate": {
            "passed": bool(gate.get("passed")),
            "scope": "single_checkpoint_capability_gate",
        },
    }


def extract_cell_metric(
    capability_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize a frozen result without comparing latent scales."""

    if capability_id == "contextworld-speed":
        return _speed_metric(payload)
    if capability_id == "contextworld-door":
        return _door_metric(payload)
    if capability_id == "contextworld-action-delay":
        return _action_delay_metric(payload)
    return _paired_metric(payload)


def _verify_declared_identity(
    declaration: Mapping[str, Any], *, root: Path, label: str
) -> dict[str, Any]:
    path = Path(str(declaration.get("path", ""))).expanduser()
    observed = _identity(path, root=root)
    expected = {
        "path": str(declaration.get("path", "")),
        "sha256": str(declaration.get("sha256", "")),
        "size_bytes": int(declaration.get("size_bytes", observed["size_bytes"])),
    }
    if observed != expected:
        raise RuntimeError(f"{label} identity mismatch: {observed} != {expected}")
    return observed


def _action_delay_recovery_context(root: Path) -> dict[str, Any]:
    import yaml

    prereg_path = root / ACTION_DELAY_RECOVERY_PREREG
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    prereg = dict(_mapping(prereg, label="Action Delay recovery prereg"))
    if prereg.get("status") != "frozen_before_recovery_scoring":
        raise ValueError("Action Delay recovery was not frozen before scoring")
    authority = _mapping(prereg.get("authority"), label="recovery authority")
    if (
        authority.get("training_authorized") is not False
        or authority.get("checkpoint_selection_authorized") is not False
        or authority.get("formal_scoreboard_mutation") is not False
        or int(authority.get("authorized_icl_cells", -1)) != 2
    ):
        raise ValueError("Action Delay recovery authority is too broad")

    freeze_path = root / ACTION_DELAY_RECOVERY_FREEZE
    freeze = _load_json(freeze_path)
    frozen_prereg = _mapping(freeze.get("preregistration"), label="frozen prereg")
    prereg_identity = _identity(ACTION_DELAY_RECOVERY_PREREG, root=root)
    if (
        prereg_identity["sha256"] != frozen_prereg.get("sha256")
        or prereg_identity["size_bytes"] != frozen_prereg.get("size_bytes")
    ):
        raise RuntimeError("Action Delay recovery prereg/freeze mismatch")

    for label, declaration in (
        ("original prereg", prereg["original_matrix_bindings"]["original_baseline_preregistration"]),
        ("original freeze", prereg["original_matrix_bindings"]["original_baseline_freeze"]),
        ("original release", prereg["release_bindings"]["original_release"]),
        ("recovery release", prereg["release_bindings"]["recovery_release"]),
    ):
        _verify_declared_identity(declaration, root=root, label=label)
    for failure in prereg["original_matrix_bindings"]["failed_native_h7_attempts"]:
        _verify_declared_identity(
            failure, root=root, label=f"{failure['family']} native-H7 failure"
        )
    for implementation in prereg["implementation"]:
        _verify_declared_identity(
            implementation,
            root=root,
            label=f"recovery implementation {implementation['path']}",
        )
    for checkpoint in prereg["checkpoints"]:
        _verify_declared_identity(
            checkpoint["weights"],
            root=root,
            label=f"recovery checkpoint {checkpoint['checkpoint_id']}",
        )

    return {
        "preregistration": prereg_identity,
        "freeze": _identity(ACTION_DELAY_RECOVERY_FREEZE, root=root),
        "release": _identity(ACTION_DELAY_RECOVERY_RELEASE, root=root),
        "history_adapter": dict(prereg["history_adapter"]),
        "failed_native_h7_attempts": [
            _identity(Path(row["path"]), root=root)
            for row in prereg["original_matrix_bindings"]["failed_native_h7_attempts"]
        ],
    }


def _action_strength_recovery_evidence(root: Path, raw_sha: str) -> dict[str, Any]:
    path = root / ACTION_STRENGTH_LEWM_RESCORE_RECOVERY
    payload = _load_json(path)
    if (
        payload.get("status") != "completed"
        or payload.get("verification", {}).get("passed") is not True
        or payload.get("bindings", {}).get("raw_receipt", {}).get("sha256")
        != raw_sha
    ):
        raise RuntimeError("Action Strength LeWM rescore recovery is invalid")
    legacy_path = Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        "contextworld-action-strength/lewm.rescore_failure.json"
    )
    legacy = _load_json(root / legacy_path)
    if (
        legacy.get("status") != "failed"
        or legacy.get("evaluation_rerun_performed") is not False
        or legacy.get("raw_receipt_preserved") is not True
    ):
        raise RuntimeError("Action Strength legacy rescore failure changed")
    return {
        "kind": "float32_exact_rescore_recovery",
        "status": "verified",
        "receipt": _identity(ACTION_STRENGTH_LEWM_RESCORE_RECOVERY, root=root),
        "retained_legacy_failed_rescore": _identity(legacy_path, root=root),
        "model_gate_passed": bool(
            payload["verification"]["recomputed_model_gate_passed"]
        ),
    }


def _action_delay_rescore_evidence(
    root: Path,
    *,
    family: str,
    checkpoint_sha: str,
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        f"contextworld-action-delay/recovery_v1/{family}.rescore.json"
    )
    payload = _load_json(root / path)
    checkpoints = payload.get("checkpoints")
    if (
        payload.get("status") != "completed"
        or payload.get("submission_kind") != "descriptive_checkpoint"
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 1
    ):
        raise RuntimeError(f"Invalid Action Delay rescore receipt: {path}")
    row = _mapping(checkpoints[0], label=f"{family} Action Delay rescore")
    if (
        row.get("checkpoint_sha256") != checkpoint_sha
        or float(row.get("physical_group_macro_accuracy", -1))
        != float(metric["value"])
        or bool(row.get("passed")) != bool(metric["gate"]["passed"])
    ):
        raise RuntimeError(f"Action Delay rescore mismatch: {path}")
    return {
        "kind": "independent_record_rescore",
        "status": "verified",
        "receipt": _identity(path, root=root),
        "model_gate_passed": bool(row["passed"]),
    }


def _portal_rescore_recovery_evidence(
    root: Path,
    *,
    family: str,
    raw_sha: str,
    checkpoint_sha: str,
) -> dict[str, Any]:
    path = PORTAL_RESCORE_RECOVERY_ROOT / (
        f"{family}_float32_rescore_recovery_v1.json"
    )
    payload = _load_json(root / path)
    bindings = _mapping(payload.get("bindings"), label="Portal recovery bindings")
    raw_binding = _mapping(
        bindings.get("raw_receipt"), label="Portal recovery raw binding"
    )
    checkpoint = _mapping(
        bindings.get("checkpoint"), label="Portal recovery checkpoint binding"
    )
    frozen_weights = _mapping(
        checkpoint.get("frozen_weights"), label="Portal frozen weights"
    )
    verification = _mapping(
        payload.get("verification"), label="Portal recovery verification"
    )
    scope = _mapping(payload.get("scope"), label="Portal recovery scope")
    if (
        payload.get("status") != "completed"
        or verification.get("passed") is not True
        or raw_binding.get("sha256") != raw_sha
        or frozen_weights.get("sha256") != checkpoint_sha
        or scope.get("model_evaluation_rerun_performed") is not False
        or scope.get("raw_receipt_rewritten") is not False
        or scope.get("legacy_failed_rescore_rewritten") is not False
        or scope.get("formal_scoreboard_mutated") is not False
    ):
        raise RuntimeError(f"Invalid Portal rescore recovery: {path}")
    legacy_identity = _verify_declared_identity(
        _mapping(
            bindings.get("retained_legacy_failed_rescore"),
            label="retained Portal legacy rescore",
        ),
        root=root,
        label=f"{family} retained Portal legacy rescore",
    )
    return {
        "kind": "float32_exact_rescore_recovery",
        "status": "verified",
        "receipt": _identity(path, root=root),
        "retained_legacy_failed_rescore": legacy_identity,
        "model_gate_passed": bool(
            verification["recomputed_model_gate_passed"]
        ),
    }


def _existing_rescore_evidence(
    root: Path,
    *,
    capability_id: str,
    family: str,
    raw_receipt: Mapping[str, Any],
    checkpoint_sha: str,
    metric: Mapping[str, Any],
) -> dict[str, Any] | None:
    if capability_id not in {
        "contextworld-door",
        "contextworld-action-strength",
        "contextworld-contact-friction",
        "contextworld-motion-damping",
    }:
        return None
    path = Path(
        "artifacts/evaluation/original_baseline_matrix_v1/"
        f"{capability_id}/{family}.rescore.json"
    )
    payload = _load_json(root / path)
    if payload.get("status") != "completed":
        raise RuntimeError(f"Incomplete rescore receipt: {path}")
    identity = _identity(path, root=root)

    if capability_id == "contextworld-door":
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise RuntimeError(f"Malformed Door rescore receipt: {path}")
        row = _mapping(checkpoints[0], label=f"{family} Door rescore")
        if (
            float(row.get("correct_target_choice_rate", -1))
            != float(metric["value"])
            or bool(row.get("checkpoint_passed"))
            != bool(metric["gate"]["passed"])
        ):
            raise RuntimeError(f"Door rescore metric mismatch: {path}")
        return {
            "kind": "legacy_record_rescore",
            "status": "metric_verified",
            "evidence_strength": "legacy_output_omits_checkpoint_and_raw_sha",
            "receipt": identity,
            "model_gate_passed": bool(row["checkpoint_passed"]),
        }

    if capability_id == "contextworld-action-strength":
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 1:
            raise RuntimeError(f"Malformed Action Strength rescore: {path}")
        row = _mapping(checkpoints[0], label="Action Strength rescore")
        if (
            row.get("checkpoint_sha256") != checkpoint_sha
            or float(row.get("correct_future_rate", -1))
            != float(metric["value"])
            or bool(row.get("passed")) != bool(metric["gate"]["passed"])
        ):
            raise RuntimeError(f"Action Strength rescore mismatch: {path}")
        return {
            "kind": "independent_record_rescore",
            "status": "verified",
            "evidence_strength": "raw_path_and_checkpoint_sha_bound",
            "receipt": identity,
            "model_gate_passed": bool(row["passed"]),
        }

    if (
        identity["sha256"] != raw_receipt["sha256"]
        or identity["size_bytes"] != raw_receipt["size_bytes"]
        or _checkpoint_sha(payload) != checkpoint_sha
        or extract_cell_metric(capability_id, payload) != metric
    ):
        raise RuntimeError(f"Development rescore round trip changed: {path}")
    return {
        "kind": "deterministic_record_rescore_round_trip",
        "status": "completed_output_byte_identical_to_raw",
        "evidence_strength": "no_distinct_rescore_provenance_in_legacy_schema",
        "receipt": identity,
        "model_gate_passed": bool(metric["gate"]["passed"]),
    }


def build_original_baseline_summary(
    *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Verify and summarize all eighteen frozen original-baseline cells."""

    root = (repo_root or repository_root()).resolve()
    base_audit = audit_original_baseline_prereg(
        repo_root=root, verify_local_checkpoints=True
    )
    if base_audit.get("status") != "passed":
        raise RuntimeError("Original-baseline preregistration audit failed")
    prereg = load_original_baseline_prereg(repo_root=root)
    checkpoint_by_id = {
        row["checkpoint_id"]: row for row in prereg["checkpoints"]
    }
    component_by_id = {
        row["capability_id"]: row for row in prereg["components"]
    }
    cell_by_pair = {
        (row["capability_id"], row["family"]): row
        for row in prereg["icl_cells"]
    }
    action_delay_recovery = _action_delay_recovery_context(root)

    cells: list[dict[str, Any]] = []
    for capability_id in CAPABILITY_ORDER:
        component = component_by_id[capability_id]
        for family in FAMILY_ORDER:
            registered = cell_by_pair[(capability_id, family)]
            checkpoint = checkpoint_by_id[registered["checkpoint_id"]]
            original_attempt = None
            if capability_id == "contextworld-action-delay":
                result_path = Path(
                    "artifacts/evaluation/original_baseline_matrix_v1/"
                    f"contextworld-action-delay/recovery_v1/{family}.json"
                )
                expected_release_sha = action_delay_recovery["release"]["sha256"]
                failure = next(
                    row
                    for row in action_delay_recovery["failed_native_h7_attempts"]
                    if Path(row["path"]).name == f"{family}.failure.json"
                )
                original_attempt = {
                    "status": "failed_before_predictions",
                    "receipt": failure,
                }
            else:
                result_path = Path(registered["output"])
                expected_release_sha = component["release_config"]["sha256"]

            absolute_result = (root / result_path).resolve()
            payload = _load_json(absolute_result)
            if payload.get("status") not in {"completed", "passed"}:
                raise RuntimeError(f"Incomplete result: {result_path}")
            observed_checkpoint_sha = _checkpoint_sha(payload)
            expected_checkpoint_sha = checkpoint["weights"]["sha256"]
            if observed_checkpoint_sha != expected_checkpoint_sha:
                raise RuntimeError(f"Checkpoint mismatch: {result_path}")
            if _release_sha(payload) != expected_release_sha:
                raise RuntimeError(f"Release mismatch: {result_path}")
            unchanged = _state_unchanged(payload)
            if unchanged is False:
                raise RuntimeError(f"Model state changed: {result_path}")
            if capability_id == "contextworld-action-delay":
                model = _mapping(payload.get("model"), label="Action Delay model")
                adapter = _mapping(
                    model.get("adapter"), label="Action Delay recovery adapter"
                )
                projection = _mapping(
                    adapter.get("projection"), label="Action Delay projection"
                )
                if (
                    adapter.get("history_adapter") != "h3_tail_projection"
                    or adapter.get("weights_modified") is not False
                    or projection.get("input_pixels") != "input_pixels[:, -3:]"
                    or projection.get("raw_action_blocks")
                    != "raw_action_blocks[:, -5:]"
                    or projection.get("positional_embedding_interpolation")
                    is not False
                ):
                    raise RuntimeError(
                        f"Undisclosed Action Delay history projection: {result_path}"
                    )

            receipt = _identity(result_path, root=root)
            metric = extract_cell_metric(capability_id, payload)
            rescore: dict[str, Any] | None = None
            if capability_id == "contextworld-action-strength" and family == "lewm":
                rescore = _action_strength_recovery_evidence(
                    root, receipt["sha256"]
                )
            elif capability_id == "contextworld-action-delay":
                rescore = _action_delay_rescore_evidence(
                    root,
                    family=family,
                    checkpoint_sha=expected_checkpoint_sha,
                    metric=metric,
                )
            elif capability_id == "contextworld-portal-exit":
                rescore = _portal_rescore_recovery_evidence(
                    root,
                    family=family,
                    raw_sha=receipt["sha256"],
                    checkpoint_sha=expected_checkpoint_sha,
                )
            else:
                rescore = _existing_rescore_evidence(
                    root,
                    capability_id=capability_id,
                    family=family,
                    raw_receipt=receipt,
                    checkpoint_sha=expected_checkpoint_sha,
                    metric=metric,
                )
            cells.append(
                {
                    "capability_id": capability_id,
                    "environment": component["environment"],
                    "family": family,
                    "checkpoint_id": registered["checkpoint_id"],
                    "checkpoint_sha256": expected_checkpoint_sha,
                    "phase": registered["phase"],
                    "receipt": receipt,
                    "record_count": _record_count(capability_id, payload),
                    "model_state_unchanged": unchanged,
                    "metric": metric,
                    "history_adapter": (
                        "h3_tail_projection"
                        if capability_id == "contextworld-action-delay"
                        else registered.get("history_adapter")
                    ),
                    "native_history7_checkpoint": (
                        False
                        if capability_id == "contextworld-action-delay"
                        else None
                    ),
                    "original_attempt": original_attempt,
                    "rescore_evidence": rescore,
                    "formal_scoreboard_eligible": False,
                }
            )

    if len(cells) != 18:
        raise AssertionError(f"Expected 18 cells, got {len(cells)}")
    if len({(row["capability_id"], row["family"]) for row in cells}) != 18:
        raise AssertionError("Original-baseline cell matrix is not unique")

    return {
        "schema_version": 1,
        "matrix_id": "contextworld_original_baseline_matrix_v1",
        "status": "completed",
        "claim_scope": "post_release_single_checkpoint_descriptive_only",
        "formal_scoreboard_mutated": False,
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "cross_model_raw_latent_comparison_permitted": False,
        "base_protocol": {
            "preregistration": _identity(DEFAULT_ORIGINAL_BASELINE_PREREG, root=root),
            "freeze": _identity(DEFAULT_ORIGINAL_BASELINE_FREEZE, root=root),
            "checkpoint_identity_audit": _identity(
                Path(
                    "artifacts/evaluation/original_baseline_matrix_v1/"
                    "checkpoint_identity_audit.json"
                ),
                root=root,
            ),
        },
        "derivation_implementation": {
            "summary_module": _identity(
                Path("contextworld/benchmarks/original_baseline_results.py"),
                root=root,
            ),
            "command_line": _identity(
                Path("scripts/finalize_contextworld_original_baseline_matrix.py"),
                root=root,
            ),
        },
        "recoveries": {
            "action_delay_h3_tail_projection": action_delay_recovery,
            "action_strength_lewm_float32_rescore": next(
                row["rescore_evidence"]
                for row in cells
                if row["capability_id"] == "contextworld-action-strength"
                and row["family"] == "lewm"
            ),
            "portal_exit_float32_rescore": [
                row["rescore_evidence"]
                for row in cells
                if row["capability_id"] == "contextworld-portal-exit"
            ],
        },
        "counts": {
            "canonical_checkpoints": 8,
            "capabilities": 9,
            "icl_cells": 18,
            "completed_cells": len(cells),
            "passing_single_checkpoint_gates": sum(
                bool(row["metric"]["gate"]["passed"]) for row in cells
            ),
            "formal_scoreboard_eligible_cells": 0,
            "authorized_cem_jobs": 0,
            "cells_with_rescore_evidence": sum(
                row["rescore_evidence"] is not None for row in cells
            ),
            "cells_without_rescore_entrypoint": sum(
                row["rescore_evidence"] is None for row in cells
            ),
        },
        "cells": cells,
        "cem_followup": {
            "status": "pending_separate_component_bound_preregistration",
            "reason": (
                "CEM catalogs, horizons, budgets, and runtime pins are "
                "component-specific and are not authorized by this ICL matrix."
            ),
        },
    }


__all__ = [
    "DEFAULT_MATRIX_SUMMARY",
    "build_original_baseline_summary",
    "extract_cell_metric",
]
