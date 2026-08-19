#!/usr/bin/env python3
"""Independently reconstruct and aggregate bound Speed PLDM Public ICL results.

This post-raw, pre-CEM recovery is additive.  It reads only the three raw
Public ICL JSON receipts, recomputes every stored track/horizon loss summary
from their retained records, and writes x-exclusive recovery receipts.  It
never loads a model, rewrites a raw result, or calls a planner/environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

import yaml

from contextworld.benchmarks.speed_icl_data import HORIZONS, load_speed_icl_release
from contextworld.benchmarks.speed_icl_score import _longest_contiguous, _loss_summary
from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_reference_completion_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
RECOVERY_NAMESPACE = Path(
    "artifacts/evaluation/history3/"
    "tworoom_speed_pldm_reference_completion_v1/"
    "formal_icl_v1/recovery_v1"
)
FORMAL_ICL_NAMESPACE = RECOVERY_NAMESPACE.parent
DEFAULT_PREREG = ROOT / "configs/benchmark/tworoom_speed_pldm_formal_icl_recovery_v1.yaml"
PRIMARY_TRACK = "unseen_interpolation"
PRIMARY_HORIZON = "1"
CLAIM_LEVEL = "behavioral_trained_reference_only"


def _resolve(value: str | Path, *, label: str = "path") -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error
    return resolved


def _logical(path: Path, *, label: str = "path") -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _file_identity(specification: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(specification, dict) or not isinstance(
        specification.get("path"), str
    ) or not isinstance(specification.get("sha256"), str):
        raise ValueError("A frozen file specification needs path and sha256")
    path = _resolve(specification["path"], label="frozen input")
    observed = _sha256(path) if path.is_file() else None
    return {
        "path": _logical(path, label="frozen input"),
        "sha256": observed,
        "expected_sha256": specification["sha256"],
        "observed_sha256": observed,
        "matched": observed == specification["sha256"],
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
    }


def _canonical_identity(specification: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Return the read-time identity for one frozen input, or fail closed."""

    identity = _file_identity(specification)
    if not identity["matched"]:
        raise RuntimeError(
            f"{label} changed: expected={identity['expected_sha256']}, "
            f"observed={identity['observed_sha256']}"
        )
    return {
        "path": identity["path"],
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _same_specification(value: Any, expected: dict[str, Any]) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("path") == expected.get("path")
        and value.get("sha256") == expected.get("sha256")
    )


def _claim_scope() -> dict[str, Any]:
    """The only claim scope this completion may emit at any formal stage."""

    return {
        "paired_single_speed_control_available": False,
        "training_attribution_claim": False,
        "public_test_reopened": False,
        "claim_level": CLAIM_LEVEL,
    }


def _validate_behavioral_claim_boundary(
    specification: dict[str, Any],
    *,
    release_id: str,
    completion_specification: dict[str, Any],
    release_specification: dict[str, Any],
    scorer_specification: dict[str, Any],
) -> dict[str, Any]:
    """Validate the frozen pre-evaluation claim boundary itself.

    A file identity alone is insufficient here: the recovery must reject a
    boundary that does not explicitly rule out paired-control and training
    attribution claims.  This function is intentionally Public-data-free.
    """

    identity = _canonical_identity(specification, label="behavioral claim boundary")
    payload = _load_yaml(
        _resolve(specification["path"], label="behavioral claim boundary")
    )
    chronology = payload.get("chronology")
    boundary = payload.get("claim_boundary")
    conditional = payload.get("conditional_evaluation")
    mutation = payload.get("mutation_boundary")
    frozen = payload.get("frozen_inputs")
    if not (
        payload.get("schema_version") == 1
        and payload.get("amendment_id")
        == "tworoom_speed_pldm_behavioral_claim_boundary_v1"
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("release_id") == release_id
        and payload.get("status")
        == "preregistered_during_fixed_training_before_development_or_public_evaluation"
        and chronology
        == {
            "fixed_training_already_running": True,
            "development_evaluation_started": False,
            "public_test_opened": False,
            "checkpoint_selection_changed": False,
            "training_budget_changed": False,
        }
        and isinstance(frozen, dict)
        and _same_specification(frozen.get("completion_config"), completion_specification)
        and _same_specification(frozen.get("speed_release"), release_specification)
        and _same_specification(frozen.get("public_scorer"), scorer_specification)
        and isinstance(boundary, dict)
        and boundary.get("paired_single_speed_pldm_controls_trained") is False
        and boundary.get("training_attribution_claim_authorized") is False
        and boundary.get("training_attributed_speed_icl_claim_authorized") is False
        and boundary.get("three_seed_behavioral_reference_authorized") is True
        and boundary.get("comparison_with_lewm_single_speed_controls_authorized")
        is False
        and boundary.get("cross_architecture_raw_latent_loss_comparison_authorized")
        is False
        and boundary.get("method_name_must_identify_pldm") is True
        and boundary.get("scoreboard_evidence_scope_if_reported") == "behavioral"
        and isinstance(boundary.get("behavioral_reference_requirement"), str)
        and boundary["behavioral_reference_requirement"].strip()
        and conditional
        == {
            "development_must_precede_public": True,
            "public_test_authorized_by_this_record": False,
            "public_test_requires_separate_passed_binding": True,
            "if_three_seed_public_behavioral_gate_passes": {
                "action_planning_cem_may_be_separately_authorized": True,
                "original_tworoom_retention_cem_may_be_separately_authorized": True,
            },
            "if_any_public_behavioral_gate_fails": {
                "cem_authorized": False,
                "cem_executed": False,
                "terminal_stop_receipt_required": True,
            },
        }
        and isinstance(mutation, dict)
        and mutation
        == {
            "training_or_checkpoint_selection_authorized": False,
            "optimizer_step_or_batch_change_authorized": False,
            "raw_data_mutation_authorized": False,
            "score_or_threshold_change_authorized": False,
            "public_test_access_authorized": False,
        }
    ):
        raise RuntimeError("Behavioral claim-boundary contract is invalid")
    return identity


def _path_identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": _logical(path, label=label),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _require_matched(specifications: dict[str, Any], *, label: str) -> None:
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError(f"{label} must be a non-empty mapping")
    for name, specification in specifications.items():
        identity = _file_identity(specification)
        if not identity["matched"]:
            raise RuntimeError(
                f"{label} changed: {name}; observed={identity['observed_sha256']}"
            )


def _entries(prereg: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = prereg.get("raw_public_icl", {}).get("checkpoints", [])
    entries = {int(row["seed"]): row for row in rows}
    if tuple(sorted(entries)) != (3072, 4096, 5120) or len(entries) != len(rows):
        raise ValueError("Speed recovery requires exactly seeds 3072/4096/5120")
    for seed, entry in entries.items():
        if (
            not isinstance(entry.get("checkpoint"), dict)
            or entry["checkpoint"].get("sha256") != entry.get("checkpoint_sha256")
            or not isinstance(entry.get("model_state_sha256"), str)
            or not isinstance(entry.get("raw_gate_passed"), bool)
        ):
            raise ValueError(f"Seed {seed} lacks a complete checkpoint/gate identity")
        identity = _file_identity(entry["checkpoint"])
        if not identity["matched"]:
            raise RuntimeError(
                f"Checkpoint changed for seed {seed}: {identity['observed_sha256']}"
            )
    return entries


def _output_root(prereg: dict[str, Any]) -> Path:
    outputs = prereg.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Recovery preregistration must declare outputs")
    root = _resolve(outputs["root"], label="recovery output root")
    if _logical(root, label="recovery output root") != RECOVERY_NAMESPACE.as_posix():
        raise ValueError("Recovery output root is not its dedicated namespace")
    return root


def _expected_seed_output(
    prereg: dict[str, Any], entries: dict[int, dict[str, Any]], seed: int
) -> tuple[Path, str]:
    if seed not in entries:
        raise ValueError(f"Seed {seed} is not preregistered")
    specification = entries[seed].get("recovery_receipt")
    if not isinstance(specification, dict) or not isinstance(specification.get("path"), str):
        raise ValueError(f"Seed {seed} lacks recovery output path")
    root = _output_root(prereg)
    path = _resolve(specification["path"], label="recovery output")
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("Recovery output is outside its dedicated namespace") from error
    if path.suffix != ".json":
        raise ValueError("Recovery output must be JSON")
    return path, _logical(path, label="recovery output")


def _expected_aggregate_output(prereg: dict[str, Any]) -> tuple[Path, str]:
    outputs = prereg.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("aggregate"), dict):
        raise ValueError("Recovery preregistration lacks aggregate output")
    path = _resolve(outputs["aggregate"]["path"], label="aggregate output")
    expected = (ROOT / FORMAL_ICL_NAMESPACE / "three_seed_aggregate.json").resolve()
    if path != expected:
        raise ValueError(
            "Aggregate output must equal the formal-ICL aggregate destination "
            f"{_logical(expected, label='aggregate output')}"
        )
    return path, _logical(path, label="aggregate output")


def _development_identities(prereg: dict[str, Any]) -> dict[str, Any]:
    """Read the three immutable no-score Development identities."""

    frozen = prereg["frozen_inputs"]
    required = {
        "development_config",
        "development_manifest",
        *(f"development_receipt_{seed}" for seed in EXPECTED_SEEDS),
    }
    if not required.issubset(frozen):
        raise ValueError("Recovery preregistration lacks frozen Development evidence")
    config = _canonical_identity(frozen["development_config"], label="development config")
    manifest = _canonical_identity(frozen["development_manifest"], label="development manifest")
    raw_entries = {
        int(row["seed"]): row
        for row in prereg.get("raw_public_icl", {}).get("checkpoints", [])
        if isinstance(row, dict) and isinstance(row.get("seed"), int)
    }
    if set(raw_entries) != set(EXPECTED_SEEDS):
        raise ValueError("Recovery preregistration lacks complete checkpoint identities")
    receipts = []
    for seed in EXPECTED_SEEDS:
        receipt = _canonical_identity(
            frozen[f"development_receipt_{seed}"],
            label=f"development receipt {seed}",
        )
        payload = _load_json(
            _resolve(
                frozen[f"development_receipt_{seed}"]["path"],
                label=f"development receipt {seed}",
            )
        )
        checks = payload.get("checks", {})
        expected_checkpoint = raw_entries[seed]
        if not (
            payload.get("development_id") == DEVELOPMENT_ID
            and payload.get("completion_id") == COMPLETION_ID
            and int(payload.get("seed", -1)) == seed
            and payload.get("status") == "passed_infrastructure_readiness"
            and payload.get("passed") is True
            and payload.get("scope") == DEVELOPMENT_SCOPE
            and payload.get("development_config") == config
            and payload.get("development_manifest") == manifest
            and payload.get("checkpoint", {}).get("sha256")
            == expected_checkpoint.get("checkpoint_sha256")
            and payload.get("checkpoint_model_state_sha256")
            == expected_checkpoint.get("model_state_sha256")
            and isinstance(checks, dict)
            and all(
                checks.get(name, {}).get("passed") is True
                for name in (
                    "strict_native_checkpoint_load",
                    "complete_heldout_manifest_coverage",
                    "prefix_autoregressive_geometry",
                    "native_future_latent_mse_finiteness",
                    "frozen_weight_audit",
                    "public_boundary",
                )
            )
            and checks.get("public_boundary", {}).get("public_payload_accessed")
            is False
            and checks.get("public_boundary", {}).get("checkpoint_selection")
            is False
        ):
            raise RuntimeError(f"Development receipt {seed} is not a passed no-score gate")
        receipts.append({"seed": seed, "receipt": receipt})
    config_payload = _load_yaml(
        _resolve(frozen["development_config"]["path"], label="development config")
    )
    manifest_payload = _load_json(
        _resolve(frozen["development_manifest"]["path"], label="development manifest")
    )
    if not (
        config_payload.get("development_id") == DEVELOPMENT_ID
        and config_payload.get("completion_id") == COMPLETION_ID
        and config_payload.get("scope") == DEVELOPMENT_SCOPE
        and manifest_payload.get("development_id") == DEVELOPMENT_ID
        and manifest_payload.get("completion_id") == COMPLETION_ID
        and manifest_payload.get("status") == "frozen_prepublic_development_manifest"
        and manifest_payload.get("passed") is True
        and manifest_payload.get("scope") == DEVELOPMENT_SCOPE
        and manifest_payload.get("development_config") == config
        and manifest_payload.get("public_payload_accessed") is False
    ):
        raise RuntimeError("Development config/manifest chain is not intact")
    return {"config": config, "manifest": manifest, "receipts": receipts}


def _assert_output(actual: Path, expected: Path, *, label: str) -> str:
    actual = _resolve(actual, label=label)
    if actual != expected:
        raise ValueError(
            f"{label} must equal preregistered destination "
            f"{_logical(expected, label=label)}, got {_logical(actual, label=label)}"
        )
    return _logical(actual, label=label)


def _validate_prereg(prereg_path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    prereg_path = _resolve(prereg_path, label="preregistration")
    prereg = _load_yaml(prereg_path)
    if not (
        prereg.get("schema_version") == 1
        and prereg.get("recovery_id") == RECOVERY_ID
        and prereg.get("completion_id") == COMPLETION_ID
        and prereg.get("status")
        == "preregistered_after_raw_public_icl_before_recovery"
    ):
        raise ValueError("Unexpected Speed recovery preregistration")
    required_inputs = {
        "completion_config",
        "evaluation_binding_config",
        "evaluation_binding_receipt",
        "release_config",
        "behavioral_claim_boundary",
        "development_config",
        "development_manifest",
        "development_receipt_3072",
        "development_receipt_4096",
        "development_receipt_5120",
    }
    frozen = prereg.get("frozen_inputs", {})
    if not required_inputs.issubset(frozen):
        raise ValueError("Recovery preregistration lacks frozen identities")
    _require_matched(frozen, label="Frozen recovery input")
    implementation = prereg.get("implementation")
    _require_matched(implementation, label="Implementation")
    if not isinstance(implementation, dict) or "frozen_icl_scorer" not in implementation:
        raise ValueError("Recovery preregistration lacks its frozen ICL scorer")
    boundary_identity = _validate_behavioral_claim_boundary(
        frozen["behavioral_claim_boundary"],
        release_id=str(prereg.get("release_id")),
        completion_specification=frozen["completion_config"],
        release_specification=frozen["release_config"],
        scorer_specification=implementation["frozen_icl_scorer"],
    )
    development = _development_identities(prereg)
    entries = _entries(prereg)
    destinations = set()
    for seed in entries:
        output, _ = _expected_seed_output(prereg, entries, seed)
        if output in destinations:
            raise ValueError("Recovery destinations must be unique")
        destinations.add(output)
    aggregate, _ = _expected_aggregate_output(prereg)
    if aggregate in destinations:
        raise ValueError("Aggregate output overlaps a recovery output")
    binding = _load_json(
        _resolve(frozen["evaluation_binding_receipt"]["path"], label="binding receipt")
    )
    if not (
        binding.get("status") == "passed_evaluation_binding_freeze"
        and binding.get("passed") is True
        and binding.get("binding", {}).get("sha256")
        == frozen["evaluation_binding_config"]["sha256"]
        and _same_specification(
            binding.get("behavioral_claim_boundary"),
            frozen["behavioral_claim_boundary"],
        )
        and binding.get("behavioral_claim_boundary") == boundary_identity
        and binding.get("claim_boundary") == _claim_scope()
        and binding.get("development") == development
    ):
        raise RuntimeError("Passed Speed evaluation binding is not intact")
    binding_config = _load_yaml(
        _resolve(frozen["evaluation_binding_config"]["path"], label="binding config")
    )
    if not (
        _same_specification(
            binding_config.get("behavioral_claim_boundary"),
            frozen["behavioral_claim_boundary"],
        )
        and _same_specification(binding_config.get("completion"), frozen["completion_config"])
        and _same_specification(binding_config.get("release"), frozen["release_config"])
        and isinstance(binding_config.get("development"), dict)
        and _same_specification(
            binding_config["development"].get("config"),
            frozen["development_config"],
        )
        and _same_specification(
            binding_config["development"].get("manifest"),
            frozen["development_manifest"],
        )
    ):
        raise RuntimeError("Speed binding config does not bind the behavioral boundary")
    bound_receipts = binding_config["development"].get("receipts")
    if not (
        isinstance(bound_receipts, list)
        and tuple(sorted(int(row.get("seed", -1)) for row in bound_receipts))
        == EXPECTED_SEEDS
        and all(
            _same_specification(
                next(
                    row["receipt"]
                    for row in bound_receipts
                    if int(row["seed"]) == seed
                ),
                frozen[f"development_receipt_{seed}"],
            )
            for seed in EXPECTED_SEEDS
        )
    ):
        raise RuntimeError("Speed binding config does not bind all Development receipts")
    return prereg, entries


def _reconstruct_tracks(
    raw: dict[str, Any], release: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    reconstructed: dict[str, dict[str, Any]] = {}
    for track in release["scope"]["public_tracks"]:
        source = raw.get("tracks", {}).get(track)
        if not isinstance(source, dict) or not isinstance(source.get("records"), list):
            raise RuntimeError(f"Raw result lacks records for track {track}")
        formal_eligible = bool(source.get("data", {}).get("full_protocol"))
        horizons: dict[str, dict[str, Any]] = {}
        passes: dict[str, bool] = {}
        for horizon in HORIZONS:
            key = str(horizon)
            summary = _loss_summary(source["records"], horizon)
            gate = bool(formal_eligible and summary["diagnostic_within_sample_pass"])
            horizons[key] = {
                **summary,
                "formal_protocol_eligible": formal_eligible,
                "formal_within_checkpoint_pass": gate if formal_eligible else None,
            }
            passes[key] = gate
        reconstructed[track] = {
            "horizons": horizons,
            "longest_contiguous_passing_horizon": (
                _longest_contiguous(passes) if formal_eligible else None
            ),
        }
    return reconstructed


def _primary_gate(reconstructed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    horizon = reconstructed[PRIMARY_TRACK]["horizons"][PRIMARY_HORIZON]
    value = horizon["formal_within_checkpoint_pass"]
    if not isinstance(value, bool):
        raise RuntimeError("Primary Speed gate is not formally eligible")
    return {
        "id": "unseen_in_range_one_step_strict_history_accuracy",
        "track": PRIMARY_TRACK,
        "horizon_action_blocks": int(PRIMARY_HORIZON),
        "value": float(
            horizon["reference_speed_balanced_strict_query_win_rate_vs_every_other"]
        ),
        "passed": value,
    }


def _snapshot(prereg_path: Path, prereg: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "preregistration": _path_identity(prereg_path, label="preregistration"),
        "frozen_inputs": {
            name: _file_identity(specification)
            for name, specification in prereg["frozen_inputs"].items()
        },
        "implementation": {
            name: _file_identity(specification)
            for name, specification in prereg["implementation"].items()
        },
        "checkpoint": _file_identity(entry["checkpoint"]),
        "raw_public_result": _file_identity(entry["raw_result"]),
    }


def _validate_raw(
    raw: dict[str, Any],
    *,
    entry: dict[str, Any],
    prereg: dict[str, Any],
    release: dict[str, Any],
) -> None:
    model = raw.get("model", {})
    completion = raw.get("completion_evaluation", {})
    raw_gate = completion.get("primary_gate", {})
    expected_binding = prereg["frozen_inputs"]
    expected_development = _development_identities(prereg)
    boundary_identity = _canonical_identity(
        expected_binding["behavioral_claim_boundary"],
        label="behavioral claim boundary",
    )
    expected_scope = {
        "public_icl_evaluated": True,
        "action_planning_cem_executed": False,
        "original_tworoom_retention_cem_executed": False,
        "checkpoint_selection_performed": False,
        **_claim_scope(),
    }
    if not (
        raw.get("schema_version") == 1
        and raw.get("benchmark") == release["release_id"]
        and raw.get("submission_kind") == "single_model"
        and raw.get("status") == "passed"
        and raw.get("full_protocol") is True
        and raw.get("release_config", {}).get("sha256")
        == expected_binding["release_config"]["sha256"]
        and model.get("training_seed") == entry["seed"]
        and model.get("training_role") == "multi_speed_target"
        and model.get("checkpoint_sha256") == entry["checkpoint_sha256"]
        and raw.get("frozen_weight_audit", {}).get("state_hash_before")
        == entry["model_state_sha256"]
        and raw.get("frozen_weight_audit", {}).get("state_hash_after")
        == entry["model_state_sha256"]
        and raw.get("frozen_weight_audit", {}).get("passed") is True
        and completion.get("completion_id") == COMPLETION_ID
        and completion.get("checkpoint", {}).get("sha256") == entry["checkpoint_sha256"]
        and completion.get("checkpoint_model_state_sha256")
        == entry["model_state_sha256"]
        and completion.get("binding", {}).get("sha256")
        == expected_binding["evaluation_binding_config"]["sha256"]
        and completion.get("binding_receipt", {}).get("sha256")
        == expected_binding["evaluation_binding_receipt"]["sha256"]
        and completion.get("development") == expected_development
        and completion.get("behavioral_claim_boundary") == boundary_identity
        and completion.get("scope") == expected_scope
        and raw_gate.get("id") == "unseen_in_range_one_step_strict_history_accuracy"
        and raw_gate.get("passed") == entry["raw_gate_passed"]
        and set(raw.get("tracks", {})) == set(release["scope"]["public_tracks"])
    ):
        raise RuntimeError(f"Raw formal ICL contract mismatch for seed {entry['seed']}")


def recover_one(
    prereg_path: Path,
    seed: int,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    prereg_path = _resolve(prereg_path, label="preregistration")
    prereg, entries = _validate_prereg(prereg_path)
    if seed not in entries:
        raise ValueError(f"Seed {seed} is not preregistered")
    expected_output, output_logical = _expected_seed_output(prereg, entries, seed)
    _assert_output(
        output_path if output_path is not None else expected_output,
        expected_output,
        label="recovery output",
    )
    entry = entries[seed]
    before = _snapshot(prereg_path, prereg, entry)
    if not before["raw_public_result"]["matched"]:
        raise RuntimeError("Raw Public ICL receipt changed before recovery")
    boundary_identity = _canonical_identity(
        prereg["frozen_inputs"]["behavioral_claim_boundary"],
        label="behavioral claim boundary",
    )
    development = _development_identities(prereg)
    release_path = _resolve(
        prereg["frozen_inputs"]["release_config"]["path"], label="release config"
    )
    release = load_speed_icl_release(release_path)
    raw_path = _resolve(entry["raw_result"]["path"], label="raw Public ICL result")
    raw = _load_json(raw_path)
    _validate_raw(raw, entry=entry, prereg=prereg, release=release)
    reconstructed = _reconstruct_tracks(raw, release)
    raw_summary = {
        track: {
            "horizons": raw["tracks"][track]["horizons"],
            "longest_contiguous_passing_horizon": raw["tracks"][track][
                "longest_contiguous_passing_horizon"
            ],
        }
        for track in release["scope"]["public_tracks"]
    }
    exact = reconstructed == raw_summary
    gate = _primary_gate(reconstructed)
    raw_gate = raw["completion_evaluation"]["primary_gate"]
    gate_exact = gate["passed"] == raw_gate["passed"] and gate["value"] == raw_gate["value"]
    if not (exact and gate_exact):
        raise RuntimeError("Speed track/horizon recovery did not exactly reproduce raw ICL")
    after = _snapshot(prereg_path, prereg, entry)
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("A frozen Speed recovery input changed while it was read")
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "completion_id": COMPLETION_ID,
        "release_id": prereg["release_id"],
        "status": "completed",
        "training_seed": seed,
        "checkpoint_sha256": entry["checkpoint_sha256"],
        "preregistration": before["preregistration"],
        "output_policy": {
            "namespace": RECOVERY_NAMESPACE.as_posix(),
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        },
        "output": {
            "path": output_logical,
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "behavioral_claim_boundary": boundary_identity,
        "development": development,
        "claim_boundary": _claim_scope(),
        "bindings": {
            "evaluation_binding_config": before["frozen_inputs"][
                "evaluation_binding_config"
            ],
            "evaluation_binding_receipt": before["frozen_inputs"][
                "evaluation_binding_receipt"
            ],
            "behavioral_claim_boundary": boundary_identity,
            "development": development,
            "release_config": before["frozen_inputs"]["release_config"],
            "checkpoint": before["checkpoint"],
            "raw_public_result": before["raw_public_result"],
        },
        "scope": {
            "model_evaluation_rerun_performed": False,
            "raw_public_result_rewritten": False,
            "public_test_reopened": False,
            "cem_executed": False,
            "exact_raw_record_reconstruction_only": True,
            **_claim_scope(),
        },
        "reconstruction": {
            "loss_record_dtype": "float64_json",
            "tracks": reconstructed,
            "primary_gate": gate,
        },
        "verification": {
            "passed": True,
            "gate_exact_equal": gate_exact,
            "all_track_horizon_metrics_exact": exact,
            "latent_summary_close": exact,
            "float32_scalar_aggregation_applicable": False,
            "float32_scalar_aggregates_bitwise_equal": None,
            "stored_model_gate_passed": raw_gate["passed"],
            "recomputed_model_gate_passed": gate["passed"],
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_recovery": unchanged,
            "identities_before_recovery_read": before,
            "identities_after_recovery_read": after,
        },
    }


def _identity_matches_spec(identity: Any, specification: dict[str, Any]) -> bool:
    observed = _file_identity(specification)
    return bool(
        observed["matched"]
        and isinstance(identity, dict)
        and identity.get("path") == observed["path"]
        and identity.get("expected_sha256") == observed["expected_sha256"]
        and identity.get("observed_sha256") == observed["observed_sha256"]
        and identity.get("matched") is True
        and identity.get("size_bytes") == observed["size_bytes"]
    )


def _validate_recovery_receipt(
    *,
    receipt: dict[str, Any],
    path: Path,
    prereg_path: Path,
    prereg: dict[str, Any],
    entries: dict[int, dict[str, Any]],
) -> int:
    try:
        seed = int(receipt["training_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Recovery receipt lacks training_seed") from error
    expected_path, expected_logical = _expected_seed_output(prereg, entries, seed)
    if path != expected_path:
        raise RuntimeError(f"Recovery receipt has noncanonical path for seed {seed}")
    entry = entries[seed]
    bindings = receipt.get("bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Recovery receipt bindings are missing")
    required_binding_specs = {
        "evaluation_binding_config": prereg["frozen_inputs"][
            "evaluation_binding_config"
        ],
        "evaluation_binding_receipt": prereg["frozen_inputs"][
            "evaluation_binding_receipt"
        ],
        "release_config": prereg["frozen_inputs"]["release_config"],
        "checkpoint": entry["checkpoint"],
        "raw_public_result": entry["raw_result"],
    }
    bindings_intact = all(
        _identity_matches_spec(bindings.get(name), specification)
        for name, specification in required_binding_specs.items()
    )
    boundary_intact = (
        bindings.get("behavioral_claim_boundary")
        == _canonical_identity(
            prereg["frozen_inputs"]["behavioral_claim_boundary"],
            label="behavioral claim boundary",
        )
    )
    expected_development = _development_identities(prereg)
    development_intact = (
        receipt.get("development") == expected_development
        and bindings.get("development") == expected_development
    )
    verification = receipt.get("verification")
    integrity = receipt.get("input_integrity")
    expected_scope = {
        "model_evaluation_rerun_performed": False,
        "raw_public_result_rewritten": False,
        "public_test_reopened": False,
        "cem_executed": False,
        "exact_raw_record_reconstruction_only": True,
        **_claim_scope(),
    }
    if not (
        receipt.get("schema_version") == 1
        and receipt.get("recovery_id") == RECOVERY_ID
        and receipt.get("completion_id") == COMPLETION_ID
        and receipt.get("release_id") == prereg.get("release_id")
        and receipt.get("status") == "completed"
        and receipt.get("checkpoint_sha256") == entry["checkpoint_sha256"]
        and receipt.get("preregistration")
        == _path_identity(prereg_path, label="preregistration")
        and receipt.get("output", {}).get("path") == expected_logical
        and receipt.get("output_policy", {}).get("namespace")
        == RECOVERY_NAMESPACE.as_posix()
        and receipt.get("output_policy", {}).get("exclusive_create_required") is True
        and bindings_intact
        and boundary_intact
        and development_intact
        and receipt.get("behavioral_claim_boundary")
        == _canonical_identity(
            prereg["frozen_inputs"]["behavioral_claim_boundary"],
            label="behavioral claim boundary",
        )
        and receipt.get("claim_boundary") == _claim_scope()
        and isinstance(verification, dict)
        and verification.get("passed") is True
        and verification.get("gate_exact_equal") is True
        and verification.get("all_track_horizon_metrics_exact") is True
        and verification.get("latent_summary_close") is True
        and verification.get("float32_scalar_aggregation_applicable") is False
        and verification.get("float32_scalar_aggregates_bitwise_equal") is None
        and isinstance(integrity, dict)
        and integrity.get("all_frozen_inputs_unchanged_during_recovery") is True
        and receipt.get("scope") == expected_scope
    ):
        raise RuntimeError(f"Recovery receipt is not intact for seed {seed}")
    return seed


def _aggregate_snapshot(
    prereg_path: Path, prereg: dict[str, Any], paths: dict[int, Path]
) -> dict[str, Any]:
    return {
        "preregistration": _path_identity(prereg_path, label="preregistration"),
        "frozen_inputs": {
            name: _file_identity(specification)
            for name, specification in prereg["frozen_inputs"].items()
        },
        "implementation": {
            name: _file_identity(specification)
            for name, specification in prereg["implementation"].items()
        },
        "recovery_receipts": {
            str(seed): _path_identity(path, label="recovery receipt")
            for seed, path in sorted(paths.items())
        },
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def aggregate(
    prereg_path: Path,
    recovery_paths: list[Path],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    prereg_path = _resolve(prereg_path, label="preregistration")
    prereg, entries = _validate_prereg(prereg_path)
    expected_output, output_logical = _expected_aggregate_output(prereg)
    _assert_output(
        output_path if output_path is not None else expected_output,
        expected_output,
        label="aggregate output",
    )
    if len(recovery_paths) != 3:
        raise ValueError("Three-seed aggregate requires exactly three receipts")
    paths = [_resolve(path, label="recovery receipt") for path in recovery_paths]
    if len(set(paths)) != 3:
        raise ValueError("Recovery receipt paths must be unique")
    recovered: dict[int, tuple[dict[str, Any], Path]] = {}
    for path in paths:
        receipt = _load_json(path)
        seed = _validate_recovery_receipt(
            receipt=receipt,
            path=path,
            prereg_path=prereg_path,
            prereg=prereg,
            entries=entries,
        )
        if seed in recovered:
            raise ValueError("Recovery receipt seeds must be unique")
        recovered[seed] = (receipt, path)
    if tuple(sorted(recovered)) != (3072, 4096, 5120):
        raise ValueError("Aggregate must contain the three preregistered seeds")
    before = _aggregate_snapshot(
        prereg_path, prereg, {seed: path for seed, (_, path) in recovered.items()}
    )
    checkpoints = []
    for seed in sorted(recovered):
        receipt, path = recovered[seed]
        gate = receipt["reconstruction"]["primary_gate"]
        checkpoints.append(
            {
                "training_seed": seed,
                "checkpoint_sha256": entries[seed]["checkpoint_sha256"],
                "recovery_receipt": _path_identity(path, label="recovery receipt"),
                "unseen_in_range_one_step_strict_history_accuracy": gate["value"],
                "value": gate["value"],
                "passed": bool(gate["passed"]),
            }
        )
    passed = all(row["passed"] for row in checkpoints)
    after = _aggregate_snapshot(
        prereg_path, prereg, {seed: path for seed, (_, path) in recovered.items()}
    )
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("An aggregate input changed while it was read")
    values = [float(row["value"]) for row in checkpoints]
    boundary_identity = _canonical_identity(
        prereg["frozen_inputs"]["behavioral_claim_boundary"],
        label="behavioral claim boundary",
    )
    development = _development_identities(prereg)
    return {
        "schema_version": 1,
        "recovery_id": RECOVERY_ID,
        "completion_id": COMPLETION_ID,
        "release_id": prereg["release_id"],
        "status": "completed",
        "evaluation_kind": "public_icl_recovery_aggregate",
        "submission_kind": "three_seed_method_recovery",
        "preregistration": before["preregistration"],
        "output_policy": {
            "namespace": FORMAL_ICL_NAMESPACE.as_posix(),
            "exclusive_create_required": True,
            "overwrite_permitted": False,
        },
        "output": {
            "path": output_logical,
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "metric": {
            "id": "unseen_in_range_one_step_strict_history_accuracy",
            "label": "训练范围内未见速度的一步严格正确率",
        },
        "behavioral_claim_boundary": boundary_identity,
        "development": development,
        "claim_boundary": _claim_scope(),
        "checkpoints": checkpoints,
        "aggregate": {"unseen_in_range_one_step_strict_history_accuracy": _stats(values)},
        "decision": {
            "formal_evaluation_completed": True,
            "formal_method_claim": False,
            "passed": passed,
            "reason": (
                "all_three_training_seeds_passed_behavioral_gate_without_training_attribution"
                if passed
                else "one_or_more_training_seeds_failed_behavioral_gate"
            ),
        },
        "cem": {
            "authorized": passed,
            "executed": False,
            "reason": (
                "authorized_only_after_three_seed_icl_gate"
                if passed
                else "not_authorized_because_three_seed_icl_gate_failed"
            ),
        },
        "input_integrity": {
            "all_aggregate_inputs_unchanged_during_read": unchanged,
            "identities_before_aggregate_read": before,
            "identities_after_aggregate_read": after,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    commands = parser.add_subparsers(dest="command", required=True)
    recover = commands.add_parser("recover")
    recover.add_argument("--seed", type=int, required=True)
    recover.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--input", action="append", type=Path, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg_path = _resolve(args.preregistration, label="preregistration")
    output = _resolve(args.output, label="output")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recovery output: {output}")
    if args.command == "recover":
        payload = recover_one(
            prereg_path,
            int(args.seed),
            output_path=output,
        )
    else:
        payload = aggregate(
            prereg_path,
            [_resolve(path, label="recovery receipt") for path in args.input],
            output_path=output,
        )
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": payload["output"]["path"],
                "decision": payload.get("decision"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
