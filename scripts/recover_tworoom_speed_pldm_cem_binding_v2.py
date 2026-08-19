#!/usr/bin/env python3
"""Recover Speed PLDM's CEM binding with one frozen-schema compatibility shim.

Recovery v1 correctly substituted the passed evaluation-binding receipt, then
stopped before writing because the original freezer's literal baseline schema
omits the truthful additive field ``strictly_reused_episodes``.  This v2
launcher preserves every frozen file and identity.  While the unchanged
freezer validates the baseline, it presents an in-memory compatibility view
with exactly that one field removed; the canonical binding still snapshots
and retains the original file identity.

No model, environment, planner, Public Test payload, or CEM evaluation is
executed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

import scripts.freeze_tworoom_speed_pldm_cem_binding_v1 as original_freezer
from scripts.recover_tworoom_speed_pldm_cem_binding_v1 import (
    validate_preregistration as validate_v1_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_cem_binding_recovery_v2"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
DEFAULT_PREREGISTRATION = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_cem_binding_recovery_v2.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1/cem_binding_v1.json"
)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _logical(path: Path) -> str:
    return original_freezer.logical_path(path.resolve(), repo_root=ROOT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _logical(path),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _same_identity(observed: Any, expected: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(observed, Mapping)
        and observed.get("path") == expected.get("path")
        and observed.get("sha256") == expected.get("sha256")
        and observed.get("size_bytes") == expected.get("size_bytes")
    )


def _require_identity(specification: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(specification, Mapping) or not isinstance(
        specification.get("path"), str
    ):
        raise ValueError(f"{label} lacks a frozen identity")
    observed = _identity(original_freezer.resolve_source(specification["path"], repo_root=ROOT))
    if not _same_identity(observed, specification):
        raise RuntimeError(f"{label} identity drifted")
    return observed


def _expected_freeze_summary(matrix_summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(matrix_summary),
        "summary_id": "contextworld_original_baseline_cem_matrix_v1",
        "status": "completed_descriptive_original_environment_cem_matrix",
        "matrix_cells": 8,
        "episodes_per_cell": 300,
        "total_matrix_episodes": 2400,
        "newly_executed_cells": 7,
        "newly_executed_episodes": 2100,
        "strictly_reused_cells": 1,
        "receipt_identities_embedded_in_summary": True,
        "all_model_state_audits_passed": True,
    }


def validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_yaml(path)
    frozen = preregistration.get("frozen_inputs")
    compatibility = preregistration.get("compatibility_view")
    implementation = preregistration.get("implementation")
    if not (
        preregistration.get("schema_version") == 1
        and preregistration.get("recovery_id") == RECOVERY_ID
        and preregistration.get("completion_id") == COMPLETION_ID
        and preregistration.get("status")
        == "registered_after_recovery_v1_validation_failure_before_cem_binding_or_execution"
        and preregistration.get("scope")
        == {
            "changes_frozen_file_content_or_identity": False,
            "changes_cem_runner_metric_or_threshold": False,
            "changes_checkpoint_or_model": False,
            "reopens_public_test": False,
            "executes_model_environment_or_planner": False,
            "executes_cem": False,
            "normalizes_one_additive_schema_field_in_memory": True,
        }
        and isinstance(frozen, Mapping)
        and isinstance(compatibility, Mapping)
        and isinstance(implementation, Mapping)
    ):
        raise ValueError("Unexpected Speed CEM-binding recovery-v2 preregistration")
    expected_frozen = {
        "recovery_v1_preregistration",
        "recovery_v1_launcher",
        "recovery_v1_failure",
        "original_cem_binding_freezer",
        "original_baseline_results_freeze",
        "original_baseline_matrix_summary",
    }
    if set(frozen) != expected_frozen:
        raise ValueError("Recovery-v2 frozen-input set is incomplete")
    observed = {
        name: _require_identity(specification, label=name)
        for name, specification in frozen.items()
    }
    launcher = _require_identity(
        implementation.get("recovery_launcher", {}), label="recovery-v2 launcher"
    )
    if launcher["path"] != _logical(Path(__file__)):
        raise RuntimeError("Recovery-v2 preregistration binds a different launcher")

    v1_path = _resolve(frozen["recovery_v1_preregistration"]["path"])
    validate_v1_preregistration(v1_path)
    failure = _load_json(_resolve(frozen["recovery_v1_failure"]["path"]))
    if not (
        failure.get("status") == "failed_before_cem_binding_or_execution"
        and failure.get("boundary", {}).get("cem_binding_written") is False
        and failure.get("boundary", {}).get("action_planning_cem_executed") is False
        and failure.get("boundary", {}).get("original_task_retention_cem_executed")
        is False
        and failure.get("failure", {}).get("stage")
        == "frozen_original_baseline_retention_chain_validation"
    ):
        raise RuntimeError("Recovery-v1 failure evidence is not intact")

    results_freeze_path = original_freezer.resolve_source(
        frozen["original_baseline_results_freeze"]["path"], repo_root=ROOT
    )
    matrix_summary_path = original_freezer.resolve_source(
        frozen["original_baseline_matrix_summary"]["path"], repo_root=ROOT
    )
    results_freeze = _load_json(results_freeze_path)
    matrix_summary = _load_json(matrix_summary_path)
    static, _ = original_freezer._validate_static_prereg(original_freezer.CEM_PREREG)
    matrix_specification = static["tracks"]["original_task_retention_cem"][
        "frozen_paired_baseline"
    ]["matrix_summary"]
    actual = results_freeze.get("matrix_summary")
    if not isinstance(actual, dict):
        raise RuntimeError("Original baseline results freeze lacks matrix_summary")
    normalized = dict(actual)
    removed = normalized.pop("strictly_reused_episodes", None)
    if not (
        compatibility
        == {
            "file": frozen["original_baseline_results_freeze"],
            "mapping": "matrix_summary",
            "field": "strictly_reused_episodes",
            "required_value": 300,
            "operation": "omit_from_in_memory_validator_view_only",
            "underlying_file_rewritten": False,
            "binding_snapshot_uses_original_file_identity": True,
        }
        and removed == 300
        and normalized == _expected_freeze_summary(matrix_specification)
        and matrix_summary.get("summary_id")
        == "contextworld_original_baseline_cem_matrix_v1"
        and matrix_summary.get("status")
        == "completed_descriptive_original_environment_cem_matrix"
    ):
        raise RuntimeError("Baseline mismatch is broader than the registered additive field")
    return preregistration, observed


def build_binding(
    *, preregistration_path: Path, preregistration: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    v1_preregistration = _load_yaml(
        _resolve(preregistration["frozen_inputs"]["recovery_v1_preregistration"]["path"])
    )
    replacement = _resolve(
        v1_preregistration["bounded_substitution"]["replacement"]["path"]
    )
    results_freeze_path = original_freezer.resolve_source(
        preregistration["frozen_inputs"]["original_baseline_results_freeze"]["path"],
        repo_root=ROOT,
    ).resolve()
    original_receipt_path = original_freezer.EVALUATION_BINDING_RECEIPT
    original_load_json = original_freezer._load_json

    def compatibility_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if Path(path).resolve() == results_freeze_path:
            payload = copy.deepcopy(payload)
            matrix = payload.get("matrix_summary")
            if not isinstance(matrix, dict) or matrix.pop("strictly_reused_episodes", None) != 300:
                raise RuntimeError("Registered baseline compatibility field changed")
        return payload

    try:
        original_freezer.EVALUATION_BINDING_RECEIPT = replacement
        original_freezer._load_json = compatibility_load_json
        payload = original_freezer.build_binding(original_freezer.CEM_PREREG)
    finally:
        original_freezer.EVALUATION_BINDING_RECEIPT = original_receipt_path
        original_freezer._load_json = original_load_json
    if not (
        payload.get("status") == "frozen_after_passed_three_seed_public_icl_before_cem"
        and payload.get("passed") is True
        and payload.get("cem") == {"authorized": True, "executed": False}
        and payload.get("scope", {}).get("model_or_environment_execution_performed")
        is False
        and payload.get("scope", {}).get("action_planning_cem_executed") is False
        and payload.get("scope", {}).get("original_tworoom_retention_cem_executed")
        is False
    ):
        raise RuntimeError("Recovered CEM binding is not a closed positive branch")
    payload["binding_recovery"] = {
        "recovery_id": RECOVERY_ID,
        "preregistration": _identity(preregistration_path),
        "recovery_v1_failure": observed["recovery_v1_failure"],
        "unchanged_original_cem_binding_freezer": observed[
            "original_cem_binding_freezer"
        ],
        "evaluation_binding_receipt_substitution_inherited_from_recovery_v1": True,
        "compatibility_view": {
            "source": observed["original_baseline_results_freeze"],
            "omitted_field_in_memory": "matrix_summary.strictly_reused_episodes",
            "required_value": 300,
            "underlying_file_rewritten": False,
            "binding_snapshot_uses_original_file_identity": True,
        },
        "prepublic_cem_protocol_changed": False,
        "model_or_environment_execution_performed": False,
        "public_test_reopened": False,
        "cem_executed": False,
    }
    return payload


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    preregistration_path = _resolve(args.preregistration)
    output = _resolve(args.output)
    if output != DEFAULT_OUTPUT:
        raise ValueError("CEM binding output must use its canonical destination")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CEM binding: {output}")
    preregistration, observed = validate_preregistration(preregistration_path)
    payload = build_binding(
        preregistration_path=preregistration_path,
        preregistration=preregistration,
        observed=observed,
    )
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": _logical(output),
                "cem_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
