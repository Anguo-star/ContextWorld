#!/usr/bin/env python3
"""Recover the canonical Speed PLDM CEM binding after a binding-receipt recovery.

The pre-Public CEM freezer and every planning protocol remain unchanged.  Its
only stale assumption is the default path of the evaluation-binding receipt,
which now contains a preserved failed attempt.  This launcher validates that
bounded history, substitutes the additive passed receipt while the unchanged
freezer builds its in-memory payload, and writes the canonical CEM binding.

No model, environment, planner, Public Test payload, or CEM evaluation is
executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

import scripts.freeze_tworoom_speed_pldm_cem_binding_v1 as original_freezer


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_cem_binding_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
DEFAULT_PREREGISTRATION = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_cem_binding_recovery_v1.yaml"
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
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


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
    observed = _identity(_resolve(specification["path"]))
    if not _same_identity(observed, specification):
        raise RuntimeError(f"{label} identity drifted")
    return observed


def validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_yaml(path)
    frozen = preregistration.get("frozen_inputs")
    substitution = preregistration.get("bounded_substitution")
    implementation = preregistration.get("implementation")
    if not (
        preregistration.get("schema_version") == 1
        and preregistration.get("recovery_id") == RECOVERY_ID
        and preregistration.get("completion_id") == COMPLETION_ID
        and preregistration.get("status")
        == "registered_after_passed_public_icl_aggregate_before_cem_binding_or_execution"
        and preregistration.get("scope")
        == {
            "changes_cem_runner_or_metric": False,
            "changes_checkpoint_or_model": False,
            "reopens_public_test": False,
            "executes_model_environment_or_planner": False,
            "executes_cem": False,
            "substitutes_only_evaluation_binding_receipt_path": True,
        }
        and isinstance(frozen, Mapping)
        and isinstance(substitution, Mapping)
        and isinstance(implementation, Mapping)
    ):
        raise ValueError("Unexpected Speed CEM-binding recovery preregistration")
    expected_frozen = {
        "cem_preregistration",
        "formal_icl_recovery_preregistration",
        "formal_icl_aggregate",
        "failed_evaluation_binding_receipt",
        "passed_evaluation_binding_receipt",
        "original_cem_binding_freezer",
    }
    if set(frozen) != expected_frozen:
        raise ValueError("CEM-binding recovery frozen-input set is incomplete")
    observed = {
        name: _require_identity(specification, label=name)
        for name, specification in frozen.items()
    }
    launcher = _require_identity(
        implementation.get("recovery_launcher", {}), label="recovery launcher"
    )
    if launcher["path"] != _logical(Path(__file__)):
        raise RuntimeError("CEM-binding recovery binds a different launcher")

    failed = _load_json(_resolve(frozen["failed_evaluation_binding_receipt"]["path"]))
    passed = _load_json(_resolve(frozen["passed_evaluation_binding_receipt"]["path"]))
    aggregate = _load_json(_resolve(frozen["formal_icl_aggregate"]["path"]))
    if not (
        failed.get("status") == "failed_evaluation_binding_freeze"
        and failed.get("passed") is False
        and failed.get("public_test", {}).get("accessed_by_binding") is False
        and passed.get("status") == "passed_evaluation_binding_freeze"
        and passed.get("passed") is True
        and passed.get("binding_freeze_recovery", {}).get("public_test_accessed")
        is False
        and aggregate.get("status") == "completed"
        and aggregate.get("decision", {}).get("passed") is True
        and aggregate.get("decision", {}).get("formal_method_claim") is False
        and aggregate.get("cem") == {
            "authorized": True,
            "executed": False,
            "reason": "authorized_only_after_three_seed_icl_gate",
        }
        and all(row.get("passed") is True for row in aggregate.get("checkpoints", []))
        and len(aggregate.get("checkpoints", [])) == 3
    ):
        raise RuntimeError("CEM-binding recovery lacks a passed behavioral ICL chain")
    if not (
        substitution.get("stale_default")
        == frozen["failed_evaluation_binding_receipt"]
        and substitution.get("replacement")
        == frozen["passed_evaluation_binding_receipt"]
        and substitution.get("original_freezer_source_unchanged") is True
        and substitution.get("canonical_cem_binding_destination_unchanged") is True
        and substitution.get("all_prepublic_cem_sources_and_decisions_unchanged") is True
    ):
        raise ValueError("CEM-binding recovery substitution is broader than registered")
    if _resolve(original_freezer.EVALUATION_BINDING_RECEIPT) != _resolve(
        substitution["stale_default"]["path"]
    ):
        raise RuntimeError("Original CEM freezer no longer has the registered stale default")
    if _resolve(preregistration.get("output", {}).get("cem_binding", "")) != DEFAULT_OUTPUT:
        raise ValueError("Recovered CEM binding must use the canonical destination")
    return preregistration, observed


def build_binding(
    *, preregistration_path: Path, preregistration: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    replacement = _resolve(
        preregistration["bounded_substitution"]["replacement"]["path"]
    )
    stale = original_freezer.EVALUATION_BINDING_RECEIPT
    try:
        original_freezer.EVALUATION_BINDING_RECEIPT = replacement
        payload = original_freezer.build_binding(original_freezer.CEM_PREREG)
    finally:
        original_freezer.EVALUATION_BINDING_RECEIPT = stale
    chain_receipt = payload.get("frozen_chain", {}).get("evaluation_binding_receipt")
    if chain_receipt != observed["passed_evaluation_binding_receipt"]:
        raise RuntimeError("Recovered CEM binding did not retain the passed receipt")
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
        raise RuntimeError("Unchanged CEM freezer did not produce a closed positive branch")
    payload["binding_recovery"] = {
        "recovery_id": RECOVERY_ID,
        "preregistration": _identity(preregistration_path),
        "unchanged_original_cem_binding_freezer": observed[
            "original_cem_binding_freezer"
        ],
        "stale_failed_evaluation_binding_receipt": observed[
            "failed_evaluation_binding_receipt"
        ],
        "substituted_passed_evaluation_binding_receipt": observed[
            "passed_evaluation_binding_receipt"
        ],
        "formal_icl_aggregate": observed["formal_icl_aggregate"],
        "substitution_scope": "evaluation_binding_receipt_path_only",
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
