#!/usr/bin/env python3
"""Freeze the terminal no-CEM branch for a failed Speed behavioral gate.

This is deliberately a receipt freezer, not an evaluator.  It can run only
after the immutable three-seed Speed Public-ICL aggregate says the behavioral
gate did not pass.  It never opens an evaluation payload, loads a model, or
starts a CEM planner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from contextworld.benchmarks.speed_pldm_infrastructure_development import (
    DEVELOPMENT_ID,
    DEVELOPMENT_SCOPE,
    EXPECTED_SEEDS,
)

ROOT = Path(__file__).resolve().parents[1]
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
RECOVERY_ID = "tworoom_speed_pldm_reference_completion_recovery_v1"
FORMAL_ROOT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1"
)
AGGREGATE_PATH = FORMAL_ROOT / "three_seed_aggregate.json"
DEFAULT_OUTPUT = FORMAL_ROOT / "cem_not_authorized_stop_v1.json"
PLANNED_CEM_ROOTS = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_action_planning_cem_v1",
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_original_tworoom_retention_cem_v1",
)
CLAIM_BOUNDARY = {
    "paired_single_speed_control_available": False,
    "training_attribution_claim": False,
    "public_test_reopened": False,
    "claim_level": "behavioral_trained_reference_only",
}


def _resolve(value: str | Path, *, label: str) -> Path:
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside the repository") from error
    return resolved


def _logical(path: Path, *, label: str) -> str:
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


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "path": _logical(path, label=label),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }


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


def _development_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not (
        isinstance(value, dict)
        and set(value) == {"path", "sha256", "size_bytes"}
        and isinstance(value.get("path"), str)
        and isinstance(value.get("sha256"), str)
        and isinstance(value.get("size_bytes"), int)
    ):
        raise ValueError(f"{label} is not an immutable file identity")
    observed = _identity(_resolve(value["path"], label=label), label=label)
    if observed != value:
        raise RuntimeError(f"{label} changed after Speed Development freeze")
    return observed


def _validate_development_chain(value: Any) -> dict[str, Any]:
    """Require the exact no-score Development chain before any CEM branch."""

    if not isinstance(value, dict) or set(value) != {"config", "manifest", "receipts"}:
        raise ValueError("Speed aggregate lacks Development chain")
    config = _development_identity(value["config"], label="development config")
    manifest = _development_identity(value["manifest"], label="development manifest")
    config_payload = _load_yaml(_resolve(config["path"], label="development config"))
    manifest_payload = _load_json(_resolve(manifest["path"], label="development manifest"))
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
        raise ValueError("Speed Development config/manifest chain is invalid")
    rows = value["receipts"]
    if not (
        isinstance(rows, list)
        and len(rows) == len(EXPECTED_SEEDS)
        and tuple(sorted(int(row.get("seed", -1)) for row in rows if isinstance(row, dict)))
        == EXPECTED_SEEDS
    ):
        raise ValueError("Speed aggregate lacks all three Development receipts")
    receipts = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        if not isinstance(row, dict) or set(row) != {"seed", "receipt"}:
            raise ValueError("Speed Development receipt declaration is invalid")
        seed = int(row["seed"])
        receipt_identity = _development_identity(
            row["receipt"], label=f"development receipt {seed}"
        )
        receipt = _load_json(
            _resolve(receipt_identity["path"], label=f"development receipt {seed}")
        )
        checks = receipt.get("checks", {})
        if not (
            receipt.get("development_id") == DEVELOPMENT_ID
            and receipt.get("completion_id") == COMPLETION_ID
            and int(receipt.get("seed", -1)) == seed
            and receipt.get("status") == "passed_infrastructure_readiness"
            and receipt.get("passed") is True
            and receipt.get("scope") == DEVELOPMENT_SCOPE
            and receipt.get("development_config") == config
            and receipt.get("development_manifest") == manifest
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
            raise ValueError(f"Speed Development receipt {seed} is invalid")
        receipts.append({"seed": seed, "receipt": receipt_identity})
    return {"config": config, "manifest": manifest, "receipts": receipts}


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


def _validate_aggregate_payload(payload: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
    """Validate the immutable failed aggregate without reading any Public data."""

    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError("Speed CEM stop requires exactly three ICL checkpoints")
    observed: dict[int, bool] = {}
    for row in checkpoints:
        if not isinstance(row, dict):
            raise ValueError("Speed ICL aggregate checkpoint must be an object")
        seed = row.get("training_seed")
        passed = row.get("passed")
        if isinstance(seed, bool) or not isinstance(seed, int) or type(passed) is not bool:
            raise ValueError("Speed ICL aggregate checkpoint lacks seed/gate")
        if seed in observed:
            raise ValueError("Speed ICL aggregate checkpoint seeds must be unique")
        observed[seed] = passed
    if set(observed) != {3072, 4096, 5120}:
        raise ValueError("Speed ICL aggregate must cover seeds 3072/4096/5120")
    decision = payload.get("decision")
    cem = payload.get("cem")
    boundary = payload.get("claim_boundary")
    boundary_identity = payload.get("behavioral_claim_boundary")
    development = _validate_development_chain(payload.get("development"))
    if not (
        payload.get("schema_version") == 1
        and payload.get("recovery_id") == RECOVERY_ID
        and payload.get("completion_id") == COMPLETION_ID
        and payload.get("status") == "completed"
        and payload.get("evaluation_kind") == "public_icl_recovery_aggregate"
        and payload.get("submission_kind") == "three_seed_method_recovery"
        and isinstance(decision, dict)
        and decision.get("formal_evaluation_completed") is True
        and decision.get("formal_method_claim") is False
        and decision.get("passed") is False
        and isinstance(decision.get("reason"), str)
        and decision["reason"].strip()
        and isinstance(cem, dict)
        and cem.get("authorized") is False
        and cem.get("executed") is False
        and isinstance(cem.get("reason"), str)
        and cem["reason"].strip()
        and boundary == CLAIM_BOUNDARY
        and isinstance(boundary_identity, dict)
        and set(boundary_identity) == {"path", "sha256", "size_bytes"}
        and isinstance(boundary_identity["path"], str)
        and isinstance(boundary_identity["sha256"], str)
        and len(boundary_identity["sha256"]) == 64
        and isinstance(boundary_identity["size_bytes"], int)
        and boundary_identity["size_bytes"] >= 0
        and not all(observed.values())
    ):
        raise ValueError("Speed failed ICL aggregate does not authorize a CEM stop")
    return sum(observed.values()), boundary_identity, development


def _assert_output(path: Path) -> Path:
    expected = DEFAULT_OUTPUT.resolve()
    actual = _resolve(path, label="stop output")
    if actual != expected:
        raise ValueError(
            "CEM-stop output must equal its dedicated destination "
            f"{_logical(expected, label='stop output')}"
        )
    return actual


def build_stop_receipt(aggregate_path: Path) -> dict[str, Any]:
    aggregate_path = _resolve(aggregate_path, label="Public ICL aggregate")
    if aggregate_path != AGGREGATE_PATH.resolve():
        raise ValueError("CEM-stop input must be the formal Speed ICL aggregate")
    if any(path.exists() for path in PLANNED_CEM_ROOTS):
        raise RuntimeError("CEM artifacts exist despite a failed Speed ICL gate")
    before_aggregate = _identity(aggregate_path, label="Public ICL aggregate")
    aggregate = _load_json(aggregate_path)
    passed_checkpoints, boundary, development = _validate_aggregate_payload(aggregate)
    boundary_path = _resolve(boundary["path"], label="behavioral claim boundary")
    observed_boundary = _identity(boundary_path, label="behavioral claim boundary")
    if observed_boundary != boundary:
        raise RuntimeError("Behavioral claim boundary changed after ICL aggregation")
    source = _identity(Path(__file__).resolve(), label="CEM-stop freezer source")
    before = {
        "public_icl_aggregate": before_aggregate,
        "behavioral_claim_boundary": observed_boundary,
        "development": development,
        "stop_freezer_source": source,
        "planned_cem_roots_absent": [
            _logical(path, label="planned CEM root") for path in PLANNED_CEM_ROOTS
        ],
    }
    after_aggregate = _identity(aggregate_path, label="Public ICL aggregate")
    after_boundary = _identity(boundary_path, label="behavioral claim boundary")
    after_payload = _load_json(aggregate_path)
    after_development = _validate_development_chain(after_payload.get("development"))
    after = {
        **before,
        "public_icl_aggregate": after_aggregate,
        "behavioral_claim_boundary": after_boundary,
        "development": after_development,
    }
    if before != after or any(path.exists() for path in PLANNED_CEM_ROOTS):
        raise RuntimeError("A CEM-stop input or planned CEM namespace changed during freeze")
    return {
        "schema_version": 1,
        "completion_id": COMPLETION_ID,
        "status": "frozen_cem_not_authorized_after_failed_three_seed_public_icl",
        "output": {
            "path": _logical(DEFAULT_OUTPUT, label="stop output"),
            "content_sha256_not_embedded_to_avoid_self_reference": True,
        },
        "public_icl_aggregate": before_aggregate,
        "public_icl": {
            "passed": False,
            "passed_checkpoints": passed_checkpoints,
            "evaluated_checkpoints": 3,
        },
        "cem": {"authorized": False, "executed": False},
        "behavioral_claim_boundary": observed_boundary,
        "development": development,
        "claim_boundary": CLAIM_BOUNDARY,
        "scope": {
            "model_evaluation_rerun_performed": False,
            "public_test_reopened": False,
            "action_planning_cem_executed": False,
            "original_tworoom_retention_cem_executed": False,
            "checkpoint_selection_performed": False,
        },
        "input_integrity": {
            "all_frozen_inputs_unchanged_during_stop_freeze": True,
            "identities_before_stop_freeze": before,
            "identities_after_stop_freeze": after,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=AGGREGATE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = _assert_output(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CEM-stop receipt: {output}")
    receipt = build_stop_receipt(args.aggregate)
    _write_exclusive(output, receipt)
    print(json.dumps({"status": receipt["status"], "output": receipt["output"]["path"]}))


if __name__ == "__main__":
    main()
