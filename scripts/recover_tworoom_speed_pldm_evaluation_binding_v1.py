#!/usr/bin/env python3
"""Recover the Speed PLDM evaluation binding without rewriting prior evidence.

The original binding attempt failed for two bounded reasons: the repository
copy of a frozen retention catalog was absent, and the original freezer
mistook preflight-only receipts for training-completion receipts.  This
recovery copies the byte-identical catalog from its frozen archive, reruns the
unchanged original freezer in memory, and validates training completion from
the Development manifest's recovery receipts and terminal reports.

No model, optimizer, Public Test payload, scorer, planner, or environment is
executed by this program.  Both the failed receipt and the original freezer
remain immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from scripts.freeze_tworoom_speed_pldm_evaluation_binding_v1 import (
    build_receipt as build_original_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_evaluation_binding_recovery_v1"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
EXPECTED_SEEDS = (3072, 4096, 5120)
EXPECTED_ORIGINAL_FAILURES = {
    "evaluator_source_cem_retention_catalog",
    "prepublic_cem_source_retention_catalog",
    *(f"training_preflight_{seed}" for seed in EXPECTED_SEEDS),
}
EXPECTED_POST_CATALOG_FAILURES = {
    f"training_preflight_{seed}" for seed in EXPECTED_SEEDS
}
DEFAULT_PREREGISTRATION = (
    ROOT
    / "configs/benchmark/tworoom_speed_pldm_evaluation_binding_recovery_v1.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "evaluation_binding_v1/recovery_v1/evaluation_binding_receipt.json"
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _same_identity(observed: Any, expected: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(observed, Mapping)
        and observed.get("path") == expected.get("path")
        and observed.get("sha256") == expected.get("sha256")
        and (
            "size_bytes" not in expected
            or observed.get("size_bytes") == expected.get("size_bytes")
        )
    )


def _require_identity(specification: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(specification.get("path"), str) or not isinstance(
        specification.get("sha256"), str
    ):
        raise ValueError(f"{label} lacks a frozen path/hash")
    observed = _identity(_resolve(specification["path"]))
    if not _same_identity(observed, specification):
        raise RuntimeError(
            f"{label} identity changed: expected={dict(specification)}, observed={observed}"
        )
    return observed


def _failed_checks(receipt: Mapping[str, Any]) -> set[str]:
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Binding receipt lacks checks")
    return {
        str(name)
        for name, row in checks.items()
        if not isinstance(row, Mapping) or row.get("passed") is not True
    }


def validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_yaml(path)
    frozen = preregistration.get("frozen_inputs")
    repair = preregistration.get("bounded_repair")
    implementation = preregistration.get("implementation")
    outputs = preregistration.get("outputs")
    if not (
        preregistration.get("schema_version") == 1
        and preregistration.get("recovery_id") == RECOVERY_ID
        and preregistration.get("completion_id") == COMPLETION_ID
        and preregistration.get("status")
        == "preregistered_after_failed_binding_before_catalog_restore_or_public_evaluation"
        and isinstance(frozen, Mapping)
        and isinstance(repair, Mapping)
        and isinstance(implementation, Mapping)
        and isinstance(outputs, Mapping)
        and preregistration.get("public_boundary")
        == {
            "public_test_accessed": False,
            "formal_icl_executed": False,
            "cem_executed": False,
            "training_or_optimizer_execution_authorized": False,
            "checkpoint_selection_authorized": False,
        }
    ):
        raise ValueError("Unexpected evaluation-binding recovery preregistration")

    required_frozen = {
        "evaluation_binding_config",
        "failed_binding_receipt",
        "original_binding_freezer",
        "development_manifest",
        "archived_retention_catalog",
    }
    if set(frozen) != required_frozen:
        raise ValueError("Recovery preregistration has an incomplete frozen-input set")
    observed = {
        name: _require_identity(specification, label=name)
        for name, specification in frozen.items()
    }
    launcher = _require_identity(implementation.get("recovery_launcher", {}), label="recovery launcher")
    if launcher["path"] != _logical(Path(__file__)):
        raise RuntimeError("Recovery preregistration binds a different launcher")

    failed = _load_json(_resolve(frozen["failed_binding_receipt"]["path"]))
    binding_identity = observed["evaluation_binding_config"]
    if not (
        failed.get("status") == "failed_evaluation_binding_freeze"
        and failed.get("passed") is False
        and failed.get("public_test", {}).get("accessed_by_binding") is False
        and failed.get("public_test", {}).get("scored_by_binding") is False
        and failed.get("next_stage", {}).get("formal_public_icl_authorized") is False
        and failed.get("binding", {}).get("path") == binding_identity["path"]
        and failed.get("binding", {}).get("sha256") == binding_identity["sha256"]
        and _failed_checks(failed) == EXPECTED_ORIGINAL_FAILURES
    ):
        raise RuntimeError("Original failed receipt is not the bounded recovery source")

    catalog = repair.get("catalog_restore")
    preflight = repair.get("preflight_completion_semantics")
    if not (
        repair.get("only_original_failed_checks_may_be_repaired") is True
        and set(repair.get("original_failed_checks", [])) == EXPECTED_ORIGINAL_FAILURES
        and isinstance(catalog, Mapping)
        and catalog.get("source") == frozen["archived_retention_catalog"]
        and catalog.get("exact_byte_copy_required") is True
        and catalog.get("content_regeneration_authorized") is False
        and isinstance(catalog.get("destination"), Mapping)
        and catalog["destination"].get("sha256")
        == frozen["archived_retention_catalog"].get("sha256")
        and catalog["destination"].get("size_bytes")
        == frozen["archived_retention_catalog"].get("size_bytes")
        and isinstance(preflight, Mapping)
        and preflight.get("preflight_receipt_role")
        == "preflight_only_not_training_completion_evidence"
        and preflight.get("historical_preflight_immutability_claimed") is False
        and preflight.get("completion_evidence")
        == "recovery_completion_receipt_and_final_training_report"
        and preflight.get("required_optimizer_steps") == 12840
        and preflight.get("terminal_report_recovery_optimizer_steps") == 0
    ):
        raise ValueError("Recovery preregistration does not describe the bounded repair")

    expected_output = _resolve(outputs.get("recovered_binding_receipt", ""))
    if expected_output != DEFAULT_OUTPUT.resolve():
        raise ValueError("Recovered receipt output is outside its dedicated namespace")
    return preregistration, observed


def _copy_catalog(preregistration: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    catalog = preregistration["bounded_repair"]["catalog_restore"]
    source = _resolve(catalog["source"]["path"])
    destination = _resolve(catalog["destination"]["path"])
    expected = catalog["destination"]
    if destination.is_file():
        observed = _identity(destination)
        if not _same_identity(observed, expected):
            raise RuntimeError("Existing retention catalog does not match the frozen archive")
        return observed, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o644)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            for block in iter(lambda: input_stream.read(8 * 1024 * 1024), b""):
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    observed = _identity(destination)
    if not _same_identity(observed, expected):
        destination.unlink(missing_ok=True)
        raise RuntimeError("Copied retention catalog does not match its frozen identity")
    return observed, False


def _validate_training_completion(
    *,
    binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    entries = {
        int(row["seed"]): row
        for row in binding.get("checkpoints", [])
        if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
    }
    manifest_entries = {
        int(row["seed"]): row
        for row in manifest.get("training_checkpoints", [])
        if isinstance(row, Mapping) and isinstance(row.get("seed"), int)
    }
    if set(entries) != set(EXPECTED_SEEDS) or set(manifest_entries) != set(EXPECTED_SEEDS):
        raise RuntimeError("Binding/manifest training seed set is incomplete")
    entry = entries[seed]
    frozen = manifest_entries[seed]
    preflight_path = _resolve(entry["preflight"]["path"])
    report_path = _resolve(entry["training_report"]["path"])
    completion_specification = frozen.get("recovery_completion_receipt")
    if not isinstance(completion_specification, Mapping):
        raise RuntimeError(f"Seed {seed} lacks recovery completion evidence")
    completion_path = _resolve(completion_specification["path"])
    preflight_identity = _identity(preflight_path)
    report_identity = _identity(report_path)
    completion_identity = _identity(completion_path)
    preflight = _load_json(preflight_path)
    report = _load_json(report_path)
    completion = _load_json(completion_path)
    fixed_contract = {
        "passed": True,
        "checkpoint_selection": "final_fixed_step",
        "early_stopping": False,
        "optimizer_steps": 12840,
        "completion_evidence": "recovery_completion_receipt_and_final_training_report",
        "preflight_receipt_role": "preflight_only_not_training_completion_evidence",
        "historical_preflight_immutability_claimed": False,
    }
    if not (
        frozen.get("checkpoint") == {
            key: value for key, value in entry["checkpoint"].items() if key != "model_state_sha256"
        }
        and frozen.get("checkpoint_config") == entry["config"]
        and frozen.get("training_report") == entry["training_report"]
        and frozen.get("loss_trace") == entry["loss_trace"]
        and frozen.get("preflight") == entry["preflight"]
        and frozen.get("fixed_training_contract") == fixed_contract
        and _same_identity(preflight_identity, entry["preflight"])
        and _same_identity(report_identity, entry["training_report"])
        and _same_identity(completion_identity, completion_specification)
        and preflight.get("schema_version") == 1
        and preflight.get("completion_id") == COMPLETION_ID
        and preflight.get("status") == "passed"
        and preflight.get("seed") == seed
        and preflight.get("training_started") is False
        and "training_completed" not in preflight
        and "training_failed" not in preflight
        and report.get("schema_version") == 1
        and report.get("passed") is True
        and report.get("training", {}).get("training_complete") is True
        and report.get("training", {}).get("global_step") == 12840
        and report.get("training", {}).get("expected_optimizer_steps") == 12840
        and report.get("training", {}).get("terminal_report_recovery_optimizer_steps") == 0
        and report.get("terminal_report_recovery", {}).get(
            "training_or_optimizer_execution"
        )
        is False
        and report.get("artifacts", {}).get("pretrained_sha256")
        == entry["checkpoint"]["sha256"]
        and report.get("artifacts", {}).get("loss_trace", {}).get("sha256")
        == entry["loss_trace"]["sha256"]
        and completion.get("schema_version") == 1
        and completion.get("completion_id") == COMPLETION_ID
        and completion.get("seed") == seed
        and completion.get("status") == "completed_fixed_budget_required_resume"
        and completion.get("passed") is True
        and completion.get("training_report") == report_identity
        and completion.get("final_checkpoint")
        == {key: value for key, value in entry["checkpoint"].items() if key != "model_state_sha256"}
        and completion.get("resume_proof", {}).get("initial_global_step") == 10272
        and completion.get("resume_proof", {}).get("final_global_step") == 12840
        and completion.get("resume_proof", {}).get("remaining_optimizer_steps_executed")
        == 2568
        and completion.get("evaluation_executed") is False
        and completion.get("public_test_accessed") is False
    ):
        raise RuntimeError(f"Seed {seed} does not satisfy the recovered completion contract")
    return {
        "passed": True,
        "preflight_receipt_role": "preflight_only_not_training_completion_evidence",
        "preflight": preflight,
        "completion_evidence": {
            "development_manifest": _identity(
                _resolve(binding["development"]["manifest"]["path"])
            ),
            "recovery_completion_receipt": completion_identity,
            "final_training_report": report_identity,
            "fixed_optimizer_steps": 12840,
            "terminal_report_recovery_optimizer_steps": 0,
        },
    }


def build_recovered_receipt(
    *, preregistration_path: Path, preregistration: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    binding_path = _resolve(
        preregistration["frozen_inputs"]["evaluation_binding_config"]["path"]
    )
    binding = _load_yaml(binding_path)
    manifest = _load_json(
        _resolve(preregistration["frozen_inputs"]["development_manifest"]["path"])
    )
    receipt = build_original_receipt(binding_path)
    failures = _failed_checks(receipt)
    if failures != EXPECTED_POST_CATALOG_FAILURES:
        raise RuntimeError(
            "Unchanged original freezer has failures outside the preregistered "
            f"preflight correction: {sorted(failures)}"
        )
    training_evidence = {}
    for seed in EXPECTED_SEEDS:
        evidence = _validate_training_completion(binding=binding, manifest=manifest, seed=seed)
        receipt["checks"][f"training_preflight_{seed}"] = evidence
        receipt["checks"][f"training_completion_recovery_{seed}"] = {
            "passed": True,
            **evidence["completion_evidence"],
        }
        training_evidence[str(seed)] = evidence["completion_evidence"]
    if _failed_checks(receipt):
        raise RuntimeError("Recovered receipt still contains failed checks")

    receipt["passed"] = True
    receipt["status"] = "passed_evaluation_binding_freeze"
    receipt["next_stage"] = {
        "formal_public_icl_authorized": True,
        "action_planning_cem_authorized_after_three_seed_icl_gate": True,
        "original_tworoom_retention_cem_authorized_after_three_seed_icl_gate": True,
    }
    receipt["binding_freeze_recovery"] = {
        "recovery_id": RECOVERY_ID,
        "preregistration": _identity(preregistration_path),
        "failed_binding_receipt": observed["failed_binding_receipt"],
        "unchanged_original_binding_freezer": observed["original_binding_freezer"],
        "catalog_restore": {
            "source": observed["archived_retention_catalog"],
            "destination": _identity(
                _resolve(
                    preregistration["bounded_repair"]["catalog_restore"]["destination"][
                        "path"
                    ]
                )
            ),
            "exact_byte_copy": True,
            "content_regenerated": False,
        },
        "training_completion_evidence": training_evidence,
        "optimizer_steps_executed_by_binding_recovery": 0,
        "training_executed_by_binding_recovery": False,
        "model_or_checkpoint_selection_performed": False,
        "public_test_accessed": False,
        "formal_icl_executed": False,
        "cem_executed": False,
    }
    return receipt


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
    parser.add_argument(
        "command", choices=("check", "recover"), help="Validate only, or recover and freeze"
    )
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    preregistration_path = _resolve(args.preregistration)
    output = _resolve(args.output)
    if output != DEFAULT_OUTPUT.resolve():
        raise ValueError("Recovered binding receipt must use its dedicated destination")
    preregistration, observed = validate_preregistration(preregistration_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite recovered binding receipt: {output}")
    if args.command == "check":
        target = _resolve(
            preregistration["bounded_repair"]["catalog_restore"]["destination"]["path"]
        )
        print(
            json.dumps(
                {
                    "status": "passed_recovery_preflight",
                    "catalog_destination_present": target.is_file(),
                    "public_test_accessed": False,
                },
                sort_keys=True,
            )
        )
        return
    catalog, preexisting = _copy_catalog(preregistration)
    receipt = build_recovered_receipt(
        preregistration_path=preregistration_path,
        preregistration=preregistration,
        observed=observed,
    )
    receipt["binding_freeze_recovery"]["catalog_restore"][
        "destination_preexisting_exact"
    ] = preexisting
    if receipt["binding_freeze_recovery"]["catalog_restore"]["destination"] != catalog:
        raise RuntimeError("Recovered catalog identity changed during binding freeze")
    _write_exclusive(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": _logical(output),
                "public_test_accessed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
