#!/usr/bin/env python3
"""Recover the Speed PLDM CEM binding with canonical path comparison.

Recovery v2 isolated the remaining incompatibility: the baseline matrix stores
absolute artifact paths, while the preregistration stores portable
``artifacts/...`` paths for the same six files.  The original freezer both
indexes by the absolute form and demands byte-for-byte equality with the
portable form.  This launcher validates resolved path, SHA-256 and byte size,
uses the absolute form only inside that legacy comparison, then restores the
portable identities in the binding payload.

The v2 one-field schema view is retained.  No frozen file is rewritten and no
model, environment, planner, Public Test payload, or CEM evaluation is run.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

import scripts.freeze_tworoom_speed_pldm_cem_binding_v1 as original_freezer
from scripts.recover_tworoom_speed_pldm_cem_binding_v2 import (
    _identity,
    _load_json,
    _load_yaml,
    _logical,
    _resolve,
    _same_identity,
    validate_preregistration as validate_v2_preregistration,
)


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_ID = "tworoom_speed_pldm_cem_binding_recovery_v3"
COMPLETION_ID = "tworoom_speed_pldm_reference_completion_v1"
EXPECTED_SEEDS = (42, 43, 44, 45, 46, 47)
DEFAULT_PREREGISTRATION = (
    ROOT / "configs/benchmark/tworoom_speed_pldm_cem_binding_recovery_v3.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/tworoom_speed_pldm_reference_completion_v1"
    / "formal_icl_v1/cem_binding_v1.json"
)


def _require_identity(specification: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(specification, Mapping) or not isinstance(
        specification.get("path"), str
    ):
        raise ValueError(f"{label} lacks a frozen identity")
    path = original_freezer.resolve_source(specification["path"], repo_root=ROOT)
    observed = _identity(path)
    if not _same_identity(observed, specification):
        raise RuntimeError(f"{label} identity drifted")
    return observed


def _baseline_declarations() -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    static, _ = original_freezer._validate_static_prereg(original_freezer.CEM_PREREG)
    baseline = static["tracks"]["original_task_retention_cem"][
        "frozen_paired_baseline"
    ]
    declared = {
        int(row["eval_seed"]): dict(row) for row in baseline["raw_receipts"]
    }
    summary_path = original_freezer.resolve_source(
        baseline["matrix_summary"]["path"], repo_root=ROOT
    )
    summary = _load_json(summary_path)
    cell = [
        row
        for row in summary.get("cells", [])
        if row.get("environment") == "tworoom" and row.get("family") == "pldm"
    ]
    if len(cell) != 1:
        raise RuntimeError("Baseline summary lacks one TwoRoom PLDM cell")
    sources = cell[0].get("sources")
    if not isinstance(sources, list) or len(sources) != 6:
        raise RuntimeError("Baseline summary lacks six receipt sources")
    summary_by_seed: dict[int, dict[str, Any]] = {}
    for seed, item in declared.items():
        declared_path = original_freezer.resolve_source(item["path"], repo_root=ROOT)
        matches = [
            dict(row)
            for row in sources
            if original_freezer.resolve_source(row.get("path", ""), repo_root=ROOT)
            == declared_path
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Baseline summary path is ambiguous for eval seed {seed}")
        summary_by_seed[seed] = matches[0]
    return declared, summary_by_seed


def validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    preregistration = _load_yaml(path)
    frozen = preregistration.get("frozen_inputs")
    comparison = preregistration.get("path_compatibility")
    implementation = preregistration.get("implementation")
    if not (
        preregistration.get("schema_version") == 1
        and preregistration.get("recovery_id") == RECOVERY_ID
        and preregistration.get("completion_id") == COMPLETION_ID
        and preregistration.get("status")
        == "registered_after_recovery_v2_path_failure_before_cem_binding_or_execution"
        and preregistration.get("scope")
        == {
            "changes_frozen_file_content_or_identity": False,
            "changes_cem_runner_metric_or_threshold": False,
            "changes_checkpoint_or_model": False,
            "reopens_public_test": False,
            "executes_model_environment_or_planner": False,
            "executes_cem": False,
            "canonicalizes_six_frozen_receipt_paths_for_comparison_only": True,
        }
        and isinstance(frozen, Mapping)
        and isinstance(comparison, Mapping)
        and isinstance(implementation, Mapping)
    ):
        raise ValueError("Unexpected Speed CEM-binding recovery-v3 preregistration")
    expected_frozen = {
        "recovery_v2_preregistration",
        "recovery_v2_launcher",
        "recovery_v2_failure",
        "original_cem_binding_freezer",
        "original_baseline_matrix_summary",
    }
    if set(frozen) != expected_frozen:
        raise ValueError("Recovery-v3 frozen-input set is incomplete")
    observed = {
        name: _require_identity(specification, label=name)
        for name, specification in frozen.items()
    }
    launcher = _require_identity(
        implementation.get("recovery_launcher", {}), label="recovery-v3 launcher"
    )
    if launcher["path"] != _logical(Path(__file__)):
        raise RuntimeError("Recovery-v3 preregistration binds a different launcher")
    v2_path = _resolve(frozen["recovery_v2_preregistration"]["path"])
    validate_v2_preregistration(v2_path)
    failure = _load_json(_resolve(frozen["recovery_v2_failure"]["path"]))
    if not (
        failure.get("status") == "failed_before_cem_binding_or_execution"
        and failure.get("failure", {}).get("stage")
        == "frozen_original_baseline_raw_receipt_identity_comparison"
        and failure.get("boundary", {}).get("cem_binding_written") is False
        and failure.get("boundary", {}).get("action_planning_cem_executed") is False
        and failure.get("boundary", {}).get("original_task_retention_cem_executed")
        is False
    ):
        raise RuntimeError("Recovery-v2 failure evidence is not intact")
    if comparison != {
        "scope": "six_frozen_tworoom_pldm_original_baseline_receipts",
        "eval_seeds": list(EXPECTED_SEEDS),
        "comparison_key": "resolved_path_sha256_size_bytes",
        "binding_output_path_form": "portable_artifacts_path",
        "path_rebinding_authorized": False,
        "underlying_file_rewritten": False,
    }:
        raise ValueError("Recovery-v3 path compatibility is broader than registered")
    declared, summary = _baseline_declarations()
    if tuple(sorted(declared)) != EXPECTED_SEEDS or tuple(sorted(summary)) != EXPECTED_SEEDS:
        raise RuntimeError("Recovery-v3 receipt seed set is incomplete")
    for seed in EXPECTED_SEEDS:
        portable = declared[seed]
        absolute = summary[seed]
        if not (
            Path(absolute["path"]).is_absolute()
            and str(portable["path"]).startswith("artifacts/")
            and original_freezer.resolve_source(portable["path"], repo_root=ROOT)
            == original_freezer.resolve_source(absolute["path"], repo_root=ROOT)
            and portable["sha256"] == absolute["sha256"]
            and portable["size_bytes"] == absolute["size_bytes"]
        ):
            raise RuntimeError(f"Baseline receipt content identity differs for seed {seed}")
    return preregistration, observed


def build_binding(
    *, preregistration_path: Path, preregistration: dict[str, Any], observed: dict[str, Any]
) -> dict[str, Any]:
    v2_preregistration = _load_yaml(
        _resolve(preregistration["frozen_inputs"]["recovery_v2_preregistration"]["path"])
    )
    v1_preregistration = _load_yaml(
        _resolve(v2_preregistration["frozen_inputs"]["recovery_v1_preregistration"]["path"])
    )
    replacement = _resolve(
        v1_preregistration["bounded_substitution"]["replacement"]["path"]
    )
    results_freeze_path = original_freezer.resolve_source(
        v2_preregistration["frozen_inputs"]["original_baseline_results_freeze"]["path"],
        repo_root=ROOT,
    ).resolve()
    declared, _ = _baseline_declarations()
    original_receipt_path = original_freezer.EVALUATION_BINDING_RECEIPT
    original_load_json = original_freezer._load_json
    original_require_identity = original_freezer._require_static_identity
    original_baseline_retention = original_freezer._baseline_retention

    def compatibility_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if Path(path).resolve() == results_freeze_path:
            payload = copy.deepcopy(payload)
            matrix = payload.get("matrix_summary")
            if not isinstance(matrix, dict) or matrix.pop("strictly_reused_episodes", None) != 300:
                raise RuntimeError("Registered baseline compatibility field changed")
        return payload

    def comparison_identity(value: Any, *, label: str) -> dict[str, Any]:
        identity = original_require_identity(value, label=label)
        if label.startswith("retention baseline raw receipt ") or label == (
            "retention baseline receipt"
        ):
            identity = dict(identity)
            identity["path"] = str(
                original_freezer.resolve_source(value["path"], repo_root=ROOT)
            )
        return identity

    def canonical_baseline_retention(
        retention: Mapping[str, Any],
        *,
        catalog_identity: Mapping[str, Any],
        retention_schedule: list[dict[str, Any]],
    ) -> dict[str, Any]:
        result = original_baseline_retention(
            retention,
            catalog_identity=catalog_identity,
            retention_schedule=retention_schedule,
        )
        for row in result["raw_receipts"]:
            seed = int(row["eval_seed"])
            row["receipt"] = original_require_identity(
                declared[seed], label=f"canonical retention baseline receipt {seed}"
            )
        return result

    try:
        original_freezer.EVALUATION_BINDING_RECEIPT = replacement
        original_freezer._load_json = compatibility_load_json
        original_freezer._require_static_identity = comparison_identity
        original_freezer._baseline_retention = canonical_baseline_retention
        payload = original_freezer.build_binding(original_freezer.CEM_PREREG)
    finally:
        original_freezer.EVALUATION_BINDING_RECEIPT = original_receipt_path
        original_freezer._load_json = original_load_json
        original_freezer._require_static_identity = original_require_identity
        original_freezer._baseline_retention = original_baseline_retention
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
    portable_receipts = payload["tracks"]["original_task_retention_cem"][
        "paired_baseline"
    ]["raw_receipts"]
    if any(
        row["receipt"] != original_require_identity(
            declared[int(row["eval_seed"])], label="portable binding receipt"
        )
        for row in portable_receipts
    ):
        raise RuntimeError("Recovered binding did not restore portable receipt identities")
    payload["binding_recovery"] = {
        "recovery_id": RECOVERY_ID,
        "preregistration": _identity(preregistration_path),
        "recovery_v2_failure": observed["recovery_v2_failure"],
        "unchanged_original_cem_binding_freezer": observed[
            "original_cem_binding_freezer"
        ],
        "evaluation_binding_receipt_substitution_inherited_from_recovery_v1": True,
        "baseline_results_freeze_schema_view_inherited_from_recovery_v2": True,
        "path_compatibility": {
            "scope": "six_frozen_tworoom_pldm_original_baseline_receipts",
            "comparison_key": "resolved_path_sha256_size_bytes",
            "binding_output_path_form": "portable_artifacts_path",
            "underlying_files_rewritten": False,
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
