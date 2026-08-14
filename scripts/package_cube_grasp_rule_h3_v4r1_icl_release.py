from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    EXPECTED_PLANNED_ARTIFACTS,
    EXPECTED_PREREGISTRATION_LOGICAL_PATH,
    assert_portable_tree,
    file_identity,
    identity_equal,
    lexical_absolute,
    read_json,
    read_yaml,
    require_no_symlink_components,
    resolve_no_symlink_contextworld_path,
    tree_identity,
    validate_historical_evidence,
    validate_registration_preregistration_contract,
)
from contextworld.paths import repository_root
from scripts.freeze_cube_grasp_rule_h3_v4r1_suite_registration import (
    DEFAULT_OUTPUT as DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    FREEZE_RECEIPT_ID,
    IMPLEMENTATION_PATHS,
    REGISTRATION_ID,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def package_release_projection(
    *,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    freeze_receipt: Path = DEFAULT_FREEZE_RECEIPT,
    output: Path | None = None,
) -> dict[str, Any]:
    repo = repository_root()
    preregistration = lexical_absolute(preregistration)
    expected_preregistration = require_no_symlink_components(
        repo / EXPECTED_PREREGISTRATION_LOGICAL_PATH,
        anchor=repo,
        label="Cube Suite-registration preregistration",
    )
    if preregistration != expected_preregistration:
        raise RuntimeError("Packaging must use the canonical preregistration path")
    freeze_receipt = lexical_absolute(freeze_receipt)
    expected_freeze_receipt = resolve_no_symlink_contextworld_path(
        EXPECTED_PLANNED_ARTIFACTS["registration_freeze_receipt"],
        repo_root=repo,
        label="registration freeze receipt",
    )
    if freeze_receipt != expected_freeze_receipt:
        raise RuntimeError("Packaging must use the canonical freeze receipt path")
    prereg = read_yaml(preregistration, label="registration preregistration")
    freeze = read_json(freeze_receipt, label="registration freeze receipt")
    validate_registration_preregistration_contract(
        prereg, preregistration_path=preregistration
    )
    if (
        not isinstance(prereg, dict)
        or prereg.get("registration_id") != REGISTRATION_ID
        or freeze.get("receipt_id") != FREEZE_RECEIPT_ID
        or freeze.get("registration_id") != REGISTRATION_ID
        or freeze.get("status") != "suite_registration_packaging_frozen"
        or freeze.get("checks_passed") is not True
        or freeze.get("receipt_path")
        != prereg["planned_artifacts"]["registration_freeze_receipt"]
        or freeze.get("planned_artifacts") != prereg["planned_artifacts"]
        or freeze.get("authorization_basis") != prereg["authorization_basis"]
        or freeze.get("authorization")
        != {
            "projection_packaging_allowed": True,
            "public_test_rerun_allowed": False,
            "training_or_checkpoint_selection_allowed": False,
            "suite_membership_before_final_audit_allowed": False,
        }
    ):
        raise RuntimeError("Registration packaging authorization drifted")
    expected_freeze_receipt = resolve_no_symlink_contextworld_path(
        prereg["planned_artifacts"]["registration_freeze_receipt"],
        repo_root=repo,
        label="registration freeze receipt",
    )
    if freeze_receipt != expected_freeze_receipt:
        raise RuntimeError("Registration freeze receipt path drifted")
    prereg_logical = preregistration.relative_to(repo).as_posix()
    if not identity_equal(
        freeze["preregistration"],
        file_identity(preregistration, logical_path=prereg_logical),
    ):
        raise RuntimeError("Registration preregistration changed after freeze")
    for name, logical in IMPLEMENTATION_PATHS.items():
        if not identity_equal(
            freeze["implementation_identities"][name],
            file_identity(
                require_no_symlink_components(
                    repo / logical,
                    anchor=repo,
                    label=f"packaging implementation {name}",
                ),
                logical_path=logical,
            ),
        ):
            raise RuntimeError(f"Packaging implementation changed after freeze: {name}")

    expected_output = resolve_no_symlink_contextworld_path(
        prereg["packaging_contract"]["projection_root"],
        repo_root=repo,
        label="release projection",
        allow_missing=True,
    )
    output = lexical_absolute(output or expected_output)
    if output != expected_output:
        raise RuntimeError("Projection output must use the preregistered namespace")
    if os.path.lexists(output):
        raise FileExistsError(f"Projection namespace is already consumed: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    started = {
        "schema_version": 1,
        "registration_id": REGISTRATION_ID,
        "projection_id": "contextworld_cube_gripper_carry_h3_v4r1_projection_v1",
        "status": "portable_projection_packaging_started_namespace_consumed",
        "preregistration": freeze["preregistration"],
        "freeze_receipt": file_identity(
            freeze_receipt,
            logical_path=prereg["planned_artifacts"]["registration_freeze_receipt"],
        ),
        "output": prereg["packaging_contract"]["projection_root"],
        "public_test_rerun": False,
        "model_or_checkpoint_read": False,
        "rerun_authorized": False,
    }
    _write_json(output / "_PACKAGING_STARTED.json", started)
    phase = "historical_evidence_validation"
    try:
        evidence = validate_historical_evidence(prereg, repo_root=repo)
        if (
            evidence["authorization_basis"] != freeze["authorization_basis"]
            or evidence["source_tables"] != freeze["source_tables"]
            or freeze.get("historical_validation")
            != {
                "public_reference": evidence["public_reference"],
                "original_task_retention": evidence["original_task_retention"],
                "data_contract": evidence["data_contract"],
            }
        ):
            raise RuntimeError("Historical evidence differs from the freeze receipt")
        source_tables: dict[str, dict[str, Any]] = {}
        source_paths: dict[str, Path] = {}
        for split, row in evidence["source_tables"].items():
            source_paths[split] = resolve_no_symlink_contextworld_path(
                row["path"],
                repo_root=repo,
                label=f"source table {split}",
            )
            source_tables[split] = {
                "source": row["path"],
                "bundled_path": f"{split}.lance",
                "pair_count": int(row["pair_count"]),
                "rows": int(row["rows"]),
                **{key: row[key] for key in ("files", "bytes", "sha256")},
            }
        reference = dict(evidence["public_reference"])
        reference["checkpoint_results"] = [
            {
                "training_seed": int(seed),
                "model_family": "lewm",
                "training_recipe": reference["training_recipe"],
                **row,
            }
            for seed, row in sorted(
                reference["checkpoint_results"].items(), key=lambda item: int(item[0])
            )
        ]
        provenance = {
            "schema_version": 1,
            "projection_id": started["projection_id"],
            "registration_id": REGISTRATION_ID,
            "release_id": prereg["release_id"],
            "component_id": prereg["component_id"],
            "status": "portable_projection_content_complete",
            "source_tables": source_tables,
            "historical_evidence": evidence["authorization_basis"],
            "data_contract": evidence["data_contract"],
            "public_reference": reference,
            "original_task_retention": evidence["original_task_retention"],
            "claim_boundary": {
                "public_reference_family": "lewm",
                "pldm_public_result_included": False,
                "public_test_rerun_during_packaging": False,
                "recovery_decision_granted_suite_registration": False,
                "suite_registration_requires_separate_final_audit": True,
            },
        }
        phase = "table_copy"
        for split, source in source_paths.items():
            shutil.copytree(source, output / f"{split}.lance", symlinks=True)
        phase = "provenance_write"
        provenance_path = output / "portable_provenance.json"
        _write_json(provenance_path, provenance)
        assert_portable_tree(output)
        if set(path.name for path in output.iterdir()) != {
            "_PACKAGING_STARTED.json",
            "train.lance",
            "loader_validation.lance",
            "validation.lance",
            "portable_provenance.json",
        }:
            raise RuntimeError("Projection root contains unexpected content")
        for split, specification in source_tables.items():
            copied = tree_identity(output / f"{split}.lance")
            expected = {
                key: specification[key]
                for key in ("files", "bytes", "sha256")
            }
            if copied != expected:
                raise RuntimeError(f"Copied table identity drifted: {split}")
        before_success = tree_identity(output)
        success = {
            "schema_version": 1,
            "projection_id": provenance["projection_id"],
            "registration_id": REGISTRATION_ID,
            "release_id": prereg["release_id"],
            "status": "portable_release_projection_published",
            "preregistration": freeze["preregistration"],
            "freeze_receipt": file_identity(
                freeze_receipt,
                logical_path=prereg["planned_artifacts"]["registration_freeze_receipt"],
            ),
            "packaging_started": file_identity(
                output / "_PACKAGING_STARTED.json",
                logical_path=prereg["packaging_contract"]["projection_root"]
                + "/_PACKAGING_STARTED.json",
            ),
            "portable_provenance": file_identity(
                provenance_path,
                logical_path=prereg["packaging_contract"]["projection_root"]
                + "/portable_provenance.json",
            ),
            "source_tables": source_tables,
            "tree_before_success_marker": before_success,
            "public_test_rerun": False,
            "model_or_checkpoint_read": False,
            "rerun_authorized": False,
        }
        phase = "success_write"
        _write_json(output / "_SUCCESS.json", success)
        assert_portable_tree(output)
        if set(path.name for path in output.iterdir()) != set(
            prereg["packaging_contract"]["projection_contains"]
        ):
            raise RuntimeError("Published projection child set drifted")
    except BaseException as error:
        failure_path = output / "_PACKAGING_FAILURE.json"
        if not os.path.lexists(failure_path):
            _write_json(
                failure_path,
                {
                    "schema_version": 1,
                    "registration_id": REGISTRATION_ID,
                    "projection_id": started["projection_id"],
                    "status": "portable_projection_packaging_failed_namespace_consumed",
                    "phase": phase,
                    "error_type": type(error).__name__,
                    "public_test_rerun": False,
                    "model_or_checkpoint_read": False,
                    "rerun_authorized": False,
                },
            )
        raise
    return {
        "schema_version": 1,
        "status": "portable_release_projection_published",
        "output": prereg["packaging_contract"]["projection_root"],
        "tree": tree_identity(output),
        "public_test_rerun": False,
        "model_or_checkpoint_read": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = package_release_projection(
        preregistration=args.preregistration,
        freeze_receipt=args.freeze_receipt,
        output=args.output,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
