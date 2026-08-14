from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    EXPECTED_PREREGISTRATION_LOGICAL_PATH,
    FREEZE_RECEIPT_ID,
    REGISTRATION_ID,
    lexical_absolute,
    read_yaml,
    require_no_symlink_components,
    resolve_no_symlink_contextworld_path,
    validate_historical_evidence,
    validate_registration_preregistration_contract,
)
from contextworld.paths import repository_root


DEFAULT_PREREGISTRATION = repository_root() / EXPECTED_PREREGISTRATION_LOGICAL_PATH
DEFAULT_OUTPUT = resolve_no_symlink_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_v1/"
    "registration_freeze_receipt_v1.json",
    label="registration freeze receipt",
    allow_missing=True,
)
IMPLEMENTATION_PATHS = {
    "registration_contract": (
        "contextworld/benchmarks/cube_grasp_rule_suite_registration.py"
    ),
    "packaging_script": "scripts/package_cube_grasp_rule_h3_v4r1_icl_release.py",
    "registration_freezer": (
        "scripts/freeze_cube_grasp_rule_h3_v4r1_suite_registration.py"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path, *, logical_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Required regular file is missing: {path}")
    return {
        "path": logical_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def freeze_registration(
    *, preregistration: Path = DEFAULT_PREREGISTRATION, output: Path = DEFAULT_OUTPUT
) -> dict[str, Any]:
    repo = repository_root()
    preregistration = lexical_absolute(preregistration)
    expected_preregistration = require_no_symlink_components(
        repo / EXPECTED_PREREGISTRATION_LOGICAL_PATH,
        anchor=repo,
        label="Cube Suite-registration preregistration",
    )
    if preregistration != expected_preregistration:
        raise RuntimeError("Freeze must use the canonical preregistration path")
    prereg = read_yaml(preregistration, label="registration preregistration")
    validate_registration_preregistration_contract(
        prereg, preregistration_path=preregistration
    )
    evidence = validate_historical_evidence(prereg, repo_root=repo)
    basis_observed = evidence["authorization_basis"]
    source_tables = evidence["source_tables"]

    projection = resolve_no_symlink_contextworld_path(
        prereg["packaging_contract"]["projection_root"],
        repo_root=repo,
        label="release projection",
        allow_missing=True,
    )
    if os.path.lexists(projection):
        raise FileExistsError(f"Projection namespace is already consumed: {projection}")
    output = lexical_absolute(output)
    expected_output = resolve_no_symlink_contextworld_path(
        prereg["planned_artifacts"]["registration_freeze_receipt"],
        repo_root=repo,
        label="registration freeze receipt",
        allow_missing=True,
    )
    if output != expected_output:
        raise RuntimeError("Freeze output must use the preregistered path")
    if os.path.lexists(output):
        raise FileExistsError(f"Registration freeze receipt exists: {output}")

    implementation_identities = {
        name: _identity(
            require_no_symlink_components(
                repo / logical,
                anchor=repo,
                label=f"packaging implementation {name}",
            ),
            logical_path=logical,
        )
        for name, logical in IMPLEMENTATION_PATHS.items()
    }
    prereg_logical = preregistration.relative_to(repo).as_posix()
    receipt = {
        "schema_version": 1,
        "receipt_id": FREEZE_RECEIPT_ID,
        "receipt_path": prereg["planned_artifacts"][
            "registration_freeze_receipt"
        ],
        "registration_id": REGISTRATION_ID,
        "status": "suite_registration_packaging_frozen",
        "checks_passed": True,
        "preregistration": _identity(
            preregistration, logical_path=prereg_logical
        ),
        "authorization_basis": basis_observed,
        "source_tables": source_tables,
        "historical_validation": {
            "public_reference": evidence["public_reference"],
            "original_task_retention": evidence["original_task_retention"],
            "data_contract": evidence["data_contract"],
        },
        "implementation_identities": implementation_identities,
        "authorization": {
            "projection_packaging_allowed": True,
            "public_test_rerun_allowed": False,
            "training_or_checkpoint_selection_allowed": False,
            "suite_membership_before_final_audit_allowed": False,
        },
        "planned_artifacts": dict(prereg["planned_artifacts"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = freeze_registration(
        preregistration=args.preregistration, output=args.output
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
