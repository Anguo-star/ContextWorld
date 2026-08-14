from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from contextworld.benchmarks.cube_grasp_rule_suite_registration import (
    file_identity,
    lexical_absolute,
    read_yaml,
    require_no_symlink_components,
    resolve_no_symlink_contextworld_path,
)
from contextworld.benchmarks.cube_grasp_rule_suite_registration_recovery import (
    RECOVERY_FREEZE_RECEIPT_ID,
    RECOVERY_PREREGISTRATION_LOGICAL_PATH,
    RECOVERY_REGISTRATION_ID,
    validate_prior_failed_registration_evidence,
    validate_registration_recovery_preregistration_contract,
)
from contextworld.paths import repository_root


ROOT = repository_root()
DEFAULT_PREREGISTRATION = ROOT / RECOVERY_PREREGISTRATION_LOGICAL_PATH
DEFAULT_OUTPUT = resolve_no_symlink_contextworld_path(
    "artifacts/evaluation/history3/"
    "cube_gripper_carry_h3_v4r1_suite_registration_recovery_v2/"
    "registration_freeze_receipt_v2.json",
    repo_root=ROOT,
    label="registration recovery freeze receipt",
    allow_missing=True,
)
IMPLEMENTATION_PATHS = {
    "historical_registration_contract": (
        "contextworld/benchmarks/cube_grasp_rule_suite_registration.py"
    ),
    "recovery_contract": (
        "contextworld/benchmarks/"
        "cube_grasp_rule_suite_registration_recovery.py"
    ),
    "suite_data_api": "contextworld/benchmarks/suite_data.py",
    "recovery_freezer": (
        "scripts/freeze_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py"
    ),
    "recovery_finalizer": (
        "scripts/finalize_cube_grasp_rule_h3_v4r1_"
        "suite_registration_recovery_v2.py"
    ),
}
EXPECTED_AUTHORIZATION = {
    "direct_one_use_export_copy_allowed": True,
    "directory_rename_allowed": False,
    "prior_failed_namespace_mutation_allowed": False,
    "prior_failed_staging_reuse_allowed": False,
    "public_test_rerun_allowed": False,
    "training_or_checkpoint_selection_allowed": False,
    "suite_membership_before_registration_decision_allowed": False,
}


def freeze_registration_recovery(
    *,
    preregistration: Path = DEFAULT_PREREGISTRATION,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    preregistration = lexical_absolute(preregistration)
    expected_preregistration = require_no_symlink_components(
        ROOT / RECOVERY_PREREGISTRATION_LOGICAL_PATH,
        anchor=ROOT,
        label="Cube Suite-registration recovery preregistration",
    )
    if preregistration != expected_preregistration:
        raise RuntimeError(
            "Recovery freeze must use the canonical preregistration path"
        )
    prereg = read_yaml(
        preregistration, label="Suite-registration recovery preregistration"
    )
    validate_registration_recovery_preregistration_contract(
        prereg, preregistration_path=preregistration
    )
    prior_evidence = validate_prior_failed_registration_evidence(
        prereg, repo_root=ROOT
    )

    output = lexical_absolute(output)
    expected_output = resolve_no_symlink_contextworld_path(
        prereg["planned_artifacts"]["registration_freeze_receipt"],
        repo_root=ROOT,
        label="registration recovery freeze receipt",
        allow_missing=True,
    )
    if output != expected_output:
        raise RuntimeError("Recovery freeze output is not preregistered")
    if os.path.lexists(output):
        raise FileExistsError(f"Recovery freeze receipt exists: {output}")
    for name, logical in prereg["planned_artifacts"].items():
        if name == "registration_freeze_receipt":
            continue
        path = resolve_no_symlink_contextworld_path(
            logical,
            repo_root=ROOT,
            label=f"planned recovery output {name}",
            allow_missing=True,
        )
        if os.path.lexists(path):
            raise FileExistsError(
                f"Recovery output namespace is already consumed: {path}"
            )

    implementation_identities = {
        name: file_identity(
            require_no_symlink_components(
                ROOT / logical,
                anchor=ROOT,
                label=f"registration recovery implementation {name}",
            ),
            logical_path=logical,
        )
        for name, logical in IMPLEMENTATION_PATHS.items()
    }
    receipt = {
        "schema_version": 1,
        "receipt_id": RECOVERY_FREEZE_RECEIPT_ID,
        "receipt_path": prereg["planned_artifacts"][
            "registration_freeze_receipt"
        ],
        "registration_id": RECOVERY_REGISTRATION_ID,
        "status": "suite_registration_infrastructure_recovery_frozen",
        "checks_passed": True,
        "preregistration": file_identity(
            preregistration,
            logical_path=RECOVERY_PREREGISTRATION_LOGICAL_PATH,
        ),
        "prior_failed_registration": prior_evidence,
        "implementation_identities": implementation_identities,
        "authorization": EXPECTED_AUTHORIZATION,
        "recovery_contract": dict(prereg["recovery_contract"]),
        "planned_artifacts": dict(prereg["planned_artifacts"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preregistration", type=Path, default=DEFAULT_PREREGISTRATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    value = freeze_registration_recovery(
        preregistration=args.preregistration,
        output=args.output,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
