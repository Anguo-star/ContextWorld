#!/usr/bin/env python3
"""Finalize the completed one-use Cube v4r1 Public campaign."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_public_contract import (  # noqa: E402
    DEFAULT_FREEZE_RECEIPT,
    DEFAULT_PREREGISTRATION,
    file_identity,
    load_public_authorization,
    read_json_nofollow,
)
from contextworld.benchmarks.cube_grasp_rule_public_score import (  # noqa: E402
    BENCHMARK_ID,
    MATRIX_STATUS,
    aggregate_public_results,
    validate_public_checkpoint_result,
    validate_public_publication,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path  # noqa: E402


def _identity_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(name) == right.get(name)
        for name in ("path", "sha256", "size_bytes")
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_x(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def finalize_public_release(
    *,
    preregistration: Path,
    freeze_receipt: Path,
    score_root: Path,
    output: Path,
) -> dict[str, Any]:
    authorization = load_public_authorization(
        preregistration_path=preregistration,
        freeze_receipt_path=freeze_receipt,
    )
    score_root = score_root.expanduser().resolve()
    output = output.expanduser().resolve()
    if score_root != authorization.score_root:
        raise ValueError("Cube Public score root differs from the frozen path")
    expected_output = resolve_contextworld_path(
        authorization.preregistration["planned_artifacts"][
            "public_release_decision"
        ]
    )
    if output != expected_output:
        raise ValueError("Cube Public decision path differs from preregistration")
    try:
        os.lstat(output)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"Cube Public decision already exists: {output}")

    publication = validate_public_publication(authorization)
    success_path = score_root / "_SUCCESS.json"
    _, success = read_json_nofollow(
        success_path, label="Cube Public scoring success marker"
    )
    failure_path = score_root / "infrastructure_failure_receipt.json"
    try:
        os.lstat(failure_path)
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("Cube Public score root contains a failure receipt")
    matrix_entry = success.get("matrix_score")
    if (
        set(success)
        != {
            "schema_version",
            "status",
            "completed_at_utc",
            "matrix_score",
            "checkpoint_results",
            "public_test",
            "rerun_authorized",
        }
        or not isinstance(success.get("completed_at_utc"), str)
        or success.get("schema_version") != 1
        or success.get("status") != MATRIX_STATUS
        or success.get("rerun_authorized") is not False
        or not isinstance(matrix_entry, Mapping)
        or success.get("public_test")
        != {
            "generated": True,
            "hashed": True,
            "opened": True,
            "read": True,
            "scored": True,
            "used_for_training_or_selection": False,
        }
    ):
        raise RuntimeError("Cube Public scoring success marker is invalid")
    matrix_path = score_root / "matrix_score.json"
    canonical_matrix_path = portable_contextworld_path(matrix_path)
    if matrix_entry.get("path") != canonical_matrix_path:
        raise RuntimeError("Cube Public matrix logical path is not canonical")
    observed_matrix = file_identity(
        matrix_path, logical_path=canonical_matrix_path
    )
    if not _identity_equal(observed_matrix, matrix_entry):
        raise RuntimeError("Cube Public matrix identity mismatch")
    _, matrix = read_json_nofollow(matrix_path, label="Cube Public matrix score")

    checkpoints = authorization.preregistration["public_evaluation"][
        "checkpoints"
    ]
    expected_seeds = sorted(int(row["training_seed"]) for row in checkpoints)
    expected_by_seed = {int(row["training_seed"]): row for row in checkpoints}
    checkpoint_results = matrix.get("checkpoint_results")
    result_entries = success.get("checkpoint_results")
    if (
        matrix.get("schema_version") != 1
        or matrix.get("benchmark") != BENCHMARK_ID
        or matrix.get("status") != MATRIX_STATUS
        or matrix.get("model_family") != "lewm"
        or matrix.get("training_seeds") != expected_seeds
        or not isinstance(checkpoint_results, list)
        or len(checkpoint_results) != 3
        or not isinstance(result_entries, list)
        or len(result_entries) != 3
        or int(matrix.get("checkpoints_required", -1)) != 3
    ):
        raise RuntimeError("Cube Public matrix contract mismatch")
    observed_result_seeds = sorted(
        int(row.get("model", {}).get("training_seed", -1))
        for row in checkpoint_results
    )
    if observed_result_seeds != expected_seeds:
        raise RuntimeError("Cube Public checkpoint-result seed set drifted")
    embedded_by_seed = {
        int(row["model"]["training_seed"]): row for row in checkpoint_results
    }
    for entry, seed in zip(result_entries, expected_seeds):
        if not isinstance(entry, Mapping):
            raise RuntimeError("Cube Public result identity is malformed")
        path = score_root / f"lewm_seed{seed}.json"
        canonical_path = portable_contextworld_path(path)
        if entry.get("path") != canonical_path:
            raise RuntimeError(
                f"Cube Public seed {seed} result logical path is not canonical"
            )
        observed = file_identity(path, logical_path=canonical_path)
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"Cube Public seed {seed} result identity mismatch")
        _, standalone = read_json_nofollow(
            path, label=f"Cube Public seed {seed} result"
        )
        if standalone != embedded_by_seed[seed]:
            raise RuntimeError(f"Cube Public seed {seed} embedded result drifted")
        expected_checkpoint = expected_by_seed[seed]
        validate_public_checkpoint_result(
            standalone,
            authorization=authorization,
            checkpoint_specification=expected_checkpoint,
        )

    expected_matrix = aggregate_public_results(
        checkpoint_results, authorization=authorization
    )
    if set(matrix) != set(expected_matrix) | {"authorization"}:
        raise RuntimeError("Cube Public matrix key set drifted")
    for name, expected_value in expected_matrix.items():
        if matrix.get(name) != expected_value:
            raise RuntimeError(f"Cube Public matrix field {name} was not recomputed")

    matrix_authorization = matrix.get("authorization")
    if not isinstance(matrix_authorization, Mapping) or set(matrix_authorization) != {
        "matrix_request",
        "public_access_started",
    }:
        raise RuntimeError("Cube Public matrix authorization block is invalid")
    request_path = score_root / "matrix_request.json"
    access_path = score_root / "public_access_started.json"
    for name, path in (
        ("matrix_request", request_path),
        ("public_access_started", access_path),
    ):
        entry = matrix_authorization[name]
        canonical_path = portable_contextworld_path(path)
        if not isinstance(entry, Mapping) or entry.get("path") != canonical_path:
            raise RuntimeError(f"Cube Public {name} path is not canonical")
        observed = file_identity(path, logical_path=canonical_path)
        if not _identity_equal(observed, entry):
            raise RuntimeError(f"Cube Public {name} identity mismatch")
    _, request = read_json_nofollow(
        request_path, label="Cube Public matrix request"
    )
    _, access = read_json_nofollow(
        access_path, label="Cube Public access marker"
    )
    expected_prereg = file_identity(
        authorization.preregistration_path,
        logical_path=authorization.preregistration["identity"][
            "preregistration_path"
        ],
    )
    expected_freeze = file_identity(
        authorization.freeze_receipt_path,
        logical_path=authorization.freeze_receipt_identity["path"],
    )
    expected_public_success = file_identity(
        authorization.public_root / "_SUCCESS.json",
        logical_path=portable_contextworld_path(
            authorization.public_root / "_SUCCESS.json"
        ),
    )
    runtime_preflight = request.get("adapter_runtime_preflight")
    expected_devices = authorization.preregistration["public_evaluation"]["devices"]
    runtime_preflight_valid = (
        isinstance(runtime_preflight, list)
        and len(runtime_preflight) == 3
        and all(
            isinstance(row, Mapping)
            and int(row.get("training_seed", -1))
            == int(checkpoint["training_seed"])
            and row.get("device") == device
            and row.get("model_state_sha256")
            == checkpoint["model_state_sha256"]
            and isinstance(row.get("adapter_id"), str)
            and bool(row.get("adapter_id"))
            and row.get("runtime_preflight_passed") is True
            for row, checkpoint, device in zip(
                runtime_preflight, checkpoints, expected_devices
            )
        )
    )
    if (
        request.get("schema_version") != 1
        or not _identity_equal(request.get("preregistration", {}), expected_prereg)
        or not _identity_equal(request.get("freeze_receipt", {}), expected_freeze)
        or not _identity_equal(
            request.get("public_success_marker", {}), expected_public_success
        )
        or request.get("checkpoints") != checkpoints
        or request.get("devices")
        != expected_devices
        or int(request.get("batch_size", -1))
        != int(authorization.preregistration["public_evaluation"]["batch_size"])
        or request.get("model_visible_fields")
        != ["history_pixels", "query_action_blocks"]
        or request.get("privileged_columns_passed_to_model") is not False
        or request.get("training_or_checkpoint_selection") is not False
        or request.get("threshold_or_recipe_changes") is not False
        or request.get("rerun_authorized") is not False
        or not runtime_preflight_valid
    ):
        raise RuntimeError("Cube Public matrix request contract drifted")
    if (
        access.get("schema_version") != 1
        or access.get("status")
        != "public_access_started_irreversible_one_use_campaign"
        or not _identity_equal(
            access.get("request", {}), matrix_authorization["matrix_request"]
        )
        or access.get("public_data_metadata_preflight_passed") is not True
        or access.get("checkpoint_identity_preflight_passed") is not True
        or access.get("adapter_runtime_preflight_passed") is not True
    ):
        raise RuntimeError("Cube Public access marker contract drifted")

    passed_count = int(expected_matrix["checkpoints_passed"])
    matrix_passed = bool(expected_matrix["passed"])
    if (
        int(matrix.get("checkpoints_passed", -1)) != passed_count
        or matrix.get("passed") is not matrix_passed
        or matrix.get("public_test")
        != {
            "generated": True,
            "hashed": True,
            "opened": True,
            "read": True,
            "scored": True,
            "used_for_training_or_selection": False,
        }
    ):
        raise RuntimeError("Cube Public matrix outcome/state mismatch")

    status = (
        "public_test_release_candidate_reference_passed"
        if matrix_passed
        else "public_test_data_and_scoring_candidate_reference_failed"
    )
    decision = {
        "schema_version": 1,
        "decision_id": "contextworld_cube_gripper_carry_h3_v4r1_public_release_decision_v1",
        "preregistration_id": authorization.preregistration[
            "preregistration_id"
        ],
        "protocol_id": authorization.preregistration["protocol_id"],
        "status": status,
        "decided_at_utc": _utc_now(),
        "authorization_chain": {
            "preregistration": file_identity(
                authorization.preregistration_path,
                logical_path=authorization.preregistration["identity"][
                    "preregistration_path"
                ],
            ),
            "freeze_receipt": authorization.freeze_receipt_identity,
            "public_data_success": file_identity(
                authorization.public_root / "_SUCCESS.json",
                logical_path=portable_contextworld_path(
                    authorization.public_root / "_SUCCESS.json"
                ),
            ),
            "public_score_success": file_identity(
                success_path,
                logical_path=portable_contextworld_path(success_path),
            ),
            "matrix_score": observed_matrix,
            "development_decision": authorization.preregistration[
                "authorization_basis"
            ]["reference_development_decision"],
            "retention_decision": authorization.preregistration[
                "authorization_basis"
            ]["original_task_retention_decision"],
        },
        "public_data": {
            "success_marker_status": publication["success"]["status"],
            "pair_count": int(
                publication["build_report"]["pair_count"]
            ),
            "split": publication["build_report"]["split"],
            "all_frozen_data_gates_passed": True,
        },
        "public_evaluation": {
            "model_family": "lewm",
            "training_seeds": expected_seeds,
            "checkpoints_passed": passed_count,
            "checkpoints_required": 3,
            "passed": matrix_passed,
        },
        "claims": {
            "public_test_completed": True,
            "public_test_score_reporting_allowed": True,
            "positive_reference_public_claim_allowed": matrix_passed,
            "local_data_and_scoring_release_packaging_allowed": True,
            "suite_registration_allowed": False,
            "public_test_rerun_allowed": False,
        },
        "public_test": {
            "access_status": "completed_one_use_public_scoring",
            "generated": True,
            "hashed": True,
            "opened": True,
            "read": True,
            "scored": True,
            "used_for_training_or_selection": False,
            "rerun_authorized": False,
        },
        "next_step": (
            "package a local release candidate and run the separate suite-registration audit"
            if matrix_passed
            else "package the data/scoring release candidate with an explicit negative reference result"
        ),
    }
    _write_json_x(output, decision)
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument(
        "--freeze-receipt", type=Path, default=DEFAULT_FREEZE_RECEIPT
    )
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = finalize_public_release(
        preregistration=args.prereg,
        freeze_receipt=args.freeze_receipt,
        score_root=args.score_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "checkpoints_passed": result["public_evaluation"][
                    "checkpoints_passed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
