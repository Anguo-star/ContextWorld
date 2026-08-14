#!/usr/bin/env python3
"""Seal the zero-step Cube v4r1 reference-training v1 infrastructure failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]

from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    file_sha256,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path  # noqa: E402


EXPECTED_ERROR = (
    "ValueError: Required upstream training input is not installed; set "
    "'CONTEXTWORLD_CUBE_H5' or provide the bundled artifact"
)
EXPECTED_TRACE = (
    "scripts/run_pusht_contact_friction_h3_train.py\", line 297, "
    "in _resolve_release_input"
)
JOB_PATTERN = re.compile(r"(?:lewm|pldm)_seed(?:17321|17322|17323)")
OUTPUT_NAME = "reference_training_v1_infrastructure_failure_receipt.json"
V1_PREREGISTRATION_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_reference_training_v1"
)
V1_PROTOCOL_ID = "cube_gripper_carry_rule_history3_v4r1_reference_training_v1"
V1_FREEZE_STATUS = "frozen_before_reference_training"


def _closed_public() -> dict[str, Any]:
    return {
        "access_status": "closed_not_read_not_scored",
        "generated": False,
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
        "validation_lance_access_allowed": False,
    }


def _load_archived_v1_prereg(prereg_path: Path) -> dict[str, Any]:
    """Load the immutable v1 contract without routing through current v2 APIs."""

    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if (
        not isinstance(prereg, dict)
        or prereg.get("schema_version") != 1
        or prereg.get("preregistration_id") != V1_PREREGISTRATION_ID
        or prereg.get("protocol_id") != V1_PROTOCOL_ID
        or prereg.get("status") != "preregistered_before_reference_training"
        or prereg.get("planned_artifacts", {}).get("training_root")
        != (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v1"
        )
    ):
        raise RuntimeError("Archived Cube reference-training v1 preregistration drifted")
    freeze_path = resolve_contextworld_path(
        prereg["planned_artifacts"]["freeze_receipt"], repo_root=ROOT
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_prereg = freeze.get("preregistration", {})
    if (
        freeze.get("schema_version") != 1
        or freeze.get("preregistration_id") != V1_PREREGISTRATION_ID
        or freeze.get("protocol_id") != V1_PROTOCOL_ID
        or freeze.get("status") != V1_FREEZE_STATUS
        or freeze.get("checks_passed") is not True
        or frozen_prereg.get("sha256") != file_sha256(prereg_path)
        or int(frozen_prereg.get("size_bytes", -1)) != prereg_path.stat().st_size
        or freeze.get("public_test") != _closed_public()
    ):
        raise RuntimeError("Archived Cube reference-training v1 freeze drifted")
    return {
        **prereg,
        "_config_path": str(prereg_path),
        "_freeze_receipt": freeze,
        "_freeze_receipt_path": str(freeze_path),
    }


def archive(*, prereg_path: Path, training_root: Path, output: Path) -> dict[str, Any]:
    prereg = _load_archived_v1_prereg(prereg_path)
    expected_root = resolve_contextworld_path(
        prereg["planned_artifacts"]["training_root"], repo_root=ROOT
    )
    if training_root.resolve() != expected_root:
        raise RuntimeError("Cube v1 failure root does not match its preregistration")
    if output.resolve() != training_root / OUTPUT_NAME:
        raise RuntimeError("Cube v1 failure receipt path drifted")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failure receipt: {output}")
    matrix_path = training_root / "matrix_report.json"
    request_path = training_root / "matrix_request.json"
    if not matrix_path.is_file() or not request_path.is_file():
        raise FileNotFoundError("Cube v1 failure is missing matrix evidence")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    expected_names = [
        f"{model}_seed{seed}"
        for model in ("lewm", "pldm")
        for seed in prereg["training"]["reference_matrix"]["training_seeds"]
    ]
    failures = matrix.get("failures")
    if (
        matrix.get("schema_version") != 1
        or matrix.get("status") != "failed"
        or matrix.get("preregistration_id") != V1_PREREGISTRATION_ID
        or matrix.get("reports") != []
        or not isinstance(failures, list)
        or [row.get("name") for row in failures] != expected_names
        or any(int(row.get("returncode", 0)) != 1 for row in failures)
        or request.get("schema_version") != 1
        or request.get("status") != "running"
    ):
        raise RuntimeError("Cube v1 matrix failure evidence drifted")
    logs: dict[str, Any] = {}
    job_directories = []
    for name in expected_names:
        if JOB_PATTERN.fullmatch(name) is None:
            raise AssertionError(name)
        job = training_root / name
        if not job.is_dir() or job.is_symlink():
            raise RuntimeError(f"Cube v1 job directory is missing: {name}")
        children = list(job.iterdir())
        if children:
            raise RuntimeError(f"Cube v1 job ran beyond pre-output validation: {name}")
        job_directories.append({"name": name, "empty": True})
        log = training_root / "logs" / f"{name}.log"
        if not log.is_file() or log.is_symlink():
            raise RuntimeError(f"Cube v1 log is missing: {name}")
        text = log.read_text(encoding="utf-8")
        if EXPECTED_ERROR not in text or EXPECTED_TRACE not in text:
            raise RuntimeError(f"Cube v1 failure traceback drifted: {name}")
        logs[name] = {
            "path": portable_contextworld_path(log, repo_root=ROOT),
            "sha256": file_sha256(log),
            "size_bytes": log.stat().st_size,
            "expected_error_present": True,
        }
    forbidden = [
        path
        for path in training_root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".pt", ".ckpt"}
            or path.name in {"config.json", "training_provenance.json", "training_report.json"}
        )
    ]
    if forbidden:
        raise RuntimeError(f"Cube v1 failure unexpectedly produced model artifacts: {forbidden}")
    receipt = {
        "schema_version": 1,
        "receipt_id": "cube_gripper_carry_h3_v4r1_reference_training_v1_failure",
        "preregistration_id": V1_PREREGISTRATION_ID,
        "status": "infrastructure_failed_before_training",
        "classified_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "input_resolver_contract_mismatch_not_scientific_failure",
        "failure_stage": "shared_trainer_resolve_original_h5_before_output_creation",
        "checks_passed": True,
        "authorization_chain": {
            "preregistration": {
                "path": portable_contextworld_path(prereg_path, repo_root=ROOT),
                "sha256": file_sha256(prereg_path),
                "size_bytes": prereg_path.stat().st_size,
            },
            "freeze_receipt": {
                "path": portable_contextworld_path(
                    Path(prereg["_freeze_receipt_path"]), repo_root=ROOT
                ),
                "sha256": file_sha256(Path(prereg["_freeze_receipt_path"])),
                "size_bytes": Path(prereg["_freeze_receipt_path"]).stat().st_size,
            },
            "matrix_request": {
                "path": portable_contextworld_path(request_path, repo_root=ROOT),
                "sha256": file_sha256(request_path),
                "size_bytes": request_path.stat().st_size,
            },
            "matrix_report": {
                "path": portable_contextworld_path(matrix_path, repo_root=ROOT),
                "sha256": file_sha256(matrix_path),
                "size_bytes": matrix_path.stat().st_size,
            },
        },
        "jobs": {
            "authorized": expected_names,
            "exit_code_one": expected_names,
            "empty_output_directories": job_directories,
            "logs": logs,
        },
        "training_state": {
            "trainer_reached_data_materialization": False,
            "model_instantiated": False,
            "forward_passes": 0,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "training_reports_created": 0,
        },
        "root_cause": {
            "prereg_input_key": "local_source",
            "shared_resolver_supported_keys": ["path", "checkpoint"],
            "frozen_input_missing": False,
            "frozen_input_identity_invalid": False,
        },
        "retry": {
            "authorized_under_v1": False,
            "v1_output_reusable": False,
            "new_preregistration_and_namespace_required": True,
        },
        "scientific_conclusion": {
            "data_failure_claim_allowed": False,
            "model_training_failure_claim_allowed": False,
            "development_score_claim_allowed": False,
        },
        "public_test": _closed_public(),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = archive(
        prereg_path=args.prereg.expanduser().resolve(),
        training_root=args.training_root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
