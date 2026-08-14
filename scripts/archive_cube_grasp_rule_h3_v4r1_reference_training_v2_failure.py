#!/usr/bin/env python3
"""Seal the zero-step Cube v4r1 reference-training v2 runtime failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]

from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    file_sha256,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path  # noqa: E402


V2_PREREGISTRATION_ID = (
    "contextworld_cube_gripper_carry_h3_v4r1_reference_training_v2"
)
V2_PROTOCOL_ID = "cube_gripper_carry_rule_history3_v4r1_reference_training_v2"
V2_FREEZE_STATUS = "frozen_before_reference_training"
OUTPUT_NAME = "reference_training_v2_infrastructure_failure_receipt.json"
EXPECTED_ERROR = (
    "TypeError: ConditionalSIGReg.__init__() got an unexpected keyword "
    "argument 'include_unpaired'"
)
EXPECTED_TRACE = (
    "scripts/run_pusht_hidden_actuation_mixed.py\", line 809, in train_variant"
)
JOB_PATTERN = re.compile(r"(?:lewm|pldm)_seed(?:17321|17322|17323)")


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


def _identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing regular failure artifact: {path}")
    return {
        "path": portable_contextworld_path(path, repo_root=ROOT),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_archived_v2_prereg(prereg_path: Path) -> dict[str, Any]:
    prereg = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if (
        not isinstance(prereg, dict)
        or prereg.get("schema_version") != 1
        or prereg.get("preregistration_id") != V2_PREREGISTRATION_ID
        or prereg.get("protocol_id") != V2_PROTOCOL_ID
        or prereg.get("status") != "preregistered_before_reference_training"
        or prereg.get("planned_artifacts", {}).get("training_root")
        != (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4r1/reference_training_v2"
        )
    ):
        raise RuntimeError("Archived Cube reference-training v2 preregistration drifted")
    freeze_path = resolve_contextworld_path(
        prereg["planned_artifacts"]["freeze_receipt"], repo_root=ROOT
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    frozen_prereg = freeze.get("preregistration", {})
    if (
        freeze.get("schema_version") != 1
        or freeze.get("preregistration_id") != V2_PREREGISTRATION_ID
        or freeze.get("protocol_id") != V2_PROTOCOL_ID
        or freeze.get("status") != V2_FREEZE_STATUS
        or freeze.get("checks_passed") is not True
        or frozen_prereg.get("sha256") != file_sha256(prereg_path)
        or int(frozen_prereg.get("size_bytes", -1)) != prereg_path.stat().st_size
        or freeze.get("public_test") != _closed_public()
    ):
        raise RuntimeError("Archived Cube reference-training v2 freeze drifted")
    return {
        **prereg,
        "_config_path": str(prereg_path),
        "_freeze_receipt": freeze,
        "_freeze_receipt_path": str(freeze_path),
    }


def _validate_job_artifacts(
    *, job: Path, model: str, seed: int, prereg: Mapping[str, Any]
) -> dict[str, Any]:
    expected_names = {"config.json", "training_provenance.json"}
    observed_names = {path.name for path in job.iterdir()}
    if observed_names != expected_names or any(path.is_symlink() for path in job.iterdir()):
        raise RuntimeError(f"Cube v2 job artifact set drifted: {job.name}")
    config_path = job / "config.json"
    provenance_path = job / "training_provenance.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_target = {
        "lewm": "stable_worldmodel.wm.lewm.LeWM",
        "pldm": "stable_worldmodel.wm.pldm.pldm.PLDM",
    }[model]
    matrix = prereg["training"]["reference_matrix"]
    expected_checkpoint = matrix["initial_checkpoints"][model]
    if (
        config.get("_target_") != expected_target
        or int(config.get("action_encoder", {}).get("input_dim", -1)) != 25
        or provenance.get("schema_version") != 1
        or provenance.get("status") != "cube_gripper_carry_reference_training"
        or provenance.get("formal_reference_recipe") is not True
        or provenance.get("model") != model
        or int(provenance.get("seed", -1)) != seed
        or int(provenance.get("optimizer_steps", -1)) != 4096
        or provenance.get("release", {}).get("release_id")
        != V2_PREREGISTRATION_ID
        or provenance.get("release", {}).get("sha256")
        != file_sha256(Path(prereg["_config_path"]))
        or provenance.get("data", {}).get("train_pairs") != 2048
        or provenance.get("data", {}).get("loader_validation_pairs") != 256
        or provenance.get("data", {}).get("independent_validation_opened")
        is not False
        or provenance.get("upstream", {}).get("initial_checkpoint", {}).get(
            "sha256"
        )
        != expected_checkpoint["sha256"]
    ):
        raise RuntimeError(f"Cube v2 pre-training provenance drifted: {job.name}")
    return {
        "config": _identity(config_path),
        "training_provenance": _identity(provenance_path),
        "action_input_dim": 25,
        "initial_checkpoint_identity_matched": True,
        "train_pairs_materialized": 2048,
        "loader_validation_pairs_materialized": 256,
    }


def archive(*, prereg_path: Path, training_root: Path, output: Path) -> dict[str, Any]:
    prereg = _load_archived_v2_prereg(prereg_path)
    expected_root = resolve_contextworld_path(
        prereg["planned_artifacts"]["training_root"], repo_root=ROOT
    )
    if training_root.resolve() != expected_root:
        raise RuntimeError("Cube v2 failure root does not match its preregistration")
    if output.resolve() != training_root / OUTPUT_NAME:
        raise RuntimeError("Cube v2 failure receipt path drifted")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failure receipt: {output}")
    request_path = training_root / "matrix_request.json"
    matrix_path = training_root / "matrix_report.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in prereg["training"]["reference_matrix"]["training_seeds"]]
    expected_jobs = [
        f"{model}_seed{seed}"
        for model in ("lewm", "pldm")
        for seed in seeds
    ]
    failures = matrix.get("failures")
    if (
        request.get("schema_version") != 1
        or request.get("status") != "running"
        or matrix.get("schema_version") != 1
        or matrix.get("status") != "failed"
        or matrix.get("preregistration_id") != V2_PREREGISTRATION_ID
        or matrix.get("reports") != []
        or not isinstance(failures, list)
        or [row.get("name") for row in failures] != expected_jobs
        or any(int(row.get("returncode", 0)) != 1 for row in failures)
    ):
        raise RuntimeError("Cube v2 matrix failure evidence drifted")

    jobs: dict[str, Any] = {}
    for name in expected_jobs:
        if JOB_PATTERN.fullmatch(name) is None:
            raise AssertionError(name)
        model, seed_text = name.split("_seed", 1)
        job = training_root / name
        if not job.is_dir() or job.is_symlink():
            raise RuntimeError(f"Cube v2 job directory is missing: {name}")
        artifacts = _validate_job_artifacts(
            job=job,
            model=model,
            seed=int(seed_text),
            prereg=prereg,
        )
        log = training_root / "logs" / f"{name}.log"
        text = log.read_text(encoding="utf-8")
        if (
            EXPECTED_ERROR not in text
            or EXPECTED_TRACE not in text
            or "Bulk-loading 2,048 complete" not in text
            or "Loading the isolated Loader Validation split" not in text
        ):
            raise RuntimeError(f"Cube v2 failure traceback drifted: {name}")
        jobs[name] = {
            "exit_code": 1,
            "log": {**_identity(log), "expected_error_present": True},
            "artifacts": artifacts,
        }
    forbidden = [
        path
        for path in training_root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".pt", ".ckpt"}
            or path.name == "training_report.json"
            or "snapshot" in path.name
        )
    ]
    if forbidden:
        raise RuntimeError(f"Cube v2 failure produced training artifacts: {forbidden}")

    receipt = {
        "schema_version": 1,
        "receipt_id": "cube_gripper_carry_h3_v4r1_reference_training_v2_failure",
        "preregistration_id": V2_PREREGISTRATION_ID,
        "status": "infrastructure_failed_after_model_load_before_forward",
        "classified_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": (
            "pinned_runtime_optional_loss_constructor_mismatch_"
            "not_scientific_failure"
        ),
        "failure_stage": (
            "shared_engine_eager_conditional_sigreg_construction_before_"
            "optimizer_and_forward"
        ),
        "checks_passed": True,
        "authorization_chain": {
            "preregistration": _identity(prereg_path),
            "freeze_receipt": _identity(Path(prereg["_freeze_receipt_path"])),
            "matrix_request": _identity(request_path),
            "matrix_report": _identity(matrix_path),
        },
        "jobs": jobs,
        "training_state": {
            "training_data_materialized": True,
            "loader_validation_materialized": True,
            "model_instantiated": True,
            "initial_checkpoint_loaded": True,
            "optimizer_instantiated": False,
            "forward_passes": 0,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "training_reports_created": 0,
            "config_files_created": 6,
            "training_provenance_files_created": 6,
        },
        "root_cause": {
            "pinned_class": "stable_worldmodel.wm.loss.ConditionalSIGReg",
            "unsupported_constructor_keyword_observed": "include_unpaired",
            "next_unsupported_constructor_keyword": "complete_haar_population",
            "shared_engine_eagerly_constructs_optional_losses": True,
            "authorized_recipes_use_conditional_sigreg": False,
            "shared_trainer_or_engine_change_required": False,
            "process_local_false_only_constructor_adapter_required": True,
        },
        "retry": {
            "authorized_under_v2": False,
            "v2_output_reusable": False,
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
