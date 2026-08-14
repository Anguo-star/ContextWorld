#!/usr/bin/env python3
"""Archive the zero-episode Cube CEM v1 EGL infrastructure failure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_original_task_retention import (  # noqa: E402
    CUBE_CEM_RETENTION_ID,
    DEFAULT_CUBE_CEM_RETENTION_PREREG,
    closed_public_contract,
    expected_cube_cem_jobs,
    file_sha256,
    load_cube_cem_retention_prereg,
    resolve_declared_path,
)
from contextworld.paths import portable_contextworld_path  # noqa: E402


ERROR_MARKER = "ImportError: Cannot initialize a headless EGL display."
BACKEND_MARKER = "MUJOCO_GL=egl, attempting to import specified OpenGL backend."


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(resolved)
    return {
        "path": portable_contextworld_path(resolved, repo_root=ROOT),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def archive(*, prereg_path: Path, attempt_root: Path, output: Path) -> dict[str, Any]:
    prereg = load_cube_cem_retention_prereg(
        prereg_path, require_freeze=True, repo_root=ROOT
    )
    expected_root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=ROOT
    )
    if attempt_root.resolve() != expected_root:
        raise RuntimeError("Cube CEM v1 attempt root drifted")
    expected_output = attempt_root / "infrastructure_failure_receipt.json"
    if output.resolve() != expected_output:
        raise RuntimeError("Cube CEM v1 failure receipt path drifted")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failure receipt: {output}")

    request_path = attempt_root / "matrix_request.json"
    report_path = attempt_root / "matrix_report.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    jobs = expected_cube_cem_jobs(prereg)
    expected_names = [row["model_name"] for row in jobs]
    if (
        request.get("schema_version") != 1
        or request.get("status") != "authorized_before_execution"
        or request.get("preregistration_id") != CUBE_CEM_RETENTION_ID
        or [row.get("name") for row in request.get("jobs", [])] != expected_names
        or request.get("public_test") != closed_public_contract()
        or report.get("schema_version") != 1
        or report.get("status") != "failed"
        or report.get("preregistration_id") != CUBE_CEM_RETENTION_ID
        or [row.get("name") for row in report.get("jobs", [])] != expected_names
        or any(int(row.get("exit_code", -1)) != 1 for row in report.get("jobs", []))
        or [row.get("name") for row in report.get("failures", [])] != expected_names
        or report.get("public_test") != closed_public_contract()
    ):
        raise RuntimeError("Cube CEM v1 matrix failure provenance drifted")

    evidence = []
    for name in expected_names:
        log = attempt_root / "logs" / f"{name}.log"
        result_dir = attempt_root / "results" / name
        text = log.read_text(encoding="utf-8")
        files = sorted(path.name for path in result_dir.iterdir() if path.is_file())
        catalog = result_dir / "query_catalog.json"
        if (
            ERROR_MARKER not in text
            or BACKEND_MARKER not in text
            or f"[cube/{name}] CEM seed=42" not in text
            or "success=" in text
            or files != ["query_catalog.json"]
            or catalog.read_bytes()
            != Path(prereg["_query_catalog_path"]).read_bytes()
        ):
            raise RuntimeError(f"Cube CEM v1 zero-episode evidence drifted: {name}")
        evidence.append(
            {
                "model_name": name,
                "log": _identity(log),
                "result_directory": str(result_dir.resolve()),
                "files": files,
                "query_catalog": _identity(catalog),
                "aggregate_report_created": False,
                "world_evaluate_calls": 0,
                "completed_eval_seeds": 0,
                "completed_episodes": 0,
                "success_records": 0,
                "error_marker": ERROR_MARKER,
            }
        )

    receipt = {
        "schema_version": 1,
        "receipt_id": "cube_gripper_carry_h3_v4r1_cem_retention_v1_egl_failure",
        "preregistration_id": CUBE_CEM_RETENTION_ID,
        "status": "archived_zero_episode_infrastructure_failure",
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "render_backend_initialization_failure_not_model_result",
        "failure_stage": "first_environment_construction_before_world_evaluate",
        "root_cause": {
            "configured_backend": "egl",
            "error": ERROR_MARKER,
            "missing_library_observed": "libOpenGL.so.0",
            "all_four_jobs_same_failure": True,
        },
        "authorization_chain": {
            "preregistration": _identity(prereg_path),
            "freeze_receipt": _identity(Path(prereg["_freeze_receipt_path"])),
            "query_catalog": _identity(Path(prereg["_query_catalog_path"])),
            "matrix_request": _identity(request_path),
            "matrix_report": _identity(report_path),
        },
        "execution": {
            "jobs_authorized": 4,
            "jobs_exit_1": 4,
            "models_strictly_loaded": 4,
            "environment_constructions_completed": 0,
            "world_evaluate_calls": 0,
            "cem_episodes_completed": 0,
            "aggregate_reports_created": 0,
            "retention_result_created": False,
            "retention_decision_created": False,
        },
        "jobs": evidence,
        "scientific_interpretation": {
            "baseline_scored": False,
            "candidates_scored": False,
            "retention_pass_or_fail_observed": False,
            "retry_in_same_namespace_authorized": False,
            "new_recovery_preregistration_required": True,
        },
        "public_test": closed_public_contract(),
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_CEM_RETENTION_PREREG
    )
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = archive(
        prereg_path=args.prereg.expanduser().resolve(),
        attempt_root=args.attempt_root.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cem_episodes_completed": receipt["execution"][
                    "cem_episodes_completed"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
