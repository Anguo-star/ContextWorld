#!/usr/bin/env python3
"""Seal the Cube v4r1 original-task retention decision without opening Public."""

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
    validate_cube_cem_retention_result,
)
from contextworld.paths import portable_contextworld_path  # noqa: E402


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": portable_contextworld_path(resolved, repo_root=ROOT),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def finalize(
    *, prereg_path: Path, result_path: Path, output: Path
) -> dict[str, Any]:
    prereg = load_cube_cem_retention_prereg(
        prereg_path, require_freeze=True, repo_root=ROOT
    )
    expected_root = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=ROOT
    )
    expected_result = expected_root / "retention_result.json"
    matrix_report_path = expected_root / "matrix_report.json"
    planned_output = resolve_declared_path(
        prereg["planned_artifacts"]["retention_decision"], repo_root=ROOT
    )
    if result_path.resolve() != expected_result:
        raise RuntimeError("Cube retention result path does not match preregistration")
    if output.resolve() != planned_output:
        raise RuntimeError("Cube retention decision path does not match preregistration")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Cube retention decision: {output}")

    validated = validate_cube_cem_retention_result(
        prereg,
        result_path=result_path,
        matrix_report_path=matrix_report_path,
        repo_root=ROOT,
    )
    result = validated["payload"]
    matrix = json.loads(matrix_report_path.read_text(encoding="utf-8"))
    expected_names = [row["model_name"] for row in expected_cube_cem_jobs(prereg)]
    if (
        matrix.get("schema_version") != 1
        or matrix.get("status") != "completed"
        or matrix.get("preregistration_id") != CUBE_CEM_RETENTION_ID
        or matrix.get("failures") != []
        or [row.get("name") for row in matrix.get("jobs", [])] != expected_names
        or any(int(row.get("exit_code", -1)) != 0 for row in matrix.get("jobs", []))
        or matrix.get("public_test") != closed_public_contract()
    ):
        raise RuntimeError("Cube CEM retention matrix report is incomplete")

    passed = bool(result["passed"])
    status = "passed_retention" if passed else "failed_retention"
    decision = {
        "schema_version": 1,
        "decision_id": "cube_gripper_carry_h3_v4r1_cem_retention_v1",
        "preregistration_id": CUBE_CEM_RETENTION_ID,
        "status": status,
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "original_cube_cem_retention_only_not_Public_or_release",
        "authorization_chain": {
            "preregistration": _identity(prereg_path),
            "freeze_receipt": _identity(Path(prereg["_freeze_receipt_path"])),
            "query_catalog": _identity(Path(prereg["_query_catalog_path"])),
            "matrix_report": _identity(matrix_report_path),
            "retention_result": _identity(result_path),
        },
        "baseline": result["baseline"],
        "comparisons": result["comparisons"],
        "passing_families": ["lewm"] if passed else [],
        "claims": {
            "reference_development_passed": True,
            "original_task_retention_passed": passed,
            "positive_reference_development_claim_allowed": passed,
            "public_test_claim_allowed": False,
            "release_claim_allowed": False,
            "suite_registration_allowed": False,
        },
        "next_step": (
            "freeze a separate one-use Public evaluation and release authorization "
            "for the LeWM three-checkpoint family"
            if passed
            else "stop; Public evaluation and release remain unauthorized"
        ),
        "public_test": closed_public_contract(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_CEM_RETENTION_PREREG
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = finalize(
        prereg_path=args.prereg.expanduser().resolve(),
        result_path=args.result.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": decision["status"],
                "passing_families": decision["passing_families"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
