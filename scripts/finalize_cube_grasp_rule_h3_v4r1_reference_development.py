#!/usr/bin/env python3
"""Seal the Cube v4r1 LeWM/PLDM Development decision without opening Public."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]

from contextworld.benchmarks.cube_grasp_rule_reference_score import (  # noqa: E402
    validate_cube_reference_development_method,
)
from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    CUBE_REFERENCE_TRAINING_ID,
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    file_sha256,
    load_cube_reference_training_prereg,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path  # noqa: E402


def _closed_public() -> dict[str, Any]:
    return {
        "access_status": "closed_not_read_not_scored",
        "generated": False,
        "opened": False,
        "read": False,
        "hashed": False,
        "scored": False,
    }


def finalize(
    *, prereg_path: Path, matrix_score_path: Path, output: Path
) -> dict[str, Any]:
    prereg = load_cube_reference_training_prereg(
        prereg_path, require_freeze=True, repo_root=ROOT
    )
    planned = resolve_contextworld_path(
        prereg["planned_artifacts"]["development_decision"], repo_root=ROOT
    )
    if output.resolve() != planned:
        raise RuntimeError("Cube Development decision output does not match prereg")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Development decision: {output}")
    expected_matrix_score = resolve_contextworld_path(
        prereg["planned_artifacts"]["development_score_root"], repo_root=ROOT
    ) / "matrix_score.json"
    if matrix_score_path.resolve() != expected_matrix_score:
        raise RuntimeError("Cube Development matrix score path is not preregistered")
    matrix = json.loads(matrix_score_path.read_text(encoding="utf-8"))
    prereg_config = Path(prereg["_config_path"])
    freeze_path = Path(prereg["_freeze_receipt_path"])
    training_root = resolve_contextworld_path(
        prereg["planned_artifacts"]["training_root"], repo_root=ROOT
    )
    training_matrix_path = training_root / "matrix_report.json"
    expected_chain = {
        "preregistration": {
            "path": str(prereg_config),
            "sha256": file_sha256(prereg_config),
        },
        "freeze_receipt": {
            "path": str(freeze_path),
            "sha256": file_sha256(freeze_path),
        },
        "training_matrix": {
            "path": str(training_matrix_path),
            "sha256": file_sha256(training_matrix_path),
        },
    }
    if (
        matrix.get("schema_version") != 1
        or matrix.get("status") != "completed"
        or matrix.get("preregistration_id") != CUBE_REFERENCE_TRAINING_ID
        or matrix.get("public_test_opened") is not False
        or set(matrix.get("methods", {})) != {"lewm", "pldm"}
        or Path(str(matrix.get("training_root", ""))).resolve() != training_root
        or matrix.get("authorization_chain") != expected_chain
    ):
        raise RuntimeError("Cube Development matrix score is incomplete or contaminated")
    expected_seeds = sorted(
        int(value)
        for value in prereg["training"]["reference_matrix"]["training_seeds"]
    )
    families: dict[str, Any] = {}
    for family in ("lewm", "pldm"):
        method = validate_cube_reference_development_method(
            matrix["methods"][family], prereg=prereg, model_family=family
        )
        observed_pass = all(
            row["gate"]["passed"] for row in method["checkpoint_results"]
        )
        if method.get("passed") is not observed_pass:
            raise RuntimeError(f"Cube {family} aggregate pass flag drifted")
        families[family] = {
            "training_recipe": method["training_recipe"],
            "training_seeds": expected_seeds,
            "checkpoints_passed": sum(
                int(row["gate"]["passed"])
                for row in method["checkpoint_results"]
            ),
            "checkpoints_required": 3,
            "passed": observed_pass,
            "aggregate": method["aggregate"],
        }
    passing = [name for name, value in families.items() if value["passed"]]
    status = "passed_development" if passing else "failed_development"
    decision = {
        "schema_version": 1,
        "decision_id": "cube_gripper_carry_h3_v4r1_reference_development_v3",
        "preregistration_id": CUBE_REFERENCE_TRAINING_ID,
        "status": status,
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "reference_model_development_only_not_CEM_or_Public",
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
            "matrix_score": {
                "path": portable_contextworld_path(matrix_score_path, repo_root=ROOT),
                "sha256": file_sha256(matrix_score_path),
                "size_bytes": matrix_score_path.stat().st_size,
            },
        },
        "families": families,
        "passing_families": passing,
        "claims": {
            "reference_development_passed": bool(passing),
            "positive_reference_claim_allowed": False,
            "original_task_retention_claim_allowed": False,
            "public_test_claim_allowed": False,
            "release_claim_allowed": False,
            "suite_registration_allowed": False,
        },
        "next_step": (
            "freeze a separate original-Cube CEM retention authorization for "
            + ", ".join(passing)
            if passing
            else "stop; a new preregistration is required for any redesign"
        ),
        "original_task_retention": {
            "authorized": False,
            "run": False,
        },
        "public_test": _closed_public(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(decision, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return decision


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG)
    parser.add_argument("--matrix-score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    decision = finalize(
        prereg_path=args.prereg.expanduser().resolve(),
        matrix_score_path=args.matrix_score.expanduser().resolve(),
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
