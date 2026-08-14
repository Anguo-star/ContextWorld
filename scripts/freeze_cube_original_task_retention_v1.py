#!/usr/bin/env python3
"""Freeze the post-Development Cube original-task CEM retention run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_original_task_retention import (  # noqa: E402
    DEFAULT_CUBE_CEM_RETENTION_PREREG,
    TOTAL_QUERIES,
    closed_public_contract,
    collect_cube_cem_static_identities,
    expected_cube_cem_jobs,
    file_sha256,
    load_cube_cem_retention_prereg,
    resolve_declared_path,
    validate_cube_cem_query_catalog,
)


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _last_json_line(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            value = json.loads(stripped)
            if isinstance(value, dict):
                return value
    raise RuntimeError("Cube CEM preflight did not emit a JSON summary")


def freeze(*, prereg_path: Path, output: Path) -> dict[str, Any]:
    prereg = load_cube_cem_retention_prereg(
        prereg_path, require_freeze=False, repo_root=ROOT
    )
    planned = prereg["planned_artifacts"]
    expected_output = resolve_declared_path(planned["freeze_receipt"], repo_root=ROOT)
    query_path = resolve_declared_path(planned["query_catalog"], repo_root=ROOT)
    retention_root = resolve_declared_path(planned["retention_root"], repo_root=ROOT)
    decision_path = resolve_declared_path(
        planned["retention_decision"], repo_root=ROOT
    )
    if output.resolve() != expected_output:
        raise RuntimeError("Cube CEM freeze output does not match preregistration")
    for path, label in (
        (output, "freeze receipt"),
        (query_path, "query catalog"),
        (retention_root, "retention root"),
        (decision_path, "retention decision"),
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Cube CEM {label}: {path}")

    static = collect_cube_cem_static_identities(prereg, repo_root=ROOT)
    evaluator = resolve_declared_path(
        prereg["identity"]["cem_evaluator"]["path"], repo_root=ROOT
    )
    stable = prereg["runtime"]["stable_worldmodel"]
    jobs = expected_cube_cem_jobs(prereg)
    preflight_command = [
        sys.executable,
        str(evaluator),
        "preflight-models",
        "--stable-worldmodel-root",
        str(stable["repo"]),
        "--expected-ref",
        str(stable["expected_ref"]),
    ]
    for job in jobs:
        preflight_command.extend(
            ["--model", f"{job['model_name']}={job['checkpoint']}"]
        )
    preflight_run = subprocess.run(
        preflight_command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    model_preflight = _last_json_line(preflight_run.stdout)
    expected_names = [job["model_name"] for job in jobs]
    if (
        [row.get("model") for row in model_preflight.get("models", [])]
        != expected_names
        or any(
            row.get("strict_load") is not True
            for row in model_preflight.get("models", [])
        )
        or model_preflight.get("runtime", {}).get("commit")
        != stable["expected_ref"]
        or model_preflight.get("runtime", {}).get("clean") is not True
    ):
        raise RuntimeError("Cube CEM model preflight drifted")

    original_h5 = prereg["data"]["original_h5"]
    evaluation = prereg["evaluation"]
    with tempfile.TemporaryDirectory(prefix="contextworld-cube-cem-query-") as raw:
        temporary_query = Path(raw) / "query_catalog.json"
        query_command = [
            sys.executable,
            str(evaluator),
            "prepare-queries",
            "--stable-worldmodel-root",
            str(stable["repo"]),
            "--expected-ref",
            str(stable["expected_ref"]),
            "--dataset",
            str(original_h5["path"]),
            "--expected-dataset-sha256",
            str(original_h5["expected_identity"]["sha256"]),
            "--eval-seeds",
            ",".join(str(value) for value in evaluation["eval_seeds"]),
            "--num-eval",
            str(evaluation["queries_per_eval_seed"]),
            "--output",
            str(temporary_query),
        ]
        query_run = subprocess.run(
            query_command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        query_preflight = _last_json_line(query_run.stdout)
        validated_query = validate_cube_cem_query_catalog(
            prereg, path=temporary_query
        )
        if (
            int(query_preflight.get("query_count", -1)) != TOTAL_QUERIES
            or query_preflight.get("sha256")
            != validated_query["identity"]["sha256"]
        ):
            raise RuntimeError("Cube CEM query materialization drifted")
        query_bytes = temporary_query.read_bytes()

    query_path.parent.mkdir(parents=True, exist_ok=True)
    with query_path.open("xb") as stream:
        stream.write(query_bytes)
    query_identity = _identity(query_path)
    authorized_jobs = [
        {
            "kind": row["kind"],
            "model_family": row["model_family"],
            "model_name": row["model_name"],
            **(
                {"training_seed": int(row["training_seed"])}
                if row["kind"] == "candidate"
                else {}
            ),
        }
        for row in jobs
    ]
    receipt = {
        "schema_version": 1,
        "status": "frozen_authorized",
        "preregistration_id": prereg["preregistration_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": _identity(prereg_path),
        "static_identities": static,
        "model_preflight": model_preflight,
        "query_catalog": query_identity,
        "authorization": {
            "jobs": authorized_jobs,
            "jobs_count": len(authorized_jobs),
            "episodes_per_job": TOTAL_QUERIES,
            "total_cem_episodes": TOTAL_QUERIES * len(authorized_jobs),
            "baseline_and_candidates_share_frozen_queries": True,
            "noninferiority_margin_successes": evaluation[
                "noninferiority_margin_successes"
            ],
        },
        "public_test": closed_public_contract(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_CEM_RETENTION_PREREG
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        prereg_path=args.prereg.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "jobs": receipt["authorization"]["jobs_count"],
                "total_cem_episodes": receipt["authorization"][
                    "total_cem_episodes"
                ],
                "query_catalog_sha256": receipt["query_catalog"]["sha256"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
