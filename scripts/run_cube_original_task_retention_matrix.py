#!/usr/bin/env python3
"""Run the frozen baseline plus three-candidate Cube CEM retention matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_original_task_retention import (  # noqa: E402
    DEFAULT_CUBE_CEM_RETENTION_PREREG,
    build_cube_cem_retention_result,
    closed_public_contract,
    expected_cube_cem_jobs,
    file_sha256,
    load_cube_cem_retention_prereg,
    resolve_declared_path,
    validate_cube_cem_job_result,
)


def _identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _write_exclusive(path: Path, payload: MappingLike) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


MappingLike = dict[str, Any]


def _require_cuda_devices(gpus: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Cube CEM retention requires CUDA")
    count = int(torch.cuda.device_count())
    try:
        indices = tuple(int(value) for value in gpus)
    except ValueError as error:
        raise ValueError("Cube CEM GPU indices must be integers") from error
    if any(index < 0 or index >= count for index in indices):
        raise RuntimeError(
            f"Requested Cube CEM GPUs {indices}, but only {count} are visible"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prereg", type=Path, default=DEFAULT_CUBE_CEM_RETENTION_PREREG
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    prereg_path = args.prereg.expanduser().resolve()
    prereg = load_cube_cem_retention_prereg(
        prereg_path, require_freeze=True, repo_root=ROOT
    )
    jobs = expected_cube_cem_jobs(prereg)
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected {len(jobs)} distinct GPU indices, got {gpus}")
    _require_cuda_devices(gpus)

    output = args.output.expanduser().resolve()
    expected_output = resolve_declared_path(
        prereg["planned_artifacts"]["retention_root"], repo_root=ROOT
    )
    if output != expected_output:
        raise RuntimeError("Cube CEM retention output does not match preregistration")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Cube CEM output: {output}")

    evaluator = resolve_declared_path(
        prereg["identity"]["cem_evaluator"]["path"], repo_root=ROOT
    )
    stable = prereg["runtime"]["stable_worldmodel"]
    original_h5 = prereg["data"]["original_h5"]
    expected_h5 = original_h5["expected_identity"]
    query_catalog = Path(prereg["_query_catalog_path"])
    evaluation = prereg["evaluation"]
    logs = output / "logs"
    results = output / "results"
    logs.mkdir(parents=True)
    results.mkdir()

    commands: list[dict[str, Any]] = []
    for job, gpu in zip(jobs, gpus, strict=True):
        name = str(job["model_name"])
        result = results / name
        log = logs / f"{name}.log"
        command = [
            sys.executable,
            str(evaluator),
            "eval",
            "--stable-worldmodel-root",
            str(stable["repo"]),
            "--expected-ref",
            str(stable["expected_ref"]),
            "--model",
            f"{name}={job['checkpoint']}",
            "--dataset",
            str(original_h5["path"]),
            "--expected-dataset-size",
            str(expected_h5["size_bytes"]),
            "--expected-dataset-sha256",
            str(expected_h5["sha256"]),
            "--query-catalog",
            str(query_catalog),
            "--output",
            str(result),
            "--eval-seeds",
            ",".join(str(value) for value in evaluation["eval_seeds"]),
            "--num-eval",
            str(evaluation["queries_per_eval_seed"]),
            "--device",
            f"cuda:{gpu}",
        ]
        commands.append(
            {
                "name": name,
                "kind": job["kind"],
                "training_seed": job.get("training_seed"),
                "gpu": gpu,
                "result": str(result),
                "log": str(log),
                "command": command,
            }
        )

    request_path = output / "matrix_request.json"
    request = {
        "schema_version": 1,
        "status": "authorized_before_execution",
        "preregistration_id": prereg["preregistration_id"],
        "authorization_chain": {
            "preregistration": _identity(prereg_path),
            "freeze_receipt": _identity(Path(prereg["_freeze_receipt_path"])),
            "query_catalog": _identity(query_catalog),
        },
        "jobs": commands,
        "public_test": closed_public_contract(),
    }
    _write_exclusive(request_path, request)

    processes: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = str(evaluation["mujoco_gl"])
    for command in commands:
        stream = Path(command["log"]).open("x", encoding="utf-8")
        process = subprocess.Popen(
            command["command"],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
        )
        processes.append({**command, "process": process, "stream": stream})

    while True:
        statuses = []
        all_done = True
        for row in processes:
            code = row["process"].poll()
            if code is None:
                all_done = False
                status = "running"
            else:
                status = f"exit_{code}"
            statuses.append(
                {"name": row["name"], "gpu": row["gpu"], "status": status}
            )
        print(json.dumps({"jobs": statuses}, sort_keys=True), flush=True)
        if all_done:
            break
        time.sleep(30)

    for row in processes:
        row["stream"].close()
    failures = [
        {"name": row["name"], "exit_code": int(row["process"].returncode)}
        for row in processes
        if int(row["process"].returncode) != 0
    ]
    report_path = output / "matrix_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "preregistration_id": prereg["preregistration_id"],
        "request": _identity(request_path),
        "jobs": [
            {
                "name": row["name"],
                "kind": row["kind"],
                "training_seed": row["training_seed"],
                "gpu": row["gpu"],
                "exit_code": int(row["process"].returncode),
                "result": row["result"],
                "log": row["log"],
            }
            for row in processes
        ],
        "failures": failures,
        "public_test": closed_public_contract(),
    }
    if failures:
        _write_exclusive(report_path, report)
        raise RuntimeError(f"Cube CEM retention matrix failed: {failures}")

    validated = [
        validate_cube_cem_job_result(
            prereg, model_name=job["model_name"], repo_root=ROOT
        )
        for job in jobs
    ]
    report["results"] = [row["report_identity"] for row in validated]
    _write_exclusive(report_path, report)
    result = build_cube_cem_retention_result(
        prereg,
        validated=validated,
        matrix_report_path=report_path,
        repo_root=ROOT,
    )
    result_path = output / "retention_result.json"
    _write_exclusive(result_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "passed": result["passed"],
                "baseline_successes": result["baseline"]["success_count"],
                "candidate_successes": [
                    row["candidate_successes"] for row in result["comparisons"]
                ],
                "result": str(result_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
