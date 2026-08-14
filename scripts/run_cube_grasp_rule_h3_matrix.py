#!/usr/bin/env python3
"""Run the frozen 2-model x 3-seed Cube gripper-carry training matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.cube_grasp_rule_reference_training import (  # noqa: E402
    DEFAULT_CUBE_REFERENCE_TRAINING_PREREG,
    file_sha256,
    load_cube_reference_training_prereg,
    validate_cube_reference_training_report,
)
from contextworld.paths import artifact_path, resolve_contextworld_path  # noqa: E402


TRAINER = ROOT / "scripts/run_cube_grasp_rule_h3_train.py"
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/cube_gripper_carry_h3_development_v4r1/"
    "reference_training_v3"
)


def _validate_completed_report(
    report: dict,
    *,
    row: dict,
    prereg: dict,
    prereg_path: Path,
) -> None:
    validated = validate_cube_reference_training_report(
        prereg,
        model_family=str(row["model"]),
        training_seed=int(row["seed"]),
        prereg_path=prereg_path,
        report_path=row["output"] / "training_report.json",
    )
    if validated["report_payload"] != report:
        raise RuntimeError("Cube training report changed during matrix validation")


def _require_cuda_devices(gpus: tuple[str, ...]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Cube formal matrix requires CUDA; no formal output was created"
        )
    count = int(torch.cuda.device_count())
    try:
        indices = tuple(int(value) for value in gpus)
    except ValueError as error:
        raise ValueError("Cube formal matrix GPU indices must be integers") from error
    if any(index < 0 or index >= count for index in indices):
        raise RuntimeError(
            f"Cube formal matrix requested GPUs {indices}, but only {count} are visible"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_CUBE_REFERENCE_TRAINING_PREREG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_cube_reference_training_prereg(release_path, require_freeze=True)
    matrix = release["training"]["reference_matrix"]
    seeds = tuple(int(value) for value in matrix["training_seeds"])
    jobs = [(model, seed) for model in ("lewm", "pldm") for seed in seeds]
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected {len(jobs)} distinct GPU indices, got {gpus}")
    output = args.output.expanduser().resolve()
    planned_output = resolve_contextworld_path(
        release["planned_artifacts"]["training_root"], repo_root=ROOT
    )
    if output != planned_output:
        raise RuntimeError("Cube formal matrix output does not match preregistration")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    environment = os.environ.copy()
    stable_repo = Path(
        release["runtime"]["stable_worldmodel"]["repo"]
    ).expanduser()
    if not stable_repo.is_absolute():
        stable_repo = (ROOT / stable_repo).resolve()
    if not stable_repo.is_dir():
        raise FileNotFoundError(f"Pinned Stable-WorldModel repo missing: {stable_repo}")
    _require_cuda_devices(gpus)
    logs = output / "logs"
    logs.mkdir(parents=True)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO": str(stable_repo),
        }
    )
    processes = []
    for (model, seed), gpu in zip(jobs, gpus, strict=True):
        name = f"{model}_seed{seed}"
        job_output = output / name
        log_path = logs / f"{name}.log"
        command = [
            sys.executable,
            str(TRAINER),
            "--release-config",
            str(release_path),
            "--model",
            model,
            "--seed",
            str(seed),
            "--output",
            str(job_output),
            "--device",
            f"cuda:{gpu}",
            "--num-workers",
            str(args.num_workers),
        ]
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        processes.append(
            {
                "name": name,
                "model": model,
                "seed": seed,
                "gpu": gpu,
                "command": command,
                "output": job_output,
                "log": log_path,
                "stream": stream,
                "process": process,
            }
        )
    (output / "matrix_request.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "release": str(release_path),
                "jobs": [
                    {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in row.items()
                        if key not in {"stream", "process"}
                    }
                    for row in processes
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    while True:
        status = []
        running = 0
        for row in processes:
            code = row["process"].poll()
            running += int(code is None)
            status.append(
                {
                    "name": row["name"],
                    "gpu": row["gpu"],
                    "status": "running" if code is None else f"exit_{code}",
                }
            )
        print(json.dumps({"jobs": status}, sort_keys=True), flush=True)
        if not running:
            break
        time.sleep(15)
    failures, reports = [], []
    for row in processes:
        row["stream"].close()
        if row["process"].returncode:
            failures.append(
                {
                    "name": row["name"],
                    "returncode": row["process"].returncode,
                    "log": str(row["log"]),
                }
            )
        else:
            report = json.loads(
                (row["output"] / "training_report.json").read_text()
            )
            _validate_completed_report(
                report,
                row=row,
                prereg=release,
                prereg_path=release_path,
            )
            reports.append(report)
    payload = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "preregistration_id": release["preregistration_id"],
        "training_root": str(output),
        "authorization_chain": {
            "preregistration": {
                "path": str(release_path),
                "sha256": file_sha256(release_path),
            },
            "freeze_receipt": {
                "path": release["_freeze_receipt_path"],
                "sha256": file_sha256(Path(release["_freeze_receipt_path"])),
            },
        },
        "reports": reports,
        "failures": failures,
    }
    result_path = output / "matrix_report.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "result": str(result_path)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
