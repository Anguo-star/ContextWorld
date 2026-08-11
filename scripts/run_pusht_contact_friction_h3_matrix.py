#!/usr/bin/env python3
"""Run the contact-friction jobs permitted by the frozen release decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    load_contact_friction_icl_release,
)
from contextworld.paths import artifact_path

TRAINER = ROOT / "scripts/run_pusht_contact_friction_h3_train.py"
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/pusht_contact_friction_h3_v1/reference_matrix"
)
MATRIX_DESCRIPTION = __doc__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=MATRIX_DESCRIPTION)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated CUDA indices matching the frozen jobs",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the frozen job matrix without creating output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    reference_matrix = release["training"]["reference_matrix"]
    failed_development = reference_matrix["status"] == "failed_development"
    seed_key = (
        "completed_development_seeds"
        if failed_development
        else "training_seeds"
    )
    seeds = tuple(int(value) for value in reference_matrix[seed_key])
    models = ("lewm",) if failed_development else ("lewm", "pldm")
    jobs = [
        (model, seed)
        for model in models
        for seed in seeds
    ]
    gpu_specification = args.gpus or ",".join(
        str(index) for index in range(len(jobs))
    )
    gpus = tuple(
        value.strip()
        for value in gpu_specification.split(",")
        if value.strip()
    )
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(
            f"Expected {len(jobs)} distinct GPU indices, got {gpus}"
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if args.dry_run:
        commands = []
        for (model, seed), gpu in zip(jobs, gpus):
            commands.append(
                [
                    sys.executable,
                    str(TRAINER),
                    "--release-config",
                    str(release_path),
                    "--model",
                    model,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output / f"{model}_seed{seed}"),
                    "--device",
                    f"cuda:{gpu}",
                    "--num-workers",
                    str(args.num_workers),
                    "--dry-run",
                ]
            )
        print(
            json.dumps(
                {
                    "status": "ready",
                    "release_id": release["release_id"],
                    "jobs": commands,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    logs = output / "logs"
    logs.mkdir(parents=True)

    processes: list[dict[str, object]] = []
    for (model, seed), gpu in zip(jobs, gpus):
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
                "started_at": time.time(),
            }
        )

    request = {
        "schema_version": 1,
        "status": "running",
        "release": str(release_path),
        "jobs": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in row.items()
                if key
                not in {"stream", "process", "started_at"}
            }
            for row in processes
        ],
    }
    (output / "matrix_request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures: list[dict[str, object]] = []
    while True:
        running = 0
        summary = []
        for row in processes:
            process = row["process"]
            assert isinstance(process, subprocess.Popen)
            code = process.poll()
            if code is None:
                running += 1
            summary.append(
                {
                    "name": row["name"],
                    "gpu": row["gpu"],
                    "status": "running" if code is None else f"exit_{code}",
                }
            )
        print(json.dumps({"jobs": summary}, sort_keys=True), flush=True)
        if running == 0:
            break
        time.sleep(30)

    reports = []
    for row in processes:
        stream = row["stream"]
        stream.close()
        process = row["process"]
        assert isinstance(process, subprocess.Popen)
        if process.returncode != 0:
            failures.append(
                {
                    "name": row["name"],
                    "returncode": process.returncode,
                    "log": str(row["log"]),
                }
            )
            continue
        report_path = Path(row["output"]) / "training_report.json"
        reports.append(
            json.loads(report_path.read_text(encoding="utf-8"))
        )

    matrix = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "release": str(release_path),
        "reports": reports,
        "failures": failures,
    }
    result_path = output / "matrix_report.json"
    result_path.write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": matrix["status"],
                "result": str(result_path),
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
