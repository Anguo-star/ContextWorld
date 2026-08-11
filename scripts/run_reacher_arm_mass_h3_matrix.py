#!/usr/bin/env python3
"""Run the frozen 2-model x 3-seed Reacher arm-mass training matrix."""

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

from contextworld.benchmarks.reacher_arm_mass_icl_data import (  # noqa: E402
    DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    load_reacher_arm_mass_icl_release,
    resolve_reacher_initial_checkpoint,
    resolve_reacher_original_h5,
    resolve_reacher_original_lance,
)
from contextworld.paths import artifact_path  # noqa: E402


TRAINER = ROOT / "scripts/run_reacher_arm_mass_h3_train.py"
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/reacher_robot_arm_mass_h3_release_v1/reference_matrix"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_REACHER_ARM_MASS_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve all six frozen jobs without creating output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_reacher_arm_mass_icl_release(release_path)
    matrix = release["training"]["reference_matrix"]
    seeds = tuple(int(value) for value in matrix["training_seeds"])
    jobs = [(model, seed) for model in ("lewm", "pldm") for seed in seeds]
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected {len(jobs)} distinct GPU indices, got {gpus}")
    original_h5 = resolve_reacher_original_h5(release, repo_root=ROOT)
    original_lance = resolve_reacher_original_lance(
        release,
        repo_root=ROOT,
    )
    checkpoints = {
        model: resolve_reacher_initial_checkpoint(
            release,
            model,
            repo_root=ROOT,
        )
        for model in ("lewm", "pldm")
    }
    for source in (original_h5, original_lance, *checkpoints.values()):
        if not source.exists():
            raise FileNotFoundError(source)
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
                    "--checkpoint",
                    str(checkpoints[model]),
                    "--original-h5",
                    str(original_h5),
                    "--original-lance",
                    str(original_lance),
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
            "--checkpoint",
            str(checkpoints[model]),
            "--original-h5",
            str(original_h5),
            "--original-lance",
            str(original_lance),
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
                if key not in {"stream", "process"}
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
        status = []
        for row in processes:
            process = row["process"]
            assert isinstance(process, subprocess.Popen)
            code = process.poll()
            if code is None:
                running += 1
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
        time.sleep(30)

    reports = []
    for row in processes:
        row["stream"].close()
        process = row["process"]
        assert isinstance(process, subprocess.Popen)
        if process.returncode:
            failures.append(
                {
                    "name": row["name"],
                    "returncode": process.returncode,
                    "log": str(row["log"]),
                }
            )
        else:
            reports.append(
                json.loads(
                    (Path(row["output"]) / "training_report.json").read_text()
                )
            )
    payload = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "release": str(release_path),
        "reports": reports,
        "failures": failures,
    }
    result_path = output / "matrix_report.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "result": str(result_path)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
