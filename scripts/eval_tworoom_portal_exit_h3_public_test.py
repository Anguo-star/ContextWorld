#!/usr/bin/env python3
"""Evaluate the selected Portal Exit LeWM recipe and the official PLDM control."""

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

from contextworld.benchmarks.portal_exit_icl_data import (  # noqa: E402
    DEFAULT_PORTAL_EXIT_RELEASE_CONFIG,
    load_portal_exit_icl_release,
)
from contextworld.benchmarks.portal_exit_icl_score import (  # noqa: E402
    score_portal_exit_icl_results,
)
from contextworld.paths import artifact_path  # noqa: E402


DEFAULT_LEWM_ROOT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/"
    "development_recipe_screen"
)
DEFAULT_PLDM_ROOT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/reference_matrix"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/public_test_matrix"
)
VARIANTS = {
    "lewm": "mixed_frozen_image_paired_future_fit_1p00",
    "pldm": "mixed_pldm_joint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    parser.add_argument("--lewm-training-root", type=Path, default=DEFAULT_LEWM_ROOT)
    parser.add_argument("--pldm-training-root", type=Path, default=DEFAULT_PLDM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_portal_exit_icl_release(release_path)
    seeds = tuple(
        int(value)
        for value in release["training"]["reference_matrix"]["training_seeds"]
    )
    steps = int(release["training"]["reference_matrix"]["common"]["optimizer_steps"])
    roots = {
        "lewm": args.lewm_training_root.expanduser().resolve(),
        "pldm": args.pldm_training_root.expanduser().resolve(),
    }
    jobs = []
    for model in ("lewm", "pldm"):
        variant = VARIANTS[model]
        for seed in seeds:
            directory = (
                f"lewm_paired_fit_seed{seed}"
                if model == "lewm"
                else f"pldm_seed{seed}"
            )
            checkpoint = roots[model] / directory / f"{variant}_step{steps}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            jobs.append(
                {
                    "model": model,
                    "seed": seed,
                    "variant": variant,
                    "checkpoint": checkpoint,
                }
            )
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus):
        raise ValueError(f"Expected {len(jobs)} distinct GPU indices, got {gpus}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    logs = output / "logs"
    checkpoints = output / "checkpoints"
    logs.mkdir(parents=True)
    checkpoints.mkdir()

    processes = []
    for job, gpu in zip(jobs, gpus, strict=True):
        name = f"{job['model']}_seed{job['seed']}"
        result = checkpoints / f"{name}.json"
        log = logs / f"{name}.log"
        command = [
            sys.executable,
            "-m",
            "contextworld.benchmarks.portal_exit_icl_cli",
            "--release-config",
            str(release_path),
            "eval",
            "--checkpoint",
            str(job["checkpoint"]),
            "--adapter",
            job["model"],
            "--model-name",
            f"portal_exit_{name}",
            "--training-recipe",
            f"portal_exit_{job['variant']}",
            "--training-seed",
            str(job["seed"]),
            "--device",
            f"cuda:{gpu}",
            "--batch-size",
            str(args.batch_size),
            "--output",
            str(result),
        ]
        stream = log.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((job, gpu, result, log, command, stream, process))
    request = {
        "schema_version": 1,
        "status": "running",
        "release_id": release["release_id"],
        "jobs": [
            {
                "model": job["model"],
                "seed": job["seed"],
                "variant": job["variant"],
                "checkpoint": str(job["checkpoint"]),
                "gpu": gpu,
            }
            for job, gpu, *_ in processes
        ],
    }
    (output / "matrix_request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    while any(process.poll() is None for *_, process in processes):
        print(
            json.dumps(
                {
                    "running": [
                        f"{job['model']}/seed{job['seed']}/gpu{gpu}"
                        for job, gpu, _, _, _, _, process in processes
                        if process.poll() is None
                    ]
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(15)
    failures = []
    for job, _, _, log, command, stream, process in processes:
        stream.close()
        if process.returncode:
            failures.append(
                {
                    "model": job["model"],
                    "seed": job["seed"],
                    "returncode": process.returncode,
                    "command": command,
                    "log": str(log),
                }
            )
    if failures:
        payload = {"schema_version": 1, "status": "failed", "failures": failures}
        (output / "matrix_score.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(f"Portal Exit Public Test failed: {failures}")

    methods = {}
    for model in ("lewm", "pldm"):
        result_paths = [
            result for job, _, result, *_ in processes if job["model"] == model
        ]
        methods[model] = score_portal_exit_icl_results(
            result_paths=result_paths,
            method_name=f"portal_exit_{VARIANTS[model]}",
            release_config=release_path,
        )
        (output / f"{model}_three_seed_score.json").write_text(
            json.dumps(methods[model], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "release_id": release["release_id"],
        "methods": methods,
    }
    result_path = output / "matrix_score.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "result": str(result_path),
                "passed": {name: row["passed"] for name, row in methods.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
