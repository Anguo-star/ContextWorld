#!/usr/bin/env python3
"""Run the frozen 4-checkpoint x 6-seed TwoRoom CEM retention matrix."""

from __future__ import annotations

import argparse
import json
import os
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
from contextworld.paths import artifact_path, resolve_contextworld_path  # noqa: E402


EVALUATOR = ROOT / "scripts/eval_tworoom_ability_catalog.py"
DEFAULT_TRAINING_ROOT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/"
    "development_recipe_screen"
)
DEFAULT_OUTPUT = artifact_path(
    "evaluation/history3/tworoom_portal_exit_h3_release_v1/"
    "original_task_retention"
)


def _completed_result(path: Path, *, eval_seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            payload.get("status") == "passed"
            and int(payload["protocol"]["eval_seed"]) == eval_seed
            and int(payload["aggregate"]["evaluations"]) == 50
            and payload["frozen_weight_audit"]["passed"] is True
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config", type=Path, default=DEFAULT_PORTAL_EXIT_RELEASE_CONFIG
    )
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_portal_exit_icl_release(release_path)
    retention = release["scoring"]["original_task_retention"]
    seeds = tuple(int(value) for value in retention["eval_seeds"])
    if len(seeds) != 6 or int(retention["episodes_per_eval_seed"]) != 50:
        raise ValueError("Portal Exit retention requires six independent 50-query cells")
    training_root = args.training_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    request_path = output / "matrix_request.json"
    if output.exists() and not request_path.is_file():
        raise FileExistsError(
            f"Refusing to resume an unrecognized output directory: {output}"
        )
    if request_path.is_file():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("release_id") != release["release_id"]:
            raise ValueError("Existing CEM output belongs to another release")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    initial = resolve_contextworld_path(
        release["training"]["initialization"]["checkpoint"], repo_root=ROOT
    )
    steps = int(release["training"]["reference_matrix"]["common"]["optimizer_steps"])
    variant = "mixed_frozen_image_paired_future_fit_1p00"
    checkpoints = [("baseline_lewm", initial)] + [
        (
            f"icl_lewm_seed{seed}",
            training_root
            / f"lewm_paired_fit_seed{seed}"
            / f"{variant}_step{steps}.pt",
        )
        for seed in release["training"]["reference_matrix"]["training_seeds"]
    ]
    missing = [path for _, path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(map(str, missing)))
    catalog = resolve_contextworld_path(
        retention["query_catalog"]["path"], repo_root=ROOT
    )
    normalizer = resolve_contextworld_path(
        release["training"]["initialization"]["frozen_normalizer"], repo_root=ROOT
    )
    gpus = tuple(value.strip() for value in args.gpus.split(",") if value.strip())
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must contain distinct device indices")

    jobs = []
    for model, checkpoint in checkpoints:
        for eval_seed in seeds:
            result = output / model / f"seed{eval_seed}.json"
            log = logs / f"{model}_seed{eval_seed}.log"
            jobs.append(
                {
                    "model": model,
                    "checkpoint": checkpoint,
                    "eval_seed": eval_seed,
                    "result": result,
                    "log": log,
                }
            )
    request = {
        "schema_version": 1,
        "status": "running",
        "release_id": release["release_id"],
        "catalog": str(catalog),
        "normalizer": str(normalizer),
        "stable_worldmodel_ref": release["runtime"]["stable_worldmodel"]["expected_ref"],
        "jobs": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in job.items()
            }
            for job in jobs
        ],
    }
    completed_before_resume = sum(
        _completed_result(job["result"], eval_seed=job["eval_seed"])
        for job in jobs
    )
    request["completed_before_resume"] = completed_before_resume
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pending = [
        job
        for job in jobs
        if not _completed_result(job["result"], eval_seed=job["eval_seed"])
    ]
    failures = []
    worker_environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    while pending:
        wave = pending[: len(gpus)]
        pending = pending[len(gpus) :]
        processes = []
        for job, gpu in zip(wave, gpus[: len(wave)], strict=True):
            job["result"].parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(EVALUATOR),
                "--catalog",
                str(catalog),
                "--checkpoint",
                str(job["checkpoint"]),
                "--normalizer",
                str(normalizer),
                "--output",
                str(job["result"]),
                "--seed",
                str(job["eval_seed"]),
                "--stablewm-repo",
                args.stablewm_repo,
                "--stablewm-ref",
                release["runtime"]["stable_worldmodel"]["expected_ref"],
                "--device",
                f"cuda:{gpu}",
                "--eval-budget",
                "50",
                "--horizon",
                "5",
                "--receding-horizon",
                "5",
                "--cem-samples",
                "300",
                "--cem-steps",
                "30",
                "--cem-topk",
                "30",
                "--expected-history-size",
                "3",
            ]
            stream = job["log"].open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=worker_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((job, gpu, command, stream, process))
        while any(process.poll() is None for *_, process in processes):
            print(
                json.dumps(
                    {
                        "running": [
                            f"{job['model']}/seed{job['eval_seed']}/gpu{gpu}"
                            for job, gpu, _, _, process in processes
                            if process.poll() is None
                        ]
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(15)
        for job, _, command, stream, process in processes:
            stream.close()
            if process.returncode:
                failures.append(
                    {
                        "model": job["model"],
                        "eval_seed": job["eval_seed"],
                        "returncode": process.returncode,
                        "command": command,
                        "log": str(job["log"]),
                    }
                )
        if failures:
            break

    report = {
        "schema_version": 1,
        "status": "failed" if failures else "completed",
        "release_id": release["release_id"],
        "jobs": len(jobs),
        "completed_results": sum(job["result"].is_file() for job in jobs),
        "failures": failures,
    }
    (output / "matrix_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
