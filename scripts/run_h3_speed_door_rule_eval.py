#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue


ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = ROOT / "scripts/eval_tworoom_speed_door_rule_h3.py"
AGGREGATE_SCRIPT = (
    ROOT / "scripts/aggregate_tworoom_speed_door_rule_h3.py"
)


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        ROOT.parents[1] / "data/world_model/context_world"
    ).resolve()


def _jobs(root: Path) -> list[dict]:
    checkpoint_root = root / "training/runs/checkpoints"
    report_root = root / "training/reports"
    jobs = [
        {
            "model_id": "H3_Original_LEWM",
            "training_model_id": "H3_Original_LEWM",
            "seed": 3072,
            "adapter": "lewm",
            "checkpoint": (
                checkpoint_root
                / "h3_origheldout_s3072/weights_final_step_6420.pt"
            ),
            "report": None,
        }
    ]
    for seed in (3072, 4096, 5120):
        door_run = (
            "h3_passage_mixed_rules_pldm_objective_"
            f"passage_formal_s{seed}"
        )
        jobs.append(
            {
                "model_id": "H3_DoorOnly_PLDM",
                "training_model_id": (
                    "H3_Passage_MixedRules_PLDMObjective"
                ),
                "seed": seed,
                "adapter": "pldm",
                "checkpoint": (
                    checkpoint_root
                    / door_run
                    / "weights_final_step_1024.pt"
                ),
                "report": report_root / f"{door_run}.json",
            }
        )
    for public_id, run_prefix in (
        ("H3_SpeedOnly_PLDM", "h3_speed_only_pldm"),
        ("H3_SpeedDoorJoint_PLDM", "h3_speed_door_joint_pldm"),
    ):
        for seed in (3072, 4096, 5120):
            run = f"{run_prefix}_passage_formal_s{seed}"
            jobs.append(
                {
                    "model_id": public_id,
                    "training_model_id": public_id,
                    "seed": seed,
                    "adapter": "pldm",
                    "checkpoint": (
                        checkpoint_root
                        / run
                        / "weights_final_step_1024.pt"
                    ),
                    "report": report_root / f"{run}.json",
                }
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ten-checkpoint composition matrix across free GPUs"
        )
    )
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.gpus <= 8:
        parser.error("--gpus must be in [1, 8]")

    root = _artifact_root()
    results_root = (
        root
        / "evaluation/history3/"
        "speed_door_rule_composition_validation_v1/results"
    )
    results_root.mkdir(parents=True, exist_ok=True)
    jobs = _jobs(root)
    missing = [
        str(value)
        for job in jobs
        for value in (job["checkpoint"], job["report"])
        if value is not None and not Path(value).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Evaluation matrix is incomplete:\n" + "\n".join(missing)
        )

    gpu_queue: Queue[int] = Queue()
    for gpu in range(args.gpus):
        gpu_queue.put(gpu)

    def run_one(job: dict) -> str:
        output = (
            results_root
            / f"{job['model_id']}_s{job['seed']}.json"
        )
        if output.exists():
            if args.resume:
                return f"reused {output.name}"
            raise FileExistsError(output)
        gpu = gpu_queue.get()
        try:
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--checkpoint",
                str(job["checkpoint"]),
                "--model-id",
                str(job["model_id"]),
                "--training-model-id",
                str(job["training_model_id"]),
                "--training-seed",
                str(job["seed"]),
                "--adapter",
                str(job["adapter"]),
                "--device",
                "cuda:0",
                "--output",
                str(output),
            ]
            if job["report"] is not None:
                command.extend(
                    ["--training-report", str(job["report"])]
                )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{job['model_id']} seed {job['seed']} failed on "
                    f"GPU {gpu}\nSTDOUT:\n{completed.stdout}\n"
                    f"STDERR:\n{completed.stderr}"
                )
            return (
                f"completed {job['model_id']} seed={job['seed']} "
                f"gpu={gpu}"
            )
        finally:
            gpu_queue.put(gpu)

    with ThreadPoolExecutor(max_workers=args.gpus) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            print(future.result(), flush=True)

    aggregate = (
        root
        / "evaluation/history3/"
        "speed_door_rule_composition_validation_v1/aggregate.json"
    )
    if aggregate.exists():
        raise FileExistsError(aggregate)
    subprocess.run(
        [
            sys.executable,
            str(AGGREGATE_SCRIPT),
            "--results-root",
            str(results_root),
            "--output",
            str(aggregate),
        ],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
