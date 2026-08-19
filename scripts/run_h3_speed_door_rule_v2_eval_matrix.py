#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/eval_tworoom_speed_door_rule_h3_v2.py"
SEEDS = (3072, 4096, 5120)
MODEL = {
    "speed_only": "H3_SpeedDoorV2_SpeedOnly_PLDM",
    "door_only": "H3_SpeedDoorV2_DoorOnly_PLDM",
    "joint": "H3_SpeedDoorV2_Joint_PLDM",
}
PREFIX = {
    "speed_only": "h3_sdr_v2_speed_only_pldm",
    "door_only": "h3_sdr_v2_door_only_pldm",
    "joint": "h3_sdr_v2_joint_pldm",
}


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        ROOT.parents[1] / "data/world_model/context_world"
    ).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("speed_only", "door_only", "joint"),
        required=True,
    )
    args = parser.parse_args()
    artifact_root = _artifact_root()
    checkpoint_root = artifact_root / "training/runs/checkpoints"
    report_root = artifact_root / "training/reports"
    result_root = (
        artifact_root
        / "evaluation/history3/speed_door_rule_composition_v2/results"
    )
    log_root = artifact_root / "training/logs"
    result_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (role, seed)
        for role in args.roles
        for seed in SEEDS
    ]
    running = []
    for gpu, (role, seed) in enumerate(jobs):
        run = f"{PREFIX[role]}_passage_formal_s{seed}"
        checkpoint = (
            checkpoint_root / run / "weights_final_step_1024.pt"
        )
        report = report_root / f"{run}.json"
        output = result_root / f"{MODEL[role]}_s{seed}.json"
        log_path = log_root / f"{run}_v2_eval.log"
        for path in (checkpoint, report):
            if not path.is_file():
                raise FileNotFoundError(path)
        if output.exists() or log_path.exists():
            raise FileExistsError(f"{output} / {log_path}")
        command = [
            "python",
            str(EVALUATOR),
            "--checkpoint",
            str(checkpoint),
            "--training-report",
            str(report),
            "--model-id",
            MODEL[role],
            "--training-seed",
            str(seed),
            "--role",
            role,
            "--adapter",
            "pldm",
            "--device",
            f"cuda:{gpu}",
            "--output",
            str(output),
        ]
        log = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append(
            (role, seed, process, log, log_path, output)
        )
        print(
            f"[v2-eval] started role={role} seed={seed} gpu={gpu}",
            flush=True,
        )
    while any(process.poll() is None for _, _, process, *_ in running):
        time.sleep(10)
        completed = sum(
            process.poll() is not None
            for _, _, process, *_ in running
        )
        print(
            f"[v2-eval] completed {completed}/{len(running)}",
            flush=True,
        )
    failures = []
    for role, seed, process, log, log_path, output in running:
        log.close()
        if process.returncode or not output.is_file():
            failures.append(
                {
                    "role": role,
                    "seed": seed,
                    "returncode": process.returncode,
                    "log": str(log_path),
                }
            )
    if failures:
        raise RuntimeError(f"v2 evaluation failures: {failures}")
    print(f"[v2-eval] all {len(running)} evaluations passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
