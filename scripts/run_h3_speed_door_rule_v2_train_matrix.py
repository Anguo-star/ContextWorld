#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_h3_speed_door_rule_v2_train.sh"
SEEDS = (3072, 4096, 5120)
RUN_PREFIX = {
    "speed_only": "h3_sdr_v2_speed_only_pldm",
    "door_only": "h3_sdr_v2_door_only_pldm",
    "joint": "h3_sdr_v2_joint_pldm",
}
MODEL_ID = {
    "speed_only": "H3_SpeedDoorV2_SpeedOnly_PLDM",
    "door_only": "H3_SpeedDoorV2_DoorOnly_PLDM",
    "joint": "H3_SpeedDoorV2_Joint_PLDM",
}


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        ROOT.parents[1] / "data/world_model/context_world"
    ).resolve()


def _gpu_rows() -> list[tuple[int, int, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = []
    for line in completed.stdout.splitlines():
        if line.strip():
            rows.append(
                tuple(int(value.strip()) for value in line.split(","))
            )
    if len(rows) != 8:
        raise RuntimeError(f"Expected eight GPUs, got {rows}")
    return rows


def _wait_for_gpus() -> None:
    while any(
        utilization > 5 or memory > 1024
        for _, utilization, memory in _gpu_rows()
    ):
        print("[v2-train] waiting for an exclusive 8-GPU window", flush=True)
        time.sleep(30)


def _tail(path: Path, lines: int = 100) -> str:
    return "\n".join(
        path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-lines:]
    )


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
    report_root = artifact_root / "training/reports"
    checkpoint_root = artifact_root / "training/runs/checkpoints"
    log_root = artifact_root / "training/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for role in args.roles:
        for seed in SEEDS:
            run = f"{RUN_PREFIX[role]}_passage_formal_s{seed}"
            report_path = report_root / f"{run}.json"
            checkpoint_path = (
                checkpoint_root / run / "weights_final_step_1024.pt"
            )
            if report_path.exists() or checkpoint_path.exists():
                raise FileExistsError(
                    "Refusing to overwrite a formal matrix member: "
                    f"{report_path} / {checkpoint_path}"
                )
            log_path = log_root / f"{run}_matrix.log"
            if log_path.exists():
                raise FileExistsError(log_path)
            _wait_for_gpus()
            environment = dict(os.environ)
            environment["TRAINING_SEED"] = str(seed)
            environment["logger_backend"] = "none"
            started = time.monotonic()
            print(
                f"[v2-train] start role={role} seed={seed}; "
                f"log={log_path}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    ["bash", str(LAUNCHER), role, "formal"],
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    time.sleep(30)
                    elapsed = int(time.monotonic() - started)
                    rows = _gpu_rows()
                    print(
                        f"[v2-train] role={role} seed={seed} "
                        f"elapsed={elapsed}s; "
                        + ", ".join(
                            f"gpu{index}={util}%/{memory}MiB"
                            for index, util, memory in rows
                        ),
                        flush=True,
                    )
            if process.returncode:
                raise RuntimeError(
                    f"Training failed role={role} seed={seed}\n"
                    + _tail(log_path)
                )
            payload = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            checks = {
                "report_passed": payload.get("passed") is True,
                "model_id_exact": payload.get("model_id")
                == MODEL_ID[role],
                "training_complete": payload.get("training", {}).get(
                    "training_complete"
                )
                is True,
                "global_step_exact": int(
                    payload.get("training", {}).get("global_step", -1)
                )
                == 1024,
                "checkpoint_exists": checkpoint_path.is_file(),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"Training report audit failed: {checks}"
                )
            print(
                f"[v2-train] completed role={role} seed={seed} "
                f"elapsed={int(time.monotonic() - started)}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
