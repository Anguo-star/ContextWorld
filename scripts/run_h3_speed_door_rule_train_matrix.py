#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_h3_speed_door_rule_train.sh"
JOBS = tuple(
    (variant, seed)
    for seed in (3072, 4096, 5120)
    for variant in ("speed", "joint")
)
RUN_PREFIX = {
    "speed": "h3_speed_only_pldm",
    "joint": "h3_speed_door_joint_pldm",
}
MODEL_ID = {
    "speed": "H3_SpeedOnly_PLDM",
    "joint": "H3_SpeedDoorJoint_PLDM",
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
        if not line.strip():
            continue
        index, utilization, memory = (
            int(value.strip()) for value in line.split(",")
        )
        rows.append((index, utilization, memory))
    if len(rows) != 8:
        raise RuntimeError(f"Expected eight GPUs, got {rows}")
    return rows


def _all_gpus_free() -> bool:
    return all(memory <= 16 for _, _, memory in _gpu_rows())


def _wait_for_exclusive_gpu_window() -> None:
    while not _all_gpus_free():
        rows = _gpu_rows()
        print(
            "[composition-train] waiting for existing GPU task: "
            + ", ".join(
                f"gpu{index}={util}%/{memory}MiB"
                for index, util, memory in rows
            ),
            flush=True,
        )
        time.sleep(30)


def _tail(path: Path, lines: int = 80) -> str:
    values = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    return "\n".join(values[-lines:])


def main() -> int:
    artifact_root = _artifact_root()
    report_root = artifact_root / "training/reports"
    checkpoint_root = artifact_root / "training/runs/checkpoints"
    log_root = artifact_root / "training/logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for variant, seed in JOBS:
        run = f"{RUN_PREFIX[variant]}_passage_formal_s{seed}"
        report_path = report_root / f"{run}.json"
        checkpoint_path = (
            checkpoint_root / run / "weights_final_step_1024.pt"
        )
        if report_path.exists() or checkpoint_path.exists():
            raise FileExistsError(
                "Refusing to overwrite a formal matrix member: "
                f"{report_path} / {checkpoint_path}"
            )
        _wait_for_exclusive_gpu_window()
        log_path = log_root / f"{run}_matrix.log"
        if log_path.exists():
            raise FileExistsError(log_path)
        environment = dict(os.environ)
        environment["TRAINING_SEED"] = str(seed)
        environment["logger_backend"] = "none"
        started = time.monotonic()
        print(
            f"[composition-train] start {variant} seed={seed}; "
            f"log={log_path}",
            flush=True,
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                ["bash", str(LAUNCHER), variant, "formal"],
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
                    f"[composition-train] running {variant} seed={seed} "
                    f"elapsed={elapsed}s; "
                    + ", ".join(
                        f"gpu{index}={util}%/{memory}MiB"
                        for index, util, memory in rows
                    ),
                    flush=True,
                )
        if process.returncode:
            raise RuntimeError(
                f"Training failed for {variant} seed={seed}\n"
                + _tail(log_path)
            )
        payload = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        checks = {
            "report_passed": payload.get("passed") is True,
            "model_id_exact": payload.get("model_id")
            == MODEL_ID[variant],
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
                f"Completed training failed report audit: {checks}"
            )
        print(
            f"[composition-train] completed {variant} seed={seed} "
            f"elapsed={int(time.monotonic() - started)}s",
            flush=True,
        )
    print("[composition-train] all six formal runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
