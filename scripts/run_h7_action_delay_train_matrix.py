#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_h7_action_delay_train.sh"
SEEDS = (3072, 4096, 5120)
VARIANTS = ("original", "single", "multi")
MODEL_ID = {
    "original": "H7_OriginalOnly",
    "single": "H7_ActionDelay_SingleControl",
    "multi": "H7_ActionDelay_Multi",
}
RUN_PREFIX = {
    "original": "h7_action_delay_original_only",
    "single": "h7_action_delay_single_control",
    "multi": "h7_action_delay_multi",
}


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parents[1] / "data/world_model/context_world").resolve()


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
    rows = [
        tuple(int(value.strip()) for value in line.split(","))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    if len(rows) != 8:
        raise RuntimeError(f"History=7 formal training requires 8 GPUs: {rows}")
    return rows


def _wait_for_exclusive_gpu_window() -> None:
    while True:
        rows = _gpu_rows()
        if all(
            utilization <= 5 and memory_mib <= 2048
            for _, utilization, memory_mib in rows
        ):
            return
        print(
            "[action-delay-h7-matrix] waiting: "
            + ", ".join(
                f"gpu{index}={utilization}%/{memory_mib}MiB"
                for index, utilization, memory_mib in rows
            ),
            flush=True,
        )
        time.sleep(30)


def _tail(path: Path, lines: int = 80) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[
            -lines:
        ]
    )


def main() -> int:
    artifact_root = _artifact_root()
    report_root = artifact_root / "training/reports"
    checkpoint_root = artifact_root / "training/runs/checkpoints"
    log_root = artifact_root / "training/logs"
    log_root.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        for variant in VARIANTS:
            run_name = f"{RUN_PREFIX[variant]}_formal_s{seed}"
            report_path = report_root / f"{run_name}.json"
            checkpoint_path = (
                checkpoint_root
                / run_name
                / "weights_final_step_1024.pt"
            )
            log_path = log_root / f"{run_name}_matrix.log"
            existing = [
                path
                for path in (report_path, checkpoint_path, log_path)
                if path.exists()
            ]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite a formal matrix member: "
                    + ", ".join(map(str, existing))
                )

            _wait_for_exclusive_gpu_window()
            started = time.monotonic()
            print(
                f"[action-delay-h7-matrix] start {variant} seed={seed}",
                flush=True,
            )
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        "bash",
                        str(LAUNCHER),
                        variant,
                        "formal",
                        str(seed),
                    ],
                    cwd=ROOT,
                    env={**os.environ, "RUN_NAME": run_name},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    time.sleep(30)
                    print(
                        f"[action-delay-h7-matrix] running {variant} "
                        f"seed={seed} elapsed="
                        f"{int(time.monotonic() - started)}s",
                        flush=True,
                    )
            if process.returncode:
                raise RuntimeError(
                    f"Training failed for {variant} seed={seed}\n"
                    + _tail(log_path)
                )

            payload = json.loads(report_path.read_text(encoding="utf-8"))
            initialization = payload.get("initialization_checkpoint", {})
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
                "temporal_expansion_passed": initialization.get(
                    "temporal_adaptation_audit", {}
                ).get("passed")
                is True,
                "checkpoint_exists": checkpoint_path.is_file(),
            }
            if not all(checks.values()):
                raise RuntimeError(
                    f"Completed training failed audit: {checks}"
                )
            print(
                f"[action-delay-h7-matrix] completed {variant} seed={seed} "
                f"elapsed={int(time.monotonic() - started)}s",
                flush=True,
            )

    print(
        "[action-delay-h7-matrix] all nine formal runs completed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
