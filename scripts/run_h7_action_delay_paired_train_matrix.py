#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_h7_action_delay_paired_train.sh"
SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
MODEL_IDS = {
    "pldm": "H7_ActionDelay_Paired_PLDM",
    "lewm": "H7_ActionDelay_Paired_LeWM",
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
    return [
        tuple(int(value.strip()) for value in line.split(","))
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _wait_for_eight_idle_gpus() -> None:
    while True:
        rows = _gpu_rows()
        if len(rows) != 8:
            raise RuntimeError(f"Expected eight GPUs, observed {rows}")
        if all(
            utilization <= 5 and memory_mib <= 2048
            for _, utilization, memory_mib in rows
        ):
            return
        print(
            "[action-delay-h7-paired-matrix] waiting for GPUs: "
            + ", ".join(
                f"{index}={utilization}%/{memory_mib}MiB"
                for index, utilization, memory_mib in rows
            ),
            flush=True,
        )
        time.sleep(30)


def _paths(
    artifact_root: Path,
    family: str,
    seed: int,
) -> tuple[str, Path, Path]:
    slug = f"h7_action_delay_paired_{family}_formal_s{seed}"
    checkpoint = (
        artifact_root
        / "training/runs/checkpoints"
        / slug
        / "weights_final_step_1024.pt"
    )
    report = artifact_root / "training/reports" / f"{slug}.json"
    return slug, checkpoint, report


def _audit_completed(
    *,
    family: str,
    checkpoint: Path,
    report: Path,
) -> None:
    if not checkpoint.is_file() or not report.is_file():
        raise FileNotFoundError(
            f"Incomplete matrix member: {checkpoint}, {report}"
        )
    payload = json.loads(report.read_text(encoding="utf-8"))
    objective = payload.get("model", {}).get("training_objective", {})
    checks = {
        "report_passed": payload.get("passed") is True,
        "model_id": payload.get("model_id") == MODEL_IDS[family],
        "training_method": payload.get("model", {}).get(
            "training_method"
        )
        == family,
        "weighted_objective": objective.get("name")
        == f"native_{family}_temporal_weighted",
        "last_weight": objective.get(
            "temporal_prediction_loss", {}
        ).get("transition_weights")
        == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0],
        "global_step": int(
            payload.get("training", {}).get("global_step", -1)
        )
        == 1024,
        "training_complete": payload.get("training", {}).get(
            "training_complete"
        )
        is True,
        "save_load_exact": payload.get("save_load_exact") is True,
        "checkpoint_path": Path(
            payload.get("artifacts", {}).get("pretrained", "")
        ).resolve()
        == checkpoint,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Matrix member failed audit: {report}; checks={checks}"
        )


def main() -> int:
    artifact_root = _artifact_root()
    log_root = (
        artifact_root / "training/logs/action_delay_h7_paired_v1"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    stablewm_repo = os.environ.get(
        "STABLEWM_REPO",
        "../stable-worldmodel",
    )

    for seed in SEEDS:
        for family in FAMILIES:
            slug, checkpoint, report = _paths(
                artifact_root,
                family,
                seed,
            )
            if checkpoint.exists() or report.exists():
                _audit_completed(
                    family=family,
                    checkpoint=checkpoint,
                    report=report,
                )
                print(
                    f"[action-delay-h7-paired-matrix] audited existing {slug}",
                    flush=True,
                )
                continue

            _wait_for_eight_idle_gpus()
            log_path = log_root / f"{slug}.log"
            if log_path.exists():
                raise FileExistsError(log_path)
            started = time.monotonic()
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        "bash",
                        str(LAUNCHER),
                        family,
                        str(seed),
                    ],
                    cwd=ROOT,
                    env={
                        **os.environ,
                        "STABLEWM_REPO": stablewm_repo,
                    },
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                while process.poll() is None:
                    time.sleep(30)
                    print(
                        f"[action-delay-h7-paired-matrix] running {slug} "
                        f"elapsed={int(time.monotonic() - started)}s",
                        flush=True,
                    )
            if process.returncode:
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()[-100:]
                )
                raise RuntimeError(f"{slug} failed\n{tail}")
            _audit_completed(
                family=family,
                checkpoint=checkpoint,
                report=report,
            )
            print(
                f"[action-delay-h7-paired-matrix] completed {slug} "
                f"elapsed={int(time.monotonic() - started)}s",
                flush=True,
            )
    print(
        "[action-delay-h7-paired-matrix] all six members passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
