#!/usr/bin/env python3
"""Train a six-checkpoint History-7 Action Delay reference matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
RECIPES = {
    "full_range_v2": {
        "launcher": ROOT
        / "scripts/run_h7_action_delay_full_range_train.sh",
        "run_stem": "h7_action_delay_full_range",
        "log_directory": "action_delay_h7_full_range_v2",
        "model_ids": {
            "pldm": "H7_ActionDelay_FullRange_PLDM",
            "lewm": "H7_ActionDelay_FullRange_LeWM",
        },
    },
    "core_v3": {
        "launcher": ROOT / "scripts/run_h7_action_delay_core_train.sh",
        "run_stem": "h7_action_delay_core_v3",
        "log_directory": "action_delay_h7_core_v3",
        "model_ids": {
            "pldm": "H7_ActionDelay_Core_v3_PLDM",
            "lewm": "H7_ActionDelay_Core_v3_LeWM",
        },
    },
    "curriculum_v4": {
        "launcher": (
            ROOT / "scripts/run_h7_action_delay_curriculum_train.sh"
        ),
        "run_stem": "h7_action_delay_curriculum_v4",
        "log_directory": "action_delay_h7_curriculum_v4",
        "model_ids": {
            "pldm": "H7_ActionDelay_Curriculum_v4_PLDM",
            "lewm": "H7_ActionDelay_Curriculum_v4_LeWM",
        },
    },
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
            "[action-delay-h7-train-matrix] waiting for GPUs: "
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
    *,
    run_stem: str,
) -> tuple[str, Path, Path]:
    slug = f"{run_stem}_{family}_formal_s{seed}"
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
    seed: int,
    checkpoint: Path,
    report: Path,
    model_ids: dict[str, str],
    recipe_name: str,
    artifact_root: Path,
) -> None:
    if not checkpoint.is_file() or not report.is_file():
        raise FileNotFoundError(
            f"Incomplete matrix member: {checkpoint}, {report}"
        )
    payload = json.loads(report.read_text(encoding="utf-8"))
    objective = payload.get("model", {}).get("training_objective", {})
    checks = {
        "report_passed": payload.get("passed") is True,
        "model_id": payload.get("model_id") == model_ids[family],
        "training_method": payload.get("model", {}).get(
            "training_method"
        )
        == family,
        "one_step_model": payload.get("model", {}).get(
            "num_preds"
        )
        == 1,
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
    if recipe_name in {"core_v3", "curriculum_v4"}:
        plan = payload.get("training", {}).get("plan", {})
        synthetic = plan.get("group_exposure", {}).get(
            "action_delay_paired",
            {},
        )
        balancing = (
            payload.get("data", {})
            .get("groups", {})
            .get("action_delay_paired", {})
            .get("train_balancing", {})
        )
        checks.update(
            {
                "core_profile": plan.get("profile") == "icl_core_v3",
                "six_balance_groups": (
                    balancing.get("balance_groups") == 6
                    and balancing.get("factor_values") == 11
                ),
                "all_synthetic_raw_clips_seen": (
                    synthetic.get("run_unique_raw_fraction") == 1.0
                    and synthetic.get("raw_clips_never_drawn") == 0
                ),
            }
        )
    if recipe_name == "curriculum_v4":
        initialization = payload.get("initialization_checkpoint", {})
        expected_initial = (
            artifact_root
            / "training/runs/checkpoints"
            / f"h7_action_delay_paired_{family}_formal_s{seed}"
            / "weights_final_step_1024.pt"
        )
        checks.update(
            {
                "curriculum_initialization_applied": (
                    initialization.get("applied") is True
                    and initialization.get("hash_audit_passed") is True
                ),
                "curriculum_initialization_path": Path(
                    initialization.get("path", "")
                ).resolve()
                == expected_initial,
                "curriculum_initialization_is_weights_only": (
                    initialization.get("resume_state_loaded") is False
                    and initialization.get("optimizer_state_loaded") is False
                    and initialization.get("scheduler_state_loaded") is False
                    and initialization.get("state_exact") is True
                    and initialization.get("temporal_adaptation") is None
                ),
            }
        )
    if not all(checks.values()):
        raise RuntimeError(
            f"Matrix member failed audit: {report}; checks={checks}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        choices=tuple(RECIPES),
        default="full_range_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recipe = RECIPES[str(args.recipe)]
    artifact_root = _artifact_root()
    log_root = (
        artifact_root / "training/logs" / recipe["log_directory"]
    )
    log_root.mkdir(parents=True, exist_ok=True)
    stablewm_repo = os.environ.get(
        "STABLEWM_REPO",
        "../stable-worldmodel",
    )

    matrix = (
        [(family, seed) for family in FAMILIES for seed in SEEDS]
        if args.recipe in {"core_v3", "curriculum_v4"}
        else [(family, seed) for seed in SEEDS for family in FAMILIES]
    )
    for family, seed in matrix:
        slug, checkpoint, report = _paths(
            artifact_root,
            family,
            seed,
            run_stem=str(recipe["run_stem"]),
        )
        if checkpoint.exists() or report.exists():
            _audit_completed(
                family=family,
                seed=seed,
                checkpoint=checkpoint,
                report=report,
                model_ids=recipe["model_ids"],
                recipe_name=str(args.recipe),
                artifact_root=artifact_root,
            )
            print(
                "[action-delay-h7-train-matrix] "
                f"audited existing {slug}",
                flush=True,
            )
            continue

        _wait_for_eight_idle_gpus()
        log_path = log_root / f"{slug}.log"
        attempt = 1
        while log_path.exists():
            attempt += 1
            log_path = log_root / f"{slug}.attempt{attempt}.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    "bash",
                    str(recipe["launcher"]),
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
                    "[action-delay-h7-train-matrix] "
                    f"running {slug} "
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
            seed=seed,
            checkpoint=checkpoint,
            report=report,
            model_ids=recipe["model_ids"],
            recipe_name=str(args.recipe),
            artifact_root=artifact_root,
        )
        print(
            "[action-delay-h7-train-matrix] "
            f"completed {slug} "
            f"elapsed={int(time.monotonic() - started)}s",
            flush=True,
        )
    print(
        "[action-delay-h7-train-matrix] all six members passed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
