#!/usr/bin/env python3
"""Evaluate the six paired History-7 Action Delay checkpoints.

Each checkpoint is scored on both the three-delay training-domain diagnostic
and the frozen 300-query, 11-delay Validation.  One checkpoint is assigned to
one GPU so the six model evaluations can run independently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (3072, 4096, 5120)
FAMILIES = ("pldm", "lewm")
DOMAIN_SCORER = ROOT / "scripts/diagnose_tworoom_action_delay_h7_checkpoint.py"
VALIDATION_SCORER = (
    ROOT / "scripts/diagnose_tworoom_action_delay_h7_validation_checkpoint.py"
)


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parents[1] / "data/world_model/context_world").resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _model_paths(
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


def _run_command(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-100:])
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}\n{tail}"
        )


def _audit_output(
    *,
    output: Path,
    label: str,
    family: str,
    validation: bool,
    expected_track_scope: str | None = None,
) -> None:
    payload = _load_json(output)
    checks = {
        "status": payload.get("status") == "completed_post_hoc_diagnostic",
        "label": payload.get("label") == label,
        "model_family": payload.get("model_family") == family,
        "state_unchanged": payload.get("model_state_sha256_before")
        == payload.get("model_state_sha256_after"),
    }
    if validation:
        checks.update(
            {
                "training_receipt": payload.get(
                    "training_receipt", {}
                ).get("passed")
                is True,
                "offline_only": payload.get("score_audit", {}).get(
                    "online_environment_calls"
                )
                == 0,
                "queries": payload.get("score_audit", {}).get("queries")
                == 300,
            }
        )
    else:
        checks["source_h1_units"] = payload.get(
            "aggregate_source_h1", {}
        ).get("source_h1_units") == 900
        checks["track_scope"] = payload.get(
            "claim_boundary", {}
        ).get("track_scope") == expected_track_scope
    _require(
        all(checks.values()),
        f"Evaluation output failed audit: {output}; checks={checks}",
    )


def _evaluate_one(
    *,
    artifact_root: Path,
    stablewm_repo: Path,
    family: str,
    seed: int,
    gpu: int,
    output_root: Path,
    log_root: Path,
    batch_size: int,
) -> tuple[str, Path, Path, Path]:
    slug, checkpoint, report = _model_paths(
        artifact_root,
        family,
        seed,
    )
    _require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    _require(report.is_file(), f"Missing training report: {report}")
    domain_output = output_root / f"{slug}_training_domain.json"
    heldout_domain_output = (
        output_root / f"{slug}_heldout_same_distribution.json"
    )
    validation_output = output_root / f"{slug}_validation.json"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    _run_command(
        [
            sys.executable,
            str(DOMAIN_SCORER),
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            family,
            "--label",
            slug,
            "--output",
            str(domain_output),
            "--stablewm-repo",
            str(stablewm_repo),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
        ],
        environment=environment,
        log_path=log_root / f"{slug}_training_domain.log",
    )
    _audit_output(
        output=domain_output,
        label=slug,
        family=family,
        validation=False,
        expected_track_scope="training_replay",
    )

    _run_command(
        [
            sys.executable,
            str(DOMAIN_SCORER),
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            family,
            "--label",
            slug,
            "--output",
            str(heldout_domain_output),
            "--stablewm-repo",
            str(stablewm_repo),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
            "--track-scope",
            "loader_validation",
        ],
        environment=environment,
        log_path=log_root / f"{slug}_heldout_same_distribution.log",
    )
    _audit_output(
        output=heldout_domain_output,
        label=slug,
        family=family,
        validation=False,
        expected_track_scope="loader_validation",
    )

    _run_command(
        [
            sys.executable,
            str(VALIDATION_SCORER),
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            family,
            "--label",
            slug,
            "--output",
            str(validation_output),
            "--training-report",
            str(report),
            "--stablewm-repo",
            str(stablewm_repo),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
        ],
        environment=environment,
        log_path=log_root / f"{slug}_validation.log",
    )
    _audit_output(
        output=validation_output,
        label=slug,
        family=family,
        validation=True,
    )
    return slug, domain_output, heldout_domain_output, validation_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stablewm-repo",
        type=Path,
        default=Path(
            os.environ.get("STABLEWM_REPO", "../stable-worldmodel")
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = _artifact_root()
    stablewm_repo = args.stablewm_repo.expanduser().resolve()
    _require(stablewm_repo.is_dir(), f"Missing StableWM repo: {stablewm_repo}")
    output_root = (
        artifact_root
        / "evaluation/history7/action_delay_paired_repair_v1/model_results"
    )
    log_root = (
        artifact_root
        / "evaluation/history7/action_delay_paired_repair_v1/logs"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        {
            "family": family,
            "seed": seed,
            "gpu": index,
        }
        for index, (seed, family) in enumerate(
            (seed, family)
            for seed in SEEDS
            for family in FAMILIES
        )
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                _evaluate_one,
                artifact_root=artifact_root,
                stablewm_repo=stablewm_repo,
                family=str(job["family"]),
                seed=int(job["seed"]),
                gpu=int(job["gpu"]),
                output_root=output_root,
                log_root=log_root,
                batch_size=int(args.batch_size),
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            (
                slug,
                domain_output,
                heldout_domain_output,
                validation_output,
            ) = future.result()
            print(
                f"[action-delay-h7-paired-eval] completed {slug} "
                f"gpu={job['gpu']}\n"
                f"  training_domain={domain_output}\n"
                f"  heldout_same_distribution={heldout_domain_output}\n"
                f"  validation={validation_output}",
                flush=True,
            )
    print(
        "[action-delay-h7-paired-eval] all six checkpoints passed audits",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
