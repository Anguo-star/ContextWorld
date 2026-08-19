#!/usr/bin/env python3
"""Evaluate six full-range checkpoints on frozen Action Delay Validation."""

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
SCORER = (
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
    _require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


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


def _audit_eval_output(
    payload: dict[str, Any],
    *,
    slug: str,
    family: str,
) -> None:
    audit = payload.get("score_audit", {})
    checks = {
        "status": payload.get("status")
        == "completed_post_hoc_diagnostic",
        "label": payload.get("label") == slug,
        "family": payload.get("model_family") == family,
        "receipt": payload.get("training_receipt", {}).get("passed")
        is True,
        "state_unchanged": payload.get("model_state_sha256_before")
        == payload.get("model_state_sha256_after"),
        "queries": audit.get("queries") == 300,
        "predictions": audit.get("model_predictions") == 3300,
        "targets": audit.get("target_encodings") == 9900,
        "loss_records": audit.get("horizon_loss_records") == 108900,
        "offline": audit.get("online_environment_calls") == 0,
    }
    _require(
        all(checks.values()),
        f"{slug} Eval audit failed: {checks}",
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
    run_stem: str,
) -> tuple[str, Path]:
    slug, checkpoint, report = _paths(
        artifact_root,
        family,
        seed,
        run_stem=run_stem,
    )
    _require(checkpoint.is_file(), f"Missing checkpoint: {checkpoint}")
    _require(report.is_file(), f"Missing training report: {report}")
    output = output_root / f"{slug}_validation.json"
    log_path = log_root / f"{slug}_validation.log"
    if output.is_file():
        _audit_eval_output(
            _load_json(output),
            slug=slug,
            family=family,
        )
        return slug, output
    _require(not output.exists(), f"Invalid Eval output path: {output}")
    attempt = 1
    while log_path.exists():
        attempt += 1
        log_path = log_root / f"{slug}_validation.attempt{attempt}.log"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--checkpoint",
            str(checkpoint),
            "--model-family",
            family,
            "--label",
            slug,
            "--output",
            str(output),
            "--training-report",
            str(report),
            "--stablewm-repo",
            str(stablewm_repo),
            "--device",
            "cuda:0",
            "--batch-size",
            str(batch_size),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"{slug} Eval failed\n"
            + "\n".join(completed.stdout.splitlines()[-100:])
        )
    _audit_eval_output(
        _load_json(output),
        slug=slug,
        family=family,
    )
    return slug, output


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
    parser.add_argument(
        "--run-stem",
        default="h7_action_delay_full_range",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5",
        help="Comma-separated physical GPU indices",
    )
    parser.add_argument(
        "--completed-training-only",
        action="store_true",
        help=(
            "Evaluate only checkpoints whose training reports already exist; "
            "a later default run safely audits these outputs and fills gaps"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = _artifact_root()
    stablewm_repo = args.stablewm_repo.expanduser().resolve()
    _require(stablewm_repo.is_dir(), f"Missing StableWM: {stablewm_repo}")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else artifact_root
        / "evaluation/history7/action_delay_full_range_v2/model_results"
    )
    log_root = output_root.parent / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    specs = [
        (family, seed)
        for seed in SEEDS
        for family in FAMILIES
    ]
    if args.completed_training_only:
        specs = [
            (family, seed)
            for family, seed in specs
            if all(
                path.is_file()
                for path in _paths(
                    artifact_root,
                    family,
                    seed,
                    run_stem=str(args.run_stem),
                )[1:]
            )
        ]
    _require(bool(specs), "No completed checkpoints selected")
    gpus = [
        int(value)
        for value in str(args.gpus).split(",")
        if value.strip()
    ]
    _require(
        len(gpus) >= len(specs) and len(gpus) == len(set(gpus)),
        "Provide at least one distinct GPU for every selected checkpoint",
    )
    jobs = [
        (family, seed, gpus[index])
        for index, (family, seed) in enumerate(specs)
    ]
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = {
            executor.submit(
                _evaluate_one,
                artifact_root=artifact_root,
                stablewm_repo=stablewm_repo,
                family=family,
                seed=seed,
                gpu=gpu,
                output_root=output_root,
                log_root=log_root,
                batch_size=int(args.batch_size),
                run_stem=str(args.run_stem),
            ): (family, seed, gpu)
            for family, seed, gpu in jobs
        }
        for future in as_completed(futures):
            family, seed, gpu = futures[future]
            slug, output = future.result()
            print(
                "[action-delay-h7-full-range-eval] "
                f"completed {slug} gpu={gpu} output={output}",
                flush=True,
            )
    print(
        "[action-delay-h7-full-range-eval] "
        f"all {len(jobs)} selected checkpoints passed audits",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
