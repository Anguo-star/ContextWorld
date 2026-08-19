#!/usr/bin/env python3
"""Run the nine formal History-7 Action Delay scorers across eight GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.synthesis.manifest import write_json


CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
)
EVALUATOR = ROOT / "scripts/eval_tworoom_action_delay_h7.py"


def _artifact_root() -> Path:
    configured = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parents[1] / "data/world_model/context_world").resolve()


def _models(config: dict[str, Any]) -> list[str]:
    return [
        str(row["slug"])
        for rows in config["models"].values()
        for row in rows
    ]


def _gpu_count() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return len(
        [line for line in completed.stdout.splitlines() if line.strip()]
    )


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = _models(config)
    gpu_count = _gpu_count()
    if gpu_count != 8:
        raise RuntimeError(f"Expected eight GPUs, found {gpu_count}")
    artifact_root = _artifact_root()
    result_root = (
        artifact_root
        / "evaluation/history7/action_delay_validation_v1/model_results"
    )
    log_root = result_root / "logs"
    matrix_report = result_root / "scoring_matrix_report.json"
    log_root.mkdir(parents=True, exist_ok=True)
    if matrix_report.exists():
        raise FileExistsError(matrix_report)
    for slug in models:
        existing = [
            path
            for path in (
                result_root / f"{slug}.json",
                log_root / f"{slug}.log",
            )
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite formal scoring output: "
                + ", ".join(map(str, existing))
            )

    stablewm_repo = os.environ.get(
        "STABLEWM_REPO", str(config["stable_worldmodel"]["repo"])
    )
    pending = list(models)
    running: dict[int, dict[str, Any]] = {}
    completed_rows = []
    started_matrix = time.monotonic()
    while pending or running:
        free_gpus = [
            gpu for gpu in range(gpu_count) if gpu not in running
        ]
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            slug = pending.pop(0)
            output = result_root / f"{slug}.json"
            log_path = log_root / f"{slug}.log"
            log = log_path.open("w", encoding="utf-8")
            command = [
                "python",
                str(EVALUATOR),
                "--model",
                slug,
                "--config",
                str(CONFIG),
                "--output",
                str(output),
                "--stablewm-repo",
                stablewm_repo,
                "--device",
                "cuda:0",
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = {
                "slug": slug,
                "process": process,
                "log_handle": log,
                "log": log_path,
                "output": output,
                "started": time.monotonic(),
            }
            print(
                f"[action-delay-h7-eval] start {slug} gpu={gpu}",
                flush=True,
            )

        time.sleep(10)
        for gpu, row in list(running.items()):
            process = row["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            row["log_handle"].close()
            elapsed = time.monotonic() - row["started"]
            if returncode:
                tail = "\n".join(
                    row["log"]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-100:]
                )
                raise RuntimeError(
                    f"Scoring failed: {row['slug']}\n{tail}"
                )
            payload = json.loads(
                row["output"].read_text(encoding="utf-8")
            )
            if (
                payload.get("status") != "completed"
                or payload.get("score_audit", {}).get("passed") is not True
            ):
                raise RuntimeError(
                    f"Scoring output failed audit: {row['slug']}"
                )
            completed_rows.append(
                {
                    "slug": row["slug"],
                    "gpu": gpu,
                    "elapsed_seconds": elapsed,
                    "output": str(row["output"]),
                    "log": str(row["log"]),
                    "status": "passed",
                }
            )
            del running[gpu]
            print(
                f"[action-delay-h7-eval] completed {row['slug']} "
                f"gpu={gpu} elapsed={int(elapsed)}s "
                f"total={len(completed_rows)}/{len(models)}",
                flush=True,
            )
        if running:
            print(
                "[action-delay-h7-eval] running "
                + ", ".join(
                    f"{row['slug']}@gpu{gpu}"
                    for gpu, row in sorted(running.items())
                ),
                flush=True,
            )

    matrix_payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "models": len(models),
        "gpus": gpu_count,
        "elapsed_seconds": time.monotonic() - started_matrix,
        "results": sorted(
            completed_rows, key=lambda row: models.index(row["slug"])
        ),
    }
    write_json(matrix_report, matrix_payload)
    print(
        f"[action-delay-h7-eval] all nine completed: {matrix_report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
