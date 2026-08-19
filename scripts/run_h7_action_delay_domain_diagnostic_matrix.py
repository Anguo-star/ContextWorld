#!/usr/bin/env python3
"""Run the nine History-7 domain diagnostics across eight GPUs."""

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

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_scoring_v1.yaml"
)
EVALUATOR = (
    ROOT / "scripts/eval_tworoom_action_delay_h7_domain_diagnostic.py"
)


def _models(config: dict[str, Any]) -> list[str]:
    return [
        str(row["slug"])
        for rows in config["models"].values()
        for row in rows
    ]


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


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = _models(config)
    gpu_rows = _gpu_rows()
    if len(gpu_rows) != 8:
        raise RuntimeError(f"Expected eight GPUs, found {gpu_rows}")
    if not all(util <= 5 and memory <= 2048 for _, util, memory in gpu_rows):
        raise RuntimeError(f"Eight idle GPUs required: {gpu_rows}")
    result_root = resolve_contextworld_path(
        config["artifacts"]["results_root"],
        repo_root=ROOT,
    )
    log_root = result_root / "logs"
    matrix_report = result_root / "matrix_report.json"
    log_root.mkdir(parents=True, exist_ok=True)
    if matrix_report.exists():
        raise FileExistsError(matrix_report)
    for slug in models:
        for path in (
            result_root / f"{slug}.json",
            log_root / f"{slug}.log",
        ):
            if path.exists():
                raise FileExistsError(path)

    stablewm_repo = os.environ.get(
        "STABLEWM_REPO",
        str(config["stable_worldmodel"]["repo"]),
    )
    pending = list(models)
    running: dict[int, dict[str, Any]] = {}
    completed_rows = []
    started = time.monotonic()
    while pending or running:
        for gpu in [value for value in range(8) if value not in running]:
            if not pending:
                break
            slug = pending.pop(0)
            output = result_root / f"{slug}.json"
            log_path = log_root / f"{slug}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
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
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[gpu] = {
                "slug": slug,
                "process": process,
                "handle": handle,
                "log": log_path,
                "output": output,
                "started": time.monotonic(),
            }
            print(
                f"[h7-domain-matrix] start {slug} gpu={gpu}",
                flush=True,
            )
        time.sleep(10)
        for gpu, row in list(running.items()):
            returncode = row["process"].poll()
            if returncode is None:
                continue
            row["handle"].close()
            if returncode:
                tail = "\n".join(
                    row["log"]
                    .read_text(encoding="utf-8", errors="replace")
                    .splitlines()[-100:]
                )
                raise RuntimeError(
                    f"Domain diagnostic failed: {row['slug']}\n{tail}"
                )
            payload = json.loads(
                row["output"].read_text(encoding="utf-8")
            )
            if (
                payload.get("status") != "completed"
                or payload.get("score_audit", {}).get("passed") is not True
            ):
                raise RuntimeError(
                    f"Invalid domain result: {row['slug']}"
                )
            completed_rows.append(
                {
                    "slug": row["slug"],
                    "gpu": gpu,
                    "elapsed_seconds": (
                        time.monotonic() - row["started"]
                    ),
                    "output": str(row["output"]),
                    "log": str(row["log"]),
                    "status": "passed",
                }
            )
            del running[gpu]
            print(
                f"[h7-domain-matrix] completed "
                f"{len(completed_rows)}/9",
                flush=True,
            )
    write_json(
        matrix_report,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "passed",
            "models": 9,
            "gpus": 8,
            "elapsed_seconds": time.monotonic() - started,
            "results": completed_rows,
        },
    )
    print(
        f"[h7-domain-matrix] all completed: {matrix_report}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
