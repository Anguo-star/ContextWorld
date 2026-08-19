#!/usr/bin/env python3
"""Run post-gate History-7 latent alignment diagnostics on eight GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
)
ENTRYPOINT = (
    ROOT / "scripts/diagnose_tworoom_action_delay_h7_latents.py"
)


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
    rows = _gpu_rows()
    if (
        len(rows) != 8
        or not all(util <= 5 and memory <= 2048 for _, util, memory in rows)
    ):
        raise RuntimeError(f"Eight idle GPUs required: {rows}")
    root = (
        _artifact_root()
        / "evaluation/history7/action_delay_validation_v1/"
        "model_results/latent_diagnostics"
    )
    log_root = root / "logs"
    report_path = root / "runner_report.json"
    log_root.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(report_path)
    for slug in models:
        for path in (root / f"{slug}.json", log_root / f"{slug}.log"):
            if path.exists():
                raise FileExistsError(path)

    stablewm_repo = os.environ.get(
        "STABLEWM_REPO", str(config["stable_worldmodel"]["repo"])
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
            output = root / f"{slug}.json"
            log_path = log_root / f"{slug}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                [
                    "python",
                    str(ENTRYPOINT),
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
                "output": output,
                "log": log_path,
                "handle": handle,
                "process": process,
                "started": time.monotonic(),
            }
            print(f"[h7-latent] start {slug} gpu={gpu}", flush=True)

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
                    f"Latent diagnostic failed: {row['slug']}\n{tail}"
                )
            payload = json.loads(
                row["output"].read_text(encoding="utf-8")
            )
            if payload.get("status") != (
                "diagnostic_completed_not_part_of_primary_gate"
            ):
                raise RuntimeError(
                    f"Invalid latent diagnostic: {row['slug']}"
                )
            completed_rows.append(
                {
                    "slug": row["slug"],
                    "gpu": gpu,
                    "elapsed_seconds": (
                        time.monotonic() - row["started"]
                    ),
                    "output": str(row["output"]),
                }
            )
            del running[gpu]
            print(
                f"[h7-latent] completed {len(completed_rows)}/9",
                flush=True,
            )

    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "models": 9,
                "elapsed_seconds": time.monotonic() - started,
                "results": completed_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[h7-latent] all completed: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
