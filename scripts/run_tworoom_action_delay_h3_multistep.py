#!/usr/bin/env python3
"""在多张 GPU 上运行七个动作延迟多步 Eval。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_multistep import (
    LOSS_RECORDS_PER_CHECKPOINT,
    PREDICTIONS_PER_CHECKPOINT,
    QUERY_COUNT,
    TARGET_ENCODINGS_PER_CHECKPOINT,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_multistep_extrap_v1.yaml"
)


def _models(
    config: dict[str, Any],
    selected: set[str] | None,
) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for group in config["models"].values()
        for row in group
        if selected is None or str(row["slug"]) in selected
    ]
    if not rows:
        raise ValueError("没有选中模型")
    return rows


def _valid(path: Path, *, slug: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    audit = payload.get("score_audit", {})
    return (
        payload.get("status") == "completed"
        and payload.get("model_slug") == slug
        and audit.get("queries") == QUERY_COUNT
        and audit.get("model_rollouts") == PREDICTIONS_PER_CHECKPOINT
        and audit.get("target_encodings")
        == TARGET_ENCODINGS_PER_CHECKPOINT
        and audit.get("horizon_loss_records")
        == LOSS_RECORDS_PER_CHECKPOINT
        and audit.get("online_environment_calls") == 0
    )


def _run(
    *,
    slug: str,
    gpu: str,
    config_path: Path,
    output: Path,
    batch_size: int,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_tworoom_action_delay_h3_multistep.py"),
        "--config",
        str(config_path),
        "--model",
        slug,
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--batch-size",
        str(batch_size),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("PYTHONUNBUFFERED", "1")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[variable] = "1"
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[start] gpu={gpu} command={json.dumps(command)}\n"
        )
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    passed = completed.returncode == 0 and _valid(output, slug=slug)
    return {
        "slug": slug,
        "gpu": gpu,
        "status": "passed" if passed else "failed",
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "output": str(output),
        "log": str(log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_root = resolve_contextworld_path(
        config["artifacts"]["results"], repo_root=ROOT
    )
    selected = set(args.models) if args.models else None
    models = _models(config, selected)
    jobs = []
    for index, model in enumerate(models):
        slug = str(model["slug"])
        output = results_root / f"{slug}.json"
        if not args.force and _valid(output, slug=slug):
            continue
        jobs.append(
            {
                "slug": slug,
                "gpu": str(args.gpus[index % len(args.gpus)]),
                "output": output,
            }
        )
    print(
        json.dumps(
            {
                "models": len(models),
                "pending": len(jobs),
                "skipped_valid": len(models) - len(jobs),
                "jobs": [
                    {
                        "slug": job["slug"],
                        "gpu": job["gpu"],
                        "output": str(job["output"]),
                    }
                    for job in jobs
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    results = []
    with ThreadPoolExecutor(
        max_workers=min(len(args.gpus), len(jobs)) or 1
    ) as executor:
        futures = [
            executor.submit(
                _run,
                slug=job["slug"],
                gpu=job["gpu"],
                config_path=config_path,
                output=job["output"],
                batch_size=int(args.batch_size),
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(
                f"[{row['status']}] gpu={row['gpu']} {row['slug']} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "models": len(models),
        "executed": len(results),
        "skipped_valid": len(models) - len(jobs),
        "failed": len(failures),
        "results": sorted(results, key=lambda row: row["slug"]),
    }
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "runner_report.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} 个多步 Eval 失败")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
