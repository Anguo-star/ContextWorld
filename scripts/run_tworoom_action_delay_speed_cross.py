#!/usr/bin/env python3
"""并行运行动作延迟 checkpoint 的冻结速度历史诊断。"""

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

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h3_speed_cross_diagnostic_v1.yaml"
)


def _models(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for rows in config["models"].values()
        for row in rows
    ]


def _valid(path: Path, *, benchmark: str, slug: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "passed"
        and payload.get("benchmark") == benchmark
        and payload.get("model", {}).get("slug") == slug
        and payload.get("online_environment_calls") == 0
        and len(payload.get("tracks", {})) == 2
        and all(
            track.get("summary", {})
            .get("count_audit", {})
            .get("passed")
            for track in payload.get("tracks", {}).values()
        )
    )


def _run(
    *,
    config_path: Path,
    slug: str,
    gpu: str,
    output: Path,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_tworoom_speed_next_latent.py"),
        "--config",
        str(config_path),
        "--model",
        slug,
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--encode-batch-size",
        "64",
        "--predictor-batch-size",
        "128",
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
    return {
        "slug": slug,
        "gpu": gpu,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
        "output": str(output),
        "log": str(log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    benchmark = str(config["benchmark"])
    results_root = resolve_contextworld_path(
        config["artifacts"]["results"], repo_root=ROOT
    )
    jobs = []
    models = _models(config)
    for index, model in enumerate(models):
        slug = str(model["slug"])
        output = results_root / f"{slug}.json"
        if not args.force and _valid(
            output, benchmark=benchmark, slug=slug
        ):
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
                "jobs": [
                    {
                        "slug": row["slug"],
                        "gpu": row["gpu"],
                        "output": str(row["output"]),
                    }
                    for row in jobs
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    rows = []
    with ThreadPoolExecutor(
        max_workers=min(len(args.gpus), len(jobs)) or 1
    ) as executor:
        futures = [
            executor.submit(
                _run,
                config_path=config_path,
                slug=job["slug"],
                gpu=job["gpu"],
                output=job["output"],
            )
            for job in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            row["status"] = (
                "passed"
                if row["returncode"] == 0
                and _valid(
                    Path(row["output"]),
                    benchmark=benchmark,
                    slug=row["slug"],
                )
                else "failed"
            )
            rows.append(row)
            print(
                f"[{row['status']}] gpu={row['gpu']} {row['slug']} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
    failures = [row for row in rows if row["status"] != "passed"]
    report = {
        "schema_version": 1,
        "benchmark": benchmark,
        "status": "failed" if failures else "passed",
        "models": len(models),
        "executed": len(rows),
        "skipped_valid": len(models) - len(jobs),
        "failed": len(failures),
        "results": sorted(rows, key=lambda row: row["slug"]),
    }
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / "runner_report.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} 个速度交叉诊断失败")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
