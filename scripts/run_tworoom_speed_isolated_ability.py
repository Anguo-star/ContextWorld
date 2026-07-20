#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_cube_eval_v2.yaml"
)


@dataclass(frozen=True)
class Job:
    mode: str
    slug: str
    checkpoint: Path
    domain: str
    seed: int | None
    output: Path
    log: Path

    @property
    def label(self) -> str:
        suffix = f"/s{self.seed}" if self.seed is not None else ""
        return f"{self.mode}/{self.slug}/{self.domain}{suffix}"


def _models(
    config: dict[str, Any], selected: set[str] | None
) -> list[dict[str, Any]]:
    rows = [
        dict(model)
        for models in config["models"].values()
        for model in models
        if selected is None or str(model["slug"]) in selected
    ]
    if not rows:
        raise ValueError("No models selected")
    return rows


def _valid(job: Job) -> bool:
    if not job.output.is_file():
        return False
    try:
        payload = json.loads(job.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "passed":
        return False
    if job.mode == "planning":
        return (
            int(payload.get("protocol", {}).get("evaluations", -1)) == 50
            and int(payload.get("protocol", {}).get("eval_seed", -1))
            == job.seed
        )
    return len(payload.get("raw_records", [])) == 600


def _jobs(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[Job]:
    root = resolve_contextworld_path(
        config["artifacts"]["original_ability_retention_root"],
        repo_root=ROOT,
    )
    modes = (
        ["planning", "rollout"]
        if args.mode == "all"
        else [args.mode]
    )
    domains = {
        "original_heldout": resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "original_ability_reconstruction/"
            "original_heldout_eval_catalog.json",
            repo_root=ROOT,
        ),
        "speed5_matched": resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "original_ability_reconstruction/"
            "speed5_matched_eval_catalog.json",
            repo_root=ROOT,
        ),
    }
    jobs = []
    for model in _models(
        config, set(args.models) if args.models else None
    ):
        slug = str(model["slug"])
        checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=ROOT
        )
        for mode in modes:
            if mode == "rollout":
                output = root / slug / "rollout_error.json"
                jobs.append(
                    Job(
                        mode=mode,
                        slug=slug,
                        checkpoint=checkpoint,
                        domain="both_domains",
                        seed=None,
                        output=output,
                        log=output.with_suffix(".log"),
                    )
                )
                continue
            for domain in args.domains:
                if domain not in domains:
                    raise ValueError(f"Unknown domain: {domain}")
                for seed in args.eval_seeds:
                    output = (
                        root
                        / slug
                        / domain
                        / f"s{int(seed)}.json"
                    )
                    jobs.append(
                        Job(
                            mode=mode,
                            slug=slug,
                            checkpoint=checkpoint,
                            domain=domain,
                            seed=int(seed),
                            output=output,
                            log=output.with_suffix(".log"),
                        )
                    )
    return jobs


def _command(
    job: Job, args: argparse.Namespace, config: dict[str, Any]
) -> list[str]:
    normalizer = resolve_contextworld_path(
        "artifacts/splits/"
        "tworoom_original_train_s3072_normalizer.json",
        repo_root=ROOT,
    )
    common = [
        args.python,
        "--checkpoint",
        str(job.checkpoint),
        "--normalizer",
        str(normalizer),
        "--output",
        str(job.output),
        "--stablewm-repo",
        str((ROOT / config["stable_worldmodel"]["repo"]).resolve()),
        "--stablewm-ref",
        str(config["stable_worldmodel"]["expected_ref"]),
        "--device",
        "cuda:0",
    ]
    if job.mode == "rollout":
        catalog = resolve_contextworld_path(
            "artifacts/evaluation/history3/"
            "original_ability_reconstruction/rollout_catalog.json",
            repo_root=ROOT,
        )
        return [
            common[0],
            str(ROOT / "scripts/eval_tworoom_rollout_error.py"),
            "--catalog",
            str(catalog),
            *common[1:],
            "--batch-size",
            "16",
        ]
    catalog_name = {
        "original_heldout": "original_heldout_eval_catalog.json",
        "speed5_matched": "speed5_matched_eval_catalog.json",
    }[job.domain]
    catalog = resolve_contextworld_path(
        "artifacts/evaluation/history3/"
        f"original_ability_reconstruction/{catalog_name}",
        repo_root=ROOT,
    )
    return [
        common[0],
        str(ROOT / "scripts/eval_tworoom_ability_catalog.py"),
        "--catalog",
        str(catalog),
        *common[1:],
        "--seed",
        str(job.seed),
        "--eval-budget",
        "50",
        "--horizon",
        "5",
        "--receding-horizon",
        "5",
        "--cem-samples",
        "300",
        "--cem-steps",
        "30",
        "--cem-topk",
        "30",
    ]


def _run_job(
    job: Job,
    *,
    gpu: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args, config)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment.setdefault("MUJOCO_GL", "egl")
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
    with job.log.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[start] gpu={gpu} command={json.dumps(command)}\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    passed = completed.returncode == 0 and _valid(job)
    return {
        "label": job.label,
        "status": "passed" if passed else "failed",
        "returncode": completed.returncode,
        "gpu": gpu,
        "elapsed_seconds": time.time() - started,
        "output": str(job.output),
        "log": str(job.log),
    }


def _gpu_queue(
    gpu: str,
    jobs: list[Job],
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for job in jobs:
        result = _run_job(
            job, gpu=gpu, args=args, config=config
        )
        results.append(result)
        print(
            f"[{result['status']}] gpu={gpu} {job.label} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    jobs = _jobs(args, config)
    missing = [
        str(job.checkpoint)
        for job in jobs
        if not job.checkpoint.is_file()
    ]
    if missing:
        raise FileNotFoundError("\n".join(sorted(set(missing))))
    pending = [job for job in jobs if args.force or not _valid(job)]
    skipped = len(jobs) - len(pending)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "jobs": len(jobs),
                "pending": len(pending),
                "skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "jobs": len(jobs),
            "pending": len(pending),
            "commands": [
                {"label": job.label, "command": _command(job, args, config)}
                for job in pending
            ],
        }
    queues = {str(gpu): [] for gpu in args.gpus}
    grouped: dict[tuple[str, str, str], list[Job]] = {}
    for job in pending:
        grouped.setdefault(
            (job.mode, job.slug, job.domain), []
        ).append(job)
    for index, key in enumerate(sorted(grouped)):
        gpu = str(args.gpus[index % len(args.gpus)])
        queues[gpu].extend(
            sorted(
                grouped[key],
                key=lambda job: job.seed if job.seed is not None else -1,
            )
        )
    results = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                _gpu_queue,
                gpu,
                gpu_jobs,
                args=args,
                config=config,
            )
            for gpu, gpu_jobs in queues.items()
            if gpu_jobs
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "mode": args.mode,
        "jobs": len(jobs),
        "skipped_valid": skipped,
        "executed": len(results),
        "failed": len(failures),
        "results": sorted(results, key=lambda row: row["label"]),
    }
    root = resolve_contextworld_path(
        config["artifacts"]["original_ability_retention_root"],
        repo_root=ROOT,
    )
    write_json(root / f"runner_{args.mode}.json", report)
    if failures:
        raise RuntimeError(f"{len(failures)} ability jobs failed")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("planning", "rollout", "all"),
        required=True,
    )
    parser.add_argument("--models", nargs="+")
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["original_heldout", "speed5_matched"],
    )
    parser.add_argument(
        "--eval-seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47],
    )
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
