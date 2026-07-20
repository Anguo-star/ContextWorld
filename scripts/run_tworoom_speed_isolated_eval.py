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
    model_group: str
    model_slug: str
    checkpoint: Path
    track: str
    catalog: Path
    query_speed: float
    eval_seed: int
    output: Path
    log: Path

    @property
    def label(self) -> str:
        return (
            f"{self.mode}/{self.model_slug}/{self.track}/"
            f"q{self.query_speed:g}/s{self.eval_seed}"
        )


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("status") not in {
        "preregistered_before_catalog_generation_and_scoring",
        "amended_before_model_scoring",
    }:
        raise ValueError("Evaluation config is not a frozen preregistration")
    return config


def _models(
    config: dict[str, Any],
    *,
    groups: set[str] | None,
    slugs: set[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    rows = []
    for group, models in config["models"].items():
        if groups is not None and group not in groups:
            continue
        for model in models:
            if slugs is not None and str(model["slug"]) not in slugs:
                continue
            rows.append((str(group), dict(model)))
    if not rows:
        raise ValueError("No models selected")
    return rows


def _valid_output(job: Job, *, num_eval: int) -> bool:
    if not job.output.is_file():
        return False
    try:
        payload = json.loads(job.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "passed":
        return False
    if float(payload.get("query_speed", -1)) != float(job.query_speed):
        return False
    if int(payload.get("eval_seed", -1)) != int(job.eval_seed):
        return False
    audit = payload.get("count_audit", {})
    if not audit.get("passed"):
        return False
    expected = num_eval * 3 if job.mode == "planning" else num_eval
    return int(audit.get("records", -1)) == expected


def _jobs(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[Job]:
    artifacts = config["artifacts"]
    tracks = {
        row["name"]: row
        for row in config["frozen_scope"]["tracks"]
        if row["name"] in args.tracks
    }
    missing_tracks = set(args.tracks) - set(tracks)
    if missing_tracks:
        raise ValueError(f"Unknown tracks: {sorted(missing_tracks)}")
    models = _models(
        config,
        groups=set(args.model_groups) if args.model_groups else None,
        slugs=set(args.models) if args.models else None,
    )
    modes = (
        ["physical", "fixed", "planning"]
        if args.mode == "all"
        else [args.mode]
    )
    seeds = (
        [int(value) for value in args.eval_seeds]
        if args.eval_seeds
        else [
            int(value)
            for value in config["formal_eval"]["eval_seeds"]
        ]
    )
    root_by_mode = {
        "physical": resolve_contextworld_path(
            artifacts["physical_transition_root"], repo_root=ROOT
        ),
        "fixed": resolve_contextworld_path(
            artifacts["fixed_candidate_root"], repo_root=ROOT
        ),
        "planning": resolve_contextworld_path(
            artifacts["closed_loop_root"], repo_root=ROOT
        ),
    }
    result = []
    for mode in modes:
        mode_root = root_by_mode[mode]
        for group, model in models:
            checkpoint = resolve_contextworld_path(
                model["checkpoint"], repo_root=ROOT
            )
            for track_name, track in tracks.items():
                catalog = resolve_contextworld_path(
                    artifacts["catalogs"][track_name], repo_root=ROOT
                )
                for query_speed in track["speeds"]:
                    for seed in seeds:
                        directory = (
                            mode_root
                            / str(model["slug"])
                            / str(track_name)
                            / f"q{float(query_speed):g}"
                        )
                        result.append(
                            Job(
                                mode=mode,
                                model_group=group,
                                model_slug=str(model["slug"]),
                                checkpoint=checkpoint,
                                track=str(track_name),
                                catalog=catalog,
                                query_speed=float(query_speed),
                                eval_seed=int(seed),
                                output=directory / f"s{seed}.json",
                                log=directory / f"s{seed}.log",
                            )
                        )
    return result


def _command(
    job: Job,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[str]:
    formal = config["formal_eval"]
    normalizer = resolve_contextworld_path(
        config["frozen_scope"]["source_training_protocol"],
        repo_root=ROOT,
    )
    training_config = yaml.safe_load(
        normalizer.read_text(encoding="utf-8")
    )
    normalizer_path = resolve_contextworld_path(
        training_config["evaluation_protocol"]["normalizer"],
        repo_root=ROOT,
    )
    entrypoint = {
        "physical": ROOT
        / "scripts/eval_tworoom_speed_physical_transition.py",
        "fixed": ROOT
        / "scripts/eval_tworoom_speed_cube_fixed_candidate.py",
        "planning": ROOT
        / "scripts/eval_tworoom_speed_cube_planning.py",
    }[job.mode]
    command = [
        args.python,
        str(entrypoint),
        "--catalog",
        str(job.catalog),
        "--checkpoint",
        str(job.checkpoint),
        "--normalizer",
        str(normalizer_path),
        "--output",
        str(job.output),
        "--query-speed",
        str(job.query_speed),
        "--seed",
        str(job.eval_seed),
        "--num-eval",
        str(args.num_eval),
        "--device",
        "cuda:0",
        "--stablewm-repo",
        str((ROOT / config["stable_worldmodel"]["repo"]).resolve()),
        "--stablewm-ref",
        str(config["stable_worldmodel"]["expected_ref"]),
    ]
    if job.mode == "physical":
        probe = formal["physical_transition"]
        command.extend(
            [
                "--oracle-speed-min",
                str(probe["oracle_speed_grid_range"][0]),
                "--oracle-speed-max",
                str(probe["oracle_speed_grid_range"][1]),
                "--oracle-speed-step",
                str(probe["oracle_speed_grid_step"]),
                "--oracle-cache-dir",
                str(job.output.parent / "_oracle_cache"),
            ]
        )
    elif job.mode == "fixed":
        fixed = formal["fixed_candidate"]
        command.extend(
            [
                "--candidates",
                str(fixed["candidates"]),
                "--horizon",
                str(fixed["horizon_action_blocks"]),
                "--skip-catalog-replay",
            ]
        )
    else:
        planner = formal["planner"]
        command.extend(
            [
                "--eval-budget",
                str(planner["eval_budget_raw_steps"]),
                "--deadline-budgets",
                *[
                    str(value)
                    for value in planner["deadline_budgets_raw_steps"]
                ],
                "--horizon",
                str(planner["horizon_action_blocks"]),
                "--receding-horizon",
                str(planner["receding_horizon_action_blocks"]),
                "--cem-num-samples",
                str(planner["cem_samples"]),
                "--cem-steps",
                str(planner["cem_steps"]),
                "--cem-topk",
                str(planner["cem_topk"]),
                "--cem-var-scale",
                str(planner["cem_var_scale"]),
                "--skip-catalog-replay",
            ]
        )
    return command


def _run_job(
    job: Job,
    *,
    gpu: str,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args=args, config=config)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("MUJOCO_GL", "egl")
    environment.setdefault("PYTHONUNBUFFERED", "1")
    started = time.time()
    last_error = None
    for attempt in range(1, args.retries + 2):
        with job.log.open("a", encoding="utf-8") as log:
            log.write(
                f"\n[start] attempt={attempt} gpu={gpu} "
                f"command={json.dumps(command)}\n"
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
        if completed.returncode == 0 and _valid_output(
            job, num_eval=args.num_eval
        ):
            return {
                "label": job.label,
                "status": "passed",
                "gpu": str(gpu),
                "attempts": attempt,
                "elapsed_seconds": time.time() - started,
                "output": str(job.output),
                "log": str(job.log),
            }
        last_error = (
            f"returncode={completed.returncode}, "
            f"valid_output={_valid_output(job, num_eval=args.num_eval)}"
        )
    return {
        "label": job.label,
        "status": "failed",
        "gpu": str(gpu),
        "attempts": args.retries + 1,
        "elapsed_seconds": time.time() - started,
        "output": str(job.output),
        "log": str(job.log),
        "error": last_error,
    }


def _run_gpu_queue(
    gpu: str,
    jobs: list[Job],
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        result = _run_job(
            job,
            gpu=gpu,
            args=args,
            config=config,
        )
        rows.append(result)
        print(
            f"[{result['status']}] gpu={result['gpu']} "
            f"{result['label']} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    jobs = _jobs(args=args, config=config)
    missing = sorted(
        {str(job.checkpoint) for job in jobs if not job.checkpoint.is_file()}
    )
    if missing:
        raise FileNotFoundError(
            "Required checkpoints are missing:\n" + "\n".join(missing)
        )
    pending = [
        job
        for job in jobs
        if args.force or not _valid_output(job, num_eval=args.num_eval)
    ]
    skipped = len(jobs) - len(pending)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "jobs": len(jobs),
                "pending": len(pending),
                "skipped_valid": skipped,
                "gpus": args.gpus,
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
            "skipped_valid": skipped,
            "commands": [
                {
                    "label": job.label,
                    "command": _command(
                        job, args=args, config=config
                    ),
                }
                for job in pending
            ],
        }

    results = []
    if pending:
        queues = {
            str(gpu): [] for gpu in args.gpus
        }
        grouped_jobs: dict[
            tuple[str, str, str, float], list[Job]
        ] = {}
        for job in pending:
            key = (
                job.mode,
                job.model_slug,
                job.track,
                job.query_speed,
            )
            grouped_jobs.setdefault(key, []).append(job)
        for group_index, key in enumerate(sorted(grouped_jobs)):
            gpu = str(args.gpus[group_index % len(args.gpus)])
            for job in sorted(
                grouped_jobs[key], key=lambda row: row.eval_seed
            ):
                queues[gpu].append(job)
                print(f"[queue] gpu={gpu} {job.label}", flush=True)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = [
                executor.submit(
                    _run_gpu_queue,
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
        "benchmark": config["benchmark"],
        "status": "failed" if failures else "passed",
        "mode": args.mode,
        "config": str(config_path),
        "jobs": len(jobs),
        "skipped_valid": skipped,
        "executed": len(results),
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": len(failures),
        "results": sorted(results, key=lambda row: row["label"]),
    }
    report_path = resolve_contextworld_path(
        config["artifacts"]["root"], repo_root=ROOT
    ) / f"runner_{args.mode}.json"
    write_json(report_path, report)
    if failures:
        raise RuntimeError(
            f"{len(failures)} evaluation jobs failed; see {report_path}"
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("physical", "fixed", "planning", "all"),
        required=True,
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["seen_for_multi", "unseen_interpolation"],
    )
    parser.add_argument("--model-groups", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
