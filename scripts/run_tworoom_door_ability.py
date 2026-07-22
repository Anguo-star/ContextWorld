#!/usr/bin/env python3
"""Run the frozen original-ability retention suite for the door benchmark.

Planning evaluates the original held-out catalog independently for every
model and evaluation seed (50 queries per seed).  Rollout evaluates the
retention protocol's frozen 1/2/3/5-step catalog once per model.  Existing
outputs are reused only when paths, hashes, protocol values, and record IDs
all match the current frozen inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)
MODES = ("planning", "rollout")


@dataclass(frozen=True)
class Model:
    group: str
    slug: str
    training_seed: int
    checkpoint: Path


@dataclass(frozen=True)
class Job:
    mode: str
    model: Model
    output: Path
    log: Path
    catalog: Path
    eval_seed: int | None = None

    @property
    def label(self) -> str:
        suffix = f"/seed{self.eval_seed}" if self.eval_seed is not None else ""
        return f"{self.mode}/{self.model.slug}{suffix}"


@dataclass(frozen=True)
class FrozenInputs:
    protocol_path: Path
    protocol_sha256: str
    normalizer: Path
    normalizer_sha256: str
    planning_catalog: Path
    planning_catalog_sha256: str
    rollout_catalog: Path
    rollout_catalog_sha256: str
    eval_seeds: tuple[int, ...]
    num_eval_per_seed: int
    planning: dict[str, int]
    rollout_horizons: tuple[int, ...]
    planning_ids_by_seed: dict[int, frozenset[str]]
    rollout_ids: frozenset[str]
    rollout_counts_by_domain_seed: dict[tuple[str, int], int]
    stablewm_ref: str

    @property
    def rollout_record_count(self) -> int:
        return len(self.rollout_ids)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _load_door_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if not str(config.get("status", "")).startswith("preregistered_"):
        raise ValueError("Door benchmark config is not a preregistration")
    required = {"benchmark", "models", "training_protocol", "ability_retention"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Door benchmark config is incomplete: {sorted(missing)}")
    return config


def _retention_protocol(
    config: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    try:
        logical_path = config["ability_retention"]["protocol"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Door config must declare ability_retention.protocol"
        ) from error
    path = resolve_contextworld_path(logical_path, repo_root=ROOT)
    protocol = _load_yaml(path)
    required = {"models", "evaluation_protocol", "artifacts", "training_protocol"}
    missing = required - set(protocol)
    if missing:
        raise ValueError(
            f"Ability-retention protocol is incomplete: {sorted(missing)}"
        )
    return path, protocol


def _stablewm_ref(protocol: dict[str, Any]) -> str:
    prefix = "stable_worldmodel_commit_"
    values = [
        str(value).removeprefix(prefix)
        for value in protocol["training_protocol"].get("fixed_components", [])
        if str(value).startswith(prefix)
    ]
    if len(values) != 1 or re.fullmatch(r"[0-9a-f]{40}", values[0]) is None:
        raise ValueError(
            "Ability-retention protocol must pin exactly one 40-character "
            "stable_worldmodel commit"
        )
    return values[0]


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _unique_ids(entries: list[dict[str, Any]], *, label: str) -> frozenset[str]:
    try:
        values = [str(row["evaluation_id"]) for row in entries]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{label} entries must contain evaluation_id") from error
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate evaluation_id values")
    return frozenset(values)


def _load_frozen_inputs(
    config: dict[str, Any],
    protocol_path: Path,
    protocol: dict[str, Any],
) -> FrozenInputs:
    evaluation = protocol["evaluation_protocol"]
    seeds = tuple(int(value) for value in evaluation["eval_seeds"])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Frozen ability-retention seeds must be nonempty and unique")
    num_eval = int(evaluation["num_eval_per_seed"])
    if num_eval <= 0:
        raise ValueError("num_eval_per_seed must be positive")
    expected_total = len(seeds) * num_eval
    declared_total = int(config["ability_retention"]["per_eval_per_model"])
    if declared_total != expected_total:
        raise ValueError(
            "Door config and retention protocol disagree on evaluations per model: "
            f"{declared_total} != {expected_total}"
        )

    planning = {
        key: int(evaluation["planning"][key])
        for key in (
            "eval_budget",
            "horizon",
            "receding_horizon",
            "cem_samples",
            "cem_steps",
            "cem_topk",
        )
    }
    if any(value <= 0 for value in planning.values()):
        raise ValueError("All frozen planning parameters must be positive")
    horizons = tuple(
        int(value) for value in evaluation["rollout_horizons_action_blocks"]
    )
    if not horizons or len(horizons) != len(set(horizons)):
        raise ValueError("Frozen rollout horizons must be nonempty and unique")

    artifacts = protocol["artifacts"]
    normalizer = resolve_contextworld_path(
        artifacts["frozen_normalizer"], repo_root=ROOT
    )
    planning_catalog = resolve_contextworld_path(
        artifacts["original_eval_catalog"], repo_root=ROOT
    )
    rollout_catalog = resolve_contextworld_path(
        artifacts["rollout_catalog"], repo_root=ROOT
    )
    required_files = (protocol_path, normalizer, planning_catalog, rollout_catalog)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Frozen ability-retention inputs are missing:\n" + "\n".join(missing)
        )

    planning_payload = json.loads(planning_catalog.read_text(encoding="utf-8"))
    if planning_payload.get("status") != "frozen":
        raise ValueError("Original held-out planning catalog is not frozen")
    planning_entries = list(planning_payload.get("entries", []))
    if len(planning_entries) != expected_total:
        raise ValueError(
            f"Expected {expected_total} planning entries, found "
            f"{len(planning_entries)}"
        )
    planning_ids_by_seed: dict[int, frozenset[str]] = {}
    for seed in seeds:
        selected = [
            row for row in planning_entries if int(row.get("eval_seed", -1)) == seed
        ]
        if len(selected) != num_eval:
            raise ValueError(
                f"Planning seed {seed} has {len(selected)} entries; expected {num_eval}"
            )
        planning_ids_by_seed[seed] = _unique_ids(
            selected, label=f"planning seed {seed}"
        )
    if set(int(row.get("eval_seed", -1)) for row in planning_entries) != set(seeds):
        raise ValueError("Planning catalog contains an unexpected evaluation seed")
    catalog_planning = planning_payload.get("protocol", {})
    for key, expected in {**planning, "num_eval_per_seed": num_eval}.items():
        if int(catalog_planning.get(key, -1)) != expected:
            raise ValueError(
                f"Planning catalog protocol mismatch for {key}: "
                f"{catalog_planning.get(key)!r} != {expected}"
            )
    if tuple(map(int, catalog_planning.get("eval_seeds", []))) != seeds:
        raise ValueError("Planning catalog evaluation seeds do not match protocol")

    rollout_payload = json.loads(rollout_catalog.read_text(encoding="utf-8"))
    if rollout_payload.get("status") != "frozen":
        raise ValueError("Rollout catalog is not frozen")
    rollout_entries = list(rollout_payload.get("entries", []))
    rollout_ids = _unique_ids(rollout_entries, label="rollout catalog")
    rollout_protocol = rollout_payload.get("protocol", {})
    if tuple(map(int, rollout_protocol.get("eval_seeds", []))) != seeds:
        raise ValueError("Rollout catalog evaluation seeds do not match protocol")
    if tuple(map(int, rollout_protocol.get("rollout_horizons", []))) != horizons:
        raise ValueError("Rollout catalog horizons do not match protocol")
    if int(rollout_protocol.get("num_eval_per_seed_per_domain", -1)) != num_eval:
        raise ValueError("Rollout catalog per-domain count does not match protocol")
    counts = Counter(
        (str(row.get("domain")), int(row.get("eval_seed", -1)))
        for row in rollout_entries
    )
    domains = {domain for domain, _ in counts}
    if "original_heldout" not in domains:
        raise ValueError("Rollout catalog does not contain original_heldout")
    expected_cells = {(domain, seed) for domain in domains for seed in seeds}
    if set(counts) != expected_cells or any(
        counts[cell] != num_eval for cell in expected_cells
    ):
        raise ValueError(
            "Rollout catalog must contain exactly num_eval_per_seed entries "
            "for every domain and seed"
        )

    return FrozenInputs(
        protocol_path=protocol_path,
        protocol_sha256=file_sha256(protocol_path),
        normalizer=normalizer,
        normalizer_sha256=file_sha256(normalizer),
        planning_catalog=planning_catalog,
        planning_catalog_sha256=file_sha256(planning_catalog),
        rollout_catalog=rollout_catalog,
        rollout_catalog_sha256=file_sha256(rollout_catalog),
        eval_seeds=seeds,
        num_eval_per_seed=num_eval,
        planning=planning,
        rollout_horizons=horizons,
        planning_ids_by_seed=planning_ids_by_seed,
        rollout_ids=rollout_ids,
        rollout_counts_by_domain_seed=dict(counts),
        stablewm_ref=_stablewm_ref(protocol),
    )


def _original_checkpoint(protocol: dict[str, Any]) -> Path:
    rows = [
        row
        for row in protocol["models"]
        if list(row.get("training_groups", [])) == ["original"]
    ]
    if len(rows) != 1 or "checkpoint" not in rows[0]:
        raise ValueError(
            "Ability-retention protocol must declare exactly one original-only model"
        )
    return resolve_contextworld_path(rows[0]["checkpoint"], repo_root=ROOT)


def _models(
    config: dict[str, Any],
    protocol: dict[str, Any],
    selected: set[str] | None,
) -> list[Model]:
    optimizer_steps = int(config["training_protocol"]["optimizer_steps"])
    rows: list[Model] = []
    for group, model_config in config["models"].items():
        seeds = tuple(int(value) for value in model_config["required_training_seeds"])
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seed in model group {group}")
        if group == "original_reference":
            if len(seeds) != 1:
                raise ValueError("Original reference must declare exactly one seed")
            checkpoint = _original_checkpoint(protocol)
            rows.append(
                Model(
                    group=str(group),
                    slug=checkpoint.parent.name,
                    training_seed=seeds[0],
                    checkpoint=checkpoint,
                )
            )
            continue
        groups = dict(model_config["training_groups"])
        synthetic = sorted(set(groups) - {"original"})
        if len(synthetic) != 1:
            raise ValueError(
                f"Expected one synthetic group for {group}: {synthetic}"
            )
        for seed in seeds:
            slug = f"h3_{synthetic[0]}_s{seed}"
            checkpoint = artifact_path(
                "training",
                "runs",
                "checkpoints",
                slug,
                f"weights_final_step_{optimizer_steps}.pt",
                repo_root=ROOT,
            )
            rows.append(
                Model(
                    group=str(group),
                    slug=slug,
                    training_seed=seed,
                    checkpoint=checkpoint,
                )
            )
    if len(rows) != 7:
        raise ValueError(
            "Door ability-retention matrix must contain 7 models "
            f"(original + fixed-door x3 + multi-door x3), found {len(rows)}"
        )
    if len({row.slug for row in rows}) != len(rows):
        raise ValueError("Door ability-retention model slugs must be unique")
    if selected is not None:
        unknown = selected - {row.slug for row in rows}
        if unknown:
            raise ValueError(f"Unknown model slugs: {sorted(unknown)}")
        rows = [row for row in rows if row.slug in selected]
    if not rows:
        raise ValueError("No models selected")
    return rows


def _benchmark_root(config: dict[str, Any]) -> Path:
    benchmark = str(config["benchmark"])
    if not benchmark.startswith("tworoom_"):
        raise ValueError(f"Unexpected benchmark name: {benchmark}")
    return artifact_path(
        "evaluation",
        "history3",
        benchmark.removeprefix("tworoom_"),
        repo_root=ROOT,
    )


def _ability_root(config: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return _benchmark_root(config) / "ability_retention"


def _selected_seeds(
    frozen: FrozenInputs, requested: list[int] | None
) -> tuple[int, ...]:
    if requested is None:
        return frozen.eval_seeds
    values = tuple(int(value) for value in requested)
    if not values or len(values) != len(set(values)):
        raise ValueError("Selected evaluation seeds must be nonempty and unique")
    unknown = set(values) - set(frozen.eval_seeds)
    if unknown:
        raise ValueError(f"Unknown evaluation seeds: {sorted(unknown)}")
    return values


def _jobs(
    args: argparse.Namespace,
    config: dict[str, Any],
    protocol: dict[str, Any],
    frozen: FrozenInputs,
) -> list[Job]:
    modes = MODES if args.mode == "all" else (args.mode,)
    models = _models(
        config, protocol, set(args.models) if args.models is not None else None
    )
    seeds = _selected_seeds(frozen, args.eval_seeds)
    root = _ability_root(config, args.artifact_root)
    jobs: list[Job] = []
    for mode in modes:
        for model in models:
            model_root = root / model.slug
            if mode == "rollout":
                output = model_root / "rollout_error.json"
                jobs.append(
                    Job(
                        mode=mode,
                        model=model,
                        catalog=frozen.rollout_catalog,
                        output=output,
                        log=output.with_suffix(".log"),
                    )
                )
                continue
            for seed in seeds:
                output = (
                    model_root
                    / "planning_original_heldout"
                    / f"seed{seed}.json"
                )
                jobs.append(
                    Job(
                        mode=mode,
                        model=model,
                        catalog=frozen.planning_catalog,
                        eval_seed=seed,
                        output=output,
                        log=output.with_suffix(".log"),
                    )
                )
    return jobs


def _command(
    job: Job,
    *,
    args: argparse.Namespace,
    frozen: FrozenInputs,
) -> list[str]:
    common = [
        "--catalog",
        str(job.catalog),
        "--checkpoint",
        str(job.model.checkpoint),
        "--normalizer",
        str(frozen.normalizer),
        "--output",
        str(job.output),
        "--stablewm-repo",
        str(args.stablewm_repo),
        "--stablewm-ref",
        frozen.stablewm_ref,
        "--device",
        "cuda:0",
    ]
    if job.mode == "rollout":
        return [
            args.python,
            str(ROOT / "scripts/eval_tworoom_rollout_error.py"),
            *common,
            "--batch-size",
            str(args.rollout_batch_size),
        ]
    if job.eval_seed is None:
        raise AssertionError("A planning job requires an evaluation seed")
    planning = frozen.planning
    return [
        args.python,
        str(ROOT / "scripts/eval_tworoom_ability_catalog.py"),
        *common,
        "--seed",
        str(job.eval_seed),
        "--eval-budget",
        str(planning["eval_budget"]),
        "--horizon",
        str(planning["horizon"]),
        "--receding-horizon",
        str(planning["receding_horizon"]),
        "--cem-samples",
        str(planning["cem_samples"]),
        "--cem-steps",
        str(planning["cem_steps"]),
        "--cem-topk",
        str(planning["cem_topk"]),
    ]


def _checkpoint_sha256(job: Job, cache: dict[Path, str]) -> str:
    path = job.model.checkpoint.resolve()
    if path not in cache:
        cache[path] = file_sha256(path)
    return cache[path]


def _valid_common(
    payload: dict[str, Any],
    job: Job,
    *,
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> bool:
    if payload.get("status") != "passed":
        return False
    checkpoint = payload.get("checkpoint", {})
    if not _same_path(checkpoint.get("path"), job.model.checkpoint):
        return False
    if checkpoint.get("sha256") != _checkpoint_sha256(job, checkpoint_hashes):
        return False
    normalizer = payload.get("normalizer", {})
    if not _same_path(normalizer.get("path"), frozen.normalizer):
        return False
    if normalizer.get("sha256") != frozen.normalizer_sha256:
        return False
    catalog = payload.get("catalog", {})
    if not _same_path(catalog.get("path"), job.catalog):
        return False
    expected_catalog_hash = (
        frozen.planning_catalog_sha256
        if job.mode == "planning"
        else frozen.rollout_catalog_sha256
    )
    if catalog.get("sha256") != expected_catalog_hash:
        return False
    if payload.get("stable_worldmodel", {}).get("commit") != frozen.stablewm_ref:
        return False
    audit = payload.get("frozen_weight_audit", {})
    return bool(
        audit.get("passed")
        and audit.get("state_dict_sha256_before")
        and audit.get("state_dict_sha256_before")
        == audit.get("state_dict_sha256_after")
    )


def _valid_output_unchecked(
    job: Job,
    *,
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> bool:
    if not job.output.is_file() or not job.model.checkpoint.is_file():
        return False
    payload = json.loads(job.output.read_text(encoding="utf-8"))
    if not _valid_common(
        payload, job, frozen=frozen, checkpoint_hashes=checkpoint_hashes
    ):
        return False
    records = list(payload.get("raw_records", []))
    ids = [str(row.get("evaluation_id")) for row in records]
    if len(ids) != len(set(ids)):
        return False
    protocol = payload.get("protocol", {})
    if int(protocol.get("action_block", -1)) != 5:
        return False
    if int(protocol.get("history_size", -1)) != 3:
        return False
    if job.mode == "planning":
        if job.eval_seed is None:
            return False
        expected_ids = frozen.planning_ids_by_seed[job.eval_seed]
        if len(records) != frozen.num_eval_per_seed or set(ids) != expected_ids:
            return False
        if any(int(row.get("eval_seed", -1)) != job.eval_seed for row in records):
            return False
        expected_protocol = {
            "eval_seed": job.eval_seed,
            "evaluations": frozen.num_eval_per_seed,
            **frozen.planning,
        }
        if any(
            int(protocol.get(key, -1)) != expected
            for key, expected in expected_protocol.items()
        ):
            return False
        aggregate = payload.get("aggregate", {})
        return int(aggregate.get("evaluations", -1)) == frozen.num_eval_per_seed

    if len(records) != frozen.rollout_record_count or set(ids) != frozen.rollout_ids:
        return False
    if tuple(map(int, protocol.get("horizons_action_blocks", []))) != (
        frozen.rollout_horizons
    ):
        return False
    counts = Counter(
        (str(row.get("domain")), int(row.get("eval_seed", -1)))
        for row in records
    )
    if dict(counts) != frozen.rollout_counts_by_domain_seed:
        return False
    horizon_keys = {str(value) for value in frozen.rollout_horizons}
    if any(set(row.get("horizons", {})) != horizon_keys for row in records):
        return False
    aggregate_cells = {
        (str(row.get("domain")), int(row.get("horizon_action_blocks", -1))): int(
            row.get("evaluations", -1)
        )
        for row in payload.get("aggregates", [])
    }
    expected_domain_counts = Counter()
    for (domain, _seed), count in frozen.rollout_counts_by_domain_seed.items():
        expected_domain_counts[domain] += count
    expected_aggregates = {
        (domain, horizon): count
        for domain, count in expected_domain_counts.items()
        for horizon in frozen.rollout_horizons
    }
    return aggregate_cells == expected_aggregates


def _valid_output(
    job: Job,
    *,
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> bool:
    try:
        return _valid_output_unchecked(
            job, frozen=frozen, checkpoint_hashes=checkpoint_hashes
        )
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _run_job(
    job: Job,
    *,
    gpu: str,
    args: argparse.Namespace,
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> dict[str, Any]:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    job.log.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args=args, frozen=frozen)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
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
    last_error = "subprocess not started"
    for attempt in range(1, args.retries + 2):
        try:
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
            valid = _valid_output(
                job, frozen=frozen, checkpoint_hashes=checkpoint_hashes
            )
            if completed.returncode == 0 and valid:
                return {
                    "label": job.label,
                    "mode": job.mode,
                    "status": "passed",
                    "gpu": str(gpu),
                    "attempts": attempt,
                    "elapsed_seconds": time.time() - started,
                    "output": str(job.output),
                    "log": str(job.log),
                }
            last_error = (
                f"returncode={completed.returncode}, valid_output={valid}"
            )
        except OSError as error:
            last_error = f"{type(error).__name__}: {error}"
    return {
        "label": job.label,
        "mode": job.mode,
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
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        row = _run_job(
            job,
            gpu=gpu,
            args=args,
            frozen=frozen,
            checkpoint_hashes=checkpoint_hashes,
        )
        rows.append(row)
        print(
            f"[{row['status']}] gpu={gpu} {row['label']} "
            f"elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return rows


def _execute_stage(
    jobs: list[Job],
    *,
    args: argparse.Namespace,
    frozen: FrozenInputs,
    checkpoint_hashes: dict[Path, str],
) -> list[dict[str, Any]]:
    queues = {str(gpu): [] for gpu in args.gpus}
    for index, job in enumerate(sorted(jobs, key=lambda row: row.label)):
        gpu = str(args.gpus[index % len(args.gpus)])
        queues[gpu].append(job)
        print(f"[queue] gpu={gpu} {job.label}", flush=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue,
                gpu,
                gpu_jobs,
                args=args,
                frozen=frozen,
                checkpoint_hashes=checkpoint_hashes,
            )
            for gpu, gpu_jobs in queues.items()
            if gpu_jobs
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.gpus:
        raise ValueError("At least one GPU queue is required")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive")
    config_path = args.config.expanduser().resolve()
    config = _load_door_config(config_path)
    protocol_path, protocol = _retention_protocol(config)
    frozen = _load_frozen_inputs(config, protocol_path, protocol)
    jobs = _jobs(args, config, protocol, frozen)
    checkpoint_hashes: dict[Path, str] = {}
    missing_checkpoints = sorted(
        {
            str(job.model.checkpoint)
            for job in jobs
            if not job.model.checkpoint.is_file()
        }
    )
    pending = [
        job
        for job in jobs
        if args.force
        or not _valid_output(
            job, frozen=frozen, checkpoint_hashes=checkpoint_hashes
        )
    ]
    skipped = len(jobs) - len(pending)
    summary = {
        "mode": args.mode,
        "jobs": len(jobs),
        "pending": len(pending),
        "skipped_valid": skipped,
        "gpus": [str(value) for value in args.gpus],
        "missing_checkpoints": missing_checkpoints,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "dry_run",
            **summary,
            "ability_retention_protocol": {
                "path": str(frozen.protocol_path),
                "sha256": frozen.protocol_sha256,
            },
            "commands": [
                {
                    "label": job.label,
                    "command": _command(job, args=args, frozen=frozen),
                }
                for job in pending
            ],
        }
    if missing_checkpoints:
        raise FileNotFoundError(
            "Required door checkpoints are missing:\n"
            + "\n".join(missing_checkpoints)
        )

    results: list[dict[str, Any]] = []
    aborted_modes: list[str] = []
    requested_modes = list(MODES) if args.mode == "all" else [args.mode]
    for mode_index, mode in enumerate(requested_modes):
        stage = [job for job in pending if job.mode == mode]
        if not stage:
            continue
        stage_results = _execute_stage(
            stage,
            args=args,
            frozen=frozen,
            checkpoint_hashes=checkpoint_hashes,
        )
        results.extend(stage_results)
        if any(row["status"] != "passed" for row in stage_results):
            aborted_modes = requested_modes[mode_index + 1 :]
            break

    failures = [row for row in results if row["status"] != "passed"]
    ability_root = _ability_root(config, args.artifact_root)
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "failed" if failures else "passed",
        "mode": args.mode,
        "door_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "ability_retention_protocol": {
            "path": str(frozen.protocol_path),
            "sha256": frozen.protocol_sha256,
        },
        "inputs": {
            "normalizer": {
                "path": str(frozen.normalizer),
                "sha256": frozen.normalizer_sha256,
            },
            "planning_catalog": {
                "path": str(frozen.planning_catalog),
                "sha256": frozen.planning_catalog_sha256,
            },
            "rollout_catalog": {
                "path": str(frozen.rollout_catalog),
                "sha256": frozen.rollout_catalog_sha256,
            },
        },
        "artifact_root": str(ability_root),
        "jobs": len(jobs),
        "skipped_valid": skipped,
        "executed": len(results),
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": len(failures),
        "aborted_modes": aborted_modes,
        "results": sorted(results, key=lambda row: row["label"]),
    }
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else ability_root / "runner_reports" / f"{args.mode}.json"
    )
    write_json(report_path, report)
    if failures:
        raise RuntimeError(
            f"{len(failures)} ability-retention jobs failed; see {report_path}"
        )
    return {**report, "report": str(report_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=(*MODES, "all"), required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
