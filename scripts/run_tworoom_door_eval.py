#!/usr/bin/env python3
"""Run the frozen TwoRoom visible-door evaluation matrix.

The runner deliberately contains no benchmark values such as door positions,
evaluation seeds, query counts, or CEM settings.  Those values are read from
the preregistered door benchmark config and forwarded to the metric-specific
entry points.  One subprocess is allowed on each visible GPU at a time.
"""

from __future__ import annotations

import argparse
import json
import os
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
from contextworld.evaluation.sealed_test_gate import (
    canonical_door_planning_catalog,
    canonical_door_split_root,
    require_canonical_split_path,
    require_sealed_test_gate,
)
from contextworld.paths import (
    artifact_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)
MODES = ("latent", "fixed", "planning")


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
    split: str
    output: Path
    log: Path
    artifact_root: Path
    track: str | None = None
    door_position: int | None = None
    eval_seed: int | None = None
    catalog: Path | None = None

    @property
    def label(self) -> str:
        if self.mode == "latent":
            return f"latent/{self.split}/{self.model.slug}"
        return (
            f"{self.mode}/{self.split}/{self.model.slug}/{self.track}/"
            f"door{self.door_position}/seed{self.eval_seed}"
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _load_config(path: Path) -> dict[str, Any]:
    config = _load_yaml(path)
    if not str(config.get("status", "")).startswith("preregistered_"):
        raise ValueError("Door evaluation config is not a preregistration")
    required = {
        "benchmark",
        "models",
        "training_protocol",
        "evaluation_data",
        "fixed_candidate_evaluation",
        "closed_loop_planning",
        "ability_retention",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Door evaluation config is incomplete: {sorted(missing)}")
    return config


def _retention_config(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = resolve_contextworld_path(
        config["ability_retention"]["protocol"], repo_root=ROOT
    )
    return path, _load_yaml(path)


def _normalizer(config: dict[str, Any]) -> Path:
    """Resolve the normalizer through paths declared by the door config."""

    direct = config.get("artifacts", {}).get("frozen_normalizer")
    if direct is not None:
        return resolve_contextworld_path(direct, repo_root=ROOT)
    _, retention = _retention_config(config)
    try:
        value = retention["artifacts"]["frozen_normalizer"]
    except KeyError as error:
        raise ValueError(
            "The door config's ability-retention protocol does not declare "
            "artifacts.frozen_normalizer"
        ) from error
    return resolve_contextworld_path(value, repo_root=ROOT)


def _original_checkpoint(config: dict[str, Any]) -> Path:
    _, retention = _retention_config(config)
    originals = [
        row
        for row in retention.get("models", [])
        if list(row.get("training_groups", [])) == ["original"]
    ]
    if len(originals) != 1 or "checkpoint" not in originals[0]:
        raise ValueError(
            "Expected exactly one original-only checkpoint in the door "
            "config's ability-retention protocol"
        )
    return resolve_contextworld_path(originals[0]["checkpoint"], repo_root=ROOT)


def _models(config: dict[str, Any], selected: set[str] | None) -> list[Model]:
    """Expand the preregistered 1 + 3 + 3 checkpoint matrix."""

    steps = int(config["training_protocol"]["optimizer_steps"])
    rows: list[Model] = []
    for group, model_config in config["models"].items():
        seeds = [int(value) for value in model_config["required_training_seeds"]]
        if group == "original_reference":
            if len(seeds) != 1:
                raise ValueError("The original reference must declare one seed")
            checkpoint = _original_checkpoint(config)
            rows.append(
                Model(
                    group=str(group),
                    slug=checkpoint.parent.name,
                    training_seed=seeds[0],
                    checkpoint=checkpoint,
                )
            )
            continue
        training_groups = dict(model_config["training_groups"])
        synthetic_groups = sorted(set(training_groups) - {"original"})
        if len(synthetic_groups) != 1:
            raise ValueError(
                f"Expected one synthetic group for {group}: {synthetic_groups}"
            )
        run_prefix = f"h3_{synthetic_groups[0]}"
        for seed in seeds:
            slug = f"{run_prefix}_s{seed}"
            checkpoint = artifact_path(
                "training",
                "runs",
                "checkpoints",
                slug,
                f"weights_final_step_{steps}.pt",
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
    if selected is not None:
        known = {row.slug for row in rows}
        unknown = selected - known
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
        "evaluation", "history3", benchmark.removeprefix("tworoom_"), repo_root=ROOT
    )


def _split_root(
    config: dict[str, Any], split: str, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    root = _benchmark_root(config)
    return root if split == "validation" else root / "sealed_test"


def _planning_catalog(
    config: dict[str, Any], split: str, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    return _benchmark_root(config) / "planning" / split / "catalog.json"


def _selected_tracks(
    config: dict[str, Any], split: str, selected: set[str] | None
) -> list[tuple[str, dict[str, Any]]]:
    rows = [
        (str(name), dict(row))
        for name, row in config["evaluation_data"]["tracks"].items()
        if str(row["split"]) == split
    ]
    if selected is not None:
        known = {name for name, _ in rows}
        unknown = selected - known
        if unknown:
            raise ValueError(
                f"Tracks do not belong to split {split}: {sorted(unknown)}"
            )
        rows = [(name, row) for name, row in rows if name in selected]
    if not rows:
        raise ValueError(f"No tracks selected for split {split}")
    return rows


def _selected_eval_seeds(
    config: dict[str, Any], requested: list[int] | None
) -> list[int]:
    frozen = [int(value) for value in config["evaluation_data"]["eval_seeds"]]
    if requested is None:
        return frozen
    selected = [int(value) for value in requested]
    unknown = set(selected) - set(frozen)
    if unknown:
        raise ValueError(f"Unknown evaluation seeds: {sorted(unknown)}")
    if len(selected) != len(set(selected)):
        raise ValueError("Evaluation seeds must be unique")
    return selected


def _jobs(args: argparse.Namespace, config: dict[str, Any]) -> list[Job]:
    modes = list(MODES) if args.mode == "all" else [args.mode]
    split_root = _split_root(config, args.split, args.artifact_root)
    planning_catalog = _planning_catalog(
        config, args.split, args.planning_catalog
    )
    models = _models(
        config, set(args.models) if args.models is not None else None
    )
    tracks = _selected_tracks(
        config,
        args.split,
        set(args.tracks) if args.tracks is not None else None,
    )
    seeds = _selected_eval_seeds(config, args.eval_seeds)
    jobs: list[Job] = []
    for mode in modes:
        for model in models:
            if mode == "latent":
                jobs.append(
                    Job(
                        mode=mode,
                        model=model,
                        split=args.split,
                        artifact_root=split_root,
                        output=split_root / f"{model.slug}.json",
                        log=split_root / "logs" / mode / f"{model.slug}.log",
                    )
                )
                continue
            for track_name, track in tracks:
                for door in map(int, track["door_positions"]):
                    for seed in seeds:
                        directory = (
                            split_root
                            / f"{mode}_results"
                            / model.slug
                            / track_name
                            / f"door{door}"
                        )
                        jobs.append(
                            Job(
                                mode=mode,
                                model=model,
                                split=args.split,
                                track=track_name,
                                door_position=door,
                                eval_seed=seed,
                                catalog=planning_catalog,
                                artifact_root=split_root,
                                output=directory / f"seed{seed}.json",
                                log=directory / f"seed{seed}.log",
                            )
                        )
    return jobs


def _command(
    job: Job,
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    normalizer: Path,
) -> list[str]:
    common = [
        args.python,
        str(
            ROOT
            / "scripts"
            / {
                "latent": "eval_tworoom_door_true_future_latent.py",
                "fixed": "eval_tworoom_door_fixed_candidates.py",
                "planning": "eval_tworoom_door_planning.py",
            }[job.mode]
        ),
        "--config",
        str(args.config.resolve()),
        "--split",
        job.split,
    ]
    if job.split == "sealed_test":
        gate_path = (
            args.sealed_test_gate.resolve()
            if args.sealed_test_gate is not None
            else resolve_contextworld_path(
                config["sealed_test_gate"]["manifest"], repo_root=ROOT
            )
        )
        common.extend(["--sealed-test-gate", str(gate_path)])
    if job.mode == "latent":
        return common + [
            "--artifact-root",
            str(job.artifact_root),
            "--model-slug",
            job.model.slug,
            "--group",
            job.model.group,
            "--training-seed",
            str(job.model.training_seed),
            "--checkpoint",
            str(job.model.checkpoint),
            "--output",
            str(job.output),
            "--normalizer",
            str(normalizer),
            "--device",
            "cuda:0",
        ]
    if job.catalog is None or job.track is None:
        raise AssertionError("A planning cell job needs a catalog and track")
    if job.door_position is None or job.eval_seed is None:
        raise AssertionError("A planning cell job needs a door and eval seed")
    protocol = (
        config["fixed_candidate_evaluation"]
        if job.mode == "fixed"
        else config["closed_loop_planning"]
    )
    command = common + [
        "--catalog",
        str(job.catalog),
        "--checkpoint",
        str(job.model.checkpoint),
        "--normalizer",
        str(normalizer),
        "--output",
        str(job.output),
        "--track",
        job.track,
        "--door-position",
        str(job.door_position),
        "--seed",
        str(job.eval_seed),
        "--num-eval",
        str(protocol["evaluations_per_door_per_seed"]),
        "--run-kind",
        "confirmation",
        "--device",
        "cuda:0",
    ]
    if job.mode == "fixed":
        command.extend(
            [
                "--candidates",
                str(protocol["candidates_per_query"]),
                "--horizon",
                str(protocol["horizon_action_blocks"]),
            ]
        )
    else:
        command.extend(
            [
                "--eval-budget",
                str(protocol["execution_budget_raw_steps"]),
                "--horizon",
                str(protocol["horizon_action_blocks"]),
                "--receding-horizon",
                str(protocol["receding_horizon_action_blocks"]),
                "--cem-num-samples",
                str(protocol["candidates"]),
                "--cem-steps",
                str(protocol["iterations"]),
                "--cem-topk",
                str(protocol["topk"]),
            ]
        )
    return command


def _same_path(left: Any, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def _cached_file_sha256(path: Path, cache: dict[Path, str] | None) -> str:
    resolved = path.expanduser().resolve()
    if cache is None:
        return file_sha256(resolved)
    if resolved not in cache:
        cache[resolved] = file_sha256(resolved)
    return cache[resolved]


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def _formal_input_integrity_errors(
    jobs: list[Job],
    *,
    config: dict[str, Any],
    config_path: Path,
) -> list[str]:
    """Audit complete formal catalogs before any GPU subprocess is launched."""

    errors: list[str] = []
    config_hash = file_sha256(config_path)
    split = jobs[0].split
    expected_tracks = {
        str(name): tuple(int(value) for value in row["door_positions"])
        for name, row in config["evaluation_data"]["tracks"].items()
        if str(row["split"]) == split
    }
    eval_seeds = tuple(
        int(value) for value in config["evaluation_data"]["eval_seeds"]
    )

    latent_roots = {
        job.artifact_root for job in jobs if job.mode == "latent"
    }
    for artifact_root in sorted(latent_roots, key=str):
        report_path = artifact_root / "catalogs" / "build_report.json"
        if not report_path.is_file():
            continue
        try:
            report = _load_json_mapping(
                report_path, label="Door visual build report"
            )
            count_key = (
                "validation_counts_per_checkpoint"
                if split == "validation"
                else "sealed_test_counts_per_checkpoint"
            )
            expected_counts = config["evaluation_data"][count_key]
            count = report.get("count_audit", {})
            checks = {
                "status": report.get("status") == "passed",
                "split": str(report.get("evaluation_split")) == split,
                "config_sha256": str(report.get("config", {}).get("sha256"))
                == config_hash,
                "formal_counts": count.get("formal_50_by_6_counts") is True,
                "eval_seeds": tuple(map(int, count.get("eval_seeds", [])))
                == eval_seeds,
                "scored_sequences": int(
                    count.get("scored_sequences_per_checkpoint", -1)
                )
                == int(expected_counts["scored_sequences"]),
                "horizon_losses": int(
                    count.get("horizon_losses_per_checkpoint", -1)
                )
                == int(expected_counts["horizon_losses"]),
                "tracks": set(report.get("tracks", {}))
                == set(expected_tracks),
            }
            if not all(checks.values()):
                errors.append(
                    f"visual build report is stale or incomplete: {checks}"
                )
                continue
            for track, doors in expected_tracks.items():
                row = report["tracks"][track]
                catalog_path = Path(str(row.get("catalog"))).expanduser().resolve()
                expected_path = (
                    artifact_root / "catalogs" / f"{track}.json"
                ).resolve()
                expected_bundles = (
                    len(doors)
                    * len(config["evaluation_data"]["offline_prediction_tasks"])
                    * int(
                        config["evaluation_data"][
                            "unique_queries_per_door_per_task"
                        ]
                    )
                )
                summary = row.get("summary", {})
                track_checks = {
                    "canonical_path": catalog_path == expected_path,
                    "catalog_exists": catalog_path.is_file(),
                    "catalog_sha256": catalog_path.is_file()
                    and file_sha256(catalog_path) == row.get("catalog_sha256"),
                    "door_positions": tuple(
                        map(int, summary.get("door_positions", []))
                    )
                    == doors,
                    "bundles": int(summary.get("bundles", -1))
                    == expected_bundles,
                    "formal_counts": summary.get("formal_50_by_6_counts")
                    is True,
                }
                if not all(track_checks.values()):
                    errors.append(
                        f"visual catalog {track} is stale or incomplete: "
                        f"{track_checks}"
                    )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"visual input audit failed at {report_path}: {error}")

    planning_catalogs = {
        job.catalog for job in jobs if job.catalog is not None
    }
    for catalog_path in sorted(planning_catalogs, key=str):
        if catalog_path is None or not catalog_path.is_file():
            continue
        try:
            catalog = _load_json_mapping(
                catalog_path, label="Door planning catalog"
            )
            protocol = catalog.get("protocol", {})
            per_seed = int(
                config["closed_loop_planning"][
                    "evaluations_per_door_per_seed"
                ]
            )
            expected_cells = {
                (track, door, seed)
                for track, doors in expected_tracks.items()
                for door in doors
                for seed in eval_seeds
            }
            cell_counts: Counter[tuple[str, int, int]] = Counter()
            cell_indices: dict[tuple[str, int, int], set[int]] = {
                cell: set() for cell in expected_cells
            }
            unexpected_cells = set()
            for row in catalog.get("bundles", []):
                cell = (
                    str(row.get("track")),
                    int(row.get("door_position", -1)),
                    int(row.get("eval_seed", -1)),
                )
                if cell not in expected_cells:
                    unexpected_cells.add(cell)
                    continue
                cell_counts[cell] += 1
                cell_indices[cell].add(int(row.get("evaluation_index", -1)))
            expected_indices = set(range(per_seed))
            checks = {
                "status": catalog.get("status") == "passed",
                "split": str(catalog.get("split_role")) == split,
                "config_sha256": str(catalog.get("config", {}).get("sha256"))
                == config_hash,
                "agent_speed": float(protocol.get("agent_speed", float("nan")))
                == float(config["closed_loop_planning"]["agent_speed"]),
                "task": protocol.get("task")
                == config["closed_loop_planning"]["query_task"],
                "history_tokens": int(protocol.get("history_tokens", -1))
                == int(config["scope"]["history_tokens"]),
                "action_block": int(
                    protocol.get("action_block_raw_steps", -1)
                )
                == int(config["scope"]["action_block_raw_steps"]),
                "candidates": int(protocol.get("candidates_per_query", -1))
                == int(config["fixed_candidate_evaluation"]["candidates_per_query"]),
                "horizon": int(
                    protocol.get("candidate_horizon_action_blocks", -1)
                )
                == int(config["fixed_candidate_evaluation"]["horizon_action_blocks"]),
                "eval_seeds": tuple(map(int, protocol.get("eval_seeds", [])))
                == eval_seeds,
                "per_seed": int(
                    protocol.get("queries_per_door_per_eval_seed", -1)
                )
                == per_seed,
                "cells": set(cell_counts) == expected_cells,
                "cell_counts": all(
                    cell_counts[cell] == per_seed for cell in expected_cells
                ),
                "cell_indices": all(
                    cell_indices[cell] == expected_indices
                    for cell in expected_cells
                ),
                "unexpected_cells": not unexpected_cells,
                "bundles": len(catalog.get("bundles", []))
                == len(expected_cells) * per_seed,
                "formal_summary": catalog.get("summary", {}).get(
                    "formal_50_by_6_per_door"
                )
                is True,
            }
            if not all(checks.values()):
                errors.append(
                    f"planning catalog is stale or incomplete: {checks}"
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            errors.append(f"planning input audit failed at {catalog_path}: {error}")
    return sorted(set(errors))


def _valid_output_unchecked(
    job: Job,
    *,
    config: dict[str, Any],
    config_path: Path,
    normalizer: Path,
    input_hashes: dict[Path, str] | None = None,
) -> bool:
    if not job.output.is_file():
        return False
    try:
        payload = json.loads(job.output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "passed":
        return False
    if job.split == "sealed_test":
        gate_path = resolve_contextworld_path(
            config["sealed_test_gate"]["manifest"], repo_root=ROOT
        )
        gate = payload.get("sealed_test_gate", {})
        if (
            gate.get("passed") is not True
            or not _same_path(gate.get("manifest"), gate_path)
            or not gate_path.is_file()
            or gate.get("manifest_sha256") != file_sha256(gate_path)
        ):
            return False
    config_row = payload.get("config", {})
    if config_row.get("sha256") != file_sha256(config_path):
        return False
    if not _same_path(
        payload.get("normalizer", {}).get("path"), normalizer
    ):
        return False
    if not normalizer.is_file() or payload.get("normalizer", {}).get(
        "sha256"
    ) != _cached_file_sha256(normalizer, input_hashes):
        return False
    if not payload.get("frozen_weight_audit", {}).get("passed"):
        return False
    count = payload.get("count_audit", {})
    if not count.get("passed"):
        return False
    if job.mode == "latent":
        count_key = (
            "validation_counts_per_checkpoint"
            if job.split == "validation"
            else "sealed_test_counts_per_checkpoint"
        )
        expected = int(
            config["evaluation_data"][count_key]["scored_sequences"]
        )
        model = payload.get("model", {})
        build_report = job.artifact_root / "catalogs" / "build_report.json"
        return (
            payload.get("evaluation_split") == job.split
            and model.get("slug") == job.model.slug
            and model.get("group") == job.model.group
            and int(model.get("training_seed", -1))
            == job.model.training_seed
            and _same_path(model.get("checkpoint"), job.model.checkpoint)
            and job.model.checkpoint.is_file()
            and model.get("checkpoint_sha256")
            == _cached_file_sha256(job.model.checkpoint, input_hashes)
            and _same_path(
                payload.get("build_report", {}).get("path"), build_report
            )
            and build_report.is_file()
            and payload.get("build_report", {}).get("sha256")
            == _cached_file_sha256(build_report, input_hashes)
            and int(count.get("scored_sequences", -1)) == expected
        )
    if payload.get("run_kind") != "confirmation":
        return False
    if job.split == "sealed_test" and payload.get("evaluation_split") != job.split:
        return False
    if payload.get("evaluation_split") not in (None, job.split):
        return False
    if payload.get("track") != job.track:
        return False
    if int(payload.get("door_position", -1)) != job.door_position:
        return False
    if int(payload.get("eval_seed", -1)) != job.eval_seed:
        return False
    if not _same_path(
        payload.get("model", {}).get("checkpoint"), job.model.checkpoint
    ):
        return False
    if not job.model.checkpoint.is_file() or payload.get("model", {}).get(
        "sha256"
    ) != _cached_file_sha256(job.model.checkpoint, input_hashes):
        return False
    if not _same_path(payload.get("catalog", {}).get("path"), job.catalog):
        return False
    if job.catalog is None or not job.catalog.is_file() or payload.get(
        "catalog", {}
    ).get("sha256") != _cached_file_sha256(job.catalog, input_hashes):
        return False
    protocol = payload.get("protocol", {})
    frozen = (
        config["fixed_candidate_evaluation"]
        if job.mode == "fixed"
        else config["closed_loop_planning"]
    )
    expected_records = int(frozen["evaluations_per_door_per_seed"])
    if (
        int(count.get("records", -1)) != expected_records
        or int(protocol.get("queries", -1)) != expected_records
        or float(protocol.get("agent_speed", float("nan")))
        != float(config["closed_loop_planning"]["agent_speed"])
    ):
        return False
    if job.mode == "fixed":
        return (
            int(protocol.get("candidates_per_query", -1))
            == int(frozen["candidates_per_query"])
            and int(protocol.get("horizon_action_blocks", -1))
            == int(frozen["horizon_action_blocks"])
        )
    return (
        int(protocol.get("eval_budget_raw_steps", -1))
        == int(frozen["execution_budget_raw_steps"])
        and int(protocol.get("horizon_action_blocks", -1))
        == int(frozen["horizon_action_blocks"])
        and int(protocol.get("receding_horizon_action_blocks", -1))
        == int(frozen["receding_horizon_action_blocks"])
        and int(protocol.get("cem_samples", -1)) == int(frozen["candidates"])
        and int(protocol.get("cem_iterations", -1))
        == int(frozen["iterations"])
        and int(protocol.get("cem_topk", -1)) == int(frozen["topk"])
    )


def _valid_output(
    job: Job,
    *,
    config: dict[str, Any],
    config_path: Path,
    normalizer: Path,
    input_hashes: dict[Path, str] | None = None,
) -> bool:
    """Return false, rather than aborting resume, for any stale partial JSON."""

    try:
        return _valid_output_unchecked(
            job,
            config=config,
            config_path=config_path,
            normalizer=normalizer,
            input_hashes=input_hashes,
        )
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        return False


def _run_job(
    job: Job,
    *,
    gpu: str,
    args: argparse.Namespace,
    config: dict[str, Any],
    normalizer: Path,
    input_hashes: dict[Path, str] | None = None,
) -> dict[str, Any]:
    job.output.parent.mkdir(parents=True, exist_ok=True)
    job.log.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args=args, config=config, normalizer=normalizer)
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
                job,
                config=config,
                config_path=args.config.resolve(),
                normalizer=normalizer,
                input_hashes=input_hashes,
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
    config: dict[str, Any],
    normalizer: Path,
    input_hashes: dict[Path, str],
) -> list[dict[str, Any]]:
    rows = []
    for job in jobs:
        result = _run_job(
            job,
            gpu=gpu,
            args=args,
            config=config,
            normalizer=normalizer,
            input_hashes=input_hashes,
        )
        rows.append(result)
        print(
            f"[{result['status']}] gpu={gpu} {result['label']} "
            f"elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return rows


def _execute_stage(
    jobs: list[Job],
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    normalizer: Path,
    input_hashes: dict[Path, str],
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
                config=config,
                normalizer=normalizer,
                input_hashes=input_hashes,
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
    config_path = args.config.resolve()
    config = _load_config(config_path)
    gate_audit = require_sealed_test_gate(
        split=args.split,
        config_path=config_path,
        config=config,
        manifest_path=args.sealed_test_gate,
        repo_root=ROOT,
    )
    require_canonical_split_path(
        args.artifact_root,
        canonical=canonical_door_split_root(
            config, split=args.split, repo_root=ROOT
        ),
        split=args.split,
        label="Door evaluation artifact root",
    )
    require_canonical_split_path(
        args.planning_catalog,
        canonical=canonical_door_planning_catalog(
            config, split=args.split, repo_root=ROOT
        ),
        split=args.split,
        label="Door evaluation planning catalog",
    )
    normalizer = _normalizer(config)
    jobs = _jobs(args, config)
    missing_checkpoints = sorted(
        {str(job.model.checkpoint) for job in jobs if not job.model.checkpoint.is_file()}
    )
    missing_inputs = []
    if not normalizer.is_file():
        missing_inputs.append(str(normalizer))
    for build_report in sorted(
        {
            job.artifact_root / "catalogs" / "build_report.json"
            for job in jobs
            if job.mode == "latent"
        },
        key=str,
    ):
        if not build_report.is_file():
            missing_inputs.append(str(build_report))
    for catalog in sorted(
        {job.catalog for job in jobs if job.catalog is not None}, key=str
    ):
        if catalog is not None and not catalog.is_file():
            missing_inputs.append(str(catalog))
    invalid_inputs = _formal_input_integrity_errors(
        jobs, config=config, config_path=config_path
    )
    input_hashes: dict[Path, str] = {}
    hash_inputs = {normalizer, *(job.model.checkpoint for job in jobs)}
    hash_inputs.update(
        job.catalog for job in jobs if job.catalog is not None
    )
    hash_inputs.update(
        job.artifact_root / "catalogs" / "build_report.json"
        for job in jobs
        if job.mode == "latent"
    )
    for path in sorted(hash_inputs, key=str):
        if path.is_file():
            _cached_file_sha256(path, input_hashes)
    pending = [
        job
        for job in jobs
        if args.force
        or not _valid_output(
            job,
            config=config,
            config_path=config_path,
            normalizer=normalizer,
            input_hashes=input_hashes,
        )
    ]
    skipped = len(jobs) - len(pending)
    summary = {
        "mode": args.mode,
        "split": args.split,
        "jobs": len(jobs),
        "pending": len(pending),
        "skipped_valid": skipped,
        "gpus": [str(value) for value in args.gpus],
        "missing_checkpoints": missing_checkpoints,
        "missing_inputs": sorted(set(missing_inputs)),
        "invalid_inputs": invalid_inputs,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "status": "dry_run",
            **summary,
            "commands": [
                {
                    "label": job.label,
                    "command": _command(
                        job,
                        args=args,
                        config=config,
                        normalizer=normalizer,
                    ),
                }
                for job in pending
            ],
        }
    missing = missing_checkpoints + sorted(set(missing_inputs))
    if missing:
        raise FileNotFoundError(
            "Required door evaluation inputs are missing:\n" + "\n".join(missing)
        )
    if invalid_inputs:
        raise RuntimeError(
            "Door evaluation inputs failed the formal preflight:\n"
            + "\n".join(invalid_inputs)
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
            config=config,
            normalizer=normalizer,
            input_hashes=input_hashes,
        )
        results.extend(stage_results)
        if any(row["status"] != "passed" for row in stage_results):
            aborted_modes = requested_modes[mode_index + 1 :]
            break
    failures = [row for row in results if row["status"] != "passed"]
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "failed" if failures else "passed",
        "mode": args.mode,
        "split": args.split,
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "normalizer": str(normalizer),
        "sealed_test_gate": gate_audit,
        "artifact_root": str(_split_root(config, args.split, args.artifact_root)),
        "planning_catalog": str(
            _planning_catalog(config, args.split, args.planning_catalog)
        ),
        "jobs": len(jobs),
        "skipped_valid": skipped,
        "executed": len(results),
        "passed": sum(row["status"] == "passed" for row in results),
        "failed": len(failures),
        "aborted_modes": aborted_modes,
        "results": sorted(results, key=lambda row: row["label"]),
    }
    report_path = (
        args.report.resolve()
        if args.report is not None
        else _split_root(config, args.split, args.artifact_root)
        / "runner_reports"
        / f"{args.mode}.json"
    )
    write_json(report_path, report)
    if failures:
        raise RuntimeError(
            f"{len(failures)} door evaluation jobs failed; see {report_path}"
        )
    return {**report, "report": str(report_path)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--split",
        choices=("validation", "sealed_test"),
        default="validation",
    )
    parser.add_argument(
        "--mode", choices=(*MODES, "all"), required=True
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--planning-catalog", type=Path)
    parser.add_argument("--sealed-test-gate", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tracks", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--eval-seeds", type=int, nargs="+")
    parser.add_argument(
        "--gpus", nargs="+", default=[str(index) for index in range(8)]
    )
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
