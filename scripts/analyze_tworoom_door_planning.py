#!/usr/bin/env python3
"""Audit and summarize the frozen TwoRoom visible-door planning results.

This entry point deliberately treats planning as supporting evidence.  It
never converts a fixed-candidate or CEM result into a claim about prediction
accuracy; the true-future latent evaluation owns that claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from scripts.analyze_tworoom_door_visual_generalization import (
    PAIRED_TRAINING_SEEDS,
    TRAINING_BINDINGS,
    _audit_runtime_identity,
    _audit_training_report_bindings,
)


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)
DEFAULT_NORMALIZER = (
    "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
)
MODES = ("fixed", "planning")
EXPECTED_BENCHMARK = {
    "fixed": "tworoom_door_fixed_candidates_v1",
    "planning": "tworoom_door_closed_loop_planning_v1",
}
EXPECTED_EVIDENCE_ROLE = {
    "fixed": "planning_action_ranking_not_latent_accuracy",
    "planning": "closed_loop_planning_not_latent_accuracy",
}


def _split_root(config: Mapping[str, Any], split: str) -> Path:
    benchmark = str(config["benchmark"])
    if not benchmark.startswith("tworoom_"):
        raise ValueError(f"Unexpected benchmark name: {benchmark}")
    root = artifact_path(
        "evaluation",
        "history3",
        benchmark.removeprefix("tworoom_"),
        repo_root=ROOT,
    )
    return root if split == "validation" else root / "sealed_test"


def _expected_tracks(
    config: Mapping[str, Any], split: str
) -> dict[str, tuple[int, ...]]:
    rows = {
        str(name): tuple(int(value) for value in row["door_positions"])
        for name, row in config["evaluation_data"]["tracks"].items()
        if str(row["split"]) == str(split)
    }
    if not rows:
        raise RuntimeError(f"No tracks are configured for split={split}")
    all_doors = [door for doors in rows.values() for door in doors]
    if len(all_doors) != len(set(all_doors)):
        raise RuntimeError("A door position belongs to two tracks in one split")
    return rows


def _resolve_result_paths(
    explicit: Iterable[Path], *, artifact_root: Path, mode: str
) -> list[Path]:
    selected: list[Path] = []
    for value in explicit:
        path = value.expanduser().resolve()
        if path.is_dir():
            selected.extend(path.rglob("*.json"))
        else:
            selected.append(path)
    if not selected:
        selected = list((artifact_root / f"{mode}_results").rglob("*.json"))
    paths = sorted({path.resolve() for path in selected})
    if not paths:
        raise FileNotFoundError(
            f"No {mode} result JSON files found below {artifact_root}"
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))
    return paths


def _training_report_paths(args: argparse.Namespace) -> list[Path]:
    if args.training_reports:
        return [
            resolve_contextworld_path(path, repo_root=ROOT)
            for path in args.training_reports
        ]
    root = resolve_contextworld_path(args.training_report_root, repo_root=ROOT)
    return [
        root / binding["report_name"] for binding in TRAINING_BINDINGS.values()
    ]


def _load_training_identities(
    report_paths: Iterable[Path], *, require_complete: bool
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], Path], str]:
    """Return checkpoint-hash identities after report-only integrity checks."""

    model_id_to_group = {
        binding["model_id"]: group
        for (group, _), binding in TRAINING_BINDINGS.items()
    }
    by_hash: dict[str, dict[str, Any]] = {}
    path_by_key: dict[tuple[str, int], Path] = {}
    stable_commits: set[str] = set()
    for path in report_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(report.get("model_id"))
        if model_id not in model_id_to_group:
            raise RuntimeError(f"Unknown door model_id in training report: {path}")
        group = model_id_to_group[model_id]
        seed = int(report["data"]["seed"])
        key = (group, seed)
        expected = TRAINING_BINDINGS.get(key)
        if expected is None:
            raise RuntimeError(f"Unexpected training report binding {key}: {path}")
        if key in path_by_key:
            raise RuntimeError(f"Repeated training report for {key}")
        checks = {
            "passed": report.get("passed") is True,
            "save_load_exact": report.get("save_load_exact") is True,
            "training_complete": report["training"].get("training_complete")
            is True,
            "model_id": model_id == expected["model_id"],
            "run_name": str(report.get("run_name")) == expected["run_name"],
            "data_seed": seed == key[1],
            "training_seed": int(report["training"]["plan"]["training_seed"])
            == key[1],
        }
        if not all(checks.values()):
            raise RuntimeError(f"Training report failed for {key}: {checks}")
        checkpoint_hash = str(report["artifacts"]["pretrained_sha256"])
        if checkpoint_hash in by_hash:
            raise RuntimeError("Two training reports bind the same checkpoint hash")
        stable_commit = str(report["stable_worldmodel"]["commit"])
        stable_commits.add(stable_commit)
        by_hash[checkpoint_hash] = {
            "slug": expected["run_name"],
            "group": group,
            "training_seed": seed,
            "checkpoint_sha256": checkpoint_hash,
            "training_report": str(path.resolve()),
            "training_report_sha256": file_sha256(path),
        }
        path_by_key[key] = path.resolve()
    if len(stable_commits) != 1:
        raise RuntimeError(
            "Training reports use different StableWorldModel commits: "
            f"{sorted(stable_commits)}"
        )
    if require_complete and set(path_by_key) != set(TRAINING_BINDINGS):
        raise RuntimeError(
            "Expected all seven training reports; observed "
            f"{sorted(path_by_key)}"
        )
    return by_hash, path_by_key, next(iter(stable_commits))


def _load_catalog(
    path: Path,
    *,
    config_hash: str,
    split: str,
    stable_commit: str,
    tracks: Mapping[str, tuple[int, ...]],
    eval_seeds: tuple[int, ...],
    per_seed: int,
    require_formal: bool,
) -> tuple[dict[str, Any], dict[tuple[str, int, int, int], dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if require_formal and catalog.get("status") != "passed":
        raise RuntimeError("Formal planning analysis refuses a smoke catalog")
    if str(catalog.get("split_role")) != str(split):
        raise RuntimeError("Planning catalog split does not match the analysis")
    if str(catalog["config"]["sha256"]) != str(config_hash):
        raise RuntimeError("Planning catalog/config hash mismatch")
    if str(catalog["stable_worldmodel"]["commit"]) != str(stable_commit):
        raise RuntimeError("Planning catalog/StableWorldModel commit mismatch")
    protocol = catalog["protocol"]
    protocol_checks = {
        "speed": float(protocol["agent_speed"]) == 5.0,
        "task": str(protocol["task"]) == "cross_room_navigation",
        "history": int(protocol["history_tokens"]) == 3,
        "action_block": int(protocol["action_block_raw_steps"]) == 5,
        "candidates": int(protocol["candidates_per_query"]) == 300,
        "horizon": int(protocol["candidate_horizon_action_blocks"]) == 10,
        "eval_seeds": tuple(map(int, protocol["eval_seeds"])) == eval_seeds,
        "per_seed": int(protocol["queries_per_door_per_eval_seed"])
        == per_seed,
    }
    if not all(protocol_checks.values()):
        raise RuntimeError(f"Planning catalog protocol mismatch: {protocol_checks}")

    expected_cells = {
        (track, door, seed)
        for track, doors in tracks.items()
        for door in doors
        for seed in eval_seeds
    }
    by_key: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    cell_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    query_ids: set[str] = set()
    for row in catalog.get("bundles", []):
        track = str(row["track"])
        door = int(row.get("door_position", row["query_factors"]["door.position"]))
        seed = int(row["eval_seed"])
        index = int(row["evaluation_index"])
        cell = (track, door, seed)
        if cell not in expected_cells:
            raise RuntimeError(f"Unexpected catalog cell: {cell}")
        key = (*cell, index)
        if key in by_key:
            raise RuntimeError(f"Repeated catalog query key: {key}")
        query_id = str(row["query_id"])
        if query_id in query_ids:
            raise RuntimeError(f"Repeated catalog query_id: {query_id}")
        if int(row["query_factors"]["door.position"]) != door:
            raise RuntimeError(f"Catalog door readback mismatch at {query_id}")
        if float(row["query_factors"]["agent.speed"]) != 5.0:
            raise RuntimeError(f"Catalog speed mismatch at {query_id}")
        by_key[key] = row
        query_ids.add(query_id)
        cell_counts[cell] += 1
    if require_formal:
        if set(cell_counts) != expected_cells:
            raise RuntimeError("Planning catalog is missing one or more formal cells")
        bad = {key: count for key, count in cell_counts.items() if count != per_seed}
        if bad:
            raise RuntimeError(f"Planning catalog cells are not 50 each: {bad}")
        expected_indices = set(range(per_seed))
        for cell in expected_cells:
            observed = {key[3] for key in by_key if key[:3] == cell}
            if observed != expected_indices:
                raise RuntimeError(f"Catalog indices are incomplete for {cell}")
    return catalog, by_key


def _assert_common_result(
    payload: Mapping[str, Any],
    *,
    path: Path,
    mode: str,
    config_hash: str,
    catalog_path: Path,
    catalog_hash: str,
    normalizer_hash: str,
    stable_commit: str,
    tracks: Mapping[str, tuple[int, ...]],
    eval_seeds: tuple[int, ...],
    per_seed: int,
    identities_by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if payload.get("status") != "passed" or payload.get("run_kind") != "confirmation":
        raise RuntimeError(f"Formal analysis refuses a smoke/partial result: {path}")
    if payload.get("benchmark") != EXPECTED_BENCHMARK[mode]:
        raise RuntimeError(f"Wrong benchmark in {path}")
    if payload.get("evidence_role") != EXPECTED_EVIDENCE_ROLE[mode]:
        raise RuntimeError(f"Wrong evidence role in {path}")
    if str(payload["config"]["sha256"]) != str(config_hash):
        raise RuntimeError(f"Config hash mismatch: {path}")
    if str(payload["catalog"]["sha256"]) != str(catalog_hash):
        raise RuntimeError(f"Catalog hash mismatch: {path}")
    if Path(str(payload["catalog"]["path"])).resolve() != catalog_path.resolve():
        raise RuntimeError(f"Catalog path mismatch: {path}")
    if not payload["catalog"]["cell_audit"].get("passed"):
        raise RuntimeError(f"Catalog cell audit failed: {path}")
    if str(payload["normalizer"]["sha256"]) != str(normalizer_hash):
        raise RuntimeError(f"Normalizer hash mismatch: {path}")
    if str(payload["stable_worldmodel"]["commit"]) != str(stable_commit):
        raise RuntimeError(f"StableWorldModel commit mismatch: {path}")
    if not payload["frozen_weight_audit"].get("passed"):
        raise RuntimeError(f"Frozen-weight audit failed: {path}")
    if (
        payload["frozen_weight_audit"].get("state_dict_sha256_before")
        != payload["frozen_weight_audit"].get("state_dict_sha256_after")
    ):
        raise RuntimeError(f"Model state changed while scoring: {path}")
    if not payload["count_audit"].get("passed"):
        raise RuntimeError(f"Count audit failed: {path}")
    if int(payload["count_audit"]["records"]) != per_seed:
        raise RuntimeError(f"Result does not contain exactly {per_seed} rows: {path}")
    if int(payload["protocol"]["queries"]) != per_seed:
        raise RuntimeError(f"Protocol query count mismatch: {path}")
    checkpoint_hash = str(payload["model"]["sha256"])
    if checkpoint_hash not in identities_by_hash:
        raise RuntimeError(f"Checkpoint is not bound to a training report: {path}")
    identity = dict(identities_by_hash[checkpoint_hash])
    track = str(payload["track"])
    door = int(payload["door_position"])
    seed = int(payload["eval_seed"])
    if track not in tracks or door not in tracks[track] or seed not in eval_seeds:
        raise RuntimeError(f"Unexpected result cell {(track, door, seed)}: {path}")
    protocol = payload["protocol"]
    common_checks = {
        "history": int(protocol["history_size"]) == 3,
        "action_block": int(protocol["action_block"]) == 5,
        "agent_speed": float(protocol["agent_speed"]) == 5.0,
    }
    if mode == "fixed":
        common_checks.update(
            {
                "candidates": int(protocol["candidates_per_query"]) == 300,
                "horizon": int(protocol["horizon_action_blocks"]) == 10,
                "raw_horizon": int(protocol["horizon_raw_steps"]) == 50,
                "frozen_bank": bool(
                    protocol["same_frozen_candidate_bank_across_models"]
                ),
            }
        )
    else:
        common_checks.update(
            {
                "budget": int(protocol["eval_budget_raw_steps"]) == 100,
                "horizon": int(protocol["horizon_action_blocks"]) == 10,
                "receding": int(protocol["receding_horizon_action_blocks"]) == 5,
                "samples": int(protocol["cem_samples"]) == 300,
                "iterations": int(protocol["cem_iterations"]) == 30,
                "topk": int(protocol["cem_topk"]) == 30,
                "paired_seed": bool(
                    protocol["same_query_and_initial_cem_seed_across_models"]
                ),
                "rolling": bool(
                    protocol["rolling_causally_aligned_natural_history3"]
                ),
            }
        )
        if not payload["rolling_history3_audit"].get("passed"):
            raise RuntimeError(f"Rolling History-3 audit failed: {path}")
    if not all(common_checks.values()):
        raise RuntimeError(f"Frozen planning protocol mismatch: {common_checks}")
    return {
        "identity": identity,
        "track": track,
        "door_position": door,
        "eval_seed": seed,
        "checkpoint_sha256": checkpoint_hash,
    }


def _validate_record(
    row: Mapping[str, Any],
    *,
    mode: str,
    track: str,
    door: int,
    seed: int,
    catalog_row: Mapping[str, Any],
) -> None:
    index = int(row["evaluation_index"])
    query_id = str(catalog_row["query_id"])
    expected_evaluation_id = f"s{seed}-e{index:03d}-{query_id}"
    common = {
        "track": str(row["track"]) == track,
        "door": int(row["door_position"]) == door,
        "seed": int(row["eval_seed"]) == seed,
        "query": str(row["query_id"]) == query_id,
        "evaluation_id": str(row["evaluation_id"]) == expected_evaluation_id,
        "task": str(row["task"]) == "cross_room_navigation",
        "speed": float(row.get("agent_speed", row.get("speed"))) == 5.0,
        "direction": str(row["direction"]) == str(catalog_row["direction"]),
        "offset": int(row["door_relative_vertical_offset_px"])
        == int(catalog_row["door_relative_vertical_offset_px"]),
        "history_pixels": str(row["history_pixels_sha256"])
        == str(catalog_row["history_pixels_sha256"]),
        "history_actions": str(row["history_actions_sha256"])
        == str(catalog_row["history_actions_sha256"]),
    }
    if not all(common.values()):
        raise RuntimeError(
            f"Record/catalog mismatch at {(track, door, seed, index)}: {common}"
        )
    final_distance = float(row["final_distance_px"])
    if not np.isfinite(final_distance) or final_distance < 0.0:
        raise RuntimeError(f"Invalid final distance at {query_id}")
    success = bool(row["success"])
    steps = row["steps_to_success"]
    if success != (steps is not None):
        raise RuntimeError(f"Success/steps mismatch at {query_id}")
    if steps is not None and int(steps) <= 0:
        raise RuntimeError(f"Invalid successful step count at {query_id}")
    if mode == "fixed":
        regret = float(row["exact_environment_endpoint_regret_px"])
        correlation = float(
            row["predicted_cost_vs_true_endpoint_distance_spearman"]
        )
        if not np.isfinite(regret) or regret < -1e-6:
            raise RuntimeError(f"Invalid endpoint regret at {query_id}")
        if not np.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
            raise RuntimeError(f"Invalid Spearman value at {query_id}")
        if str(row["candidate_bank_raw_sha256"]) != str(
            catalog_row["fixed_candidate_raw_actions_sha256"]
        ):
            raise RuntimeError(f"Candidate bank/catalog mismatch at {query_id}")
    else:
        if str(row["condition"]) != "natural_history3":
            raise RuntimeError(f"Unexpected CEM input condition at {query_id}")
        if int(row["cem_seed"]) != int(catalog_row["cem_seed"]):
            raise RuntimeError(f"CEM seed/catalog mismatch at {query_id}")
        context = row["fixed_context"]
        if str(context["pixels_sha256"]) != str(
            catalog_row["history_pixels_sha256"]
        ) or str(context["raw_actions_sha256"]) != str(
            catalog_row["history_actions_sha256"]
        ):
            raise RuntimeError(f"CEM context/catalog mismatch at {query_id}")
        if row["rolling_history3_audit"] is None or not row[
            "rolling_history3_audit"
        ].get("passed"):
            raise RuntimeError(f"Per-query rolling audit failed at {query_id}")


def _load_results(
    paths: Iterable[Path],
    *,
    mode: str,
    config_hash: str,
    catalog_path: Path,
    catalog_hash: str,
    catalog_index: Mapping[tuple[str, int, int, int], Mapping[str, Any]],
    normalizer_hash: str,
    stable_commit: str,
    tracks: Mapping[str, tuple[int, ...]],
    eval_seeds: tuple[int, ...],
    per_seed: int,
    identities_by_hash: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    observed_cells: set[tuple[str, str, int, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        common = _assert_common_result(
            payload,
            path=path,
            mode=mode,
            config_hash=config_hash,
            catalog_path=catalog_path,
            catalog_hash=catalog_hash,
            normalizer_hash=normalizer_hash,
            stable_commit=stable_commit,
            tracks=tracks,
            eval_seeds=eval_seeds,
            per_seed=per_seed,
            identities_by_hash=identities_by_hash,
        )
        identity = common["identity"]
        track = common["track"]
        door = common["door_position"]
        seed = common["eval_seed"]
        cell_key = (str(identity["slug"]), track, door, seed)
        if cell_key in observed_cells:
            raise RuntimeError(f"Repeated model/result cell: {cell_key}")
        observed_cells.add(cell_key)
        rows = list(payload.get("records", []))
        if len(rows) != per_seed:
            raise RuntimeError(f"Result records are not exactly {per_seed}: {path}")
        indices = [int(row["evaluation_index"]) for row in rows]
        if set(indices) != set(range(per_seed)) or len(indices) != len(set(indices)):
            raise RuntimeError(f"Evaluation indices are incomplete: {path}")
        query_ids = [str(row["query_id"]) for row in rows]
        if len(query_ids) != len(set(query_ids)):
            raise RuntimeError(f"Repeated query_id inside a result cell: {path}")
        directions = [str(row["direction"]) for row in rows]
        if directions.count("left_to_right") != 25 or directions.count(
            "right_to_left"
        ) != 25:
            raise RuntimeError(f"Direction balance is not 25+25: {path}")
        for row in rows:
            index = int(row["evaluation_index"])
            catalog_key = (track, door, seed, index)
            if catalog_key not in catalog_index:
                raise RuntimeError(f"Result query missing from catalog: {catalog_key}")
            _validate_record(
                row,
                mode=mode,
                track=track,
                door=door,
                seed=seed,
                catalog_row=catalog_index[catalog_key],
            )
        loaded.append(
            {
                "path": str(path.resolve()),
                "path_sha256": file_sha256(path),
                "payload": payload,
                "model": identity,
                "track": track,
                "door_position": door,
                "eval_seed": seed,
                "records": rows,
            }
        )
    return loaded


def _audit_matrix(
    results: Iterable[Mapping[str, Any]],
    *,
    tracks: Mapping[str, tuple[int, ...]],
    eval_seeds: tuple[int, ...],
    require_complete: bool,
) -> dict[str, Any]:
    rows = list(results)
    expected_models = {
        binding["run_name"] for binding in TRAINING_BINDINGS.values()
    }
    expected_cells = {
        (model, track, door, seed)
        for model in expected_models
        for track, doors in tracks.items()
        for door in doors
        for seed in eval_seeds
    }
    observed_cells = {
        (
            str(row["model"]["slug"]),
            str(row["track"]),
            int(row["door_position"]),
            int(row["eval_seed"]),
        )
        for row in rows
    }
    observed_models = {str(row["model"]["slug"]) for row in rows}
    unexpected = observed_cells - expected_cells
    if unexpected:
        raise RuntimeError(f"Unexpected planning result cells: {sorted(unexpected)}")
    complete = observed_cells == expected_cells and observed_models == expected_models
    if require_complete and not complete:
        missing = expected_cells - observed_cells
        raise RuntimeError(
            "Planning result matrix is incomplete: "
            f"models={sorted(observed_models)}, missing_cells={len(missing)}"
        )
    return {
        "passed": complete if require_complete else True,
        "complete_formal_matrix": complete,
        "expected_models": 7,
        "observed_models": len(observed_models),
        "expected_result_files": len(expected_cells),
        "observed_result_files": len(observed_cells),
        "expected_records_per_file": 50,
        "observed_records": sum(len(row["records"]) for row in rows),
        "missing_result_files": len(expected_cells - observed_cells),
    }


def _record_pairing_signature(
    row: Mapping[str, Any], *, mode: str
) -> tuple[Any, ...]:
    common = (
        str(row["query_id"]),
        str(row["evaluation_id"]),
        str(row["direction"]),
        int(row["door_relative_vertical_offset_px"]),
        str(row["history_pixels_sha256"]),
        str(row["history_actions_sha256"]),
    )
    if mode == "fixed":
        return common + (
            str(row["candidate_bank_raw_sha256"]),
            str(row["candidate_bank_normalized_sha256"]),
        )
    context = row["fixed_context"]
    return common + (
        int(row["cem_seed"]),
        str(row["cem_rng_state_sha256_before"]),
        str(context["normalized_actions_sha256"]),
    )


def _audit_cross_model_pairing(
    results: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    require_complete: bool,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int, int], list[tuple[str, tuple[Any, ...]]]] = (
        defaultdict(list)
    )
    for result in results:
        for row in result["records"]:
            key = (
                str(result["track"]),
                int(result["door_position"]),
                int(result["eval_seed"]),
                int(row["evaluation_index"]),
            )
            grouped[key].append(
                (
                    str(result["model"]["slug"]),
                    _record_pairing_signature(row, mode=mode),
                )
            )
    failures = []
    for key, rows in grouped.items():
        signatures = {signature for _, signature in rows}
        models = {model for model, _ in rows}
        if len(signatures) != 1 or (require_complete and len(models) != 7):
            failures.append(
                {"query_key": list(key), "models": len(models), "signatures": len(signatures)}
            )
    if failures:
        raise RuntimeError(f"Cross-model query pairing failed: {failures[:3]}")
    return {
        "passed": True,
        "paired_queries": len(grouped),
        "models_per_query": 7 if require_complete else None,
        "query_identity_and_history_paired": True,
        "fixed_candidate_bank_paired": mode == "fixed",
        "cem_seed_and_initial_rng_state_paired": mode == "planning",
    }


def _continuous_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("A continuous metric needs finite observations")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _aggregate_model_records(
    records: Iterable[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot summarize an empty model/track/door cell")
    success = np.asarray([bool(row["success"]) for row in rows])
    crossing = np.asarray([bool(row["doorway_crossing"]) for row in rows])
    distances = [float(row["final_distance_px"]) for row in rows]
    successful_steps = [
        int(row["steps_to_success"])
        for row in rows
        if row["steps_to_success"] is not None
    ]
    summary: dict[str, Any] = {
        "evaluations": len(rows),
        "successes": int(success.sum()),
        "success_rate_percent": float(100.0 * success.mean()),
        "doorway_crossings": int(crossing.sum()),
        "doorway_crossing_rate_percent": float(100.0 * crossing.mean()),
        "final_distance_px": _continuous_summary(distances),
        "steps_to_success_success_only": (
            _continuous_summary(successful_steps) if successful_steps else None
        ),
        "by_eval_seed": {},
    }
    if mode == "fixed":
        summary.update(
            {
                "endpoint_regret_px_lower_is_better": _continuous_summary(
                    float(row["exact_environment_endpoint_regret_px"])
                    for row in rows
                ),
                "tie_aware_spearman_higher_is_better": _continuous_summary(
                    float(
                        row[
                            "predicted_cost_vs_true_endpoint_distance_spearman"
                        ]
                    )
                    for row in rows
                ),
            }
        )
    for seed in sorted({int(row["eval_seed"]) for row in rows}):
        selected = [row for row in rows if int(row["eval_seed"]) == seed]
        summary["by_eval_seed"][str(seed)] = {
            "evaluations": len(selected),
            "success_rate_percent": float(
                100.0 * np.mean([bool(row["success"]) for row in selected])
            ),
            "doorway_crossing_rate_percent": float(
                100.0
                * np.mean([bool(row["doorway_crossing"]) for row in selected])
            ),
            "mean_final_distance_px": float(
                np.mean([float(row["final_distance_px"]) for row in selected])
            ),
        }
    return summary


def _by_track_door_model(
    results: Iterable[Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    identities: dict[str, Mapping[str, Any]] = {}
    for result in results:
        slug = str(result["model"]["slug"])
        identities[slug] = result["model"]
        grouped[(str(result["track"]), int(result["door_position"]), slug)].extend(
            result["records"]
        )
    output: dict[str, Any] = {}
    for (track, door, slug), rows in sorted(grouped.items()):
        door_row = output.setdefault(track, {"doors": {}})["doors"].setdefault(
            str(door), {"models": {}}
        )
        door_row["models"][slug] = {
            "model": dict(identities[slug]),
            "metrics": _aggregate_model_records(rows, mode=mode),
        }
    return output


def _bootstrap_delta_summary(
    target: np.ndarray,
    control: np.ndarray,
    *,
    indices: np.ndarray,
    higher_is_better: bool,
    percent: bool = False,
    tie_atol: float = 1e-12,
) -> dict[str, Any]:
    target_values = np.asarray(target, dtype=np.float64)
    control_values = np.asarray(control, dtype=np.float64)
    if target_values.shape != control_values.shape or target_values.size == 0:
        raise ValueError("Paired metrics must have equal non-zero shapes")
    difference = target_values - control_values
    scale = 100.0 if percent else 1.0
    bootstrap = np.mean(difference[indices], axis=1) * scale
    better = difference > tie_atol if higher_is_better else difference < -tie_atol
    worse = difference < -tie_atol if higher_is_better else difference > tie_atol
    ties = ~(better | worse)
    return {
        "pairs": int(difference.size),
        "direction": "higher_is_better" if higher_is_better else "lower_is_better",
        "target": _continuous_summary(target_values * scale),
        "fixed_door_control": _continuous_summary(control_values * scale),
        "target_minus_fixed_door_control_mean": float(np.mean(difference) * scale),
        "target_minus_fixed_door_control_bootstrap_95_ci": [
            float(value) for value in np.percentile(bootstrap, [2.5, 97.5])
        ],
        "target_better_pairs": int(better.sum()),
        "fixed_door_control_better_pairs": int(worse.sum()),
        "ties": int(ties.sum()),
    }


def _paired_cell(
    target_rows: Mapping[tuple[int, int, str], Mapping[str, Any]],
    control_rows: Mapping[tuple[int, int, str], Mapping[str, Any]],
    *,
    mode: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if set(target_rows) != set(control_rows):
        raise RuntimeError("Target/control query keys differ")
    keys = sorted(target_rows)
    target = [target_rows[key] for key in keys]
    control = [control_rows[key] for key in keys]
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, len(keys), size=(int(bootstrap_samples), len(keys)))
    metrics: dict[str, Any] = {
        "success_rate_percentage_points": _bootstrap_delta_summary(
            np.asarray([bool(row["success"]) for row in target], dtype=float),
            np.asarray([bool(row["success"]) for row in control], dtype=float),
            indices=indices,
            higher_is_better=True,
            percent=True,
        ),
        "doorway_crossing_rate_percentage_points": _bootstrap_delta_summary(
            np.asarray([bool(row["doorway_crossing"]) for row in target], dtype=float),
            np.asarray([bool(row["doorway_crossing"]) for row in control], dtype=float),
            indices=indices,
            higher_is_better=True,
            percent=True,
        ),
        "final_distance_px": _bootstrap_delta_summary(
            np.asarray([float(row["final_distance_px"]) for row in target]),
            np.asarray([float(row["final_distance_px"]) for row in control]),
            indices=indices,
            higher_is_better=False,
        ),
    }
    if mode == "fixed":
        metrics.update(
            {
                "endpoint_regret_px": _bootstrap_delta_summary(
                    np.asarray(
                        [
                            float(row["exact_environment_endpoint_regret_px"])
                            for row in target
                        ]
                    ),
                    np.asarray(
                        [
                            float(row["exact_environment_endpoint_regret_px"])
                            for row in control
                        ]
                    ),
                    indices=indices,
                    higher_is_better=False,
                ),
                "tie_aware_spearman": _bootstrap_delta_summary(
                    np.asarray(
                        [
                            float(
                                row[
                                    "predicted_cost_vs_true_endpoint_distance_spearman"
                                ]
                            )
                            for row in target
                        ]
                    ),
                    np.asarray(
                        [
                            float(
                                row[
                                    "predicted_cost_vs_true_endpoint_distance_spearman"
                                ]
                            )
                            for row in control
                        ]
                    ),
                    indices=indices,
                    higher_is_better=True,
                ),
            }
        )
    else:
        common_success = [
            index
            for index, (target_row, control_row) in enumerate(zip(target, control, strict=True))
            if bool(target_row["success"]) and bool(control_row["success"])
        ]
        if common_success:
            success_indices = rng.integers(
                0,
                len(common_success),
                size=(int(bootstrap_samples), len(common_success)),
            )
            metrics["steps_to_success_common_success_queries"] = (
                _bootstrap_delta_summary(
                    np.asarray(
                        [int(target[index]["steps_to_success"]) for index in common_success]
                    ),
                    np.asarray(
                        [int(control[index]["steps_to_success"]) for index in common_success]
                    ),
                    indices=success_indices,
                    higher_is_better=False,
                )
            )
        else:
            metrics["steps_to_success_common_success_queries"] = {
                "pairs": 0,
                "available": False,
                "reason": "no_queries_succeeded_for_both_models",
            }
    return {
        "query_pairs": len(keys),
        "bootstrap_unit": "paired_query",
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "metrics": metrics,
    }


def _paired_target_control(
    results: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    tracks: Mapping[str, tuple[int, ...]],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    lookup: dict[
        tuple[str, int, int, str],
        dict[tuple[int, int, str], Mapping[str, Any]],
    ] = defaultdict(dict)
    identity: dict[tuple[str, int], str] = {}
    for result in results:
        group = str(result["model"]["group"])
        seed = int(result["model"]["training_seed"])
        identity[(group, seed)] = str(result["model"]["slug"])
        key = (str(result["track"]), int(result["door_position"]), seed, group)
        for row in result["records"]:
            query_key = (
                int(row["eval_seed"]),
                int(row["evaluation_index"]),
                str(row["query_id"]),
            )
            if query_key in lookup[key]:
                raise RuntimeError(f"Repeated paired-query key: {key}/{query_key}")
            lookup[key][query_key] = row
    output: dict[str, Any] = {}
    cell_index = 0
    for training_seed in PAIRED_TRAINING_SEEDS:
        control_slug = identity.get(("fixed_door_control", training_seed))
        target_slug = identity.get(("multi_door_target", training_seed))
        if control_slug is None or target_slug is None:
            continue
        seed_output = {
            "training_seed": int(training_seed),
            "target_model": target_slug,
            "fixed_door_control_model": control_slug,
            "by_track": {},
        }
        for track, doors in tracks.items():
            track_output = seed_output["by_track"].setdefault(track, {"doors": {}})
            for door in doors:
                control = lookup.get(
                    (track, door, training_seed, "fixed_door_control")
                )
                target = lookup.get(
                    (track, door, training_seed, "multi_door_target")
                )
                if not control or not target:
                    continue
                track_output["doors"][str(door)] = _paired_cell(
                    target,
                    control,
                    mode=mode,
                    bootstrap_seed=int(bootstrap_seed + 1009 * cell_index),
                    bootstrap_samples=bootstrap_samples,
                )
                cell_index += 1
        output[str(training_seed)] = seed_output
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in MODES:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("benchmark") != "tworoom_door_visual_generalization_v1":
        raise ValueError(f"Unexpected door benchmark config: {config_path}")
    config_hash = file_sha256(config_path)
    tracks = _expected_tracks(config, args.split)
    eval_seeds = tuple(map(int, config["evaluation_data"]["eval_seeds"]))
    if len(eval_seeds) != 6 or len(set(eval_seeds)) != 6:
        raise RuntimeError("Door planning requires exactly six unique eval seeds")
    protocol_name = (
        "fixed_candidate_evaluation" if args.mode == "fixed" else "closed_loop_planning"
    )
    per_seed = int(config[protocol_name]["evaluations_per_door_per_seed"])
    if per_seed != 50:
        raise RuntimeError("Door planning requires an independent 50-query cell")
    require_formal = not args.allow_partial

    report_paths = _training_report_paths(args)
    identities_by_hash, report_by_key, report_stable_commit = (
        _load_training_identities(report_paths, require_complete=require_formal)
    )
    expected_normalizer = resolve_contextworld_path(
        args.expected_normalizer, repo_root=ROOT
    )
    if not expected_normalizer.is_file():
        raise FileNotFoundError(expected_normalizer)
    normalizer_hash = file_sha256(expected_normalizer)
    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else _split_root(config, args.split)
    )
    benchmark_root = _split_root(config, "validation")
    catalog_path = (
        args.catalog.resolve()
        if args.catalog is not None
        else benchmark_root / "planning" / args.split / "catalog.json"
    )
    catalog, catalog_index = _load_catalog(
        catalog_path,
        config_hash=config_hash,
        split=args.split,
        stable_commit=report_stable_commit,
        tracks=tracks,
        eval_seeds=eval_seeds,
        per_seed=per_seed,
        require_formal=require_formal,
    )
    catalog_hash = file_sha256(catalog_path)
    paths = _resolve_result_paths(
        args.results, artifact_root=artifact_root, mode=args.mode
    )
    results = _load_results(
        paths,
        mode=args.mode,
        config_hash=config_hash,
        catalog_path=catalog_path,
        catalog_hash=catalog_hash,
        catalog_index=catalog_index,
        normalizer_hash=normalizer_hash,
        stable_commit=report_stable_commit,
        tracks=tracks,
        eval_seeds=eval_seeds,
        per_seed=per_seed,
        identities_by_hash=identities_by_hash,
    )
    matrix = _audit_matrix(
        results,
        tracks=tracks,
        eval_seeds=eval_seeds,
        require_complete=require_formal,
    )
    pairing = _audit_cross_model_pairing(
        results, mode=args.mode, require_complete=require_formal
    )

    representative: dict[str, dict[str, Any]] = {}
    for result in results:
        model = result["model"]
        slug = str(model["slug"])
        representative.setdefault(
            slug,
            {
                "model": dict(model),
                "normalizer": {"sha256": normalizer_hash},
                "stable_worldmodel": {"commit": report_stable_commit},
            },
        )
    runtime_identity = _audit_runtime_identity(
        representative, expected_normalizer_sha256=normalizer_hash
    )
    selected_report_paths = [
        report_by_key[
            (str(row["model"]["group"]), int(row["model"]["training_seed"]))
        ]
        for row in representative.values()
    ]
    training_bindings = _audit_training_report_bindings(
        representative,
        report_paths=selected_report_paths,
        stable_worldmodel_commit=report_stable_commit,
        require_complete=require_formal,
    )
    formal = bool(
        require_formal
        and matrix["complete_formal_matrix"]
        and training_bindings["complete_formal_binding"]
    )
    by_track = _by_track_door_model(results, mode=args.mode)
    paired = _paired_target_control(
        results,
        mode=args.mode,
        tracks=tracks,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "analysis": f"{args.mode}_planning_supporting_evidence",
        "evaluation_split": args.split,
        "status": "passed" if formal else "partial_analysis_only",
        "formal_analysis": formal,
        "config": {"path": str(config_path), "sha256": config_hash},
        "catalog": {
            "path": str(catalog_path.resolve()),
            "sha256": catalog_hash,
            "status": catalog.get("status"),
        },
        "expected_normalizer": {
            "path": str(expected_normalizer),
            "sha256": normalizer_hash,
        },
        "input_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "model_matrix_audit": matrix,
        "query_pairing_audit": pairing,
        "runtime_identity_audit": runtime_identity,
        "training_report_binding_audit": training_bindings,
        "metric_contract": {
            "mode": args.mode,
            "tracks_never_pooled": True,
            "doors_never_pooled_for_target_control_inference": True,
            "original_reference_role": "descriptive_only",
            "paired_comparison": (
                "multi_door_target_minus_paired_fixed_door_control_by_training_seed"
            ),
            "bootstrap_unit": "paired_query",
            "bootstrap_samples": int(args.bootstrap_samples),
            "fixed_tie_aware_spearman": args.mode == "fixed",
            "planning_is_supporting_evidence_not_prediction_accuracy": True,
            "planning_alone_cannot_establish_better_prediction": True,
        },
        "by_track_door_model": by_track,
        "paired_target_vs_fixed_door_control": paired,
        "interpretation": {
            "integrity_passed": formal,
            "scientific_effect_direction_preregistered_as_gate": False,
            "standalone_prediction_claim_allowed": False,
            "joint_prediction_to_planning_claim_requires_true_future_prediction_gate": True,
            "partial_results_are_never_formal": True,
        },
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else artifact_root / f"{args.mode}_final_summary.json"
    )
    write_json(output, payload)
    return {**payload, "output": str(output)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize TwoRoom visible-door planning results."
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--split", choices=("validation", "sealed_test"), default="validation"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--results", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-normalizer", default=DEFAULT_NORMALIZER)
    parser.add_argument(
        "--training-report-root", default="artifacts/training/reports"
    )
    parser.add_argument("--training-reports", nargs="*", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072242)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "analysis": result["analysis"],
                "split": result["evaluation_split"],
                "output": result["output"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
