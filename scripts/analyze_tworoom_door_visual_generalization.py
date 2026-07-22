#!/usr/bin/env python3
"""Aggregate paired visual-door true-future latent evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.door_visual import (
    HORIZONS,
    TASKS,
    VALIDATION_TRACKS,
    longest_contiguous_horizon,
    paired_normalized_effects,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.sealed_test_gate import (
    canonical_door_split_root,
    require_canonical_split_path,
    require_path_within_split_root,
    require_sealed_test_gate,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


PAIRED_TRAINING_SEEDS = (3072, 4096, 5120)
DEFAULT_NORMALIZER = "artifacts/splits/tworoom_original_train_s3072_normalizer.json"
TRAINING_BINDINGS = {
    ("original_reference", 3072): {
        "model_id": "M_origheldout",
        "run_name": "h3_origheldout_s3072",
        "report_name": "h3_origheldout_s3072.json",
    },
    **{
        ("fixed_door_control", seed): {
            "model_id": "M_door_fixed49_v2",
            "run_name": f"h3_door_fixed49_v2_s{seed}",
            "report_name": f"h3_door_fixed49_v2_s{seed}.json",
        }
        for seed in PAIRED_TRAINING_SEEDS
    },
    **{
        ("multi_door_target", seed): {
            "model_id": "M_door_multi_v2",
            "run_name": f"h3_door_multi_v2_s{seed}",
            "report_name": f"h3_door_multi_v2_s{seed}.json",
        }
        for seed in PAIRED_TRAINING_SEEDS
    },
}


def _flatten_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for track in result["tracks"].values()
        for row in track["records"]
    ]


def _load_results(
    paths: list[Path],
    *,
    config_hash: str,
    build_report_hash: str,
    evaluation_split: str,
    stable_worldmodel_commit: str,
    require_formal: bool,
) -> dict[str, dict[str, Any]]:
    results = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["config"]["sha256"] != config_hash:
            raise RuntimeError(f"Config hash mismatch: {path}")
        if payload["build_report"]["sha256"] != build_report_hash:
            raise RuntimeError(f"Build report hash mismatch: {path}")
        if str(payload.get("evaluation_split")) != str(evaluation_split):
            raise RuntimeError(f"Evaluation split mismatch: {path}")
        if require_formal and payload.get("status") != "passed":
            raise RuntimeError(f"Formal analysis refuses a smoke result: {path}")
        if (
            str(payload["stable_worldmodel"]["commit"])
            != str(stable_worldmodel_commit)
        ):
            raise RuntimeError(f"StableWorldModel commit mismatch: {path}")
        if payload["online_environment_calls"] != 0:
            raise RuntimeError(f"Online environment used during scoring: {path}")
        if not payload["frozen_weight_audit"]["passed"]:
            raise RuntimeError(f"Frozen weight audit failed: {path}")
        if not payload["count_audit"]["passed"]:
            raise RuntimeError(f"Count audit failed: {path}")
        if payload["protocol"]["raw_latent_mse_cross_checkpoint_ranking"]:
            raise RuntimeError(f"Forbidden raw-MSE ranking enabled: {path}")
        for track_name, track in payload["tracks"].items():
            for condition, audit in track[
                "autoregressive_prefix_audit"
            ].items():
                if not audit["passed"]:
                    raise RuntimeError(
                        f"Prefix audit failed: {path}/{track_name}/{condition}"
                    )
        slug = str(payload["model"]["slug"])
        if slug in results:
            raise RuntimeError(f"Repeated model slug: {slug}")
        results[slug] = {**payload, "result_path": str(path)}
    return results


def _audit_model_matrix(
    results: dict[str, dict[str, Any]], *, allow_partial: bool
) -> dict[str, Any]:
    observed: dict[str, set[int]] = defaultdict(set)
    observed_keys = []
    for result in results.values():
        key = (
            str(result["model"]["group"]),
            int(result["model"]["training_seed"]),
        )
        observed_keys.append(key)
        observed[key[0]].add(key[1])
    expected = {
        "original_reference": {3072},
        "fixed_door_control": set(PAIRED_TRAINING_SEEDS),
        "multi_door_target": set(PAIRED_TRAINING_SEEDS),
    }
    duplicate_keys = sorted(
        {key for key in observed_keys if observed_keys.count(key) > 1}
    )
    exact = (
        {group: observed.get(group, set()) for group in expected} == expected
        and not duplicate_keys
        and len(results) == len(TRAINING_BINDINGS)
    )
    if not allow_partial and not exact:
        raise RuntimeError(
            f"Expected complete 1+3+3 model matrix, observed={dict(observed)}"
        )
    return {
        "passed": exact if not allow_partial else True,
        "complete_formal_matrix": exact,
        "observed_training_seeds_by_group": {
            group: sorted(values) for group, values in observed.items()
        },
        "expected_training_seeds_by_group": {
            group: sorted(values) for group, values in expected.items()
        },
        "duplicate_group_seed_labels": [list(key) for key in duplicate_keys],
    }


def _audit_runtime_identity(
    results: dict[str, dict[str, Any]], *, expected_normalizer_sha256: str
) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No model results were provided for analysis")
    normalizer_hashes = {
        str(row["normalizer"]["sha256"]) for row in results.values()
    }
    if normalizer_hashes != {str(expected_normalizer_sha256)}:
        raise RuntimeError(
            "All formal models must use the frozen original-train normalizer: "
            f"observed={sorted(normalizer_hashes)}"
        )
    checkpoint_hashes = [
        str(row["model"]["checkpoint_sha256"]) for row in results.values()
    ]
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise RuntimeError("Two model labels point to the same checkpoint hash")
    stable_commits = {
        str(row["stable_worldmodel"]["commit"]) for row in results.values()
    }
    if len(stable_commits) != 1:
        raise RuntimeError(
            f"Model results use different StableWorldModel commits: {stable_commits}"
        )
    return {
        "passed": True,
        "normalizer_sha256": str(expected_normalizer_sha256),
        "stable_worldmodel_commit": next(iter(stable_commits)),
        "unique_checkpoint_hashes": len(set(checkpoint_hashes)),
        "models": len(results),
    }


def _audit_training_report_bindings(
    results: dict[str, dict[str, Any]],
    *,
    report_paths: list[Path],
    stable_worldmodel_commit: str,
    expected_data_split_seed: int,
    require_complete: bool,
) -> dict[str, Any]:
    reports: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
    model_id_to_group = {
        row["model_id"]: group for (group, _), row in TRAINING_BINDINGS.items()
    }
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_id = str(payload.get("model_id"))
        if model_id not in model_id_to_group:
            raise RuntimeError(f"Unknown door training-report model_id in {path}")
        group = model_id_to_group[model_id]
        training_seed = int(payload["training"]["plan"]["training_seed"])
        key = (group, training_seed)
        if key not in TRAINING_BINDINGS:
            raise RuntimeError(f"Unexpected training-report binding {key}: {path}")
        if key in reports:
            raise RuntimeError(f"Repeated training report for {key}")
        reports[key] = (path, payload)

    result_by_key = {
        (
            str(row["model"]["group"]),
            int(row["model"]["training_seed"]),
        ): row
        for row in results.values()
    }
    required_keys = set(TRAINING_BINDINGS) if require_complete else set(result_by_key)
    if set(reports) != required_keys:
        raise RuntimeError(
            "Training-report matrix does not match the required model matrix: "
            f"required={sorted(required_keys)}, observed={sorted(reports)}"
        )
    if not required_keys <= set(result_by_key):
        raise RuntimeError("A required training report has no corresponding result")

    bindings = {}
    for key in sorted(required_keys):
        path, report = reports[key]
        expected = TRAINING_BINDINGS[key]
        result = result_by_key[key]
        training_plan = report["training"]["plan"]
        checks = {
            "training_report_passed": report.get("passed") is True,
            "training_complete": report["training"].get("training_complete") is True,
            "save_load_exact": report.get("save_load_exact") is True,
            "model_id": str(report.get("model_id")) == expected["model_id"],
            "run_name": str(report.get("run_name")) == expected["run_name"],
            "data_seed": int(report["data"]["seed"])
            == expected_data_split_seed,
            "plan_data_split_seed": int(training_plan["data_split_seed"])
            == expected_data_split_seed,
            "training_seed": int(training_plan["training_seed"]) == key[1],
            "stable_worldmodel_commit": str(
                report["stable_worldmodel"]["commit"]
            )
            == str(stable_worldmodel_commit),
            "checkpoint_sha256": str(
                report["artifacts"]["pretrained_sha256"]
            )
            == str(result["model"]["checkpoint_sha256"]),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Training-report binding failed for {key}: {checks}")
        bindings[f"{key[0]}/s{key[1]}"] = {
            "training_report": str(path),
            "training_report_sha256": file_sha256(path),
            "checkpoint_sha256": str(report["artifacts"]["pretrained_sha256"]),
            "checks": checks,
        }
    return {
        "passed": True,
        "complete_formal_binding": set(reports) == set(TRAINING_BINDINGS),
        "bindings": bindings,
    }


def _find_by_group_seed(
    results: dict[str, dict[str, Any]], group: str, seed: int
) -> dict[str, Any] | None:
    selected = [
        row
        for row in results.values()
        if row["model"]["group"] == group
        and int(row["model"]["training_seed"]) == int(seed)
    ]
    if len(selected) > 1:
        raise RuntimeError(f"Repeated group/training seed: {group}/{seed}")
    return selected[0] if selected else None


def _paired_seed_summary(
    *,
    seed: int,
    control: dict[str, Any],
    target: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired = paired_normalized_effects(
        _flatten_records(control),
        _flatten_records(target),
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for cell in paired["cells"].values():
        grouped[
            (
                str(cell["track"]),
                str(cell["task"]),
                str(cell["input_condition"]),
                int(cell["horizon"]),
            )
        ].append(cell)
    summaries = {}
    for key, rows in sorted(grouped.items()):
        track, task, condition, horizon = key
        cell_id = f"{track}/{task}/{condition}/h{horizon}"
        summaries[cell_id] = {
            "track": track,
            "task": task,
            "input_condition": condition,
            "horizon": horizon,
            "door_positions": sorted(int(row["door_position"]) for row in rows),
            "door_balanced_mean_control_minus_target": float(
                np.mean([row["mean_control_minus_target"] for row in rows])
            ),
            "door_balanced_paired_query_win_rate": float(
                np.mean([row["paired_query_win_rate"] for row in rows])
            ),
            "every_door_all_eval_seed_directions_positive": all(
                row["all_eval_seed_directions_positive"] for row in rows
            ),
            "by_door_position": {
                str(row["door_position"]): row for row in rows
            },
        }
    return {
        "training_seed": int(seed),
        "control_model": control["model"]["slug"],
        "target_model": target["model"]["slug"],
        "query_pairs": len(paired["effects"]) // len(HORIZONS),
        "horizon_effects": len(paired["effects"]),
        "summaries": summaries,
    }


def _formal_decision(
    paired_by_seed: dict[str, dict[str, Any]],
    *,
    complete_matrix: bool,
    tracks: tuple[str, ...] = VALIDATION_TRACKS,
    evaluation_split: str = "validation",
) -> dict[str, Any]:
    by_track_horizon = {
        track: {str(horizon): False for horizon in HORIZONS}
        for track in tracks
    }
    details = {}
    if complete_matrix:
        for track in tracks:
            for horizon in HORIZONS:
                task_passes = {}
                for task in TASKS:
                    cell_id = f"{track}/{task}/natural_history3/h{horizon}"
                    by_seed = {
                        seed: bool(
                            paired_by_seed[str(seed)]["summaries"][cell_id][
                                "every_door_all_eval_seed_directions_positive"
                            ]
                        )
                        for seed in PAIRED_TRAINING_SEEDS
                    }
                    task_passes[task] = {
                        "passed": all(by_seed.values()),
                        "by_training_seed": by_seed,
                    }
                passed = all(row["passed"] for row in task_passes.values())
                by_track_horizon[track][str(horizon)] = passed
                details[f"{track}/h{horizon}"] = {
                    "passed": passed,
                    "by_task": task_passes,
                    "input_condition": "natural_history3",
                }
    longest = {
        track: longest_contiguous_horizon(by_track_horizon[track])
        for track in tracks
    }
    validation_primary = bool(
        evaluation_split == "validation"
        and complete_matrix
        and all(by_track_horizon[track]["1"] for track in tracks)
    )
    low = by_track_horizon.get("test_extrapolation_low", {}).get("1", False)
    high = by_track_horizon.get("test_extrapolation_high", {}).get("1", False)
    return {
        "evaluation_split": evaluation_split,
        "formal_matrix_complete": complete_matrix,
        "visible_geometry_generalization_validation_gate_passed": (
            validation_primary if evaluation_split == "validation" else None
        ),
        "formal_pass_by_track_and_horizon": by_track_horizon,
        "longest_contiguous_passing_horizon_by_track": longest,
        "formal_details": details,
        "primary_input_condition": "natural_history3",
        "primary_horizon": 1,
        "required_prediction_tasks": list(TASKS),
        "query_only_is_descriptive_and_not_a_hard_gate": True,
        "door_position_icl_claimed": False,
        "sealed_low_extrapolation_h1_passed": (
            bool(low) if evaluation_split == "sealed_test" else None
        ),
        "sealed_high_extrapolation_h1_passed": (
            bool(high) if evaluation_split == "sealed_test" else None
        ),
        "sealed_bidirectional_extrapolation_h1_passed": (
            bool(low and high) if evaluation_split == "sealed_test" else None
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = file_sha256(config_path)
    gate_audit = require_sealed_test_gate(
        split=args.split,
        config_path=config_path,
        config=config,
        manifest_path=getattr(args, "sealed_test_gate", None),
        repo_root=ROOT,
    )
    artifact_root = require_canonical_split_path(
        args.artifact_root,
        canonical=canonical_door_split_root(
            config, split=args.split, repo_root=ROOT
        ),
        split=args.split,
        label="Door latent analysis root",
    )
    build_report_path = artifact_root / "catalogs" / "build_report.json"
    build_report_hash = file_sha256(build_report_path)
    build_report = json.loads(build_report_path.read_text(encoding="utf-8"))
    if build_report["config"]["sha256"] != config_hash:
        raise RuntimeError("Build report/config hash mismatch")
    if not args.allow_partial and build_report.get("status") != "passed":
        raise RuntimeError("Formal analysis refuses a smoke build report")
    evaluation_split = str(build_report.get("evaluation_split", "validation"))
    if evaluation_split != args.split:
        raise RuntimeError("Build report/analyzer split mismatch")
    stable_worldmodel_commit = str(build_report["stable_worldmodel"]["commit"])
    paths = [
        require_path_within_split_root(
            path,
            split_root=artifact_root,
            split=args.split,
            label="Door latent result",
        )
        for path in args.results
    ]
    if not paths:
        paths = sorted(
            path.resolve()
            for path in artifact_root.glob("*.json")
            if path.name != "final_summary.json"
        )
    results = _load_results(
        paths,
        config_hash=config_hash,
        build_report_hash=build_report_hash,
        evaluation_split=evaluation_split,
        stable_worldmodel_commit=stable_worldmodel_commit,
        require_formal=not args.allow_partial,
    )
    matrix = _audit_model_matrix(results, allow_partial=args.allow_partial)
    normalizer_path = resolve_contextworld_path(
        args.expected_normalizer, repo_root=ROOT
    )
    expected_normalizer_sha256 = file_sha256(normalizer_path)
    runtime_identity = _audit_runtime_identity(
        results, expected_normalizer_sha256=expected_normalizer_sha256
    )
    if args.training_reports:
        training_report_paths = [
            resolve_contextworld_path(path, repo_root=ROOT)
            for path in args.training_reports
        ]
    elif args.allow_partial:
        training_report_paths = []
    else:
        report_root = resolve_contextworld_path(
            args.training_report_root, repo_root=ROOT
        )
        training_report_paths = [
            report_root / binding["report_name"]
            for binding in TRAINING_BINDINGS.values()
        ]
    training_bindings = (
        _audit_training_report_bindings(
            results,
            report_paths=training_report_paths,
            stable_worldmodel_commit=stable_worldmodel_commit,
            expected_data_split_seed=int(
                config["training_protocol"]["data_split_seed"]
            ),
            require_complete=not args.allow_partial,
        )
        if training_report_paths
        else {
            "passed": False,
            "complete_formal_binding": False,
            "bindings": {},
            "reason": "partial_analysis_without_training_reports",
        }
    )
    paired_by_seed = {}
    for seed_index, seed in enumerate(PAIRED_TRAINING_SEEDS):
        control = _find_by_group_seed(results, "fixed_door_control", seed)
        target = _find_by_group_seed(results, "multi_door_target", seed)
        if control is None or target is None:
            if args.allow_partial:
                continue
            raise RuntimeError(f"Missing paired target/control seed {seed}")
        paired_by_seed[str(seed)] = _paired_seed_summary(
            seed=seed,
            control=control,
            target=target,
            bootstrap_seed=args.bootstrap_seed + 10_000 * seed_index,
            bootstrap_samples=args.bootstrap_samples,
        )
    formal_analysis = bool(
        not args.allow_partial
        and build_report.get("status") == "passed"
        and matrix["complete_formal_matrix"]
        and training_bindings["complete_formal_binding"]
    )
    formal_tracks = tuple(build_report["tracks"])
    decision = _formal_decision(
        paired_by_seed,
        complete_matrix=formal_analysis,
        tracks=formal_tracks,
        evaluation_split=evaluation_split,
    )
    total_sequences = sum(
        int(row["count_audit"]["scored_sequences"])
        for row in results.values()
    )
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "evaluation_split": evaluation_split,
        "sealed_test_gate": gate_audit,
        "status": (
            "passed" if formal_analysis else "partial_analysis_only"
        ),
        "config": {"path": str(config_path), "sha256": config_hash},
        "build_report": {
            "path": str(build_report_path),
            "sha256": build_report_hash,
        },
        "input_files": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "model_matrix_audit": matrix,
        "runtime_identity_audit": runtime_identity,
        "training_report_binding_audit": training_bindings,
        "expected_normalizer": {
            "path": str(normalizer_path),
            "sha256": expected_normalizer_sha256,
        },
        "count_audit": {
            "models": len(results),
            "scored_sequences": total_sequences,
            "horizon_losses": total_sequences * len(HORIZONS),
            "online_environment_calls": 0,
        },
        "metric_contract": {
            "raw_native_latent_mse": "within_checkpoint_only",
            "cross_model_metric": (
                "prediction_mse_divided_by_same_checkpoint_unchanged_frame_baseline"
            ),
            "paired_query_win_rate": "target_normalized_error_below_control",
            "tracks_never_pooled": True,
            "horizons_never_pooled": True,
            "bootstrap": {
                "samples": args.bootstrap_samples,
                "role": "descriptive_pointwise_not_formal_gate",
            },
        },
        "models": {
            slug: {
                "model": row["model"],
                "result": row["result_path"],
                "checkpoint_summary": row["checkpoint_summary"],
            }
            for slug, row in results.items()
        },
        "paired_training_seed_effects": paired_by_seed,
        "decision": decision,
    }
    output = args.output.resolve() if args.output else artifact_root / "final_summary.json"
    write_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml",
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--split", choices=("validation", "sealed_test"), default="validation"
    )
    parser.add_argument("--sealed-test-gate", type=Path)
    parser.add_argument("--results", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--expected-normalizer",
        default=DEFAULT_NORMALIZER,
        help="Frozen original-train normalizer used by every formal model.",
    )
    parser.add_argument(
        "--training-report-root",
        default="artifacts/training/reports",
        help="Directory containing the seven default formal training reports.",
    )
    parser.add_argument(
        "--training-reports",
        nargs="*",
        type=Path,
        help=(
            "Explicit seven training-report JSON paths. If omitted, formal "
            "analysis derives their frozen names from --training-report-root."
        ),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=2026072204)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
