from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json

from .icl_sensitive import sha256_file
from .icl_sensitive_analysis import exact_paired_sign_test


MODEL_ORDER = ("speedfull", "single_speed_mixed_control")
CONDITIONS = ("slow", "correct", "fast")
PREDICTION_HORIZONS_RAW_STEPS = (5, 10, 15, 20, 25)
PRIMARY_PREDICTION_HORIZONS_RAW_STEPS = (5, 10, 15, 25)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("benchmark") != "tworoom_planner_mechanism_attribution_v1":
        raise ValueError(f"Unexpected benchmark in {path}")
    if config.get("status") != "preregistered_before_execution":
        raise ValueError(f"Unexpected preregistration status in {path}")
    return config


def _load_passed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Evaluation did not pass: {path}")
    return payload


def _verified_path(
    spec: Mapping[str, Any], *, repo_root: Path, label: str
) -> Path:
    path = resolve_contextworld_path(spec["path"], repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    observed = sha256_file(path)
    if observed != str(spec["sha256"]):
        raise RuntimeError(
            f"{label} hash mismatch: {observed} != {spec['sha256']}"
        )
    return path


def _record_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row["eval_seed"]), str(row["evaluation_id"])


def _mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values]
    if not selected:
        raise ValueError("Cannot average an empty sequence")
    return float(np.mean(selected))


def _continuous_summary(values: Iterable[float]) -> dict[str, Any]:
    selected = np.asarray(list(values), dtype=np.float64)
    if selected.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p25": float(np.percentile(selected, 25)),
        "p75": float(np.percentile(selected, 75)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def paired_continuous_comparison(
    left: Iterable[float],
    right: Iterable[float],
    *,
    left_label: str,
    right_label: str,
    bootstrap_seed: int,
    bootstrap_resamples: int = 10_000,
    tie_atol: float = 1e-12,
) -> dict[str, Any]:
    left_values = np.asarray(list(left), dtype=np.float64)
    right_values = np.asarray(list(right), dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("Paired arrays must have the same non-zero shape")
    differences = left_values - right_values
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        len(differences),
        size=(int(bootstrap_resamples), len(differences)),
    )
    bootstrap = np.mean(differences[indices], axis=1)
    left_lower = int(np.sum(differences < -tie_atol))
    right_lower = int(np.sum(differences > tie_atol))
    ties = int(len(differences) - left_lower - right_lower)
    return {
        "pairs": int(len(differences)),
        left_label: _continuous_summary(left_values),
        right_label: _continuous_summary(right_values),
        f"{left_label}_minus_{right_label}": {
            **_continuous_summary(differences),
            "evaluation_bootstrap_95_ci": [
                float(value)
                for value in np.percentile(bootstrap, [2.5, 97.5])
            ],
        },
        f"{left_label}_lower_pairs": left_lower,
        f"{right_label}_lower_pairs": right_lower,
        "ties": ties,
        "paired_exact_sign_test": exact_paired_sign_test(
            left_lower, right_lower
        ),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(bootstrap_resamples),
    }


def paired_binary_comparison(
    left: Iterable[bool],
    right: Iterable[bool],
    *,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    left_values = np.asarray(list(left), dtype=np.bool_)
    right_values = np.asarray(list(right), dtype=np.bool_)
    if left_values.shape != right_values.shape or left_values.size == 0:
        raise ValueError("Paired arrays must have the same non-zero shape")
    left_only = int(np.sum(left_values & ~right_values))
    right_only = int(np.sum(~left_values & right_values))
    both = int(np.sum(left_values & right_values))
    neither = int(np.sum(~left_values & ~right_values))
    total = int(left_values.size)
    left_successes = int(np.sum(left_values))
    right_successes = int(np.sum(right_values))
    return {
        "pairs": total,
        left_label: {
            "successes": left_successes,
            "success_rate_percent": 100.0 * left_successes / total,
        },
        right_label: {
            "successes": right_successes,
            "success_rate_percent": 100.0 * right_successes / total,
        },
        f"{left_label}_minus_{right_label}_success_rate_points": (
            100.0 * (left_successes - right_successes) / total
        ),
        f"{left_label}_only_successes": left_only,
        f"{right_label}_only_successes": right_only,
        "both_successes": both,
        "neither_successes": neither,
        "paired_exact_sign_test": exact_paired_sign_test(
            left_only, right_only
        ),
    }


def _trajectory_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    trace = row["trajectory"]
    distances = np.asarray(trace["goal_distances"], dtype=np.float64)
    if distances.size != int(trace["raw_steps_executed"]) + 1:
        raise RuntimeError(
            f"Trajectory length mismatch at {row['evaluation_id']}"
        )
    initial = float(distances[0])
    final = float(distances[-1])
    return {
        "final_distance": final,
        "best_distance": float(np.min(distances)),
        "normalized_progress": (
            (initial - final) / initial if initial > 0.0 else 0.0
        ),
        "normalized_distance_auc": float(
            trace["normalized_distance_auc"]
        ),
        "distance_auc_raw_step_mean": float(
            trace["distance_auc_raw_step_mean"]
        ),
        "path_length": float(trace["path_length"]),
        "progress_per_path_length": float(
            trace["progress_per_path_length"]
        ),
        "raw_steps_executed": float(trace["raw_steps_executed"]),
    }


def summarize_closed_loop_condition(
    rows: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    selected = [rows[key] for key in sorted(rows)]
    if not selected:
        raise ValueError("No closed-loop records")
    total = len(selected)
    successes = [bool(row["success"]) for row in selected]
    metrics = [_trajectory_metrics(row) for row in selected]
    success_steps = [
        int(row["trajectory"]["steps_to_success"])
        for row in selected
        if row["success"]
    ]
    path_efficiencies = [
        float(row["trajectory"]["path_efficiency_success_only"])
        for row in selected
        if row["success"]
    ]
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        by_seed[int(row["eval_seed"])].append(row)
    return {
        "evaluations": total,
        "successes": int(sum(successes)),
        "success_rate_percent": 100.0 * sum(successes) / total,
        "final_distance_px": _continuous_summary(
            value["final_distance"] for value in metrics
        ),
        "best_distance_px": _continuous_summary(
            value["best_distance"] for value in metrics
        ),
        "normalized_progress": _continuous_summary(
            value["normalized_progress"] for value in metrics
        ),
        "normalized_distance_auc": _continuous_summary(
            value["normalized_distance_auc"] for value in metrics
        ),
        "distance_auc_raw_step_mean_px": _continuous_summary(
            value["distance_auc_raw_step_mean"] for value in metrics
        ),
        "path_length_px": _continuous_summary(
            value["path_length"] for value in metrics
        ),
        "progress_per_path_length": _continuous_summary(
            value["progress_per_path_length"] for value in metrics
        ),
        "raw_steps_executed": _continuous_summary(
            value["raw_steps_executed"] for value in metrics
        ),
        "steps_to_success_success_only": (
            _continuous_summary(success_steps) if success_steps else None
        ),
        "path_efficiency_success_only": (
            _continuous_summary(path_efficiencies)
            if path_efficiencies
            else None
        ),
        "by_eval_seed": {
            str(seed): {
                "evaluations": len(seed_rows),
                "successes": int(
                    sum(bool(row["success"]) for row in seed_rows)
                ),
                "success_rate_percent": float(
                    100.0
                    * sum(bool(row["success"]) for row in seed_rows)
                    / len(seed_rows)
                ),
                "mean_final_distance_px": _mean(
                    row["final_distance"] for row in seed_rows
                ),
            }
            for seed, seed_rows in sorted(by_seed.items())
        },
    }


def _paired_values(
    rows: Mapping[str, Mapping[tuple[int, str], Mapping[str, Any]]],
    left: str,
    right: str,
    extractor: Any,
) -> tuple[list[Any], list[Any]]:
    left_keys = sorted(rows[left])
    if set(left_keys) != set(rows[right]):
        raise RuntimeError(f"{left}/{right} record keys differ")
    return (
        [extractor(rows[left][key]) for key in left_keys],
        [extractor(rows[right][key]) for key in left_keys],
    )


def _assert_shared_row_metadata(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    fields = (
        "evaluation_id",
        "evaluation_index",
        "repeat_index",
        "query_id",
        "speed",
        "template_id",
        "goal_state",
        "cem_seed",
        "cem_rng_state_sha256_before",
    )
    mismatches = [field for field in fields if left[field] != right[field]]
    if mismatches:
        raise RuntimeError(f"{label} pairing mismatch: {mismatches}")


def _assert_trajectory_prefix(
    shorter: Mapping[str, Any],
    longer: Mapping[str, Any],
    *,
    label: str,
) -> None:
    short_trace = shorter["trajectory"]
    long_trace = longer["trajectory"]
    short_steps = int(short_trace["raw_steps_executed"])
    for field in ("actions", "states", "goal_distances"):
        short_values = short_trace[field]
        expected = (
            long_trace[field][:short_steps]
            if field == "actions"
            else long_trace[field][: short_steps + 1]
        )
        if short_values != expected:
            raise RuntimeError(f"{label} trajectory {field} prefix differs")
    if shorter["success"] and (
        shorter["success"] != longer["success"]
        or short_trace["steps_to_success"]
        != long_trace["steps_to_success"]
    ):
        raise RuntimeError(f"{label} successful prefix is not preserved")


def _load_closed_loop(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    model_paths: Mapping[str, Path],
) -> tuple[
    dict[str, dict[int, dict[str, dict[tuple[int, str], dict[str, Any]]]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    seeds = [int(value) for value in config["shared_protocol"]["eval_seeds"]]
    per_condition = int(
        config["shared_protocol"]["evaluations_per_condition_per_seed"]
    )
    root = resolve_contextworld_path(
        Path(config["artifacts"]["root"]) / "closed_loop",
        repo_root=repo_root,
    )
    all_rows: dict[
        str, dict[int, dict[str, dict[tuple[int, str], dict[str, Any]]]]
    ] = {}
    inputs: list[dict[str, Any]] = []
    reference_schedules: dict[tuple[str, int], Any] = {}
    reference_metadata: dict[
        tuple[str, int, str, tuple[int, str]], Mapping[str, Any]
    ] = {}
    budgets_by_model = {
        "speedfull": (50, 75, 100),
        "single_speed_mixed_control": (50,),
    }
    directions = ("wrong_slow", "wrong_fast")
    for model_id in MODEL_ORDER:
        slug = model_paths[model_id].parent.name
        all_rows[model_id] = {}
        for budget in budgets_by_model[model_id]:
            conditions: dict[
                str, dict[tuple[int, str], dict[str, Any]]
            ] = {name: {} for name in CONDITIONS}
            for direction in directions:
                for seed in seeds:
                    path = (
                        root
                        / f"budget_{budget}"
                        / slug
                        / f"{direction}_n50_s{seed}.json"
                    )
                    payload = _load_passed(path)
                    protocol = payload["protocol"]
                    expected_protocol = {
                        "eval_seed": seed,
                        "eval_budget": budget,
                        "action_block": 5,
                        "horizon": 5,
                        "receding_horizon": 5,
                        "cem_num_samples": 300,
                        "cem_steps": 30,
                        "cem_topk": 30,
                    }
                    mismatches = {
                        key: (value, protocol.get(key))
                        for key, value in expected_protocol.items()
                        if protocol.get(key) != value
                    }
                    if mismatches:
                        raise RuntimeError(
                            f"Protocol mismatch in {path}: {mismatches}"
                        )
                    expected_checkpoint = config["frozen_inputs"]["models"][
                        model_id
                    ]["sha256"]
                    if payload["checkpoint"]["sha256"] != expected_checkpoint:
                        raise RuntimeError(
                            f"Checkpoint hash mismatch in {path}"
                        )
                    if (
                        payload["stable_worldmodel"]["commit"]
                        != config["frozen_inputs"][
                            "stable_worldmodel_commit"
                        ]
                    ):
                        raise RuntimeError(
                            f"StableWorldModel commit mismatch in {path}"
                        )
                    records = list(payload["records"])
                    if len(records) != 2 * per_condition:
                        raise RuntimeError(
                            f"Expected {2 * per_condition} rows in {path}"
                        )
                    schedule = payload["selection"]["schedule"]
                    schedule_key = (direction, seed)
                    if schedule_key in reference_schedules:
                        if schedule != reference_schedules[schedule_key]:
                            raise RuntimeError(
                                f"Schedule mismatch in {path}"
                            )
                    else:
                        reference_schedules[schedule_key] = schedule
                    for raw_condition in ("correct", "wrong"):
                        selected = [
                            row
                            for row in records
                            if row["condition"] == raw_condition
                        ]
                        if len(selected) != per_condition:
                            raise RuntimeError(
                                f"Wrong condition count in {path}"
                            )
                        condition = (
                            "correct"
                            if raw_condition == "correct"
                            else direction.removeprefix("wrong_")
                        )
                        if condition == "correct" and direction == "wrong_fast":
                            canonical = conditions["correct"]
                            for row in selected:
                                key = _record_key(row)
                                if key not in canonical:
                                    raise RuntimeError(
                                        f"Missing canonical correct row {key}"
                                    )
                                left = canonical[key]
                                _assert_shared_row_metadata(
                                    left,
                                    row,
                                    label=f"{path}/correct",
                                )
                                for field in (
                                    "success",
                                    "final_distance",
                                    "final_state",
                                    "trajectory",
                                    "cem_rng_state_sha256_after",
                                ):
                                    if left[field] != row[field]:
                                        raise RuntimeError(
                                            f"Correct condition differs "
                                            f"between directions: {path}/{key}/{field}"
                                        )
                            continue
                        for row in selected:
                            key = _record_key(row)
                            if key in conditions[condition]:
                                raise RuntimeError(
                                    f"Duplicate {condition} row in {path}: {key}"
                                )
                            trace = row.get("trajectory", {})
                            required_trace = (
                                "states",
                                "actions",
                                "goal_distances",
                                "raw_steps_executed",
                                "steps_to_success",
                                "path_length",
                                "progress_per_path_length",
                                "distance_auc_raw_step_mean",
                                "normalized_distance_auc",
                            )
                            missing = [
                                field
                                for field in required_trace
                                if field not in trace
                            ]
                            if missing:
                                raise RuntimeError(
                                    f"Missing trace fields in {path}: {missing}"
                                )
                            if int(trace["raw_steps_executed"]) > budget:
                                raise RuntimeError(
                                    f"Trace exceeds budget in {path}"
                                )
                            if bool(row["success"]) != (
                                trace["steps_to_success"] is not None
                            ):
                                raise RuntimeError(
                                    f"Success/step mismatch in {path}"
                                )
                            conditions[condition][key] = row
                            metadata_key = (
                                direction,
                                seed,
                                raw_condition,
                                key,
                            )
                            if metadata_key in reference_metadata:
                                _assert_shared_row_metadata(
                                    reference_metadata[metadata_key],
                                    row,
                                    label=f"{path}/{key}",
                                )
                            else:
                                reference_metadata[metadata_key] = row
                    inputs.append(
                        {
                            "path": portable_contextworld_path(
                                path, repo_root=repo_root
                            ),
                            "sha256": sha256_file(path),
                            "model": model_id,
                            "budget_raw_steps": budget,
                            "direction": direction,
                            "eval_seed": seed,
                            "raw_records": len(records),
                        }
                    )
            expected = len(seeds) * per_condition
            if any(len(rows) != expected for rows in conditions.values()):
                raise RuntimeError(
                    f"Condition total mismatch for {model_id}/budget={budget}"
                )
            for key in sorted(conditions["correct"]):
                _assert_shared_row_metadata(
                    conditions["correct"][key],
                    conditions["slow"][key],
                    label=f"{model_id}/budget={budget}/correct-vs-slow/{key}",
                )
                _assert_shared_row_metadata(
                    conditions["correct"][key],
                    conditions["fast"][key],
                    label=f"{model_id}/budget={budget}/correct-vs-fast/{key}",
                )
            all_rows[model_id][budget] = conditions

    prefix_checks = 0
    speedfull = all_rows["speedfull"]
    for short_budget, long_budget in ((50, 75), (75, 100), (50, 100)):
        for condition in CONDITIONS:
            for key, shorter in speedfull[short_budget][condition].items():
                longer = speedfull[long_budget][condition][key]
                _assert_trajectory_prefix(
                    shorter,
                    longer,
                    label=(
                        f"SpeedFull/{condition}/{key}/"
                        f"{short_budget}->{long_budget}"
                    ),
                )
                prefix_checks += 1
    audit = {
        "files": len(inputs),
        "raw_records": sum(int(row["raw_records"]) for row in inputs),
        "unique_condition_records": sum(
            len(rows)
            for model in all_rows.values()
            for budget in model.values()
            for rows in budget.values()
        ),
        "correct_direction_duplicates_verified": (
            sum(len(values) for values in budgets_by_model.values())
            * len(seeds)
            * per_condition
        ),
        "trajectory_prefix_checks_passed": prefix_checks,
        "independent_per_eval_condition": expected,
        "eval_seeds": seeds,
        "passed": True,
    }
    return all_rows, inputs, audit


def _fixed_prediction_summary(
    records: list[Mapping[str, Any]], *, seed_offset: int
) -> dict[str, Any]:
    by_horizon: dict[str, Any] = {}
    for block_index, horizon in enumerate(PREDICTION_HORIZONS_RAW_STEPS):
        values = {
            condition: [
                float(
                    row["conditions"][condition][
                        "prediction_mse_to_true_by_block"
                    ][block_index]
                )
                for row in records
            ]
            for condition in CONDITIONS
        }
        by_horizon[str(horizon)] = {
            "conditions": {
                condition: _continuous_summary(selected)
                for condition, selected in values.items()
            },
            "correct_vs_slow": paired_continuous_comparison(
                values["correct"],
                values["slow"],
                left_label="correct",
                right_label="slow",
                bootstrap_seed=seed_offset + horizon * 101,
            ),
            "correct_vs_fast": paired_continuous_comparison(
                values["correct"],
                values["fast"],
                left_label="correct",
                right_label="fast",
                bootstrap_seed=seed_offset + horizon * 103,
            ),
        }
    aggregate_over_horizons = {}
    horizon_groups = {
        "primary_5_10_15_25": PRIMARY_PREDICTION_HORIZONS_RAW_STEPS,
        "all_observed_5_10_15_20_25": PREDICTION_HORIZONS_RAW_STEPS,
    }
    for group_index, (name, horizons) in enumerate(horizon_groups.items()):
        indices = [
            PREDICTION_HORIZONS_RAW_STEPS.index(horizon)
            for horizon in horizons
        ]
        values = {
            condition: [
                float(
                    np.mean(
                        np.asarray(
                            row["conditions"][condition][
                                "prediction_mse_to_true_by_block"
                            ],
                            dtype=np.float64,
                        )[indices]
                    )
                )
                for row in records
            ]
            for condition in CONDITIONS
        }
        aggregate_over_horizons[name] = {
            "horizons_raw_steps": list(horizons),
            "aggregation": (
                "mean native latent MSE across the listed horizons for each "
                "paired evaluation"
            ),
            "conditions": {
                condition: _continuous_summary(selected)
                for condition, selected in values.items()
            },
            "correct_vs_slow": paired_continuous_comparison(
                values["correct"],
                values["slow"],
                left_label="correct",
                right_label="slow",
                bootstrap_seed=seed_offset + 701 + group_index,
            ),
            "correct_vs_fast": paired_continuous_comparison(
                values["correct"],
                values["fast"],
                left_label="correct",
                right_label="fast",
                bootstrap_seed=seed_offset + 711 + group_index,
            ),
        }
    primary_pass = {
        wrong: all(
            by_horizon[str(horizon)][f"correct_vs_{wrong}"][
                f"correct_minus_{wrong}"
            ]["mean"]
            < 0.0
            for horizon in PRIMARY_PREDICTION_HORIZONS_RAW_STEPS
        )
        for wrong in ("slow", "fast")
    }
    return {
        "metric": "native_latent_mse_to_true_query_speed_rollout",
        "scale_boundary": (
            "Only within-model, paired context comparisons are meaningful; "
            "latent MSE scales are not compared across model checkpoints."
        ),
        "probe": (
            "The exact correct-speed oracle's selected 25-step action "
            "sequence, shared across the three contexts."
        ),
        "observed_horizons_raw_steps": list(
            PREDICTION_HORIZONS_RAW_STEPS
        ),
        "primary_horizons_raw_steps": list(
            PRIMARY_PREDICTION_HORIZONS_RAW_STEPS
        ),
        "by_horizon_raw_steps": by_horizon,
        "aggregate_over_horizons": aggregate_over_horizons,
        "correct_mean_mse_lower_at_every_primary_horizon": primary_pass,
    }


def _fixed_model_summary(
    records: list[Mapping[str, Any]], *, seed_offset: int
) -> dict[str, Any]:
    condition_summary = {}
    for condition in CONDITIONS:
        rows = [row["conditions"][condition] for row in records]
        success = [bool(row["selected_true_success"]) for row in rows]
        terminal_success = [
            float(row["selected_true_final_distance"]) < 16.0
            for row in rows
        ]
        condition_summary[condition] = {
            "evaluations": len(rows),
            "selected_candidate_ever_successes": int(sum(success)),
            "selected_candidate_ever_success_rate_percent": (
                100.0 * sum(success) / len(rows)
            ),
            "selected_candidate_terminal_successes": int(
                sum(terminal_success)
            ),
            "selected_candidate_terminal_success_rate_percent": (
                100.0 * sum(terminal_success) / len(rows)
            ),
            "selected_candidate_true_final_distance_px": _continuous_summary(
                float(row["selected_true_final_distance"]) for row in rows
            ),
            "cost_rank_spearman_vs_true_terminal_distance": (
                _continuous_summary(
                    float(row["spearman_cost_vs_true_final_distance"])
                    for row in rows
                )
            ),
        }

    pair_keys = (("slow", "correct"), ("correct", "fast"), ("slow", "fast"))
    context_rank = {
        f"{left}_{right}": _continuous_summary(
            float(row["context_cost_rank_spearman"][f"{left}_{right}"])
            for row in records
        )
        for left, right in pair_keys
    }
    topk = {
        pair: _continuous_summary(
            float(row["topk_overlap"][pair]) for row in records
        )
        for pair in ("slow_correct", "correct_fast")
    }
    selected_agreement = {}
    for left, right in pair_keys:
        agreements = sum(
            int(row["conditions"][left]["selected_candidate"])
            == int(row["conditions"][right]["selected_candidate"])
            for row in records
        )
        selected_agreement[f"{left}_{right}"] = {
            "agreements": int(agreements),
            "pairs": len(records),
            "agreement_rate_percent": 100.0 * agreements / len(records),
        }

    selected_success_comparisons = {}
    selected_terminal_distance_comparisons = {}
    for wrong in ("slow", "fast"):
        left = [
            bool(row["conditions"]["correct"]["selected_true_success"])
            for row in records
        ]
        right = [
            bool(row["conditions"][wrong]["selected_true_success"])
            for row in records
        ]
        selected_success_comparisons[f"correct_vs_{wrong}"] = (
            paired_binary_comparison(
                left,
                right,
                left_label="correct",
                right_label=wrong,
            )
        )
        selected_terminal_distance_comparisons[
            f"correct_vs_{wrong}"
        ] = paired_continuous_comparison(
            [
                float(
                    row["conditions"]["correct"][
                        "selected_true_final_distance"
                    ]
                )
                for row in records
            ],
            [
                float(
                    row["conditions"][wrong][
                        "selected_true_final_distance"
                    ]
                )
                for row in records
            ],
            left_label="correct",
            right_label=wrong,
            bootstrap_seed=seed_offset + 801 + (wrong == "fast"),
        )
    return {
        "conditions": condition_summary,
        "context_cost_rank_spearman": context_rank,
        "top30_overlap": topk,
        "selected_candidate_agreement": selected_agreement,
        "selected_ever_success_comparisons": selected_success_comparisons,
        "selected_true_terminal_distance_comparisons": (
            selected_terminal_distance_comparisons
        ),
        "prediction_accuracy": _fixed_prediction_summary(
            records, seed_offset=seed_offset
        ),
    }


def _oracle_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    conditions = {}
    for condition in CONDITIONS:
        rows = [row["oracle"][condition] for row in records]
        ever = [bool(row["selected_true_success"]) for row in rows]
        terminal = [
            float(row["selected_true_final_distance"]) < 16.0
            for row in rows
        ]
        conditions[condition] = {
            "evaluations": len(rows),
            "ever_successes": int(sum(ever)),
            "ever_success_rate_percent": 100.0 * sum(ever) / len(rows),
            "terminal_successes": int(sum(terminal)),
            "terminal_success_rate_percent": (
                100.0 * sum(terminal) / len(rows)
            ),
            "true_final_distance_px": _continuous_summary(
                float(row["selected_true_final_distance"]) for row in rows
            ),
            "steps_to_first_success_success_only": (
                _continuous_summary(
                    int(row["selected_true_steps_to_success"])
                    for row in rows
                    if row["selected_true_success"]
                )
                if any(ever)
                else None
            ),
        }
    comparisons = {}
    all_no_worse = {}
    for wrong_index, wrong in enumerate(("slow", "fast")):
        correct_distance = [
            float(row["oracle"]["correct"]["selected_true_final_distance"])
            for row in records
        ]
        wrong_distance = [
            float(row["oracle"][wrong]["selected_true_final_distance"])
            for row in records
        ]
        comparisons[f"correct_vs_{wrong}_true_final_distance"] = (
            paired_continuous_comparison(
                correct_distance,
                wrong_distance,
                left_label="correct",
                right_label=wrong,
                bootstrap_seed=91001 + wrong_index,
            )
        )
        all_no_worse[wrong] = all(
            correct <= mismatch + 1e-6
            for correct, mismatch in zip(correct_distance, wrong_distance)
        )
    reachability_passed = (
        conditions["correct"]["terminal_success_rate_percent"] >= 95.0
    )
    return {
        "selection_rule": (
            "Minimize exact terminal distance after 25 raw steps under the "
            "assumed speed; execute the selected sequence at query speed."
        ),
        "success_semantics": (
            "ever_success means the trajectory entered the 16 px success "
            "radius at any step; terminal_success means it remained inside "
            "at step 25. The exact fixed rollout does not stop on entry."
        ),
        "conditions": conditions,
        "correct_selected_true_terminal_distance_no_worse_per_pair": (
            all_no_worse
        ),
        "correct_candidate_bank_terminal_reachability_gate": {
            "requirement_percent": 95.0,
            "observed_percent": conditions["correct"][
                "terminal_success_rate_percent"
            ],
            "passed": reachability_passed,
        },
        "comparisons": comparisons,
        "positive_control_passed": reachability_passed and all(
            all_no_worse.values()
        ),
    }


def _load_fixed_candidates(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    model_paths: Mapping[str, Path],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    seeds = [int(value) for value in config["shared_protocol"]["eval_seeds"]]
    per_seed = int(
        config["shared_protocol"]["evaluations_per_condition_per_seed"]
    )
    root = resolve_contextworld_path(
        Path(config["artifacts"]["root"]) / "fixed_candidates",
        repo_root=repo_root,
    )
    model_records: dict[str, list[dict[str, Any]]] = {}
    inputs: list[dict[str, Any]] = []
    reference: dict[tuple[int, str], dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        slug = model_paths[model_id].parent.name
        selected: list[dict[str, Any]] = []
        for seed in seeds:
            path = root / slug / f"fixed_candidates_n50_s{seed}.json"
            payload = _load_passed(path)
            if payload.get("candidate_generation_version") != (
                "goal_directed_mixture_v1"
            ):
                raise RuntimeError(
                    f"Unexpected candidate generation in {path}"
                )
            if int(payload["eval_seed"]) != seed:
                raise RuntimeError(f"Eval seed mismatch in {path}")
            expected_checkpoint = config["frozen_inputs"]["models"][
                model_id
            ]["sha256"]
            if payload["model"]["sha256"] != expected_checkpoint:
                raise RuntimeError(
                    f"Checkpoint hash mismatch in {path}"
                )
            if (
                payload["stable_worldmodel"]["commit"]
                != config["frozen_inputs"]["stable_worldmodel_commit"]
            ):
                raise RuntimeError(
                    f"StableWorldModel commit mismatch in {path}"
                )
            records = list(payload["records"])
            if len(records) != per_seed:
                raise RuntimeError(
                    f"Expected {per_seed} fixed rows in {path}"
                )
            for row in records:
                if tuple(sorted(row["conditions"])) != tuple(
                    sorted(CONDITIONS)
                ):
                    raise RuntimeError(f"Condition mismatch in {path}")
                for condition in CONDITIONS:
                    mse = row["conditions"][condition][
                        "prediction_mse_to_true_by_block"
                    ]
                    if len(mse) != len(PREDICTION_HORIZONS_RAW_STEPS):
                        raise RuntimeError(
                            f"Prediction horizon mismatch in {path}"
                        )
                key = (seed, str(row["evaluation_id"]))
                shared = {
                    "evaluation_index": row["evaluation_index"],
                    "query_id": row["query_id"],
                    "query_speed": row["query_speed"],
                    "candidate_bank_sha256": row[
                        "candidate_bank_sha256"
                    ],
                    "oracle": row["oracle"],
                }
                if key in reference:
                    if shared != reference[key]:
                        raise RuntimeError(
                            f"Fixed candidate pairing differs at {path}/{key}"
                        )
                else:
                    reference[key] = shared
            selected.extend(records)
            inputs.append(
                {
                    "path": portable_contextworld_path(
                        path, repo_root=repo_root
                    ),
                    "sha256": sha256_file(path),
                    "model": model_id,
                    "eval_seed": seed,
                    "query_records": len(records),
                    "context_evaluations": len(records) * 3,
                }
            )
        expected = len(seeds) * per_seed
        if len(selected) != expected:
            raise RuntimeError(
                f"Fixed total mismatch for {model_id}: {len(selected)}"
            )
        model_records[model_id] = selected
    speedfull_ids = {
        str(row["evaluation_id"]) for row in model_records["speedfull"]
    }
    control_ids = {
        str(row["evaluation_id"])
        for row in model_records["single_speed_mixed_control"]
    }
    if speedfull_ids != control_ids:
        raise RuntimeError("Fixed candidate cross-model schedule differs")
    declared = tuple(
        int(value)
        for value in config["directional_prediction_accuracy"][
            "horizons_raw_steps"
        ]
    )
    primary = tuple(
        int(value)
        for value in config["directional_prediction_accuracy"][
            "primary_horizons_raw_steps"
        ]
    )
    audit = {
        "files": len(inputs),
        "query_records": sum(
            int(row["query_records"]) for row in inputs
        ),
        "context_evaluations": sum(
            int(row["context_evaluations"]) for row in inputs
        ),
        "paired_query_records_per_model": len(
            model_records["speedfull"]
        ),
        "candidate_banks_and_oracles_shared_across_models": True,
        "observed_prediction_horizons_raw_steps": list(
            PREDICTION_HORIZONS_RAW_STEPS
        ),
        "declared_prediction_horizons_raw_steps": list(declared),
        "primary_prediction_horizons_all_observed": set(primary).issubset(
            PREDICTION_HORIZONS_RAW_STEPS
        ),
        "protocol_deviation": {
            "present": declared != PREDICTION_HORIZONS_RAW_STEPS,
            "description": (
                "The preregistration's broad horizon list included raw steps "
                "1/2/3, but this action-chunk model emits one prediction per "
                "5 raw actions. The implementation therefore observes raw "
                "steps 5/10/15/20/25. All preregistered primary horizons "
                "5/10/15/25 are present; step 20 is descriptive."
            ),
            "primary_inference_affected": not set(primary).issubset(
                PREDICTION_HORIZONS_RAW_STEPS
            ),
        },
        "passed": True,
    }
    return model_records, inputs, audit


def _closed_loop_analysis(
    rows: Mapping[
        str,
        Mapping[
            int, Mapping[str, Mapping[tuple[int, str], Mapping[str, Any]]]
        ],
    ]
) -> dict[str, Any]:
    models = {}
    for model_index, model_id in enumerate(MODEL_ORDER):
        budgets = {}
        for budget, conditions in sorted(rows[model_id].items()):
            condition_summaries = {
                condition: summarize_closed_loop_condition(
                    conditions[condition]
                )
                for condition in CONDITIONS
            }
            success_comparisons = {}
            for left, right in (
                ("fast", "slow"),
                ("correct", "slow"),
                ("correct", "fast"),
            ):
                left_values, right_values = _paired_values(
                    conditions,
                    left,
                    right,
                    lambda row: bool(row["success"]),
                )
                success_comparisons[f"{left}_vs_{right}"] = (
                    paired_binary_comparison(
                        left_values,
                        right_values,
                        left_label=left,
                        right_label=right,
                    )
                )
            continuous_comparisons = {}
            for metric_name, extractor in (
                (
                    "final_distance_px",
                    lambda row: float(row["final_distance"]),
                ),
                (
                    "normalized_progress",
                    lambda row: _trajectory_metrics(row)[
                        "normalized_progress"
                    ],
                ),
                (
                    "normalized_distance_auc",
                    lambda row: float(
                        row["trajectory"]["normalized_distance_auc"]
                    ),
                ),
            ):
                metric_rows = {}
                for left, right in (
                    ("fast", "slow"),
                    ("correct", "slow"),
                    ("correct", "fast"),
                ):
                    left_values, right_values = _paired_values(
                        conditions, left, right, extractor
                    )
                    metric_rows[f"{left}_vs_{right}"] = (
                        paired_continuous_comparison(
                            left_values,
                            right_values,
                            left_label=left,
                            right_label=right,
                            bootstrap_seed=(
                                73000
                                + model_index * 1000
                                + budget * 10
                                + sum(map(ord, metric_name + left + right))
                            ),
                        )
                    )
                continuous_comparisons[metric_name] = metric_rows
            common_success_steps = {}
            for left, right in (
                ("fast", "slow"),
                ("correct", "slow"),
                ("fast", "correct"),
            ):
                keys = [
                    key
                    for key in sorted(conditions[left])
                    if conditions[left][key]["success"]
                    and conditions[right][key]["success"]
                ]
                common_success_steps[f"{left}_vs_{right}"] = (
                    paired_continuous_comparison(
                        [
                            int(
                                conditions[left][key]["trajectory"][
                                    "steps_to_success"
                                ]
                            )
                            for key in keys
                        ],
                        [
                            int(
                                conditions[right][key]["trajectory"][
                                    "steps_to_success"
                                ]
                            )
                            for key in keys
                        ],
                        left_label=left,
                        right_label=right,
                        bootstrap_seed=(
                            76000
                            + model_index * 1000
                            + budget * 10
                            + sum(map(ord, left + right))
                        ),
                    )
                )
            budgets[str(budget)] = {
                "conditions": condition_summaries,
                "paired_success_comparisons": success_comparisons,
                "paired_continuous_comparisons": continuous_comparisons,
                "paired_steps_to_success_on_common_successes": (
                    common_success_steps
                ),
            }
        models[model_id] = {"by_eval_budget_raw_steps": budgets}

    speedfull = rows["speedfull"]
    budget_sensitivity = {}
    for condition in CONDITIONS:
        condition_rows = {}
        for budget in (50, 75, 100):
            selected = speedfull[budget][condition]
            condition_rows[str(budget)] = {
                "successes": int(
                    sum(bool(row["success"]) for row in selected.values())
                ),
                "success_rate_percent": float(
                    100.0
                    * sum(bool(row["success"]) for row in selected.values())
                    / len(selected)
                ),
            }
        gains = {}
        for short, long in ((50, 75), (75, 100), (50, 100)):
            keys = sorted(speedfull[short][condition])
            gains[f"{short}_to_{long}"] = paired_binary_comparison(
                [
                    bool(speedfull[long][condition][key]["success"])
                    for key in keys
                ],
                [
                    bool(speedfull[short][condition][key]["success"])
                    for key in keys
                ],
                left_label=f"budget_{long}",
                right_label=f"budget_{short}",
            )
        budget_sensitivity[condition] = {
            "by_budget": condition_rows,
            "paired_budget_gains": gains,
        }
    fast_minus_slow = {
        str(budget): models["speedfull"]["by_eval_budget_raw_steps"][
            str(budget)
        ]["paired_success_comparisons"]["fast_vs_slow"][
            "fast_minus_slow_success_rate_points"
        ]
        for budget in (50, 75, 100)
    }
    fast_minus_correct = {
        str(budget): -models["speedfull"]["by_eval_budget_raw_steps"][
            str(budget)
        ]["paired_success_comparisons"]["correct_vs_fast"][
            "correct_minus_fast_success_rate_points"
        ]
        for budget in (50, 75, 100)
    }
    context_effect_change = {}
    for effect_name, positive, negative in (
        ("fast_minus_slow", "fast", "slow"),
        ("fast_minus_correct", "fast", "correct"),
    ):
        effects = {}
        for budget in (50, 75, 100):
            keys = sorted(speedfull[budget][positive])
            effects[budget] = [
                100.0
                * (
                    float(
                        bool(speedfull[budget][positive][key]["success"])
                    )
                    - float(
                        bool(speedfull[budget][negative][key]["success"])
                    )
                )
                for key in keys
            ]
        context_effect_change[effect_name] = {
            f"budget_{long}_minus_budget_{short}": (
                paired_continuous_comparison(
                    effects[long],
                    effects[short],
                    left_label=f"effect_budget_{long}",
                    right_label=f"effect_budget_{short}",
                    bootstrap_seed=77000 + short + long,
                )
            )
            for short, long in ((50, 75), (75, 100), (50, 100))
        }
    return {
        "models": models,
        "speedfull_budget_sensitivity": {
            "conditions": budget_sensitivity,
            "fast_minus_slow_success_rate_points": fast_minus_slow,
            "fast_minus_correct_success_rate_points": fast_minus_correct,
            "fast_advantage_change_50_to_100_points": (
                fast_minus_slow["100"] - fast_minus_slow["50"]
            ),
            "context_effect_budget_change_points": context_effect_change,
        },
    }


def run_analysis(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    repo_root: Path,
    write_output: bool = True,
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    normalizer = _verified_path(
        frozen["normalizer"], repo_root=repo_root, label="normalizer"
    )
    catalogs = {
        label: _verified_path(
            spec, repo_root=repo_root, label=f"{label} catalog"
        )
        for label, spec in frozen["catalogs"].items()
    }
    model_paths = {
        model_id: _verified_path(
            frozen["models"][model_id],
            repo_root=repo_root,
            label=f"{model_id} checkpoint",
        )
        for model_id in MODEL_ORDER
    }
    fixed_records, fixed_inputs, fixed_audit = _load_fixed_candidates(
        config=config, repo_root=repo_root, model_paths=model_paths
    )
    closed_rows, closed_inputs, closed_audit = _load_closed_loop(
        config=config, repo_root=repo_root, model_paths=model_paths
    )
    fixed_analysis = {
        "oracle": _oracle_summary(fixed_records["speedfull"]),
        "models": {
            model_id: _fixed_model_summary(
                fixed_records[model_id],
                seed_offset=81000 + model_index * 1000,
            )
            for model_index, model_id in enumerate(MODEL_ORDER)
        },
    }
    closed_analysis = _closed_loop_analysis(closed_rows)

    speedfull_prediction = fixed_analysis["models"]["speedfull"][
        "prediction_accuracy"
    ]
    control_prediction = fixed_analysis["models"][
        "single_speed_mixed_control"
    ]["prediction_accuracy"]
    speedfull_primary_prediction = speedfull_prediction[
        "aggregate_over_horizons"
    ]["primary_5_10_15_25"]
    control_primary_prediction = control_prediction[
        "aggregate_over_horizons"
    ]["primary_5_10_15_25"]
    correct_prediction_lower_slow = speedfull_prediction[
        "correct_mean_mse_lower_at_every_primary_horizon"
    ]["slow"]
    correct_prediction_lower_fast = speedfull_prediction[
        "correct_mean_mse_lower_at_every_primary_horizon"
    ]["fast"]
    budget_effect = closed_analysis["speedfull_budget_sensitivity"]
    fast_effect = budget_effect["fast_minus_slow_success_rate_points"]
    fast_correct_effect = budget_effect[
        "fast_minus_correct_success_rate_points"
    ]
    shrank = float(fast_effect["100"]) < float(fast_effect["50"])
    fixed_fast_slow_rank = fixed_analysis["models"]["speedfull"][
        "context_cost_rank_spearman"
    ]["slow_fast"]["mean"]
    fixed_fast_correct_rank = fixed_analysis["models"]["speedfull"][
        "context_cost_rank_spearman"
    ]["correct_fast"]["mean"]
    speedfull_closed = closed_analysis["models"]["speedfull"][
        "by_eval_budget_raw_steps"
    ]
    fast_correct_p_50 = speedfull_closed["50"][
        "paired_success_comparisons"
    ]["correct_vs_fast"]["paired_exact_sign_test"]["two_sided_p_value"]
    fast_correct_p_100 = speedfull_closed["100"][
        "paired_success_comparisons"
    ]["correct_vs_fast"]["paired_exact_sign_test"]["two_sided_p_value"]
    common_success_fast_correct_50 = speedfull_closed["50"][
        "paired_steps_to_success_on_common_successes"
    ]["fast_vs_correct"]["fast_minus_correct"]
    context_effect_change = budget_effect[
        "context_effect_budget_change_points"
    ]
    fast_correct_shrink = context_effect_change["fast_minus_correct"][
        "budget_100_minus_budget_50"
    ]["effect_budget_100_minus_effect_budget_50"]
    fast_slow_shrink = context_effect_change["fast_minus_slow"][
        "budget_100_minus_budget_50"
    ]["effect_budget_100_minus_effect_budget_50"]
    terminal_25_correct_fast = speedfull_prediction[
        "by_horizon_raw_steps"
    ]["25"]["correct_vs_fast"]
    fixed_selected_correct_fast = fixed_analysis["models"]["speedfull"][
        "selected_true_terminal_distance_comparisons"
    ]["correct_vs_fast"]
    conclusion = {
        "prediction_accuracy_separated_from_endpoint_success": True,
        "speedfull_correct_prediction_mean_mse_lower_than_slow_at_all_primary_horizons": (
            correct_prediction_lower_slow
        ),
        "speedfull_correct_prediction_mean_mse_lower_than_fast_at_all_primary_horizons": (
            correct_prediction_lower_fast
        ),
        "speedfull_primary_horizon_average_prediction_mse": {
            "correct": speedfull_primary_prediction["conditions"][
                "correct"
            ]["mean"],
            "slow": speedfull_primary_prediction["conditions"]["slow"][
                "mean"
            ],
            "fast": speedfull_primary_prediction["conditions"]["fast"][
                "mean"
            ],
            "correct_minus_fast": speedfull_primary_prediction[
                "correct_vs_fast"
            ]["correct_minus_fast"]["mean"],
            "correct_minus_fast_bootstrap_95_ci": (
                speedfull_primary_prediction["correct_vs_fast"][
                    "correct_minus_fast"
                ]["evaluation_bootstrap_95_ci"]
            ),
        },
        "speedfull_25_step_prediction_boundary": {
            "correct_minus_fast_mean_mse": (
                terminal_25_correct_fast["correct_minus_fast"]["mean"]
            ),
            "bootstrap_95_ci": terminal_25_correct_fast[
                "correct_minus_fast"
            ]["evaluation_bootstrap_95_ci"],
            "correct_lower_pairs": terminal_25_correct_fast[
                "correct_lower_pairs"
            ],
            "fast_lower_pairs": terminal_25_correct_fast[
                "fast_lower_pairs"
            ],
            "interpretation": (
                "Correct has lower pooled mean error, but the paired "
                "bootstrap interval crosses zero and Fast is lower in more "
                "individual 25-step probes. The robust accuracy claim is "
                "therefore the trajectory-average result, not uniform "
                "per-query superiority at the terminal horizon."
            ),
        },
        "single_speed_control_primary_horizon_average_prediction_mse": {
            "correct": control_primary_prediction["conditions"]["correct"][
                "mean"
            ],
            "slow": control_primary_prediction["conditions"]["slow"][
                "mean"
            ],
            "fast": control_primary_prediction["conditions"]["fast"][
                "mean"
            ],
            "correct_minus_fast": control_primary_prediction[
                "correct_vs_fast"
            ]["correct_minus_fast"]["mean"],
            "correct_minus_fast_bootstrap_95_ci": (
                control_primary_prediction["correct_vs_fast"][
                    "correct_minus_fast"
                ]["evaluation_bootstrap_95_ci"]
            ),
        },
        "speedfull_fast_context_finite_budget_success_advantage_at_50_steps_points": (
            float(fast_effect["50"])
        ),
        "speedfull_fast_minus_correct_success_advantage_points": {
            key: float(value)
            for key, value in fast_correct_effect.items()
        },
        "speedfull_fast_minus_correct_paired_p_value": {
            "50": float(fast_correct_p_50),
            "100": float(fast_correct_p_100),
        },
        "speedfull_fast_minus_correct_change_50_to_100_points": {
            "mean": fast_correct_shrink["mean"],
            "bootstrap_95_ci": fast_correct_shrink[
                "evaluation_bootstrap_95_ci"
            ],
        },
        "speedfull_fast_minus_correct_steps_on_common_successes_at_50": {
            "pairs": speedfull_closed["50"][
                "paired_steps_to_success_on_common_successes"
            ]["fast_vs_correct"]["pairs"],
            "mean_raw_steps": common_success_fast_correct_50["mean"],
            "bootstrap_95_ci": common_success_fast_correct_50[
                "evaluation_bootstrap_95_ci"
            ],
            "interpretation": (
                "Positive values mean Fast took more real environment steps. "
                "Its higher 50-step success rate is therefore not evidence "
                "that the same commonly solved episodes finished earlier."
            ),
        },
        "speedfull_fast_minus_slow_advantage_shrank_by_100_steps": shrank,
        "speedfull_fast_minus_slow_change_50_to_100_points": float(
            budget_effect["fast_advantage_change_50_to_100_points"]
        ),
        "speedfull_fast_minus_slow_change_50_to_100_bootstrap_95_ci": (
            fast_slow_shrink["evaluation_bootstrap_95_ci"]
        ),
        "fixed_candidate_context_rank_similarity": {
            "correct_fast_mean_spearman": fixed_fast_correct_rank,
            "slow_fast_mean_spearman": fixed_fast_slow_rank,
        },
        "fixed_candidate_correct_minus_fast_true_terminal_distance_px": {
            "mean": fixed_selected_correct_fast["correct_minus_fast"][
                "mean"
            ],
            "bootstrap_95_ci": fixed_selected_correct_fast[
                "correct_minus_fast"
            ]["evaluation_bootstrap_95_ci"],
        },
        "finite_execution_budget_contributes": (
            fast_correct_shrink["evaluation_bootstrap_95_ci"][1] < 0.0
        ),
        "finite_execution_budget_fully_explains_fast_over_slow": False,
        "mechanism_statement": (
            "A fast context can improve finite-budget endpoint success by "
            "changing the model's predicted displacement and the trajectories "
            "that CEM searches/selects, without being the most accurate "
            "prediction of the true query-speed rollout or reducing the real "
            "steps on commonly solved episodes. Prediction accuracy, "
            "deadline success, and realized efficiency are therefore "
            "distinct benchmark axes."
        ),
        "causal_boundary": (
            "The budget sweep isolates the environment execution-step budget "
            "while keeping CEM horizon and sample count fixed. It does not "
            "represent unlimited planning and cannot by itself separate CEM "
            "sampling, horizon, and replanning effects."
        ),
        "training_seed_boundary": (
            "Both checkpoints use one training seed. The result is a "
            "mechanism-level stage conclusion, not a population claim across "
            "training runs."
        ),
    }
    implementation_paths = [
        repo_root / "contextworld/evaluation/icl_planning.py",
        repo_root / "contextworld/evaluation/planner_mechanism.py",
        repo_root / "contextworld/evaluation/planner_mechanism_analysis.py",
        repo_root / "scripts/eval_tworoom_icl_planning.py",
        repo_root / "scripts/eval_tworoom_fixed_candidate_mechanism.py",
        repo_root / "scripts/run_tworoom_fixed_candidate_mechanism.sh",
        repo_root / "scripts/run_tworoom_planner_mechanism_eval.sh",
        repo_root / "scripts/analyze_tworoom_planner_mechanism.py",
    ]
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "evidence_level": "mechanism_validation_single_training_seed",
        "config": {
            "path": portable_contextworld_path(
                config_path, repo_root=repo_root
            ),
            "sha256": sha256_file(config_path),
        },
        "frozen_input_audit": {
            "stable_worldmodel_commit": frozen[
                "stable_worldmodel_commit"
            ],
            "normalizer": {
                "path": portable_contextworld_path(
                    normalizer, repo_root=repo_root
                ),
                "sha256": sha256_file(normalizer),
            },
            "catalogs": {
                label: {
                    "path": portable_contextworld_path(
                        path, repo_root=repo_root
                    ),
                    "sha256": sha256_file(path),
                }
                for label, path in catalogs.items()
            },
            "models": {
                model_id: {
                    "path": portable_contextworld_path(
                        path, repo_root=repo_root
                    ),
                    "sha256": sha256_file(path),
                }
                for model_id, path in model_paths.items()
            },
            "passed": True,
        },
        "implementation_audit": {
            "files": [
                {
                    "path": portable_contextworld_path(
                        path, repo_root=repo_root
                    ),
                    "sha256": sha256_file(path),
                }
                for path in implementation_paths
            ],
            "passed": all(path.is_file() for path in implementation_paths),
        },
        "protocol_and_count_audit": {
            "fixed_candidate": fixed_audit,
            "closed_loop": closed_audit,
            "all_required_primary_prediction_horizons_observed": (
                fixed_audit["primary_prediction_horizons_all_observed"]
            ),
            "all_audits_passed": (
                fixed_audit["passed"]
                and closed_audit["passed"]
                and fixed_audit[
                    "primary_prediction_horizons_all_observed"
                ]
            ),
        },
        "fixed_candidate_diagnostic": fixed_analysis,
        "closed_loop_budget_diagnostic": closed_analysis,
        "stage_conclusion": conclusion,
        "input_files": {
            "fixed_candidate": fixed_inputs,
            "closed_loop": closed_inputs,
        },
    }
    if write_output:
        output = resolve_contextworld_path(
            config["artifacts"]["final_summary"], repo_root=repo_root
        )
        write_json(output, result)
        result["output"] = portable_contextworld_path(
            output, repo_root=repo_root
        )
    return result


__all__ = [
    "PREDICTION_HORIZONS_RAW_STEPS",
    "PRIMARY_PREDICTION_HORIZONS_RAW_STEPS",
    "load_config",
    "paired_binary_comparison",
    "paired_continuous_comparison",
    "run_analysis",
    "summarize_closed_loop_condition",
]
