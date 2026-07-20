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
from .planner_mechanism_analysis import (
    paired_binary_comparison,
    paired_continuous_comparison,
    summarize_closed_loop_condition,
)


CONFIGURATION_ORDER = (
    "baseline",
    "horizon10",
    "samples600",
    "iterations60",
)
VARIANT_ORDER = CONFIGURATION_ORDER[1:]
CONDITIONS = ("slow", "correct", "fast")
RAW_STEPS = tuple(range(5, 51, 5))


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("benchmark") != "tworoom_cem_config_impact_v1":
        raise ValueError(f"Unexpected benchmark in {path}")
    if config.get("status") != "preregistered_before_formal_execution":
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


def _continuous_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def seed_stratified_bootstrap(
    values: Iterable[float],
    seeds: Iterable[int],
    *,
    resamples: int,
    random_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    value_array = np.asarray(list(values), dtype=np.float64)
    seed_array = np.asarray(list(seeds), dtype=np.int64)
    if value_array.shape != seed_array.shape or value_array.size == 0:
        raise ValueError("Values and seeds must have the same non-zero shape")
    groups = [
        value_array[seed_array == seed]
        for seed in sorted(np.unique(seed_array).tolist())
    ]
    if any(group.size == 0 for group in groups):
        raise ValueError("Every seed stratum must contain observations")
    rng = np.random.default_rng(int(random_seed))
    bootstrap = np.empty(int(resamples), dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, int(resamples), chunk_size):
        stop = min(start + chunk_size, int(resamples))
        chunk = np.zeros(stop - start, dtype=np.float64)
        for group in groups:
            indices = rng.integers(
                0,
                group.size,
                size=(stop - start, group.size),
            )
            chunk += np.mean(group[indices], axis=1)
        bootstrap[start:stop] = chunk / len(groups)
    tail = (1.0 - float(confidence_level)) / 2.0
    return {
        "estimate": float(np.mean(value_array)),
        "confidence_interval": [
            float(value)
            for value in np.quantile(
                bootstrap,
                [tail, 1.0 - tail],
            )
        ],
        "confidence_level": float(confidence_level),
        "method": "paired_seed_stratified_bootstrap",
        "strata": int(len(groups)),
        "stratum_sizes": [int(group.size) for group in groups],
        "resamples": int(resamples),
        "random_seed": int(random_seed),
    }


def paired_sign_flip_test(
    differences: Iterable[float],
    *,
    resamples: int,
    random_seed: int,
) -> dict[str, Any]:
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot test an empty difference vector")
    observed = abs(float(np.mean(values)))
    rng = np.random.default_rng(int(random_seed))
    extreme = 0
    chunk_size = 5_000
    for start in range(0, int(resamples), chunk_size):
        count = min(chunk_size, int(resamples) - start)
        signs = rng.integers(
            0,
            2,
            size=(count, values.size),
            dtype=np.int8,
        )
        signs = signs.astype(np.float64) * 2.0 - 1.0
        statistics = np.abs(np.mean(signs * values[None], axis=1))
        extreme += int(np.sum(statistics >= observed - 1e-15))
    return {
        "observed_absolute_mean": observed,
        "two_sided_p": float((extreme + 1) / (int(resamples) + 1)),
        "method": "paired_sign_flip_monte_carlo",
        "resamples": int(resamples),
        "random_seed": int(random_seed),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (label, value) in enumerate(ordered):
        candidate = min(1.0, (total - index) * float(value))
        running = max(running, candidate)
        adjusted[label] = running
    return adjusted


def _assert_metadata_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str,
) -> None:
    fields = (
        "evaluation_id",
        "evaluation_index",
        "repeat_index",
        "eval_seed",
        "query_id",
        "source_scenario_id",
        "template_id",
        "speed",
        "door_position",
        "goal_state",
        "cem_seed",
        "cem_rng_state_sha256_before",
    )
    mismatches = [
        field for field in fields if left.get(field) != right.get(field)
    ]
    if mismatches:
        raise RuntimeError(f"Metadata mismatch at {label}: {mismatches}")


def _configuration_root(
    config: Mapping[str, Any],
    configuration: str,
    *,
    repo_root: Path,
) -> Path:
    spec = config["configurations"][configuration]
    if configuration == "baseline":
        return resolve_contextworld_path(
            spec["source_root"], repo_root=repo_root
        )
    return resolve_contextworld_path(
        Path(config["artifacts"]["closed_loop"])
        / configuration
        / "h3_speedfull_s3072",
        repo_root=repo_root,
    )


def _load_closed_loop(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[
    dict[
        str,
        dict[str, dict[tuple[int, str], dict[str, Any]]],
    ],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
]:
    shared = config["shared_protocol"]
    seeds = [int(value) for value in shared["eval_seeds"]]
    per_seed = int(shared["evaluations_per_condition_per_seed"])
    expected_per_condition = len(seeds) * per_seed
    directions = [str(value) for value in shared["directional_evals"]]
    model_hash = str(config["frozen_inputs"]["model"]["sha256"])
    stable_commit = str(
        config["frozen_inputs"]["stable_worldmodel_commit"]
    )
    rows_by_configuration: dict[
        str,
        dict[str, dict[tuple[int, str], dict[str, Any]]],
    ] = {}
    raw_rows_by_configuration: dict[str, list[dict[str, Any]]] = {}
    inputs: list[dict[str, Any]] = []
    reference_schedules: dict[tuple[str, int], Any] = {}
    reference_metadata: dict[
        tuple[str, int, str, tuple[int, str]], Mapping[str, Any]
    ] = {}

    for configuration in CONFIGURATION_ORDER:
        spec = config["configurations"][configuration]
        root = _configuration_root(
            config, configuration, repo_root=repo_root
        )
        conditions: dict[
            str, dict[tuple[int, str], dict[str, Any]]
        ] = {condition: {} for condition in CONDITIONS}
        raw_rows_by_configuration[configuration] = []
        for direction in directions:
            if direction not in ("wrong_slow", "wrong_fast"):
                raise ValueError(f"Unexpected direction {direction}")
            for seed in seeds:
                path = root / f"{direction}_n50_s{seed}.json"
                payload = _load_passed(path)
                protocol = payload["protocol"]
                expected_protocol = {
                    "eval_seed": seed,
                    "eval_budget": int(
                        shared["eval_budget_raw_steps"]
                    ),
                    "action_block": int(
                        shared["action_block_raw_steps"]
                    ),
                    "horizon": int(
                        spec["horizon_action_blocks"]
                    ),
                    "receding_horizon": int(
                        shared["receding_horizon_action_blocks"]
                    ),
                    "cem_batch_size": int(shared["cem_batch_size"]),
                    "cem_num_samples": int(spec["cem_samples"]),
                    "cem_steps": int(spec["cem_iterations"]),
                    "cem_topk": int(spec["cem_topk"]),
                    "cem_var_scale": float(shared["cem_var_scale"]),
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
                if payload["checkpoint"]["sha256"] != model_hash:
                    raise RuntimeError(
                        f"Checkpoint hash mismatch in {path}"
                    )
                if (
                    payload["stable_worldmodel"]["commit"]
                    != stable_commit
                ):
                    raise RuntimeError(
                        f"StableWorldModel mismatch in {path}"
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
                records = list(payload["records"])
                if len(records) != 2 * per_seed:
                    raise RuntimeError(
                        f"Expected {2 * per_seed} records in {path}"
                    )
                raw_rows_by_configuration[configuration].extend(records)
                for raw_condition in ("correct", "wrong"):
                    selected = [
                        row
                        for row in records
                        if row["condition"] == raw_condition
                    ]
                    if len(selected) != per_seed:
                        raise RuntimeError(
                            f"Condition count mismatch in {path}"
                        )
                    condition = (
                        "correct"
                        if raw_condition == "correct"
                        else direction.removeprefix("wrong_")
                    )
                    if condition == "correct" and direction == "wrong_fast":
                        for row in selected:
                            key = _record_key(row)
                            canonical = conditions["correct"].get(key)
                            if canonical is None:
                                raise RuntimeError(
                                    f"Missing canonical correct row {key}"
                                )
                            _assert_metadata_equal(
                                canonical,
                                row,
                                label=f"{path}/{key}/correct",
                            )
                            for field in (
                                "success",
                                "final_distance",
                                "final_state",
                                "trajectory",
                                "cem_rng_state_sha256_after",
                            ):
                                if canonical[field] != row[field]:
                                    raise RuntimeError(
                                        "Correct duplicate differs at "
                                        f"{path}/{key}/{field}"
                                    )
                        continue
                    for row in selected:
                        key = _record_key(row)
                        if key in conditions[condition]:
                            raise RuntimeError(
                                f"Duplicate {condition} row at {path}/{key}"
                            )
                        trace = row.get("trajectory", {})
                        required_trace = (
                            "raw_steps_executed",
                            "states",
                            "actions",
                            "goal_distances",
                            "steps_to_success",
                            "path_length",
                            "normalized_distance_auc",
                        )
                        missing = [
                            name
                            for name in required_trace
                            if name not in trace
                        ]
                        if missing:
                            raise RuntimeError(
                                f"Missing trace fields in {path}: {missing}"
                            )
                        if int(trace["raw_steps_executed"]) > int(
                            shared["eval_budget_raw_steps"]
                        ):
                            raise RuntimeError(
                                f"Trace exceeds budget in {path}"
                            )
                        if bool(row["success"]) != (
                            trace["steps_to_success"] is not None
                        ):
                            raise RuntimeError(
                                f"Success/steps mismatch in {path}"
                            )
                        conditions[condition][key] = row
                        metadata_key = (
                            direction,
                            seed,
                            raw_condition,
                            key,
                        )
                        if metadata_key in reference_metadata:
                            _assert_metadata_equal(
                                reference_metadata[metadata_key],
                                row,
                                label=(
                                    f"{configuration}/{direction}/"
                                    f"{seed}/{key}"
                                ),
                            )
                        else:
                            reference_metadata[metadata_key] = row
                inputs.append(
                    {
                        "configuration": configuration,
                        "direction": direction,
                        "eval_seed": seed,
                        "path": portable_contextworld_path(
                            path, repo_root=repo_root
                        ),
                        "sha256": sha256_file(path),
                        "raw_records": len(records),
                    }
                )
        for condition, selected in conditions.items():
            if len(selected) != expected_per_condition:
                raise RuntimeError(
                    f"{configuration}/{condition}: expected "
                    f"{expected_per_condition}, got {len(selected)}"
                )
        keys = set(conditions["correct"])
        if any(set(conditions[name]) != keys for name in CONDITIONS):
            raise RuntimeError(
                f"Condition keys differ for {configuration}"
            )
        for key in sorted(keys):
            _assert_metadata_equal(
                conditions["correct"][key],
                conditions["slow"][key],
                label=f"{configuration}/{key}/correct-vs-slow",
            )
            _assert_metadata_equal(
                conditions["correct"][key],
                conditions["fast"][key],
                label=f"{configuration}/{key}/correct-vs-fast",
            )
        rows_by_configuration[configuration] = conditions

    audit = {
        "configurations": len(rows_by_configuration),
        "files": len(inputs),
        "raw_records": sum(row["raw_records"] for row in inputs),
        "raw_records_by_configuration": {
            name: len(rows)
            for name, rows in raw_rows_by_configuration.items()
        },
        "unique_condition_records": sum(
            len(rows)
            for configuration in rows_by_configuration.values()
            for rows in configuration.values()
        ),
        "condition_records_per_configuration": {
            name: {
                condition: len(rows)
                for condition, rows in configuration.items()
            }
            for name, configuration in rows_by_configuration.items()
        },
        "correct_direction_duplicates_verified": (
            len(CONFIGURATION_ORDER) * len(seeds) * per_seed
        ),
        "identical_query_schedules_across_configurations": True,
        "identical_starting_cem_seeds_across_configurations": True,
        "passed": True,
    }
    return (
        rows_by_configuration,
        inputs,
        audit,
        raw_rows_by_configuration,
    )


def _normalized_progress(row: Mapping[str, Any]) -> float:
    distances = np.asarray(
        row["trajectory"]["goal_distances"], dtype=np.float64
    )
    initial = float(distances[0])
    return (
        float((initial - distances[-1]) / initial)
        if initial > 0.0
        else 0.0
    )


def _paired_condition_values(
    conditions: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, Any]]
    ],
    left: str,
    right: str,
    extractor: Any,
) -> tuple[list[Any], list[Any]]:
    keys = sorted(conditions[left])
    if set(keys) != set(conditions[right]):
        raise RuntimeError(f"{left}/{right} keys differ")
    return (
        [extractor(conditions[left][key]) for key in keys],
        [extractor(conditions[right][key]) for key in keys],
    )


def _configuration_analysis(
    *,
    configuration: str,
    conditions: Mapping[
        str, Mapping[tuple[int, str], Mapping[str, Any]]
    ],
    raw_rows: list[dict[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    condition_summaries = {
        condition: summarize_closed_loop_condition(conditions[condition])
        for condition in CONDITIONS
    }
    success_comparisons = {}
    for left, right in (
        ("fast", "slow"),
        ("correct", "slow"),
        ("correct", "fast"),
    ):
        left_values, right_values = _paired_condition_values(
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
    continuous = {}
    metrics = {
        "final_distance_px": lambda row: float(row["final_distance"]),
        "normalized_progress": _normalized_progress,
        "normalized_distance_auc": lambda row: float(
            row["trajectory"]["normalized_distance_auc"]
        ),
    }
    for metric_index, (metric, extractor) in enumerate(metrics.items()):
        continuous[metric] = {}
        for left, right in (
            ("fast", "slow"),
            ("correct", "slow"),
            ("correct", "fast"),
        ):
            left_values, right_values = _paired_condition_values(
                conditions, left, right, extractor
            )
            continuous[metric][f"{left}_vs_{right}"] = (
                paired_continuous_comparison(
                    left_values,
                    right_values,
                    left_label=left,
                    right_label=right,
                    bootstrap_seed=(
                        82000
                        + CONFIGURATION_ORDER.index(configuration) * 1000
                        + metric_index * 100
                        + sum(map(ord, left + right))
                    ),
                )
            )
    common_steps = {}
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
        if keys:
            common_steps[f"{left}_vs_{right}"] = (
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
                        85000
                        + CONFIGURATION_ORDER.index(configuration) * 1000
                        + sum(map(ord, left + right))
                    ),
                )
            )
        else:
            common_steps[f"{left}_vs_{right}"] = {
                "pairs": 0,
                "status": "no_common_successes",
            }
    per_seed = {}
    for seed in sorted(
        {key[0] for key in conditions["correct"]}
    ):
        selected_keys = [
            key for key in sorted(conditions["correct"]) if key[0] == seed
        ]
        rates = {
            condition: (
                100.0
                * sum(
                    bool(conditions[condition][key]["success"])
                    for key in selected_keys
                )
                / len(selected_keys)
            )
            for condition in CONDITIONS
        }
        per_seed[str(seed)] = {
            "evaluations_per_condition": len(selected_keys),
            "success_rate_percent": rates,
            "fast_minus_slow_percentage_points": (
                rates["fast"] - rates["slow"]
            ),
            "fast_minus_correct_percentage_points": (
                rates["fast"] - rates["correct"]
            ),
        }
    get_cost_calls = int(
        sum(int(row["get_cost_calls"]) for row in raw_rows)
    )
    cem_solve_calls = int(
        sum(int(row["cem_solve_calls"]) for row in raw_rows)
    )
    samples = int(spec["cem_samples"])
    iterations = int(spec["cem_iterations"])
    horizon = int(spec["horizon_action_blocks"])
    return {
        "conditions": condition_summaries,
        "paired_success_comparisons": success_comparisons,
        "paired_continuous_comparisons": continuous,
        "paired_steps_to_success_on_common_successes": common_steps,
        "per_eval_seed": per_seed,
        "compute": {
            "raw_records": len(raw_rows),
            "elapsed_seconds": _continuous_summary(
                float(row["elapsed_seconds"]) for row in raw_rows
            ),
            "total_elapsed_gpu_seconds": float(
                sum(float(row["elapsed_seconds"]) for row in raw_rows)
            ),
            "cem_solve_calls": cem_solve_calls,
            "get_cost_calls": get_cost_calls,
            "candidate_cost_evaluations": get_cost_calls * samples,
            "candidate_rollout_blocks": (
                get_cost_calls * samples * horizon
            ),
            "expected_get_cost_calls_from_solver": (
                cem_solve_calls * iterations
            ),
            "get_cost_call_audit_passed": (
                get_cost_calls == cem_solve_calls * iterations
            ),
        },
    }


def _primary_effect_analysis(
    *,
    config: Mapping[str, Any],
    rows: Mapping[
        str,
        Mapping[
            str, Mapping[tuple[int, str], Mapping[str, Any]]
        ],
    ],
) -> dict[str, Any]:
    analysis = config["primary_analysis"]
    bootstrap_spec = analysis["bootstrap"]
    randomization_spec = analysis["randomization_test"]
    keys = sorted(rows["baseline"]["slow"])
    seeds = [key[0] for key in keys]
    gap_vectors = {
        configuration: np.asarray(
            [
                100.0
                * (
                    float(
                        bool(rows[configuration]["fast"][key]["success"])
                    )
                    - float(
                        bool(rows[configuration]["slow"][key]["success"])
                    )
                )
                for key in keys
            ],
            dtype=np.float64,
        )
        for configuration in CONFIGURATION_ORDER
    }
    gap_results = {}
    for index, configuration in enumerate(CONFIGURATION_ORDER):
        gap_results[configuration] = seed_stratified_bootstrap(
            gap_vectors[configuration],
            seeds,
            resamples=int(bootstrap_spec["resamples"]),
            random_seed=int(bootstrap_spec["seed"]) + index,
            confidence_level=float(
                bootstrap_spec["confidence_level"]
            ),
        )
    condition_changes = {}
    for variant_index, variant in enumerate(VARIANT_ORDER):
        condition_changes[variant] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            variant_values = [
                bool(rows[variant][condition][key]["success"])
                for key in keys
            ]
            baseline_values = [
                bool(rows["baseline"][condition][key]["success"])
                for key in keys
            ]
            differences = [
                100.0 * (float(left) - float(right))
                for left, right in zip(
                    variant_values, baseline_values
                )
            ]
            condition_changes[variant][condition] = {
                "paired_success": paired_binary_comparison(
                    variant_values,
                    baseline_values,
                    left_label=variant,
                    right_label="baseline",
                ),
                "success_rate_change_percentage_points": (
                    seed_stratified_bootstrap(
                        differences,
                        seeds,
                        resamples=int(bootstrap_spec["resamples"]),
                        random_seed=(
                            int(bootstrap_spec["seed"])
                            + 200
                            + variant_index * 10
                            + condition_index
                        ),
                        confidence_level=float(
                            bootstrap_spec["confidence_level"]
                        ),
                    )
                ),
                "role": (
                    "descriptive decomposition of the preregistered "
                    "Fast-minus-Slow estimand; not a separate decision gate"
                ),
            }
    changes = {}
    raw_p_values = {}
    baseline = gap_vectors["baseline"]
    for index, variant in enumerate(VARIANT_ORDER):
        difference = gap_vectors[variant] - baseline
        bootstrap = seed_stratified_bootstrap(
            difference,
            seeds,
            resamples=int(bootstrap_spec["resamples"]),
            random_seed=int(bootstrap_spec["seed"]) + 100 + index,
            confidence_level=float(
                bootstrap_spec["confidence_level"]
            ),
        )
        randomization = paired_sign_flip_test(
            difference,
            resamples=int(randomization_spec["resamples"]),
            random_seed=int(randomization_spec["seed"]) + index,
        )
        raw_p_values[variant] = randomization["two_sided_p"]
        per_seed = {}
        for seed in sorted(set(seeds)):
            selected = difference[
                np.asarray(seeds, dtype=np.int64) == seed
            ]
            per_seed[str(seed)] = float(np.mean(selected))
        changes[variant] = {
            "variant_minus_baseline_gap_percentage_points": (
                bootstrap
            ),
            "gap_reduction_percentage_points": float(
                -bootstrap["estimate"]
            ),
            "paired_randomization_test": randomization,
            "per_eval_seed_change_percentage_points": per_seed,
        }
    adjusted = holm_adjust(raw_p_values)
    reduction_threshold = float(
        analysis["practical_reduction_threshold_percentage_points"]
    )
    baseline_gap = float(gap_results["baseline"]["estimate"])
    primary_fraction = float(
        analysis["primary_driver_fraction_of_baseline_gap"]
    )
    equivalence_margin = float(
        analysis["practical_equivalence_margin_percentage_points"]
    )
    for variant in VARIANT_ORDER:
        change = changes[variant]
        interval = change[
            "variant_minus_baseline_gap_percentage_points"
        ]["confidence_interval"]
        reduction = float(change["gap_reduction_percentage_points"])
        adjusted_p = float(adjusted[variant])
        contributes = bool(
            reduction >= reduction_threshold
            and interval[1] < 0.0
            and adjusted_p
            < float(config["primary_analysis"]["multiplicity"]["alpha"])
        )
        gap_interval = gap_results[variant]["confidence_interval"]
        changes[variant]["holm_adjusted_p"] = adjusted_p
        changes[variant]["contributes"] = contributes
        changes[variant]["primary_driver"] = bool(
            contributes
            and reduction
            >= primary_fraction * abs(baseline_gap)
        )
        changes[variant]["gap_resolved_within_margin"] = bool(
            gap_interval[0] >= -equivalence_margin
            and gap_interval[1] <= equivalence_margin
        )
    return {
        "estimand": analysis["estimand"],
        "pairs": len(keys),
        "fast_minus_slow_gap_percentage_points": gap_results,
        "variant_effects": changes,
        "condition_success_change_from_baseline": condition_changes,
        "holm_family": list(VARIANT_ORDER),
        "raw_randomization_p_values": raw_p_values,
        "holm_adjusted_p_values": adjusted,
        "practical_reduction_threshold_percentage_points": (
            reduction_threshold
        ),
        "primary_driver_threshold_percentage_points": (
            primary_fraction * abs(baseline_gap)
        ),
        "practical_equivalence_margin_percentage_points": (
            equivalence_margin
        ),
    }


def _load_long_horizon_prediction(
    *,
    config: Mapping[str, Any],
    repo_root: Path,
) -> tuple[
    dict[tuple[int, str], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    spec = config["long_horizon_prediction_diagnostic"]
    root = resolve_contextworld_path(
        Path(config["artifacts"]["long_horizon_prediction"])
        / "h3_speedfull_s3072",
        repo_root=repo_root,
    )
    model_hash = str(config["frozen_inputs"]["model"]["sha256"])
    normalizer_hash = str(
        config["frozen_inputs"]["normalizer"]["sha256"]
    )
    stable_commit = str(
        config["frozen_inputs"]["stable_worldmodel_commit"]
    )
    expected_per_seed = int(spec["evaluations_per_seed"])
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    inputs = []
    for seed_value in spec["eval_seeds"]:
        seed = int(seed_value)
        path = root / f"long_horizon_prediction_n50_s{seed}.json"
        payload = _load_passed(path)
        if payload["model"]["sha256"] != model_hash:
            raise RuntimeError(f"Prediction model mismatch in {path}")
        if payload["normalizer"]["sha256"] != normalizer_hash:
            raise RuntimeError(
                f"Prediction normalizer mismatch in {path}"
            )
        if (
            payload["stable_worldmodel"]["commit"]
            != stable_commit
        ):
            raise RuntimeError(
                f"Prediction StableWorldModel mismatch in {path}"
            )
        protocol = payload["protocol"]
        if (
            int(protocol["prediction_horizon_raw_steps"]) != 50
            or list(protocol["observed_raw_steps"]) != list(RAW_STEPS)
            or int(protocol["evaluations"]) != expected_per_seed
        ):
            raise RuntimeError(
                f"Prediction protocol mismatch in {path}"
            )
        records = list(payload["records"])
        if len(records) != expected_per_seed:
            raise RuntimeError(
                f"Prediction count mismatch in {path}"
            )
        for row in records:
            key = _record_key(row)
            if key in rows:
                raise RuntimeError(
                    f"Duplicate prediction record {key}"
                )
            if not row["probe"]["last_25_raw_actions_are_zero"]:
                raise RuntimeError(f"Non-zero tail in {path}/{key}")
            for condition in CONDITIONS:
                values = row["conditions"][condition][
                    "prediction_mse_to_true_by_block"
                ]
                if len(values) != len(RAW_STEPS):
                    raise RuntimeError(
                        f"Prediction horizon mismatch in {path}/{key}"
                    )
            rows[key] = row
        inputs.append(
            {
                "eval_seed": seed,
                "path": portable_contextworld_path(
                    path, repo_root=repo_root
                ),
                "sha256": sha256_file(path),
                "records": len(records),
            }
        )
    expected = int(spec["records"])
    if len(rows) != expected:
        raise RuntimeError(
            f"Expected {expected} prediction rows, got {len(rows)}"
        )
    audit = {
        "files": len(inputs),
        "records": len(rows),
        "context_evaluations": len(rows) * len(CONDITIONS),
        "observed_raw_steps": list(RAW_STEPS),
        "all_zero_action_tails_verified": True,
        "passed": True,
    }
    return rows, inputs, audit


def _prediction_analysis(
    rows: Mapping[tuple[int, str], Mapping[str, Any]]
) -> dict[str, Any]:
    keys = sorted(rows)
    by_horizon = {}
    for block_index, raw_step in enumerate(RAW_STEPS):
        condition_values = {
            condition: [
                float(
                    rows[key]["conditions"][condition][
                        "prediction_mse_to_true_by_block"
                    ][block_index]
                )
                for key in keys
            ]
            for condition in CONDITIONS
        }
        by_horizon[str(raw_step)] = {
            "conditions": {
                condition: _continuous_summary(values)
                for condition, values in condition_values.items()
            },
            "correct_vs_slow": paired_continuous_comparison(
                condition_values["correct"],
                condition_values["slow"],
                left_label="correct",
                right_label="slow",
                bootstrap_seed=91000 + raw_step,
            ),
            "correct_vs_fast": paired_continuous_comparison(
                condition_values["correct"],
                condition_values["fast"],
                left_label="correct",
                right_label="fast",
                bootstrap_seed=92000 + raw_step,
            ),
        }
    groups = {
        "first_25": tuple(range(0, 5)),
        "last_25": tuple(range(5, 10)),
        "full_50": tuple(range(0, 10)),
    }
    group_results = {}
    for group_index, (group, indices) in enumerate(groups.items()):
        condition_values = {
            condition: [
                float(
                    np.mean(
                        [
                            rows[key]["conditions"][condition][
                                "prediction_mse_to_true_by_block"
                            ][index]
                            for index in indices
                        ]
                    )
                )
                for key in keys
            ]
            for condition in CONDITIONS
        }
        group_results[group] = {
            "raw_steps": [RAW_STEPS[index] for index in indices],
            "conditions": {
                condition: _continuous_summary(values)
                for condition, values in condition_values.items()
            },
            "correct_vs_slow": paired_continuous_comparison(
                condition_values["correct"],
                condition_values["slow"],
                left_label="correct",
                right_label="slow",
                bootstrap_seed=93000 + group_index,
            ),
            "correct_vs_fast": paired_continuous_comparison(
                condition_values["correct"],
                condition_values["fast"],
                left_label="correct",
                right_label="fast",
                bootstrap_seed=94000 + group_index,
            ),
        }
    terminal_reachable = [
        float(
            row["probe"]["exact_correct_speed_final_distance_at_25"]
        )
        < 16.0
        for row in rows.values()
    ]
    late = group_results["last_25"]
    correct_slow_interval = late["correct_vs_slow"][
        "correct_minus_slow"
    ]["evaluation_bootstrap_95_ci"]
    correct_fast_interval = late["correct_vs_fast"][
        "correct_minus_fast"
    ]["evaluation_bootstrap_95_ci"]
    return {
        "metric": "native_latent_mse_to_true_query_speed_rollout",
        "by_horizon_raw_steps": by_horizon,
        "aggregate_over_horizons": group_results,
        "probe_exact_correct_speed_terminal_reachability": {
            "success_radius_px": 16.0,
            "successes": int(sum(terminal_reachable)),
            "records": len(terminal_reachable),
            "rate_percent": (
                100.0 * sum(terminal_reachable) / len(terminal_reachable)
            ),
        },
        "late_horizon_correct_lower_than_slow": bool(
            correct_slow_interval[1] < 0.0
        ),
        "late_horizon_correct_lower_than_fast": bool(
            correct_fast_interval[1] < 0.0
        ),
    }


def _stage_conclusion(
    primary: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, Any]:
    effects = primary["variant_effects"]
    horizon = bool(effects["horizon10"]["contributes"])
    samples = bool(effects["samples600"]["contributes"])
    iterations = bool(effects["iterations60"]["contributes"])
    search = samples or iterations
    primary_drivers = [
        name
        for name in VARIANT_ORDER
        if effects[name]["primary_driver"]
    ]
    resolved = [
        name
        for name in VARIANT_ORDER
        if effects[name]["gap_resolved_within_margin"]
    ]
    if horizon and search:
        resource_result = "both_lookahead_and_search_contribute"
    elif horizon:
        resource_result = "limited_lookahead_contributes"
    elif search:
        resource_result = "insufficient_cem_search_contributes"
    else:
        resource_result = "tested_resource_increases_do_not_explain_gap"
    if (
        not horizon
        and not search
        and prediction["late_horizon_correct_lower_than_slow"]
    ):
        remaining = (
            "model_side_slow_context_prediction_mismatch_supported; "
            "cost_and_untested_replanning_remain_unseparated"
        )
    else:
        remaining = (
            "interpret_with_closed_loop_effects_and_long_horizon_prediction"
        )
    return {
        "lookahead_contributes": horizon,
        "search_width_contributes": samples,
        "search_depth_contributes": iterations,
        "resource_mechanism_result": resource_result,
        "primary_driver_configurations": primary_drivers,
        "gap_resolved_configurations": resolved,
        "late_horizon_correct_prediction_lower_than_slow": prediction[
            "late_horizon_correct_lower_than_slow"
        ],
        "late_horizon_correct_prediction_lower_than_fast": prediction[
            "late_horizon_correct_lower_than_fast"
        ],
        "remaining_interpretation": remaining,
        "evidence_boundary": (
            "single training seed, Validation mechanism evidence; "
            "not final Test"
        ),
    }


def run_analysis(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    repo_root: Path,
    write_output: bool = True,
) -> dict[str, Any]:
    frozen = config["frozen_inputs"]
    verified_inputs = {
        "model": _verified_path(
            frozen["model"], repo_root=repo_root, label="model"
        ),
        "normalizer": _verified_path(
            frozen["normalizer"],
            repo_root=repo_root,
            label="normalizer",
        ),
        "wrong_slow_catalog": _verified_path(
            frozen["catalogs"]["wrong_slow"],
            repo_root=repo_root,
            label="wrong_slow catalog",
        ),
        "wrong_fast_catalog": _verified_path(
            frozen["catalogs"]["wrong_fast"],
            repo_root=repo_root,
            label="wrong_fast catalog",
        ),
        "prior_mechanism_summary": _verified_path(
            frozen["prior_mechanism_summary"],
            repo_root=repo_root,
            label="prior mechanism summary",
        ),
    }
    (
        closed_rows,
        closed_inputs,
        closed_audit,
        raw_rows,
    ) = _load_closed_loop(config=config, repo_root=repo_root)
    configuration_results = {
        name: _configuration_analysis(
            configuration=name,
            conditions=closed_rows[name],
            raw_rows=raw_rows[name],
            spec=config["configurations"][name],
        )
        for name in CONFIGURATION_ORDER
    }
    primary = _primary_effect_analysis(
        config=config,
        rows=closed_rows,
    )
    prediction_rows, prediction_inputs, prediction_audit = (
        _load_long_horizon_prediction(
            config=config, repo_root=repo_root
        )
    )
    prediction = _prediction_analysis(prediction_rows)
    stage_conclusion = _stage_conclusion(primary, prediction)
    output_path = resolve_contextworld_path(
        config["artifacts"]["final_summary"],
        repo_root=repo_root,
    )
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "evidence_level": "validation_planner_mechanism",
        "preregistration": {
            "path": portable_contextworld_path(
                config_path, repo_root=repo_root
            ),
            "sha256": sha256_file(config_path),
            "status": config["status"],
            "date": str(config["preregistered_date"]),
        },
        "protocol_and_count_audit": {
            "closed_loop": closed_audit,
            "long_horizon_prediction": prediction_audit,
            "full_50x6_per_eval_and_condition": True,
            "passed": (
                closed_audit["passed"] and prediction_audit["passed"]
            ),
        },
        "closed_loop_by_configuration": configuration_results,
        "primary_fast_minus_slow_effect": primary,
        "long_horizon_prediction_diagnostic": prediction,
        "stage_conclusion": stage_conclusion,
        "inputs": {
            "frozen": {
                label: {
                    "path": portable_contextworld_path(
                        path, repo_root=repo_root
                    ),
                    "sha256": sha256_file(path),
                }
                for label, path in verified_inputs.items()
            },
            "closed_loop": closed_inputs,
            "long_horizon_prediction": prediction_inputs,
        },
        "output": portable_contextworld_path(
            output_path, repo_root=repo_root
        ),
    }
    if write_output:
        write_json(output_path, result)
    return result


__all__ = [
    "CONFIGURATION_ORDER",
    "CONDITIONS",
    "VARIANT_ORDER",
    "holm_adjust",
    "load_config",
    "paired_sign_flip_test",
    "run_analysis",
    "seed_stratified_bootstrap",
]
