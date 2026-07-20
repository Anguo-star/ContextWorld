#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


HORIZONS = (1, 2, 3, 5, 10)


def _load_passed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Result did not pass: {path}")
    return payload


def _recover_speed_from_context(
    *,
    states: np.ndarray,
    next_states: np.ndarray,
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if (
        states.shape != next_states.shape
        or states.ndim != 2
        or states.shape[-1] != 2
        or actions.shape != (states.shape[0], 5, 2)
    ):
        raise ValueError(
            "Expected states/next_states [T,2] and actions [T,5,2]"
        )
    block_actions = np.sum(actions, axis=1)
    denominators = np.sum(block_actions**2, axis=-1)
    if np.any(denominators <= 1.0e-12):
        raise ValueError("Context contains a zero-information action block")
    displacements = next_states - states
    estimates = (
        np.sum(displacements * block_actions, axis=-1) / denominators
    )
    residuals = np.linalg.norm(
        displacements - estimates[:, None] * block_actions,
        axis=-1,
    )
    return estimates, residuals


def _context_identifiability_audit(catalog_path: Path) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    speed_errors = []
    motion_residuals = []
    visible_field_checks = []
    query_hashes: dict[str, set[str]] = defaultdict(set)
    payloads = set()
    for bundle in catalog["bundles"]:
        visible_field_checks.append(
            set(bundle.get("model_visible_fields", []))
            == {"pixels", "action"}
        )
        query_hashes[str(bundle["static_query_id"])].add(
            str(bundle["query_pixels_sha256"])
        )
        payload_path = resolve_contextworld_path(
            str(bundle["payload"]), repo_root=ROOT
        )
        payloads.add(str(payload_path))
        with np.load(payload_path, allow_pickle=False) as payload:
            for condition, condition_row in bundle["conditions"].items():
                expected_speed = float(
                    condition_row["factors"]["agent.speed"]
                )
                estimates, residuals = _recover_speed_from_context(
                    states=payload[
                        f"context_b2_{condition}_states"
                    ],
                    next_states=payload[
                        f"context_b2_{condition}_next_states"
                    ],
                    actions=payload[
                        f"context_b2_{condition}_actions"
                    ],
                )
                speed_errors.extend(
                    np.abs(estimates - expected_speed).tolist()
                )
                motion_residuals.extend(residuals.tolist())
    static_pixels_identical = all(
        len(hashes) == 1 for hashes in query_hashes.values()
    )
    maximum_speed_error = float(max(speed_errors))
    maximum_motion_residual = float(max(motion_residuals))
    passed = bool(
        speed_errors
        and maximum_speed_error <= 1.0e-4
        and maximum_motion_residual <= 1.0e-4
        and static_pixels_identical
        and all(visible_field_checks)
    )
    return {
        "catalog": str(catalog_path),
        "track": str(catalog["track"]),
        "bundles": len(catalog["bundles"]),
        "payloads": len(payloads),
        "context_transitions": len(speed_errors),
        "speed_recovery_mae": float(np.mean(speed_errors)),
        "maximum_speed_recovery_error": maximum_speed_error,
        "maximum_free_motion_residual_px": maximum_motion_residual,
        "static_query_pixels_identical_across_query_speeds": (
            static_pixels_identical
        ),
        "model_input_excludes_speed_and_state": all(
            visible_field_checks
        ),
        "passed": passed,
    }


def _exact_sign_test(positive: int, negative: int) -> float:
    trials = int(positive + negative)
    if trials == 0:
        return 1.0
    tail = min(int(positive), int(negative))
    probability = sum(
        math.comb(trials, value) for value in range(tail + 1)
    ) / (2.0**trials)
    return float(min(1.0, 2.0 * probability))


def _bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    resamples: int,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        raise ValueError("Cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    means = np.mean(array[indices], axis=1)
    return [
        float(value) for value in np.percentile(means, [2.5, 97.5])
    ]


def _holm_adjust(rows: list[dict[str, Any]], *, alpha: float) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float(
            item[1]["cluster_sign_test_two_sided_p"]
        ),
    )
    running = 0.0
    total = len(ordered)
    for rank, (_, row) in enumerate(ordered):
        raw = float(row["cluster_sign_test_two_sided_p"])
        adjusted = min(1.0, (total - rank) * raw)
        running = max(running, adjusted)
        row["holm_adjusted_p"] = float(running)
        row["holm_passed"] = bool(running <= alpha)


def _contrast(
    records: list[dict[str, Any]],
    *,
    same_condition: str,
    other_condition: str,
    value: Callable[[dict[str, Any], str], float],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    differences = [
        float(value(record, other_condition))
        - float(value(record, same_condition))
        for record in records
    ]
    by_seed: dict[int, list[float]] = defaultdict(list)
    by_unit: dict[str, list[float]] = defaultdict(list)
    for record, difference in zip(records, differences):
        seed = int(record["eval_seed"])
        by_seed[seed].append(difference)
        family = str(
            record.get("action_probe", {}).get("family", "shared")
        )
        unit = f"{record['static_query_id']}:{family}"
        by_unit[unit].append(difference)
    unit_effects = [
        float(np.mean(values)) for values in by_unit.values()
    ]
    positive = sum(value > 1e-12 for value in unit_effects)
    negative = sum(value < -1e-12 for value in unit_effects)
    by_seed_means = {
        str(seed): float(np.mean(values))
        for seed, values in sorted(by_seed.items())
    }
    return {
        "other_minus_same_mean": float(np.mean(differences)),
        "bootstrap_95_ci": _bootstrap_ci(
            unit_effects,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
        "eval_seed_means": by_seed_means,
        "positive_eval_seeds": int(
            sum(value > 0.0 for value in by_seed_means.values())
        ),
        "eval_seeds": len(by_seed_means),
        "cluster_units": len(unit_effects),
        "positive_units": positive,
        "negative_units": negative,
        "ties": len(unit_effects) - positive - negative,
        "cluster_sign_test_two_sided_p": _exact_sign_test(
            positive, negative
        ),
        "passed_directional_stability": bool(
            np.mean(differences) > 0.0
            and all(value > 0.0 for value in by_seed_means.values())
        ),
    }


def _condition_names(
    records: list[dict[str, Any]],
) -> tuple[str, list[str]]:
    first = records[0]["conditions"]
    same = [
        name
        for name, row in first.items()
        if row["history_relation"] == "same"
    ]
    if len(same) != 1:
        raise RuntimeError(f"Expected one same-speed history: {same}")
    others = [name for name in first if name != same[0]]
    if len(others) != 2:
        raise RuntimeError(f"Expected two other histories: {others}")
    return same[0], others


def _physical_value(
    metric: str,
    *,
    horizon: int | None,
) -> Callable[[dict[str, Any], str], float]:
    def value(record: dict[str, Any], condition: str) -> float:
        rows = record["conditions"][condition]["by_horizon"]
        if horizon is not None:
            return float(rows[str(horizon)][metric])
        return float(
            np.mean(
                [float(rows[str(step)][metric]) for step in HORIZONS]
            )
        )

    return value


def _physical_row_summary(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    same, others = _condition_names(records)
    condition_means = {}
    for condition in records[0]["conditions"]:
        history_speed = float(
            records[0]["conditions"][condition]["history_speed"]
        )
        condition_means[condition] = {
            "history_speed": history_speed,
            "history_relation": records[0]["conditions"][condition][
                "history_relation"
            ],
            "by_horizon": {
                str(horizon): {
                    **{
                        metric: float(
                            np.mean(
                                [
                                    record["conditions"][condition][
                                        "by_horizon"
                                    ][str(horizon)][metric]
                                    for record in records
                                ]
                            )
                        )
                        for metric in (
                            "inferred_speed",
                            "position_error_px",
                            "displacement_magnitude_error_px",
                            "displacement_direction_error_deg",
                            "latent_mse_to_true_query_future",
                            "latent_mse_to_nearest_oracle",
                        )
                    },
                    "predicted_displacement_magnitude_px": float(
                        np.mean(
                            [
                                np.linalg.norm(
                                    record["conditions"][condition][
                                        "by_horizon"
                                    ][str(horizon)][
                                        "predicted_displacement"
                                    ]
                                )
                                for record in records
                            ]
                        )
                    ),
                    "true_displacement_magnitude_px": float(
                        np.mean(
                            [
                                np.linalg.norm(
                                    record["conditions"][condition][
                                        "by_horizon"
                                    ][str(horizon)][
                                        "true_displacement"
                                    ]
                                )
                                for record in records
                            ]
                        )
                    ),
                    "inferred_speed_mae_to_query": float(
                        np.mean(
                            [
                                abs(
                                    record["conditions"][condition][
                                        "by_horizon"
                                    ][str(horizon)][
                                        "inferred_minus_query_speed"
                                    ]
                                )
                                for record in records
                            ]
                        )
                    ),
                    "inferred_speed_mae_to_history": float(
                        np.mean(
                            [
                                abs(
                                    record["conditions"][condition][
                                        "by_horizon"
                                    ][str(horizon)][
                                        "inferred_minus_history_speed"
                                    ]
                                )
                                for record in records
                            ]
                        )
                    ),
                }
                for horizon in HORIZONS
            },
        }

    comparisons = {}
    for other_index, other in enumerate(others):
        comparison_seed = bootstrap_seed + 100 * other_index
        comparisons[other] = {
            "one_block_position": _contrast(
                records,
                same_condition=same,
                other_condition=other,
                value=_physical_value(
                    "position_error_px", horizon=1
                ),
                bootstrap_seed=comparison_seed + 1,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "one_block_displacement_magnitude": _contrast(
                records,
                same_condition=same,
                other_condition=other,
                value=_physical_value(
                    "displacement_magnitude_error_px", horizon=1
                ),
                bootstrap_seed=comparison_seed + 2,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "trajectory_position": _contrast(
                records,
                same_condition=same,
                other_condition=other,
                value=_physical_value(
                    "position_error_px", horizon=None
                ),
                bootstrap_seed=comparison_seed + 3,
                bootstrap_resamples=bootstrap_resamples,
            ),
            "trajectory_latent": _contrast(
                records,
                same_condition=same,
                other_condition=other,
                value=_physical_value(
                    "latent_mse_to_true_query_future",
                    horizon=None,
                ),
                bootstrap_seed=comparison_seed + 4,
                bootstrap_resamples=bootstrap_resamples,
            ),
        }

    low = min(
        condition_means,
        key=lambda name: condition_means[name]["history_speed"],
    )
    high = max(
        condition_means,
        key=lambda name: condition_means[name]["history_speed"],
    )
    response_by_horizon = {}
    history_speed_span = float(
        condition_means[high]["history_speed"]
        - condition_means[low]["history_speed"]
    )
    if history_speed_span <= 0.0:
        raise RuntimeError("History speed span must be positive")
    for horizon in HORIZONS:
        differences = [
            float(
                record["conditions"][high]["by_horizon"][
                    str(horizon)
                ]["inferred_speed"]
            )
            - float(
                record["conditions"][low]["by_horizon"][
                    str(horizon)
                ]["inferred_speed"]
            )
            for record in records
        ]
        displacement_differences = [
            float(
                np.linalg.norm(
                    record["conditions"][high]["by_horizon"][
                        str(horizon)
                    ]["predicted_displacement"]
                )
            )
            - float(
                np.linalg.norm(
                    record["conditions"][low]["by_horizon"][
                        str(horizon)
                    ]["predicted_displacement"]
                )
            )
            for record in records
        ]
        by_seed: dict[int, list[float]] = defaultdict(list)
        displacement_by_seed: dict[int, list[float]] = defaultdict(list)
        for record, difference, displacement_difference in zip(
            records, differences, displacement_differences
        ):
            by_seed[int(record["eval_seed"])].append(difference)
            displacement_by_seed[int(record["eval_seed"])].append(
                displacement_difference
            )
        seed_means = {
            str(seed): float(np.mean(values))
            for seed, values in sorted(by_seed.items())
        }
        displacement_seed_means = {
            str(seed): float(np.mean(values))
            for seed, values in sorted(displacement_by_seed.items())
        }
        response_by_horizon[str(horizon)] = {
            "high_minus_low_inferred_speed": float(
                np.mean(differences)
            ),
            "history_speed_span": history_speed_span,
            "inferred_speed_response_gain": float(
                np.mean(differences) / history_speed_span
            ),
            "high_minus_low_predicted_displacement_px": float(
                np.mean(displacement_differences)
            ),
            "eval_seed_means": seed_means,
            "predicted_displacement_eval_seed_means": (
                displacement_seed_means
            ),
            "positive_eval_seeds": int(
                sum(value > 0.0 for value in seed_means.values())
            ),
            "passed_directional_stability": bool(
                np.mean(differences) > 0.0
                and all(value > 0.0 for value in seed_means.values())
            ),
        }
    one_block_gate = all(
        row["one_block_position"]["passed_directional_stability"]
        and row["one_block_displacement_magnitude"][
            "passed_directional_stability"
        ]
        for row in comparisons.values()
    )
    trajectory_gate = all(
        row["trajectory_position"]["passed_directional_stability"]
        and row["trajectory_latent"]["passed_directional_stability"]
        for row in comparisons.values()
    )
    return {
        "evaluations_per_condition": len(records),
        "same_speed_condition": same,
        "other_conditions": others,
        "condition_means": condition_means,
        "same_speed_benefit": comparisons,
        "history_speed_response": response_by_horizon,
        "gates": {
            "one_block_same_speed_lowest": one_block_gate,
            "trajectory_same_speed_lowest": trajectory_gate,
            "history_response_at_one_block": response_by_horizon["1"][
                "passed_directional_stability"
            ],
            "passed": bool(
                one_block_gate
                and trajectory_gate
                and response_by_horizon["1"][
                    "passed_directional_stability"
                ]
            ),
        },
    }


def _fixed_row_summary(
    records: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    same, others = _condition_names(records)

    def regret(record: dict[str, Any], condition: str) -> float:
        return float(
            record["conditions"][condition][
                "exact_query_dynamics_regret_px"
            ]
        )

    comparisons = {
        other: _contrast(
            records,
            same_condition=same,
            other_condition=other,
            value=regret,
            bootstrap_seed=bootstrap_seed + index,
            bootstrap_resamples=bootstrap_resamples,
        )
        for index, other in enumerate(others)
    }
    return {
        "evaluations_per_condition": len(records),
        "same_speed_condition": same,
        "condition_means": {
            condition: {
                "history_speed": float(
                    records[0]["conditions"][condition][
                        "history_speed"
                    ]
                ),
                "mean_exact_query_dynamics_regret_px": float(
                    np.mean(
                        [
                            regret(record, condition)
                            for record in records
                        ]
                    )
                ),
                "success_rate": float(
                    np.mean(
                        [
                            bool(
                                record["conditions"][condition][
                                    "selected_true_success"
                                ]
                            )
                            for record in records
                        ]
                    )
                ),
            }
            for condition in records[0]["conditions"]
        },
        "same_speed_benefit": comparisons,
        "gates": {
            "same_speed_lowest_regret": all(
                row["passed_directional_stability"]
                for row in comparisons.values()
            )
        },
    }


def _planning_row_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition"])].append(record)
    return {
        "evaluations_per_condition": len(next(iter(grouped.values()))),
        "conditions": {
            condition: {
                "history_speed": float(rows[0]["history_speed"]),
                "history_relation": rows[0]["history_relation"],
                "success_rate": float(
                    np.mean([bool(row["success"]) for row in rows])
                ),
                "mean_final_distance_px": float(
                    np.mean(
                        [float(row["final_distance"]) for row in rows]
                    )
                ),
                "mean_normalized_distance_auc": float(
                    np.mean(
                        [
                            float(
                                row["trajectory"][
                                    "normalized_distance_auc"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),
                "deadline_success_curve": {
                    budget: float(
                        np.mean(
                            [
                                bool(
                                    row["trajectory"][
                                        "success_by_budget_raw_steps"
                                    ][budget]
                                )
                                for row in rows
                            ]
                        )
                    )
                    for budget in rows[0]["trajectory"][
                        "success_by_budget_raw_steps"
                    ]
                },
            }
            for condition, rows in sorted(grouped.items())
        },
    }


def _result_files(
    *,
    root: Path,
    slug: str,
    track: str,
    query_speed: float,
    eval_seeds: list[int],
) -> list[Path]:
    return [
        root
        / slug
        / track
        / f"q{float(query_speed):g}"
        / f"s{seed}.json"
        for seed in eval_seeds
    ]


def _records(
    paths: list[Path],
    *,
    checkpoint_sha256: str,
    mode: str,
    num_eval: int,
) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        payload = _load_passed(path)
        if payload["model"]["sha256"] != checkpoint_sha256:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        expected = num_eval * 3 if mode == "planning" else num_eval
        if int(payload["count_audit"]["records"]) != expected:
            raise RuntimeError(f"Count mismatch: {path}")
        result.extend(payload["records"])
    return result


def _ability_records(
    *,
    root: Path,
    slug: str,
    domain: str,
    eval_seeds: list[int],
    checkpoint_sha256: str,
    num_eval: int,
) -> list[dict[str, Any]]:
    records = []
    for seed in eval_seeds:
        path = root / slug / domain / f"s{seed}.json"
        if not path.is_file() and slug == "h3_origheldout_s3072":
            legacy_stem = {
                "original_heldout": "planning_original_heldout",
                "speed5_matched": "planning_speed5_matched",
            }[domain]
            path = (
                root.parent.parent
                / "original_ability_reconstruction"
                / slug
                / f"{legacy_stem}_s{seed}.json"
            )
        payload = _load_passed(path)
        if payload["checkpoint"]["sha256"] != checkpoint_sha256:
            raise RuntimeError(
                f"Ability checkpoint hash mismatch: {path}"
            )
        if (
            int(payload["protocol"]["eval_seed"]) != seed
            or int(payload["protocol"]["evaluations"]) != num_eval
        ):
            raise RuntimeError(f"Ability protocol mismatch: {path}")
        records.extend(payload["raw_records"])
    return records


def _ability_comparison(
    candidate: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    def lookup(
        rows: list[dict[str, Any]],
    ) -> dict[tuple[int, str], dict[str, Any]]:
        result = {
            (int(row["eval_seed"]), str(row["evaluation_id"])): row
            for row in rows
        }
        if len(result) != len(rows):
            raise RuntimeError("Duplicate ability evaluation IDs")
        return result

    candidate_by_key = lookup(candidate)
    reference_by_key = lookup(reference)
    if candidate_by_key.keys() != reference_by_key.keys():
        raise RuntimeError("Ability evaluation schedules differ")
    success_differences = []
    distance_differences = []
    for key in sorted(candidate_by_key):
        left = candidate_by_key[key]
        right = reference_by_key[key]
        for field in (
            "initial_state",
            "goal_state",
            "room_relation",
            "source_kind",
            "source_path",
            "episode",
            "start_step",
        ):
            if left[field] != right[field]:
                raise RuntimeError(
                    f"Ability pairing differs at {key}/{field}"
                )
        success_differences.append(
            float(bool(left["success"])) - float(bool(right["success"]))
        )
        distance_differences.append(
            float(left["final_distance"])
            - float(right["final_distance"])
        )
    success_ci = [
        100.0 * value
        for value in _bootstrap_ci(
            success_differences,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        )
    ]
    distance_ci = _bootstrap_ci(
        distance_differences,
        seed=bootstrap_seed + 1,
        resamples=bootstrap_resamples,
    )
    strata = {}
    for room_relation in sorted(
        {str(row["room_relation"]) for row in reference}
    ):
        candidate_rows = [
            row
            for row in candidate
            if row["room_relation"] == room_relation
        ]
        reference_rows = [
            row
            for row in reference
            if row["room_relation"] == room_relation
        ]
        reference_successes = sum(
            bool(row["success"]) for row in reference_rows
        )
        candidate_successes = sum(
            bool(row["success"]) for row in candidate_rows
        )
        strata[room_relation] = {
            "evaluations": len(reference_rows),
            "reference_successes": int(reference_successes),
            "candidate_successes": int(candidate_successes),
            "solvable_stratum_collapsed": bool(
                reference_successes > 0 and candidate_successes == 0
            ),
        }
    gates = {
        "success_ci_lower_at_least_minus_5pp": success_ci[0] >= -5.0,
        "final_distance_ci_upper_at_most_plus_5px": (
            distance_ci[1] <= 5.0
        ),
        "no_solvable_stratum_collapse": not any(
            row["solvable_stratum_collapsed"]
            for row in strata.values()
        ),
    }
    return {
        "paired_evaluations": len(success_differences),
        "candidate_minus_reference_success_rate_points": float(
            100.0 * np.mean(success_differences)
        ),
        "success_difference_bootstrap_95_ci_points": success_ci,
        "candidate_minus_reference_mean_final_distance_px": float(
            np.mean(distance_differences)
        ),
        "final_distance_difference_bootstrap_95_ci_px": distance_ci,
        "strata": strata,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _mean_physical_benefit(track: dict[str, Any]) -> float:
    values = []
    for row in track["by_query_speed"].values():
        for comparison in row["physical"]["same_speed_benefit"].values():
            values.append(
                comparison["one_block_position"][
                    "other_minus_same_mean"
                ]
            )
    return float(np.mean(values))


def _mean_displacement_calibration_benefit(
    track: dict[str, Any],
) -> float:
    values = []
    for row in track["by_query_speed"].values():
        for comparison in row["physical"]["same_speed_benefit"].values():
            values.append(
                comparison["one_block_displacement_magnitude"][
                    "other_minus_same_mean"
                ]
            )
    return float(np.mean(values))


def _mean_history_response(track: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                row["physical"]["history_speed_response"]["1"][
                    "high_minus_low_inferred_speed"
                ]
                for row in track["by_query_speed"].values()
            ]
        )
    )


def _mean_history_response_gain(track: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                row["physical"]["history_speed_response"]["1"][
                    "inferred_speed_response_gain"
                ]
                for row in track["by_query_speed"].values()
            ]
        )
    )


def _mean_displacement_response(track: dict[str, Any]) -> float:
    return float(
        np.mean(
            [
                row["physical"]["history_speed_response"]["1"][
                    "high_minus_low_predicted_displacement_px"
                ]
                for row in track["by_query_speed"].values()
            ]
        )
    )


def _mean_fixed_benefit(track: dict[str, Any]) -> float:
    values = []
    for row in track["by_query_speed"].values():
        for comparison in row["fixed_candidate"][
            "same_speed_benefit"
        ].values():
            values.append(comparison["other_minus_same_mean"])
    return float(np.mean(values))


def _member_scalar_summary(
    values: dict[str, float],
) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "by_model": {
            slug: float(value) for slug, value in values.items()
        },
    }


def _model_group_summaries(
    model_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(
        list
    )
    for slug, row in model_results.items():
        grouped[str(row["model_group"])].append((slug, row))
    display_names = {
        "original_reference": "原始单速基线",
        "single_speed_control": "匹配单速控制",
        "multi_speed_target": "匹配多速度模型",
    }
    result = {}
    for group, members in grouped.items():
        first = members[0][1]
        tracks = {}
        for track in first["tracks"]:
            first_track = first["tracks"][track]
            summary_effects = {
                key: _member_scalar_summary(
                    {
                        slug: row["tracks"][track][
                            "summary_effects"
                        ][key]
                        for slug, row in members
                    }
                )
                for key in first_track["summary_effects"]
            }
            gates = {
                key: {
                    "passed_models": int(
                        sum(
                            bool(
                                row["tracks"][track]["gates"][key]
                            )
                            for _, row in members
                        )
                    ),
                    "models": len(members),
                }
                for key in first_track["gates"]
            }
            by_query_speed = {}
            for speed, first_speed_row in first_track[
                "by_query_speed"
            ].items():
                conditions = {}
                for condition, first_condition in first_speed_row[
                    "physical"
                ]["condition_means"].items():
                    h1_metrics = first_condition["by_horizon"]["1"]
                    conditions[condition] = {
                        "history_speed": float(
                            first_condition["history_speed"]
                        ),
                        "history_relation": str(
                            first_condition["history_relation"]
                        ),
                        "physical_one_block": {
                            metric: _member_scalar_summary(
                                {
                                    slug: row["tracks"][track][
                                        "by_query_speed"
                                    ][speed]["physical"][
                                        "condition_means"
                                    ][condition]["by_horizon"]["1"][
                                        metric
                                    ]
                                    for slug, row in members
                                }
                            )
                            for metric in h1_metrics
                            if isinstance(
                                h1_metrics[metric], (int, float)
                            )
                        },
                        "fixed_candidate": {
                            metric: _member_scalar_summary(
                                {
                                    slug: row["tracks"][track][
                                        "by_query_speed"
                                    ][speed]["fixed_candidate"][
                                        "condition_means"
                                    ][condition][metric]
                                    for slug, row in members
                                }
                            )
                            for metric in (
                                "mean_exact_query_dynamics_regret_px",
                                "success_rate",
                            )
                        },
                        "closed_loop": {
                            metric: _member_scalar_summary(
                                {
                                    slug: row["tracks"][track][
                                        "by_query_speed"
                                    ][speed]["planning"]["conditions"][
                                        condition
                                    ][metric]
                                    for slug, row in members
                                }
                            )
                            for metric in (
                                "success_rate",
                                "mean_final_distance_px",
                                "mean_normalized_distance_auc",
                            )
                        },
                    }
                    deadline_source = first_speed_row["planning"][
                        "conditions"
                    ][condition]["deadline_success_curve"]
                    conditions[condition]["closed_loop"][
                        "deadline_success_curve"
                    ] = {
                        budget: _member_scalar_summary(
                            {
                                slug: row["tracks"][track][
                                    "by_query_speed"
                                ][speed]["planning"]["conditions"][
                                    condition
                                ]["deadline_success_curve"][budget]
                                for slug, row in members
                            }
                        )
                        for budget in deadline_source
                    }
                by_query_speed[speed] = {
                    "conditions": conditions,
                }
            tracks[track] = {
                "summary_effects": summary_effects,
                "gates": gates,
                "by_query_speed": by_query_speed,
            }
        result[group] = {
            "display_name": display_names.get(group, group),
            "models": [slug for slug, _ in members],
            "training_seeds": [
                int(row["training_seed"]) for _, row in members
            ],
            "tracks": tracks,
            "ability_retention": {
                "passed_models": int(
                    sum(
                        bool(row["ability_retention"]["passed"])
                        for _, row in members
                    )
                ),
                "models": len(members),
            },
        }
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = config["artifacts"]
    artifact_root = resolve_contextworld_path(
        artifacts["root"], repo_root=ROOT
    )
    support_audit_path = artifact_root / "support_audit.json"
    support_audit = _load_passed(support_audit_path)
    if not (
        support_audit["manifest_pairing"]["passed"]
        and support_audit["actual_reset_goal_pairing"]["passed"]
        and support_audit["exposure_equality"]["passed"]
        and int(support_audit["paired_scenarios"]) == 608
    ):
        raise RuntimeError("Matched speed training-data audit failed")
    roots = {
        "physical": resolve_contextworld_path(
            artifacts["physical_transition_root"], repo_root=ROOT
        ),
        "fixed": resolve_contextworld_path(
            artifacts["fixed_candidate_root"], repo_root=ROOT
        ),
        "planning": resolve_contextworld_path(
            artifacts["closed_loop_root"], repo_root=ROOT
        ),
        "ability": resolve_contextworld_path(
            artifacts["original_ability_retention_root"],
            repo_root=ROOT,
        ),
    }
    eval_seeds = [
        int(value) for value in config["formal_eval"]["eval_seeds"]
    ]
    num_eval = int(
        config["formal_eval"]["evaluations_per_matrix_cell_per_seed"]
    )
    tracks = {
        row["name"]: [float(value) for value in row["speeds"]]
        for row in config["frozen_scope"]["tracks"]
        if row["name"] in args.tracks
    }
    context_identifiability = {
        track: _context_identifiability_audit(
            resolve_contextworld_path(
                artifacts["catalogs"][track], repo_root=ROOT
            )
        )
        for track in tracks
    }
    if not all(
        row["passed"] for row in context_identifiability.values()
    ):
        raise RuntimeError("Speed context identifiability audit failed")
    models = [
        (group, model)
        for group, rows in config["models"].items()
        for model in rows
    ]
    model_results = {}
    ability_raw: dict[
        str, dict[str, list[dict[str, Any]]]
    ] = {}
    for model_index, (group, model) in enumerate(models):
        slug = str(model["slug"])
        checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=ROOT
        )
        checkpoint_hash = file_sha256(checkpoint)
        ability_raw[slug] = {
            domain: _ability_records(
                root=roots["ability"],
                slug=slug,
                domain=domain,
                eval_seeds=eval_seeds,
                checkpoint_sha256=checkpoint_hash,
                num_eval=num_eval,
            )
            for domain in ("original_heldout", "speed5_matched")
        }
        rollout_path = (
            roots["ability"] / slug / "rollout_error.json"
        )
        if (
            not rollout_path.is_file()
            and slug == "h3_origheldout_s3072"
        ):
            rollout_path = (
                roots["ability"].parent.parent
                / "original_ability_reconstruction"
                / slug
                / "rollout_error.json"
            )
        rollout = _load_passed(rollout_path)
        if rollout["checkpoint"]["sha256"] != checkpoint_hash:
            raise RuntimeError(
                f"Rollout checkpoint hash mismatch: {rollout_path}"
            )
        track_results = {}
        for track_index, (track, speeds) in enumerate(tracks.items()):
            rows_by_speed = {}
            for speed_index, speed in enumerate(speeds):
                mode_records = {
                    mode: _records(
                        _result_files(
                            root=root,
                            slug=slug,
                            track=track,
                            query_speed=speed,
                            eval_seeds=eval_seeds,
                        ),
                        checkpoint_sha256=checkpoint_hash,
                        mode=mode,
                        num_eval=num_eval,
                    )
                    for mode, root in (
                        (name, roots[name])
                        for name in ("physical", "fixed", "planning")
                    )
                }
                if (
                    len(mode_records["physical"]) != num_eval
                    * len(eval_seeds)
                    or len(mode_records["fixed"]) != num_eval
                    * len(eval_seeds)
                    or len(mode_records["planning"]) != num_eval
                    * len(eval_seeds)
                    * 3
                ):
                    raise RuntimeError(
                        f"Formal count mismatch: {slug}/{track}/{speed}"
                    )
                seed_base = (
                    args.bootstrap_seed
                    + 100000 * model_index
                    + 10000 * track_index
                    + 1000 * speed_index
                )
                rows_by_speed[f"{speed:g}"] = {
                    "physical": _physical_row_summary(
                        mode_records["physical"],
                        bootstrap_seed=seed_base,
                        bootstrap_resamples=args.bootstrap_resamples,
                    ),
                    "fixed_candidate": _fixed_row_summary(
                        mode_records["fixed"],
                        bootstrap_seed=seed_base + 500,
                        bootstrap_resamples=args.bootstrap_resamples,
                    ),
                    "planning": _planning_row_summary(
                        mode_records["planning"]
                    ),
                }
            multiplicity_family = []
            for speed_row in rows_by_speed.values():
                for comparison in speed_row["physical"][
                    "same_speed_benefit"
                ].values():
                    multiplicity_family.extend(
                        [
                            comparison["one_block_position"],
                            comparison[
                                "one_block_displacement_magnitude"
                            ],
                            comparison["trajectory_position"],
                            comparison["trajectory_latent"],
                        ]
                    )
                multiplicity_family.extend(
                    speed_row["fixed_candidate"][
                        "same_speed_benefit"
                    ].values()
                )
            _holm_adjust(
                multiplicity_family,
                alpha=float(config["decisions"]["multiplicity"][
                    "familywise_alpha"
                ]),
            )
            track_results[track] = {
                "by_query_speed": rows_by_speed,
                "gates": {
                    "history_response": all(
                        row["physical"]["gates"][
                            "history_response_at_one_block"
                        ]
                        for row in rows_by_speed.values()
                    ),
                    "physical_calibration": all(
                        row["physical"]["gates"]["passed"]
                        and all(
                            comparison["one_block_position"][
                                "holm_passed"
                            ]
                            and comparison[
                                "one_block_displacement_magnitude"
                            ]["holm_passed"]
                            and comparison["trajectory_position"][
                                "holm_passed"
                            ]
                            and comparison["trajectory_latent"][
                                "holm_passed"
                            ]
                            for comparison in row["physical"][
                                "same_speed_benefit"
                            ].values()
                        )
                        for row in rows_by_speed.values()
                    ),
                    "fixed_candidate_calibration": all(
                        row["fixed_candidate"]["gates"][
                            "same_speed_lowest_regret"
                        ]
                        and all(
                            comparison["holm_passed"]
                            for comparison in row[
                                "fixed_candidate"
                            ]["same_speed_benefit"].values()
                        )
                        for row in rows_by_speed.values()
                    ),
                },
                "multiplicity": {
                    "method": "holm",
                    "familywise_alpha": float(
                        config["decisions"]["multiplicity"][
                            "familywise_alpha"
                        ]
                    ),
                    "primary_comparisons": len(
                        multiplicity_family
                    ),
                },
            }
            track_results[track]["summary_effects"] = {
                "mean_one_block_same_speed_benefit_px": (
                    _mean_physical_benefit(track_results[track])
                ),
                "mean_one_block_same_speed_displacement_benefit_px": (
                    _mean_displacement_calibration_benefit(
                        track_results[track]
                    )
                ),
                "mean_high_minus_low_inferred_speed": (
                    _mean_history_response(track_results[track])
                ),
                "mean_inferred_speed_response_gain": (
                    _mean_history_response_gain(track_results[track])
                ),
                "mean_high_minus_low_predicted_displacement_px": (
                    _mean_displacement_response(track_results[track])
                ),
                "mean_same_speed_fixed_candidate_benefit_px": (
                    _mean_fixed_benefit(track_results[track])
                ),
            }
        model_results[slug] = {
            "model_group": group,
            "training_seed": int(model["training_seed"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "tracks": track_results,
            "ability_retention": {
                "rollout_error": {
                    "path": str(rollout_path),
                    "aggregates": rollout["aggregates"],
                }
            },
        }

    training_audit = {}
    for slug, row in model_results.items():
        report_path = resolve_contextworld_path(
            f"artifacts/training/reports/{slug}.json",
            repo_root=ROOT,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_steps = (
            6420
            if row["model_group"] == "original_reference"
            else 12840
        )
        plan = report["training"]["plan"]
        exposures = {
            group: int(exposure["total_draws"])
            for group, exposure in plan["group_exposure"].items()
        }
        passed = bool(
            report["passed"]
            and report["save_load_exact"]
            and report["artifacts"]["pretrained_sha256"]
            == row["checkpoint_sha256"]
            and int(report["training"]["global_step"])
            == expected_steps
            and int(report["training"]["expected_optimizer_steps"])
            == expected_steps
            and int(report["training"]["scheduler_last_epoch"])
            == expected_steps
            and int(
                report["training"][
                    "seed_before_model_initialization"
                ]
            )
            == int(row["training_seed"])
            and int(report["model"]["history_size"]) == 3
            and int(report["model"]["action_block"]) == 5
            and len(exposures)
            == (
                1
                if row["model_group"] == "original_reference"
                else 2
            )
            and all(draws == 6574080 for draws in exposures.values())
            and all(
                bool(exposure["exposure_is_exact"])
                for exposure in plan["group_exposure"].values()
            )
        )
        if not passed:
            raise RuntimeError(f"Training audit failed: {slug}")
        training_audit[slug] = {
            "path": str(report_path),
            "training_seed": int(
                report["training"]["seed_before_model_initialization"]
            ),
            "optimizer_steps": expected_steps,
            "scheduler_steps": int(
                report["training"]["scheduler_last_epoch"]
            ),
            "group_draws": exposures,
            "save_load_exact": bool(report["save_load_exact"]),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "passed": True,
        }

    reference_slug = next(
        slug
        for slug, row in model_results.items()
        if row["model_group"] == "original_reference"
    )
    for model_index, (slug, row) in enumerate(model_results.items()):
        comparisons = {
            domain: _ability_comparison(
                ability_raw[slug][domain],
                ability_raw[reference_slug][domain],
                bootstrap_seed=(
                    args.bootstrap_seed + 700000 + 1000 * model_index
                ),
                bootstrap_resamples=args.bootstrap_resamples,
            )
            for domain in ("original_heldout", "speed5_matched")
        }
        row["ability_retention"]["planning_vs_original_reference"] = (
            comparisons
        )
        row["ability_retention"]["passed"] = all(
            comparison["passed"]
            for comparison in comparisons.values()
        )

    causality_path = (
        artifact_root / "temporal_causality_audit.json"
    )
    causality = _load_passed(causality_path)
    expected_hashes = {
        slug: row["checkpoint_sha256"]
        for slug, row in model_results.items()
    }
    if set(causality.get("models", {})) != set(expected_hashes):
        raise RuntimeError(
            "Temporal causality audit model set does not match evaluation"
        )
    for slug, checkpoint_hash in expected_hashes.items():
        audit_row = causality["models"][slug]
        if (
            audit_row["checkpoint_sha256"] != checkpoint_hash
            or not audit_row["passed"]
        ):
            raise RuntimeError(
                f"Temporal causality audit mismatch: {slug}"
            )

    by_group_seed = {
        (row["model_group"], row["training_seed"]): (slug, row)
        for slug, row in model_results.items()
    }
    paired_training = {}
    for seed in (3072, 4096, 5120):
        single_slug, single = by_group_seed[("single_speed_control", seed)]
        multi_slug, multi = by_group_seed[("multi_speed_target", seed)]
        paired_training[str(seed)] = {
            "single_model": single_slug,
            "multi_model": multi_slug,
            "tracks": {
                track: {
                    "multi_minus_single_one_block_calibration_benefit_px": (
                        _mean_physical_benefit(multi["tracks"][track])
                        - _mean_physical_benefit(single["tracks"][track])
                    ),
                    (
                        "multi_minus_single_one_block_displacement_"
                        "calibration_benefit_px"
                    ): (
                        _mean_displacement_calibration_benefit(
                            multi["tracks"][track]
                        )
                        - _mean_displacement_calibration_benefit(
                            single["tracks"][track]
                        )
                    ),
                    "multi_minus_single_history_response": (
                        _mean_history_response(multi["tracks"][track])
                        - _mean_history_response(single["tracks"][track])
                    ),
                    "multi_minus_single_history_response_gain": (
                        _mean_history_response_gain(
                            multi["tracks"][track]
                        )
                        - _mean_history_response_gain(
                            single["tracks"][track]
                        )
                    ),
                    "multi_minus_single_displacement_response_px": (
                        _mean_displacement_response(
                            multi["tracks"][track]
                        )
                        - _mean_displacement_response(
                            single["tracks"][track]
                        )
                    ),
                    "multi_minus_single_fixed_candidate_benefit_px": (
                        _mean_fixed_benefit(multi["tracks"][track])
                        - _mean_fixed_benefit(single["tracks"][track])
                    ),
                }
                for track in tracks
            },
        }

    def stable_positive(track: str, key: str) -> bool:
        return all(
            row["tracks"][track][key] > 0.0
            for row in paired_training.values()
        )

    attribution = {
        track: {
            "one_block_calibration_stable": stable_positive(
                track,
                "multi_minus_single_one_block_calibration_benefit_px",
            ),
            "one_block_displacement_calibration_stable": stable_positive(
                track,
                (
                    "multi_minus_single_one_block_displacement_"
                    "calibration_benefit_px"
                ),
            ),
            "history_response_stable": stable_positive(
                track, "multi_minus_single_history_response"
            ),
            "displacement_response_stable": stable_positive(
                track,
                "multi_minus_single_displacement_response_px",
            ),
            "fixed_candidate_calibration_stable": stable_positive(
                track,
                "multi_minus_single_fixed_candidate_benefit_px",
            ),
        }
        for track in tracks
    }
    multi_models = [
        row
        for row in model_results.values()
        if row["model_group"] == "multi_speed_target"
    ]

    def all_multi(track: str, gate: str) -> bool:
        return all(row["tracks"][track]["gates"][gate] for row in multi_models)

    seen = "seen_for_multi"
    unseen = "unseen_interpolation"
    level_a = all_multi(seen, "history_response")
    level_b = bool(
        level_a
        and all_multi(seen, "physical_calibration")
        and attribution[seen]["one_block_calibration_stable"]
        and attribution[seen][
            "one_block_displacement_calibration_stable"
        ]
        and attribution[seen]["history_response_stable"]
        and attribution[seen]["displacement_response_stable"]
    )
    level_c = bool(
        level_b
        and all_multi(seen, "fixed_candidate_calibration")
        and attribution[seen]["fixed_candidate_calibration_stable"]
    )
    multi_ability_retention = all(
        row["ability_retention"]["passed"] for row in multi_models
    )
    level_d = bool(
        level_c
        and all(attribution[seen].values())
        and multi_ability_retention
    )
    level_e = bool(
        level_d
        and all_multi(unseen, "history_response")
        and all_multi(unseen, "physical_calibration")
        and all_multi(unseen, "fixed_candidate_calibration")
        and all(attribution[unseen].values())
    )
    output = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": str(config_path),
        "formal_count_audit": {
            "eval_seeds": eval_seeds,
            "evaluations_per_cell_per_seed": num_eval,
            "evaluations_per_cell": num_eval * len(eval_seeds),
            "models": len(models),
            "tracks": list(tracks),
            "passed": True,
        },
        "matched_training_data_audit": {
            "path": str(support_audit_path),
            "paired_scenarios": int(
                support_audit["paired_scenarios"]
            ),
            "actual_rows": support_audit["actual_rows"],
            "actual_speed_support": support_audit[
                "actual_speed_support"
            ],
            "exposure_equality": support_audit["exposure_equality"],
            "manifest_pairing_passed": support_audit[
                "manifest_pairing"
            ]["passed"],
            "reset_and_goal_pairing_passed": support_audit[
                "actual_reset_goal_pairing"
            ]["passed"],
            "passed": True,
        },
        "training_audit": training_audit,
        "context_identifiability_audit": context_identifiability,
        "temporal_causality_audit": {
            "path": str(causality_path),
            "status": causality["status"],
            "models": {
                slug: {
                    "maximum_change_at_or_before_boundary": row[
                        "probe"
                    ]["maximum_change_at_or_before_boundary"],
                    "future_perturbation_changed_a_future_output": row[
                        "probe"
                    ][
                        "future_perturbation_changed_a_future_output"
                    ],
                    "frozen_weight_audit_passed": row[
                        "frozen_weight_audit"
                    ]["passed"],
                    "passed": row["passed"],
                }
                for slug, row in causality["models"].items()
            },
            "passed": True,
        },
        "models": model_results,
        "model_group_summaries": _model_group_summaries(
            model_results
        ),
        "paired_training_seed_attribution": paired_training,
        "attribution_gates": attribution,
        "decision_levels": {
            "A_history_sensitive_seen": level_a,
            "B_physically_calibrated_seen": level_b,
            "C_candidate_planning_calibrated_seen": level_c,
            "D_training_method_with_ability_retention": level_d,
            "E_unseen_speed_interpolation": level_e,
            "multi_speed_original_ability_retention": (
                multi_ability_retention
            ),
        },
        "statistical_protocol": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "pairing": "within identical query, action probe, and eval seed",
            "cluster_unit": "static query geometry × action-probe family",
            "training_seed_is_highest_level_replication": True,
        },
    }
    output_path = resolve_contextworld_path(
        artifacts["final_summary"], repo_root=ROOT
    )
    write_json(output_path, output)
    return {**output, "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/tworoom_speed_cube_eval_v2.yaml"
        ),
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["seen_for_multi", "unseen_interpolation"],
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026072023)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": result["output"],
                "count_audit": result["formal_count_audit"],
                "attribution_gates": result["attribution_gates"],
                "decision_levels": result[
                    "decision_levels"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
