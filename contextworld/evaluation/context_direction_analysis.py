from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json

from .icl_sensitive import sha256_file
from .icl_sensitive_analysis import (
    exact_paired_sign_test,
    summarize_paired_rows,
)


def direction_result_path(
    root: Path,
    *,
    direction: str,
    seed: int,
) -> Path:
    if direction not in {"wrong_slow", "wrong_fast"}:
        raise ValueError(f"Unknown direction: {direction}")
    return root / f"{direction}_n50_s{int(seed)}.json"


@lru_cache(maxsize=None)
def _cached_sha256(path: str) -> str:
    return sha256_file(Path(path))


def _load_passed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Evaluation did not pass: {path}")
    return payload


def _assert_protocol(
    payload: dict[str, Any],
    *,
    seed: int,
    config: dict[str, Any],
    expected_checkpoint: Path,
    expected_normalizer: Path,
    expected_query_ids: list[str],
    expected_templates: list[str],
    path: Path,
) -> None:
    planning = config["frozen_scope"]["planning"]
    protocol = payload["protocol"]
    expected_protocol = {
        "action_block": int(planning["action_block_raw_steps"]),
        "history_size": 3,
        "eval_seed": int(seed),
        "eval_budget": int(planning["eval_budget_raw_steps"]),
        "horizon": int(planning["horizon_action_blocks"]),
        "receding_horizon": int(
            planning["receding_horizon_action_blocks"]
        ),
        "cem_batch_size": 1,
        "cem_num_samples": int(planning["cem_samples"]),
        "cem_steps": int(planning["cem_steps"]),
        "cem_topk": int(planning["cem_topk"]),
    }
    mismatches = {
        key: {"expected": value, "observed": protocol.get(key)}
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if not np.isclose(
        float(protocol.get("cem_var_scale", float("nan"))),
        float(planning["cem_var_scale"]),
        rtol=0.0,
        atol=0.0,
    ):
        mismatches["cem_var_scale"] = {
            "expected": float(planning["cem_var_scale"]),
            "observed": protocol.get("cem_var_scale"),
        }
    if Path(protocol["normalization_source"]).resolve() != (
        expected_normalizer.resolve()
    ):
        mismatches["normalization_source"] = {
            "expected": str(expected_normalizer.resolve()),
            "observed": protocol["normalization_source"],
        }
    if mismatches:
        raise RuntimeError(
            f"Frozen planning protocol mismatch in {path}: {mismatches}"
        )

    if payload.get("run_kind") != "confirmation":
        raise RuntimeError(f"Non-confirmatory result: {path}")
    if payload["stable_worldmodel"]["commit"] != str(
        config["frozen_scope"]["stable_worldmodel_commit"]
    ):
        raise RuntimeError(f"StableWM commit mismatch in {path}")
    if not payload["frozen_weight_audit"]["passed"]:
        raise RuntimeError(f"Frozen-weight audit failed in {path}")
    if not payload["pairing_audit"]["passed"]:
        raise RuntimeError(f"Within-eval pairing audit failed in {path}")

    checkpoint = expected_checkpoint.resolve()
    observed_checkpoint = Path(
        payload["checkpoint"]["path"]
    ).resolve()
    if observed_checkpoint != checkpoint:
        raise RuntimeError(
            f"Checkpoint mismatch in {path}: "
            f"{observed_checkpoint} != {checkpoint}"
        )
    if payload["checkpoint"]["sha256"] != _cached_sha256(str(checkpoint)):
        raise RuntimeError(f"Checkpoint hash mismatch in {path}")

    selection = payload["selection"]
    expected_formal = config["formal_eval"]
    if selection["speeds"] != [
        float(value)
        for value in config["frozen_scope"]["query_speeds"]
    ]:
        raise RuntimeError(f"Query speeds differ in {path}")
    if selection["templates"] != expected_templates:
        raise RuntimeError(f"Template selection differs in {path}")
    if selection["query_ids"] != expected_query_ids:
        raise RuntimeError(f"Query selection differs in {path}")
    if int(selection["unique_base_queries"]) != int(
        expected_formal["expected_unique_base_queries_per_eval"]
    ):
        raise RuntimeError(f"Unique query count differs in {path}")
    if int(selection["evaluations_per_condition"]) != int(
        expected_formal["evaluations_per_condition_per_seed"]
    ):
        raise RuntimeError(f"Per-condition count differs in {path}")


def _query_metadata(
    catalog: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for bundle in catalog["bundles"]:
        query_id = str(bundle["query_id"])
        if query_id in metadata:
            raise RuntimeError(f"Duplicate query ID: {query_id}")
        metadata[query_id] = {
            "speed": float(bundle["query_factors"]["agent.speed"]),
            "distance_bin": int(bundle["template"]["distance_bin"]),
            "template_id": str(bundle["template"]["template_id"]),
        }
    return metadata


def _records_by_condition(
    payload: dict[str, Any],
    *,
    expected_count: int,
    expected_query_ids: set[str],
    seed: int,
    path: Path,
) -> dict[str, list[dict[str, Any]]]:
    by_condition = {
        condition: [
            {**row, "eval_seed": int(seed)}
            for row in payload["records"]
            if row["condition"] == condition
        ]
        for condition in ("correct", "wrong")
    }
    if len(payload["records"]) != 2 * expected_count or any(
        len(rows) != expected_count
        for rows in by_condition.values()
    ):
        raise RuntimeError(f"Expected {expected_count} records/condition: {path}")
    for condition, rows in by_condition.items():
        keys = {
            (int(row["eval_seed"]), str(row["evaluation_id"]))
            for row in rows
        }
        if len(keys) != expected_count:
            raise RuntimeError(
                f"Duplicate {condition} evaluation IDs in {path}"
            )
        if any(
            str(row["query_id"]) not in expected_query_ids
            for row in rows
        ):
            raise RuntimeError(f"Unknown query ID in {path}")
    return by_condition


def _correct_record_signature(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "evaluation_id",
            "evaluation_index",
            "repeat_index",
            "eval_seed",
            "query_id",
            "source_scenario_id",
            "template_id",
            "speed",
            "door_position",
            "condition",
            "success",
            "final_state",
            "goal_state",
            "final_distance",
            "cem_seed",
            "cem_rng_state_sha256_before",
            "cem_rng_state_sha256_after",
            "cem_solve_calls",
            "get_cost_calls",
            "fixed_context",
            "factor_readback",
        )
    }


def _record_lookup(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    selected = list(rows)
    result = {
        (int(row["eval_seed"]), str(row["evaluation_id"])): row
        for row in selected
    }
    if len(result) != len(selected):
        raise RuntimeError("Condition records contain duplicate schedule keys")
    return result


def _paired_rows(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    first_label: str,
    second_label: str,
) -> list[dict[str, Any]]:
    first_lookup = _record_lookup(first)
    second_lookup = _record_lookup(second)
    if first_lookup.keys() != second_lookup.keys():
        raise RuntimeError(
            f"{first_label}/{second_label} evaluation schedules differ"
        )
    rows: list[dict[str, Any]] = []
    for key in sorted(first_lookup):
        first_row = first_lookup[key]
        second_row = second_lookup[key]
        shared = (
            "query_id",
            "evaluation_index",
            "repeat_index",
            "speed",
            "template_id",
            "cem_seed",
            "cem_rng_state_sha256_before",
            "goal_state",
        )
        if any(
            first_row[field] != second_row[field]
            for field in shared
        ):
            raise RuntimeError(
                f"{first_label}/{second_label} pairing differs at {key}"
            )
        rows.append(
            {
                "eval_seed": key[0],
                "evaluation_id": key[1],
                "query_id": str(first_row["query_id"]),
                "speed": float(first_row["speed"]),
                "template_id": str(first_row["template_id"]),
                "correct_success": bool(first_row["success"]),
                "wrong_success": bool(second_row["success"]),
                "wrong_minus_correct_final_distance": float(
                    second_row["final_distance"]
                    - first_row["final_distance"]
                ),
            }
        )
    return rows


def _strata(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_seed": defaultdict(list),
        "by_speed": defaultdict(list),
        "by_distance": defaultdict(list),
        "by_template": defaultdict(list),
    }
    for row in rows:
        query = metadata[str(row["query_id"])]
        grouped["by_seed"][str(int(row["eval_seed"]))].append(row)
        grouped["by_speed"][f"{float(row['speed']):g}"].append(row)
        grouped["by_distance"][
            str(int(query["distance_bin"]))
        ].append(row)
        grouped["by_template"][
            str(query["template_id"])
        ].append(row)
    result: dict[str, Any] = {}
    for name, values in grouped.items():
        if name == "by_template":
            ordered = sorted(values.items())
        else:
            ordered = sorted(
                values.items(), key=lambda item: float(item[0])
            )
        result[name] = {
            key: summarize_paired_rows(selected)
            for key, selected in ordered
        }
    return result


def _condition_summary(
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            raise ValueError("Cannot summarize zero records")
        successes = sum(bool(row["success"]) for row in selected)
        initial_distances = np.asarray(
            [
                float(
                    metadata[str(row["query_id"])]["distance_bin"]
                )
                for row in selected
            ],
            dtype=np.float64,
        )
        final_distances = np.asarray(
            [float(row["final_distance"]) for row in selected],
            dtype=np.float64,
        )
        normalized_remaining = final_distances / initial_distances
        return {
            "evaluations": len(selected),
            "successes": int(successes),
            "success_rate_percent": float(
                100.0 * successes / len(selected)
            ),
            "mean_initial_distance": float(np.mean(initial_distances)),
            "mean_final_distance": float(np.mean(final_distances)),
            "mean_normalized_remaining_distance": float(
                np.mean(normalized_remaining)
            ),
            "mean_normalized_progress_percent": float(
                100.0 * np.mean(1.0 - normalized_remaining)
            ),
            "final_distance_threshold_curve_percent": {
                str(radius): float(
                    100.0 * np.mean(final_distances < float(radius))
                )
                # The environment terminates once it enters the official
                # 16 px success radius, so smaller radii are not observable.
                for radius in (16, 24, 32, 48)
            },
        }

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_seed": defaultdict(list),
        "by_speed": defaultdict(list),
        "by_distance": defaultdict(list),
    }
    for row in rows:
        query = metadata[str(row["query_id"])]
        groups["by_seed"][str(int(row["eval_seed"]))].append(row)
        groups["by_speed"][f"{float(row['speed']):g}"].append(row)
        groups["by_distance"][
            str(int(query["distance_bin"]))
        ].append(row)
    return {
        **summarize(rows),
        **{
            name: {
                key: summarize(selected)
                for key, selected in sorted(
                    values.items(), key=lambda item: float(item[0])
                )
            }
            for name, values in groups.items()
        },
    }


def _direction_counts(
    effects: Iterable[float],
) -> dict[str, Any]:
    selected = [float(value) for value in effects]
    positive = sum(value > 1e-12 for value in selected)
    negative = sum(value < -1e-12 for value in selected)
    ties = len(selected) - positive - negative
    return {
        "units": len(selected),
        "positive": positive,
        "ties": ties,
        "negative": negative,
        "exact_sign_test_ignoring_ties": exact_paired_sign_test(
            positive, negative
        ),
    }


def _secondary_diagnostics(
    *,
    conditions: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    correct_vs_slow = comparisons["correct_vs_wrong_slow"]
    correct_vs_fast = comparisons["correct_vs_wrong_fast"]
    fast_vs_slow = comparisons["wrong_fast_vs_wrong_slow"]
    template_rows = []
    for template_id in sorted(
        fast_vs_slow["strata"]["by_template"]
    ):
        slow_rate = float(
            correct_vs_slow["strata"]["by_template"][template_id][
                "wrong_slow"
            ]["success_rate_percent"]
        )
        correct_rate = float(
            correct_vs_slow["strata"]["by_template"][template_id][
                "correct"
            ]["success_rate_percent"]
        )
        fast_rate = float(
            correct_vs_fast["strata"]["by_template"][template_id][
                "wrong_fast"
            ]["success_rate_percent"]
        )
        template_rows.append(
            {
                "template_id": template_id,
                "wrong_slow_success_rate_percent": slow_rate,
                "correct_success_rate_percent": correct_rate,
                "wrong_fast_success_rate_percent": fast_rate,
                "wrong_fast_minus_wrong_slow_points": (
                    fast_rate - slow_rate
                ),
                "nondecreasing_with_context_speed": (
                    slow_rate <= correct_rate <= fast_rate
                ),
                "strictly_increasing_with_context_speed": (
                    slow_rate < correct_rate < fast_rate
                ),
                "all_conditions_at_floor": (
                    slow_rate == correct_rate == fast_rate == 0.0
                ),
                "all_conditions_at_ceiling": (
                    slow_rate == correct_rate == fast_rate == 100.0
                ),
            }
        )

    fast_slow_strata = fast_vs_slow["strata"]
    reporting_units = {
        name: _direction_counts(
            float(
                row[
                    "wrong_fast_minus_wrong_slow_success_rate_points"
                ]
            )
            for row in fast_slow_strata[name].values()
        )
        for name in (
            "by_seed",
            "by_speed",
            "by_distance",
            "by_template",
        )
    }
    slow_progress = float(
        conditions["wrong_slower_context_3p1"][
            "mean_normalized_progress_percent"
        ]
    )
    correct_progress = float(
        conditions["correct_speed_context"][
            "mean_normalized_progress_percent"
        ]
    )
    fast_progress = float(
        conditions["wrong_faster_context_7p0"][
            "mean_normalized_progress_percent"
        ]
    )
    return {
        "status": "descriptive_posthoc_secondary_not_a_frozen_gate",
        "normalized_progress": {
            "wrong_slow_percent": slow_progress,
            "correct_percent": correct_progress,
            "wrong_fast_percent": fast_progress,
            "correct_minus_wrong_slow_points": (
                correct_progress - slow_progress
            ),
            "wrong_fast_minus_correct_points": (
                fast_progress - correct_progress
            ),
            "wrong_fast_minus_wrong_slow_points": (
                fast_progress - slow_progress
            ),
        },
        "context_speed_ordering_by_geometry": {
            "templates": len(template_rows),
            "nondecreasing_templates": sum(
                row["nondecreasing_with_context_speed"]
                for row in template_rows
            ),
            "strictly_increasing_templates": sum(
                row["strictly_increasing_with_context_speed"]
                for row in template_rows
            ),
            "all_floor_templates": sum(
                row["all_conditions_at_floor"]
                for row in template_rows
            ),
            "all_ceiling_templates": sum(
                row["all_conditions_at_ceiling"]
                for row in template_rows
            ),
            "context_sensitive_templates": sum(
                not row["all_conditions_at_floor"]
                and not row["all_conditions_at_ceiling"]
                and abs(
                    row["wrong_fast_minus_wrong_slow_points"]
                )
                > 1e-12
                for row in template_rows
            ),
            "rows": template_rows,
        },
        "direction_consistency_across_reporting_units": reporting_units,
        "statistical_boundary": (
            "The frozen pair-level exact sign tests treat distinct CEM "
            "evaluations as paired stochastic trials. Seed-, speed-, "
            "distance-, and geometry-level sign summaries are reported "
            "descriptively to expose clustering; future benchmark releases "
            "must add hierarchical interval estimates."
        ),
    }


def _labeled_pair_summary(
    rows: list[dict[str, Any]],
    *,
    first_label: str,
    second_label: str,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    raw = summarize_paired_rows(rows)
    return {
        "pairs": raw["pairs"],
        first_label: raw["correct"],
        second_label: raw["wrong"],
        f"{first_label}_minus_{second_label}_success_rate_points": raw[
            "correct_minus_wrong_success_rate_points"
        ],
        f"{first_label}_only_successes": raw[
            "correct_only_successes"
        ],
        f"{second_label}_only_successes": raw[
            "wrong_only_successes"
        ],
        "both_successes": raw["both_successes"],
        "neither_successes": raw["neither_successes"],
        "paired_sign_test": raw["paired_sign_test"],
        f"{second_label}_minus_{first_label}_mean_final_distance": raw[
            "wrong_minus_correct_mean_final_distance"
        ],
        "_raw_strata": _strata(rows, metadata=metadata),
    }


def _relabel_strata(
    strata: dict[str, Any],
    *,
    first_label: str,
    second_label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum, values in strata.items():
        result[stratum] = {}
        for key, raw in values.items():
            result[stratum][key] = {
                "pairs": raw["pairs"],
                first_label: raw["correct"],
                second_label: raw["wrong"],
                (
                    f"{first_label}_minus_{second_label}_"
                    "success_rate_points"
                ): raw["correct_minus_wrong_success_rate_points"],
                f"{first_label}_only_successes": raw[
                    "correct_only_successes"
                ],
                f"{second_label}_only_successes": raw[
                    "wrong_only_successes"
                ],
                "both_successes": raw["both_successes"],
                "neither_successes": raw["neither_successes"],
                "paired_sign_test": raw["paired_sign_test"],
                (
                    f"{second_label}_minus_{first_label}_"
                    "mean_final_distance"
                ): raw["wrong_minus_correct_mean_final_distance"],
            }
    return result


def labeled_pair_summary(
    rows: list[dict[str, Any]],
    *,
    first_label: str,
    second_label: str,
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    result = _labeled_pair_summary(
        rows,
        first_label=first_label,
        second_label=second_label,
        metadata=metadata,
    )
    strata = result.pop("_raw_strata")
    result["strata"] = _relabel_strata(
        strata,
        first_label=first_label,
        second_label=second_label,
    )
    return result


def directional_decisions(
    *,
    correct_vs_slow: dict[str, Any],
    correct_vs_fast: dict[str, Any],
    fast_vs_slow: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    specs = config["decisions_frozen_before_execution"]
    correctness_spec = specs["correctness_aligned_planning_icl"]
    comparisons = {
        "correct_over_wrong_slow": (
            correct_vs_slow,
            "correct_minus_wrong_slow_success_rate_points",
            "correct_only_successes",
            "wrong_slow_only_successes",
        ),
        "correct_over_wrong_fast": (
            correct_vs_fast,
            "correct_minus_wrong_fast_success_rate_points",
            "correct_only_successes",
            "wrong_fast_only_successes",
        ),
    }
    correctness_gates: dict[str, Any] = {}
    for name, (summary, effect_key, first_only, second_only) in (
        comparisons.items()
    ):
        correctness_gates[name] = {
            "effect_at_least_minimum": float(summary[effect_key])
            >= float(correctness_spec["minimum_effect_each_pp"]),
            "paired_exact_sign_test_passed": float(
                summary["paired_sign_test"]["two_sided_p_value"]
            )
            <= float(
                correctness_spec[
                    "paired_exact_sign_test_p_max_each"
                ]
            ),
            "correct_only_greater": int(summary[first_only])
            > int(summary[second_only]),
        }
        correctness_gates[name]["passed"] = all(
            correctness_gates[name].values()
        )
    correctness_passed = all(
        gate["passed"] for gate in correctness_gates.values()
    )

    bias_spec = specs["higher_speed_prompt_bias_confirmation"]
    bias_gates = {
        "effect_at_least_minimum": float(
            fast_vs_slow[
                "wrong_fast_minus_wrong_slow_success_rate_points"
            ]
        )
        >= float(bias_spec["minimum_effect_pp"]),
        "paired_exact_sign_test_passed": float(
            fast_vs_slow["paired_sign_test"]["two_sided_p_value"]
        )
        <= float(bias_spec["paired_exact_sign_test_p_max"]),
        "fast_only_greater": int(
            fast_vs_slow["wrong_fast_only_successes"]
        )
        > int(fast_vs_slow["wrong_slow_only_successes"]),
    }
    bias_passed = all(bias_gates.values())
    bias_gates["passed"] = bias_passed

    if correctness_passed and bias_passed:
        classification = (
            "correctness_alignment_and_higher_speed_bias_both_present"
        )
    elif correctness_passed:
        classification = "correctness_aligned_planning_icl"
    elif bias_passed:
        classification = "higher_speed_prompt_bias"
    else:
        classification = "neither_frozen_gate_passed"
    return {
        "correctness_aligned_planning_icl": {
            "frozen_specification": correctness_spec,
            "comparisons": correctness_gates,
            "established": correctness_passed,
        },
        "higher_speed_prompt_bias": {
            "frozen_specification": bias_spec,
            "gates": bias_gates,
            "confirmed": bias_passed,
        },
        "classification": classification,
    }


def run_directional_analysis(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
    catalog_source_config_path: Path | None = None,
    result_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    build_report_path = resolve_contextworld_path(
        artifacts["catalog_build_report"], repo_root=repo_root
    )
    build_report = _load_passed(build_report_path)
    config_hash = sha256_file(config_path)
    source_config_path = (
        catalog_source_config_path or config_path
    ).resolve()
    source_config_hash = sha256_file(source_config_path)
    if build_report["config"]["sha256"] != source_config_hash:
        raise RuntimeError("Catalogs were built from a different config")
    if not build_report["cross_catalog_audit"]["passed"]:
        raise RuntimeError("Cross-catalog pairing audit did not pass")

    catalog_paths = {
        "wrong_slow": resolve_contextworld_path(
            artifacts["wrong_slow_catalog"], repo_root=repo_root
        ),
        "wrong_fast": resolve_contextworld_path(
            artifacts["wrong_fast_catalog"], repo_root=repo_root
        ),
    }
    catalogs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in catalog_paths.items()
    }
    for name, path in catalog_paths.items():
        if sha256_file(path) != build_report["catalogs"][name]["sha256"]:
            raise RuntimeError(f"{name} catalog changed after build")

    metadata = _query_metadata(catalogs["wrong_slow"])
    fast_metadata = _query_metadata(catalogs["wrong_fast"])
    if metadata != fast_metadata:
        raise RuntimeError("Directional catalog query metadata differs")
    expected_templates = [
        str(row["template_id"])
        for row in catalogs["wrong_slow"]["geometry_bank"]
    ]
    expected_query_ids = [
        str(bundle["query_id"])
        for bundle in catalogs["wrong_slow"]["bundles"]
    ]
    if len(expected_query_ids) != int(
        config["formal_eval"]["expected_unique_base_queries_per_eval"]
    ):
        raise RuntimeError("Catalog query count differs from preregistration")

    raw_root = resolve_contextworld_path(
        artifacts["raw_results"], repo_root=repo_root
    )
    expected_checkpoint = resolve_contextworld_path(
        config["frozen_scope"]["model"]["checkpoint"],
        repo_root=repo_root,
    )
    expected_normalizer = resolve_contextworld_path(
        config["frozen_scope"]["normalizer"], repo_root=repo_root
    )
    seeds = [
        int(value) for value in config["formal_eval"]["eval_seeds"]
    ]
    per_seed = int(
        config["formal_eval"]["evaluations_per_condition_per_seed"]
    )
    condition_records: dict[str, dict[str, list[dict[str, Any]]]] = {
        direction: {"correct": [], "wrong": []}
        for direction in ("wrong_slow", "wrong_fast")
    }
    inputs: list[dict[str, Any]] = []
    schedules: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for direction in ("wrong_slow", "wrong_fast"):
        for seed in seeds:
            path = direction_result_path(
                raw_root, direction=direction, seed=seed
            )
            payload = _load_passed(path)
            _assert_protocol(
                payload,
                seed=seed,
                config=config,
                expected_checkpoint=expected_checkpoint,
                expected_normalizer=expected_normalizer,
                expected_query_ids=expected_query_ids,
                expected_templates=expected_templates,
                path=path,
            )
            selected = _records_by_condition(
                payload,
                expected_count=per_seed,
                expected_query_ids=set(expected_query_ids),
                seed=seed,
                path=path,
            )
            for condition in ("correct", "wrong"):
                condition_records[direction][condition].extend(
                    selected[condition]
                )
            schedules[(direction, seed)] = payload["selection"][
                "schedule"
            ]
            inputs.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "direction": direction,
                    "eval_seed": seed,
                    "evaluations_per_condition": per_seed,
                }
            )

    correct_outputs_identical = 0
    for seed in seeds:
        if (
            schedules[("wrong_slow", seed)]
            != schedules[("wrong_fast", seed)]
        ):
            raise RuntimeError(
                f"Directional evaluation schedules differ for seed {seed}"
            )
        slow_correct = {
            str(row["evaluation_id"]): row
            for row in condition_records["wrong_slow"]["correct"]
            if int(row["eval_seed"]) == seed
        }
        fast_correct = {
            str(row["evaluation_id"]): row
            for row in condition_records["wrong_fast"]["correct"]
            if int(row["eval_seed"]) == seed
        }
        if slow_correct.keys() != fast_correct.keys():
            raise RuntimeError(
                f"Correct output schedules differ for seed {seed}"
            )
        for evaluation_id in slow_correct:
            if _correct_record_signature(
                slow_correct[evaluation_id]
            ) != _correct_record_signature(fast_correct[evaluation_id]):
                raise RuntimeError(
                    "Identical correct context produced a different output: "
                    f"seed={seed}, evaluation={evaluation_id}"
                )
            correct_outputs_identical += 1

    expected_total = len(seeds) * per_seed
    if correct_outputs_identical != expected_total:
        raise RuntimeError("Cross-eval correct output count differs")
    correct = condition_records["wrong_slow"]["correct"]
    wrong_slow = condition_records["wrong_slow"]["wrong"]
    wrong_fast = condition_records["wrong_fast"]["wrong"]
    if any(
        len(rows) != expected_total
        for rows in (correct, wrong_slow, wrong_fast)
    ):
        raise RuntimeError("Total condition count differs from 50×6")

    correct_vs_slow_rows = _paired_rows(
        correct,
        wrong_slow,
        first_label="correct",
        second_label="wrong_slow",
    )
    correct_vs_fast_rows = _paired_rows(
        correct,
        wrong_fast,
        first_label="correct",
        second_label="wrong_fast",
    )
    fast_vs_slow_rows = _paired_rows(
        wrong_fast,
        wrong_slow,
        first_label="wrong_fast",
        second_label="wrong_slow",
    )
    comparisons = {
        "correct_vs_wrong_slow": labeled_pair_summary(
            correct_vs_slow_rows,
            first_label="correct",
            second_label="wrong_slow",
            metadata=metadata,
        ),
        "correct_vs_wrong_fast": labeled_pair_summary(
            correct_vs_fast_rows,
            first_label="correct",
            second_label="wrong_fast",
            metadata=metadata,
        ),
        "wrong_fast_vs_wrong_slow": labeled_pair_summary(
            fast_vs_slow_rows,
            first_label="wrong_fast",
            second_label="wrong_slow",
            metadata=metadata,
        ),
    }
    decisions = directional_decisions(
        correct_vs_slow=comparisons["correct_vs_wrong_slow"],
        correct_vs_fast=comparisons["correct_vs_wrong_fast"],
        fast_vs_slow=comparisons["wrong_fast_vs_wrong_slow"],
        config=config,
    )
    condition_summaries = {
        "correct_speed_context": _condition_summary(
            correct, metadata=metadata
        ),
        "wrong_slower_context_3p1": _condition_summary(
            wrong_slow, metadata=metadata
        ),
        "wrong_faster_context_7p0": _condition_summary(
            wrong_fast, metadata=metadata
        ),
    }
    secondary_diagnostics = _secondary_diagnostics(
        conditions=condition_summaries,
        comparisons=comparisons,
    )

    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "heldout_direction_confirmation",
        "status": "passed",
        "evidence_level": "preregistered_heldout_mechanism_confirmation",
        "config": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "catalog_source_config": {
            "path": str(source_config_path),
            "sha256": source_config_hash,
        },
        "catalog_build_report": {
            "path": str(build_report_path),
            "sha256": sha256_file(build_report_path),
        },
        "catalogs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "wrong_context_speed": float(
                    catalogs[name]["protocol"]["wrong_speed_override"]
                ),
            }
            for name, path in catalog_paths.items()
        },
        "protocol_and_count_audit": {
            "eval_seeds": seeds,
            "evaluations_per_condition_per_seed": per_seed,
            "evaluations_per_condition_per_eval": expected_total,
            "conditions_per_eval": ["correct", "wrong"],
            "raw_records_per_eval": 2 * expected_total,
            "base_queries_per_eval": len(expected_query_ids),
            "two_evals_each_independently_are_50x6": True,
            "identical_schedules_across_evals": True,
            "identical_correct_outputs_across_evals": (
                correct_outputs_identical
            ),
            "passed": True,
        },
        "conditions": condition_summaries,
        "paired_comparisons": comparisons,
        "decisions": decisions,
        "secondary_diagnostics": secondary_diagnostics,
        "conclusion": {
            "classification": decisions["classification"],
            "correctness_aligned_planning_icl_established": decisions[
                "correctness_aligned_planning_icl"
            ]["established"],
            "higher_speed_prompt_bias_confirmed": decisions[
                "higher_speed_prompt_bias"
            ]["confirmed"],
        },
        "inputs": inputs,
    }
    if result_metadata:
        overlap = set(result).intersection(result_metadata)
        if overlap:
            raise ValueError(
                "Result metadata cannot replace analysis fields: "
                f"{sorted(overlap)}"
            )
        result.update(result_metadata)
    output_path = resolve_contextworld_path(
        artifacts["formal_summary"], repo_root=repo_root
    )
    write_json(output_path, result)
    return {**result, "output": str(output_path)}


__all__ = [
    "direction_result_path",
    "directional_decisions",
    "labeled_pair_summary",
    "run_directional_analysis",
]
