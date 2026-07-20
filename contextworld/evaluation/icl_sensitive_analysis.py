from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json

from .icl_catalog import validate_context_query_catalog
from .icl_sensitive import sha256_file


def speed_slug(speed: float) -> str:
    return f"{float(speed):g}".replace(".", "p").replace("-", "m")


def calibration_result_path(
    root: Path,
    *,
    model_slug: str,
    speed: float,
    seed: int,
) -> Path:
    return (
        root
        / model_slug
        / f"paired_speed{speed_slug(speed)}_n72_s{int(seed)}.json"
    )


def formal_result_path(
    root: Path,
    *,
    model_slug: str,
    condition: str,
    seed: int,
) -> Path:
    if condition not in {"none", "paired"}:
        raise ValueError(f"Unknown formal result condition: {condition}")
    return root / model_slug / f"{condition}_n50_s{int(seed)}.json"


def exact_paired_sign_test(
    correct_only: int,
    wrong_only: int,
) -> dict[str, Any]:
    discordant = int(correct_only + wrong_only)
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "two_sided_p_value": 1.0,
        }
    smaller = min(int(correct_only), int(wrong_only))
    tail = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2**discordant)
    return {
        "discordant_pairs": discordant,
        "two_sided_p_value": float(min(1.0, 2.0 * tail)),
    }


def _load_passed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Evaluation did not pass: {path}")
    return payload


def _assert_eval_protocol(
    payload: dict[str, Any],
    *,
    seed: int,
    planning: dict[str, Any],
    paired: bool,
    path: Path,
) -> None:
    protocol = payload["protocol"]
    expected = {
        "eval_seed": int(seed),
        "eval_budget": int(planning["eval_budget_raw_steps"]),
        "horizon": int(planning["horizon_action_blocks"]),
        "receding_horizon": int(
            planning["receding_horizon_action_blocks"]
        ),
        "cem_steps": int(planning["cem_steps"]),
        "cem_topk": int(planning["cem_topk"]),
    }
    expected[
        "cem_num_samples" if paired else "cem_samples"
    ] = int(planning["cem_samples"])
    mismatches = {
        key: {
            "expected": value,
            "observed": protocol.get(key),
        }
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if paired and not np.isclose(
        float(protocol.get("cem_var_scale", float("nan"))),
        float(planning["cem_var_scale"]),
        rtol=0.0,
        atol=0.0,
    ):
        mismatches["cem_var_scale"] = {
            "expected": float(planning["cem_var_scale"]),
            "observed": protocol.get("cem_var_scale"),
        }
    if mismatches:
        raise RuntimeError(
            f"Frozen planning protocol mismatch in {path}: {mismatches}"
        )


def _assert_checkpoint(
    payload: dict[str, Any],
    *,
    expected: Path,
    path: Path,
) -> None:
    expected = expected.resolve()
    observed = Path(payload["checkpoint"]["path"]).resolve()
    if observed != expected:
        raise RuntimeError(
            f"Checkpoint path mismatch in {path}: "
            f"{observed} != {expected}"
        )
    expected_sha = _cached_sha256(str(expected))
    if payload["checkpoint"]["sha256"] != expected_sha:
        raise RuntimeError(f"Checkpoint hash mismatch in {path}")


@lru_cache(maxsize=None)
def _cached_sha256(path: str) -> str:
    return sha256_file(Path(path))


def _query_metadata(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metadata = {}
    for bundle in catalog["bundles"]:
        query_id = str(bundle["query_id"])
        if query_id in metadata:
            raise RuntimeError(f"Duplicate query_id in catalog: {query_id}")
        metadata[query_id] = {
            "distance_bin": int(bundle["template"]["distance_bin"]),
            "template_id": str(bundle["template"]["template_id"]),
            "speed": float(bundle["query_factors"]["agent.speed"]),
        }
    return metadata


def summarize_paired_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        raise ValueError("Cannot summarize zero paired rows")
    correct_successes = sum(
        bool(row["correct_success"]) for row in selected
    )
    wrong_successes = sum(bool(row["wrong_success"]) for row in selected)
    correct_only = sum(
        bool(row["correct_success"]) and not bool(row["wrong_success"])
        for row in selected
    )
    wrong_only = sum(
        bool(row["wrong_success"]) and not bool(row["correct_success"])
        for row in selected
    )
    total = len(selected)
    return {
        "pairs": total,
        "correct": {
            "successes": int(correct_successes),
            "success_rate_percent": float(
                100.0 * correct_successes / total
            ),
        },
        "wrong": {
            "successes": int(wrong_successes),
            "success_rate_percent": float(
                100.0 * wrong_successes / total
            ),
        },
        "pooled_correct_wrong_success_rate_percent": float(
            100.0 * (correct_successes + wrong_successes) / (2 * total)
        ),
        "correct_minus_wrong_success_rate_points": float(
            100.0 * (correct_successes - wrong_successes) / total
        ),
        "correct_only_successes": int(correct_only),
        "wrong_only_successes": int(wrong_only),
        "both_successes": int(
            sum(
                bool(row["correct_success"])
                and bool(row["wrong_success"])
                for row in selected
            )
        ),
        "neither_successes": int(
            sum(
                not bool(row["correct_success"])
                and not bool(row["wrong_success"])
                for row in selected
            )
        ),
        "paired_sign_test": exact_paired_sign_test(
            int(correct_only), int(wrong_only)
        ),
        "wrong_minus_correct_mean_final_distance": float(
            np.mean(
                [
                    float(row["wrong_minus_correct_final_distance"])
                    for row in selected
                ]
            )
        ),
    }


def _rank_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        -float(row["correct_minus_wrong_success_rate_points"]),
        abs(
            float(row["pooled_correct_wrong_success_rate_percent"])
            - 50.0
        ),
        int(row["distance_bin"]),
    )


def select_distance_bins(
    summaries: list[dict[str, Any]],
    *,
    spacing: int,
    maximum_bins: int,
) -> list[int]:
    eligible = sorted(
        [row for row in summaries if bool(row["eligible"])],
        key=_rank_key,
    )
    if not eligible:
        return []
    selected = [int(eligible[0]["distance_bin"])]
    if maximum_bins <= 1:
        return selected
    neighbors = [
        row
        for row in eligible[1:]
        if abs(int(row["distance_bin"]) - selected[0]) == int(spacing)
    ]
    if neighbors:
        selected.append(int(sorted(neighbors, key=_rank_key)[0]["distance_bin"]))
    return sorted(selected)


def _selected_heldout_catalog(
    heldout_bank: dict[str, Any],
    *,
    selected_bins: list[int],
    calibration_selection: dict[str, Any],
) -> dict[str, Any]:
    selected = copy.deepcopy(heldout_bank)
    selected_set = {int(value) for value in selected_bins}
    selected["bundles"] = [
        bundle
        for bundle in selected["bundles"]
        if int(bundle["template"]["distance_bin"]) in selected_set
    ]
    selected["geometry_bank"] = [
        geometry
        for geometry in selected["geometry_bank"]
        if int(geometry["distance_bin"]) in selected_set
    ]
    selected["summary"].update(
        {
            "bundles": len(selected["bundles"]),
            "by_family": {"speed": len(selected["bundles"])},
            "physical_payloads": len(selected["bundles"]),
            "distance_bins": sorted(selected_set),
            "base_geometries": len(selected["geometry_bank"]),
        }
    )
    selected["selection"] = {
        "kind": "preregistered_distance_calibration",
        "selected_distance_bins": sorted(selected_set),
        "calibration_selection_sha256": calibration_selection[
            "selection_fingerprint"
        ],
        "heldout_geometry_was_not_scored_during_calibration": True,
    }
    return selected


def run_calibration_selection(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    calibration_catalog_path = resolve_contextworld_path(
        config["artifacts"]["calibration_catalog"],
        repo_root=repo_root,
    )
    heldout_bank_path = resolve_contextworld_path(
        config["artifacts"]["heldout_bank_catalog"],
        repo_root=repo_root,
    )
    calibration_root = resolve_contextworld_path(
        config["artifacts"]["calibration_results"],
        repo_root=repo_root,
    )
    output_path = resolve_contextworld_path(
        config["artifacts"]["calibration_selection"],
        repo_root=repo_root,
    )
    selected_catalog_path = resolve_contextworld_path(
        config["artifacts"]["selected_heldout_catalog"],
        repo_root=repo_root,
    )
    calibration_catalog = json.loads(
        calibration_catalog_path.read_text(encoding="utf-8")
    )
    heldout_bank = json.loads(
        heldout_bank_path.read_text(encoding="utf-8")
    )
    query_metadata = _query_metadata(calibration_catalog)
    model_slug = str(config["calibration"]["model"]["slug"])
    speeds = [
        float(value) for value in config["frozen_scope"]["agent_speeds"]
    ]
    seeds = [int(value) for value in config["calibration"]["eval_seeds"]]
    expected_per_result = int(
        config["calibration"]["evaluations_per_speed_per_seed"]
    )
    planning = config["frozen_scope"]["planning"]
    calibration_checkpoint = resolve_contextworld_path(
        config["calibration"]["model"]["checkpoint"],
        repo_root=repo_root,
    )
    rows: list[dict[str, Any]] = []
    inputs = []
    for speed in speeds:
        for seed in seeds:
            path = calibration_result_path(
                calibration_root,
                model_slug=model_slug,
                speed=speed,
                seed=seed,
            )
            payload = _load_passed(path)
            _assert_eval_protocol(
                payload,
                seed=seed,
                planning=planning,
                paired=True,
                path=path,
            )
            _assert_checkpoint(
                payload,
                expected=calibration_checkpoint,
                path=path,
            )
            pairs = list(payload["aggregate"]["pairs"])
            if len(pairs) != expected_per_result:
                raise RuntimeError(
                    f"Expected {expected_per_result} pairs in {path}, "
                    f"found {len(pairs)}"
                )
            for pair in pairs:
                metadata = query_metadata.get(str(pair["query_id"]))
                if metadata is None:
                    raise RuntimeError(
                        f"Unknown calibration query: {pair['query_id']}"
                    )
                if not np.isclose(
                    float(pair["speed"]), speed, rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError(f"Speed mismatch in {path}")
                rows.append(
                    {
                        **pair,
                        **metadata,
                        "eval_seed": seed,
                    }
                )
            inputs.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "speed": speed,
                    "eval_seed": seed,
                    "pairs": len(pairs),
                }
            )

    selection_spec = config["calibration"][
        "distance_selection_frozen_before_execution"
    ]
    distance_summaries = []
    distance_bins = [
        int(value)
        for value in config["catalog_generation"]["distance_bins_px"]
    ]
    expected_pairs_per_bin = (
        len(speeds)
        * int(
            config["catalog_generation"]["calibration"][
                "variants_per_distance"
            ]
        )
        * len(seeds)
        * int(
            config["calibration"][
                "expected_repeats_per_base_query_per_seed"
            ]
        )
    )
    for distance in distance_bins:
        distance_rows = [
            row for row in rows if int(row["distance_bin"]) == distance
        ]
        if len(distance_rows) != expected_pairs_per_bin:
            raise RuntimeError(
                f"Distance {distance}: expected {expected_pairs_per_bin} "
                f"pairs, found {len(distance_rows)}"
            )
        summary = summarize_paired_rows(distance_rows)
        pooled_fraction = (
            summary["pooled_correct_wrong_success_rate_percent"] / 100.0
        )
        gates = {
            "pooled_above_min": pooled_fraction
            >= float(
                selection_spec[
                    "pooled_correct_wrong_success_rate_min"
                ]
            ),
            "pooled_below_max": pooled_fraction
            <= float(
                selection_spec[
                    "pooled_correct_wrong_success_rate_max"
                ]
            ),
            "effect_at_least_minimum": float(
                summary["correct_minus_wrong_success_rate_points"]
            )
            >= float(
                selection_spec[
                    "correct_minus_wrong_success_rate_min_pp"
                ]
            ),
            "correct_only_greater_than_wrong_only": int(
                summary["correct_only_successes"]
            )
            > int(summary["wrong_only_successes"]),
        }
        by_speed = {}
        for speed in speeds:
            speed_rows = [
                row
                for row in distance_rows
                if np.isclose(
                    float(row["speed"]), speed, rtol=0.0, atol=1e-6
                )
            ]
            by_speed[f"{speed:g}"] = summarize_paired_rows(speed_rows)
        distance_summaries.append(
            {
                "distance_bin": distance,
                **summary,
                "by_speed": by_speed,
                "eligibility_gates": gates,
                "eligible": all(gates.values()),
            }
        )

    selected_bins = select_distance_bins(
        distance_summaries,
        spacing=int(selection_spec["distance_bin_spacing_px"]),
        maximum_bins=int(selection_spec["maximum_selected_bins"]),
    )
    selection_fingerprint_payload = {
        "config_sha256": sha256_file(config_path),
        "calibration_catalog_sha256": sha256_file(
            calibration_catalog_path
        ),
        "heldout_bank_catalog_sha256": sha256_file(heldout_bank_path),
        "input_sha256": [row["sha256"] for row in inputs],
        "selected_distance_bins": selected_bins,
    }
    selection_fingerprint = hashlib.sha256(
        json.dumps(
            selection_fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    status = "passed" if selected_bins else "no_eligible_distance_bin"
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "calibration_distance_selection",
        "status": status,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "catalogs": {
            "calibration": {
                "path": str(calibration_catalog_path),
                "sha256": sha256_file(calibration_catalog_path),
            },
            "heldout_bank": {
                "path": str(heldout_bank_path),
                "sha256": sha256_file(heldout_bank_path),
                "scored_during_calibration": False,
            },
        },
        "inputs": inputs,
        "expected_and_observed": {
            "result_files": len(inputs),
            "total_pairs": len(rows),
            "pairs_per_distance_bin": expected_pairs_per_bin,
        },
        "frozen_selection_specification": selection_spec,
        "distance_bins": distance_summaries,
        "selected_distance_bins": selected_bins,
        "selection_fingerprint": selection_fingerprint,
        "formal_eval_authorized": bool(selected_bins),
        "interpretation": (
            "At least one preregistered ICL-sensitive calibration distance "
            "passed all gates; formal evaluation must use only the untouched "
            "heldout bank."
            if selected_bins
            else
            "No preregistered distance passed all gates. Formal four-model "
            "evaluation must stop without relaxing thresholds."
        ),
    }
    if selected_bins:
        selected_catalog = _selected_heldout_catalog(
            heldout_bank,
            selected_bins=selected_bins,
            calibration_selection=result,
        )
        write_json(selected_catalog_path, selected_catalog)
        validation = validate_context_query_catalog(
            selected_catalog_path,
            repo_root=repo_root,
            replay_simulator=False,
            family="speed",
        )
        if not validation["passed"]:
            raise RuntimeError(
                "Selected heldout catalog validation failed: "
                f"{validation['failures'][:5]}"
            )
        result["selected_heldout_catalog"] = {
            "path": str(selected_catalog_path),
            "sha256": sha256_file(selected_catalog_path),
            "bundles": len(selected_catalog["bundles"]),
            "templates": [
                geometry["template_id"]
                for geometry in selected_catalog["geometry_bank"]
            ],
            "validation": validation,
        }
    write_json(output_path, result)
    return result


def _group_summary(
    records: list[dict[str, Any]],
    *,
    query_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize zero records")

    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(bool(row["success"]) for row in selected)
        return {
            "evaluations": len(selected),
            "successes": int(successes),
            "success_rate_percent": float(
                100.0 * successes / len(selected)
            ),
            "mean_final_distance": float(
                np.mean([float(row["final_distance"]) for row in selected])
            ),
        }

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_seed": defaultdict(list),
        "by_distance": defaultdict(list),
        "by_speed": defaultdict(list),
    }
    for row in records:
        metadata = query_metadata[str(row["query_id"])]
        groups["by_seed"][str(int(row["eval_seed"]))].append(row)
        groups["by_distance"][
            str(int(metadata["distance_bin"]))
        ].append(row)
        groups["by_speed"][f"{float(row['speed']):g}"].append(row)
    return {
        **summarize(records),
        **{
            group_name: {
                key: summarize(selected)
                for key, selected in sorted(
                    grouped.items(),
                    key=lambda item: float(item[0]),
                )
            }
            for group_name, grouped in groups.items()
        },
    }


def _paired_from_condition_records(
    correct: list[dict[str, Any]],
    wrong: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    correct_lookup = {
        (int(row["eval_seed"]), str(row["evaluation_id"])): row
        for row in correct
    }
    wrong_lookup = {
        (int(row["eval_seed"]), str(row["evaluation_id"])): row
        for row in wrong
    }
    if correct_lookup.keys() != wrong_lookup.keys():
        raise RuntimeError("Correct/wrong formal schedules differ")
    rows = []
    for key in sorted(correct_lookup):
        correct_row = correct_lookup[key]
        wrong_row = wrong_lookup[key]
        rows.append(
            {
                "eval_seed": key[0],
                "evaluation_id": key[1],
                "query_id": correct_row["query_id"],
                "speed": float(correct_row["speed"]),
                "correct_success": bool(correct_row["success"]),
                "wrong_success": bool(wrong_row["success"]),
                "wrong_minus_correct_final_distance": float(
                    wrong_row["final_distance"]
                    - correct_row["final_distance"]
                ),
            }
        )
    return rows


def _paired_strata(
    rows: list[dict[str, Any]],
    *,
    query_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_seed": defaultdict(list),
        "by_distance": defaultdict(list),
        "by_speed": defaultdict(list),
    }
    for row in rows:
        metadata = query_metadata[str(row["query_id"])]
        groups["by_seed"][str(int(row["eval_seed"]))].append(row)
        groups["by_distance"][
            str(int(metadata["distance_bin"]))
        ].append(row)
        groups["by_speed"][f"{float(row['speed']):g}"].append(row)
    return {
        group_name: {
            key: summarize_paired_rows(selected)
            for key, selected in sorted(
                grouped.items(),
                key=lambda item: float(item[0]),
            )
        }
        for group_name, grouped in groups.items()
    }


def _condition_vs_none(
    condition_records: list[dict[str, Any]],
    none_records: list[dict[str, Any]],
    *,
    condition_name: str,
) -> dict[str, Any]:
    condition_lookup = {
        (int(row["eval_seed"]), str(row["evaluation_id"])): row
        for row in condition_records
    }
    none_lookup = {
        (int(row["eval_seed"]), str(row["evaluation_id"])): row
        for row in none_records
    }
    if condition_lookup.keys() != none_lookup.keys():
        raise RuntimeError(f"{condition_name}/none formal schedules differ")
    condition_only = 0
    none_only = 0
    for key in condition_lookup:
        condition_success = bool(condition_lookup[key]["success"])
        none_success = bool(none_lookup[key]["success"])
        condition_only += int(condition_success and not none_success)
        none_only += int(none_success and not condition_success)
    total = len(condition_lookup)
    condition_successes = sum(
        bool(row["success"]) for row in condition_records
    )
    none_successes = sum(bool(row["success"]) for row in none_records)
    return {
        "pairs": total,
        "condition_only_successes": condition_only,
        "none_only_successes": none_only,
        "condition_minus_none_success_rate_points": float(
            100.0 * (condition_successes - none_successes) / total
        ),
        "paired_sign_test": exact_paired_sign_test(
            condition_only, none_only
        ),
    }


def run_formal_analysis(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    selection_path = resolve_contextworld_path(
        config["artifacts"]["calibration_selection"],
        repo_root=repo_root,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("formal_eval_authorized"):
        raise RuntimeError(
            "Calibration did not authorize formal evaluation"
        )
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["selected_heldout_catalog"],
        repo_root=repo_root,
    )
    formal_root = resolve_contextworld_path(
        config["artifacts"]["formal_results"],
        repo_root=repo_root,
    )
    output_path = resolve_contextworld_path(
        config["artifacts"]["formal_summary"],
        repo_root=repo_root,
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    query_metadata = _query_metadata(catalog)
    seeds = [
        int(value) for value in config["formal_heldout"]["eval_seeds"]
    ]
    expected_per_condition = int(
        config["formal_heldout"]["evaluations_per_seed_per_condition"]
    )
    planning = config["frozen_scope"]["planning"]
    model_summaries: dict[str, Any] = {}
    all_inputs = []
    for model in config["formal_heldout"]["models"]:
        slug = str(model["slug"])
        checkpoint = resolve_contextworld_path(
            model["checkpoint"], repo_root=repo_root
        )
        conditions: dict[str, list[dict[str, Any]]] = {
            "none": [],
            "correct": [],
            "wrong": [],
        }
        model_inputs = []
        for seed in seeds:
            noctx_path = formal_result_path(
                formal_root,
                model_slug=slug,
                condition="none",
                seed=seed,
            )
            paired_path = formal_result_path(
                formal_root,
                model_slug=slug,
                condition="paired",
                seed=seed,
            )
            noctx = _load_passed(noctx_path)
            paired = _load_passed(paired_path)
            _assert_eval_protocol(
                noctx,
                seed=seed,
                planning=planning,
                paired=False,
                path=noctx_path,
            )
            _assert_eval_protocol(
                paired,
                seed=seed,
                planning=planning,
                paired=True,
                path=paired_path,
            )
            _assert_checkpoint(
                noctx,
                expected=checkpoint,
                path=noctx_path,
            )
            _assert_checkpoint(
                paired,
                expected=checkpoint,
                path=paired_path,
            )
            if len(noctx["records"]) != expected_per_condition:
                raise RuntimeError(f"Wrong no-context count: {noctx_path}")
            paired_by_condition = {
                condition: [
                    row
                    for row in paired["records"]
                    if row["condition"] == condition
                ]
                for condition in ("correct", "wrong")
            }
            if any(
                len(rows) != expected_per_condition
                for rows in paired_by_condition.values()
            ):
                raise RuntimeError(f"Wrong paired count: {paired_path}")
            for row in noctx["records"]:
                row["eval_seed"] = seed
                if str(row["query_id"]) not in query_metadata:
                    raise RuntimeError(
                        f"Unknown formal query in {noctx_path}"
                    )
            for condition, records in paired_by_condition.items():
                for row in records:
                    row["eval_seed"] = seed
                    if str(row["query_id"]) not in query_metadata:
                        raise RuntimeError(
                            f"Unknown formal query in {paired_path}"
                        )
                conditions[condition].extend(records)
            conditions["none"].extend(noctx["records"])
            model_inputs.extend(
                [
                    {
                        "path": str(noctx_path),
                        "sha256": sha256_file(noctx_path),
                        "condition": "none",
                        "eval_seed": seed,
                    },
                    {
                        "path": str(paired_path),
                        "sha256": sha256_file(paired_path),
                        "condition": "correct_wrong",
                        "eval_seed": seed,
                    },
                ]
            )
        paired_rows = _paired_from_condition_records(
            conditions["correct"], conditions["wrong"]
        )
        paired_summary = summarize_paired_rows(paired_rows)
        paired_summary["strata"] = _paired_strata(
            paired_rows,
            query_metadata=query_metadata,
        )
        paired_summary["unique_base_queries"] = len(
            {row["query_id"] for row in paired_rows}
        )
        summary = {
            "display_name": model["display_name"],
            "slug": slug,
            "role": model["role"],
            "conditions": {
                condition: _group_summary(
                    records, query_metadata=query_metadata
                )
                for condition, records in conditions.items()
            },
            "paired_correct_wrong": paired_summary,
            "context_vs_none": {
                condition: _condition_vs_none(
                    conditions[condition],
                    conditions["none"],
                    condition_name=condition,
                )
                for condition in ("correct", "wrong")
            },
            "inputs": model_inputs,
        }
        model_summaries[str(model["display_name"])] = summary
        all_inputs.extend(model_inputs)

    primary_name = str(
        config["formal_heldout"][
            "primary_decision_frozen_before_execution"
        ]["model"]
    )
    primary = model_summaries[primary_name]["paired_correct_wrong"]
    decision_spec = config["formal_heldout"][
        "primary_decision_frozen_before_execution"
    ]
    correct_fraction = (
        primary["correct"]["success_rate_percent"] / 100.0
    )
    decision_gates = {
        "effect_at_least_minimum": float(
            primary["correct_minus_wrong_success_rate_points"]
        )
        >= float(decision_spec["correct_minus_wrong_success_rate_min_pp"]),
        "correct_only_greater_than_wrong_only": int(
            primary["correct_only_successes"]
        )
        > int(primary["wrong_only_successes"]),
        "paired_exact_sign_test_passed": float(
            primary["paired_sign_test"]["two_sided_p_value"]
        )
        <= float(decision_spec["paired_exact_sign_test_p_max"]),
        "correct_success_above_minimum": correct_fraction
        >= float(decision_spec["correct_success_rate_min"]),
        "correct_success_below_maximum": correct_fraction
        <= float(decision_spec["correct_success_rate_max"]),
    }
    primary_passed = all(decision_gates.values())
    control_effects = {
        name: float(
            summary["paired_correct_wrong"][
                "correct_minus_wrong_success_rate_points"
            ]
        )
        for name, summary in model_summaries.items()
        if name != primary_name
    }
    best_control = max(control_effects, key=control_effects.get)
    margin = float(
        primary["correct_minus_wrong_success_rate_points"]
        - control_effects[best_control]
    )
    specificity_minimum = float(
        config["formal_heldout"]["specificity_diagnostic"][
            "primary_effect_minus_best_control_min_pp"
        ]
    )
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "formal_heldout_four_model_eval",
        "status": "passed",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "calibration_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selected_distance_bins": selection[
                "selected_distance_bins"
            ],
        },
        "selected_heldout_catalog": {
            "path": str(catalog_path),
            "sha256": sha256_file(catalog_path),
            "bundles": len(catalog["bundles"]),
            "scored_during_calibration": False,
        },
        "protocol": {
            "eval_seeds": seeds,
            "evaluations_per_seed_per_condition": expected_per_condition,
            "total_evaluations_per_model_per_condition": (
                len(seeds) * expected_per_condition
            ),
            "conditions": ["none", "correct", "wrong"],
            "planning": config["frozen_scope"]["planning"],
        },
        "models": model_summaries,
        "primary_decision": {
            "model": primary_name,
            "frozen_specification": decision_spec,
            "gates": decision_gates,
            "planning_success_icl_established": primary_passed,
        },
        "specificity_diagnostic": {
            "control_effects_points": control_effects,
            "best_control": best_control,
            "primary_minus_best_control_effect_points": margin,
            "minimum_points": specificity_minimum,
            "passed": margin >= specificity_minimum,
            "required_for_primary_decision": False,
        },
        "conclusion": (
            "Heldout planning-success ICL is established under the frozen "
            "CEM and preregistered decision rule."
            if primary_passed
            else
            "Heldout planning-success ICL is not established under the "
            "frozen CEM and preregistered decision rule."
        ),
        "inputs": all_inputs,
    }
    write_json(output_path, result)
    return result


def run_calibration_diagnostics(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run explicitly post-hoc decomposition after the frozen v1 stop."""

    selection_path = resolve_contextworld_path(
        config["artifacts"]["calibration_selection"],
        repo_root=repo_root,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "no_eligible_distance_bin":
        raise RuntimeError(
            "Post-hoc stop diagnostics are only defined for a stopped v1"
        )
    catalog_path = resolve_contextworld_path(
        config["artifacts"]["calibration_catalog"],
        repo_root=repo_root,
    )
    calibration_root = resolve_contextworld_path(
        config["artifacts"]["calibration_results"],
        repo_root=repo_root,
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    metadata = _query_metadata(catalog)
    wrong_speed_by_query = {
        str(bundle["query_id"]): float(
            bundle["conditions"]["wrong"]["factors"]["agent.speed"]
        )
        for bundle in catalog["bundles"]
    }
    geometry_by_template = {
        str(geometry["template_id"]): geometry
        for geometry in catalog["geometry_bank"]
    }
    model_slug = str(config["calibration"]["model"]["slug"])
    rows = []
    input_paths = []
    for speed in config["frozen_scope"]["agent_speeds"]:
        for seed in config["calibration"]["eval_seeds"]:
            path = calibration_result_path(
                calibration_root,
                model_slug=model_slug,
                speed=float(speed),
                seed=int(seed),
            )
            payload = _load_passed(path)
            for pair in payload["aggregate"]["pairs"]:
                query_id = str(pair["query_id"])
                rows.append(
                    {
                        **pair,
                        **metadata[query_id],
                        "eval_seed": int(seed),
                        "wrong_context_speed": wrong_speed_by_query[
                            query_id
                        ],
                    }
                )
            input_paths.append(
                {"path": str(path), "sha256": sha256_file(path)}
            )

    difficulty_bins = [
        int(row["distance_bin"])
        for row in selection["distance_bins"]
        if bool(row["eligibility_gates"]["pooled_above_min"])
        and bool(row["eligibility_gates"]["pooled_below_max"])
    ]
    difficulty_rows = [
        row
        for row in rows
        if int(row["distance_bin"]) in set(difficulty_bins)
    ]

    by_speed = {}
    for speed in sorted({float(row["speed"]) for row in rows}):
        selected = [
            row
            for row in difficulty_rows
            if np.isclose(
                float(row["speed"]), speed, rtol=0.0, atol=1e-6
            )
        ]
        by_speed[f"{speed:g}"] = {
            **summarize_paired_rows(selected),
            "wrong_context_speed": float(
                selected[0]["wrong_context_speed"]
            ),
            "wrong_context_direction": (
                "faster"
                if float(selected[0]["wrong_context_speed"]) > speed
                else "slower"
            ),
        }

    wrong_faster = [
        row
        for row in difficulty_rows
        if float(row["wrong_context_speed"]) > float(row["speed"])
    ]
    wrong_slower = [
        row
        for row in difficulty_rows
        if float(row["wrong_context_speed"]) < float(row["speed"])
    ]

    high_prompt_rows = []
    for row in difficulty_rows:
        wrong_is_higher = (
            float(row["wrong_context_speed"]) > float(row["speed"])
        )
        high_prompt_rows.append(
            {
                **row,
                "correct_success": (
                    bool(row["wrong_success"])
                    if wrong_is_higher
                    else bool(row["correct_success"])
                ),
                "wrong_success": (
                    bool(row["correct_success"])
                    if wrong_is_higher
                    else bool(row["wrong_success"])
                ),
                "wrong_minus_correct_final_distance": (
                    -float(row["wrong_minus_correct_final_distance"])
                    if wrong_is_higher
                    else float(row["wrong_minus_correct_final_distance"])
                ),
            }
        )
    high_prompt_summary = summarize_paired_rows(high_prompt_rows)
    high_prompt_summary = {
        "pairs": high_prompt_summary["pairs"],
        "higher_speed_context": high_prompt_summary["correct"],
        "lower_speed_context": high_prompt_summary["wrong"],
        "higher_minus_lower_success_rate_points": high_prompt_summary[
            "correct_minus_wrong_success_rate_points"
        ],
        "higher_only_successes": high_prompt_summary[
            "correct_only_successes"
        ],
        "lower_only_successes": high_prompt_summary[
            "wrong_only_successes"
        ],
        "paired_sign_test": high_prompt_summary["paired_sign_test"],
        "lower_minus_higher_mean_final_distance": high_prompt_summary[
            "wrong_minus_correct_mean_final_distance"
        ],
    }

    by_template = {}
    for template_id in sorted(geometry_by_template):
        selected = [
            row for row in rows if row["template_id"] == template_id
        ]
        geometry = geometry_by_template[template_id]
        delta = (
            np.asarray(geometry["goal_state"], dtype=np.float64)
            - np.asarray(geometry["reset_state"], dtype=np.float64)
        )
        summary = summarize_paired_rows(selected)
        by_template[template_id] = {
            **summary,
            "distance_bin": int(geometry["distance_bin"]),
            "reset_state": geometry["reset_state"],
            "goal_state": geometry["goal_state"],
            "unit_goal_direction": (
                delta / float(geometry["distance_bin"])
            ).tolist(),
        }
    geometry_heterogeneity = {}
    for distance in sorted(
        {int(row["distance_bin"]) for row in selection["distance_bins"]}
    ):
        summaries = [
            summary
            for summary in by_template.values()
            if int(summary["distance_bin"]) == distance
        ]
        pooled = [
            float(
                summary[
                    "pooled_correct_wrong_success_rate_percent"
                ]
            )
            for summary in summaries
        ]
        geometry_heterogeneity[str(distance)] = {
            "templates": len(summaries),
            "minimum_template_pooled_success_percent": min(pooled),
            "maximum_template_pooled_success_percent": max(pooled),
            "within_distance_range_points": max(pooled) - min(pooled),
        }

    wrong_faster_summary = summarize_paired_rows(wrong_faster)
    wrong_slower_summary = summarize_paired_rows(wrong_slower)
    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "posthoc_calibration_stop_diagnostics",
        "status": "passed",
        "evidence_level": "exploratory_posthoc_not_formal_confirmation",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "frozen_v1_selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "status": selection["status"],
            "formal_eval_authorized": False,
        },
        "difficulty_eligible_ignoring_context_effect": {
            "distance_bins": difficulty_bins,
            "pairs": len(difficulty_rows),
            "definition": (
                "Only the preregistered pooled-success 10%-90% gates are "
                "used; this does not alter the failed primary decision."
            ),
        },
        "by_actual_speed": by_speed,
        "wrong_context_direction_groups": {
            "wrong_context_faster_than_query": {
                "actual_speeds": sorted(
                    {float(row["speed"]) for row in wrong_faster}
                ),
                **wrong_faster_summary,
            },
            "wrong_context_slower_than_query": {
                "actual_speeds": sorted(
                    {float(row["speed"]) for row in wrong_slower}
                ),
                **wrong_slower_summary,
            },
            "correct_minus_wrong_effect_interaction_points": float(
                wrong_slower_summary[
                    "correct_minus_wrong_success_rate_points"
                ]
                - wrong_faster_summary[
                    "correct_minus_wrong_success_rate_points"
                ]
            ),
        },
        "prompt_speed_bias_relabeling": {
            **high_prompt_summary,
            "interpretation": (
                "This relabels each pair by higher-versus-lower context "
                "speed irrespective of factual correctness. It is post-hoc."
            ),
        },
        "geometry_heterogeneity": geometry_heterogeneity,
        "by_template": by_template,
        "diagnostic_conclusion": (
            "Usable difficulty exists, but correctness does not control the "
            "planning effect. The sign flips with whether the wrong prompt "
            "is faster or slower than the true dynamics, and geometry "
            "difficulty varies strongly within a fixed Euclidean distance."
        ),
        "inputs": input_paths,
    }
    output = selection_path.parent / "calibration_diagnostics_posthoc.json"
    write_json(output, result)
    return {**result, "output": str(output)}


__all__ = [
    "calibration_result_path",
    "exact_paired_sign_test",
    "formal_result_path",
    "run_calibration_selection",
    "run_calibration_diagnostics",
    "run_formal_analysis",
    "select_distance_bins",
    "speed_slug",
    "summarize_paired_rows",
]
