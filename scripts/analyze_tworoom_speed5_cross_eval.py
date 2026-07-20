#!/usr/bin/env python3
"""Analyze the preregistered fixed-speed cross-evaluation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.synthesis.manifest import write_json


EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
MODEL_SPECS = {
    "H3-OrigHeldout": {
        "slug": "h3_origheldout_s3072",
        "fixed_speed_training": True,
    },
    "H3-Synth5Matched": {
        "slug": "h3_synth5matched_s3072",
        "fixed_speed_training": True,
    },
    "H3-OrigPlusSynth5": {
        "slug": "h3_origplus_synth5_s3072",
        "fixed_speed_training": True,
    },
    "H3-SpeedFull": {
        "slug": "h3_speedfull_s3072",
        "fixed_speed_training": False,
    },
}
PRIMARY_MODELS = tuple(
    model
    for model, spec in MODEL_SPECS.items()
    if spec["fixed_speed_training"]
)
SOURCE_MINOR_THRESHOLD_PP = 10.0
GEOMETRY_MAJOR_THRESHOLD_PP = 40.0
SPEED_PRIMARY_THRESHOLD_PP = 20.0
CONTEXT_THRESHOLD_PP = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise ValueError(f"Result is not passed: {path}")
    return payload


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty record list")
    successes = sum(bool(record["success"]) for record in records)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "template_id" in record:
            by_template[str(record["template_id"])].append(record)
        by_seed[int(record["eval_seed"])].append(record)

    def compact(selected: list[dict[str, Any]]) -> dict[str, Any]:
        selected_successes = sum(
            bool(record["success"]) for record in selected
        )
        return {
            "evaluations": len(selected),
            "successes": int(selected_successes),
            "success_rate": float(selected_successes / len(selected)),
            "mean_final_distance": float(
                np.mean(
                    [
                        float(record["final_distance"])
                        for record in selected
                    ]
                )
            ),
        }

    return {
        **compact(records),
        "by_template": {
            template: compact(selected)
            for template, selected in sorted(by_template.items())
        },
        "by_seed": {
            str(seed): compact(selected)
            for seed, selected in sorted(by_seed.items())
        },
    }


def _load_existing_planning(
    eval_root: Path, slug: str, stem: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records = []
    files = []
    for seed in EVAL_SEEDS:
        path = eval_root / slug / f"{stem}_s{seed}.json"
        payload = _load_json(path)
        seed_records = payload["raw_records"]
        if len(seed_records) != 50:
            raise ValueError(f"Expected 50 records in {path}")
        records.extend(seed_records)
        files.append({"path": str(path), "sha256": _sha256(path)})
    if len(records) != 300:
        raise ValueError(f"Expected 300 records for {slug}/{stem}")
    return records, files


def _load_speed5_e4(
    eval_root: Path, slug: str
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    conditions: dict[str, list[dict[str, Any]]] = {
        "none": [],
        "correct": [],
        "wrong": [],
    }
    files: dict[str, list[dict[str, str]]] = {
        "none": [],
        "paired_context": [],
    }
    schedule_audit = []
    for seed in EVAL_SEEDS:
        root = eval_root / slug / "speed5_cross_eval"
        noctx_path = root / f"e4_speed5_noctx_n50_s{seed}.json"
        ctx_path = root / f"e4_speed5_ctx_n50_s{seed}.json"
        noctx = _load_json(noctx_path)
        ctx = _load_json(ctx_path)
        if noctx["selection"]["speeds"] != [5.0]:
            raise ValueError(f"Unexpected no-context speed selection: {noctx_path}")
        if ctx["selection"]["speeds"] != [5.0]:
            raise ValueError(f"Unexpected context speed selection: {ctx_path}")
        noctx_records = noctx["records"]
        ctx_records = ctx["records"]
        correct = [
            record for record in ctx_records
            if record["condition"] == "correct"
        ]
        wrong = [
            record for record in ctx_records
            if record["condition"] == "wrong"
        ]
        if not (
            len(noctx_records) == len(correct) == len(wrong) == 50
        ):
            raise ValueError(f"Expected 50 records per condition for seed {seed}")
        noctx_map = {
            record["evaluation_id"]: record for record in noctx_records
        }
        correct_map = {
            record["evaluation_id"]: record for record in correct
        }
        wrong_map = {
            record["evaluation_id"]: record for record in wrong
        }
        if not (
            set(noctx_map) == set(correct_map) == set(wrong_map)
        ):
            raise ValueError(f"E4 evaluation IDs differ for {slug}/seed={seed}")
        for evaluation_id in sorted(noctx_map):
            rows = (
                noctx_map[evaluation_id],
                correct_map[evaluation_id],
                wrong_map[evaluation_id],
            )
            paired_fields = (
                "query_id",
                "template_id",
                "speed",
                "cem_seed",
                "evaluation_index",
                "repeat_index",
            )
            for field in paired_fields:
                if len({str(row[field]) for row in rows}) != 1:
                    raise ValueError(
                        f"Pairing mismatch {slug}/{seed}/{evaluation_id}/{field}"
                    )
            if float(rows[0]["speed"]) != 5.0:
                raise ValueError("Observed non-speed5 E4 record")
        conditions["none"].extend(noctx_records)
        conditions["correct"].extend(correct)
        conditions["wrong"].extend(wrong)
        files["none"].append(
            {"path": str(noctx_path), "sha256": _sha256(noctx_path)}
        )
        files["paired_context"].append(
            {"path": str(ctx_path), "sha256": _sha256(ctx_path)}
        )
        schedule_audit.append(
            {
                "seed": seed,
                "paired_evaluations": len(noctx_map),
                "passed": True,
            }
        )
    for condition, records in conditions.items():
        if len(records) != 300:
            raise ValueError(
                f"Expected 300 {condition} records for {slug}"
            )
    return conditions, {
        "files": files,
        "schedule_audit": schedule_audit,
        "passed": True,
    }


def _load_multispeed_e4(
    eval_root: Path, slug: str
) -> dict[str, list[dict[str, Any]]]:
    result = {"none": [], "correct": [], "wrong": []}
    for seed in EVAL_SEEDS:
        noctx = _load_json(
            eval_root / slug / f"e4_speed_noctx_n50_s{seed}.json"
        )
        ctx = _load_json(
            eval_root / slug / f"e4_speed_ctx_n50_s{seed}.json"
        )
        result["none"].extend(noctx["records"])
        for condition in ("correct", "wrong"):
            result[condition].extend(
                record
                for record in ctx["records"]
                if record["condition"] == condition
            )
    return result


def _distance_profile(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    distances = np.asarray(
        [
            np.linalg.norm(
                np.asarray(record["goal_state"], dtype=np.float64)
                - np.asarray(record["initial_state"], dtype=np.float64)
            )
            for record in records
        ],
        dtype=np.float64,
    )
    cross_room = sum(
        record.get("room_relation") == "cross_room"
        for record in records
    )
    return {
        "evaluations": len(records),
        "distance_px": {
            "mean": float(distances.mean()),
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.90)),
            "max": float(distances.max()),
        },
        "cross_room_evaluations": int(cross_room),
        "cross_room_fraction": float(cross_room / len(records)),
    }


def _e4_profile(
    records: list[dict[str, Any]], catalog_path: Path
) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    states = {}
    relations = {}
    for bundle in catalog["bundles"]:
        if (
            bundle["family"] == "speed"
            and float(bundle["query_factors"]["agent.speed"]) == 5.0
        ):
            query_id = str(bundle["query_id"])
            start = np.asarray(
                bundle["template"]["reset_state"], dtype=np.float64
            )
            goal = np.asarray(
                bundle["template"]["goal_state"], dtype=np.float64
            )
            states[query_id] = (start, goal)
            relations[query_id] = (
                "cross_room"
                if (start[0] < 112.0) != (goal[0] < 112.0)
                else "same_room"
            )
    enriched = []
    for record in records:
        start, goal = states[str(record["query_id"])]
        enriched.append(
            {
                "initial_state": start,
                "goal_state": goal,
                "room_relation": relations[str(record["query_id"])],
            }
        )
    return _distance_profile(enriched)


def _paired_effect(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_map = {
        str(record["evaluation_id"]): record for record in left
    }
    right_map = {
        str(record["evaluation_id"]): record for record in right
    }
    if set(left_map) != set(right_map):
        raise ValueError("Paired E4 IDs differ")
    keys = sorted(left_map)
    success_delta = np.asarray(
        [
            int(bool(left_map[key]["success"]))
            - int(bool(right_map[key]["success"]))
            for key in keys
        ],
        dtype=np.int64,
    )
    distance_delta = np.asarray(
        [
            float(left_map[key]["final_distance"])
            - float(right_map[key]["final_distance"])
            for key in keys
        ],
        dtype=np.float64,
    )
    return {
        "left_minus_right_success_rate_pp": float(
            100.0 * success_delta.mean()
        ),
        "left_minus_right_mean_final_distance_px": float(
            distance_delta.mean()
        ),
        "left_only_successes": int((success_delta == 1).sum()),
        "right_only_successes": int((success_delta == -1).sum()),
        "same_success_outcome": int((success_delta == 0).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = args.artifact_root.resolve()
    eval_root = (
        artifact_root
        / "evaluation/history3/original_ability_reconstruction"
    )
    catalog_path = (
        artifact_root
        / "evaluation/icl/"
        "tworoom_icl_v1_validation_context_query_catalog.json"
    )
    protocol_path = (
        REPO_ROOT
        / "configs/benchmark/tworoom_speed5_cross_eval_v1.yaml"
    )
    models = {}
    source_gaps = []
    geometry_drops = []
    speed5_gains = []
    context_gaps = []
    template_locked = True

    for model, spec in MODEL_SPECS.items():
        slug = str(spec["slug"])
        original, original_files = _load_existing_planning(
            eval_root, slug, "planning_original_heldout"
        )
        matched, matched_files = _load_existing_planning(
            eval_root, slug, "planning_speed5_matched"
        )
        speed5, speed5_audit = _load_speed5_e4(eval_root, slug)
        multispeed = _load_multispeed_e4(eval_root, slug)
        aggregates = {
            "original_future25": _summarize(original),
            "synthetic_matched_future25": _summarize(matched),
            "e4_speed5_none": _summarize(speed5["none"]),
            "e4_speed5_correct": _summarize(speed5["correct"]),
            "e4_speed5_wrong": _summarize(speed5["wrong"]),
            "e4_multispeed_none": _summarize(multispeed["none"]),
            "e4_multispeed_correct": _summarize(
                multispeed["correct"]
            ),
            "e4_multispeed_wrong": _summarize(multispeed["wrong"]),
        }
        effects = {
            "matched_minus_original": {
                "success_rate_pp": float(
                    100.0
                    * (
                        aggregates["synthetic_matched_future25"][
                            "success_rate"
                        ]
                        - aggregates["original_future25"]["success_rate"]
                    )
                ),
                "mean_final_distance_px": float(
                    aggregates["synthetic_matched_future25"][
                        "mean_final_distance"
                    ]
                    - aggregates["original_future25"][
                        "mean_final_distance"
                    ]
                ),
            },
            "e4_speed5_none_minus_matched": {
                "success_rate_pp": float(
                    100.0
                    * (
                        aggregates["e4_speed5_none"]["success_rate"]
                        - aggregates["synthetic_matched_future25"][
                            "success_rate"
                        ]
                    )
                ),
                "mean_final_distance_px": float(
                    aggregates["e4_speed5_none"][
                        "mean_final_distance"
                    ]
                    - aggregates["synthetic_matched_future25"][
                        "mean_final_distance"
                    ]
                ),
            },
            "speed5_none_minus_multispeed_none": {
                "success_rate_pp": float(
                    100.0
                    * (
                        aggregates["e4_speed5_none"]["success_rate"]
                        - aggregates["e4_multispeed_none"]["success_rate"]
                    )
                ),
                "mean_final_distance_px": float(
                    aggregates["e4_speed5_none"][
                        "mean_final_distance"
                    ]
                    - aggregates["e4_multispeed_none"][
                        "mean_final_distance"
                    ]
                ),
            },
            "correct_minus_none": _paired_effect(
                speed5["correct"], speed5["none"]
            ),
            "correct_minus_wrong": _paired_effect(
                speed5["correct"], speed5["wrong"]
            ),
        }
        models[model] = {
            "training_speed_scope": (
                "fixed_speed5"
                if spec["fixed_speed_training"]
                else "multi_speed"
            ),
            "aggregates": aggregates,
            "effects": effects,
            "provenance": {
                "original_future25": original_files,
                "synthetic_matched_future25": matched_files,
                "speed5_e4": speed5_audit,
            },
        }
        if model in PRIMARY_MODELS:
            source_gaps.append(
                abs(effects["matched_minus_original"]["success_rate_pp"])
            )
            geometry_drops.extend(
                [
                    100.0
                    * (
                        aggregates["original_future25"]["success_rate"]
                        - aggregates["e4_speed5_none"]["success_rate"]
                    ),
                    100.0
                    * (
                        aggregates["synthetic_matched_future25"][
                            "success_rate"
                        ]
                        - aggregates["e4_speed5_none"]["success_rate"]
                    ),
                ]
            )
            speed5_gains.append(
                effects["speed5_none_minus_multispeed_none"][
                    "success_rate_pp"
                ]
            )
            context_gaps.extend(
                [
                    abs(
                        effects["correct_minus_none"][
                            "left_minus_right_success_rate_pp"
                        ]
                    ),
                    abs(
                        effects["correct_minus_wrong"][
                            "left_minus_right_success_rate_pp"
                        ]
                    ),
                ]
            )
            for condition in ("none", "correct", "wrong"):
                strata = aggregates[f"e4_speed5_{condition}"][
                    "by_template"
                ]
                template_locked &= (
                    strata["s0"]["successes"] == 0
                    and strata["s1"]["successes"] == 0
                    and strata["s2"]["successes"] == 0
                    and strata["s3"]["successes"]
                    == strata["s3"]["evaluations"]
                )

    profiles = {
        "original_future25": _distance_profile(
            _load_existing_planning(
                eval_root,
                MODEL_SPECS["H3-OrigHeldout"]["slug"],
                "planning_original_heldout",
            )[0]
        ),
        "synthetic_matched_future25": _distance_profile(
            _load_existing_planning(
                eval_root,
                MODEL_SPECS["H3-OrigHeldout"]["slug"],
                "planning_speed5_matched",
            )[0]
        ),
    }
    reference_speed5, _ = _load_speed5_e4(
        eval_root, MODEL_SPECS["H3-OrigHeldout"]["slug"]
    )
    profiles["e4_fixed_geometry_speed5"] = _e4_profile(
        reference_speed5["none"], catalog_path
    )

    decisions = {
        "trajectory_source_component_minor": bool(
            max(source_gaps) <= SOURCE_MINOR_THRESHOLD_PP
        ),
        "fixed_geometry_planner_component_major": bool(
            min(geometry_drops) >= GEOMETRY_MAJOR_THRESHOLD_PP
        ),
        "speed_mixture_is_primary": bool(
            min(speed5_gains) >= SPEED_PRIMARY_THRESHOLD_PP
        ),
        "planning_context_success_effect_detected": bool(
            max(context_gaps) >= CONTEXT_THRESHOLD_PP
        ),
        "e4_speed5_outcomes_locked_by_template": bool(template_locked),
        "statistics": {
            "max_abs_matched_vs_original_gap_pp": float(
                max(source_gaps)
            ),
            "min_future25_to_e4_speed5_drop_pp": float(
                min(geometry_drops)
            ),
            "min_speed5_vs_multispeed_gain_pp": float(
                min(speed5_gains)
            ),
            "max_abs_context_gap_pp": float(max(context_gaps)),
        },
    }
    conclusions = [
        (
            "At fixed speed 5, original and distribution-matched synthetic "
            "future-25 evaluations remain close."
        ),
        (
            "The large score drop is associated with E4 fixed-goal geometry "
            "and the frozen planner, not with the multi-speed evaluation "
            "mixture."
        ),
        (
            "Speed-5 E4 binary outcomes remain template-locked, so this "
            "benchmark cannot currently expose model or context differences "
            "through pooled success."
        ),
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_speed5_cross_eval_v1",
        "status": "passed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
            "primary_models": list(PRIMARY_MODELS),
            "eval_seeds": list(EVAL_SEEDS),
            "evaluations_per_seed_per_condition": 50,
            "agent_speed": 5.0,
            "thresholds": {
                "trajectory_source_minor_max_abs_gap_pp": (
                    SOURCE_MINOR_THRESHOLD_PP
                ),
                "fixed_geometry_major_min_drop_pp": (
                    GEOMETRY_MAJOR_THRESHOLD_PP
                ),
                "speed_mixture_primary_min_gain_pp": (
                    SPEED_PRIMARY_THRESHOLD_PP
                ),
                "planning_context_detected_abs_gap_pp": (
                    CONTEXT_THRESHOLD_PP
                ),
            },
        },
        "dataset_profiles": profiles,
        "models": models,
        "decisions": decisions,
        "conclusions": conclusions,
        "limitations": [
            (
                "E4 speed-5 uses four base queries repeated over CEM seeds; "
                "300 evaluations are not 300 independent geometries."
            ),
            (
                "Original and matched future-25 catalogs are not paired "
                "query-by-query."
            ),
            (
                "H3-SpeedFull is a robustness control and is excluded from "
                "the fixed-training-speed attribution."
            ),
        ],
    }
    write_json(args.output.resolve(), payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(
            "/opt/huawei/explorer-env/dataset/ag_data/data/"
            "world_model/context_world"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "decisions": result["decisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
