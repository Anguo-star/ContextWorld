from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json

from .context_direction_analysis import (
    direction_result_path,
    run_directional_analysis,
)
from .icl_sensitive import sha256_file


CONTROL_STATUS = "preregistered_before_control_execution"
MODEL_ORDER = (
    "M_origheldout",
    "M_synth5matched",
    "M_origplus_synth5",
    "M_speedfull",
)


def load_attribution_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("status") != CONTROL_STATUS:
        raise ValueError(
            f"Expected status={CONTROL_STATUS}, got {payload.get('status')}"
        )
    return payload


def _verified_path(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    path = resolve_contextworld_path(spec["path"], repo_root=repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    observed = sha256_file(path)
    expected = str(spec["sha256"])
    if observed != expected:
        raise RuntimeError(
            f"{label} hash mismatch: observed={observed}, expected={expected}"
        )
    return path


def audit_static_inputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source = config["frozen_source"]
    source_config_path = _verified_path(
        source["directional_config"],
        repo_root=repo_root,
        label="directional config",
    )
    normalizer_path = _verified_path(
        source["normalizer"], repo_root=repo_root, label="normalizer"
    )
    build_report_path = _verified_path(
        source["catalog_build_report"],
        repo_root=repo_root,
        label="catalog build report",
    )
    catalog_paths = {
        name: _verified_path(
            spec, repo_root=repo_root, label=f"{name} catalog"
        )
        for name, spec in source["catalogs"].items()
    }
    speedfull_summary_path = _verified_path(
        source["existing_speedfull_summary"],
        repo_root=repo_root,
        label="existing SpeedFull summary",
    )
    model_paths: dict[str, dict[str, Any]] = {}
    model_ids = [str(row["model_id"]) for row in config["models"]]
    if tuple(model_ids) != MODEL_ORDER:
        raise RuntimeError(
            f"Frozen model order differs: {model_ids} != {MODEL_ORDER}"
        )
    for model in config["models"]:
        model_id = str(model["model_id"])
        checkpoint = _verified_path(
            model["checkpoint"],
            repo_root=repo_root,
            label=f"{model_id} checkpoint",
        )
        model_paths[model_id] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "raw_results": str(
                resolve_contextworld_path(
                    model["raw_results"], repo_root=repo_root
                )
            ),
            "directional_summary": str(
                resolve_contextworld_path(
                    model["directional_summary"], repo_root=repo_root
                )
            ),
            "execute": bool(model["execute"]),
        }

    protocol = config["evaluation_protocol"]
    expected_new = (
        int(protocol["controls_to_execute"])
        * int(protocol["evals_per_model"])
        * int(protocol["conditions_per_eval"])
        * int(protocol["evaluations_per_condition_per_eval"])
    )
    if expected_new != int(protocol["expected_new_raw_records"]):
        raise RuntimeError(
            "Frozen record arithmetic differs: "
            f"{expected_new} != {protocol['expected_new_raw_records']}"
        )

    source_config = yaml.safe_load(
        source_config_path.read_text(encoding="utf-8")
    )
    if source_config["frozen_scope"]["normalizer"] != (
        source["normalizer"]["path"]
    ):
        raise RuntimeError("Directional source normalizer path differs")
    if source_config["formal_eval"]["eval_seeds"] != protocol["eval_seeds"]:
        raise RuntimeError("Directional source eval seeds differ")
    if int(
        source_config["formal_eval"][
            "evaluations_per_condition_per_seed"
        ]
    ) != int(protocol["evaluations_per_condition_per_seed"]):
        raise RuntimeError("Directional per-seed count differs")
    if (
        source_config["frozen_scope"]["stable_worldmodel_commit"]
        != source["stable_worldmodel_commit"]
    ):
        raise RuntimeError("StableWorldModel commit differs")

    return {
        "status": "passed",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "source_directional_config": str(source_config_path),
        "normalizer": str(normalizer_path),
        "catalog_build_report": str(build_report_path),
        "catalogs": {
            name: str(path) for name, path in catalog_paths.items()
        },
        "existing_speedfull_summary": str(speedfull_summary_path),
        "models": model_paths,
        "expected_new_raw_records": expected_new,
    }


def _effective_directional_config(
    *,
    source_config: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(source_config)
    result["frozen_scope"]["model"] = {
        "display_name": str(model["display_name"]),
        "slug": str(model["slug"]),
        "checkpoint": str(model["checkpoint"]["path"]),
    }
    result["artifacts"]["raw_results"] = str(model["raw_results"])
    result["artifacts"]["formal_summary"] = str(
        model["directional_summary"]
    )
    return result


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed":
        raise RuntimeError(f"Result did not pass: {path}")
    return payload


def _record_key(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["eval_seed"]), str(row["evaluation_id"])


def _model_records(
    *,
    model: dict[str, Any],
    config: dict[str, Any],
    repo_root: Path,
) -> tuple[
    dict[str, dict[tuple[int, str], dict[str, Any]]],
    dict[tuple[str, int], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    protocol = config["evaluation_protocol"]
    raw_root = resolve_contextworld_path(
        model["raw_results"], repo_root=repo_root
    )
    expected_checkpoint_sha = str(model["checkpoint"]["sha256"])
    seeds = [int(value) for value in protocol["eval_seeds"]]
    per_seed = int(protocol["evaluations_per_condition_per_seed"])
    conditions: dict[
        str, dict[tuple[int, str], dict[str, Any]]
    ] = {
        "wrong_slow": {},
        "correct": {},
        "wrong_fast": {},
    }
    schedules: dict[tuple[str, int], list[dict[str, Any]]] = {}
    inputs: list[dict[str, Any]] = []
    for direction in ("wrong_slow", "wrong_fast"):
        for seed in seeds:
            path = direction_result_path(
                raw_root, direction=direction, seed=seed
            )
            payload = _load_json(path)
            if payload["checkpoint"]["sha256"] != expected_checkpoint_sha:
                raise RuntimeError(
                    f"Checkpoint differs for {model['model_id']}: {path}"
                )
            if int(payload["protocol"]["eval_seed"]) != seed:
                raise RuntimeError(f"Eval seed differs: {path}")
            records = list(payload["records"])
            if len(records) != 2 * per_seed:
                raise RuntimeError(
                    f"Expected {2 * per_seed} records: {path}"
                )
            schedules[(direction, seed)] = payload["selection"][
                "schedule"
            ]
            for condition in ("correct", "wrong"):
                selected = [
                    row
                    for row in records
                    if row["condition"] == condition
                ]
                if len(selected) != per_seed:
                    raise RuntimeError(
                        f"Expected {per_seed} {condition} records: {path}"
                    )
                label = (
                    "correct"
                    if condition == "correct"
                    else direction
                )
                if label == "correct" and direction == "wrong_fast":
                    continue
                for row in selected:
                    key = _record_key(row)
                    if key in conditions[label]:
                        raise RuntimeError(
                            f"Duplicate {label} record: {key}"
                        )
                    conditions[label][key] = row
            inputs.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "direction": direction,
                    "eval_seed": seed,
                    "records": len(records),
                }
            )
    expected_total = len(seeds) * per_seed
    if any(
        len(rows) != expected_total for rows in conditions.values()
    ):
        raise RuntimeError(
            f"Condition totals differ for {model['model_id']}"
        )
    return conditions, schedules, inputs


def _replan_summary(
    rows: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    selected = list(rows.values())
    by_calls: dict[str, dict[str, int]] = {}
    for calls in sorted({int(row["cem_solve_calls"]) for row in selected}):
        subset = [
            row
            for row in selected
            if int(row["cem_solve_calls"]) == calls
        ]
        by_calls[str(calls)] = {
            "evaluations": len(subset),
            "successes": sum(bool(row["success"]) for row in subset),
            "failures": sum(not bool(row["success"]) for row in subset),
        }
    return {
        "evaluations": len(selected),
        "successes": sum(bool(row["success"]) for row in selected),
        "by_cem_solve_calls": by_calls,
        "exact_steps_to_success_available": False,
    }


def _gate(
    summary: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    effect = float(
        summary["wrong_fast_minus_wrong_slow_success_rate_points"]
    )
    p_value = float(
        summary["paired_sign_test"]["two_sided_p_value"]
    )
    fast_only = int(summary["wrong_fast_only_successes"])
    slow_only = int(summary["wrong_slow_only_successes"])
    gates = {
        "effect_at_least_minimum": effect
        >= float(spec["minimum_effect_pp"]),
        "paired_exact_sign_test_passed": p_value
        <= float(spec["paired_exact_sign_test_p_max"]),
        "fast_only_greater": fast_only > slow_only,
    }
    return {
        "effect_points": effect,
        "paired_p_value": p_value,
        "fast_only": fast_only,
        "slow_only": slow_only,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _cross_model_effect(
    *,
    target: dict[str, dict[tuple[int, str], dict[str, Any]]],
    control: dict[str, dict[tuple[int, str], dict[str, Any]]],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    keys = sorted(target["wrong_fast"])
    for condition in ("wrong_slow", "wrong_fast"):
        if set(target[condition]) != set(control[condition]):
            raise RuntimeError(
                f"Cross-model {condition} schedules differ"
            )
    differences = []
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key in keys:
        shared_fields = (
            "query_id",
            "evaluation_index",
            "repeat_index",
            "speed",
            "template_id",
            "cem_seed",
            "cem_rng_state_sha256_before",
            "goal_state",
        )
        for condition in ("wrong_slow", "wrong_fast"):
            target_row = target[condition][key]
            control_row = control[condition][key]
            if any(
                target_row[field] != control_row[field]
                for field in shared_fields
            ):
                raise RuntimeError(
                    f"Cross-model pairing differs at {key}/{condition}"
                )
        target_delta = float(
            bool(target["wrong_fast"][key]["success"])
        ) - float(bool(target["wrong_slow"][key]["success"]))
        control_delta = float(
            bool(control["wrong_fast"][key]["success"])
        ) - float(bool(control["wrong_slow"][key]["success"]))
        difference = target_delta - control_delta
        differences.append(difference)
        by_seed[int(key[0])].append(difference)
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        len(values),
        size=(int(bootstrap_resamples), len(values)),
    )
    boot = 100.0 * np.mean(values[indices], axis=1)
    return {
        "paired_evaluations": len(values),
        "target_minus_control_context_effect_points": float(
            100.0 * np.mean(values)
        ),
        "evaluation_bootstrap_95_ci_points": [
            float(value)
            for value in np.percentile(boot, [2.5, 97.5])
        ],
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_resamples": int(bootstrap_resamples),
        "by_eval_seed_points": {
            str(seed): float(100.0 * np.mean(selected))
            for seed, selected in sorted(by_seed.items())
        },
        "statistical_boundary": (
            "Descriptive paired evaluation-level bootstrap; the formal "
            "attribution gate is the preregistered pass/fail pattern across "
            "models, and final claims still require multiple training seeds."
        ),
    }


def run_model_attribution_analysis(
    *,
    config: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    static_audit = audit_static_inputs(
        config=config, config_path=config_path, repo_root=repo_root
    )
    source_config_path = Path(
        static_audit["source_directional_config"]
    )
    source_config = yaml.safe_load(
        source_config_path.read_text(encoding="utf-8")
    )
    models = {
        str(row["model_id"]): row for row in config["models"]
    }
    summaries: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        model = models[model_id]
        summary_path = resolve_contextworld_path(
            model["directional_summary"], repo_root=repo_root
        )
        if bool(model["execute"]):
            effective = _effective_directional_config(
                source_config=source_config, model=model
            )
            summaries[model_id] = run_directional_analysis(
                config=effective,
                config_path=config_path,
                repo_root=repo_root,
                catalog_source_config_path=source_config_path,
                result_metadata={
                    "model_attribution_component": {
                        "model_id": model_id,
                        "display_name": str(model["display_name"]),
                        "role": str(model["role"]),
                        "factor_training_support": str(
                            model["factor_training_support"]
                        ),
                    }
                },
            )
        else:
            summaries[model_id] = _load_json(summary_path)

    model_records: dict[
        str, dict[str, dict[tuple[int, str], dict[str, Any]]]
    ] = {}
    model_schedules: dict[
        str, dict[tuple[str, int], list[dict[str, Any]]]
    ] = {}
    raw_inputs: dict[str, list[dict[str, Any]]] = {}
    for model_id in MODEL_ORDER:
        records, schedules, inputs = _model_records(
            model=models[model_id],
            config=config,
            repo_root=repo_root,
        )
        model_records[model_id] = records
        model_schedules[model_id] = schedules
        raw_inputs[model_id] = inputs
    target_schedules = model_schedules["M_speedfull"]
    for model_id, schedules in model_schedules.items():
        if schedules != target_schedules:
            raise RuntimeError(
                f"Cross-model evaluation schedules differ: {model_id}"
            )

    gate_spec = config["decisions_frozen_before_control_execution"][
        "stable_fast_over_slow_gate"
    ]
    model_results: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        model = models[model_id]
        summary = summaries[model_id]
        fast_slow = summary["paired_comparisons"][
            "wrong_fast_vs_wrong_slow"
        ]
        seed_strata = fast_slow["strata"]["by_seed"]
        model_results[model_id] = {
            "display_name": str(model["display_name"]),
            "role": str(model["role"]),
            "factor_training_support": str(
                model["factor_training_support"]
            ),
            "optimizer_steps": int(model["optimizer_steps"]),
            "checkpoint_sha256": str(model["checkpoint"]["sha256"]),
            "directional_summary": {
                "path": str(
                    resolve_contextworld_path(
                        model["directional_summary"],
                        repo_root=repo_root,
                    )
                ),
                "sha256": sha256_file(
                    resolve_contextworld_path(
                        model["directional_summary"],
                        repo_root=repo_root,
                    )
                ),
            },
            "conditions": summary["conditions"],
            "paired_comparisons": summary["paired_comparisons"],
            "stable_fast_over_slow_gate": _gate(
                fast_slow, gate_spec
            ),
            "fast_minus_slow_by_eval_seed_points": {
                seed: float(
                    row[
                        "wrong_fast_minus_wrong_slow_success_rate_points"
                    ]
                )
                for seed, row in seed_strata.items()
            },
            "positive_eval_seeds": sum(
                float(
                    row[
                        "wrong_fast_minus_wrong_slow_success_rate_points"
                    ]
                )
                > 0.0
                for row in seed_strata.values()
            ),
            "replan_stage": {
                condition: _replan_summary(rows)
                for condition, rows in model_records[model_id].items()
            },
            "raw_inputs": raw_inputs[model_id],
        }

    attribution_spec = config[
        "decisions_frozen_before_control_execution"
    ]["integrated_recipe_attribution_support"]
    target_id = str(attribution_spec["target_model"])
    control_ids = [
        str(value) for value in attribution_spec["control_models"]
    ]
    target_passed = bool(
        model_results[target_id]["stable_fast_over_slow_gate"][
            "passed"
        ]
    )
    control_gate_passes = {
        model_id: bool(
            model_results[model_id]["stable_fast_over_slow_gate"][
                "passed"
            ]
        )
        for model_id in control_ids
    }
    attribution_supported = (
        target_passed
        and not any(control_gate_passes.values())
    )
    cross_model = {
        model_id: _cross_model_effect(
            target=model_records[target_id],
            control=model_records[model_id],
            bootstrap_seed=3072 + index,
            bootstrap_resamples=10000,
        )
        for index, model_id in enumerate(control_ids)
    }

    new_records = sum(
        int(row["records"])
        for model_id in control_ids
        for row in raw_inputs[model_id]
    )
    total_records = sum(
        int(row["records"])
        for model_id in MODEL_ORDER
        for row in raw_inputs[model_id]
    )
    expected_new = int(
        config["evaluation_protocol"]["expected_new_raw_records"]
    )
    if new_records != expected_new:
        raise RuntimeError(
            f"New record total differs: {new_records} != {expected_new}"
        )

    result = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "validation_model_attribution",
        "status": "passed",
        "config": static_audit["config"],
        "evidence_boundary": config["evidence_boundary"],
        "static_input_audit": static_audit,
        "protocol_and_count_audit": {
            "models": len(MODEL_ORDER),
            "control_models_executed": len(control_ids),
            "evals_per_model": int(
                config["evaluation_protocol"]["evals_per_model"]
            ),
            "conditions_per_eval": int(
                config["evaluation_protocol"]["conditions_per_eval"]
            ),
            "eval_seeds": config["evaluation_protocol"]["eval_seeds"],
            "evaluations_per_condition_per_seed": int(
                config["evaluation_protocol"][
                    "evaluations_per_condition_per_seed"
                ]
            ),
            "evaluations_per_condition_per_eval": int(
                config["evaluation_protocol"][
                    "evaluations_per_condition_per_eval"
                ]
            ),
            "new_control_raw_records": new_records,
            "all_four_model_raw_records": total_records,
            "cross_model_schedules_identical": True,
            "passed": True,
        },
        "models": model_results,
        "cross_model_context_effect": cross_model,
        "decisions": {
            "frozen_specification": attribution_spec,
            "target_gate_passed": target_passed,
            "control_gate_passes": control_gate_passes,
            "only_target_gate_passed": attribution_supported,
            "integrated_recipe_attribution_supported": (
                attribution_supported
            ),
            "classification": (
                "speedfull_only_stable_fast_over_slow"
                if attribution_supported
                else "fast_over_slow_not_specific_to_speedfull"
            ),
        },
        "conclusion": {
            "allowed_claim": (
                attribution_spec["interpretation_if_passed"]
                if attribution_supported
                else attribution_spec["interpretation_if_any_control_passes"]
            ),
            "single_training_seed_boundary": True,
            "final_test": False,
        },
    }
    output_path = resolve_contextworld_path(
        config["artifacts"]["final_summary"], repo_root=repo_root
    )
    write_json(output_path, result)
    return {**result, "output": str(output_path)}


__all__ = [
    "audit_static_inputs",
    "load_attribution_config",
    "run_model_attribution_analysis",
]
