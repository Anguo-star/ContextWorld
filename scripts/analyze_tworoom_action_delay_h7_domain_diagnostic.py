#!/usr/bin/env python3
"""Aggregate nine History-7 Action Delay domain diagnostics."""

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

from contextworld.evaluation.action_delay_h7_domain_score import (
    RATE_METRICS,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_scoring_v1.yaml"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": (
            float(array.std(ddof=1)) if len(array) > 1 else 0.0
        ),
    }


def _models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for role, rows in config["models"].items():
        for row in rows:
            slug = str(row["slug"])
            _require(slug not in result, f"模型 slug 重复：{slug}")
            result[slug] = {**row, "role": role}
    return result


def _load_results(
    config: dict[str, Any],
    *,
    config_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    result_root = resolve_contextworld_path(
        config["artifacts"]["results_root"],
        repo_root=ROOT,
    )
    expected_config_hash = file_sha256(config_path)
    results = {}
    hashes = {}
    checkpoint_hashes = set()
    for slug, model in _models(config).items():
        path = result_root / f"{slug}.json"
        _require(path.is_file(), f"缺少训练域诊断结果：{path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks = {
            "status_completed": payload.get("status") == "completed",
            "model_slug_exact": payload.get("model_slug") == slug,
            "model_id_exact": payload.get("model_id")
            == model["model_id"],
            "role_exact": payload.get("training_role")
            == model["role"],
            "seed_exact": int(payload.get("training_seed", -1))
            == int(model["training_seed"]),
            "config_hash_exact": payload.get("identity", {}).get(
                "scoring_config_sha256"
            )
            == expected_config_hash,
            "training_receipt_passed": payload.get(
                "training_receipt", {}
            ).get("passed")
            is True,
            "score_audit_passed": payload.get(
                "score_audit", {}
            ).get("passed")
            is True,
            "model_state_frozen": payload.get(
                "model_state_sha256_before"
            )
            == payload.get("model_state_sha256_after"),
        }
        _require(
            all(checks.values()),
            f"{slug}: 结果审计失败 {checks}",
        )
        checkpoint_hash = payload["identity"]["checkpoint_sha256"]
        _require(
            checkpoint_hash not in checkpoint_hashes,
            f"checkpoint 重复：{slug}",
        )
        checkpoint_hashes.add(checkpoint_hash)
        results[slug] = payload
        hashes[slug] = file_sha256(path)
    return results, hashes


def _source_h1(
    result: dict[str, Any],
    track: str,
) -> dict[str, Any]:
    return result["tracks"][track]["by_horizon"]["1"][
        "source_supervised_target"
    ]


def _track_gate(
    config: dict[str, Any],
    result: dict[str, Any],
    track: str,
) -> dict[str, Any]:
    gate = config["diagnosis_gate"]["source_supervised_h1"]
    source = _source_h1(result, track)
    overall = source["overall"]
    seeds = source["by_diagnostic_eval_seed"]
    checks = {
        "target_selection_rate": overall[
            "exact_target_selection_rate"
        ]
        >= float(gate["exact_target_selection_rate_minimum"]),
        "history_selection_rate": overall[
            "exact_history_selection_rate"
        ]
        >= float(gate["exact_history_selection_rate_minimum"]),
        "strict_history_win_rate": overall[
            "matching_history_strict_win_rate"
        ]
        >= float(gate["matching_history_strict_win_rate_minimum"]),
        "every_eval_seed_target_selection": all(
            row["exact_target_selection_rate"]
            >= float(gate["every_eval_seed_rate_minimum"])
            for row in seeds.values()
        ),
        "every_eval_seed_history_selection": all(
            row["exact_history_selection_rate"]
            >= float(gate["every_eval_seed_rate_minimum"])
            for row in seeds.values()
        ),
        "every_eval_seed_history_margin_positive": all(
            row["mean_history_margin"] > 0.0
            for row in seeds.values()
        ),
    }
    target_fitted = (
        checks["target_selection_rate"]
        and checks["every_eval_seed_target_selection"]
    )
    history_bound = (
        checks["history_selection_rate"]
        and checks["strict_history_win_rate"]
        and checks["every_eval_seed_history_selection"]
        and checks["every_eval_seed_history_margin_positive"]
    )
    return {
        "metrics": overall,
        "checks": checks,
        "target_fitted": target_fitted,
        "history_bound": history_bound,
        "passed": target_fitted and history_bound,
    }


def _paired_track_comparison(
    single: dict[str, Any],
    multi: dict[str, Any],
    track: str,
) -> dict[str, Any]:
    single_values = _source_h1(single, track)["overall"]
    multi_values = _source_h1(multi, track)["overall"]
    metrics = {
        metric: {
            "single": float(single_values[metric]),
            "multi": float(multi_values[metric]),
            "delta": float(
                multi_values[metric] - single_values[metric]
            ),
            "passed": float(multi_values[metric])
            > float(single_values[metric]),
        }
        for metric in RATE_METRICS
    }
    return {
        "metrics": metrics,
        "passed": all(row["passed"] for row in metrics.values()),
    }


def _group_summary(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slug, model in _models(config).items():
        grouped[model["role"]].append(results[slug])
    tracks = list(config["evaluation"]["tracks"])
    return {
        role: {
            "models": len(rows),
            "training_seeds": sorted(
                int(row["training_seed"]) for row in rows
            ),
            "tracks": {
                track: {
                    "source_supervised_h1": {
                        metric: _mean_std(
                            [
                                float(
                                    _source_h1(row, track)["overall"][
                                        metric
                                    ]
                                )
                                for row in rows
                            ]
                        )
                        for metric in (
                            *RATE_METRICS,
                            "mean_history_loss_ratio",
                        )
                    },
                    "trajectory_overall": {
                        metric: _mean_std(
                            [
                                float(
                                    row["tracks"][track][
                                        "trajectory"
                                    ]["overall"][metric]
                                )
                                for row in rows
                            ]
                        )
                        for metric in (
                            *RATE_METRICS,
                            "mean_history_loss_ratio",
                        )
                    },
                    "trajectory_alignment": {
                        metric: _mean_std(
                            [
                                float(
                                    row["tracks"][track][
                                        "latent_alignment"
                                    ]["trajectory"][metric]
                                )
                                for row in rows
                            ]
                        )
                        for metric in (
                            "prediction_to_target_pair_magnitude_ratio",
                            "pair_direction_cosine_mean",
                            "pair_direction_positive_fraction",
                        )
                    },
                }
                for track in tracks
            },
        }
        for role, rows in grouped.items()
    }


def _seed_decisions(
    config: dict[str, Any],
    results: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lookup = {
        (model["role"], int(model["training_seed"])): slug
        for slug, model in _models(config).items()
    }
    train_tracks = [
        track
        for track in config["evaluation"]["tracks"]
        if str(track).startswith("training_replay")
    ]
    loader_tracks = [
        track
        for track in config["evaluation"]["tracks"]
        if str(track).startswith("loader_validation")
    ]
    decisions = {}
    paired = {}
    for seed in config["diagnosis_gate"]["required_training_seeds"]:
        seed = int(seed)
        single_slug = lookup[("single_delay_control", seed)]
        multi_slug = lookup[("multi_delay_target", seed)]
        single = results[single_slug]
        multi = results[multi_slug]
        track_gates = {
            track: _track_gate(config, multi, track)
            for track in config["evaluation"]["tracks"]
        }
        comparisons = {
            track: _paired_track_comparison(
                single,
                multi,
                track,
            )
            for track in config["evaluation"]["tracks"]
        }
        paired[str(seed)] = comparisons
        training_target_fitted = all(
            track_gates[track]["target_fitted"]
            for track in train_tracks
        )
        training_history_bound = all(
            track_gates[track]["history_bound"]
            for track in train_tracks
        )
        loader_target_fitted = all(
            track_gates[track]["target_fitted"]
            for track in loader_tracks
        )
        loader_history_bound = all(
            track_gates[track]["history_bound"]
            for track in loader_tracks
        )
        paired_training_passed = all(
            comparisons[track]["passed"] for track in train_tracks
        )
        if not training_target_fitted:
            diagnosis = "source_supervised_h1_not_fitted"
        elif not training_history_bound:
            diagnosis = "h1_fitted_without_delay_history_binding"
        elif not paired_training_passed:
            diagnosis = "training_fit_not_attributable_to_multi_delay"
        elif not loader_target_fitted:
            diagnosis = "loader_validation_target_generalization_failed"
        elif not loader_history_bound:
            diagnosis = "loader_validation_history_binding_failed"
        else:
            diagnosis = "training_domain_passed_check_formal_distribution"
        decisions[str(seed)] = {
            "multi_slug": multi_slug,
            "single_slug": single_slug,
            "track_gates": track_gates,
            "checks": {
                "training_target_fitted": training_target_fitted,
                "training_history_bound": training_history_bound,
                "loader_target_fitted": loader_target_fitted,
                "loader_history_bound": loader_history_bound,
                "paired_training_passed": paired_training_passed,
            },
            "diagnosis": diagnosis,
        }
    return decisions, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(
        config.get("benchmark")
        == "tworoom_action_delay_history7_domain_diagnostic_scoring_v1",
        "不是冻结的训练域诊断评分配置",
    )
    results, result_hashes = _load_results(
        config,
        config_path=config_path,
    )
    decisions, paired = _seed_decisions(config, results)
    diagnoses = {
        value["diagnosis"] for value in decisions.values()
    }
    if len(diagnoses) == 1:
        diagnosis = next(iter(diagnoses))
    else:
        diagnosis = "mixed_training_seed_diagnoses"
    formal_summary_path = resolve_contextworld_path(
        config["source_identity"]["formal_validation_summary"]["path"],
        repo_root=ROOT,
    )
    formal_summary = json.loads(
        formal_summary_path.read_text(encoding="utf-8")
    )
    formal_gate_passed = bool(
        formal_summary["primary_prediction_gate"]["passed"]
    )
    claim_by_diagnosis = {
        "source_supervised_h1_not_fitted": (
            "The final multi-delay checkpoints do not reliably fit even "
            "the source-supervised h1 transitions on exact training "
            "geometries."
        ),
        "h1_fitted_without_delay_history_binding": (
            "The source h1 transition is fitted, but the prediction is "
            "not bound to the matching delay history."
        ),
        "training_fit_not_attributable_to_multi_delay": (
            "The training transition is fitted, but the multi-delay "
            "model does not consistently outperform its paired "
            "single-delay control."
        ),
        "loader_validation_target_generalization_failed": (
            "The source transition is fitted on training geometries but "
            "does not generalize to same-distribution loader geometry."
        ),
        "loader_validation_history_binding_failed": (
            "The target future generalizes to loader geometry, but the "
            "model does not bind it to the matching delay history."
        ),
        "training_domain_passed_check_formal_distribution": (
            "The training-domain diagnostic passes; the remaining formal "
            "failure is specific to the Benchmark distribution or its "
            "larger delay support."
        ),
        "mixed_training_seed_diagnoses": (
            "Training seeds do not support one stable root-cause class."
        ),
    }
    payload = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "diagnosis": diagnosis,
        "claim": claim_by_diagnosis[diagnosis],
        "formal_validation_gate_passed": formal_gate_passed,
        "chance_reference_three_way": 1.0 / 3.0,
        "seed_decisions": decisions,
        "paired_multi_vs_single": paired,
        "group_summary": _group_summary(config, results),
        "identity": {
            "scoring_config": str(config_path),
            "scoring_config_sha256": file_sha256(config_path),
            "aggregation_entrypoint": str(Path(__file__).resolve()),
            "aggregation_entrypoint_sha256": file_sha256(
                Path(__file__).resolve()
            ),
            "result_files": result_hashes,
            "formal_validation_summary": {
                "path": str(formal_summary_path),
                "sha256": file_sha256(formal_summary_path),
            },
        },
    }
    output = resolve_contextworld_path(
        args.output or config["artifacts"]["aggregate"],
        repo_root=ROOT,
    )
    write_json(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "diagnosis": payload["diagnosis"],
                "claim": payload["claim"],
                "seed_decisions": {
                    seed: {
                        "checks": row["checks"],
                        "diagnosis": row["diagnosis"],
                    }
                    for seed, row in decisions.items()
                },
                "output": str(output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
