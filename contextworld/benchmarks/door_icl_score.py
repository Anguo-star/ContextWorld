from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import DoorICLModelAdapter
from contextworld.benchmarks.door_icl_data import (
    DEFAULT_DOOR_RELEASE_CONFIG,
    DoorICLEvalDataset,
    load_door_icl_release,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
)
from contextworld.benchmarks.test_gate_completion import (
    load_test_gate_completion_config,
    threshold_source_block,
    unit_bootstrap_draws,
)
from contextworld.evaluation.hidden_passage_validation import (
    HISTORY_CONDITIONS,
    OTHER_HISTORY,
    SAME_HISTORY,
    TRUE_RULES,
    canonical_sha256,
    file_sha256,
    paired_effect_rows,
    score_validation_assets,
    summarize_validation_records,
)
from contextworld.paths import repository_root


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    if not rows:
        raise ValueError("Cannot average an empty collection")
    return float(np.mean(np.asarray(rows, dtype=np.float64)))


def door_gate_completion_metrics(
    *,
    predicted: np.ndarray,
    encoded_targets: np.ndarray,
    query_ids: list[str],
    static_query_ids: list[str],
    config: dict[str, Any],
    full_protocol: bool,
) -> dict[str, Any]:
    """Additive anti-shortcut gates for the Door Rule Public Test.

    Every frozen payload holds the query's true future under both rules, and
    the model rolls out one prediction per history condition.  Pairing the
    passable-evidence history with the blocked-evidence history against the
    two rule futures reproduces the paired hidden-dynamics probe of the six
    fully gated components: swapping only the demonstrated rule evidence must
    move the prediction along the true inter-rule future displacement.

    All thresholds come from the additive gate-completion config; the frozen
    decision gate (two-target accuracy, history win, stratified bootstrap)
    stays untouched.
    """

    section = config["door"]
    gates = section["gates"]
    bootstrap = section["bootstrap"]
    predicted_all = np.asarray(predicted, dtype=np.float64)
    targets = np.asarray(encoded_targets, dtype=np.float64)
    if predicted_all.ndim != 3 or targets.ndim != 3:
        raise ValueError("Door gate latents must be (queries, member, dim)")
    if len(predicted_all) != len(query_ids) or len(targets) != len(
        query_ids
    ):
        raise ValueError("One latent row per query is required")
    if len(predicted_all) != len(static_query_ids):
        raise ValueError("One static query id per latent row is required")
    if tuple(TRUE_RULES) != ("passable", "blocked"):
        raise RuntimeError("Door true-rule order changed")
    if tuple(HISTORY_CONDITIONS[:2]) != (
        "observed_passable",
        "observed_blocked",
    ):
        raise RuntimeError("Door evidence-history order changed")

    pred_passable = predicted_all[:, 0]
    pred_blocked = predicted_all[:, 1]
    target_passable = targets[:, 0]
    target_blocked = targets[:, 1]

    latent_response, response_records = paired_latent_response_metrics(
        pair_ids=[str(query_id) for query_id in query_ids],
        predicted_first=pred_passable,
        predicted_second=pred_blocked,
        target_first=target_passable,
        target_second=target_blocked,
    )
    switch = (
        np.sum(
            (pred_blocked - pred_passable)
            * (target_blocked - target_passable),
            axis=-1,
        )
        > 0.0
    )

    def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.square(left - right).mean(axis=-1)

    def _evidence_prediction(condition: str) -> np.ndarray:
        if condition == "observed_passable":
            return pred_passable
        if condition == "observed_blocked":
            return pred_blocked
        raise RuntimeError(f"Unknown evidence history: {condition}")

    correct_history_by_rule: dict[str, list[bool]] = {}
    correct_target_by_rule: dict[str, list[bool]] = {}
    for rule in TRUE_RULES:
        same_prediction = _evidence_prediction(SAME_HISTORY[rule])
        other_prediction = _evidence_prediction(OTHER_HISTORY[rule])
        true_target = (
            target_passable if rule == "passable" else target_blocked
        )
        other_target = (
            target_blocked if rule == "passable" else target_passable
        )
        correct_history_by_rule[rule] = (
            _mse(same_prediction, true_target)
            < _mse(other_prediction, true_target)
        ).tolist()
        correct_target_by_rule[rule] = (
            _mse(same_prediction, true_target)
            < _mse(same_prediction, other_target)
        ).tolist()

    unit_rows = np.asarray(
        [
            [
                float(switch[index]),
                float(correct_target_by_rule["passable"][index]),
                float(correct_target_by_rule["blocked"][index]),
            ]
            for index in range(len(query_ids))
        ],
        dtype=np.float64,
    )
    draws = unit_bootstrap_draws(
        unit_rows,
        resamples=int(bootstrap["resamples"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    tail = 0.5 * (1.0 - float(bootstrap["confidence_level"]))
    lower_bounds = {
        "context_switch_rate": float(np.quantile(draws[:, 0], tail)),
        "worst_rule_correct_target_choice_rate": float(
            np.quantile(np.min(draws[:, 1:], axis=1), tail)
        ),
    }
    uncertainty = {
        "method": "paired_static_query_bootstrap",
        "unit": str(bootstrap["unit"]),
        "resamples": int(bootstrap["resamples"]),
        "confidence_level": float(bootstrap["confidence_level"]),
        "random_seed": int(bootstrap["random_seed"]),
        "lower_bounds": lower_bounds,
    }

    by_rule_target = {
        rule: float(np.mean(correct_target_by_rule[rule]))
        for rule in TRUE_RULES
    }
    metrics = {
        "queries": len(query_ids),
        "pair_count": len(query_ids),
        "decision_count": 2 * len(query_ids),
        "response_pair_count": len(response_records),
        "context_switch_rate": float(switch.mean()),
        "correct_history_rate": float(
            np.mean(
                [
                    value
                    for rule in TRUE_RULES
                    for value in correct_history_by_rule[rule]
                ]
            )
        ),
        "correct_target_choice_rate_by_true_rule": by_rule_target,
        "worst_rule_correct_target_choice_rate": float(
            min(by_rule_target.values())
        ),
        "latent_response": latent_response,
    }
    checks = {
        "context_switch_rate": bool(
            metrics["context_switch_rate"]
            >= float(gates["context_switch_rate_minimum"])
        ),
        "worst_rule_correct_target_choice_rate": bool(
            metrics["worst_rule_correct_target_choice_rate"]
            >= float(
                gates["worst_rule_correct_target_choice_rate_minimum"]
            )
        ),
    }
    checks.update(
        paired_latent_response_gate_checks(metrics, thresholds=gates)
    )
    uncertainty_checks = {
        name: bool(
            uncertainty["lower_bounds"][name] >= float(minimum)
        )
        for name, minimum in dict(bootstrap["lower_bound_minimum"]).items()
    }
    return {
        "schema_version": 1,
        "component": "door",
        "threshold_source": threshold_source_block(config),
        "threshold_provenance": str(section["threshold_provenance"]).strip(),
        "additive_only": True,
        "full_protocol": bool(full_protocol),
        "metrics": metrics,
        "uncertainty": uncertainty,
        "checks": {**checks, **uncertainty_checks},
        "passed": bool(all(checks.values()) and all(uncertainty_checks.values())),
    }


def _smoke_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    paired = paired_effect_rows(records)
    by_rule = {}
    for rule in TRUE_RULES:
        rows = [row for row in paired if row["true_rule"] == rule]
        by_rule[rule] = {
            "queries": len(rows),
            "matching_history_two_target_accuracy": _mean(
                float(row["same_history_true_target_closer"])
                for row in rows
            ),
            "matching_vs_opposite_history_win_rate": _mean(
                float(row["same_vs_other_advantage"] > 0.0)
                for row in rows
            ),
            "mean_matching_vs_opposite_history_advantage": _mean(
                row["same_vs_other_advantage"] for row in rows
            ),
        }
    return {
        "formal_protocol_eligible": False,
        "decision": {
            "passed": None,
            "reason": (
                "A reduced smoke run is descriptive only. The formal gate "
                "requires all 300 frozen queries."
            ),
        },
        "by_true_rule": by_rule,
        "records": len(records),
    }


def evaluate_door_icl_model(
    *,
    adapter: DoorICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
    repo_root: Path | None = None,
    eval_seeds: list[int] | tuple[int, ...] | None = None,
    limit_per_seed: int | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    """Evaluate one frozen model on offline door-rule Validation arrays."""

    root = (repo_root or repository_root()).resolve()
    release = load_door_icl_release(release_config)
    dataset = DoorICLEvalDataset(
        release=release,
        repo_root=root,
        eval_seeds=eval_seeds,
        limit_per_seed=limit_per_seed,
    )
    scored = score_validation_assets(
        adapter,
        dataset.raw_assets,
        batch_size=int(batch_size),
        return_latents=True,
    )
    latents = scored["latents"]
    gate_completion = door_gate_completion_metrics(
        predicted=latents["predicted"],
        encoded_targets=latents["encoded_targets"],
        query_ids=list(latents["query_ids"]),
        static_query_ids=list(latents["static_query_ids"]),
        config=load_test_gate_completion_config(),
        full_protocol=dataset.is_full_protocol,
    )
    if dataset.is_full_protocol:
        summary = summarize_validation_records(
            scored["records"],
            eval_seeds=release["evaluation"]["eval_seeds"],
            unique_queries_per_seed=int(
                release["evaluation"]["queries_per_eval_seed"]
            ),
            gates=release["scoring"]["gates"],
        )
        formal_pass = bool(summary["decision"]["passed"])
    else:
        summary = _smoke_summary(scored["records"])
        formal_pass = None
    release_path = Path(release["_config_path"])
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_history3_door_rule_icl_v1",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "catalog_sha256": release["evaluation"]["catalog_sha256"],
            "content_manifest_sha256": release["evaluation"][
                "content_manifest_sha256"
            ],
            "normalizer_sha256": release["evaluation"][
                "normalizer_sha256"
            ],
            "sealed_test_included": False,
        },
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": (
                None if training_seed is None else int(training_seed)
            ),
            "adapter": adapter.metadata,
        },
        "data": dataset.describe(),
        "full_protocol": dataset.is_full_protocol,
        "formal_checkpoint_passed": formal_pass,
        "score_audit": scored["score_audit"],
        "summary": summary,
        "gate_completion": gate_completion,
    }
    if include_records:
        payload["records"] = scored["records"]
    return payload


def _load_and_verify_result(
    path: Path,
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != "tworoom_history3_door_rule_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Door ICL result: {path}")
    identity = payload.get("release", {})
    expected_config_hash = file_sha256(Path(release["_config_path"]))
    expected = {
        "release_id": release["release_id"],
        "release_config_sha256": expected_config_hash,
        "catalog_sha256": release["evaluation"]["catalog_sha256"],
        "content_manifest_sha256": release["evaluation"][
            "content_manifest_sha256"
        ],
        "normalizer_sha256": release["evaluation"]["normalizer_sha256"],
        "sealed_test_included": False,
    }
    if identity != expected:
        raise RuntimeError(
            f"Result release identity mismatch for {path}: {identity}"
        )
    if payload.get("full_protocol") is not True:
        raise ValueError(
            f"Formal scoring requires all frozen queries: {path}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"Result must retain records for independent rescoring: {path}"
        )
    recomputed = summarize_validation_records(
        records,
        eval_seeds=release["evaluation"]["eval_seeds"],
        unique_queries_per_seed=int(
            release["evaluation"]["queries_per_eval_seed"]
        ),
        gates=release["scoring"]["gates"],
    )
    if canonical_sha256(recomputed) != canonical_sha256(payload["summary"]):
        raise RuntimeError(f"Stored Door ICL summary changed: {path}")
    if bool(payload["formal_checkpoint_passed"]) != bool(
        recomputed["decision"]["passed"]
    ):
        raise RuntimeError(f"Stored Door ICL decision changed: {path}")
    return payload


def _reader_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    rules = payload["summary"]["by_true_rule"]
    target_accuracy = _mean(
        rules[rule]["overall"]["same_history_two_target_accuracy"]
        for rule in TRUE_RULES
    )
    history_win = _mean(
        rules[rule]["overall"][
            "matching_vs_opposite_history_win_rate"
        ]
        for rule in TRUE_RULES
    )
    return {
        "correct_target_choice_rate": target_accuracy,
        "correct_history_win_rate": history_win,
        "checkpoint_passed": bool(
            payload["summary"]["decision"]["passed"]
        ),
        "by_true_rule": {
            rule: {
                "correct_target_choice_rate": rules[rule]["overall"][
                    "same_history_two_target_accuracy"
                ],
                "correct_history_win_rate": rules[rule]["overall"][
                    "matching_vs_opposite_history_win_rate"
                ],
            }
            for rule in TRUE_RULES
        },
    }


def score_door_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Independently rescore one checkpoint or aggregate three model seeds."""

    release = load_door_icl_release(release_config)
    paths = [Path(value).expanduser().resolve() for value in result_paths]
    if not paths:
        raise ValueError("At least one Door ICL result is required")
    if len(paths) not in {1, 3}:
        raise ValueError(
            "Use one result for a descriptive checkpoint score or three "
            "training seeds for a method-level claim"
        )
    results = [
        _load_and_verify_result(path, release=release) for path in paths
    ]
    checkpoint_hashes = [
        str(result["model"]["adapter"].get("checkpoint_sha256", ""))
        for result in results
    ]
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in checkpoint_hashes
    ):
        raise ValueError(
            "Every result must bind a lowercase 64-character checkpoint "
            "SHA-256"
        )
    if len(checkpoint_hashes) != len(set(checkpoint_hashes)):
        raise ValueError("Each result must use a different checkpoint")
    seeds = [result["model"]["training_seed"] for result in results]
    if len(paths) == 3:
        recipes = {
            str(result["model"]["training_recipe"])
            for result in results
        }
        adapter_ids = {
            str(result["model"]["adapter"].get("adapter_id"))
            for result in results
        }
        if len(recipes) != 1 or len(adapter_ids) != 1:
            raise ValueError(
                "A three-seed method score cannot mix training recipes or "
                "model adapters"
            )
        expected = {
            int(value)
            for value in release["training"]["paired_training_seeds"]
        }
        if set(seeds) != expected:
            raise ValueError(
                f"Method scoring requires training seeds {sorted(expected)}"
            )
    metrics = [_reader_metrics(result) for result in results]
    method_passed = bool(
        len(results) == 3
        and all(value["checkpoint_passed"] for value in metrics)
    )
    grouped_names: dict[str, list[int | None]] = defaultdict(list)
    for result in results:
        grouped_names[str(result["model"]["name"])].append(
            result["model"]["training_seed"]
        )
    return {
        "schema_version": 1,
        "benchmark": "tworoom_history3_door_rule_icl_v1",
        "submission_kind": (
            "single_checkpoint_score"
            if len(results) == 1
            else "three_seed_method_score"
        ),
        "status": "completed",
        "release_id": release["release_id"],
        "method_name": str(method_name),
        "models": dict(grouped_names),
        "result_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "checkpoint_sha256": checkpoint_hash,
            }
            for path, checkpoint_hash in zip(paths, checkpoint_hashes)
        ],
        "checkpoints": metrics,
        "mean_correct_target_choice_rate": _mean(
            value["correct_target_choice_rate"] for value in metrics
        ),
        "mean_correct_history_win_rate": _mean(
            value["correct_history_win_rate"] for value in metrics
        ),
        "passed_checkpoints": sum(
            value["checkpoint_passed"] for value in metrics
        ),
        "required_checkpoints_for_method_claim": 3,
        "formal_claim_level": (
            "three_seed_method_result"
            if len(results) == 3
            else "descriptive_checkpoint_result"
        ),
        "method_passed": (
            method_passed if len(results) == 3 else None
        ),
        "sealed_test_included": False,
    }


__all__ = [
    "evaluate_door_icl_model",
    "score_door_icl_results",
]
