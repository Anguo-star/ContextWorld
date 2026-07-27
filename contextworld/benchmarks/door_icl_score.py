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
from contextworld.evaluation.hidden_passage_validation import (
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
