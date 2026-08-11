from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from contextworld.benchmarks.action_delay_icl_data import (
    DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    ActionDelayICLEvalDataset,
    load_action_delay_icl_release,
)
from contextworld.benchmarks.adapters import ActionDelayICLModelAdapter
from contextworld.evaluation.action_delay_h7_core import (
    summarize_action_delay_h1_physical,
)
from contextworld.evaluation.action_delay_h7_score import (
    score_h7_validation_assets,
    summarize_h7_validation_records,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import repository_root


def _gate(
    core: dict[str, Any],
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    thresholds = release["scoring"]["primary_gate"]
    checks = {
        "physical_group_macro_accuracy": (
            core["physical_group_macro_accuracy"]
            >= float(
                thresholds[
                    "physical_group_macro_accuracy_minimum"
                ]
            )
        ),
        "minimum_physical_group_accuracy": (
            core["minimum_physical_group_accuracy"]
            >= float(
                thresholds["minimum_physical_group_accuracy"]
            )
        ),
        "bootstrap_lower_bound": (
            core["paired_query_bootstrap_95_percent_interval"]["lower"]
            >= float(
                thresholds[
                    "paired_query_bootstrap_95_percent_lower_bound_minimum"
                ]
            )
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _summaries(
    records: list[dict[str, Any]],
    *,
    release: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary = summarize_h7_validation_records(records)
    uncertainty = release["scoring"]["uncertainty"]
    core = summarize_action_delay_h1_physical(
        summary["by_horizon"]["1"]["query_metrics"],
        bootstrap_resamples=int(uncertainty["resamples"]),
        bootstrap_seed=int(uncertainty["random_seed"]),
    )
    return summary, core, _gate(core, release=release)


def evaluate_action_delay_icl_model(
    *,
    adapter: ActionDelayICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 128,
    include_records: bool = True,
) -> dict[str, Any]:
    """Run the complete frozen 300-query Action Delay Public Test."""

    root = (repo_root or repository_root()).resolve()
    release = load_action_delay_icl_release(release_config)
    dataset = ActionDelayICLEvalDataset(
        release=release,
        repo_root=root,
    )
    if not dataset.is_full_protocol:
        raise RuntimeError(
            "Formal Action Delay scoring requires all 300 frozen queries"
        )
    before = adapter.frozen_state_hash()
    scored = score_h7_validation_assets(
        adapter,
        dataset.raw_assets,
        batch_size=int(batch_size),
    )
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Action Delay scoring")
    summary, core, gate = _summaries(
        scored["records"],
        release=release,
    )
    release_path = Path(release["_config_path"])
    payload = {
        "schema_version": 1,
        "benchmark": "tworoom_history7_action_delay_icl_v1",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "catalog_sha256": release["evaluation"][
                "catalog_sha256"
            ],
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
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "score_audit": scored["score_audit"],
        "core_h1": core,
        "gate": gate,
    }
    if include_records:
        payload["records"] = scored["records"]
    return payload


def _load_and_rescore(
    path: Path,
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != "tworoom_history7_action_delay_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Action Delay result: {path}")
    expected_identity = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(
            Path(release["_config_path"])
        ),
        "catalog_sha256": release["evaluation"]["catalog_sha256"],
        "content_manifest_sha256": release["evaluation"][
            "content_manifest_sha256"
        ],
        "normalizer_sha256": release["evaluation"][
            "normalizer_sha256"
        ],
        "sealed_test_included": False,
    }
    if payload.get("release") != expected_identity:
        raise RuntimeError(f"Release identity mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(
            f"Independent rescoring requires retained records: {path}"
        )
    _, core, gate = _summaries(records, release=release)
    if core != payload.get("core_h1") or gate != payload.get("gate"):
        raise RuntimeError(f"Stored Action Delay score changed: {path}")
    return payload


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def score_action_delay_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Independently rescore one checkpoint or one three-seed method."""

    release = load_action_delay_icl_release(release_config)
    paths = [Path(value).expanduser().resolve() for value in result_paths]
    if len(paths) not in {1, 3}:
        raise ValueError(
            "Provide one result for a descriptive checkpoint or three "
            "results for a method-level claim"
        )
    results = [
        _load_and_rescore(path, release=release) for path in paths
    ]
    checkpoint_hashes = [
        str(result["model"]["adapter"].get("checkpoint_sha256", ""))
        for result in results
    ]
    if (
        any(len(value) != 64 for value in checkpoint_hashes)
        or len(set(checkpoint_hashes)) != len(checkpoint_hashes)
    ):
        raise ValueError(
            "Every result must bind a distinct checkpoint SHA-256"
        )
    seeds = [result["model"]["training_seed"] for result in results]
    if len(paths) == 3:
        if any(seed is None for seed in seeds) or len(set(seeds)) != 3:
            raise ValueError(
                "A method score requires three distinct training seeds"
            )
        recipes = {
            str(result["model"]["training_recipe"])
            for result in results
        }
        adapters = {
            str(result["model"]["adapter"].get("adapter_id"))
            for result in results
        }
        if len(recipes) != 1 or len(adapters) != 1:
            raise ValueError(
                "A method score cannot mix recipes or adapter families"
            )
    per_checkpoint = [
        {
            "path": str(path),
            "checkpoint_sha256": result["model"]["adapter"][
                "checkpoint_sha256"
            ],
            "training_seed": result["model"]["training_seed"],
            "physical_group_macro_accuracy": result["core_h1"][
                "physical_group_macro_accuracy"
            ],
            "minimum_physical_group_accuracy": result["core_h1"][
                "minimum_physical_group_accuracy"
            ],
            "bootstrap_lower_bound": result["core_h1"][
                "paired_query_bootstrap_95_percent_interval"
            ]["lower"],
            "passed": bool(result["gate"]["passed"]),
        }
        for path, result in zip(paths, results, strict=True)
    ]
    formal_method = len(paths) == 3
    return {
        "schema_version": 1,
        "benchmark": "tworoom_history7_action_delay_icl_v1",
        "submission_kind": (
            "three_seed_method"
            if formal_method
            else "descriptive_checkpoint"
        ),
        "status": "completed",
        "method_name": str(method_name),
        "release_id": release["release_id"],
        "checkpoints": per_checkpoint,
        "aggregate": {
            metric: _stats(row[metric] for row in per_checkpoint)
            for metric in (
                "physical_group_macro_accuracy",
                "minimum_physical_group_accuracy",
                "bootstrap_lower_bound",
            )
        },
        "decision": {
            "passed": (
                all(row["passed"] for row in per_checkpoint)
                if formal_method
                else None
            ),
            "passed_checkpoints": sum(
                row["passed"] for row in per_checkpoint
            ),
            "required_checkpoints": 3 if formal_method else None,
            "claim": (
                "method_level_action_delay_icl"
                if formal_method
                else "descriptive_checkpoint_only"
            ),
        },
    }


__all__ = [
    "evaluate_action_delay_icl_model",
    "score_action_delay_icl_results",
]
