from __future__ import annotations

import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from contextworld.benchmarks.adapters import CubeGraspRuleICLModelAdapter
from contextworld.benchmarks.cube_grasp_rule_icl_score import (
    _validate_cube_adapter_protocol,
    cube_grasp_rule_prediction_gate,
    cube_grasp_rule_prediction_metrics,
)
from contextworld.benchmarks.cube_grasp_rule_v4r1_icl_data import (
    DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    EXPECTED_EXTERNAL_EVALUATION_POLICY,
    CubeGraspRuleV4R1ICLEvalDataset,
    file_sha256,
    load_cube_grasp_rule_v4r1_icl_release,
)
from contextworld.paths import repository_root


CUBE_GRASP_RULE_V4R1_BENCHMARK_ID = (
    "cube_history3_gripper_carry_icl_v4r1"
)
EXTERNAL_RESULT_CLAIM_BOUNDARY = {
    "external_result": True,
    "external_evaluation_allowed": True,
    "formal_reference_mutation": False,
    "formal_scoreboard_eligible": False,
    "reference_rerun": False,
}


def validate_cube_grasp_rule_v4r1_external_evaluation_policy(
    release: Mapping[str, Any],
) -> dict[str, bool]:
    """Fail closed before any Public data or checkpoint is opened."""

    claim = release.get("claim_boundary")
    policy = claim.get("external_evaluation") if isinstance(claim, Mapping) else None
    if not isinstance(policy, Mapping) or dict(policy) != (
        EXPECTED_EXTERNAL_EVALUATION_POLICY
    ):
        raise RuntimeError("Cube v4r1 external-evaluation policy is not authorized")
    return dict(EXPECTED_EXTERNAL_EVALUATION_POLICY)


def validate_cube_grasp_rule_v4r1_external_checkpoint_identity(
    release: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_path: str | Path | None = None,
    model_state_sha256: str | None = None,
) -> None:
    """Prevent the frozen formal reference from being rerun as external."""

    checkpoint_sha256 = str(checkpoint_sha256)
    if len(checkpoint_sha256) != 64:
        raise ValueError("Cube v4r1 external checkpoint requires a SHA-256 identity")
    canonical = release["reference_method"]["checkpoints"]
    canonical_checkpoint_hashes = {str(row["sha256"]) for row in canonical}
    canonical_state_hashes = {str(row["model_state_sha256"]) for row in canonical}
    canonical_paths = {str(row["path"]) for row in canonical}
    observed_path = str(checkpoint_path) if checkpoint_path is not None else None
    if (
        checkpoint_sha256 in canonical_checkpoint_hashes
        or model_state_sha256 in canonical_state_hashes
        or observed_path in canonical_paths
    ):
        raise RuntimeError(
            "Frozen Cube formal-reference checkpoints cannot be rerun through "
            "the external evaluation API"
        )


def evaluate_cube_grasp_rule_v4r1_icl_model(
    *,
    adapter: CubeGraspRuleICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
    layout: Literal["auto", "source", "bundle"] = "auto",
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_cube_grasp_rule_v4r1_icl_release(release_config)
    validate_cube_grasp_rule_v4r1_external_evaluation_policy(release)
    if not str(model_name).strip() or not str(training_recipe).strip():
        raise ValueError("Cube v4r1 external model and recipe names must be non-empty")
    metadata = adapter.metadata
    if not isinstance(metadata, Mapping):
        raise ValueError("Cube v4r1 adapter metadata must be a mapping")
    validate_cube_grasp_rule_v4r1_external_checkpoint_identity(
        release,
        checkpoint_sha256=str(metadata.get("checkpoint_sha256", "")),
        checkpoint_path=metadata.get("checkpoint"),
    )
    _validate_cube_adapter_protocol(adapter)
    before = adapter.frozen_state_hash()
    validate_cube_grasp_rule_v4r1_external_checkpoint_identity(
        release,
        checkpoint_sha256=str(metadata.get("checkpoint_sha256", "")),
        checkpoint_path=metadata.get("checkpoint"),
        model_state_sha256=before,
    )
    dataset = CubeGraspRuleV4R1ICLEvalDataset(
        release=release, repo_root=root, layout=layout
    )
    arrays = dataset.arrays
    histories = np.concatenate(
        [arrays.cannot_hold_pixels[:, :3], arrays.can_hold_pixels[:, :3]]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    predicted = adapter.rollout_latents(histories, actions, batch_size=batch_size)
    count = arrays.pair_count
    if (
        predicted.ndim != 3
        or predicted.shape[0] != 2 * count
        or predicted.shape[1] != 1
        or not np.isfinite(predicted).all()
    ):
        raise RuntimeError(
            "Cube v4r1 adapter must return finite latents with shape "
            "(2 * pair_count, 1, latent_dim)"
        )
    targets = np.concatenate(
        [arrays.cannot_hold_pixels[:, 3], arrays.can_hold_pixels[:, 3]]
    )
    encoded = adapter.encode_pixels(targets, batch_size=batch_size)
    if (
        encoded.shape != (2 * count, predicted.shape[2])
        or not np.isfinite(encoded).all()
    ):
        raise RuntimeError("Cube v4r1 encoded targets do not match predictions")
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Cube v4r1 evaluation")
    metrics, records = cube_grasp_rule_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_cannot_hold=predicted[:count, 0],
        predicted_can_hold=predicted[count:, 0],
        target_cannot_hold=encoded[:count],
        target_can_hold=encoded[count:],
    )
    release_path = Path(release["_config_path"])
    result = {
        "schema_version": 1,
        "benchmark": CUBE_GRASP_RULE_V4R1_BENCHMARK_ID,
        "submission_kind": "external_single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "portable_provenance_sha256": release["data"]["artifacts"]
            ["portable_provenance"]["sha256"],
        },
        "model": {
            "name": str(model_name),
            "training_recipe": str(training_recipe),
            "training_seed": training_seed,
            "adapter": dict(metadata),
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "data": dataset.describe(),
        "metrics": metrics,
        "gate": cube_grasp_rule_prediction_gate(metrics, release=release),
        "claim_boundary": dict(EXTERNAL_RESULT_CLAIM_BOUNDARY),
    }
    if include_records:
        result["records"] = records
    return result


def score_cube_grasp_rule_v4r1_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_CUBE_GRASP_RULE_V4R1_RELEASE_CONFIG,
) -> dict[str, Any]:
    release = load_cube_grasp_rule_v4r1_icl_release(release_config)
    validate_cube_grasp_rule_v4r1_external_evaluation_policy(release)
    if not str(method_name).strip():
        raise ValueError("Cube v4r1 external method name must be non-empty")
    release_path = Path(release["_config_path"])
    expected_release = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(release_path),
        "portable_provenance_sha256": release["data"]["artifacts"]
        ["portable_provenance"]["sha256"],
    }
    results: list[dict[str, Any]] = []
    for value in result_paths:
        path = Path(value).expanduser().resolve()
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(row, dict)
            or set(row)
            not in (
                {
                    "schema_version",
                    "benchmark",
                    "submission_kind",
                    "status",
                    "release",
                    "model",
                    "data",
                    "metrics",
                    "gate",
                    "claim_boundary",
                },
                {
                    "schema_version",
                    "benchmark",
                    "submission_kind",
                    "status",
                    "release",
                    "model",
                    "data",
                    "metrics",
                    "gate",
                    "claim_boundary",
                    "records",
                },
            )
            or row.get("schema_version") != 1
            or row.get("benchmark") != CUBE_GRASP_RULE_V4R1_BENCHMARK_ID
            or row.get("submission_kind") != "external_single_checkpoint"
            or row.get("status") != "completed"
            or row.get("release") != expected_release
            or row.get("claim_boundary") != EXTERNAL_RESULT_CLAIM_BOUNDARY
        ):
            raise ValueError(f"Unsupported Cube v4r1 result: {path}")
        seed = row.get("model", {}).get("training_seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Cube v4r1 result has no integer seed: {path}")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or "latent_response" not in metrics:
            raise ValueError(f"Cube v4r1 result lacks current metrics: {path}")
        recomputed = cube_grasp_rule_prediction_gate(metrics, release=release)
        if row.get("gate") != recomputed:
            raise RuntimeError(f"Cube v4r1 result gate drifted: {path}")
        results.append(row)
    required = int(release["scoring"]["method_level"]["training_seeds_required"])
    seeds = [int(row["model"]["training_seed"]) for row in results]
    if len(results) != required or len(set(seeds)) != required:
        raise ValueError(f"Cube v4r1 method scoring requires {required} seeds")
    model_names = {str(row["model"].get("name", "")) for row in results}
    recipes = {str(row["model"].get("training_recipe", "")) for row in results}
    if len(model_names) != 1 or "" in model_names or len(recipes) != 1 or "" in recipes:
        raise ValueError("Cube v4r1 method results must share one model and recipe")
    names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_rule_correct_future_rate",
        "other_minus_correct_mse_margin_mean",
        "joint_icl_pair_success_rate",
    )
    return {
        "schema_version": 1,
        "benchmark": CUBE_GRASP_RULE_V4R1_BENCHMARK_ID,
        "submission_kind": "external_three_seed_method",
        "status": "completed",
        "method_name": str(method_name),
        "training_seeds": sorted(seeds),
        "checkpoint_results": results,
        "aggregate": {
            name: {
                "mean": float(
                    statistics.mean(row["metrics"][name] for row in results)
                ),
                "minimum": float(min(row["metrics"][name] for row in results)),
                "maximum": float(max(row["metrics"][name] for row in results)),
            }
            for name in names
        },
        "passed": all(row["gate"]["passed"] for row in results),
        "claim_boundary": dict(EXTERNAL_RESULT_CLAIM_BOUNDARY),
    }


__all__ = [
    "CUBE_GRASP_RULE_V4R1_BENCHMARK_ID",
    "EXTERNAL_RESULT_CLAIM_BOUNDARY",
    "evaluate_cube_grasp_rule_v4r1_icl_model",
    "score_cube_grasp_rule_v4r1_icl_results",
    "validate_cube_grasp_rule_v4r1_external_checkpoint_identity",
    "validate_cube_grasp_rule_v4r1_external_evaluation_policy",
]
