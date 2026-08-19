from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.action_strength_icl_data import (
    DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    ActionStrengthICLEvalDataset,
    file_sha256,
    load_action_strength_icl_release,
)
from contextworld.benchmarks.adapters import (
    ActionStrengthICLModelAdapter,
    validate_adapter_protocol,
)
from contextworld.benchmarks.paired_latent_response import (
    paired_latent_response_gate_checks,
    paired_latent_response_metrics,
    paired_latent_response_summaries_close,
    summarize_paired_latent_response_records,
)
from contextworld.paths import repository_root, resolve_contextworld_path


def _prediction_contract_payload(release: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen inputs that define Action Strength prediction.

    Reference results intentionally do not participate.  This avoids a
    circular identity where publishing a newly scored checkpoint changes the
    release-file hash and invalidates the checkpoint result itself.
    """

    evaluation = release["evaluation"]
    runtime = release["runtime"]
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "scope": release["scope"],
        "runtime": {
            "supported_adapters": runtime["supported_adapters"],
            "extension_contract": runtime["extension_contract"],
            "stable_worldmodel_expected_ref": runtime[
                "stable_worldmodel"
            ]["expected_ref"],
        },
        "training": {
            "manifest_sha256": release["training"]["manifest_sha256"],
            "train_pairs": release["training"]["train_pairs"],
            "validation_pairs": release["training"]["validation_pairs"],
        },
        "evaluation": {
            name: evaluation[name]
            for name in (
                "track",
                "pair_count",
                "condition_count",
                "lance_table",
                "minimum_true_future_block_gap_px",
                "source_episode_overlap_with_training_or_development",
                "query_image_hash_overlap_with_training_or_development",
                "online_environment_calls_for_prediction_track",
                "manifest_sha256",
                "action_normalization",
            )
        },
        "scoring": release["scoring"]["hidden_future_prediction"],
    }


def _prediction_contract_sha256(release: dict[str, Any]) -> str:
    encoded = json.dumps(
        _prediction_contract_payload(release),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mse(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.square(left - right).mean(axis=-1)


def _prediction_metrics(
    *,
    pair_ids: tuple[str, ...],
    predicted_low: np.ndarray,
    predicted_high: np.ndarray,
    target_low: np.ndarray,
    target_high: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    low_to_low = _mse(predicted_low, target_low)
    low_to_high = _mse(predicted_low, target_high)
    high_to_low = _mse(predicted_high, target_low)
    high_to_high = _mse(predicted_high, target_high)

    low_target = low_to_low < low_to_high
    high_target = high_to_high < high_to_low
    low_history = low_to_low < high_to_low
    high_history = high_to_high < low_to_high
    switch = np.sum(
        (predicted_high - predicted_low) * (target_high - target_low),
        axis=-1,
    ) > 0
    target_decisions = np.concatenate([low_target, high_target])
    history_decisions = np.concatenate([low_history, high_history])
    correct_losses = np.concatenate([low_to_low, high_to_high])
    incorrect_losses = np.concatenate([low_to_high, high_to_low])
    summary = {
        "pair_count": len(pair_ids),
        "decision_count": 2 * len(pair_ids),
        "correct_future_rate": float(target_decisions.mean()),
        "correct_history_rate": float(history_decisions.mean()),
        "rule_switch_rate": float(switch.mean()),
        "low_strength_correct_future_rate": float(low_target.mean()),
        "high_strength_correct_future_rate": float(high_target.mean()),
        "worst_strength_correct_future_rate": float(
            min(low_target.mean(), high_target.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(incorrect_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (incorrect_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    latent_response, latent_response_records = (
        paired_latent_response_metrics(
            pair_ids=pair_ids,
            predicted_first=predicted_low,
            predicted_second=predicted_high,
            target_first=target_low,
            target_second=target_high,
        )
    )
    summary["latent_response"] = latent_response
    calibrated_response = np.asarray(
        [
            row["calibrated_response_success"]
            for row in latent_response_records
        ],
        dtype=bool,
    )
    joint_icl_pair_success = (
        low_target
        & high_target
        & low_history
        & high_history
        & calibrated_response
    )
    summary["joint_icl_pair_success_rate"] = float(
        joint_icl_pair_success.mean()
    )
    records = [
        {
            "pair_id": pair_id,
            "low_strength": {
                "correct_future_mse": float(low_to_low[index]),
                "other_future_mse": float(low_to_high[index]),
                "correct_future": bool(low_target[index]),
                "correct_history": bool(low_history[index]),
            },
            "high_strength": {
                "correct_future_mse": float(high_to_high[index]),
                "other_future_mse": float(high_to_low[index]),
                "correct_future": bool(high_target[index]),
                "correct_history": bool(high_history[index]),
            },
            "rule_switch_correct": bool(switch[index]),
            "joint_icl_pair_success": bool(
                joint_icl_pair_success[index]
            ),
            "latent_response": {
                name: value
                for name, value in latent_response_records[index].items()
                if name != "pair_id"
            },
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    return summary, records


def _prediction_gate(
    metrics: dict[str, Any],
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    thresholds = release["scoring"]["hidden_future_prediction"]["gates"]
    checks = {
        "correct_future_rate": (
            metrics["correct_future_rate"]
            >= float(thresholds["correct_future_rate_minimum"])
        ),
        "correct_history_rate": (
            metrics["correct_history_rate"]
            >= float(thresholds["correct_history_rate_minimum"])
        ),
        "rule_switch_rate": (
            metrics["rule_switch_rate"]
            >= float(thresholds["rule_switch_rate_minimum"])
        ),
        "worst_strength_correct_future_rate": (
            metrics["worst_strength_correct_future_rate"]
            >= float(
                thresholds[
                    "worst_strength_correct_future_rate_minimum"
                ]
            )
        ),
    }
    checks.update(
        paired_latent_response_gate_checks(
            metrics, thresholds=thresholds
        )
    )
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_action_strength_icl_model(
    *,
    adapter: ActionStrengthICLModelAdapter,
    model_name: str,
    training_recipe: str,
    training_seed: int | None,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    repo_root: Path | None = None,
    batch_size: int = 64,
    include_records: bool = True,
) -> dict[str, Any]:
    """Score one frozen checkpoint on all 256 independent query pairs."""

    root = (repo_root or repository_root()).resolve()
    release = load_action_strength_icl_release(release_config)
    dataset = ActionStrengthICLEvalDataset(
        release=release,
        repo_root=root,
    )
    arrays = dataset.arrays
    if not dataset.is_full_protocol:
        raise RuntimeError(
            "Formal Action Strength scoring requires all 256 pairs"
        )
    validate_adapter_protocol(
        adapter,
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        minimum_future_action_blocks=1,
        task_name="Action Strength v1",
    )

    histories = np.concatenate(
        [arrays.low_pixels[:, :3], arrays.high_pixels[:, :3]],
        axis=0,
    )
    context_and_query = np.concatenate(
        [
            arrays.raw_action_blocks[:, :3],
            arrays.raw_action_blocks[:, :3],
        ],
        axis=0,
    )
    before = adapter.frozen_state_hash()
    predicted = adapter.rollout_latents(
        histories,
        context_and_query,
        batch_size=int(batch_size),
    )
    if predicted.ndim != 3 or predicted.shape[1] != 1:
        raise RuntimeError(
            "Action Strength adapter must return one predicted future"
        )
    predicted = predicted[:, 0]
    futures = np.concatenate(
        [arrays.low_pixels[:, 3], arrays.high_pixels[:, 3]],
        axis=0,
    )
    encoded = adapter.encode_pixels(futures, batch_size=int(batch_size))
    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during Action Strength scoring")

    count = arrays.pair_count
    metrics, records = _prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_low=predicted[:count],
        predicted_high=predicted[count:],
        target_low=encoded[:count],
        target_high=encoded[count:],
    )
    gate = _prediction_gate(metrics, release=release)
    release_path = Path(release["_config_path"])
    payload = {
        "schema_version": 1,
        "benchmark": "pusht_history3_action_strength_icl_v1",
        "submission_kind": "single_checkpoint",
        "status": "completed",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256_at_evaluation": file_sha256(
                release_path
            ),
            "prediction_contract_sha256": (
                _prediction_contract_sha256(release)
            ),
            "training_manifest_sha256": release["training"][
                "manifest_sha256"
            ],
            "confirmation_manifest_sha256": release["evaluation"][
                "manifest_sha256"
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
        "metrics": metrics,
        "gate": gate,
    }
    if include_records:
        payload["records"] = records
    return payload


def _rescore_prediction_result(
    path: Path,
    *,
    release: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark")
        != "pusht_history3_action_strength_icl_v1"
        or payload.get("submission_kind") != "single_checkpoint"
        or payload.get("status") != "completed"
    ):
        raise ValueError(f"Unsupported Action Strength result: {path}")
    expected_release_identity = {
        "release_id": release["release_id"],
        "prediction_contract_sha256": _prediction_contract_sha256(
            release
        ),
        "training_manifest_sha256": release["training"][
            "manifest_sha256"
        ],
        "confirmation_manifest_sha256": release["evaluation"][
            "manifest_sha256"
        ],
        "sealed_test_included": False,
    }
    observed_release = payload.get("release")
    receipt = (
        observed_release.get("release_config_sha256_at_evaluation")
        if isinstance(observed_release, dict)
        else None
    )
    identity_matches = bool(
        isinstance(observed_release, dict)
        and all(
            observed_release.get(name) == value
            for name, value in expected_release_identity.items()
        )
        and isinstance(receipt, str)
        and len(receipt) == 64
    )
    if not identity_matches:
        raise RuntimeError(f"Release identity mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 256:
        raise ValueError(
            "Independent rescoring requires all 256 pair records"
        )
    low_target = np.asarray(
        [row["low_strength"]["correct_future"] for row in records],
        dtype=bool,
    )
    high_target = np.asarray(
        [row["high_strength"]["correct_future"] for row in records],
        dtype=bool,
    )
    low_history = np.asarray(
        [row["low_strength"]["correct_history"] for row in records],
        dtype=bool,
    )
    high_history = np.asarray(
        [row["high_strength"]["correct_history"] for row in records],
        dtype=bool,
    )
    switch = np.asarray(
        [row["rule_switch_correct"] for row in records],
        dtype=bool,
    )
    # The evaluator produces these per-example MSE values from float32
    # latent arrays.  Preserve that dtype when JSON records are rescored so
    # the public command reproduces the evaluator's aggregate exactly.
    correct_losses = np.asarray(
        [
            row["low_strength"]["correct_future_mse"]
            for row in records
        ]
        + [
            row["high_strength"]["correct_future_mse"]
            for row in records
        ],
        dtype=np.float32,
    )
    other_losses = np.asarray(
        [
            row["low_strength"]["other_future_mse"]
            for row in records
        ]
        + [
            row["high_strength"]["other_future_mse"]
            for row in records
        ],
        dtype=np.float32,
    )
    metrics = {
        "pair_count": len(records),
        "decision_count": 2 * len(records),
        "correct_future_rate": float(
            np.concatenate([low_target, high_target]).mean()
        ),
        "correct_history_rate": float(
            np.concatenate([low_history, high_history]).mean()
        ),
        "rule_switch_rate": float(switch.mean()),
        "low_strength_correct_future_rate": float(low_target.mean()),
        "high_strength_correct_future_rate": float(high_target.mean()),
        "worst_strength_correct_future_rate": float(
            min(low_target.mean(), high_target.mean())
        ),
        "correct_future_mse_mean": float(correct_losses.mean()),
        "other_future_mse_mean": float(other_losses.mean()),
        "other_minus_correct_mse_margin_mean": float(
            (other_losses - correct_losses).mean()
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    response_rows_present = [
        isinstance(row.get("latent_response"), dict) for row in records
    ]
    if not any(response_rows_present):
        raise ValueError(
            "Legacy Action Strength result lacks mandatory latent response "
            f"metrics and must be rescored from its checkpoint: {path}"
        )
    if any(response_rows_present):
        if not all(response_rows_present):
            raise RuntimeError(
                f"Incomplete Action Strength latent response records: {path}"
            )
        metrics["latent_response"] = (
            summarize_paired_latent_response_records(
                [
                    {
                        "pair_id": row["pair_id"],
                        **row["latent_response"],
                    }
                    for row in records
                ]
            )
        )
        calibrated_response = np.asarray(
            [
                row["latent_response"][
                    "calibrated_response_success"
                ]
                for row in records
            ],
            dtype=bool,
        )
        joint = (
            low_target
            & high_target
            & low_history
            & high_history
            & calibrated_response
        )
        if not all(
            row.get("joint_icl_pair_success") == bool(joint[index])
            for index, row in enumerate(records)
        ):
            raise RuntimeError(
                f"Invalid Action Strength joint ICL records: {path}"
            )
        metrics["joint_icl_pair_success_rate"] = float(joint.mean())
    gate = _prediction_gate(metrics, release=release)
    stored_metrics = payload.get("metrics")
    scalar_metric_names = set(metrics) - {"latent_response"}
    metrics_match = (
        isinstance(stored_metrics, dict)
        and set(stored_metrics) == set(metrics)
        and all(
            (
                value == stored_metrics[name]
                if isinstance(value, int)
                else np.isclose(
                    value,
                    stored_metrics[name],
                    rtol=1e-7,
                    atol=1e-9,
                )
            )
            for name, value in metrics.items()
            if name in scalar_metric_names
        )
        and (
            "latent_response" not in metrics
            or paired_latent_response_summaries_close(
                stored_metrics["latent_response"],
                metrics["latent_response"],
            )
        )
    )
    if not metrics_match or gate != payload.get("gate"):
        raise RuntimeError(f"Stored Action Strength score changed: {path}")
    return payload


def _stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    return {
        "mean": float(statistics.fmean(rows)),
        "sample_std": (
            float(statistics.stdev(rows)) if len(rows) > 1 else 0.0
        ),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def score_action_strength_icl_results(
    *,
    result_paths: Iterable[Path | str],
    method_name: str,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Rescore one checkpoint or aggregate three independent train seeds."""

    release = load_action_strength_icl_release(release_config)
    paths = [Path(value).expanduser().resolve() for value in result_paths]
    if len(paths) not in {1, 3}:
        raise ValueError(
            "Provide one result for a descriptive checkpoint or three "
            "results for a method-level claim"
        )
    results = [
        _rescore_prediction_result(path, release=release) for path in paths
    ]
    hashes = [
        str(result["model"]["adapter"].get("checkpoint_sha256", ""))
        for result in results
    ]
    if (
        any(len(value) != 64 for value in hashes)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError("Every result must bind a distinct checkpoint hash")
    seeds = [result["model"]["training_seed"] for result in results]
    if len(paths) == 3:
        if any(seed is None for seed in seeds) or len(set(seeds)) != 3:
            raise ValueError("A method claim requires three training seeds")
        recipes = {
            str(result["model"]["training_recipe"]) for result in results
        }
        if len(recipes) != 1:
            raise ValueError("A method score cannot mix training recipes")
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "rule_switch_rate",
        "worst_strength_correct_future_rate",
    )
    if all(
        "joint_icl_pair_success_rate" in result["metrics"]
        for result in results
    ):
        metric_names += ("joint_icl_pair_success_rate",)
    checkpoints = [
        {
            "path": str(path),
            "checkpoint_sha256": result["model"]["adapter"][
                "checkpoint_sha256"
            ],
            "training_seed": result["model"]["training_seed"],
            **{name: result["metrics"][name] for name in metric_names},
            "passed": bool(result["gate"]["passed"]),
        }
        for path, result in zip(paths, results, strict=True)
    ]
    formal = len(paths) == 3
    return {
        "schema_version": 1,
        "benchmark": "pusht_history3_action_strength_icl_v1",
        "submission_kind": (
            "three_seed_method" if formal else "descriptive_checkpoint"
        ),
        "status": "completed",
        "method_name": str(method_name),
        "release_id": release["release_id"],
        "checkpoints": checkpoints,
        "aggregate": {
            metric: _stats(row[metric] for row in checkpoints)
            for metric in metric_names
        },
        "decision": {
            "passed": formal and all(row["passed"] for row in checkpoints),
            "formal_method_claim": formal,
            "reason": (
                "all_three_training_seeds_passed"
                if formal and all(row["passed"] for row in checkpoints)
                else (
                    "one_or_more_training_seeds_failed"
                    if formal
                    else "single_checkpoint_is_descriptive_only"
                )
            ),
        },
    }


def score_action_strength_planning_submission(
    *,
    submission_path: Path | str,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Score selected action amplitudes against the frozen physical oracle.

    A planner submission contains 512 records with ``condition_id`` and
    ``selected_amplitude``.  The scorer never trusts model-provided hidden
    labels or oracle amplitudes.
    """

    root = (repo_root or repository_root()).resolve()
    release = load_action_strength_icl_release(release_config)
    submission_path = Path(submission_path).expanduser().resolve()
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    records = submission.get("records")
    if not isinstance(records, list):
        raise ValueError("Planning submission must contain records")
    by_id = {}
    for row in records:
        condition_id = str(row["condition_id"])
        selected_value = row.get("selected_amplitude")
        if selected_value is None and isinstance(row.get("execution"), dict):
            selected_value = row["execution"].get("amplitude")
        if selected_value is None:
            raise ValueError(
                f"Planning record has no selected amplitude: {condition_id}"
            )
        amplitude = float(selected_value)
        if condition_id in by_id or not 0.0 <= amplitude <= 1.0:
            raise ValueError(
                f"Invalid or duplicate planning record {condition_id}"
            )
        by_id[condition_id] = amplitude

    oracle_specification = release["evaluation"]["planning_oracle"]
    oracle_path = resolve_contextworld_path(
        oracle_specification["path"], repo_root=root
    )
    if file_sha256(oracle_path) != oracle_specification["sha256"]:
        raise RuntimeError("Frozen Action Strength oracle hash changed")
    oracle_rows = json.loads(oracle_path.read_text(encoding="utf-8"))
    oracle_by_id = {str(row["condition_id"]): row for row in oracle_rows}
    if set(by_id) != set(oracle_by_id) or len(by_id) != 512:
        raise ValueError(
            "Planning submission must cover all 512 frozen conditions"
        )

    pair_oracles: dict[int, dict[str, float]] = {}
    for row in oracle_rows:
        pair_oracles.setdefault(int(row["pair_index"]), {})[
            str(row["mode"])
        ] = float(row["best"]["amplitude"])
    scored = []
    for condition_id, selected in by_id.items():
        oracle = oracle_by_id[condition_id]
        mode = str(oracle["mode"])
        pair_index = int(oracle["pair_index"])
        other_mode = "high_gain" if mode == "low_gain" else "low_gain"
        own = pair_oracles[pair_index][mode]
        other = pair_oracles[pair_index][other_mode]
        own_regret = abs(selected - own)
        other_regret = abs(selected - other)
        scored.append(
            {
                "condition_id": condition_id,
                "pair_index": pair_index,
                "strength": (
                    "lower" if mode == "low_gain" else "higher"
                ),
                "selected_amplitude": selected,
                "oracle_amplitude": own,
                "absolute_amplitude_regret": own_regret,
                "correct_strength_action_region": own_regret < other_regret,
            }
        )
    accuracy = float(
        np.mean(
            [row["correct_strength_action_region"] for row in scored]
        )
    )
    threshold = float(
        release["scoring"]["action_planning"][
            "correct_action_region_rate_minimum"
        ]
    )
    return {
        "schema_version": 1,
        "benchmark": "pusht_history3_action_strength_planning_v1",
        "submission_kind": "frozen_physical_oracle_action_selection",
        "status": "completed",
        "release_id": release["release_id"],
        "submission": {
            "path": str(submission_path),
            "sha256": file_sha256(submission_path),
        },
        "oracle": {
            "path": str(oracle_path),
            "sha256": oracle_specification["sha256"],
        },
        "summary": {
            "condition_count": len(scored),
            "pair_count": len(pair_oracles),
            "correct_action_region_rate": accuracy,
            "mean_absolute_amplitude_regret": float(
                np.mean(
                    [
                        row["absolute_amplitude_regret"]
                        for row in scored
                    ]
                )
            ),
        },
        "gate": {
            "minimum": threshold,
            "passed": accuracy >= threshold,
        },
        "records": scored,
    }


def score_action_strength_retention_report(
    *,
    report_path: Path | str,
    model_name: str,
    release_config: Path | str = DEFAULT_ACTION_STRENGTH_RELEASE_CONFIG,
) -> dict[str, Any]:
    """Validate and score one model from the frozen standard PushT CEM run."""

    release = load_action_strength_icl_release(release_config)
    report_path = Path(report_path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = report.get("protocol", {})
    expected = release["scoring"]["original_task_retention"]
    protocol_checks = {
        "eval_seeds": protocol.get("eval_seeds") == expected["eval_seeds"],
        "queries_per_seed": (
            protocol.get("num_eval_per_seed")
            == expected["queries_per_seed"]
        ),
        "history_len": protocol.get("history_len") == 3,
        "horizon": protocol.get("horizon") == 5,
        "receding_horizon": protocol.get("receding_horizon") == 5,
        "action_block": protocol.get("action_block") == 5,
        "cem_samples": protocol.get("cem_samples") == 300,
        "cem_iterations": protocol.get("cem_iterations") == 30,
        "cem_topk": protocol.get("cem_topk") == 30,
    }
    catalog = report.get("query_catalog", {})
    catalog_check = (
        catalog.get("sha256") == expected["query_catalog_sha256"]
    )
    matches = [
        row for row in report.get("models", []) if row.get("model") == model_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one retention model named {model_name!r}"
        )
    model = matches[0]
    aggregate = model.get("aggregate", {})
    successes = int(aggregate.get("success_count", -1))
    evaluations = int(aggregate.get("evaluation_count", -1))
    minimum = int(expected["noninferiority_minimum_successes"])
    passed = bool(
        all(protocol_checks.values())
        and catalog_check
        and evaluations == expected["independent_cem_episodes"]
        and successes >= minimum
    )
    return {
        "schema_version": 1,
        "benchmark": "pusht_action_strength_original_task_retention_v1",
        "submission_kind": "standard_pusht_cem_retention",
        "status": "completed" if passed else "failed",
        "release_id": release["release_id"],
        "report": {
            "path": str(report_path),
            "sha256": file_sha256(report_path),
        },
        "model": {
            "name": model_name,
            "checkpoint_sha256": model.get("checkpoint_sha256"),
        },
        "protocol_checks": {
            **protocol_checks,
            "query_catalog": catalog_check,
        },
        "score": {
            "successes": successes,
            "evaluations": evaluations,
            "success_rate": (
                successes / evaluations if evaluations > 0 else None
            ),
            "standard_only_reference_successes": int(
                expected["standard_only_successes"]
            ),
            "difference_from_standard_only": (
                successes - int(expected["standard_only_successes"])
            ),
        },
        "gate": {
            "minimum_successes": minimum,
            "passed": passed,
        },
    }


__all__ = [
    "evaluate_action_strength_icl_model",
    "score_action_strength_icl_results",
    "score_action_strength_planning_submission",
    "score_action_strength_retention_report",
]
