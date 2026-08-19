from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import SpeedICLModelAdapter


ACCURACY_METRICS = (
    "speed_history_accuracy",
    "door_history_accuracy",
    "joint_history_accuracy",
    "speed_target_accuracy",
    "door_target_accuracy",
    "joint_target_accuracy",
)


def factor_key(factor: tuple[float, str]) -> str:
    speed, rule = factor
    return f"speed_{float(speed):04.1f}_{rule}".replace(".", "p")


def _adapter_protocol_audit(
    adapter: SpeedICLModelAdapter,
) -> dict[str, Any]:
    protocol = adapter.protocol
    checks = {
        "history_tokens_are_three": int(protocol.history_tokens) == 3,
        "raw_steps_per_action_block_are_five": (
            int(protocol.action_block_raw_steps) == 5
        ),
        "action_dimension_is_two": int(protocol.action_dim) == 2,
        "at_least_one_future_is_supported": (
            int(protocol.future_action_blocks) >= 1
        ),
        "native_target_encoder_is_used": bool(
            protocol.native_target_encoder
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Adapter protocol mismatch: {checks}")
    return {"passed": True, "checks": checks}


def _strict_choice(
    losses: np.ndarray,
    *,
    correct_index: int,
    candidates: Iterable[int],
    epsilon: float,
) -> tuple[bool, float]:
    candidate_indices = tuple(map(int, candidates))
    if correct_index not in candidate_indices:
        raise ValueError("Correct index is absent from candidate set")
    wrong = tuple(
        index for index in candidate_indices if index != correct_index
    )
    if not wrong:
        raise ValueError("A strict choice needs at least one alternative")
    correct_loss = float(losses[correct_index])
    nearest_wrong = float(min(losses[index] for index in wrong))
    margin = nearest_wrong - correct_loss
    return bool(margin > float(epsilon)), margin


def score_validation_assets(
    adapter: SpeedICLModelAdapter,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
    epsilon: float,
) -> dict[str, Any]:
    """Score the frozen six-condition matrix without constructing an env."""

    if not assets:
        raise ValueError("Cannot score an empty Validation catalog")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    protocol_audit = _adapter_protocol_audit(adapter)
    factors = tuple(assets[0]["histories"])
    if len(factors) != 6:
        raise ValueError(f"Expected six factor conditions, got {factors}")
    speeds = tuple(dict.fromkeys(float(speed) for speed, _ in factors))
    rules = tuple(dict.fromkeys(str(rule) for _, rule in factors))
    if len(speeds) != 3 or len(rules) != 2:
        raise ValueError(
            f"Expected three speeds and two rules, got {speeds}/{rules}"
        )
    for asset in assets:
        if (
            tuple(asset["histories"]) != factors
            or tuple(asset["actions"]) != factors
            or tuple(asset["targets"]) != factors
        ):
            raise RuntimeError(
                f"Factor order changed in query {asset['query_id']}"
            )

    state_before = adapter.frozen_state_hash()
    samples = [
        (asset, factor)
        for asset in assets
        for factor in factors
    ]
    input_pixels = np.stack(
        [asset["histories"][factor] for asset, factor in samples]
    ).astype(np.uint8)
    raw_actions = np.stack(
        [asset["actions"][factor] for asset, factor in samples]
    ).astype(np.float32)
    predictions = np.asarray(
        adapter.rollout_latents(
            input_pixels,
            raw_actions,
            batch_size=int(batch_size),
        )
    )
    if predictions.ndim < 3 or predictions.shape[:2] != (
        len(samples),
        1,
    ):
        raise RuntimeError(
            "Adapter must return one future latent per factor history; "
            f"observed={predictions.shape}"
        )
    predictions = predictions[:, 0].reshape(
        len(assets), len(factors), -1
    ).astype(np.float64)

    target_pixels = np.stack(
        [
            asset["targets"][factor]
            for asset in assets
            for factor in factors
        ]
    ).astype(np.uint8)
    encoded_targets = np.asarray(
        adapter.encode_pixels(
            target_pixels,
            batch_size=int(batch_size),
        )
    ).reshape(len(assets), len(factors), -1)
    encoded_targets = encoded_targets.astype(np.float64)
    if predictions.shape[-1] != encoded_targets.shape[-1]:
        raise RuntimeError(
            "Prediction and target latent dimensions differ: "
            f"{predictions.shape[-1]} != {encoded_targets.shape[-1]}"
        )
    if not (
        np.isfinite(predictions).all()
        and np.isfinite(encoded_targets).all()
    ):
        raise RuntimeError("Prediction or target latent is non-finite")

    losses = np.mean(
        np.square(
            predictions[:, :, None, :]
            - encoded_targets[:, None, :, :]
        ),
        axis=-1,
    )
    pairwise_target_mse = np.stack(
        [
            np.mean(
                np.square(
                    encoded_targets[:, left]
                    - encoded_targets[:, right]
                ),
                axis=-1,
            )
            for left in range(len(factors))
            for right in range(left + 1, len(factors))
        ],
        axis=1,
    )

    rows: list[dict[str, Any]] = []
    for asset_index, asset in enumerate(assets):
        for target_index, true_factor in enumerate(factors):
            true_speed, true_rule = true_factor
            history_losses = losses[asset_index, :, target_index]
            matching_prediction_losses = losses[
                asset_index, target_index, :
            ]
            same_rule_indices = [
                index
                for index, (_, rule) in enumerate(factors)
                if rule == true_rule
            ]
            same_speed_indices = [
                index
                for index, (speed, _) in enumerate(factors)
                if speed == true_speed
            ]
            all_indices = range(len(factors))
            speed_history, speed_history_margin = _strict_choice(
                history_losses,
                correct_index=target_index,
                candidates=same_rule_indices,
                epsilon=epsilon,
            )
            door_history, door_history_margin = _strict_choice(
                history_losses,
                correct_index=target_index,
                candidates=same_speed_indices,
                epsilon=epsilon,
            )
            joint_history, joint_history_margin = _strict_choice(
                history_losses,
                correct_index=target_index,
                candidates=all_indices,
                epsilon=epsilon,
            )
            speed_target, speed_target_margin = _strict_choice(
                matching_prediction_losses,
                correct_index=target_index,
                candidates=same_rule_indices,
                epsilon=epsilon,
            )
            door_target, door_target_margin = _strict_choice(
                matching_prediction_losses,
                correct_index=target_index,
                candidates=same_speed_indices,
                epsilon=epsilon,
            )
            joint_target, joint_target_margin = _strict_choice(
                matching_prediction_losses,
                correct_index=target_index,
                candidates=all_indices,
                epsilon=epsilon,
            )
            rows.append(
                {
                    "evaluation_id": (
                        f"{asset['query_id']}/{factor_key(true_factor)}"
                    ),
                    "query_id": str(asset["query_id"]),
                    "eval_seed": int(asset["eval_seed"]),
                    "evaluation_index": int(asset["evaluation_index"]),
                    "direction": str(asset["direction"]),
                    "door_position": int(asset["door_position"]),
                    "template_id": str(asset["template_id"]),
                    "true_speed": float(true_speed),
                    "true_rule": str(true_rule),
                    "true_condition": factor_key(true_factor),
                    "matching_history_true_latent_mse": float(
                        losses[
                            asset_index,
                            target_index,
                            target_index,
                        ]
                    ),
                    "speed_history_accuracy": speed_history,
                    "door_history_accuracy": door_history,
                    "joint_history_accuracy": joint_history,
                    "speed_target_accuracy": speed_target,
                    "door_target_accuracy": door_target,
                    "joint_target_accuracy": joint_target,
                    "speed_history_margin": speed_history_margin,
                    "door_history_margin": door_history_margin,
                    "joint_history_margin": joint_history_margin,
                    "speed_target_margin": speed_target_margin,
                    "door_target_margin": door_target_margin,
                    "joint_target_margin": joint_target_margin,
                    "history_loss_by_condition": {
                        factor_key(factor): float(
                            history_losses[index]
                        )
                        for index, factor in enumerate(factors)
                    },
                    "matching_history_target_loss_by_condition": {
                        factor_key(factor): float(
                            matching_prediction_losses[index]
                        )
                        for index, factor in enumerate(factors)
                    },
                }
            )

    state_after = adapter.frozen_state_hash()
    expected_records = len(assets) * len(factors)
    if len(rows) != expected_records:
        raise RuntimeError(
            f"Expected {expected_records} records, got {len(rows)}"
        )
    if state_before != state_after:
        raise RuntimeError("Adapter state changed during frozen scoring")
    by_condition = Counter(
        str(row["true_condition"]) for row in rows
    )
    by_seed_condition = Counter(
        (int(row["eval_seed"]), str(row["true_condition"]))
        for row in rows
    )
    if any(
        by_condition[factor_key(factor)] != len(assets)
        for factor in factors
    ):
        raise RuntimeError("True-condition score counts are incomplete")
    return {
        "records": rows,
        "score_audit": {
            "passed": True,
            "unique_queries": len(assets),
            "factor_conditions": len(factors),
            "model_predictions": len(samples),
            "target_encodings": len(target_pixels),
            "loss_comparisons": int(losses.size),
            "records": len(rows),
            "expected_records": expected_records,
            "records_per_true_condition": {
                factor_key(factor): by_condition[factor_key(factor)]
                for factor in factors
            },
            "records_per_eval_seed_and_true_condition": {
                f"{seed}/{condition}": count
                for (seed, condition), count in sorted(
                    by_seed_condition.items()
                )
            },
            "online_environment_calls": 0,
            "model_input_keys": ["pixels", "action"],
            "privileged_fields_passed_to_adapter": [],
            "epsilon": float(epsilon),
            "adapter_protocol": protocol_audit,
            "frozen_state_hash_before": state_before,
            "frozen_state_hash_after": state_after,
            "target_latent_separation": {
                "pairs_per_query": int(pairwise_target_mse.shape[1]),
                "minimum_mse": float(pairwise_target_mse.min()),
                "mean_mse": float(pairwise_target_mse.mean()),
                "median_mse": float(np.median(pairwise_target_mse)),
                "maximum_mse": float(pairwise_target_mse.max()),
                "pairs_at_or_below_epsilon": int(
                    np.sum(pairwise_target_mse <= epsilon)
                ),
                "all_pairs_above_epsilon": bool(
                    np.all(pairwise_target_mse > epsilon)
                ),
            },
        },
    }


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty score cell")
    return {
        "samples": len(rows),
        **{
            metric: float(
                np.mean([bool(row[metric]) for row in rows])
            )
            for metric in ACCURACY_METRICS
        },
        "matching_history_true_latent_mse": float(
            np.mean(
                [
                    float(row["matching_history_true_latent_mse"])
                    for row in rows
                ]
            )
        ),
        "mean_margins": {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "speed_history_margin",
                "door_history_margin",
                "joint_history_margin",
                "speed_target_margin",
                "door_target_margin",
                "joint_target_margin",
            )
        },
    }


def summarize_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_seed[int(row["eval_seed"])].append(row)
        by_condition[str(row["true_condition"])].append(row)
    return {
        "overall": _cell_summary(records),
        "by_eval_seed": {
            str(seed): _cell_summary(rows)
            for seed, rows in sorted(by_seed.items())
        },
        "by_true_condition": {
            condition: _cell_summary(rows)
            for condition, rows in sorted(by_condition.items())
        },
        "raw_native_latent_mse_cross_checkpoint_comparison_allowed": False,
        "accuracy_metrics_cross_checkpoint_comparison_allowed": True,
    }


def evaluate_checkpoint_gate(
    *,
    summary: dict[str, Any],
    score_audit: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    overall_thresholds = dict(gates["checkpoint"]["overall"])
    seed_thresholds = dict(gates["checkpoint"]["every_eval_seed"])
    overall_checks = {
        metric: bool(
            float(summary["overall"][metric]) >= float(threshold)
        )
        for metric, threshold in overall_thresholds.items()
    }
    per_seed_checks = {
        seed: {
            metric: bool(float(cell[metric]) >= float(threshold))
            for metric, threshold in seed_thresholds.items()
        }
        for seed, cell in summary["by_eval_seed"].items()
    }
    minimum_separation = float(
        gates["minimum_target_pair_latent_mse"]
    )
    observed_separation = float(
        score_audit["target_latent_separation"]["minimum_mse"]
    )
    checks = {
        "all_overall_accuracy_thresholds_pass": all(
            overall_checks.values()
        ),
        "all_eval_seed_accuracy_thresholds_pass": all(
            all(cell.values()) for cell in per_seed_checks.values()
        ),
        "all_target_pairs_are_latently_distinct": (
            observed_separation > minimum_separation
        ),
        "strict_ties_fail_in_metric_definition": bool(
            gates["strict_ties_fail"]
        ),
        "offline_scoring_only": (
            int(score_audit["online_environment_calls"]) == 0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "overall": {
            "thresholds": overall_thresholds,
            "checks": overall_checks,
        },
        "every_eval_seed": {
            "thresholds": seed_thresholds,
            "checks": per_seed_checks,
        },
        "target_latent_separation": {
            "minimum_required_exclusive": minimum_separation,
            "minimum_observed": observed_separation,
            "passed": observed_separation > minimum_separation,
        },
    }


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    required_joint_training_seeds: Iterable[int],
) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot aggregate zero checkpoint results")
    content_ids = {
        result["asset_audit"]["content_manifest_sha256"]
        for result in results
    }
    if len(content_ids) != 1:
        raise RuntimeError("Results use different Validation catalogs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities = set()
    for result in results:
        identity = (
            str(result["model_id"]),
            int(result["training_seed"]),
        )
        if identity in identities:
            raise RuntimeError(f"Duplicate result {identity}")
        identities.add(identity)
        grouped[identity[0]].append(result)

    model_summaries = {}
    for model_id, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["training_seed"]))
        model_summaries[model_id] = {
            "training_seeds": [
                int(row["training_seed"]) for row in ordered
            ],
            "checkpoints": len(ordered),
            "checkpoint_gate_passes": [
                bool(row["checkpoint_gate"]["passed"])
                for row in ordered
            ],
            "all_checkpoint_gates_pass": all(
                row["checkpoint_gate"]["passed"] for row in ordered
            ),
            "mean_over_training_seeds": {
                metric: float(
                    np.mean(
                        [
                            row["summary"]["overall"][metric]
                            for row in ordered
                        ]
                    )
                )
                for metric in ACCURACY_METRICS
            },
            "per_training_seed": {
                str(row["training_seed"]): {
                    metric: float(
                        row["summary"]["overall"][metric]
                    )
                    for metric in ACCURACY_METRICS
                }
                | {
                    "checkpoint_gate_passed": bool(
                        row["checkpoint_gate"]["passed"]
                    )
                }
                for row in ordered
            },
        }

    joint_id = "H3_SpeedDoorJoint_PLDM"
    required = tuple(sorted(map(int, required_joint_training_seeds)))
    observed_joint = tuple(
        sorted(model_summaries.get(joint_id, {}).get("training_seeds", ()))
    )
    joint_seed_results = grouped.get(joint_id, [])
    method_checks = {
        "all_required_joint_seeds_present": observed_joint == required,
        "every_joint_seed_passes_checkpoint_gate": (
            len(joint_seed_results) == len(required)
            and all(
                result["checkpoint_gate"]["passed"]
                for result in joint_seed_results
            )
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": "tworoom_speed_door_rule_history3_validation_v1",
        "validation_content_manifest_sha256": next(iter(content_ids)),
        "models": model_summaries,
        "method_gate": {
            "passed": all(method_checks.values()),
            "checks": method_checks,
            "required_joint_training_seeds": list(required),
            "observed_joint_training_seeds": list(observed_joint),
        },
        "conclusion": (
            "History=3 Speed×门规则组合 ICL 通过预注册方法门槛。"
            if all(method_checks.values())
            else "History=3 Speed×门规则组合 ICL 未通过预注册方法门槛。"
        ),
    }


__all__ = [
    "ACCURACY_METRICS",
    "aggregate_results",
    "evaluate_checkpoint_gate",
    "factor_key",
    "score_validation_assets",
    "summarize_records",
]
