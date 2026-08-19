from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np

from contextworld.benchmarks.adapters import SpeedICLModelAdapter

from .speed_door_rule_v2_validation import passable_target_key


PRIMARY_METRICS = (
    "passable_speed_future_accuracy",
    "door_future_accuracy",
    "physical_future_macro_accuracy",
    "blocked_speed_suppression_win_rate",
)


def _strict_choice(
    losses: np.ndarray,
    *,
    correct_index: int,
    candidates: Iterable[int],
    epsilon: float,
) -> tuple[bool, float]:
    candidates = tuple(map(int, candidates))
    if correct_index not in candidates:
        raise ValueError("Correct target is absent from candidates")
    wrong = tuple(index for index in candidates if index != correct_index)
    if not wrong:
        raise ValueError("A strict choice needs an alternative")
    margin = float(min(losses[index] for index in wrong)) - float(
        losses[correct_index]
    )
    return bool(margin > float(epsilon)), margin


def _adapter_audit(adapter: SpeedICLModelAdapter) -> dict[str, Any]:
    protocol = adapter.protocol
    checks = {
        "history_tokens_are_three": int(protocol.history_tokens) == 3,
        "raw_steps_per_action_block_are_five": (
            int(protocol.action_block_raw_steps) == 5
        ),
        "action_dimension_is_two": int(protocol.action_dim) == 2,
        "two_future_blocks_supported": (
            int(protocol.future_action_blocks) >= 2
        ),
        "native_target_encoder_used": bool(
            protocol.native_target_encoder
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2 adapter protocol mismatch: {checks}")
    return {"passed": True, "checks": checks}


def score_v2_assets(
    adapter: SpeedICLModelAdapter,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
    epsilon: float,
) -> dict[str, Any]:
    if not assets:
        raise ValueError("Cannot score empty v2 assets")
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    protocol_audit = _adapter_audit(adapter)
    factors = tuple(assets[0]["histories"])
    speeds = tuple(
        dict.fromkeys(float(speed) for speed, _ in factors)
    )
    if len(factors) != 6 or len(speeds) != 3:
        raise ValueError("v2 scorer requires three speeds × two rules")
    horizons = tuple(assets[0]["targets"])
    if horizons != (1, 2):
        raise ValueError(f"v2 scorer requires h1/h2, got {horizons}")
    target_keys = tuple(assets[0]["targets"][1])
    expected_targets = ("blocked",) + tuple(
        passable_target_key(speed) for speed in speeds
    )
    if target_keys != expected_targets:
        raise ValueError(
            f"Unexpected v2 target order: {target_keys}"
        )
    for asset in assets:
        if (
            tuple(asset["histories"]) != factors
            or tuple(asset["actions"]) != factors
            or tuple(asset["targets"]) != horizons
            or any(
                tuple(asset["targets"][horizon]) != target_keys
                for horizon in horizons
            )
        ):
            raise RuntimeError(
                f"v2 asset order drift: {asset['query_id']}"
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
        len(horizons),
    ):
        raise RuntimeError(
            "v2 adapter must return h1/h2 predictions, got "
            f"{predictions.shape}"
        )
    predictions = predictions.reshape(
        len(assets), len(factors), len(horizons), -1
    ).transpose(0, 2, 1, 3)
    predictions = predictions.astype(np.float64)

    target_pixels = np.stack(
        [
            asset["targets"][horizon][target_key]
            for asset in assets
            for horizon in horizons
            for target_key in target_keys
        ]
    ).astype(np.uint8)
    encoded_targets = np.asarray(
        adapter.encode_pixels(
            target_pixels, batch_size=int(batch_size)
        )
    ).reshape(
        len(assets), len(horizons), len(target_keys), -1
    )
    encoded_targets = encoded_targets.astype(np.float64)
    if predictions.shape[-1] != encoded_targets.shape[-1]:
        raise RuntimeError("v2 prediction/target latent dimensions differ")
    if not (
        np.isfinite(predictions).all()
        and np.isfinite(encoded_targets).all()
    ):
        raise RuntimeError("v2 prediction or target is non-finite")

    losses = np.mean(
        np.square(
            predictions[:, :, :, None, :]
            - encoded_targets[:, :, None, :, :]
        ),
        axis=-1,
    )
    target_pair_mse = np.stack(
        [
            np.mean(
                np.square(
                    encoded_targets[:, :, left]
                    - encoded_targets[:, :, right]
                ),
                axis=-1,
            )
            for left, right in combinations(
                range(len(target_keys)), 2
            )
        ],
        axis=-1,
    )
    minimum_target_pair_mse = float(np.min(target_pair_mse))
    if minimum_target_pair_mse <= float(epsilon):
        raise RuntimeError(
            "v2 physical target latents are not distinguishable: "
            f"{minimum_target_pair_mse}"
        )

    factor_index = {
        factor: index for index, factor in enumerate(factors)
    }
    target_index = {
        target_key: index
        for index, target_key in enumerate(target_keys)
    }
    condition_records = []
    suppression_records = []
    for asset_index, asset in enumerate(assets):
        for horizon_index, horizon in enumerate(horizons):
            for speed, rule in factors:
                history_index = factor_index[(speed, rule)]
                prediction_losses = losses[
                    asset_index, horizon_index, history_index
                ]
                correct_key = (
                    passable_target_key(speed)
                    if rule == "passable"
                    else "blocked"
                )
                correct_target = target_index[correct_key]
                passable_indices = [
                    target_index[passable_target_key(value)]
                    for value in speeds
                ]
                speed_future = None
                speed_future_margin = None
                if rule == "passable":
                    speed_future, speed_future_margin = _strict_choice(
                        prediction_losses,
                        correct_index=correct_target,
                        candidates=passable_indices,
                        epsilon=epsilon,
                    )
                door_future, door_future_margin = _strict_choice(
                    prediction_losses,
                    correct_index=correct_target,
                    candidates=(
                        target_index["blocked"],
                        target_index[passable_target_key(speed)],
                    ),
                    epsilon=epsilon,
                )
                physical_future, physical_future_margin = _strict_choice(
                    prediction_losses,
                    correct_index=correct_target,
                    candidates=range(len(target_keys)),
                    epsilon=epsilon,
                )

                target_losses_by_history = losses[
                    asset_index, horizon_index, :, correct_target
                ]
                speed_history = None
                speed_history_margin = None
                if rule == "passable":
                    speed_history, speed_history_margin = _strict_choice(
                        target_losses_by_history,
                        correct_index=history_index,
                        candidates=[
                            factor_index[(value, "passable")]
                            for value in speeds
                        ],
                        epsilon=epsilon,
                    )
                door_history, door_history_margin = _strict_choice(
                    target_losses_by_history,
                    correct_index=history_index,
                    candidates=(
                        factor_index[(speed, "passable")],
                        factor_index[(speed, "blocked")],
                    ),
                    epsilon=epsilon,
                )
                condition_records.append(
                    {
                        "evaluation_id": (
                            f"{asset['query_id']}/h{horizon}/"
                            f"{speed:g}/{rule}"
                        ),
                        "query_id": str(asset["query_id"]),
                        "eval_seed": int(asset["eval_seed"]),
                        "evaluation_index": int(
                            asset["evaluation_index"]
                        ),
                        "direction": str(asset["direction"]),
                        "door_position": int(asset["door_position"]),
                        "template_id": str(asset["template_id"]),
                        "horizon": int(horizon),
                        "speed": float(speed),
                        "rule": str(rule),
                        "physical_target_class": correct_key,
                        "passable_speed_future_accuracy": speed_future,
                        "door_future_accuracy": door_future,
                        "physical_future_accuracy": physical_future,
                        "passable_speed_history_guidance": speed_history,
                        "door_history_guidance": door_history,
                        "passable_speed_future_margin": speed_future_margin,
                        "door_future_margin": door_future_margin,
                        "physical_future_margin": physical_future_margin,
                        "passable_speed_history_margin": (
                            speed_history_margin
                        ),
                        "door_history_margin": door_history_margin,
                        "matching_true_future_latent_mse": float(
                            prediction_losses[correct_target]
                        ),
                        "target_loss_by_physical_class": {
                            key: float(
                                prediction_losses[target_index[key]]
                            )
                            for key in target_keys
                        },
                    }
                )

            for lower, higher in combinations(speeds, 2):
                blocked_lower = predictions[
                    asset_index,
                    horizon_index,
                    factor_index[(lower, "blocked")],
                ]
                blocked_higher = predictions[
                    asset_index,
                    horizon_index,
                    factor_index[(higher, "blocked")],
                ]
                passable_lower = predictions[
                    asset_index,
                    horizon_index,
                    factor_index[(lower, "passable")],
                ]
                passable_higher = predictions[
                    asset_index,
                    horizon_index,
                    factor_index[(higher, "passable")],
                ]
                blocked_distance = float(
                    np.mean(
                        np.square(blocked_lower - blocked_higher)
                    )
                )
                passable_distance = float(
                    np.mean(
                        np.square(passable_lower - passable_higher)
                    )
                )
                margin = passable_distance - blocked_distance
                suppression_records.append(
                    {
                        "evaluation_id": (
                            f"{asset['query_id']}/h{horizon}/"
                            f"{lower:g}-{higher:g}"
                        ),
                        "query_id": str(asset["query_id"]),
                        "eval_seed": int(asset["eval_seed"]),
                        "evaluation_index": int(
                            asset["evaluation_index"]
                        ),
                        "horizon": int(horizon),
                        "lower_speed": float(lower),
                        "higher_speed": float(higher),
                        "blocked_prediction_pair_mse": blocked_distance,
                        "passable_prediction_pair_mse": passable_distance,
                        "blocked_speed_suppression_win": bool(
                            margin > float(epsilon)
                        ),
                        "blocked_speed_suppression_margin": margin,
                    }
                )

    state_after = adapter.frozen_state_hash()
    if state_after != state_before:
        raise RuntimeError("v2 scoring changed model state")
    return {
        "condition_records": condition_records,
        "suppression_records": suppression_records,
        "score_audit": {
            "adapter_protocol": protocol_audit,
            "unique_base_queries": len(assets),
            "factor_histories": len(factors),
            "horizons": len(horizons),
            "physical_target_classes": len(target_keys),
            "model_prediction_sequences": len(samples),
            "model_prediction_endpoints": (
                len(samples) * len(horizons)
            ),
            "target_encodings": (
                len(assets) * len(horizons) * len(target_keys)
            ),
            "loss_comparisons": (
                len(assets)
                * len(horizons)
                * len(factors)
                * len(target_keys)
            ),
            "minimum_target_pair_latent_mse": (
                minimum_target_pair_mse
            ),
            "model_state_unchanged": True,
        },
    }


def _mean_boolean(
    records: list[dict[str, Any]], key: str
) -> float:
    values = [
        bool(record[key])
        for record in records
        if record.get(key) is not None
    ]
    if not values:
        raise ValueError(f"No values for {key}")
    return float(np.mean(values))


def _condition_summary(
    condition_records: list[dict[str, Any]],
    suppression_records: list[dict[str, Any]],
) -> dict[str, float]:
    speed_records = [
        row
        for row in condition_records
        if row["rule"] == "passable"
    ]
    class_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in condition_records:
        class_rows[str(row["physical_target_class"])].append(row)
    class_accuracy = {
        key: _mean_boolean(rows, "physical_future_accuracy")
        for key, rows in sorted(class_rows.items())
    }
    return {
        "passable_speed_future_accuracy": _mean_boolean(
            speed_records, "passable_speed_future_accuracy"
        ),
        "door_future_accuracy": _mean_boolean(
            condition_records, "door_future_accuracy"
        ),
        "door_anchor_future_accuracy": _mean_boolean(
            [
                row
                for row in condition_records
                if np.isclose(float(row["speed"]), 5.1)
            ],
            "door_future_accuracy",
        ),
        "physical_future_macro_accuracy": float(
            np.mean(list(class_accuracy.values()))
        ),
        "blocked_speed_suppression_win_rate": _mean_boolean(
            suppression_records,
            "blocked_speed_suppression_win",
        ),
        "passable_speed_history_guidance": _mean_boolean(
            speed_records, "passable_speed_history_guidance"
        ),
        "door_history_guidance": _mean_boolean(
            condition_records, "door_history_guidance"
        ),
        "physical_future_accuracy_by_class": class_accuracy,
    }


def summarize_v2_scores(
    condition_records: list[dict[str, Any]],
    suppression_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not condition_records or not suppression_records:
        raise ValueError("Cannot summarize empty v2 records")
    horizons = sorted(
        {int(row["horizon"]) for row in condition_records}
    )
    eval_seeds = sorted(
        {int(row["eval_seed"]) for row in condition_records}
    )
    by_horizon = {}
    for horizon in horizons:
        conditions = [
            row
            for row in condition_records
            if int(row["horizon"]) == horizon
        ]
        suppressions = [
            row
            for row in suppression_records
            if int(row["horizon"]) == horizon
        ]
        by_seed = {}
        for seed in eval_seeds:
            seed_conditions = [
                row
                for row in conditions
                if int(row["eval_seed"]) == seed
            ]
            seed_suppressions = [
                row
                for row in suppressions
                if int(row["eval_seed"]) == seed
            ]
            by_seed[str(seed)] = {
                **_condition_summary(
                    seed_conditions, seed_suppressions
                ),
                "condition_records": len(seed_conditions),
                "suppression_records": len(seed_suppressions),
            }
        by_condition = {}
        for speed in sorted(
            {float(row["speed"]) for row in conditions}
        ):
            for rule in ("passable", "blocked"):
                rows = [
                    row
                    for row in conditions
                    if np.isclose(float(row["speed"]), speed)
                    and row["rule"] == rule
                ]
                by_condition[f"{speed:g}/{rule}"] = {
                    "physical_future_accuracy": _mean_boolean(
                        rows, "physical_future_accuracy"
                    ),
                    "door_future_accuracy": _mean_boolean(
                        rows, "door_future_accuracy"
                    ),
                    "passable_speed_future_accuracy": (
                        _mean_boolean(
                            rows,
                            "passable_speed_future_accuracy",
                        )
                        if rule == "passable"
                        else None
                    ),
                    "samples": len(rows),
                }
        by_horizon[f"h{horizon}"] = {
            "overall": {
                **_condition_summary(conditions, suppressions),
                "condition_records": len(conditions),
                "suppression_records": len(suppressions),
            },
            "by_eval_seed": by_seed,
            "by_condition": by_condition,
        }
    return {
        "by_horizon": by_horizon,
        "condition_records": len(condition_records),
        "suppression_records": len(suppression_records),
    }


def evaluate_v2_checkpoint_gate(
    *,
    summary: dict[str, Any],
    config: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    if role not in {"speed_only", "door_only", "joint", "descriptive"}:
        raise ValueError(f"Unknown v2 checkpoint role {role!r}")
    thresholds = config["decision_gates"]["checkpoint"]
    if role == "speed_only":
        required = ("passable_speed_future_accuracy",)
    elif role == "door_only":
        required = ("door_anchor_future_accuracy",)
    elif role == "joint":
        required = PRIMARY_METRICS
    else:
        required = ()
    checks = {}
    for horizon, payload in summary["by_horizon"].items():
        for metric in required:
            threshold_key = (
                "door_future_accuracy"
                if metric == "door_anchor_future_accuracy"
                else metric
            )
            overall_value = float(payload["overall"][metric])
            overall_threshold = float(
                thresholds["overall"][threshold_key]
            )
            checks[f"{horizon}/overall/{metric}"] = {
                "value": overall_value,
                "threshold": overall_threshold,
                "passed": overall_value >= overall_threshold,
            }
            for seed, seed_payload in payload[
                "by_eval_seed"
            ].items():
                value = float(seed_payload[metric])
                threshold = float(
                    thresholds["every_eval_seed"][threshold_key]
                )
                checks[f"{horizon}/seed{seed}/{metric}"] = {
                    "value": value,
                    "threshold": threshold,
                    "passed": value >= threshold,
                }
    return {
        "role": role,
        "required_metrics": list(required),
        "passed": (
            True
            if role == "descriptive"
            else bool(checks)
            and all(row["passed"] for row in checks.values())
        ),
        "checks": checks,
    }


__all__ = [
    "PRIMARY_METRICS",
    "evaluate_v2_checkpoint_gate",
    "score_v2_assets",
    "summarize_v2_scores",
]
