from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contextworld.paths import resolve_contextworld_path

from .action_delay import array_sha256, canonical_sha256
from .action_delay_h7_validation import (
    ARRAY_KEYS,
    DELAYS,
    EVAL_SEEDS,
    FUTURE_HORIZONS,
    HISTORY_TOKENS,
    QUERY_COUNT,
    QUERIES_PER_SEED,
    file_sha256,
)


TRAINING_SEEN_DELAYS = (0, 4, 8)
INTERPOLATION_DELAYS = (1, 2, 3, 5, 6, 7)
HIGH_EXTRAPOLATION_DELAYS = (9, 10)
MODEL_PREDICTIONS_PER_CHECKPOINT = QUERY_COUNT * len(DELAYS)
TARGET_ENCODINGS_PER_CHECKPOINT = (
    QUERY_COUNT * len(DELAYS) * len(FUTURE_HORIZONS)
)
HORIZON_LOSS_RECORDS_PER_CHECKPOINT = (
    QUERY_COUNT
    * len(DELAYS)
    * len(DELAYS)
    * len(FUTURE_HORIZONS)
)
TRAJECTORY_COMPARISONS_PER_CHECKPOINT = (
    QUERY_COUNT * len(DELAYS) * len(DELAYS)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _target_track(delay: int) -> str:
    if delay in TRAINING_SEEN_DELAYS:
        return "training_seen"
    if delay in INTERPOLATION_DELAYS:
        return "within_range_unseen"
    if delay in HIGH_EXTRAPOLATION_DELAYS:
        return "above_range_unseen"
    raise ValueError(f"Unknown delay: {delay}")


def physical_future_group(delay: int, horizon: int | str) -> int:
    """Return the physically identifiable future group.

    At h1, delays 5..10 are all stationary and therefore form one group.
    The three-step trajectory and h2/h3 distinguish every delay.
    """

    if horizon == 1:
        return min(int(delay), 5)
    if horizon in (2, 3, "trajectory"):
        return int(delay)
    raise ValueError(f"Unsupported horizon: {horizon}")


def load_h7_validation_assets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog_path = Path(catalog_path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    content = {
        "benchmark": catalog.get("benchmark"),
        "protocol": catalog.get("protocol"),
        "queries": catalog.get("queries"),
    }
    _require(
        catalog.get("status") == "frozen_before_model_scoring",
        "History=7 Validation catalog is not frozen",
    )
    _require(
        catalog.get("content_manifest_sha256") == canonical_sha256(content),
        "History=7 Validation catalog content hash changed",
    )
    _require(
        len(catalog.get("queries", ())) == QUERY_COUNT,
        "History=7 Validation must contain 300 distinct queries",
    )
    _require(
        tuple(catalog["protocol"]["delay_values"]) == DELAYS,
        "History=7 Validation delay support changed",
    )
    _require(
        tuple(catalog["protocol"]["future_horizons_action_blocks"])
        == FUTURE_HORIZONS,
        "History=7 Validation horizons changed",
    )

    assets: list[dict[str, Any]] = []
    for row in catalog["queries"]:
        asset_path = resolve_contextworld_path(
            row["asset"], repo_root=repo_root
        )
        _require(
            file_sha256(asset_path) == row["asset_sha256"],
            f"History=7 asset file hash changed: {asset_path}",
        )
        with np.load(asset_path, allow_pickle=False) as bundle:
            arrays = {name: bundle[name].copy() for name in bundle.files}
        _require(
            tuple(sorted(arrays)) == tuple(sorted(ARRAY_KEYS)),
            f"History=7 asset keys changed: {asset_path}",
        )
        hashes = {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        }
        _require(
            hashes == row["array_sha256"],
            f"History=7 asset array hash changed: {asset_path}",
        )
        _require(
            canonical_sha256(hashes) == row["payload_sha256"],
            f"History=7 asset payload hash changed: {asset_path}",
        )
        _require(
            arrays["history_pixels"].shape[0] == len(DELAYS)
            and arrays["history_pixels"].shape[1] == HISTORY_TOKENS,
            f"Unexpected History=7 pixels: {asset_path}",
        )
        _require(
            arrays["action_blocks"].shape[:2]
            == (HISTORY_TOKENS - 1 + len(FUTURE_HORIZONS), 5),
            f"Unexpected History=7 actions: {asset_path}",
        )
        _require(
            arrays["true_future_pixels"].shape[:2]
            == (len(DELAYS), len(FUTURE_HORIZONS)),
            f"Unexpected History=7 true futures: {asset_path}",
        )
        assets.append({**row, **arrays})
    return catalog, assets


def score_h7_validation_assets(
    adapter: Any,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    _require(len(assets) == QUERY_COUNT, "Expected 300 History=7 assets")
    protocol = adapter.protocol
    _require(
        int(protocol.history_tokens) == HISTORY_TOKENS,
        "Adapter is not History=7",
    )
    _require(
        int(protocol.action_block_raw_steps) == 5,
        "Adapter action block is not five raw steps",
    )
    _require(
        int(protocol.future_action_blocks) >= len(FUTURE_HORIZONS),
        "Adapter cannot produce three future action blocks",
    )

    input_pixels = np.concatenate(
        [asset["history_pixels"] for asset in assets], axis=0
    )
    action_blocks = np.concatenate(
        [
            np.repeat(
                asset["action_blocks"][None],
                len(DELAYS),
                axis=0,
            )
            for asset in assets
        ],
        axis=0,
    )
    predicted = np.asarray(
        adapter.rollout_latents(
            input_pixels,
            action_blocks,
            batch_size=int(batch_size),
        ),
        dtype=np.float32,
    )
    _require(
        predicted.shape[0] == MODEL_PREDICTIONS_PER_CHECKPOINT
        and predicted.shape[1] == len(FUTURE_HORIZONS),
        f"Unexpected History=7 rollout shape: {predicted.shape}",
    )

    target_pixels = np.concatenate(
        [asset["true_future_pixels"] for asset in assets], axis=0
    )
    flat_targets = target_pixels.reshape(
        -1, *target_pixels.shape[-3:]
    )
    encoded = np.asarray(
        adapter.encode_pixels(flat_targets, batch_size=int(batch_size)),
        dtype=np.float32,
    )
    _require(
        encoded.shape[0] == TARGET_ENCODINGS_PER_CHECKPOINT,
        f"Unexpected History=7 target encodings: {encoded.shape}",
    )
    predicted = predicted.reshape(
        QUERY_COUNT,
        len(DELAYS),
        len(FUTURE_HORIZONS),
        -1,
    )
    encoded = encoded.reshape(
        QUERY_COUNT,
        len(DELAYS),
        len(FUTURE_HORIZONS),
        -1,
    )

    records: list[dict[str, Any]] = []
    for query_index, asset in enumerate(assets):
        for history_index, history_delay in enumerate(DELAYS):
            for target_index, target_delay in enumerate(DELAYS):
                for horizon_index, horizon in enumerate(FUTURE_HORIZONS):
                    prediction = predicted[
                        query_index, history_index, horizon_index
                    ]
                    target = encoded[
                        query_index, target_index, horizon_index
                    ]
                    records.append(
                        {
                            "query_id": str(asset["query_id"]),
                            "eval_seed": int(asset["eval_seed"]),
                            "evaluation_index": int(
                                asset["evaluation_index"]
                            ),
                            "room": str(asset["room"]),
                            "direction": str(asset["direction"]),
                            "history_delay": int(history_delay),
                            "target_delay": int(target_delay),
                            "target_track": _target_track(target_delay),
                            "horizon": int(horizon),
                            "target_physical_group": physical_future_group(
                                target_delay, horizon
                            ),
                            "latent_mse": float(
                                np.mean((prediction - target) ** 2)
                            ),
                        }
                    )
    _require(
        len(records) == HORIZON_LOSS_RECORDS_PER_CHECKPOINT,
        "History=7 horizon loss record count changed",
    )
    return {
        "records": records,
        "score_audit": {
            "queries": len(assets),
            "model_predictions": len(input_pixels),
            "target_encodings": len(flat_targets),
            "trajectory_comparisons": (
                TRAJECTORY_COMPARISONS_PER_CHECKPOINT
            ),
            "horizon_loss_records": len(records),
            "online_environment_calls": 0,
            "model_visible_fields": ["pixels", "action"],
            "privileged_fields_passed_to_adapter": [],
        },
    }


def _aggregate_metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
    _require(bool(values), "Cannot aggregate empty History=7 metrics")
    return {
        "query_target_units": len(values),
        "mean_matching_history_loss": float(
            np.mean([row["matching_history_loss"] for row in values])
        ),
        "mean_other_history_loss": float(
            np.mean([row["other_history_mean_loss"] for row in values])
        ),
        "mean_history_margin": float(
            np.mean([row["history_margin"] for row in values])
        ),
        "mean_history_loss_ratio": float(
            np.mean([row["history_loss_ratio"] for row in values])
        ),
        "matching_history_strict_win_rate": float(
            np.mean([row["matching_history_strict_win"] for row in values])
        ),
        "exact_history_selection_rate": float(
            np.mean([row["exact_history_selection_correct"] for row in values])
        ),
        "exact_target_selection_rate": float(
            np.mean([row["exact_target_selection_correct"] for row in values])
        ),
        "physical_history_group_selection_rate": float(
            np.mean(
                [
                    row["physical_history_group_selection_correct"]
                    for row in values
                ]
            )
        ),
        "physical_target_group_selection_rate": float(
            np.mean(
                [
                    row["physical_target_group_selection_correct"]
                    for row in values
                ]
            )
        ),
    }


def _summarize_query_matrices(
    matrices: Iterable[
        tuple[dict[str, Any], int | str, dict[tuple[int, int], float]]
    ],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for exemplar, horizon, losses in matrices:
        _require(
            len(losses) == len(DELAYS) ** 2,
            f"Incomplete 11x11 loss matrix: {exemplar['query_id']}",
        )
        for target_delay in DELAYS:
            matching = float(losses[(target_delay, target_delay)])
            other = [
                float(losses[(history_delay, target_delay)])
                for history_delay in DELAYS
                if history_delay != target_delay
            ]
            selected_history = min(
                DELAYS,
                key=lambda delay: (
                    losses[(delay, target_delay)],
                    delay,
                ),
            )
            selected_target = min(
                DELAYS,
                key=lambda delay: (
                    losses[(target_delay, delay)],
                    delay,
                ),
            )
            target_group = physical_future_group(target_delay, horizon)
            metrics.append(
                {
                    "query_id": str(exemplar["query_id"]),
                    "eval_seed": int(exemplar["eval_seed"]),
                    "evaluation_index": int(
                        exemplar["evaluation_index"]
                    ),
                    "room": str(exemplar["room"]),
                    "direction": str(exemplar["direction"]),
                    "horizon": horizon,
                    "target_delay": int(target_delay),
                    "target_track": _target_track(target_delay),
                    "target_physical_group": target_group,
                    "matching_history_loss": matching,
                    "other_history_mean_loss": float(np.mean(other)),
                    "history_margin": float(np.mean(other) - matching),
                    "history_loss_ratio": float(
                        matching / max(float(np.mean(other)), 1e-12)
                    ),
                    "matching_history_strict_win": matching < min(other),
                    "selected_history": int(selected_history),
                    "exact_history_selection_correct": (
                        selected_history == target_delay
                    ),
                    "physical_history_group_selection_correct": (
                        physical_future_group(selected_history, horizon)
                        == target_group
                    ),
                    "selected_target": int(selected_target),
                    "exact_target_selection_correct": (
                        selected_target == target_delay
                    ),
                    "physical_target_group_selection_correct": (
                        physical_future_group(selected_target, horizon)
                        == target_group
                    ),
                }
            )
    return metrics


def _breakdowns(
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "overall": _aggregate_metrics(metrics),
        "by_target_delay": {
            str(delay): _aggregate_metrics(
                [row for row in metrics if row["target_delay"] == delay]
            )
            for delay in DELAYS
        },
        "by_track": {
            track: _aggregate_metrics(
                [row for row in metrics if row["target_track"] == track]
            )
            for track in (
                "training_seen",
                "within_range_unseen",
                "above_range_unseen",
            )
        },
        "by_target_delay_and_eval_seed": {
            str(delay): {
                str(seed): _aggregate_metrics(
                    [
                        row
                        for row in metrics
                        if row["target_delay"] == delay
                        and row["eval_seed"] == seed
                    ]
                )
                for seed in EVAL_SEEDS
            }
            for delay in DELAYS
        },
        "by_target_delay_and_direction": {
            str(delay): {
                direction: _aggregate_metrics(
                    [
                        row
                        for row in metrics
                        if row["target_delay"] == delay
                        and row["direction"] == direction
                    ]
                )
                for direction in ("up", "down")
            }
            for delay in DELAYS
        },
    }


def summarize_h7_validation_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(
        len(records) == HORIZON_LOSS_RECORDS_PER_CHECKPOINT,
        "Incomplete History=7 loss records",
    )
    grouped: dict[
        tuple[str, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in records:
        grouped[(str(row["query_id"]), int(row["horizon"]))].append(row)
    _require(
        len(grouped) == QUERY_COUNT * len(FUTURE_HORIZONS),
        "Incomplete History=7 query/horizon groups",
    )

    horizon_matrices = []
    trajectory_groups: dict[
        str, dict[tuple[int, int], list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    trajectory_exemplars: dict[str, dict[str, Any]] = {}
    for (query_id, horizon), rows in sorted(grouped.items()):
        losses = {
            (int(row["history_delay"]), int(row["target_delay"])): float(
                row["latent_mse"]
            )
            for row in rows
        }
        horizon_matrices.append((rows[0], horizon, losses))
        trajectory_exemplars[query_id] = rows[0]
        for pair, loss in losses.items():
            trajectory_groups[query_id][pair].append(loss)

    horizon_metrics = _summarize_query_matrices(horizon_matrices)
    trajectory_matrices = []
    for query_id in sorted(trajectory_groups):
        loss_lists = trajectory_groups[query_id]
        _require(
            all(len(values) == len(FUTURE_HORIZONS) for values in loss_lists.values()),
            f"Incomplete three-step trajectory: {query_id}",
        )
        trajectory_matrices.append(
            (
                trajectory_exemplars[query_id],
                "trajectory",
                {
                    pair: float(np.mean(values))
                    for pair, values in loss_lists.items()
                },
            )
        )
    trajectory_metrics = _summarize_query_matrices(
        trajectory_matrices
    )
    return {
        "trajectory": {
            **_breakdowns(trajectory_metrics),
            "query_metrics": trajectory_metrics,
        },
        "by_horizon": {
            str(horizon): {
                **_breakdowns(
                    [
                        row
                        for row in horizon_metrics
                        if row["horizon"] == horizon
                    ]
                ),
                "query_metrics": [
                    row
                    for row in horizon_metrics
                    if row["horizon"] == horizon
                ],
            }
            for horizon in FUTURE_HORIZONS
        },
    }


__all__ = [
    "HIGH_EXTRAPOLATION_DELAYS",
    "HORIZON_LOSS_RECORDS_PER_CHECKPOINT",
    "INTERPOLATION_DELAYS",
    "MODEL_PREDICTIONS_PER_CHECKPOINT",
    "TARGET_ENCODINGS_PER_CHECKPOINT",
    "TRAINING_SEEN_DELAYS",
    "TRAJECTORY_COMPARISONS_PER_CHECKPOINT",
    "load_h7_validation_assets",
    "physical_future_group",
    "score_h7_validation_assets",
    "summarize_h7_validation_records",
]
