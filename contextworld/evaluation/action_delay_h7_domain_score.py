from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from contextworld.paths import resolve_contextworld_path

from .action_delay import array_sha256, canonical_sha256
from .action_delay_h7_domain_diagnostic import (
    ARRAY_KEYS,
    DELAYS,
    DIAGNOSTIC_EVAL_SEEDS,
    FUTURE_HORIZONS,
    HISTORY_TOKENS,
    QUERIES_PER_TRACK,
)
from .action_delay_h7_validation import file_sha256


RATE_METRICS = (
    "exact_history_selection_rate",
    "exact_target_selection_rate",
    "matching_history_strict_win_rate",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_domain_catalog(
    catalog_path: Path,
) -> dict[str, Any]:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    content = {
        "benchmark": catalog.get("benchmark"),
        "protocol": catalog.get("protocol"),
        "queries": catalog.get("queries"),
    }
    _require(
        catalog.get("status") == "frozen_before_model_scoring",
        "Domain diagnostic catalog is not frozen",
    )
    _require(
        canonical_sha256(content)
        == catalog.get("content_manifest_sha256"),
        "Domain diagnostic catalog content hash changed",
    )
    _require(
        tuple(catalog["protocol"]["delay_values"]) == DELAYS,
        "Domain diagnostic delay support changed",
    )
    _require(
        len(catalog["queries"]) == 1800,
        "Domain diagnostic must contain 1,800 queries",
    )
    counts = Counter(row["track"] for row in catalog["queries"])
    _require(
        set(counts.values()) == {QUERIES_PER_TRACK}
        and len(counts) == 6,
        f"Domain diagnostic track counts changed: {counts}",
    )
    return catalog


def load_domain_track_assets(
    catalog: dict[str, Any],
    *,
    track: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows = [
        row for row in catalog["queries"] if row["track"] == track
    ]
    _require(
        len(rows) == QUERIES_PER_TRACK,
        f"{track}: expected 300 queries",
    )
    assets = []
    for row in rows:
        path = resolve_contextworld_path(
            row["asset"],
            repo_root=repo_root,
        )
        _require(
            file_sha256(path) == row["asset_sha256"],
            f"Asset file hash changed: {path}",
        )
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {name: bundle[name].copy() for name in bundle.files}
        _require(
            tuple(sorted(arrays)) == tuple(sorted(ARRAY_KEYS)),
            f"Asset keys changed: {path}",
        )
        hashes = {
            name: array_sha256(value)
            for name, value in sorted(arrays.items())
        }
        _require(
            hashes == row["array_sha256"],
            f"Asset array hashes changed: {path}",
        )
        _require(
            canonical_sha256(hashes) == row["payload_sha256"],
            f"Asset payload hash changed: {path}",
        )
        _require(
            tuple(map(int, arrays["history_delays"])) == DELAYS
            and tuple(map(int, arrays["target_delays"])) == DELAYS,
            f"Asset delay order changed: {path}",
        )
        assets.append({**row, **arrays})
    return assets


def _aggregate_query_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(bool(rows), "Cannot aggregate empty query metrics")
    return {
        "query_target_units": len(rows),
        "mean_matching_history_loss": float(
            np.mean([row["matching_history_loss"] for row in rows])
        ),
        "mean_other_history_loss": float(
            np.mean([row["other_history_mean_loss"] for row in rows])
        ),
        "mean_history_margin": float(
            np.mean([row["history_margin"] for row in rows])
        ),
        "mean_history_loss_ratio": float(
            np.mean([row["history_loss_ratio"] for row in rows])
        ),
        "matching_history_strict_win_rate": float(
            np.mean([row["matching_history_strict_win"] for row in rows])
        ),
        "exact_history_selection_rate": float(
            np.mean([row["exact_history_selection_correct"] for row in rows])
        ),
        "exact_target_selection_rate": float(
            np.mean([row["exact_target_selection_correct"] for row in rows])
        ),
    }


def _query_metrics(
    losses: np.ndarray,
    assets: list[dict[str, Any]],
    *,
    horizon: int | str,
) -> list[dict[str, Any]]:
    _require(
        losses.shape == (
            QUERIES_PER_TRACK,
            len(DELAYS),
            len(DELAYS),
        ),
        f"Unexpected loss matrix shape: {losses.shape}",
    )
    rows: list[dict[str, Any]] = []
    for query_index, asset in enumerate(assets):
        source_delay = int(asset["source_delay"])
        for target_index, target_delay in enumerate(DELAYS):
            matching = float(
                losses[query_index, target_index, target_index]
            )
            other = [
                float(losses[query_index, history_index, target_index])
                for history_index in range(len(DELAYS))
                if history_index != target_index
            ]
            selected_history_index = int(
                np.argmin(losses[query_index, :, target_index])
            )
            selected_target_index = int(
                np.argmin(losses[query_index, target_index, :])
            )
            rows.append(
                {
                    "query_id": str(asset["query_id"]),
                    "track": str(asset["track"]),
                    "source_split": str(asset["source_split"]),
                    "source_delay": source_delay,
                    "diagnostic_eval_seed": int(
                        asset["diagnostic_eval_seed"]
                    ),
                    "evaluation_index": int(
                        asset["evaluation_index"]
                    ),
                    "room": str(asset["room"]),
                    "direction": str(asset["direction"]),
                    "horizon": horizon,
                    "target_delay": int(target_delay),
                    "is_source_supervised_target": (
                        int(target_delay) == source_delay
                    ),
                    "matching_history_loss": matching,
                    "other_history_mean_loss": float(np.mean(other)),
                    "history_margin": float(np.mean(other) - matching),
                    "history_loss_ratio": float(
                        matching / max(float(np.mean(other)), 1e-12)
                    ),
                    "matching_history_strict_win": matching < min(other),
                    "selected_history": int(
                        DELAYS[selected_history_index]
                    ),
                    "exact_history_selection_correct": (
                        selected_history_index == target_index
                    ),
                    "selected_target": int(
                        DELAYS[selected_target_index]
                    ),
                    "exact_target_selection_correct": (
                        selected_target_index == target_index
                    ),
                }
            )
    return rows


def _breakdowns(
    rows: list[dict[str, Any]],
    *,
    source_delay: int,
) -> dict[str, Any]:
    source_rows = [
        row
        for row in rows
        if int(row["target_delay"]) == int(source_delay)
    ]
    return {
        "overall": _aggregate_query_metrics(rows),
        "by_target_delay": {
            str(delay): _aggregate_query_metrics(
                [
                    row
                    for row in rows
                    if int(row["target_delay"]) == delay
                ]
            )
            for delay in DELAYS
        },
        "by_diagnostic_eval_seed": {
            str(seed): _aggregate_query_metrics(
                [
                    row
                    for row in rows
                    if int(row["diagnostic_eval_seed"]) == seed
                ]
            )
            for seed in DIAGNOSTIC_EVAL_SEEDS
        },
        "source_supervised_target": {
            "overall": _aggregate_query_metrics(source_rows),
            "by_diagnostic_eval_seed": {
                str(seed): _aggregate_query_metrics(
                    [
                        row
                        for row in source_rows
                        if int(row["diagnostic_eval_seed"]) == seed
                    ]
                )
                for seed in DIAGNOSTIC_EVAL_SEEDS
            },
        },
        "query_metrics": rows,
    }


def _alignment_metrics(
    predicted: np.ndarray,
    targets: np.ndarray,
) -> dict[str, Any]:
    _require(
        predicted.shape == targets.shape
        and predicted.shape[:2] == (
            QUERIES_PER_TRACK,
            len(DELAYS),
        ),
        f"Unexpected alignment arrays: {predicted.shape}/{targets.shape}",
    )
    pairs = ((0, 1), (0, 2), (1, 2))
    left = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    right = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    prediction_delta = predicted[:, left] - predicted[:, right]
    target_delta = targets[:, left] - targets[:, right]
    prediction_norm_sq = np.sum(prediction_delta**2, axis=-1)
    target_norm_sq = np.sum(target_delta**2, axis=-1)
    dot = np.sum(prediction_delta * target_delta, axis=-1)
    valid = (prediction_norm_sq > 1e-18) & (target_norm_sq > 1e-18)
    cosine = (
        dot[valid]
        / np.sqrt(
            prediction_norm_sq[valid] * target_norm_sq[valid]
        )
        if np.any(valid)
        else np.asarray([], dtype=np.float32)
    )
    prediction_pair_mse = float(np.mean(prediction_delta**2))
    target_pair_mse = float(np.mean(target_delta**2))
    alignment_mse = float(
        np.mean((prediction_delta - target_delta) ** 2)
    )
    prediction_centered = predicted - predicted.mean(
        axis=1, keepdims=True
    )
    target_centered = targets - targets.mean(axis=1, keepdims=True)
    centered_dot = np.sum(
        prediction_centered * target_centered,
        axis=tuple(range(1, predicted.ndim)),
    )
    prediction_centered_norm = np.sum(
        prediction_centered**2,
        axis=tuple(range(1, predicted.ndim)),
    )
    target_centered_norm = np.sum(
        target_centered**2,
        axis=tuple(range(1, predicted.ndim)),
    )
    centered_valid = (
        prediction_centered_norm > 1e-18
    ) & (target_centered_norm > 1e-18)
    centered_cosine = (
        centered_dot[centered_valid]
        / np.sqrt(
            prediction_centered_norm[centered_valid]
            * target_centered_norm[centered_valid]
        )
        if np.any(centered_valid)
        else np.asarray([], dtype=np.float32)
    )
    return {
        "queries": QUERIES_PER_TRACK,
        "delay_pairs_per_query": len(pairs),
        "target_pair_mse": target_pair_mse,
        "prediction_pair_mse": prediction_pair_mse,
        "prediction_to_target_pair_magnitude_ratio": float(
            np.sqrt(
                prediction_pair_mse / max(target_pair_mse, 1e-18)
            )
        ),
        "pair_delta_alignment_mse": alignment_mse,
        "pair_delta_alignment_mse_over_target_pair_mse": float(
            alignment_mse / max(target_pair_mse, 1e-18)
        ),
        "pair_direction_cosine_mean": (
            float(np.mean(cosine)) if cosine.size else 0.0
        ),
        "pair_direction_cosine_median": (
            float(np.median(cosine)) if cosine.size else 0.0
        ),
        "pair_direction_positive_fraction": float(
            np.mean(dot[valid] > 0) if np.any(valid) else 0.0
        ),
        "pair_direction_gain_mean": float(
            np.mean(dot[valid] / target_norm_sq[valid])
            if np.any(valid)
            else 0.0
        ),
        "centered_delay_pattern_cosine_mean": float(
            np.mean(centered_cosine)
            if centered_cosine.size
            else 0.0
        ),
        "centered_delay_pattern_cosine_median": float(
            np.median(centered_cosine)
            if centered_cosine.size
            else 0.0
        ),
        "target_centered_variance": float(
            np.mean(target_centered**2)
        ),
        "prediction_centered_variance": float(
            np.mean(prediction_centered**2)
        ),
    }


def score_domain_track(
    adapter: Any,
    assets: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    _require(
        len(assets) == QUERIES_PER_TRACK,
        "Domain track must contain 300 assets",
    )
    tracks = {str(asset["track"]) for asset in assets}
    source_delays = {int(asset["source_delay"]) for asset in assets}
    _require(
        len(tracks) == 1 and len(source_delays) == 1,
        "Domain track identity is mixed",
    )
    source_delay = next(iter(source_delays))
    input_pixels = np.concatenate(
        [asset["history_pixels"] for asset in assets],
        axis=0,
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
    ).reshape(
        QUERIES_PER_TRACK,
        len(DELAYS),
        len(FUTURE_HORIZONS),
        -1,
    )
    target_pixels = np.concatenate(
        [asset["true_future_pixels"] for asset in assets],
        axis=0,
    )
    encoded = np.asarray(
        adapter.encode_pixels(
            target_pixels.reshape(
                -1, *target_pixels.shape[-3:]
            ),
            batch_size=int(batch_size),
        ),
        dtype=np.float32,
    ).reshape(
        QUERIES_PER_TRACK,
        len(DELAYS),
        len(FUTURE_HORIZONS),
        -1,
    )
    losses = np.mean(
        (
            predicted[:, :, None, :, :]
            - encoded[:, None, :, :, :]
        )
        ** 2,
        axis=-1,
    )
    _require(
        losses.shape
        == (
            QUERIES_PER_TRACK,
            len(DELAYS),
            len(DELAYS),
            len(FUTURE_HORIZONS),
        ),
        f"Unexpected domain loss tensor: {losses.shape}",
    )
    trajectory_rows = _query_metrics(
        losses.mean(axis=-1),
        assets,
        horizon="trajectory",
    )
    horizon_rows = {
        str(horizon): _query_metrics(
            losses[:, :, :, index],
            assets,
            horizon=horizon,
        )
        for index, horizon in enumerate(FUTURE_HORIZONS)
    }
    alignment = {
        f"h{horizon}": _alignment_metrics(
            predicted[:, :, index],
            encoded[:, :, index],
        )
        for index, horizon in enumerate(FUTURE_HORIZONS)
    }
    alignment["trajectory"] = _alignment_metrics(
        predicted.reshape(QUERIES_PER_TRACK, len(DELAYS), -1),
        encoded.reshape(QUERIES_PER_TRACK, len(DELAYS), -1),
    )
    return {
        "track": next(iter(tracks)),
        "source_delay": source_delay,
        "trajectory": _breakdowns(
            trajectory_rows,
            source_delay=source_delay,
        ),
        "by_horizon": {
            horizon: _breakdowns(
                rows,
                source_delay=source_delay,
            )
            for horizon, rows in horizon_rows.items()
        },
        "latent_alignment": alignment,
        "loss_tensor_sha256": array_sha256(losses),
        "audit": {
            "queries": len(assets),
            "model_predictions": len(input_pixels),
            "target_encodings": (
                QUERIES_PER_TRACK
                * len(DELAYS)
                * len(FUTURE_HORIZONS)
            ),
            "trajectory_comparisons": (
                QUERIES_PER_TRACK * len(DELAYS) ** 2
            ),
            "horizon_loss_records": int(losses.size),
            "online_environment_calls": 0,
            "model_visible_fields": ["pixels", "action"],
            "privileged_fields_passed_to_adapter": [],
        },
    }


__all__ = [
    "RATE_METRICS",
    "load_domain_catalog",
    "load_domain_track_assets",
    "score_domain_track",
]
