#!/usr/bin/env python3
"""Diagnose strict motion-damping learnability without opening Public Test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMMotionDampingAdapter,
    StableWorldModelPLDMMotionDampingAdapter,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    _read_lance_pairs,
    file_sha256,
    load_motion_damping_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_score import (  # noqa: E402
    motion_damping_prediction_metrics,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


ADAPTERS = {
    "lewm": StableWorldModelLeWMMotionDampingAdapter,
    "pldm": StableWorldModelPLDMMotionDampingAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="NAME:ADAPTER=PATH where ADAPTER is lewm or pldm",
    )
    parser.add_argument("--train-pairs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _parse_checkpoints(values: list[str]) -> dict[str, tuple[str, Path]]:
    rows: dict[str, tuple[str, Path]] = {}
    for value in values:
        if "=" not in value or ":" not in value.split("=", 1)[0]:
            raise ValueError("--checkpoint must use NAME:ADAPTER=PATH")
        identity, raw_path = value.split("=", 1)
        name, adapter = identity.split(":", 1)
        path = Path(raw_path).expanduser().resolve()
        if name in rows or adapter not in ADAPTERS or not path.is_file():
            raise ValueError(f"Invalid checkpoint specification {value!r}")
        rows[name] = (adapter, path)
    if not rows:
        raise ValueError("At least one checkpoint is required")
    return rows


def _visible_history_features(arrays) -> tuple[np.ndarray, np.ndarray]:
    # `state` contains rendered agent/block pose, not hidden damping or
    # velocity. Differences across x0/x1/x2 are therefore recoverable from
    # RGB and form an audit-only upper bound on history identifiability.
    faster = arrays.faster_decay_states[:, :3].reshape(arrays.pair_count, -1)
    no_extra = arrays.no_extra_decay_states[:, :3].reshape(
        arrays.pair_count, -1
    )
    features = np.concatenate([faster, no_extra]).astype(np.float64)
    labels = np.concatenate(
        [
            np.zeros(arrays.pair_count, dtype=np.int64),
            np.ones(arrays.pair_count, dtype=np.int64),
        ]
    )
    return features, labels


def _visible_motion_ratios(arrays) -> tuple[np.ndarray, np.ndarray]:
    def ratio(states: np.ndarray) -> np.ndarray:
        block_xy = states[:, :3, 2:4]
        first = np.linalg.norm(block_xy[:, 1] - block_xy[:, 0], axis=-1)
        second = np.linalg.norm(block_xy[:, 2] - block_xy[:, 1], axis=-1)
        return second / np.maximum(first, 1e-12)

    return ratio(arrays.faster_decay_states), ratio(
        arrays.no_extra_decay_states
    )


def _model_metrics(
    *,
    adapter,
    arrays,
    batch_size: int,
) -> dict[str, Any]:
    histories = np.concatenate(
        [
            arrays.faster_decay_pixels[:, :3],
            arrays.no_extra_decay_pixels[:, :3],
        ]
    )
    actions = np.concatenate(
        [arrays.raw_action_blocks[:, :3], arrays.raw_action_blocks[:, :3]]
    )
    futures = np.concatenate(
        [arrays.faster_decay_pixels[:, 3], arrays.no_extra_decay_pixels[:, 3]]
    )
    predictions = adapter.rollout_latents(
        histories, actions, batch_size=batch_size
    )[:, 0]
    targets = adapter.encode_pixels(futures, batch_size=batch_size)
    count = arrays.pair_count
    metrics, _ = motion_damping_prediction_metrics(
        pair_ids=arrays.pair_ids,
        predicted_faster_decay=predictions[:count],
        predicted_no_extra_decay=predictions[count:],
        target_faster_decay=targets[:count],
        target_no_extra_decay=targets[count:],
    )
    target_pair_mse = np.square(
        targets[:count] - targets[count:]
    ).mean(axis=-1)
    unrelated_target_mse = np.square(
        targets[:count] - np.roll(targets[count:], 1, axis=0)
    ).mean(axis=-1)
    prediction_pair_mse = np.square(
        predictions[:count] - predictions[count:]
    ).mean(axis=-1)
    metrics["target_pair_mse_mean"] = float(target_pair_mse.mean())
    metrics["unrelated_target_mse_mean"] = float(
        unrelated_target_mse.mean()
    )
    metrics["target_pair_to_unrelated_mse_ratio"] = float(
        target_pair_mse.mean() / max(unrelated_target_mse.mean(), 1e-12)
    )
    metrics["prediction_pair_mse_mean"] = float(prediction_pair_mse.mean())
    metrics["prediction_to_target_pair_mse_ratio"] = float(
        prediction_pair_mse.mean() / max(target_pair_mse.mean(), 1e-12)
    )
    return metrics


def main() -> None:
    args = parse_args()
    checkpoints = _parse_checkpoints(args.checkpoint)
    release = load_motion_damping_icl_release(args.release_config)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    train_count = int(release["data"]["pair_counts"]["train"])
    development_count = int(
        release["data"]["pair_counts"]["loader_validation"]
    )
    if not 0 < args.train_pairs <= train_count:
        raise ValueError("--train-pairs is outside the Training split")
    train = _read_lance_pairs(
        data_root / release["data"]["lance_tables"]["train"],
        expected_pairs=train_count,
        expected_split="train",
    )
    development = _read_lance_pairs(
        data_root / release["data"]["lance_tables"]["loader_validation"],
        expected_pairs=development_count,
        expected_split="loader_validation",
    )
    selected = np.arange(args.train_pairs)
    # Restrict only model fitting diagnostics; the visible-state classifier
    # uses every Training pair and the independent Development split.
    train_subset = type(train)(
        pair_ids=tuple(train.pair_ids[index] for index in selected),
        faster_decay_pixels=train.faster_decay_pixels[selected],
        no_extra_decay_pixels=train.no_extra_decay_pixels[selected],
        raw_action_blocks=train.raw_action_blocks[selected],
        faster_decay_states=train.faster_decay_states[selected],
        no_extra_decay_states=train.no_extra_decay_states[selected],
        faster_decay_physics_states=train.faster_decay_physics_states[selected],
        no_extra_decay_physics_states=train.no_extra_decay_physics_states[selected],
    )
    train_features, train_labels = _visible_history_features(train)
    train_fast_ratio, train_no_extra_ratio = _visible_motion_ratios(train)
    development_features, development_labels = _visible_history_features(
        development
    )
    development_fast_ratio, development_no_extra_ratio = (
        _visible_motion_ratios(development)
    )
    del train
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=5000, random_state=14321),
    )
    classifier.fit(train_features, train_labels)
    classifier_audit = {
        "feature_source": "rendered_agent_and_block_pose_across_x0_x1_x2",
        "hidden_velocity_or_damping_used": False,
        "training_examples": int(train_labels.size),
        "development_examples": int(development_labels.size),
        "training_accuracy": float(
            classifier.score(train_features, train_labels)
        ),
        "development_accuracy": float(
            classifier.score(development_features, development_labels)
        ),
    }
    ratio_threshold = 0.5 * (
        float(train_fast_ratio.max()) + float(train_no_extra_ratio.min())
    )
    train_ratio_predictions = np.concatenate(
        [
            train_fast_ratio >= ratio_threshold,
            train_no_extra_ratio >= ratio_threshold,
        ]
    )
    development_ratio_predictions = np.concatenate(
        [
            development_fast_ratio >= ratio_threshold,
            development_no_extra_ratio >= ratio_threshold,
        ]
    )
    motion_ratio_oracle = {
        "feature": "block_displacement_x1_to_x2_divided_by_x0_to_x1",
        "recoverable_from_rgb_history": True,
        "hidden_velocity_or_damping_used": False,
        "threshold_frozen_from_training": ratio_threshold,
        "faster_decay_training_range": [
            float(train_fast_ratio.min()),
            float(train_fast_ratio.max()),
        ],
        "no_extra_decay_training_range": [
            float(train_no_extra_ratio.min()),
            float(train_no_extra_ratio.max()),
        ],
        "training_accuracy": float(
            (train_ratio_predictions == train_labels).mean()
        ),
        "development_accuracy": float(
            (development_ratio_predictions == development_labels).mean()
        ),
    }
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    model_rows = {}
    for name, (adapter_name, path) in checkpoints.items():
        adapter = ADAPTERS[adapter_name].from_checkpoint(
            path,
            action_mean=normalization["mean"],
            action_std=normalization["std_population"],
            repo_root=ROOT,
            stablewm_repo=runtime["repo"],
            stablewm_ref=runtime["expected_ref"],
            device=args.device,
        )
        model_rows[name] = {
            "adapter": adapter_name,
            "checkpoint": str(path),
            "checkpoint_sha256": file_sha256(path),
            "training_subset": _model_metrics(
                adapter=adapter,
                arrays=train_subset,
                batch_size=args.batch_size,
            ),
            "development": _model_metrics(
                adapter=adapter,
                arrays=development,
                batch_size=args.batch_size,
            ),
        }
        del adapter
        try:
            import torch

            torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
    payload = {
        "schema_version": 1,
        "status": "completed_training_and_development_only",
        "diagnostic_only": True,
        "public_test_opened": False,
        "release_id": release["release_id"],
        "manifest_sha256": release["data"]["manifest_sha256"],
        "training_model_subset_pairs": args.train_pairs,
        "development_pairs": development_count,
        "visible_history_classifier": classifier_audit,
        "visible_motion_ratio_oracle": motion_ratio_oracle,
        "models": model_rows,
    }
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
