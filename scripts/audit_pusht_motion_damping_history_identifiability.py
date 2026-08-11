#!/usr/bin/env python3
"""Audit whether RGB-visible History=3 uniquely reveals motion damping."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import lance
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    DAMPING_MODES,
    load_motion_damping_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


# PushT renders the block with pygame ``LightSlateGray`` (119, 136, 153).
# Its polygon renderer applies the fixed 1.2 light-color transform before the
# image is resized and JPEG encoded, giving the reference below.  Neither the
# reference nor the radius is fitted to a split or a hidden-mode label.
BLOCK_RENDER_RGB = np.asarray([142.8, 163.2, 183.6], dtype=np.float64)
BLOCK_COLOR_RADIUS = 35.0


def _block_centroid_from_rgb(value: bytes) -> tuple[np.ndarray, int]:
    """Return the block centroid using only a fixed RGB color mask."""

    with Image.open(BytesIO(value)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64)
    distance = np.linalg.norm(rgb - BLOCK_RENDER_RGB, axis=-1)
    weights = np.clip(BLOCK_COLOR_RADIUS - distance, 0.0, None)
    selected = int(np.count_nonzero(weights))
    total = float(weights.sum())
    if selected == 0 or total <= 0.0:
        raise RuntimeError("Fixed block-color mask selected no RGB pixels")
    y, x = np.indices(weights.shape, dtype=np.float64)
    return np.asarray(
        [(weights * x).sum() / total, (weights * y).sum() / total],
        dtype=np.float64,
    ), selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _split_ratios(
    path: Path,
    *,
    expected_split: str,
    expected_pairs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "pair_id",
            "hidden_mode",
            "split",
        ]
    )
    episodes = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    steps = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    pixels = table["pixels"].to_pylist()
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    splits = table["split"].to_pylist()
    ratios = {mode: [] for mode in DAMPING_MODES}
    selected_pixel_counts: list[int] = []
    seen_pairs = set()
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(steps[rows])]
        if not np.array_equal(steps[rows], np.arange(20)):
            raise RuntimeError(f"Incomplete episode {episode}")
        frame_rows = rows[[0, 5, 10]]
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        pair_values = {str(pair_ids[index]) for index in rows}
        if (
            len(mode_values) != 1
            or split_values != {expected_split}
            or len(pair_values) != 1
        ):
            raise RuntimeError(f"Episode metadata changed in {episode}")
        mode = mode_values.pop()
        pair_id = pair_values.pop()
        if mode not in ratios:
            raise RuntimeError(f"Unexpected mode {mode}")
        centroids = []
        for index in frame_rows:
            centroid, selected = _block_centroid_from_rgb(pixels[index])
            centroids.append(centroid)
            selected_pixel_counts.append(selected)
        block_xy = np.stack(centroids)
        first = float(np.linalg.norm(block_xy[1] - block_xy[0]))
        second = float(np.linalg.norm(block_xy[2] - block_xy[1]))
        if first <= 0.0:
            raise RuntimeError(f"Zero first displacement in {pair_id}/{mode}")
        ratios[mode].append(second / first)
        seen_pairs.add(pair_id)
    if len(seen_pairs) != expected_pairs:
        raise RuntimeError(f"Unexpected pair count in {expected_split}")
    return (
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in ratios.items()
        },
        np.asarray(selected_pixel_counts, dtype=np.int64),
    )


def _summary(
    ratios: dict[str, np.ndarray], *, threshold: float
) -> dict[str, Any]:
    faster = ratios["faster_decay"]
    no_extra = ratios["no_extra_decay"]
    correct = int((faster < threshold).sum() + (no_extra >= threshold).sum())
    count = int(faster.size + no_extra.size)
    return {
        "condition_count": count,
        "faster_decay": {
            "count": int(faster.size),
            "minimum": float(faster.min()),
            "mean": float(faster.mean()),
            "maximum": float(faster.max()),
        },
        "no_extra_decay": {
            "count": int(no_extra.size),
            "minimum": float(no_extra.min()),
            "mean": float(no_extra.mean()),
            "maximum": float(no_extra.max()),
        },
        "correct_count": correct,
        "accuracy": correct / count,
    }


def main() -> None:
    args = parse_args()
    release_path = args.release_config.expanduser().resolve()
    release = load_motion_damping_icl_release(release_path)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    split_names = ("train", "loader_validation", "validation")
    split_outputs = {
        split: _split_ratios(
            data_root / release["data"]["lance_tables"][split],
            expected_split=split,
            expected_pairs=int(release["data"]["pair_counts"][split]),
        )
        for split in split_names
    }
    ratios = {split: value[0] for split, value in split_outputs.items()}
    selected_pixel_counts = {
        split: value[1] for split, value in split_outputs.items()
    }
    train = ratios["train"]
    faster_upper = float(train["faster_decay"].max())
    no_extra_lower = float(train["no_extra_decay"].min())
    if not faster_upper < no_extra_lower:
        raise RuntimeError("Training motion-ratio classes overlap")
    threshold = 0.5 * (faster_upper + no_extra_lower)
    summaries = {
        split: _summary(values, threshold=threshold)
        for split, values in ratios.items()
    }
    passed = all(value["accuracy"] == 1.0 for value in summaries.values())
    payload = {
        "schema_version": 2,
        "status": "passed" if passed else "failed",
        "release": {
            "release_id": release["release_id"],
            "config_path": str(release_path),
            "manifest_sha256": release["data"]["manifest_sha256"],
        },
        "feature": {
            "name": "rgb_only_block_motion_decay_ratio",
            "formula": (
                "norm(rgb_centroid_x2-rgb_centroid_x1)/"
                "norm(rgb_centroid_x1-rgb_centroid_x0)"
            ),
            "history_frames": ["x0", "x1", "x2"],
            "input_feature_columns": ["pixels"],
            "scoring_label_column": "hidden_mode",
            "forbidden_input_columns": [
                "state",
                "physics_state",
                "hidden_motion_damping",
                "hidden_mode",
            ],
            "hidden_mode_used_as_input_feature": False,
            "state_or_physics_used_as_input_feature": False,
            "segmentation": {
                "method": "fixed_render_color_distance_weighted_centroid",
                "block_render_rgb": BLOCK_RENDER_RGB.tolist(),
                "color_radius": BLOCK_COLOR_RADIUS,
                "parameters_fitted_from_training_or_labels": False,
                "selected_pixels_per_frame": {
                    split: {
                        "minimum": int(values.min()),
                        "mean": float(values.mean()),
                        "maximum": int(values.max()),
                    }
                    for split, values in selected_pixel_counts.items()
                },
            },
        },
        "threshold": {
            "selection_split": "Training",
            "faster_decay_training_upper_bound": faster_upper,
            "no_extra_decay_training_lower_bound": no_extra_lower,
            "frozen_value": threshold,
        },
        "splits": summaries,
        "public_test_role": (
            "data_identifiability_audit_only_not_recipe_or_checkpoint_selection"
        ),
        "model_or_checkpoint_loaded": False,
        "passed": passed,
    }
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "threshold": threshold,
                "accuracies": {
                    key: value["accuracy"] for key, value in summaries.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
