#!/usr/bin/env python3
"""Audit contact-friction identifiability from released model inputs.

The audit deliberately uses only the RGB frames and actions exposed to a
benchmark model.  It reports RGB-only, action-only, and RGB-plus-action
decoders separately.  This is a frozen-data observability check, not a model
evaluation and not evidence that a world model has learned the capability.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any

import lance
import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    FRICTION_MODES,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


SPLIT_NAMES = ("train", "loader_validation", "validation")
MODEL_FRAME_STEPS = (0, 5, 10)
# Renderer defaults, registered in stable_worldmodel.envs.pusht.env.PushT.
RENDER_COLORS = np.asarray(
    [
        [65.0, 105.0, 225.0],   # RoyalBlue pusher
        [119.0, 136.0, 153.0],  # LightSlateGray object
        [144.0, 238.0, 144.0],  # LightGreen goal; used to disambiguate masks
    ],
    dtype=np.float64,
)
MAXIMUM_COLOR_DISTANCE = 70.0
MINIMUM_DESCRIPTIVE_HELDOUT_ACCURACY = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _decode_rgb(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _mask_geometry(image: np.ndarray, color_index: int) -> np.ndarray:
    values = image.astype(np.float64)
    distance = np.linalg.norm(
        values[:, :, None, :] - RENDER_COLORS[None, None, :, :],
        axis=-1,
    )
    nearest = np.argmin(distance, axis=-1)
    mask = (
        (nearest == color_index)
        & (distance[:, :, color_index] <= MAXIMUM_COLOR_DISTANCE)
    )
    row, column = np.nonzero(mask)
    if row.size < 20:
        raise RuntimeError(
            f"RGB color mask {color_index} has only {row.size} pixels"
        )
    height, width = image.shape[:2]
    x = column.astype(np.float64) / width
    y = row.astype(np.float64) / height
    center_x = float(x.mean())
    center_y = float(y.mean())
    dx = x - center_x
    dy = y - center_y
    return np.asarray(
        [
            center_x,
            center_y,
            row.size / (height * width),
            np.mean(dx * dx),
            np.mean(dx * dy),
            np.mean(dy * dy),
        ],
        dtype=np.float64,
    )


def _visible_geometry(value: bytes) -> np.ndarray:
    image = _decode_rgb(value)
    return np.concatenate(
        [
            _mask_geometry(image, 0),
            _mask_geometry(image, 1),
        ]
    )


def _read_split(
    path: Path,
    *,
    expected_split: str,
    expected_pairs: int,
) -> dict[str, Any]:
    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "pair_id",
            "hidden_mode",
            "split",
        ]
    )
    episodes = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    steps = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    pixels = table["pixels"].to_pylist()
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    splits = table["split"].to_pylist()

    conditions: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(steps[rows])]
        if not np.array_equal(steps[rows], np.arange(20)):
            raise RuntimeError(f"Incomplete episode {episode}")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        if (
            len(pair_values) != 1
            or len(mode_values) != 1
            or split_values != {expected_split}
        ):
            raise RuntimeError(f"Episode metadata changed in {episode}")
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in FRICTION_MODES:
            raise RuntimeError(f"Unexpected mode {mode!r}")
        frame_rows = [
            rows[np.flatnonzero(steps[rows] == step)[0]]
            for step in MODEL_FRAME_STEPS
        ]
        geometry = [_visible_geometry(pixels[index]) for index in frame_rows]
        pair = conditions.setdefault(pair_id, {})
        if mode in pair:
            raise RuntimeError(f"Duplicate {pair_id}/{mode}")
        pair[mode] = {
            "geometry": geometry,
            # This is exactly the four 5-step action blocks materialized as
            # four 10-D model action tokens by the reference loader.
            "action_blocks": actions[rows].reshape(4, 10),
        }

    if len(conditions) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairs in {expected_split}, got "
            f"{len(conditions)}"
        )
    x0_rows: list[np.ndarray] = []
    history_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    model_input_rows: list[np.ndarray] = []
    labels: list[int] = []
    paired_x0_geometry_exact = True
    paired_actions_exact = True
    for pair_id in sorted(conditions):
        pair = conditions[pair_id]
        if set(pair) != set(FRICTION_MODES):
            raise RuntimeError(f"Incomplete pair {pair_id}")
        paired_x0_geometry_exact &= np.array_equal(
            pair[FRICTION_MODES[0]]["geometry"][0],
            pair[FRICTION_MODES[1]]["geometry"][0],
        )
        paired_actions_exact &= np.array_equal(
            pair[FRICTION_MODES[0]]["action_blocks"],
            pair[FRICTION_MODES[1]]["action_blocks"],
        )
        for label, mode in enumerate(FRICTION_MODES):
            x0, x1, x2 = pair[mode]["geometry"]
            action = pair[mode]["action_blocks"].reshape(-1)
            history = np.concatenate([x0, x1 - x0, x2 - x1])
            x0_rows.append(x0)
            history_rows.append(history)
            action_rows.append(action)
            model_input_rows.append(np.concatenate([history, action]))
            labels.append(label)
    return {
        "x0": np.stack(x0_rows),
        "history_rgb": np.stack(history_rows),
        "action_only": np.stack(action_rows),
        "model_input": np.stack(model_input_rows),
        "labels": np.asarray(labels, dtype=np.int64),
        "pair_count": len(conditions),
        "condition_count": len(labels),
        "paired_x0_geometry_exact": bool(paired_x0_geometry_exact),
        "paired_actions_exact": bool(paired_actions_exact),
    }


def _score(
    classifier: Any,
    *,
    features: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, float]:
    return {
        split: float(classifier.score(features[split], labels[split]))
        for split in SPLIT_NAMES
    }


def _decoder_scores(
    feature_name: str,
    *,
    raw: dict[str, dict[str, Any]],
    transformed: dict[str, dict[str, np.ndarray]],
    labels: dict[str, np.ndarray],
) -> dict[str, Any]:
    knn: dict[str, Any] = {}
    for neighbors in (1, 3):
        classifier = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights="distance",
        ).fit(transformed[feature_name]["train"], labels["train"])
        knn[str(neighbors)] = _score(
            classifier,
            features=transformed[feature_name],
            labels=labels,
        )
    extra_trees: dict[str, Any] = {}
    for minimum_leaf in (1, 2, 5, 10):
        classifier = ExtraTreesClassifier(
            n_estimators=256,
            min_samples_leaf=minimum_leaf,
            max_features="sqrt",
            random_state=20260803,
            n_jobs=-1,
        ).fit(raw["train"][feature_name], labels["train"])
        extra_trees[str(minimum_leaf)] = _score(
            classifier,
            features={
                split: values[feature_name]
                for split, values in raw.items()
            },
            labels=labels,
        )
    heldout = [
        float(row[split])
        for family in (knn, extra_trees)
        for row in family.values()
        for split in ("loader_validation", "validation")
    ]
    return {
        "knn_distance_weighted": knn,
        "extra_trees": extra_trees,
        "minimum_heldout_accuracy": min(heldout),
        "maximum_heldout_accuracy": max(heldout),
        "all_heldout_at_or_above_descriptive_threshold": bool(
            all(
                score >= MINIMUM_DESCRIPTIVE_HELDOUT_ACCURACY
                for score in heldout
            )
        ),
        "all_scores_are_perfect": bool(all(score == 1.0 for score in heldout)),
    }


def main() -> None:
    args = parse_args()
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"],
        repo_root=ROOT,
    )
    splits = {
        split: _read_split(
            data_root / release["data"]["lance_tables"][split],
            expected_split=split,
            expected_pairs=int(release["data"]["pair_counts"][split]),
        )
        for split in SPLIT_NAMES
    }
    labels = {split: values["labels"] for split, values in splits.items()}
    transformed: dict[str, dict[str, np.ndarray]] = {}
    scaler_receipts: dict[str, Any] = {}
    for name in ("x0", "history_rgb", "action_only", "model_input"):
        scaler = StandardScaler().fit(splits["train"][name])
        transformed[name] = {
            split: scaler.transform(values[name])
            for split, values in splits.items()
        }
        scaler_receipts[name] = {
            "fit_split": "Training",
            "dimension": int(scaler.mean_.size),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    x0_model = KNeighborsClassifier(
        n_neighbors=1,
        weights="distance",
    ).fit(transformed["x0"]["train"], labels["train"])
    x0_scores = _score(
        x0_model,
        features=transformed["x0"],
        labels=labels,
    )
    rgb_decoders = _decoder_scores(
        "history_rgb",
        raw=splits,
        transformed=transformed,
        labels=labels,
    )
    action_decoders = _decoder_scores(
        "action_only",
        raw=splits,
        transformed=transformed,
        labels=labels,
    )
    model_input_decoders = _decoder_scores(
        "model_input",
        raw=splits,
        transformed=transformed,
        labels=labels,
    )

    x0_gate = (
        x0_scores["loader_validation"] <= 0.55
        and x0_scores["validation"] <= 0.55
    )
    action_shortcut_gate = bool(
        action_decoders["maximum_heldout_accuracy"] <= 0.55
    )
    history_signal_gate = bool(
        rgb_decoders[
            "all_heldout_at_or_above_descriptive_threshold"
        ]
        and model_input_decoders[
            "all_heldout_at_or_above_descriptive_threshold"
        ]
    )
    passed = bool(
        all(
            values["paired_x0_geometry_exact"]
            for values in splits.values()
        )
        and all(values["paired_actions_exact"] for values in splits.values())
        and x0_gate
        and action_shortcut_gate
        and history_signal_gate
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "role": (
            "released_model_input_data_identifiability_audit_not_model_"
            "evaluation"
        ),
        "release": {
            "release_id": release["release_id"],
            "config_path": str(release_path),
            "manifest_sha256": release["data"]["manifest_sha256"],
            "artifact_tree_sha256": release["data"]["artifact_tree"][
                "sha256"
            ],
        },
        "feature_contract": {
            "input_columns": ["pixels", "action"],
            "model_frame_steps": list(MODEL_FRAME_STEPS),
            "known_renderer_colors_rgb": {
                "pusher_royal_blue": RENDER_COLORS[0].astype(int).tolist(),
                "object_light_slate_gray": (
                    RENDER_COLORS[1].astype(int).tolist()
                ),
                "goal_light_green_disambiguation_only": (
                    RENDER_COLORS[2].astype(int).tolist()
                ),
            },
            "maximum_jpeg_color_distance": MAXIMUM_COLOR_DISTANCE,
            "per_body_geometry": (
                "normalized centroid, area, and 2D central second moments"
            ),
            "x0_only": "RGB geometry at x0",
            "history_rgb": "concat(RGB geometry x0, x1-x0, x2-x1)",
            "action_only": (
                "four released 5-step action blocks flattened from the same "
                "40 scalar values received by the reference model"
            ),
            "model_input": "concat(history_rgb, action_only)",
            "feature_boundary_excludes": [
                "state",
                "physics_state",
                "proprio",
                "n_contacts",
                "hidden_contact_friction",
                "hidden_mode",
                "future x3 pixels",
            ],
            "hidden_mode_used_only_as_supervised_target_after_feature_extraction": True,
        },
        "fit_contract": {
            "classifiers_and_standardization_fit_split": "Training",
            "development_used_for_fit_or_threshold_selection": False,
            "public_test_used_for_fit_or_threshold_selection": False,
            "model_or_checkpoint_loaded": False,
            "scalers": scaler_receipts,
        },
        "counts": {
            split: {
                "pair_count": values["pair_count"],
                "condition_count": values["condition_count"],
                "paired_x0_geometry_exact": values[
                    "paired_x0_geometry_exact"
                ],
                "paired_actions_exact": values["paired_actions_exact"],
            }
            for split, values in splits.items()
        },
        "x0_only_knn_1": {
            "scores": x0_scores,
            "maximum_development_or_public_accuracy": 0.55,
            "passed": x0_gate,
        },
        "rgb_only_history_decoders": rgb_decoders,
        "action_only_decoders": action_decoders,
        "released_model_input_decoders": model_input_decoders,
        "history_identifiability_gate": {
            "descriptive_minimum_heldout_accuracy": (
                MINIMUM_DESCRIPTIVE_HELDOUT_ACCURACY
            ),
            "threshold_role": (
                "data-observability description only; not a benchmark model "
                "prediction gate and not used to select a checkpoint"
            ),
            "rgb_only_simple_decoders_are_perfect": rgb_decoders[
                "all_scores_are_perfect"
            ],
            "released_model_input_simple_decoders_are_perfect": (
                model_input_decoders["all_scores_are_perfect"]
            ),
            "history_signal_present": history_signal_gate,
            "action_only_shortcut_absent": action_shortcut_gate,
            "passed": history_signal_gate and action_shortcut_gate,
        },
        "public_test_role": (
            "data_identifiability_audit_only_not_recipe_or_checkpoint_selection"
        ),
        "model_or_checkpoint_loaded": False,
        "passed": passed,
    }
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
                "x0_only": x0_scores,
                "rgb_only_history_decoders": rgb_decoders,
                "action_only_decoders": action_decoders,
                "released_model_input_decoders": model_input_decoders,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
