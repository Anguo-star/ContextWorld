#!/usr/bin/env python3
"""Test whether frozen Stable-WorldModel latents retain a history signal.

The decoder is fitted on Training and scored only on Development.  Public Test
is never opened.  This separates data observability from the predictor's
ability to learn the same rule from the frozen image representation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (  # noqa: E402
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelLeWMMotionDampingAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (  # noqa: E402
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    _read_lance_pairs as read_contact_pairs,
    load_contact_friction_icl_release,
)
from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    _read_lance_pairs as read_damping_pairs,
    load_motion_damping_icl_release,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capability",
        choices=("contact_friction", "motion_damping"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def encode_split(
    adapter: Any,
    arrays: Any,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    rows = []
    x0_rows = []
    for pixels in (arrays.low_pixels, arrays.high_pixels):
        z0, z1, z2 = (
            adapter.encode_pixels(pixels[:, index], batch_size=batch_size)
            for index in range(3)
        )
        x0_rows.append(z0)
        rows.append(np.concatenate([z0, z1 - z0, z2 - z1], axis=-1))
    count = arrays.pair_count
    return {
        "x0": np.concatenate(x0_rows),
        "history": np.concatenate(rows),
        "labels": np.concatenate(
            [
                np.zeros(count, dtype=np.int64),
                np.ones(count, dtype=np.int64),
            ]
        ),
    }


def scores(
    train: dict[str, np.ndarray], development: dict[str, np.ndarray]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for feature_name in ("x0", "history"):
        x_train = train[feature_name]
        x_dev = development[feature_name]
        y_train = train["labels"]
        y_dev = development["labels"]
        classifiers = {
            "logistic_l2": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    random_state=20260803,
                ),
            ),
            "knn_1": make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=1),
            ),
            "extra_trees_leaf_1": ExtraTreesClassifier(
                n_estimators=256,
                min_samples_leaf=1,
                max_features="sqrt",
                random_state=20260803,
                n_jobs=-1,
            ),
            "extra_trees_leaf_10": ExtraTreesClassifier(
                n_estimators=256,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=20260803,
                n_jobs=-1,
            ),
        }
        result[feature_name] = {}
        for name, classifier in classifiers.items():
            classifier.fit(x_train, y_train)
            result[feature_name][name] = {
                "training_accuracy": float(classifier.score(x_train, y_train)),
                "development_accuracy": float(classifier.score(x_dev, y_dev)),
            }
    return result


def history_tree_learning_curve(
    train: dict[str, np.ndarray], development: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Measure held-out decoding as paired Training coverage increases."""

    total_pairs = int(train["labels"].size // 2)
    pair_counts = [
        value
        for value in (128, 256, 512, 1024, 2048, total_pairs)
        if value <= total_pairs
    ]
    pair_counts = list(dict.fromkeys(pair_counts))
    rows = {}
    for pair_count in pair_counts:
        indices = np.concatenate(
            [
                np.arange(pair_count),
                total_pairs + np.arange(pair_count),
            ]
        )
        tree = ExtraTreesClassifier(
            n_estimators=256,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=20260803,
            n_jobs=-1,
        ).fit(train["history"][indices], train["labels"][indices])
        knn = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=3, weights="distance"),
        ).fit(train["history"][indices], train["labels"][indices])
        rows[str(pair_count)] = {
            "training_conditions": int(indices.size),
            "extra_trees_leaf_10": {
                "training_accuracy": float(
                    tree.score(
                        train["history"][indices],
                        train["labels"][indices],
                    )
                ),
                "development_accuracy": float(
                    tree.score(
                        development["history"], development["labels"]
                    )
                ),
            },
            "knn_3_distance_weighted": {
                "training_accuracy": float(
                    knn.score(
                        train["history"][indices],
                        train["labels"][indices],
                    )
                ),
                "development_accuracy": float(
                    knn.score(
                        development["history"], development["labels"]
                    )
                ),
            },
        }
    return {
        "classifiers": [
            "extra_trees_256_min_samples_leaf_10",
            "standardized_knn_3_distance_weighted",
        ],
        "paired_prefix_order": True,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.capability == "contact_friction":
        release_path = args.release_config or DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG
        release = load_contact_friction_icl_release(release_path)
        reader = read_contact_pairs
        adapter_type = StableWorldModelLeWMContactFrictionAdapter
    else:
        release_path = args.release_config or DEFAULT_MOTION_DAMPING_RELEASE_CONFIG
        release = load_motion_damping_icl_release(release_path)
        reader = read_damping_pairs
        adapter_type = StableWorldModelLeWMMotionDampingAdapter
    root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=ROOT
    )
    arrays = {}
    for split in ("train", "loader_validation"):
        arrays[split] = reader(
            root / release["data"]["lance_tables"][split],
            expected_pairs=int(release["data"]["pair_counts"][split]),
            expected_split=split,
        )
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    adapter = adapter_type.from_checkpoint(
        args.checkpoint.expanduser().resolve(),
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=args.device,
    )
    train = encode_split(adapter, arrays["train"], batch_size=args.batch_size)
    development = encode_split(
        adapter, arrays["loader_validation"], batch_size=args.batch_size
    )
    payload = {
        "schema_version": 1,
        "status": "completed_training_and_development_only",
        "public_test_opened": False,
        "capability": args.capability,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "feature": "concat(z0,z1-z0,z2-z1) from frozen encoder/projector",
        "condition_counts": {
            "train": int(train["labels"].size),
            "loader_validation": int(development["labels"].size),
        },
        "scores": scores(train, development),
        "history_tree_learning_curve": history_tree_learning_curve(
            train, development
        ),
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
