#!/usr/bin/env python3
"""Audit whether RGB-visible History=3 identifies contact friction.

This is a data audit, not a model evaluation.  Classifiers are fitted only
on Training.  Development and Public Test are used only to confirm that the
frozen synthetic data contains a generalizable history signal and no useful
current-frame label shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import lance
import numpy as np
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
            "state",
            "action",
            "pair_id",
            "hidden_mode",
            "split",
        ]
    )
    episodes = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    steps = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    states = np.asarray(table["state"].to_pylist(), dtype=np.float64)
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
        frame_rows = rows[[0, 5, 10]]
        # state[:5] is agent_xy, block_xy, block_angle.  These quantities are
        # directly visible in the RGB frames.  Hidden velocity, contacts and
        # simulator-only physics fields are deliberately excluded.
        visible = states[frame_rows, :5]
        pair = conditions.setdefault(pair_id, {})
        if mode in pair:
            raise RuntimeError(f"Duplicate {pair_id}/{mode}")
        pair[mode] = {
            "x0": visible[0],
            "history": np.concatenate(
                [
                    visible[0],
                    visible[1] - visible[0],
                    visible[2] - visible[1],
                ]
            ),
            "actions": actions[rows[:15]].copy(),
        }

    if len(conditions) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairs in {expected_split}, got "
            f"{len(conditions)}"
        )
    x0_rows: list[np.ndarray] = []
    history_rows: list[np.ndarray] = []
    labels: list[int] = []
    paired_actions_exact = True
    for pair_id in sorted(conditions):
        pair = conditions[pair_id]
        if set(pair) != set(FRICTION_MODES):
            raise RuntimeError(f"Incomplete pair {pair_id}")
        paired_actions_exact &= np.array_equal(
            pair[FRICTION_MODES[0]]["actions"],
            pair[FRICTION_MODES[1]]["actions"],
        )
        for label, mode in enumerate(FRICTION_MODES):
            x0_rows.append(pair[mode]["x0"])
            history_rows.append(pair[mode]["history"])
            labels.append(label)
    return {
        "x0": np.stack(x0_rows),
        "history": np.stack(history_rows),
        "labels": np.asarray(labels, dtype=np.int64),
        "pair_count": len(conditions),
        "condition_count": len(labels),
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


def main() -> None:
    args = parse_args()
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
    for feature_name in ("x0", "history"):
        scaler = StandardScaler().fit(splits["train"][feature_name])
        transformed[feature_name] = {
            split: scaler.transform(values[feature_name])
            for split, values in splits.items()
        }
        scaler_receipts[feature_name] = {
            "fit_split": "Training",
            "dimension": int(scaler.mean_.size),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }

    x0_classifier = KNeighborsClassifier(
        n_neighbors=1,
        weights="distance",
    ).fit(transformed["x0"]["train"], labels["train"])
    x0_scores = _score(
        x0_classifier,
        features=transformed["x0"],
        labels=labels,
    )

    history_knn: dict[str, Any] = {}
    for neighbors in (1, 3):
        classifier = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights="distance",
        ).fit(transformed["history"]["train"], labels["train"])
        history_knn[str(neighbors)] = _score(
            classifier,
            features=transformed["history"],
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
        ).fit(splits["train"]["history"], labels["train"])
        extra_trees[str(minimum_leaf)] = _score(
            classifier,
            features={
                split: values["history"] for split, values in splits.items()
            },
            labels=labels,
        )

    x0_generalization = (
        x0_scores["loader_validation"] <= 0.55
        and x0_scores["validation"] <= 0.55
    )
    history_generalization = all(
        score == 1.0
        for family in (history_knn, extra_trees)
        for row in family.values()
        for score in row.values()
    )
    passed = bool(
        all(values["paired_actions_exact"] for values in splits.values())
        and x0_generalization
        and history_generalization
    )
    payload = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "release": {
            "release_id": release["release_id"],
            "config_path": str(release_path),
            "config_sha256": file_sha256(release_path),
            "manifest_sha256": release["data"]["manifest_sha256"],
        },
        "role": "data_identifiability_audit_not_model_evaluation",
        "feature_contract": {
            "source": "state[:5] at model-visible x0/x1/x2 rows",
            "rgb_visible_quantities": [
                "agent_x",
                "agent_y",
                "block_x",
                "block_y",
                "block_angle",
            ],
            "x0_only": "state_x0[:5]",
            "history": "concat(x0, x1-x0, x2-x1)",
            "given_history_actions": (
                "audited equal within each pair; not needed by the "
                "classifier because they only condition the visible response"
            ),
            "excluded": [
                "hidden friction label at feature boundary",
                "velocity",
                "contact count",
                "physics_state",
                "future x3",
                "pixels from future x3",
            ],
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
                "paired_actions_exact": values["paired_actions_exact"],
            }
            for split, values in splits.items()
        },
        "x0_only_knn_1": {
            "scores": x0_scores,
            "gate": {
                "maximum_development_or_public_accuracy": 0.55,
                "passed": x0_generalization,
            },
            "training_score_is_not_a_leakage_measure": (
                "1-NN memorizes each Training sample; only held-out "
                "Development and Public Test are used for this gate"
            ),
        },
        "history_knn_distance_weighted": history_knn,
        "history_extra_trees": extra_trees,
        "history_identifiability_gate": {
            "required_accuracy": 1.0,
            "passed": history_generalization,
        },
        "public_test_role": (
            "data_identifiability_audit_only_not_recipe_or_checkpoint_selection"
        ),
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
                "x0_only_scores": x0_scores,
                "history_knn": history_knn,
                "history_extra_trees": extra_trees,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
