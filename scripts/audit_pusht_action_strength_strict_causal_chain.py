#!/usr/bin/env python3
"""Audit natural x0->x1->x2->x3 continuity and legacy input identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import lance
import numpy as np


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / "stable-worldmodel"
for source_root in (CONTEXTWORLD_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation.pusht_replay_matched_hidden_actuation import (  # noqa: E402
    ReplayMatchedHiddenActuationTemplate,
    simulate_replay_matched_hidden_actuation,
    validate_replay_matched_pair,
)
from stable_worldmodel.data.formats.lance import _encode_frame  # noqa: E402


DEFAULT_ARTIFACT_ROOT = Path(
    "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
    "context_world/synthesis"
)
MODEL_FRAME_ROWS = (0, 5, 10, 15)
RAW_ROWS_PER_EPISODE = 20
MODES = ("low_gain", "high_gain")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _template(raw: dict[str, Any]) -> ReplayMatchedHiddenActuationTemplate:
    names = {field.name for field in fields(ReplayMatchedHiddenActuationTemplate)}
    return ReplayMatchedHiddenActuationTemplate(
        **{name: raw[name] for name in names}
    )


def _selected_indices(pair_count: int, limit: int | None) -> list[int]:
    if limit is None or limit >= pair_count:
        return list(range(pair_count))
    if limit <= 0:
        raise ValueError("--pairs-per-dataset must be positive")
    return sorted(
        set(
            map(
                int,
                np.linspace(0, pair_count - 1, num=limit, dtype=np.int64),
            )
        )
    )


def _dataset_audit(
    *,
    label: str,
    root: Path,
    split: str,
    pairs_per_dataset: int | None,
    jpeg_quality: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((root / "manifest.json").read_text())
    split_report = manifest["splits"][split]
    pairs = split_report["pairs"]
    indices = _selected_indices(len(pairs), pairs_per_dataset)
    dataset = lance.dataset(root / f"{split}.lance")
    legacy_digest = hashlib.sha256()
    strict_digest = hashlib.sha256()
    pixel_mismatches: list[dict[str, Any]] = []
    action_mismatches: list[dict[str, Any]] = []
    causal_audits: list[dict[str, Any]] = []

    for pair_index in indices:
        template = _template(pairs[pair_index]["template"])
        rollouts = {
            mode: simulate_replay_matched_hidden_actuation(
                template,
                mode=mode,
                resolution=int(manifest["resolution"]),
            )
            for mode in MODES
        }
        causal_audit = validate_replay_matched_pair(
            rollouts["low_gain"],
            rollouts["high_gain"],
        )
        causal_audits.append(causal_audit)

        for mode_index, mode in enumerate(MODES):
            rollout = rollouts[mode]
            episode_index = 2 * pair_index + mode_index
            model_rows = [
                episode_index * RAW_ROWS_PER_EPISODE + frame
                for frame in MODEL_FRAME_ROWS
            ]
            raw_rows = [
                episode_index * RAW_ROWS_PER_EPISODE + step
                for step in range(RAW_ROWS_PER_EPISODE)
            ]
            stored_pixels = dataset.take(
                model_rows,
                columns=["pixels", "pair_id", "hidden_mode"],
            ).to_pylist()
            stored_actions = dataset.take(
                raw_rows,
                columns=["action", "pair_id", "hidden_mode"],
            ).to_pylist()
            for frame_index, (frame_row, stored) in enumerate(
                zip(MODEL_FRAME_ROWS, stored_pixels)
            ):
                if stored["pair_id"] != template.template_id:
                    raise RuntimeError("Stored pair order does not match manifest")
                if stored["hidden_mode"] != mode:
                    raise RuntimeError("Stored mode order does not match manifest")
                strict_blob = _encode_frame(
                    np.asarray(rollout["rows"]["pixels"][frame_row]),
                    jpeg_quality,
                )
                legacy_blob = bytes(stored["pixels"])
                legacy_digest.update(legacy_blob)
                strict_digest.update(strict_blob)
                if legacy_blob != strict_blob:
                    pixel_mismatches.append(
                        {
                            "pair_index": pair_index,
                            "mode": mode,
                            "model_frame_index": frame_index,
                            "legacy_sha256": _sha256(legacy_blob),
                            "strict_sha256": _sha256(strict_blob),
                        }
                    )
            legacy_actions = np.asarray(
                [row["action"] for row in stored_actions],
                dtype=np.float32,
            )
            strict_actions = np.asarray(
                rollout["raw_actions"],
                dtype=np.float32,
            )
            legacy_digest.update(legacy_actions.tobytes())
            strict_digest.update(strict_actions.tobytes())
            if not np.array_equal(legacy_actions, strict_actions):
                action_mismatches.append(
                    {
                        "pair_index": pair_index,
                        "mode": mode,
                        "maximum_absolute_difference": float(
                            np.max(np.abs(legacy_actions - strict_actions))
                        ),
                    }
                )

    result = {
        "label": label,
        "legacy_root": str(root),
        "split": split,
        "pair_count": len(indices),
        "pair_selection": (
            "all" if len(indices) == len(pairs) else indices
        ),
        "trajectory_count": 2 * len(indices),
        "model_frames_compared": 8 * len(indices),
        "raw_action_rows_compared": 40 * len(indices),
        "pixel_mismatch_count": len(pixel_mismatches),
        "action_mismatch_count": len(action_mismatches),
        "legacy_model_visible_sha256": legacy_digest.hexdigest(),
        "strict_model_visible_sha256": strict_digest.hexdigest(),
        "model_visible_hash_exactly_same_as_pre_fix": (
            legacy_digest.digest() == strict_digest.digest()
            and not pixel_mismatches
            and not action_mismatches
        ),
        "pixel_mismatches": pixel_mismatches[:20],
        "action_mismatches": action_mismatches[:20],
        "state_installations_after_x0": int(
            sum(
                row["state_installations_after_x0"]
                for row in causal_audits
            )
        ),
        "query_simulator_recreated": bool(
            any(
                row["query_simulator_recreated"]
                for row in causal_audits
            )
        ),
        "max_pair_full_state_gap": float(
            max(
                row["query_physics_max_abs_gap"]
                for row in causal_audits
            )
        ),
        "max_pair_query_pixel_difference": int(
            max(
                row["pair_query_pixel_difference"]
                for row in causal_audits
            )
        ),
        "max_pair_query_action_difference": float(
            max(
                row["pair_query_action_difference"]
                for row in causal_audits
            )
        ),
        "min_history_effect": float(
            min(row["history_effect"] for row in causal_audits)
        ),
        "min_true_future_effect": float(
            min(row["true_future_effect"] for row in causal_audits)
        ),
    }
    return result, causal_audits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )
    parser.add_argument("--pairs-per-dataset", type=int, default=8)
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.all_pairs else int(args.pairs_per_dataset)
    root = args.artifact_root.expanduser().resolve()
    specifications = (
        (
            "replay_training",
            root / "pusht_hidden_actuation_replay_matched_h3_v2",
            "train",
        ),
        (
            "replay_development",
            root / "pusht_hidden_actuation_replay_matched_h3_v2",
            "validation",
        ),
        (
            "frozen_public_test",
            root / "pusht_hidden_actuation_replay_matched_confirm_h3_v3",
            "validation",
        ),
    )
    datasets: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for label, dataset_root, split in specifications:
        result, values = _dataset_audit(
            label=label,
            root=dataset_root,
            split=split,
            pairs_per_dataset=limit,
            jpeg_quality=int(args.jpeg_quality),
        )
        datasets.append(result)
        audits.extend(values)

    summary = {
        "pair_count": len(audits),
        "state_installations_after_x0": int(
            sum(row["state_installations_after_x0"] for row in audits)
        ),
        "query_simulator_recreated": bool(
            any(row["query_simulator_recreated"] for row in audits)
        ),
        "max_pair_full_state_gap": float(
            max(row["query_physics_max_abs_gap"] for row in audits)
        ),
        "max_pair_query_pixel_difference": int(
            max(row["pair_query_pixel_difference"] for row in audits)
        ),
        "max_pair_query_action_difference": float(
            max(row["pair_query_action_difference"] for row in audits)
        ),
        "min_history_effect": float(
            min(row["history_effect"] for row in audits)
        ),
        "min_true_future_effect": float(
            min(row["true_future_effect"] for row in audits)
        ),
        "full_state_tolerance": float(
            min(row["query_physics_tolerance"] for row in audits)
        ),
        "full_state_dimensions": int(audits[0]["full_state_dimensions"]),
        "full_state_components": audits[0]["full_state_components"],
        "model_visible_hash_exactly_same_as_pre_fix": all(
            row["model_visible_hash_exactly_same_as_pre_fix"]
            for row in datasets
        ),
    }
    summary["passed"] = (
        summary["state_installations_after_x0"] == 0
        and not summary["query_simulator_recreated"]
        and summary["max_pair_full_state_gap"]
        <= summary["full_state_tolerance"]
        and summary["max_pair_query_pixel_difference"] == 0
        and summary["max_pair_query_action_difference"] == 0.0
        and summary["model_visible_hash_exactly_same_as_pre_fix"]
    )
    report = {
        "schema_version": 1,
        "scope": "PushT action-strength strict causal-chain audit",
        "datasets": datasets,
        "summary": summary,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
