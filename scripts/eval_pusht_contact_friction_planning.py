#!/usr/bin/env python3
"""Score one checkpoint on frozen contact-friction action selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelPLDMContactFrictionAdapter,
)
from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    ContactFrictionICLEvalDataset,
    file_sha256,
    load_contact_friction_icl_release,
)


ADAPTERS = {
    "lewm": StableWorldModelLeWMContactFrictionAdapter,
    "pldm": StableWorldModelPLDMContactFrictionAdapter,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--training-seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--condition-chunk-size", type=int, default=16)
    return parser.parse_args()


def _build_adapter(args: argparse.Namespace, release: dict[str, Any]):
    normalization = release["evaluation"]["action_normalization"]
    runtime = release["runtime"]["stable_worldmodel"]
    return ADAPTERS[args.adapter].from_checkpoint(
        args.checkpoint,
        action_mean=normalization["mean"],
        action_std=normalization["std_population"],
        repo_root=ROOT,
        stablewm_repo=runtime["repo"],
        stablewm_ref=runtime["expected_ref"],
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    if args.condition_chunk_size <= 0:
        raise ValueError("--condition-chunk-size must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    oracle_path = args.oracle.expanduser().resolve()
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    expected_identity = {
        "release_id": release["release_id"],
        "release_config_sha256": file_sha256(release_path),
        "data_manifest_sha256": release["data"]["manifest_sha256"],
    }
    if oracle.get("release") != expected_identity:
        raise RuntimeError("Planning oracle release identity mismatch")
    dataset = ContactFrictionICLEvalDataset(
        release=release,
        repo_root=ROOT,
    )
    arrays = dataset.arrays
    oracle_rows = oracle["pairs"]
    if tuple(row["pair_id"] for row in oracle_rows) != arrays.pair_ids:
        raise RuntimeError("Planning oracle pair order mismatch")
    scales = np.asarray(
        oracle["candidate_actions"]["scales"],
        dtype=np.float32,
    )
    candidate_count = len(scales)
    adapter = _build_adapter(args, release)
    before = adapter.frozen_state_hash()
    goals = adapter.encode_pixels(
        arrays.low_pixels[:, 3],
        batch_size=args.batch_size,
    )

    conditions = [
        (mode, index)
        for mode in ("low_friction", "high_friction")
        for index in range(arrays.pair_count)
    ]
    selected_indices = []
    selected_costs = []
    for start in range(0, len(conditions), args.condition_chunk_size):
        chunk = conditions[start : start + args.condition_chunk_size]
        histories = np.stack(
            [
                (
                    arrays.low_pixels[index, :3]
                    if mode == "low_friction"
                    else arrays.high_pixels[index, :3]
                )
                for mode, index in chunk
            ]
        )
        repeated_histories = np.repeat(
            histories,
            candidate_count,
            axis=0,
        )
        action_rows = []
        goal_rows = []
        for _mode, index in chunk:
            fixed = arrays.raw_action_blocks[index, :2]
            query = (
                scales[:, None, None]
                * arrays.raw_action_blocks[index, 2][None]
            )
            fixed_rows = np.broadcast_to(
                fixed[None],
                (candidate_count, 2, 5, 2),
            )
            action_rows.append(
                np.concatenate([fixed_rows, query[:, None]], axis=1)
            )
            goal_rows.append(
                np.broadcast_to(
                    goals[index],
                    (candidate_count, goals.shape[-1]),
                )
            )
        actions = np.concatenate(action_rows, axis=0)
        repeated_goals = np.concatenate(goal_rows, axis=0)
        predicted = adapter.rollout_latents(
            repeated_histories,
            actions,
            batch_size=args.batch_size,
        )[:, 0]
        costs = np.square(predicted - repeated_goals).mean(axis=-1)
        costs = costs.reshape(len(chunk), candidate_count)
        selected_indices.extend(np.argmin(costs, axis=1).tolist())
        selected_costs.extend(np.min(costs, axis=1).tolist())
        print(
            f"scored {min(start + len(chunk), len(conditions))}/"
            f"{len(conditions)} conditions",
            flush=True,
        )

    after = adapter.frozen_state_hash()
    if before != after:
        raise RuntimeError("Model state changed during planning evaluation")
    records = []
    correct = []
    by_mode: dict[str, list[bool]] = {
        "low_friction": [],
        "high_friction": [],
    }
    for condition_index, ((mode, pair_index), selected, model_cost) in enumerate(
        zip(
            conditions,
            selected_indices,
            selected_costs,
            strict=True,
        )
    ):
        physical = oracle_rows[pair_index]["modes"][mode]
        acceptable = set(physical["acceptable_candidate_indices"])
        passed = selected in acceptable
        correct.append(passed)
        by_mode[mode].append(passed)
        records.append(
            {
                "condition_index": condition_index,
                "pair_id": arrays.pair_ids[pair_index],
                "mode": mode,
                "selected_candidate_index": selected,
                "selected_scale": float(scales[selected]),
                "selected_model_cost": float(model_cost),
                "selected_physical_cost_px_equivalent": float(
                    physical["physical_costs_px_equivalent"][selected]
                ),
                "oracle_best_scale": physical["best_scale"],
                "oracle_best_physical_cost_px_equivalent": physical[
                    "best_physical_cost_px_equivalent"
                ],
                "acceptable_candidate_indices": sorted(acceptable),
                "correct_action_region": passed,
            }
        )
    low_selected = np.asarray(
        selected_indices[: arrays.pair_count],
        dtype=np.int64,
    )
    high_selected = np.asarray(
        selected_indices[arrays.pair_count :],
        dtype=np.int64,
    )
    rate = float(np.mean(correct))
    metrics = {
        "pair_count": arrays.pair_count,
        "condition_count": len(conditions),
        "correct_action_region_rate": rate,
        "low_friction_correct_action_region_rate": float(
            np.mean(by_mode["low_friction"])
        ),
        "high_friction_correct_action_region_rate": float(
            np.mean(by_mode["high_friction"])
        ),
        "worst_friction_correct_action_region_rate": float(
            min(
                np.mean(by_mode["low_friction"]),
                np.mean(by_mode["high_friction"]),
            )
        ),
        "selected_scale_changes_with_history_rate": float(
            np.mean(low_selected != high_selected)
        ),
        "selected_scale_moves_in_oracle_direction_rate": float(
            np.mean(low_selected > high_selected)
        ),
        "current_frame_only_accuracy_bound": 0.5,
    }
    gate = {
        "minimum": 0.90,
        "passed": rate >= 0.90,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "pusht_history3_contact_friction_planning_v1",
        "release": expected_identity,
        "oracle": {
            "path": str(oracle_path),
            "sha256": file_sha256(oracle_path),
        },
        "model": {
            "name": args.model_name,
            "training_seed": args.training_seed,
            "adapter": adapter.metadata,
            "state_sha256_before": before,
            "state_sha256_after": after,
        },
        "metrics": metrics,
        "gate": gate,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(output),
                "metrics": metrics,
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
