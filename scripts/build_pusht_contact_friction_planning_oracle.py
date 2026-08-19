#!/usr/bin/env python3
"""Build the real-simulator action oracle for contact-friction planning."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.benchmarks.contact_friction_icl_data import (
    DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    ContactFrictionICLEvalDataset,
    file_sha256,
    load_contact_friction_icl_release,
)
from contextworld.evaluation.pusht_contact_friction_h3 import (
    ENDPOINT_MODES,
    ContactFrictionTemplate,
    simulate_query_future,
)


ANGLE_RADIUS_PX = 40.0


def _visible_state(snapshot: np.ndarray) -> np.ndarray:
    value = np.asarray(snapshot, dtype=np.float64)
    return np.asarray(
        [
            value[0],
            value[1],
            value[6],
            value[7],
            ANGLE_RADIUS_PX * value[10],
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-config",
        type=Path,
        default=DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-scale", type=float, default=0.10)
    parser.add_argument("--maximum-scale", type=float, default=1.60)
    parser.add_argument("--candidate-count", type=int, default=61)
    parser.add_argument("--acceptable-topk", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.candidate_count < 3
        or not 0.0 <= args.minimum_scale < args.maximum_scale
        or not 0 < args.acceptable_topk < args.candidate_count
    ):
        raise ValueError("Invalid candidate scale grid or top-k")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    release_path = args.release_config.expanduser().resolve()
    release = load_contact_friction_icl_release(release_path)
    dataset = ContactFrictionICLEvalDataset(
        release=release,
        repo_root=ROOT,
    )
    arrays = dataset.arrays
    manifest = json.loads(
        (dataset.root / "manifest.json").read_text(encoding="utf-8")
    )
    pairs = manifest["splits"]["validation"]["pairs"]
    if len(pairs) != arrays.pair_count:
        raise RuntimeError("Manifest and Validation pair counts differ")
    scales = np.linspace(
        args.minimum_scale,
        args.maximum_scale,
        args.candidate_count,
        dtype=np.float64,
    )
    scale_one = int(np.argmin(np.abs(scales - 1.0)))
    if not np.isclose(scales[scale_one], 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("The candidate grid must contain scale 1.0")

    rows = []
    started = time.monotonic()
    minimum_region_index_gap = args.candidate_count
    maximum_low_goal_error = 0.0
    best_high_scales = []
    for index, (pair_id, pair) in enumerate(
        zip(arrays.pair_ids, pairs, strict=True)
    ):
        template = ContactFrictionTemplate(**pair["template"])
        if pair_id != template.template_id:
            raise RuntimeError(
                f"Pair ordering mismatch: {pair_id} != {template.template_id}"
            )
        base_query = np.asarray(template.query_actions, dtype=np.float64)
        goal = _visible_state(arrays.low_physics_states[index, 3])
        mode_rows = {}
        for mode in ENDPOINT_MODES:
            costs = []
            contacts = []
            for scale in scales:
                candidate = np.clip(base_query * scale, -1.0, 1.0)
                candidate_template = replace(
                    template,
                    query_actions=tuple(map(tuple, candidate.tolist())),
                )
                result = simulate_query_future(
                    candidate_template,
                    mode=mode,
                    canonical_query_snapshot=(
                        template.canonical_query_snapshot
                    ),
                    resolution=32,
                    render_pixels=False,
                )
                costs.append(
                    float(
                        np.linalg.norm(
                            _visible_state(result["future_snapshot"]) - goal
                        )
                    )
                )
                contacts.append(
                    int(np.count_nonzero(result["contact_counts"]))
                )
            costs_array = np.asarray(costs, dtype=np.float64)
            order = np.argsort(costs_array, kind="stable")
            acceptable = np.sort(order[: args.acceptable_topk])
            best = int(order[0])
            mode_rows[mode] = {
                "physical_costs_px_equivalent": costs_array.tolist(),
                "contact_steps": contacts,
                "best_candidate_index": best,
                "best_scale": float(scales[best]),
                "best_physical_cost_px_equivalent": float(
                    costs_array[best]
                ),
                "acceptable_candidate_indices": acceptable.tolist(),
                "acceptable_scales": [
                    float(scales[value]) for value in acceptable
                ],
            }
        low_set = set(
            mode_rows["low_friction"]["acceptable_candidate_indices"]
        )
        high_set = set(
            mode_rows["high_friction"]["acceptable_candidate_indices"]
        )
        if low_set & high_set:
            raise RuntimeError(
                f"Current-frame-only regions overlap for {pair_id}"
            )
        if mode_rows["low_friction"]["best_candidate_index"] != scale_one:
            raise RuntimeError(f"Scale 1.0 is not the low oracle for {pair_id}")
        low_goal_error = mode_rows["low_friction"][
            "best_physical_cost_px_equivalent"
        ]
        if low_goal_error > 1e-3:
            raise RuntimeError(
                f"Stored low future is not reproduced for {pair_id}: "
                f"{low_goal_error}"
            )
        maximum_low_goal_error = max(maximum_low_goal_error, low_goal_error)
        minimum_region_index_gap = min(
            minimum_region_index_gap,
            min(abs(left - right) for left in low_set for right in high_set),
        )
        best_high_scales.append(
            mode_rows["high_friction"]["best_scale"]
        )
        rows.append(
            {
                "pair_id": pair_id,
                "goal_source": "low_friction_scale_1_true_future",
                "modes": mode_rows,
            }
        )
        if (index + 1) % 32 == 0:
            print(
                f"oracle {index + 1}/{arrays.pair_count}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "benchmark": "pusht_history3_contact_friction_planning_v1",
        "release": {
            "release_id": release["release_id"],
            "release_config_sha256": file_sha256(release_path),
            "data_manifest_sha256": release["data"]["manifest_sha256"],
        },
        "candidate_actions": {
            "definition": "stored_query_action_times_scalar",
            "minimum_scale": float(scales[0]),
            "maximum_scale": float(scales[-1]),
            "candidate_count": len(scales),
            "scales": scales.tolist(),
            "acceptable_region": (
                f"best_{args.acceptable_topk}_real_simulator_candidates"
            ),
            "acceptable_topk": args.acceptable_topk,
        },
        "physical_cost": {
            "fields": [
                "agent_position_xy",
                "block_position_xy",
                "block_angle_times_40px",
            ],
            "metric": "euclidean_px_equivalent",
            "angle_radius_px": ANGLE_RADIUS_PX,
        },
        "audit": {
            "pair_count": arrays.pair_count,
            "condition_count": 2 * arrays.pair_count,
            "all_low_high_acceptable_regions_disjoint": True,
            "current_frame_only_accuracy_bound": 0.5,
            "minimum_region_index_gap": minimum_region_index_gap,
            "maximum_low_goal_reproduction_error_px_equivalent": (
                maximum_low_goal_error
            ),
            "high_friction_best_scale": {
                "minimum": float(min(best_high_scales)),
                "mean": float(np.mean(best_high_scales)),
                "maximum": float(max(best_high_scales)),
            },
            "online_environment_calls_for_model_scoring": 0,
        },
        "pairs": rows,
        "elapsed_seconds": time.monotonic() - started,
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
                "audit": payload["audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
