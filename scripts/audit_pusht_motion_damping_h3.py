#!/usr/bin/env python3
"""Replay the frozen PushT motion-damping History-3 feasibility audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contextworld.evaluation.pusht_motion_damping_h3 as damping  # noqa: E402
from contextworld.evaluation import pusht_contact_friction_h3 as friction  # noqa: E402
from contextworld.evaluation.pusht_motion_damping_h3 import (  # noqa: E402
    ENDPOINT_MODES,
    evaluate_template,
    make_confirmation_templates,
    make_motion_damping_env,
)
from contextworld.synthesis.manifest import write_json  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/evaluation/history3/"
    "pusht_motion_damping_h3_strict_causal_v3/audit.json"
)


def _solver_cache_diagnostic(*, resolution: int) -> dict:
    """Compare continuous x3 with a clean simulator for audit only.

    The clean branch deliberately restores the naturally reached x2.  It is
    never called by the data builder and must not be used to generate rows.
    """

    rows = []
    for template in make_confirmation_templates():
        for mode in ENDPOINT_MODES:
            continuous = damping._simulate_continuous_causal_chain(
                template,
                mode=mode,
                resolution=resolution,
                render_pixels=True,
            )
            clean, _ = make_motion_damping_env(
                template, mode=mode, resolution=resolution
            )
            try:
                friction.restore_body_snapshot(
                    clean, continuous["query_snapshot"]
                )
                clean_start_arbiters = len(clean.space._get_arbiters())
                for action in np.asarray(template.query_actions, dtype=np.float64):
                    friction._step_and_count_agent_block_contacts(clean, action)
                clean_future = friction.body_snapshot(clean)
                clean_future_pixels = np.asarray(
                    clean.render(), dtype=np.uint8
                ).copy()
                clean_end_arbiters = len(clean.space._get_arbiters())
            finally:
                clean.close()
            state_gap = float(
                np.max(
                    np.abs(
                        friction._snapshot_delta(
                            continuous["future_snapshot"], clean_future
                        )
                    )
                )
            )
            pixel_difference = int(
                np.max(
                    np.abs(
                        continuous["future_pixels"].astype(np.int16)
                        - clean_future_pixels.astype(np.int16)
                    )
                )
            )
            rows.append(
                {
                    "template_id": template.template_id,
                    "mode": mode,
                    "continuous_max_arbiter_count": int(
                        max(
                            np.max(
                                continuous["history_arbiter_counts"], initial=0
                            ),
                            np.max(
                                continuous["query_arbiter_counts"], initial=0
                            ),
                        )
                    ),
                    "consecutive_arbiter_free_raw_steps_before_x2": 10,
                    "clean_start_arbiter_count": clean_start_arbiters,
                    "clean_end_arbiter_count": clean_end_arbiters,
                    "continuous_vs_clean_x3_full_state_gap": state_gap,
                    "continuous_vs_clean_x3_pixel_difference": pixel_difference,
                }
            )
    return {
        "purpose": "diagnostic_only_not_used_for_data_generation",
        "condition_count": len(rows),
        "minimum_consecutive_arbiter_free_raw_steps_before_x2": min(
            row["consecutive_arbiter_free_raw_steps_before_x2"] for row in rows
        ),
        "maximum_continuous_arbiter_count": max(
            row["continuous_max_arbiter_count"] for row in rows
        ),
        "maximum_clean_start_arbiter_count": max(
            row["clean_start_arbiter_count"] for row in rows
        ),
        "maximum_continuous_vs_clean_x3_full_state_gap": max(
            row["continuous_vs_clean_x3_full_state_gap"] for row in rows
        ),
        "maximum_continuous_vs_clean_x3_pixel_difference": max(
            row["continuous_vs_clean_x3_pixel_difference"] for row in rows
        ),
        "conditions": rows,
        "passed": all(
            row["continuous_max_arbiter_count"] == 0
            and row["clean_start_arbiter_count"] == 0
            and row["continuous_vs_clean_x3_full_state_gap"] <= 1e-8
            and row["continuous_vs_clean_x3_pixel_difference"] == 0
            for row in rows
        ),
    }


def build_report(*, resolution: int) -> dict:
    first = [
        evaluate_template(template, resolution=resolution)
        for template in make_confirmation_templates()
    ]
    second = [
        evaluate_template(template, resolution=resolution)
        for template in make_confirmation_templates()
    ]
    deterministic = all(left == right for left, right in zip(first, second, strict=True))
    passed = sum(int(row["passed"]) for row in first)
    solver_cache = _solver_cache_diagnostic(resolution=resolution)
    return {
        "schema_version": 1,
        "benchmark": "pusht_motion_damping_history3_strict_causal_v3",
        "status": (
            "completed_history3_physical_feasibility_passed"
            if passed == 8 and deterministic
            else "failed"
        ),
        "claim_limit": "physical_identifiability_not_model_icl",
        "history3_confirmation": {
            "templates": len(first),
            "orientations": 4,
            "positions_per_orientation": 2,
            "passed_templates": passed,
            "pair_count": len(first),
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "query_full_state_tolerance": damping.QUERY_STATE_TOLERANCE,
            "query_full_state_dimensions": (
                "agent_position_velocity_angle_angular_velocity_and_"
                "block_position_velocity_angle_angular_velocity"
            ),
            "max_pair_full_state_gap": max(
                row["max_pair_full_state_gap"] for row in first
            ),
            "max_pair_query_pixel_difference": max(
                row["max_pair_query_pixel_difference"] for row in first
            ),
            "max_pair_query_action_difference": max(
                row["max_pair_query_action_difference"] for row in first
            ),
            "max_query_reference_deviation": max(
                value
                for row in first
                for value in row["query_reference_deviation"].values()
            ),
            "query_reference_tolerance": damping.QUERY_REFERENCE_TOLERANCE,
            "min_history_effect": min(
                row["history_visible_response_gap"]["px_equivalent"]
                for row in first
            ),
            "history_gap_minimum": 3.0,
            "min_true_future_effect": min(
                row["future_gap"]["block_position_px"] for row in first
            ),
            "future_gap_minimum": 2.0,
            "all_deterministic": deterministic,
            "maximum_arbiter_count_from_x0_through_x3": max(
                row["maximum_arbiter_count_from_x0_through_x3"]
                for row in first
            ),
            "x0_rgb_hash_multisets_identical_across_modes": (
                Counter(
                    row["hashes"]["faster_decay_initial_pixels"]
                    for row in first
                )
                == Counter(
                    row["hashes"]["no_extra_decay_initial_pixels"]
                    for row in first
                )
            ),
            "template_results": first,
        },
        "audit": {"resolution": int(resolution)},
        "solver_cache_diagnostic": solver_cache,
        "passed": bool(passed == 8 and deterministic and solver_cache["passed"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(resolution=int(args.resolution))
    write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
