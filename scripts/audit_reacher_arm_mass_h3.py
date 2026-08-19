#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

from contextworld.evaluation.reacher_arm_mass_h3 import (  # noqa: E402
    ReacherArmMassSimulator,
    make_candidate,
)
from contextworld.paths import resolve_contextworld_path  # noqa: E402
from contextworld.synthesis.manifest import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/reacher_arm_mass_h3_feasibility_v1.yaml"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or resolve_contextworld_path(
        config["output"], repo_root=ROOT
    )
    required = int(config["gates"]["minimum_passing_templates"])
    simulator = ReacherArmMassSimulator()
    accepted = []
    attempted = 0
    try:
        while len(accepted) < required:
            candidate = make_candidate(
                split="feasibility",
                index=attempted,
                catalog_seed=2026080201,
            )
            result = simulator.build_pair(candidate)
            attempted += 1
            if result is not None:
                accepted.append(
                    {
                        "candidate": result["candidate"],
                        "audit": result["audit"],
                    }
                )
            if attempted > 16 * required:
                raise RuntimeError(
                    f"Only {len(accepted)}/{required} candidates passed"
                )
    finally:
        simulator.close()
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "completed",
        "config": str(args.config.resolve()),
        "attempted_candidates": attempted,
        "passing_templates": len(accepted),
        "acceptance_rate": len(accepted) / attempted,
        "minimum_history_qpos_gap": min(
            row["audit"]["history_qpos_gap"] for row in accepted
        ),
        "minimum_true_future_qpos_gap": min(
            row["audit"]["future_qpos_gap"] for row in accepted
        ),
        "maximum_query_state_gap": max(
            row["audit"]["query_state_gap"] for row in accepted
        ),
        "minimum_history_changed_rgb_values": min(
            row["audit"]["history_changed_rgb_values"]
            for row in accepted
        ),
        "minimum_future_changed_rgb_values": min(
            row["audit"]["future_changed_rgb_values"]
            for row in accepted
        ),
        "all_query_pixels_bitwise_equal": all(
            row["audit"]["query_pixels_equal"] for row in accepted
        ),
        "all_actions_bitwise_equal": all(
            row["audit"]["actions_equal"] for row in accepted
        ),
        "passed": len(accepted) == required,
        "templates": accepted,
    }
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "templates"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
