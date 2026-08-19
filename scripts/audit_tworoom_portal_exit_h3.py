#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.portal_exit_h3 import (  # noqa: E402
    make_template,
    simulate_portal_exit_clip,
    validate_portal_exit_pair,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/benchmark/tworoom_portal_exit_h3_feasibility_v1.yaml",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / config["output"]
    count = int(config["catalog"]["templates"])
    seed = int(config["catalog"]["seed"])
    rows = []
    replay_checks = []
    for index in range(count):
        template = make_template(split="feasibility", index=index, catalog_seed=seed)
        near = simulate_portal_exit_clip(template, mode="near_border")
        farther = simulate_portal_exit_clip(template, mode="farther_from_border")
        audit = validate_portal_exit_pair(near, farther)
        replay = simulate_portal_exit_clip(template, mode="near_border")
        replay_equal = all(
            np.array_equal(near[key], replay[key])
            for key in (
                "history_pixels", "history_states", "history_actions",
                "query_pixels", "query_state", "query_action",
                "future_pixels", "future_state",
            )
        )
        replay_checks.append(replay_equal)
        rows.append({"index": index, "template": template.__dict__, "audit": audit})
    result = {
        "benchmark": config["benchmark"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": file_sha256(args.config),
        "templates": count,
        "rule_rollouts": 2 * count,
        "minimum_history_exit_gap_px": min(
            row["audit"]["middle_state_gap_px"] for row in rows
        ),
        "minimum_true_future_gap_px": min(
            row["audit"]["future_state_gap_px"] for row in rows
        ),
        "maximum_query_state_gap": max(
            row["audit"]["maximum_query_state_gap"] for row in rows
        ),
        "failed_templates": [
            row["template"]["template_id"]
            for row in rows if not row["audit"]["passed"]
        ],
        "exact_replay_passed": all(replay_checks),
        "passed": all(row["audit"]["passed"] for row in rows)
        and all(replay_checks),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "templates", "minimum_history_exit_gap_px",
        "minimum_true_future_gap_px", "maximum_query_state_gap",
        "exact_replay_passed", "passed",
    )}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
