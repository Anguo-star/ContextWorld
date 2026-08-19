#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.speed_door_rule_score import (
    aggregate_results,
)
from contextworld.evaluation.speed_door_rule_validation import file_sha256
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_speed_door_rule_h3_validation_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen composition checkpoint matrix"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    results_root = resolve_contextworld_path(
        args.results_root or config["artifacts"]["results_root"],
        repo_root=ROOT,
    )
    paths = sorted(results_root.glob("*.json"))
    output = resolve_contextworld_path(
        args.output or config["artifacts"]["aggregate"],
        repo_root=ROOT,
    )
    paths = [path for path in paths if path.resolve() != output.resolve()]
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    aggregate = aggregate_results(
        results,
        required_joint_training_seeds=config["decision_gates"]["method"][
            "training_seeds"
        ],
    )
    aggregate["inputs"] = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in paths
    ]
    aggregate["identity"] = {
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
    }
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
