#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h3_data import (
    build_training_release,
)
from contextworld.evaluation.action_delay_validation import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_training_data_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired single-delay and multi-delay History=3 "
            "training releases"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    workers = int(
        args.workers
        if args.workers is not None
        else config["collection"]["workers"]
    )
    if workers <= 0:
        raise ValueError("--workers must be positive")
    report = build_training_release(
        config=config,
        repo_root=ROOT,
        workers=workers,
        resume=args.resume,
    )
    report["identity"]["config"] = portable_contextworld_path(
        config_path,
        repo_root=ROOT,
    )
    report["identity"]["config_sha256"] = file_sha256(config_path)
    report["identity"]["sources"] = {
        name: {
            "path": portable_contextworld_path(path, repo_root=ROOT),
            "sha256": file_sha256(path),
        }
        for name, path in {
            "environment": (
                ROOT / "contextworld/evaluation/action_delay_env.py"
            ),
            "physical_protocol": (
                ROOT / "contextworld/evaluation/action_delay.py"
            ),
            "training_builder": (
                ROOT / "contextworld/evaluation/action_delay_h3_data.py"
            ),
            "entrypoint": Path(__file__).resolve(),
        }.items()
    }
    output_root = resolve_contextworld_path(
        config["output_root"],
        repo_root=ROOT,
    )
    write_json(output_root / "build_report.json", report)
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "checks": report["checks"],
                "physical_counts": report["physical_counts"],
                "delay_support": report["delay_support"],
                "validation_exclusion_audit": report[
                    "validation_exclusion_audit"
                ],
                "artifacts_by_group": report["artifacts_by_group"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
