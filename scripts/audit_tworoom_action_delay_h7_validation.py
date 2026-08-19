#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_validation import (
    audit_validation_release,
    file_sha256,
)
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.config import load_config
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reopen the formal H7 Action Delay Validation and replay all "
            "3,300 physical trajectories"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
        help="reopen hashes without replaying the physical environment",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    catalog_path = resolve_contextworld_path(
        (
            args.catalog
            if args.catalog is not None
            else Path(config["outputs"]["validation_root"]) / "catalog.json"
        ),
        repo_root=ROOT,
    )
    report = audit_validation_release(
        config=config,
        repo_root=ROOT,
        catalog_path=catalog_path,
        replay_physics=not args.integrity_only,
    )
    report["identity"] = {
        "config": {
            "path": portable_contextworld_path(
                config_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(config_path),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "sources": {
            name: {
                "path": portable_contextworld_path(path, repo_root=ROOT),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "environment": (
                    ROOT / "contextworld/evaluation/action_delay_env.py"
                ),
                "history_protocol": (
                    ROOT
                    / "contextworld/evaluation/action_delay_long_history.py"
                ),
                "validation_builder": (
                    ROOT
                    / "contextworld/evaluation/"
                    "action_delay_h7_validation.py"
                ),
                "audit_entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }
    output_path = catalog_path.parent / "audit_report.json"
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "mode": report["mode"],
                "checks": report["checks"],
                "counts": report["counts"],
                "audit_report": portable_contextworld_path(
                    output_path,
                    repo_root=ROOT,
                ),
                "audit_report_sha256": file_sha256(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
