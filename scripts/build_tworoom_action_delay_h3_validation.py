#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_validation import (
    build_validation_release,
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
    ROOT
    / "configs/benchmark/tworoom_action_delay_h3_validation_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen 50x6 five-delay Validation release"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    output_root = resolve_contextworld_path(
        (
            args.output_root
            if args.output_root is not None
            else config["artifacts"]["output_root"]
        ),
        repo_root=ROOT,
    )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Validation release {output_root}"
        )
    report = build_validation_release(
        config=config,
        repo_root=ROOT,
        output_root=output_root,
    )
    report["identity"] = {
        "config": portable_contextworld_path(
            config_path,
            repo_root=ROOT,
        ),
        "config_sha256": file_sha256(config_path),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "sources": {
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
                "validation_builder": (
                    ROOT
                    / "contextworld/evaluation/action_delay_validation.py"
                ),
                "entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }
    write_json(output_root / "build_report.json", report)
    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "status": report["status"],
                "checks": report["checks"],
                "counts": report["counts"],
                "catalog": report["catalog"],
                "catalog_sha256": report["catalog_sha256"],
                "training_exclusion_manifest_sha256": report[
                    "training_exclusion_manifest_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
