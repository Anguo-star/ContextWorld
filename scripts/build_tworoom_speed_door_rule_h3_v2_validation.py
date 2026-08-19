#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.speed_door_rule_v2_validation import (
    build_v2_validation_catalog,
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
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen 300-query h1/h2 Speed × Door Rule v2 "
            "offline Validation catalog"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    output_root = resolve_contextworld_path(
        args.output_root or config["artifacts"]["validation_root"],
        repo_root=ROOT,
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    catalog, exclusion, report = build_v2_validation_catalog(
        config=config,
        repo_root=ROOT,
        output_root=output_root,
    )
    catalog_path = output_root / "catalog.json"
    exclusion_path = output_root / "training_exclusion_manifest.json"
    report_path = output_root / "build_report.json"
    write_json(catalog_path, catalog)
    write_json(exclusion_path, exclusion)
    report["identity"] = {
        "config": portable_contextworld_path(
            config_path, repo_root=ROOT
        ),
        "config_sha256": file_sha256(config_path),
        "stable_worldmodel_commit": str(
            config["stable_worldmodel"]["commit"]
        ),
        "catalog": portable_contextworld_path(
            catalog_path, repo_root=ROOT
        ),
        "catalog_sha256": file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path, repo_root=ROOT
        ),
        "training_exclusion_manifest_sha256": file_sha256(
            exclusion_path
        ),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
