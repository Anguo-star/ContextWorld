#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_domain_diagnostic import (
    audit_domain_diagnostic_release,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_data_v1.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reopen and physically replay every H7 domain diagnostic asset"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument(
        "--integrity-only",
        action="store_true",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source_training_release"]
    training_config_path = resolve_contextworld_path(
        source["config"]["path"],
        repo_root=ROOT,
    )
    training_catalog_path = resolve_contextworld_path(
        source["multi_delay_catalog"]["path"],
        repo_root=ROOT,
    )
    training_config = yaml.safe_load(
        training_config_path.read_text(encoding="utf-8")
    )
    training_catalog = json.loads(
        training_catalog_path.read_text(encoding="utf-8")
    )
    catalog_path = resolve_contextworld_path(
        args.catalog
        or Path(config["outputs"]["root"]) / "catalog.json",
        repo_root=ROOT,
    )
    report = audit_domain_diagnostic_release(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
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
        "catalog": {
            "path": portable_contextworld_path(
                catalog_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(catalog_path),
        },
        "entrypoint": {
            "path": portable_contextworld_path(
                Path(__file__).resolve(),
                repo_root=ROOT,
            ),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    output = catalog_path.parent / "audit_report.json"
    write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
