#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_h7_domain_diagnostic import (
    build_domain_diagnostic_release,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.paths import (
    portable_contextworld_path,
    resolve_contextworld_path,
)
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_h7_domain_diagnostic_data_v1.yaml"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build six paired 50x6 History=7 Action Delay domain "
            "diagnostic tracks"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stablewm-repo")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 1),
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = _load(config_path)
    source = config["source_training_release"]
    training_config_path = resolve_contextworld_path(
        source["config"]["path"],
        repo_root=ROOT,
    )
    build_report_path = resolve_contextworld_path(
        source["build_report"]["path"],
        repo_root=ROOT,
    )
    training_catalog_path = resolve_contextworld_path(
        source["multi_delay_catalog"]["path"],
        repo_root=ROOT,
    )
    for path, expected in (
        (training_config_path, source["config"]["sha256"]),
        (build_report_path, source["build_report"]["sha256"]),
        (
            training_catalog_path,
            source["multi_delay_catalog"]["sha256"],
        ),
    ):
        if file_sha256(path) != str(expected):
            raise RuntimeError(f"Frozen source identity changed: {path}")
    training_config = _load(training_config_path)
    training_catalog = json.loads(
        training_catalog_path.read_text(encoding="utf-8")
    )
    formal_catalog_path = (
        resolve_contextworld_path(
            training_config["validation_exclusion"]["manifest"],
            repo_root=ROOT,
        ).parent
        / "catalog.json"
    )
    formal_validation_catalog = json.loads(
        formal_catalog_path.read_text(encoding="utf-8")
    )
    stablewm_repo = str(
        args.stablewm_repo
        or source["stable_worldmodel"]["repo"]
    )
    _, observed_repo, observed_commit = load_stable_worldmodel(
        ROOT,
        stablewm_repo,
        str(source["stable_worldmodel"]["commit"]),
    )
    output_root = resolve_contextworld_path(
        args.output_root or config["outputs"]["root"],
        repo_root=ROOT,
    )
    if output_root.exists():
        raise FileExistsError(output_root)
    _, report = build_domain_diagnostic_release(
        config=config,
        training_config=training_config,
        training_catalog=training_catalog,
        formal_validation_catalog=formal_validation_catalog,
        repo_root=ROOT,
        output_root=output_root,
        workers=int(args.workers),
        stablewm_repo=str(observed_repo),
        stablewm_commit=observed_commit,
    )
    report["identity"] = {
        "config": {
            "path": portable_contextworld_path(
                config_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(config_path),
        },
        "source_training_config": {
            "path": portable_contextworld_path(
                training_config_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(training_config_path),
        },
        "source_training_build_report": {
            "path": portable_contextworld_path(
                build_report_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(build_report_path),
        },
        "source_training_catalog": {
            "path": portable_contextworld_path(
                training_catalog_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(training_catalog_path),
        },
        "formal_validation_catalog": {
            "path": portable_contextworld_path(
                formal_catalog_path,
                repo_root=ROOT,
            ),
            "sha256": file_sha256(formal_catalog_path),
        },
        "stable_worldmodel": {
            "repo": str(observed_repo),
            "commit": observed_commit,
        },
        "sources": {
            name: {
                "path": portable_contextworld_path(
                    path,
                    repo_root=ROOT,
                ),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "environment": (
                    ROOT / "contextworld/evaluation/action_delay_env.py"
                ),
                "history_protocol": (
                    ROOT
                    / "contextworld/evaluation/"
                    "action_delay_long_history.py"
                ),
                "source_training_builder": (
                    ROOT
                    / "contextworld/evaluation/action_delay_h7_data.py"
                ),
                "diagnostic_builder": (
                    ROOT
                    / "contextworld/evaluation/"
                    "action_delay_h7_domain_diagnostic.py"
                ),
                "entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }
    write_json(output_root / "build_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "counts": report["counts"],
                "audit": report["audit"],
                "catalog": report["catalog"],
                "catalog_sha256": report["catalog_sha256"],
                "build_report": portable_contextworld_path(
                    output_root / "build_report.json",
                    repo_root=ROOT,
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
