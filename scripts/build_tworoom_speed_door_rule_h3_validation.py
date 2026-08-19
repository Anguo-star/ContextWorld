#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.speed_door_rule_validation import (
    build_validation_catalog,
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
    / "configs/benchmark/tworoom_speed_door_rule_h3_validation_v1.yaml"
)


def _assert_relevant_sources_clean(
    stable_repo: Path,
    sources: list[str],
) -> None:
    for source in sources:
        if not (stable_repo / source).is_file():
            raise FileNotFoundError(stable_repo / source)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(stable_repo),
            "diff",
            "--quiet",
            "HEAD",
            "--",
            *sources,
        ],
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError("Could not audit Stable-WorldModel sources")
    if result.returncode == 1:
        raise RuntimeError(
            "A Stable-WorldModel source used by Validation differs from HEAD"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen 300-query History=3 Speed × Door Rule "
            "offline Validation catalog"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    if (
        config.get("status")
        != "preregistered_before_catalog_generation_training_or_model_scoring"
    ):
        raise ValueError("Validation config is not preregistered")
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    sources = [
        str(value)
        for value in config["stable_worldmodel"]["relevant_sources"]
    ]
    _assert_relevant_sources_clean(stable_repo, sources)
    output_root = resolve_contextworld_path(
        args.output_root or config["artifacts"]["output_root"],
        repo_root=ROOT,
    )
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    catalog, report = build_validation_catalog(
        config=config,
        repo_root=ROOT,
        output_root=output_root,
    )
    catalog_path = output_root / "catalog.json"
    exclusion_path = output_root / "training_exclusion_manifest.json"
    report_path = output_root / "build_report.json"
    write_json(catalog_path, catalog)
    write_json(
        exclusion_path,
        {
            "schema_version": 1,
            "benchmark": str(config["benchmark"]),
            "purpose": (
                "Fail-closed exclusion of all Eval-only doors and selected "
                "query pixels from composition training data"
            ),
            "eval_only_door_positions": catalog["summary"][
                "eval_only_doors"
            ],
            "eval_speeds": catalog["summary"]["eval_speeds"],
            "training_speeds": catalog["summary"]["training_speeds"],
            "query_records": [
                {
                    "query_id": bundle["query_id"],
                    "template_id": bundle["template"]["template_id"],
                    "eval_seed": bundle["eval_seed"],
                    "door_position": bundle["door_position"],
                    "direction": bundle["direction"],
                    "query_pixels_sha256": bundle[
                        "query_pixels_sha256"
                    ],
                }
                for bundle in catalog["bundles"]
            ],
            "query_count": len(catalog["bundles"]),
            "content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
        },
    )
    report["identity"] = {
        "config": portable_contextworld_path(
            config_path, repo_root=ROOT
        ),
        "config_sha256": file_sha256(config_path),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "stable_worldmodel_sources": {
            source: {
                "path": source,
                "sha256": file_sha256(stable_repo / source),
            }
            for source in sources
        },
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
        "contextworld_sources": {
            name: {
                "path": portable_contextworld_path(path, repo_root=ROOT),
                "sha256": file_sha256(path),
            }
            for name, path in {
                "environment": (
                    ROOT
                    / "contextworld/evaluation/hidden_passage_env.py"
                ),
                "protocol": (
                    ROOT
                    / "contextworld/evaluation/"
                    "speed_door_rule_composition.py"
                ),
                "validation": (
                    ROOT
                    / "contextworld/evaluation/"
                    "speed_door_rule_validation.py"
                ),
                "entrypoint": Path(__file__).resolve(),
            }.items()
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
