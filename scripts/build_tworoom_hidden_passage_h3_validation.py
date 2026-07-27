#!/usr/bin/env python3
"""Build the frozen History-3 hidden-passage Validation dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.hidden_passage_validation import (
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
    / "configs/benchmark/tworoom_hidden_passage_h3_validation_v2.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build 300 disjoint frozen queries for the History-3 "
            "hidden-passage 2x3 Validation matrix"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace output owned by this same benchmark configuration",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    allowed_statuses = {
        "preregistered_before_independent_catalog_generation_and_scoring",
        "diagnostic_frozen_before_catalog_generation_and_scoring",
    }
    if config.get("status") not in allowed_statuses:
        raise ValueError(
            "Validation/diagnostic config is not frozen before scoring"
        )
    _, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    if stable_commit != str(config["stable_worldmodel"]["commit"]):
        raise RuntimeError(
            "Stable-WorldModel commit mismatch: "
            f"{stable_commit} != {config['stable_worldmodel']['commit']}"
        )

    configured_output = (
        args.output_root
        if args.output_root is not None
        else Path(config["artifacts"]["output_root"])
    )
    output_root = resolve_contextworld_path(
        configured_output,
        repo_root=ROOT,
    )
    if output_root.exists():
        if not args.refresh_existing:
            raise FileExistsError(
                f"Refusing to overwrite Validation output {output_root}"
            )
        prior_report = output_root / "build_report.json"
        if not prior_report.is_file():
            raise FileNotFoundError(
                "--refresh-existing requires the prior build_report.json"
            )
        prior = json.loads(prior_report.read_text(encoding="utf-8"))
        if prior.get("benchmark") != config["benchmark"]:
            raise ValueError(
                "Existing output belongs to another benchmark: "
                f"{prior.get('benchmark')!r}"
            )
        shutil.rmtree(output_root)
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
    query_domain = str(
        config["data"]["generation"].get("query_domain", "eval_only")
    )
    write_json(
        exclusion_path,
        {
            "schema_version": 1,
            "benchmark": config["benchmark"],
            "purpose": (
                (
                    "fail-closed training exclusion for every frozen "
                    "eval-only door position and selected Validation query"
                )
                if query_domain == "eval_only"
                else (
                    "identity manifest for frozen training-seen diagnostic "
                    "queries; not a training exclusion"
                )
            ),
            "query_domain": query_domain,
            "eval_only_door_positions": [
                int(value)
                for value in config["data"]["generation"][
                    "eval_only_door_positions"
                ]
            ],
            "query_records": catalog["training_exclusion_manifest"],
            "query_count": len(catalog["training_exclusion_manifest"]),
            "content_manifest_sha256": catalog[
                "content_manifest_sha256"
            ],
        },
    )
    report["identity"] = {
        "config": portable_contextworld_path(config_path, repo_root=ROOT),
        "config_sha256": file_sha256(config_path),
        "stable_worldmodel_repo": str(stable_repo),
        "stable_worldmodel_commit": stable_commit,
        "catalog": portable_contextworld_path(catalog_path, repo_root=ROOT),
        "catalog_sha256": file_sha256(catalog_path),
        "training_exclusion_manifest": portable_contextworld_path(
            exclusion_path,
            repo_root=ROOT,
        ),
        "training_exclusion_manifest_sha256": file_sha256(exclusion_path),
    }
    write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
