#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.speed_door_rule_v2_feasibility import (
    audit_v2_query_bundles,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.config import load_config
from contextworld.synthesis.stablewm import load_stable_worldmodel


DEFAULT_CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only physical prototype audit for the History=3 "
            "Speed × Door Rule v2 design"
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--limit",
        type=int,
        help="Audit only the first N paired queries for a quick smoke run",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    load_stable_worldmodel(
        ROOT,
        str(config["stable_worldmodel"]["repo"]),
        str(config["stable_worldmodel"]["commit"]),
    )
    source = config["evaluation"]["paired_query_source"]
    catalog_path = resolve_contextworld_path(
        source["catalog"], repo_root=ROOT
    )
    observed_sha256 = file_sha256(catalog_path)
    if observed_sha256 != str(source["catalog_sha256"]):
        raise RuntimeError(
            "Paired v1 query catalog hash mismatch: "
            f"{observed_sha256} != {source['catalog_sha256']}"
        )
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    bundles = list(catalog["bundles"])
    require_full_catalog = args.limit is None
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        bundles = bundles[: args.limit]

    report = audit_v2_query_bundles(
        config=config,
        bundles=bundles,
        require_full_catalog=require_full_catalog,
    )
    report["source_catalog"] = str(catalog_path)
    report["source_catalog_sha256"] = observed_sha256
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
