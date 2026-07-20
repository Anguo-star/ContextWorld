#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.natural_history import (
    build_natural_history_catalog,
    validate_natural_history_catalog,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"
SOURCES = [
    {
        "family": "speed",
        "catalog": "artifacts/synthesis/catalogs/tworoom_speed_pixel_v2.json",
        "regime": "validation_interp",
    },
    {
        "family": "door",
        "catalog": "artifacts/synthesis/catalogs/tworoom_door_pixel_v1.json",
        "regime": "validation_interp",
    },
    {
        "family": "speed_door_composition",
        "catalog": "artifacts/synthesis/catalogs/tworoom_speed_door_composition_v1.json",
        "regime": "validation_unseen_combination",
    },
]


def run(args: argparse.Namespace) -> dict:
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    args.report = resolve_contextworld_path(args.report, repo_root=REPO_ROOT)
    swm, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    catalog = build_natural_history_catalog(
        swm=swm,
        repo_root=REPO_ROOT,
        sources=SOURCES,
        generator_seed=args.seed,
        clips_per_scenario=args.clips_per_scenario,
    )
    catalog["stable_worldmodel"] = {
        "repo": str(stable_repo),
        "commit": stable_commit,
    }
    write_json(args.output.resolve(), catalog)
    report = validate_natural_history_catalog(
        catalog, swm=swm, repo_root=REPO_ROOT
    )
    report["catalog"] = str(args.output.resolve())
    write_json(args.report.resolve(), report)
    if not report["passed"]:
        raise RuntimeError(f"Natural-history catalog validation failed: {report}")
    return {
        "catalog": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "clips": catalog["counts"]["clips"],
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic training-equivalent TwoRoom validation clips"
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--clips-per-scenario", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_natural_history_v1_validation_catalog.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_natural_history_v1_validation_report.json",
            repo_root=REPO_ROOT,
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
