#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import (
    build_tworoom_icl_validation_catalog,
    validate_context_query_catalog,
)
from contextworld.paths import artifact_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def run(args: argparse.Namespace) -> dict:
    args.speed_manifest = resolve_contextworld_path(
        args.speed_manifest, repo_root=REPO_ROOT
    )
    args.door_manifest = resolve_contextworld_path(
        args.door_manifest, repo_root=REPO_ROOT
    )
    args.composition_manifest = resolve_contextworld_path(
        args.composition_manifest, repo_root=REPO_ROOT
    )
    args.output = resolve_contextworld_path(args.output, repo_root=REPO_ROOT)
    args.payload_root = resolve_contextworld_path(
        args.payload_root, repo_root=REPO_ROOT
    )
    args.report = resolve_contextworld_path(args.report, repo_root=REPO_ROOT)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    manifests = {
        "speed": args.speed_manifest.resolve(),
        "door": args.door_manifest.resolve(),
        "speed_door_composition": args.composition_manifest.resolve(),
    }
    catalog = build_tworoom_icl_validation_catalog(
        repo_root=REPO_ROOT,
        manifest_paths=manifests,
        output_catalog=args.output.resolve(),
        payload_root=args.payload_root.resolve(),
        stable_worldmodel_commit=stable_commit,
        generator_seed=args.seed,
    )
    validation = validate_context_query_catalog(
        args.output.resolve(),
        repo_root=REPO_ROOT,
        replay_simulator=not args.skip_simulator_replay,
    )
    validation["stable_worldmodel"] = {
        "repo": str(stable_repo),
        "commit": stable_commit,
    }
    write_json(args.report.resolve(), validation)
    if not validation["passed"]:
        raise RuntimeError(
            f"ContextQueryCatalog validation failed: {validation['failures'][:5]}"
        )
    return {
        "catalog": str(args.output.resolve()),
        "report": str(args.report.resolve()),
        "bundles": catalog["summary"]["bundles"],
        "passed": validation["passed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build strict paired TwoRoom T0/T1 validation bundles"
    )
    parser.add_argument(
        "--speed-manifest",
        type=Path,
        default=artifact_path(
            "synthesis/manifests/tworoom_speed_pixel_v2.jsonl",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--door-manifest",
        type=Path,
        default=artifact_path(
            "synthesis/manifests/tworoom_door_pixel_v1.jsonl",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--composition-manifest",
        type=Path,
        default=artifact_path(
            "synthesis/manifests/tworoom_speed_door_composition_v1.jsonl",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_context_query_catalog.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_payloads",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=artifact_path(
            "evaluation/icl/tworoom_icl_v1_validation_catalog_validation.json",
            repo_root=REPO_ROOT,
        ),
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--skip-simulator-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
