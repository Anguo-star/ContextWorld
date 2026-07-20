#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import (
    validate_context_query_catalog,
)
from contextworld.evaluation.icl_sensitive import (
    build_speed_icl_sensitive_catalog,
    geometry_pair_set,
    sha256_file,
)
from contextworld.paths import resolve_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel


PINNED_STABLEWM = "5864b74980f6ed328fd0045e777b3865962eff43"


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload["status"] != "preregistered_before_execution":
        raise ValueError("Expected a preregistered config")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    _, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    expected_commit = config["frozen_scope"]["stable_worldmodel_commit"]
    if stable_commit != expected_commit:
        raise RuntimeError(
            f"StableWM commit mismatch: {stable_commit} != {expected_commit}"
        )

    artifact_root = resolve_contextworld_path(
        config["artifacts"]["root"], repo_root=REPO_ROOT
    )
    catalog_root = artifact_root / "catalogs"
    payload_root = artifact_root / "payloads"
    distances = [
        int(value)
        for value in config["catalog_generation"]["distance_bins_px"]
    ]
    speeds = tuple(
        float(value)
        for value in config["frozen_scope"]["agent_speeds"]
    )
    door_position = int(config["frozen_scope"]["door_position"])

    catalogs: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, key in (
        ("calibration", "calibration"),
        ("heldout_bank", "heldout_bank"),
    ):
        spec = config["catalog_generation"][key]
        output = resolve_contextworld_path(
            config["artifacts"][
                "calibration_catalog"
                if name == "calibration"
                else "heldout_bank_catalog"
            ],
            repo_root=REPO_ROOT,
        )
        paths[name] = output
        catalogs[name] = build_speed_icl_sensitive_catalog(
            repo_root=REPO_ROOT,
            output_catalog=output,
            payload_root=payload_root / name,
            split=str(spec["split"]),
            distances=distances,
            variants_per_distance=int(spec["variants_per_distance"]),
            geometry_seed=int(spec["geometry_seed"]),
            catalog_seed=int(spec["catalog_seed"]),
            stable_worldmodel_commit=stable_commit,
            speeds=speeds,
            door_position=door_position,
        )

    calibration_pairs = geometry_pair_set(catalogs["calibration"])
    heldout_pairs = geometry_pair_set(catalogs["heldout_bank"])
    overlap = calibration_pairs & heldout_pairs
    calibration_ids = {
        bundle["query_id"] for bundle in catalogs["calibration"]["bundles"]
    }
    heldout_ids = {
        bundle["query_id"] for bundle in catalogs["heldout_bank"]["bundles"]
    }
    query_overlap = calibration_ids & heldout_ids
    if overlap or query_overlap:
        raise RuntimeError(
            f"Calibration/heldout leakage: pairs={len(overlap)}, "
            f"query_ids={len(query_overlap)}"
        )

    referenced_payloads = {
        resolve_contextworld_path(bundle["payload"], repo_root=REPO_ROOT)
        for catalog in catalogs.values()
        for bundle in catalog["bundles"]
    }
    stale_payloads = [
        path
        for path in payload_root.glob("*/*.npz")
        if path.resolve() not in referenced_payloads
    ]
    for path in stale_payloads:
        path.unlink()

    validations = {}
    for name, path in paths.items():
        validations[name] = validate_context_query_catalog(
            path,
            repo_root=REPO_ROOT,
            replay_simulator=not args.skip_simulator_replay,
            family="speed",
        )
        if not validations[name]["passed"]:
            raise RuntimeError(
                f"{name} validation failed: "
                f"{validations[name]['failures'][:5]}"
            )

    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "status": "passed",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "catalogs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "summary": catalogs[name]["summary"],
                "validation": validations[name],
            }
            for name, path in paths.items()
        },
        "disjointness": {
            "calibration_geometry_pairs": len(calibration_pairs),
            "heldout_geometry_pairs": len(heldout_pairs),
            "overlapping_geometry_pairs": len(overlap),
            "overlapping_query_ids": len(query_overlap),
            "passed": not overlap and not query_overlap,
        },
        "payload_hygiene": {
            "referenced_payloads": len(referenced_payloads),
            "stale_payloads_removed": len(stale_payloads),
        },
    }
    report_path = catalog_root / "catalog_build_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build preregistered calibration and heldout ICL-sensitive banks"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT
        / "configs/benchmark/tworoom_speed_icl_sensitive_eval_v1.yaml",
    )
    parser.add_argument("--stablewm-repo", default="../stable-worldmodel")
    parser.add_argument("--stablewm-ref", default=PINNED_STABLEWM)
    parser.add_argument("--skip-simulator-replay", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "report": result["report"],
                "disjointness": result["disjointness"],
                "catalogs": {
                    name: value["summary"]
                    for name, value in result["catalogs"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
