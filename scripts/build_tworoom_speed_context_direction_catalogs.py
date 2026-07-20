#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contextworld.evaluation.icl_catalog import (
    validate_context_query_catalog,
)
from contextworld.evaluation.icl_sensitive import (
    SensitiveGeometry,
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


def _geometry_from_json(payload: dict[str, Any]) -> SensitiveGeometry:
    return SensitiveGeometry(
        template_id=str(payload["template_id"]),
        distance_bin=int(payload["distance_bin"]),
        geometry_variant=int(payload["geometry_variant"]),
        reset_state=tuple(float(value) for value in payload["reset_state"]),
        goal_state=tuple(float(value) for value in payload["goal_state"]),
        context_direction=tuple(
            float(value) for value in payload["context_direction"]
        ),
        query_action=tuple(
            float(value) for value in payload["query_action"]
        ),
    )


def _load_array_payload(
    bundle: dict[str, Any],
) -> dict[str, np.ndarray]:
    path = resolve_contextworld_path(bundle["payload"], repo_root=REPO_ROOT)
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
        }


def _cross_catalog_audit(
    slow: dict[str, Any],
    fast: dict[str, Any],
    *,
    slow_speed: float,
    fast_speed: float,
) -> dict[str, Any]:
    slow_bundles = {
        str(bundle["query_id"]): bundle for bundle in slow["bundles"]
    }
    fast_bundles = {
        str(bundle["query_id"]): bundle for bundle in fast["bundles"]
    }
    if len(slow_bundles) != len(slow["bundles"]):
        raise RuntimeError("Duplicate query IDs in wrong-slower catalog")
    if len(fast_bundles) != len(fast["bundles"]):
        raise RuntimeError("Duplicate query IDs in wrong-faster catalog")
    if slow_bundles.keys() != fast_bundles.keys():
        raise RuntimeError("Directional catalogs have different query IDs")

    identical_query_payloads = 0
    identical_correct_contexts = 0
    wrong_contexts_differ = 0
    for query_id in sorted(slow_bundles):
        slow_bundle = slow_bundles[query_id]
        fast_bundle = fast_bundles[query_id]
        for key in (
            "query_id",
            "simulator_seed",
            "template",
            "query_factors",
            "source_manifest_fingerprint",
        ):
            if slow_bundle[key] != fast_bundle[key]:
                raise RuntimeError(
                    f"{query_id}: catalog field differs: {key}"
                )
        if (
            slow_bundle["conditions"]["correct"]
            != fast_bundle["conditions"]["correct"]
        ):
            raise RuntimeError(f"{query_id}: correct context metadata differs")
        observed_slow = float(
            slow_bundle["conditions"]["wrong"]["factors"]["agent.speed"]
        )
        observed_fast = float(
            fast_bundle["conditions"]["wrong"]["factors"]["agent.speed"]
        )
        if not np.isclose(
            observed_slow, slow_speed, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"{query_id}: wrong-slower speed mismatch")
        if not np.isclose(
            observed_fast, fast_speed, rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(f"{query_id}: wrong-faster speed mismatch")

        slow_payload = _load_array_payload(slow_bundle)
        fast_payload = _load_array_payload(fast_bundle)
        query_keys = (
            "query_pixels",
            "query_action",
            "query_state",
            "target_pixels",
            "target_state",
        )
        correct_keys = tuple(
            key
            for key in slow_payload
            if key.startswith("context_b")
            and "_correct_" in key
        )
        if not all(
            np.array_equal(slow_payload[key], fast_payload[key])
            for key in query_keys
        ):
            raise RuntimeError(f"{query_id}: query payload differs")
        identical_query_payloads += 1
        if not correct_keys or not all(
            np.array_equal(slow_payload[key], fast_payload[key])
            for key in correct_keys
        ):
            raise RuntimeError(f"{query_id}: correct context payload differs")
        identical_correct_contexts += 1

        wrong_keys = tuple(
            key
            for key in slow_payload
            if key.startswith("context_b")
            and "_wrong_" in key
            and key.endswith(("pixels", "next_pixels", "states", "next_states"))
        )
        if not wrong_keys or not any(
            not np.array_equal(slow_payload[key], fast_payload[key])
            for key in wrong_keys
        ):
            raise RuntimeError(
                f"{query_id}: slow and fast wrong contexts are identical"
            )
        wrong_contexts_differ += 1

    return {
        "passed": True,
        "query_ids_identical": True,
        "query_id_count": len(slow_bundles),
        "query_payloads_identical": identical_query_payloads,
        "correct_context_payloads_identical": identical_correct_contexts,
        "wrong_context_payloads_different": wrong_contexts_differ,
        "simulator_seeds_identical": True,
        "wrong_slow_speed": float(slow_speed),
        "wrong_fast_speed": float(fast_speed),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = _load_config(config_path)
    frozen = config["frozen_scope"]
    artifacts = config["artifacts"]
    artifact_root = resolve_contextworld_path(
        artifacts["root"], repo_root=REPO_ROOT
    )
    raw_root = resolve_contextworld_path(
        artifacts["raw_results"], repo_root=REPO_ROOT
    )
    existing_scores = sorted(raw_root.glob("*.json"))
    if existing_scores:
        raise RuntimeError(
            "Refusing to rebuild preregistered catalogs after score files "
            f"exist: {existing_scores[:3]}"
        )

    _, stable_repo, stable_commit = load_stable_worldmodel(
        REPO_ROOT, args.stablewm_repo, args.stablewm_ref
    )
    if stable_commit != str(frozen["stable_worldmodel_commit"]):
        raise RuntimeError(
            "StableWM commit mismatch: "
            f"{stable_commit} != {frozen['stable_worldmodel_commit']}"
        )

    source_path = resolve_contextworld_path(
        frozen["source_heldout_bank"], repo_root=REPO_ROOT
    )
    source_hash = sha256_file(source_path)
    expected_source_hash = str(frozen["source_heldout_bank_sha256"])
    if source_hash != expected_source_hash:
        raise RuntimeError(
            f"Source heldout bank hash mismatch: {source_hash}"
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if int(source["geometry_seed"]) != int(frozen["geometry_seed"]):
        raise RuntimeError("Source heldout geometry seed mismatch")

    distance_bins = [int(value) for value in frozen["distance_bins_px"]]
    distance_set = set(distance_bins)
    geometries = [
        _geometry_from_json(row)
        for row in source["geometry_bank"]
        if int(row["distance_bin"]) in distance_set
    ]
    expected_geometries = (
        len(distance_bins) * int(frozen["variants_per_distance"])
    )
    if len(geometries) != expected_geometries:
        raise RuntimeError(
            "Unexpected source heldout geometry count: "
            f"{len(geometries)} != {expected_geometries}"
        )
    source_subset_pairs = {
        tuple(
            np.round(
                np.asarray(
                    [*row.reset_state, *row.goal_state],
                    dtype=np.float64,
                ),
                6,
            ).tolist()
        )
        for row in geometries
    }

    payload_root = artifact_root / "payloads"
    evaluation_specs = {
        str(spec["name"]): spec
        for spec in config["catalogs"]["evaluations"]
    }
    paths = {
        "wrong_slow": resolve_contextworld_path(
            artifacts["wrong_slow_catalog"], repo_root=REPO_ROOT
        ),
        "wrong_fast": resolve_contextworld_path(
            artifacts["wrong_fast_catalog"], repo_root=REPO_ROOT
        ),
    }
    catalogs: dict[str, dict[str, Any]] = {}
    for name in ("wrong_slow", "wrong_fast"):
        wrong_speed = float(
            evaluation_specs[name]["wrong_context_speed"]
        )
        catalogs[name] = build_speed_icl_sensitive_catalog(
            repo_root=REPO_ROOT,
            output_catalog=paths[name],
            payload_root=payload_root / name,
            split=str(config["catalogs"]["split"]),
            distances=distance_bins,
            variants_per_distance=int(frozen["variants_per_distance"]),
            geometry_seed=int(frozen["geometry_seed"]),
            catalog_seed=int(frozen["catalog_seed"]),
            stable_worldmodel_commit=stable_commit,
            speeds=tuple(float(value) for value in frozen["query_speeds"]),
            door_position=int(frozen["door_position"]),
            wrong_speed_override=wrong_speed,
            benchmark_name=str(config["benchmark"]),
            track_name="T1_speed_context_direction_confirmation",
            protocol_name="tworoom_speed_context_direction_v2",
            regime="same_room_heldout_direction_confirmation",
            geometries_override=geometries,
        )
        if geometry_pair_set(catalogs[name]) != source_subset_pairs:
            raise RuntimeError(
                f"{name} does not exactly reuse the source heldout geometries"
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

    validations: dict[str, dict[str, Any]] = {}
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

    cross_audit = _cross_catalog_audit(
        catalogs["wrong_slow"],
        catalogs["wrong_fast"],
        slow_speed=float(frozen["wrong_slow_speed"]),
        fast_speed=float(frozen["wrong_fast_speed"]),
    )
    report = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "stage": "directional_catalog_build_before_scoring",
        "status": "passed",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "commit": stable_commit,
        },
        "source_heldout_bank": {
            "path": str(source_path),
            "sha256": source_hash,
            "selected_distance_bins": distance_bins,
            "selected_geometry_pairs": len(source_subset_pairs),
            "previously_scored": False,
        },
        "catalogs": {
            name: {
                "path": str(paths[name]),
                "sha256": sha256_file(paths[name]),
                "summary": catalogs[name]["summary"],
                "validation": validations[name],
            }
            for name in ("wrong_slow", "wrong_fast")
        },
        "cross_catalog_audit": cross_audit,
        "count_audit": {
            "expected_base_queries_per_eval": int(
                config["formal_eval"]["expected_unique_base_queries_per_eval"]
            ),
            "observed_base_queries_per_eval": {
                name: len(catalog["bundles"])
                for name, catalog in catalogs.items()
            },
            "expected_evaluations_per_condition_per_eval": int(
                config["formal_eval"][
                    "expected_evaluations_per_condition_per_eval"
                ]
            ),
            "evaluations_per_condition_per_seed": int(
                config["formal_eval"][
                    "evaluations_per_condition_per_seed"
                ]
            ),
            "eval_seeds": [
                int(value)
                for value in config["formal_eval"]["eval_seeds"]
            ],
        },
        "payload_hygiene": {
            "referenced_payloads": len(referenced_payloads),
            "stale_payloads_removed": len(stale_payloads),
        },
    }
    report_path = resolve_contextworld_path(
        artifacts["catalog_build_report"], repo_root=REPO_ROOT
    )
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the preregistered wrong-slower and wrong-faster heldout "
            "catalogs before any model score is produced"
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            REPO_ROOT
            / "configs/benchmark/"
            "tworoom_speed_context_direction_eval_v2.yaml"
        ),
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
                "source_heldout_bank": result["source_heldout_bank"],
                "cross_catalog_audit": result["cross_catalog_audit"],
                "catalogs": {
                    name: value["summary"]
                    for name, value in result["catalogs"].items()
                },
                "count_audit": result["count_audit"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
