#!/usr/bin/env python3
# Hard-gate disjointness audit for the Development Action Delay H7 payload.
# 
# The Development release (built by
# build_tworoom_action_delay_h7_dev_structural_parity_v1.py) must be start
# disjoint from the frozen Public Test catalog: the intersection of the 300
# (room, direction, x, y) start tuples must be empty, and so must the
# coordinate, query-id, template-id, and query-pixel-hash intersections.
# The audit also reopens every Development asset to confirm file hashes,
# the eleven-delay family per query, and the Test stratification (50 queries
# per eval seed, 25/25 rooms, 25/25 directions, every delay seeing all 300
# queries).  Exit status is non-zero when any check fails.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay import array_sha256, canonical_sha256
from contextworld.evaluation.action_delay_h7_validation import (
    ARRAY_KEYS,
    DELAYS,
    file_sha256,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.manifest import write_json

TEST_CATALOG = (
    ROOT
    / "artifacts/evaluation/history7/action_delay_validation_v1/catalog.json"
)
TEST_CATALOG_SHA256 = (
    "5a3fc1a53c05cc4e6f97f7b7544c48aaaf6033550b70a63899eab35ee46fafca"
)
DEV_EVAL_SEEDS = (52, 53, 54, 55, 56, 57)
TEST_EVAL_SEEDS = (42, 43, 44, 45, 46, 47)
GRID = {
    "left_x": range(28, 94, 3),
    "right_x": range(133, 199, 3),
    "up_y": range(30, 147, 3),
    "down_y": range(78, 195, 3),
}

def _start_tuples(
    queries: list[dict[str, Any]],
) -> set[tuple[str, str, float, float]]:
    return {
        (
            str(row["room"]),
            str(row["direction"]),
            float(row["template"]["reset_state"][0]),
            float(row["template"]["reset_state"][1]),
        )
        for row in queries
    }


def _coordinates(queries: list[dict[str, Any]]) -> set[tuple[float, float]]:
    return {
        (
            float(row["template"]["reset_state"][0]),
            float(row["template"]["reset_state"][1]),
        )
        for row in queries
    }


def _grid_membership(dev_queries: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in dev_queries:
        x_position, y_position = (
            float(row["template"]["reset_state"][0]),
            float(row["template"]["reset_state"][1]),
        )
        room = str(row["room"])
        direction = str(row["direction"])
        x_range = GRID["left_x"] if room == "left" else GRID["right_x"]
        y_range = GRID["up_y"] if direction == "up" else GRID["down_y"]
        if int(x_position) not in x_range or int(y_position) not in y_range:
            failures.append(str(row["query_id"]))
    return failures

def run(args: argparse.Namespace) -> dict[str, Any]:
    dev_catalog_path = args.dev_catalog.resolve()
    dev_catalog = json.loads(dev_catalog_path.read_text(encoding="utf-8"))
    if dev_catalog.get("status") != "frozen_before_model_scoring":
        raise ValueError("Development catalog is not frozen")
    protocol = dev_catalog["protocol"]
    dev_queries = list(dev_catalog["queries"])
    if tuple(map(int, protocol["eval_seeds"])) != DEV_EVAL_SEEDS:
        raise ValueError("Development catalog eval seeds are not 52-57")
    if list(map(int, protocol["delay_values"])) != list(DELAYS):
        raise ValueError("Development catalog delays are not 0-10")
    if int(protocol["history_tokens"]) != 7:
        raise ValueError("Development catalog is not History=7")
    test_catalog_path = args.test_catalog.resolve()
    test_sha = file_sha256(test_catalog_path)
    if test_sha != TEST_CATALOG_SHA256:
        raise RuntimeError(f"Public Test catalog sha mismatch: {test_sha}")
    test_catalog = json.loads(test_catalog_path.read_text(encoding="utf-8"))
    test_queries = list(test_catalog["queries"])

    dev_start_tuples = _start_tuples(dev_queries)
    test_start_tuples = _start_tuples(test_queries)
    dev_coordinates = _coordinates(dev_queries)
    test_coordinates = _coordinates(test_queries)
    start_intersection = sorted(
        str(list(value)) for value in dev_start_tuples & test_start_tuples
    )
    coordinate_intersection = sorted(
        str(list(value)) for value in dev_coordinates & test_coordinates
    )
    query_id_intersection = sorted(
        {str(row["query_id"]) for row in dev_queries}
        & {str(row["query_id"]) for row in test_queries}
    )
    template_id_intersection = sorted(
        {str(row["template"]["template_id"]) for row in dev_queries}
        & {str(row["template"]["template_id"]) for row in test_queries}
    )
    pixel_hash_intersection = sorted(
        {str(row["query_pixels_sha256"]) for row in dev_queries}
        & {str(row["query_pixels_sha256"]) for row in test_queries}
    )
    simulator_seed_intersection = sorted(
        str(value)
        for value in {int(row["template"]["simulator_seed"]) for row in dev_queries}
        & {int(row["template"]["simulator_seed"]) for row in test_queries}
    )

    by_seed = Counter(int(row["eval_seed"]) for row in dev_queries)
    by_seed_room = Counter(
        (int(row["eval_seed"]), str(row["room"])) for row in dev_queries
    )
    by_seed_direction = Counter(
        (int(row["eval_seed"]), str(row["direction"])) for row in dev_queries
    )
    delay_counts: Counter = Counter()
    asset_failures: list[str] = []
    for row in dev_queries:
        asset_path = resolve_contextworld_path(
            str(row["asset"]), repo_root=ROOT
        )
        if not asset_path.is_file():
            asset_failures.append(str(row["query_id"]) + ":missing")
            continue
        if file_sha256(asset_path) != row["asset_sha256"]:
            asset_failures.append(str(row["query_id"]) + ":sha")
            continue
        with np.load(asset_path, allow_pickle=False) as payload:
            arrays = {name: payload[name].copy() for name in payload.files}
        if tuple(sorted(arrays)) != tuple(sorted(ARRAY_KEYS)):
            asset_failures.append(str(row["query_id"]) + ":keys")
            continue
        hashes = {
            name: array_sha256(value) for name, value in sorted(arrays.items())
        }
        if hashes != row["array_sha256"] or canonical_sha256(hashes) != row[
            "payload_sha256"
        ]:
            asset_failures.append(str(row["query_id"]) + ":payload")
            continue
        history = np.asarray(arrays["history_delays"], dtype=np.int64)
        targets = np.asarray(arrays["target_delays"], dtype=np.int64)
        if not np.array_equal(history, np.asarray(DELAYS)) or not np.array_equal(
            targets, np.asarray(DELAYS)
        ):
            asset_failures.append(str(row["query_id"]) + ":delay_family")
            continue
        for delay in DELAYS:
            delay_counts[int(delay)] += 1
    grid_failures = _grid_membership(dev_queries)
    checks = {
        "exact_300_dev_queries": len(dev_queries) == 300,
        "exact_300_test_queries": len(test_queries) == 300,
        "start_tuples_disjoint": not start_intersection,
        "coordinates_disjoint": not coordinate_intersection,
        "query_ids_disjoint": not query_id_intersection,
        "template_ids_disjoint": not template_id_intersection,
        "query_pixel_hashes_disjoint": not pixel_hash_intersection,
        "simulator_seeds_disjoint": not simulator_seed_intersection,
        "eval_seeds_are_52_through_57": tuple(sorted(by_seed))
        == DEV_EVAL_SEEDS,
        "no_test_eval_seed_present": not (
            set(by_seed) & set(TEST_EVAL_SEEDS)
        ),
        "fifty_queries_per_eval_seed": all(
            count == 50 for count in by_seed.values()
        ) and len(by_seed) == 6,
        "rooms_25_25_per_eval_seed": all(
            by_seed_room[(seed, "left")] == 25
            and by_seed_room[(seed, "right")] == 25
            for seed in DEV_EVAL_SEEDS
        ),
        "directions_25_25_per_eval_seed": all(
            by_seed_direction[(seed, "up")] == 25
            and by_seed_direction[(seed, "down")] == 25
            for seed in DEV_EVAL_SEEDS
        ),
        "every_delay_sees_all_300_queries": delay_counts
        == Counter({int(delay): 300 for delay in DELAYS}),
        "every_asset_reopens_with_exact_hashes": not asset_failures,
        "starts_on_declared_grid": not grid_failures,
    }
    report = {
        "schema_version": 1,
        "benchmark": dev_catalog.get("benchmark"),
        "audit": "tworoom_action_delay_h7_dev_disjointness_v1",
        "dev_catalog": portable_contextworld_path(
            dev_catalog_path, repo_root=ROOT
        ),
        "dev_catalog_sha256": file_sha256(dev_catalog_path),
        "test_catalog": portable_contextworld_path(
            test_catalog_path, repo_root=ROOT
        ),
        "test_catalog_sha256": test_sha,
        "checks": checks,
        "passed": all(checks.values()),
        "intersections": {
            "start_tuples": start_intersection,
            "coordinates": coordinate_intersection,
            "query_ids": query_id_intersection,
            "template_ids": template_id_intersection,
            "query_pixels_sha256": pixel_hash_intersection,
            "simulator_seeds": simulator_seed_intersection,
        },
        "counts": {
            "dev_queries_by_eval_seed": {
                str(seed): by_seed[seed] for seed in sorted(by_seed)
            },
            "dev_queries_room": {
                f"s{seed}/{room}": by_seed_room[(seed, room)]
                for seed in sorted(by_seed)
                for room in ("left", "right")
            },
            "dev_queries_direction": {
                f"s{seed}/{direction}": by_seed_direction[(seed, direction)]
                for seed in sorted(by_seed)
                for direction in ("up", "down")
            },
            "dev_queries_per_delay": {
                str(delay): delay_counts[int(delay)] for delay in DELAYS
            },
            "dev_distinct_start_tuples": len(dev_start_tuples),
            "test_distinct_start_tuples": len(test_start_tuples),
            "asset_failures": asset_failures,
            "grid_failures": grid_failures,
        },
    }
    if args.output is not None:
        write_json(Path(args.output).resolve(), report)
    return report

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Development Action Delay H7 payload for start-point "
            "disjointness from the Public Test and Test-structure counts"
        ),
    )
    parser.add_argument(
        "--dev-catalog",
        type=Path,
        default=(
            ROOT
            / "artifacts/evaluation/history7/"
            / "action_delay_dev_structural_parity_v1/catalog.json"
        ),
    )
    parser.add_argument("--test-catalog", type=Path, default=TEST_CATALOG)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "failed_checks": [
                    name
                    for name, value in result["checks"].items()
                    if not value
                ],
                "counts": result["counts"],
                "intersections": {
                    name: len(values)
                    for name, values in result["intersections"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
