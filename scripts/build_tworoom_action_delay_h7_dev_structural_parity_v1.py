#!/usr/bin/env python3
# Build the Public-Test-structured Development Action Delay H7 release.
# 
# configs/benchmark/tworoom_action_delay_h7_dev_structural_parity_v1.yaml is
# field-identical to the frozen Public Test config tworoom_action_delay_h7_v1.yaml
# except for the two sampling identities (eval seeds 52-57 instead of 42-47 and
# catalog seed 20260910 instead of 20260728) and the split label.  The frozen
# builder module contextworld.evaluation.action_delay_h7_validation hardcodes
# the Test eval seeds, so this entry point rebinds the module constants at
# runtime (the frozen file itself is never modified) and adds the one step the
# Test build did not need: every Public Test start coordinate is removed from
# the candidate pools before selection, so the Development queries are start
# disjoint from the Public Test by construction.  Before use, the copied
# selector is proven identical to the frozen selector on an empty exclusion
# set, and the config is machine-diffed against the Test config section by
# section.

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import contextworld.evaluation.action_delay_h7_validation as validation
from contextworld.paths import portable_contextworld_path
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

TEST_CONFIG = ROOT / "configs/benchmark/tworoom_action_delay_h7_v1.yaml"
TEST_CATALOG = (
    ROOT
    / "artifacts/evaluation/history7/action_delay_validation_v1/catalog.json"
)
TEST_CATALOG_SHA256 = (
    "5a3fc1a53c05cc4e6f97f7b7544c48aaaf6033550b70a63899eab35ee46fafca"
)
DEV_STATUS = "development_structural_parity_candidate"
DEV_EVAL_SEEDS = (52, 53, 54, 55, 56, 57)
DEV_CATALOG_SEED = 20260910
DEV_SPLIT = "development_structural_parity"
STRUCTURAL_SECTIONS = ("environment", "history_protocol", "scoring")
STRUCTURAL_VALIDATION_KEYS = (
    "queries_per_seed",
    "independent_queries_per_delay",
    "delay_values",
    "tracks",
    "future_horizons_action_blocks",
    "future_actions",
    "offline_true_futures",
    "online_environment_during_model_scoring",
    "query_geometry",
)


def select_dev_validation_assignments(
    config: dict[str, Any],
    *,
    excluded_coordinates: set[tuple[float, float]],
) -> list[validation.ActionDelayH7ValidationAssignment]:
    """Frozen selector copy whose coordinate pool excludes the Test starts."""

    module = validation
    module._validate_protocol(config)
    validation_section = config["validation"]
    geometry = validation_section["query_geometry"]
    catalog_seed = int(validation_section["catalog_seed"])
    x_by_room = {
        "left": module._range_from_specification(geometry["left_x"]),
        "right": module._range_from_specification(geometry["right_x"]),
    }
    y_by_direction = {
        "up": module._range_from_specification(geometry["up_y"]),
        "down": module._range_from_specification(geometry["down_y"]),
    }
    expected_rooms = {
        str(key): int(value)
        for key, value in geometry["rooms_per_seed"].items()
    }
    expected_directions = {
        str(key): int(value)
        for key, value in geometry["directions_per_seed"].items()
    }
    if expected_rooms != {"left": 25, "right": 25}:
        raise ValueError("Each Eval seed must contain 25 queries per room")
    if expected_directions != {"up": 25, "down": 25}:
        raise ValueError("Each Eval seed must contain 25 queries per direction")

    selected: list[validation.ActionDelayH7ValidationAssignment] = []
    used_coordinates: set[tuple[float, float]] = set(excluded_coordinates)
    used_simulator_seeds: set[int] = set()
    for eval_seed in module.EVAL_SEEDS:
        assignment_rng = np.random.default_rng(
            np.random.SeedSequence([catalog_seed, eval_seed, 0xA7D])
        )
        rooms = module._balanced_values(
            "left",
            "right",
            count=module.QUERIES_PER_SEED,
            rng=assignment_rng,
        )
        directions = module._balanced_values(
            "up",
            "down",
            count=module.QUERIES_PER_SEED,
            rng=assignment_rng,
        )
        candidate_pools: dict[tuple[str, str], list[tuple[int, int]]] = {}
        cursors: Counter = Counter()
        for room in ("left", "right"):
            for direction in ("up", "down"):
                values = [
                    (x_position, y_position)
                    for x_position in x_by_room[room]
                    for y_position in y_by_direction[direction]
                ]
                pool_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            catalog_seed,
                            eval_seed,
                            0 if room == "left" else 1,
                            0 if direction == "up" else 1,
                        ]
                    )
                )
                permutation = pool_rng.permutation(len(values))
                candidate_pools[(room, direction)] = [
                    values[index] for index in permutation
                ]

        for evaluation_index, (room, direction) in enumerate(
            zip(rooms, directions, strict=True)
        ):
            key = (room, direction)
            pool = candidate_pools[key]
            while True:
                cursor = cursors[key]
                if cursor >= len(pool):
                    raise RuntimeError(
                        f"Exhausted H7 Validation coordinates for {key}"
                    )
                x_position, y_position = pool[cursor]
                cursors[key] += 1
                coordinate = (float(x_position), float(y_position))
                if coordinate not in used_coordinates:
                    break
            used_coordinates.add(coordinate)
            simulator_seed = int(
                np.random.SeedSequence(
                    [catalog_seed, eval_seed, evaluation_index, 0x51A]
                ).generate_state(1)[0]
            )
            if simulator_seed in used_simulator_seeds:
                raise RuntimeError("H7 Validation simulator seed repeated")
            used_simulator_seeds.add(simulator_seed)
            goal_state = (
                (190.0, 200.0 if y_position < 112 else 24.0)
                if room == "left"
                else (30.0, 200.0 if y_position < 112 else 24.0)
            )
            query_id = (
                f"action-delay-h7-val-s{eval_seed}-q"
                f"{evaluation_index:02d}"
            )
            selected.append(
                validation.ActionDelayH7ValidationAssignment(
                    query_id=query_id,
                    eval_seed=eval_seed,
                    evaluation_index=evaluation_index,
                    room=room,
                    template=validation.LongHistoryDelayTemplate(
                        template_id=query_id,
                        direction=direction,
                        reset_state=coordinate,
                        goal_state=goal_state,
                        simulator_seed=simulator_seed,
                    ),
                )
            )
    if len(selected) != module.QUERY_COUNT:
        raise RuntimeError(
            f"Expected {module.QUERY_COUNT} queries, got {len(selected)}"
        )
    if len(used_coordinates) - len(excluded_coordinates) != module.QUERY_COUNT:
        raise RuntimeError("H7 Validation query coordinates repeat")
    return selected


def structural_parity_report(
    dev: dict[str, Any], test: dict[str, Any]
) -> dict[str, Any]:
    problems: list[str] = []
    for section in STRUCTURAL_SECTIONS:
        if dev.get(section) != test.get(section):
            problems.append(section)
    dev_validation = dev["validation"]
    test_validation = test["validation"]
    for key in STRUCTURAL_VALIDATION_KEYS:
        if dev_validation.get(key) != test_validation.get(key):
            problems.append("validation." + str(key))
    if dev_validation.get("eval_seeds") == test_validation.get("eval_seeds"):
        problems.append("validation.eval_seeds must differ from the Test")
    if dev_validation.get("catalog_seed") == test_validation.get(
        "catalog_seed"
    ):
        problems.append("validation.catalog_seed must differ from the Test")
    return {
        "compared_sections": list(STRUCTURAL_SECTIONS),
        "compared_validation_keys": list(STRUCTURAL_VALIDATION_KEYS),
        "differences": problems,
        "passed": not problems,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("status") != DEV_STATUS:
        raise ValueError("Development config status is not the dev candidate")
    validation_section = config["validation"]
    if tuple(map(int, validation_section["eval_seeds"])) != DEV_EVAL_SEEDS:
        raise ValueError("Development eval seeds are not 52-57")
    if int(validation_section["catalog_seed"]) != DEV_CATALOG_SEED:
        raise ValueError("Development catalog seed is not 20260910")
    if str(validation_section["split"]) != DEV_SPLIT:
        raise ValueError("Development split label is unexpected")
    test_config = yaml.safe_load(TEST_CONFIG.read_text(encoding="utf-8"))
    parity = structural_parity_report(config, test_config)
    if not parity["passed"]:
        raise RuntimeError(f"Config is not Test-structured: {parity}")

    test_catalog_path = args.test_catalog.resolve()
    test_sha = validation.file_sha256(test_catalog_path)
    if test_sha != TEST_CATALOG_SHA256:
        raise RuntimeError(f"Public Test catalog sha mismatch: {test_sha}")
    test_catalog = json.loads(test_catalog_path.read_text(encoding="utf-8"))
    test_queries = list(test_catalog["queries"])
    if len(test_queries) != 300:
        raise RuntimeError("Public Test catalog does not hold 300 queries")
    test_coordinates = _coordinates(test_queries)
    test_start_tuples = _start_tuples(test_queries)

    swm, stable_repo, stable_commit = load_stable_worldmodel(
        ROOT,
        str(args.stablewm_repo),
        str(config["stable_worldmodel"]["commit"]),
    )

    frozen_seeds = validation.EVAL_SEEDS
    validation.EVAL_SEEDS = DEV_EVAL_SEEDS
    try:
        identity_check = select_dev_validation_assignments(
            config, excluded_coordinates=set()
        )
        frozen_check = validation.select_validation_assignments(config)
        if identity_check != frozen_check:
            raise RuntimeError(
                "Copied selector diverges from the frozen selector"
            )
        assignments = select_dev_validation_assignments(
            config, excluded_coordinates=set(test_coordinates)
        )
    finally:
        validation.EVAL_SEEDS = frozen_seeds
    assignment_coordinates = {
        tuple(float(value) for value in item.template.reset_state)
        for item in assignments
    }
    reused = sorted(
        str(list(value))
        for value in assignment_coordinates & test_coordinates
    )
    if reused:
        raise RuntimeError(f"Exclusion failed: {reused[:5]}")

    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    builder_config = dict(config)
    builder_config["stable_worldmodel"] = {
        **config["stable_worldmodel"],
        "repo": str(stable_repo),
    }
    frozen_selector = validation.select_validation_assignments
    validation.EVAL_SEEDS = DEV_EVAL_SEEDS
    validation.select_validation_assignments = lambda builder_cfg: (
        select_dev_validation_assignments(
            builder_cfg, excluded_coordinates=set(test_coordinates)
        )
    )
    try:
        catalog, exclusion_manifest, report = validation.build_validation_release(
            config=builder_config,
            repo_root=ROOT,
            output_root=output_root,
            workers=int(args.workers),
            stable_worldmodel_commit=stable_commit,
        )
    finally:
        validation.EVAL_SEEDS = frozen_seeds
        validation.select_validation_assignments = frozen_selector

    dev_queries = list(catalog["queries"])
    dev_start_tuples = _start_tuples(dev_queries)
    dev_coordinates = _coordinates(dev_queries)
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
    pixel_hash_intersection = sorted(
        {str(row["query_pixels_sha256"]) for row in dev_queries}
        & {str(row["query_pixels_sha256"]) for row in test_queries}
    )
    by_seed = Counter(int(row["eval_seed"]) for row in dev_queries)
    by_seed_room = Counter(
        (int(row["eval_seed"]), str(row["room"])) for row in dev_queries
    )
    by_seed_direction = Counter(
        (int(row["eval_seed"]), str(row["direction"])) for row in dev_queries
    )
    strata_ok = (
        len(by_seed) == 6
        and all(count == 50 for count in by_seed.values())
        and all(
            by_seed_room[(seed, "left")] == 25
            and by_seed_room[(seed, "right")] == 25
            and by_seed_direction[(seed, "up")] == 25
            and by_seed_direction[(seed, "down")] == 25
            for seed in sorted(by_seed)
        )
    )
    disjointness = {
        "test_catalog": portable_contextworld_path(
            test_catalog_path, repo_root=ROOT
        ),
        "test_catalog_sha256": test_sha,
        "test_queries": len(test_queries),
        "dev_queries": len(dev_queries),
        "start_tuple_intersection": start_intersection,
        "coordinate_intersection": coordinate_intersection,
        "query_id_intersection": query_id_intersection,
        "query_pixels_sha256_intersection": pixel_hash_intersection,
        "excluded_test_coordinates": len(test_coordinates),
        "counts_by_eval_seed": {
            str(seed): by_seed[seed] for seed in sorted(by_seed)
        },
        "counts_room": {
            f"s{seed}/{room}": by_seed_room[(seed, room)]
            for seed in sorted(by_seed)
            for room in ("left", "right")
        },
        "counts_direction": {
            f"s{seed}/{direction}": by_seed_direction[(seed, direction)]
            for seed in sorted(by_seed)
            for direction in ("up", "down")
        },
        "stratification_matches_test_structure": bool(strata_ok),
    }
    disjointness["passed"] = (
        not start_intersection
        and not coordinate_intersection
        and not query_id_intersection
        and not pixel_hash_intersection
        and strata_ok
        and report["audit"]["passed"]
    )
    identity = {
        "config": {
            "path": portable_contextworld_path(config_path, repo_root=ROOT),
            "sha256": validation.file_sha256(config_path),
        },
        "test_config": {
            "path": portable_contextworld_path(TEST_CONFIG, repo_root=ROOT),
            "sha256": validation.file_sha256(TEST_CONFIG),
        },
        "stable_worldmodel": {"repo": str(stable_repo), "commit": stable_commit},
        "selector_identity_check_passed": True,
    }
    summary = {
        "schema_version": 1,
        "benchmark": config["benchmark"],
        "split": DEV_SPLIT,
        "status": "passed" if disjointness["passed"] else "failed",
        "structural_parity": parity,
        "disjointness_audit": disjointness,
        "identity": identity,
        "builder_report": report,
        "output_root": portable_contextworld_path(output_root, repo_root=ROOT),
    }
    write_json(
        output_root / "dev_structural_parity_build_report.json", summary
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Development Action Delay H7 release with Public Test "
            "structure and start-point isolation"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "configs/benchmark/tworoom_action_delay_h7_dev_structural_parity_v1.yaml"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stablewm-repo", type=Path, required=True)
    parser.add_argument("--test-catalog", type=Path, default=TEST_CATALOG)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_root": result["output_root"],
                "disjointness_passed": result["disjointness_audit"]["passed"],
                "start_tuple_intersection": len(
                    result["disjointness_audit"]["start_tuple_intersection"]
                ),
                "counts_by_eval_seed": result["disjointness_audit"][
                    "counts_by_eval_seed"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
