from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.evaluation.door_planning import (
    aggregate_door_records,
    deterministic_cem_seed,
    doorway_crossing,
    load_door_planning_cell,
    simulate_door_candidates,
    summarize_fixed_candidate_selection,
    validate_door_planning_catalog_header,
)
from contextworld.evaluation.icl_model import file_sha256
from contextworld.evaluation.planner_mechanism import (
    array_sha256,
    simulate_tworoom_candidates,
)
from scripts.build_tworoom_door_planning_catalogs import (
    _catalog_benchmark,
    _relative_templates,
    _scripted_actions,
)
from scripts import eval_tworoom_door_fixed_candidates as fixed_evaluator
from scripts import eval_tworoom_door_planning as planning_evaluator


ROOT = Path(__file__).resolve().parents[1]
DOOR_CONFIG = (
    ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
)


def test_planning_catalog_uses_the_evaluator_benchmark_identity() -> None:
    config = yaml.safe_load(DOOR_CONFIG.read_text(encoding="utf-8"))
    assert _catalog_benchmark(config) == config["benchmark"]


def _formal_catalog_header(config: dict) -> dict:
    fixed = config["fixed_candidate_evaluation"]
    planning = config["closed_loop_planning"]
    return {
        "benchmark": config["benchmark"],
        "status": "passed",
        "split_role": "validation",
        "config": {"sha256": file_sha256(DOOR_CONFIG)},
        "stable_worldmodel": {
            "commit": "5864b74980f6ed328fd0045e777b3865962eff43"
        },
        "protocol": {
            "agent_speed": planning["agent_speed"],
            "task": planning["query_task"],
            "history_tokens": config["scope"]["history_tokens"],
            "action_block_raw_steps": config["scope"][
                "action_block_raw_steps"
            ],
            "candidates_per_query": fixed["candidates_per_query"],
            "candidate_horizon_action_blocks": fixed[
                "horizon_action_blocks"
            ],
            "eval_seeds": config["evaluation_data"]["eval_seeds"],
            "queries_per_door_per_eval_seed": planning[
                "evaluations_per_door_per_seed"
            ],
        },
        "summary": {"formal_50_by_6_per_door": True},
    }


def test_direct_evaluator_catalog_header_binds_config_split_status_and_protocol() -> None:
    config = yaml.safe_load(DOOR_CONFIG.read_text(encoding="utf-8"))
    header = _formal_catalog_header(config)
    audit = validate_door_planning_catalog_header(
        header,
        config=config,
        config_sha256=file_sha256(DOOR_CONFIG),
        split="validation",
        run_kind="confirmation",
    )
    assert audit["passed"]
    assert all(audit["checks"].values())

    mutations = []
    stale = copy.deepcopy(header)
    stale["config"]["sha256"] = "stale"
    mutations.append(stale)
    wrong_split = copy.deepcopy(header)
    wrong_split["split_role"] = "sealed_test"
    mutations.append(wrong_split)
    smoke_status = copy.deepcopy(header)
    smoke_status["status"] = "smoke_only"
    mutations.append(smoke_status)
    wrong_protocol = copy.deepcopy(header)
    wrong_protocol["protocol"]["candidates_per_query"] = 299
    mutations.append(wrong_protocol)
    for changed in mutations:
        with pytest.raises(RuntimeError, match="catalog header mismatch"):
            validate_door_planning_catalog_header(
                changed,
                config=config,
                config_sha256=file_sha256(DOOR_CONFIG),
                split="validation",
                run_kind="confirmation",
            )

    smoke = copy.deepcopy(header)
    smoke["status"] = "smoke_only"
    smoke["protocol"]["queries_per_door_per_eval_seed"] = 1
    smoke["summary"]["formal_50_by_6_per_door"] = False
    assert validate_door_planning_catalog_header(
        smoke,
        config=config,
        config_sha256=file_sha256(DOOR_CONFIG),
        split="validation",
        run_kind="smoke",
    )["passed"]


@pytest.mark.parametrize(
    "evaluator",
    [fixed_evaluator, planning_evaluator],
    ids=["fixed_candidates", "closed_loop_cem"],
)
def test_direct_evaluator_rejects_stale_catalog_before_cell_or_model_load(
    tmp_path: Path, monkeypatch, evaluator
) -> None:
    monkeypatch.setenv("CONTEXTWORLD_ARTIFACT_ROOT", str(tmp_path))
    catalog = (
        tmp_path
        / "evaluation/history3/door_visual_generalization_v1"
        / "planning/validation/catalog.json"
    )
    catalog.parent.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(DOOR_CONFIG.read_text(encoding="utf-8"))
    stale = _formal_catalog_header(config)
    stale["config"]["sha256"] = "stale"
    catalog.write_text(json.dumps(stale), encoding="utf-8")
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("stale catalog reached the payload loader")

    monkeypatch.setattr(evaluator, "load_door_planning_cell", forbidden_loader)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            evaluator.__name__,
            "--config",
            str(DOOR_CONFIG),
            "--split",
            "validation",
            "--catalog",
            str(catalog),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--normalizer",
            str(tmp_path / "missing-normalizer.json"),
            "--output",
            str(tmp_path / "unused.json"),
            "--track",
            "validation_seen",
            "--door-position",
            "49",
            "--seed",
            "42",
        ],
    )
    with pytest.raises(RuntimeError, match="catalog header mismatch"):
        evaluator.run(evaluator.parse_args())
    assert not called


def test_doorway_crossing_requires_opening_and_goal_direction() -> None:
    valid = np.asarray(
        [[90.0, 80.0], [110.0, 80.0], [115.0, 80.0], [140.0, 80.0]],
        dtype=np.float32,
    )
    result = doorway_crossing(
        valid,
        door_position=80,
        goal_state=np.asarray([180.0, 80.0], dtype=np.float32),
    )
    assert result == {"crossed": True, "first_crossing_raw_step": 2}

    outside = valid.copy()
    outside[:, 1] = 120.0
    assert not doorway_crossing(
        outside,
        door_position=80,
        goal_state=np.asarray([180.0, 120.0], dtype=np.float32),
    )["crossed"]
    assert not doorway_crossing(
        valid,
        door_position=80,
        goal_state=np.asarray([40.0, 80.0], dtype=np.float32),
    )["crossed"]

    # Exact StableWM endpoint convention: this diagonal transition starts
    # just outside the opening but its proposed endpoint is inside, so the
    # environment accepts the cross-wall step.
    diagonal_endpoint_entry = np.asarray(
        [[110.0, 65.0], [115.0, 64.5]], dtype=np.float32
    )
    assert doorway_crossing(
        diagonal_endpoint_entry,
        door_position=49,
        goal_state=np.asarray([180.0, 49.0], dtype=np.float32),
    )["crossed"]


def test_door_candidate_simulation_matches_established_dynamics() -> None:
    rng = np.random.default_rng(7)
    actions = rng.uniform(-1, 1, size=(8, 20, 2)).astype(np.float32)
    query = np.asarray([75.0, 89.0], dtype=np.float32)
    goal = np.asarray([175.0, 89.0], dtype=np.float32)
    door = simulate_door_candidates(
        query_state=query,
        goal_state=goal,
        raw_actions=actions,
        door_position=89,
    )
    established = simulate_tworoom_candidates(
        query_state=query,
        goal_state=goal,
        raw_actions=actions,
        speed=5.0,
        door_position=89,
    )
    assert np.array_equal(door["final_states"], established["final_states"])
    assert np.array_equal(
        door["final_distances"], established["final_distances"]
    )
    assert np.array_equal(door["success"], established["success"])


def _write_cell(tmp_path: Path) -> Path:
    bundles = []
    for index, (direction, offset) in enumerate(
        (
            ("left_to_right", 0),
            ("left_to_right", 20),
            ("left_to_right", 40),
            ("right_to_left", 0),
            ("right_to_left", 20),
            ("right_to_left", 40),
        )
    ):
        left = direction == "left_to_right"
        query_state = np.asarray(
            [70.0 if left else 154.0, 89.0 + offset], dtype=np.float32
        )
        goal_state = np.asarray(
            [154.0 if left else 70.0, 89.0 + offset], dtype=np.float32
        )
        query_pixels = np.full((8, 8, 3), index, dtype=np.uint8)
        arrays = {
            "query_pixels": query_pixels,
            "goal_pixels": np.full((8, 8, 3), 200 + index, dtype=np.uint8),
            "history_pixels": np.stack(
                [query_pixels + np.uint8(1), query_pixels + np.uint8(2)]
            ),
            "history_actions": np.zeros((2, 5, 2), dtype=np.float32),
            "query_state": query_state,
            "goal_state": goal_state,
            "fixed_candidate_raw_actions": np.full(
                (2, 10, 2), index / 10.0, dtype=np.float32
            ),
        }
        payload_path = tmp_path / f"query-{index}.npz"
        np.savez_compressed(payload_path, **arrays)
        bundle = {
            "query_id": f"query-{index}",
            "track": "validation_seen",
            "task": "cross_room_navigation",
            "eval_seed": 42,
            "evaluation_index": index,
            "query_factors": {"agent.speed": 5.0, "door.position": 89},
            "direction": direction,
            "door_relative_vertical_offset_px": offset,
            "payload": str(payload_path),
            "payload_sha256": file_sha256(payload_path),
        }
        bundle.update(
            {f"{key}_sha256": array_sha256(value) for key, value in arrays.items()}
        )
        bundles.append(bundle)
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps({"schema_version": 1, "bundles": bundles}),
        encoding="utf-8",
    )
    return catalog_path


def test_planning_catalog_cell_validates_pairing_and_hashes(
    tmp_path: Path,
) -> None:
    path = _write_cell(tmp_path)
    cell = load_door_planning_cell(
        path,
        repo_root=tmp_path,
        track="validation_seen",
        door_position=89,
        eval_seed=42,
        candidates=2,
        horizon=2,
        expected_queries=6,
    )
    assert cell.audit["passed"]
    assert cell.audit["unique_query_pixel_hashes"] == 6
    assert set(cell.audit["geometry_strata"].values()) == {1}
    assert len({asset["cem_seed"] for asset in cell.assets}) == 6

    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["bundles"][0]["query_pixels_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    try:
        load_door_planning_cell(
            path,
            repo_root=tmp_path,
            track="validation_seen",
            door_position=89,
            eval_seed=42,
            candidates=2,
            horizon=2,
            expected_queries=6,
        )
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("Corrupt array hash was accepted")


def test_fixed_candidate_and_closed_loop_summary_fields() -> None:
    actions = np.zeros((2, 20, 2), dtype=np.float32)
    actions[0, :, 0] = 1.0
    dynamics = simulate_door_candidates(
        query_state=np.asarray([70.0, 89.0]),
        goal_state=np.asarray([170.0, 89.0]),
        raw_actions=actions,
        door_position=89,
    )
    selected = summarize_fixed_candidate_selection(
        np.asarray([0.0, 1.0]), dynamics
    )
    assert selected["selected_true_success"]
    assert selected["selected_true_doorway_crossing"]
    assert selected["exact_environment_endpoint_regret_px"] == 0.0

    base = {
        "success": True,
        "final_distance_px": 5.0,
        "steps_to_success": 18,
        "doorway_crossing": True,
        "direction": "left_to_right",
        "door_relative_vertical_offset_px": 0,
    }
    aggregate = aggregate_door_records([base, {**base, "success": False,
        "final_distance_px": 30.0, "steps_to_success": None}])
    assert aggregate["success_rate"] == 0.5
    assert aggregate["doorway_crossing_rate"] == 1.0
    assert aggregate["steps_to_success_success_only"]["count"] == 1


def test_cem_seed_does_not_depend_on_model() -> None:
    first = deterministic_cem_seed(
        eval_seed=42, evaluation_index=3, query_id="door-query"
    )
    second = deterministic_cem_seed(
        eval_seed=42, evaluation_index=3, query_id="door-query"
    )
    different = deterministic_cem_seed(
        eval_seed=43, evaluation_index=3, query_id="door-query"
    )
    assert first == second
    assert first != different


def test_formal_direction_offset_schedule_is_balanced_per_seed_and_overall() -> None:
    seeds = [42, 43, 44, 45, 46, 47]
    templates = _relative_templates(seeds, 50)
    overall: dict[tuple[str, int], int] = {}
    for seed in seeds:
        rows = [
            row
            for (row_seed, _), row in templates.items()
            if row_seed == seed
        ]
        assert len(rows) == 50
        assert sum(row["direction"] == "left_to_right" for row in rows) == 25
        assert sum(row["direction"] == "right_to_left" for row in rows) == 25
        counts = {}
        for row in rows:
            key = (
                row["direction"],
                row["door_relative_vertical_offset_px"],
            )
            counts[key] = counts.get(key, 0) + 1
            overall[key] = overall.get(key, 0) + 1
        assert max(counts.values()) - min(counts.values()) == 1
    assert set(overall.values()) == {50}


def test_scripted_oracle_covers_all_22_doors_and_300_queries_per_door() -> None:
    doors = [
        49, 89, 129, 169, 53, 85, 117, 149,
        61, 77, 93, 109, 125, 141, 157, 165,
        24, 32, 40, 180, 190, 199,
    ]
    templates = _relative_templates([42, 43, 44, 45, 46, 47], 50)
    checked = 0
    for door in doors:
        interior_sign = 1.0 if door < 112 else -1.0
        for template in templates.values():
            y = float(
                door
                + interior_sign
                * int(template["door_relative_vertical_offset_px"])
            )
            query = np.asarray(
                [template["query_x"], y], dtype=np.float32
            )
            goal = np.asarray(
                [template["goal_x"], y], dtype=np.float32
            )
            candidate_zero = _scripted_actions(
                query_state=query,
                goal_state=goal,
                door_position=door,
                doorway_y_delta=-12.0,
            )[None]
            result = simulate_door_candidates(
                query_state=query,
                goal_state=goal,
                raw_actions=candidate_zero,
                door_position=door,
            )
            assert result["success"][0], (door, template)
            assert result["doorway_crossed"][0], (door, template)
            assert result["final_distances"][0] == 0.0, (door, template)
            checked += 1
    assert checked == 22 * 300
