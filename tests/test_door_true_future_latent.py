from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.evaluation.door_visual import (
    HORIZONS,
    INPUT_CONDITIONS,
    TASKS,
    assign_eval_partitions,
    checkpoint_cell_summary,
    door_support_audit,
    formal_template_assignments,
    future_actions,
    longest_contiguous_horizon,
    make_query_geometry,
    natural_history_actions,
    paired_normalized_effects,
)
from scripts.analyze_tworoom_door_visual_generalization import TRAINING_BINDINGS
from scripts.analyze_tworoom_door_visual_generalization import _audit_runtime_identity
from scripts.analyze_tworoom_door_visual_generalization import (
    _audit_training_report_bindings,
)
from scripts.analyze_tworoom_door_visual_generalization import _formal_decision
from scripts.analyze_tworoom_door_visual_generalization import _load_results
from scripts.build_tworoom_door_visual_catalogs import _prediction_payload_views
from scripts.build_tworoom_door_visual_catalogs import _task_oracle
from scripts.build_tworoom_door_visual_catalogs import _selected_track_rows
from scripts.eval_tworoom_door_true_future_latent import _rollout


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmark/tworoom_door_visual_generalization_v1.yaml"
EVAL_SEEDS = [42, 43, 44, 45, 46, 47]


def test_frozen_door_validation_contract_and_counts() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    evaluation = config["evaluation_data"]
    audit = door_support_audit(config)
    assert audit["passed"]
    assert evaluation["eval_seeds"] == EVAL_SEEDS
    assert evaluation["unique_queries_per_door_per_task_per_seed"] == 50
    assert evaluation["unique_queries_per_door_per_task"] == 300
    assert evaluation["future_horizons_action_blocks"] == list(HORIZONS)
    assert tuple(evaluation["offline_prediction_tasks"]) == TASKS
    assert tuple(evaluation["input_conditions"]) == INPUT_CONDITIONS
    assert evaluation["validation_counts_per_checkpoint"] == {
        "door_positions": 8,
        "prediction_tasks": 2,
        "input_conditions": 2,
        "cells": 32,
        "scored_sequences": 9600,
        "horizon_losses": 38400,
    }
    assert evaluation["input_conditions"]["query_only"][
        "primary_decision_gate"
    ] is False
    assert evaluation["input_conditions"]["natural_history3"][
        "primary_decision_gate"
    ] is True
    assert config["prediction_metrics"]["raw_latent_mse_cross_checkpoint_ranking"] == (
        "forbidden"
    )


def test_door_query_geometry_and_actions_are_task_diagnostic() -> None:
    passage = make_query_geometry(
        door_position=89,
        task="doorway_passage",
        direction="left_to_right",
        template_index=7,
        seed=123,
    )
    contact = make_query_geometry(
        door_position=89,
        task="wall_contact",
        direction="left_to_right",
        template_index=7,
        seed=123,
    )
    assert abs(passage.query_state[1] - 89) <= 5.0
    assert abs(contact.query_state[1] - 89) >= 22.0
    history = natural_history_actions(passage)
    future = future_actions("left_to_right")
    assert history.shape == (2, 5, 2)
    assert future.shape == (5, 5, 2)
    assert np.array_equal(history[1], -history[0])
    assert np.max(np.abs(history)) == pytest.approx(0.25)
    assert np.max(np.abs(future)) == pytest.approx(0.5)
    assert np.all(future[..., 0] > 0)
    assert np.all(future[..., 1] == 0)


def test_task_outcome_oracle_distinguishes_passage_and_wall_contact() -> None:
    passage = make_query_geometry(
        door_position=89,
        task="doorway_passage",
        direction="left_to_right",
        template_index=0,
        seed=1,
    )
    contact = make_query_geometry(
        door_position=89,
        task="wall_contact",
        direction="left_to_right",
        template_index=0,
        seed=1,
    )
    passage_rollout = {
        "next_states": np.asarray(
            [[0, 0], [0, 0], [105, 89], [115, 89], [125, 89], [140, 89], [152, 89]],
            dtype=np.float32,
        )
    }
    contact_rollout = {
        "next_states": np.asarray(
            [[0, 0], [0, 0], [99.5, 120], [99.5, 120], [99.5, 120], [99.5, 120], [99.5, 120]],
            dtype=np.float32,
        )
    }
    assert _task_oracle(passage, passage_rollout)["passed"]
    assert _task_oracle(contact, contact_rollout)["passed"]


def test_actual_third_history_frame_is_query_despite_tiny_return_drift() -> None:
    first = np.zeros((4, 4, 3), dtype=np.uint8)
    middle = np.full((4, 4, 3), 2, dtype=np.uint8)
    actual_third = first.copy()
    actual_third[0, 0, 0] = 1
    future_1 = np.full((4, 4, 3), 3, dtype=np.uint8)
    future_2 = np.full((4, 4, 3), 4, dtype=np.uint8)
    rollout = {
        "pixels": np.stack([first, middle, actual_third, future_1]),
        "next_pixels": np.stack([middle, actual_third, future_1, future_2]),
        "states": np.asarray(
            [[10.0, 20.0], [10.0, 21.0], [10.0, 20.000002], [11.0, 20.0]],
            dtype=np.float32,
        ),
        "next_states": np.asarray(
            [[10.0, 21.0], [10.0, 20.000002], [11.0, 20.0], [12.0, 20.0]],
            dtype=np.float32,
        ),
    }
    views = _prediction_payload_views(rollout)
    assert not np.array_equal(views["history_pixels"][0], views["query_pixels"])
    assert np.array_equal(views["query_pixels"], actual_third)
    assert np.array_equal(views["future_pixels"][0], actual_third)
    assert views["nominal_history_return_drift_px"] < 1e-4


def test_eval_partition_assignment_is_disjoint_and_paired() -> None:
    assignments = formal_template_assignments(
        eval_seeds=[42, 43], per_seed=2, assignment_seed=99
    )
    bundles = []
    for door in (49, 89):
        for task in TASKS:
            for template_index in range(4):
                bundles.append(
                    {
                        "door_position": door,
                        "task": task,
                        "template_index": template_index,
                        "direction": assignments[template_index][2],
                    }
                )
    assign_eval_partitions(
        bundles,
        eval_seeds=[42, 43],
        per_seed=2,
        assignment_seed=99,
    )
    assignment_by_template = {}
    for row in bundles:
        observed = (row["eval_seed"], row["evaluation_index"])
        assignment_by_template.setdefault(row["template_index"], observed)
        assert assignment_by_template[row["template_index"]] == observed
    for door in (49, 89):
        for task in TASKS:
            selected = [
                row
                for row in bundles
                if row["door_position"] == door and row["task"] == task
            ]
            assert {row["eval_seed"] for row in selected} == {42, 43}
            assert len({row["template_index"] for row in selected}) == 4
            for eval_seed in (42, 43):
                counts = {
                    direction: sum(
                        row["eval_seed"] == eval_seed
                        and row["direction"] == direction
                        for row in selected
                    )
                    for direction in ("left_to_right", "right_to_left")
                }
                assert counts == {"left_to_right": 1, "right_to_left": 1}


def test_formal_300_templates_are_25_plus_25_in_every_eval_seed() -> None:
    assignments = formal_template_assignments(
        eval_seeds=EVAL_SEEDS,
        per_seed=50,
        assignment_seed=2026072203,
    )
    assert len(assignments) == 300
    for eval_seed in EVAL_SEEDS:
        counts = {
            direction: sum(
                seed == eval_seed and observed_direction == direction
                for seed, _, observed_direction in assignments.values()
            )
            for direction in ("left_to_right", "right_to_left")
        }
        assert counts == {"left_to_right": 25, "right_to_left": 25}


def test_builder_defaults_to_validation_and_explicitly_supports_sealed_test() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validation = _selected_track_rows(config, "validation")
    sealed = _selected_track_rows(config, "sealed_test")
    assert set(validation) == {"validation_seen", "validation_interpolation"}
    assert {name: len(row["door_positions"]) for name, row in sealed.items()} == {
        "test_interpolation": 8,
        "test_extrapolation_low": 3,
        "test_extrapolation_high": 3,
    }
    for door in (24, 199):
        for direction in ("left_to_right", "right_to_left"):
            geometry = make_query_geometry(
                door_position=door,
                task="doorway_passage",
                direction=direction,
                template_index=0,
                seed=123,
            )
            assert 21.0 <= geometry.query_state[0] <= 203.0
            assert 21.0 <= geometry.query_state[1] <= 203.0


def _model_records(error: float) -> list[dict]:
    rows = []
    for track in ("validation_seen", "validation_interpolation"):
        for task in TASKS:
            for condition in INPUT_CONDITIONS:
                for door in (49, 89, 129, 169):
                    for seed in EVAL_SEEDS:
                        for index in range(2):
                            rows.append(
                                {
                                    "query_id": f"{track}-{task}-{door}-{seed}-{index}",
                                    "static_query_id": f"{track}-{task}-{door}-{seed}-{index}",
                                    "template_id": f"template-{index}",
                                    "track": track,
                                    "task": task,
                                    "direction": (
                                        "left_to_right"
                                        if index == 0
                                        else "right_to_left"
                                    ),
                                    "input_condition": condition,
                                    "door_position": door,
                                    "eval_seed": seed,
                                    "evaluation_index": index,
                                    "latent_mse_by_horizon": {
                                        str(horizon): error * horizon
                                        for horizon in HORIZONS
                                    },
                                    "unchanged_baseline_mse_by_horizon": {
                                        str(horizon): 2.0 * horizon
                                        for horizon in HORIZONS
                                    },
                                    "normalized_error_by_horizon": {
                                        str(horizon): error / 2.0
                                        for horizon in HORIZONS
                                    },
                                }
                            )
    return rows


def test_checkpoint_summary_keeps_raw_mse_inside_checkpoint() -> None:
    summary = checkpoint_cell_summary(_model_records(1.0), eval_seeds=EVAL_SEEDS)
    assert not summary["raw_latent_mse_cross_checkpoint_comparison_allowed"]
    first = next(iter(summary["cells"].values()))
    assert first["native_latent_mse"] > 0
    assert first["unchanged_baseline_mse"] > first["native_latent_mse"]
    assert first["mean_normalized_error"] == pytest.approx(0.5)
    assert first["beats_unchanged_baseline_rate"] == 1.0


def test_paired_normalized_effect_uses_exact_queries_and_seed_partitions() -> None:
    control = _model_records(1.2)
    target = _model_records(0.8)
    result = paired_normalized_effects(
        control,
        target,
        bootstrap_seed=5,
        bootstrap_samples=100,
    )
    assert len(result["effects"]) == len(control) * len(HORIZONS)
    for cell in result["cells"].values():
        assert cell["mean_control_minus_target"] == pytest.approx(0.2)
        assert cell["paired_query_win_rate"] == 1.0
        assert cell["all_eval_seed_directions_positive"]
        assert set(cell["by_eval_seed"]) == set(map(str, EVAL_SEEDS))


def test_formal_gate_uses_history3_both_tasks_and_not_query_only() -> None:
    paired_by_seed = {}
    for training_seed in (3072, 4096, 5120):
        summaries = {}
        for track in ("validation_seen", "validation_interpolation"):
            for task in TASKS:
                for condition in INPUT_CONDITIONS:
                    for horizon in HORIZONS:
                        summaries[f"{track}/{task}/{condition}/h{horizon}"] = {
                            "every_door_all_eval_seed_directions_positive": True
                        }
        paired_by_seed[str(training_seed)] = {"summaries": summaries}
    # Query-only failure must remain descriptive.
    paired_by_seed["3072"]["summaries"][
        "validation_seen/doorway_passage/query_only/h1"
    ]["every_door_all_eval_seed_directions_positive"] = False
    decision = _formal_decision(paired_by_seed, complete_matrix=True)
    assert decision["visible_geometry_generalization_validation_gate_passed"]
    assert decision["query_only_is_descriptive_and_not_a_hard_gate"]
    # A History-3 task failure is a real primary-gate failure.
    paired_by_seed["3072"]["summaries"][
        "validation_seen/wall_contact/natural_history3/h1"
    ]["every_door_all_eval_seed_directions_positive"] = False
    decision = _formal_decision(paired_by_seed, complete_matrix=True)
    assert not decision["visible_geometry_generalization_validation_gate_passed"]
    assert not decision["formal_pass_by_track_and_horizon"]["validation_seen"]["1"]


def test_query_only_and_history3_rollout_extract_five_future_predictions() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def rollout(self, observations, actions, history_size):
            batch = actions.shape[0]
            token_count = actions.shape[2] + 1
            values = torch.arange(
                token_count, device=actions.device, dtype=actions.dtype
            ).view(1, 1, token_count, 1)
            return {"predicted_emb": values.expand(batch, 1, token_count, 4)}

    registry = {
        "q": np.zeros((224, 224, 3), dtype=np.uint8),
        "h0": np.zeros((224, 224, 3), dtype=np.uint8),
        "h1": np.ones((224, 224, 3), dtype=np.uint8),
    }
    model = FakeModel()
    query_only = [
        {
            "history_size": 1,
            "input_pixel_keys": ["q"],
            "normalized_actions": np.zeros((5, 10), dtype=np.float32),
        }
    ]
    history3 = [
        {
            "history_size": 3,
            "input_pixel_keys": ["h0", "h1", "q"],
            "normalized_actions": np.zeros((7, 10), dtype=np.float32),
        }
    ]
    assert _rollout(model, query_only, registry, device="cpu").shape == (1, 5, 4)
    assert _rollout(model, history3, registry, device="cpu").shape == (1, 5, 4)


def test_longest_contiguous_horizon_stops_at_first_failure() -> None:
    assert longest_contiguous_horizon(
        {"1": True, "2": True, "3": False, "5": True}
    ) == 2
    assert longest_contiguous_horizon(
        {"1": True, "2": True, "3": True, "5": True}
    ) == 5


def test_formal_analyzer_refuses_smoke_result_and_split_mismatch(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    payload = {
        "status": "smoke_only",
        "evaluation_split": "validation",
        "config": {"sha256": "config"},
        "build_report": {"sha256": "build"},
        "stable_worldmodel": {"commit": "stable"},
        "online_environment_calls": 0,
        "frozen_weight_audit": {"passed": True},
        "count_audit": {"passed": True},
        "protocol": {"raw_latent_mse_cross_checkpoint_ranking": False},
        "tracks": {},
        "model": {"slug": "model"},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="smoke result"):
        _load_results(
            [result_path],
            config_hash="config",
            build_report_hash="build",
            evaluation_split="validation",
            stable_worldmodel_commit="stable",
            require_formal=True,
        )
    payload["status"] = "passed"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="split mismatch"):
        _load_results(
            [result_path],
            config_hash="config",
            build_report_hash="build",
            evaluation_split="sealed_test",
            stable_worldmodel_commit="stable",
            require_formal=True,
        )


def test_runtime_identity_and_training_reports_bind_every_checkpoint(
    tmp_path: Path,
) -> None:
    stable_commit = "stable"
    normalizer_hash = "normalizer"
    results = {}
    report_paths = []
    for index, (key, expected) in enumerate(TRAINING_BINDINGS.items()):
        group, seed = key
        checkpoint_hash = f"checkpoint-{index}"
        slug = expected["run_name"]
        results[slug] = {
            "model": {
                "slug": slug,
                "group": group,
                "training_seed": seed,
                "checkpoint_sha256": checkpoint_hash,
            },
            "normalizer": {"sha256": normalizer_hash},
            "stable_worldmodel": {"commit": stable_commit},
        }
        report = {
            "passed": True,
            "save_load_exact": True,
            "model_id": expected["model_id"],
            "run_name": expected["run_name"],
            "data": {"seed": seed},
            "training": {
                "training_complete": True,
                "plan": {"training_seed": seed},
            },
            "stable_worldmodel": {"commit": stable_commit},
            "artifacts": {"pretrained_sha256": checkpoint_hash},
        }
        path = tmp_path / expected["report_name"]
        path.write_text(json.dumps(report), encoding="utf-8")
        report_paths.append(path)
    identity = _audit_runtime_identity(
        results, expected_normalizer_sha256=normalizer_hash
    )
    assert identity["passed"]
    binding = _audit_training_report_bindings(
        results,
        report_paths=report_paths,
        stable_worldmodel_commit=stable_commit,
        require_complete=True,
    )
    assert binding["complete_formal_binding"]
    first = next(iter(results.values()))
    first["normalizer"]["sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="frozen original-train normalizer"):
        _audit_runtime_identity(
            results, expected_normalizer_sha256=normalizer_hash
        )
    first["normalizer"]["sha256"] = normalizer_hash
    result_values = list(results.values())
    result_values[1]["model"]["checkpoint_sha256"] = result_values[0]["model"][
        "checkpoint_sha256"
    ]
    with pytest.raises(RuntimeError, match="same checkpoint hash"):
        _audit_runtime_identity(
            results, expected_normalizer_sha256=normalizer_hash
        )
