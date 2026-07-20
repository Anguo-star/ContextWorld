from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from contextworld.synthesis.config import scenario_requests
from scripts.analyze_tworoom_speedtask_quality import paired_factor_cross
from scripts.analyze_tworoom_training_data_gap import (
    TraceTable,
    paired_speed_reuse,
    summarize_trace,
)
from scripts.analyze_tworoom_trajectory_quality import EpisodeTable


ROOT = Path(__file__).resolve().parents[1]
E4_SPEEDS = {3.1, 3.3, 3.5, 4.1, 5.0, 5.1, 5.9, 7.0}


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_speedtask_restores_task_semantics_and_expands_geometry() -> None:
    config = _load("configs/synthesis/tworoom_speed_task_v1.yaml")
    requests = scenario_requests(config)
    train = [request for request in requests if request.split == "train"]
    val = [request for request in requests if request.split == "val"]
    train_speeds = {float(request.atoms[0].value) for request in train}

    assert config["collection"]["reset_constraints"] == {
        "target_room": "opposite",
        "exclude_wall_zone": True,
        "minimum_initial_distance": 40.0,
    }
    assert E4_SPEEDS <= train_speeds
    assert len(train_speeds) == 32
    assert len({request.seed_group for request in train}) == 16
    assert len(train) == 512
    assert sum(request.episodes for request in train) == 16_384
    assert len(val) == 32
    assert sum(request.episodes for request in val) == 512


def test_speedtask_training_keeps_model_recipe_fixed() -> None:
    benchmark = _load("configs/benchmark/tworoom_speed_task_v1.yaml")
    assert benchmark["training_protocol"]["group_sampling"]["M_speed"] == {
        "original": 0.5,
        "speed": 0.5,
    }
    assert benchmark["training_protocol"]["reference"] == "H3-SpeedSeen-s3072"
    assert benchmark["models"][0]["display_name"] == "H3-SpeedTask"
    assert benchmark["data_quality"]["groups"]["speed"][
        "minimum_train_scenarios"
    ] == 512
    assert benchmark["training_result"]["status"] == "passed"
    assert benchmark["evaluation_result"]["e4"]["correct_only_successes"] == 0
    assert benchmark["evaluation_result"]["training_data_gap"][
        "original_independent_geometries"
    ] == 10_000


def test_speedtask_launchers_are_portable() -> None:
    data = (ROOT / "scripts/run_tworoom_speedtask_data.sh").read_text(
        encoding="utf-8"
    )
    train = (ROOT / "scripts/run_h3_speedtask_train.sh").read_text(
        encoding="utf-8"
    )
    evaluate = (ROOT / "scripts/run_h3_speedtask_eval.sh").read_text(
        encoding="utf-8"
    )
    e4 = (ROOT / "scripts/run_h3_speedtask_e4_parallel.sh").read_text(
        encoding="utf-8"
    )
    assert "tworoom_speed_task_v1.yaml" in data
    assert 'SHARDS="${SHARDS:-32}"' in data
    assert "tworoom_speed_task_v1.yaml" in train
    assert "h3_speedtask_s${TRAINING_SEED}" in train
    assert 'STABLEWM_REPO="${STABLEWM_REPO:-../stable-worldmodel}"' in train
    assert "h3_speedtask_s3072/weights_final_step_6420.pt" in evaluate
    assert "run_h3_speedtask_e4_parallel.sh" in evaluate
    assert 'OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"' in evaluate
    assert "[H3-SpeedTask E4]" in e4
    assert "--num-eval 50" in e4
    assert 'OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"' in e4


def test_speedtask_quality_report_detects_complete_geometry_factor_cross() -> None:
    episodes = EpisodeTable(
        start=np.asarray([[20, 50], [20, 50], [190, 170], [190, 170]]),
        goal=np.asarray([[190, 170], [190, 170], [20, 50], [20, 50]]),
        final=np.asarray([[180, 160], [180, 160], [30, 60], [30, 60]]),
        lengths=np.asarray([80, 60, 90, 70]),
        terminated=np.asarray([False, True, False, True]),
        truncated=np.asarray([True, False, True, False]),
        speed=np.asarray([3.0, 7.0, 3.0, 7.0]),
    )

    paired = paired_factor_cross(episodes)

    assert paired["factor_values"] == [3.0, 7.0]
    assert paired["independent_geometries"] == 2
    assert paired["complete_factor_cross_geometries"] == 2
    assert paired["minimum_episodes_per_factor"] == 2
    assert paired["duplicate_factor_rows_within_geometry"] == 0


def test_training_gap_separates_episode_count_from_geometry_support() -> None:
    trace = TraceTable(
        state=np.asarray(
            [[20, 50], [23, 50], [26, 50], [20, 50], [20, 57], [20, 64]],
            dtype=np.float32,
        ),
        goal=np.asarray([[190, 50]] * 6, dtype=np.float32),
        action=np.asarray(
            [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
            dtype=np.float32,
        ),
        offsets=np.asarray([0, 3]),
        lengths=np.asarray([3, 3]),
        terminated=np.asarray([False, False]),
        truncated=np.asarray([True, True]),
        speed=np.asarray([3.0, 7.0], dtype=np.float32),
        reset=np.asarray([[20, 50], [20, 50]], dtype=np.float32),
        reset_is_observed=True,
        scenario=np.asarray(["slow", "fast"], dtype=object),
    )

    summary = summarize_trace(trace)
    reuse = paired_speed_reuse(trace)

    assert summary["episodes"] == 2
    assert summary["geometry"]["unique_reset_goal_pairs_5dp"] == 1
    assert summary["controller_and_transition"][
        "collision_or_boundary_residual_fraction"
    ] == 0.0
    assert reuse["independent_reset_goal_geometries"] == 1
    assert reuse["episodes_per_geometry_minimum"] == 2
    assert reuse["unique_speeds_per_geometry_minimum"] == 2
