from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from contextworld.evaluation.action_delay_env import (
    action_delay_steps_value,
    make_extended_action_delay_env,
)
from contextworld.evaluation.action_delay_long_history import (
    CANDIDATE_HISTORY_TOKENS,
    DELAY_VALUES,
    build_long_history_feasibility,
    history_action_blocks,
    make_templates,
    simulate_template,
    validate_history_candidate,
)
from contextworld.synthesis.config import load_config
from scripts.train_tworoom_step1 import (
    _project_lewm_model_batch,
    _training_sequence_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/benchmark/"
    "tworoom_action_delay_long_history_feasibility_v1.yaml"
)
H7_CONFIG = (
    ROOT / "configs/benchmark/tworoom_action_delay_h7_v1.yaml"
)


def test_extended_environment_does_not_widen_frozen_h3_validator() -> None:
    assert action_delay_steps_value(5) == 5
    try:
        action_delay_steps_value(6)
    except ValueError:
        pass
    else:
        raise AssertionError("Frozen History=3 validator accepted delay 6")

    env = make_extended_action_delay_env(
        max_delay_steps=10,
        render_mode="rgb_array",
    )
    try:
        env.reset(
            seed=3,
            options={
                "variation": (),
                "variation_values": {
                    "agent.speed": np.asarray([7.0], dtype=np.float32),
                    "action.delay_steps": 10,
                },
                "state": np.asarray([45.0, 55.0], dtype=np.float32),
                "target_state": np.asarray(
                    [190.0, 190.0],
                    dtype=np.float32,
                ),
            },
        )
        assert env.action_delay_steps == 10
        assert env.pending_actions().shape == (10, 2)
    finally:
        env.close()


def test_history_action_schedule_has_no_teleport_or_hidden_action() -> None:
    blocks = history_action_blocks(
        history_tokens=7,
        direction="up",
        action_magnitude=0.5,
    )
    assert blocks.shape == (6, 5, 2)
    np.testing.assert_array_equal(blocks[2], -blocks[0])
    np.testing.assert_array_equal(blocks[1], np.zeros((5, 2)))
    np.testing.assert_array_equal(blocks[3:], np.zeros((3, 5, 2)))


def test_history6_is_minimum_and_history7_adds_stable_boundary() -> None:
    template = make_templates(
        catalog_seed=20260728,
        starts_per_direction=1,
    )[0]
    by_history = {}
    for history_tokens in CANDIDATE_HISTORY_TOKENS:
        rollouts = {
            delay: simulate_template(
                template,
                history_tokens=history_tokens,
                delay_steps=delay,
                agent_speed=7.0,
                action_magnitude=0.5,
                maximum_delay_steps=10,
            )
            for delay in DELAY_VALUES
        }
        by_history[history_tokens] = validate_history_candidate(
            template,
            rollouts,
        )

    assert not by_history[5]["physical_alignment_passed"]
    assert by_history[6]["physical_alignment_passed"]
    assert not by_history[6]["robust_query_boundary_passed"]
    assert by_history[7]["robust_query_boundary_passed"]
    assert by_history[9]["robust_query_boundary_passed"]


def test_full_length_selection_report(tmp_path: Path) -> None:
    config = load_config(CONFIG)
    catalog, report = build_long_history_feasibility(
        config=config,
        repo_root=ROOT,
        output_root=tmp_path,
    )
    assert report["status"] == "passed"
    assert report["selection"] == {
        "physical_minimum_history_tokens": 6,
        "formal_history_tokens": 7,
        "reason": (
            "the formal history adds one all-zero transition after the "
            "latest delayed recovery command has executed"
        ),
    }
    assert report["candidate_summary"]["5"][
        "physical_alignment_passed"
    ] is False
    assert report["candidate_summary"]["6"][
        "physical_alignment_passed"
    ] is True
    assert report["candidate_summary"]["6"][
        "robust_query_boundary_passed"
    ] is False
    assert report["candidate_summary"]["7"][
        "robust_query_boundary_passed"
    ] is True
    assert catalog["selection"]["formal_history_tokens"] == 7


def test_history7_training_shape_is_supported_by_controlled_runner() -> None:
    contract = _training_sequence_contract(H7_CONFIG)
    assert contract == {
        "history_tokens": 7,
        "num_preds": 1,
        "sequence_steps": 8,
        "raw_steps_per_action_block": 5,
    }
    projected = _project_lewm_model_batch(
        {
            "pixels": torch.zeros(2, 8, 3, 224, 224),
            "action": torch.zeros(2, 8, 10),
            "state": torch.ones(2, 8, 2),
        },
        sequence_steps=8,
    )
    assert tuple(projected) == ("pixels", "action")


def test_history7_protocol_does_not_claim_unrun_model_results() -> None:
    config = load_config(H7_CONFIG)
    assert config["claim_boundary"] == {
        "current_status": "protocol_frozen_not_yet_executed",
        "history3_result": "one_step_feasibility_only",
        "history7_model_result": "not_yet_available",
        "planning_result": "not_yet_available",
    }
    assert config["validation"]["independent_queries_per_delay"] == 300
    assert config["validation"]["future_horizons_action_blocks"] == [
        1,
        2,
        3,
    ]
    assert config["scoring"]["physical_equivalence"]["horizon1"][
        "distinct_future_groups"
    ] == 6
    assert config["scoring"]["physical_equivalence"]["horizon2"][
        "distinct_future_groups"
    ] == 11
