from __future__ import annotations

import numpy as np
import pytest
import torch

from contextworld.evaluation.icl_planning import (
    CONTEXT_ACTIONS_KEY,
    CONTEXT_PIXELS_KEY,
    FixedContextCostModel,
    FixedContextPolicy,
    PairedQueryDataset,
    QueryEpisode,
)


class _CaptureCost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.info = None
        self.actions = None

    def get_cost(self, info, actions):
        self.info = info
        self.actions = actions
        return torch.zeros(actions.shape[:2])


def test_fixed_context_is_prepended_outside_cem_variables():
    base = _CaptureCost()
    model = FixedContextCostModel(base, history_size=3)
    context_pixels = torch.randn(2, 4, 2, 3, 8, 8)
    current_pixels = torch.randn(2, 4, 1, 3, 8, 8)
    context_actions = torch.randn(2, 4, 2, 10)
    candidates = torch.randn(2, 4, 5, 10)

    costs = model.get_cost(
        {
            "pixels": current_pixels,
            CONTEXT_PIXELS_KEY: context_pixels,
            CONTEXT_ACTIONS_KEY: context_actions,
        },
        candidates,
    )

    assert costs.shape == (2, 4)
    assert base.info["pixels"].shape == (2, 4, 3, 3, 8, 8)
    assert base.actions.shape == (2, 4, 7, 10)
    assert torch.equal(base.info["pixels"][:, :, :2], context_pixels)
    assert torch.equal(base.info["pixels"][:, :, 2:], current_pixels)
    assert torch.equal(base.actions[:, :, :2], context_actions)
    assert torch.equal(base.actions[:, :, 2:], candidates)
    assert CONTEXT_PIXELS_KEY not in base.info
    assert CONTEXT_ACTIONS_KEY not in base.info
    assert model.get_cost_calls == 1


def test_paired_query_dataset_exposes_two_row_query_goal_episode():
    episode = QueryEpisode(
        query_id="q0",
        scenario_id="scenario",
        template_id="s0",
        speed=3.1,
        door_position=49,
        simulator_seed=7,
        query_pixels=np.zeros((8, 8, 3), dtype=np.uint8),
        goal_pixels=np.ones((8, 8, 3), dtype=np.uint8),
        query_state=np.asarray([55.0, 70.0], dtype=np.float32),
        goal_state=np.asarray([190.0, 190.0], dtype=np.float32),
    )
    dataset = PairedQueryDataset([episode])

    chunk = dataset.load_chunk(
        np.asarray([0]), np.asarray([0]), np.asarray([2])
    )[0]
    assert chunk["pixels"].shape == (2, 3, 8, 8)
    assert dataset.lengths.tolist() == [2]
    assert dataset.offsets.tolist() == [0, 2]
    assert dataset.get_row_data(0)["variation_agent_speed"].tolist() == pytest.approx([3.1])
    assert dataset.get_row_data(0)["variation_door_position"].tolist() == [49, 49, 49]


class _TracePolicy:
    def get_action(self, info, **kwargs):
        return np.asarray([[0.25, -0.5]], dtype=np.float32)


def test_fixed_context_policy_records_raw_state_and_action():
    trace = []
    policy = FixedContextPolicy(
        _TracePolicy(),
        context_pixels=np.zeros((1, 2, 8, 8, 3), dtype=np.uint8),
        context_actions=np.zeros((1, 2, 10), dtype=np.float32),
        trace_steps=trace,
    )
    action = policy.get_action(
        {"state": np.asarray([[[10.0, 20.0]]], dtype=np.float32)}
    )
    assert np.allclose(action, [[0.25, -0.5]])
    assert trace == [{"state": [10.0, 20.0], "action": [0.25, -0.5]}]
