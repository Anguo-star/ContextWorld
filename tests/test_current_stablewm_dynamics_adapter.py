from __future__ import annotations

import numpy as np
import torch

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    StableWorldModelLeWMContactFrictionAdapter,
)
from contextworld.evaluation.protocol import ColumnStandardizer


class _CurrentDynamics(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.received: dict[str, object] = {}

    def rollout(self, info, action_sequence, history_size=None):
        self.received = {
            "pixels_shape": tuple(info["pixels"].shape),
            "action_history_shape": tuple(info["action_history"].shape),
            "future_action_shape": tuple(action_sequence.shape),
            "history_size": history_size,
        }
        batch, samples, future = action_sequence.shape[:3]
        return {
            "predicted_emb": torch.zeros(
                batch,
                samples,
                int(history_size) + future,
                4,
                device=action_sequence.device,
            )
        }


class _LegacyDynamics(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.received: dict[str, object] = {}

    def rollout(self, info, action_sequence, history_size=None):
        self.received = {
            "info_keys": tuple(sorted(info)),
            "action_shape": tuple(action_sequence.shape),
            "history_size": history_size,
        }
        batch, samples, action_tokens = action_sequence.shape[:3]
        future = action_tokens - int(history_size) + 1
        return {
            "predicted_emb": torch.zeros(
                batch,
                samples,
                int(history_size) + future,
                4,
                device=action_sequence.device,
            )
        }


def _adapter(model: torch.nn.Module):
    adapter = object.__new__(StableWorldModelLeWMContactFrictionAdapter)
    adapter.model = model
    adapter._protocol = AdapterProtocol(
        history_tokens=3,
        action_block_raw_steps=5,
        action_dim=2,
        future_action_blocks=5,
    )
    adapter.action_standardizer = ColumnStandardizer(
        mean=np.zeros((1, 2), dtype=np.float32),
        std=np.ones((1, 2), dtype=np.float32),
    )
    adapter.device = "cpu"
    return adapter


def test_lewm_adapter_splits_executed_history_from_future_actions() -> None:
    model = _CurrentDynamics()
    adapter = _adapter(model)

    predicted = adapter.rollout_latents(
        np.zeros((2, 3, 224, 224, 3), dtype=np.uint8),
        np.zeros((2, 3, 5, 2), dtype=np.float32),
        batch_size=2,
    )

    assert predicted.shape == (2, 1, 4)
    assert model.received == {
        "pixels_shape": (2, 1, 3, 3, 224, 224),
        "action_history_shape": (2, 1, 2, 10),
        "future_action_shape": (2, 1, 1, 10),
        "history_size": 3,
    }


def test_lewm_adapter_preserves_legacy_combined_action_protocol() -> None:
    model = _LegacyDynamics()
    adapter = _adapter(model)

    predicted = adapter.rollout_latents(
        np.zeros((2, 3, 224, 224, 3), dtype=np.uint8),
        np.zeros((2, 3, 5, 2), dtype=np.float32),
        batch_size=2,
    )

    assert predicted.shape == (2, 1, 4)
    assert model.received == {
        "info_keys": ("pixels",),
        "action_shape": (2, 1, 3, 10),
        "history_size": 3,
    }
