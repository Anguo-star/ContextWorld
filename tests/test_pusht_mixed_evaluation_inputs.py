from __future__ import annotations

import numpy as np
import torch

from scripts import run_pusht_hidden_actuation_mixed as mixed


ACTION_STATS = {
    "mean": np.zeros(2, dtype=np.float32),
    "std": np.ones(2, dtype=np.float32),
}


def _pair() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    low_pixels = torch.zeros(4, 3, 224, 224, dtype=torch.uint8)
    high_pixels = low_pixels.clone()
    high_pixels[1] = 1
    high_pixels[3] = 2
    low_state = torch.zeros(4, 7)
    high_state = low_state.clone()
    high_state[3, 2] = 1.0
    action = torch.zeros(4, 10)
    return (
        {"pixels": low_pixels, "action": action, "state": low_state},
        {"pixels": high_pixels, "action": action.clone(), "state": high_state},
    )


def test_validation_lance_fallback_constructs_complete_development_pairs(
    tmp_path, monkeypatch
) -> None:
    validation = tmp_path / "validation.lance"
    validation.mkdir()
    low, high = _pair()

    class FakeLanceDataset:
        def __init__(self, *, path, frameskip, num_steps, keys_to_load) -> None:
            assert path == validation
            assert frameskip == 5
            assert num_steps == 4
            assert keys_to_load == ["pixels", "action", "state"]
            self.rows = [low, high]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return self.rows[index]

    monkeypatch.setattr(mixed, "LanceDataset", FakeLanceDataset)
    evaluation, source = mixed.load_hidden_evaluation(
        tmp_path,
        action_stats=ACTION_STATS,
    )

    assert source == "development_validation_lance"
    assert evaluation["low_pixels"].shape == (1, 4, 3, 224, 224)
    assert evaluation["high_pixels"].shape == (1, 4, 3, 224, 224)
    assert evaluation["action"].shape == (1, 4, 10)
    assert torch.equal(evaluation["low_states"][0], low["state"])
    assert torch.equal(evaluation["high_states"][0], high["state"])


def test_legacy_eval_payloads_remain_supported(tmp_path) -> None:
    payload_root = tmp_path / "eval_payloads"
    payload_root.mkdir()
    low, high = _pair()
    np.savez(
        payload_root / "pair.npz",
        low_pixels=low["pixels"].permute(0, 2, 3, 1).numpy(),
        high_pixels=high["pixels"].permute(0, 2, 3, 1).numpy(),
        low_actions=low["action"].numpy(),
        high_actions=high["action"].numpy(),
        low_states=low["state"].numpy(),
        high_states=high["state"].numpy(),
    )

    evaluation, source = mixed.load_hidden_evaluation(
        tmp_path,
        action_stats=ACTION_STATS,
    )

    assert source == "legacy_eval_payloads"
    assert evaluation["low_pixels"].shape == (1, 4, 3, 224, 224)
    assert evaluation["action"].shape == (1, 4, 10)
