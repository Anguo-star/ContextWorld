from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMHistory7Adapter,
)
from contextworld.evaluation.action_delay_h7_score import (
    HORIZON_LOSS_RECORDS_PER_CHECKPOINT,
    score_h7_validation_assets,
    summarize_h7_validation_records,
)
from contextworld.evaluation.action_delay_h7_validation import (
    DELAYS,
    EVAL_SEEDS,
    QUERY_COUNT,
)


class _OracleAdapter:
    protocol = SimpleNamespace(
        history_tokens=7,
        action_block_raw_steps=5,
        action_dim=2,
        future_action_blocks=3,
    )

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del raw_action_blocks, batch_size
        delays = input_pixels[:, 0, 0, 0, 0].astype(np.float32)
        return np.stack(
            [np.minimum(delays, 5.0), delays, delays], axis=1
        )[..., None]

    def encode_pixels(
        self,
        pixels: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        del batch_size
        return pixels[:, 0, 0, 0].astype(np.float32)[:, None]


def _assets() -> list[dict[str, object]]:
    assets = []
    for query_index in range(QUERY_COUNT):
        history = np.zeros((len(DELAYS), 7, 1, 1, 3), dtype=np.uint8)
        future = np.zeros((len(DELAYS), 3, 1, 1, 3), dtype=np.uint8)
        for delay in DELAYS:
            history[delay, ..., 0] = delay
            future[delay, 0, ..., 0] = min(delay, 5)
            future[delay, 1:, ..., 0] = delay
        assets.append(
            {
                "query_id": f"q{query_index:03d}",
                "eval_seed": EVAL_SEEDS[query_index // 50],
                "evaluation_index": query_index % 50,
                "room": "left" if query_index % 2 == 0 else "right",
                "direction": "up" if query_index % 2 == 0 else "down",
                "history_pixels": history,
                "action_blocks": np.zeros((9, 5, 2), dtype=np.float32),
                "true_future_pixels": future,
            }
        )
    return assets


def test_history7_oracle_scores_exact_trajectory_and_physical_h1() -> None:
    scored = score_h7_validation_assets(
        _OracleAdapter(),
        _assets(),
        batch_size=128,
    )
    assert len(scored["records"]) == HORIZON_LOSS_RECORDS_PER_CHECKPOINT
    summary = summarize_h7_validation_records(scored["records"])
    trajectory = summary["trajectory"]["overall"]
    assert trajectory["exact_history_selection_rate"] == 1.0
    assert trajectory["exact_target_selection_rate"] == 1.0
    assert trajectory["matching_history_strict_win_rate"] == 1.0

    h1 = summary["by_horizon"]["1"]
    assert h1["overall"]["physical_target_group_selection_rate"] == 1.0
    assert (
        h1["by_target_delay"]["6"]["exact_target_selection_rate"] == 0.0
    )
    assert (
        h1["by_target_delay"]["6"][
            "physical_target_group_selection_rate"
        ]
        == 1.0
    )


def test_history7_adapter_and_scoring_protocol_are_frozen() -> None:
    assert StableWorldModelLeWMHistory7Adapter.required_history_tokens == 7
    assert (
        StableWorldModelLeWMHistory7Adapter.maximum_future_action_blocks
        == 3
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "configs/benchmark/tworoom_action_delay_h7_scoring_v1.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_any_history7_model_scoring"
    assert sum(len(rows) for rows in config["models"].values()) == 9
    assert config["evaluation"]["expected_counts_per_checkpoint"] == {
        "model_predictions": 3300,
        "target_encodings": 9900,
        "trajectory_comparisons": 36300,
        "horizon_loss_records": 108900,
    }
