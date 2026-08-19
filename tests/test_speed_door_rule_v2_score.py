from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    SpeedICLModelAdapter,
)
from contextworld.evaluation.speed_door_rule_v2_score import (
    evaluate_v2_checkpoint_gate,
    score_v2_assets,
    summarize_v2_scores,
)
from contextworld.synthesis.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/benchmark/tworoom_speed_door_rule_h3_v2.yaml"
)
SPEEDS = (3.1, 5.1, 7.0)
FACTORS = tuple(
    (speed, rule)
    for speed in SPEEDS
    for rule in ("passable", "blocked")
)
TARGETS = (
    "blocked",
    "passable_s03p1",
    "passable_s05p1",
    "passable_s07p0",
)
CODE = {
    "blocked": np.asarray([10, 10], dtype=np.uint8),
    "passable_s03p1": np.asarray([60, 30], dtype=np.uint8),
    "passable_s05p1": np.asarray([130, 30], dtype=np.uint8),
    "passable_s07p0": np.asarray([220, 30], dtype=np.uint8),
}


def _target(speed: float, rule: str) -> str:
    if rule == "blocked":
        return "blocked"
    return {
        3.1: "passable_s03p1",
        5.1: "passable_s05p1",
        7.0: "passable_s07p0",
    }[speed]


def _assets() -> list[dict]:
    assets = []
    for index, seed in enumerate((42, 43)):
        histories = {}
        actions = {}
        targets = {1: {}, 2: {}}
        for factor in FACTORS:
            history = np.zeros((3, 4, 4, 3), dtype=np.uint8)
            history[1, 0, 0, :2] = CODE[_target(*factor)]
            histories[factor] = history
            actions[factor] = np.zeros((4, 5, 2), dtype=np.float32)
        for horizon in (1, 2):
            for target in TARGETS:
                pixels = np.zeros((4, 4, 3), dtype=np.uint8)
                pixels[0, 0, :2] = CODE[target]
                targets[horizon][target] = pixels
        assets.append(
            {
                "query_id": f"q{index}",
                "eval_seed": seed,
                "evaluation_index": 0,
                "direction": "left_to_right",
                "door_position": 30 + index * 4,
                "template_id": f"t{index}",
                "histories": histories,
                "actions": actions,
                "targets": targets,
            }
        )
    return assets


class _FakeAdapter(SpeedICLModelAdapter):
    def __init__(self, *, history_sensitive: bool = True) -> None:
        self.history_sensitive = history_sensitive
        self._protocol = AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=2,
        )

    @property
    def protocol(self) -> AdapterProtocol:
        return self._protocol

    @property
    def metadata(self) -> dict:
        return {"protocol": asdict(self.protocol)}

    def encode_pixels(
        self, pixels: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        return np.asarray(pixels)[:, 0, 0, :2].astype(np.float32)

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        if self.history_sensitive:
            value = np.asarray(input_pixels)[:, 1, 0, 0, :2]
        else:
            value = np.zeros((len(input_pixels), 2), dtype=np.float32)
        return np.repeat(value[:, None].astype(np.float32), 2, axis=1)

    def frozen_state_hash(self) -> str:
        return "unchanged"


def test_perfect_composition_scores_every_primary_metric() -> None:
    scored = score_v2_assets(
        _FakeAdapter(),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    assert len(scored["condition_records"]) == 24
    assert len(scored["suppression_records"]) == 12
    assert scored["score_audit"]["model_prediction_sequences"] == 12
    assert scored["score_audit"]["model_prediction_endpoints"] == 24
    summary = summarize_v2_scores(
        scored["condition_records"], scored["suppression_records"]
    )
    for horizon in ("h1", "h2"):
        overall = summary["by_horizon"][horizon]["overall"]
        assert overall["passable_speed_future_accuracy"] == 1.0
        assert overall["door_future_accuracy"] == 1.0
        assert overall["physical_future_macro_accuracy"] == 1.0
        assert overall["blocked_speed_suppression_win_rate"] == 1.0
        assert overall["passable_speed_history_guidance"] == 1.0
        assert overall["door_history_guidance"] == 1.0


def test_history_invariant_predictions_fail_formal_joint_gate() -> None:
    scored = score_v2_assets(
        _FakeAdapter(history_sensitive=False),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    summary = summarize_v2_scores(
        scored["condition_records"], scored["suppression_records"]
    )
    gate = evaluate_v2_checkpoint_gate(
        summary=summary,
        config=load_config(CONFIG),
        role="joint",
    )
    assert not gate["passed"]


def test_role_gates_use_only_the_registered_prerequisite_metric() -> None:
    scored = score_v2_assets(
        _FakeAdapter(),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    summary = summarize_v2_scores(
        scored["condition_records"], scored["suppression_records"]
    )
    config = load_config(CONFIG)
    speed_gate = evaluate_v2_checkpoint_gate(
        summary=summary, config=config, role="speed_only"
    )
    door_gate = evaluate_v2_checkpoint_gate(
        summary=summary, config=config, role="door_only"
    )
    assert speed_gate["passed"]
    assert speed_gate["required_metrics"] == [
        "passable_speed_future_accuracy"
    ]
    assert door_gate["passed"]
    assert door_gate["required_metrics"] == [
        "door_anchor_future_accuracy"
    ]
