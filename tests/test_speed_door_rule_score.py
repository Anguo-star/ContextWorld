from __future__ import annotations

from dataclasses import asdict

import numpy as np

from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    SpeedICLModelAdapter,
)
from contextworld.evaluation.speed_door_rule_score import (
    ACCURACY_METRICS,
    aggregate_results,
    evaluate_checkpoint_gate,
    score_validation_assets,
    summarize_records,
)


FACTORS = tuple(
    (speed, rule)
    for speed in (3.1, 5.1, 7.0)
    for rule in ("passable", "blocked")
)


def _code(factor: tuple[float, str]) -> np.ndarray:
    speed, rule = factor
    return np.asarray(
        [
            {3.1: 20, 5.1: 100, 7.0: 220}[speed],
            {"passable": 30, "blocked": 210}[rule],
        ],
        dtype=np.uint8,
    )


def _assets() -> list[dict]:
    output = []
    for index, seed in enumerate((42, 43)):
        histories = {}
        actions = {}
        targets = {}
        for factor in FACTORS:
            history = np.zeros((3, 4, 4, 3), dtype=np.uint8)
            history[1, 0, 0, :2] = _code(factor)
            histories[factor] = history
            actions[factor] = np.zeros((3, 5, 2), dtype=np.float32)
            target = np.zeros((4, 4, 3), dtype=np.uint8)
            target[0, 0, :2] = _code(factor)
            targets[factor] = target
        output.append(
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
    return output


class _FakeAdapter(SpeedICLModelAdapter):
    def __init__(self, *, history_sensitive: bool = True) -> None:
        self.history_sensitive = history_sensitive
        self._protocol = AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=5,
            action_dim=2,
            future_action_blocks=1,
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
        return value.astype(np.float32)[:, None]

    def frozen_state_hash(self) -> str:
        return "unchanged"


def _gates() -> dict:
    overall = {metric: 0.75 for metric in ACCURACY_METRICS}
    by_seed = {metric: 0.65 for metric in ACCURACY_METRICS}
    return {
        "strict_ties_fail": True,
        "minimum_target_pair_latent_mse": 1.0e-12,
        "checkpoint": {
            "overall": overall,
            "every_eval_seed": by_seed,
        },
    }


def test_perfect_factor_switch_scores_all_six_metrics() -> None:
    scored = score_validation_assets(
        _FakeAdapter(),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    assert len(scored["records"]) == 12
    assert scored["score_audit"]["model_predictions"] == 12
    assert scored["score_audit"]["target_encodings"] == 12
    assert scored["score_audit"]["loss_comparisons"] == 72
    summary = summarize_records(scored["records"])
    assert all(summary["overall"][metric] == 1.0 for metric in ACCURACY_METRICS)
    gate = evaluate_checkpoint_gate(
        summary=summary,
        score_audit=scored["score_audit"],
        gates=_gates(),
    )
    assert gate["passed"]


def test_history_invariant_predictions_fail_history_metrics() -> None:
    scored = score_validation_assets(
        _FakeAdapter(history_sensitive=False),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    summary = summarize_records(scored["records"])
    assert summary["overall"]["speed_history_accuracy"] == 0.0
    assert summary["overall"]["door_history_accuracy"] == 0.0
    assert summary["overall"]["joint_history_accuracy"] == 0.0
    assert not evaluate_checkpoint_gate(
        summary=summary,
        score_audit=scored["score_audit"],
        gates=_gates(),
    )["passed"]


def test_method_gate_requires_all_three_joint_seeds() -> None:
    scored = score_validation_assets(
        _FakeAdapter(),
        _assets(),
        batch_size=4,
        epsilon=1.0e-12,
    )
    summary = summarize_records(scored["records"])
    checkpoint_gate = evaluate_checkpoint_gate(
        summary=summary,
        score_audit=scored["score_audit"],
        gates=_gates(),
    )
    rows = [
        {
            "model_id": "H3_SpeedDoorJoint_PLDM",
            "training_seed": seed,
            "asset_audit": {"content_manifest_sha256": "same"},
            "summary": summary,
            "checkpoint_gate": checkpoint_gate,
        }
        for seed in (3072, 4096, 5120)
    ]
    aggregate = aggregate_results(
        rows,
        required_joint_training_seeds=(3072, 4096, 5120),
    )
    assert aggregate["method_gate"]["passed"]
    assert not aggregate_results(
        rows[:2],
        required_joint_training_seeds=(3072, 4096, 5120),
    )["method_gate"]["passed"]
