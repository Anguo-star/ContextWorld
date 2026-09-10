"""Additive Public Test gate completion for speed, action delay, and door.

Perfect history-following predictions must pass every new anti-shortcut
gate; constant (history-blind) predictions must fail.  The thresholds come
from ``configs/benchmark/contextworld_test_gate_completion_v1.yaml`` and are
additive to the frozen release gates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from contextworld.benchmarks.action_delay_icl_score import (
    action_delay_gate_completion_metrics,
)
from contextworld.benchmarks.door_icl_score import (
    door_gate_completion_metrics,
)
from contextworld.benchmarks.speed_icl_score import (
    speed_gate_completion_metrics,
)
from contextworld.benchmarks.test_gate_completion import (
    DEFAULT_GATE_COMPLETION_CONFIG,
    load_test_gate_completion_config,
    unit_bootstrap_draws,
)


DIM = 8
SPEEDS = (1.0, 2.0, 3.0)
CONDITIONS = {
    1.0: "history_low",
    2.0: "history_mid",
    3.0: "history_high",
}


@pytest.fixture(scope="module")
def config() -> dict:
    return load_test_gate_completion_config()


def _speed_inputs(
    perfect: bool, *, families: int = 6
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(11)
    groups: dict[str, dict] = {}
    records: list[dict] = []
    for family in range(families):
        static_id = f"static-{family}"
        targets = {speed: rng.normal(size=DIM) * speed for speed in SPEEDS}
        members = {}
        for speed in SPEEDS:
            predictions = {
                condition: (
                    targets[condition_speed].copy()
                    if perfect
                    else np.full(DIM, 0.5)
                )
                for condition_speed, condition in CONDITIONS.items()
            }
            members[speed] = {
                "query_id": f"{static_id}@{speed}",
                "eval_seed": 42,
                "matching_condition": CONDITIONS[speed],
                "condition_speeds": {
                    condition: condition_speed
                    for condition_speed, condition in CONDITIONS.items()
                },
                "condition_predictions": predictions,
                "target": targets[speed],
            }
            for condition_speed, condition in CONDITIONS.items():
                own_target_loss = float(
                    np.mean((predictions[condition] - targets[speed]) ** 2)
                )
                records.append(
                    {
                        "query_id": f"{static_id}@{speed}",
                        "reference_speed": speed,
                        "condition": condition,
                        "matching_condition": CONDITIONS[speed],
                        "latent_mse_by_horizon": {
                            str(horizon): own_target_loss
                            for horizon in (1, 2, 3, 5)
                        },
                    }
                )
        groups[static_id] = members
    return groups, records


def _action_delay_inputs(
    perfect: bool, *, queries: int = 5
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    rng = np.random.default_rng(12)
    encoded_one = np.stack(
        [rng.normal(size=DIM) * (1.0 + delay / 4.0) for delay in range(11)]
    )
    encoded = np.repeat(encoded_one[None], queries, axis=0)
    if perfect:
        predicted = encoded.copy()
        wins = np.ones((queries, 11), dtype=bool)
    else:
        predicted = np.full((queries, 11, DIM), 0.25)
        wins = np.zeros((queries, 11), dtype=bool)
    query_ids = [f"ad-{index}" for index in range(queries)]
    return predicted, encoded, query_ids, wins


def _door_inputs(
    perfect: bool, *, queries: int = 6
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(13)
    target_passable = rng.normal(size=(queries, DIM)) + 2.0
    target_blocked = rng.normal(size=(queries, DIM)) - 2.0
    targets = np.stack([target_passable, target_blocked], axis=1)
    if perfect:
        predicted = targets.copy()
    else:
        predicted = np.zeros((queries, 3, DIM))
    query_ids = [f"door-{index}" for index in range(queries)]
    return predicted, targets, query_ids, list(query_ids)


def test_gate_completion_config_loads_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    config = load_test_gate_completion_config()
    assert config["release_id"] == "contextworld_test_gate_completion_v1"
    for component in ("speed", "action_delay", "door"):
        assert config[component]["gates"]["response_gain_minimum"] == 0.50
        assert (
            config[component]["gates"][
                "normalized_response_error_strict_maximum"
            ]
            == 1.00
        )

    payload = yaml.safe_load(
        Path(DEFAULT_GATE_COMPLETION_CONFIG).read_text(encoding="utf-8")
    )
    payload["speed"]["gates"]["response_gain_minimum"] = 0.10
    tampered = tmp_path / "tampered_gate_completion.yaml"
    tampered.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="response-gain"):
        load_test_gate_completion_config(tampered)

    payload["speed"]["gates"]["response_gain_minimum"] = 0.50
    payload["speed"]["gates"]["normalized_response_error_strict_maximum"] = 1.5
    tampered.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normalized-response-error"):
        load_test_gate_completion_config(tampered)


def test_speed_gate_completion_passes_perfect_and_fails_constant(
    config: dict,
) -> None:
    groups, records = _speed_inputs(perfect=True)
    block = speed_gate_completion_metrics(
        groups=groups,
        records=records,
        config=config,
        full_protocol=True,
    )
    assert block["passed"] is True
    assert all(block["checks"].values())
    metrics = block["metrics"]
    assert metrics["correct_history_rate"] == 1.0
    assert (
        metrics["worst_reference_speed_correct_history_rate"] == 1.0
    )
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(1.0)
    assert metrics["latent_response"]["normalized_response_error"] == (
        pytest.approx(0.0, abs=1e-9)
    )
    assert block["uncertainty"]["lower_bounds"][
        "correct_history_rate"
    ] == pytest.approx(1.0)
    assert block["uncertainty"]["lower_bounds"][
        "context_switch_rate"
    ] == pytest.approx(1.0)

    groups, records = _speed_inputs(perfect=False)
    block = speed_gate_completion_metrics(
        groups=groups,
        records=records,
        config=config,
        full_protocol=True,
    )
    assert block["passed"] is False
    metrics = block["metrics"]
    assert metrics["correct_history_rate"] == 0.0
    assert metrics["context_switch_rate"] == 0.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(0.0)
    assert metrics["latent_response"]["normalized_response_error"] >= 1.0
    assert block["checks"]["response_gain"] is False
    assert block["checks"]["normalized_response_error"] is False
    assert block["uncertainty"]["lower_bounds"]["correct_history_rate"] < 0.70


def test_speed_gate_completion_counts_incomplete_families(
    config: dict,
) -> None:
    groups, records = _speed_inputs(perfect=True)
    solo = dict(groups)
    solo["static-solo"] = {
        1.0: dict(
            groups["static-0"][1.0], query_id="static-solo@1.0"
        )
    }
    block = speed_gate_completion_metrics(
        groups=solo,
        records=records,
        config=config,
        full_protocol=False,
    )
    assert block["metrics"]["incomplete_static_query_families"] == 1
    assert block["metrics"]["static_query_families"] == 6
    assert block["full_protocol_track"] is False


def test_action_delay_gate_completion_passes_perfect_and_fails_constant(
    config: dict,
) -> None:
    predicted, encoded, query_ids, wins = _action_delay_inputs(
        perfect=True
    )
    block = action_delay_gate_completion_metrics(
        predicted_h1=predicted,
        encoded_h1=encoded,
        query_ids=query_ids,
        history_strict_wins=wins,
        config=config,
    )
    assert block["passed"] is True
    metrics = block["metrics"]
    assert metrics["pair_count"] == len(query_ids) * 40
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(1.0)
    assert block["uncertainty"]["lower_bounds"][
        "correct_history_rate"
    ] == pytest.approx(1.0)

    predicted, encoded, query_ids, wins = _action_delay_inputs(
        perfect=False
    )
    block = action_delay_gate_completion_metrics(
        predicted_h1=predicted,
        encoded_h1=encoded,
        query_ids=query_ids,
        history_strict_wins=wins,
        config=config,
    )
    assert block["passed"] is False
    metrics = block["metrics"]
    assert metrics["correct_history_rate"] == 0.0
    assert metrics["context_switch_rate"] == 0.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(0.0)
    assert block["checks"]["target_latent_separation"] is True
    assert block["checks"]["response_gain"] is False


def test_action_delay_gate_pairs_never_compare_stationary_futures(
    config: dict,
) -> None:
    predicted, encoded, query_ids, wins = _action_delay_inputs(
        perfect=True
    )
    block = action_delay_gate_completion_metrics(
        predicted_h1=predicted,
        encoded_h1=encoded,
        query_ids=query_ids,
        history_strict_wins=wins,
        config=config,
    )
    pairs = block["metrics"]["delay_pairs"]
    assert len(pairs) == 40
    stationary = {f"{a}>{b}" for a in range(5, 11) for b in range(a + 1, 11)}
    assert stationary.isdisjoint(pairs)
    assert "0>5" in pairs and "4>10" in pairs

    with pytest.raises(ValueError, match="share a"):
        action_delay_gate_completion_metrics(
            predicted_h1=predicted[:, :, None, :],
            encoded_h1=encoded,
            query_ids=query_ids,
            history_strict_wins=wins,
            config=config,
        )


def test_door_gate_completion_passes_perfect_and_fails_constant(
    config: dict,
) -> None:
    predicted, targets, query_ids, static_ids = _door_inputs(perfect=True)
    block = door_gate_completion_metrics(
        predicted=predicted,
        encoded_targets=targets,
        query_ids=query_ids,
        static_query_ids=static_ids,
        config=config,
        full_protocol=True,
    )
    assert block["passed"] is True
    metrics = block["metrics"]
    assert metrics["context_switch_rate"] == 1.0
    assert metrics["worst_rule_correct_target_choice_rate"] == 1.0
    assert metrics["correct_history_rate"] == 1.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(1.0)
    assert block["uncertainty"]["lower_bounds"][
        "worst_rule_correct_target_choice_rate"
    ] == pytest.approx(1.0)

    predicted, targets, query_ids, static_ids = _door_inputs(perfect=False)
    block = door_gate_completion_metrics(
        predicted=predicted,
        encoded_targets=targets,
        query_ids=query_ids,
        static_query_ids=static_ids,
        config=config,
        full_protocol=True,
    )
    assert block["passed"] is False
    metrics = block["metrics"]
    assert metrics["context_switch_rate"] == 0.0
    # A history-blind prediction only "wins" a rule by target-norm luck and
    # never both rules at once, so the worst rule stays far below 0.70.
    assert (
        metrics["worst_rule_correct_target_choice_rate"] < 0.70
    )
    assert metrics["correct_history_rate"] == 0.0
    assert metrics["latent_response"]["response_gain"] == pytest.approx(0.0)
    assert metrics["latent_response"]["normalized_response_error"] >= 1.0
    assert block["checks"]["response_gain"] is False
    assert block["checks"]["normalized_response_error"] is False


def test_unit_bootstrap_draws_resample_whole_units() -> None:
    rows = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    draws = unit_bootstrap_draws(
        rows, resamples=200, random_seed=2026
    )
    assert draws.shape == (200, 2)
    assert np.all(draws[:, 0] == 1.0)
    assert np.all(draws[:, 1] == 0.0)
    with pytest.raises(ValueError, match="positive"):
        unit_bootstrap_draws(rows, resamples=0, random_seed=1)
