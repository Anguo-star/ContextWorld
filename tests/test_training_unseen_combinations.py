from types import SimpleNamespace

from contextworld.synthesis.models import AtomRequest
from contextworld.synthesis.validator import (
    validate_training_unseen_combinations,
)


def _scenario(
    scenario_id: str,
    split: str,
    speed: float,
    door: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=scenario_id,
        split=split,
        atoms=(
            AtomRequest("agent_speed", speed),
            AtomRequest("door_position", door),
        ),
    )


def _config() -> dict:
    return {
        "atoms": ["agent_speed", "door_position"],
        "evaluation_splits": ["val", "test"],
    }


def test_training_unseen_combinations_keep_atomic_support() -> None:
    scenarios = [
        _scenario("train_a", "train", 3.0, 61),
        _scenario("train_b", "train", 3.0, 85),
        _scenario("train_c", "train", 4.0, 61),
        _scenario("val", "val", 4.0, 85),
        _scenario("test", "test", 4.0, 109),
        _scenario("train_d", "train", 3.0, 109),
    ]

    result = validate_training_unseen_combinations(scenarios, _config())

    assert result["passed"]
    assert result["combination_counts"] == {
        "test": 1,
        "train": 4,
        "val": 1,
    }
    assert result["train_evaluation_overlap"] == []
    assert result["missing_train_atomic_support"] == []


def test_combination_overlap_is_a_hard_failure() -> None:
    scenarios = [
        _scenario("train_a", "train", 3.0, 61),
        _scenario("train_b", "train", 4.0, 85),
        _scenario("val_copy", "val", 3.0, 61),
    ]

    result = validate_training_unseen_combinations(scenarios, _config())

    assert not result["passed"]
    assert result["train_evaluation_overlap"] == [
        {"agent_speed": 3.0, "door_position": 61}
    ]


def test_unseen_atomic_value_is_not_mislabeled_as_combination_generalization() -> None:
    scenarios = [
        _scenario("train_a", "train", 3.0, 61),
        _scenario("train_b", "train", 4.0, 85),
        _scenario("test_unseen_value", "test", 5.0, 61),
    ]

    result = validate_training_unseen_combinations(scenarios, _config())

    assert not result["passed"]
    assert result["missing_train_atomic_support"] == [
        {
            "scenario_id": "test_unseen_value",
            "split": "test",
            "atom": "agent_speed",
            "value": 5.0,
        }
    ]
