from pathlib import Path

import numpy as np
import pytest

from contextworld.synthesis.compiler import ScenarioCompiler
from contextworld.synthesis.models import AtomRequest, ScenarioRequest
from contextworld.synthesis.validator import validate_paired_seed_crossing


def compiler(tmp_path: Path) -> ScenarioCompiler:
    return ScenarioCompiler(
        experiment="unit",
        task="tworoom",
        env_id="swm/TwoRoom-v1",
        seed=17,
        episodes=1,
        max_episode_steps=12,
        image_shape=(224, 224),
        output_root=tmp_path,
    )


def test_compilation_is_deterministic(tmp_path: Path) -> None:
    request = ScenarioRequest(
        name="combo",
        split="train",
        atoms=(
            AtomRequest("agent_speed", 3.0),
            AtomRequest("door_position", 70),
        ),
    )
    first = compiler(tmp_path).compile_all([request])[0]
    second = compiler(tmp_path).compile_all([request])[0]

    assert first.scenario_id == second.scenario_id
    assert first.fingerprint == second.fingerprint
    assert first.env_seed == second.env_seed
    assert first.policy_seed == second.policy_seed
    assert first.variation == (
        "agent.position",
        "target.position",
        "agent.speed",
        "door.position",
    )
    np.testing.assert_array_equal(
        first.variation_values["door.position"], [70, 70, 70]
    )


def test_atom_change_changes_identity(tmp_path: Path) -> None:
    requests = [
        ScenarioRequest(
            name=name,
            split="train",
            atoms=(AtomRequest("agent_speed", value),),
        )
        for name, value in (("speed_slow", 3.0), ("speed_fast", 7.0))
    ]
    scenarios = compiler(tmp_path).compile_all(requests)
    assert scenarios[0].scenario_id != scenarios[1].scenario_id


def test_pixel_codec_is_part_of_scenario_identity(tmp_path: Path) -> None:
    request = ScenarioRequest(
        name="speed",
        split="train",
        atoms=(AtomRequest("agent_speed", 3.0),),
    )
    jpeg = compiler(tmp_path).compile_all([request])[0]
    png_compiler = ScenarioCompiler(
        experiment="unit",
        task="tworoom",
        env_id="swm/TwoRoom-v1",
        seed=17,
        episodes=1,
        max_episode_steps=12,
        image_shape=(224, 224),
        output_root=tmp_path,
        pixel_codec={"format": "png", "compress_level": 1},
    )
    png = png_compiler.compile_all([request])[0]

    assert jpeg.scenario_id != png.scenario_id
    assert png.pixel_codec["lossless"] is True


def test_reset_constraints_are_part_of_scenario_identity(tmp_path: Path) -> None:
    request = ScenarioRequest(
        name="speed",
        split="train",
        atoms=(AtomRequest("agent_speed", 5.0),),
    )
    unconstrained = compiler(tmp_path).compile_all([request])[0]
    constrained_compiler = ScenarioCompiler(
        experiment="unit",
        task="tworoom",
        env_id="swm/TwoRoom-v1",
        seed=17,
        episodes=1,
        max_episode_steps=12,
        image_shape=(224, 224),
        output_root=tmp_path,
        reset_constraints={
            "target_room": "opposite",
            "exclude_wall_zone": True,
            "minimum_initial_distance": 40.0,
        },
    )
    constrained = constrained_compiler.compile_all([request])[0]

    assert constrained.reset_constraints["target_room"] == "opposite"
    assert constrained.scenario_id != unconstrained.scenario_id


def test_scenario_reset_constraints_override_compiler_default(tmp_path: Path) -> None:
    request = ScenarioRequest(
        name="same_room_speed",
        split="train",
        atoms=(AtomRequest("agent_speed", 5.0),),
        reset_constraints={"target_room": "same"},
    )
    scenario = ScenarioCompiler(
        experiment="unit",
        task="tworoom",
        env_id="swm/TwoRoom-v1",
        seed=17,
        episodes=1,
        max_episode_steps=12,
        image_shape=(224, 224),
        output_root=tmp_path,
        reset_constraints={"target_room": "opposite"},
    ).compile_all([request])[0]

    assert scenario.reset_constraints["target_room"] == "same"


def test_duplicate_factor_is_rejected(tmp_path: Path) -> None:
    request = ScenarioRequest(
        name="duplicate",
        split="train",
        atoms=(
            AtomRequest("agent_speed", 3.0),
            AtomRequest("agent_speed", 7.0),
        ),
    )
    with pytest.raises(ValueError, match="mutates 'agent.speed' twice"):
        compiler(tmp_path).compile_all([request])


def test_seed_group_creates_paired_scenarios(tmp_path: Path) -> None:
    requests = [
        ScenarioRequest(
            name=name,
            split="train",
            atoms=(AtomRequest("agent_speed", speed),),
            regime="train",
            seed_group="paired-train",
        )
        for name, speed in (("paired_slow", 3.0), ("paired_fast", 6.0))
    ]
    scenarios = compiler(tmp_path).compile_all(requests)

    assert scenarios[0].env_seed == scenarios[1].env_seed
    assert scenarios[0].policy_seed == scenarios[1].policy_seed
    assert scenarios[0].scenario_id != scenarios[1].scenario_id
    assert scenarios[0].seed_group == "paired-train"


def test_multiple_seed_blocks_are_fully_crossed(tmp_path: Path) -> None:
    requests = [
        ScenarioRequest(
            name=f"speed_{group[-1]}_{int(speed)}",
            split="train",
            atoms=(AtomRequest("agent_speed", speed),),
            episodes=3,
            seed_group=group,
        )
        for group in ("block-a", "block-b")
        for speed in (3.0, 7.0)
    ]
    scenarios = compiler(tmp_path).compile_all(requests)

    result = validate_paired_seed_crossing(
        scenarios,
        {
            "atom": "agent_speed",
            "splits": {
                "train": {
                    "minimum_seed_groups": 2,
                    "minimum_factor_values": 2,
                    "minimum_paired_resets_per_factor": 6,
                }
            },
        },
    )

    assert result["passed"]
    assert result["splits"]["train"]["observed"] == {
        "seed_groups": 2,
        "factor_values": 2,
        "paired_resets_per_factor": 6,
    }


@pytest.mark.parametrize("value", [1.0, 11.0, float("nan")])
def test_unsafe_speed_is_rejected(tmp_path: Path, value: float) -> None:
    request = ScenarioRequest(
        name="bad_speed",
        split="test",
        atoms=(AtomRequest("agent_speed", value),),
    )
    with pytest.raises(ValueError, match="outside"):
        compiler(tmp_path).compile_all([request])
