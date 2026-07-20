from types import SimpleNamespace

from contextworld.synthesis.atoms import (
    PIXEL_EFFECT_CONTRACTS,
    tworoom_atom_registry,
)
from contextworld.synthesis.models import AtomRequest
from contextworld.synthesis.validator import (
    run_required_atom_oracles,
    validate_atom_oracle_coverage,
)


def test_every_tworoom_atom_declares_pixel_effect_oracles() -> None:
    registry = tworoom_atom_registry()

    assert registry["agent_speed"].pixel_effect == "temporal_dynamics"
    assert registry["agent_speed"].required_oracles == (
        "speed_frame_skip_oracle",
    )
    assert registry["door_position"].pixel_effect == "single_frame_geometry"
    assert registry["door_position"].required_oracles == (
        "door_position_pixel_oracle",
        "door_position_passage_oracle",
    )


def test_missing_oracle_config_is_a_hard_coverage_failure() -> None:
    scenario = SimpleNamespace(
        atoms=(AtomRequest(kind="door_position", value=70),)
    )

    result = validate_atom_oracle_coverage(
        [scenario], tworoom_atom_registry(), validation_config={}
    )

    assert not result["passed"]
    assert result["missing_config"] == {
        "door_position": [
            "door_position_pixel_oracle",
            "door_position_passage_oracle",
        ]
    }


def test_declared_oracle_without_runner_is_a_hard_failure() -> None:
    scenario = SimpleNamespace(
        atoms=(AtomRequest(kind="future_color", value=[1, 2, 3]),)
    )
    registry = {
        "future_color": SimpleNamespace(
            pixel_effect="single_frame_appearance",
            required_oracles=("future_color_oracle",),
        )
    }

    result = validate_atom_oracle_coverage(
        [scenario],
        registry,
        validation_config={"future_color_oracle": {}},
    )

    assert not result["passed"]
    assert result["missing_implementation"] == {
        "future_color": ["future_color_oracle"]
    }


def test_incomplete_oracle_evidence_cannot_pass_atom_contract() -> None:
    scenario = SimpleNamespace(
        atoms=(AtomRequest(kind="future_color", value=[1, 2, 3]),)
    )
    registry = {
        "future_color": SimpleNamespace(
            pixel_effect="single_frame_appearance",
            required_oracles=("future_color_oracle",),
        )
    }

    result = run_required_atom_oracles(
        [scenario],
        registry,
        validation_config={"future_color_oracle": {}},
        oracle_runners={
            "future_color_oracle": lambda _: {
                "passed": True,
                "evidence": {"factor_readback": True},
            }
        },
    )

    assert not result["passed"]
    assert result["atom_contracts"]["future_color"]["missing_evidence"] == sorted(
        name
        for name in PIXEL_EFFECT_CONTRACTS["single_frame_appearance"]
        if name != "factor_readback"
    )
