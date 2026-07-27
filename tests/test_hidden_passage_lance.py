from __future__ import annotations

import numpy as np

from contextworld.evaluation.hidden_passage import (
    make_templates,
    simulate_template,
)
from contextworld.evaluation.hidden_passage_env import (
    PASSAGE_RULES,
    make_hidden_passage_env,
)
from contextworld.evaluation.hidden_passage_lance import (
    RAW_STEPS,
    _collection_actions,
    _model_blocks,
    audit_hidden_passage_lance_pairs,
)
from contextworld.synthesis.atoms import (
    AtomValidationError,
    PassageOpenAtom,
)


def _template(direction: str):
    goal = (190.0, 63.0) if direction == "left_to_right" else (30.0, 63.0)
    return make_templates(
        door_positions=[49],
        directions=[direction],
        doorway_offsets_px=[14.0],
        catalog_seed=23,
        goal_state=goal,
    )[0]


def test_world_action_rotation_produces_exact_history3_blocks() -> None:
    reference = simulate_template(_template("left_to_right"), rule="passable")
    collection = _collection_actions(reference)
    stored = np.concatenate([collection[1:], collection[:1]], axis=0)

    assert collection.shape == (RAW_STEPS, 2)
    assert np.array_equal(
        stored.reshape(4, 5, 2),
        _model_blocks(reference),
    )


def test_hidden_passage_pair_contract_survives_model_projection() -> None:
    assets = {}
    for direction in ("left_to_right", "right_to_left"):
        for rule in ("passable", "blocked"):
            reference = simulate_template(_template(direction), rule=rule)
            assets[f"{direction}-{rule}"] = {
                "model_pixels": np.concatenate(
                    [
                        reference["history_pixels"],
                        reference["target_pixels"][None],
                    ],
                    axis=0,
                ),
                "model_actions": _model_blocks(reference),
                "model_proprio": np.concatenate(
                    [
                        reference["history_states"],
                        reference["target_state"][None],
                    ],
                    axis=0,
                ),
            }

    audit = audit_hidden_passage_lance_pairs(assets)

    assert audit["passed"]
    assert set(audit["directions"]) == {
        "left_to_right",
        "right_to_left",
    }
    assert all(
        row["middle_state_gap_px"] == 8.5
        and row["future_state_gap_px"] == 25.0
        for row in audit["directions"].values()
    )


def test_hidden_passage_restore_applies_rule_before_state_and_goal() -> None:
    env = make_hidden_passage_env(render_mode="rgb_array")
    try:
        env.reset(seed=1)
        env.restore_contextworld_hidden_passage(
            passage_open=PASSAGE_RULES["blocked"],
            state=np.asarray([98.0, 63.0], dtype=np.float32),
            goal_state=np.asarray([190.0, 63.0], dtype=np.float32),
        )
        assert env.passage_open == PASSAGE_RULES["blocked"]
        assert env._contextworld_hidden_passage_readback == 0
        assert np.array_equal(
            env.agent_position.detach().cpu().numpy(),
            np.asarray([98.0, 63.0], dtype=np.float32),
        )
        assert np.array_equal(
            env.target_position.detach().cpu().numpy(),
            np.asarray([190.0, 63.0], dtype=np.float32),
        )
    finally:
        env.close()


def test_passage_open_atom_is_strict_and_scalar() -> None:
    atom = PassageOpenAtom()
    assert atom.compile(0).variation_value == 0
    assert atom.compile(True).variation_value == 1
    for invalid in (-1, 2, 0.5, "open", [0, 1]):
        try:
            atom.compile(invalid)
        except AtomValidationError:
            pass
        else:
            raise AssertionError(f"Accepted invalid passage rule {invalid!r}")
