from pathlib import Path

from contextworld.synthesis.compiler import ScenarioCompiler
from contextworld.synthesis.models import AtomRequest, ScenarioRequest
from contextworld.synthesis.validator import validate_split_isolation


def _compile(tmp_path: Path, requests: list[ScenarioRequest]):
    compiler = ScenarioCompiler(
        experiment="split-test",
        task="tworoom",
        env_id="swm/TwoRoom-v1",
        seed=4,
        episodes=1,
        max_episode_steps=4,
        image_shape=(32, 32),
        output_root=tmp_path,
    )
    return compiler.compile_all(requests)


def test_recombined_context_is_not_exact_split_leakage(tmp_path: Path) -> None:
    scenarios = _compile(
        tmp_path,
        [
            ScenarioRequest(
                "seen", "train", (AtomRequest("agent_speed", 3.0),)
            ),
            ScenarioRequest(
                "train_unseen_combo",
                "test",
                (
                    AtomRequest("agent_speed", 3.0),
                    AtomRequest("door_position", 70),
                ),
            ),
        ],
    )
    assert validate_split_isolation(scenarios)["passed"]


def test_exact_atom_set_across_splits_is_rejected(tmp_path: Path) -> None:
    scenarios = _compile(
        tmp_path,
        [
            ScenarioRequest(
                "train_copy", "train", (AtomRequest("agent_speed", 3.0),)
            ),
            ScenarioRequest(
                "test_copy", "test", (AtomRequest("agent_speed", 3.0),)
            ),
        ],
    )
    assert not validate_split_isolation(scenarios)["passed"]
