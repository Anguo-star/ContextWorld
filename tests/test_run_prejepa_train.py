"""The prejepa launcher hands off to Stable-WorldModel without reimplementing it.

Two things are worth holding still. The first is that this really is a launcher:
it composes upstream's own training script and sets only what the benchmark
fixes, so the objective stays upstream's. The second is the per-task geometry,
where the three asymmetries (action_delay's history 7, cube's action width 5,
and the frameskip multiplier) are exactly what a hand-written command gets
wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_prejepa_train as launcher  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout skeleton holding just the training script."""

    train = tmp_path / "scripts/train"
    train.mkdir(parents=True)
    (train / "prejepa.py").write_text("", encoding="utf-8")
    (train / "lewm.py").write_text("", encoding="utf-8")
    return tmp_path


def _overrides(task: str, **keywords: object) -> dict[str, str]:
    defaults: dict[str, object] = {
        "task": task,
        "run_name": "r",
        "dataset": "d",
        "seed": 3072,
        "output": None,
        "batch_size": None,
        "num_workers": None,
        "devices": None,
        "max_epochs": None,
        "precision": None,
        "override": [],
    }
    defaults.update(keywords)
    import argparse

    args = argparse.Namespace(**defaults)
    pairs = {}
    for entry in launcher.build_overrides(args):
        key, _, value = entry.partition("=")
        pairs[key] = value
    return pairs


class TestTaskGeometry:
    def test_all_nine_benchmark_tasks_are_launchable(self) -> None:
        assert sorted(launcher.TASK_GEOMETRY) == [
            "action_delay",
            "action_strength",
            "contact_friction",
            "cube_gripper_carry",
            "door",
            "motion_damping",
            "portal_exit",
            "robot_arm_mass",
            "speed",
        ]

    def test_action_delay_asks_for_history_seven(self) -> None:
        """The other eight tasks are history 3; this one is not."""

        assert _overrides("action_delay")["wm.history_size"] == "7"
        assert _overrides("speed")["wm.history_size"] == "3"

    def test_cube_widens_the_action_encoder(self) -> None:
        """Cube's raw action width is 5, every other task's is 2.

        The encoder input is the raw width times the frameskip, matching how
        the benchmark packs an action block.
        """

        assert _overrides("cube_gripper_carry")[
            "model.action_encoder.input_dim"
        ] == "25"
        assert _overrides("speed")["model.action_encoder.input_dim"] == "10"

    @pytest.mark.parametrize("task", sorted(launcher.TASK_GEOMETRY))
    def test_every_task_pins_geometry_and_frameskip(self, task: str) -> None:
        pairs = _overrides(task)

        assert pairs["frameskip"] == "5"
        assert int(pairs["wm.history_size"]) > 0
        assert int(pairs["model.action_encoder.input_dim"]) > 0


class TestItRemainsALauncher:
    def test_it_sets_no_loss_or_objective_override(self) -> None:
        """The objective is upstream's. Setting it here would fork the recipe."""

        for task in launcher.TASK_GEOMETRY:
            for key in _overrides(task):
                assert not key.startswith("loss")
                assert "regularizer" not in key
                assert "idm" not in key

    def test_hardware_flags_are_omitted_unless_asked_for(self) -> None:
        """An unset knob must keep the upstream default, not a copy of it."""

        pairs = _overrides("speed")

        assert "batch_size" not in pairs
        assert "trainer.devices" not in pairs
        assert "trainer.max_epochs" not in pairs

    def test_hardware_flags_pass_through_when_given(self) -> None:
        pairs = _overrides("speed", batch_size=64, devices=8, max_epochs=5)

        assert pairs["batch_size"] == "64"
        assert pairs["trainer.devices"] == "8"
        assert pairs["trainer.max_epochs"] == "5"

    def test_arbitrary_overrides_are_forwarded_last(self) -> None:
        """Hydra takes the last value, so an explicit override must win."""

        import argparse

        args = argparse.Namespace(
            task="speed", run_name="r", dataset="d", seed=1, output=None,
            batch_size=32, num_workers=None, devices=None, max_epochs=None,
            precision=None, override=["batch_size=99"],
        )
        entries = launcher.build_overrides(args)

        assert entries[-1] == "batch_size=99"


class TestScriptResolution:
    def test_it_defaults_to_the_family_script(self, repo: Path) -> None:
        resolved = launcher.resolve_train_script(repo, None)

        assert resolved == repo / "scripts/train/prejepa.py"

    def test_an_explicit_script_wins(self, repo: Path) -> None:
        """This is what makes a fork or a new family reachable."""

        target = repo / "scripts/train/lewm.py"

        assert launcher.resolve_train_script(repo, str(target)) == target

    def test_the_environment_variable_is_honoured(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = repo / "scripts/train/lewm.py"
        monkeypatch.setenv("CONTEXTWORLD_TRAIN_SCRIPT", str(target))

        assert launcher.resolve_train_script(repo, None) == target

    def test_a_missing_script_fails_loudly(self, repo: Path) -> None:
        with pytest.raises(SystemExit, match="not found"):
            launcher.resolve_train_script(repo, str(repo / "nope.py"))

    def test_a_checkout_without_training_code_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("STABLEWM_REPO", raising=False)
        monkeypatch.delenv(
            "CONTEXTWORLD_STABLE_WORLDMODEL_REPO", raising=False
        )

        with pytest.raises(SystemExit, match="not found"):
            launcher.resolve_stablewm_repo(str(tmp_path))
