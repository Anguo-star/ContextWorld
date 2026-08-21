"""The prejepa launcher hands off to Stable-WorldModel without reimplementing it.

Two things are worth holding still. The first is that this really is a launcher:
it composes upstream's own training script and sets only what the benchmark
fixes, so the objective stays upstream's. The second is the per-task geometry,
where the three asymmetries (action_delay's history 7, cube's action width 5,
and the frameskip multiplier) are exactly what a hand-written command gets
wrong.
"""

from __future__ import annotations

import argparse
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


def _namespace(**keywords: object) -> argparse.Namespace:
    """A parsed command line with everything the launcher reads."""

    defaults: dict[str, object] = {
        "task": "speed",
        "run_name": "r",
        "dataset": "d",
        "seed": 3072,
        "output": None,
        "batch_size": None,
        "num_workers": None,
        "devices": None,
        "accumulate": None,
        "max_epochs": None,
        "precision": None,
        "override": [],
    }
    defaults.update(keywords)
    return argparse.Namespace(**defaults)


def _overrides(task: str, **keywords: object) -> dict[str, str]:
    pairs = {}
    for entry in launcher.build_overrides(_namespace(task=task, **keywords)):
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

    def test_action_width_is_derived_by_upstream(self) -> None:
        """PreJEPA builds extra encoders after loading the dataset.

        ``model.action_encoder`` is not a key in prejepa.yaml; passing it was
        both stale and a Hydra composition error for Cube.
        """

        for task in launcher.TASK_GEOMETRY:
            assert not any(
                "action_encoder.input_dim" in key for key in _overrides(task)
            )

    @pytest.mark.parametrize("task", sorted(launcher.TASK_GEOMETRY))
    def test_every_task_pins_geometry_and_frameskip(self, task: str) -> None:
        pairs = _overrides(task)

        assert pairs["frameskip"] == "5"
        assert int(pairs["wm.history_size"]) > 0
        assert pairs["++wandb.enabled"] == "false"


class TestItRemainsALauncher:
    def test_checkpoints_are_isolated_by_run_name(self) -> None:
        """prejepa.yaml defaults ``subdir`` to null, which makes concurrent
        jobs share config and resume state below STABLEWM_HOME/checkpoints."""

        pairs = _overrides("speed", run_name="speed_prejepa_s3072")

        assert pairs["subdir"] == "speed_prejepa_s3072"
        assert pairs["output_model_name"] == "speed_prejepa_s3072"

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

        entries = launcher.build_overrides(
            _namespace(batch_size=32, override=["batch_size=99"])
        )

        assert entries[-1] == "batch_size=99"


class TestBaselineBatchSize:
    """lewm/pldm are aligned by upstream default; prejepa is the odd one out.

    ``lewm.yaml`` and ``pldm.yaml`` both ship ``batch_size: 128`` with no
    gradient accumulation. ``prejepa.yaml`` ships 32, so it is the single
    value an operator has to override to train it like the baselines.
    """

    def test_the_baseline_batch_size_is_reachable(self) -> None:
        pairs = _overrides("speed", batch_size=launcher.BASELINE_BATCH_SIZE)

        assert pairs["batch_size"] == "128"

    def test_accumulation_reaches_the_upstream_trainer(self) -> None:
        """``pl.Trainer(**cfg.trainer)`` accepts it. Upstream uses none, but
        fewer GPUs than the recipe assumed is exactly when it is needed."""

        pairs = _overrides("speed", accumulate=2)

        assert pairs["trainer.accumulate_grad_batches"] == "2"

    def test_accumulation_is_omitted_when_unset(self) -> None:
        assert "trainer.accumulate_grad_batches" not in _overrides("speed")


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
