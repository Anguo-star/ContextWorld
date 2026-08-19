"""External models must reach the frozen protocol without disturbing it.

Two things are being pinned here, and the second matters more than the first.

The first is that the external entry point works: every task is reachable, the
adapter comes from the registry, and the result is labelled unofficial.

The second is that the frozen task CLIs stay frozen.  Each release
configuration records the ``sha256`` of the sources that produced its numbers,
so a well-meaning edit to a task CLI silently invalidates the provenance of a
published result.  ``test_frozen_release_source_pins_still_match`` turns that
into a direct, named failure instead of a confusing audit error much later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from contextworld.benchmarks import external_model_cli
from contextworld.benchmarks.adapter_registry import AdapterRequest
from contextworld.benchmarks.external_model_cli import (
    RESULT_KIND,
    TASKS,
    build_request,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/benchmark"


class TestTaskBindings:
    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_every_binding_resolves(self, task: str) -> None:
        """A typo in the table would otherwise surface only at runtime."""

        binding = TASKS[task]
        assert callable(binding.load_scorer())
        families = binding.load_builtins()
        assert set(families) == {"lewm", "pldm"}
        assert all(isinstance(value, type) for value in families.values())

    def test_the_nine_benchmark_tasks_are_all_reachable(self) -> None:
        assert sorted(TASKS) == [
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

    def test_statistics_tasks_declare_which_deviation_they_use(self) -> None:
        """PushT-family tasks are not uniform: portal_exit is unbiased."""

        for task, binding in TASKS.items():
            if binding.action_source == "statistics":
                assert binding.std_key in {"std_population", "std_unbiased"}, task
            else:
                assert binding.std_key is None, task
        assert TASKS["portal_exit"].std_key == "std_unbiased"


class TestRequestConstruction:
    def _args(self, **overrides: Any) -> argparse.Namespace:
        base = {
            "checkpoint": Path("/tmp/model.ckpt"),
            "device": "cpu",
            "stablewm_repo": None,
            "stablewm_ref": None,
        }
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_normalizer_task_builds_a_normalizer_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            external_model_cli,
            "resolve_contextworld_path",
            lambda *a, **k: tmp_path / "norm.json",
        )
        release = {
            "runtime": {"stable_worldmodel": {"repo": "r", "expected_ref": "c"}},
            "evaluation": {"normalizer": "norm.json"},
        }
        request = build_request(TASKS["speed"], release, self._args())
        assert request.action_normalizer == tmp_path / "norm.json"
        assert request.action_mean is None
        assert request.runtime["stablewm_ref"] == "c"

    def test_statistics_task_reads_the_frozen_normalization(self) -> None:
        release = {
            "runtime": {"stable_worldmodel": {"repo": "r", "expected_ref": "c"}},
            "evaluation": {
                "action_normalization": {
                    "mean": [1.0, 2.0],
                    "std_population": [3.0, 4.0],
                    "std_unbiased": [5.0, 6.0],
                }
            },
        }
        request = build_request(TASKS["action_strength"], release, self._args())
        assert request.action_mean == [1.0, 2.0]
        assert request.action_std == [3.0, 4.0]
        assert request.action_normalizer is None

    def test_portal_exit_uses_the_unbiased_deviation(self) -> None:
        """It differs from its siblings, so the binding is checked directly."""

        release = {
            "runtime": {"stable_worldmodel": {"repo": "r"}},
            "evaluation": {
                "action_normalization": {
                    "mean": [1.0],
                    "std_population": [3.0],
                    "std_unbiased": [5.0],
                }
            },
        }
        request = build_request(TASKS["portal_exit"], release, self._args())
        assert request.action_std == [5.0]

    def test_action_geometry_is_not_taken_from_the_command_line(self) -> None:
        """An external model must not be able to pick its own normalization."""

        parsed = external_model_cli.parse_args(
            [
                "--task", "speed",
                "--adapter", "pkg:Cls",
                "--checkpoint", "/tmp/x.pt",
                "--model-name", "m",
            ]
        )
        assert not hasattr(parsed, "action_mean")
        assert not hasattr(parsed, "action_std")
        assert not hasattr(parsed, "normalizer")


class TestResultLabelling:
    def test_result_is_stamped_unofficial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding = TASKS["speed"]
        monkeypatch.setattr(
            type(binding), "load_release",
            lambda self: {"release_id": "frozen-speed-v1"},
        )
        monkeypatch.setattr(
            type(binding), "load_builtins", lambda self: {}
        )
        monkeypatch.setattr(
            external_model_cli, "build_adapter", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            external_model_cli, "build_request", lambda *a, **k: None
        )
        monkeypatch.setattr(
            type(binding),
            "load_scorer",
            lambda self: (lambda **kwargs: {"icl_score": 0.5}),
        )

        payload = external_model_cli.run(
            argparse.Namespace(
                task="speed",
                adapter="pkg:Cls",
                model_name="my-model",
                training_recipe="external_method",
                training_seed=None,
                batch_size=8,
                checkpoint=Path("/tmp/x.pt"),
                device="cpu",
                stablewm_repo=None,
                stablewm_ref=None,
            )
        )
        assert payload["result_kind"] == RESULT_KIND == "external_unofficial"
        assert payload["official_scoreboard_row"] is False
        assert payload["release_id"] == "frozen-speed-v1"
        # The scorer's payload is nested, never spread into the envelope, so
        # an external result cannot be replayed as a frozen submission.
        assert payload["result"] == {"icl_score": 0.5}
        assert "icl_score" not in payload

    def test_speed_receives_its_three_batch_sizes(self) -> None:
        keywords = external_model_cli._scorer_keywords(
            TASKS["speed"],
            argparse.Namespace(
                model_name="m",
                training_recipe="r",
                training_seed=1,
                batch_size=16,
            ),
        )
        assert keywords["encode_batch_size"] == 16
        assert keywords["rollout_batch_size"] == 16
        assert keywords["bundle_batch_size"] == 16
        assert "batch_size" not in keywords
        # speed names the argument differently from every other task
        assert keywords["training_role"] == "r"

    def test_other_tasks_receive_a_single_batch_size(self) -> None:
        keywords = external_model_cli._scorer_keywords(
            TASKS["door"],
            argparse.Namespace(
                model_name="m",
                training_recipe="r",
                training_seed=1,
                batch_size=16,
            ),
        )
        assert keywords["batch_size"] == 16
        assert keywords["training_recipe"] == "r"
        assert "encode_batch_size" not in keywords


class TestArgumentParsing:
    def test_unknown_task_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            external_model_cli.parse_args(
                [
                    "--task", "no_such_task",
                    "--adapter", "pkg:Cls",
                    "--checkpoint", "/tmp/x.pt",
                    "--model-name", "m",
                ]
            )

    def test_adapter_is_required_and_unrestricted(self) -> None:
        parsed = external_model_cli.parse_args(
            [
                "--task", "door",
                "--adapter", "some_package.mod:Adapter",
                "--checkpoint", "/tmp/x.pt",
                "--model-name", "m",
            ]
        )
        assert parsed.adapter == "some_package.mod:Adapter"


def _release_configs_with_source_pins() -> list[tuple[Path, dict[str, str]]]:
    found: list[tuple[Path, dict[str, str]]] = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(payload, dict):
            continue
        pins = (
            payload.get("runtime", {})
            .get("contextworld", {})
            .get("source_sha256")
        )
        if isinstance(pins, dict) and pins:
            found.append((path, pins))
    return found


def test_release_configs_with_source_pins_are_discoverable() -> None:
    """Guards the guard: an empty sweep would make the next test vacuous."""

    assert _release_configs_with_source_pins(), (
        "no release configuration exposed runtime.contextworld.source_sha256; "
        "the frozen-source check below would silently verify nothing"
    )


@pytest.mark.parametrize(
    "config_path",
    [path for path, _ in _release_configs_with_source_pins()],
    ids=lambda path: path.name,
)
def test_frozen_release_source_pins_still_match(config_path: Path) -> None:
    """Editing a hash-pinned source invalidates a published result's provenance.

    This is the failure mode that made the external entry point necessary: the
    obvious way to support a third model family is to widen ``--adapter`` on
    each task CLI, and that silently breaks every pin recorded here.  The
    external path exists precisely so these stay untouched.
    """

    pins = dict(_release_configs_with_source_pins())[config_path]
    drifted = []
    for relative, expected in pins.items():
        source = ROOT / relative
        if not source.is_file():
            drifted.append(f"{relative}: missing from the checkout")
            continue
        observed = hashlib.sha256(source.read_bytes()).hexdigest()
        if observed != expected:
            drifted.append(
                f"{relative}: pinned {expected[:12]}… but file is "
                f"{observed[:12]}…"
            )
    assert not drifted, (
        f"{config_path.name} pins sources that have changed:\n  "
        + "\n  ".join(drifted)
    )


def test_the_external_entry_point_is_not_itself_pinned() -> None:
    """It must stay editable, which is the entire point of separating it."""

    module = "contextworld/benchmarks/external_model_cli.py"
    pinning = [
        path.name
        for path, pins in _release_configs_with_source_pins()
        if module in pins
    ]
    assert not pinning, (
        f"{module} became hash-pinned by {pinning}; external-model support "
        "would then be frozen against the very configs it must not disturb"
    )
