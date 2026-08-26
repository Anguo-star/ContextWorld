"""The external evaluator must stay on the public Development boundary.

Two things are being pinned here, and the second matters more than the first.

The first is that the external entry point works: every task is reachable, the
adapter comes from the registry, and the result is explicitly Development-only.

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
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from contextworld.benchmarks import external_model_cli
from contextworld.benchmarks.adapter_registry import AdapterRequest
from contextworld.benchmarks.adapters import LatentWorldModelAdapter
from contextworld.benchmarks.external_model_cli import (
    RESULT_KIND,
    TASKS,
    _BUILTIN_FAMILIES,
    _builtins_for_run,
    build_request,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/benchmark"
PACKAGE_PIN_CORRECTION = (
    CONFIG_DIR / "contextworld_historical_package_pin_correction_v1.yaml"
)


class TestTaskBindings:
    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_every_binding_resolves(self, task: str) -> None:
        """A typo in the table would otherwise surface only at runtime."""

        binding = TASKS[task]
        assert callable(binding.load_scorer())
        families = binding.load_builtins()
        assert set(families) == set(_BUILTIN_FAMILIES)
        assert all(isinstance(value, type) for value in families.values())

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_every_family_covers_every_task(self, task: str) -> None:
        """Each built-in family must reach all nine tasks with real geometry.

        The class names are assembled by string interpolation, so a family
        that is missing one task's variant fails only when that task runs.
        """

        families = TASKS[task].load_builtins()
        for name, adapter in families.items():
            assert issubclass(adapter, LatentWorldModelAdapter), name
            assert not getattr(adapter, "__abstractmethods__", None), name
            assert adapter.required_history_tokens > 0, name
            assert adapter.raw_action_dim > 0, name

    def test_families_agree_on_geometry_within_a_task(self) -> None:
        """A family swap must not silently change what a task evaluates.

        Geometry belongs to the task, not the model, so every family bound to
        one task must declare identical history, horizon and action width. If
        they diverge, two families' numbers for that task are not comparable.
        """

        for task, binding in TASKS.items():
            geometries = {
                name: (
                    adapter.required_history_tokens,
                    adapter.maximum_future_action_blocks,
                    adapter.raw_action_dim,
                    adapter.action_input_dim,
                )
                for name, adapter in binding.load_builtins().items()
            }
            assert len(set(geometries.values())) == 1, (task, geometries)

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

    def test_cube_binding_uses_the_current_v4r1_release(self) -> None:
        release = TASKS["cube_gripper_carry"].load_release()

        assert release["release_id"] == (
            "contextworld_cube_gripper_carry_icl_history3_v4r1"
        )

    def test_statistics_tasks_declare_which_deviation_they_use(self) -> None:
        """PushT-family tasks are not uniform: portal_exit is unbiased."""

        for task, binding in TASKS.items():
            if binding.action_source == "statistics":
                assert binding.std_key in {"std_population", "std_unbiased"}, task
            else:
                assert binding.std_key is None, task
        assert TASKS["portal_exit"].std_key == "std_unbiased"

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_explicit_normalized_zero_uses_task_matched_diagnostic_prejepa(
        self, task: str
    ) -> None:
        binding = TASKS[task]
        regular = binding.load_builtins()["prejepa"]
        diagnostic = _builtins_for_run(
            binding,
            argparse.Namespace(
                task=task,
                adapter="prejepa",
                prejepa_missing_context_policy="normalized_zero",
                history_adapter="native",
            ),
        )["prejepa"]

        assert diagnostic is not regular
        assert diagnostic.required_history_tokens == regular.required_history_tokens
        assert diagnostic.maximum_future_action_blocks == (
            regular.maximum_future_action_blocks
        )
        assert diagnostic.raw_action_dim == regular.raw_action_dim
        assert diagnostic.missing_context_strategy == "normalized_zero"

    def test_action_delay_h3_tail_is_an_explicit_prejepa_override(self) -> None:
        adapter = _builtins_for_run(
            TASKS["action_delay"],
            argparse.Namespace(
                task="action_delay",
                adapter="prejepa",
                prejepa_missing_context_policy="normalized_zero",
                history_adapter="h3_tail_projection",
            ),
        )["prejepa"]

        assert adapter.__name__ == (
            "StableWorldModelPreJEPADiagnosticActionDelayH3TailAdapter"
        )
        assert adapter.required_history_tokens == 7


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
    def test_result_is_stamped_development_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding = TASKS["speed"]
        monkeypatch.setattr(
            type(binding), "load_builtins", lambda self: {}
        )
        monkeypatch.setattr(
            external_model_cli, "build_adapter", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            external_model_cli, "build_development_request", lambda *a, **k: None
        )
        monkeypatch.setattr(
            external_model_cli,
            "evaluate_bundle_development_model",
            lambda **kwargs: {"metrics": {"icl_score": 0.5}},
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
                benchmark_root="/tmp/ContextWorld-v1",
                evaluation_split="development",
            )
        )
        assert payload["result_kind"] == RESULT_KIND == "development_only_not_public_test"
        assert payload["official_scoreboard_row"] is False
        # The evaluator payload is nested, never spread into the envelope, so
        # a Development result cannot be replayed as a held-out result.
        assert payload["result"] == {"metrics": {"icl_score": 0.5}}
        assert "icl_score" not in payload
        assert "not a held-out Public Test score" in payload["note"]

    def test_normalized_zero_result_stays_development_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding = TASKS["speed"]
        monkeypatch.setattr(type(binding), "load_builtins", lambda self: {})
        monkeypatch.setattr(
            external_model_cli, "build_adapter", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            external_model_cli, "build_development_request", lambda *a, **k: None
        )
        monkeypatch.setattr(
            external_model_cli,
            "evaluate_bundle_development_model",
            lambda **kwargs: {"metrics": {"icl_score": 0.5}},
        )

        payload = external_model_cli.run(
            argparse.Namespace(
                task="speed",
                adapter="prejepa",
                model_name="dino-wm",
                training_recipe="external_method",
                training_seed=None,
                batch_size=8,
                checkpoint=Path("/tmp/x.pt"),
                device="cpu",
                stablewm_repo=None,
                stablewm_ref=None,
                benchmark_root="/tmp/ContextWorld-v1",
                evaluation_split="development",
                prejepa_missing_context_policy="normalized_zero",
                history_adapter="native",
            )
        )

        assert payload["result_kind"] == RESULT_KIND
        assert payload["official_scoreboard_row"] is False
        assert payload["diagnostic"] == {
            "classification": "diagnostic",
            "prejepa_missing_context_policy": "normalized_zero",
            "frozen_v1_compatible": False,
            "history_adapter": "native",
        }

    def test_public_split_is_rejected_before_model_or_data_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            external_model_cli,
            "build_adapter",
            lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("Public Test must be rejected before adapter build")
            ),
        )
        with pytest.raises(ValueError, match="Public Test is not available"):
            external_model_cli.run(
                argparse.Namespace(
                    task="contact_friction",
                    adapter="pkg:Cls",
                    model_name="my-model",
                    training_recipe="external_method",
                    training_seed=None,
                    batch_size=8,
                    checkpoint=Path("/tmp/x.pt"),
                    device="cpu",
                    stablewm_repo=None,
                    stablewm_ref=None,
                    benchmark_root="/tmp/ContextWorld-v1",
                    evaluation_split="public",
                )
            )

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

    def test_public_adapter_cache_never_falls_back_to_private_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        checkpoint = tmp_path / "checkpoints" / "run" / "weights.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        monkeypatch.setenv(
            "CONTEXTWORLD_ARTIFACT_ROOT", "/private/context_world"
        )
        monkeypatch.delenv("CONTEXTWORLD_MODEL_CACHE_ROOT", raising=False)
        monkeypatch.delenv("STABLEWM_HOME", raising=False)
        args = argparse.Namespace(checkpoint=checkpoint)

        expected = tmp_path / ".contextworld-eval-cache"
        with external_model_cli._public_model_cache_scope(args) as cache_root:
            assert cache_root == expected
            assert Path(os.environ["CONTEXTWORLD_ARTIFACT_ROOT"]) == expected

        assert os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] == "/private/context_world"

    def test_public_adapter_cache_override_must_be_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CONTEXTWORLD_MODEL_CACHE_ROOT", "relative/cache")
        with pytest.raises(ValueError, match="must be absolute"):
            external_model_cli._public_model_cache_root(
                argparse.Namespace(checkpoint=tmp_path / "weights.pt")
            )


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

    def test_normalized_zero_requires_the_builtin_prejepa_adapter(self) -> None:
        with pytest.raises(SystemExit):
            external_model_cli.parse_args(
                [
                    "--task", "speed",
                    "--adapter", "lewm",
                    "--checkpoint", "/tmp/x.pt",
                    "--model-name", "m",
                    "--prejepa-missing-context-policy", "normalized_zero",
                ]
            )

    def test_h3_tail_projection_is_limited_to_action_delay_prejepa(self) -> None:
        with pytest.raises(SystemExit):
            external_model_cli.parse_args(
                [
                    "--task", "speed",
                    "--adapter", "prejepa",
                    "--checkpoint", "/tmp/x.pt",
                    "--model-name", "m",
                    "--history-adapter", "h3_tail_projection",
                ]
            )

        parsed = external_model_cli.parse_args(
            [
                "--task", "action_delay",
                "--adapter", "prejepa",
                "--checkpoint", "/tmp/x.pt",
                "--model-name", "m",
                "--history-adapter", "h3_tail_projection",
            ]
        )
        assert parsed.history_adapter == "h3_tail_projection"


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


def _corrected_non_runtime_package_pins() -> dict[str, str]:
    payload = yaml.safe_load(PACKAGE_PIN_CORRECTION.read_text(encoding="utf-8"))
    assert payload["status"] == "accepted_metadata_correction"
    assert payload["scope"] == {
        "classification_only": True,
        "historical_files_rewritten": False,
        "model_results_changed": False,
        "public_test_access_changed": False,
        "training_or_evaluation_reexecuted": False,
    }
    invalid = payload["finding"]["invalid_sha256"]
    assert payload["finding"]["role_after_correction"] == (
        "historical_packaging_metadata_not_runtime_source"
    )
    return {
        row["config"]["path"]: invalid
        for row in payload["affected_records"]
        if row["field"] == "runtime.contextworld.source_sha256.pyproject.toml"
    }


def test_release_configs_with_source_pins_are_discoverable() -> None:
    """Guards the guard: an empty sweep would make the next test vacuous."""

    assert _release_configs_with_source_pins(), (
        "no release configuration exposed runtime.contextworld.source_sha256; "
        "the frozen-source check below would silently verify nothing"
    )


def test_historical_package_pin_correction_binds_unchanged_predecessors() -> None:
    payload = yaml.safe_load(PACKAGE_PIN_CORRECTION.read_text(encoding="utf-8"))
    records = [
        row["config"] for row in payload["affected_records"]
    ] + [payload["predecessor_binding"]["integrity_reseal_v2_decision"]]
    for record in records:
        path = ROOT / record["path"]
        assert path.is_file()
        assert path.stat().st_size == record["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert payload["finding"]["invalid_sha256"] != payload["finding"][
        "actual_sha256_at_introduction_commit"
    ]


def test_historical_package_pin_correction_covers_every_live_config_occurrence() -> None:
    """The metadata exception must be exhaustive, exact, and non-wildcarded."""

    payload = yaml.safe_load(PACKAGE_PIN_CORRECTION.read_text(encoding="utf-8"))
    invalid = payload["finding"]["invalid_sha256"]
    listed = {row["config"]["path"] for row in payload["affected_records"]}
    observed = {
        path.relative_to(ROOT).as_posix()
        for path in CONFIG_DIR.glob("*.yaml")
        if not path.name.startswith("contextworld_historical_package_pin_correction_")
        and invalid in path.read_text(encoding="utf-8")
    }

    assert observed == listed


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
    corrected_package_pins = _corrected_non_runtime_package_pins()
    config_relative = config_path.relative_to(ROOT).as_posix()
    drifted = []
    for relative, expected in pins.items():
        if (
            relative == "pyproject.toml"
            and corrected_package_pins.get(config_relative) == expected
        ):
            # This exact impossible historical packaging pin is preserved in
            # the predecessor YAML but no longer misclassified as executable
            # runtime source. The correction record is hash-bound above.
            continue
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
