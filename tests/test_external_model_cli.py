"""The external evaluator must enforce public Development/Test roles.

Two things are being pinned here, and the second matters more than the first.

The first is that the external entry point works: every task is reachable, the
adapter comes from the registry, and the result is explicitly Development-only.

The second is that the frozen task CLIs stay frozen.  Each release
configuration records the ``sha256`` of the sources that produced its numbers,
so a well-meaning edit to a task CLI silently invalidates the provenance of a
published result.  ``test_frozen_release_source_pins_still_match`` turns that
into a direct, named failure instead of a confusing audit error much later.

That check reads pins from both schemas the release contracts use —
``runtime.contextworld.source_sha256`` and the top-level ``identity:`` section
— and grades each pin in three tiers: byte-equal pins pass, byte drift with an
unchanged semantic fingerprint (see ``contextworld.benchmarks.source_fingerprint``)
passes with a warning, and semantic drift fails unless a correction record
registers the accepted new fingerprints.  Pins pointing outside this repository
pin the third-party runtime and are governed by ``expected_ref`` instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

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
from contextworld.benchmarks.source_fingerprint import (
    fingerprint_file,
    pinned_source_from_history,
    semantic_fingerprint,
    load_additive_scoring_extension_transitions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/benchmark"
PACKAGE_PIN_CORRECTION = (
    CONFIG_DIR / "contextworld_historical_package_pin_correction_v1.yaml"
)
RUNTIME_SOURCE_PIN_CORRECTION = (
    CONFIG_DIR / "contextworld_runtime_source_pin_correction_v1.yaml"
)
RELEASE_CONFIG_GLOB = "*_icl_release_v1.yaml"
EXPECTED_RELEASE_CONFIG_COUNT = 10


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
        assert payload["runtime_fingerprint"]["schema_version"] == 1
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

    def test_public_test_is_an_explicit_offline_final_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        binding = TASKS["contact_friction"]
        monkeypatch.setattr(type(binding), "load_builtins", lambda self: {})
        monkeypatch.setattr(
            external_model_cli,
            "build_adapter",
            lambda *a, **k: object(),
        )
        monkeypatch.setattr(
            external_model_cli, "build_development_request", lambda *a, **k: None
        )
        monkeypatch.setattr(
            external_model_cli,
            "_public_test_bundle_binding",
            lambda *a, **k: {"task": "contact_friction", "manifest_payload_files": 3},
        )
        monkeypatch.setattr(
            type(binding),
            "load_scorer",
            lambda self: lambda **kwargs: {
                "metrics": {"correct_future_rate": 0.9},
                "gate": {"passed": False},
            },
        )
        payload = external_model_cli.run(
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
        assert payload["result_kind"] == "public_test_offline_final_report_v1"
        assert payload["evaluation_split"] == "test"
        assert payload["official_scoreboard_row"] is False
        assert payload["runtime_fingerprint"]["schema_version"] == 1
        assert payload["result"]["gate"]["passed"] is False
        assert "must not be fed back into tuning" in payload["note"]

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

    def test_h3_tail_projection_is_limited_to_action_delay(self) -> None:
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

    @pytest.mark.parametrize("family", ("lewm", "pldm"))
    def test_h3_tail_projection_explicitly_selects_h3_builtin_family(
        self, family: str
    ) -> None:
        parsed = external_model_cli.parse_args(
            [
                "--task", "action_delay",
                "--adapter", family,
                "--checkpoint", "/tmp/x.pt",
                "--model-name", "m",
                "--history-adapter", "h3_tail_projection",
            ]
        )
        adapter_class = _builtins_for_run(
            TASKS["action_delay"], parsed
        )[family]
        assert adapter_class.__name__ == (
            "StableWorldModelLeWMH3TailProjectionAdapter"
            if family == "lewm"
            else "StableWorldModelPLDMH3TailProjectionAdapter"
        )


def _pins_from_release_payload(payload: dict[str, Any]) -> dict[str, str]:
    """Collect every source pin a release contract carries, in either schema.

    The contracts pin sources in two shapes: ``runtime.contextworld.
    source_sha256`` as a path-to-hash mapping, and a top-level ``identity:``
    section whose entries each name one ``path``/``sha256`` pair (written
    either as a list or as a mapping of role names to pairs).  A scanner that
    recognises only one shape silently stops auditing the other, which is
    exactly how eight of the ten contracts went unverified.
    """

    pins: dict[str, str] = {}
    runtime_pins = (
        payload.get("runtime", {}).get("contextworld", {}).get("source_sha256")
    )
    if isinstance(runtime_pins, dict):
        pins.update(
            {str(path): str(sha) for path, sha in runtime_pins.items()}
        )
    identity = payload.get("identity")
    if isinstance(identity, list):
        entries: Iterable[Any] = identity
    elif isinstance(identity, dict):
        entries = identity.values()
    else:
        entries = []
    for entry in entries:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and isinstance(entry.get("sha256"), str)
        ):
            pins[entry["path"]] = entry["sha256"]
    return pins


def _release_configs_with_source_pins() -> list[tuple[Path, dict[str, str]]]:
    found: list[tuple[Path, dict[str, str]]] = []
    for path in sorted(CONFIG_DIR.glob(RELEASE_CONFIG_GLOB)):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        if not isinstance(payload, dict):
            continue
        pins = _pins_from_release_payload(payload)
        if pins:
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
        if row["field"]
        in (
            "runtime.contextworld.source_sha256.pyproject.toml",
            "identity.package.sha256",
        )
    }


def _accepted_runtime_source_pin_corrections() -> dict[
    tuple[str, str, str], dict[str, Any]
]:
    """Load registered (config, path, pinned sha) acceptances, with teeth.

    The same ``accepted_metadata_correction`` pattern as the package-pin
    record above, generalized beyond pyproject.toml: each row accepts one
    pinned byte sha whose current file state is hash-bound here.  If the
    file moves again, the registered state stops matching and the rule
    fails, so a correction can never silently rot into a blanket exemption.
    """

    payload = yaml.safe_load(
        RUNTIME_SOURCE_PIN_CORRECTION.read_text(encoding="utf-8")
    )
    assert payload["status"] == "accepted_metadata_correction"
    assert payload["scope"] == {
        "classification_only": False,
        "historical_files_rewritten": False,
        "model_results_changed": False,
        "public_test_access_changed": False,
        "training_or_evaluation_reexecuted": False,
    }
    finding = payload["finding"]
    assert finding["classification"] == (
        "runtime_source_fingerprint_updated_behavior_unchanged"
    )
    assert finding["drift_commits"], "correction must name its drift commits"
    inertness = finding["inertness_verification"]
    assert inertness["pinned_runtime_refs"], (
        "correction must name the runtime refs it was verified against"
    )
    for probe in inertness["probes"].values():
        assert probe == {"action_history_occurrences": 0}
    accepted: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in payload["accepted_transitions"]:
        accepted[(row["config"], row["path"], row["pinned_sha256"])] = row
    superseded = payload["superseded_release_lineage"]
    for row in superseded["affected_records"]:
        accepted[(superseded["release_config"], row["path"], row["pinned_sha256"])] = row
    for row in payload["historical_execution_receipts"]:
        accepted[(row["config"], row["path"], row["pinned_sha256"])] = row
    # The additive Public Test gate completion is a different claim and lives in
    # its own record with its own strict loader: new fields emitted, every
    # pre-existing field bit-identical, every sealed result re-executed.  Its
    # assertions run inside that loader, so a row edited into a blanket
    # exemption fails there rather than silently widening this audit.
    additive = load_additive_scoring_extension_transitions(
        ROOT
        / "configs/benchmark"
        / "contextworld_additive_test_gate_completion_pin_transition_v1.yaml"
    )
    overlap = accepted.keys() & additive.keys()
    assert not overlap, f"pin registered in both correction records: {sorted(overlap)}"
    accepted.update(additive)
    return accepted


def _audit_release_source_pins(
    *,
    root: Path,
    pins: Mapping[str, str],
    config_relative: str,
    corrected_package_pins: Mapping[str, str],
    accepted_corrections: Mapping[tuple[str, str, str], dict[str, Any]],
    pinned_source_resolver: Callable[[str, str], str | None],
) -> tuple[list[str], list[str], list[str]]:
    """Grade every pin: (failures, semantic-only warnings, external pins).

    A pin fails only when evaluation behaviour could plausibly have changed:
    the byte sha drifted, no accepted correction registers the new state, and
    the semantic fingerprint either moved or cannot be compared.  Pure
    documentation and layout drift downgrades to a warning, and pins on the
    third-party runtime snapshot are reported separately because their live
    contract is ``expected_ref``, not the local checkout state.
    """

    failures: list[str] = []
    semantic_warnings: list[str] = []
    external: list[str] = []
    for relative, expected in pins.items():
        if not (root / relative).resolve().is_relative_to(root.resolve()):
            external.append(relative)
            continue
        acceptance = accepted_corrections.get(
            (config_relative, relative, expected)
        )
        source = root / relative
        if not source.is_file():
            if acceptance and acceptance.get("accepted_state") == (
                "absent_from_checkout"
            ):
                continue
            failures.append(f"{relative}: missing from the checkout")
            continue
        observed = fingerprint_file(source)
        if observed.byte_sha256 == expected:
            continue
        if (
            relative == "pyproject.toml"
            and corrected_package_pins.get(config_relative) == expected
        ):
            # This exact impossible historical packaging pin is preserved in
            # the predecessor YAML but no longer misclassified as executable
            # runtime source. The correction record is hash-bound above.
            continue
        if acceptance is not None:
            if (
                acceptance.get("accepted_current_sha256")
                == observed.byte_sha256
                and acceptance.get("accepted_current_semantic_sha256")
                == observed.semantic_sha256
            ):
                continue
            failures.append(
                f"{relative}: pinned {expected[:12]}… but file is "
                f"{observed.byte_sha256[:12]}…, and the registered state in "
                f"{RUNTIME_SOURCE_PIN_CORRECTION.name} "
                f"({acceptance.get('accepted_current_sha256', '?')[:12]}…) no "
                "longer matches either — the source moved after the "
                "correction was accepted; re-assess before trusting results"
            )
            continue
        if observed.semantic_sha256 is not None:
            pinned_source = pinned_source_resolver(relative, expected)
            if pinned_source is not None and semantic_fingerprint(
                pinned_source
            ) == observed.semantic_sha256:
                semantic_warnings.append(
                    f"{relative}: byte sha256 drifted "
                    f"({expected[:12]}… -> {observed.byte_sha256[:12]}…) but "
                    "the semantic fingerprint is unchanged — documentation or "
                    "layout-only drift; update the pin record when convenient"
                )
                continue
        semantic = (
            observed.semantic_sha256[:12] + "…"
            if observed.semantic_sha256
            else "n/a for non-Python source"
        )
        failures.append(
            f"{relative}: pinned {expected[:12]}… but file is "
            f"{observed.byte_sha256[:12]}… (semantic {semantic}); evaluation "
            "behaviour may have changed — assess whether the results sealed "
            "against this pin need re-running"
        )
    return failures, semantic_warnings, external


def test_release_configs_with_source_pins_are_discoverable() -> None:
    """Guards the guard: an empty sweep would make the next test vacuous.

    The count is pinned to ten because that is how many release contracts
    exist; when an eleventh is added this assertion forces it to be wired
    into the sweep consciously instead of appearing as a silent gap.  When
    it fails *low*, a contract stopped matching the scanner — the exact
    regression that left eight of these ten unaudited.
    """

    found = _release_configs_with_source_pins()

    assert len(found) == EXPECTED_RELEASE_CONFIG_COUNT, (
        f"expected {EXPECTED_RELEASE_CONFIG_COUNT} release contracts with "
        f"source pins, found {len(found)}: "
        + ", ".join(path.name for path, _ in found)
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

    A pin now fails only when it could change evaluation behaviour.  Byte
    drift alone — a comment, a docstring, a reformat — downgrades to a
    warning that the pin record should be refreshed, because the semantic
    fingerprint proves the parsed program is unchanged.  Semantic drift
    fails unless a correction record registers the accepted new state.
    """

    pins = dict(_release_configs_with_source_pins())[config_path]
    config_relative = config_path.relative_to(ROOT).as_posix()
    failures, semantic_warnings, _external = _audit_release_source_pins(
        root=ROOT,
        pins=pins,
        config_relative=config_relative,
        corrected_package_pins=_corrected_non_runtime_package_pins(),
        accepted_corrections=_accepted_runtime_source_pin_corrections(),
        pinned_source_resolver=lambda relative, pinned: (
            pinned_source_from_history(ROOT, relative, pinned)
        ),
    )
    for message in semantic_warnings:
        warnings.warn(message, stacklevel=2)
    assert not failures, (
        f"{config_path.name} pins sources whose evaluation behaviour may "
        "have changed:\n  " + "\n  ".join(failures)
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


def test_both_pin_schemas_are_collected_from_a_release_payload() -> None:
    """One contract shape must never hide the other again."""

    runtime_schema = {
        "runtime": {
            "contextworld": {
                "source_sha256": {"contextworld/paths.py": "a" * 64}
            }
        }
    }
    identity_list_schema = {
        "identity": [
            {"path": "contextworld/paths.py", "sha256": "b" * 64},
            {"path": "pyproject.toml", "sha256": "c" * 64},
        ]
    }
    identity_mapping_schema = {
        "identity": {
            "public_api": {"path": "contextworld/paths.py", "sha256": "d" * 64},
            "package": {"path": "pyproject.toml", "sha256": "e" * 64},
        }
    }

    assert _pins_from_release_payload(runtime_schema) == {
        "contextworld/paths.py": "a" * 64
    }
    assert _pins_from_release_payload(identity_list_schema) == {
        "contextworld/paths.py": "b" * 64,
        "pyproject.toml": "c" * 64,
    }
    assert _pins_from_release_payload(identity_mapping_schema) == {
        "contextworld/paths.py": "d" * 64,
        "pyproject.toml": "e" * 64,
    }


class TestPinDriftTiers:
    """Every grading branch, driven on a synthetic contract.

    These are the properties the audit rule is worthless without: byte drift
    with unchanged semantics must not fail, and semantic drift must.
    """

    SOURCE = "contextworld/frozen.py"
    ORIGINAL = '"""Frozen scorer."""\n\nLIMIT = 3\n'

    def _audit(
        self,
        tmp_path: Path,
        content: str,
        *,
        pinned: str | None = None,
        accepted: Mapping[tuple[str, str, str], dict[str, Any]] | None = None,
        resolver: Callable[[str, str], str | None] | None = None,
        suffix: str = ".py",
    ) -> tuple[list[str], list[str], list[str]]:
        source = tmp_path / f"contextworld/frozen{suffix}"
        source.parent.mkdir(exist_ok=True)
        source.write_text(content, encoding="utf-8")
        relative = f"contextworld/frozen{suffix}"
        return _audit_release_source_pins(
            root=tmp_path,
            pins={relative: pinned or hashlib.sha256(content.encode()).hexdigest()},
            config_relative="configs/benchmark/synthetic_icl_release_v1.yaml",
            corrected_package_pins={},
            accepted_corrections=accepted or {},
            pinned_source_resolver=resolver or (lambda rel, sha: None),
        )

    def test_byte_sha_match_passes_silently(self, tmp_path: Path) -> None:
        assert self._audit(tmp_path, self.ORIGINAL) == ([], [], [])

    def test_documentation_drift_passes_with_a_warning(
        self, tmp_path: Path
    ) -> None:
        pinned = hashlib.sha256(self.ORIGINAL.encode()).hexdigest()
        drifted = self.ORIGINAL + "# rationale note\n"

        failures, warned, _ = self._audit(
            tmp_path,
            drifted,
            pinned=pinned,
            resolver=lambda rel, sha: self.ORIGINAL,
        )

        assert failures == []
        assert len(warned) == 1
        assert "semantic fingerprint is unchanged" in warned[0]
        assert pinned[:12] in warned[0]

    def test_semantic_drift_fails_even_when_bytes_almost_match(
        self, tmp_path: Path
    ) -> None:
        pinned = hashlib.sha256(self.ORIGINAL.encode()).hexdigest()
        retuned = self.ORIGINAL.replace("LIMIT = 3", "LIMIT = 4")

        failures, warned, _ = self._audit(
            tmp_path,
            retuned,
            pinned=pinned,
            resolver=lambda rel, sha: self.ORIGINAL,
        )

        assert warned == []
        assert len(failures) == 1
        assert "evaluation behaviour may have changed" in failures[0]

    def test_unprovable_drift_fails_conservatively(self, tmp_path: Path) -> None:
        """No recoverable pinned blob means semantic equality cannot be shown."""

        pinned = hashlib.sha256(self.ORIGINAL.encode()).hexdigest()
        redocumented = self.ORIGINAL.replace(
            "Frozen scorer.", "Frozen scorer, revised."
        )

        failures, _, _ = self._audit(
            tmp_path, redocumented, pinned=pinned, resolver=lambda rel, sha: None
        )

        assert len(failures) == 1

    def test_non_python_drift_has_no_semantic_pardon(
        self, tmp_path: Path
    ) -> None:
        launcher = "#!/bin/sh\nexec python -m frozen\n"
        pinned = hashlib.sha256(launcher.encode()).hexdigest()

        failures, warned, _ = self._audit(
            tmp_path,
            launcher + "# edited\n",
            pinned=pinned,
            suffix=".sh",
        )

        assert warned == []
        assert len(failures) == 1

    def test_registered_acceptance_passes_only_in_the_registered_state(
        self, tmp_path: Path
    ) -> None:
        pinned = hashlib.sha256(self.ORIGINAL.encode()).hexdigest()
        drifted = self.ORIGINAL + "# rationale note\n"
        drifted_file = tmp_path / "contextworld/frozen.py"
        drifted_file.parent.mkdir(exist_ok=True)
        drifted_file.write_text(drifted, encoding="utf-8")
        accepted_state = fingerprint_file(drifted_file)
        acceptance = {
            (
                "configs/benchmark/synthetic_icl_release_v1.yaml",
                self.SOURCE,
                pinned,
            ): {
                "accepted_current_sha256": accepted_state.byte_sha256,
                "accepted_current_semantic_sha256": (
                    accepted_state.semantic_sha256
                ),
            }
        }

        assert self._audit(
            tmp_path, drifted, pinned=pinned, accepted=acceptance
        ) == ([], [], [])

        moved_again = drifted.replace("LIMIT = 3", "LIMIT = 4")
        failures, _, _ = self._audit(
            tmp_path, moved_again, pinned=pinned, accepted=acceptance
        )
        assert len(failures) == 1
        assert RUNTIME_SOURCE_PIN_CORRECTION.name in failures[0]

    def test_missing_sources_fail_unless_registered_absent(
        self, tmp_path: Path
    ) -> None:
        pinned = "f" * 64
        config = "configs/benchmark/synthetic_icl_release_v1.yaml"
        relative = "contextworld/gone.py"
        (tmp_path / "contextworld").mkdir(exist_ok=True)

        def audit(
            accepted: Mapping[tuple[str, str, str], dict[str, Any]]
        ) -> list[str]:
            failures, _, _ = _audit_release_source_pins(
                root=tmp_path,
                pins={relative: pinned},
                config_relative=config,
                corrected_package_pins={},
                accepted_corrections=accepted,
                pinned_source_resolver=lambda rel, sha: None,
            )
            return failures

        assert audit({}) == [f"{relative}: missing from the checkout"]

        assert audit(
            {
                (config, relative, pinned): {
                    "accepted_state": "absent_from_checkout"
                }
            }
        ) == []

    def test_pins_outside_the_repository_are_reported_not_byte_checked(
        self, tmp_path: Path
    ) -> None:
        result = _audit_release_source_pins(
            root=tmp_path,
            pins={"../stable-worldmodel/stable_worldmodel/wm/lewm.py": "0" * 64},
            config_relative="configs/benchmark/synthetic_icl_release_v1.yaml",
            corrected_package_pins={},
            accepted_corrections={},
            pinned_source_resolver=lambda rel, sha: None,
        )

        assert result == (
            [],
            [],
            ["../stable-worldmodel/stable_worldmodel/wm/lewm.py"],
        )


def test_external_runtime_lineage_pins_stay_ref_governed() -> None:
    """Pins outside this repository pin the stable-worldmodel snapshot.

    The sibling checkout legitimately moves between refs during development,
    so byte-checking it here would fail on every switch; the live contract
    for that runtime is ``runtime.stable_worldmodel.expected_ref``, which
    the scorer enforces when it loads frozen checkpoints.  What this rule
    does enforce is that such pins are recognized and enumerated rather
    than silently skipped.
    """

    expected = {
        "../stable-worldmodel/scripts/train/config/lewm.yaml",
        "../stable-worldmodel/scripts/train/config/pldm.yaml",
        "../stable-worldmodel/stable_worldmodel/wm/lewm/lewm.py",
        "../stable-worldmodel/stable_worldmodel/wm/pldm/pldm.py",
        "../stable-worldmodel/stable_worldmodel/wm/loss.py",
        "../stable-worldmodel/stable_worldmodel/wm/utils.py",
    }
    observed: dict[str, set[str]] = {}
    for path, pins in _release_configs_with_source_pins():
        external = {
            relative
            for relative in pins
            if not (ROOT / relative).resolve().is_relative_to(ROOT.resolve())
        }
        if external:
            observed[path.name] = external

    assert observed == {
        "pusht_motion_damping_icl_release_v1.yaml": expected
    }
    motion_damping = yaml.safe_load(
        (CONFIG_DIR / "pusht_motion_damping_icl_release_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert motion_damping["runtime"]["stable_worldmodel"]["expected_ref"], (
        "a contract pinning the third-party runtime by file hash must also "
        "pin the ref that governs it"
    )


def test_runtime_source_pin_correction_binds_accepted_states() -> None:
    """Every registered acceptance must describe the checkout as it is now.

    This keeps the correction honest: because the accepted byte and semantic
    fingerprints are re-derived from the live sources on every run, the
    record can never rot into a blanket exemption for a file that has since
    moved on.
    """

    accepted = _accepted_runtime_source_pin_corrections()
    assert accepted, "correction record exposed no accepted transitions"

    for (config_relative, relative, pinned_sha), row in accepted.items():
        assert (ROOT / config_relative).is_file(), config_relative
        if row.get("accepted_state") == "absent_from_checkout":
            assert not (ROOT / relative).exists(), relative
            continue
        source = ROOT / relative
        assert source.is_file(), relative
        observed = fingerprint_file(source)
        assert observed.byte_sha256 == row["accepted_current_sha256"], relative
        assert observed.semantic_sha256 == (
            row["accepted_current_semantic_sha256"]
        ), relative
        assert pinned_sha != row["accepted_current_sha256"], (
            f"{relative}: registered a no-op transition; either the pin or "
            "the record is wrong"
        )
