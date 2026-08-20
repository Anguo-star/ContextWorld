"""The ``prejepa`` family reaches the benchmark without touching pinned bytes.

Two properties are worth holding still. The first is that ``adapters.py``
stays byte-identical to what the frozen release configs pin -- adding a model
family must never be a reason to break a published result's provenance. The
second is the rollout shim, which reconciles three differences between
``prejepa`` and the ``lewm``/``pldm`` rollout surface. Each of those has a
silent failure mode, so each gets a test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from contextworld.benchmarks.adapters import LatentWorldModelAdapter
from contextworld.benchmarks.prejepa_adapters import (
    _PreJEPARolloutShim,
    StableWorldModelPreJEPAAdapter,
    StableWorldModelPreJEPACubeGraspRuleAdapter,
    StableWorldModelPreJEPAHistory7Adapter,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "contextworld/benchmarks/adapters.py"


class _FakePreJEPA:
    """Stands in for the Stable-WorldModel module, recording its calls."""

    def __init__(self, result: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result if result is not None else {
            "predicted_visual": "visual",
            "predicted_proprio": "proprio",
        }

    def rollout(self, info: Any, action_sequence: Any) -> dict[str, Any]:
        self.calls.append({"info": info, "actions": action_sequence})
        return self._result


def test_adapters_module_still_matches_its_frozen_pins() -> None:
    """Adding a family must not invalidate a published result's provenance.

    The speed and door releases both record this file's sha256. A prejepa
    adapter written directly into it would have broken both.

    Only the live component releases are checked. The suite-level manifests
    (``contextworld_icl_suite_v1``, ``_v2``, ``_v2_recovery_v2``) froze on
    2026-08-14 as historical commit markers and are *expected* to carry older
    hashes -- see ``test_suite_export_manifest_selection.py``.
    """

    live = hashlib.sha256(ADAPTERS.read_bytes()).hexdigest()
    historical = {
        "contextworld_icl_suite_v1.yaml",
        "contextworld_icl_suite_v2.yaml",
        "contextworld_icl_suite_v2_recovery_v2.yaml",
    }
    pinned: dict[str, str] = {}
    for config in sorted((ROOT / "configs/benchmark").glob("*.yaml")):
        if config.name in historical:
            continue
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "source_sha256" and isinstance(value, dict):
                        for path, digest in value.items():
                            if str(path).endswith("benchmarks/adapters.py"):
                                pinned[config.name] = digest
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(payload)

    assert pinned, "no live config pins adapters.py; this guard would be vacuous"
    assert {name: live for name in pinned} == pinned


class TestRolloutShim:
    def test_it_exposes_the_predicted_emb_key_the_base_adapter_reads(
        self,
    ) -> None:
        """The base adapter indexes ``predicted_emb``; prejepa has no such key."""

        shim = _PreJEPARolloutShim(_FakePreJEPA())

        result = shim.rollout({"pixels": "p"}, "actions")

        assert result["predicted_emb"] == "visual"

    def test_it_selects_the_visual_stream(self) -> None:
        """Targets from ``encode_pixels`` are visual-only.

        Scoring the concatenated action or proprio slots would compare a
        model's own action encoding against itself, which is not the
        capability the benchmark measures.
        """

        shim = _PreJEPARolloutShim(_FakePreJEPA())

        result = shim.rollout({"pixels": "p"}, "actions")

        assert result["predicted_emb"] != "proprio"
        assert set(result) == {"predicted_emb"}

    def test_it_swallows_the_history_size_argument(self) -> None:
        """``prejepa.rollout`` takes no ``history_size``; the base passes one."""

        model = _FakePreJEPA()
        shim = _PreJEPARolloutShim(model)

        shim.rollout({"pixels": "p"}, "actions", history_size=3)

        assert len(model.calls) == 1

    def test_it_drops_a_cached_initial_embedding(self) -> None:
        """A stale cache would score one bundle from another's initial state.

        ``prejepa`` keys its cache on ``info['id']`` and ``info['step_idx']``.
        ContextWorld's query bundles carry neither, so a surviving cache could
        be reused across unrelated bundles.
        """

        model = _FakePreJEPA()
        model._init_cached_info = {"stale": True}
        shim = _PreJEPARolloutShim(model)

        shim.rollout({"pixels": "p"}, "actions")

        assert not hasattr(model, "_init_cached_info")

    def test_a_missing_visual_stream_is_an_error_not_a_silent_wrong_score(
        self,
    ) -> None:
        shim = _PreJEPARolloutShim(_FakePreJEPA({"predicted_proprio": "p"}))

        with pytest.raises(RuntimeError, match="predicted_visual"):
            shim.rollout({"pixels": "p"}, "actions")

    def test_it_forwards_unknown_attributes_to_the_model(self) -> None:
        """The adapter reads ``parameters()`` and the state dict through this."""

        model = _FakePreJEPA()
        model.marker = "forwarded"

        assert _PreJEPARolloutShim(model).marker == "forwarded"


class TestFamilyGeometry:
    @pytest.mark.parametrize(
        "adapter,history,action_dim",
        [
            (StableWorldModelPreJEPAAdapter, 3, 2),
            (StableWorldModelPreJEPAHistory7Adapter, 7, 2),
            (StableWorldModelPreJEPACubeGraspRuleAdapter, 3, 5),
        ],
    )
    def test_geometry_is_inherited_from_the_task_not_the_family(
        self, adapter: type, history: int, action_dim: int
    ) -> None:
        """Swapping families must not change what a task evaluates."""

        assert adapter.required_history_tokens == history
        assert adapter.raw_action_dim == action_dim
        assert adapter.model_config_name == "prejepa"

    def test_every_variant_is_concrete_and_in_the_hierarchy(self) -> None:
        from contextworld.benchmarks import prejepa_adapters

        exported = [
            getattr(prejepa_adapters, name)
            for name in prejepa_adapters.__all__
        ]

        assert len(exported) == 8
        for adapter in exported:
            assert issubclass(adapter, LatentWorldModelAdapter)
            assert not adapter.__abstractmethods__


def test_the_help_text_advertises_every_built_in_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A family the CLI accepts but never mentions is a family nobody uses."""

    from contextworld.benchmarks.external_model_cli import (
        _BUILTIN_FAMILIES,
        parse_args,
    )

    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = capsys.readouterr().out

    for family in _BUILTIN_FAMILIES:
        assert family in help_text
