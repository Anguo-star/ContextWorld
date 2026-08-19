"""An external model must be able to reach the scorers through the CLI.

The scoring boundary was always model-independent, but the command line was
not: ``--adapter`` accepted only ``lewm`` and ``pldm``.  These tests pin the
behaviour of the resolver that replaces that restriction -- in particular that
it accepts genuinely external classes, that it refuses anything which does not
satisfy the adapter contract, and that an installed third-party package cannot
take over the names of the frozen reference baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from contextworld.benchmarks.adapter_registry import (
    ENTRY_POINT_GROUP,
    AdapterConstructionError,
    AdapterRequest,
    AdapterResolutionError,
    build_adapter,
    resolve_adapter_class,
)
from contextworld.benchmarks.adapters import (
    AdapterProtocol,
    LatentWorldModelAdapter,
    StableWorldModelLeWMAdapter,
    StableWorldModelPLDMAdapter,
)


BUILTINS = {
    "lewm": StableWorldModelLeWMAdapter,
    "pldm": StableWorldModelPLDMAdapter,
}


class _ConcreteAdapter(LatentWorldModelAdapter):
    """Implements the five-member contract and nothing else.

    Deliberately carries no constructor of either kind, so subclasses below
    can add exactly one and the resolver's preference order is observable.
    """

    def __init__(self, request: AdapterRequest | None = None) -> None:
        self.request = request

    @property
    def protocol(self) -> AdapterProtocol:
        return AdapterProtocol(
            history_tokens=3,
            action_block_raw_steps=1,
            action_dim=2,
            future_action_blocks=1,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {"adapter_id": "external_test_v1"}

    def encode_pixels(self, pixels: np.ndarray, *, batch_size: int) -> np.ndarray:
        return np.zeros((len(pixels), 4), dtype=np.float32)

    def rollout_latents(
        self,
        input_pixels: np.ndarray,
        raw_action_blocks: np.ndarray,
        *,
        batch_size: int,
    ) -> np.ndarray:
        return np.zeros((len(input_pixels), 1, 4), dtype=np.float32)

    def frozen_state_hash(self) -> str:
        return "0" * 64


class ExternalAdapter(_ConcreteAdapter):
    """A complete adapter that has nothing to do with Stable-WorldModel.

    It exists to prove the boundary is real: no torch, no checkpoint format,
    no upstream repository pin, and it still resolves and constructs.
    """

    @classmethod
    def from_contextworld_request(
        cls, request: AdapterRequest
    ) -> "ExternalAdapter":
        return cls(request)


class IncompleteAdapter(LatentWorldModelAdapter):
    """Subclasses the contract but never implements it."""


class NotAnAdapter:
    """Right shape of name, wrong contract."""


class _RecordingAdapter(_ConcreteAdapter):
    """Captures the keyword shape a sealed built-in constructor receives.

    It stands in for the Stable-WorldModel adapters, which cannot be modified
    and therefore must be reachable through ``from_checkpoint`` alone.
    """

    captured: dict[str, Any] = {}

    @classmethod
    def from_checkpoint(cls, checkpoint: Path, **keywords: Any):
        cls.captured = {"checkpoint": checkpoint, **keywords}
        return cls()


def _request(**overrides: Any) -> AdapterRequest:
    base: dict[str, Any] = {
        "task": "speed",
        "checkpoint": Path("/tmp/model.ckpt"),
        "device": "cpu",
        "repo_root": Path("/tmp/repo"),
        "runtime": {"stablewm_repo": "repo", "stablewm_ref": "abc123"},
    }
    base.update(overrides)
    return AdapterRequest(**base)


class TestBuiltinResolution:
    def test_builtin_names_resolve(self) -> None:
        assert resolve_adapter_class("lewm", builtins=BUILTINS) is (
            StableWorldModelLeWMAdapter
        )
        assert resolve_adapter_class("pldm", builtins=BUILTINS) is (
            StableWorldModelPLDMAdapter
        )

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert resolve_adapter_class("  lewm  ", builtins=BUILTINS) is (
            StableWorldModelLeWMAdapter
        )

    def test_an_installed_package_cannot_hijack_a_baseline_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``lewm`` names a frozen baseline behind published scoreboard rows.

        If an installed distribution could rebind it, a published number would
        silently start meaning a different model.  Built-ins win.
        """

        class _Hijack:
            name = "lewm"

            def load(self) -> type:
                return ExternalAdapter

        monkeypatch.setattr(
            "contextworld.benchmarks.adapter_registry._entry_points",
            lambda: {"lewm": _Hijack()},
        )
        assert resolve_adapter_class("lewm", builtins=BUILTINS) is (
            StableWorldModelLeWMAdapter
        )


class TestImportPathResolution:
    def test_an_external_class_resolves_by_import_path(self) -> None:
        spec = f"{__name__}:ExternalAdapter"
        assert resolve_adapter_class(spec, builtins=BUILTINS) is ExternalAdapter

    def test_unimportable_module_is_reported(self) -> None:
        with pytest.raises(AdapterResolutionError, match="could not import"):
            resolve_adapter_class("no_such_module_xyz:Thing", builtins=BUILTINS)

    def test_missing_attribute_is_reported(self) -> None:
        with pytest.raises(AdapterResolutionError, match="no attribute"):
            resolve_adapter_class(f"{__name__}:Nonexistent", builtins=BUILTINS)

    def test_half_written_import_path_is_reported(self) -> None:
        with pytest.raises(AdapterResolutionError, match="not a usable import path"):
            resolve_adapter_class(f"{__name__}:", builtins=BUILTINS)


class TestEntryPointResolution:
    def test_registered_entry_point_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Entry:
            name = "my-model"

            def load(self) -> type:
                return ExternalAdapter

        monkeypatch.setattr(
            "contextworld.benchmarks.adapter_registry._entry_points",
            lambda: {"my-model": _Entry()},
        )
        assert resolve_adapter_class("my-model", builtins=BUILTINS) is (
            ExternalAdapter
        )

    def test_failing_entry_point_names_the_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Broken:
            name = "broken"

            def load(self) -> type:
                raise RuntimeError("dependency missing")

        monkeypatch.setattr(
            "contextworld.benchmarks.adapter_registry._entry_points",
            lambda: {"broken": _Broken()},
        )
        with pytest.raises(AdapterResolutionError, match=ENTRY_POINT_GROUP):
            resolve_adapter_class("broken", builtins=BUILTINS)


class TestRejection:
    def test_unknown_name_lists_what_is_available(self) -> None:
        with pytest.raises(AdapterResolutionError) as excinfo:
            resolve_adapter_class("lwem", builtins=BUILTINS)
        message = str(excinfo.value)
        assert "lewm" in message and "pldm" in message
        assert "import path" in message

    def test_empty_specification_is_rejected(self) -> None:
        with pytest.raises(AdapterResolutionError, match="no adapter"):
            resolve_adapter_class("   ", builtins=BUILTINS)

    def test_non_class_is_rejected(self) -> None:
        with pytest.raises(AdapterResolutionError, match="not a\n?\\s*class"):
            resolve_adapter_class(
                "contextworld.benchmarks.adapter_registry:ENTRY_POINT_GROUP",
                builtins=BUILTINS,
            )

    def test_class_outside_the_contract_is_rejected(self) -> None:
        with pytest.raises(AdapterResolutionError, match="does not subclass"):
            resolve_adapter_class(f"{__name__}:NotAnAdapter", builtins=BUILTINS)

    def test_still_abstract_class_is_rejected_with_missing_members(self) -> None:
        with pytest.raises(AdapterResolutionError) as excinfo:
            resolve_adapter_class(f"{__name__}:IncompleteAdapter", builtins=BUILTINS)
        message = str(excinfo.value)
        assert "still abstract" in message
        assert "encode_pixels" in message and "rollout_latents" in message


class TestAdapterRequest:
    def test_both_action_shapes_at_once_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="one or the other"):
            _request(
                action_normalizer=Path("/tmp/norm.json"),
                action_mean=[0.0, 0.0],
                action_std=[1.0, 1.0],
            )

    def test_mean_without_std_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="together"):
            _request(action_mean=[0.0, 0.0])

    def test_either_shape_alone_is_accepted(self) -> None:
        assert _request(action_normalizer=Path("/tmp/n.json")).action_std is None
        assert _request(action_mean=[0.0], action_std=[1.0]).action_normalizer is None


class TestConstruction:
    def test_external_contract_is_preferred(self) -> None:
        request = _request()
        adapter = build_adapter(
            f"{__name__}:ExternalAdapter", builtins=BUILTINS, request=request
        )
        assert isinstance(adapter, ExternalAdapter)
        assert adapter.request is request

    def test_external_contract_wins_when_a_class_offers_both(self) -> None:
        """A class carrying both constructors must not be built the old way."""

        class _Both(_ConcreteAdapter):
            used: list[str] = []

            @classmethod
            def from_contextworld_request(cls, request: AdapterRequest):
                cls.used.append("request")
                return cls(request)

            @classmethod
            def from_checkpoint(cls, checkpoint: Path, **keywords: Any):
                cls.used.append("checkpoint")
                return cls()

        build_adapter(
            "both",
            builtins={"both": _Both},
            request=_request(action_mean=[0.0], action_std=[1.0]),
        )
        assert _Both.used == ["request"]

    def test_normalizer_shape_reaches_from_checkpoint(self) -> None:
        _RecordingAdapter.captured = {}
        request = _request(action_normalizer=Path("/tmp/norm.json"))
        build_adapter(
            f"{__name__}:_RecordingAdapter",
            builtins={"rec": _RecordingAdapter},
            request=request,
        )
        captured = _RecordingAdapter.captured
        assert captured["normalizer"] == Path("/tmp/norm.json")
        assert captured["stablewm_repo"] == "repo"
        assert captured["stablewm_ref"] == "abc123"
        assert "action_mean" not in captured

    def test_action_statistics_shape_reaches_from_checkpoint(self) -> None:
        _RecordingAdapter.captured = {}
        request = _request(action_mean=[0.0, 1.0], action_std=[2.0, 3.0])
        build_adapter(
            "rec", builtins={"rec": _RecordingAdapter}, request=request
        )
        captured = _RecordingAdapter.captured
        assert captured["action_mean"] == [0.0, 1.0]
        assert captured["action_std"] == [2.0, 3.0]
        assert "normalizer" not in captured

    def test_builtin_shape_without_action_information_is_reported(self) -> None:
        with pytest.raises(AdapterConstructionError, match="neither"):
            build_adapter(
                "rec", builtins={"rec": _RecordingAdapter}, request=_request()
            )

    def test_class_with_no_usable_constructor_is_reported(self) -> None:
        with pytest.raises(AdapterConstructionError, match="neither"):
            build_adapter(
                "nc", builtins={"nc": _ConcreteAdapter}, request=_request()
            )

    def test_constructor_returning_a_foreign_object_is_reported(self) -> None:
        class _WrongReturn(ExternalAdapter):
            @classmethod
            def from_contextworld_request(cls, request: AdapterRequest):
                return "not an adapter"

        with pytest.raises(AdapterConstructionError, match="not a LatentWorldModel"):
            build_adapter(
                "wr", builtins={"wr": _WrongReturn}, request=_request()
            )


def test_the_registry_does_not_require_a_deep_learning_stack() -> None:
    """Resolution must work in a light install, or CI cannot exercise it.

    Checked in a fresh interpreter rather than against ``sys.modules``: by the
    time this module runs, a sibling test has usually imported torch already,
    so an in-process assertion would pass or fail on test ordering rather than
    on the property being claimed.
    """

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import contextworld.benchmarks.adapter_registry as registry\n"
            "heavy = [m for m in ('torch', 'lancedb', 'stable_worldmodel')\n"
            "         if m in sys.modules]\n"
            "assert not heavy, heavy\n"
            "print(registry.ENTRY_POINT_GROUP)\n",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "contextworld.adapters"
