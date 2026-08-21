"""PreJEPA uses its native rollout without widening the frozen ICL inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from contextworld.benchmarks.adapter_registry import AdapterRequest
from contextworld.benchmarks.adapters import LatentWorldModelAdapter
from contextworld.benchmarks.prejepa_adapters import (
    PreJEPAInputContractError,
    StableWorldModelPreJEPAAdapter,
    StableWorldModelPreJEPACubeGraspRuleAdapter,
    StableWorldModelPreJEPAHistory7Adapter,
)
from contextworld.evaluation.protocol import ColumnStandardizer


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "contextworld/benchmarks/adapters.py"


def test_adapters_module_still_matches_its_frozen_pins() -> None:
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


class _FakeEmbedder:
    def __init__(self, in_chans: int, emb_dim: int = 10) -> None:
        self.in_chans = in_chans
        self.emb_dim = emb_dim


def _fake_model(*, history: int = 3, action_width: int = 10, state=False):
    torch = pytest.importorskip("torch")

    class FakePreJEPA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.ones(()))
            self.history_size = history
            self.extra_encoders = {
                "action": _FakeEmbedder(action_width),
            }
            if state:
                self.extra_encoders = {
                    "proprio": _FakeEmbedder(2),
                    **self.extra_encoders,
                }
            self.encode_calls: list[list[str] | None] = []
            self.rollout_calls: list[tuple[dict[str, Any], Any]] = []

        def encode(self, info, *, emb_keys=None, **_):
            self.encode_calls.append(emb_keys)
            batch, frames = info["pixels"].shape[:2]
            pixels_emb = torch.ones(batch, frames, 2, 3)
            return {**info, "pixels_emb": pixels_emb, "emb": pixels_emb}

        def rollout(self, info, future_actions):
            self.rollout_calls.append((info, future_actions))
            batch, samples, frames = info["pixels"].shape[:3]
            future = future_actions.shape[2]
            predicted = torch.arange(
                batch * samples * (frames + future) * 2 * 3,
                dtype=torch.float32,
            ).reshape(batch, samples, frames + future, 2, 3)
            return {"predicted_pixels_emb": predicted}

    return FakePreJEPA()


def _adapter(model, *, cls=StableWorldModelPreJEPAAdapter):
    return cls(
        model=model,
        checkpoint=Path(__file__),
        stable_repo=ROOT,
        stable_commit="a" * 40,
        action_standardizer=ColumnStandardizer(
            np.zeros((1, cls.raw_action_dim), dtype=np.float32),
            np.ones((1, cls.raw_action_dim), dtype=np.float32),
        ),
        device="cpu",
    )


def test_native_visual_encode_excludes_action_encoder() -> None:
    model = _fake_model()
    adapter = _adapter(model)

    encoded = adapter.encode_pixels(
        np.zeros((2, 8, 8, 3), dtype=np.uint8), batch_size=2
    )

    assert encoded.shape == (2, 2, 3)
    assert model.encode_calls == [[]]


def test_native_rollout_splits_past_and_strictly_future_actions() -> None:
    model = _fake_model()
    model._init_cached_info = {"stale": True}
    adapter = _adapter(model)
    pixels = np.zeros((2, 3, 8, 8, 3), dtype=np.uint8)
    # H=3 consumes two past blocks; three blocks remain as future actions.
    actions = np.zeros((2, 5, 5, 2), dtype=np.float32)

    predicted = adapter.rollout_latents(pixels, actions, batch_size=2)

    info, future = model.rollout_calls[0]
    assert info["pixels"].shape == (2, 1, 3, 3, 8, 8)
    assert info["action_history"].shape == (2, 1, 2, 10)
    assert future.shape == (2, 1, 3, 10)
    assert predicted.shape == (2, 3, 2, 3)
    assert not hasattr(model, "_init_cached_info")


def test_state_conditioned_checkpoint_is_rejected_without_fabricated_state() -> None:
    with pytest.raises(PreJEPAInputContractError, match="only pixels and actions"):
        _adapter(_fake_model(state=True))


def test_checkpoint_history_must_match_the_frozen_task() -> None:
    with pytest.raises(PreJEPAInputContractError, match="history_size=3"):
        _adapter(_fake_model(history=3), cls=StableWorldModelPreJEPAHistory7Adapter)


def test_request_constructor_uses_native_loader_not_get_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextworld.benchmarks import prejepa_adapters

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"native")
    (tmp_path / "config.json").write_text(json.dumps({}), encoding="utf-8")
    model = _fake_model()
    calls = []

    class Utils:
        @staticmethod
        def load_pretrained(name, cache_dir=None):
            calls.append((name, cache_dir))
            return model

    swm = SimpleNamespace(wm=SimpleNamespace(utils=Utils()))
    monkeypatch.setattr(
        prejepa_adapters,
        "load_stable_worldmodel",
        lambda *_: (swm, tmp_path, "b" * 40),
    )
    request = AdapterRequest(
        task="speed",
        checkpoint=checkpoint,
        device="cpu",
        repo_root=ROOT,
        action_mean=(0.0, 0.0),
        action_std=(1.0, 1.0),
        runtime={"stablewm_repo": str(tmp_path), "stablewm_ref": "b" * 40},
    )

    adapter = StableWorldModelPreJEPAAdapter.from_contextworld_request(request)

    assert isinstance(adapter, LatentWorldModelAdapter)
    assert calls and calls[0][0] == str(checkpoint)


@pytest.mark.parametrize(
    "adapter,history,action_dim",
    [
        (StableWorldModelPreJEPAAdapter, 3, 2),
        (StableWorldModelPreJEPAHistory7Adapter, 7, 2),
        (StableWorldModelPreJEPACubeGraspRuleAdapter, 3, 5),
    ],
)
def test_geometry_is_inherited_from_the_task(
    adapter: type, history: int, action_dim: int
) -> None:
    assert adapter.required_history_tokens == history
    assert adapter.raw_action_dim == action_dim
    assert adapter.model_config_name == "prejepa"


def test_the_help_text_advertises_every_built_in_family(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from contextworld.benchmarks.external_model_cli import (
        _BUILTIN_FAMILIES,
        parse_args,
    )

    with pytest.raises(SystemExit):
        parse_args(["--help"])
    help_text = capsys.readouterr().out
    for family in _BUILTIN_FAMILIES:
        assert family in help_text
