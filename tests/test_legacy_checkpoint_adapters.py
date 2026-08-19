from __future__ import annotations

from pathlib import Path

import hydra.utils
from omegaconf import OmegaConf
import pytest
import torch

import contextworld.benchmarks.adapters as adapters
from contextworld.benchmarks.adapters import (
    StableWorldModelLeWMActionStrengthAdapter,
    StableWorldModelLeWMAdapter,
    StableWorldModelLeWMCubeGraspRuleAdapter,
    StableWorldModelLeWMContactFrictionAdapter,
    StableWorldModelLeWMHistory7Adapter,
    StableWorldModelLeWMMotionDampingAdapter,
    StableWorldModelLeWMPortalExitAdapter,
    StableWorldModelLeWMReacherArmMassAdapter,
    StableWorldModelPLDMActionStrengthAdapter,
    StableWorldModelPLDMAdapter,
    StableWorldModelPLDMCubeGraspRuleAdapter,
    StableWorldModelPLDMContactFrictionAdapter,
    StableWorldModelPLDMHistory7Adapter,
    StableWorldModelPLDMMotionDampingAdapter,
    StableWorldModelPLDMPortalExitAdapter,
    StableWorldModelPLDMReacherArmMassAdapter,
)


class _StrictModel:
    def __init__(self) -> None:
        self.loaded_state: dict[str, torch.Tensor] | None = None
        self.strict: bool | None = None

    def load_state_dict(
        self, state: dict[str, torch.Tensor], *, strict: bool
    ) -> None:
        self.loaded_state = state
        self.strict = strict


@pytest.mark.parametrize(
    ("adapter_class", "family", "action_input_dim"),
    (
        (StableWorldModelLeWMAdapter, "lewm", 10),
        (StableWorldModelPLDMAdapter, "pldm", 10),
        (StableWorldModelLeWMHistory7Adapter, "lewm", 10),
        (StableWorldModelPLDMHistory7Adapter, "pldm", 10),
        (StableWorldModelLeWMActionStrengthAdapter, "lewm", 10),
        (StableWorldModelPLDMActionStrengthAdapter, "pldm", 10),
        (StableWorldModelLeWMContactFrictionAdapter, "lewm", 10),
        (StableWorldModelPLDMContactFrictionAdapter, "pldm", 10),
        (StableWorldModelLeWMMotionDampingAdapter, "lewm", 10),
        (StableWorldModelPLDMMotionDampingAdapter, "pldm", 10),
        (StableWorldModelLeWMPortalExitAdapter, "lewm", 10),
        (StableWorldModelPLDMPortalExitAdapter, "pldm", 10),
        (StableWorldModelLeWMReacherArmMassAdapter, "lewm", 10),
        (StableWorldModelPLDMReacherArmMassAdapter, "pldm", 10),
        (StableWorldModelLeWMCubeGraspRuleAdapter, "lewm", 25),
        (StableWorldModelPLDMCubeGraspRuleAdapter, "pldm", 25),
    ),
)
def test_legacy_checkpoint_routes_by_adapter_family_and_loads_strictly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    adapter_class: type[StableWorldModelLeWMAdapter],
    family: str,
    action_input_dim: int,
) -> None:
    checkpoint = tmp_path / "original-baseline.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.tensor([1.0]),
                "trainer.global_step": torch.tensor(9),
            }
        },
        checkpoint,
    )
    config = OmegaConf.create(
        {"model": {"action_encoder": {"input_dim": "???"}}}
    )
    seen: dict[str, object] = {}
    model = _StrictModel()

    def load_config(path: Path):
        seen["config_path"] = path
        return config

    monkeypatch.setattr(OmegaConf, "load", load_config)
    monkeypatch.setattr(hydra.utils, "instantiate", lambda _: model)

    loaded = adapters._load_model(
        checkpoint,
        stable_worldmodel=object(),
        stable_repo=tmp_path / "stable-worldmodel",
        repo_root=tmp_path,
        model_config_name=adapter_class.model_config_name,
        action_input_dim=adapter_class.action_input_dim,
    )

    assert loaded is model
    assert seen["config_path"] == (
        tmp_path / "stable-worldmodel/scripts/train/config" / f"{family}.yaml"
    )
    assert config.model.action_encoder.input_dim == action_input_dim
    assert model.strict is True
    assert model.loaded_state is not None
    assert set(model.loaded_state) == {"weight"}
    assert torch.equal(model.loaded_state["weight"], torch.tensor([1.0]))


def test_native_pt_loading_path_keeps_existing_semantics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "native.pt"
    checkpoint.write_bytes(b"native-checkpoint")
    native_model = object()
    seen: dict[str, object] = {}

    def load_native(
        received_checkpoint: Path,
        received_stable_worldmodel: object,
        *,
        cache_dir: Path,
    ) -> object:
        seen["checkpoint"] = received_checkpoint
        seen["stable_worldmodel"] = received_stable_worldmodel
        seen["cache_dir"] = cache_dir
        return native_model

    monkeypatch.setattr(adapters, "load_pretrained_cost_model", load_native)
    stable_worldmodel = object()
    loaded = adapters._load_model(
        checkpoint,
        stable_worldmodel=stable_worldmodel,
        stable_repo=tmp_path / "unused-stable-worldmodel",
        repo_root=tmp_path,
        model_config_name="pldm",
        action_input_dim=10,
    )

    assert loaded is native_model
    assert seen == {
        "checkpoint": checkpoint.resolve(),
        "stable_worldmodel": stable_worldmodel,
        "cache_dir": adapters.artifact_path(
            "evaluation/model_cache", repo_root=tmp_path
        ),
    }
