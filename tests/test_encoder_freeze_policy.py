"""Focused checks for full-model warmstart plus encoder-only freezing."""

from types import SimpleNamespace

import pytest
import torch

from contextworld.training.encoder_policy import (
    attach_encoder_freeze,
    freeze_encoder,
    validate_encoder_freeze_policy,
)
from contextworld.training.initialization import apply_initialization, validate_initialization
from scripts import run_stablewm_train as launcher


class FakeWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.5)
        )
        self.projector = torch.nn.Linear(4, 4)
        self.predictor = torch.nn.Linear(4, 4)
        self.action_encoder = torch.nn.Linear(2, 4)
        self.pred_proj = torch.nn.Linear(4, 4)


class FakePreJEPA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4)
        )
        self.predictor = torch.nn.Linear(4, 4)
        self.extra_encoders = torch.nn.ModuleDict({"action": torch.nn.Linear(2, 4)})


def _checkpoint(tmp_path, model):
    path = tmp_path / "original.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)
    return path


def _args(**changes):
    values = dict(
        training_track="encoder_freeze_v1", init_checkpoint=None,
        freeze_encoder=False,
        family="lewm", component="action_strength", original_env=None,
        method="native", rollout_steps=1, dataset=None, component_payload=None,
        mix_original_weight=None, mix_synthetic_weight=None,
        component_epoch_size=None, run_name="run", resume="auto",
        eval_icl_only=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_lewm_freezes_encoder_only_and_callback_preserves_projector_training(tmp_path):
    source = FakeWM()
    path = _checkpoint(tmp_path, source)
    target = FakeWM()
    apply_initialization(target, validate_initialization(str(path)))
    for name in ("encoder", "projector", "predictor", "action_encoder", "pred_proj"):
        for key, value in getattr(source, name).state_dict().items():
            assert torch.equal(getattr(target, name).state_dict()[key], value)
    receipt = freeze_encoder(target, family="lewm")
    assert receipt["modules"] == ["encoder"]
    assert not any(p.requires_grad for p in target.encoder.parameters())
    assert all(p.requires_grad for p in target.projector.parameters())
    assert all(p.requires_grad for p in target.predictor.parameters())
    assert validate_encoder_freeze_policy(family="lewm", checkpoint=str(path))["semantics"] == (
        "full_model_warmstart_encoder_only_frozen"
    )
    trainer = SimpleNamespace(callbacks=[])
    callback = attach_encoder_freeze(trainer, target, family="lewm")
    target.train()
    assert target.encoder.training
    callback.on_train_batch_start(None, None, None, 0)
    assert not target.encoder.training
    assert target.projector.training


def test_prejepa_freezes_backbone_only():
    model = FakePreJEPA()
    record = freeze_encoder(model, family="prejepa")
    assert record["modules"] == ["backbone"]
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.predictor.parameters())
    assert all(p.requires_grad for p in model.extra_encoders.parameters())


def test_launcher_encoder_freeze_requires_full_model_checkpoint(tmp_path):
    source = FakeWM()
    path = _checkpoint(tmp_path, source)
    request = _args(freeze_encoder=True, init_checkpoint=str(path))
    launcher._validate_training_track(request)
    assert request.initialization_identity["semantics"] == "model_weights_only_fresh_optimizer"
    assert request.encoder_freeze_identity["modules"] == ["encoder"]

    with pytest.raises(SystemExit, match="requires a full-model"):
        launcher._validate_training_track(_args(freeze_encoder=True))
