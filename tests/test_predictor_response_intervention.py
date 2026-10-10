"""Unit checks for the grouped response intervention contract."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location('intervention', SCRIPT / 'run_predictor_response_intervention.py')
intervention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(intervention)


def test_response_is_centered_within_each_query():
    target = torch.zeros((2, 2, 3, 1))
    prediction = target.clone()
    prediction[0, :, -1, 0] = torch.tensor([1., -1.])
    prediction[1, :, -1, 0] = 10.
    native, rest, response = intervention.query_losses(prediction, target)
    assert response.item() == pytest.approx(1 / 6)
    assert native.item() == pytest.approx((2 + 200) / 12)
    assert (rest + response).item() == pytest.approx(native.item())
    assert intervention.loss_for_arm(prediction, target, 1, 1)[0].item() == pytest.approx(native.item())


def test_groups_are_complete_and_keep_original_query_order():
    data = dict(query_ids=np.array(['b','a','b','a']), labels=np.array([0,0,1,1]),
                rawactions=np.zeros((4,3,1,2)))
    ids, groups = intervention.group_indices(data)
    assert ids == ['b','a']
    assert [g.tolist() for g in groups] == [[0,2],[1,3]]
    data['labels'][3] = 0
    with pytest.raises(ValueError, match='incomplete'):
        intervention.group_indices(data)


def test_zero_lr_arms_identical_and_only_predictor_updates():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1,1,bias=False)
            self.predictor = torch.nn.Linear(1,1,bias=False)
        def forward(self,x):
            return self.predictor(self.encoder(x))
    model = Tiny()
    before = intervention.frozen_hash(model)
    initial = {k:v.clone() for k,v in model.predictor.state_dict().items()}
    x = torch.tensor([[[[0.]],[[1.]]],[[[2.]],[[3.]]]])
    y = torch.zeros_like(x)
    endpoints = []
    for weights in [(1.,1.),(0.5,2.)]:
        model.predictor.load_state_dict(initial)
        for p in model.parameters(): p.requires_grad_(False)
        model.predictor.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.predictor.parameters(),lr=0.)
        optimizer.zero_grad()
        intervention.loss_for_arm(model(x),y,*weights)[0].backward()
        optimizer.step()
        endpoints.append({k:v.clone() for k,v in model.predictor.state_dict().items()})
        assert intervention.frozen_hash(model) == before
        assert model.encoder.weight.grad is None
    assert all(torch.equal(endpoints[0][k],endpoints[1][k]) for k in initial)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in model.predictor.parameters()}


def test_nonzero_update_keeps_encoder_fixed():
    encoder = torch.nn.Linear(1, 1, bias=False)
    predictor = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        encoder.weight.fill_(2.)
        predictor.weight.fill_(1.)
    for p in encoder.parameters(): p.requires_grad_(False)
    frozen = encoder.weight.detach().clone()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    x = torch.ones((2, 2, 3, 1))
    target = torch.zeros_like(x)
    loss, _ = intervention.loss_for_arm(predictor(encoder(x)), target, 1., 2.)
    loss.backward()
    optimizer.step()
    assert torch.equal(encoder.weight, frozen)
    assert predictor.weight.item() != 1.


def test_calibration_backprops_both_components(monkeypatch):
    class Adapter:
        pass
    adapter = Adapter()
    adapter.model = torch.nn.Module()
    adapter.model.predictor = torch.nn.Linear(1,1,bias=False)
    adapter.model.pred_proj = torch.nn.Identity()
    with torch.no_grad(): adapter.model.predictor.weight.fill_(2.)
    data = {'source_groups':np.array([f'g{i}' for i in range(16)]),
            'history':np.zeros((16,3,1)), 'queryfuture':np.zeros((16,1))}
    groups = [np.array([i]) for i in range(16)]
    ids = [f'q{i}' for i in range(16)]
    def predict(_, __, ___, rows):
        x = torch.tensor([[[1.],[2.]],[[2.],[3.]]])
        return adapter.model.predictor(x), torch.zeros_like(x)
    monkeypatch.setattr(intervention,'predict_query',predict)
    result = intervention.calibrate(adapter,'lewm',data,groups,ids)
    assert result['rest_gradient_norm'] > 0
    assert result['response_gradient_norm'] > 0
    assert result['lambda_response'] >= 1
    assert len(result['source_groups']) == 16
