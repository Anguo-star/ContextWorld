"""Focused checks for complete-query PLDM batches and frozen state."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_pldm_fixed_representation_intervention as intervention
import run_predictor_response_intervention as base
import summarize_pldm_fixed_representation_intervention as summary


def test_grouped_batch_loss_and_gradient_equal_query_average(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = torch.nn.Linear(2, 2, bias=False)
    model = Tiny()
    class Adapter:
        pass
    adapter = Adapter()
    adapter.model = model
    def fake_prediction(_, __, history, future, actions):
        h = torch.as_tensor(history)
        y = torch.as_tensor(future)
        prediction = model.predictor(h)
        return prediction, torch.cat((h[:, 1:], y), 1), None
    monkeypatch.setattr(intervention.native, '_prediction', fake_prediction)
    rng = np.random.default_rng(4)
    cache = dict(history=rng.normal(size=(6, 3, 2)).astype('f'),
                 target=rng.normal(size=(6, 2)).astype('f'))
    raw = dict(rawactions=np.zeros((6, 3, 1, 2), 'f'))
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    together, _ = intervention.grouped_loss(adapter, cache, raw, groups, [2, 0, 1], (0.7, 2.3))
    gradient = torch.autograd.grad(together, model.predictor.weight)[0]
    individual = [intervention.grouped_loss(adapter, cache, raw, groups, [i], (0.7, 2.3))[0]
                  for i in (2, 0, 1)]
    average = torch.stack(individual).mean()
    expected = torch.autograd.grad(average, model.predictor.weight)[0]
    assert torch.allclose(together, average, atol=1e-6)
    assert torch.allclose(gradient, expected, atol=1e-6)


def test_swapped_group_maps_to_opposite_history(monkeypatch):
    seen = []
    def fake_prediction(_, __, history, future, actions):
        seen.append(history.copy())
        p = torch.from_numpy(history.copy())
        return p, p, None
    monkeypatch.setattr(intervention.native, '_prediction', fake_prediction)
    cache = dict(history=np.arange(12, dtype='f').reshape(2, 3, 2),
                 target=np.zeros((2, 2), 'f'))
    raw = dict(rawactions=np.zeros((2, 3, 1, 2), 'f'))
    adapter = type('Adapter', (), {})()
    intervention.batch_prediction(adapter, cache, raw, [np.array([0, 1])], [0], swapped=True)
    np.testing.assert_array_equal(seen[0][0], cache['history'][1])
    np.testing.assert_array_equal(seen[0][1], cache['history'][0])


def test_interleaved_query_rows_use_actual_swap_indices(monkeypatch):
    def fake_batch(_, cache, raw, groups, chosen, swapped=False):
        rows = np.concatenate([groups[i] for i in chosen])
        prediction = torch.from_numpy(cache['correct_prediction'][rows]).reshape(len(chosen), 2, 1, 1)
        return prediction, prediction
    monkeypatch.setattr(intervention, 'batch_prediction', fake_batch)
    raw = {}
    cache = dict(target=np.array([[0.], [10.], [1.], [11.]], dtype='f'),
                 correct_prediction=np.array([[0.], [10.], [1.], [11.]], dtype='f'),
                 source_groups=np.array(['a', 'b', 'a', 'b']))
    groups = [np.array([0, 2]), np.array([1, 3])]
    adapter = type('Adapter', (), {'model': torch.nn.Linear(1, 1)})()
    result = intervention.evaluate(adapter, cache, raw, ['a', 'b'], groups)
    assert result['summary']['E_over_B'] == pytest.approx(0.)
    assert result['per_query'][0]['W'] == pytest.approx(2.)
    assert result['per_query'][1]['W'] == pytest.approx(2.)


def test_frozen_state_and_restore():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.predictor = torch.nn.Linear(2, 2)
            self.pred_proj = torch.nn.Linear(2, 2)
    model = Tiny()
    frozen = base.frozen_hash(model)
    saved = base.trainable_state(model)
    for p in model.parameters(): p.requires_grad_(False)
    model.predictor.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    assert {id(p) for p in base.trainable_parameters(model)} == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(base.trainable_parameters(model), lr=1e-3)
    optimizer.zero_grad()
    model.pred_proj(model.predictor(model.encoder(torch.ones(2, 2)))).square().mean().backward()
    optimizer.step()
    assert base.frozen_hash(model) == frozen
    base.restore_trainable_state(model, saved)
    assert all(torch.equal(model.state_dict()[key], value) for key, value in saved.items())


def test_groups_reject_reordered_source():
    raw = dict(query_ids=np.array(['a', 'a']), source_groups=np.array(['g', 'g']),
               labels=np.array([0, 1]), rawactions=np.zeros((2, 3, 1, 2)),
               history_pixels=np.zeros((2, 3, 2, 2, 3)))
    cache = dict(query_ids=raw['query_ids'].copy(), source_groups=np.array(['g', 'other']))
    with pytest.raises(ValueError, match='source_groups'):
        intervention.groups_for(raw, cache)


def test_summary_pools_energy_before_dividing():
    rows = [dict(B=1., E=2., R=1., C=1., W=3.),
            dict(B=9., E=9., R=3., C=6., W=18.)]
    values = summary.pooled(rows)
    assert values['NRE'] == pytest.approx(.4)
    assert values['E_over_B'] == pytest.approx(1.1)
    assert values['G'] == pytest.approx(1.)
    assert values['NRE'] != pytest.approx((1. + 3./9.)/2.)


def test_source_paired_bootstrap_mirror_duplication_and_zero_delta():
    control = [dict(query_id=f'q{i}', source_group=f'g{i//2}', B=float(i+2),
                    E=float(i+3), R=1., C=float(i+2), W=float(i+5)) for i in range(6)]
    weighted = [dict(row, E=row['E']+.5, R=row['R']+.5, W=row['W']+1.) for row in control]
    original = summary.paired_delta(control, weighted)
    mirror_control = [dict(row, query_id=row['query_id']+'-mirror') for row in control]
    mirror_weighted = [dict(row, query_id=row['query_id']+'-mirror') for row in weighted]
    doubled = summary.paired_delta(control + mirror_control, weighted + mirror_weighted)
    assert doubled['source_groups'] == original['source_groups'] == 3
    for key in ('NRE', 'E_over_B', 'G'):
        a = original['weighted_minus_control'][key]
        b = doubled['weighted_minus_control'][key]
        assert b['estimate'] == pytest.approx(a['estimate'])
        assert b['ci95'] == pytest.approx(a['ci95'])
    zero = summary.paired_delta(control, [dict(row) for row in control])
    for effect in zero['weighted_minus_control'].values():
        assert effect['estimate'] == 0.
        assert effect['ci95'] == [0., 0.]


def test_paired_summary_rejects_query_identity_mismatch():
    row = dict(query_id='a', source_group='g', B=1., E=1., R=1., C=0., W=2.)
    with pytest.raises(AssertionError):
        summary.paired_delta([row], [dict(row, query_id='other')])
