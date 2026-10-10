"""Check diagnostic identities against known signals and exact derivatives."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/diagnose_delay_native_objective.py'
spec = importlib.util.spec_from_file_location('delay_native_objective', SOURCE)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


@pytest.mark.parametrize('patch_axes', [(), (2,)])
def test_weak_response_with_common_bias_has_known_errors(patch_axes):
    shape = (3, 7, *patch_axes, 4)
    condition = torch.tensor([-1., 0., 1.]).reshape(3, *([1] * (len(shape) - 1)))
    target = condition.expand(shape)
    pred = 0.001 * target + 4
    d = diag._loss_decomposition(pred, target)
    response = torch.full((7,), (0.999 ** 2) * 2 / 3, dtype=torch.float64)
    common = torch.full((7,), 16., dtype=torch.float64)
    assert torch.allclose(d['response_by_position'], response, atol=1e-6)
    assert torch.allclose(d['common_by_position'], common, atol=1e-6)
    assert torch.allclose(d['native_by_position'], response + common, atol=1e-6)


def test_component_gradients_sum_to_exact_mse_derivative():
    target = torch.tensor([-1., 0., 1.])[:, None, None] * torch.arange(1., 8.)[None, :, None]
    pred = (0.001 * target + 4).detach().requires_grad_(True)
    d = diag._loss_decomposition(pred, target)
    components = [d['response_by_position'][-1] / 7,
                  d['common_by_position'][-1] / 7,
                  d['native_by_position'][:-1].sum() / 7]
    parts = [torch.autograd.grad(x, pred, retain_graph=True)[0] for x in components]
    actual = torch.autograd.grad(d['native'], pred)[0]
    expected = 2 * (pred.detach() - target) / pred.numel()
    assert torch.allclose(actual, expected, atol=1e-7)
    assert torch.allclose(sum(parts), expected, atol=1e-7)


def test_zero_and_unused_gradients_are_not_interpreted_as_alignment():
    stats = diag._pairwise_gradient_summary({'zero': [torch.zeros(1), None],
                                            'one': [torch.ones(1), None]})
    assert stats['pairs']['zero__one']['cosine'] is None
    assert diag._array_norm([None, torch.tensor([3., 4.])]) == 5
    a = [torch.tensor([1000.])]
    b = [torch.tensor([-999.99])]
    reference = [a[0] + b[0] + 0.00001]
    closure = diag._gradient_closure(reference, [a, b])
    assert closure['relative_l2'] > closure['relative_to_component_norm_sum']
