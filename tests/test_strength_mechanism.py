"""Paired uncertainty must preserve repeated source groups and condition balance."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('strength_summary',Path(__file__).resolve().parents[1]/'scripts/analyze_strength_mechanism.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def rows(correct):
    # Group A has two pairs; group B has one. Pair-level resampling is wrong.
    return [dict(pair_id=f'p{i//2}',label=i%2,source_group=('A' if i<4 else 'B'),
                 correct=bool(ok),prediction=(i%2 if ok else 1-i%2)) for i,ok in enumerate(correct)]


def test_identical_paired_readouts_have_exactly_zero_uncertainty():
    a=rows([1,0,1,1,0,1])
    result=module.paired_difference(a,list(reversed(a)))
    assert result['difference_pp']==0
    assert result['ci95_pp']==[0,0]
    assert result['source_groups']==2


def test_source_bootstrap_retains_all_rows_and_paired_labels():
    a=rows([0]*6);b=rows([1,1,1,1,0,0])
    result=module.paired_difference(a,b)
    assert result['difference_pp']==pytest.approx(100*4/6)
    # AA, AB, BA, BB have differences 100, 66.67, 66.67, 0.
    assert result['ci95_pp']==[0,100]
    assert module.paired_difference(b,a)['difference_pp']==pytest.approx(-100*4/6)


def test_uniform_paired_improvement_has_zero_noise():
    result=module.paired_difference(rows([0]*6),rows([1]*6))
    assert result['difference_pp']==100
    assert result['ci95_pp']==[100,100]


def test_pairing_rejects_changed_source_and_missing_condition():
    a=rows([1]*6);b=rows([1]*6);b[0]['source_group']='B'
    with pytest.raises(ValueError,match='source group differs'):
        module.paired_difference(a,b)
    with pytest.raises(ValueError,match='identities differ'):
        module.paired_difference(a,a[:-1])


def test_h3_binary_loss_components_match_exact_derivative():
    torch=pytest.importorskip('torch')
    source=Path(__file__).resolve().parents[1]/'scripts/diagnose_delay_native_objective.py'
    s=importlib.util.spec_from_file_location('strength_loss_helper',source)
    diag=importlib.util.module_from_spec(s);s.loader.exec_module(diag)
    target=torch.tensor([-1.,1.])[:,None,None]*torch.tensor([1.,2.,3.])[None,:,None]
    prediction=(.2*target+4).detach().requires_grad_(True)
    d=diag._loss_decomposition(prediction,target)
    components=[d['response_by_position'][-1]/3,d['common_by_position'][-1]/3,
                d['native_by_position'][:2].sum()/3]
    assert d['response_by_position'][-1].item()==pytest.approx(.8**2*9)
    assert d['common_by_position'][-1].item()==pytest.approx(16)
    gradients=[torch.autograd.grad(v,prediction,retain_graph=True)[0] for v in components]
    actual=torch.autograd.grad(d['native'],prediction)[0]
    expected=2*(prediction.detach()-target)/prediction.numel()
    assert torch.allclose(actual,expected,atol=1e-7)
    assert torch.allclose(sum(gradients),expected,atol=1e-7)
