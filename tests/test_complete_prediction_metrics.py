"""Known counterexamples distinguish full prediction from response alone."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from complete_prediction_metrics import array_energies,pair_record_energies,aggregate,summarize,paired_contrast,energy_growth


def test_correct_response_with_common_bias_is_not_accurate_prediction():
    z=np.array([[-1.,0.],[1.,0.]])
    p=z+np.array([0.,10.])
    r=aggregate([array_energies(p,z)])
    assert r['response_error']==0
    assert r['complete_error']==r['common_bias_error']==100
    assert r['history_error_reduction']==4  # common bias cancels; never add it.


@pytest.mark.parametrize('k',[2,3,4])
def test_condition_blind_bound_and_all_pair_response_identity(k):
    rng=np.random.default_rng(4)
    z=rng.normal(size=(k,7,11));p=rng.normal(size=z.shape)
    r=aggregate([array_energies(p,z)])
    pair_error=sum(np.square((p[j]-p[i])-(z[j]-z[i])).sum() for i in range(k) for j in range(i+1,k))
    pair_energy=sum(np.square(z[j]-z[i]).sum() for i in range(k) for j in range(i+1,k))
    assert r['response_error']==pytest.approx(pair_error/pair_energy)
    common=np.broadcast_to(z.mean(axis=0),z.shape)
    assert aggregate([array_energies(common,z)])['complete_error']==pytest.approx(1)
    assert aggregate([array_energies(common+3,z)])['complete_error']>1
    gain=((p-p.mean(axis=0))*(z-z.mean(axis=0))).sum()/np.square(z-z.mean(axis=0)).sum()
    assert r['history_error_reduction']==pytest.approx(2*k/(k-1)*gain)


def test_pair_record_reconstruction_matches_full_arrays():
    rng=np.random.default_rng(3);p=rng.normal(size=(2,8));z=rng.normal(size=(2,8))
    dz=z[1]-z[0];dp=p[1]-p[0]
    record={name:dict(correct_future_mse=np.square(p[i]-z[i]).mean(),other_future_mse=np.square(p[i]-z[1-i]).mean()) for i,name in enumerate(('low_strength','high_strength'))}
    record['latent_response']=dict(target_response_mse=np.square(dz).mean(),normalized_response_error=np.square(dp-dz).sum()/np.square(dz).sum(),response_gain=(dp*dz).sum()/np.square(dz).sum())
    assert aggregate([pair_record_energies(record)])==pytest.approx(aggregate([array_energies(p,z)]))


def test_zero_separation_actions_still_contribute_prediction_error():
    z=np.array([[[-1.],[0.]],[[1.],[0.]]]);p=z.copy();p[:,1]=10
    r=array_energies(p,z)
    assert r['zero_separation_comparisons']==1
    assert aggregate([r])['complete_error']==100
    with pytest.raises(ValueError,match='no target separation'):
        aggregate([array_energies(np.ones((2,1)),np.zeros((2,1)))])


def test_cluster_bootstrap_and_paired_identity():
    records=[dict(query_id=str(i),**array_energies(np.array([[-.5],[.5]]),np.array([[-1.],[1.]]))) for i in range(6)]
    r=summarize(records)
    assert r['complete_error']==.25
    assert r['ci95']['complete_error']==[.25,.25]
    diff=paired_contrast(records,records)
    assert diff['complete_error']==dict(after_minus_before=0.,ci95=[0.,0.])
    with pytest.raises(ValueError,match='align'):
        paired_contrast(records,records[::-1])


def test_energy_growth_does_not_hide_behind_changing_reference():
    z=np.array([[-1.],[1.]])
    before=[dict(query_id=str(i),**array_energies(z+1,z)) for i in range(6)]
    after=[dict(query_id=str(i),**array_energies(3*z+2,3*z)) for i in range(6)]
    # Raw squared error grows fourfold, yet normalized error falls (4/9).
    assert aggregate(after)['complete_error'] < aggregate(before)['complete_error']
    r=energy_growth(before,after)
    assert r['complete_energy']==dict(after_over_before=4.,ci95=[4.,4.])
    assert r['baseline_energy']==dict(after_over_before=9.,ci95=[9.,9.])


def test_published_energy_sums_and_existing_response_scores_agree():
    import json
    root=Path(__file__).resolve().parents[1]/'docs/research/data'
    data=json.loads((root/'icl_complete_prediction_v1.json').read_text())
    old=json.loads((root/'speed_planning_horizon_probe_v1.json').read_text())
    for row in data['rows']:
        fields=['dataset','model','scheme','raw_steps']+[f for f in ('track','arm') if f in row]
        records=[r for r in data['per_scene'] if all(r.get(f)==row[f] for f in fields)]
        result=aggregate(records)
        assert result==pytest.approx({k:row[k] for k in result})
        assert row['scenes']==len(records)
        if row['dataset']=='speed_horizon':
            prior=next(r for r in old['rows'] if all(r[f]==row[f] for f in ('scheme','arm','raw_steps')))
            assert row['response_error']==pytest.approx(1-prior['response_score']/100,abs=1e-9)
    assert len(data['rows'])==58 and data['no_training']
    assert data['new_model_prediction_conditions']==512
