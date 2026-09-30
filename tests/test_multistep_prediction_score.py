import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multistep_prediction_score import score,query_energies,aggregate_runs


def from_arrays(p,z):
    # [condition, candidate, horizon, feature]
    return dict(scene_id='q',horizons=[5,10,15,20,25],conditions=len(z),candidates=z.shape[1],
        target_separation_energy=((z-z.mean(0))**2).sum((0,1,3)).tolist(),
        modes={'free':{'prediction_error_sum':((p-z)**2).sum((0,1,3)).tolist()}})


def arrays():
    z=np.broadcast_to(np.array([-1.,1.])[:,None,None,None],(2,1,5,1)).copy()
    return z


def compute(p,z):
    q=query_energies(from_arrays(p,z));return score(q['numerator'],q['denominator'])


def test_perfect_and_best_condition_blind_predictions():
    z=arrays();assert compute(z,z)==100
    assert compute(np.zeros_like(z),z)==0
    assert compute(np.ones_like(z),z)<0


def test_tiny_correct_response_does_not_get_full_credit():
    z=arrays();assert compute(.001*z,z)==pytest.approx(.1999)


def test_correct_response_does_not_hide_common_bias():
    z=arrays();assert compute(z+3,z)==pytest.approx(-800)


def test_last_frame_accuracy_cannot_hide_earlier_errors():
    z=arrays();p=np.zeros_like(z);p[:,:,-1]=z[:,:,-1]
    assert compute(p,z)==pytest.approx(20)


def test_zero_separation_is_not_a_perfect_score():
    with pytest.raises(ValueError):compute(np.zeros_like(arrays()),np.zeros_like(arrays()))


def test_condition_candidate_duplication_keeps_query_weight():
    z=arrays();p=.4*z
    a=query_energies(from_arrays(p,z));b=query_energies(from_arrays(np.repeat(p,3,1),np.repeat(z,3,1)))
    assert a['numerator']==pytest.approx(b['numerator'])
    assert a['denominator']==pytest.approx(b['denominator'])


def test_energy_ratio_is_not_mean_of_per_query_ratios():
    assert score([1.,1.],[1.,9.])==pytest.approx(80)


def test_repetitions_are_averaged_after_per_checkpoint_normalization():
    base=dict(task='speed',family='lewm',regime='scratch',panel_sha256='p',queries=[dict(query_id='q',numerator=1.,denominator=2.)])
    a=dict(base,id='a',training_seed=1,checkpoint_sha256='a')
    b=dict(base,id='b',training_seed=2,checkpoint_sha256='b',queries=[dict(query_id='q',numerator=9.,denominator=10.)])
    r=aggregate_runs([a,b]);assert r['score']==pytest.approx(30);assert r['training_repetitions']==2
    assert r['ci95']==pytest.approx([30,30])


def test_primary_uses_free_predictions_never_teacher_forced_future():
    z=arrays();record=from_arrays(np.zeros_like(z),z)
    record['modes']['full']={'prediction_error_sum':[0.]*5}
    q=query_energies(record)
    assert score(q['numerator'],q['denominator'])==0
