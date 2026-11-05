import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from multistep_prediction_score import score,query_energies,aggregate_runs
from diagnose_cross_task_decisions import wrong_history_prediction_error_sum


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
    record['modes']['full']={'prediction_error_sum':[2.,0.,0.,0.,0.]}
    q=query_energies(record)
    assert score(q['numerator'],q['denominator'])==0


def test_feedback_diagnostic_keeps_one_denominator_across_depths():
    # Increasing physical separation must not hide a growing free error.
    base=dict(task='speed',family='lewm',regime='scratch',panel_sha256='p',
              id='a',training_seed=1,checkpoint_sha256='a',queries=[dict(
              query_id='q',numerator=3.,denominator=10.,
              horizon_numerators=[1.,2.,3.,4.,5.],
              horizon_denominators=[1.,2.,5.,12.,30.],
              real_input_horizon_numerators=[1.,1.,1.,1.,1.])])
    r=aggregate_runs([base]);d=r['error_diagnostics']
    assert d['free']['mean']==pytest.approx([.1,.2,.3,.4,.5])
    assert d['feedback_gap']['mean']==pytest.approx([0.,.1,.2,.3,.4])
    assert r['score']==pytest.approx(70)


def test_real_observation_diagnostic_does_not_replace_primary_score():
    z=arrays();record=from_arrays(np.zeros_like(z),z)
    record['modes']['full']={'prediction_error_sum':[2.,0.,0.,0.,0.]}
    q=query_energies(record)
    assert q['real_input_horizon_numerators']==[1.,0.,0.,0.,0.]
    assert score(q['numerator'],q['denominator'])==0


def test_inconsistent_first_step_refresh_is_rejected():
    z=arrays();record=from_arrays(np.zeros_like(z),z)
    record['modes']['full']={'prediction_error_sum':[0.]*5}
    with pytest.raises(ValueError,match='first steps'):
        query_energies(record)


def test_compressed_query_records_roundtrip_without_mutating_payload(tmp_path):
    import gzip,json
    from multistep_prediction_score import write_outputs,sha
    p=dict(runs=[dict(id='r',queries=[dict(query_id='q',numerator=2.)])],rows=[])
    out=tmp_path/'scores.json';write_outputs(p,out,split_queries=True)
    saved=json.loads(out.read_text());ref=saved['query_records']
    data=tmp_path/ref['path']
    assert sha(data)==ref['sha256']
    assert json.loads(gzip.decompress(data.read_bytes()))=={'r':p['runs'][0]['queries']}
    assert 'queries' in p['runs'][0] and 'queries' not in saved['runs'][0]


def test_shared_source_queries_are_bootstrapped_together():
    # Two mirror views of a single source must never become two independent draws.
    queries=[dict(query_id='a',bootstrap_cluster='same-source',numerator=1.,denominator=1.),
             dict(query_id='b',bootstrap_cluster='same-source',numerator=0.,denominator=1.)]
    run=dict(task='motion_damping',family='lewm',regime='scratch',panel_sha256='p',
             id='a',training_seed=1,checkpoint_sha256='a',queries=queries)
    r=aggregate_runs([run])
    assert r['score']==50 and r['bootstrap_clusters']==1
    assert r['ci95']==[50.,50.]


def test_new_diagnostics_preserve_common_bias_and_history_signal():
    z=arrays()
    # Perfect response with shared bias: response NRE is zero, E/B is nine.
    p=z+3
    record=from_arrays(p,z)
    record['modes']['free']['response_error_sum']=[0.]*5
    record['modes']['free']['wrong_history_prediction_error_sum']=wrong_history_prediction_error_sum(p,z)
    q=query_energies(record)
    assert q['response_numerator']==0
    assert q['wrong_history_numerator']==pytest.approx(13)
    base=dict(task='speed',family='lewm',regime='scratch',panel_sha256='p',id='a',
              training_seed=1,checkpoint_sha256='a',queries=[q])
    row=aggregate_runs([base])
    assert row['complete_error_over_reference']==pytest.approx(9)
    assert row['response_nre']==0
    assert row['normalized_history_gain']==pytest.approx(4)
    assert row['score']==pytest.approx(-800)


def test_aggregate_diagnostics_sum_energy_before_division():
    queries=[dict(query_id='a',numerator=1.,denominator=1.,wrong_history_numerator=2.,response_numerator=0.),
             dict(query_id='b',numerator=1.,denominator=9.,wrong_history_numerator=5.,response_numerator=9.)]
    run=dict(task='speed',family='lewm',regime='scratch',panel_sha256='p',id='a',
             training_seed=1,checkpoint_sha256='a',queries=queries)
    row=aggregate_runs([run])
    assert row['complete_error_over_reference']==pytest.approx(.2)
    assert row['normalized_history_gain']==pytest.approx(.5)
    assert row['response_nre']==pytest.approx(.9)


def test_test_split_rejects_legacy_development_provenance(tmp_path):
    import json
    from multistep_prediction_score import read_run
    panel=tmp_path/'panel';result=tmp_path/'result';panel.mkdir();result.mkdir()
    (panel/'manifest.json').write_text(json.dumps({'data_version':'development','scenes':[]}))
    from multistep_prediction_score import sha
    (result/'receipt.json').write_text(json.dumps({'state_hash_before':'a','state_hash_after':'a',
        'no_training':True,'panel_sha256':sha(panel/'manifest.json'),'entry_hashes':{}}))
    with pytest.raises(ValueError,match='split'):
        read_run({},result,panel,'test')


def test_wrong_history_numerical_cases():
    z=arrays()
    cases=[(z,8.),(np.zeros_like(z),2.),(z+3,26.),(-z,0.)]
    for p,expected in cases:
        raw=wrong_history_prediction_error_sum(p,z)
        assert raw==pytest.approx([expected]*5)
        record=from_arrays(p,z)
        record['modes']['free']['wrong_history_prediction_error_sum']=raw
        q=query_energies(record)
        assert q['wrong_history_numerator']==pytest.approx(expected/2)
    assert query_energies(from_arrays(np.zeros_like(z),z))['numerator']==pytest.approx(1)


def test_wrong_history_k3_matches_brute_force_with_candidates_and_horizons():
    rng=np.random.default_rng(11)
    z=rng.normal(size=(3,4,5,2))
    p=rng.normal(size=z.shape)
    observed=wrong_history_prediction_error_sum(p,z)
    expected=[]
    for t in range(5):
        e=0.
        for k in range(3):
            for h in range(3):
                if h!=k:e+=np.square(p[h,:,t]-z[k,:,t]).sum()/2
        expected.append(e)
    assert observed==pytest.approx(expected)


def test_metric_csv_has_intervals_and_blank_legacy_fields(tmp_path):
    import csv
    from multistep_prediction_score import write_outputs
    q=dict(query_id='q',numerator=1.,denominator=2.,wrong_history_numerator=3.,response_numerator=1.)
    run=dict(task='speed',family='lewm',regime='scratch',panel_sha256='p',id='a',
             training_seed=1,checkpoint_sha256='a',queries=[q])
    row=aggregate_runs([run]);out=tmp_path/'metrics.json'
    write_outputs(dict(runs=[run],rows=[row]),out)
    with out.with_suffix('.csv').open() as f:record=next(csv.DictReader(f))
    assert float(record['complete_error_over_reference_ci95_low'])==pytest.approx(.5)
    assert float(record['normalized_history_gain_ci95_high'])==pytest.approx(1)
    legacy=aggregate_runs([dict(run,queries=[dict(query_id='q',numerator=1.,denominator=2.)])])
    write_outputs(dict(runs=[run],rows=[legacy]),out)
    with out.with_suffix('.csv').open() as f:record=next(csv.DictReader(f))
    assert record['normalized_history_gain']=='' and record['response_nre']==''
