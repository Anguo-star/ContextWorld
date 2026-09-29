"""Guard interpretation of pairwise controls and numerical preferences."""
import copy
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_speed_timed_arrival import cost_order
from summarize_speed_search_diagnosis import summarize


def test_cost_ties_are_symmetric_and_not_search_gaps():
    assert cost_order(1.,1.+1e-6)=='tie'
    assert cost_order(1.+1e-6,1.)=='tie'
    assert cost_order(2.,1.)=='reference'
    assert cost_order(1.,2.)=='cem'
    with pytest.raises(ValueError):
        cost_order(float('nan'),1.)


def fixture_rows():
    data=json.loads((ROOT/'docs/research/data/speed_timed_arrival_search_v1.json').read_text())
    return [r for r in data['per_condition'] if r['scheme']=='T1']


def test_worse_physical_reference_does_not_certify_useful_search_gap():
    rows=fixture_rows()
    rows[0]['physical_reference_better']=False
    with pytest.raises(ValueError,match='physical improvement'):
        summarize(rows)


def test_duplicate_scene_cannot_replace_missing_scene():
    rows=fixture_rows()
    rows[-1]=copy.deepcopy(rows[0])
    with pytest.raises(ValueError,match='complete scenes'):
        summarize(rows)


def test_public_counts_come_from_exact_executed_plan_costs():
    data=json.loads((ROOT/'docs/research/data/speed_timed_arrival_search_v1.json').read_text())
    for summary in data['rows']:
        rows=[r for r in data['per_condition'] if r['scheme']==summary['scheme']]
        for r in rows:
            cem,ref=r['candidates']['cem'],r['candidates']['reference']
            assert r['predicted_preference']==cost_order(cem['native_terminal_goal_cost'],ref['native_terminal_goal_cost'])
            assert r['encoded_true_preference']==cost_order(cem['encoded_true_goal_cost_by_step'][-1],ref['encoded_true_goal_cost_by_step'][-1])
            assert r['physical_reference_better']==(ref['terminal_distance'] < cem['terminal_distance'])
        actual=summarize(rows)
        assert actual=={k:v for k,v in summary.items() if k!='scheme'}


def test_cost_tie_retains_existing_plan_without_crediting_reference():
    rows=fixture_rows()
    for r in rows:
        r['predicted_preference']='tie'
    result=summarize(rows)
    assert result['tied_cost_count']==18
    assert result['search_gap_count']==result['model_cost_misranking_count']==0
    assert result['two_candidate_distance']==result['cem_distance']
    assert result['two_candidate_success_count']==result['cem_success_count']
