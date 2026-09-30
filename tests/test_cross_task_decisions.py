import numpy as np
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_cross_task_decisions import costs_summary

def test_correct_history_can_reduce_physical_regret():
    physical=np.array([[[0.],[4.]],[[4.],[0.]]])
    learned=physical.copy()
    result=costs_summary(learned,physical)
    assert result['matched_regret']==[0.]
    assert result['history_benefit']==[4.]
    shared=physical.mean(0).min(0)-physical.min(1).mean(0)
    assert shared.tolist()==[2.]

def test_different_futures_do_not_imply_different_best_actions():
    physical=np.array([[[0.],[4.]],[[2.],[9.]]])
    gap=physical.mean(0).min(0)-physical.min(1).mean(0)
    assert gap.tolist()==[0.]
    result=costs_summary(physical,physical)
    assert result['history_benefit']==[0.]

def test_fixed_deadline_penalizes_passing_goal_then_overshooting():
    physical=np.array([[[0.,8.],[3.,0.]],[[1.,7.],[3.,1.]]])
    result=costs_summary(physical,physical)
    assert result['selected_indices']==[[0,1],[0,1]]
    assert result['matched_cost']==[.5,.5]

def test_wrong_latent_cost_does_not_count_as_planning_success():
    physical=np.array([[[0.],[10.]],[[10.],[0.]]])
    result=costs_summary(-physical,physical)
    assert result['matched_regret']==[10.]
    assert result['history_benefit']==[-10.]


def test_error_ratio_bootstrap_preserves_source_query_pairing():
    from summarize_cross_task_decisions import ratio_ci
    result=ratio_ci([1.,2.,10.],[2.,4.,20.])
    assert result['mean']==.5
    assert result['ci95']==[.5,.5]
