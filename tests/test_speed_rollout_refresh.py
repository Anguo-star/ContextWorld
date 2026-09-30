import sys
from pathlib import Path
import numpy as np
import torch
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_speed_rollout_refresh import context_window, infer_modes, visual_speed_estimate
from summarize_speed_rollout_refresh import error_ratio_interval, summarize


def test_refresh_alignment_and_no_future_leakage():
    truth=torch.arange(8.).reshape(1,8,1)
    frames=[torch.tensor([[x]]) for x in (100.,101.,102.,103.,104.)]
    assert context_window(frames,truth,2,'free').flatten().tolist()==[102,103,104]
    assert context_window(frames,truth,2,'current').flatten().tolist()==[102,103,4]
    assert context_window(frames,truth,2,'past').flatten().tolist()==[2,3,104]
    assert context_window(frames,truth,2,'full').flatten().tolist()==[2,3,4]
    assert [x.item() for x in frames]==[100,101,102,103,104]
    with pytest.raises(ValueError): context_window(frames,truth,2,'unknown')


class BiasedTransition:
    def action_encoder(self,x): return x
    def predict(self,h,a): return h+a+1


def test_known_accumulation_and_action_alignment():
    h=np.arange(3,dtype=np.float32).reshape(1,3,1)
    # true action from state i is i+1; ensures shifted action slices fail.
    a=np.arange(1,8,dtype=np.float32).reshape(1,7,1)
    y=np.array([5,9,14,20,27],np.float32).reshape(1,5,1)
    out=infer_modes(BiasedTransition(),h,y,a)
    np.testing.assert_allclose((out['free']-y).ravel(),[1,2,3,4,5])
    np.testing.assert_allclose((out['full']-y).ravel(),np.ones(5))
    np.testing.assert_array_equal(out['current'],out['full'])
    np.testing.assert_array_equal(out['past'],out['free'])
    assert all(np.array_equal(v[:,0],out['free'][:,0]) for v in out.values())


def test_visual_speed_uses_separate_transitions_not_roundtrip_sum():
    pixels=np.zeros((3,30,30,3),np.uint8)
    for i,x in enumerate([20,10,20]): pixels[i,15,x,0]=255
    a=np.zeros((2,5,2),np.float32);a[0,:,0]=-.5;a[1,:,0]=.5
    assert visual_speed_estimate(pixels,a)['estimated_speed']==4
    with pytest.raises(ValueError):visual_speed_estimate(pixels,np.zeros_like(a))


def test_ratio_aggregates_energy_and_uses_paired_scenes():
    b=np.arange(1.,7.)
    result=error_ratio_interval(b*.25,b)
    assert result['value']==.25 and result['ci95']==[.25,.25]
    np.testing.assert_allclose(error_ratio_interval(np.ones(6),b)['value'],6/21)
    assert error_ratio_interval(np.ones(6),b)==error_ratio_interval(np.ones(6),b)
    with pytest.raises(ValueError): error_ratio_interval(b,np.zeros(6))
    with pytest.raises(ValueError): summarize([], 'T1')
