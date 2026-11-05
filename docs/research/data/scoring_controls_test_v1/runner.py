#!/usr/bin/env python3
"""Test-split constructed-prediction controls; no predictor rollout or training."""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np

ROOT=Path('/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld')
sys.path.insert(0,str(ROOT))
sys.modules.setdefault('flash_attn',None)
from contextworld.benchmarks.action_strength_icl_data import ActionStrengthICLEvalDataset, load_action_strength_icl_release
from contextworld.benchmarks.action_strength_icl_score import _prediction_metrics
from contextworld.benchmarks.adapters import StableWorldModelLeWMActionStrengthAdapter

OUT=Path('/tmp/cw_scoring_control_test_v1')
OUT.mkdir(exist_ok=True)
plan=json.loads((ROOT/'configs/benchmark/current_reference_test_v1.json').read_text())
runmap={r['id']:r for r in plan['runs']}
ids=['action_strength/lewm/original/s3073','action_strength/lewm/frozen/s3072']
if len(sys.argv)>1:
    ids=[sys.argv[1]]
release=load_action_strength_icl_release(ROOT/'configs/benchmark/pusht_action_strength_icl_release_v1.yaml')
arr=ActionStrengthICLEvalDataset(release=release,repo_root=ROOT).arrays
assert arr.pair_count==256
normal=release['evaluation']['action_normalization']
future=np.concatenate([arr.low_pixels[:,3],arr.high_pixels[:,3]],axis=0)
results={}
for rid in ids:
    run=runmap[rid]
    ck=Path(run['checkpoint'])
    assert hashlib.sha256(ck.read_bytes()).hexdigest()==run['checkpoint_sha256']
    adapter=StableWorldModelLeWMActionStrengthAdapter.from_checkpoint(
        ck,action_mean=normal['mean'],action_std=normal['std_population'],
        repo_root=ROOT,stablewm_repo=run['stable_repo'],stablewm_ref=run['stable_ref'],device='cuda:0')
    h0=adapter.frozen_state_hash()
    target=np.asarray(adapter.encode_pixels(future,batch_size=32),dtype=np.float64)
    assert h0==adapter.frozen_state_hash()
    a,b=target[:256],target[256:]
    assert np.all(np.mean((b-a)**2,axis=1)>0)
    m=(a+b)/2
    cases={
      'true_target':(a,b,100,100,0),
      'same_target_A':(a,a,50,0,1),
      'same_midpoint':(m,m,0,0,1),
      'swapped_targets':(b,a,0,0,4),
      'tiny_correct_response':(m+.001*(a-m),m+.001*(b-m),100,100,.998001),
    }
    scored={}
    for name,(pa,pb,accuracy,history,nre) in cases.items():
        summary,_=_prediction_metrics(pair_ids=arr.pair_ids,predicted_low=pa,predicted_high=pb,target_low=a,target_high=b)
        actual=[100*summary['correct_future_rate'],100*summary['correct_history_rate'],summary['latent_response']['normalized_response_error']]
        expected=[accuracy,history,nre]
        assert np.allclose(actual,expected,rtol=0,atol=2e-9),(rid,name,actual,expected)
        scored[name]={'accuracy_percent':actual[0],'history_percent':actual[1],'NRE':actual[2]}
    results[rid]={'checkpoint_sha256':run['checkpoint_sha256'],'pair_count':256,'target_shape':list(target.shape),'minimum_target_separation_mse':float(np.min(np.mean((b-a)**2,axis=1))),'scenarios':scored}
    del adapter
    print(rid,'passed',flush=True)
payload={'schema':'contextworld.test-constructed-prediction-controls.v1','split':'public_test','query':'action_strength, 256 paired scenes, nonzero action candidate, five-step future','runner':str(Path(__file__).resolve()),'scorer_sha256':hashlib.sha256((ROOT/'contextworld/benchmarks/action_strength_icl_score.py').read_bytes()).hexdigest(),'results':results}
out=OUT/(ids[0].replace('/','_')+'.json')
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(out)
