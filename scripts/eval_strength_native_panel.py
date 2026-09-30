#!/usr/bin/env python3
"""Recover native-action predictions on an existing Strength candidate panel.

Only the original query action is evaluated. No candidate search, training,
new data synthesis, or Public Test access is performed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha,write
from eval_speed_action_selection import load_adapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stable-repo',required=True)
    p.add_argument('--stable-ref',required=True)
    p.add_argument('--family',choices=['lewm','pldm'],default='lewm')
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'native_icl.json').exists():raise FileExistsError(a.output/'native_icl.json')
    manifest=json.loads((a.panel/'manifest.json').read_text())
    assert sha(a.checkpoint)==a.expected_sha256
    assert len(manifest['pairs'])==256
    adapter=load_adapter(a,dict(protocol=dict(normalization=manifest['normalization'])))
    from contextworld.benchmarks.action_strength_icl_score import _prediction_metrics
    before=adapter.frozen_state_hash()
    predictions,targets,ids=[],[],[]
    for start in range(0,256,8):
        histories,actions,futures=[],[],[]
        for row in manifest['pairs'][start:start+8]:
            path=a.panel/row['path'];assert sha(path)==row['sha256']
            with np.load(path,allow_pickle=False) as data:
                assert np.array_equal(data['history_pixels'][0,-1],data['history_pixels'][1,-1])
                assert data['amplitude_grid'][-1]==1
                histories.extend(data['history_pixels'])
                actions.extend(np.repeat(data['actions'][None],2,axis=0))
                futures.extend(data['future_pixels'][:,-1])
            ids.append(row['pair_id'])
        predictions.extend(adapter.rollout_latents(np.asarray(histories),np.asarray(actions),batch_size=16)[:,0])
        targets.extend(adapter.encode_pixels(np.asarray(futures),batch_size=16))
        if (start+8)%64==0:print('pairs',start+8,flush=True)
    predictions=np.asarray(predictions).reshape(256,2,-1);targets=np.asarray(targets).reshape(256,2,-1)
    metrics,records=_prediction_metrics(pair_ids=ids,predicted_low=predictions[:,0],predicted_high=predictions[:,1],target_low=targets[:,0],target_high=targets[:,1])
    after=adapter.frozen_state_hash();assert before==after
    np.savez_compressed(a.output/'native_predictions.npz',prediction=predictions,target=targets,pair_ids=np.asarray(ids))
    write(a.output/'native_icl.json',dict(metrics=metrics,records=records,optimizer_steps=0,public_test_accessed=False,evaluation_split='development'))
    write(a.output/'receipt.json',dict(checkpoint_sha256=a.expected_sha256,panel_manifest_sha256=sha(a.panel/'manifest.json'),
        state_before=before,state_after=after,optimizer_steps=0,public_test_accessed=False,pair_count=256,
        adapter=adapter.metadata,output_sha256=sha(a.output/'native_icl.json'),predictions_sha256=sha(a.output/'native_predictions.npz')))
    print('complete',metrics['correct_future_rate'],flush=True)

if __name__=='__main__':main()
