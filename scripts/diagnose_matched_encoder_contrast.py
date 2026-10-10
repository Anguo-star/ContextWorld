#!/usr/bin/env python3
"""Matched existing LeWM joint/frozen checkpoint contrast on native Train/Dev panels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from diagnose_task_history_readout import helper, native_history, _fit_dual, _predict_dual, sha, write

TASKS = ('contact_friction', 'cube_gripper_carry', 'action_strength', 'robot_arm_mass')

def centered_energy(values, groups):
    """Sum within-query squared latent deviations; groups have matched conditions."""
    unique, inverse = np.unique(groups, return_inverse=True)
    count = np.bincount(inverse)
    assert len(unique) > 0 and np.all(count == count[0]) and count[0] > 1
    mean = np.zeros((len(unique), values.shape[-1]), np.float64)
    np.add.at(mean, inverse, values)
    mean /= count[:, None]
    return float(np.square(values - mean[inverse]).sum(dtype=np.float64))

def conditional_fraction(values, groups):
    values = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    within = centered_energy(values, groups)
    total = float(np.square(values-values.mean(axis=0,keepdims=True)).sum(dtype=np.float64))
    assert np.isfinite(within) and np.isfinite(total) and total > 0
    assert 0 <= within <= total*(1+1e-9)
    return {'within_query_energy':within, 'global_centered_energy':total,
            'between_query_energy':total-within, 'conditional_variance_fraction':within/total}

def readout(train_x, dev_x, train_y, dev_y):
    from sklearn.metrics import balanced_accuracy_score
    classes = np.unique(train_y)
    assert len(classes) >= 2 and np.isin(dev_y, classes).all()
    target = np.where(train_y[:, None] == classes[None], 1., -1.)
    if len(classes) == 2: target = target[:, 1]
    scores = _predict_dual(_fit_dual(train_x, target), dev_x)
    pred = classes[(scores > 0).astype(int)] if len(classes) == 2 else classes[scores.argmax(1)]
    return {'accuracy':float(np.mean(pred == dev_y)),
            'balanced_accuracy':float(balanced_accuracy_score(dev_y,pred)),
            'classes':classes.tolist()}

def run(task, regime, args):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = json.loads(args.models.read_text())
    spec = next(x for x in models if x['id'] == f'{task}/lewm/{regime}/s3072')
    panel = args.panels/task
    manifest_path=panel/'manifest.json'; manifest=json.loads(manifest_path.read_text())
    norm=manifest['normalization']
    output=args.output/task/regime; output.mkdir(parents=True,exist_ok=True)
    adapter=helper().load_adapter(spec,norm,output,args.device)
    adapter.model.eval(); before=adapter.frozen_state_hash()
    splits={}
    for split in ('training','development'):
        info=manifest['splits'][split]
        path=panel/info['path']; assert sha(path)==info['sha256']
        with np.load(path,allow_pickle=False) as raw:
            data={key:raw[key] for key in raw.files}
        h,y=helper().encode_unique(adapter,data['history_pixels'],data['queryfuture_pixels'])
        splits[split]=(data,h,y)
        print(task,regime,split,h.shape,flush=True)
    assert before==adapter.frozen_state_hash()
    train,htrain,_=splits['training']; dev,hdev,y=splits['development']
    assert len(np.unique(train['labels']))>=2
    full=readout(native_history(htrain),native_history(hdev),train['labels'],dev['labels'])
    current=readout(htrain[:,-1],hdev[:,-1],train['labels'],dev['labels'])
    # First predicted endpoint uses the native 3-frame context and the same raw actions.
    pred=helper().predict_all(adapter,hdev,y[:,None],dev['action_blocks'],'lewm',modes=('free',))['free'][:,0]
    groups=dev['query_ids']; target_variance=conditional_fraction(y,groups)
    history_variance=conditional_fraction(hdev.reshape(len(hdev),-1),groups)
    target_energy=target_variance['within_query_energy']; assert target_energy>0
    # Center the prediction error within each matched query; this measures response fidelity.
    unique,inverse=np.unique(groups,return_inverse=True)
    residual=pred-y
    means=np.zeros((len(unique),residual.shape[-1]),np.float64)
    np.add.at(means,inverse,residual);means/=np.bincount(inverse)[:,None]
    response_error=float(np.square(residual-means[inverse]).sum(dtype=np.float64))
    endpoint_error=float(np.square(residual).mean(dtype=np.float64))
    result={'schema':'contextworld.matched_encoder_contrast.v1','task':task,'regime':regime,
      'model_id':spec['id'],'checkpoint_sha256':spec['checkpoint_sha256'],
      'panel_manifest_sha256':sha(manifest_path),'split_sha256':{s:manifest['splits'][s]['sha256'] for s in splits},
      'model_state_hash_before':before,'model_state_hash_after':adapter.frozen_state_hash(),
      'no_model_update':True,'no_test_read':True,'readout':'StandardScaler(Train)+Ridge(alpha=1), same hidden labels, Dev evaluation',
      'readout_full_history':full,'readout_current_only':current,
      'readout_history_gain_balanced_accuracy':full['balanced_accuracy']-current['balanced_accuracy'],
      'target_variance':target_variance,'history_variance':history_variance,
      'response':{'target_energy':target_energy,'response_error':response_error,
        'response_nre':response_error/target_energy,'endpoint_mse':endpoint_error,
        'response_error_mse':response_error/residual.size,
        'common_error_mse':endpoint_error-response_error/residual.size,
        'latent_dim':residual.shape[-1],
        'dev_rows':len(dev['labels']),'matched_query_groups':len(unique)},
      'limitations':'Joint versus frozen compares total training-policy effects; readout accuracy does not prove rollout sufficiency.'}
    write(output/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--regime',choices=('joint','frozen'),required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--models',type=Path,default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    p.add_argument('--panels',type=Path,default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    p.add_argument('--output',type=Path,default=Path('/tmp/cw-matched-encoder-20261010'))
    a=p.parse_args();run(a.task,a.regime,a)
if __name__=='__main__':main()
