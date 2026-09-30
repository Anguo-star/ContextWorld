#!/usr/bin/env python3
"""Summarize privileged refresh diagnostics using paired scene resampling."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write
from diagnose_speed_rollout_refresh import MODES


def error_ratio_interval(error, baseline):
    """Ratios of summed errors, not the mean of per-condition error ratios."""
    error=np.asarray(error,dtype=np.float64); baseline=np.asarray(baseline,dtype=np.float64)
    if error.shape != baseline.shape or error.ndim != 1 or len(error)!=6:
        raise ValueError('Expected six paired scene energies')
    if not np.isfinite(error).all() or not np.isfinite(baseline).all() or np.any(error<0) or np.any(baseline<=0):
        raise ValueError('Invalid error energies')
    indices=np.random.default_rng(20260930).integers(0,6,size=(10000,6))
    samples=error[indices].sum(1)/baseline[indices].sum(1)
    return dict(value=float(error.sum()/baseline.sum()), ci95=np.quantile(samples,[.025,.975]).tolist())


def summarize(scene_rows, scheme):
    if len(scene_rows)!=6 or any(len(rows)!=3 or {r['speed_index'] for r in rows}!={0,1,2} for rows in scene_rows):
        raise ValueError('Expected all six scenes and three speeds')
    for rows in scene_rows:
        if any(set(r['modes'])!=set(MODES) for r in rows):
            raise ValueError('Incomplete intervention modes')
    free=np.asarray([[r['modes']['free']['prediction_squared_error'] for r in rows] for rows in scene_rows],np.float64)
    if free.shape!=(6,3,2,5): raise ValueError(free.shape)
    out=[]
    for mode in MODES:
        err=np.asarray([[r['modes'][mode]['prediction_squared_error'] for r in rows] for rows in scene_rows],np.float64)
        if err.shape!=free.shape or not np.array_equal(err[...,0],free[...,0]):
            raise ValueError('First step must agree for all modes')
        ratios=[error_ratio_interval(err[...,t].mean((1,2)),free[...,t].mean((1,2))) for t in range(5)]
        candidate_ratios={name:error_ratio_interval(err[:,:,i,-1].mean(1),free[:,:,i,-1].mean(1)) for i,name in enumerate(('cem','reference'))}
        allrows=[r for rows in scene_rows for r in rows]
        collision_free=[r for r in allrows if not r['contacts'][0]]
        preferences={name:sum(r['modes'][mode]['terminal_preference']==name for r in allrows) for name in ('reference','cem','tie')}
        out.append(dict(scheme=scheme,mode=mode,conditions=18,scenes=6,
            mean_squared_latent_error_by_depth=err.mean((0,1,2)).tolist(),
            error_relative_to_free_by_depth=ratios,candidate_terminal_error_ratios=candidate_ratios,
            terminal_preference_counts=preferences,
            collision_free_cem_conditions=len(collision_free),
            collision_free_reference_preferences=sum(r['modes'][mode]['terminal_preference']=='reference' for r in collision_free)))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--previous-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    panel=a.source_root/'panel'; m=json.loads((panel/'manifest.json').read_text())
    protocol=json.loads((a.root/'protocol.json').read_text())
    rows=[]; inputs=[]; per_condition=[]; identities={}; architecture={}; visual={}
    for scheme in ('T0','T1'):
        scene_rows=[]
        for entry in m['queries']:
            q=entry['query_id'];path=a.root/scheme/(q+'.json');x=json.loads(path.read_text())
            assert x['query_id']==q and x['panel_sha256']==sha(panel/'manifest.json')
            assert x['source_data_sha256']==entry['sha256']==sha(panel/entry['path'])
            assert x['saved_plan_sha256']==sha(a.source_root/scheme/(q+'.json'))
            assert x['previous_diagnosis_sha256']==sha(a.previous_root/scheme/(q+'.json'))
            assert x['protocol_sha256']==sha(a.root/'protocol.json') and x['array_sha256']==sha(path.with_suffix('.npz'))
            assert x['state_hash_before']==x['state_hash_after'] and x['no_training'] and x['new_searches']==0
            identities.setdefault(scheme,x['checkpoint_sha256']); assert identities[scheme]==x['checkpoint_sha256']
            architecture.setdefault(scheme,x['architecture']); assert architecture[scheme]==x['architecture']
            scene_rows.append(x['rows'])
            for r in x['rows']:
                key=(q,r['speed_index'])
                vh=dict(query_id=q,speed_index=r['speed_index'],actual_speed=r['speed'],**r['visual_history'])
                visual.setdefault(key,vh);assert visual[key]==vh
                per_condition.append(dict(scheme=scheme,query_id=q,**r))
            inputs.append(dict(path=str(path.relative_to(a.root)),sha256=sha(path),array_sha256=x['array_sha256'],
                source_data_sha256=x['source_data_sha256'],saved_plan_sha256=x['saved_plan_sha256'],previous_diagnosis_sha256=x['previous_diagnosis_sha256']))
        rows.extend(summarize(scene_rows,scheme))
    v=list(visual.values()); speeds=np.asarray([3.4,4.8,6.9])
    maxerr=max(abs(x['estimated_speed']-x['actual_speed']) for x in v)
    resolved=sum(int(np.argmin(abs(speeds-x['estimated_speed'])))==x['speed_index'] for x in v)
    write(a.output,dict(schema=protocol['schema'],protocol=protocol,protocol_sha256=sha(a.root/'protocol.json'),
        panel_sha256=sha(panel/'manifest.json'),checkpoint_sha256=identities,architecture=architecture,
        visual_history_check=dict(conditions=len(v),correct_condition_assignments=resolved,max_speed_error=maxerr,per_condition=v),
        rows=rows,per_condition=per_condition,inputs=inputs))
    for r in rows:
        print(r['scheme'],r['mode'],r['error_relative_to_free_by_depth'][-1],r['terminal_preference_counts'])

if __name__=='__main__': main()
