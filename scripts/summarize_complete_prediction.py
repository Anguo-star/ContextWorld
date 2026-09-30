#!/usr/bin/env python3
"""Recompute complete prediction accuracy from existing frozen-model outputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write
from complete_prediction_metrics import array_energies,pair_record_energies,summarize,paired_contrast,energy_growth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--speed-root',type=Path,required=True)
    p.add_argument('--horizon-root',type=Path,required=True)
    p.add_argument('--strength-inventory',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    public=repo/'docs/research/data'
    sources={k:json.loads((public/f).read_text()) for k,f in (
        ('speed','speed_action_selection_v1.json'),('horizon','speed_planning_horizon_probe_v1.json'),
        ('strength','icl_action_selection_v1.json'))}
    rows,per_scene,inputs,contrasts=[],[],[],[]
    datasets={}
    manifest=json.loads((a.speed_root/'panel/manifest.json').read_text())
    assert sha(a.speed_root/'panel/manifest.json')==sources['speed']['panel_manifest_sha256']
    # Original query actions, never chosen by the model: no candidate-selection bias.
    for scheme in ('T0','T1'):
        checkpoint=sources['speed']['models'][scheme]['checkpoint_sha256']
        receipts=[json.loads(x.read_text()) for x in (a.speed_root/scheme).glob('receipt_*.json')]
        assert receipts and len(receipts)==receipts[0]['shards']
        assert {r['shard'] for r in receipts}==set(range(len(receipts)))
        assert sum(r['queries'] for r in receipts)==len(manifest['queries'])
        assert all(r['completed'] and r['state_hash_before']==r['state_hash_after'] for r in receipts)
        for track in sources['speed']['protocol']['tracks']:
            selected=[q for q in manifest['queries'] if q['track']==track]
            records=[]
            for q in selected:
                path=a.speed_root/scheme/q['path']
                with np.load(path,allow_pickle=False) as data:
                    assert str(data['checkpoint_sha256'])==checkpoint
                    assert str(data['panel_sha256'])==sources['speed']['panel_manifest_sha256']
                    assert str(data['query_id'])==q['query_id']
                    r=array_energies(data['native_prediction'],data['native_target'])
                records.append(dict(query_id=q['query_id'],**r))
                inputs.append(dict(dataset='speed_native',path=str(path.relative_to(a.speed_root)),sha256=sha(path)))
            assert len(records)==300
            identity=dict(dataset='speed_native',model='lewm',scheme=scheme,track=track,raw_steps=5,checkpoint_sha256=checkpoint)
            rows.append(dict(**identity,**summarize(records)))
            per_scene.extend(dict(**identity,**r) for r in records)
            datasets['speed_native',scheme,track]=records
            print('speed',scheme,track,'complete',flush=True)
    for track in sources['speed']['protocol']['tracks']:
        contrasts.append(dict(dataset='speed_native',track=track,before='T0',after='T1',
                              metrics=paired_contrast(datasets['speed_native','T0',track],datasets['speed_native','T1',track])))
    # Preserve all existing action arms and horizons; do not select on outcomes.
    horizon_inputs={r['path']:r['sha256'] for r in sources['horizon']['inputs']}
    queries=sorted(x.name for x in (a.horizon_root/'panel').iterdir() if x.is_dir())
    assert len(queries)==6
    for scheme in ('T0','T1'):
        for q in queries:
            receipt=json.loads((a.horizon_root/scheme/q/'receipt.json').read_text())
            assert receipt['state_hash_before']==receipt['state_hash_after']
            assert receipt['canonical_rollout_checks_passed']
            assert receipt['source_manifest_sha256']==sha(a.horizon_root/'panel'/q/'manifest.json')
            assert receipt['checkpoint_sha256']==sources['speed']['models'][scheme]['checkpoint_sha256']
        for arm in ('structured','issued','clipped','small'):
            steps={t:[] for t in (5,10,15,20,25)}
            for q in queries:
                path=a.horizon_root/scheme/q/(arm+'.npz')
                h=sha(path);assert h==horizon_inputs[str(path.relative_to(a.horizon_root))]
                with np.load(path,allow_pickle=False) as data:
                    for t in steps:
                        steps[t].append(dict(query_id=q,**array_energies(data['prediction'][:,:,t//5-1],data['target'][:,:,t//5-1])))
                inputs.append(dict(dataset='speed_horizon',path=str(path.relative_to(a.horizon_root)),sha256=h))
            for step,records in steps.items():
                identity=dict(dataset='speed_horizon',model='lewm',scheme=scheme,arm=arm,raw_steps=step,
                              checkpoint_sha256=sources['speed']['models'][scheme]['checkpoint_sha256'])
                rows.append(dict(**identity,**summarize(records)))
                per_scene.extend(dict(**identity,**r) for r in records)
                datasets['speed_horizon',scheme,arm,step]=records
            contrasts.append(dict(dataset='speed_horizon',scheme=scheme,arm=arm,before_raw_steps=5,after_raw_steps=25,
                                  metrics=paired_contrast(steps[5],steps[25]),energy_growth=energy_growth(steps[5],steps[25])))
            print('horizon',scheme,arm,'complete',flush=True)
        for step in (5,25):
            contrasts.append(dict(dataset='speed_horizon',scheme=scheme,raw_steps=step,before='issued',after='clipped',
                metrics=paired_contrast(datasets['speed_horizon',scheme,'issued',step],datasets['speed_horizon',scheme,'clipped',step])))
    inventory=json.loads(a.strength_inventory.read_text())
    entries=inventory['rows'] if isinstance(inventory,dict) else inventory
    for entry in entries:
        path=Path(entry['path']);payload=json.loads(path.read_text())
        rid=entry['id']
        public_row=next(r for r in sources['strength']['rows'] if r['training_comparison_id']==rid)
        receipt=json.loads(Path(entry['receipt']).read_text())
        assert receipt['checkpoint_sha256']==entry['checkpoint_sha256']
        assert receipt['panel_manifest_sha256']==sources['strength']['panel_manifest_sha256']
        assert (receipt.get('state_before') and receipt['state_before']==receipt['state_after']) or receipt.get('frozen_state_unchanged') is True
        assert receipt['optimizer_steps']==0 and receipt['public_test_accessed'] is False
        assert entry['checkpoint_sha256']==public_row['checkpoint_sha256']
        if 'native_icl_source_sha256' in public_row:
            assert sha(path)==public_row['native_icl_source_sha256']
        records=[dict(query_id=r['pair_id'],**pair_record_energies(r)) for r in payload['records']]
        assert len(records)==256 and len({r['query_id'] for r in records})==256
        scheme={'original':'T0','scratch':'T1','joint':'T2','frozen':'T3'}[rid.split('/')[-1]]
        identity=dict(dataset='strength_native',model=rid.split('/')[1],scheme=scheme,raw_steps=5,training_comparison_id=rid,
                      checkpoint_sha256=public_row['checkpoint_sha256'])
        rows.append(dict(**identity,**summarize(records)))
        per_scene.extend(dict(**identity,**r) for r in records)
        inputs.append(dict(dataset='strength_native',training_comparison_id=rid,filename=path.name,sha256=sha(path)))
        datasets['strength_native',identity['model'],scheme]=records
        print('strength',rid,'complete',flush=True)
    assert len(entries)==len(sources['strength']['rows'])==10
    for model in ('lewm','pldm'):
        for before,after in (('T0','T1'),('T0','T2'),('T0','T3'),('T2','T3')):
            contrasts.append(dict(dataset='strength_native',model=model,before=before,after=after,
                                  metrics=paired_contrast(datasets['strength_native',model,before],datasets['strength_native',model,after])))
    contrasts.append(dict(dataset='strength_native',model='dinowm',before='T0',after='T1',
                          metrics=paired_contrast(datasets['strength_native','dinowm','T0'],datasets['strength_native','dinowm','T1'])))
    write(a.output,dict(schema='contextworld.complete_prediction.results.v1',protocol=json.loads(a.protocol.read_text()),
        protocol_sha256=sha(a.protocol),no_training=True,new_model_prediction_conditions=sum(512 for e in entries if e.get('new_inference')),new_inference_scope='Only LeWM Strength T0 original query actions; all other predictions reused.',
        availability='Aggregates and per-scene error energies are included here. Original prediction arrays/records are identified by hashes; no new public dataset release is implied.',
        rows=rows,contrasts=contrasts,per_scene=per_scene,inputs=inputs))
    print('Saved',len(rows),'result rows and',len(per_scene),'scene records',flush=True)

if __name__=='__main__':main()
