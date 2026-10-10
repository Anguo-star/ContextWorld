#!/usr/bin/env python3
"""Reproduce the native H7 first-endpoint diagnostic without model training.

--export-root reads existing history-validation results. Otherwise all summary
statistics and tables reproduce from the published sufficient statistics.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
from render_training_comparison import aggregate,MODEL_ZH
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'docs/research/data'
STEM='delay_first_step_mechanism_v1'; RECORDS=DATA/(STEM+'_records.json.gz'); OUT=DATA/(STEM+'.json');DOC=ROOT/'docs/ICL_Metric_Study.md';MARK='DELAY_FIRST_STEP_MECHANISM'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def metrics(energy):
    # Columns are B, response error, matched error, other-history mean error.
    b,r,e,w=np.moveaxis(np.asarray(energy),-1,0)
    assert np.all(b>0)
    R=r/b; g=(w-e)/b*10/22; q=R-1+2*g
    assert np.all(q>=-1e-10),f'Negative conditional prediction variance: {np.min(q)}'
    return np.stack((R,g,np.sqrt(np.maximum(q,0)),e/b),axis=-1)

def export(root,panel_root,encoding):
    source=read(DATA/'icl_measurement_validation_v1.json'); enc=read(encoding)
    assert enc['state_hash_before']==enc['state_hash_after'] and enc['no_training']
    manifest=read(panel_root/'manifest.json');queue=[]
    for entry in manifest['scenes']:
        with np.load(panel_root/entry['path']) as z:
            queue.append({'scene_id':entry['scene_id'],'nonzero_pending_commands':int(np.count_nonzero(z['aux_pending_actions_at_query'])),'max_queue_length':int(z['aux_pending_action_lengths'].max())})
    entries={e['scene_id']:e for e in manifest['scenes']}; enc_rows={r['scene_id']:r for r in enc['rows']}
    assert len(queue)==len(enc_rows)==len(entries)==300
    target_checks=[]; runs=[]
    for run in source['runs']:
        spec=run['model']
        if spec['task']!='action_delay':continue
        receipt=run['history_receipt'];directory=root/'results'/spec['id'];rows=[]
        assert receipt['no_training'] and receipt['state_hash_before']==receipt['state_hash_after']
        for name,digest in sorted(receipt['entry_hashes'].items()):
            p=directory/name;assert sha(p)==digest
            d=read(p);entry=entries[d['scene_id']]
            assert d['checkpoint_sha256']==spec['checkpoint_sha256'] and d['source_sha256']==entry['sha256']
            assert d['manifest_sha256']==receipt['panel_sha256'] and d['horizons'][0]==5
            assert d['conditions']==11
            size=d['conditions']*d['candidates']
            energy=[d['target_separation_energy'][0]/size,d['modes']['free']['response_error_sum'][0]/size,d['matched']['energy_mean'][0],d['wrong_allother']['energy_mean'][0]]
            rows.append({'scene_id':d['scene_id'],'source_sha256':d['source_sha256'],'energy':energy})
            if spec['id']==enc['model_id']:
                assert spec['checkpoint_sha256']==enc['checkpoint_sha256'] and receipt['panel_sha256']==enc['panel_sha256']
                assert enc_rows[d['scene_id']]['source_sha256']==d['source_sha256']
                fp=directory/d['features_file'];assert sha(fp)==d['features_file_sha256']
                native=enc_rows[d['scene_id']]['native_query_candidates'][0]
                with np.load(fp) as z:y=z['target'][:,native,0].astype(np.float64)
                # Distinct pooled features imply distinct full native features;
                # do not use these pooled distances for the native error score.
                representatives=y[:6];dist=[float(np.sum((representatives[i]-representatives[j])**2)) for i in range(6) for j in range(i+1,6)]
                stationary_equal=all(np.array_equal(y[5],y[k]) for k in range(6,11))
                target_checks.append({'scene_id':d['scene_id'],'distinct_physical_group_pairs':sum(x>0 for x in dist),'pairs':15,'stationary_group_equal':stationary_equal,'pooled_separation_min':min(dist),'witness_space':'DINO 4x4 spatial means; distinctness only','candidate_index':native})
        assert len(rows)==300
        runs.append({'id':spec['id'],'family':spec['family'],'training_seed':spec['training_seed'],'checkpoint_sha256':spec['checkpoint_sha256'],'panel_sha256':receipt['panel_sha256'],'rows':rows})
    assert len(runs)==9 and len(target_checks)==300
    payload={'schema':'contextworld.delay_first_step_mechanism.records.v1','runs':runs,'history_encoding':enc,'target_distinctness':target_checks,'queue_check':queue}
    RECORDS.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0))

def summarize(raw):
    original=read(DATA/'icl_training_study_v2.json');out=[]
    for family in ['lewm','pldm','dinowm']:
        runs=sorted([r for r in raw['runs'] if r['family']==family],key=lambda r:r['training_seed'])
        ids=[q['scene_id'] for q in runs[0]['rows']]; assert all([q['scene_id'] for q in r['rows']]==ids for r in runs)
        weights=np.random.default_rng(20261009).multinomial(300,np.full(300,1/300),size=1000)
        values=[];boot=[];details=[]
        for run in runs:
            energy=np.asarray([q['energy'] for q in run['rows']]); m=metrics(energy.sum(axis=0));values.append(m);boot.append(metrics(weights@energy))
            details.append({'id':run['id'],'response_error':m[0],'response_gain':m[1],'response_amplitude_ratio':m[2],'complete_error_ratio':m[3]})
        mean=np.mean(values,axis=0); ci=np.quantile(np.mean(boot,axis=0),[.025,.975],axis=0);sd=np.std(values,axis=0,ddof=1)
        old=next(r for r in original['rows'] if r['task']=='action_delay' and r['model']==family and r['regime']=='scratch');ag=aggregate(old)
        out.append({'family':family,'training_repetitions':len(runs),'scenes':300,'original_choice':ag['scores']['main'],'original_response_nre':ag['scores']['nre'],
          'metrics':{k:{'mean':float(mean[j]),'ci95':ci[:,j].tolist(),'training_sd':float(sd[j])} for j,k in enumerate(['response_error','response_gain','response_amplitude_ratio','complete_error_ratio'])},'runs':details})
    h=raw['history_encoding'];hr=h['rows'];tr=raw['target_distinctness'];qr=raw['queue_check']
    assert all(r['raw_history_distinct_pairs']==r['encoded_history_distinct_pairs']==r['temporal_difference_distinct_pairs']==55 for r in hr)
    assert all(r['distinct_physical_group_pairs']==15 and r['stationary_group_equal'] for r in tr)
    assert all(r['nonzero_pending_commands']==0 for r in qr)
    return {'schema':'contextworld.delay_first_step_mechanism.v1','scope':'300 Development scenes, native H7 T1, three training repetitions per family. First endpoint is raw step 5, no autoregressive feedback yet.',
      'protocol':{'conditions':11,'physical_future_groups_at_raw_step5':6,'candidate_actions':13,'weights':'equal conditions and candidates within scene, ratio of summed energies per checkpoint, then equal training-repetition mean',
       'metric_identity':'g=(K-1)/(2K)*(Ewrong-Ematched)/B; amplitude=sqrt(R-1+2g). K=11. B uses raw-step-5 targets only, not all-horizon B.',
       'bootstrap':'1000 paired source-scene draws shared across training repetitions; checkpoints fixed; no cross-encoder physical-error ranking',
       'original_score':'Six physical-group original-query choice and response NRE retained separately; its weights and candidate actions differ from the diagnostic.',
       'interpretation':'Representation distinctness excludes exact information erasure only; not a trained decoder or proof of sufficient geometry. Response amplitude is a diagnostic, not an extra benchmark score.'},
      'sources':{RECORDS.name:sha(RECORDS),'icl_training_study_v2.json':sha(DATA/'icl_training_study_v2.json')},'rows':out,
      'representation_check':{'model_id':h['model_id'],'checkpoint_sha256':h['checkpoint_sha256'],'panel_sha256':h['panel_sha256'],'scenes':len(hr),'history_pairs':sum(r['pairs'] for r in hr),'distinct_history_pairs':sum(r['encoded_history_distinct_pairs'] for r in hr),'distinct_temporal_difference_pairs':sum(r['temporal_difference_distinct_pairs'] for r in hr),'target_group_pairs':sum(r['pairs'] for r in tr),'distinct_target_group_pairs':sum(r['distinct_physical_group_pairs'] for r in tr),'weights_unchanged':h['state_hash_before']==h['state_hash_after']},
      'queue_check':{'scenes':len(qr),'nonzero_pending_commands':sum(r['nonzero_pending_commands'] for r in qr),'note':'Queue length varies with delay but every pending command at the query is zero; queue-length metadata is never given to the model.'}}

def render(d):
    lines=['| 模型 | 原查询选择率 (%) ↑ | 原查询响应 NRE ↓ | 扩展动作响应误差 ↓ | 扩展动作响应幅度比（理想为 1） | 扩展动作方向增益（理想为 1） |','|---|---:|---:|---:|---:|---:|']
    for r in d['rows']:
        m=r['metrics'];amp=m['response_amplitude_ratio'];lo,hi=amp['ci95']
        vals=[MODEL_ZH[r['family']],f"{r['original_choice']['mean']:.2f}",f"{r['original_response_nre']['mean']:.5f}",f"{m['response_error']['mean']:.5f}",f"{amp['mean']:.5f} [{lo:.5f}, {hi:.5f}]",f"{m['response_gain']['mean']:.6f}"]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-root',type=Path);p.add_argument('--history-encoding',type=Path);p.add_argument('--panel-root',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    assert not (a.check and a.export_root)
    if a.export_root:export(a.export_root,a.panel_root,a.history_encoding)
    with gzip.open(RECORDS,'rt') as f:raw=json.load(f)
    d=summarize(raw);text=json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n';doc=DOC.read_text();start=f'<!-- BEGIN {MARK} -->';end=f'<!-- END {MARK} -->';assert doc.count(start)==doc.count(end)==1;old=doc.split(start)[1].split(end)[0];new='\n'+render(d)+'\n'
    if a.check:assert OUT.read_text()==text;assert old==new
    else:OUT.write_text(text);DOC.write_text(doc.replace(start+old+end,start+new+end))
    print(render(d));print('verified' if a.check else 'written')

if __name__=='__main__':main()
