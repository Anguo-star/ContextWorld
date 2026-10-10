#!/usr/bin/env python3
"""Reproduce split diagnostics from published per-source sufficient statistics."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'docs/research/data';STEM='delay_train_development_v1';MARK='DELAY_TRAIN_DEVELOPMENT'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ratios(energy):
    b,r,e,w,p,common=np.moveaxis(np.asarray(energy),-1,0)
    assert np.all(b>0) and np.all(p>=0)
    g=(w-e)/b*10/22
    assert np.allclose(e,r+common,rtol=1e-8,atol=1e-7)
    assert np.allclose(r,p+b-2*g*b,rtol=1e-8,atol=1e-7)
    return np.stack([r/b,g,np.sqrt(p/b),e/b,common/b],axis=-1)

def export(run_root):
    panel=read(run_root/'manifest.json');runs=[]
    for path in sorted((run_root/'results').glob('*/*/result.json')):
        r=read(path);model=r['model'];assert r['panel_sha256']==sha(run_root/'manifest.json') and r['no_training'] and r['state_hash_before']==r['state_hash_after']
        for split in ['training','development']:
            entries={e['scene_id']:e for e in panel['splits'][split]['scenes']};rows=[x for x in r['rows'] if x['split']==split]
            assert set(x['scene_id'] for x in rows)==set(entries)
            assert all(x['source_sha256']==entries[x['scene_id']]['sha256'] for x in rows)
        runs.append({'id':model['id'],'family':model['family'],'training_seed':model['training_seed'],'checkpoint_sha256':model['checkpoint_sha256'],
          'checkpoint_name':Path(model['checkpoint']).name,'state_hash':r['state_hash_before'],'canonical_checks':r['canonical_checks'],'rows':r['rows']})
    assert len(runs)==9 and len(set(r['id'] for r in runs))==9
    runtime=read(run_root/'runtime_environment.json');assert runtime['optimizer_split_torch_matches']
    raw={'schema':'contextworld.delay_train_development.records.v1','panel':panel,'runtime_environment':runtime,'runs':runs}
    target=DATA/(STEM+'_records.json.gz');target.write_bytes(gzip.compress(json.dumps(raw,separators=(',',':'),ensure_ascii=False).encode(),mtime=0))

def summarize(raw):
    panel=raw['panel'];tables=[];names=['response_error','response_gain','response_amplitude_ratio','complete_error_ratio','common_bias_ratio']
    bootstrap={}
    for split in ['training','development']:
        entries=panel['splits'][split]['scenes'];groups={}
        for i,e in enumerate(entries):groups.setdefault(e['room']+'/'+e['direction'],[]).append(i)
        rng=np.random.default_rng(20261009+(split=='development'));w=np.zeros((1000,len(entries)),np.int64)
        for indices in groups.values():w[:,indices]=rng.multinomial(len(indices),np.full(len(indices),1/len(indices)),size=1000)
        bootstrap[split]=w
    for family in ['lewm','pldm','dinowm']:
        runs=sorted([r for r in raw['runs'] if r['family']==family],key=lambda r:r['training_seed']);assert len(runs)==3
        by_split={};per_run=[]
        for split in ['training','development']:
            ordered_ids=[e['scene_id'] for e in panel['splits'][split]['scenes']];vals=[];boots=[];choices=[];choice_boot=[]
            for run in runs:
                rr={r['scene_id']:r for r in run['rows'] if r['split']==split};energy=np.asarray([rr[k]['energy'] for k in ordered_ids]);choice=np.asarray([rr[k]['six_group_choice'] for k in ordered_ids])
                value=ratios(energy.sum(0));boot=ratios(bootstrap[split]@energy)
                vals.append(value);boots.append(boot);choices.append(choice.mean()*100);choice_boot.append(bootstrap[split]@choice/len(choice)*100)
                per_run.append({'id':run['id'],'split':split,'metrics':dict(zip(names,value.tolist())),'six_group_choice_pct':float(choice.mean()*100)})
            mean=np.mean(vals,0);ci=np.quantile(np.mean(boots,0),[.025,.975],axis=0);sd=np.std(vals,axis=0,ddof=1)
            by_split[split]={'scenes':len(ordered_ids),'metrics':{k:{'mean':float(mean[i]),'ci95':ci[:,i].tolist(),'training_sd':float(sd[i])} for i,k in enumerate(names)},
              'six_group_choice_pct':{'mean':float(np.mean(choices)),'ci95':np.quantile(np.mean(choice_boot,0),[.025,.975]).tolist()},'_boot':np.mean(boots,0)}
        gap=by_split['development']['_boot']-by_split['training']['_boot'];difference=by_split['development']['metrics']['response_error']['mean']-by_split['training']['metrics']['response_error']['mean']
        for split in by_split:by_split[split].pop('_boot')
        tables.append({'family':family,'training_repetitions':3,'splits':by_split,'development_minus_training_response_error':{'mean':difference,'ci95':np.quantile(gap[:,0],[.025,.975]).tolist()},'runs':per_run})
    return {'schema':'contextworld.delay_train_development.v1','scope':'Existing native H7 T1 frozen checkpoints; raw first endpoint 5; 128 real Training and 128 Development scenes; three checkpoint seeds per family; no training.',
      'protocol':{**panel['protocol'],'bootstrap':'1000 source-group draws stratified by room/direction, independently by split and shared across fixed checkpoints; no uncertainty from new training.',
        'metrics':'Native checkpoint target space. Response R=conditional-centered error/target-centered energy; gain and amplitude ideal1; complete E/B includes common prediction bias. No cross-encoder physical-error ranking.',
        'aggregation':'Sum scene energies per split/checkpoint, divide by its target energy; equal checkpoint mean. Choice six-group macro per scene; response eleven delays equal.'},
      'source':{**panel['source'],'runtime_environment':raw['runtime_environment']},'split_selection':{k:{key:v for key,v in x.items() if key!='scenes'} for k,x in panel['splits'].items()},
      'sources':{STEM+'_records.json.gz':sha(DATA/(STEM+'_records.json.gz'))},'rows':tables}

def render(d):
    lines=['| 模型 | 训练集选择率 (%) ↑ | Development 选择率 (%) ↑ | 训练集响应误差 ↓ | Development 响应误差 ↓ | 训练集 / Development 响应幅度比（理想为 1） | 训练集 / Development 完整误差 ↓ |','|---|---:|---:|---:|---:|---:|---:|']
    model_names={'lewm':'LeWM','pldm':'PLDM','dinowm':'DINO-WM'}
    for row in d['rows']:
        t=row['splits']['training'];v=row['splits']['development']
        def err(s):
            m=s['metrics']['response_error'];lo,hi=m['ci95'];return f"{m['mean']:.5f} [{lo:.5f}, {hi:.5f}]"
        vals=[model_names[row['family']],f"{t['six_group_choice_pct']['mean']:.2f}",f"{v['six_group_choice_pct']['mean']:.2f}",err(t),err(v),
          f"{t['metrics']['response_amplitude_ratio']['mean']:.5f} / {v['metrics']['response_amplitude_ratio']['mean']:.5f}",f"{t['metrics']['complete_error_ratio']['mean']:.3f} / {v['metrics']['complete_error_ratio']['mean']:.3f}"]
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if a.run_root:export(a.run_root)
    result=summarize(json.loads(gzip.decompress((DATA/(STEM+'_records.json.gz')).read_bytes())))
    target=DATA/(STEM+'.json');text=json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n';doc=ROOT/'docs/ICL_Metric_Study.md';before=doc.read_text();start=f'<!-- BEGIN {MARK} -->';end=f'<!-- END {MARK} -->';block=render(result)
    assert before.count(start)==before.count(end)==1
    if a.check:
        assert target.read_text()==text
        assert before.split(start,1)[1].split(end,1)[0].strip()==block
        print('Delay Training/Development: 9 frozen runs, 256 source scenes; data and table consistent.')
    else:
        target.write_text(text);prefix,suffix=before.split(start,1);_,suffix=suffix.split(end,1);doc.write_text(prefix+start+'\n'+block+'\n'+end+suffix);print(block)

if __name__=='__main__':main()
