"""Summarize fixed-tolerance accuracy from exported Development readouts.

These are measurement-validation scores, never a claim of a calibrated
physical leaderboard. No world-model inference or optimization is performed.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from prediction_accuracy import geometry_position_errors, score_errors, clustered_paired_bootstrap_ci

NAMES = {'action_strength':'推手移动幅度','contact_friction':'接触摩擦','motion_damping':'运动阻尼','speed':'速度','door':'门通行规则'}
SCHEMES = {'original':'T0','scratch':'T1','joint':'T2','frozen':'T3'}
FAMILIES = {'lewm':'LeWM','dinowm':'DINO-WM','pldm':'PLDM'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def summarize_scene(truth, matched, calibration, geometry, thresholds):
    if truth.shape != matched.shape or truth.shape != calibration.shape or truth.ndim != 4:
        raise ValueError('Aligned condition/candidate/horizon/component arrays required')
    k = truth.shape[0]
    if k < 2:
        raise ValueError('History comparison requires at least two conditions')
    paired_truth = np.stack([truth[i] for i in range(k) for j in range(k) if i != j])
    wrong = np.stack([matched[j] for i in range(k) for j in range(k) if i != j])
    arrays = {'matched':(truth,matched),'wrong':(paired_truth,wrong),'calibration':(truth,calibration)}
    objects = {}
    for mode,(target,prediction) in arrays.items():
        for obj,errors in geometry_position_errors(target,prediction,geometry).items():
            objects.setdefault(obj,{})[mode] = score_errors(errors,thresholds).to_dict()
    return objects

def aggregate(scenes, obj):
    groups = [s['source_group'] for s in scenes]
    values = {mode:np.array([s['objects'][obj][mode]['score'] for s in scenes]) for mode in ['matched','wrong','calibration']}
    def ci(a,b):
        out = clustered_paired_bootstrap_ci(a,b,groups,n_bootstrap=1000,seed=20261008)
        return [out.lower,out.upper]
    out={}
    for mode in values:
        records=[s['objects'][obj][mode] for s in scenes]
        out[mode] = {name:np.mean([r[name] for r in records],axis=0).tolist() for name in ['score','tolerance_curve','horizon_curve','tolerance_horizon_curve','max_horizon_score','max_horizon_tolerance_curve']}
        out[mode]['raw_rmse']=float(np.sqrt(np.mean([r['raw_rmse']**2 for r in records])))
        out[mode]['score_ci95']=ci(values[mode],np.zeros(len(scenes)))
        curves=np.array([r['tolerance_curve'] for r in records]); horizon=np.array([r['horizon_curve'] for r in records])
        out[mode]['threshold_sensitivity']={'finest_three':float(curves[:,:3].mean()),'all_five':float(curves.mean()),'coarsest_three':float(curves[:,-3:].mean())}
        out[mode]['horizon_ci95']=[ci(horizon[:,i],np.zeros(len(scenes))) for i in range(horizon.shape[1])]
    out['history_gain']=float(np.mean(values['matched']-values['wrong']))
    out['history_gain_ci95']=ci(values['matched'],values['wrong'])
    out['calibration_strictest_tolerance_coverage']=out['calibration']['tolerance_curve'][0]
    out['necessary_calibration_screen_passed']=bool(out['calibration_strictest_tolerance_coverage']>=95.)
    out['predicted_latent_readout_validity_verified']=False
    return out

def build(root):
    protocol=json.loads((root/'protocol.json').read_text())
    manifest=json.loads((root/'readout/run_manifest.json').read_text())
    rows=[]; query_rows=[]; model_specs={}
    for unit in manifest['units']:
        for s in json.loads(Path(unit['models_manifest']).read_text()): model_specs[s['id']]=s
        identifier=unit['id']; task,family,regime,seed=identifier.split('/')
        geometry='pusht' if task in ['action_strength','contact_friction','motion_damping'] else 'tworoom'
        thresholds=protocol['thresholds'][geometry]; scenes=[]
        for file in sorted((root/'readout'/identifier).glob('*.npz')):
            with np.load(file,allow_pickle=False) as d:
                objects=summarize_scene(d['truth'],d['matched'],d['calibration'],geometry,thresholds)
                scenes.append({'scene_id':str(d['scene_id'].item()),'source_group':str(d['source_group'].item()),'fold':int(d['fold'].item()),'objects':objects,'array_sha256':sha(file)})
        expected=json.loads(Path(unit['source_physical_json']).read_text())['query_metrics']
        if len(scenes)!=len(expected) or {s['scene_id'] for s in scenes}!={str(s['scene_id']) for s in expected}:
            raise ValueError(f'Incomplete or duplicate scene coverage: {identifier}')
        expected_groups={str(s['scene_id']):str(s['source_group']) for s in expected}
        if any(expected_groups[s['scene_id']]!=s['source_group'] for s in scenes): raise ValueError('Source groups changed')
        specs=model_specs[identifier]
        row={'id':identifier,'task':task,'family':family,'regime':regime,'training_seed':int(seed[1:]),'checkpoint_sha256':specs['checkpoint_sha256'],
             'suite':unit['suite'],'panel_manifest_sha256':unit['panel_identity']['manifest_sha256'],'scenes':len(scenes),'source_groups':len({s['source_group'] for s in scenes}),
             'thresholds':thresholds,'primary_object':protocol['primary_object'][task],'objects':{obj:aggregate(scenes,obj) for obj in scenes[0]['objects']}}
        rows.append(row); query_rows.append({'id':identifier,'scenes':scenes})
    comparisons=[]
    for task in NAMES:
        for family in FAMILIES:
            units=[r for r in rows if r['task']==task and r['family']==family]; old=next((r for r in units if r['regime']=='original'),None)
            if old is None:continue
            qa={q['scene_id']:q for run in query_rows if run['id']==old['id'] for q in run['scenes']}
            for new in units:
                if new is old:continue
                if new['panel_manifest_sha256']!=old['panel_manifest_sha256']:raise ValueError('Cannot compare different panels')
                qb={q['scene_id']:q for run in query_rows if run['id']==new['id'] for q in run['scenes']}
                if qa.keys()!=qb.keys(): raise ValueError('Unpaired training comparison')
                ids=sorted(qa)
                if any(qa[i]['source_group']!=qb[i]['source_group'] for i in ids): raise ValueError('Unpaired source groups')
                objects={}
                for obj in old['objects']:
                    a=[qb[i]['objects'][obj]['matched']['score'] for i in ids];b=[qa[i]['objects'][obj]['matched']['score'] for i in ids]
                    delta=clustered_paired_bootstrap_ci(a,b,[qa[i]['source_group'] for i in ids],n_bootstrap=1000,seed=20261008).to_dict()
                    objects[obj]=delta
                comparisons.append({'task':task,'family':family,'before':old['id'],'after':new['id'],'objects':objects})
    return {'schema':'contextworld.prediction_accuracy_validation.v1','protocol':protocol,'coverage':{'checkpoint_task_units':len(rows),'checkpoint_scenes':sum(r['scenes'] for r in rows),'tasks':list(NAMES)},
            'interpretation':'Auxiliary physical readout accuracy; calibration and prediction-manifold validity required before physical model ranking.','rows':rows,'training_comparisons':comparisons},query_rows

def render(data):
    lines=['| 任务 | 模型 | 方案 | 真值读出校准分 ↑ | 预测读出分 ↑ [95% CI] | 正确历史收益 ↑ [95% CI] | 校准误差小于最严容差的比例 |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in sorted(data['rows'],key=lambda r:(list(NAMES).index(r['task']),r['family'],SCHEMES[r['regime']])):
        obj=r['objects'][r['primary_object']];m=obj['matched'];ci=m['score_ci95'];g=obj['history_gain_ci95']
        lines.append(f"| {NAMES[r['task']]} | {FAMILIES[r['family']]} | {SCHEMES[r['regime']]} | {obj['calibration']['score']:.2f} | {m['score']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {obj['history_gain']:+.2f} [{g[0]:+.2f}, {g[1]:+.2f}] | {obj['calibration_strictest_tolerance_coverage']:.2f}% |")
    return '\n'.join(lines)+'\n'

def render_controls(data):
    lines = ['| 任务 | 场景数 | 真值预测 | 共同未来均值 | 错误规律未来 | 延迟一个观测间隔 |', '|---|---:|---:|---:|---:|---:|']
    for task in NAMES:
        metrics = [r for r in data['threshold_sensitivity'] if r['task']==task and r['task_relevant_object'] and r['threshold_subset']=='all5']
        def value(control):
            rows=[r for r in metrics if r['control']==control]
            if not rows:return '—'
            if len(rows)!=5:raise ValueError(f'Incomplete five-horizon control: {task}/{control}')
            return f"{np.mean([r['accuracy_percent'] for r in rows]):.2f}"
        count=data['created_from']['tasks'][task]['scene_count']
        lines.append(f"| {NAMES[task]} | {count} | {value('exact_truth')} | {value('condition_mean_common_privileged')} | {value('wrong_condition_true_future')} | {value('temporal_lag_prepend_current')} |")
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path);p.add_argument('--controls',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    result=args.output.with_suffix('.json')
    if args.check:data=json.loads(result.read_text())
    else:
        data,queries=build(args.root);result.parent.mkdir(parents=True,exist_ok=True)
        result.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with gzip.open(args.output.parent/(args.output.name+'_queries.json.gz'),'wt',encoding='utf-8') as f:json.dump(queries,f,ensure_ascii=False,allow_nan=False)
        records=[]
        for row in data['rows']:
            for obj,v in row['objects'].items():
                records.append({k:row[k] for k in ['id','task','family','regime','suite','scenes','source_groups']}|{'object':obj,'calibration_score':v['calibration']['score'],'matched_score':v['matched']['score'],'wrong_score':v['wrong']['score'],'history_gain':v['history_gain'],'calibration_strictest_coverage':v['calibration_strictest_tolerance_coverage'],'necessary_calibration_screen_passed':v['necessary_calibration_screen_passed'],'matched_rmse':v['matched']['raw_rmse'],'calibration_rmse':v['calibration']['raw_rmse']})
        with args.output.with_suffix('.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    table=render(data)
    if args.report:
        text=args.report.read_text()
        sections={'PREDICTION_ACCURACY_VALIDATION':table}
        if args.controls:sections['PREDICTION_ACCURACY_CONTROLS']=render_controls(json.loads(args.controls.read_text()))
        for marker,content in sections.items():
            begin=f'<!-- BEGIN {marker} -->';end=f'<!-- END {marker} -->'
            left,rest=text.split(begin);old,right=rest.split(end)
            if args.check:
                if old!='\n'+content:raise SystemExit(f'Generated {marker} table is stale')
            else:text=left+begin+'\n'+content+end+right
        if not args.check:args.report.write_text(text)
    print(table)
if __name__=='__main__':main()
