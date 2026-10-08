"""Report a complete, explicitly listed pilot on a new future-action distribution.

Uses the established native-latent error and source-group bootstrap functions.
It never pools this panel with the frozen benchmark and never trains a model.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from summarize_icl_measurement_validation import (
    read_history_run, aggregate_history, aggregate_physical,
    _paired_indices, _json_safe, file_sha256,
)

TASK_NAMES = {'action_strength': '推手移动幅度', 'contact_friction': '接触摩擦', 'motion_damping': '运动阻尼'}
FAMILIES = {'lewm': 'LeWM', 'pldm': 'PLDM', 'dinowm': 'DINO-WM'}
SCHEMES = {'original': 'T0', 'scratch': 'T1', 'joint': 'T2', 'frozen': 'T3'}
MARKER = 'VISIBLE_FUTURE_PILOT'

def object_readout(payload: dict) -> dict:
    result = {}
    for name, columns in [('agent', [0, 1]), ('block', [2, 3, 4, 5])]:
        result[name] = {'units': 'px' if name == 'agent' else 'px-equivalent'}
        for key, output_key in [('oracle_true_latent','calibration_rmse'), ('predicted_latent_matched','predicted_rmse'), ('predicted_latent_mismatched','wrong_rmse')]:
            mse = [np.square(np.asarray(q[key]['rmse_by_dimension'])[columns]).mean() for q in payload['query_metrics']]
            result[name][output_key] = float(np.sqrt(np.mean(mse)))
    return result

def summarize(root: Path, panels: Path, reps: int = 1000) -> dict:
    specs = json.loads((root / 'models.json').read_text())
    validation_path = root/'visible_validation.json'
    validation = json.loads(validation_path.read_text())
    if not validation['passed'] or set(validation['tasks']) != set(TASK_NAMES):
        raise ValueError('All three panels must pass data validation before publication')
    data_rows = []
    for task, row in validation['tasks'].items():
        if not row['passed']:raise ValueError(f'Data validation failed: {task}')
        data_rows.append(dict(task=task,source_count=row['checks']['coverage']['new_count'],
            unique_candidates_range=row['checks']['candidate_count_range'],
            geometry_failures=row['checks']['full_shape_unsafe_candidate_slots'],
            exact_aliases=row['local_exact_pixel_aliases']['qualifying_alias_pair_count'],
            gaps={version:{obj:float(np.mean([h['objects'][obj]['mean_world_gap'] for h in row[version+'_conditional_signal']['per_horizon']])) for obj in ('agent','block')} for version in ('legacy','new')}))
    rows = []
    for spec in specs:
        run = read_history_run(spec, root/'results'/spec['id'], panels/spec['task'], root/'absent_bootstrap_map.json')
        if len(run['queries']) != run['expected_queries'] or run['receipt_scenes'] != run['expected_queries']:
            raise ValueError(f"Incomplete query coverage: {spec['id']}")
        groups = sorted({r['source_group'] for r in run['queries']})
        indices = _paired_indices(groups, reps, 20261008)
        physical_path = root/'physical'/(spec['id']+'.json')
        physical = json.loads(physical_path.read_text())
        ids = {q['scene_id'] for q in run['queries']}
        if {str(q['scene_id']) for q in physical['query_metrics']} != ids:
            raise ValueError(f"Physical query coverage mismatch: {spec['id']}")
        groups_by_scene = {q['scene_id']: q['source_group'] for q in run['queries']}
        rows.append(dict(id=spec['id'],task=spec['task'],family=spec['family'],regime=spec['regime'],
            training_seed=spec['training_seed'],checkpoint_sha256=spec['checkpoint_sha256'],
            panel_sha256=run['panel_sha256'],receipt_sha256=run['receipt_sha256'],
            physical_result_sha256=file_sha256(physical_path),
            object_readout=object_readout(physical),
            history=aggregate_history([run],indices),
            physical=aggregate_physical([physical],groups_by_scene,indices)))
    rows.sort(key=lambda x: (list(TASK_NAMES).index(x['task']),x['family'],SCHEMES[x['regime']]))
    comparisons=[]
    for task in TASK_NAMES:
        by_regime={r['regime']:r for r in rows if r['task']==task and r['family']=='dinowm'}
        if not {'original','scratch'} <= set(by_regime):continue
        old,new=by_regime['original']['history'],by_regime['scratch']['history']
        same_denominator=np.isclose(old['denominator_B_total_mean_over_runs'],new['denominator_B_total_mean_over_runs'],rtol=1e-10,atol=1e-12)
        if not same_denominator:raise ValueError('DINO fixed-target error denominator changed')
        comparisons.append(dict(task=task,family='dinowm',comparison='T1 versus T0 on this new panel',
            matched_native_error_decrease_percent=100*(1-new['matched_error_ratio']/old['matched_error_ratio']),
            correct_vs_wrong_history_error_decrease_percent=100*new['gain']/new['wrong_error_ratio'],
            equal_native_target_denominator=True))
    return _json_safe(dict(schema='contextworld.visible_future_pilot.v1',
        model_world_weights_updated=False,evaluation_split='development',
        scope='Selected existing checkpoints; a new future-action distribution, not a replacement of frozen benchmark scores.',
        models_sha256=file_sha256(root/'models.json'),protocol=json.loads((root/'protocol.json').read_text()),
        weighting='Equal conditions, unique candidates and horizons within each scene; ratio of scene-mean error sums; source-group bootstrap.',
        readout_scope='Auxiliary Development crossfit only; true-latent calibration and predicted-latent readout errors remain separate.',
        expected_units=len(specs),completed_units=len(rows),data_validation_sha256=file_sha256(validation_path),data_validation=data_rows,comparisons=comparisons,rows=rows))

def render(payload: dict) -> str:
    lines=['| 任务 | 模型 | 方案 | 多步预测分 ↑ [95% CI] | 正确历史误差收益 ↑ | 真实 latent 校准 RMSE：推手 / 方块 ↓ | 预测 latent 读出 RMSE：推手 / 方块 ↓ |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in payload['rows']:
        h,p=r['history'],r['physical']; ci=h['bootstrap']['score_ci95']; o=r['object_readout']
        lines.append(f"| {TASK_NAMES[r['task']]} | {FAMILIES[r['family']]} | {SCHEMES[r['regime']]} | {h['score']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] | {h['gain']:.3f} | {o['agent']['calibration_rmse']:.2f} / {o['block']['calibration_rmse']:.2f} | {o['agent']['predicted_rmse']:.2f} / {o['block']['predicted_rmse']:.2f} |")
    return '\n'.join(lines)+'\n'

def render_data(payload: dict) -> str:
    lines=['| 任务 | 来源数 | 唯一候选数 / 来源 | 越界候选数 | 推手条件距离：旧 → 新 | 方块条件距离：旧 → 新 |',
           '|---|---:|---:|---:|---:|---:|']
    for r in sorted(payload['data_validation'],key=lambda x:list(TASK_NAMES).index(x['task'])):
        g=r['gaps']; low,high=r['unique_candidates_range']
        lines.append(f"| {TASK_NAMES[r['task']]} | {r['source_count']} | {low}–{high} | {r['geometry_failures']} | {g['legacy']['agent']:.2f} → {g['new']['agent']:.2f} | {g['legacy']['block']:.2f} → {g['new']['block']:.2f} |")
    return '\n'.join(lines)+'\n'

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--panels',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--report',type=Path)
    p.add_argument('--check',action='store_true');p.add_argument('--bootstrap-reps',type=int,default=1000);a=p.parse_args()
    output=a.output.with_suffix('.json')
    if a.check:
        payload=json.loads(output.read_text())
    else:
        if a.root is None or a.panels is None:p.error('--root and --panels required when generating')
        payload=summarize(a.root,a.panels,a.bootstrap_reps);output.parent.mkdir(parents=True,exist_ok=True)
        (output.parent/'icl_visible_future_validation_v1.json').write_bytes((a.root/'visible_validation.json').read_bytes())
        output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        flat=[]
        for r in payload['rows']:
            h,v=r['history'],r['physical']
            flat.append({k:r[k] for k in ('id','task','family','regime','training_seed')} | {
                'source_scenes':h['n_queries'],'source_groups':h['n_source_groups'],'score':h['score'],
                'score_ci95_low':h['bootstrap']['score_ci95'][0],'score_ci95_high':h['bootstrap']['score_ci95'][1],
                'history_gain':h['gain'],'true_latent_calibration_rmse':v['oracle_readout_rmse'],
                'predicted_latent_readout_rmse':v['predicted_rmse']} | {f'{obj}_{metric}':r['object_readout'][obj][metric] for obj in ('agent','block') for metric in ('calibration_rmse','predicted_rmse','wrong_rmse')})
        with a.output.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");writer.writeheader();writer.writerows(flat)
    table=render(payload)
    if a.report:
        text=a.report.read_text()
        for marker,content in [(MARKER,table),('VISIBLE_FUTURE_DATA',render_data(payload))]:
            begin=f'<!-- BEGIN {marker} -->';end=f'<!-- END {marker} -->'
            left,rest=text.split(begin);old,right=rest.split(end)
            if a.check:
                if old != '\n'+content:raise SystemExit(f'Generated {marker} table is stale')
            else:text=left+begin+'\n'+content+end+right
        if not a.check:a.report.write_text(text)
    print(f"{payload['completed_units']}/{payload['expected_units']} checkpoint-task units")
    print(table)

if __name__=='__main__':main()
