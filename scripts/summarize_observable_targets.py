#!/usr/bin/env python3
"""Export the fixed observable-target diagnostic and generate its report table."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

TASK_NAMES = {'action_strength':'推手移动幅度','contact_friction':'接触摩擦','motion_damping':'运动阻尼'}
BRANCHES = {'calibration':'oracle_true_latent','matched':'predicted_latent_matched','mismatched':'predicted_latent_mismatched'}

def export(root: Path, prefix: Path) -> str:
    specs=json.loads((root/'models.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[]; summaries=[]; details=[]; keyed={}
    for spec in specs:
        data=json.loads((root/'results'/(spec['id']+'.json')).read_text())
        parity=data.get('reference_check',data.get('full_aggregate_parity'))
        if not parity or not parity['passed_atol_1e-8']:
            raise ValueError(f"Full-data parity failed: {spec['id']}")
        if 'mse_contribution_decomposition' in data:
            if not data['mse_contribution_decomposition']['all_branches_conserve']:
                raise ValueError(f"MSE decomposition failed: {spec['id']}")
        keyed[spec['id']]=data
        details.append({'model':spec,'result':data})
        summary={k:v for k,v in data.items() if k not in {'query_metrics','query_metrics_by_branch','folds'}}
        if 'summary' in summary['coverage']:
            summary['coverage']={k:v for k,v in summary['coverage'].items() if not k.endswith('_counts')}
        summary['source_files']={k:v for k,v in summary['source_files'].items() if k not in {'feature_archives','panel_archives'}}
        summaries.append({'model':spec,'result':summary})
        categories=data['summaries'] if 'all_rows' in data['summaries'] else {'all_rows':data['summaries']}
        for category,branches in categories.items():
            for obj in data['target']['objects']:
                row={'id':spec['id'],'task':spec['task'],'family':spec['family'],'regime':spec['regime'],'category':category,'object':obj,'units':'px' if obj=='pusher' else data['target']['units']}
                for label,branch in BRANCHES.items():
                    metric=branches[branch]['objects'][obj]
                    row[label+'_rmse']=metric['raw_rmse']
                    row[label+'_ci95']=json.dumps(metric['bootstrap_ci95_raw_rmse'])
                    row[label+'_n_queries']=metric['n_queries_valid']
                rows.append(row)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    detail_path=prefix.with_name(prefix.name+'_details.json.gz')
    with detail_path.open('wb') as stream:
        with gzip.GzipFile(fileobj=stream,mode='wb',filename='',mtime=0) as gz:
            gz.write(json.dumps({'protocol':protocol,'runs':details},ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())
    payload={'protocol':protocol,'completed_runs':len(specs),'tasks':sorted({s['task'] for s in specs}),'all_full_data_parity_passed':True,'runs':summaries,'details':{'file':detail_path.name,'sha256':hashlib.sha256(detail_path.read_bytes()).hexdigest()}}
    prefix.with_suffix('.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with prefix.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['<!-- BEGIN OBSERVABLE_TARGET_DIAGNOSTIC -->',
        '| 任务 | 完整配对画布内覆盖 | 条件差异能量保留：推手 / 方块 | 推手校准 RMSE：全量 → 画布内 | 方块校准 RMSE：全量 → 画布内 |',
        '|---|---:|---:|---:|---:|']
    for task,label in TASK_NAMES.items():
        data=keyed[f'{task}/dinowm/scratch/s3072']
        category='paired_all_centers_in_canvas'
        fraction=data['coverage']['summary']['paired_in_canvas_fraction']
        energy=data['condition_signal_energy']['categories'][category]['fraction_of_all_rows_energy_by_object']
        energy_text=' / '.join('—' if energy[o] is None else f'{100*energy[o]:.1f}%' for o in ['pusher','block'])
        pairs=[]
        for obj in ['pusher','block']:
            a=data['summaries']['all_rows']['oracle_true_latent']['objects'][obj]['raw_rmse']
            b=data['summaries'][category]['oracle_true_latent']['objects'][obj]['raw_rmse']
            pairs.append(f'{a:.2f} → {b:.2f}')
        lines.append(f'| {label} | {100*fraction:.2f}% | {energy_text} | {pairs[0]} | {pairs[1]} |')
    lines.append('<!-- END OBSERVABLE_TARGET_DIAGNOSTIC -->')
    markdown='\n'.join(lines)+'\n'
    prefix.with_suffix('.md').write_text(markdown)
    return markdown

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    p.add_argument('--report',type=Path,help='Replace the existing generated diagnostic block.')
    args=p.parse_args();block=export(args.root,args.output_prefix)
    if args.report:
        text=args.report.read_text();start=text.index('<!-- BEGIN OBSERVABLE_TARGET_DIAGNOSTIC -->');end=text.index('<!-- END OBSERVABLE_TARGET_DIAGNOSTIC -->',start)+len('<!-- END OBSERVABLE_TARGET_DIAGNOSTIC -->')
        args.report.write_text(text[:start]+block.rstrip()+text[end:])
    print(f'Exported results to {args.output_prefix}')
if __name__=='__main__':
    main()
