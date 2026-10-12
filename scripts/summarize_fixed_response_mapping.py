#!/usr/bin/env python3
"""Recompute fixed-response mapping diagnostics from archived predictions."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TASKS=('contact_friction','motion_damping')
MODES=('native','linear_cv','rbf_cv','rbf_interpolation','current_control')
BEGIN='<!-- BEGIN FIXED_RESPONSE_MAPPING -->'
END='<!-- END FIXED_RESPONSE_MAPPING -->'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(target,prediction,pairs):
    y=np.asarray(target,np.float64)[pairs]
    p=np.asarray(prediction,np.float64)[pairs]
    assert y.shape==p.shape and y.shape[1]==2
    assert np.isfinite(y).all() and np.isfinite(p).all()
    np.testing.assert_allclose(y.mean(1),0,atol=1e-12)
    centered=p-p.mean(1,keepdims=True)
    return {'B':np.square(y).sum((1,2)),
            'R':np.square(centered-y).sum((1,2)),
            'S':np.square(centered[:,::-1]-y).sum((1,2)),
            'E':np.square(p-y).sum((1,2))}


def aggregate(stats,groups):
    keys,inv=np.unique(groups,return_inverse=True)
    sums={k:np.bincount(inv,weights=v,minlength=len(keys)) for k,v in stats.items()}
    assert sums['B'].sum()>0
    draw=np.random.default_rng(20261012).integers(len(keys),size=(2000,len(keys)))
    b=sums['B'][draw].sum(1);assert np.all(b>0)
    boot=sums['R'][draw].sum(1)/b
    total={k:float(v.sum()) for k,v in sums.items()}
    return dict(queries=len(groups),source_groups=len(keys),NRE=total['R']/total['B'],
                NRE_ci95=np.quantile(boot,[.025,.975]).tolist(),
                swapped_NRE=total['S']/total['B'],centered_target_error_ratio=total['E']/total['B'])


def build(source):
    protocol=json.loads((source/'protocol.json').read_text())
    assert protocol['schema']=='contextworld.fixed_response_mapping.v1'
    rows=[];identities={}
    for task in TASKS:
        directory=source/task
        result=json.loads((directory/'result.json').read_text())
        assert result['completion_state']=='complete' and result['task']==task
        assert result['schema']==protocol['schema'] and result['design']==protocol['design']
        assert result['no_test_read'] and result['no_checkpoint_update']
        assert result['cross_split_exact_input_overlap_count']==0
        assert result['protocol_sha256']==sha(source/'protocol.json')
        assert result['input_protocol_sha256']==protocol['input_protocol_sha256']
        assert result['array_sha256']==sha(directory/'predictions.npz')
        identities[task]={'result_sha256':sha(directory/'result.json'),
                         'array_sha256':result['array_sha256'],'selected':result['selected']}
        with np.load(directory/'predictions.npz',allow_pickle=False) as d:
            assert set(d['training_source_groups'].astype(str)).isdisjoint(d['development_source_groups'].astype(str))
            for split,count in [('training',512),('development',256)]:
                pairs=d[f'{split}_pair_rows'];y=d[f'{split}_target']
                groups=d[f'{split}_source_groups'].astype(str)
                ids=d[f'{split}_query_ids'].astype(str)
                assert pairs.shape==(count,2) and len(set(ids))==count and len(groups)==count
                np.testing.assert_array_equal(np.sort(pairs.ravel()),np.arange(count*2))
                for mode in MODES:
                    prediction=d[f'{split}_{mode}_prediction']
                    stats=statistics(y,prediction,pairs)
                    value=aggregate(stats,groups)
                    reported=result['metrics'][split][mode]
                    np.testing.assert_allclose(value['NRE'],reported['nre'],rtol=1e-10,atol=1e-12)
                    np.testing.assert_allclose(value['swapped_NRE'],reported['swapped_nre'],rtol=1e-10,atol=1e-12)
                    if mode!='native':
                        np.testing.assert_allclose(value['centered_target_error_ratio'],reported['row_centered_target_error'],rtol=1e-10,atol=1e-12)
                    if mode=='native':value['centered_target_error_ratio']=None
                    if mode=='current_control':np.testing.assert_allclose(value['NRE'],1,rtol=0,atol=1e-10)
                    rows.append(dict(task=task,mode=mode,split=split,**value))
    return dict(schema='contextworld.fixed_response_mapping_summary.v1',
                protocol_sha256=sha(source/'protocol.json'),design=protocol['design'],
                identities=identities,rows=rows,
                scope='Training-supervised centered future targets; single-trajectory history/actions at inference; fixed PLDM T3 representation; not complete future accuracy.' )


def table(summary):
    labels={'native':'原生 T3 预测器','linear_cv':'线性映射（Training 交叉验证）',
            'rbf_cv':'非线性映射（Training 交叉验证）','rbf_interpolation':'非线性近插值对照',
            'current_control':'仅当前画面与动作'}
    names={'contact_friction':'接触摩擦','motion_damping':'运动阻尼'}
    lines=['| 任务 | 预测方式 | Training 响应 NRE ↓ | Dev 响应 NRE ↓ [95% 区间] |',
           '|---|---|---:|---:|']
    for task in TASKS:
        for mode in MODES:
            t=next(x for x in summary['rows'] if (x['task'],x['mode'],x['split'])==(task,mode,'training'))
            d=next(x for x in summary['rows'] if (x['task'],x['mode'],x['split'])==(task,mode,'development'))
            value='<0.001' if t['NRE']<.0005 else f"{t['NRE']:.3f}"
            lo,hi=d['NRE_ci95']
            lines.append(f"| {names[task]} | {labels[mode]} | {value} | {d['NRE']:.3f} [{lo:.3f}, {hi:.3f}] |")
    return '\n'.join(lines)+'\n'


def csv_text(rows):
    output=io.StringIO(newline='')
    w=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return output.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'docs/research/data/fixed_response_mapping_v1')
    p.add_argument('--check',action='store_true');p.add_argument('--render-doc',action='store_true')
    p.add_argument('--doc',type=Path,default=ROOT/'docs/ICL_Metric_Study.md')
    args=p.parse_args()
    if not args.check and args.source is None:p.error('--source required')
    source=args.output_dir/'sources' if args.check else args.source
    summary=build(source)
    contents={'summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n',
              'summary.csv':csv_text(summary['rows']),'results.md':table(summary)}
    if args.check:
        for name,value in contents.items():assert (args.output_dir/name).read_text()==value,name
    else:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        dest=args.output_dir/'sources';dest.mkdir(exist_ok=True)
        shutil.copy2(source/'protocol.json',dest/'protocol.json')
        for task in TASKS:
            (dest/task).mkdir(exist_ok=True)
            for name in ('result.json','predictions.npz'):
                shutil.copy2(source/task/name,dest/task/name)
        for name,value in contents.items():(args.output_dir/name).write_text(value)
    if args.render_doc:
        old=args.doc.read_text();assert old.count(BEGIN)==old.count(END)==1
        before,rest=old.split(BEGIN);_,after=rest.split(END)
        new=before+BEGIN+'\n'+contents['results.md']+END+after
        if args.check:assert old==new,'Document table differs'
        else:args.doc.write_text(new)
    print(contents['results.md'],end='')


if __name__=='__main__':main()
