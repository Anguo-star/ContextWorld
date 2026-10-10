#!/usr/bin/env python3
"""Index existing nine-task evaluations; no inference and no change to frozen scores.

Use --cache-root once to export native response sufficient statistics. Subsequent
runs and --check reproduce the inventory from published JSON/gzip files only.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from render_training_comparison import (TASK_ORDER, TASK_ZH, MODEL_ORDER, MODEL_ZH,
    REGIME_ORDER, REGIME_ZH, ordered_current, display_stats)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/research/data'
STEM = 'nine_task_root_cause_inventory_v1'
QUERY = DATA / (STEM + '_responses.json.gz')
DOC = ROOT / 'docs/ICL_Metric_Study.md'
MARK = 'NINE_TASK_ROOT_CAUSE_INVENTORY'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def key(row):
    return row['task'], row.get('family', row.get('model')), row['regime']

def export_responses(cache, coverage, queries):
    specs = {s['id']: s for s in read(cache / 'models.json')}
    out = {}
    for run in coverage['runs']:
        rid = run['id']; spec = specs[rid]
        directory = Path(spec['result_dir'])
        receipt_path = directory / 'receipt.json'; receipt = read(receipt_path)
        assert sha(receipt_path) == run['receipt_sha256'], rid
        assert receipt['panel_sha256'] == run['panel_sha256'], rid
        assert receipt['no_training'] and receipt['state_hash_before'] == receipt['state_hash_after'], rid
        records = {}
        for name, digest in receipt['entry_hashes'].items():
            path = directory / name
            assert sha(path) == digest, path
            r = read(path)
            assert r['checkpoint_sha256'] == run['checkpoint_sha256'], path
            assert r['horizons'] == [5, 10, 15, 20, 25], path
            n = r['conditions'] * r['candidates']
            records[r['scene_id']] = {
                'query_id': r['scene_id'], 'source_sha256': r['source_sha256'],
                'free': (np.asarray(r['modes']['free']['response_error_sum']) / n).tolist(),
                'real_input': (np.asarray(r['modes']['full']['response_error_sum']) / n).tolist(),
            }
            q = next_q[rid][r['scene_id']]
            np.testing.assert_allclose(np.asarray(r['target_separation_energy']) / n, q['horizon_denominators'], rtol=1e-12)
            np.testing.assert_allclose(np.asarray(r['modes']['free']['prediction_error_sum']) / n, q['horizon_numerators'], rtol=1e-12)
        assert set(records) == {q['query_id'] for q in queries[rid]}, rid
        out[rid] = [records[q['query_id']] for q in queries[rid]]
    QUERY.write_bytes(gzip.compress(json.dumps(out, separators=(',', ':'), allow_nan=False).encode(), mtime=0))

def response_stats(row, queries, responses):
    ids = row['run_ids']; first = queries[ids[0]]
    clusters = sorted({q['bootstrap_cluster'] for q in first})
    cluster_idx = {c: i for i, c in enumerate(clusters)}
    index = np.asarray([cluster_idx[q['bootstrap_cluster']] for q in first])
    values = []; raw = []
    for rid in ids:
        qs = queries[rid]; rs = responses[rid]
        assert [q['query_id'] for q in qs] == [r['query_id'] for r in rs]
        assert [q['query_id'] for q in qs] == [q['query_id'] for q in first]
        assert [q['bootstrap_cluster'] for q in qs] == [q['bootstrap_cluster'] for q in first]
        b = np.asarray([q['horizon_denominators'] for q in qs])
        e = np.asarray([q['horizon_numerators'] for q in qs])
        full_e = np.asarray([q['real_input_horizon_numerators'] for q in qs])
        r = np.asarray([q['free'] for q in rs]); full_r = np.asarray([q['real_input'] for q in rs])
        assert np.isfinite([b,e,r,full_e,full_r]).all() and min(b.min(),e.min(),r.min(),full_r.min()) >= 0
        assert np.min(e-r) >= -1e-6 and np.min(full_e-full_r) >= -1e-6, rid
        np.testing.assert_allclose(r[:,0], full_r[:,0], rtol=1e-6, atol=1e-9)
        denom = b.sum(); bh = b.sum(axis=0)
        assert denom > 0 and np.all(bh > 0)
        # All-time reference for error growth; own-time reference for response precision.
        vals = {'response_ratio': r.sum()/denom,
                'common_bias_ratio': (e-r).sum()/denom,
                'complete_error_ratio': e.sum()/denom,
                'response_by_horizon': (r.sum(axis=0)/bh).tolist(),
                'real_input_response_by_horizon': (full_r.sum(axis=0)/bh).tolist(),
                'complete_by_horizon_shared_reference': (e.sum(axis=0)/(denom/5)).tolist(),
                'real_input_complete_by_horizon_shared_reference': (full_e.sum(axis=0)/(denom/5)).tolist()}
        values.append(vals)
        energies = np.column_stack((r.sum(axis=1), (e-r).sum(axis=1), b.sum(axis=1)))
        grouped = np.zeros((len(clusters), 3)); np.add.at(grouped,index,energies); raw.append(grouped)
    mean = {k: np.mean([v[k] for v in values],axis=0).tolist() for k in values[0]}
    np.testing.assert_allclose(mean['complete_error_ratio'], 1-row['score']/100, rtol=1e-12)
    np.testing.assert_allclose(mean['complete_by_horizon_shared_reference'], row['error_diagnostics']['free']['mean'], rtol=1e-12)
    # Same source draws across all training repetitions, preserving within-source dependence.
    rng = np.random.default_rng(20261009)
    draws = rng.multinomial(len(clusters), np.full(len(clusters),1/len(clusters)), size=1000)
    boot = []
    for raw_run in raw:
        sums = draws @ raw_run
        assert np.all(sums[:,2]>0)
        boot.append(sums[:,:2]/sums[:,2,None])
    ci = np.quantile(np.mean(boot,axis=0),[.025,.975],axis=0)
    mean['response_ci95'] = ci[:,0].tolist(); mean['common_bias_ci95'] = ci[:,1].tolist()
    mean['response_training_sd'] = float(np.std([v['response_ratio'] for v in values],ddof=1)) if len(ids)>1 else None
    return mean

def render(payload):
    lines = []
    for task in TASK_ORDER:
        lines += [f'<details>\n<summary>{TASK_ZH[task]}：模型与训练方案</summary>\n',
          '| 模型 | 方案 | 重复数 | 原协议选择率 (%) ↑ | 多步响应误差 ↓ | 完整误差 ↓ | 历史收益 ↑ | 完整误差：第5 → 25步 / 真实输入第25步 |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
        for r in payload['rows']:
            if r['task'] != task: continue
            score = r['original_protocol']['main']['mean']
            sc = '—' if score is None else f'{score:.2f}'
            if any('h3_tail_delay_reference' in note for note in (r['original_protocol']['notes'] or [])):
                sc += '†'
            d = r.get('multistep')
            if d:
                lo,hi = d['response_ci95']; free=d['complete_by_horizon_shared_reference']; real=d['real_input_complete_by_horizon_shared_reference']
                val=[str(r['training_repetitions']),sc,f"{d['response_ratio']:.3f} [{lo:.3f}, {hi:.3f}]",f"{d['complete_error_ratio']:.3f}",f"{d['history_gain']:.3g}",f'{free[0]:.3f} → {free[-1]:.3f} / {real[-1]:.3f}']
            else: val=['—',sc,'未覆盖','—','—','—']
            lines.append('| '+' | '.join([MODEL_ZH[r['family']],REGIME_ZH[r['regime']]]+val)+' |')
        lines += ['\n</details>\n']
    return '\n'.join(lines).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-root',type=Path);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    assert not (args.cache_root and args.check), '--check never writes'
    coverage=read(DATA/'multistep_prediction_coverage_v2.json')
    history=read(DATA/'icl_measurement_validation_v1.json')
    training=read(DATA/'icl_training_study_v2.json')
    with gzip.open(DATA/'multistep_prediction_coverage_v2_queries.json.gz','rt') as f: queries=json.load(f)
    global next_q
    next_q={rid:{q['query_id']:q for q in qs} for rid,qs in queries.items()}
    if args.cache_root: export_responses(args.cache_root,coverage,queries)
    with gzip.open(QUERY,'rt') as f: responses=json.load(f)
    assert set(responses)==set(queries)
    mult={key(r):r for r in coverage['rows']}; hist={key(r):r for r in history['rows']}
    assert set(mult)==set(hist)
    rows=[]
    for task in TASK_ORDER:
        for source in ordered_current(training['rows'],task):
            k=key(source); metrics,agg=display_stats(source)
            row={'task':task,'family':source['model'],'regime':source['regime'],
                 'original_protocol':{'row_id':source['id'],'main':metrics['main'],'nre':metrics['nre'],'cem':metrics['cem'],
                    'notes':source.get('comparability_notes'), 'status':source['measurement_status']},
                 'multistep_status':'available' if k in mult else 'not_available'}
            if k in mult:
                m=mult[k]; h=hist[k]; d=response_stats(m,queries,responses)
                # Independent history inference differs slightly by numerical batching.
                np.testing.assert_allclose(d['complete_error_ratio'],h['matched_error_ratio'],rtol=1e-6,atol=1e-6)
                d.update(complete_error_ci95=[1-m['ci95'][1]/100,1-m['ci95'][0]/100],
                         history_gain=h['gain'],history_gain_ci95=h['history_bootstrap']['gain_ci95'],
                         history_gain_by_horizon_shared_reference=h['gain_by_horizon'])
                row.update(multistep=d,training_repetitions=m['training_repetitions'],run_ids=m['run_ids'],
                           scenes=m['scenes'],source_groups=m['bootstrap_clusters'],panel_sha256=m['panel_sha256'])
            rows.append(row)
    payload={'schema':'contextworld.nine_task_root_cause_inventory.v1',
      'scope':'Development only; Speed unseen interpolation; original single-query scores and multiaction rollout diagnostics remain different protocols',
      'coverage':{'tasks':9,'cells':len(rows),'multistep_cells':len(mult),'checkpoint_runs':len(coverage['runs']),
                  'unique_scenes':sum(next(r['scenes'] for r in coverage['rows'] if r['task']==t) for t in TASK_ORDER)},
      'protocol':{'physical_steps':[5,10,15,20,25],
        'response':'Centered conditional prediction error / centered target energy; equals paired response NRE for balanced binary conditions on the same panel. Not the original task score.',
        'weighting':'Equal conditions, candidate actions and horizons within scene; ratio of summed energies within checkpoint; then equal mean across training repetitions.',
        'reference':'Task-specific native latent condition mean. Zero conditional response has response ratio 1; perfect conditional response 0. Complete error includes common bias.',
        'history_gain':'(wrong-history error - matched-history error) / same all-horizon target energy; uniform other histories, from existing history validation.',
        'uncertainty':'1000 source-group paired bootstrap draws, seed 20261009; checkpoints fixed. Training SD separate; no cross-encoder physical ranking.',
        'real_input':'Privileged real-observation diagnostic corrects state AND refreshes evidence; not a uniquely identified causal effect.',
        'delay':'11 delays weighted equally in this multi-action diagnostic; differs from original six physical-group macro score.',
        'zero_separation':'No scenes/horizons filtered. Aggregate denominators must be positive; zero aggregate is undefined.'},
      'sources':{f:sha(DATA/f) for f in ['icl_training_study_v2.json','multistep_prediction_coverage_v2.json','multistep_prediction_coverage_v2_queries.json.gz','icl_measurement_validation_v1.json',QUERY.name]},
      'rows':rows}
    outputs={DATA/(STEM+'.json'):json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n'}
    buf=io.StringIO();fields=['task','family','regime','multistep_status','training_repetitions','scenes','original_choice','original_nre','original_cem','response_ratio','response_ci95_low','response_ci95_high','complete_error_ratio','common_bias_ratio','history_gain','history_ci95_low','history_ci95_high','free_first','free_last','real_input_last','response_first','response_last','real_input_response_last']
    writer=csv.DictWriter(buf,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for r in rows:
        x={f:r.get(f) for f in fields[:6]}; d=r.get('multistep')
        x.update(original_choice=r['original_protocol']['main']['mean'],original_nre=r['original_protocol']['nre']['mean'],original_cem=r['original_protocol']['cem']['mean'])
        if d:
            x.update({f:d[f] for f in ['response_ratio','complete_error_ratio','common_bias_ratio','history_gain']})
            x.update(response_ci95_low=d['response_ci95'][0],response_ci95_high=d['response_ci95'][1],history_ci95_low=d['history_gain_ci95'][0],history_ci95_high=d['history_gain_ci95'][1],free_first=d['complete_by_horizon_shared_reference'][0],free_last=d['complete_by_horizon_shared_reference'][-1],real_input_last=d['real_input_complete_by_horizon_shared_reference'][-1],response_first=d['response_by_horizon'][0],response_last=d['response_by_horizon'][-1],real_input_response_last=d['real_input_response_by_horizon'][-1])
        writer.writerow(x)
    outputs[DATA/(STEM+'.csv')]=buf.getvalue()
    doc=DOC.read_text(); start=f'<!-- BEGIN {MARK} -->';end=f'<!-- END {MARK} -->'
    assert doc.count(start)==doc.count(end)==1
    old=doc.split(start)[1].split(end)[0];new='\n'+render(payload)+'\n'
    if args.check:
        for p,s in outputs.items(): assert p.read_text()==s, f'Stale: {p}'
        assert old==new, 'Stale inventory documentation'
    else:
        for p,s in outputs.items():p.write_text(s)
        DOC.write_text(doc.replace(start+old+end,start+new+end))
    print(json.dumps(payload['coverage']), 'verified' if args.check else 'written')

if __name__=='__main__':main()
