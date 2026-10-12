#!/usr/bin/env python3
"""Recompute the fixed-representation PLDM objective contrast from query records."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('contact_friction', 'motion_damping')
ARMS = ('native_visual', 'response_weighted')
STEPS = ('0', '64', '256')
BEGIN = '<!-- BEGIN PLDM_FIXED_REPRESENTATION_INTERVENTION -->'
END = '<!-- END PLDM_FIXED_REPRESENTATION_INTERVENTION -->'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pooled(rows):
    sums = {k: sum(r[k] for r in rows) for k in ('B', 'E', 'R', 'C', 'W')}
    assert sums['B'] > 0
    np.testing.assert_allclose(sums['E'], sums['R'] + sums['C'], rtol=1e-10)
    return dict(NRE=sums['R']/sums['B'], E_over_B=sums['E']/sums['B'],
                C_over_B=sums['C']/sums['B'], G=(sums['W']-sums['E'])/sums['B'])


def paired_delta(control, weighted):
    assert [(r['query_id'], r['source_group']) for r in control] == [(r['query_id'], r['source_group']) for r in weighted]
    np.testing.assert_array_equal([r['B'] for r in control], [r['B'] for r in weighted])
    labels = np.asarray([r['source_group'] for r in control])
    keys, inverse = np.unique(labels, return_inverse=True)
    stats = np.zeros((len(keys), 4), dtype=np.float64)
    for i, (a,b) in enumerate(zip(control, weighted)):
        stats[inverse[i]] += [a['B'], b['R']-a['R'], b['E']-a['E'], (b['W']-b['E'])-(a['W']-a['E'])]
    draws = np.random.default_rng(20261012).integers(len(keys), size=(2000, len(keys)))
    samples = stats[draws].sum(axis=1)
    ratios = samples[:, 1:]/samples[:, :1]
    totals = stats.sum(axis=0)
    return dict(source_groups=len(keys), bootstrap_seed=20261012, bootstrap_resamples=2000,
                weighted_minus_control={key: {'estimate':float(totals[i+1]/totals[0]),
                    'ci95':np.quantile(ratios[:, i], [.025,.975]).tolist()}
                    for i,key in enumerate(('NRE','E_over_B','G'))})


def build(source, archived):
    protocol_bytes = (source/'protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol['schema'] == 'contextworld.pldm_fixed_representation_intervention.v1'
    assert protocol['design']['steps'] == protocol['design']['primary_checkpoint'] == 256
    rows, effects, identities, source_bytes = [], {}, {}, {}
    for task in TASKS:
        path = source/task/('result.json.gz' if archived else 'result.json')
        raw = gzip.decompress(path.read_bytes()) if archived else path.read_bytes()
        result = json.loads(raw); source_bytes[task] = raw
        assert result['completion_state'] == 'complete' and result['task'] == task
        assert result['protocol_sha256'] == digest(protocol_bytes)
        assert result['binding'] == protocol['tasks'][task]
        assert result['code_sha256'] == protocol['code_sha256']
        assert result['restored_model_state_hash'] == result['source_model_state_hash']
        assert not result['public_test_accessed'] and set(result['arms']) == set(ARMS)
        assert len(result['batch_query_ids']) == 256
        assert all(len(batch) == 8 for batch in result['batch_query_ids'])
        visits = Counter(query for batch in result['batch_query_ids'] for query in batch)
        assert len(visits) == 512 and set(visits.values()) == {4}
        identities[task] = {'checkpoint_sha256':result['binding']['checkpoint_sha256'],
            'result_sha256':digest(raw), 'batch_order_sha256':result['batch_order_sha256'],
            'fixed_module_hashes':result['fixed_module_hashes'], 'calibration':result['calibration'],
            'first_update_norm':{arm:result['arms'][arm]['first_update_norm'] for arm in ARMS}}
        baseline = result['arms']['native_visual']['snapshots']['0']
        assert baseline == result['arms']['response_weighted']['snapshots']['0']
        for arm in ARMS:
            record = result['arms'][arm]
            assert record['frozen_state_hash_after'] == result['frozen_state_hash']
            assert len(record['training_losses']) == 256
            assert set(record['snapshots']) == set(STEPS)
            for step in STEPS:
                for split, count in (('training',512),('development',256)):
                    entry = record['snapshots'][step][split]; queries = entry['per_query']
                    assert len(queries) == count and len({q['query_id'] for q in queries}) == count
                    for a,b in zip(baseline[split]['per_query'],queries):
                        assert (a['query_id'],a['source_group'],a['B']) == (b['query_id'],b['source_group'],b['B'])
                        assert np.isfinite([b[k] for k in ('B','E','R','C','W')]).all()
                        assert all(b[k]>=0 for k in ('B','E','R','C','W'))
                        np.testing.assert_allclose(b['E'],b['C']+b['R'],rtol=1e-10,atol=1e-10)
                    values = pooled(queries)
                    for key in ('NRE','E_over_B','G'):
                        np.testing.assert_allclose(values[key],entry['summary'][key],rtol=1e-10,atol=1e-10)
                    rows.append(dict(task=task,arm=arm,step=int(step),split=split,
                        queries=count,source_groups=len({q['source_group'] for q in queries}),**values))
        effects[task] = {split:paired_delta(result['arms'][ARMS[0]]['snapshots']['256'][split]['per_query'],
                                            result['arms'][ARMS[1]]['snapshots']['256'][split]['per_query'])
                         for split in ('training','development')}
    return dict(schema='contextworld.pldm_fixed_representation_intervention_summary.v1',
        protocol_sha256=digest(protocol_bytes),design=protocol['design'],rows=rows,
        effects=effects,identities=identities,
        scope='Fixed T3 representation; native query endpoint at physical step 5; Training fitting, Development evaluation; no Test or original-environment CEM.'), source_bytes, protocol_bytes


def csv_text(rows):
    out=io.StringIO(newline=''); writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return out.getvalue()


def table(summary):
    names={'contact_friction':'接触摩擦','motion_damping':'运动阻尼'}
    rows=summary['rows']
    def find(task,arm,step,split):return next(x for x in rows if (x['task'],x['arm'],x['step'],x['split'])==(task,arm,step,split))
    lines=['| 任务 | 设置 | Training 响应 NRE ↓ | Dev 响应 NRE ↓ | Dev 完整 E/B ↓ | Dev 历史收益 G ↑ |','|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for arm,step,label in [('native_visual',0,'初始 T3'),('native_visual',256,'视觉 MSE 对照'),('response_weighted',256,'响应加权')]:
            train=find(task,arm,step,'training');dev=find(task,arm,step,'development')
            lines.append(f"| {names[task]} | {label} | {train['NRE']:.3f} | {dev['NRE']:.3f} | {dev['E_over_B']:.3f} | {dev['G']:.3f} |")
    lines+=['','| 任务 | 加权 − 对照：响应 NRE [95% 区间] | 加权 − 对照：完整 E/B [95% 区间] |','|---|---:|---:|']
    for task in TASKS:
        effect=summary['effects'][task]['development']['weighted_minus_control']
        values=[]
        for key in ('NRE','E_over_B'):
            x=effect[key];values.append(f"{x['estimate']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]")
        lines.append(f"| {names[task]} | {' | '.join(values)} |")
    return '\n'.join(lines)+'\n'


def update_doc(path, content, check):
    old=path.read_text();assert old.count(BEGIN)==old.count(END)==1
    before,rest=old.split(BEGIN);_,after=rest.split(END)
    new=before+BEGIN+'\n'+content+END+after
    if check:assert old==new,'Generated document table differs'
    else:path.write_text(new)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'docs/research/data/pldm_fixed_representation_intervention_v1')
    p.add_argument('--check',action='store_true')
    p.add_argument('--render-doc',action='store_true')
    p.add_argument('--doc',type=Path,default=ROOT/'docs/ICL_Metric_Study.md')
    args=p.parse_args()
    if not args.check and args.source is None:p.error('--source is required when archiving')
    summary,raw,protocol=build(args.output_dir/'sources' if args.check else args.source,args.check)
    texts={'summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n',
           'summary.csv':csv_text(summary['rows']),'results.md':table(summary)}
    if args.check:
        for name,content in texts.items():assert (args.output_dir/name).read_text()==content,name
    else:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'sources').mkdir(exist_ok=True)
        (args.output_dir/'sources/protocol.json').write_bytes(protocol)
        for task,data in raw.items():
            path=args.output_dir/'sources'/task;path.mkdir(exist_ok=True)
            (path/'result.json.gz').write_bytes(gzip.compress(data,mtime=0))
        for name,content in texts.items():(args.output_dir/name).write_text(content)
    if args.render_doc:update_doc(args.doc,texts['results.md'],args.check)
    print(texts['results.md'],end='')


if __name__=='__main__':main()
