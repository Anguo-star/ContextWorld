"""Score complete free-running trajectories from frozen prediction receipts.

Score = 100 * (1 - complete squared error / condition-mean reference error).
Average conditions, candidates and the five fixed horizons within each source
query before summing energies; never average per-query error ratios. Training
replicates contribute equally after each receives its own normalized score.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

HORIZONS = (5, 10, 15, 20, 25)
BOOTSTRAP_SAMPLES = 4000
BOOTSTRAP_SEED = 20260930
SCHEMA = 'contextworld.multistep_complete_prediction.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(numerator, denominator):
    n, d = float(np.sum(numerator)), float(np.sum(denominator))
    if not np.isfinite([n,d]).all() or n < 0 or d <= 0:
        raise ValueError('Score requires finite nonnegative error and positive target separation')
    return 100.0 * (1.0 - n / d)


def query_energies(record):
    if tuple(record['horizons']) != HORIZONS:
        raise ValueError('Unexpected horizon identity; use fixed 5,10,15,20,25 raw steps')
    n=np.asarray(record['modes']['free']['prediction_error_sum'], dtype=np.float64)
    d=np.asarray(record['target_separation_energy'], dtype=np.float64)
    size=int(record['conditions'])*int(record['candidates'])
    if n.shape!=(5,) or d.shape!=(5,) or size<=0 or np.any(n<0) or np.any(d<0) or not np.isfinite([n,d]).all():
        raise ValueError('Invalid per-query energy record')
    result = dict(query_id=record['scene_id'], numerator=float(n.mean()/size), denominator=float(d.mean()/size),
                horizon_numerators=(n/size).tolist(), horizon_denominators=(d/size).tolist(),
                conditions=int(record['conditions']), candidates=int(record['candidates']),
                zero_separation_horizons=int(np.sum(d==0)))
    if 'full' in record['modes']:
        full=np.asarray(record['modes']['full']['prediction_error_sum'],dtype=np.float64)
        if full.shape!=(5,) or np.any(full<0) or not np.isfinite(full).all():
            raise ValueError('Invalid real-observation diagnostic energy')
        if not np.isclose(full[0],n[0],rtol=1e-7,atol=1e-10):
            raise ValueError('Free and real-observation first steps must agree')
        result['real_input_horizon_numerators']=(full/size).tolist()
    return result


def read_run(spec, result_dir, panel_dir):
    receipt_path=result_dir/'receipt.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt['state_hash_before']!=receipt['state_hash_after'] or not receipt['no_training']:
        raise ValueError('Weights changed during evaluation')
    manifest_path=panel_dir/'manifest.json'
    if receipt['panel_sha256']!=sha(manifest_path):
        raise ValueError('Panel identity mismatch')
    manifest=json.loads(manifest_path.read_text())
    entries={x.get('scene_id',x.get('pair_id',x.get('query_id'))):x for x in manifest['scenes']}
    queries=[]
    for name,expected in receipt['entry_hashes'].items():
        path=result_dir/name
        if sha(path)!=expected:
            raise ValueError(f'Result hash mismatch: {name}')
        row=json.loads(path.read_text());entry=entries[row['scene_id']]
        if row['checkpoint_sha256']!=spec['checkpoint_sha256'] or row['source_sha256']!=entry['sha256']:
            raise ValueError('Checkpoint or query identity mismatch')
        if sha(path.with_suffix('.npz'))!=row['array_sha256']:
            raise ValueError('Prediction array hash mismatch')
        query=query_energies(row)
        query['bootstrap_cluster']=entry.get('bootstrap_cluster',row['scene_id'])
        queries.append(query)
    queries.sort(key=lambda q:q['query_id'])
    if len(queries)!=len(entries) or len({q['query_id'] for q in queries})!=len(entries):
        raise ValueError('Incomplete or duplicated query coverage')
    value=score([q['numerator'] for q in queries],[q['denominator'] for q in queries])
    return dict(id=spec['id'],task=spec['task'],family=spec['family'],regime=spec['regime'],
                training_seed=spec['training_seed'],checkpoint_sha256=spec['checkpoint_sha256'],
                panel_sha256=receipt['panel_sha256'],receipt_sha256=sha(receipt_path),
                data_version=manifest.get('data_version','unversioned'),
                score=value,queries=queries,weights_unchanged=True)


def aggregate_runs(runs):
    if not runs:
        raise ValueError('No runs')
    identity={(r['task'],r['family'],r['regime'],r['panel_sha256']) for r in runs}
    if len(identity)!=1 or len({r['checkpoint_sha256'] for r in runs})!=len(runs):
        raise ValueError('Incompatible or duplicated training repetitions')
    ids=[q['query_id'] for q in runs[0]['queries']]
    if any([q['query_id'] for q in r['queries']]!=ids for r in runs):
        raise ValueError('Training repetitions must share exactly the same queries')
    cluster_ids=[q.get('bootstrap_cluster',q['query_id']) for q in runs[0]['queries']]
    if any([q.get('bootstrap_cluster',q['query_id']) for q in r['queries']]!=cluster_ids for r in runs):
        raise ValueError('Training repetitions must share source clusters')
    groups=[[i for i,c in enumerate(cluster_ids) if c==key] for key in sorted(set(cluster_ids))]
    def clustered(values):
        return np.stack([values[index].sum(axis=0) for index in groups])
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    ix=rng.integers(0,len(groups),(BOOTSTRAP_SAMPLES,len(groups)))
    values=[];draws=[]
    for r in runs:
        n=np.array([q['numerator'] for q in r['queries']]);d=np.array([q['denominator'] for q in r['queries']])
        nc,dc=clustered(n),clustered(d)
        denom=dc[ix].sum(1)
        if np.any(denom<=0):
            raise ValueError('Bootstrap has zero target separation; score cannot be reported')
        values.append(score(n,d));draws.append(100*(1-nc[ix].sum(1)/denom))
    estimates=np.mean(draws,axis=0)
    result = dict(task=runs[0]['task'],family=runs[0]['family'],regime=runs[0]['regime'],
                score=float(np.mean(values)),ci95=np.quantile(estimates,[.025,.975]).tolist(),
                training_repetitions=len(runs),training_sd=float(np.std(values,ddof=1)) if len(runs)>1 else None,
                training_seeds=[r['training_seed'] for r in runs],scenes=len(ids),run_ids=[r['id'] for r in runs],
                panel_sha256=runs[0]['panel_sha256'],bootstrap_clusters=len(groups))
    if all('real_input_horizon_numerators' in q for r in runs for q in r['queries']):
        means=[];boot=[]
        for r in runs:
            d=np.array([q['denominator'] for q in r['queries']])
            free=np.array([q['horizon_numerators'] for q in r['queries']])
            real=np.array([q['real_input_horizon_numerators'] for q in r['queries']])
            # One all-horizon denominator is shared at every depth and by both branches.
            # A depth-dependent denominator could mask growth in prediction error.
            means.append(np.stack([free.sum(0),real.sum(0),(free-real).sum(0)])/d.sum())
            fc,rc,dc=clustered(free),clustered(real),clustered(d)
            denom=dc[ix].sum(1)[:,None]
            boot.append(np.stack([fc[ix].sum(1)/denom,rc[ix].sum(1)/denom,
                                  (fc-rc)[ix].sum(1)/denom]))
        mean=np.mean(means,axis=0);draws=np.mean(boot,axis=0)
        result['error_diagnostics']={}
        for i,key in enumerate(('free','real_input','feedback_gap')):
            result['error_diagnostics'][key]=dict(mean=mean[i].tolist(),
                ci95=np.quantile(draws[i],[.025,.975],axis=0).T.tolist())
    return result


def collect(specs, results_root, panels_root):
    runs=[];missing=[]
    for spec in specs:
        root=Path(spec.get('result_dir') or results_root/spec['id'])
        if not (root/'receipt.json').is_file():
            missing.append(dict(id=spec['id'],reason='prediction results unavailable'));continue
        runs.append(read_run(spec,root,panels_root/spec['task']))
    rows=[]
    for key in sorted({(r['task'],r['family'],r['regime']) for r in runs}):
        group=[r for r in runs if (r['task'],r['family'],r['regime'])==key]
        rows.append(aggregate_runs(group))
    present={(r['task'],r['family'],r['regime']) for r in rows}
    combinations=[(f,g) for f in ('lewm','pldm','dinowm')
                  for g in ('original','scratch','joint','frozen')
                  if not (f=='dinowm' and g=='joint')]
    unavailable=[]
    for task in sorted({s['task'] for s in specs}):
        for family,regime in combinations:
            if (task,family,regime) not in present:
                reason=('No matching native H7 checkpoint; H3 projections are excluded'
                        if task=='action_delay' else
                        'No matching strict pixels-and-actions checkpoint in the published training comparison')
                unavailable.append(dict(task=task,family=family,regime=regime,reason=reason))
    return dict(schema=SCHEMA,metric='multistep_complete_prediction_score',evaluation_split='development',
                data_versions=sorted({r['data_version'] for r in runs}),
                protocol=dict(horizons_raw_steps=list(HORIZONS),horizon_weights=[.2]*5,
                    prediction='free autoregression; initial real history only; no future-observation refresh',
                    reference='per-query per-candidate per-horizon mean true latent across equally weighted hidden conditions',
                    aggregation='average condition/candidate/time energies within query, sum across queries, normalize per checkpoint, then mean over matching training repetitions',
                    negative_scores='retained; score has maximum100 and no finite minimum; not a percentage accuracy',
                    bootstrap=dict(samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED,unit='source cluster shared across all conditions/candidates and training repetitions; source episode or mirrored-scene family when known, otherwise query',scope='source-sampling uncertainty conditional on the evaluated checkpoints; training repetitions are not resampled'),
                    score_space='each checkpoint native frozen evaluation representation',
                    scope='registered Development candidate-action panels; per-task source coverage is reported below; not Test',
                    error_diagnostics='real-input predictions use privileged future observations; error at every depth is normalized by the same all-horizon target-mean energy; free-minus-real is a sensitivity contrast, not an additive causal decomposition'),
                coverage=dict(evaluated_runs=len(runs),evaluated_combinations=len(rows),
                    scenes_by_task={t:sorted({r['scenes'] for r in rows if r['task']==t})
                                    for t in sorted({r['task'] for r in rows})},
                    unavailable_combinations=unavailable,
                    excluded_scheme='DINO-WM joint encoder training is not part of these experiments'),
                rows=rows,runs=runs,missing=missing)


def write_outputs(payload, output, split_queries=False):
    output.parent.mkdir(parents=True,exist_ok=True)
    if split_queries:
        payload=json.loads(json.dumps(payload))
        queries={r['id']:r.pop('queries') for r in payload['runs']}
        path=output.with_name(output.stem+'_queries.json.gz')
        path.write_bytes(gzip.compress(json.dumps(queries,ensure_ascii=False,
                           separators=(',',':'),allow_nan=False).encode(),mtime=0))
        payload['query_records']=dict(path=path.name,sha256=sha(path),format='gzip-compressed JSON keyed by run id',
                                      count=sum(map(len,queries.values())))
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['task','family','regime','score','ci95_low','ci95_high','training_repetitions','training_sd','scenes']
    with output.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in payload['rows']:
            row={k:r[k] for k in fields if k in r}
            row.update(ci95_low=r['ci95'][0],ci95_high=r['ci95'][1]);w.writerow(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--results-root',type=Path,required=True)
    p.add_argument('--panels-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-query-records',action='store_true',help='Save per-query energies separately as compressed JSON')
    a=p.parse_args();specs=json.loads(a.models.read_text())
    if isinstance(specs,dict):specs=specs['specs']
    data=collect(specs,a.results_root,a.panels_root);write_outputs(data,a.output,a.split_query_records)
    print(f"Scored {len(data['runs'])} runs into {len(data['rows'])} task/model/scheme rows; missing {len(data['missing'])}")

if __name__=='__main__':
    main()
