"""Score complete free-running trajectories from frozen prediction receipts.

Score = 100 * (1 - complete squared error / condition-mean reference error).
Average conditions, candidates and the five fixed horizons within each source
query before summing energies; never average per-query error ratios. Training
replicates contribute equally after each receives its own normalized score.
"""
from __future__ import annotations
import argparse
import csv
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
    return dict(query_id=record['scene_id'], numerator=float(n.mean()/size), denominator=float(d.mean()/size),
                horizon_numerators=(n/size).tolist(), horizon_denominators=(d/size).tolist(),
                conditions=int(record['conditions']), candidates=int(record['candidates']),
                zero_separation_horizons=int(np.sum(d==0)))


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
        queries.append(query_energies(row))
    queries.sort(key=lambda q:q['query_id'])
    if len(queries)!=len(entries) or len({q['query_id'] for q in queries})!=len(entries):
        raise ValueError('Incomplete or duplicated query coverage')
    value=score([q['numerator'] for q in queries],[q['denominator'] for q in queries])
    return dict(id=spec['id'],task=spec['task'],family=spec['family'],regime=spec['regime'],
                training_seed=spec['training_seed'],checkpoint_sha256=spec['checkpoint_sha256'],
                panel_sha256=receipt['panel_sha256'],receipt_sha256=sha(receipt_path),
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
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    ix=rng.integers(0,len(ids),(BOOTSTRAP_SAMPLES,len(ids)))
    values=[];draws=[]
    for r in runs:
        n=np.array([q['numerator'] for q in r['queries']]);d=np.array([q['denominator'] for q in r['queries']])
        denom=d[ix].sum(1)
        if np.any(denom<=0):
            raise ValueError('Bootstrap has zero target separation; score cannot be reported')
        values.append(score(n,d));draws.append(100*(1-n[ix].sum(1)/denom))
    estimates=np.mean(draws,axis=0)
    return dict(task=runs[0]['task'],family=runs[0]['family'],regime=runs[0]['regime'],
                score=float(np.mean(values)),ci95=np.quantile(estimates,[.025,.975]).tolist(),
                training_repetitions=len(runs),training_sd=float(np.std(values,ddof=1)) if len(runs)>1 else None,
                training_seeds=[r['training_seed'] for r in runs],scenes=len(ids),run_ids=[r['id'] for r in runs],
                panel_sha256=runs[0]['panel_sha256'])


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
                protocol=dict(horizons_raw_steps=list(HORIZONS),horizon_weights=[.2]*5,
                    prediction='free autoregression; initial real history only; no future-observation refresh',
                    reference='per-query per-candidate per-horizon mean true latent across equally weighted hidden conditions',
                    aggregation='average condition/candidate/time energies within query, sum across queries, normalize per checkpoint, then mean over matching training repetitions',
                    negative_scores='retained; score has maximum100 and no finite minimum; not a percentage accuracy',
                    bootstrap=dict(samples=BOOTSTRAP_SAMPLES,seed=BOOTSTRAP_SEED,unit='source query shared across all conditions/candidates and training repetitions',scope='query uncertainty conditional on the evaluated checkpoints; training repetitions are not resampled'),
                    score_space='each checkpoint native frozen evaluation representation',
                    scope='fixed Development candidate-action panels; six Speed scenes and sixteen per other task; not the full benchmark or Test'),
                coverage=dict(evaluated_runs=len(runs),evaluated_combinations=len(rows),
                    unavailable_combinations=unavailable,
                    excluded_scheme='DINO-WM joint encoder training is not part of these experiments'),
                rows=rows,runs=runs,missing=missing)


def write_outputs(payload, output):
    output.parent.mkdir(parents=True,exist_ok=True)
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
    a=p.parse_args();specs=json.loads(a.models.read_text())
    if isinstance(specs,dict):specs=specs['specs']
    data=collect(specs,a.results_root,a.panels_root);write_outputs(data,a.output)
    print(f"Scored {len(data['runs'])} runs into {len(data['rows'])} task/model/scheme rows; missing {len(data['missing'])}")

if __name__=='__main__':
    main()
