#!/usr/bin/env python3
"""Summarize paired initial-history effects in fixed-deadline CEM plans."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write


def scene_metrics(plans):
    assert {p['history_index'] for p in plans} == {0, 1, 2}
    indexed = {(p['history_index'], o['speed_index']): o for p in plans for o in p['outcomes']}
    assert len(indexed) == 9
    rows = {}
    for label, keys in [('correct', [(i, i) for i in range(3)]),
                        ('wrong', [(h, s) for s in range(3) for h in range(3) if h != s])]:
        observations = [indexed[k] for k in keys]
        rows[label+'_distance'] = float(np.mean([x['terminal_distance'] for x in observations]))
        rows[label+'_success_percent'] = 100*float(np.mean([x['success'] for x in observations]))
        rows[label+'_endpoint_percent'] = 100*float(np.mean([x.get('endpoint_within_tolerance', x['terminal_distance'] <= 2) for x in observations]))
        rows[label+'_contact_percent'] = 100*float(np.mean([bool(x['contact_steps']) for x in observations]))
    rows['history_distance_benefit'] = rows['wrong_distance']-rows['correct_distance']
    rows['history_success_gain_pp'] = rows['correct_success_percent']-rows['wrong_success_percent']
    return rows


def summarize(records):
    keys = [k for k in records[0] if k not in ('query_id', 'scheme')]
    result = {k: float(np.mean([r[k] for r in records])) for k in keys}
    indices = np.random.default_rng(20260929).integers(0, len(records), (10000, len(records)))
    for key in ('history_distance_benefit', 'history_success_gain_pp'):
        values = np.asarray([r[key] for r in records])
        result[key+'_ci95'] = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.root/'panel/manifest.json').read_text())
    manifest_hash = sha(a.root/'panel/manifest.json')
    records, models, inputs = [], {}, []
    for scheme in ('T0', 'T1'):
        for query in manifest['queries']:
            path = a.root/scheme/(query['query_id']+'.json')
            payload = json.loads(path.read_text())
            assert payload['panel_sha256'] == manifest_hash
            assert payload['state_hash_before'] == payload['state_hash_after']
            if scheme in models:
                assert models[scheme]['checkpoint_sha256'] == payload['checkpoint_sha256']
            models[scheme] = payload['model']
            records.append(dict(query_id=query['query_id'], scheme=scheme, **scene_metrics(payload['plans'])))
            inputs.append(dict(path=str(path.relative_to(a.root)), sha256=sha(path)))
    assert len(manifest['queries']) == 6 and len(records) == 12
    rows = [dict(scheme=s, **summarize([r for r in records if r['scheme'] == s])) for s in ('T0', 'T1')]
    comparisons = {}
    rng = np.random.default_rng(20260929)
    idx = rng.integers(0, 6, (10000, 6))
    for metric in ('correct_distance', 'correct_success_percent'):
        values = {s: np.array([r[metric] for r in records if r['scheme'] == s]) for s in ('T0', 'T1')}
        difference = values['T1']-values['T0']
        comparisons[metric] = dict(t1_minus_t0=float(difference.mean()), ci95=np.quantile(difference[idx].mean(axis=1), [.025, .975]).tolist())
    write(a.output, dict(schema='contextworld.speed_timed_arrival.results.v1',
                         protocol=manifest['protocol'], validation=manifest['validation'],
                         panel_sha256=manifest_hash, scenes=6, correct_trials_per_model=18,
                         wrong_trials_per_model=36, unique_cem_plans_per_model=18,
                         models=models, rows=rows, per_scene=records, model_comparison=comparisons, inputs=inputs,
                         statistics='average speed/history conditions within scene, then average scenes; 10000 paired bootstrap resamples over 6 scenes',
                         interpretation='single 25-step plan with terminal 2px tolerance and no contact; distinct from original 50-step first-entry CEM'))
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
