#!/usr/bin/env python3
"""Summarize exact-plan cost rankings without treating controls as CEM runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write


def summarize(rows):
    scenes = {r['query_id'] for r in rows}
    if (len(rows) != 18 or len(scenes) != 6 or
            any({r['speed_index'] for r in rows if r['query_id']==q} != {0,1,2} for q in scenes)):
        raise ValueError('Expected six complete scenes with three speed conditions each')
    if any(not r['physical_reference_better'] for r in rows):
        raise ValueError('Cannot interpret pairwise search gap without physical improvement')
    counts = {k: sum(r['predicted_preference']==k for r in rows) for k in ('cem','reference','tie')}
    for r in rows:
        if not r['candidates']['reference']['success']:
            raise ValueError('Reference must be verified feasible')
    correct = [r['candidates']['cem'] for r in rows]
    reference = [r['candidates']['reference'] for r in rows]
    # Conservative tie policy: retain saved CEM plan. This is a post-hoc two-candidate
    # choice, not a new CEM score and not a deployable speed-blind controller.
    chosen = [r['candidates']['reference' if r['predicted_preference']=='reference' else 'cem'] for r in rows]
    return dict(conditions=len(rows), search_gap_count=counts['reference'], model_cost_misranking_count=counts['cem'],
        tied_cost_count=counts['tie'], encoded_true_reference_preference_count=sum(r['encoded_true_preference']=='reference' for r in rows),
        collision_free_cem_count=sum(not x['contact_steps'] for x in correct),
        collision_free_misranking_count=sum(r['predicted_preference']=='cem' and not r['candidates']['cem']['contact_steps'] for r in rows),
        cem_distance=float(np.mean([x['terminal_distance'] for x in correct])),
        reference_distance_max=max(x['terminal_distance'] for x in reference),
        cem_success_count=sum(x['success'] for x in correct), reference_success_count=sum(x['success'] for x in reference),
        two_candidate_distance=float(np.mean([x['terminal_distance'] for x in chosen])),
        two_candidate_success_count=sum(x['success'] for x in chosen),
        native_cost_check_max_difference=max(r['public_native_cost_max_difference'] for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    manifest=json.loads((a.source_root/'panel/manifest.json').read_text())
    panel_hash=sha(a.source_root/'panel/manifest.json')
    allrows, inputs, checkpoints=[],[],{}
    for scheme in ('T0','T1'):
        for q in manifest['queries']:
            path=a.root/scheme/(q['query_id']+'.json')
            savedpath=a.source_root/scheme/(q['query_id']+'.json')
            payload=json.loads(path.read_text())
            assert payload['panel_sha256']==panel_hash
            assert payload['saved_plan_sha256']==sha(savedpath)
            assert payload['state_hash_before']==payload['state_hash_after']
            checkpoints.setdefault(scheme,payload['checkpoint_sha256'])
            assert checkpoints[scheme]==payload['checkpoint_sha256']
            assert {r['speed_index'] for r in payload['rows']}=={0,1,2}
            allrows.extend(dict(scheme=scheme,query_id=q['query_id'],**r) for r in payload['rows'])
            inputs.append(dict(path=str(path.relative_to(a.root)),sha256=sha(path),saved_plan_sha256=sha(savedpath)))
    summary=[dict(scheme=s,**summarize([r for r in allrows if r['scheme']==s])) for s in ('T0','T1')]
    # Keep public provenance and measurements, while large action/state arrays remain
    # in the referenced per-scene artifacts.
    concise=[]
    for r in allrows:
        r=dict(r,candidates={k:{f:v for f,v in c.items() if f not in ('raw_actions','states')} for k,c in r['candidates'].items()})
        concise.append(r)
    write(a.output,dict(schema='contextworld.speed_timed_arrival.search_diagnosis.v1',
        protocol=json.loads((a.root/'protocol.json').read_text()),panel_sha256=panel_hash,
        checkpoint_sha256=checkpoints,rows=summary,per_condition=concise,inputs=inputs,
        inference='Pairwise evidence for search gap or predicted-cost misranking; no global optimality or exclusive root-cause claim.',
        aggregation='Equal weighting of three speeds within each of six scenes. Counts are descriptive, not independent significance tests.'))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
