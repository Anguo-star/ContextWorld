#!/usr/bin/env python3
"""Aggregate paired closed-loop trials without counting conditions as scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_speed_action_selection import sha, write


def summarize(panel, arms):
    manifest = json.loads((panel/'manifest.json').read_text())
    rows, identities, scene_sets, records = [], {}, [], []
    for name, directory in arms.items():
        loaded = [json.loads(p.read_text()) for p in sorted(directory.glob('*_s*_h*.json'))]
        if not loaded:
            raise ValueError(f'No results in {directory}')
        assert {x['panel_sha256'] for x in loaded} == {sha(panel/'manifest.json')}
        hashes = {x['checkpoint_sha256'] for x in loaded}
        assert len(hashes) == 1
        assert all(x['state_hash_before'] == x['state_hash_after'] for x in loaded)
        assert all(x['budget'] == loaded[0]['budget'] for x in loaded)
        identities[name] = dict(checkpoint_sha256=hashes.pop(), model=loaded[0]['model'])
        scenes = sorted({x['query_id'] for x in loaded})
        scene_sets.append(scenes)
        lookup = {(x['query_id'], x['result']['speed_index'], x['result']['history_index']): x['result'] for x in loaded}
        assert len(lookup) == len(loaded) == len(scenes)*9
        for si in range(3):
            paired = []
            for qid in scenes:
                correct = lookup[qid, si, si]
                wrong = [lookup[qid, si, hi] for hi in range(3) if hi != si]
                assert len({correct['cem_seed'], *(x['cem_seed'] for x in wrong)}) == 1
                def mean(key):
                    return float(np.mean([x[key] for x in wrong]))
                paired.append(dict(query_id=qid, success=float(correct['success']), wrong_success=mean('success'),
                                   distance=correct['final_distance'], wrong_distance=mean('final_distance'),
                                   steps=correct['steps'], wrong_steps=mean('steps'),
                                   distance_auc=correct['distance_auc'], wrong_distance_auc=mean('distance_auc')))
                records.append(dict(arm=name, speed=correct['speed'], **paired[-1]))
            avg = lambda key: float(np.mean([x[key] for x in paired]))
            rows.append(dict(arm=name, speed=correct['speed'], scenes=len(scenes),
                             success_percent=100*avg('success'), wrong_success_percent=100*avg('wrong_success'),
                             final_distance=avg('distance'), wrong_final_distance=avg('wrong_distance'),
                             distance_gain=avg('wrong_distance')-avg('distance'),
                             steps=avg('steps'), wrong_steps=avg('wrong_steps'),
                             distance_auc=avg('distance_auc'), wrong_distance_auc=avg('wrong_distance_auc')))
    assert all(x == scene_sets[0] for x in scene_sets)
    return dict(schema='contextworld.speed_cem_initial_evidence.v1', status='pilot',
                evaluation_split='development', scenes=len(scene_sets[0]), conditions_per_scene=3,
                initial_histories_per_condition=3, protocol=manifest['protocol'],
                panel_sha256=sha(panel/'manifest.json'), budget=loaded[0]['budget'],
                models=identities, rows=rows, paired_records=records,
                notes=['Incorrect initial histories are averaged equally.',
                       'After the first plan, every arm uses its own continuously observed history.',
                       'Scene is the statistical unit; speed conditions are not independent scenes.',
                       'This is a diagnostic pilot, not a full 6x50 checkpoint score.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--t0', type=Path, required=True)
    p.add_argument('--t1', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.panel, {'T0': a.t0, 'T1': a.t1})
    write(a.output, result)
    print(json.dumps(result['rows'], indent=2))


if __name__ == '__main__':
    main()
