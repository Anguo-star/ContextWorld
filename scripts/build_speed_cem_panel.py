#!/usr/bin/env python3
"""Build a Development planning panel with distant goals and real histories.

Retain every unseen-interpolation scene from the verified Speed source panel.
The original scene goal replaces the short-distance action-selection goal.
An oracle checks reachability only; its state inputs never enter model inference.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_speed_action_selection import sha, write


def init_worker(stable_repo, stable_ref):
    import torch
    torch.set_num_threads(1)
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    load_stable_worldmodel(ROOT, stable_repo, stable_ref)


def build_one(job):
    source_panel, output, row = job
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    original = source_panel/row['path']
    assert sha(original) == row['sha256']
    source = source_panel/'source'/row['path']
    data = np.load(source, allow_pickle=False)
    assert np.allclose(data['speeds'], [3.4, 4.8, 6.9])
    assert np.array_equal(data['frames'][:, 2], np.broadcast_to(data['frames'][0, 2], data['frames'][:, 2].shape))
    assert np.array_equal(data['actions'], np.broadcast_to(data['actions'][0], data['actions'].shape))
    assert np.allclose(data['states'][:, 10], data['states'][0, 10], atol=2e-5, rtol=0)
    assert np.array_equal(data['old_goal'], np.broadcast_to(data['old_goal'][0], data['old_goal'].shape))
    query, goal = data['states'][0, 10], data['old_goal'][0]
    distance = float(np.linalg.norm(query-goal))
    assert distance > 16
    env = TwoRoomEnv(render_mode='rgb_array')
    audits, goal_pixels = [], None
    try:
        for si, speed in enumerate(data['speeds']):
            env.reset(seed=int(data['eval_seed']), options={
                'variation': (), 'variation_values': _factor_options({'agent.speed': float(speed), 'door.position': 49}),
                'state': data['states'][si, 0].copy(), 'target_state': goal.copy()})
            pix = env._target_img.numpy().transpose(1, 2, 0).astype(np.uint8)
            if goal_pixels is not None:
                assert np.array_equal(goal_pixels, pix)
            goal_pixels = pix
            max_error = 0.
            for step in range(11):
                max_error = max(max_error, float(np.max(np.abs(env.agent_position.numpy()-data['states'][si, step]))))
                if step in (0, 5, 10):
                    assert np.array_equal(env.render(), data['frames'][si, step//5])
                if step < 10:
                    _, _, ended, _, _ = env.step(data['actions'][si].reshape(15, 2)[step])
                    assert not ended, 'Goal reached in initial context'
            assert max_error < 2e-5
            # Continue from the real context, without a query-state reset.
            oracle_states = [env.agent_position.numpy().copy()]
            oracle_actions = []
            reached = False
            for _ in range(50):
                action = np.clip((goal-env.agent_position.numpy())/speed, -1, 1).astype(np.float32)
                _, _, reached, _, _ = env.step(action)
                oracle_states.append(env.agent_position.numpy().copy())
                oracle_actions.append(action.tolist())
                if reached:
                    break
            assert reached, f"Unreachable scene: {row['query_id']} speed={speed}"
            audits.append(dict(speed=float(speed), replay_state_max=max_error, replay_pixels_exact=True,
                               oracle_steps=len(oracle_actions), oracle_final_distance=float(np.linalg.norm(env.agent_position.numpy()-goal)),
                               oracle_states=np.asarray(oracle_states).tolist(), oracle_actions=oracle_actions))
    finally:
        env.close()
    dest = output/(row['query_id']+'.npz')
    np.savez_compressed(dest, initial_states=data['states'][:, 0], history_pixels=data['frames'][:, :3],
                        context_actions=data['actions'][:, :2], query_state=query, goal_state=goal,
                        goal_pixels=goal_pixels, speeds=data['speeds'], eval_seed=data['eval_seed'],
                        evaluation_index=data['evaluation_index'])
    return dict(query_id=row['query_id'], path=dest.name, sha256=sha(dest), source_sha256=sha(source),
                eval_seed=int(data['eval_seed']), evaluation_index=int(data['evaluation_index']),
                initial_distance=distance, validation=audits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--workers', type=int, default=12)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    source = json.loads((a.source_panel/'manifest.json').read_text())
    rows = sorted((x for x in source['queries'] if x['track'] == 'unseen_interpolation'),
                  key=lambda x: (x['eval_seed'], x['evaluation_index']))
    assert len(rows) == 300
    assert {(x['eval_seed'], x['evaluation_index']) for x in rows} == {(s, i) for s in range(42, 48) for i in range(50)}
    a.output.mkdir(parents=True)
    protocol = dict(schema='contextworld.speed_cem_initial_evidence.v1', evaluation_split='development',
                    track='unseen_interpolation', speeds=[3.4, 4.8, 6.9], scenes=300, seeds=list(range(42, 48)),
                    scenes_per_seed=50, source_manifest_sha256=sha(a.source_panel/'manifest.json'),
                    normalization=source['protocol']['normalization'], stable_ref=a.stable_ref,
                    query_rule='all registered Development scenes, without model-score filtering',
                    goal_rule='original shared scene goal stored in source old_goal; not short action-selection target',
                    goal_offset_note='The original H5 uses a 25-step row offset to choose a goal; this panel stores an explicit shared goal instead.',
                    history_control='replace initial history only; subsequent replans use actual live history',
                    success_distance_px=16, evaluation_budget_raw_steps=50,
                    oracle_role='physical feasibility check only, never model input',
                    test_payload_accessed=False, no_training=True)
    write(a.output/'protocol.json', protocol)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=init_worker, initargs=(a.stable_repo, a.stable_ref)) as pool:
        result = list(pool.map(build_one, [(a.source_panel, a.output, x) for x in rows], chunksize=4))
    audits = [y for x in result for y in x['validation']]
    validation = dict(scenes=len(result), physical_conditions=len(audits),
                      all_reachable=True, replay_pixels_exact=all(x['replay_pixels_exact'] for x in audits),
                      replay_state_max=max(x['replay_state_max'] for x in audits),
                      initial_distance_range=[min(x['initial_distance'] for x in result), max(x['initial_distance'] for x in result)],
                      oracle_steps_range=[min(x['oracle_steps'] for x in audits), max(x['oracle_steps'] for x in audits)])
    write(a.output/'manifest.json', dict(protocol=protocol, validation=validation, queries=result))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
