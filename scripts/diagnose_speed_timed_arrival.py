#!/usr/bin/env python3
"""Compare saved CEM plans with independently verified feasible controls.

No search or training: rescore both actions with the original native objective,
then compare that ranking with encoded true futures and physical goal error.
Privileged speed specifies the diagnostic reference, never the model input.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write
from eval_speed_action_selection import load_adapter


def cost_order(cem, reference, *, atol=1e-5, rtol=1e-5):
    """Keep numerical ties distinct from either preference."""
    if not np.isfinite([cem, reference]).all():
        raise ValueError('Non-finite candidate costs')
    if abs(cem-reference) <= atol+rtol*max(abs(cem), abs(reference)):
        return 'tie'
    return 'reference' if reference < cem else 'cem'


def replay(data, actions, si):
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    env = TwoRoomEnv(render_mode='rgb_array')
    speed = float(data['speeds'][si])
    try:
        env.reset(seed=int(data['eval_seed']), options={
            'variation': (), 'variation_values': _factor_options({'agent.speed': speed, 'door.position': 49}),
            'state': data['initial_states'][si].copy(), 'target_state': data['goal_state'].copy()})
        for action in data['context_actions'][si].reshape(10, 2):
            env.step(action)
        assert np.allclose(env.agent_position.numpy(), data['query_state'], atol=2e-5, rtol=0)
        assert np.array_equal(env.render(), data['history_pixels'][si, -1])
        states = [env.agent_position.numpy().copy()]
        frames, contacts = [], []
        for t, action in enumerate(actions):
            expected = states[-1]+np.float32(speed)*np.clip(action, -1, 1)
            env.step(action)
            states.append(env.agent_position.numpy().copy())
            if float(np.max(np.abs(states[-1]-expected))) > 1e-4:
                contacts.append(t)
            if (t+1) % 5 == 0:
                frames.append(env.render().copy())
        assert len(states) == 26 and len(frames) == 5
        distance = float(np.linalg.norm(states[-1]-data['goal_state']))
        return dict(states=np.asarray(states), pixels=np.asarray(frames), contacts=contacts,
                    distance=distance, success=distance <= 2 and not contacts)
    finally:
        env.close()


def evaluate(adapter, data, saved):
    import torch
    from contextworld.benchmarks.adapters import _preprocess_pixels
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    goal = adapter.encode_pixels(data['goal_pixels'][None], batch_size=1)[0]
    evaluator = ShootingCostEvaluator(adapter.model, GoalMSE(), encode_goal=None)
    rows = []
    for si in range(3):
        plan = next(p for p in saved['plans'] if p['history_index'] == si)
        prior = next(o for o in plan['outcomes'] if o['speed_index'] == si)
        cem = np.asarray(plan['raw_actions'], np.float32)
        reference = np.tile((data['goal_state']-data['query_state'])/(25*data['speeds'][si]), (25, 1)).astype(np.float32)
        assert np.max(np.abs(reference)) <= 1
        candidates = np.stack([cem, reference])
        true = [replay(data, a, si) for a in candidates]
        assert np.array_equal(true[0]['states'], np.asarray(prior['states'], np.float32)), 'Saved CEM replay changed'
        assert true[0]['contacts'] == prior['contact_steps']
        assert true[1]['success'], 'Known feasible reference failed'
        histories = np.repeat(data['history_pixels'][si:si+1], 2, axis=0)
        raw = np.concatenate([np.repeat(data['context_actions'][si:si+1], 2, axis=0),
                              np.clip(candidates, -1, 1).reshape(2, 5, 5, 2)], axis=1)
        predictions = adapter.rollout_latents(histories, raw, batch_size=2)
        targets = adapter.encode_pixels(np.stack([x['pixels'] for x in true]).reshape(-1, 224, 224, 3), batch_size=10).reshape(predictions.shape)
        predicted_cost = ((predictions-goal)**2).sum(axis=-1)
        encoded_true_cost = ((targets-goal)**2).sum(axis=-1)
        prediction_error = ((predictions-targets)**2).sum(axis=-1)
        # Independently confirm the public rollout produces the search's exact objective.
        frames = _preprocess_pixels(data['history_pixels'][si], device='cpu').reshape(1, 1, 3, 3, 224, 224).expand(1, 2, 3, 3, 224, 224)
        normalized = torch.from_numpy(adapter._normalize_actions(raw))
        with torch.inference_mode():
            native_cost = evaluator.get_cost(dict(pixels=frames, action_history=normalized[None, :, :2],
                   goal_emb=torch.from_numpy(goal)[None, None]), normalized[None, :, 2:])[0].numpy()
        assert np.allclose(native_cost, predicted_cost[:, -1], atol=1e-5, rtol=1e-5)
        order = cost_order(*native_cost)
        rows.append(dict(speed_index=si, speed=float(data['speeds'][si]),
            predicted_preference=order, encoded_true_preference=cost_order(*encoded_true_cost[:, -1]),
            physical_reference_better=bool(true[1]['distance'] < true[0]['distance']),
            public_native_cost_max_difference=float(np.max(np.abs(native_cost-predicted_cost[:, -1]))),
            candidates={name:dict(raw_actions=candidates[j].tolist(),
                predicted_goal_cost_by_step=predicted_cost[j].tolist(),
                native_terminal_goal_cost=float(native_cost[j]),
                encoded_true_goal_cost_by_step=encoded_true_cost[j].tolist(),
                prediction_squared_error_by_step=prediction_error[j].tolist(),
                terminal_distance=true[j]['distance'], success=true[j]['success'],
                contact_steps=true[j]['contacts'], states=true[j]['states'].tolist())
                for j,name in enumerate(('cem','reference'))}))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--saved-plan', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--family', choices=['lewm'], default='lewm')
    p.add_argument('--threads', type=int, default=2)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    saved=json.loads(a.saved_plan.read_text())
    dest=a.output/(saved['query_id']+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    assert sha(a.checkpoint) == a.expected_sha256 == saved['checkpoint_sha256']
    assert sha(a.panel/'manifest.json') == saved['panel_sha256']
    manifest=json.loads((a.panel/'manifest.json').read_text())
    row=next(r for r in manifest['queries'] if r['query_id']==saved['query_id'])
    assert sha(a.panel/row['path']) == row['sha256']
    data=np.load(a.panel/row['path'], allow_pickle=False)
    adapter=load_adapter(a, manifest)
    before=adapter.frozen_state_hash()
    rows=evaluate(adapter, data, saved)
    after=adapter.frozen_state_hash()
    assert before==after
    write(dest, dict(query_id=saved['query_id'], checkpoint_sha256=a.expected_sha256,
        panel_sha256=saved['panel_sha256'], saved_plan_sha256=sha(a.saved_plan), model=adapter.metadata,
        state_hash_before=before, state_hash_after=after, evaluation_split='development',
        no_training=True, new_searches=0, matched_history_only=True, rows=rows))
    print(saved['query_id'], 'complete', flush=True)

if __name__=='__main__':
    main()
