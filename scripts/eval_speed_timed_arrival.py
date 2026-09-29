#!/usr/bin/env python3
"""Execute one native CEM plan in a fixed-deadline Speed diagnostic.

The CEM budget and full action sequence parameterization are unchanged. Future
model commands are clipped before normalization to match simulator execution.
No privileged state, speed or oracle control is provided to model or search.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_speed_action_selection import sha, write
from eval_speed_action_selection import load_adapter
from eval_speed_cem_panel import native_budget


def execute(data, raw_actions, speed_index):
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    env = TwoRoomEnv(render_mode='rgb_array')
    speed = float(data['speeds'][speed_index])
    try:
        env.reset(seed=int(data['eval_seed']), options={
            'variation': (), 'variation_values': _factor_options({'agent.speed': speed, 'door.position': 49}),
            'state': data['initial_states'][speed_index].copy(), 'target_state': data['goal_state'].copy()})
        for action in data['context_actions'][speed_index].reshape(10, 2):
            env.step(action)
        assert np.allclose(env.agent_position.numpy(), data['query_state'], atol=2e-5, rtol=0)
        assert np.array_equal(env.render(), data['history_pixels'][speed_index, -1])
        states = [env.agent_position.numpy().copy()]
        contacts = []
        for t, action in enumerate(raw_actions):
            expected = states[-1]+np.float32(speed)*np.clip(action, -1, 1)
            env.step(action)  # First-entry termination is not this diagnostic's criterion.
            states.append(env.agent_position.numpy().copy())
            if float(np.max(np.abs(states[-1]-expected))) > 1e-4:
                contacts.append(t)
        assert len(states) == 26
        distance = float(np.linalg.norm(states[-1]-data['goal_state']))
        return dict(speed_index=speed_index, speed=speed, terminal_distance=distance,
                    endpoint_within_tolerance=distance <= 2, success=distance <= 2 and not contacts,
                    contact_steps=contacts, states=np.asarray(states).tolist(),
                    endpoint_matches_free_dynamics=bool(np.allclose(states[-1], data['query_state']+
                        np.float32(speed)*np.clip(raw_actions, -1, 1).sum(axis=0), atol=2e-4, rtol=0)))
    finally:
        env.close()


def solve(adapter, data, history_index, seed, budget, normalization):
    import gymnasium as gym
    import torch
    from contextworld.benchmarks.adapters import _preprocess_pixels
    from stable_worldmodel import PlanConfig
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.planning.solver import CEMSolver

    mean, std = (torch.tensor(normalization[k], dtype=torch.float32) for k in ('mean', 'std'))
    goal = torch.from_numpy(adapter.encode_pixels(data['goal_pixels'][None], batch_size=1))[None]
    evaluator = ShootingCostEvaluator(adapter.model, GoalMSE(), encode_goal=None)

    class Cost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.evaluator = evaluator
            self.calls = 0

        def get_cost(self, info, candidates):
            raw = candidates.reshape(*candidates.shape[:-1], 5, 2)*std+mean
            effective = ((raw.clamp(-1, 1)-mean)/std).reshape_as(candidates)
            self.calls += 1
            return self.evaluator.get_cost(dict(info, goal_emb=goal), effective)

    cost = Cost()
    config = PlanConfig(horizon=5, receding_horizon=5, action_block=5, history_len=3, warm_start=True)
    solver = CEMSolver(cost=cost, device='cpu', seed=seed,
                       **{k: budget[k] for k in ('batch_size', 'num_samples', 'var_scale', 'n_steps', 'topk')})
    solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, 2), dtype=np.float32), n_envs=1, config=config)
    frames = _preprocess_pixels(data['history_pixels'][history_index], device='cpu').reshape(1, 3, 3, 224, 224)
    past = torch.from_numpy(adapter.action_standardizer.transform(
        data['context_actions'][history_index].reshape(10, 2)).astype(np.float32).reshape(1, 2, 10))
    with torch.inference_mode():
        info = dict(pixels=frames, emb=adapter.model.encode({'pixels': frames})['emb'], action_history=past)
        candidates = torch.randn(1, 4, 5, 10, generator=torch.Generator().manual_seed(seed))
        expanded = {k: v[:, None].expand(1, 4, *v.shape[1:]) for k, v in info.items()}
        cached = cost.get_cost(expanded, candidates)
        uncached = cost.get_cost({k: v for k, v in expanded.items() if k != 'emb'}, candidates)
        assert torch.allclose(cached, uncached, atol=1e-5, rtol=1e-5)
        output = solver.solve(info)
    raw = adapter.action_standardizer.inverse_transform(output['actions'][0].numpy().reshape(25, 2)).astype(np.float32)
    assert cost.calls == budget['n_steps']+2
    return dict(history_index=history_index, history_speed=float(data['speeds'][history_index]),
                raw_actions=raw.tolist(), effective_actions=np.clip(raw, -1, 1).tolist(),
                predicted_elite_cost=output['costs'], clipped_coordinate_fraction=float(np.mean(np.abs(raw) > 1)),
                search_cost_calls=cost.calls-2, cached_native_cost_max_difference=float((cached-uncached).abs().max()),
                outcomes=[execute(data, raw, si) for si in range(3)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--query-id', required=True)
    p.add_argument('--family', choices=['lewm'], default='lewm')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True, parents=True)
    dest = a.output/(a.query_id+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    assert sha(a.checkpoint) == a.expected_sha256
    manifest = json.loads((a.panel/'manifest.json').read_text())
    row = next(r for r in manifest['queries'] if r['query_id'] == a.query_id)
    path = a.panel/row['path']
    assert sha(path) == row['sha256']
    data = np.load(path, allow_pickle=False)
    adapter = load_adapter(a, manifest)
    before = adapter.frozen_state_hash()
    budget = native_budget(a.stable_repo)
    seed = int(np.random.SeedSequence([int(data['eval_seed']), int(data['evaluation_index'])]).generate_state(1)[0])
    plans = []
    for hi in range(3):
        plans.append(solve(adapter, data, hi, seed, budget, manifest['protocol']['normalization']))
        print(a.query_id, 'history', hi, 'completed', flush=True)
    after = adapter.frozen_state_hash()
    assert before == after
    write(dest, dict(query_id=a.query_id, plans=plans, model=adapter.metadata, cem_seed=seed,
                     checkpoint_sha256=a.expected_sha256, panel_sha256=sha(a.panel/'manifest.json'),
                     state_hash_before=before, state_hash_after=after,
                     search_budget={k: v for k, v in budget.items() if k != 'eval_budget'},
                     executed_raw_steps=25, no_training=True, evaluation_split='development'))


if __name__ == '__main__':
    main()
