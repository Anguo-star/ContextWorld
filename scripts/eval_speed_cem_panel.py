#!/usr/bin/env python3
"""Evaluate frozen models with native CEM and continuously updated real history.

Correct and incorrect histories differ only at the first solve. Later solves
use each rollout's own observations and issued actions. This measures the value
of initial evidence, not the effect of supplying wrong dynamics indefinitely.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_speed_action_selection import sha, write
from eval_speed_action_selection import load_adapter


def native_budget(stable_repo):
    import yaml
    root = Path(stable_repo) / 'scripts/plan/config'
    env = yaml.safe_load((root / 'tworoom.yaml').read_text())
    cem = yaml.safe_load((root / 'solver/cem.yaml').read_text())
    budget = {k: cem[k] for k in ('batch_size', 'num_samples', 'var_scale', 'n_steps', 'topk')}
    budget.update(env['plan_config'])
    budget['eval_budget'] = env['eval']['eval_budget']
    expected = dict(batch_size=1, num_samples=300, var_scale=1.0, n_steps=30,
                    topk=30, horizon=5, receding_horizon=5, action_block=5, eval_budget=50)
    if budget != expected:
        raise ValueError(f'Original planning configuration changed: {budget}')
    return budget


def solve_episode(adapter, data, speed_index, history_index, seed, budget):
    import gymnasium as gym
    import torch
    from contextworld.benchmarks.adapters import _preprocess_pixels
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel import PlanConfig
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.planning.solver import CEMSolver

    model = adapter.model
    goal_pixels = data['goal_pixels']
    with torch.inference_mode():
        goal = torch.from_numpy(adapter.encode_pixels(goal_pixels[None], batch_size=1))[None]
    evaluator = ShootingCostEvaluator(model, GoalMSE(), encode_goal=None)

    class Cost(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.evaluator = evaluator
            self.calls = 0

        def get_cost(self, info, candidates):
            info['goal_emb'] = goal
            self.calls += 1
            return self.evaluator.get_cost(info, candidates)

    cost = Cost()
    solver = CEMSolver(cost=cost, device='cpu', seed=seed,
                       **{k: budget[k] for k in ('batch_size', 'num_samples', 'var_scale', 'n_steps', 'topk')})
    config = PlanConfig(horizon=budget['horizon'], receding_horizon=budget['receding_horizon'],
                        action_block=budget['action_block'], history_len=3, warm_start=True)
    solver.configure(action_space=gym.spaces.Box(-1., 1., shape=(1, 2), dtype=np.float32), n_envs=1, config=config)
    env = TwoRoomEnv(render_mode='rgb_array')
    issued, states, live_pixels, plans, checks = [], [], [], [], []
    success = False
    try:
        env.reset(seed=int(data['eval_seed']), options={
            'variation': (), 'variation_values': _factor_options({'agent.speed': float(data['speeds'][speed_index]), 'door.position': 49}),
            'state': data['initial_states'][speed_index].copy(), 'target_state': data['goal_state'].copy()})
        # Reach the query by executing the real prefix; never reset at query.
        for action in data['context_actions'][speed_index].reshape(10, 2):
            _, _, terminated, _, _ = env.step(action)
            if terminated:
                raise ValueError('Goal reached during context prefix')
        assert np.allclose(env.agent_position.numpy(), data['query_state'], atol=2e-5, rtol=0)
        assert np.array_equal(env.render(), data['history_pixels'][speed_index, -1])
        states.append(env.agent_position.numpy().copy())
        live_pixels.append(env.render().copy())
        while len(issued) < budget['eval_budget'] and not success:
            step = len(issued)
            if step == 0:
                frames = data['history_pixels'][history_index].copy()
                raw_past = data['context_actions'][history_index].reshape(10, 2)
                frame_steps = [-10, -5, 0]
            else:
                frame_steps = [step-10, step-5, step]
                frames = np.stack([live_pixels[t] for t in frame_steps])
                raw_past = np.asarray(issued[-10:])
            transformed = _preprocess_pixels(frames, device='cpu').reshape(1, 3, 3, 224, 224)
            past = torch.from_numpy(adapter.action_standardizer.transform(raw_past).astype(np.float32).reshape(1, 2, 10))
            with torch.inference_mode():
                emb = model.encode({'pixels': transformed})['emb']
                info = {'pixels': transformed, 'emb': emb, 'action_history': past}
                # Compare caching against native image encoding over all five rollout steps.
                rng = torch.Generator().manual_seed(seed + step)
                candidates = torch.randn(1, 4, 5, 10, generator=rng)
                expanded = {k: v[:, None].expand(1, 4, *v.shape[1:]) for k, v in info.items()}
                cached = evaluator.get_cost(dict(expanded, goal_emb=goal), candidates)
                uncached = {k: v for k, v in expanded.items() if k != 'emb'}
                uncached = evaluator.get_cost(dict(uncached, goal_emb=goal), candidates)
                assert torch.allclose(cached, uncached, atol=1e-5, rtol=1e-5)
                checks.append(float((cached-uncached).abs().max()))
                output = solver.solve(info)
            # Original horizon == receding horizon: no unused tail is carried over.
            raw_plan = adapter.action_standardizer.inverse_transform(output['actions'][0].numpy().reshape(-1, 2))
            plans.append({'step': step, 'frame_steps': frame_steps, 'raw_actions': raw_plan.tolist(),
                          'predicted_elite_cost': output['costs'], 'context': 'initial' if step == 0 else 'live'})
            for action in raw_plan:
                if len(issued) >= budget['eval_budget']:
                    break
                _, _, success, _, _ = env.step(action)
                issued.append(action.copy())
                states.append(env.agent_position.numpy().copy())
                live_pixels.append(env.render().copy())
                if success:
                    break
        states = np.asarray(states)
        distances = np.linalg.norm(states-data['goal_state'], axis=-1)
        return dict(speed_index=speed_index, history_index=history_index, correct_history=speed_index == history_index,
                    speed=float(data['speeds'][speed_index]), history_speed=float(data['speeds'][history_index]),
                    cem_seed=seed, success=bool(success), steps=len(issued), final_distance=float(distances[-1]),
                    distance_auc=float(np.mean(distances)/distances[0]), states=states.tolist(),
                    actions=np.asarray(issued).tolist(), distances=distances.tolist(), plans=plans,
                    issued_action_clip_fraction=float(np.mean(np.abs(issued) > 1)),
                    cost_calls=cost.calls, cache_check_max=max(checks), history_mode='initial_evidence_then_live_history')
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--family', choices=['lewm', 'pldm'], default='lewm')
    p.add_argument('--query-id', required=True)
    p.add_argument('--speed-index', type=int, required=True)
    p.add_argument('--history-index', type=int, required=True)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    assert sha(a.checkpoint) == a.expected_sha256
    manifest = json.loads((a.panel/'manifest.json').read_text())
    row = next(x for x in manifest['queries'] if x['query_id'] == a.query_id)
    path = a.panel/row['path']
    assert sha(path) == row['sha256']
    a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output/f'{a.query_id}_s{a.speed_index}_h{a.history_index}.json'
    if dest.exists():
        raise FileExistsError(dest)
    adapter = load_adapter(a, manifest)
    before = adapter.frozen_state_hash()
    budget = native_budget(a.stable_repo)
    data = np.load(path, allow_pickle=False)
    seed = int(np.random.SeedSequence([int(data['eval_seed']), int(data['evaluation_index'])]).generate_state(1)[0])
    result = solve_episode(adapter, data, a.speed_index, a.history_index, seed, budget)
    after = adapter.frozen_state_hash()
    assert before == after
    write(dest, dict(query_id=a.query_id, result=result, checkpoint_sha256=a.expected_sha256,
                     panel_sha256=sha(a.panel/'manifest.json'), model=adapter.metadata, budget=budget,
                     state_hash_before=before, state_hash_after=after, evaluation_split='development'))
    print('completed', dest, result['success'], result['final_distance'], flush=True)


if __name__ == '__main__':
    main()
