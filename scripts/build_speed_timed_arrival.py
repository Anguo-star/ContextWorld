#!/usr/bin/env python3
"""Build a small Speed task whose precise terminal decision requires history.

All six preselected scenes are retained. Goal geometry and acceptance criteria
are fixed before model inference. Only simulator controls validate feasibility;
neither their actions nor privileged state are supplied to model CEM.
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


def speed_blind_bound(distance, speeds, tolerance=2.):
    """Optimal shared collision-free displacement and its success ceiling.

    Project the common cumulative action onto the goal direction. Off-axis
    motion only increases distances. The objective is sum(v*abs(b-D/v)), whose
    minimizer is a speed-weighted median. Goal intervals bound success even
    when arbitrary two-dimensional actions are allowed.
    """
    v = np.asarray(speeds, dtype=np.float64)
    if v.ndim != 1 or not len(v) or np.any(v <= 0) or distance < 0 or tolerance < 0:
        raise ValueError('Positive speeds and nonnegative distance/tolerance required')
    order = np.argsort(distance/v)
    at = int(np.searchsorted(np.cumsum(v[order]), v.sum()/2))
    displacement = float((distance/v)[order[at]])
    lower, upper = (distance-tolerance)/v, (distance+tolerance)/v
    overlap = max(int(np.sum((lower <= x) & (x <= upper))) for x in np.concatenate([lower, upper]))
    return dict(cumulative_action=displacement,
                mean_distance=float(np.mean(np.abs(v*displacement-distance))),
                maximum_success_fraction=overlap/len(v),
                goal_intervals=np.stack([lower, upper], axis=1).tolist())


def build_query(source, row, output):
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    path = source/row['path']
    assert sha(path) == row['sha256']
    with np.load(path, allow_pickle=False) as payload:
        data = {k: payload[k].copy() for k in payload.files}
    v = data['speeds']
    assert np.allclose(v, [3.4, 4.8, 6.9])
    assert np.array_equal(data['history_pixels'][:, -1], np.broadcast_to(data['history_pixels'][0, -1], data['history_pixels'][:, -1].shape))
    assert np.array_equal(data['context_actions'], np.broadcast_to(data['context_actions'][0], data['context_actions'].shape))
    direction = data['goal_state']-data['query_state']
    direction /= np.linalg.norm(direction)
    goal = data['query_state']+np.float32(32)*direction
    gap = goal-data['query_state']
    distance = float(np.linalg.norm(gap))
    bound = speed_blind_bound(distance, v)
    controls, goal_pixels, replay_error = [], None, 0.
    env = TwoRoomEnv(render_mode='rgb_array')
    try:
        for si, speed in enumerate(v):
            for kind in ('speed_aware', 'shared'):
                env.reset(seed=int(data['eval_seed']), options={
                    'variation': (), 'variation_values': _factor_options({'agent.speed': float(speed), 'door.position': 49}),
                    'state': data['initial_states'][si].copy(), 'target_state': goal.copy()})
                pixels = env._target_img.numpy().transpose(1, 2, 0).astype(np.uint8)
                if goal_pixels is not None:
                    assert np.array_equal(goal_pixels, pixels)
                goal_pixels = pixels
                for t in range(11):
                    if t in (0, 5, 10):
                        assert np.array_equal(env.render(), data['history_pixels'][si, t//5])
                    if t < 10:
                        env.step(data['context_actions'][si].reshape(10, 2)[t])
                replay_error = max(replay_error, float(np.max(np.abs(env.agent_position.numpy()-data['query_state']))))
                assert replay_error < 2e-5
                action = (gap/(25*speed) if kind == 'speed_aware' else gap/distance*bound['cumulative_action']/25).astype(np.float32)
                assert np.max(np.abs(action)) <= 1
                states, contacts = [env.agent_position.numpy().copy()], []
                for t in range(25):
                    expected = states[-1]+np.float32(speed)*action
                    env.step(action)
                    states.append(env.agent_position.numpy().copy())
                    if float(np.max(np.abs(states[-1]-expected))) > 1e-4:
                        contacts.append(t)
                error = float(np.linalg.norm(states[-1]-goal))
                assert not contacts
                if kind == 'speed_aware':
                    assert error < 1e-3
                controls.append(dict(kind=kind, speed=float(speed), action=action.tolist(),
                                     terminal_distance=error, contacts=contacts, states=np.asarray(states).tolist()))
    finally:
        env.close()
    shared = np.mean([c['terminal_distance'] for c in controls if c['kind'] == 'shared'])
    assert abs(shared-bound['mean_distance']) < 1e-3
    assert bound['maximum_success_fraction'] == 1/3
    data.update(old_goal_state=data['goal_state'].copy(), goal_state=goal, goal_pixels=goal_pixels)
    dest = output/(row['query_id']+'.npz')
    np.savez_compressed(dest, **data)
    return dict(query_id=row['query_id'], path=dest.name, sha256=sha(dest), source_sha256=sha(path),
                eval_seed=row['eval_seed'], evaluation_index=row['evaluation_index'],
                query_to_goal_distance=distance, history_replay_max_error=replay_error,
                history_pixels_exact=True, blind_bound=bound, controls=controls)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    import torch
    torch.set_num_threads(1)
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    load_stable_worldmodel(ROOT, a.stable_repo, a.stable_ref)
    source = json.loads((a.source_panel/'manifest.json').read_text())
    rows = sorted((r for r in source['queries'] if r['evaluation_index'] == 0), key=lambda r: r['eval_seed'])
    assert len(rows) == 6 and {r['eval_seed'] for r in rows} == set(range(42, 48))
    protocol = dict(schema='contextworld.speed_timed_arrival.v1', evaluation_split='development',
                    selection='index0 in each of seeds42-47, six preselected scenes retained',
                    speeds=[3.4, 4.8, 6.9], goal_distance_px=32, goal_rule='toward original shared scene goal',
                    deadline_raw_steps=25, terminal_tolerance_px=2, success='terminal distance<=2 and no contact',
                    contact_rule='native next position differs from collision-free transition by >1e-4',
                    normalization=source['protocol']['normalization'], stable_ref=a.stable_ref,
                    source_manifest_sha256=sha(a.source_panel/'manifest.json'),
                    cem=dict(num_samples=300, n_steps=30, topk=30, var_scale=1., horizon=5, action_block=5, history_len=3),
                    action_handling='full native action sequence; future model inputs clip raw commands then normalize',
                    objective='native latent goal MSE; contact checked only in execution scoring',
                    evidence='one25stepplan; no feedback before deadline; all history controls share query and goal',
                    oracle_scope='all collision-free common open-loop action sequences, including randomized mixtures',
                    oracle_actions_provided_to_planner=False, first_entry_termination_ignored=True,
                    no_training=True, test_payload_accessed=False)
    a.output.mkdir(parents=True)
    write(a.output/'protocol.json', protocol)
    results = [build_query(a.source_panel, r, a.output) for r in rows]
    aware = [c for r in results for c in r['controls'] if c['kind'] == 'speed_aware']
    validation = dict(scenes=6, speed_conditions=18, oracle_success_percent=100.,
                      oracle_terminal_error_max=max(c['terminal_distance'] for c in aware),
                      controls_contact_free=True, history_pixels_exact=True,
                      history_replay_max_error=max(r['history_replay_max_error'] for r in results),
                      optimal_shared_mean_error=float(np.mean([r['blind_bound']['mean_distance'] for r in results])),
                      history_blind_success_ceiling_percent=100/3)
    write(a.output/'manifest.json', dict(protocol=protocol, validation=validation, queries=results))
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
