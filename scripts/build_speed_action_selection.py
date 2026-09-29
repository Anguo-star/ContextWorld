#!/usr/bin/env python3
"""Build a fixed-candidate Speed panel from the registered Development payload.

All existing Development queries are retained. The candidate/goal rule is fixed
before model inference. No training or Test payload is read. Each saved query
contains all hidden speeds in its track, sharing the current state and goal.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TRACKS = ('seen_for_multi', 'unseen_interpolation', 'extrapolation_low', 'extrapolation_high')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def load_source(path):
    import lance
    from PIL import Image
    table = lance.dataset(path)
    meta = table.to_table(filter='step_idx = 10', columns=[
                          'episode_idx', 'dev_static_query_id', 'dev_reference_speed', 'dev_condition_speed', 'dev_eval_seed', 'dev_evaluation_index']).to_pylist()
    selected = [r for r in meta if np.isclose(
        r['dev_reference_speed'][0], r['dev_condition_speed'][0], atol=1e-5)]
    assert len(selected) == 300 and len({r['dev_static_query_id'] for r in selected}) == 300
    indices = [r['episode_idx']*40+i for r in selected for i in range(16)]
    rows = table.take(indices, columns=['episode_idx', 'step_idx',
                      'pixels', 'action', 'state', 'goal_state']).to_pylist()
    result = {}
    for j, m in enumerate(selected):
        rr = rows[j*16:(j+1)*16]
        assert [r['step_idx'] for r in rr] == list(range(16))
        assert all(r['episode_idx'] == m['episode_idx'] for r in rr)
        frames = np.stack(
            [np.asarray(Image.open(io.BytesIO(rr[i]['pixels'])).convert('RGB')) for i in (0, 5, 10, 15)])
        result[m['dev_static_query_id']] = dict(frames=frames, states=np.asarray([r['state'] for r in rr], np.float32), actions=np.asarray([r['action'] for r in rr[:15]], np.float32).reshape(3, 5, 2), old_goal=np.asarray(
            rr[0]['goal_state'], np.float32), speed=round(float(m['dev_condition_speed'][0]), 4), eval_seed=int(m['dev_eval_seed'][0]), evaluation_index=int(m['dev_evaluation_index'][0]), source_episode=int(m['episode_idx']))
    return result


def init_worker(source_dir, stable_repo, stable_ref):
    global SOURCES, ENV
    import torch
    torch.set_num_threads(1)
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    load_stable_worldmodel(ROOT, stable_repo, stable_ref)
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    ENV = TwoRoomEnv(render_mode='rgb_array')
    SOURCES = Path(source_dir)


def build_query(item):
    from contextworld.evaluation.icl_catalog import _factor_options
    track, qid, output = item
    data = np.load(SOURCES/track/f'{qid}.npz', allow_pickle=False)
    histories = data['frames'][:, :3]
    actions = data['actions']
    states = data['states']
    speeds = data['speeds']
    assert np.all(histories[:, -1] == histories[0, -1])
    assert np.allclose(states[:, 10], states[0, 10], atol=1e-5, rtol=0)
    assert np.array_equal(actions, np.broadcast_to(actions[0], actions.shape))
    # Candidate construction uses only the shared original action sequence.
    grid = np.linspace(0, 1.5, 31, dtype=np.float32)
    candidates = grid[:, None, None]*actions[0, 2][None]
    assert np.max(np.abs(candidates)) <= 1
    goal = states[0, 15].copy()  # slowest condition's original-action endpoint
    query = states[0, 10].copy()
    futures = []
    final_states = []
    replay_state_max = 0.
    replay_pixel_max = 0
    for si, speed in enumerate(speeds):
        ENV.reset(seed=int(data['eval_seed']), options={'variation': (), 'variation_values': _factor_options({'agent.speed': float(
            speed), 'door.position': 49}), 'state': states[si, 0].copy(), 'target_state': data['old_goal'][si].copy()})
        for j in range(16):
            replay_state_max = max(replay_state_max, float(
                np.max(np.abs(ENV.agent_position.numpy()-states[si, j]))))
            if j in (0, 5, 10, 15):
                pix = ENV.render().copy()
                replay_pixel_max = max(replay_pixel_max, int(
                    np.abs(pix.astype(np.int16)-data['frames'][si, j//5].astype(np.int16)).max()))
            if j < 15:
                ENV.step(actions[si].reshape(15, 2)[j])
        assert replay_state_max < 2e-5, (track, qid, 'state replay', replay_state_max)
        assert replay_pixel_max == 0, (track, qid, 'pixel replay', replay_pixel_max)
        fp = []
        fs = []
        for ca in candidates:
            ENV._set_state(query.copy())
            for action in ca:
                ENV.step(action)
            fs.append(ENV.agent_position.numpy().copy())
            fp.append(ENV.render().copy())
        assert np.array_equal(fp[0], histories[si, -1])
        assert np.array_equal(fp[20], data['frames'][si, 3])
        futures.append(fp)
        final_states.append(fs)
    final_states = np.asarray(final_states, np.float32)
    physical = np.linalg.norm(final_states.astype(np.float64)-goal.astype(np.float64), axis=-1)
    best = np.min(physical, axis=-1)
    blind = float(np.min(physical.mean(axis=0))-best.mean())
    dest = Path(output)/track/f'{qid}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, history_pixels=histories, actions=actions[0], candidate_actions=candidates, amplitude_grid=grid, future_pixels=np.asarray(
        futures, np.uint8), goal_pixels=data['frames'][0, 3], physical_cost=physical, query_state=query, goal_state=goal, future_states=final_states, speeds=speeds, query_id=qid)
    return {'track': track, 'query_id': qid, 'path': f'{track}/{qid}.npz', 'sha256': sha(dest), 'eval_seed': int(data['eval_seed']), 'evaluation_index': int(data['evaluation_index']), 'speeds': speeds.tolist(), 'history_blind_regret_lower_bound': blind, 'oracle_candidate_cost_mean': float(best.mean()), 'optimal_candidates': np.argmin(physical, axis=-1).tolist(), 'replay_state_max': replay_state_max, 'replay_pixel_max': replay_pixel_max}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    from contextworld.benchmarks.bundle_development import resolve_development_payload, development_action_normalization
    payload = resolve_development_payload(a.benchmark_root, task='speed')
    if (a.output / "manifest.json").exists():
        raise FileExistsError("Completed panel already exists; choose a new output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    source = a.output/'source'
    source.mkdir(exist_ok=True)
    protocol = {'schema': 'contextworld.speed_action_selection.v1', 'evaluation_split': 'development', 'tracks': list(TRACKS), 'source_manifest_sha256': payload.manifest_sha256, 'task_registry_sha256': payload.task_registry_sha256, 'source_members': [str(p.relative_to(a.benchmark_root)) for p in payload.members], 'candidate_amplitudes': np.linspace(0, 1.5, 31).tolist(
    ), 'candidate_rule': 'shared original five-step query actions multiplied by amplitude', 'goal_rule': 'slowest track speed original-query endpoint, shared across speeds', 'query_rule': 'all 300 registered Development static queries per track; no score filtering', 'cost': 'Euclidean final agent-to-goal distance in pixels', 'history_control': 'each other speed history in same track; average over all wrong histories', 'normalization': dict(zip(('mean', 'std'), development_action_normalization(payload))), 'stable_repo': a.stable_repo, 'stable_ref': a.stable_ref, 'raw_steps': 5, 'pilot': a.pilot, 'no_training': True, 'test_payload_accessed': False}
    write(a.output/'protocol.json', protocol)
    jobs = []
    for track in TRACKS:
        sources = [load_source(p)
                   for p in payload.members if p.name.startswith(f'twmsdev-{track}-')]
        sources.sort(key=lambda x: next(iter(x.values()))['speed'])
        ids = sorted(sources[0])
        assert all(set(x) == set(ids) for x in sources)
        if a.pilot:
            ids = [ids[0], ids[-1]]
        (source/track).mkdir(exist_ok=True)
        for qid in ids:
            rr = [x[qid] for x in sources]
            assert len({x['eval_seed'] for x in rr}) == 1
            np.savez_compressed(source/track/f'{qid}.npz', frames=np.stack([x['frames'] for x in rr]), states=np.stack([x['states'] for x in rr]), actions=np.stack(
                [x['actions'] for x in rr]), old_goal=np.stack([x['old_goal'] for x in rr]), speeds=np.asarray([x['speed'] for x in rr]), eval_seed=rr[0]['eval_seed'], evaluation_index=rr[0]['evaluation_index'])
            jobs.append((track, qid, str(a.output)))
        print('source ready', track, len(ids), flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker, initargs=(str(source), a.stable_repo, a.stable_ref)) as pool:
        records = []
        for i, r in enumerate(pool.map(build_query, jobs, chunksize=4)):
            records.append(r)
            if (i+1) % 50 == 0:
                print('built', i+1, len(jobs), flush=True)
    summary = {}
    for track in TRACKS:
        rr = [r for r in records if r['track'] == track]
        summary[track] = {'queries': len(rr), 'physical_conditions': len(rr)*len(rr[0]['speeds']), 'history_blind_regret_lower_bound': float(np.mean([r['history_blind_regret_lower_bound'] for r in rr])),
                          'fraction_requiring_different_candidates': float(np.mean([len(set(r['optimal_candidates'])) > 1 for r in rr])), 'oracle_candidate_cost_mean': float(np.mean([r['oracle_candidate_cost_mean'] for r in rr]))}
    write(a.output/'manifest.json', {'protocol': protocol, 'protocol_sha256': sha(
        a.output/'protocol.json'), 'summary': summary, 'queries': records})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
