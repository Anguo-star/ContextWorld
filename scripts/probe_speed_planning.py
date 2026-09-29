#!/usr/bin/env python3
"""Diagnose action-range and rollout-depth effects with frozen Speed models.

Build a common candidate bank before scoring. Clipping changes only the model
input for the same physical futures; the small-action arm changes the physical
trajectory as well. These contrasts do not modify the benchmark or CEM policy.
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


def candidate_banks(a, entry, data, norm):
    import torch
    seed = int(np.random.SeedSequence([int(data['eval_seed']), int(data['evaluation_index'])]).generate_state(1)[0])
    random = torch.randn(1, 300, 5, 10, generator=torch.Generator().manual_seed(seed)).numpy()[0].reshape(300, 5, 5, 2)
    random[0] = 0  # native CEM's first candidate is its initial mean
    random = random*np.asarray(norm['std'], np.float32)+np.asarray(norm['mean'], np.float32)
    selected, identities = [], []
    for arm in ('T0', 'T1'):
        for si in range(3):
            for hi in range(3):
                path = a.pilot_root/arm/f"{entry['query_id']}_s{si}_h{hi}.json"
                result = json.loads(path.read_text())
                assert result['panel_sha256'] == sha(a.cem_panel/'manifest.json')
                selected.append(np.asarray(result['result']['plans'][0]['raw_actions'], np.float32).reshape(5, 5, 2))
                identities.append(dict(path=f'{arm}/{path.name}', sha256=sha(path)))
    issued = np.concatenate([random, np.stack(selected)]).astype(np.float32)
    clipped = np.clip(issued, -1, 1)
    source = a.candidate_panel/'unseen_interpolation'/f"{entry['query_id']}.npz"
    manifest = json.loads((a.candidate_panel/'manifest.json').read_text())
    source_row = next(x for x in manifest['queries'] if x['track'] == 'unseen_interpolation' and x['query_id'] == entry['query_id'])
    assert sha(source) == source_row['sha256']
    with np.load(source, allow_pickle=False) as old:
        structured = np.repeat(old['candidate_actions'][:, None], 5, axis=1)
        original_future = old['future_pixels'].copy()
    return dict(structured=structured, issued=issued, clipped=clipped, small=clipped*np.float32(.525)), original_future, identities


def simulate(data, actions):
    from contextworld.evaluation.icl_catalog import _factor_options
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    env = TwoRoomEnv(render_mode='rgb_array')
    frames, states = [], []
    try:
        for si, speed in enumerate(data['speeds']):
            pf, ps = [], []
            for candidate in actions:
                env.reset(seed=int(data['eval_seed']), options={
                    'variation': (), 'variation_values': _factor_options({'agent.speed': float(speed), 'door.position': 49}),
                    'state': data['initial_states'][si].copy(), 'target_state': data['goal_state'].copy()})
                for action in data['context_actions'][si].reshape(10, 2):
                    env.step(action)
                assert np.array_equal(env.render(), data['history_pixels'][si, -1])
                ff, ss = [], []
                for block in candidate:
                    for action in block:
                        env.step(action)
                    ff.append(env.render().copy())
                    ss.append(env.agent_position.numpy().copy())
                pf.append(ff)
                ps.append(ss)
            frames.append(pf)
            states.append(ps)
    finally:
        env.close()
    return np.asarray(frames, np.uint8), np.asarray(states, np.float32)


def build(a):
    import torch
    torch.set_num_threads(a.threads)
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    load_stable_worldmodel(ROOT, a.stable_repo, a.stable_ref)
    source = json.loads((a.cem_panel/'manifest.json').read_text())
    entry = next(x for x in source['queries'] if x['query_id'] == a.query_id)
    assert entry['evaluation_index'] == 0
    path = a.cem_panel/entry['path']
    assert sha(path) == entry['sha256']
    data = np.load(path, allow_pickle=False)
    banks, expected, identities = candidate_banks(a, entry, data, source['protocol']['normalization'])
    output = a.output/a.query_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'context.npz', **{k: data[k] for k in data.files})
    records = {}
    for name in ('structured', 'issued', 'small'):
        pixels, states = simulate(data, banks[name])
        if name == 'structured':
            assert np.array_equal(pixels[:, :, 0], expected), 'Original one-block positive control differs'
        if name == 'issued':
            # Native environment clips every command. Check equivalence independently.
            indices = [0, 1, 100, 299, 317]
            clipped_pixels, clipped_states = simulate(data, banks['clipped'][indices])
            assert np.array_equal(pixels[:, indices], clipped_pixels)
            assert np.array_equal(states[:, indices], clipped_states)
        cost = np.linalg.norm(states-data['goal_state'], axis=-1)
        dest = output/f'{name}.npz'
        np.savez_compressed(dest, actions=banks[name], future_pixels=pixels, future_states=states, physical_cost=cost)
        records[name] = dict(path=dest.name, sha256=sha(dest), candidates=len(banks[name]))
        print(a.query_id, name, len(banks[name]), flush=True)
    protocol = dict(schema='contextworld.speed_planning_horizon_probe.v1', evaluation_split='development',
                    normalization=source['protocol']['normalization'], query_id=a.query_id,
                    cem_panel_sha256=sha(a.cem_panel/'manifest.json'), stable_ref=a.stable_ref,
                    horizons_raw_steps=[5, 10, 15, 20, 25],
                    candidates='31 original-direction candidates; 300 initial CEM samples plus all 18 archived first plans',
                    arms={'structured': 'repeat the original five-step block along its query direction',
                          'issued': 'native CEM raw commands; simulator applies its native clipping',
                          'clipped': 'same physical futures as issued; clip only future model inputs to [-1,1]',
                          'small': 'scale clipped actions by 0.525 for both model and simulation'},
                    comparison='all checkpoints and histories score the same saved candidates and goals',
                    selection='evaluation_index=0 in each of the six Development seeds, no performance filtering',
                    state_observation='physical state used only by simulator and scorer; predictor sees images and actions',
                    no_training=True, test_payload_accessed=False)
    write(output/'manifest.json', dict(protocol=protocol, records=records, context_sha256=sha(output/'context.npz'),
                                     source_plans=identities, original_one_block_reproduced=True, clipping_equivalence_verified=True,
                                     clipping_equivalence_checked_candidate_indices=[0, 1, 100, 299, 317],
                                     clipping_equivalence_basis="native simulator clips all actions; listed candidates independently replayed"))


def predict(adapter, context, actions):
    import torch
    h = torch.from_numpy(adapter.encode_pixels(context['history_pixels'].reshape(-1, 224, 224, 3), batch_size=16).reshape(3, 3, -1))
    outputs, checks = [], []
    with torch.inference_mode():
        for hi in range(3):
            raw = np.concatenate([np.broadcast_to(context['context_actions'][hi], (len(actions), 2, 5, 2)), actions], axis=1)
            norm = adapter._normalize_actions(raw)
            pred = []
            for start in range(0, len(actions), 32):
                aa = adapter.model.action_encoder(torch.from_numpy(norm[start:start+32]))
                frames = list(h[hi][None].expand(len(aa), -1, -1).unbind(dim=1))
                for t in range(5):
                    frames.append(adapter.model.predict(torch.stack(frames[-3:], dim=1), aa[:, t:t+3])[:, -1])
                pred.append(torch.stack(frames[3:], dim=1).numpy())
            prediction = np.concatenate(pred)
            check = adapter.rollout_latents(np.repeat(context['history_pixels'][hi:hi+1], 2, axis=0), raw[[0, -1]], batch_size=2)
            error = check-prediction[[0, -1]]
            max_abs = np.max(np.abs(error), axis=(0, 2))
            relative_l2 = np.sqrt(np.square(error).sum(axis=(0, 2))/np.square(check).sum(axis=(0, 2)))
            # CPU kernels vary slightly with encoder/predictor batch size;
            # bound absolute AND relative error at every autoregressive depth.
            assert np.all(max_abs < 1e-4) and np.all(relative_l2 < 1e-5), (max_abs, relative_l2)
            checks.append(dict(history_index=hi, candidate_indices=[0, len(actions)-1],
                               max_abs_by_depth=max_abs.tolist(), relative_l2_by_depth=relative_l2.tolist()))
            outputs.append(prediction)
    return np.asarray(outputs), checks


def evaluate(a):
    manifest = json.loads((a.panel/a.query_id/'manifest.json').read_text())
    assert sha(a.checkpoint) == a.expected_sha256
    assert sha(a.panel/a.query_id/'context.npz') == manifest['context_sha256']
    a.output.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(a, manifest)
    before = adapter.frozen_state_hash()
    context = np.load(a.panel/a.query_id/'context.npz', allow_pickle=False)
    start = adapter.encode_pixels(context['history_pixels'][:, -1], batch_size=16)
    goal = adapter.encode_pixels(context['goal_pixels'][None], batch_size=16)[0]
    dest = a.output/a.query_id
    dest.mkdir(exist_ok=False)
    checks = {}
    for bank in ('structured', 'issued', 'small'):
        row = manifest['records'][bank]
        path = a.panel/a.query_id/row['path']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as data:
            physical = data['physical_cost'].copy()
            actions = data['actions'].copy()
            pixels = data['future_pixels']
            target = adapter.encode_pixels(pixels.reshape(-1, 224, 224, 3), batch_size=32).reshape(3, len(actions), 5, -1)
        for arm in (('issued', 'clipped') if bank == 'issued' else (bank,)):
            model_actions = np.clip(actions, -1, 1) if arm == 'clipped' else actions
            predictions, checks[arm] = predict(adapter, context, model_actions)
            np.savez_compressed(dest/f'{arm}.npz', prediction=predictions, target=target, initial=start, goal=goal,
                                physical_cost=physical, model_actions=model_actions)
            print(a.query_id, arm, 'scored', flush=True)
    after = adapter.frozen_state_hash()
    assert before == after
    write(dest/'receipt.json', dict(checkpoint_sha256=a.expected_sha256, model=adapter.metadata,
                                   source_manifest_sha256=sha(a.panel/a.query_id/'manifest.json'),
                                   state_hash_before=before, state_hash_after=after, canonical_rollout_checks_passed=True, canonical_rollout_checks=checks,
                                   canonical_tolerances=dict(max_abs=1e-4, relative_l2=1e-5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['build', 'evaluate'])
    p.add_argument('--cem-panel', type=Path)
    p.add_argument('--candidate-panel', type=Path)
    p.add_argument('--pilot-root', type=Path)
    p.add_argument('--panel', type=Path)
    p.add_argument('--query-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--expected-sha256')
    p.add_argument('--family', choices=['lewm'], default='lewm')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    (build if a.mode == 'build' else evaluate)(a)


if __name__ == '__main__':
    main()
