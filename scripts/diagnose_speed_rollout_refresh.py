#!/usr/bin/env python3
"""Separate free-rollout errors from one-step errors on archived Speed plans.

True-frame and partial latent refresh are privileged diagnostics, not planning
policies. Neither checkpoint weights nor the original scores are changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from build_speed_action_selection import sha, write
from diagnose_speed_timed_arrival import replay, cost_order
from eval_speed_action_selection import load_adapter

MODES = ('free', 'current', 'past', 'full')


def context_window(frames, truth, step, mode):
    """Select aligned H3 input; partial replacements are ephemeral interventions."""
    import torch
    if mode not in MODES:
        raise ValueError(mode)
    window = torch.stack(frames[-3:], dim=1).clone()
    real = truth[:, step:step+3]
    if mode in ('current', 'full'):
        window[:, -1] = real[:, -1]
    if mode in ('past', 'full'):
        window[:, :2] = real[:, :2]
    return window


def infer_modes(model, history, targets, normalized_actions):
    import torch
    h = torch.as_tensor(history)
    y = torch.as_tensor(targets)
    truth = torch.cat([h, y], dim=1)
    with torch.inference_mode():
        actions = model.action_encoder(torch.as_tensor(normalized_actions))
        output = {}
        for mode in MODES:
            frames = list(h.unbind(dim=1))
            for t in range(y.shape[1]):
                window = context_window(frames, truth, t, mode)
                frames.append(model.predict(window, actions[:, t:t+3])[:, -1])
            output[mode] = torch.stack(frames[3:], dim=1).numpy()
    return output


def visual_speed_estimate(pixels, actions):
    """Renderer-specific, training-free red-agent localization, no state input."""
    rgb = np.asarray(pixels, np.int16)
    contrast = rgb[..., 0]-np.maximum(rgb[..., 1], rgb[..., 2])
    positions = []
    for frame in contrast:
        if frame.max() <= 0:
            raise ValueError('No visible red agent')
        yx = np.argwhere(frame == frame.max()).mean(axis=0)
        positions.append(yx[::-1])
    positions = np.asarray(positions)
    u = np.clip(actions, -1, 1).sum(axis=1)
    energy = float(np.square(u).sum())
    if energy <= 0:
        raise ValueError('History actions cannot identify speed')
    speed = float((np.diff(positions, axis=0)*u).sum()/energy)
    return dict(estimated_speed=speed, pixel_positions=positions.tolist(), action_energy=energy)


def evaluate(adapter, data, saved, previous):
    goal = adapter.encode_pixels(data['goal_pixels'][None], batch_size=1)[0]
    rows, arrays = [], {}
    for si in range(3):
        plan = next(p for p in saved['plans'] if p['history_index'] == si)
        old = next(p for p in previous['rows'] if p['speed_index'] == si)
        cem = np.asarray(plan['raw_actions'], np.float32)
        ref = np.tile((data['goal_state']-data['query_state'])/(25*data['speeds'][si]), (25, 1)).astype(np.float32)
        candidates = np.stack([cem, ref])
        true = [replay(data, x, si) for x in candidates]
        for i, name in enumerate(('cem', 'reference')):
            assert np.array_equal(true[i]['states'], np.asarray(old['candidates'][name]['states'], np.float32))
        histories = np.repeat(data['history_pixels'][si:si+1], 2, axis=0)
        history = adapter.encode_pixels(histories.reshape(-1, 224, 224, 3), batch_size=6).reshape(2, 3, -1)
        targets = adapter.encode_pixels(np.stack([x['pixels'] for x in true]).reshape(-1, 224, 224, 3), batch_size=10).reshape(2, 5, -1)
        raw = np.concatenate([np.repeat(data['context_actions'][si:si+1], 2, axis=0),
                              np.clip(candidates, -1, 1).reshape(2, 5, 5, 2)], axis=1)
        outputs = infer_modes(adapter.model, history, targets, adapter._normalize_actions(raw))
        canonical = adapter.rollout_latents(histories, raw, batch_size=2)
        diff = outputs['free']-canonical
        max_abs = float(np.abs(diff).max())
        rel = float(np.linalg.norm(diff)/np.linalg.norm(canonical))
        assert max_abs < 1e-4 and rel < 1e-5, (max_abs, rel)
        # Step 1 never sees a predicted input; all four modes must agree exactly.
        assert all(np.array_equal(x[:, 0], outputs['free'][:, 0]) for x in outputs.values())
        target_cost = np.square(targets-goal).sum(axis=-1)
        initial_error = np.square(targets-history[:, -1:]).sum(axis=-1)
        row = dict(speed_index=si, speed=float(data['speeds'][si]),
            visual_history=visual_speed_estimate(data['history_pixels'][si], data['context_actions'][si]),
            canonical_max_abs=max_abs, canonical_relative_l2=rel,
            target_goal_cost=target_cost.tolist(), persistence_error=initial_error.tolist(),
            physical_distances=[x['distance'] for x in true], contacts=[x['contacts'] for x in true], modes={})
        for name, pred in outputs.items():
            pc = np.square(pred-goal).sum(axis=-1)
            err = np.square(pred-targets).sum(axis=-1)
            if name == 'free':
                old_pc = np.asarray([old['candidates'][n]['predicted_goal_cost_by_step'] for n in ('cem','reference')])
                assert np.allclose(pc, old_pc, atol=1e-4, rtol=2e-5)
                assert cost_order(*pc[:, -1]) == old['predicted_preference']
            row['modes'][name] = dict(prediction_squared_error=err.tolist(), predicted_goal_cost=pc.tolist(),
                                      terminal_preference=cost_order(*pc[:, -1]))
            arrays[f's{si}_{name}'] = pred
        arrays[f's{si}_target'] = targets
        arrays[f's{si}_history'] = history
        arrays[f's{si}_goal'] = goal
        arrays[f's{si}_actions'] = raw
        rows.append(row)
    return rows, arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('panel', 'saved-plan', 'previous-diagnosis', 'checkpoint', 'output', 'protocol'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--family', choices=['lewm'], default='lewm')
    p.add_argument('--threads', type=int, default=2)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    saved=json.loads(a.saved_plan.read_text()); previous=json.loads(a.previous_diagnosis.read_text())
    dest=a.output/(saved['query_id']+'.json')
    if dest.exists():
        raise FileExistsError(dest)
    assert sha(a.checkpoint) == a.expected_sha256 == saved['checkpoint_sha256'] == previous['checkpoint_sha256']
    assert sha(a.panel/'manifest.json') == saved['panel_sha256'] == previous['panel_sha256']
    assert sha(a.saved_plan) == previous['saved_plan_sha256']
    manifest=json.loads((a.panel/'manifest.json').read_text())
    entry=next(x for x in manifest['queries'] if x['query_id']==saved['query_id'])
    assert sha(a.panel/entry['path']) == entry['sha256']
    with np.load(a.panel/entry['path'], allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    adapter=load_adapter(a, manifest)
    before=adapter.frozen_state_hash()
    rows, arrays=evaluate(adapter, data, saved, previous)
    after=adapter.frozen_state_hash(); assert before==after
    npz=dest.with_suffix('.npz'); np.savez_compressed(npz, **arrays)
    predictor=adapter.model.predictor
    architecture=dict(history_tokens=int(predictor.num_frames), positional_tokens=int(predictor.pos_embedding.shape[1]),
        predictor_parameters=sum(p.numel() for p in predictor.parameters()),
        encoder_parameters=sum(p.numel() for p in adapter.model.encoder.parameters()))
    write(dest, dict(query_id=saved['query_id'], checkpoint_sha256=a.expected_sha256, panel_sha256=saved['panel_sha256'],
        source_data_sha256=entry['sha256'], saved_plan_sha256=sha(a.saved_plan), previous_diagnosis_sha256=sha(a.previous_diagnosis),
        protocol_sha256=sha(a.protocol), array_sha256=sha(npz), architecture=architecture, rows=rows,
        model=adapter.metadata, state_hash_before=before, state_hash_after=after, no_training=True,
        new_searches=0, evaluation_split='development'))
    print(saved['query_id'], 'complete', flush=True)

if __name__=='__main__':
    main()
