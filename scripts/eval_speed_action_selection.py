#!/usr/bin/env python3
"""Frozen LeWM/PLDM evaluation on a prebuilt Speed action-selection panel.

The cached one-step predictor path is checked against the public adapter on the
first and last query of every shard. All raw candidate costs are retained.
"""
from __future__ import annotations
from build_speed_action_selection import sha, write
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_adapter(a, manifest):
    import torch
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260929)
    sys.modules.setdefault('flash_attn', None)
    from contextworld.benchmarks.adapters import StableWorldModelLeWMAdapter, StableWorldModelPLDMAdapter, _load_model
    from contextworld.evaluation.protocol import ColumnStandardizer
    from contextworld.synthesis.stablewm import load_stable_worldmodel
    swm, repo, commit = load_stable_worldmodel(ROOT, a.stable_repo, a.stable_ref)
    if a.checkpoint.suffix == '.pt':
        model = swm.wm.utils.load_pretrained(
            str(a.checkpoint), cache_dir=str(a.output/'model_cache'))
    else:
        model = _load_model(a.checkpoint, stable_worldmodel=swm, stable_repo=repo,
                            repo_root=ROOT, model_config_name=a.family, action_input_dim=10)
    cls = StableWorldModelLeWMAdapter if a.family == 'lewm' else StableWorldModelPLDMAdapter
    norm = manifest['protocol']['normalization']
    return cls(model=model, checkpoint=a.checkpoint, stable_repo=repo, stable_commit=commit, device='cpu', action_standardizer=ColumnStandardizer(np.asarray(norm['mean'], np.float32)[None], np.asarray(norm['std'], np.float32)[None]))


def infer(adapter, data, canonical=False):
    import torch
    histories = data['history_pixels']
    C = len(histories)
    K = len(data['candidate_actions'])
    h = adapter.encode_pixels(
        histories.reshape(-1, *histories.shape[-3:]), batch_size=16).reshape(C, 3, -1)
    goal = adapter.encode_pixels(data['goal_pixels'][None], batch_size=16)[0]
    future = adapter.encode_pixels(
        data['future_pixels'].reshape(-1, *histories.shape[-3:]), batch_size=16).reshape(C, K, -1)
    raw = np.broadcast_to(data['actions'], (K, 3, 5, 2)).copy()
    raw[:, 2] = data['candidate_actions']
    norm = adapter._normalize_actions(raw)
    predictions = []
    with torch.inference_mode():
        for si in range(C):
            pred = []
            for start in range(0, K, 16):
                aa = torch.from_numpy(norm[start:start+16])
                hh = torch.from_numpy(h[si]).unsqueeze(0).expand(len(aa), -1, -1)
                pred.append(adapter.model.predict(
                    hh, adapter.model.action_encoder(aa))[:, -1].numpy())
            predictions.append(np.concatenate(pred))
    predictions = np.asarray(predictions)
    pc = ((predictions-goal)**2).mean(axis=-1)
    ec = ((future-goal)**2).mean(axis=-1)
    check = None
    if canonical:
        pixel_batch = np.repeat(histories, K, axis=0)
        expected = adapter.rollout_latents(pixel_batch, np.tile(
            raw, (C, 1, 1, 1)), batch_size=8)[:, 0].reshape(C, K, -1)
        maximum = float(np.max(np.abs(expected-predictions)))
        assert np.allclose(expected, predictions, atol=2e-5, rtol=2e-5), maximum
        expected_cost = ((expected-goal)**2).mean(axis=-1)
        assert np.array_equal(np.argmin(expected_cost, axis=-1), np.argmin(pc, axis=-1))
        check = {'max_latent_difference': maximum, 'selected_candidates_identical': True}
    return pc, ec, predictions[:, 20], future[:, 20], check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stable-repo', required=True)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--family', choices=['lewm', 'pldm'], default='lewm')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    assert sha(a.checkpoint) == a.expected_sha256, 'checkpoint differs'
    manifest = json.loads((a.panel/'manifest.json').read_text())
    panel_sha = sha(a.panel/'manifest.json')
    a.output.mkdir(parents=True, exist_ok=True)
    adapter = load_adapter(a, manifest)
    before = adapter.frozen_state_hash()
    indices = list(range(a.shard, len(manifest['queries']), a.shards))
    assert indices
    checks = []
    for ni, idx in enumerate(indices):
        entry = manifest['queries'][idx]
        path = a.panel/entry['path']
        assert sha(path) == entry['sha256']
        data = np.load(path, allow_pickle=False)
        pred, true, native, target, check = infer(
            adapter, data, canonical=ni in (0, len(indices)-1))
        dest = a.output/entry['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, predicted_cost=pred, encoded_true_cost=true,
                            physical_cost=data['physical_cost'], native_prediction=native, native_target=target, checkpoint_sha256=a.expected_sha256, panel_sha256=panel_sha, query_id=entry['query_id'])
        if check is not None:
            checks.append(dict(query_id=entry['query_id'], **check))
        if (ni+1) % 20 == 0:
            print(a.shard, ni+1, len(indices), flush=True)
    after = adapter.frozen_state_hash()
    assert before == after
    write(a.output/f'receipt_{a.shard:02d}.json', dict(completed=True, shard=a.shard, shards=a.shards, queries=len(indices), checkpoint=adapter.metadata,
          checkpoint_sha256=a.expected_sha256, panel_sha256=panel_sha, state_hash_before=before, state_hash_after=after, canonical_checks=checks))
    print('complete', a.shard, len(indices), flush=True)


if __name__ == '__main__':
    main()
