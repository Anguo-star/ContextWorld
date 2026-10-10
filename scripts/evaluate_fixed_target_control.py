#!/usr/bin/env python3
"""Evaluate a LeWM dynamics checkpoint in the original, fixed T0 latent space."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

CW = Path(__file__).resolve().parents[1]
TASKS = ('contact_friction', 'robot_arm_mass')
SEED = 20261010


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def readout_module():
    directory = CW / 'scripts'
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('diagnose_task_history_readout', directory / 'diagnose_task_history_readout.py')
    readout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readout)
    return readout


def pair_order(data):
    labels = data['labels']
    queries = data['query_ids'].astype(str)
    order = np.lexsort((labels, queries))
    assert len(order) % 2 == 0
    assert np.array_equal(queries[order][::2], queries[order][1::2])
    assert np.all(labels[order][::2] == 0) and np.all(labels[order][1::2] == 1)
    assert np.array_equal(data['source_groups'][order][::2], data['source_groups'][order][1::2])
    return order


def terms(prediction, target):
    p = np.asarray(prediction, np.float64)
    t = np.asarray(target, np.float64)
    return np.square(p - t).sum(axis=-1), np.square(t).sum(axis=-1)


def ratio(prediction, target):
    error, energy = terms(prediction, target)
    assert energy.sum() > 0
    return float(error.sum() / energy.sum())


def bootstrap_delta(old, new, groups, *, ratio_denominator=None, resamples=2000):
    """Paired source-group resampling; positive delta means new is worse."""
    keys, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    old = np.asarray(old, np.float64)
    new = np.asarray(new, np.float64)
    assert old.shape == new.shape == inverse.shape
    sums = np.zeros((len(keys), 3), np.float64)
    np.add.at(sums[:, 0], inverse, new - old)
    np.add.at(sums[:, 1], inverse, 1)
    if ratio_denominator is not None:
        np.add.at(sums[:, 2], inverse, ratio_denominator)
    rng = np.random.default_rng(SEED)
    draw = rng.integers(len(keys), size=(resamples, len(keys)))
    sampled = sums[draw].sum(axis=1)
    estimate = sampled[:, 0] / (sampled[:, 2] if ratio_denominator is not None else sampled[:, 1])
    return {'delta': float((new - old).sum() / (np.asarray(ratio_denominator).sum() if ratio_denominator is not None else len(old))),
            'ci95': np.quantile(estimate, [.025, .975]).tolist(), 'source_groups': len(keys),
            'resamples': resamples, 'seed': SEED}


def visual_state(checkpoint):
    import torch
    raw = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = raw.get('state_dict', raw) if isinstance(raw, dict) else raw
    assert isinstance(state, dict)
    found = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        parts = key.split('.')
        for name in ('encoder', 'projector'):
            if name in parts:
                found['.'.join(parts[parts.index(name):])] = value.detach().cpu()
                break
    assert any(k.startswith('encoder.') for k in found) and any(k.startswith('projector.') for k in found)
    return found


def compare_visual(t0, candidate):
    a, b = visual_state(t0), visual_state(candidate)
    assert set(a) == set(b), f'visual state keys differ: {set(a) ^ set(b)}'
    unequal = [k for k in a if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not a[k].equal(b[k])]
    assert not unequal, f'encoder/projector tensors or buffers changed: {unequal[:8]}'
    return {'tensor_and_buffer_count': len(a), 'bitwise_equal': True}


def read_panel(path, expected_hash):
    assert sha(path) == expected_hash, f'panel hash mismatch: {path}'
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k] for k in raw.files}


def cache(task, split, spec, data, panel_hash):
    base = Path('/tmp/cw-conditional-map-20261010') / task
    meta_path = base / 'original_meta.json'
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    entry = meta.get(f'{split}_latent', {})
    path = base / f'original_{split}.npz'
    if meta.get('model_id') != spec['id'] or meta.get('checkpoint_sha256') != spec['checkpoint_sha256'] or meta.get('state_hash_before') != meta.get('state_hash_after') or not path.exists() or sha(path) != entry.get('sha256'):
        return None
    # Original cache has no panel digest. Recomputed encodings are checked against it below.
    with np.load(path, allow_pickle=False) as raw:
        cached = {k: raw[k] for k in raw.files}
    if cached['target'].shape[0] != len(data['labels']):
        return None
    return cached


def worker(args, spec, manifest):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    readout = readout_module()
    helper = readout.helper()
    adapter = helper.load_adapter(spec, manifest['normalization'], args.output / '_model_cache', args.device)
    adapter.model.eval()
    before = adapter.frozen_state_hash()
    for split in ('training', 'development'):
        entry = manifest['splits'][split]
        data = read_panel(Path(args.panels) / args.task / entry['path'], entry['sha256'])
        order = pair_order(data)
        assert np.array_equal(data['history_pixels'][order, -1][::2], data['history_pixels'][order, -1][1::2]), 'paired current pixels differ'
        assert np.array_equal(data['action_blocks'][order, -1][::2], data['action_blocks'][order, -1][1::2]), 'paired query actions differ'
        history, target = helper.encode_unique(adapter, data['history_pixels'], data['queryfuture_pixels'])
        prediction = helper.predict_all(adapter, history, target[:, None], data['action_blocks'], 'lewm', modes=('free',))['free'][:, 0]
        swapped_history = history.copy()
        swapped_history[order] = history[order][np.arange(len(order)) ^ 1]
        swapped_actions = data['action_blocks'].copy()
        swapped_actions[order, :history.shape[1] - 1] = data['action_blocks'][order][np.arange(len(order)) ^ 1, :history.shape[1] - 1]
        swapped = helper.predict_all(adapter, swapped_history, target[:, None], swapped_actions, 'lewm', modes=('free',))['free'][:, 0]
        np.savez_compressed(args.output / f'_{args.role}_{split}.npz', history=history, target=target, prediction=prediction, swapped=swapped)
    after = adapter.frozen_state_hash()
    assert before == after, 'model state changed during evaluation'
    (args.output / f'_{args.role}_state.json').write_text(json.dumps({'before': before, 'after': after}))


def load_worker(path):
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k] for k in raw.files}


def evaluate_split(data, baseline, candidate):
    order = pair_order(data)
    target = baseline['target']
    assert np.array_equal(target, candidate['target']), 'T0 and candidate encode pixels differently in float32'
    old = baseline['prediction']; new = candidate['prediction']
    groups = data['source_groups'].astype(str)
    response_target = target[order][1::2] - target[order][::2]
    old_response = old[order][1::2] - old[order][::2]
    new_response = new[order][1::2] - new[order][::2]
    pair_groups = groups[order][::2]
    _, energy = terms(old_response, response_target)
    old_response_error, _ = terms(old_response, response_target)
    new_response_error, _ = terms(new_response, response_target)
    old_full, _ = terms(old, target)
    new_full, _ = terms(new, target)
    # B is the conditional mean of the fixed target for each matched query.
    baseline_mean = np.empty_like(target)
    baseline_mean[order[::2]] = baseline_mean[order[1::2]] = (target[order[::2]] + target[order[1::2]]) / 2
    b_error, _ = terms(baseline_mean, target)
    actions_equal = np.array_equal(data['action_blocks'][order][::2], data['action_blocks'][order][1::2])
    swapped_old = baseline['swapped'][order]
    swapped_new = candidate['swapped'][order]
    old_swapped_error, _ = terms(swapped_old, target[order])
    new_swapped_error, _ = terms(swapped_new, target[order])
    def summary(pred, full, swapped):
        swapped_error, _ = terms(swapped, target[order])
        return {'response_nre': ratio(pred[order][1::2] - pred[order][::2], response_target),
                'full_error': float(full.mean()), 'full_error_over_B': float(full.sum() / b_error.sum()),
                'history_benefit': float((swapped_error - full[order]).mean()),
                'history_benefit_over_B': float((swapped_error - full[order]).sum() / b_error.sum())}
    return {'pairs': len(pair_groups), 'rows': len(order), 'action_blocks_equal_per_pair': actions_equal,
            'B_conditional_mean_error': float(b_error.mean()), 'old': summary(old, old_full, swapped_old),
            'new': summary(new, new_full, swapped_new),
            'delta_response_nre': bootstrap_delta(old_response_error, new_response_error, pair_groups, ratio_denominator=energy),
            'delta_full_error': bootstrap_delta(old_full, new_full, groups),
            'delta_full_error_over_B': bootstrap_delta(old_full, new_full, groups, ratio_denominator=b_error),
            'delta_history_benefit_over_B': bootstrap_delta(old_swapped_error - old_full[order],
                new_swapped_error - new_full[order], groups[order], ratio_denominator=b_error[order])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--stablewm-repo', required=True, type=Path)
    p.add_argument('--stable-ref', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    p.add_argument('--panels', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--role', choices=('t0', 'new'), help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.panels / args.task / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    models = json.loads(args.models.read_text())
    t0 = next(m for m in models if m['id'] == f'{args.task}/lewm/original/s3073')
    assert sha(t0['checkpoint']) == t0['checkpoint_sha256']
    assert args.checkpoint.suffix == '.pt' or args.checkpoint == Path(t0['checkpoint'])
    new = dict(t0, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint), stable_repo=str(args.stablewm_repo), stable_ref=args.stable_ref)
    if args.role:
        worker(args, t0 if args.role == 't0' else new, manifest)
        return
    visual = compare_visual(t0['checkpoint'], args.checkpoint)
    for role in ('t0', 'new'):
        command = [sys.executable, str(Path(__file__).resolve()), '--task', args.task, '--checkpoint', str(args.checkpoint),
                   '--stablewm-repo', str(args.stablewm_repo), '--stable-ref', args.stable_ref, '--output', str(args.output),
                   '--models', str(args.models), '--panels', str(args.panels), '--device', args.device, '--role', role]
        subprocess.run(command, check=True)
    result = {'schema': 'contextworld.fixed_target_control.v1', 'task': args.task, 't0_checkpoint_sha256': t0['checkpoint_sha256'],
              'new_checkpoint_sha256': new['checkpoint_sha256'], 'stable_ref': args.stable_ref, 'visual_identity': visual,
              'panel_manifest_sha256': sha(manifest_path), 'splits': {}, 'fixed_target': 'T0 original/s3073 encoder plus projector; float32 latent equality required',
              'no_fitting': True, 'no_test_read': True}
    for split in ('training', 'development'):
        entry = manifest['splits'][split]
        data = read_panel(args.panels / args.task / entry['path'], entry['sha256'])
        baseline = load_worker(args.output / f'_t0_{split}.npz')
        candidate = load_worker(args.output / f'_new_{split}.npz')
        old_cache = cache(args.task, split, t0, data, entry['sha256'])
        cache_comparison = None
        if old_cache is not None:
            cache_comparison = {'target_max_abs_difference': float(np.max(np.abs(old_cache['target'] - baseline['target']))),
                'prediction_max_abs_difference': float(np.max(np.abs(old_cache['prediction'] - baseline['prediction']))) if 'prediction' in old_cache else None}
        result['splits'][split] = {'panel_sha256': entry['sha256'], 't0_cache_sha_and_meta_valid': old_cache is not None,
                                   't0_cache_comparison': cache_comparison,
                                   **evaluate_split(data, baseline, candidate)}
    result['model_state_unchanged'] = {role: json.loads((args.output / f'_{role}_state.json').read_text()) for role in ('t0', 'new')}
    destination = args.output / 'result.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(destination)


if __name__ == '__main__':
    main()
