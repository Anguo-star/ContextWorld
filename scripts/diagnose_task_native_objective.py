#!/usr/bin/env python3
"""No-update, Training-only native visual-objective diagnostic across tasks.

The input is an extracted raw RGB/action cache. One query contains all its
conditions; selection takes the first 16 distinct Training source groups in
manifest order. Model outputs never influence selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import diagnose_delay_native_objective as generic

NAMES = ('query_response', 'query_common', 'earlier_native', 'query_native', 'native_all')


def _prediction(adapter, family, history, future, raw_actions):
    import torch
    model = adapter.model
    count, horizon = history.shape[:2]
    h = torch.as_tensor(history, dtype=torch.float32, device=adapter.device).detach()
    y = torch.as_tensor(future, dtype=torch.float32, device=adapter.device).detach()
    action = torch.as_tensor(adapter._normalize_actions(raw_actions), device=adapter.device)
    target = torch.cat((h[:, 1:], y), dim=1).detach()
    if family == 'dinowm':
        if tuple(model.extra_encoders.keys()) != ('action',):
            raise ValueError(f'DINO requires only action stream: {tuple(model.extra_encoders)}')
        channels = int(model.backbone.config.hidden_size)
        if h.shape[-1] % channels or y.shape[-1] != h.shape[-1]:
            raise ValueError('DINO visual latent must contain complete equal-size patches')
        patches = h.shape[-1] // channels
        visual = h.reshape(count, horizon, patches, channels)
        target = target.reshape(count, horizon, patches, channels)
        action_emb = model.extra_encoders['action'](action)
        predictor_input = torch.cat((visual, action_emb[:, :, None, :].expand(-1, -1, patches, -1)), -1)
        prediction = model.predict(predictor_input)[..., :channels]
        endpoint = prediction[:, -1].reshape(count, -1)
    elif family in ('lewm', 'pldm'):
        prediction = model.predict(h, model.action_encoder(action))
        endpoint = prediction[:, -1]
    else:
        raise ValueError(f'Unsupported native visual objective family: {family}')
    if prediction.shape != target.shape:
        raise ValueError(f'Native output shape {prediction.shape} differs from shifted target {target.shape}')
    return prediction, target, endpoint.detach().cpu().numpy()


def _endpoint_check(direct, canonical, label):
    delta = direct - canonical
    max_abs = float(np.abs(delta).max())
    relative = float(np.linalg.norm(delta) / max(np.linalg.norm(canonical), 1e-12))
    if max_abs >= generic.CANONICAL_MAX_ABS or relative >= generic.CANONICAL_RELATIVE_L2:
        raise RuntimeError(f'{label} endpoint mismatch: max_abs={max_abs}, relative_l2={relative}')
    return {'max_abs': max_abs, 'relative_l2': relative}


def evaluate(adapter, helper, family, history, future, actions, named_parameters, query_id):
    import torch
    count, horizon = history.shape[:2]
    if count < 2 or future.shape[:2] != (count, 1) or actions.shape[:2] != (count, horizon):
        raise ValueError(f'Invalid query shapes: {history.shape}, {future.shape}, {actions.shape}')
    prediction, target, direct = _prediction(adapter, family, history, future, actions)
    d = generic._loss_decomposition(prediction, target)
    losses = dict(zip(NAMES, (
        d['response_by_position'][-1]/horizon,
        d['common_by_position'][-1]/horizon,
        d['native_by_position'][:-1].sum()/horizon,
        d['native_by_position'][-1]/horizon,
        d['native'],
    )))
    loss_residual = float((losses['native_all'] - sum(losses[n] for n in NAMES[:3])).detach().cpu())
    position_residual = float((d['native_by_position']-d['response_by_position']-d['common_by_position']).abs().max().detach().cpu())
    if max(abs(loss_residual), position_residual) > generic.MSE_CLOSURE_ATOL:
        raise RuntimeError(f'{query_id}: native loss decomposition failed')
    canonical = helper.predict_all(adapter, history, future, actions, family, modes=('free',))['free'][:, 0]
    canonical_check = _endpoint_check(direct, canonical, 'canonical latent')
    params = tuple(p for _, p in named_parameters)
    gradients = {name: torch.autograd.grad(losses[name], params, retain_graph=index < len(NAMES)-1, allow_unused=True)
                 for index, name in enumerate(NAMES)}
    closure_final = generic._gradient_closure(gradients['query_native'], [gradients['query_response'], gradients['query_common']])
    closure_all = generic._gradient_closure(gradients['native_all'], [gradients[n] for n in NAMES[:3]])
    for closure in (closure_final, closure_all):
        if not math.isfinite(closure['relative_to_component_norm_sum']) or closure['relative_to_component_norm_sum'] > generic.GRADIENT_CLOSURE_RELATIVE_MAX or closure['max_abs'] > 1e-5:
            raise RuntimeError(f'{query_id}: gradient closure failed: {closure}')
    scalar = lambda x: float(x.detach().cpu())
    summary = generic._pairwise_gradient_summary(gradients)
    response_norm = summary['norms']['query_response']
    native_norm = summary['norms']['native_all']
    response_pair = summary['pairs']['query_response__native_all']
    target_energy = scalar(d['target_response_energy_by_position'][-1])
    row = {
        'query_id': str(query_id), 'condition_count': count, 'position_count': horizon,
        'losses': {n: scalar(v) for n, v in losses.items()},
        'query_response_loss_share': scalar(losses['query_response']/losses['native_all']) if scalar(losses['native_all']) > 0 else None,
        'native_query_response_nre': scalar(d['response_by_position'][-1])/target_energy if target_energy > 0 else None,
        'loss_reconstruction_residual': loss_residual, 'position_decomposition_max_abs': position_residual,
        'per_position': {n: d[n].detach().cpu().tolist() for n in d if n != 'native'},
        'canonical_adapter_endpoint': canonical_check, 'gradient': summary,
        'gradient_reconstruction': {'query_native': closure_final, 'native_all': closure_all},
        'query_response_to_native_gradient_norm_ratio': response_norm/native_norm if native_norm > 0 else None,
        'query_response_native_gradient_dot': response_pair['dot'],
        'query_response_native_gradient_cosine': response_pair['cosine'],
        'query_response_native_gradient_negative_cosine': response_pair['cosine'] is not None and response_pair['cosine'] < 0,
    }
    return row, gradients, direct


def _cache_key(raw, *names):
    for name in names:
        if name in raw.files:
            return name
    raise KeyError(f'Cache lacks any of {names}; available keys: {raw.files}')


def _select(raw, manifest, split_name):
    query_key = _cache_key(raw, 'query_ids', 'pair_ids')
    group_key = _cache_key(raw, 'source_groups')
    split = manifest.get('splits', {}).get(split_name, {})
    fit = split.get('query_ids', split.get('pair_ids', manifest.get('train_query_ids')))
    if fit is None:
        raise ValueError(f'{split_name} manifest must list selected query IDs')
    first_by_group = {}
    query_ids = raw[query_key].astype(str)
    for query_id in fit:
        indices = np.flatnonzero(query_ids == str(query_id))
        if len(indices) < 2:
            raise ValueError(f'{query_id}: expected multiple condition rows')
        these_groups = set(raw[group_key][indices].astype(str).tolist())
        if len(these_groups) != 1:
            raise ValueError(f'{query_id}: inconsistent source group')
        group = next(iter(these_groups))
        first_by_group.setdefault(group, str(query_id))
    ordered_groups = sorted(first_by_group, key=lambda group: hashlib.sha256(f'20261010:{group}'.encode()).hexdigest())
    groups = ordered_groups[:16]
    chosen = [first_by_group[group] for group in groups]
    if len(chosen) != 16:
        raise ValueError(f'Need 16 distinct {split_name} source groups; got {len(chosen)}')
    return query_key, chosen, groups, len(fit)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--id', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--split', choices=('training', 'development'), default='development')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    specs = json.loads(args.models.read_text())
    spec = next((s for s in specs if s['id'] == args.id), None)
    if spec is None:
        raise ValueError(f'Model ID not found: {args.id}')
    if spec['family'] not in ('lewm', 'pldm', 'dinowm'):
        raise ValueError(f'Native visual objective unavailable for {args.id}')
    manifest_path = args.cache/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    raw_path = args.cache/manifest.get('splits', {}).get(args.split, {}).get('path', f'{args.split}.npz')
    raw_sha = generic._sha256(raw_path)
    expected_sha = manifest.get('splits', {}).get(args.split, {}).get('sha256', manifest.get(f'{args.split}_sha256'))
    if expected_sha is None or raw_sha != expected_sha:
        raise ValueError(f'{args.split} cache hash missing or mismatched')
    raw = np.load(raw_path, allow_pickle=False)
    query_key, chosen, groups, fit_count = _select(raw, manifest, args.split)
    history_key = _cache_key(raw, 'history_pixels')
    future_key = _cache_key(raw, 'future_pixels', 'queryfuture_pixels')
    action_key = _cache_key(raw, 'action_blocks', 'rawactions')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'result.json').exists():
        raise FileExistsError('Result exists')
    selection = {'split': args.split, 'seed': 20261010,
                 'algorithm': 'SHA256(seed:source_group) order; first manifest query per selected group; 16 distinct groups',
                 'fit_query_count': fit_count, 'query_ids': chosen, 'source_groups': groups,
                 'manifest_sha256': generic._sha256(manifest_path), 'raw_sha256': raw_sha}
    generic._write_json(args.output/'selection.json', selection)
    helper = generic._load_helper()
    adapter = helper.load_adapter(spec, manifest['normalization'], args.output, args.device)
    model = adapter.model
    model.eval()
    before = adapter.frozen_state_hash()
    flags = [p.requires_grad for p in model.parameters()]
    for p in model.parameters(): p.requires_grad_(False)
    model.predictor.requires_grad_(True)
    named = [('predictor.'+n, p) for n, p in model.predictor.named_parameters() if p.requires_grad]
    if not named: raise RuntimeError('Predictor has no trainable parameters')
    totals = {n: [torch.zeros_like(p, device='cpu') for _, p in named] for n in NAMES}
    sums = {n: 0.0 for n in NAMES}
    rows = []
    try:
        for query_id in chosen:
            indices = np.flatnonzero(raw[query_key].astype(str) == query_id)
            if 'conditions' in raw.files and len(set(raw['conditions'][indices].astype(str))) != len(indices):
                raise ValueError(f'{query_id}: duplicate conditions')
            pixels = raw[history_key][indices]
            futures = raw[future_key][indices]
            if futures.ndim == pixels.ndim-1: futures = futures[:, None]
            actions = raw[action_key][indices]
            horizon = pixels.shape[1]
            if actions.ndim == 3:
                if actions.shape[1] % horizon:
                    raise ValueError(f'{query_id}: flattened action block does not match history length')
                actions = actions.reshape(len(indices), horizon, actions.shape[1]//horizon, actions.shape[2])
            if actions.shape[:2] != (len(indices), horizon) or futures.shape[:2] != (len(indices), 1):
                raise ValueError(f'{query_id}: raw RGB/action shapes inconsistent')
            if not np.array_equal(pixels[:, -1], np.broadcast_to(pixels[:1, -1], pixels[:, -1].shape)):
                raise ValueError(f'{query_id}: paired conditions do not share current RGB')
            if not np.array_equal(actions, np.broadcast_to(actions[:1], actions.shape)):
                raise ValueError(f'{query_id}: paired conditions do not share action blocks')
            history, future = helper.encode_unique(adapter, pixels, futures)
            row, gradients, direct = evaluate(adapter, helper, spec['family'], history, future, actions, named, query_id)
            if not rows:
                actual = adapter.rollout_latents(pixels, actions, batch_size=2)[:, 0]
                row['actual_public_adapter_endpoint'] = _endpoint_check(direct, actual, 'public RGB adapter')
            row['source'] = {'row_indices': indices.tolist(), 'source_groups': raw['source_groups'][indices].astype(str).tolist(),
                             'conditions': raw['conditions'][indices].astype(str).tolist() if 'conditions' in raw.files else None}
            rows.append(row)
            for name, value in gradients.items():
                generic._accumulate_gradient(totals[name], value)
                sums[name] += generic._tensor_tuple_norm(value)
            print(args.id, len(rows), flush=True)
        after = adapter.frozen_state_hash()
        if before != after: raise RuntimeError('Model state changed')
    finally:
        for p, flag in zip(model.parameters(), flags): p.requires_grad_(flag)
    response_energy = sum(r['per_position']['target_response_energy_by_position'][-1] for r in rows)
    response_error = sum(r['per_position']['response_by_position'][-1] for r in rows)
    cosines = [r['query_response_native_gradient_cosine'] for r in rows if r['query_response_native_gradient_cosine'] is not None]
    norm_ratios = [r['query_response_to_native_gradient_norm_ratio'] for r in rows if r['query_response_to_native_gradient_norm_ratio'] is not None]
    shares = [r['query_response_loss_share'] for r in rows if r['query_response_loss_share'] is not None]
    native_sum = sum(r['losses']['native_all'] for r in rows)
    trainer_name = 'prejepa' if spec['family'] == 'dinowm' else spec['family']
    trainer_path = Path(spec['stable_repo'])/'scripts'/'train'/f'{trainer_name}.py'
    if not trainer_path.is_file():
        raise FileNotFoundError(f'Pinned native trainer source unavailable: {trainer_path}')
    trainer_alignment = {
        'pinned_source': str(trainer_path), 'pinned_source_sha256': generic._sha256(trainer_path),
        'native_visual_loss': 'mean squared prediction error across every shifted visual position',
        'visual_component_only': True, 'action_stream_excluded_from_loss': spec['family'] == 'dinowm',
        'native_positions': rows[0]['position_count'],
    }
    result = {'schema': 'contextworld.task_native_visual_objective.v1', 'model': spec, 'selection': selection,
              'query_count': len(rows), 'condition_count': sum(r['condition_count'] for r in rows),
              'position_count': rows[0]['position_count'], 'optimizer_steps': 0, 'public_test_accessed': False,
              'scope': f'{args.split} only; shifted native targets; visual prediction MSE component only; predictor gradients only',
              'cuda_precision': {'matmul_allow_tf32': False, 'cudnn_allow_tf32': False, 'float32_matmul_precision': 'highest'},
              'state_hash_before': before, 'state_hash_after': after,
              'script_sha256': generic._sha256(Path(__file__)), 'generic_helper_sha256': generic._sha256(Path(generic.__file__)),
              'trainer_alignment': trainer_alignment,
              'predictor_parameter_count': sum(p.numel() for _, p in named),
              'summary': {'energy_pooled_native_query_response_nre': response_error/response_energy if response_energy > 0 else None,
                          'zero_query_target_energy_count': sum(r['per_position']['target_response_energy_by_position'][-1] == 0 for r in rows),
                          'mean_query_response_loss_share': float(np.mean(shares)) if shares else None,
                          'aggregate_query_response_loss_share': sum(r['losses']['query_response'] for r in rows)/native_sum if native_sum > 0 else None,
                          'mean_query_response_to_native_gradient_norm_ratio': float(np.mean(norm_ratios)) if norm_ratios else None,
                          'negative_query_response_native_cosine_count': sum(r['query_response_native_gradient_negative_cosine'] for r in rows),
                          'defined_query_response_native_cosine_count': len(cosines),
                          'mean_query_response_native_cosine': float(np.mean(cosines)) if cosines else None},
              'mean_gradient': generic._aggregate_gradient_summary(totals, sums, len(rows)),
              'per_query': rows,
              'limitation': 'Frozen-checkpoint posthoc gradients cannot independently establish optimization causality'}
    generic._write_json(args.output/'result.json', result)


if __name__ == '__main__':
    main()
