#!/usr/bin/env python3
"""Locked, paired PLDM predictor intervention using fixed visual representations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import diagnose_task_native_objective as native
import diagnose_native_policy_contrast as contrast
import run_predictor_response_intervention as base

TASKS = ('contact_friction', 'motion_damping')
SPLITS = ('training', 'development')
DESIGN = dict(steps=256, batch_queries=8, seed=20261012,
              checkpoints=[0, 64, 256], primary_checkpoint=256,
              learning_rate=1e-5, weight_decay=.001, gradient_clip_norm=1.,
              arms=['native_visual', 'response_weighted'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fixed_module_hashes(model):
    modules = {}
    for name, value in model.state_dict().items():
        if name.startswith(('predictor.', 'pred_proj.')):
            continue
        group = name.split('.', 1)[0]
        digest = modules.setdefault(group, hashlib.sha256())
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return {name: digest.hexdigest() for name, digest in modules.items()}


def groups_for(raw, cache):
    """Return paired row indices, rejecting any reordered or unmatched cache."""
    for key in ('query_ids', 'source_groups'):
        if not np.array_equal(raw[key].astype(str), cache[key].astype(str)):
            raise ValueError(f'{key} cache/raw order mismatch')
    data = dict(query_ids=raw['query_ids'], labels=raw['labels'], rawactions=raw['rawactions'])
    ids, groups = base.group_indices(data)
    if any(len(rows) != 2 for rows in groups):
        raise ValueError('PLDM intervention requires exactly two conditions per query')
    if any(not np.array_equal(raw['history_pixels'][rows, -1],
                              np.broadcast_to(raw['history_pixels'][rows[:1], -1], raw['history_pixels'][rows, -1].shape))
           for rows in groups):
        raise ValueError('Paired current RGB differs')
    return ids, groups


def batch_prediction(adapter, cache, raw, groups, chosen, swapped=False):
    """One model call over complete queries; native loss reshapes to [Q,2,H,D]."""
    rows = np.concatenate([groups[i] for i in chosen])
    h = cache['history'][rows].copy()
    if swapped:
        h = h.reshape(len(chosen), 2, *h.shape[1:])[:, ::-1].reshape(h.shape)
    y = cache['target'][rows, None]
    actions = base.raw_actions(adapter, dict(rawactions=raw['rawactions'], history=cache['history']), rows)
    prediction, target, _ = native._prediction(adapter, 'pldm', h, y, actions)
    return prediction.reshape(len(chosen), 2, *prediction.shape[1:]), target.reshape(len(chosen), 2, *target.shape[1:])


def grouped_loss(adapter, cache, raw, groups, chosen, weights):
    prediction, target = batch_prediction(adapter, cache, raw, groups, chosen)
    return base.loss_for_arm(prediction, target, *weights)


def evaluate(adapter, cache, raw, ids, groups, batch_size=8):
    import torch
    adapter.model.eval()
    target = cache['target']
    correct = np.empty_like(target)
    with torch.no_grad():
        for start in range(0, len(groups), batch_size):
            chosen = list(range(start, min(start + batch_size, len(groups))))
            prediction, _ = batch_prediction(adapter, cache, raw, groups, chosen)
            rows = np.concatenate([groups[i] for i in chosen])
            correct[rows] = prediction[:, :, -1].reshape(len(rows), -1).cpu().numpy()
    pair_rows = np.stack(groups)
    swap = np.arange(len(target))
    swap[pair_rows[:, 0]] = pair_rows[:, 1]
    swap[pair_rows[:, 1]] = pair_rows[:, 0]
    swapped = correct[swap]
    stats = contrast.pair_statistics(target, correct, swapped, pair_rows)
    per_query = []
    for i, query_id in enumerate(ids):
        row = {key: float(value[i]) for key, value in stats.items()}
        row.update(query_id=query_id, source_group=str(cache['source_groups'][groups[i][0]]),
                   row_indices=groups[i].tolist(), E_over_B=row['E']/row['B'] if row['B'] else None,
                   NRE=row['R']/row['B'] if row['B'] else None,
                   G=(row['W']-row['E'])/row['B'] if row['B'] else None,
                   history_gain=row['W']-row['E'])
        per_query.append(row)
    return dict(summary=contrast.aggregate(stats), per_query=per_query)


def validate_cache(adapter, cache, raw, groups, split, previous):
    """Bind cached latents to RGB and independently reproduce cached endpoints."""
    import torch
    sample = np.concatenate((groups[0], groups[-1]))
    with torch.no_grad():
        h = adapter.encode_pixels(raw['history_pixels'][sample].reshape(-1, *raw['history_pixels'].shape[-3:]), batch_size=16)
        h = h.reshape(len(sample), cache['history'].shape[1], -1)
        y = adapter.encode_pixels(raw['future_pixels'][sample].reshape(-1, *raw['future_pixels'].shape[-3:]), batch_size=16)
    np.testing.assert_allclose(h, cache['history'][sample], rtol=1e-5, atol=3e-4)
    np.testing.assert_allclose(y, cache['target'][sample], rtol=1e-5, atol=3e-4)
    correct, _ = batch_prediction(adapter, cache, raw, groups, [0, len(groups)-1])
    wrong, _ = batch_prediction(adapter, cache, raw, groups, [0, len(groups)-1], swapped=True)
    np.testing.assert_allclose(correct[:, :, -1].reshape(len(sample), -1).detach().cpu(),
                               cache['correct_prediction'][sample], rtol=1e-5, atol=3e-4)
    np.testing.assert_allclose(wrong[:, :, -1].reshape(len(sample), -1).detach().cpu(),
                               cache['swapped_prediction'][sample], rtol=1e-5, atol=3e-4)


def main():
    import torch
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New per-task output directory')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    p.add_argument('--panels', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    p.add_argument('--cache-root', type=Path, default=Path('/tmp/cw-native-policy-contrast-20261011'))
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('schema') != 'contextworld.pldm_fixed_representation_intervention.v1' or protocol.get('design') != DESIGN:
        raise ValueError('Locked protocol schema/design mismatch')
    code = protocol['code_sha256']
    for key, path in [('runner', Path(__file__)),
                      ('intervention_helpers', Path(base.__file__)),
                      ('native_prediction', Path(native.__file__))]:
        if code[key] != sha(path):
            raise ValueError(f'{key} script SHA mismatch')
    binding = protocol['tasks'][args.task]
    model_id = f'{args.task}/pldm/frozen/s3072'
    specs = json.loads(args.models.read_text())
    spec = next((s for s in specs if s['id'] == model_id), None)
    if spec is None or binding['model_id'] != model_id or spec['checkpoint_sha256'] != binding['checkpoint_sha256'] or sha(spec['checkpoint']) != binding['checkpoint_sha256']:
        raise ValueError('Model/checkpoint binding mismatch')
    panel_dir = args.panels / args.task
    panel_manifest_path = panel_dir / 'manifest.json'
    if sha(panel_manifest_path) != binding['panel_manifest_sha256']:
        raise ValueError('Panel manifest SHA mismatch')
    manifest = json.loads(panel_manifest_path.read_text())
    previous_path = args.cache_root / args.task / 'pldm/frozen/result.json'
    if sha(previous_path) != binding['previous_result_sha256']:
        raise ValueError('Previous result SHA mismatch')
    previous = json.loads(previous_path.read_text())
    if previous['model_id'] != model_id or previous['checkpoint_sha256'] != binding['checkpoint_sha256'] or previous['panel_manifest_sha256'] != binding['panel_manifest_sha256']:
        raise ValueError('Previous result model/panel mismatch')
    data = {}; raw = {}; ids = {}; groups = {}
    for split in SPLITS:
        raw_path = panel_dir / f'{split}.npz'
        cache_path = args.cache_root / args.task / 'pldm/frozen' / f'{split}_native_latents.npz'
        if sha(raw_path) != binding['split_sha256'][split] or sha(raw_path) != manifest['splits'][split]['sha256'] or sha(raw_path) != previous['split_sha256'][split]:
            raise ValueError(f'{split} raw SHA mismatch')
        if sha(cache_path) != binding['cache_sha256'][split]:
            raise ValueError(f'{split} native cache SHA mismatch')
        with np.load(raw_path, allow_pickle=False) as a: raw[split] = {k: a[k] for k in a.files}
        with np.load(cache_path, allow_pickle=False) as a: data[split] = {k: a[k] for k in a.files}
        ids[split], groups[split] = groups_for(raw[split], data[split])
        if ids[split] != list(map(str, manifest['splits'][split]['query_ids'])):
            raise ValueError(f'{split} manifest query order mismatch')
        if not np.isfinite(data[split]['history']).all() or not np.isfinite(data[split]['target']).all():
            raise ValueError(f'{split} nonfinite cache')
    torch.manual_seed(DESIGN['seed'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    args.output.mkdir(parents=True)
    helper = native.generic._load_helper()
    adapter = helper.load_adapter(spec, manifest['normalization'], args.output, args.device)
    model = adapter.model
    model.eval()
    original_hash = adapter.frozen_state_hash()
    if original_hash != previous['model_state_hash_before'] or original_hash != previous['model_state_hash_after']:
        raise ValueError('Cached predictions are bound to a different frozen model state')
    frozen_hash = base.frozen_hash(model)
    module_hashes = fixed_module_hashes(model)
    initial = base.trainable_state(model)
    original_flags = [p.requires_grad for p in model.parameters()]
    result = dict(schema=protocol['schema'], task=args.task, model_id=model_id,
                  protocol_sha256=sha(args.protocol), binding=binding, code_sha256=code,
                  source_model_state_hash=original_hash, frozen_state_hash=frozen_hash,
                  fixed_module_hashes=module_hashes,
                  fixed_representation_only=True,
                  native_target='shifted history plus final future',
                  objective='full shifted visual MSE versus Training-calibrated response weighting',
                  optimizer='AdamW', optimizer_betas=[0.9, .999], optimizer_epsilon=1e-8,
                  public_test_accessed=False, completion_state='running', arms={})
    try:
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.predictor.requires_grad_(True)
        if hasattr(model, 'pred_proj'): model.pred_proj.requires_grad_(True)
        for split in SPLITS:
            validate_cache(adapter, data[split], raw[split], groups[split], split, previous)
        step0 = {split: evaluate(adapter, data[split], raw[split], ids[split], groups[split]) for split in SPLITS}
        if base.frozen_hash(model) != frozen_hash or fixed_module_hashes(model) != module_hashes or adapter.frozen_state_hash() != original_hash:
            raise RuntimeError('Model state changed during step-0 evaluation')
        baseline_difference = {}
        for split in SPLITS:
            expected = previous['metrics'][split]['summary']
            baseline_difference[split] = {}
            for key in ('E_over_B', 'NRE', 'G', 'main_choice', 'history_wins'):
                actual = step0[split]['summary'][key]
                difference = actual - expected[key]
                baseline_difference[split][key] = dict(current=actual, previous=expected[key], difference=difference)
                tolerance = 1/len(groups[split]) if key in ('main_choice', 'history_wins') else None
                matched = abs(difference) <= tolerance if tolerance is not None else np.isclose(actual, expected[key], rtol=1e-5, atol=1e-6)
                if not matched:
                    raise ValueError(f'{split} baseline {key} disagrees with previous result')
        result['baseline_difference'] = baseline_difference
        calibration = base.calibrate(adapter, 'pldm',
            dict(history=data['training']['history'], queryfuture=data['training']['target'],
                 rawactions=raw['training']['rawactions'], source_groups=data['training']['source_groups']),
            groups['training'], ids['training'])
        # base.calibrate calls base.predict_query, whose future key is queryfuture.
        result['calibration'] = calibration
        schedule = base.batch_schedule(len(groups['training']), DESIGN['steps'], DESIGN['batch_queries'], DESIGN['seed'])
        result['batch_query_ids'] = [[ids['training'][i] for i in batch] for batch in schedule]
        result['batch_order_sha256'] = hashlib.sha256(np.asarray(schedule, dtype='<i4').tobytes()).hexdigest()
        for arm, weights in [('native_visual', (1., 1.)),
                             ('response_weighted', tuple(calibration['reweighted_coefficients']))]:
            base.restore_trainable_state(model, initial)
            model.eval()
            if adapter.frozen_state_hash() != original_hash or base.frozen_hash(model) != frozen_hash:
                raise RuntimeError(f'{arm} did not start from original model state')
            params = base.trainable_parameters(model)
            optimizer = torch.optim.AdamW(params, lr=DESIGN['learning_rate'],
                                          weight_decay=DESIGN['weight_decay'])
            snapshots = {'0': step0}
            losses = []
            first_update_norm = None
            write(args.output / f'progress_{arm}.json',
                  dict(arm=arm, completed_update=0, snapshots=snapshots,
                       training_losses=losses, first_update_norm=first_update_norm))
            for step in range(1, DESIGN['steps'] + 1):
                if step == 1: before = [v.detach().clone() for v in params]
                optimizer.zero_grad(set_to_none=True)
                loss, components = grouped_loss(adapter, data['training'], raw['training'],
                                                groups['training'], schedule[step-1], weights)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, DESIGN['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                losses.append(dict(objective=float(loss.detach()), native=float(components[0].detach()),
                                   rest=float(components[1].detach()), response=float(components[2].detach())))
                if step == 1:
                    first_update_norm = sum(float((v.detach()-b).double().square().sum()) for v,b in zip(params,before))**.5
                if step in DESIGN['checkpoints']:
                    snapshots[str(step)] = {split: evaluate(adapter, data[split], raw[split], ids[split], groups[split]) for split in SPLITS}
                    if base.frozen_hash(model) != frozen_hash:
                        raise RuntimeError('Frozen representation changed')
                    if fixed_module_hashes(model) != module_hashes:
                        raise RuntimeError('Fixed module state changed')
                    write(args.output / f'progress_{arm}.json',
                          dict(arm=arm, completed_update=step, snapshots=snapshots,
                               training_losses=losses, first_update_norm=first_update_norm))
            arm_dir = args.output / arm
            arm_dir.mkdir()
            torch.save(base.trainable_state(model), arm_dir / 'predictor.pt')
            result['arms'][arm] = dict(weights=weights, snapshots=snapshots,
                                       training_losses=losses, first_update_norm=first_update_norm,
                                       predictor_checkpoint=str(arm_dir / 'predictor.pt'),
                                       predictor_checkpoint_sha256=sha(arm_dir / 'predictor.pt'),
                                       frozen_state_hash_after=base.frozen_hash(model))
            write(args.output / 'result.json', result)
        base.restore_trainable_state(model, initial)
        if adapter.frozen_state_hash() != original_hash:
            raise RuntimeError('Original model state not restored')
        result['restored_model_state_hash'] = original_hash
        result['completion_state'] = 'complete'
        write(args.output / 'result.json', result)
    finally:
        base.restore_trainable_state(model, initial)
        for parameter, flag in zip(model.parameters(), original_flags): parameter.requires_grad_(flag)


if __name__ == '__main__':
    main()
