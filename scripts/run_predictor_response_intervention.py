#!/usr/bin/env python3
"""Matched predictor-only native visual intervention on fixed paired panels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import diagnose_task_native_objective as native
import diagnose_task_history_readout as readout


def group_indices(data):
    """Keep every condition of a query together, in first-occurrence order."""
    key = 'query_ids' if 'query_ids' in data else 'pair_ids'
    ids = data[key].astype(str)
    order = list(dict.fromkeys(ids.tolist()))
    groups = [np.flatnonzero(ids == query) for query in order]
    expected = set(data['labels'][groups[0]].tolist())
    for query, rows in zip(order, groups):
        if len(rows) < 2 or set(data['labels'][rows].tolist()) != expected:
            raise ValueError(f'{query}: incomplete condition group')
        if len(set(data['labels'][rows].tolist())) != len(rows):
            raise ValueError(f'{query}: duplicate labels')
        if not np.array_equal(data['rawactions'][rows], np.broadcast_to(data['rawactions'][rows[:1]], data['rawactions'][rows].shape)):
            raise ValueError(f'{query}: actions differ across conditions')
    return order, groups


def query_losses(prediction, target):
    """Return native loss and final response loss, each averaged over whole queries."""
    import torch
    if prediction.shape != target.shape or prediction.ndim < 4 or prediction.shape[1] < 2:
        raise ValueError('Expected equal [query, condition, time, feature...] tensors')
    error = prediction - target
    native_loss = error.square().mean()
    centered = error - error.mean(dim=1, keepdim=True)
    response = centered[:, :, -1].square().mean() / prediction.shape[2]
    rest = native_loss - response
    return native_loss, rest, response


def loss_for_arm(prediction, target, rest_weight, response_weight):
    native_loss, rest, response = query_losses(prediction, target)
    return rest_weight * rest + response_weight * response, (native_loss, rest, response)


def arrays(path, expected_sha):
    if readout.sha(path) != expected_sha:
        raise ValueError(f'Native cache hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    if not np.isfinite(result['history']).all() or not np.isfinite(result['queryfuture']).all():
        raise ValueError('Nonfinite latent cache')
    return result


def raw_actions(adapter, data, rows):
    action = data['rawactions'][rows]
    horizon = data['history'].shape[1]
    if action.ndim == 3:
        if action.shape[1] % horizon:
            raise ValueError('Action block length does not divide horizon')
        action = action.reshape(len(rows), horizon, action.shape[1] // horizon, action.shape[2])
    if action.shape[1] != horizon:
        raise ValueError('Action horizon mismatch')
    return action


def predict_query(adapter, family, data, rows, histories=None):
    import torch
    h = data['history'][rows] if histories is None else histories
    y = data['queryfuture'][rows, None]
    a = raw_actions(adapter, data, rows)
    prediction, target, _ = native._prediction(adapter, family, h, y, a)
    return prediction, target


def evaluate(adapter, family, data, groups, batch_queries=8):
    import torch
    model = adapter.model
    model.eval()
    rows_out = []
    with torch.no_grad():
        for rows in groups:
            prediction, target = predict_query(adapter, family, data, rows)
            endpoint = prediction[:, -1].reshape(len(rows), -1).double()
            truth = target[:, -1].reshape(len(rows), -1).double()
            residual = endpoint - truth
            center = residual - residual.mean(0, keepdim=True)
            energy = (truth - truth.mean(0, keepdim=True)).square().mean()
            endpoint_mse = residual.square().mean()
            response_mse = center.square().mean()
            common_mse = residual.mean(0).square().mean()
            native_loss, rest, response = query_losses(prediction[None], target[None])
            if not np.array_equal(data['history'][rows, -1], np.broadcast_to(data['history'][rows[:1], -1], data['history'][rows, -1].shape)):
                raise ValueError('Matched query current latents differ')
            distances = (endpoint[:, None] - truth[None]).square().mean(-1)
            swapped_mse = (distances.sum()-distances.diagonal().sum())/(len(rows)*(len(rows)-1))
            rows_out.append(dict(query_id=str(data['query_ids'][rows[0]]) if 'query_ids' in data else str(data['pair_ids'][rows[0]]),
                source_group=str(data['source_groups'][rows[0]]), row_indices=rows.tolist(),
                condition_count=len(rows), native_visual_mse=float(native_loss), rest_mse=float(rest),
                response_training_component=float(response), endpoint_mse=float(endpoint_mse),
                response_mse=float(response_mse), common_mse=float(common_mse),
                target_response_energy=float(energy), swapped_history_endpoint_mse=float(swapped_mse),
                matched_history_gain=float(swapped_mse-endpoint_mse),
                condition_endpoint_mse=residual.square().mean(-1).cpu().tolist()))
    def mean(key): return float(np.mean([row[key] for row in rows_out]))
    energy = sum(row['target_response_energy'] for row in rows_out)
    response_error = sum(row['response_mse'] for row in rows_out)
    return dict(query_count=len(groups), condition_count=sum(len(g) for g in groups),
        native_visual_mse=mean('native_visual_mse'), endpoint_mse=mean('endpoint_mse'),
        common_mse=mean('common_mse'), response_mse=mean('response_mse'),
        response_nre=response_error/energy if energy else None,
        swapped_history_endpoint_mse=mean('swapped_history_endpoint_mse'),
        matched_history_gain=mean('matched_history_gain'), per_query=rows_out)


def batch_schedule(query_count, steps, batch_queries, seed):
    rng = np.random.default_rng(seed)
    sequence = []
    while len(sequence) < steps * batch_queries:
        sequence.extend(rng.permutation(query_count).tolist())
    return [sequence[i*batch_queries:(i+1)*batch_queries] for i in range(steps)]


def train_step(adapter, family, data, groups, selected, optimizer, weights):
    import torch
    optimizer.zero_grad(set_to_none=True)
    pieces = []
    # Query-sized forward passes bound DINO patch activation memory. Division by
    # query count gives exactly the mean objective of this complete-group batch.
    for index in selected:
        prediction, target = predict_query(adapter, family, data, groups[index])
        loss, components = loss_for_arm(prediction[None], target[None], *weights)
        (loss / len(selected)).backward()
        pieces.append(tuple(float(x.detach()) for x in components))
    torch.nn.utils.clip_grad_norm_(trainable_parameters(adapter.model), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return np.mean(pieces, axis=0).tolist()


def calibrate(adapter, family, data, groups, ids):
    """Compute fixed Training-only response/rest coefficients from 16 source groups."""
    import torch
    first = {}
    for index, rows in enumerate(groups):
        source = str(data['source_groups'][rows[0]])
        first.setdefault(source, index)
    ordered = sorted(first, key=lambda source: hashlib.sha256(f'20261010:{source}'.encode()).hexdigest())
    selected = [first[source] for source in ordered[:16]]
    if len(selected) != 16:
        raise ValueError('Calibration needs 16 distinct Training source groups')
    params = trainable_parameters(adapter.model)
    totals = [torch.zeros_like(p) for p in params], [torch.zeros_like(p) for p in params]
    losses = []
    for index in selected:
        prediction, target = predict_query(adapter, family, data, groups[index])
        _, rest, response = query_losses(prediction[None], target[None])
        losses.append([float(rest.detach()), float(response.detach())])
        for number, (components, total) in enumerate(((rest, totals[0]), (response, totals[1]))):
            gradients = torch.autograd.grad(components, params, allow_unused=True, retain_graph=(number == 0))
            for sum_tensor, gradient in zip(total, gradients):
                if gradient is not None: sum_tensor.add_(gradient.detach(), alpha=1/16)
    def dot(a,b): return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
    gr, gs = totals
    nr, ns = dot(gr,gr)**0.5, dot(gs,gs)**0.5
    if not np.isfinite([nr,ns]).all() or min(nr,ns) <= 0:
        raise RuntimeError('Nonfinite or zero calibration gradient')
    lam = max(1., nr/ns)
    if not np.isfinite(lam) or lam > 1e6:
        raise RuntimeError('Calibration lambda too large')
    combined = [r+lam*s for r,s in zip(gr,gs)]
    original = [r+s for r,s in zip(gr,gs)]
    denominator = dot(original,original)**0.5
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError('Zero original calibration gradient')
    scale = (dot(combined,combined)**0.5)/denominator
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError('Nonfinite calibration scale')
    return dict(source_groups=ordered[:16], query_ids=[ids[i] for i in selected],
        rest_gradient_norm=nr, response_gradient_norm=ns, gradient_cosine=dot(gr,gs)/(nr*ns),
        mean_rest_loss=float(np.mean(losses,axis=0)[0]), mean_response_loss=float(np.mean(losses,axis=0)[1]),
        lambda_response=lam, scale=scale, reweighted_coefficients=[1/scale,lam/scale])


def frozen_hash(model):
    import torch
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(('predictor.','pred_proj.')):
            continue
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def trainable_parameters(model):
    return list(model.predictor.parameters()) + list(model.pred_proj.parameters()) if hasattr(model, 'pred_proj') else list(model.predictor.parameters())


def trainable_state(model):
    return {name:value.detach().clone() for name,value in model.state_dict().items()
            if name.startswith(('predictor.','pred_proj.'))}


def restore_trainable_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError('Trainable state restore mismatch')
    current = model.state_dict()
    if not all(torch_equal(current[name], value) for name, value in state.items()):
        raise RuntimeError('Trainable state restore did not reproduce source values')


def torch_equal(left, right):
    import torch
    return torch.equal(left.detach(), right.detach())


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', required=True)
    parser.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    parser.add_argument('--readout-root', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/readout'))
    parser.add_argument('--panel-root', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--batch-queries', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261010)
    parser.add_argument('--locked-protocol', type=Path, default=Path('/tmp/cw-response-intervention-20261010/protocol.json'))
    args = parser.parse_args()
    if args.steps != 256 or args.batch_queries != 8:
        raise ValueError('Locked protocol requires 256 updates and 8 whole queries per batch')
    checkpoints = [0,64,256]
    if args.output.exists():
        raise FileExistsError(f'Output must be new: {args.output}')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    specs = json.loads(args.models.read_text())
    spec = next((s for s in specs if s['id'] == args.id), None)
    if spec is None or spec['id'] not in (f'{task}/{family}/scratch/s3072' for task, family in
            [('contact_friction','lewm'),('cube_gripper_carry','lewm'),('robot_arm_mass','lewm'),('action_delay','dinowm')]):
        raise ValueError('ID outside locked intervention scope')
    cache_dir = args.readout_root / spec['task'] / spec['family'] / 'scratch/s3072'
    readout_result = json.loads((cache_dir/'result.json').read_text())
    if readout_result['model_id'] != args.id or readout_result['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('Readout/model binding mismatch')
    panel_manifest = args.panel_root/spec['task']/'manifest.json'
    current_manifest_hash = readout.sha(panel_manifest)
    if current_manifest_hash != readout_result['panel_manifest_sha256']:
        receipt_path = cache_dir/'manifest_binding_validation.json'
        if not receipt_path.exists():
            raise ValueError('Panel manifest hash mismatch without validation receipt')
        receipt = json.loads(receipt_path.read_text())
        if (receipt['model_id'] != args.id or receipt['old_manifest_sha256'] != readout_result['panel_manifest_sha256']
                or receipt['current_manifest_sha256'] != current_manifest_hash or not receipt['model_readout_unchanged']):
            raise ValueError('Invalid panel manifest rebinding receipt')
        for split in ('training','development'):
            check = receipt['checks'][split]
            if check['native_cache_sha256'] != readout_result['native_cache'][split]['sha256'] or not check['cached_metadata_and_actions_equal']:
                raise ValueError('Panel rebinding receipt does not bind native cache')
    task_manifest = json.loads(panel_manifest.read_text())
    if task_manifest['task'] != spec['task']:
        raise ValueError('Task manifest mismatch')
    data = {}; groups = {}; ids = {}
    for split in ('training', 'development'):
        info = readout_result['native_cache'][split]
        data[split] = arrays(Path(info['path']), info['sha256'])
        ids[split], groups[split] = group_indices(data[split])
        if data[split]['history'].shape[1] != (7 if spec['task'] == 'action_delay' else 3):
            raise ValueError('Unexpected native history horizon')
    if len(groups['training']) != (128 if spec['task'] == 'action_delay' else 512):
        raise ValueError('Training fit panel query count mismatch')
    args.output.mkdir(parents=True)
    helper = native.generic._load_helper()
    adapter = helper.load_adapter(spec, task_manifest['normalization'], args.output, args.device)
    model = adapter.model
    model.eval()
    source_hash = adapter.frozen_state_hash()
    frozen_before = frozen_hash(model)
    initial_predictor = trainable_state(model)
    schedule = batch_schedule(len(groups['training']), args.steps, args.batch_queries, args.seed)
    schedule_hash = hashlib.sha256(np.asarray(schedule,dtype='<i4').tobytes()).hexdigest()
    for p in model.parameters(): p.requires_grad_(False)
    model.predictor.requires_grad_(True)
    if hasattr(model,'pred_proj'): model.pred_proj.requires_grad_(True)
    calibration = calibrate(adapter,spec['family'],data['training'],groups['training'],ids['training'])
    lr, weight_decay = ((1e-4,0.0) if spec['family']=='dinowm' else (1e-5,0.001))
    result = dict(schema='contextworld.predictor_response_intervention.v1', model_id=args.id,
        model_spec=spec, readout_result_sha256=readout.sha(cache_dir/'result.json'),
        panel_manifest_sha256=current_manifest_hash,
        cache_sha256={s:readout_result['native_cache'][s]['sha256'] for s in data},
        script_sha256=readout.sha(__file__), locked_protocol_sha256=readout.sha(args.locked_protocol), public_test_accessed=False,
        visual_component_only=True, native_target='shifted history plus final future',
        predictor_mode='eval, gradients enabled', fixed_encoder_action_encoder=True,
        seed=args.seed, learning_rate=lr, weight_decay=weight_decay, gradient_clip_norm=1.0,
        optimizer='AdamW', optimizer_betas=[0.9,0.999], optimizer_epsilon=1e-8,
        calibration=calibration, batch_order_sha256=schedule_hash,
        batch_query_ids=[[ids['training'][i] for i in batch] for batch in schedule],
        updates=args.steps, batch_queries=args.batch_queries,
        checkpoints=checkpoints, completion_state='running', source_model_state_hash=source_hash, frozen_state_hash=frozen_before,
        arms={})
    readout.write(args.output/'protocol.json',{key:result[key] for key in ('schema','model_id','panel_manifest_sha256','cache_sha256','locked_protocol_sha256','seed','learning_rate','weight_decay','optimizer','optimizer_betas','optimizer_epsilon','gradient_clip_norm','calibration','batch_order_sha256','updates','batch_queries','checkpoints')})
    for arm, weights in [('native',(1.0,1.0)),('response_weighted',tuple(calibration['reweighted_coefficients']))]:
        restore_trainable_state(model,initial_predictor)
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        model.predictor.requires_grad_(True)
        if hasattr(model,'pred_proj'): model.pred_proj.requires_grad_(True)
        params = trainable_parameters(model)
        if not params or not all(p.requires_grad for p in params):
            raise RuntimeError('Predictor trainability failure')
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=(0.9,0.999), eps=1e-8)
        if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in params}:
            raise RuntimeError('Optimizer includes nonpredictor parameters')
        snapshots = {}; training_losses = []; first_update_norm = None
        for step in range(args.steps+1):
            if step in checkpoints:
                snapshots[str(step)] = {split:evaluate(adapter,spec['family'],data[split],groups[split]) for split in data}
                if frozen_hash(model) != frozen_before:
                    raise RuntimeError('Frozen model components changed')
                if step:
                    for split in data:
                        baseline = snapshots['0'][split]['endpoint_mse']
                        snapshots[str(step)][split]['endpoint_mse_over_step0'] = snapshots[str(step)][split]['endpoint_mse']/baseline if baseline else None
                readout.write(args.output/f'progress_{arm}.json',dict(arm=arm,completed_update=step,snapshots=snapshots))
            if step == args.steps: break
            if step == 0:
                before_step = [p.detach().clone() for p in params]
            training_losses.append(train_step(adapter,spec['family'],data['training'],groups['training'],schedule[step],optimizer,weights))
            if step == 0:
                first_update_norm = sum(float((p.detach()-q).double().square().sum()) for p,q in zip(params,before_step))**0.5
        arm_dir = args.output/arm
        arm_dir.mkdir()
        torch.save(trainable_state(model),arm_dir/'predictor.pt')
        result['arms'][arm] = dict(rest_weight=weights[0],response_weight=weights[1],
            training_components=training_losses, first_update_norm=first_update_norm, snapshots=snapshots,
            predictor_checkpoint=str(arm_dir/'predictor.pt'),frozen_state_hash_after=frozen_hash(model))
        readout.write(args.output/'result.json',result)
    restore_trainable_state(model,initial_predictor)
    if adapter.frozen_state_hash() != source_hash:
        raise RuntimeError('Source model not restored')
    result['restored_model_state_hash'] = adapter.frozen_state_hash()
    result['completion_state'] = 'complete'
    readout.write(args.output/'result.json',result)


if __name__ == '__main__':
    main()
