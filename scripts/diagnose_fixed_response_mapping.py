#!/usr/bin/env python3
"""CPU-only supervised fixed-response map on frozen native PLDM latents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

SCHEMA = 'contextworld.fixed_response_mapping.v1'
DESIGN = dict(alpha_grid=[1e-6, 1e-4, 1e-2, 1.0, 100.0], bandwidth_multipliers=[0.25, 1.0, 4.0], folds=4,
              scaler_floor=1e-6, interpolation_alpha=1e-8, interpolation_bandwidth_multiplier=1.0,
              kernels=['linear', 'rbf'], target='pair_centered_future_latent', history='z0,dz1,dz2',
              actions='raw_15x2', selection='pooled_training_fold_pair_difference_squared_error_over_pair_difference_energy',
              source_identity='motion_damping_split_scoped_original_group; contact_friction_original_group; exact_cross_split_history_actions_row_rejection')
SPLITS = ('training', 'development')
MODES = ('native', 'linear_cv', 'rbf_cv', 'rbf_interpolation', 'current_control')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pair_rows(ids, labels, groups):
    ids, labels, groups = map(np.asarray, (ids, labels, groups))
    if not (len(ids) == len(labels) == len(groups)):
        raise ValueError('row count mismatch')
    rows = []
    for query in np.unique(ids):
        r = np.flatnonzero(ids == query)
        if len(r) != 2 or sorted(labels[r].tolist()) != [0, 1] or groups[r[0]] != groups[r[1]]:
            raise ValueError(f'invalid pair: {query}')
        rows.append(r[np.argsort(labels[r])])
    return np.asarray(rows, np.int64)


def centered_targets(future, rows):
    future = np.asarray(future, np.float64)
    result = np.empty_like(future)
    result[rows] = future[rows] - future[rows].mean(axis=1, keepdims=True)
    return result


def features(history, actions, current=False):
    h = np.asarray(history, np.float64)
    a = np.asarray(actions, np.float64)
    if h.ndim != 3 or h.shape[1] != 3 or a.shape != (len(h), 15, 2):
        raise ValueError('expected H3 latent history and 15x2 actions')
    z = h[:, -1] if current else np.stack((h[:, 0], h[:, 1]-h[:, 0], h[:, 2]-h[:, 1]), axis=1).reshape(len(h), -1)
    return np.concatenate((z.reshape(len(h), -1), a.reshape(len(h), -1)), axis=1)


def scale(train, other, floor=1e-6):
    mean = train.mean(axis=0)
    std = np.maximum(train.std(axis=0), floor)
    return (train-mean)/std, (other-mean)/std, mean, std


def squared_distances(x, z):
    d = np.maximum(np.sum(x*x, axis=1)[:, None] + np.sum(z*z, axis=1)[None, :] - 2*x@z.T, 0)
    return d


def median_distance(x):
    d = squared_distances(x, x)
    vals = d[np.triu_indices(len(x), 1)]
    vals = vals[vals > 0]
    if not len(vals):
        raise ValueError('all Training inputs identical')
    return float(np.median(vals))


def kernel(x, z, kind, multiplier=1.0, median=None):
    if kind == 'linear':
        return x@z.T/x.shape[1]
    return np.exp(-squared_distances(x, z)/(2*multiplier*median))


def solve_many(k, y, alphas):
    eig, vectors = np.linalg.eigh(k)
    projected = vectors.T@y
    return {float(alpha): vectors@(projected/(eig[:, None]+alpha)) for alpha in alphas}


def pair_response_error(target, prediction, rows):
    target_diff = target[rows[:, 1]] - target[rows[:, 0]]
    prediction_diff = prediction[rows[:, 1]] - prediction[rows[:, 0]]
    return float(np.square(prediction_diff-target_diff).sum()), float(np.square(target_diff).sum())


def cv_select(x, y, groups, query_ids, kind, multipliers=None, alphas=None):
    alphas = DESIGN['alpha_grid'] if alphas is None else alphas
    multipliers = [1.0] if kind == 'linear' else (DESIGN['bandwidth_multipliers'] if multipliers is None else multipliers)
    splits = list(GroupKFold(n_splits=DESIGN['folds']).split(x, groups=groups))
    query_ids = np.asarray(query_ids)
    if len(query_ids) != len(x):
        raise ValueError('query ID row count mismatch')
    scores = {(m, a): [0.0, 0.0] for m in multipliers for a in alphas}
    for train, held in splits:
        xt, xv, _, _ = scale(x[train], x[held])
        med = median_distance(xt) if kind == 'rbf' else None
        # Form held-out pairs from IDs alone; label order is immaterial for squared response error.
        pair_indices = []
        for query in np.unique(query_ids[held]):
            positions = np.flatnonzero(query_ids[held] == query)
            if len(positions) != 2:
                raise ValueError(f'incomplete held-out query: {query}')
            pair_indices.append(positions)
        held_rows = np.asarray(pair_indices, np.int64)
        target_diff = y[held][held_rows[:, 1]] - y[held][held_rows[:, 0]]
        energy = float(np.square(target_diff).sum())
        for multiplier in multipliers:
            k = kernel(xt, xt, kind, multiplier, med)
            cross = kernel(xv, xt, kind, multiplier, med)
            fitted = solve_many(k, y[train], alphas)
            for alpha in alphas:
                prediction = cross@fitted[float(alpha)]
                error, _ = pair_response_error(y[held], prediction, held_rows)
                scores[(multiplier, alpha)][0] += error
                scores[(multiplier, alpha)][1] += energy
    records = [dict(multiplier=m, alpha=a, nre=scores[(m, a)][0]/scores[(m, a)][1],
                    squared_error=scores[(m, a)][0], centered_energy=scores[(m, a)][1])
               for m in multipliers for a in alphas]
    # Prespecified grid order resolves exact score ties.
    selected = min(records, key=lambda r: r['nre'])
    return selected, records, splits


def fit_predict(train_x, dev_x, train_y, kind, alpha, multiplier=1.0):
    xt, xd, mean, std = scale(train_x, dev_x)
    med = median_distance(xt) if kind == 'rbf' else None
    coefficient = solve_many(kernel(xt, xt, kind, multiplier, med), train_y, [alpha])[float(alpha)]
    train_prediction = kernel(xt, xt, kind, multiplier, med)@coefficient
    dev_prediction = kernel(xd, xt, kind, multiplier, med)@coefficient
    return train_prediction, dev_prediction, dict(mean=mean, std=std, coefficient=coefficient,
                                                   train_scaled=xt, median=np.array(np.nan if med is None else med))


def source_identities(task, split, original_groups):
    groups = np.asarray(original_groups).astype(str)
    if task == 'motion_damping':
        return np.asarray([f'{split}:{group}' for group in groups])
    return groups


def input_fingerprints(history, actions):
    # Exact bytes of frozen history and raw actions, excluding all future/label data.
    h = np.ascontiguousarray(history)
    a = np.ascontiguousarray(actions)
    return np.asarray([hashlib.sha256(h[i].tobytes()+a[i].tobytes()).hexdigest()
                       for i in range(len(h))])


def reject_cross_split_duplicate_inputs(training, development):
    overlap = set(training['input_fingerprints']) & set(development['input_fingerprints'])
    if overlap:
        raise ValueError(f'{len(overlap)} exact history+action inputs reused across Training/Dev')


def load_split(task, split, input_protocol, panels, cache_root):
    binding = input_protocol['tasks'][task]
    cache_path = cache_root/task/'pldm'/'frozen'/f'{split}_native_latents.npz'
    raw_path = panels/task/f'{split}.npz'
    if sha(cache_path) != binding['cache_sha256'][split] or sha(raw_path) != binding['split_sha256'][split]:
        raise ValueError(f'{split} source hash mismatch')
    with np.load(cache_path, allow_pickle=False) as z:
        cache = {k: z[k] for k in ('history', 'target', 'correct_prediction', 'query_ids', 'source_groups')}
    with np.load(raw_path, allow_pickle=False) as z:
        raw = {k: z[k] for k in ('labels', 'query_ids', 'source_groups', 'rawactions')}
    for key in ('query_ids', 'source_groups'):
        if not np.array_equal(cache[key], raw[key]):
            raise ValueError(f'{split} {key} mismatch')
    rows = pair_rows(raw['query_ids'], raw['labels'], raw['source_groups'])
    if not np.array_equal(raw['rawactions'][rows[:, 0]], raw['rawactions'][rows[:, 1]]):
        raise ValueError('paired actions mismatch')
    original_groups = raw['source_groups'][rows[:, 0]]
    groups = source_identities(task, split, original_groups)
    return dict(rows=rows, ids=raw['query_ids'][rows[:, 0]], groups=groups, original_groups=original_groups,
                target=centered_targets(cache['target'], rows), native=cache['correct_prediction'],
                full=features(cache['history'], raw['rawactions']),
                current=features(cache['history'], raw['rawactions'], current=True),
                input_fingerprints=input_fingerprints(cache['history'], raw['rawactions']),
                hashes=dict(cache=sha(cache_path), raw=sha(raw_path)))


def metrics(y, p, rows):
    truth = y[rows[:, 1]]-y[rows[:, 0]]
    response = p[rows[:, 1]]-p[rows[:, 0]]
    energy = float(np.square(truth).sum())
    return dict(nre=float(np.square(response-truth).sum()/energy),
                swapped_nre=float(np.square(-response-truth).sum()/energy),
                row_centered_target_error=float(np.square(p-y).sum()/np.square(y).sum()))


def run(args):
    if args.task not in ('contact_friction', 'motion_damping'):
        raise ValueError('unsupported task')
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = json.loads(args.protocol.read_text())
    if protocol['schema'] != SCHEMA or protocol['design'] != DESIGN or protocol['runner_sha256'] != sha(Path(__file__)) or protocol['input_protocol_sha256'] != sha(args.input_protocol):
        raise ValueError('protocol binding mismatch')
    input_protocol = json.loads(args.input_protocol.read_text())
    data = {s: load_split(args.task, s, input_protocol, args.panels, args.cache_root) for s in SPLITS}
    if set(data['training']['groups']) & set(data['development']['groups']):
        raise ValueError('Training/Dev source overlap')
    reject_cross_split_duplicate_inputs(data['training'], data['development'])
    t, d = data['training'], data['development']
    row_groups = np.empty(len(t['target']), dtype=t['groups'].dtype)
    row_groups[t['rows']] = t['groups'][:, None]
    row_ids = np.empty(len(t['target']), dtype=t['ids'].dtype)
    row_ids[t['rows']] = t['ids'][:, None]
    arrays = {}
    for split, item in data.items():
        for k in ('target', 'rows', 'ids', 'groups', 'original_groups'):
            arrays[f'{split}_{dict(rows="pair_rows", ids="query_ids", groups="source_groups", original_groups="original_source_groups").get(k,k)}'] = item[k]
        arrays[f'{split}_native_prediction'] = item['native']
    selected, scores = {}, {}
    fitted = {}
    with threadpool_limits(limits=2, user_api='blas'):
        for kind, mode in (('linear', 'linear_cv'), ('rbf', 'rbf_cv')):
            choice, records, folds = cv_select(t['full'], t['target'], row_groups, row_ids, kind)
            selected[mode], scores[mode] = choice, records
            pred_t, pred_d, model = fit_predict(t['full'], d['full'], t['target'], kind, choice['alpha'], choice['multiplier'])
            arrays[f'training_{mode}_prediction'], arrays[f'development_{mode}_prediction'] = pred_t, pred_d
            fitted[mode] = model
        for mode, input_key, alpha in (('rbf_interpolation', 'full', DESIGN['interpolation_alpha']),
                                       ('current_control', 'current', selected['rbf_cv']['alpha'])):
            multiplier = DESIGN['interpolation_bandwidth_multiplier'] if mode == 'rbf_interpolation' else selected['rbf_cv']['multiplier']
            pt, pd, model = fit_predict(t[input_key], d[input_key], t['target'], 'rbf', alpha, multiplier)
            arrays[f'training_{mode}_prediction'], arrays[f'development_{mode}_prediction'] = pt, pd
            fitted[mode] = model
    args.output.mkdir(parents=True)
    array_path = args.output/'predictions.npz'
    np.savez_compressed(array_path, **arrays)
    coefficient_path = args.output/'coefficients.npz'
    np.savez_compressed(coefficient_path, **{f'{mode}_{name}': val for mode, model in fitted.items() for name, val in model.items()})
    result = dict(schema=SCHEMA, task=args.task, protocol_sha256=sha(args.protocol),
                  input_protocol_sha256=sha(args.input_protocol), design=DESIGN, selected=selected,
                  cv_scores=scores, source_hashes={s: data[s]['hashes'] for s in SPLITS},
                  bindings=input_protocol['tasks'][args.task],
                  source_identity_scope=('split-local original source groups in motion_damping panel' if args.task == 'motion_damping' else 'original cross-split source groups'),
                  cross_split_exact_input_overlap_count=0,
                  counts={s: dict(queries=len(data[s]['rows']), source_groups=len(set(data[s]['groups']))) for s in SPLITS},
                  metrics={s: {mode: (dict(metrics(data[s]['target'], arrays[f'{s}_{mode}_prediction'], data[s]['rows']), row_centered_target_error=None) if mode == 'native' else metrics(data[s]['target'], arrays[f'{s}_{mode}_prediction'], data[s]['rows'])) for mode in MODES} for s in SPLITS},
                  array_sha256=sha(array_path), coefficients_sha256=sha(coefficient_path),
                  completion_state='complete', no_test_read=True, no_checkpoint_update=True,
                  interpretation='Privileged centered future targets supervise Training only; inference receives single-row history and actions, no labels, pair mean, or pair features. This is a response map, not a full future model or physical accuracy claim.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=('contact_friction', 'motion_damping'))
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--input-protocol', type=Path, default=Path('docs/research/data/pldm_fixed_representation_intervention_v1/sources/protocol.json'))
    p.add_argument('--panels', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    p.add_argument('--cache-root', type=Path, default=Path('/tmp/cw-native-policy-contrast-20261011'))
    run(p.parse_args())


if __name__ == '__main__':
    main()
