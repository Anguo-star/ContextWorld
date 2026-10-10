#!/usr/bin/env python3
"""Fixed full-native history readout on Training and Development panels only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

SEED = 20261010
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def helper():
    spec = importlib.util.spec_from_file_location('cross_task_helper', REPO / 'scripts/diagnose_cross_task_decisions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_history(history):
    """Invertible time difference coordinates, preserving every native dimension."""
    assert history.ndim == 3 and history.shape[1] in (3, 7)
    return np.concatenate((history[:, :1], np.diff(history, axis=1)), axis=1).reshape(len(history), -1)


def _fit_dual(train, target, alpha=1.0):
    """StandardScaler + Ridge with intercept, in sample space and float64."""
    from scipy.linalg import solve
    x = np.asarray(train)
    assert x.ndim == 2
    mean = np.empty(x.shape[1], dtype=np.float64)
    scale = np.empty(x.shape[1], dtype=np.float64)
    for start in range(0, x.shape[1], 8192):
        stop = min(start + 8192, x.shape[1])
        block = x[:, start:stop].astype(np.float64)
        mean[start:stop] = block.mean(0)
        scale[start:stop] = block.std(0)
    scale[scale < 1e-12] = 1
    y = np.asarray(target, dtype=np.float64)
    ymean = y.mean(0)
    gram = np.zeros((len(x), len(x)), dtype=np.float64)
    for start in range(0, x.shape[1], 8192):
        stop = min(start + 8192, x.shape[1])
        block = (x[:, start:stop].astype(np.float64) - mean[start:stop]) / scale[start:stop]
        gram += block @ block.T
    gram.flat[::len(x) + 1] += alpha
    coefficients = solve(gram, y - ymean, assume_a='pos', overwrite_a=True, check_finite=False)
    return mean, scale, x, coefficients, ymean


def _predict_dual(fit, values):
    mean, scale, train, coefficients, ymean = fit
    values = np.asarray(values)
    cross = np.zeros((len(values), len(train)), dtype=np.float64)
    for start in range(0, train.shape[1], 8192):
        stop = min(start + 8192, train.shape[1])
        a = (values[:, start:stop].astype(np.float64) - mean[start:stop]) / scale[start:stop]
        b = (train[:, start:stop].astype(np.float64) - mean[start:stop]) / scale[start:stop]
        cross += a @ b.T
    return cross @ coefficients + ymean


def verify_dual():
    """Check exact sklearn Ridge/RidgeClassifier conventions before a real fit."""
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(37, 51)); x[:, 3] = 2; x[:, 4] = x[:, 5]
    v = rng.normal(size=(19, 51))
    for y in (rng.normal(size=37), rng.integers(0, 2, size=37), rng.integers(0, 4, size=37)):
        if np.issubdtype(y.dtype, np.floating):
            reference = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x, y).predict(v)
            target = y
            measured = _predict_dual(_fit_dual(x, target), v)
            assert np.allclose(reference, measured, atol=1e-9, rtol=1e-9)
        else:
            classes = np.unique(y)
            target = np.where(y[:, None] == classes[None], 1.0, -1.0)
            if len(classes) == 2:
                target = target[:, 1]
            reference = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0)).fit(x, y).decision_function(v)
            measured = _predict_dual(_fit_dual(x, target), v)
            assert np.allclose(reference, measured, atol=1e-9, rtol=1e-9)


def bootstrap(values, groups, statistic):
    keys, inverse = np.unique(groups.astype(str), return_inverse=True)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(keys), size=(2000, len(keys)))
    group_sum = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64), minlength=len(keys))
    group_count = np.bincount(inverse, minlength=len(keys))
    estimates = group_sum[draws].sum(1) / group_count[draws].sum(1)
    return dict(seed=SEED, resamples=2000, source_groups=len(keys), interval_95=np.quantile(estimates, [.025, .975]).tolist())


def bootstrap_balanced(labels, prediction, groups, classes):
    keys, inverse = np.unique(groups.astype(str), return_inverse=True)
    counts = np.zeros((len(keys), len(classes)), dtype=np.int64)
    correct = np.zeros_like(counts)
    for index, cls in enumerate(classes):
        selected = labels == cls
        counts[:, index] = np.bincount(inverse[selected], minlength=len(keys))
        correct[:, index] = np.bincount(inverse[selected & (labels == prediction)], minlength=len(keys))
    draws = np.random.default_rng(SEED).integers(len(keys), size=(2000, len(keys)))
    sampled_count = counts[draws].sum(1)
    sampled_correct = correct[draws].sum(1)
    available = sampled_count > 0
    rates = np.divide(sampled_correct, sampled_count, out=np.zeros_like(sampled_correct, dtype=np.float64), where=available)
    balanced = np.divide((rates * available).sum(1), available.sum(1),
        out=np.full(len(draws), np.nan), where=available.sum(1) > 0)
    return dict(seed=SEED, resamples=2000, source_groups=len(keys),
        statistic='balanced_accuracy_over_classes_present_in_each_resample',
        interval_95=np.nanquantile(balanced, [.025, .975]).tolist(),
        resamples_missing_at_least_one_class=int(np.sum(~available.all(1))))


def classify(train_x, dev_x, train, dev):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    classes = np.unique(train['labels'])
    assert len(classes) >= 2 and np.isin(dev['labels'], classes).all(), 'Unseen class: use a physical regression target'
    y = np.where(train['labels'][:, None] == classes[None], 1., -1.)
    if len(classes) == 2:
        y = y[:, 1]
    fit = _fit_dual(train_x, y)
    results = {}
    for split, x, data in [('training', train_x, train), ('development', dev_x, dev)]:
        scores = _predict_dual(fit, x)
        prediction = classes[(scores > 0).astype(int)] if len(classes) == 2 else classes[np.argmax(scores, axis=1)]
        labels = data['labels']; groups = data['source_groups']; correct = labels == prediction
        result = dict(accuracy=float(correct.mean()), balanced_accuracy=float(balanced_accuracy_score(labels, prediction)),
            class_accuracy={str(c): float(correct[labels == c].mean()) for c in np.unique(labels)},
            confusion_matrix=confusion_matrix(labels, prediction, labels=classes).tolist(),
            source_group_bootstrap=bootstrap_balanced(labels, prediction, groups, classes),
            predictions=[dict(pair_id=str(p), source_group=str(g), label=int(a), prediction=int(b), correct=bool(a == b))
                for p, g, a, b in zip(data['pair_ids'], groups, labels, prediction)])
        results[split] = result
    return results


def regress(train_x, dev_x, train, dev):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    target = np.asarray(train['physical_target'], dtype=np.float64)
    assert target.ndim == 1 and np.isfinite(target).all()
    mean = float(target.mean())
    fit = _fit_dual(train_x, target)
    results = {}
    for split, x, data in [('training', train_x, train), ('development', dev_x, dev)]:
        actual = np.asarray(data['physical_target'], dtype=np.float64)
        prediction = _predict_dual(fit, x).reshape(-1)
        baseline = np.full_like(actual, mean)
        result = dict(mae=float(mean_absolute_error(actual, prediction)), mse=float(mean_squared_error(actual, prediction)),
            fixed_training_mean=mean, baseline_mae=float(mean_absolute_error(actual, baseline)),
            baseline_mse=float(mean_squared_error(actual, baseline)),
            r2=float(r2_score(actual, prediction)),
            source_group_bootstrap=bootstrap(np.abs(actual - prediction), data['source_groups'], lambda v: float(v.mean())),
            predictions=[dict(pair_id=str(p), source_group=str(g), target=float(a), prediction=float(b),
                baseline_prediction=mean) for p, g, a, b in zip(data['pair_ids'], data['source_groups'], actual, prediction)])
        results[split] = result
    return results


def controls(data, declaration=None):
    declaration = declaration or {}
    group_key = declaration.get('matched_group_key', 'query_ids' if 'query_ids' in data else 'pair_ids')
    pairs = np.asarray(data[group_key]).astype(str)
    unique, counts = np.unique(pairs, return_counts=True)
    paired = unique[counts > 1]
    rgb_equal = action_equal = complete_label_groups = 0
    classes = np.unique(data['labels'])
    for pair in paired:
        rows = np.flatnonzero(pairs == pair)
        rgb_equal += int(all(np.array_equal(data['history_pixels'][rows[0], -1], data['history_pixels'][row, -1]) for row in rows[1:]))
        action_equal += int(all(np.array_equal(data['rawactions'][rows[0]], data['rawactions'][row]) for row in rows[1:]))
        complete_label_groups += int(len(rows) == len(classes) and np.array_equal(np.sort(data['labels'][rows]), classes))
    if declaration.get('current_action_matched', False):
        assert len(paired), 'Declared matched groups missing'
        assert rgb_equal == len(paired), 'Matched current image differs'
        assert action_equal == len(paired), 'Matched action blocks differ'
    complete = len(paired) > 0 and counts.min() > 1 and complete_label_groups == len(paired)
    matched = complete and rgb_equal == len(paired) and action_equal == len(paired)
    return dict(group_key=group_key, declared_current_action_matched=bool(declaration.get('current_action_matched', False)),
        repeated_groups=len(paired), current_rgb_equal_groups=rgb_equal, raw_action_equal_groups=action_equal,
        complete_label_groups=complete_label_groups,
        current_only_deterministic_balanced_accuracy_ceiling=1 / len(classes) if matched else None)


def run(args):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    verify_dual()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = json.loads(args.models.read_text())
    spec = next(row for row in models if row['id'] == args.id)
    manifest_path = args.panel / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    task_manifest = manifest.get('tasks', {}).get(spec['task'], manifest)
    normalization = task_manifest.get('normalization', manifest.get('normalization'))
    assert normalization and 'splits' in task_manifest
    args.output.mkdir(parents=True, exist_ok=True)
    adapter = helper().load_adapter(spec, normalization, args.output, args.device)
    adapter.model.eval()
    before = adapter.frozen_state_hash()
    data = {}; latent_files = {}; matched = {}
    for split in ('training', 'development'):
        info = task_manifest['splits'][split]
        source = args.panel / info['path']
        assert sha(source) == info['sha256']
        with np.load(source, allow_pickle=False) as raw:
            item = {key: raw[key] for key in raw.files}
        assert item['history_pixels'].ndim == 5 and item['history_pixels'].shape[-3:] == (224, 224, 3)
        assert item['queryfuture_pixels'].shape == (len(item['history_pixels']), 224, 224, 3)
        matched[split] = controls(item, info.get('matched_control'))
        history, future = helper().encode_unique(adapter, item['history_pixels'], item['queryfuture_pixels'])
        assert history.ndim == 3 and history.shape[0] == len(item['history_pixels'])
        assert future.shape == (len(history), history.shape[-1])
        assert np.isfinite(history).all() and np.isfinite(future).all()
        path = args.output / f'{split}_native.npz'
        retained = ('rawactions', 'pair_ids', 'query_ids', 'modes', 'labels', 'conditions',
            'physical_target', 'physical_group_labels', 'source_groups', 'episode_ids')
        np.savez(path, history=history, queryfuture=future,
            **{key: item[key] for key in retained if key in item})
        latent_files[split] = dict(path=str(path), sha256=sha(path), shape=list(history.shape), future_shape=list(future.shape))
        data[split] = {key: item[key] for key in retained if key in item}
        data[split]['history'] = history
        print(args.id, split, 'native', history.shape, flush=True)
    after = adapter.frozen_state_hash()
    assert before == after, 'Frozen world-model state changed'
    train_x = native_history(data['training']['history'])
    dev_x = native_history(data['development']['history'])
    assert train_x.shape[1] == dev_x.shape[1]
    regression = spec['task'] == 'speed'
    readout = regress(train_x, dev_x, data['training'], data['development']) if regression else classify(train_x, dev_x, data['training'], data['development'])
    readouts = {'history/ridge': readout}
    if spec['task'] == 'action_delay' and all('physical_group_labels' in data[s] for s in ('training', 'development')):
        grouped = {split: {**item, 'labels': item['physical_group_labels']} for split, item in data.items()}
        readouts['history/ridge_six_physical_groups'] = classify(train_x, dev_x, grouped['training'], grouped['development'])
    result = dict(schema='contextworld.task_full_native_history_readout.v1', model_id=args.id,
        task=spec['task'], family=spec['family'], checkpoint=spec['checkpoint'], checkpoint_sha256=spec['checkpoint_sha256'],
        panel_manifest_sha256=sha(manifest_path), program_sha256=sha(__file__),
        model_state_hash_before=before, model_state_hash_after=after, no_worldmodel_update=True, no_test_read=True,
        feature='concat(z0,z1-z0,...), all native coordinates; no pooling or projection',
        readout='StandardScaler(training only) + Ridge(alpha=1, intercept), exact dual solve in float64',
        sklearn_equivalence_verified=True, target='physical_speed_regression' if regression else 'exact_hidden_mode_classification',
        seed=SEED, native_cache=latent_files, matched_controls=matched, readouts=readouts,
        limitations='Readability is not rollout sufficiency; a fixed linear failure cannot establish absent encoder information.')
    write(args.output / 'result.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/panel'))
    parser.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    parser.add_argument('--id', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
