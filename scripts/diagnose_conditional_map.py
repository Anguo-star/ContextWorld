#!/usr/bin/env python3
"""Train-only paired-difference map between existing LeWM joint and frozen latents."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from diagnose_task_history_readout import helper, sha, write
from diagnose_matched_encoder_contrast import conditional_fraction

SEED = 20261010
TASKS = ('contact_friction', 'robot_arm_mass')
ALPHAS = (1e-6, 1e-4, 1e-2, 1., 1e2, 1e4)


def pair_diff(values, data):
    labels = np.asarray(data['labels'])
    queries = np.asarray(data['query_ids']).astype(str)
    assert set(np.unique(labels)) == {0, 1}
    order = np.lexsort((labels, queries))
    q = queries[order]
    l = labels[order]
    assert len(q) % 2 == 0 and np.array_equal(
        q[::2], q[1::2]) and np.all(l[::2] == 0) and np.all(l[1::2] == 1)
    groups = np.asarray(data['source_groups']).astype(str)[order]
    assert np.array_equal(groups[::2], groups[1::2])
    return np.asarray(values)[order][1::2]-np.asarray(values)[order][::2], q[::2], groups[::2]


def metric(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    p = float(np.square(pred).sum())
    t = float(np.square(target).sum())
    cross = float((pred*target).sum())
    assert t > 0
    return dict(nre=float(np.square(pred-target).sum()/t), amplitude=(p/t)**.5,
                gain=cross/t, alignment=cross/np.sqrt(p*t) if p > 0 else 0., prediction_energy=p, target_energy=t)


def grouped_bootstrap(pred, target, groups, n=2000):
    keys, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(SEED)
    terms = np.stack([np.square(pred-target).sum(1), np.square(pred).sum(1),
                     np.square(target).sum(1), (pred*target).sum(1)], 1)
    agg = np.zeros((len(keys), 4))
    np.add.at(agg, inv, terms)
    draw = rng.integers(len(keys), size=(n, len(keys)))
    sums = agg[draw].sum(1)
    nre = sums[:, 0]/sums[:, 2]
    gain = np.sqrt(sums[:, 1]/sums[:, 2])
    align = sums[:, 3]/np.sqrt(sums[:, 1]*sums[:, 2])
    return dict(seed=SEED, resamples=n, source_groups=len(keys), nre_ci95=np.quantile(nre, [.025, .975]).tolist(), amplitude_ci95=np.quantile(gain, [.025, .975]).tolist(), gain_ci95=np.quantile(sums[:, 3]/sums[:, 2], [.025, .975]).tolist(), alignment_ci95=np.quantile(align, [.025, .975]).tolist())


def kernel(x, z, alpha):
    from scipy.linalg import solve
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    gram = x@x.T
    gram.flat[::len(x)+1] += alpha
    return x.T@solve(gram, z, assume_a='pos', check_finite=False)


def select_alpha(x, y, groups):
    from sklearn.model_selection import GroupKFold
    unique = np.unique(groups)
    folds = min(5, len(unique))
    assert folds >= 2
    scores = {}
    for alpha in ALPHAS:
        errors = []
        energy = []
        for tr, va in GroupKFold(n_splits=folds).split(x, groups=groups):
            coef = kernel(x[tr], y[tr], alpha)
            errors.append(float(np.square(x[va]@coef-y[va]).sum()))
            energy.append(float(np.square(y[va]).sum()))
        scores[str(alpha)] = sum(errors)/sum(energy)
    choice = min(ALPHAS, key=lambda a: (scores[str(a)], a))
    return choice, scores


def run(task, args):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    panel = args.panels/task
    manifest_path = panel/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    specs = json.loads(args.models.read_text())
    out = args.output/task
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for split in ('training', 'development'):
        info = manifest['splits'][split]
        path = panel/info['path']
        assert sha(path) == info['sha256']
        with np.load(path, allow_pickle=False) as raw:
            data[split] = {k: raw[k] for k in raw.files}
    import subprocess
    import sys
    latent = {}
    model_meta = {}
    for regime in ('joint', 'frozen'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', regime, '--task', task, '--device',
                       args.device, '--models', str(args.models), '--panels', str(args.panels), '--output', str(args.output)], check=True)
        latent[regime] = {}
        for split in ('training', 'development'):
            path = out/f'{regime}_{split}.npz'
            with np.load(path, allow_pickle=False) as raw:
                latent[regime][split] = {
                    'target': raw['target'], 'pred': raw['prediction'] if 'prediction' in raw else None}
        model_meta[regime] = json.loads(
            (out/f'{regime}_meta.json').read_text())
    pairs = {}
    native = {}
    for split in ('training', 'development'):
        d = data[split]
        pairs[split] = {}
        for regime in ('joint', 'frozen'):
            pairs[split][regime] = {}
            for kind in ('target', 'pred'):
                v = latent[regime][split][kind]
                if v is not None:
                    diff, q, g = pair_diff(v, d)
                    pairs[split][regime][kind] = diff
                    pairs[split][regime]['queries'] = q
                    pairs[split][regime]['groups'] = g
        assert np.array_equal(
            pairs[split]['joint']['queries'], pairs[split]['frozen']['queries'])
        assert np.array_equal(
            pairs[split]['joint']['groups'], pairs[split]['frozen']['groups'])
    for regime in ('joint', 'frozen'):
        d = data['development']
        actual = pairs['development'][regime]['target']
        pred = pairs['development'][regime]['pred']
        native[regime] = dict(target_conditional_fraction=conditional_fraction(latent[regime]['development']['target'], d['query_ids'])[
                              'conditional_variance_fraction'], response=metric(pred, actual), paired_queries=len(actual))
    x = pairs['training']['joint']['target']
    y = pairs['training']['frozen']['target']
    groups = pairs['training']['joint']['groups']
    alpha, cv = select_alpha(x, y, groups)
    coef = kernel(x, y, alpha)
    xdev = pairs['development']['joint']['target']
    ydev = pairs['development']['frozen']['target']
    mapped = xdev@coef
    fidelity = metric(mapped, ydev)
    fidelity['source_group_bootstrap'] = grouped_bootstrap(
        mapped, ydev, pairs['development']['joint']['groups'])
    gate = bool(fidelity['nre'] < .1 and .8 < fidelity['gain']
                < 1.2 and fidelity['alignment'] > .95)
    pred_joint = pairs['development']['joint']['pred']
    mapped_pred = pred_joint@coef
    support = dict(train_true_norm_q=np.quantile(np.linalg.norm(x, axis=1), [.05, .5, .95]).tolist(), dev_true_norm_q=np.quantile(np.linalg.norm(
        xdev, axis=1), [.05, .5, .95]).tolist(), dev_pred_norm_q=np.quantile(np.linalg.norm(pred_joint, axis=1), [.05, .5, .95]).tolist())
    # Near-zero responses remain within the map's linear domain; screen only excessive norms.
    norms = np.linalg.norm(pred_joint, axis=1)
    hi = np.quantile(np.linalg.norm(x, axis=1), .99)
    support['above_train_99_fraction'] = float(np.mean(norms > hi))
    result = dict(schema='contextworld.conditional_map.v1', task=task, model_meta=model_meta, panel_manifest_sha256=sha(manifest_path), split_sha256={s: manifest['splits'][s]['sha256'] for s in data}, pair_definition='label 1 minus label 0, matched query IDs', map='no-intercept Ridge on Training true target differences only',
                  ridge_alpha=alpha, train_group_cv_nre=cv, native=native, true_target_map_fidelity=fidelity, gate_pass=gate, prediction_support=support, no_model_update=True, no_test_read=True, limitations='A failed Train-fitted linear bridge does not establish loss of physical information. Affine map is diagnostic, not a causal encoder ablation.')
    if gate and support['above_train_99_fraction'] < .1:
        result['common_reference_predictor'] = dict(mapped_joint=metric(mapped_pred, ydev), native_frozen=metric(
            pairs['development']['frozen']['pred'], ydev), interpretation='diagnostic comparison in frozen T3 target space')
    else:
        result['common_reference_predictor'] = 'uninterpretable: true-target fidelity gate or prediction support screen failed'
    write(out/'result.json', result)
    return result


def worker(task, regime, args):
    import torch
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    panel = args.panels/task
    manifest = json.loads((panel/'manifest.json').read_text())
    out = args.output/task
    out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.models.read_text())
    model_seed = '3073' if regime == 'original' else '3072'
    spec = next(s for s in specs if s['id'] ==
                f'{task}/lewm/{regime}/s{model_seed}')
    adapter = helper().load_adapter(
        spec, manifest['normalization'], out, args.device)
    adapter.model.eval()
    before = adapter.frozen_state_hash()
    meta = dict(
        model_id=spec['id'], checkpoint_sha256=spec['checkpoint_sha256'], state_hash_before=before)
    for split in ('training', 'development'):
        info = manifest['splits'][split]
        path = panel/info['path']
        assert sha(path) == info['sha256']
        with np.load(path, allow_pickle=False) as raw:
            d = {k: raw[k] for k in raw.files}
        h, y = helper().encode_unique(
            adapter, d['history_pixels'], d['queryfuture_pixels'])
        pred = helper().predict_all(adapter, h, y[:, None], d['action_blocks'], 'lewm', modes=(
            'free',))['free'][:, 0] if split == 'development' else None
        path = out/f'{regime}_{split}.npz'
        np.savez_compressed(
            path, target=y, **({'prediction': pred} if pred is not None else {}))
        meta[f'{split}_latent'] = dict(path=str(path), sha256=sha(path), target_shape=list(
            y.shape), prediction_shape=list(pred.shape) if pred is not None else None)
        print(task, regime, split, y.shape, flush=True)
    meta['state_hash_after'] = adapter.frozen_state_hash()
    assert before == meta['state_hash_after']
    write(out/f'{regime}_meta.json', meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=TASKS, action='append')
    p.add_argument('--worker', choices=('joint', 'frozen', 'original'))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--models', type=Path,
                   default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    p.add_argument('--panels', type=Path,
                   default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    p.add_argument('--output', type=Path,
                   default=Path('/tmp/cw-conditional-map-20261010'))
    a = p.parse_args()

    if a.worker:
        assert a.task and len(a.task) == 1
        worker(a.task[0], a.worker, a)
    else:
        for task in a.task or TASKS:
            run(task, a)


if __name__ == '__main__':
    main()
