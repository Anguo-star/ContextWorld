#!/usr/bin/env python3
"""Read-only native first-endpoint diagnostics on matched Train/Dev queries."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_task_history_readout import helper, native_history, sha, write  # noqa: E402
from diagnose_matched_encoder_contrast import conditional_fraction, readout  # noqa: E402


def matched_pairs(data):
    """Return query rows in label order and the reverse-row permutation."""
    ids = np.asarray(data['query_ids']).astype(str)
    labels = np.asarray(data['labels'])
    groups = np.asarray(data['source_groups']).astype(str)
    order = []
    for query in np.unique(ids):
        rows = np.flatnonzero(ids == query)
        assert len(rows) == 2 and len(np.unique(labels[rows])) == 2, query
        rows = rows[np.argsort(labels[rows])]
        assert groups[rows[0]] == groups[rows[1]], query
        assert np.array_equal(data['history_pixels'][rows[0], -1], data['history_pixels'][rows[1], -1]), query
        assert np.array_equal(data['rawactions'][rows[0]], data['rawactions'][rows[1]]), query
        if 'action_blocks' in data:
            assert np.array_equal(data['action_blocks'][rows[0]], data['action_blocks'][rows[1]]), query
        order.append(rows)
    pair_rows = np.asarray(order, dtype=np.int64)
    assert len(pair_rows) * 2 == len(ids)
    swap = np.arange(len(ids))
    swap[pair_rows[:, 0]] = pair_rows[:, 1]
    swap[pair_rows[:, 1]] = pair_rows[:, 0]
    return pair_rows, swap


def pair_statistics(target, correct, swapped, pair_rows):
    """Per-query squared energies. The two-condition ANOVA is E=C+R."""
    y = np.asarray(target, np.float64)[pair_rows]
    p = np.asarray(correct, np.float64)[pair_rows]
    w = np.asarray(swapped, np.float64)[pair_rows]
    assert y.shape == p.shape == w.shape and y.shape[1] == 2
    center_y = y.mean(1)
    center_p = p.mean(1)
    b = np.square(y - center_y[:, None]).sum(axis=(1, 2), dtype=np.float64)
    c = 2 * np.square(center_p - center_y).sum(axis=1, dtype=np.float64)
    r = np.square((p - center_p[:, None]) - (y - center_y[:, None])).sum(axis=(1, 2), dtype=np.float64)
    e = np.square(p - y).sum(axis=(1, 2), dtype=np.float64)
    wrong = np.square(w - y).sum(axis=(1, 2), dtype=np.float64)
    np.testing.assert_allclose(e, c + r, rtol=1e-10, atol=1e-8)
    # Main choice is nearest native target of the same query; only within-checkpoint.
    distances = np.square(p[:, :, None] - y[:, None, :]).sum(axis=-1, dtype=np.float64)
    main_choice = ((distances[:, 0, 0] < distances[:, 0, 1]).astype(np.float64)
                   + (distances[:, 1, 1] < distances[:, 1, 0]).astype(np.float64)) / 2
    row_correct = np.square(p-y).sum(axis=2, dtype=np.float64)
    row_wrong = np.square(w-y).sum(axis=2, dtype=np.float64)
    return dict(B=b, E=e, W=wrong, C=c, R=r, main_choice=main_choice,
                history_wins=(row_wrong > row_correct).mean(axis=1))


def aggregate(stats):
    sums = {key: float(np.asarray(value, np.float64).sum(dtype=np.float64)) for key, value in stats.items()}
    assert sums['B'] > 0
    return dict(energy=sums, E_over_B=sums['E']/sums['B'], NRE=sums['R']/sums['B'],
                G=(sums['W']-sums['E'])/sums['B'],
                main_choice=float(np.mean(stats['main_choice'], dtype=np.float64)),
                history_wins=float(np.mean(stats['history_wins'], dtype=np.float64)))


def source_bootstrap(stats, groups, draws=2000, seed=20261010):
    keys, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    sums = {k: np.bincount(inverse, weights=np.asarray(v, np.float64), minlength=len(keys))
            for k, v in stats.items()}
    sampled = np.random.default_rng(seed).integers(len(keys), size=(draws, len(keys)))
    b = sums['B'][sampled].sum(1)
    assert np.all(b > 0)
    e = sums['E'][sampled].sum(1)
    intervals = {'E_over_B': e/b, 'NRE': sums['R'][sampled].sum(1)/b,
                 'G': (sums['W'][sampled].sum(1)-e)/b}
    return dict(source_groups=len(keys), seed=seed, resamples=draws,
                interval_95={k: np.quantile(v, [.025, .975]).tolist() for k, v in intervals.items()})


def tree_readout(train_x, dev_x, train_y, dev_y):
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import balanced_accuracy_score
    model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features='sqrt',
                                 random_state=20261011, n_jobs=2)
    model.fit(train_x, train_y)
    prediction = model.predict(dev_x)
    return dict(accuracy=float(np.mean(prediction == dev_y)),
                balanced_accuracy=float(balanced_accuracy_score(dev_y, prediction)),
                classes=np.unique(train_y).tolist())


def run(args):
    import torch
    from contextworld.evaluation.protocol import _prepare_optional_flash_attention
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2, user_api='blas')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _prepare_optional_flash_attention()
    candidates = [x for x in json.loads(args.models.read_text())
                  if x['task'] == args.task and x['family'] == args.family and
                  x['regime'] == args.regime and
                  x['training_seed'] == (3073 if args.regime == 'original' else 3072)]
    assert len(candidates) == 1, f'Expected one matching model, found {len(candidates)}'
    spec = candidates[0]
    assert sha(spec['checkpoint']) == spec['checkpoint_sha256']
    panel = args.panels / args.task
    manifest_path = panel / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    output = args.output / args.task / args.family / args.regime
    output.mkdir(parents=True, exist_ok=True)
    adapter = helper().load_adapter(spec, manifest['normalization'], output, args.device)
    adapter.model.eval()
    before = adapter.frozen_state_hash()
    splits = {}
    for split, expected in (('training', 512), ('development', 256)):
        info = manifest['splits'][split]
        path = panel / info['path']
        assert sha(path) == info['sha256']
        with np.load(path, allow_pickle=False) as raw:
            data = {key: raw[key] for key in raw.files}
        rows, swap = matched_pairs(data)
        assert len(rows) == expected and len(data['labels']) == expected * 2
        assert len(np.unique(data['source_groups'][rows[:, 0]])) == info['source_group_count']
        h, y = helper().encode_unique(adapter, data['history_pixels'], data['queryfuture_pixels'])
        actions = data['rawactions'] if 'action_blocks' not in data else data['action_blocks']
        with torch.inference_mode():
            p = helper().predict_all(adapter, h, y[:, None], actions, args.family, modes=('free',))['free'][:, 0]
            w = helper().predict_all(adapter, h[swap], y[:, None], actions, args.family, modes=('free',))['free'][:, 0]
        swap_difference = float(np.max(np.abs(w-p[swap])))
        assert swap_difference < 3e-4, swap_difference
        indices = np.array([0, len(h)-1])
        native = adapter.rollout_latents(data['history_pixels'][indices],
                                         data['action_blocks'][indices, :h.shape[1]], batch_size=2)[:, 0]
        rollout_difference = float(np.max(np.abs(native-p[indices])))
        assert rollout_difference < 3e-4, rollout_difference
        stats = pair_statistics(y, p, w, rows)
        ids = data['query_ids'][rows[:, 0]].astype(str)
        groups = data['source_groups'][rows[:, 0]].astype(str)
        stats_path = output / f'{split}_query_stats.npz'
        np.savez_compressed(stats_path, query_ids=ids, source_groups=groups,
                            labels=data['labels'][rows], **stats)
        if args.save_latents:
            np.savez_compressed(output / f'{split}_native_latents.npz', history=h, target=y,
                                correct_prediction=p, swapped_prediction=w,
                                query_ids=data['query_ids'], source_groups=data['source_groups'])
        splits[split] = dict(data=data, history=h, target=y, pair_rows=rows, stats=stats,
                             summary=aggregate(stats), bootstrap=source_bootstrap(stats, groups),
                             target_conditional_fraction=conditional_fraction(y, data['query_ids']),
                             query_stats_sha256=sha(stats_path),
                             max_abs_swap_prediction_difference=swap_difference,
                             max_abs_native_rollout_difference=rollout_difference)
    train, dev = splits['training'], splits['development']
    train_x = native_history(train['history']); dev_x = native_history(dev['history'])
    ty, dy = train['data']['labels'], dev['data']['labels']
    train_current, dev_current = train['history'][:, -1], dev['history'][:, -1]
    readouts = dict(history=dict(ridge=readout(train_x, dev_x, ty, dy),
                                 extra_trees=tree_readout(train_x, dev_x, ty, dy)),
                    current=dict(ridge=readout(train_current, dev_current, ty, dy),
                                 extra_trees=tree_readout(train_current, dev_current, ty, dy)))
    after = adapter.frozen_state_hash()
    assert before == after
    result = dict(schema='contextworld.native_policy_contrast.v1', model_id=spec['id'],
                  task=args.task, family=args.family, regime=args.regime,
                  checkpoint_sha256=spec['checkpoint_sha256'], panel_manifest_sha256=sha(manifest_path),
                  split_sha256={s: manifest['splits'][s]['sha256'] for s in splits},
                  model_state_hash_before=before, model_state_hash_after=after,
                  no_model_update=True, no_test_read=True,
                  metrics={s: {k: v for k, v in item.items() if k in
                           ('summary', 'bootstrap', 'target_conditional_fraction', 'query_stats_sha256',
                            'max_abs_swap_prediction_difference', 'max_abs_native_rollout_difference')}
                           for s, item in splits.items()},
                  readout=readouts,
                  readout_note='Training-fitted StandardScaler+Ridge and fixed ExtraTrees(200, leaf=2, sqrt, seed=20261011) on native visual history/current only; historical actions are omitted. Auxiliary readability, not native ICL.',
                  comparison_note='All latent metrics are within-checkpoint normalized; do not rank physical accuracy across changing target spaces.')
    write(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('contact_friction', 'motion_damping'), required=True)
    parser.add_argument('--family', choices=('pldm', 'lewm'), required=True)
    parser.add_argument('--regime', choices=('original', 'joint', 'frozen'), required=True)
    parser.add_argument('--models', type=Path, default=Path('/tmp/cw-icl-validity-20261007/models.json'))
    parser.add_argument('--panels', type=Path, default=Path('/tmp/cw-cross-task-mechanism-20261010/native'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/cw-native-policy-contrast-20261011'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save-latents', action='store_true',
                        help='Cache native history, target, and both predictions for CPU reanalysis')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
