#!/usr/bin/env python3
"""Summarize frozen Speed candidate choices using query-level paired intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_speed_action_selection import TRACKS, sha, write


def selection_metrics(physical, predicted, encoded):
    """Return per-query metrics; equal weights over actual speed conditions.

    Arrays have shape [query, history/actual-condition, candidate]. Ties choose
    the first candidate for every condition. Wrong histories are averaged over
    every other speed, so no particular unfavorable control is selected.
    """
    physical, predicted, encoded = [
        np.asarray(value, np.float64) for value in (physical, predicted, encoded)
    ]
    if not (physical.shape == predicted.shape == encoded.shape):
        raise ValueError('Cost shapes differ')
    if physical.ndim != 3 or physical.shape[1] < 2 or physical.shape[2] < 2:
        raise ValueError('Expected [query, condition>=2, candidate>=2]')
    if not all(np.isfinite(x).all() for x in (physical, predicted, encoded)):
        raise ValueError('Non-finite candidate costs')
    n, conditions, _ = physical.shape
    choice = np.argmin(predicted, axis=-1)
    actual = np.arange(conditions)
    lower = physical.min(axis=-1)
    # Costs for every actual-condition x history-condition assignment.
    costs = physical[np.arange(n)[:, None, None], actual[None, :, None], choice[:, None, :]]
    correct = costs[:, actual, actual]
    wrong = (costs.sum(axis=-1) - correct) / (conditions - 1)
    encoded_choice = np.argmin(encoded, axis=-1)
    encoded_cost = np.take_along_axis(physical, encoded_choice[..., None], axis=-1)[..., 0]
    return {
        'correct_regret': (correct - lower).mean(axis=-1),
        'wrong_regret': (wrong - lower).mean(axis=-1),
        'history_benefit': (wrong - correct).mean(axis=-1),
        'encoded_true_regret': (encoded_cost - lower).mean(axis=-1),
        'correct_physical_cost': correct.mean(axis=-1),
        'oracle_physical_cost': lower.mean(axis=-1),
        'history_blind_lower_bound': physical.mean(axis=1).min(axis=-1) - lower.mean(axis=-1),
    }


def interval(values):
    values = np.asarray(values, np.float64)
    samples = np.random.default_rng(20260929).integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
    return {'mean': float(values.mean()), 'ci95': [float(low), float(high)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--t0', type=Path, required=True)
    parser.add_argument('--t1', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.panel / 'manifest.json').read_text())
    panel_sha = sha(args.panel / 'manifest.json')
    rows, arrays, identities = [], {}, {}
    for scheme, directory in [('T0', args.t0), ('T1', args.t1)]:
        receipts = [json.loads(p.read_text()) for p in sorted(directory.glob('receipt_*.json'))]
        assert receipts and len(receipts) == receipts[0]['shards']
        assert {r['shard'] for r in receipts} == set(range(len(receipts)))
        assert sum(r['queries'] for r in receipts) == len(manifest['queries'])
        assert all(r['completed'] and r['panel_sha256'] == panel_sha for r in receipts)
        assert len({r['state_hash_before'] for r in receipts}) == 1
        assert all(r['state_hash_before'] == r['state_hash_after'] for r in receipts)
        assert all(c['selected_candidates_identical']
                   for r in receipts for c in r['canonical_checks'])
        identities[scheme] = receipts[0]['checkpoint']
        identities[scheme]['canonical_checks'] = sum(len(r['canonical_checks']) for r in receipts)
        identities[scheme]['state_hash'] = receipts[0]['state_hash_before']
        for track in TRACKS:
            queries = [r for r in manifest['queries'] if r['track'] == track]
            physical, predicted, encoded, native, target = [], [], [], [], []
            for entry in queries:
                data = np.load(directory / entry['path'], allow_pickle=False)
                assert str(data['query_id']) == entry['query_id']
                assert str(data['panel_sha256']) == panel_sha
                assert str(data['checkpoint_sha256']) == identities[scheme]['checkpoint_sha256']
                physical.append(data['physical_cost'])
                predicted.append(data['predicted_cost'])
                encoded.append(data['encoded_true_cost'])
                native.append(data['native_prediction'])
                target.append(data['native_target'])
            metrics = selection_metrics(physical, predicted, encoded)
            arrays[scheme, track] = metrics
            p, t = np.asarray(native, np.float64), np.asarray(target, np.float64)
            losses = ((p[:, :, None] - t[:, None, :]) ** 2).mean(axis=-1)
            diag = np.diagonal(losses, axis1=1, axis2=2).copy()
            ix = np.arange(losses.shape[1])
            losses[:, ix, ix] = np.inf
            strict = float((diag < losses.min(axis=1)).mean() * 100)
            rows.append({
                'model': 'pldm' if 'pldm' in identities[scheme]['adapter_id'] else 'lewm',
                'scheme': scheme, 'track': track,
                'queries': len(queries), 'conditions': len(queries[0]['speeds']),
                'metrics': {key: interval(value) for key, value in metrics.items()},
                'native_action_strict_history_win_percent': strict,
                'per_query': [dict(query_id=entry['query_id'], **{key: float(value[i]) for key, value in metrics.items()}) for i, entry in enumerate(queries)],
            })
    if identities['T0']['adapter_id'] != identities['T1']['adapter_id']:
        raise ValueError('T0/T1 must use the same model family')
    comparisons = {
        track: interval(arrays['T0', track]['correct_regret'] -
                        arrays['T1', track]['correct_regret'])
        for track in TRACKS
    }
    for identity in identities.values():
        path = Path(identity.pop('checkpoint'))
        identity['checkpoint_name'] = '/'.join(path.parts[-2:])
        identity.pop('stable_worldmodel_repo', None)
    public_protocol = dict(manifest['protocol'])
    public_protocol.pop('stable_repo', None)
    result = {
        'schema_version': 'contextworld.speed_action_selection_results.v1',
        'evaluation_split': 'development', 'unit': 'pixels',
        'uncertainty': '10000 percentile bootstrap resamples of static queries; conditions remain clustered',
        'scope': 'Specific existing LeWM checkpoints; not a training-seed population estimate or closed-loop CEM.',
        'panel_manifest_sha256': panel_sha, 'panel_summary': manifest['summary'],
        'protocol': public_protocol, 'models': identities, 'rows': rows,
        'training_benefit_t0_minus_t1': comparisons,
    }
    write(args.output, result)
    for r in rows:
        print(r['scheme'], r['track'], {k: round(v['mean'], 4) for k, v in r['metrics'].items(
        )}, 'native', r['native_action_strict_history_win_percent'])
    print('training benefit', json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
