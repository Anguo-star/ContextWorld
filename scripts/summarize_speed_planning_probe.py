#!/usr/bin/env python3
"""Summarize fixed-candidate rollout diagnostics at the scene level."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from build_speed_action_selection import sha, write


def decision_opportunity(cost, states):
    """Value of speed-specific choices versus an optimal shared candidate.

    Both selectors know the candidate consequences. Only the former knows the
    actual speed. This is not a bound on a model's correct/wrong-history gain.
    """
    rows = []
    for t in range(cost.shape[2]):
        c, s = cost[:, :, t], states[:, :, t]
        best = c.min(axis=1)
        shared = c.mean(axis=0).min()
        rows.append(dict(raw_steps=5*(t+1), oracle_distance=float(best.mean()),
                         shared_candidate_oracle_distance=float(shared),
                         context_value=float(shared-best.mean()),
                         common_candidate_within_one_px=bool(np.any(np.all(c <= best[:, None]+1, axis=0))),
                         mean_pair_future_separation=float(np.mean([
                             np.linalg.norm(s[j]-s[i], axis=-1).mean()
                             for i, j in itertools.combinations(range(3), 2)]))))
    return rows


def measurements(path):
    with np.load(path, allow_pickle=False) as d:
        pred, target, initial, goal, cost = [np.asarray(d[k], np.float64) for k in ('prediction', 'target', 'initial', 'goal', 'physical_cost')]
    # Arrays: history/speed × candidate × rollout depth × latent dimension.
    assert pred.shape == target.shape and pred.shape[:3] == cost.shape
    pred_cost = np.square(pred-goal).sum(axis=-1)
    true_cost = np.square(target-goal).sum(axis=-1)
    rows = []
    for t in range(5):
        p, z = pred[:, :, t], target[:, :, t]
        delta_p = np.concatenate([p[j]-p[i] for i, j in itertools.combinations(range(3), 2)])
        delta_z = np.concatenate([z[j]-z[i] for i, j in itertools.combinations(range(3), 2)])
        target_energy = float(np.square(delta_z).sum())
        persistence_energy = float(np.square(z-initial[:, None]).sum())
        assert target_energy > 0 and persistence_energy > 0
        correct, wrong, encoded = [], [], []
        for si in range(3):
            physical = cost[si, :, t]
            best = physical.min()
            correct.append(float(physical[pred_cost[si, :, t].argmin()]-best))
            wrong.append(float(np.mean([physical[pred_cost[hi, :, t].argmin()]-best for hi in range(3) if hi != si])))
            encoded.append(float(physical[true_cost[si, :, t].argmin()]-best))
        rows.append(dict(raw_steps=5*(t+1), prediction_squared_error=float(np.square(p-z).sum()),
                         persistence_squared_error=persistence_energy,
                         response_squared_error=float(np.square(delta_p-delta_z).sum()),
                         target_response_energy=target_energy, response_dot=float((delta_p*delta_z).sum()),
                         latent_zero_response_fraction=float(np.mean(np.square(delta_z).sum(axis=-1) == 0)),
                         regret=float(np.mean(correct)), wrong_regret=float(np.mean(wrong)),
                         history_benefit=float(np.mean(wrong)-np.mean(correct)),
                         encoded_regret=float(np.mean(encoded)), oracle_distance=float(cost[:, :, t].min(axis=1).mean())))
    return rows


def aggregate(rows):
    total = lambda key: float(sum(x[key] for x in rows))
    return dict(prediction_error_ratio=total('prediction_squared_error')/total('persistence_squared_error'),
                response_score=100*(1-total('response_squared_error')/total('target_response_energy')),
                gain=total('response_dot')/total('target_response_energy'),
                **{k: float(np.mean([r[k] for r in rows])) for k in
                   ('regret', 'wrong_regret', 'history_benefit', 'encoded_regret', 'oracle_distance', 'latent_zero_response_fraction')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    panel = a.root/'panel'
    queries = sorted(p.name for p in panel.iterdir() if p.is_dir())
    assert len(queries) == 6
    arms = ('structured', 'issued', 'clipped', 'small')
    records, rows, models, inputs, action_coverage, opportunity = [], [], {}, [], [], []
    for qid in queries:
        for bank in ('structured', 'issued', 'small'):
            with np.load(panel/qid/f'{bank}.npz', allow_pickle=False) as d:
                opportunity.extend(dict(query_id=qid, arm=bank, **x) for x in
                                   decision_opportunity(d['physical_cost'].astype(np.float64), d['future_states'].astype(np.float64)))
        with np.load(panel/qid/'issued.npz', allow_pickle=False) as d:
            actions = d['actions']
            action_coverage.append(dict(query_id=qid, candidates=len(actions),
                                        clipped_coordinate_fraction=float(np.mean(np.abs(actions) > 1)),
                                        affected_candidate_fraction=float(np.mean(np.any(np.abs(actions) > 1, axis=(1, 2, 3))))))
    for scheme in ('T0', 'T1'):
        for qid in queries:
            receipt = json.loads((a.root/scheme/qid/'receipt.json').read_text())
            assert receipt['source_manifest_sha256'] == sha(panel/qid/'manifest.json')
            assert receipt['state_hash_before'] == receipt['state_hash_after']
            assert receipt['canonical_rollout_checks_passed']
            if scheme in models:
                assert models[scheme]['checkpoint_sha256'] == receipt['checkpoint_sha256']
            models[scheme] = receipt['model']
            # This is the sole same-future intervention: assert that physical
            # outcomes and latent targets have not changed between its arms.
            with np.load(a.root/scheme/qid/'issued.npz') as issued, np.load(a.root/scheme/qid/'clipped.npz') as clipped:
                for key in ('target', 'initial', 'goal', 'physical_cost'):
                    assert np.array_equal(issued[key], clipped[key]), (scheme, qid, key)
                assert np.array_equal(np.clip(issued['model_actions'], -1, 1), clipped['model_actions'])
            for arm in arms:
                path = a.root/scheme/qid/f'{arm}.npz'
                inputs.append(dict(path=str(path.relative_to(a.root)), sha256=sha(path)))
                records.extend(dict(scheme=scheme, arm=arm, query_id=qid, **x) for x in measurements(path))
        for arm in arms:
            for step in (5, 10, 15, 20, 25):
                selected = [x for x in records if x['scheme'] == scheme and x['arm'] == arm and x['raw_steps'] == step]
                assert len(selected) == 6
                rows.append(dict(scheme=scheme, arm=arm, raw_steps=step, **aggregate(selected)))
    # Paired scene bootstrap for clipping. Targets, candidates and physical costs
    # are identical in the two arms, so this contrast changes only model inputs.
    contrasts = []
    rng = np.random.default_rng(20260929)
    resamples = rng.integers(0, 6, size=(10000, 6))
    for scheme in ('T0', 'T1'):
        for step in (5, 25):
            rr = {arm: [next(x for x in records if x['query_id'] == q and x['scheme'] == scheme and x['arm'] == arm and x['raw_steps'] == step) for q in queries] for arm in ('issued', 'clipped')}
            for metric in ('prediction_error_ratio', 'response_score', 'regret'):
                samples = []
                for idx in resamples:
                    ag = {k: aggregate([v[i] for i in idx]) for k, v in rr.items()}
                    samples.append(ag['clipped'][metric]-ag['issued'][metric])
                contrasts.append(dict(scheme=scheme, raw_steps=step, metric=metric,
                                      clipped_minus_issued=aggregate(rr['clipped'])[metric]-aggregate(rr['issued'])[metric],
                                      paired_scene_ci95=np.quantile(samples, [.025, .975]).tolist()))
    protocol = json.loads((panel/queries[0]/'manifest.json').read_text())['protocol']
    protocol.pop('query_id')
    write(a.output, dict(schema='contextworld.speed_planning_horizon_probe.results.v1', protocol=protocol,
                         scenes=6, speed_conditions=[3.4, 4.8, 6.9], models=models, rows=rows,
                         paired_clipping_contrasts=contrasts, per_scene=records,
                         action_coverage=action_coverage, inputs=inputs,
                         decision_opportunity=opportunity,
                         definitions={'prediction_error_ratio': 'sum squared prediction error / sum squared error of persisting current encoded frame; 0 perfect,1 persistence',
                                      'response_score': '100*(1-response squared error/true response energy), pooled over speed pairs and candidates',
                                      'regret': 'physical candidate regret in pixels, averaged over speeds then scenes',
                                      'oracle_distance': 'distance of best candidate; candidate coverage differs across action banks',
                                      'context_value': 'min_k mean_speed J_speed(k) - mean_speed min_k J_speed(k); physical value of conditioning an otherwise optimal candidate choice, not an upper bound on model history gain'},
                         coverage='diagnostic on six fixed Development scenes; not a closed-loop intervention'))
    print(json.dumps([r for r in rows if r['raw_steps'] in (5, 25)], indent=2))


if __name__ == '__main__':
    main()
