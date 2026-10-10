#!/usr/bin/env python3
"""Collect and render the four-cell predictor response intervention."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'docs/research/data/response_intervention_v1'
SOURCE = Path('/tmp/cw-response-intervention-20261010/results')
DOC = ROOT / 'docs/ICL_Metric_Study.md'
CELLS = [('contact_friction', 'lewm'), ('cube_gripper_carry', 'lewm'),
         ('robot_arm_mass', 'lewm'), ('action_delay', 'dinowm')]
NAMES = {'contact_friction': '接触摩擦', 'cube_gripper_carry': 'Cube 夹爪携带',
         'robot_arm_mass': '机械臂质量', 'action_delay': '动作延迟'}
FAMILIES = {'lewm': 'LeWM', 'dinowm': 'DINO-WM'}
STEPS = ('0', '64', '256')
BEGIN = '<!-- BEGIN RESPONSE_INTERVENTION -->'
END = '<!-- END RESPONSE_INTERVENTION -->'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def collect():
    for filename in ('protocol.json', 'input_inventory.json'):
        src = SOURCE.parent / filename
        require(src.is_file(), f'missing {src}')
        out = RAW / 'sources' / filename
        data = src.read_bytes()
        if out.exists():
            require(out.read_bytes() == data, f'immutable source changed: {out}')
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
    for task, family in CELLS:
        source = SOURCE / task / family
        dest = RAW / 'sources' / task / family
        for filename in ('result.json', 'protocol.json'):
            src = source / filename
            require(src.is_file(), f'missing {src}')
            out = dest / filename
            data = src.read_bytes()
            if out.exists():
                require(out.read_bytes() == data, f'immutable source changed: {out}')
            else:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)


def pooled(rows):
    require(bool(rows), 'empty query sample')
    energy = sum(r['target_response_energy'] for r in rows)
    require(energy > 0, 'zero target response energy')
    return {'response_nre': sum(r['response_mse'] for r in rows) / energy,
            'endpoint_mse': sum(r['endpoint_mse'] for r in rows) / len(rows),
            'common_mse': sum(r['common_mse'] for r in rows) / len(rows)}


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_group']].append(row)
    require(bool(groups), 'missing source groups')
    return groups


def resampled(rows, groups, indices):
    by_group = grouped(rows)
    return [row for idx in indices for row in by_group[groups[idx]]]


def paired_delta(native, weighted, initial=None, repetitions=4000, seed=20261010):
    ng, wg = grouped(native), grouped(weighted)
    ig = grouped(initial) if initial is not None else None
    require(ng.keys() == wg.keys() and (ig is None or ng.keys() == ig.keys()), 'paired source groups differ')
    keys = sorted(ng)
    require(all([r['query_id'] for r in ng[k]] == [r['query_id'] for r in wg[k]] and
                (ig is None or [r['query_id'] for r in ng[k]] == [r['query_id'] for r in ig[k]])
                for k in keys), 'paired query identities differ')
    def arrays(groups):
        return np.asarray([[sum(r['response_mse'] for r in groups[k]),
                            sum(r['target_response_energy'] for r in groups[k]),
                            sum(r['endpoint_mse'] for r in groups[k]), len(groups[k])]
                           for k in keys], dtype=float)
    n, w = arrays(ng), arrays(wg)
    i = arrays(ig) if ig is not None else n
    require(np.allclose(n[:, 1], w[:, 1], rtol=0, atol=1e-10), 'paired target energies differ')
    require(np.all(i[:, 2] > 0), 'nonpositive initial endpoint MSE')
    def metrics(a, b, c):
        nre = b[..., 0] / b[..., 1] - a[..., 0] / a[..., 1]
        mse = (b[..., 2] / b[..., 3] - a[..., 2] / a[..., 3]) / (c[..., 2] / c[..., 3])
        return nre, mse
    point_nre, point_mse = metrics(n.sum(0), w.sum(0), i.sum(0))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(repetitions, len(keys)))
    boot_nre, boot_mse = metrics(n[draws].sum(1), w[draws].sum(1), i[draws].sum(1))
    return {'weighted_minus_native': float(point_nre),
            'ci95': [float(x) for x in np.quantile(boot_nre, [0.025, 0.975])],
            'weighted_minus_native_endpoint_mse_over_initial': float(point_mse),
            'endpoint_mse_difference_ci95': [float(x) for x in np.quantile(boot_mse, [0.025, 0.975])],
            'source_groups': len(keys), 'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed}


def verify_cell(result, protocol, task, family):
    expected = f'{task}/{family}/scratch/s3072'
    require(result['schema'] == 'contextworld.predictor_response_intervention.v1', 'schema mismatch')
    require(result.get('completion_state') == 'complete', 'incomplete result')
    require(result.get('restored_model_state_hash') == result['source_model_state_hash'], 'source state not restored')
    require(result['model_id'] == protocol['model_id'] == expected, 'model ID mismatch')
    require(result['updates'] == 256 and result['checkpoints'] == [0, 64, 256], 'update schedule mismatch')
    require(result['batch_queries'] == 8 and len(result['batch_query_ids']) == 256, 'batch count mismatch')
    require(bool(result['batch_order_sha256']), 'batch schedule hash missing')
    require(not result['public_test_accessed'] and result['fixed_encoder_action_encoder'], 'scope mismatch')
    for key in protocol:
        require(result[key] == protocol[key], f'protocol mismatch: {key}')
    require(set(result['arms']) == {'native', 'response_weighted'}, 'arm coverage mismatch')
    native, weighted = (result['arms'][arm] for arm in ('native', 'response_weighted'))
    require(native['rest_weight'] == native['response_weight'] == 1, 'native weights mismatch')
    require([weighted['rest_weight'], weighted['response_weight']] == result['calibration']['reweighted_coefficients'], 'calibration mismatch')
    for arm in (native, weighted):
        require(arm['frozen_state_hash_after'] == result['frozen_state_hash'], 'frozen state changed')
        require(arm['first_update_norm'] is not None, 'first update missing')
        require(set(arm['snapshots']) == set(STEPS), 'snapshot coverage mismatch')
        require(set(arm['snapshots']['0']) == {'training', 'development'}, 'split coverage mismatch')
        for step in STEPS:
            for split in ('training', 'development'):
                snap = arm['snapshots'][step][split]
                rows = snap['per_query']
                require(len(rows) == snap['query_count'], 'query count mismatch')
                if split == 'development':
                    require(snap['query_count'] == (300 if task == 'action_delay' else 256), 'full Development coverage mismatch')
                else:
                    require(snap['query_count'] == (128 if task == 'action_delay' else 512), 'Training coverage mismatch')
                p = pooled(rows)
                for key in ('response_nre', 'endpoint_mse', 'common_mse'):
                    require(np.isclose(p[key], snap[key], rtol=1e-8, atol=1e-10), f'pooled {key} mismatch')
    require(native['snapshots']['0'] == weighted['snapshots']['0'], 'step-0 arms differ')
    for split in ('training', 'development'):
        reference = native['snapshots']['0'][split]['per_query']
        identity = [(r['query_id'], r['source_group'], r['condition_count'], r['target_response_energy']) for r in reference]
        for arm in (native, weighted):
            for step in STEPS:
                rows = arm['snapshots'][step][split]['per_query']
                require(len(rows) == len(identity), 'query identity coverage mismatch')
                require(all((r['query_id'], r['source_group'], r['condition_count']) == x[:3] and
                            np.isclose(r['target_response_energy'], x[3], rtol=0, atol=1e-10)
                            for r, x in zip(rows, identity)), 'query identity or target energy changed')
    return native, weighted


def compact(result, protocol, task, family):
    native, weighted = verify_cell(result, protocol, task, family)
    base = native['snapshots']['0']['development']['endpoint_mse']
    require(base > 0, 'zero step-0 endpoint MSE')
    snapshots = {}
    for label, arm in [('native', native), ('response_weighted', weighted)]:
        snapshots[label] = {}
        for step in STEPS:
            dev = arm['snapshots'][step]['development']
            train = arm['snapshots'][step]['training']
            snapshots[label][step] = {'full_development_response_nre': dev['response_nre'],
                                      'endpoint_mse': dev['endpoint_mse'],
                                      'endpoint_mse_over_common_step0': dev['endpoint_mse'] / base,
                                      'common_mse': dev['common_mse'],
                                      'training_response_nre': train['response_nre'],
                                      'development_queries': dev['query_count'],
                                      'training_queries': train['query_count']}
    delta = paired_delta(native['snapshots']['256']['development']['per_query'],
                         weighted['snapshots']['256']['development']['per_query'],
                         native['snapshots']['0']['development']['per_query'])
    return {'task': task, 'family': family, 'model_id': result['model_id'],
            'checkpoint_sha256': result['model_spec']['checkpoint_sha256'],
            'panel_manifest_sha256': result['panel_manifest_sha256'],
            'cache_sha256': result['cache_sha256'], 'batch_order_sha256': result['batch_order_sha256'],
            'locked_protocol_sha256': result['locked_protocol_sha256'],
            'source_model_state_hash': result['source_model_state_hash'],
            'frozen_state_hash': result['frozen_state_hash'],
            'calibration': result['calibration'],
            'first_update_norm': {label: arm['first_update_norm'] for label, arm in [('native', native), ('response_weighted', weighted)]},
            'snapshots': snapshots, 'paired_step256_response_nre': delta}


def render_table(cells):
    lines = ['| 任务 | 模型 | 全量响应 NRE：初始／原损失／响应加权 ↓ | 第 256 步完整查询误差比：原损失／加权 ↓ | 加权−原损失 NRE [95% CI] |',
             '|---|---|---:|---:|---:|']
    for cell in cells:
        n, w = cell['snapshots']['native'], cell['snapshots']['response_weighted']
        d = cell['paired_step256_response_nre']
        lines.append(f"| {NAMES[cell['task']]} | {FAMILIES[cell['family']]} | "
                     f"{n['0']['full_development_response_nre']:.4f}／{n['256']['full_development_response_nre']:.4f}／{w['256']['full_development_response_nre']:.4f} | "
                     f"{n['256']['endpoint_mse_over_common_step0']:.3f} / {w['256']['endpoint_mse_over_common_step0']:.3f} | "
                     f"{d['weighted_minus_native']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    return '\n'.join(lines) + '\n'


def outputs():
    cells, inventory = [], {}
    for task, family in CELLS:
        path = RAW / 'sources' / task / family
        rp, pp = path / 'result.json', path / 'protocol.json'
        require(rp.is_file() and pp.is_file(), f'missing collected source: {path}')
        result, protocol = json.loads(rp.read_text()), json.loads(pp.read_text())
        cells.append(compact(result, protocol, task, family))
        inventory[f'{task}/{family}'] = {'result': {'path': str(rp.relative_to(RAW)), 'sha256': sha(rp)},
                                         'protocol': {'path': str(pp.relative_to(RAW)), 'sha256': sha(pp)}}
    locked = RAW / 'sources/protocol.json'
    input_inventory = RAW / 'sources/input_inventory.json'
    require(locked.is_file() and input_inventory.is_file(), 'locked protocol or input inventory missing')
    for cell in cells:
        require(cell['locked_protocol_sha256'] == sha(locked), 'locked protocol binding mismatch')
    inventory['locked_protocol'] = {'path': str(locked.relative_to(RAW)), 'sha256': sha(locked)}
    inventory['input_inventory'] = {'path': str(input_inventory.relative_to(RAW)), 'sha256': sha(input_inventory)}
    summary = {'schema': 'contextworld.response_intervention_summary.v1',
               'scope': 'Four fixed T1 checkpoints; paired native versus response-weighted predictor updates; Training calibration and full Development evaluation; no Test',
               'metric_definition': {'response_nre': 'sum query response MSE / sum query target response energy',
                                     'endpoint_mse_ratio': 'mean Development endpoint MSE / common step-0 mean endpoint MSE for both independently initialized arms',
                                     'paired_ci95': 'paired source-group bootstrap; same sampled groups in both arms; pooled numerator and denominator recomputed, including step-0 endpoint MSE denominator'},
               'source_inventory': inventory, 'cells': cells}
    table = render_table(cells)
    context_lines = ['| 任务 | 模型 | 第 256 步原损失：Training／Development 响应 NRE | 第 256 步加权：Training／Development 响应 NRE |',
                     '|---|---|---:|---:|']
    for c in cells:
        n, w = c['snapshots']['native']['256'], c['snapshots']['response_weighted']['256']
        context_lines.append(f"| {NAMES[c['task']]} | {FAMILIES[c['family']]} | {n['training_response_nre']:.4f}／{n['full_development_response_nre']:.4f} | {w['training_response_nre']:.4f}／{w['full_development_response_nre']:.4f} |")
    context = '\n'.join(context_lines) + '\n'
    sio = io.StringIO()
    fields = ['task', 'family', 'initial_nre', 'native_256_nre', 'weighted_256_nre', 'native_256_endpoint_mse_ratio', 'weighted_256_endpoint_mse_ratio', 'weighted_minus_native_nre', 'ci95_low', 'ci95_high', 'weighted_minus_native_endpoint_mse_over_initial', 'endpoint_mse_difference_ci95_low', 'endpoint_mse_difference_ci95_high', 'native_256_training_nre', 'weighted_256_training_nre']
    writer = csv.DictWriter(sio, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for c in cells:
        n, w, d = c['snapshots']['native'], c['snapshots']['response_weighted'], c['paired_step256_response_nre']
        writer.writerow(dict(task=c['task'], family=c['family'], initial_nre=n['0']['full_development_response_nre'], native_256_nre=n['256']['full_development_response_nre'], weighted_256_nre=w['256']['full_development_response_nre'], native_256_endpoint_mse_ratio=n['256']['endpoint_mse_over_common_step0'], weighted_256_endpoint_mse_ratio=w['256']['endpoint_mse_over_common_step0'], weighted_minus_native_nre=d['weighted_minus_native'], ci95_low=d['ci95'][0], ci95_high=d['ci95'][1], weighted_minus_native_endpoint_mse_over_initial=d['weighted_minus_native_endpoint_mse_over_initial'], endpoint_mse_difference_ci95_low=d['endpoint_mse_difference_ci95'][0], endpoint_mse_difference_ci95_high=d['endpoint_mse_difference_ci95'][1], native_256_training_nre=n['256']['training_response_nre'], weighted_256_training_nre=w['256']['training_response_nre']))
    return {RAW / 'summary.json': encoded(summary), RAW / 'summary.csv': sio.getvalue().encode(), RAW / 'table.md': table.encode(), RAW / 'training_development_context.md': context.encode()}


def marked(doc, table):
    require(doc.count(BEGIN) == doc.count(END) == 1, 'RESPONSE_INTERVENTION marker missing or duplicated')
    front, rest = doc.split(BEGIN, 1)
    _, back = rest.split(END, 1)
    return front + BEGIN + '\n' + table.rstrip() + '\n' + END + back


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect()
    generated = outputs()
    doc = DOC.read_text()
    rendered = marked(doc, generated[RAW / 'table.md'].decode())
    if args.check:
        for path, data in generated.items():
            require(path.is_file() and path.read_bytes() == data, f'stale output: {path}')
        require(doc == rendered, 'study table is stale')
        print('checked response intervention summary and Study marker')
    else:
        for path, data in generated.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        DOC.write_text(rendered)
        print('built response intervention summary and Study marker')


if __name__ == '__main__':
    main()
