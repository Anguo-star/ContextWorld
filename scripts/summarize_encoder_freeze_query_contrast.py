#!/usr/bin/env python3
"""Recompute paired PLDM encoder-freeze response summaries from archived NPZ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/research/data/native_epoch_checkpoints_v1'
OUTPUT = ROOT / 'docs/research/data/encoder_freeze_query_contrast_v1'
DOC = ROOT / 'docs/ICL_Metric_Study.md'
KEYS = ('query_ids', 'source_groups', 'B', 'E', 'C', 'R')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(task, epoch, split, regime, expected):
    path = SOURCE / 'sources' / f'epoch{epoch}' / task / 'pldm' / regime / f'{split}_query_stats.npz'
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in KEYS}
    n = len(data['query_ids'])
    assert n == expected and all(len(v) == n for v in data.values()), path
    assert len(set(data['query_ids'].tolist())) == n, path
    for key in ('B', 'E', 'C', 'R'):
        assert np.isfinite(data[key]).all(), (path, key)
    assert (data['B'] > 0).all() and (data['E'] >= 0).all() and (data['C'] >= 0).all() and (data['R'] >= 0).all(), path
    assert np.allclose(data['E'], data['C'] + data['R'], rtol=1e-9, atol=1e-8), path
    return data, {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}


def build():
    with (SOURCE / 'summary.csv').open(newline='') as file:
        baseline = {(row['task'], int(row['epoch']), row['split'], row['regime']): row for row in csv.DictReader(file)}
    rows = []
    for task in ('contact_friction', 'motion_damping'):
        for epoch in (5, 10):
            for split in ('training', 'development'):
                n = 512 if split == 'training' else 256
                arrays = {}; provenance = {}
                for regime in ('joint', 'frozen'):
                    arrays[regime], provenance[regime] = load(task, epoch, split, regime, n)
                a, b = arrays['joint'], arrays['frozen']
                assert np.array_equal(a['query_ids'], b['query_ids']), (task, epoch, split, 'query IDs')
                assert np.array_equal(a['source_groups'], b['source_groups']), (task, epoch, split, 'source groups')
                groups = np.unique(a['source_groups'])
                assert len(groups) == (n if task == 'contact_friction' else n // 2), (task, epoch, split)
                response = {}
                for regime, data in arrays.items():
                    ratio = data['R'] / data['B']
                    nre = float(data['R'].sum() / data['B'].sum())
                    reference = baseline[(task, epoch, 'loader_fit' if split == 'training' else 'development', regime)]
                    assert abs(nre - float(reference['NRE'])) < 1e-10, (task, epoch, split, regime)
                    assert int(reference['pairs']) == n and int(reference['source_groups']) == len(groups)
                    response[regime] = {
                        'checkpoint_sha256': reference['checkpoint_sha256'],
                        'npz': provenance[regime],
                        'aggregate_nre': nre,
                        'query_mean_ratio': float(np.mean(ratio)),
                        'query_median_ratio': float(np.median(ratio)),
                    }
                    if split == 'development' and epoch == 10:
                        top = np.argsort(-ratio, kind='stable')[:2]
                        response[regime]['top_two_query_ratios'] = [
                            {'query_id': str(data['query_ids'][i]), 'ratio': float(ratio[i]),
                             'B': float(data['B'][i]), 'contribution_to_query_mean': float(ratio[i] / n)}
                            for i in top
                        ]
                        response[regime]['top_two_fraction_of_query_mean'] = float(ratio[top].sum() / ratio.sum())
                wins = losses = ties = 0
                # One native source ratio per arm: sum R / sum B over all its query rows.
                for group in groups:
                    mask = a['source_groups'] == group
                    x = a['R'][mask].sum() / a['B'][mask].sum()
                    y = b['R'][mask].sum() / b['B'][mask].sum()
                    wins += int(y < x)
                    losses += int(y > x)
                    ties += int(y == x)
                assert wins + losses + ties == len(groups)
                rows.append({
                    'task': task, 'epoch': epoch, 'split': split, 'queries': n,
                    'sources': len(groups), 'arms': response,
                    't3_lower_sources': wins, 't3_higher_sources': losses,
                    'tied_sources': ties, 't3_lower_source_percent': 100 * wins / len(groups),
                })
    return {'schema': 'contextworld.encoder_freeze_query_contrast.v1',
            'source_summary': {'path': str((SOURCE / 'summary.csv').relative_to(ROOT)), 'sha256': sha256(SOURCE / 'summary.csv')},
            'source_rule': 'For each source and arm, native source NRE = sum(R)/sum(B) across its query rows; compare T3 with T2, preserving Damping forward/reverse mirrors as one source.',
            'ratio_rule': 'Each query ratio is R/B in its own checkpoint target space. Aggregate NRE is sum(R)/sum(B); query mean and median are unweighted.',
            'scope': 'Existing PLDM training-source and Development endpoint caches only; no new model inference, training, or bootstrap.',
            'rows': rows}


def render_csv(data):
    out = io.StringIO(newline='')
    columns = ['task', 'epoch', 'split', 'queries', 'sources', 't2_nre', 't3_nre', 't2_query_mean', 't3_query_mean', 't2_query_median', 't3_query_median', 't3_lower_sources', 't3_higher_sources', 'tied_sources', 't3_lower_source_percent']
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    for row in data['rows']:
        a, b = row['arms']['joint'], row['arms']['frozen']
        writer.writerow({'task': row['task'], 'epoch': row['epoch'], 'split': row['split'], 'queries': row['queries'], 'sources': row['sources'], 't2_nre': a['aggregate_nre'], 't3_nre': b['aggregate_nre'], 't2_query_mean': a['query_mean_ratio'], 't3_query_mean': b['query_mean_ratio'], 't2_query_median': a['query_median_ratio'], 't3_query_median': b['query_median_ratio'], 't3_lower_sources': row['t3_lower_sources'], 't3_higher_sources': row['t3_higher_sources'], 'tied_sources': row['tied_sources'], 't3_lower_source_percent': row['t3_lower_source_percent']})
    return out.getvalue()


def render_table(data):
    lines = ['| 任务 | Epoch | 聚合 NRE T2 / T3 ↓ | 查询比值中位数 T2 / T3 ↓ | T3 来源响应比值较低 |',
             '|---|---:|---:|---:|---:|']
    labels = {'contact_friction': '接触摩擦', 'motion_damping': '运动阻尼'}
    for row in data['rows']:
        if row['split'] != 'development':
            continue
        a, b = row['arms']['joint'], row['arms']['frozen']
        lines.append(f"| {labels[row['task']]} | {row['epoch']} | {a['aggregate_nre']:.3f} / {b['aggregate_nre']:.3f} | {a['query_median_ratio']:.3f} / {b['query_median_ratio']:.3f} | {row['t3_lower_sources']}/{row['sources']} ({row['t3_lower_source_percent']:.1f}%) |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and verify archived outputs and an optional document block')
    args = parser.parse_args()
    data = build()
    outputs = {'summary.json': json.dumps(data, ensure_ascii=False, indent=2) + '\n', 'summary.csv': render_csv(data), 'table.md': render_table(data)}
    if args.check:
        for name, content in outputs.items():
            assert (OUTPUT / name).read_text() == content, name
        doc = DOC.read_text()
        start, end = '<!-- BEGIN ENCODER_FREEZE_QUERY_CONTRAST -->', '<!-- END ENCODER_FREEZE_QUERY_CONTRAST -->'
        if start in doc or end in doc:
            assert doc.count(start) == doc.count(end) == 1
            block = doc.split(start, 1)[1].split(end, 1)[0].strip()
            assert block == outputs['table.md'].strip(), 'document table block'
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            (OUTPUT / name).write_text(content)
    print('encoder freeze query contrast: OK' if args.check else f'Wrote {OUTPUT}')


if __name__ == '__main__':
    main()
