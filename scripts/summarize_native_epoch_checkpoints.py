#!/usr/bin/env python3
"""Recompute saved-epoch diagnostics from query energies and loader membership."""
from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
from pathlib import Path

import numpy as np

from diagnose_native_policy_contrast import aggregate, source_bootstrap
from diagnose_task_history_readout import sha, write

TASKS = ('contact_friction', 'motion_damping')
REGIMES = ('joint', 'frozen')
ENERGIES = ('B', 'E', 'W', 'C', 'R', 'main_choice', 'history_wins')
BEGIN = '<!-- BEGIN NATIVE_EPOCH_CHECKPOINTS -->'
END = '<!-- END NATIVE_EPOCH_CHECKPOINTS -->'


def read_npz(path):
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def summarize(root):
    rows, provenance = [], []
    for task in TASKS:
        membership = read_npz(root / 'membership' / f'{task}.npz')
        identity = json.loads((root / 'membership' / f'{task}.json').read_text())
        for regime in REGIMES:
            epochs = []
            for epoch in (5, 10):
                base = root / f'epoch{epoch}' / task / 'pldm' / regime
                result = json.loads((base / 'result.json').read_text())
                assert (result['task'], result['family'], result['regime']) == (task, 'pldm', regime)
                assert result['no_model_update'] and result['no_test_read']
                assert result['model_state_hash_before'] == result['model_state_hash_after']
                expected_checkpoint = identity['checkpoint_sha256' if regime == 'joint' else 'frozen_checkpoint_sha256']
                assert (result['registered_checkpoint_sha256'] if epoch == 5 else result['checkpoint_sha256']) == expected_checkpoint
                assert result['split_sha256']['training'] == identity['panel_npz_sha256']
                epochs.append(result)
                for split in ('training', 'development'):
                    path = base / f'{split}_query_stats.npz'
                    assert sha(path) == result['metrics'][split]['query_stats_sha256']
                    data = read_npz(path)
                    assert len(data['query_ids']) == (512 if split == 'training' else 256)
                    np.testing.assert_allclose(data['E'], data['C'] + data['R'], rtol=1e-10, atol=1e-7)
                    assert np.all(data['B'] > 0)
                    full_stats = {key: data[key] for key in ENERGIES}
                    full = aggregate(full_stats)
                    for key in ('NRE', 'E_over_B', 'G', 'main_choice'):
                        np.testing.assert_allclose(full[key], result['metrics'][split]['summary'][key], rtol=1e-10)
                    if split == 'training':
                        np.testing.assert_array_equal(data['query_ids'], membership['pair_query_ids'])
                        mask = membership['both_fit']
                        assert mask.dtype == bool and mask.shape == (512,) and mask.any()
                    else:
                        mask = np.ones(256, dtype=bool)
                    stats = {key: data[key][mask] for key in ENERGIES}
                    value = aggregate(stats)
                    boot = source_bootstrap(stats, data['source_groups'][mask])
                    rows.append(dict(task=task, family='pldm', regime=regime, epoch=epoch,
                                     split='loader_fit' if split == 'training' else split,
                                     pairs=int(mask.sum()), source_groups=boot['source_groups'],
                                     NRE=value['NRE'], E_over_B=value['E_over_B'],
                                     C_over_B=value['energy']['C']/value['energy']['B'],
                                     G=value['G'], main_percent=100*value['main_choice'],
                                     checkpoint_sha256=result['checkpoint_sha256'],
                                     interval_95=boot['interval_95']))
            assert epochs[0]['split_sha256'] == epochs[1]['split_sha256']
            assert epochs[0]['panel_manifest_sha256'] == epochs[1]['panel_manifest_sha256']
            assert epochs[0]['registered_checkpoint_sha256'] == epochs[1]['checkpoint_sha256']
        provenance.append(dict(task=task, membership=identity,
                               membership_npz_sha256=sha(root/'membership'/f'{task}.npz')))
    return dict(schema='contextworld.native_epoch_checkpoints.summary.v1', rows=rows,
                membership=provenance, no_training=True, no_test_read=True,
                scope='Original full-training checkpoints at epochs 5 and 10; first endpoint, five physical steps after current state.',
                limitations='Loader-fit membership does not prove exact minibatch consumption. Each checkpoint has its own target space. No claim of physical accuracy or convergence from two checkpoints.')


def render(summary):
    rows = summary['rows']
    names = {'contact_friction':'接触摩擦', 'motion_damping':'运动阻尼'}
    lines = [BEGIN, '| 任务 / 训练方案 | 训练子集 NRE ↓ | Dev NRE ↓ | 训练子集 E/B ↓ | Dev E/B ↓ | Dev 选择率（%）↑ |',
             '|---|---:|---:|---:|---:|---:|']
    for task in TASKS:
        for regime in REGIMES:
            def change(split, metric, digits=3):
                selected = [next(x for x in rows if (x['task'],x['regime'],x['split'],x['epoch']) == (task,regime,split,e)) for e in (5,10)]
                return ' → '.join(f'{x[metric]:.{digits}f}' for x in selected)
            label = 'T2 联合更新' if regime == 'joint' else 'T3 冻结 Encoder'
            lines.append(f'| {names[task]} / {label} | {change("loader_fit","NRE")} | {change("development","NRE")} | {change("loader_fit","E_over_B")} | {change("development","E_over_B")} | {change("development","main_percent",2)} |')
    return '\n'.join(lines + [END, ''])


def csv_text(summary):
    stream = io.StringIO(newline='')
    fields = [key for key in summary['rows'][0] if key != 'interval_95']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
    writer.writeheader(); writer.writerows(summary['rows'])
    return stream.getvalue()


def replace_block(path, table):
    text = path.read_text()
    assert text.count(BEGIN) == text.count(END) == 1
    before, rest = text.split(BEGIN)
    _, after = rest.split(END)
    return before + table.strip() + after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch5', type=Path, default=Path('/tmp/cw-native-epoch-20261012/epoch5'))
    parser.add_argument('--epoch10', type=Path, default=Path('/tmp/cw-native-policy-contrast-20261011'))
    parser.add_argument('--membership', type=Path, default=Path('/tmp/cw-native-epoch-20261012/membership'))
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'docs/research/data/native_epoch_checkpoints_v1')
    parser.add_argument('--doc', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.output/'sources'
    if not args.check:
        for task in TASKS:
            for suffix in ('.json','.npz'):
                dest = root/'membership'/f'{task}{suffix}'
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(args.membership/f'{task}{suffix}', dest)
            for epoch, input_root in ((5,args.epoch5),(10,args.epoch10)):
                for regime in REGIMES:
                    source = input_root/task/'pldm'/regime
                    dest = root/f'epoch{epoch}'/task/'pldm'/regime
                    dest.mkdir(parents=True,exist_ok=True)
                    for file in ('result.json','training_query_stats.npz','development_query_stats.npz'):
                        shutil.copy2(source/file, dest/file)
    summary = summarize(root)
    table, csv = render(summary), csv_text(summary)
    if args.check:
        assert json.loads((args.output/'summary.json').read_text()) == summary
        assert (args.output/'summary.csv').read_text() == csv
        assert (args.output/'table.md').read_text() == table
        if args.doc:
            assert args.doc.read_text() == replace_block(args.doc,table)
        print('Verified: eight checkpoints, sixteen split summaries, membership masks and rendered table.')
    else:
        write(args.output/'summary.json', summary)
        (args.output/'summary.csv').write_text(csv)
        (args.output/'table.md').write_text(table)
        if args.doc:
            args.doc.write_text(replace_block(args.doc,table))
        print(table)


if __name__ == '__main__':
    main()
