#!/usr/bin/env python3
"""Validate six native-policy caches and render a compact, checkpoint-local summary."""
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
REGIMES = ('original', 'joint', 'frozen')


def close(actual, expected, name, atol=1e-9):
    if not np.allclose(actual, expected, rtol=1e-9, atol=atol):
        raise AssertionError(f'{name}: {actual!r} != {expected!r}')


def read_unit(root, task, regime):
    base = root / task / 'pldm' / regime
    result = json.loads((base / 'result.json').read_text())
    assert result['task'] == task and result['family'] == 'pldm' and result['regime'] == regime
    assert result['model_state_hash_before'] == result['model_state_hash_after']
    assert result['no_model_update'] and result['no_test_read']
    for split, count in (('training', 512), ('development', 256)):
        path = base / f'{split}_query_stats.npz'
        expected = result['metrics'][split]
        assert sha(path) == expected['query_stats_sha256']
        with np.load(path, allow_pickle=False) as raw:
            cache = {k: raw[k] for k in raw.files}
        assert len(cache['query_ids']) == count and len(set(cache['query_ids'].tolist())) == count
        assert cache['labels'].shape == (count, 2)
        assert np.all(cache['labels'][:, 0] != cache['labels'][:, 1])
        stats = {k: cache[k] for k in ('B', 'E', 'W', 'C', 'R', 'main_choice', 'history_wins')}
        close(stats['E'], stats['C'] + stats['R'], f'{task}/{regime}/{split} decomposition', 1e-7)
        assert np.all(stats['B'] > 0)
        observed = aggregate(stats)
        for key in ('E_over_B', 'NRE', 'G', 'main_choice', 'history_wins'):
            close(observed[key], expected['summary'][key], f'{task}/{regime}/{split}/{key}')
        for key in stats:
            close(observed['energy'][key], expected['summary']['energy'][key], f'{task}/{regime}/{split}/{key} sum')
        boot = source_bootstrap(stats, cache['source_groups'])
        assert boot['source_groups'] == expected['bootstrap']['source_groups']
        assert boot['seed'] == expected['bootstrap']['seed'] and boot['resamples'] == expected['bootstrap']['resamples']
        for key in ('E_over_B', 'NRE', 'G'):
            close(boot['interval_95'][key], expected['bootstrap']['interval_95'][key], f'{task}/{regime}/{split}/{key} CI')
        result.setdefault('_cache', {})[split] = cache
    close(result['readout']['current']['ridge']['balanced_accuracy'], .5, 'current Ridge balanced accuracy')
    close(result['readout']['current']['extra_trees']['balanced_accuracy'], .5, 'current ExtraTrees balanced accuracy')
    return result


def benchmark_reference(units):
    path = Path(__file__).resolve().parents[1] / 'docs/research/data/icl_training_study_v2.json'
    study = json.loads(path.read_text())['rows']
    checks = []
    for item in units:
        matches = [x for x in study if x.get('task') == item['task'] and
                   x.get('model') == 'pldm' and x.get('regime') == item['regime'] and
                   x.get('checkpoint_sha256') == item['checkpoint_sha256']]
        assert len(matches) == 1, item['model_id']
        match = matches[0]
        measured = 100 * item['metrics']['development']['summary']['main_choice']
        nre = item['metrics']['development']['summary']['NRE']
        main_delta = measured-match['scores']['main']
        nre_delta = nre-match['scores']['nre']
        assert abs(main_delta) <= 1.0, (item['model_id'], main_delta)
        assert abs(nre_delta) < .01, (item['model_id'], nre_delta)
        checks.append(dict(model_id=item['model_id'], benchmark_main_percent=match['scores']['main'],
                           native_main_percent=measured, main_difference_pp=main_delta,
                           benchmark_nre=match['scores']['nre'], native_nre=nre,
                           nre_difference=nre_delta))
    assert len(checks) == 6
    return checks


def render(rows):
    lines = ['<!-- BEGIN NATIVE_POLICY_CONTRAST -->',
             '| 任务 / 方案 | Dev 历史 Ridge / ExtraTrees | 响应 NRE Training 来源 / Dev ↓ | 完整 E/B Training 来源 / Dev ↓ | Dev 共同误差 C/B ↓ | Dev 历史收益 G ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    names = {'contact_friction': '接触摩擦', 'motion_damping': '运动阻尼'}
    for row in rows:
        lines.append(f"| {names[row['task']]} / {row['regime_label']} | {row['history_ridge_percent']:.1f}% / {row['history_extra_trees_percent']:.1f}% | {row['train_nre']:.3f} / {row['dev_nre']:.3f} | {row['train_e_over_b']:.3f} / {row['dev_e_over_b']:.3f} | {row['dev_c_over_b']:.3f} | {row['dev_g']:.3f} |")
    lines += ['<!-- END NATIVE_POLICY_CONTRAST -->',
              'Training 的 512 对是固定来源查询，不代表每条都作为训练样本直接出现。Ridge 与 ExtraTrees 仅从视觉历史读取条件，不输入历史动作。E=C+R，NRE=R/B，历史收益 G=(W−E)/B；所有比例均在各检查点自己的目标空间内计算，不作跨空间物理精度排名。来源组 bootstrap 区间保存在 summary.json；不同目标空间之间不报告差值区间。']
    return '\n'.join(lines) + '\n'


def build(args):
    units = [read_unit(args.input_dir, task, regime) for task in TASKS for regime in REGIMES]
    for task in TASKS:
        related = [x for x in units if x['task'] == task]
        for candidate in related[1:]:
            assert candidate['panel_manifest_sha256'] == related[0]['panel_manifest_sha256']
            assert candidate['split_sha256'] == related[0]['split_sha256']
            for split in ('training', 'development'):
                left, right = related[0]['_cache'][split], candidate['_cache'][split]
                assert np.array_equal(left['query_ids'], right['query_ids'])
                assert np.array_equal(left['source_groups'], right['source_groups'])
        train_sources = set(related[0]['_cache']['training']['source_groups'].tolist())
        dev_sources = set(related[0]['_cache']['development']['source_groups'].tolist())
        # Source labels may be split-specific; record the observed intersection.
        related[0]['_source_overlap_count'] = len(train_sources & dev_sources)
    rows = []; sources = []
    for item in units:
        train = item['metrics']['training']['summary']; dev = item['metrics']['development']['summary']
        row = dict(task=item['task'], family='pldm', regime=item['regime'],
                   regime_label={'original':'T0','joint':'T2','frozen':'T3'}[item['regime']],
                   checkpoint_sha256=item['checkpoint_sha256'],
                   history_ridge_percent=100*item['readout']['history']['ridge']['balanced_accuracy'],
                   history_extra_trees_percent=100*item['readout']['history']['extra_trees']['balanced_accuracy'],
                   train_nre=train['NRE'], dev_nre=dev['NRE'],
                   train_e_over_b=train['E_over_B'], dev_e_over_b=dev['E_over_B'],
                   dev_c_over_b=dev['energy']['C']/dev['energy']['B'], dev_g=dev['G'],
                   dev_main_choice_percent=100*dev['main_choice'])
        rows.append(row)
        sources.append(dict(model_id=item['model_id'], checkpoint_sha256=item['checkpoint_sha256'],
                            panel_manifest_sha256=item['panel_manifest_sha256'], split_sha256=item['split_sha256'],
                            metrics=item['metrics'], readout=item['readout']))
    checks = benchmark_reference(units)
    summary = dict(schema='contextworld.native_policy_contrast.summary.v1', rows=rows,
                   benchmark_main_choice_checks=checks, sources=sources,
                   train_development_source_label_overlap={task: next(x['_source_overlap_count'] for x in units if x['task']==task)
                                                           for task in TASKS},
                   limitations='Same-checkpoint target-space metrics only; no cross-space physical ranking or paired delta CI.')
    return summary, render(rows), units


def csv_text(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def document_text(path, markdown):
    content = path.read_text()
    start, end = '<!-- BEGIN NATIVE_POLICY_CONTRAST -->', '<!-- END NATIVE_POLICY_CONTRAST -->'
    assert content.count(start) == content.count(end) == 1
    before, tail = content.split(start, 1)
    _, after = tail.split(end, 1)
    return before + markdown.split(end, 1)[0] + end + after


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, default=Path('/tmp/cw-native-policy-contrast-20261011'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--render-doc', action='store_true')
    p.add_argument('--doc', type=Path, default=Path(__file__).resolve().parents[1]/'docs/ICL_Metric_Study.md', help='Replace an existing NATIVE_POLICY_CONTRAST marker block')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    source_root = args.output_dir/'sources' if args.check else args.input_dir
    args.input_dir = source_root
    summary, markdown, units = build(args)
    if args.check:
        existing = json.loads((args.output_dir/'summary.json').read_text())
        assert existing == summary
        assert (args.output_dir/'summary.csv').read_text() == csv_text(summary['rows'])
        if args.render_doc:
            assert (args.output_dir/'native_policy_contrast.md').read_text() == markdown
            assert args.doc.read_text() == document_text(args.doc, markdown)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in units:
        source = args.input_dir/item['task']/'pldm'/item['regime']
        destination = args.output_dir/'sources'/item['task']/'pldm'/item['regime']
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('result.json', 'training_query_stats.npz', 'development_query_stats.npz'):
            shutil.copy2(source/name, destination/name)
    write(args.output_dir/'summary.json', summary)
    (args.output_dir/'summary.csv').write_text(csv_text(summary['rows']))
    if args.render_doc:
        (args.output_dir/'native_policy_contrast.md').write_text(markdown)
        args.doc.write_text(document_text(args.doc, markdown))


if __name__ == '__main__':
    main()
