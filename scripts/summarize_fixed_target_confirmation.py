#!/usr/bin/env python3
"""Archive protocol-bound fixed-target Test confirmations without opening Test panels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'configs/benchmark/fixed_target_confirmation_v1.json'
DEFAULT_RESULTS = Path('/tmp/cw-fixed-target-confirmation-v1/results')
DEFAULT_OUTPUT = ROOT / 'docs/research/data/fixed_target_confirmation_v1'
DEFAULT_DOC = ROOT / 'docs/ICL_Metric_Study.md'
BEGIN = '<!-- BEGIN FIXED_TARGET_CONFIRMATION -->'
END = '<!-- END FIXED_TARGET_CONFIRMATION -->'
METRICS = {'response_nre': 'nre', 'full_error_over_B': 'eb', 'history_benefit_over_B': 'g'}
FIELDS = ('task', 'seed', 'pairs', 'panel_sha256', 'nre_t0', 'nre_after', 'eb_t0', 'eb_after', 'g_t0', 'g_after', 'result_sha256')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def finite(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f'{label}: expected finite number')
    return float(value)


def checkpoint_for(task_spec):
    for key in ('completed_checkpoint', 'completed_checkpoint_path', 'checkpoint', 'checkpoint_path'):
        if key in task_spec:
            return Path(task_spec[key])
    raise ValueError('protocol task lacks completed checkpoint path')


def expected_checkpoint_hash(task_spec):
    for key in ('completed_checkpoint_sha256', 'checkpoint_sha256'):
        if key in task_spec:
            return task_spec[key]
    raise ValueError('protocol task lacks completed checkpoint SHA256')


def validate_panel_manifest(path, task, spec, protocol):
    manifest = read_json(path)
    source = protocol['source']
    require(manifest.get('schema') == 'contextworld.fixed_target_test_panel.v1' and manifest.get('task') == task,
            f'{path}: panel identity')
    require(manifest.get('source_root') == source['root'], f'{path}: source root')
    require(manifest.get('source_manifest_sha256') == source['manifest_sha256'] and
            manifest.get('source_registry_sha256') == source['registry_sha256'], f'{path}: source hashes')
    require(manifest.get('source_lance') == str(Path(source['root']) / spec['test_payload']), f'{path}: Test payload')
    expected_files = [{'path': f['path'], 'bytes': f['bytes'], 'sha256': f['sha256']} for f in spec['test_files']]
    require(manifest.get('verified_source_files') == expected_files, f'{path}: Test file inventory')
    require(manifest.get('normalization') == spec['normalization'] and
            manifest.get('normalization_manifest') == spec['normalization_manifest'] and
            manifest.get('normalization_manifest_sha256') == spec['normalization_manifest_sha256'],
            f'{path}: normalization')
    splits = manifest.get('splits', {})
    require(set(splits) == {'test'} and splits['test'].get('queries') == 256 and splits['test'].get('rows') == 512,
            f'{path}: panel splits or size')
    return manifest, digest(path)


def validate_result(path, task, seed, spec, checkpoint, expected_hash, panel_manifest, panel_manifest_hash):
    result = read_json(path)
    require(result.get('schema') == 'contextworld.fixed_target_control.v1', f'{path}: schema')
    require(result.get('task') == task, f'{path}: task')
    require(result.get('t0_checkpoint_sha256') == spec['baseline_sha256'], f'{path}: baseline hash')
    require(result.get('stable_ref') == spec['stable_ref'], f'{path}: pinned source ref')
    require(result.get('new_checkpoint_sha256') == expected_hash == digest(checkpoint), f'{path}: checkpoint hash')
    require(result.get('visual_identity') == {'tensor_and_buffer_count': 207, 'bitwise_equal': True}, f'{path}: visual state')
    require(result.get('no_fitting') is True and result.get('no_test_read') is False, f'{path}: evaluation flags')
    require(set(result.get('splits', {})) == {'test'}, f'{path}: expected Test only')
    require(result.get('panel_manifest_sha256') == panel_manifest_hash, f'{path}: panel manifest hash')
    states = result.get('model_state_unchanged', {})
    require(set(states) == {'t0', 'new'} and all(x.get('before') == x.get('after') and x.get('before') for x in states.values()), f'{path}: model state changed')
    test = result['splits']['test']
    require(test.get('pairs') == 256 and test.get('rows') == 512 and test.get('action_blocks_equal_per_pair') is True, f'{path}: Test panel size or pairing')
    panel = test.get('panel_sha256')
    require(panel == panel_manifest['splits']['test']['sha256'], f'{path}: panel hash')
    values = {'task': task, 'seed': seed, 'pairs': 256, 'panel_sha256': panel, 'result_sha256': digest(path)}
    for field, short in METRICS.items():
        values[f'{short}_t0'] = finite(test['old'][field], f'{path}: old {field}')
        values[f'{short}_after'] = finite(test['new'][field], f'{path}: new {field}')
    paired = {}
    for source, name in (('delta_response_nre', 'nre'), ('delta_full_error_over_B', 'eb'), ('delta_history_benefit_over_B', 'g')):
        delta = test[source]
        ci = delta['ci95']
        require(len(ci) == 2 and all(math.isfinite(x) for x in ci), f'{path}: paired CI')
        paired[name] = {'delta': finite(delta['delta'], f'{path}: {source} delta'), 'ci95': ci,
                        'source_groups': delta['source_groups'], 'resamples': delta['resamples']}
    return result, values, paired


def build(protocol_path, results_root, queue_path):
    protocol = read_json(protocol_path)
    protocol_hash = digest(protocol_path)
    tasks = protocol['tasks']
    seeds = protocol['training']['seeds']
    repeat_seeds = protocol['training']['repeat_seeds']
    require(len(seeds) == len(set(seeds)) and set(repeat_seeds) <= set(seeds), 'protocol seeds')
    initial = [s for s in seeds if s not in repeat_seeds]
    require(len(initial) == 1, 'expected one completed seed per task')
    queue = read_json(queue_path) if queue_path and queue_path.exists() else None
    if queue:
        require(queue.get('protocol_sha256') == protocol_hash, 'queue protocol hash differs')
    archive = {'schema': 'contextworld.fixed_target_confirmation_summary.v1',
               'protocol': str(protocol_path), 'protocol_sha256': protocol_hash,
               'scope': 'fixed-target native one-step Test, 256 pairs per task; available seeds only',
               'queue': str(queue_path) if queue_path else None,
               'queue_status': queue.get('status') if queue else 'unavailable', 'tasks': {}}
    rows = []
    for task, spec in tasks.items():
        panel_path = Path(protocol['paths']['panels']) / task / 'manifest.json'
        panel_manifest, panel_manifest_hash = validate_panel_manifest(panel_path, task, spec, protocol)
        entries, missing = [], []
        for seed in seeds:
            if seed in repeat_seeds:
                job = (queue or {}).get('jobs', {}).get(f'{task}/s{seed}')
                path = Path(job['result']) if job and job.get('result') else None
                checkpoint = Path(job['checkpoint']) if job and job.get('checkpoint') else None
                expected = job.get('checkpoint_sha256') if job else None
                status = job.get('status') if job else 'not_started'
                if status != 'completed' or path is None or not path.is_file():
                    missing.append({'seed': seed, 'status': status, 'result': str(path) if path else None})
                    continue
                require(checkpoint is not None and checkpoint.is_file() and expected, f'{task}/s{seed}: missing checkpoint')
            else:
                path = results_root / task / f's{seed}' / 'result.json'
                if not path.is_file():
                    missing.append({'seed': seed, 'status': 'result_missing', 'result': str(path)})
                    continue
                checkpoint, expected = checkpoint_for(spec), expected_checkpoint_hash(spec)
            result, values, paired = validate_result(path, task, seed, spec, checkpoint, expected,
                                                     panel_manifest, panel_manifest_hash)
            if seed in repeat_seeds:
                require(job['checkpoint_sha256'] == result['new_checkpoint_sha256'], f'{task}/s{seed}: queue checkpoint')
            entries.append({'seed': seed, 'result': str(path), 'checkpoint': str(checkpoint),
                            'checkpoint_sha256': expected, 'panel_manifest_sha256': result['panel_manifest_sha256'],
                            'panel_sha256': values['panel_sha256'], 'metrics': {k: values[k] for k in values if k.endswith(('_t0', '_after'))},
                            'within_seed_paired_ci95': paired, 'result_sha256': values['result_sha256']})
            rows.append(values)
        require(len({e['panel_sha256'] for e in entries}) <= 1, f'{task}: different Test panels')
        require(len({e['panel_manifest_sha256'] for e in entries}) <= 1, f'{task}: different panel manifests')
        require(len({tuple(e['metrics'][f'{short}_t0'] for short in METRICS.values()) for e in entries}) <= 1,
                f'{task}: T0 metrics differ across seeds on the same panel')
        aggregate = {}
        for short in METRICS.values():
            for stage in ('t0', 'after'):
                series = [e['metrics'][f'{short}_{stage}'] for e in entries]
                aggregate[f'{short}_{stage}'] = {'mean': series[0] if stage == 't0' else mean(series),
                                                 'seed_std': stdev(series) if stage == 'after' and len(series) > 1 else None} if series else None
        archive['tasks'][task] = {'planned_seeds': seeds, 'available_seeds': [e['seed'] for e in entries],
                                  'missing': missing, 'n': len(entries), 'results': entries, 'aggregate': aggregate}
    return archive, rows


def render_csv(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def render_table(archive):
    lines = ['| 任务 | 训练种子 n | 响应 NRE ↓ T0→训练后 | 完整 E/B ↓ T0→训练后 | 历史收益 G ↑ 训练后 |',
             '|---|---:|---:|---:|---:|']
    names = {'contact_friction': '接触摩擦', 'robot_arm_mass': '机械臂质量'}
    def fmt(x):
        if x is None:
            return '—'
        s = f"{x['mean']:.3f}"
        return s + (f" ± {x['seed_std']:.3f}" if x['seed_std'] is not None else '')
    for task, item in archive['tasks'].items():
        a = item['aggregate']
        lines.append(f"| {names.get(task, task)} | {item['n']} | {fmt(a['nre_t0'])} → {fmt(a['nre_after'])} | {fmt(a['eb_t0'])} → {fmt(a['eb_after'])} | {fmt(a['g_after'])} |")
    lines.append('')
    lines.append('T0 是各任务固定的训练前检查点；n 为训练重复数，训练后 ± 为样本标准差（仅 n>1）。配对来源组 bootstrap 区间见 [JSON](research/data/fixed_target_confirmation_v1/summary.json)，描述给定检查点下的场景不确定性。')
    return '\n'.join(lines) + '\n'


def update_doc(path, content, check):
    source = path.read_text()
    require(source.count(BEGIN) == source.count(END) == 1, f'{path}: expected one generated block')
    before, rest = source.split(BEGIN, 1)
    current, after = rest.split(END, 1)
    replacement = '\n' + content.rstrip() + '\n'
    if check:
        require(current == replacement, f'{path}: generated block differs')
    else:
        path.write_text(before + BEGIN + replacement + END + after)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    p.add_argument('--results-root', type=Path, default=DEFAULT_RESULTS)
    p.add_argument('--queue', type=Path)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--doc', type=Path, default=DEFAULT_DOC)
    p.add_argument('--check', action='store_true')
    p.add_argument('--render-doc', action='store_true')
    args = p.parse_args()
    protocol = read_json(args.protocol)
    queue_path = args.queue or Path(protocol['paths']['repeat_root']) / 'queue_status.json'
    archive, rows = build(args.protocol, args.results_root, queue_path)
    summary = json.dumps(archive, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    csv_text = render_csv(rows)
    json_path, csv_path = args.output_dir / 'summary.json', args.output_dir / 'summary.csv'
    if args.check:
        require(json_path.read_text() == summary and csv_path.read_text() == csv_text, 'saved summary differs from sources')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(summary)
        csv_path.write_text(csv_text)
    if args.render_doc:
        update_doc(args.doc, render_table(archive), args.check)
    print(render_table(archive), end='')


if __name__ == '__main__':
    main()
