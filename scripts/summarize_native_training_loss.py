#!/usr/bin/env python3
"""Recompute sampled native PLDM losses and matched checkpoint ICL NRE."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'docs/research/data/native_training_loss_v1'
METRICS = ('fit__loss', 'fit__pred_loss', 'fit__std_loss', 'fit__std_t_loss',
           'fit__cov_loss', 'fit__temp_align_loss')
WEIGHTS = {'fit__pred_loss': 1., 'fit__std_loss': 18., 'fit__std_t_loss': .7,
           'fit__cov_loss': 12., 'fit__temp_align_loss': .2}
FIELDS = ('run', 'cloud_id', 'cohort', 'task', 'variant', 'metric', 'checkpoint_epoch',
          'mean', 'median', 'n', 'step_min', 'step_max', 'unmatched_metric_steps')


def identity(name):
    task = 'contact_friction' if name.startswith('contact_friction_pldm_') else 'motion_damping'
    if 'num32k_v1_warmstart3073_s3072' in name:
        return 'Encoder冻结对照', task, 'T2 联合训练', 'joint'
    if 'encoder_frozen_v1_s3072' in name:
        return 'Encoder冻结对照', task, 'T3 冻结 Encoder', 'frozen'
    if 'targetgrad_v1_warmstart3073' in name:
        return '目标端梯度对照', task, ('停止目标梯度' if 'target_stopgrad' in name else '保留目标梯度'), ('target_stopgrad' if 'target_stopgrad' in name else 'native')
    raise ValueError(name)


def parse_curve(blob, name):
    reader = csv.DictReader(io.StringIO(blob.decode('utf-8-sig')))
    if not reader.fieldnames or 'step' not in reader.fieldnames:
        raise ValueError(f'{name}: invalid header')
    field = next((f for f in reader.fieldnames if f.endswith('_step') and f != 'step'), None)
    if field is None:
        raise ValueError(f'{name}: no value column')
    result = {}
    for row in reader:
        if not row['step'] or not row[field]:
            continue
        step = int(row['step'])
        value = float(row[field])
        if step < 0 or step in result or not math.isfinite(value):
            raise ValueError(f'{name}: duplicate, negative, or nonfinite step/value at {step}')
        result[step] = value
    return result


def score_map():
    scores = {}
    for filename, cohort in (('native_epoch_checkpoints_v1/summary.csv', 'Encoder冻结对照'),
                             ('pldm_target_gradient_control_v1/summary.csv', '目标端梯度对照')):
        with (BASE / 'docs/research/data' / filename).open(newline='') as handle:
            for row in csv.DictReader(handle):
                if int(row['epoch']) not in (5, 10):
                    continue
                variant = row['regime'] if cohort == 'Encoder冻结对照' else row['variant']
                split = 'dev' if row['split'] == 'development' else 'train'
                key = (cohort, row['task'], variant, int(row['epoch']), split)
                if key in scores:
                    raise ValueError(f'duplicate ICL score: {key}')
                scores[key] = float(row['NRE'])
    return scores


def fmt(value):
    return f'{value:.3f}'


def make_outputs():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    runs = manifest['runs']
    if len(runs) != 8 or len({r['name'] for r in runs}) != 8:
        raise ValueError('expected eight unique runs')
    scores = score_map()
    rows, checks, comparisons = [], [], []
    for run in sorted(runs, key=lambda r: r['name']):
        name = run['name']
        cohort, task, label, variant = identity(name)
        if set(run['metrics']) != {'epoch', *METRICS}:
            raise ValueError(f'{name}: expected seven metric files')
        curves = {}
        for metric, meta in run['metrics'].items():
            rel = Path(meta['path'])
            if rel.is_absolute() or '..' in rel.parts or rel != Path('raw') / name / (metric + '.csv.gz'):
                raise ValueError(f'{name}: invalid metric path')
            blob = gzip.decompress((DATA / rel).read_bytes())
            if hashlib.sha256(blob).hexdigest() != meta['sha256']:
                raise ValueError(f'{name}: SHA256 mismatch: {metric}')
            curves[metric] = parse_curve(blob, name + '/' + metric)
        epoch = curves['epoch']
        for step, value in epoch.items():
            if not value.is_integer() or int(value) != step // 3483:
                raise ValueError(f'{name}: epoch mismatch at step {step}')
        for metric in METRICS:
            points = curves[metric]
            unmatched = set(points) - set(epoch)
            if unmatched:
                raise ValueError(f'{name}/{metric}: {len(unmatched)} unmatched steps')
            buckets = defaultdict(list)
            for step, value in points.items():
                buckets[int(epoch[step]) + 1].append((step, value))
            for e in range(1, 11):
                pairs = sorted(buckets[e])
                if not pairs:
                    raise ValueError(f'{name}/{metric}: no recorded samples in epoch {e}')
                vals = [v for _, v in pairs]
                rows.append(dict(zip(FIELDS, (name, run['cuid'], cohort, task, label, metric, e,
                                              statistics.mean(vals), statistics.median(vals), len(vals),
                                              pairs[0][0], pairs[-1][0], 0))))
        common = set(epoch).intersection(*(set(curves[m]) for m in METRICS))
        residuals = []
        for step in sorted(common):
            actual = curves['fit__loss'][step]
            expected = sum(WEIGHTS[m] * curves[m][step] for m in WEIGHTS)
            residuals.append(abs(actual - expected))
            if not math.isclose(actual, expected, rel_tol=.001, abs_tol=.001):
                raise ValueError(f'{name}: weighted-loss mismatch at step {step}')
        checks.append({'run': name, 'common_steps': len(common),
                       'max_abs_residual': max(residuals) if residuals else None,
                       'failed_steps': 0})
        for e in (5, 10):
            def mean(metric):
                return next(r['mean'] for r in rows if r['run'] == name and r['metric'] == metric and r['checkpoint_epoch'] == e)
            key = (cohort, task, variant, e)
            if (*key, 'train') not in scores or (*key, 'dev') not in scores:
                raise ValueError(f'missing ICL NRE: {key}')
            comparisons.append({'run': name, 'cloud_id': run['cuid'], 'cohort': cohort,
                                'task': task, 'variant': label, 'epoch': e,
                                'loss': mean('fit__loss'), 'pred_loss': mean('fit__pred_loss'),
                                'train_NRE': scores[(*key, 'train')], 'dev_NRE': scores[(*key, 'dev')]})
    if len(rows) != 8 * 6 * 10:
        raise ValueError(f'expected 480 summary rows, got {len(rows)}')
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    csv_bytes = output.getvalue().encode()
    json_bytes = (json.dumps({'rows': rows, 'formula_checks': checks, 'comparisons': comparisons,
                              'epoch_rule': 'checkpoint_epoch=floor(step/3483)+1',
                              'sampling_note': 'Means and medians of logged training batches only; exact step joins, no interpolation.',
                              'formula_tolerance': {'relative': .001, 'absolute': .001}},
                             ensure_ascii=False, indent=2) + '\n').encode()
    lines = ['# PLDM 原生训练损失与条件响应', '',
             '损失为该轮已记录训练批次的均值，非整轮全样本均值；NRE 越低越好。'
             '箭头表示第 5 → 10 轮。训练 NRE 与 Development NRE 来自既有 ICL 评测。'
             '训练前检查点没有对应训练损失。', '',
             '| 对照 | 任务 | 方案 | 总 loss ↓ | 预测 loss ↓ | 训练 NRE ↓ | Development NRE ↓ |',
             '|---|---|---|---:|---:|---:|---:|']
    by_run = defaultdict(dict)
    for c in comparisons:
        by_run[c['run']][c['epoch']] = c
    for name in sorted(by_run, key=lambda n: (0 if 'targetgrad' not in n else 1, 0 if n.startswith('contact') else 1, 1 if ('frozen' in n or 'stopgrad' in n) else 0)):
        a, b = by_run[name][5], by_run[name][10]
        arrow = lambda k: (f'{a[k]:.5f} → {b[k]:.5f}' if k == 'pred_loss' else f'{a[k]:.4f} → {b[k]:.4f}' if k == 'loss' else f'{a[k]:.3f} → {b[k]:.3f}')
        lines.append(f"| {a['cohort']} | { {'contact_friction': '接触摩擦', 'motion_damping': '运动阻尼'}[a['task']] } | {a['variant']} | {arrow('loss')} | {arrow('pred_loss')} | {arrow('train_NRE')} | {arrow('dev_NRE')} |")
    lines += ['', '来源：本目录 `raw/` 与 `manifest.json`；训练实现 `stable-worldmodel/scripts/train/pldm.py`；'
              'ICL 数据 `native_epoch_checkpoints_v1/summary.csv` 和 `pldm_target_gradient_control_v1/summary.csv`。'
              '仅在六条 loss 曲线的共同 step 上以 bf16 容差核对加权式，不作精确重构或因果解释。', '']
    return {'summary.csv': csv_bytes, 'summary.json': json_bytes,
            'table.md': '\n'.join(lines).encode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='validate sources and compare generated files without writing')
    args = parser.parse_args()
    outputs = make_outputs()
    for name, content in outputs.items():
        path = DATA / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f'out of date: {path}')
        else:
            path.write_bytes(content)
    doc = BASE / 'docs/ICL_Metric_Study.md'
    text = doc.read_text()
    begin = '<!-- BEGIN NATIVE_TRAINING_LOSS -->'
    end = '<!-- END NATIVE_TRAINING_LOSS -->'
    if begin in text and end in text:
        table = outputs['table.md'].decode().split('| 对照 |', 1)[1].split('\n\n来源：', 1)[0].strip()
        block = begin + '\n| 对照 | ' + table + '\n' + end
        start, stop = text.index(begin), text.index(end) + len(end)
        expected = text[:start] + block + text[stop:]
        if args.check and expected != text:
            raise SystemExit('out of date: native training loss documentation block')
        if not args.check:
            doc.write_text(expected)
    print('verified 8 runs, 56 source CSVs, 480 metric-epochs, and 16 matched ICL checkpoints' if args.check else 'wrote summary.csv, summary.json, table.md')


if __name__ == '__main__':
    main()
