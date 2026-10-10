#!/usr/bin/env python3
"""Render the same-output metric comparison in the research report."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- BEGIN SAME_OUTPUT_METRIC_COMPARISON -->'
END = '<!-- END SAME_OUTPUT_METRIC_COMPARISON -->'
TASKS = {'action_strength': '推手移动幅度', 'contact_friction': '接触摩擦',
         'motion_damping': '运动阻尼'}
FAMILIES = {'lewm': 'LeWM', 'dinowm': 'DINO-WM'}
SCHEMES = {'original': 'T0', 'scratch': 'T1', 'frozen': 'T3'}


def estimate(row, name, scale=1, digits=2):
    point = row['point_estimates']['canonical_all5'][name] * scale
    low, high = (x * scale for x in row['source_group_bootstrap_95ci']['canonical_all5'][name])
    return f'{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]'


def render(data):
    rows = data['models']
    if len(rows) != 8 or len({r['model_id'] for r in rows}) != 8:
        raise ValueError('Expected eight distinct checkpoint groups')
    if not all(c['raw_S_all5_matches_pilot'] and c['raw_G_all5_matches_pilot']
               for c in data['pilot_raw_S_G_checks']):
        raise ValueError('Existing pilot scores do not match')
    for r in rows:
        if r['n_queries'] != 256:
            raise ValueError('Incomplete source coverage')
        for key in ('S', 'G'):
            if not math.isclose(r['point_estimates']['raw_all5'][key],
                                r['point_estimates']['canonical_all5'][key],
                                rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError('Tie correction changed an error metric materially')
    lines = [BEGIN,
             '| 任务 | 模型 | 方案 | 选择率 R (%) ↑ | 响应 NRE ↓ | 共同偏差比 ↓ | 完整误差比 ↓ | 历史收益 G ↑ |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in sorted(rows, key=lambda r: (list(TASKS).index(r['task']),
                                         list(FAMILIES).index(r['family']),
                                         list(SCHEMES).index(r['regime']))):
        point = r['point_estimates']['canonical_all5']
        parts = r['response_decomposition']
        nre, bias, full = (parts[key] for key in
                           ('nre', 'common_bias_ratio', 'complete_error_ratio'))
        if not all(math.isfinite(value) and value >= -1e-8 for value in (nre, bias, full)):
            raise ValueError('Invalid native error decomposition')
        if not math.isclose(full, nre + bias, rel_tol=2e-6, abs_tol=1e-6):
            raise ValueError('Complete error does not equal response plus common bias')
        if not math.isclose(full, 1 - point['S']/100, rel_tol=2e-6, abs_tol=1e-6):
            raise ValueError('Decomposition does not match the existing complete score')
        lines.append(f"| {TASKS[r['task']]} | {FAMILIES[r['family']]} | {SCHEMES[r['regime']]} | "
                     f"{100*point['R']:.2f} | {nre:.3f} | {bias:.3f} | {full:.3f} | {point['G']:.3f} |")
    lines += ['', '表中为点估计；各项来源组 95% 区间见结果文件。完整误差比等于响应 NRE 加共同偏差比，三者不是独立证据。', '', '**同一编码器下的对照。** DINO-WM 从 T0 到 T1：', '']
    for task in TASKS:
        baseline = next(r for r in rows if r['task'] == task and r['family'] == 'dinowm' and r['regime'] == 'original')
        trained = next(r for r in rows if r['task'] == task and r['family'] == 'dinowm' and r['regime'] == 'scratch')
        before = baseline['point_estimates']['canonical_all5']
        after = trained['point_estimates']['canonical_all5']
        error0, error1 = 1 - before['S']/100, 1 - after['S']/100
        training_improvement = 100 * (1-error1/error0)
        history_improvement = 100 * after['G']/(error1 + after['G'])
        response0 = baseline['response_decomposition']['nre']
        response1 = trained['response_decomposition']['nre']
        lines.append(f'- {TASKS[task]}：完整误差降低 **{training_improvement:.2f}%**，'
                     f'响应 NRE 从 **{response0:.3f} → {response1:.3f}**；'
                     f'在 T1 内，正确历史比错误历史降低误差 **{history_improvement:.2f}%**。')
    lines += ['', '这两个百分数采用不同参照：前者比较训练方案，后者比较同一检查点的历史条件。'
              'Damping 的完整误差降低，但响应 NRE 上升；改善来自共同预测偏差的下降，条件响应没有同步改善。'
              '收益为正也不表示预测已达到应用所需精度。', END]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'docs/research/data/same_output_metric_comparison_v2.json'
    report = ROOT / 'docs/ICL_Metric_Study.md'
    text = report.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError('Missing or duplicate comparison marker')
    updated = text[:text.index(BEGIN)] + render(json.loads(source.read_text())) + text[text.index(END)+len(END):]
    if args.check:
        if updated != text:
            raise ValueError('Comparison table or interpretation is stale')
        print('Checked eight complete checkpoint groups and three same-encoder contrasts')
    else:
        report.write_text(updated)
        print('Rendered same-output metric comparison')


if __name__ == '__main__':
    main()
