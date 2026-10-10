#!/usr/bin/env python3
"""Render the paired prediction/decision diagnostic from published JSON.

This renderer does no inference and does not modify existing benchmark scores.
Run with --check to verify the report block and the complete 25-row CSV.
"""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN MINIMAL_MEASUREMENT_VALIDATION -->'
END = '<!-- END MINIMAL_MEASUREMENT_VALIDATION -->'
TASKS = {'action_strength': '推手移动幅度', 'motion_damping': '运动阻尼', 'robot_arm_mass': '机械臂质量'}
FAMILIES = {'lewm': 'LeWM', 'pldm': 'PLDM', 'dinowm': 'DINO-WM'}
REGIMES = {'original': 'T0', 'scratch': 'T1', 'frozen': 'T3'}
EXAMPLES = ['action_strength/pldm/frozen', 'motion_damping/lewm/scratch', 'robot_arm_mass/pldm/frozen']


def score_stat(error):
    return {'mean': 100 * (1 - error['mean']),
            'ci95': [100 * (1 - error['ci95'][1]), 100 * (1 - error['ci95'][0])]}


def scale_stat(stat, scale):
    return {'mean': scale * stat['mean'], 'ci95': [scale * v for v in stat['ci95']]}


def complete_rows(data):
    rows = []
    for r in data['model_results']:
        h = r['horizons'][-1]
        assert h['raw_steps'] == 25
        a = r['all_horizon']
        is_mass = r['task'] == 'robot_arm_mass'
        factor = 1000 if is_mass else 1
        row = {'task': r['task'], 'model_id': r['model_id'], 'checkpoint_sha256': r['checkpoint_sha256'],
               'training_scheme': REGIMES[r['regime']], 'scenes': r['matched_scene_count'],
               'source_clusters': r['source_cluster_count'], 'decision_unit': 'mm' if is_mass else 'px-equivalent'}
        values = {
            'score_points': score_stat(a['free_normalized_squared_error']),
            'history_score_gain_points': scale_stat(a['history_benefit_wrong_minus_matched'], 100),
            'full_prediction_error_ratio': a['free_normalized_squared_error'],
            'privileged_real_input_error_ratio': a['real_input_normalized_squared_error'],
            'paired_real_minus_free_error_ratio': a['paired_real_minus_free_error'],
            'decision_gap_25': scale_stat(h['physical_decision_gap_best_shared_vs_oracle'], factor),
            'physical_regret_25': scale_stat(h['decision_modes']['free']['physical_regret'], factor),
            'best_shared_advantage_25': scale_stat(h['decision_modes']['free']['best_shared_advantage'], factor),
            'true_encoded_goal_regret_25': scale_stat(h['encoded_true_regret'], factor),
        }
        for name, stat in values.items():
            row[name] = stat['mean']
            row[name + '_ci95_low'], row[name + '_ci95_high'] = stat['ci95']
        rows.append(row)
    return rows


def render_table(rows):
    by_id = {r['model_id']: r for r in rows}
    lines = [START,
             '| 任务与模型 | 方案 | 多步预测分↑ | 正确历史带来的分数增量↑（95% 区间） | 决策单位 | 决策差距 G（25 步） | 模型 regret（25 步）↓ |',
             '|---|---|---:|---:|---|---:|---:|']
    for key in EXAMPLES:
        r = by_id[key]
        task, family, _ = key.split('/')
        gain = (f"{r['history_score_gain_points']:.2f} "
                f"[{r['history_score_gain_points_ci95_low']:.2f}, {r['history_score_gain_points_ci95_high']:.2f}]")
        unit = 'mm' if r['decision_unit'] == 'mm' else 'px 等效'
        lines.append(f"| {TASKS[task]} / {FAMILIES[family]} | {r['training_scheme']} | {r['score_points']:.2f} | {gain} | {unit} | {r['decision_gap_25']:.3f} | {r['physical_regret_25']:.3f} |")
    lines.append(END)
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    data_dir = args.root / 'docs/research/data'
    data = json.loads((data_dir / 'minimal_measurement_validation_v1.json').read_text())
    rows = complete_rows(data)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    csv_text = stream.getvalue()
    report = args.root / 'docs/ICL_Metric_Study.md'
    text = report.read_text()
    assert text.count(START) == text.count(END) == 1
    a, b = text.index(START), text.index(END) + len(END)
    updated = text[:a] + render_table(rows) + text[b:]
    csv_path = data_dir / 'minimal_measurement_validation_v1.csv'
    if args.check:
        assert text == updated, 'Paired measurement report block is stale'
        assert csv_path.read_bytes() == csv_text.encode(), 'Paired measurement CSV is stale'
        print(f'Checked {len(rows)} aligned checkpoint rows and the report block')
    else:
        report.write_text(updated)
        csv_path.write_bytes(csv_text.encode())
        print(f'Rendered {len(rows)} aligned checkpoint rows and the report block')


if __name__ == '__main__':
    main()
