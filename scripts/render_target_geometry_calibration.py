#!/usr/bin/env python3
"""Render the real-trajectory geometry diagnostic; does not compute ICL scores."""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN TARGET_GEOMETRY_CALIBRATION -->'
END = '<!-- END TARGET_GEOMETRY_CALIBRATION -->'
FAMILIES = {'lewm': 'LeWM', 'pldm': 'PLDM'}
REGIMES = {'original': 'T0', 'scratch': 'T1', 'joint': 'T2', 'frozen': 'T3'}
TASKS = ['action_strength', 'motion_damping', 'robot_arm_mass']


def rate(scheme, name='reversal_over_both_strict'):
    return 100 * scheme['macro_by_scene'][name]['equal_scene_mean']


def render(data, aliases):
    cells = {(c['task'], c['family'], c['regime']): c for c in data['cells']}
    lines = [START,
      '| 模型 | 方案 | 推手移动幅度 | 运动阻尼 | 机械臂质量 |',
      '|---|---|---:|---:|---:|']
    for family, family_label in FAMILIES.items():
        for regime, scheme_label in REGIMES.items():
            values = []
            for task in TASKS:
                c = cells[(task, family, regime)]['comparison_schemes']
                full = c['all_other_trajectories']
                if task == 'robot_arm_mass':
                    values.append(f'{rate(full):.2f}')
                else:
                    values.append(f"{rate(full):.2f} / {rate(c['task_relevant_geometry_all_other']):.2f}")
            lines.append(f'| {family_label} | {scheme_label} | ' + ' | '.join(values) + ' |')
    fulls = [c['comparison_schemes']['all_other_trajectories'] for c in data['cells']]
    strong = [rate(s['strong_subset']) for s in fulls]
    ties = [rate(s, 'latent_tie_fraction_over_physical_strict') for s in fulls]
    lines += ['', f'在物理 RMS 误差至少相差两倍、且较大误差至少为一个任务单位的子集中，全几何反序率仍为 {min(strong):.2f}%–{max(strong):.2f}%。物理可排序时的 latent 平局率为 {min(ties):.2f}%–{max(ties):.2f}%；平局没有记成正确排序。', '',
      '| 任务 | 存在整段同图、几何 RMS 相差至少一个单位的场景 | 同图轨迹的最大几何 RMS 差异 |',
      '|---|---:|---:|']
    names = dict(zip(TASKS, ['推手移动幅度', '运动阻尼', '机械臂质量']))
    for row in aliases['tasks']:
        unit = 'mm' if row['task']=='robot_arm_mass' else 'px 等效'
        count = row['scenes_with_full_sequence_alias_rms_at_least_one_unit']
        maximum = row['full_trajectory']['max_rms_among_pixel_identical_pairs']
        lines.append(f"| {names[row['task']]} | {count} / {row['scene_count_expected_and_processed']} | {maximum:.2f} {unit} |")
    lines.append(END)
    return '\n'.join(lines)


def csv_text(data):
    rows=[]
    for c in data['cells']:
        for scheme_name, scheme in c['comparison_schemes'].items():
            for subset, stats in [('all',scheme), ('rms_factor_two_and_one_unit',scheme['strong_subset'])]:
                r={k:c[k] for k in ['task','family','regime','model_id','checkpoint_sha256','scene_count','source_cluster_count']}
                r.update(comparison_scheme=scheme_name, subset=subset)
                for name, record in stats['macro_by_scene'].items():
                    r[name+'_percent']=100*record['equal_scene_mean'] if record['equal_scene_mean'] is not None else None
                    ci=record['source_cluster_bootstrap_ci95']
                    r[name+'_ci95_low']=None if ci['lower_2_5'] is None else 100*ci['lower_2_5']
                    r[name+'_ci95_high']=None if ci['upper_97_5'] is None else 100*ci['upper_97_5']
                    r[name+'_eligible_scenes']=record['eligible_scenes']
                r.update(stats['aggregate_pair_counts']);rows.append(r)
    out=io.StringIO(newline='');w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return out.getvalue()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--check',action='store_true');a=p.parse_args()
    d=a.root/'docs/research/data'
    data=json.loads((d/'target_geometry_calibration_v1.json').read_text())
    aliases=json.loads((d/'target_geometry_aliases_v1.json').read_text())
    report=a.root/'docs/ICL_Metric_Study.md';text=report.read_text()
    assert text.count(START)==text.count(END)==1
    i,j=text.index(START),text.index(END)+len(END)
    new=text[:i]+render(data,aliases)+text[j:]
    target=d/'target_geometry_calibration_v1.csv';csv_data=csv_text(data).encode()
    if a.check:
        assert new==text,'Geometry report block is stale'
        assert target.read_bytes()==csv_data,'Geometry CSV is stale'
        print('Geometry report and complete CSV match 24 representation configurations')
    else:
        report.write_text(new);target.write_bytes(csv_data)
        print('Rendered geometry report and complete CSV')


if __name__=='__main__':main()
