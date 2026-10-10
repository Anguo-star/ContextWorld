#!/usr/bin/env python3
"""Render known-output scorer controls in the measurement research document."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN PAIRED_SCORING_CONTROLS -->'
END = '<!-- END PAIRED_SCORING_CONTROLS -->'
CASES = [
    ('oracle_correct_target', '分别输出真实目标'),
    ('ignore_history_predict_A', '两条件都输出 z_A'),
    ('ignore_history_predict_B', '两条件都输出 z_B'),
    ('shared_midpoint_strict_tie', '两条件都输出中点 m'),
    ('swapped_targets_wrong_history', '交换两条件真实目标'),
    ('tiny_correct_response_0.001', '分别输出 m + 0.001 × (z − m)'),
]


def render(data):
    if not data['validation']['scorer_vs_independent_handcheck']:
        raise ValueError('Independent scorer validation did not pass')
    if data['selection']['candidate_index'] != 1:
        raise ValueError('This table requires the nonzero candidate-1 control')
    lines = [START,
             '| 构造输出 | 未来选择率 (%) | History (%) | Joint (%) | Gain | NRE |',
             '|---|---:|---:|---:|---:|---:|']
    for key, label in CASES:
        values = []
        for identity in ['action_strength/lewm/original/s3073',
                         'action_strength/lewm/frozen/s3072']:
            result = data['model_scenarios'][identity + '::' + key]
            if (result['n_scenes'] != 256 or result['exact_zero_target_separation_count']
                    or result['official_full_cohort_status'] != 'defined'
                    or result['official_single_pair_handcheck_mismatch_count']):
                raise ValueError(f'Incomplete or invalid control: {identity}/{key}')
            score = result['official_full_cohort_summary']
            values.append([score['correct_future_rate'], score['correct_history_rate'],
                           score['joint_icl_pair_success_rate'],
                           score['latent_response']['response_gain'],
                           score['latent_response']['normalized_response_error']])
        if not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                   for a, b in zip(*values)):
            raise ValueError('Do not combine representations with different results')
        future, history, joint, gain, nre = values[0]
        lines.append(f'| {label} | {100*future:.2f} | {100*history:.2f} | '
                     f'{100*joint:.2f} | {gain:.3f} | {nre:.6f} |')
    return '\n'.join(lines + [END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    data = json.loads((ROOT / 'docs/research/data/paired_scoring_controls_v1.json').read_text())
    report = ROOT / 'docs/ICL_Metric_Study.md'
    text = report.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Missing or duplicate control table marker')
    updated = text[:text.index(START)] + render(data) + text[text.index(END)+len(END):]
    if args.check:
        if text != updated:
            raise ValueError('Paired scoring control table is stale')
        print('Checked 6 controls in 2 representations, each covering all 256 scenes')
    else:
        report.write_text(updated)
        print('Rendered paired scorer controls in the research document')


if __name__ == '__main__':
    main()
