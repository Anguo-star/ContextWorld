#!/usr/bin/env python3
"""Validate and render the public matched LeWM T2/T3 encoder contrast."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/research/data/matched_encoder_contrast_v1'
DOC = ROOT / 'docs/ICL_Metric_Study.md'
TASKS = ('action_strength', 'contact_friction', 'cube_gripper_carry', 'robot_arm_mass')
REGIMES = ('joint', 'frozen')
BEGIN = '<!-- BEGIN MATCHED_ENCODER_CONTRAST -->'
END = '<!-- END MATCHED_ENCODER_CONTRAST -->'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(data: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((data / 'summary.json').read_text())
    require(summary.get('schema') == 'contextworld.matched_encoder_contrast_summary.v1', 'wrong summary schema')
    require(set(summary['tasks']) == set(TASKS), 'unexpected summary tasks')
    freeze = json.loads((data / 'encoder_freeze_validation.json').read_text())
    require(freeze.get('schema') == 'contextworld.encoder_freeze_checkpoint_validation.v1', 'wrong encoder validation schema')
    require(set(freeze['tasks']) == set(TASKS), 'unexpected encoder validation tasks')
    for task in TASKS:
        record = freeze['tasks'][task]
        matched = summary['tasks'][task]
        require(record['init_file_sha256'] == matched['matched_init_weights_sha256'], f'{task}: encoder initialization hash mismatch')
        require(record['checkpoint_sha256']['T2'] == matched['joint']['checkpoint_sha256'], f'{task}: T2 checkpoint hash mismatch')
        require(record['checkpoint_sha256']['T3'] == matched['frozen']['checkpoint_sha256'], f'{task}: T3 checkpoint hash mismatch')
        require(record['encoder_tensor_hash_sha256']['T0'] == record['encoder_tensor_hash_sha256']['T3'], f'{task}: frozen encoder differs from T0')
        require(record['encoder_tensor_hash_sha256']['T0'] != record['encoder_tensor_hash_sha256']['T2'], f'{task}: joint encoder unchanged')
        require(record['encoder_key_count'] == 198 and record['changed_encoder_keys_vs_T0'] == {'T2': 198, 'T3': 0}, f'{task}: unexpected encoder key comparison')
    rows = []
    for task in TASKS:
        group = summary['tasks'][task]
        require(all(len(group[k]) == 64 for k in ('matched_init_weights_sha256', 'matched_training_dataset_manifest_sha256', 'matched_panel_manifest_sha256')), f'{task}: missing match hashes')
        for regime in REGIMES:
            source = json.loads((data / 'sources' / task / f'{regime}.json').read_text())
            reported = group[regime]
            require(source['schema'] == 'contextworld.matched_encoder_contrast.v1', f'{task}/{regime}: schema')
            require((source['task'], source['regime']) == (task, regime), f'{task}/{regime}: identity')
            require(source['no_test_read'] is True and source['no_model_update'] is True, f'{task}/{regime}: safety flags')
            require(source['model_state_hash_before'] == source['model_state_hash_after'], f'{task}/{regime}: model changed')
            require(source['panel_manifest_sha256'] == group['matched_panel_manifest_sha256'], f'{task}/{regime}: panel mismatch')
            require(source['checkpoint_sha256'] == reported['checkpoint_sha256'], f'{task}/{regime}: checkpoint mismatch')
            for source_key, summary_key in (('response', 'response'), ('readout_full_history', 'readout_full_history'), ('readout_current_only', 'readout_current_only'), ('target_variance', 'target_variance'), ('history_variance', 'history_variance')):
                require(source[source_key] == reported[summary_key], f'{task}/{regime}: {source_key} mismatch')
            require(source['readout_history_gain_balanced_accuracy'] == reported['history_gain_balanced_accuracy'], f'{task}/{regime}: history gain mismatch')
            fraction = source['target_variance']['conditional_variance_fraction']
            accuracy = source['readout_full_history']['balanced_accuracy']
            nre = source['response']['response_nre']
            require(all(math.isfinite(v) for v in (fraction, accuracy, nre)), f'{task}/{regime}: nonfinite measure')
            require(0 <= fraction <= 1 and 0 <= accuracy <= 1 and nre >= 0, f'{task}/{regime}: invalid measure')
            require(math.isclose(fraction, source['target_variance']['within_query_energy'] / source['target_variance']['global_centered_energy'], rel_tol=1e-10), f'{task}/{regime}: fraction definition')
            require(math.isclose(nre, source['response']['response_error'] / source['response']['target_energy'], rel_tol=1e-10), f'{task}/{regime}: NRE definition')
            rows.append({'task': task, 'training': 'T2' if regime == 'joint' else 'T3', 'target_conditional_fraction_pct': fraction * 100, 'history_ridge_balanced_accuracy_pct': accuracy * 100, 'native_response_nre': nre, 'source': f'sources/{task}/{regime}.json'})
    published = json.loads((ROOT / 'docs/research/data/icl_training_study_v2.json').read_text())
    index = {row['id']: row for row in published['rows']}
    for row in rows:
        regime = 'joint' if row['training'] == 'T2' else 'frozen'
        anchor = index[f"{row['task']}/lewm/{regime}"]
        match = summary['tasks'][row['task']]
        require(anchor['checkpoint_sha256'] == match[regime]['checkpoint_sha256'], f"{row['task']}/{regime}: published checkpoint mismatch")
        require(anchor['init_weights_sha256'] == match['matched_init_weights_sha256'], f"{row['task']}/{regime}: initialization mismatch")
        require(anchor['training_dataset_manifest_sha256'] == match['matched_training_dataset_manifest_sha256'], f"{row['task']}/{regime}: training data mismatch")
        require(anchor['training_seed'] == 3072 and anchor['training_epochs'] == 10, f"{row['task']}/{regime}: unmatched schedule")
        row['published_main_score_pct'] = anchor['scores']['main']
    with (ROOT / 'docs/research/data/icl_measurement_validation_v1.csv').open(newline='') as stream:
        multistep = list(csv.DictReader(stream))
    for row in rows:
        regime = 'joint' if row['training'] == 'T2' else 'frozen'
        matches = [item for item in multistep if (item['task'], item['family'], item['regime']) == (row['task'], 'lewm', regime)]
        require(len(matches) == 1, f"{row['task']}/{regime}: expected one multistep row")
        item = matches[0]
        require(item['training_repetitions'] == '1' and item['n_queries'] == '256', f"{row['task']}/{regime}: wrong multistep coverage")
        ratio = float(item['matched_error_ratio'])
        require(math.isfinite(ratio) and ratio >= 0, f"{row['task']}/{regime}: invalid multistep E/B")
        require(math.isclose(float(item['score']), 100 * (1 - ratio), rel_tol=1e-9, abs_tol=1e-9), f"{row['task']}/{regime}: E/B score identity")
        row['multistep_complete_e_over_b'] = ratio
    return summary, rows


def csv_text(rows: list[dict]) -> str:
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: f'{v:.10g}' if isinstance(v, float) else v for k, v in row.items()})
    return out.getvalue()


def table(rows: list[dict]) -> str:
    labels = {'action_strength': '推手移动幅度', 'contact_friction': '接触摩擦', 'cube_gripper_carry': 'Cube 夹爪携带', 'robot_arm_mass': '机械臂质量'}
    lines = [
        '| 任务 | 目标条件差异占比（%） | 历史线性读出（%） | 原生响应 NRE ↓ | 原协议主分（%） ↑ | 多步完整 E/B ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for task in TASKS:
        joint, frozen = (r for r in rows if r['task'] == task)
        lines.append(f"| {labels[task]} | {joint['target_conditional_fraction_pct']:.3f} → {frozen['target_conditional_fraction_pct']:.3f} | {joint['history_ridge_balanced_accuracy_pct']:.2f} → {frozen['history_ridge_balanced_accuracy_pct']:.2f} | {joint['native_response_nre']:.3f} → {frozen['native_response_nre']:.3f} | {joint['published_main_score_pct']:.2f} → {frozen['published_main_score_pct']:.2f} | {joint['multistep_complete_e_over_b']:.3f} → {frozen['multistep_complete_e_over_b']:.3f} |")
    return '\n'.join(lines)


def render_doc(rows: list[dict], *, check: bool) -> None:
    content = DOC.read_text()
    require(content.count(BEGIN) == 1 and content.count(END) == 1, 'document needs exactly one BEGIN/END marker pair')
    start = content.index(BEGIN) + len(BEGIN)
    stop = content.index(END)
    require(stop >= start, 'reversed markers')
    updated = content[:start] + '\n' + table(rows) + '\n' + content[stop:]
    if check:
        require(content == updated, 'document table is stale')
    elif content != updated:
        DOC.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA, help='source bundle (default: public data directory)')
    parser.add_argument('--check', action='store_true', help='verify committed CSV and marked document table')
    parser.add_argument('--render-doc', action='store_true', help='write marked document table')
    args = parser.parse_args()
    require(not (args.check and args.render_doc), '--check and --render-doc conflict')
    _, rows = load(args.data_dir)
    csv_path = args.data_dir / 'summary.csv'
    expected = csv_text(rows)
    if args.check:
        require(csv_path.read_text() == expected, 'summary.csv is stale')
        render_doc(rows, check=True)
    else:
        csv_path.write_text(expected)
        if args.render_doc:
            render_doc(rows, check=False)
    print(table(rows))


if __name__ == '__main__':
    main()
