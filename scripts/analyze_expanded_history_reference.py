#!/usr/bin/env python3
"""Validate current history-reference receipts and generate the scoped report block."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

TASKS = {'friction': '接触摩擦', 'damping': '运动阻尼', 'cube': 'Cube 夹爪携带'}
BEGIN = '<!-- BEGIN EXPANDED_HISTORY_PREREQUISITES -->'
END = '<!-- END EXPANDED_HISTORY_PREREQUISITES -->'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(left, right, context):
    require(np.isclose(left, right, atol=1e-12, rtol=0), context)

def bootstrap(predictions):
    """Recompute the frozen source-group bootstrap from condition predictions."""
    groups = defaultdict(list)
    for row in predictions:
        require(row['correct'] == (row['mode'] == row['prediction']), 'prediction correctness mismatch')
        groups[row['source_group']].append(row['correct'])
    keys = sorted(groups)
    total = np.array([len(groups[key]) for key in keys])
    correct = np.array([sum(groups[key]) for key in keys])
    draw = np.random.default_rng(20261009).integers(len(keys), size=(4000, len(keys)))
    values = correct[draw].sum(1) / total[draw].sum(1)
    return {'resamples': 4000, 'seed': 20261009, 'groups': len(keys), 'lower': float(np.quantile(values, .025)), 'upper': float(np.quantile(values, .975))}

def validate_reference(task, report, coverage):
    """Check denominator integrity, split identity, controls and derived statistics."""
    require(report['task'] == task, 'task mismatch')
    require(report['input_contract']['Test_read'] is False, 'Test must remain unread')
    require(set(report['manifest_sha256_before']) == {'manifest.jsonl', 'manifest.sha256', 'task_registry.json'}, 'release identity file set mismatch')
    require(report['manifest_sha256_before'] == report['manifest_sha256_after'], 'release manifest changed')
    for split in ('training', 'development'):
        receipt = report[split]
        require(receipt['versions_before'] == receipt['versions_after'], 'Lance versions changed')
        expected_pairs = coverage['pair_count'] if split == 'training' else 256
        expected_rows = coverage['row_count'] if split == 'training' else 512 * coverage['rows_per_episode']
        require(receipt['metadata_pairs'] == expected_pairs, f'{task}/{split} full pair denominator mismatch')
        require(receipt['metadata_rows'] == expected_rows, f'{task}/{split} full row denominator mismatch')
        require(receipt['path'] == coverage[split + '_member'], 'supervision member identity mismatch')
        require(receipt['metadata_conditions'] == 2 * receipt['metadata_pairs'], 'full metadata conditions mismatch')
        require(receipt['selected_conditions'] == 2 * receipt['selected_pairs'], 'selection pair completeness mismatch')
    require(report['release'] == coverage['release'], 'supervision release mismatch')
    require(coverage['eligible_synthetic_window_count'] == report['training']['metadata_conditions'], 'eligible supervision window denominator mismatch')
    registry = [row for row in coverage['release_evidence'] if row['path'] == 'task_registry.json']
    require(len(registry) == 1 and registry[0]['sha256'] == report['manifest_sha256_before']['task_registry.json'], 'supervision registry identity mismatch')
    summary = {'task': task, 'release': report['release'], 'training_package_pairs': report['training']['metadata_pairs'], 'fit_pairs': report['training']['selected_pairs'], 'fit_source_groups': report['training']['selected_group_count'], 'development_pairs': report['development']['selected_pairs'], 'development_source_groups': report['development']['selected_group_count'], 'input_sha256': {split: report[split]['selected_history_feature_and_label_sha256'] for split in ('training', 'development')}, 'manifest_sha256': report['manifest_sha256_before']}
    summary['recipe'] = report['recipe']
    summary['sampling'] = report['sampling']
    if task == 'cube':
        summary['full_train_development_metadata_overlap'] = report['full_train_development_metadata_overlap']
    failed = any(report[split]['mask_or_extraction_failure_count'] for split in ('training', 'development'))
    for split in ('training', 'development'):
        require(report[split]['mask_or_extraction_failure_count'] == len(report[split]['mask_or_extraction_failures']), 'extraction failure count mismatch')
    if failed:
        require(report['status'] == 'failed_extraction_no_rows_silently_dropped', 'extraction failure status mismatch')
        summary.update(status='incomplete_extraction', interpretation='提取未完成；不能据此判断历史无辨识性。')
        return summary
    require(report['status'] == 'completed_empirical_reference_not_mathematical_identifiability', 'reference run incomplete')
    predictions = report['development_predictions']
    dev = report['development']
    require(dev['selected_pairs'] == dev['metadata_pairs'] == 256, 'Development must use all 256 pairs')
    require(len(predictions) == dev['selected_conditions'] == 512, 'Development must contain 512 condition predictions')
    pairs = defaultdict(list)
    for row in predictions:
        pairs[row['pair_id']].append(row)
    require(len(pairs) == 256, 'Development pair prediction denominator mismatch')
    require(all(len(rows) == 2 and len({row['mode'] for row in rows}) == 2 and len({row['source_group'] for row in rows}) == 1 for rows in pairs.values()), 'incomplete or misgrouped Development predictions')
    counts = Counter(row['mode'] for row in predictions)
    require(len(counts) == 2 and set(counts.values()) == {256}, 'Development condition denominators mismatch')
    accuracy = sum(row['correct'] for row in predictions) / 512
    scores = report['development_scores']
    close(accuracy, scores['accuracy'], 'Development accuracy mismatch')
    per = {mode: sum(row['correct'] for row in predictions if row['mode'] == mode) / count for mode, count in counts.items()}
    for mode in per:
        close(per[mode], scores['condition_accuracy'][mode], 'condition score mismatch')
    close(min(per.values()), scores['worst_condition_accuracy'], 'worst condition score mismatch')
    interval = bootstrap(predictions)
    stored = report['development_group_bootstrap_95ci']
    for key, value in interval.items():
        close(value, stored[key], f'bootstrap {key} mismatch')
    require(interval['groups'] == dev['selected_group_count'], 'bootstrap group denominator mismatch')
    for split in ('training', 'development'):
        n = report[split]['selected_pairs']
        for name, control in report['controls'][split].items():
            require(control['pairs'] == n and 0 <= control['count'] <= n, 'control denominator mismatch')
            close(control['fraction'], control['count'] / n, 'control fraction mismatch')
            if name in ('current_only_equal', 'action_only_equal', 'current_plus_action_equal'):
                close(control['deterministic_classifier_theoretical_accuracy_upper_bound_not_trained_score'], 1 - .5 * control['count'] / n, 'deterministic theoretical ceiling mismatch')
    ceiling = report['controls']['development']['current_plus_action_equal']['deterministic_classifier_theoretical_accuracy_upper_bound_not_trained_score']
    summary.update(status='completed_empirical_reference', development_accuracy=accuracy, development_accuracy_95ci=interval, worst_condition_accuracy=min(per.values()), current_plus_action_theoretical_upper_bound=ceiling, condition_denominators=dict(counts))
    return summary

def render(rows):
    """Render reference-classification evidence separately from world-model scores."""
    lines = [BEGIN, '识别率衡量参考分类器区分隐藏规则的表现，不是世界模型预测成绩。方括号为来源组 bootstrap 的 95% 区间；“当前图像＋动作”列是根据配对输入相同计算的理论上限，不是另一个训练分数。', '', '| 任务 | 训练包配对数 | 拟合配对数 / 来源组 | Development 配对数 / 来源组 | 规则识别率 (%) ↑ [95% 区间] | 最弱条件 (%) ↑ | 当前图像＋动作上限 (%) |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        score = worst = ceiling = '未完成'
        if row['status'] == 'completed_empirical_reference':
            ci = row['development_accuracy_95ci']
            score = f"{100 * row['development_accuracy']:.2f} [{100 * ci['lower']:.2f}, {100 * ci['upper']:.2f}]"
            worst = f"{100 * row['worst_condition_accuracy']:.2f}"
            ceiling = f"{100 * row['current_plus_action_theoretical_upper_bound']:.2f}"
        lines.append(f"| {TASKS[row['task']]} | {row['training_package_pairs']:,} | {row['fit_pairs']:,}/{row['fit_source_groups']:,} | {row['development_pairs']:,}/{row['development_source_groups']:,} | {score} | {worst} | {ceiling} |")
    return '\n'.join(lines + [END])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    repo = args.repo.resolve()
    directory = repo / 'docs/research/data'
    coverage = json.loads((directory / 'expanded_supervision_coverage_v1.json').read_text())
    rows = []
    for task in TASKS:
        source = directory / f'expanded_history_reference_v1_{task}.json'
        report = json.loads(source.read_text())
        names = {'friction': 'contact_friction', 'damping': 'motion_damping', 'cube': 'cube_gripper_carry'}
        entries = [row for row in coverage['tasks'] if row['task'] == names[task]]
        require(len(entries) == 1, 'supervision task missing or duplicated')
        row = validate_reference(task, report, entries[0])
        row['source'] = str(source.relative_to(repo))
        row['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        rows.append(row)
    output = {'version': 'expanded_history_prerequisites_v1', 'scope': 'current Training-fit Development reference classification; not world-model results', 'coverage_source': 'docs/research/data/expanded_supervision_coverage_v1.json', 'coverage_source_sha256': hashlib.sha256((directory / 'expanded_supervision_coverage_v1.json').read_bytes()).hexdigest(), 'tasks': rows}
    expected = json.dumps(output, ensure_ascii=False, indent=2) + '\n'
    destination = directory / 'expanded_history_prerequisites_v1.json'
    document = repo / 'docs/ICL_Metric_Study.md'
    text = document.read_text()
    require(text.count(BEGIN) == text.count(END) == 1, 'report requires exactly one existing generated marker block')
    first = text.index(BEGIN)
    last = text.index(END, first) + len(END)
    updated = text[:first] + render(rows) + text[last:]
    if args.check:
        require(destination.exists() and destination.read_text() == expected, 'generated prerequisite JSON is stale')
        require(updated == text, 'generated prerequisite report block is stale')
    else:
        destination.write_text(expected)
        document.write_text(updated)

if __name__ == '__main__':
    main()
