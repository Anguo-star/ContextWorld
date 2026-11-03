#!/usr/bin/env python3
"""Summarize ContextWorld Test accuracy, history benefit, and response error."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('docs/research/data/current_reference_test_v1')
SUMMARY = DATA / 'summary.json'
TASKS = ('speed', 'action_strength', 'robot_arm_mass', 'action_delay',
         'contact_friction', 'motion_damping', 'cube_gripper_carry', 'door', 'portal_exit')
FAMILIES = ('lewm', 'pldm', 'dinowm')
TASK_NAMES = {'speed': 'Speed', 'action_strength': 'Pusher displacement',
              'robot_arm_mass': 'Arm mass', 'action_delay': 'Action delay',
              'contact_friction': 'Contact friction', 'motion_damping': 'Motion damping',
              'cube_gripper_carry': 'Gripper carry rule', 'door': 'Door traversability',
              'portal_exit': 'Portal exit'}
MODEL_NAMES = {'lewm': 'LeWM', 'pldm': 'PLDM', 'dinowm': 'DINO-WM'}
LABELS = {'original': 'Original', 'scratch': 'Mixed',
          'joint': 'Original → Mixed Joint', 'frozen': 'Original → Mixed Frozen Encoder'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    assert values and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
    return {'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values)}


def fmt(item, decimals):
    value = f"{item['mean']:.{decimals}f}"
    return value + (f" ± {item['std']:.{decimals}f}" if item['std'] is not None else '')


def main():
    source = ROOT / SUMMARY
    original = json.loads(source.read_text())
    assert original['evaluation_split'] == 'test'
    assert original['expected_runs'] == original['completed_runs'] == 136
    assert original['expected_cells'] == len(original['cells']) == 86
    seen, cells = set(), []
    for cell in original['cells']:
        task, family, regime = (cell[k] for k in ('task', 'family', 'regime'))
        assert task in TASKS and family in FAMILIES and regime in LABELS
        assert cell['status'] == 'complete' and cell['expected'] == cell['completed'] == len(cell['runs'])
        runs = []
        for run in cell['runs']:
            assert run['status'] == 'completed' and run['id'] not in seen
            seen.add(run['id'])
            assert run['id'] == f"{task}/{family}/{regime}/s{run['training_seed']}"
            path = ROOT / run['result']
            assert digest(path) == run['result_sha256'], run['id']
            raw = json.loads(path.read_text())
            result = raw['result']
            model = result['model']
            assert raw['evaluation_split'] == 'test' and raw['task'] == task
            assert raw['bundle']['task'] == task
            assert model.get('training_role', model.get('training_recipe')) == regime
            assert model['training_seed'] == run['training_seed']
            assert model.get('checkpoint_sha256', model.get('adapter', {}).get('checkpoint_sha256')) == run['checkpoint_sha256']
            assert raw['model_name'] == model['name']
            if task == 'speed':
                metrics = result['tracks']['unseen_interpolation']['gate_completion']['metrics']
            elif task in ('door', 'action_delay'):
                metrics = result['gate_completion']['metrics']
            else:
                metrics = result['metrics']
            runs.append({'id': run['id'], 'seed': run['training_seed'],
                         'checkpoint_sha256': run['checkpoint_sha256'],
                         'result': run['result'], 'result_sha256': run['result_sha256'],
                         'accuracy': run['main_score'],
                         'history': 100 * metrics['correct_history_rate'],
                         'nre': metrics['latent_response']['normalized_response_error']})
        assert all(0 <= r['accuracy'] <= 100 and 0 <= r['history'] <= 100 and r['nre'] >= 0 for r in runs)
        scores = {key: summary([r[key] for r in runs]) for key in ('accuracy', 'history', 'nre')}
        for key in ('mean', 'std', 'n'):
            old, new = cell['main'][key], scores['accuracy'][key]
            assert (old is None and new is None) or (old is not None and new is not None and
                    (old == new if key == 'n' else math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-12))), (task, family, regime, key)
        cells.append({'task': task, 'family': family, 'regime': regime,
                      'n': len(runs), 'scores': scores, 'runs': runs})
    assert len(seen) == 136 and sum(c['regime'] == 'original' for c in cells) == 24
    cells.sort(key=lambda c: (TASKS.index(c['task']), FAMILIES.index(c['family']),
                              list(LABELS).index(c['regime'])))
    output = {'evaluation_split': 'test', 'source_summary': str(SUMMARY),
              'source_summary_sha256': digest(source), 'expected_runs': 136,
              'completed_runs': len(seen), 'cells': cells}
    base = ROOT / DATA
    (base / 'metrics.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    with (base / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(['task', 'family', 'regime', 'n'] +
                        [f'{metric}_{field}' for metric in ('accuracy', 'history', 'nre')
                         for field in ('mean', 'std', 'n')])
        for cell in cells:
            writer.writerow([cell[k] for k in ('task', 'family', 'regime', 'n')] +
                            [cell['scores'][metric][field] for metric in ('accuracy', 'history', 'nre')
                             for field in ('mean', 'std', 'n')])
    lines = ['# ContextWorld Test adaptation metrics', '',
             'These tables report Test accuracy, history benefit, and response error for 136 runs in 86 comparable task, model, and training configurations. Values are mean ± sample standard deviation across training seeds; n gives the number of seeds. A single seed has no standard deviation.', '',
             'Accuracy and History are percentages (higher is better). History is the proportion of cases where matching history yields lower prediction error than alternative history. Speed Accuracy and History use the same unseen interpolation gate score. NRE is normalized latent response error (lower is better).', '',
             'Only measured configurations are listed. DINO-WM continuation is labeled Original → Mixed. Multi-step prediction results are reported separately on Development.', '']
    for task in TASKS:
        lines += [f'## {TASK_NAMES[task]}', '',
                  '| Model | Training | n | Accuracy (%) ↑ | History (%) ↑ | NRE ↓ |',
                  '|---|---|---:|---:|---:|---:|']
        for cell in (c for c in cells if c['task'] == task):
            label = 'Original → Mixed' if cell['family'] == 'dinowm' and cell['regime'] in ('joint', 'frozen') else LABELS[cell['regime']]
            lines.append(f"| {MODEL_NAMES[cell['family']]} | {label} | {cell['n']} | " +
                         ' | '.join(fmt(cell['scores'][k], 3 if k == 'nre' else 2)
                                    for k in ('accuracy', 'history', 'nre')) + ' |')
        lines.append('')
    lines += ['Source: [summary.json](summary.json). Machine readable values: [metrics.json](metrics.json) and [metrics.csv](metrics.csv).', '']
    (base / 'metrics.md').write_text('\n'.join(lines))
    print(f"Validated {len(cells)} cells and {len(seen)} Test runs; wrote metrics.json, metrics.csv, metrics.md")


if __name__ == '__main__':
    main()
