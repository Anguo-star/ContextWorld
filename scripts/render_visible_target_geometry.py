#!/usr/bin/env python3
"""Render the visible-trajectory calibration table and complete summary CSV."""
import argparse
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BEGIN VISIBLE_TARGET_GEOMETRY -->'
END = '<!-- END VISIBLE_TARGET_GEOMETRY -->'


def estimate(record):
    ci = record['source_cluster_bootstrap_ci95']
    return (f"{100 * record['equal_scene_mean']:.2f} "
            f"[{100 * ci['lower_2_5']:.2f}, {100 * ci['upper_97_5']:.2f}]")


def render(data):
    lines = [START,
             '| 表示 | 原数据反序率 | 画布内反序率 | 画布内大差异反序率 |',
             '|---|---:|---:|---:|']
    for regime, label in [('original', 'LeWM T0'), ('frozen', 'LeWM T3')]:
        metrics = data['cells'][regime]['geometry_metrics']
        values = [metrics['task_agent_xy']['old'],
                  metrics['task_agent_xy']['visible'],
                  metrics['task_agent_xy_strong_subset']['visible']]
        lines.append('| ' + label + ' | ' + ' | '.join(
            estimate(v['reversal_over_both_strict']) for v in values) + ' |')
    example = data['first_visible_task_agent_xy_strong_reversal']
    p, z = example['physical_rms'], example['native_mean_squared_distance']
    lines += ['',
              f"一个画布内实例中，相对同一参照轨迹，两条真实轨迹的推手 RMS 距离分别为 "
              f"{p['alternative_a']:.2f} 与 {p['alternative_b']:.2f} px，"
              f"对应 latent 平方距离却为 {z['alternative_a']:.2f} 与 {z['alternative_b']:.2f}。"
              '物理偏差更大的轨迹在表示空间中反而更近。', END]
    return '\n'.join(lines)


def csv_text(data):
    rows = []
    for regime, cell in data['cells'].items():
        for geometry, comparisons in cell['geometry_metrics'].items():
            for comparison, metrics in comparisons.items():
                for metric, record in metrics.items():
                    ci = record['source_cluster_bootstrap_ci95']
                    rows.append(dict(
                        task=data['task'], family='lewm', regime=regime,
                        checkpoint_sha256=cell['checkpoint_sha256'],
                        scene_count=cell['scene_count'], source_clusters=cell['source_cluster_count'],
                        geometry=geometry, comparison=comparison, metric=metric,
                        mean_percent=100*record['equal_scene_mean'],
                        ci95_low_percent=100*ci['lower_2_5'],
                        ci95_high_percent=100*ci['upper_97_5'],
                        defined_scenes=record['defined_scenes'],
                        undefined_scenes=record['undefined_scenes']
                    ))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    directory = args.root / 'docs/research/data'
    data = json.loads((directory / 'visible_target_geometry_v1.json').read_text())
    report = args.root / 'docs/ICL_Metric_Study.md'
    text = report.read_text()
    assert text.count(START) == text.count(END) == 1
    i, j = text.index(START), text.index(END) + len(END)
    rendered = text[:i] + render(data) + text[j:]
    csv_path = directory / 'visible_target_geometry_v1.csv'
    csv_bytes = csv_text(data).encode()
    if args.check:
        assert rendered == text, 'Visible geometry table is stale'
        assert csv_path.read_bytes() == csv_bytes, 'Visible geometry CSV is stale'
        print('Visible geometry table and CSV match both complete 256-scene configurations')
    else:
        report.write_text(rendered)
        csv_path.write_bytes(csv_bytes)
        print('Rendered visible geometry table and CSV')


if __name__ == '__main__':
    main()
