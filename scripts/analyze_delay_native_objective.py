"""Rebuild the small Delay native-objective diagnostic from public records."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/research/data'
BEGIN = '<!-- BEGIN DELAY_NATIVE_OBJECTIVE -->'
END = '<!-- END DELAY_NATIVE_OBJECTIVE -->'


def build():
    rows = []
    identities = None
    for family, display in [('lewm', 'LeWM'), ('pldm', 'PLDM'), ('dinowm', 'DINO-WM')]:
        path = DATA / f'delay_native_objective_v1_{family}.json'
        raw = path.read_bytes()
        d = json.loads(raw)
        assert d['validation']['state_unchanged']
        assert d['scope']['scene_count'] == 16
        scene_ids = [x['scene_id'] for x in d['scenes']]
        if identities is None:
            identities = scene_ids
        assert identities == scene_ids
        for scene in d['scenes']:
            for closure in scene['gradient']['closures'].values():
                assert closure['max_abs'] <= 1e-5
                assert closure['relative_to_component_norm_sum'] <= 1e-3
        a = d['aggregate_sufficient_statistics']
        g = d['predictor_gradient_aggregate']
        norms = g['mean_gradient_norms']
        energy = a['mean_target_response_energy_by_position'][-1]
        assert energy > 0
        rows.append(dict(
            family=family, display=display,
            query_response_error=a['mean_response_mse_by_position'][-1] / energy,
            query_response_loss_share_percent=100 * a['mean_response_mse_by_position'][-1] / sum(a['mean_native_mse_by_position']),
            response_to_native_gradient_norm_percent=100 * norms['final_response'] / norms['native_all_seven'],
            response_native_gradient_cosine=g['pairwise']['final_response__native_all_seven']['cosine'],
            source_file=path.name, source_sha256=hashlib.sha256(raw).hexdigest(),
        ))
    result = dict(schema='contextworld.delay_native_objective_summary.v1', scope='Same 16 stratified Training scenes, 11 delay conditions, T1 seed3072; fixed weights, no optimization', rows=rows)
    lines = [BEGIN,
        '| 模型 | 查询响应误差↓ | 查询响应损失占比（%） | 响应／整体梯度范数比（%） | 响应与整体梯度余弦 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        cosine = r['response_native_gradient_cosine']
        lines.append(f"| {r['display']} | {r['query_response_error']:.5f} | {r['query_response_loss_share_percent']:.3f} | {r['response_to_native_gradient_norm_percent']:.5f} | {cosine:.3f} |")
    lines.append(END)
    return result, '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    result, table = build()
    output = DATA / 'delay_native_objective_v1.json'
    doc = ROOT / 'docs/ICL_Metric_Study.md'
    current = doc.read_text()
    assert current.count(BEGIN) == current.count(END) == 1
    start, end = current.index(BEGIN), current.index(END) + len(END)
    expected = current[:start] + table + current[end:]
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.check:
        assert output.read_text() == serialized
        assert current == expected
        print('Three source records and generated diagnostic table are consistent.')
    else:
        output.write_text(serialized)
        doc.write_text(expected)
        print('Wrote summary and diagnostic table.')

if __name__ == '__main__':
    main()
