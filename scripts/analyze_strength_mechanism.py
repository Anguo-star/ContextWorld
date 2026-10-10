#!/usr/bin/env python3
"""Validate frozen Strength measurements and regenerate their public summary."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/research/data'
REGIMES = ('scratch', 'joint', 'frozen')
LABELS = dict(zip(REGIMES, ('T1', 'T2', 'T3')))
MARKER = 'STRENGTH_REPRESENTATION_OBJECTIVE'
SEED = 20261010


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(rows):
    result = {}
    for row in rows:
        key = (row['pair_id'], row['label'])
        require(key not in result, 'Duplicate classification identity')
        require(row['label'] in (0, 1), 'Unexpected condition label')
        require(row['correct'] == (row['label'] == row['prediction']), 'Incorrect saved correctness')
        result[key] = row
    return result


def paired_difference(left, right, *, resamples=2000, seed=SEED):
    """Right minus left BA, retaining every row of each sampled source group."""
    a, b = identities(left), identities(right)
    require(a.keys() == b.keys(), 'Paired readout identities differ')
    groups = sorted({r['source_group'] for r in a.values()})
    counts = np.zeros((len(groups), 2), dtype=np.int64)
    deltas = np.zeros_like(counts)
    lookup = {g: i for i, g in enumerate(groups)}
    for key, row in a.items():
        other = b[key]
        require(row['source_group'] == other['source_group'], 'Paired source group differs')
        i, c = lookup[row['source_group']], row['label']
        counts[i, c] += 1
        deltas[i, c] += int(other['correct']) - int(row['correct'])
    require(np.all(counts > 0), 'Each source group must retain both conditions')
    draws = np.random.default_rng(seed).integers(len(groups), size=(resamples, len(groups)))
    sampled = (deltas[draws].sum(axis=1) / counts[draws].sum(axis=1)).mean(axis=1) * 100
    mean = float((deltas.sum(axis=0) / counts.sum(axis=0)).mean() * 100)
    return {'difference_pp': mean, 'ci95_pp': np.quantile(sampled, [.025, .975]).tolist(),
            'source_groups': len(groups), 'resamples': resamples, 'seed': seed}


def validate_objective(record, panel):
    selection = record['selection']
    selected, groups = selection['selected_pairs'], selection['source_groups']
    require(selection['split'] == 'training' and len(selected) == len(set(selected)) == 16,
            'Expected 16 distinct Training pairs')
    require(len(groups) == len(set(groups)) == 16 and selection['source_group_count'] == 16,
            'Expected 16 distinct Training source groups')
    require(set(selected) <= set(panel['splits']['training']['pair_ids']), 'Gradient pair outside fit panel')
    require(record['pair_count'] == 16 and record['condition_count'] == 32, 'Gradient coverage differs')
    require(record['optimizer_steps'] == 0 and not record['public_test_accessed'], 'Unexpected update/Test access')
    require(record['state_hash_before'] == record['state_hash_after'], 'Objective model changed')
    rows = record['per_pair']
    require([r['pair_id'] for r in rows] == selected, 'Gradient selected pair order differs')
    for row, group in zip(rows, groups):
        require(row['condition_count'] == 2 and row['position_count'] == 3, 'Expected H3/K2 loss')
        source = row['source']
        require(source['pair_id'] == row['pair_id'] and source['source_groups'] == [group, group], 'Gradient source differs')
        require(source['raw_sha256'] == panel['splits']['training']['sha256'], 'Gradient raw panel differs')
        require(source['modes'] == ['low_gain', 'high_gain'], 'Gradient condition order differs')
        for key in ('source_episode_ids', 'source_step_ids', 'source_row_ids'):
            require(len(source[key]) == 2 and source[key][0] == source[key][1], 'Gradient pair provenance differs')
        losses = row['losses']
        require(abs(losses['native_all_three'] - sum(losses[n] for n in ('final_response', 'final_common', 'earlier_two'))) <= 2e-6,
                'Loss decomposition does not close')
        require(abs(row['loss_reconstruction_residual']) <= 2e-6 and row['position_decomposition_max_abs'] <= 2e-6,
                'Saved loss closure exceeds tolerance')
        pos = row['per_position']
        require(all(len(pos[n]) == 3 for n in ('native_by_position', 'response_by_position', 'common_by_position', 'target_response_energy_by_position')),
                'Missing three native target positions')
        require(np.allclose(np.asarray(pos['native_by_position']), np.asarray(pos['response_by_position']) + pos['common_by_position'], rtol=0, atol=2e-6), 'Position loss does not close')
        require(abs(losses['final_response'] - pos['response_by_position'][-1]/3) <= 2e-6 and
                abs(losses['native_all_three'] - np.mean(pos['native_by_position'])) <= 2e-6,
                'Three-position native weighting differs')
        closure = row['gradient_reconstruction']
        require(closure['max_abs'] <= 1e-5 and closure['relative_to_component_norm_sum'] <= 1e-3,
                'Gradient decomposition does not close')
        for endpoint in ('canonical_adapter_endpoint', 'actual_public_adapter_endpoint'):
            if endpoint in row:
                require(row[endpoint]['max_abs'] < 3e-4 and row[endpoint]['relative_l2'] < 3e-5,
                        'Adapter endpoint exceeds tolerance')
    require('actual_public_adapter_endpoint' in rows[0], 'Missing actual public adapter validation')
    energy = sum(r['per_position']['target_response_energy_by_position'][-1] for r in rows)
    require(energy > 0, 'Undefined pooled response energy')
    pooled = sum(r['per_position']['response_by_position'][-1] for r in rows) / energy
    share = sum(r['losses']['final_response'] for r in rows) / sum(r['losses']['native_all_three'] for r in rows)
    mg = record['mean_gradient']
    norms = mg['mean_gradient_norms']
    require(norms['native_all_three'] > 0, 'Undefined whole visual gradient norm')
    cos = mg['pairwise']['final_response__native_all_three']['cosine']
    require(cos is not None and -1 <= cos <= 1, 'Undefined average gradient cosine')
    return {'energy_pooled_query_response_nre': pooled, 'response_loss_share_percent': share*100,
            'response_to_visual_gradient_norm_percent': norms['final_response']/norms['native_all_three']*100,
            'mean_gradient_cosine': cos,
            'negative_pair_cosines': sum(r['gradient']['pairs']['final_response__native_all_three']['cosine'] < 0 for r in rows),
            'pairs': 16, 'mean_gradient': mg}


def build(data=DATA):
    sources = {}
    def read(name):
        path = data / name
        sources[name] = sha(path)
        return json.loads(path.read_text())
    panel_name = 'strength_mechanism_v1_panel.json'
    panel = read(panel_name)
    require(panel['release'] == 'ContextWorld-action-strength-32k-v1' and panel['no_test_read'], 'Unexpected panel release/split')
    require(panel['full_metadata_source_episode_overlap_count'] == 0, 'Training/Development sources overlap')
    require(panel['splits']['training']['pairs'] == 512 and panel['splits']['training']['selected_source_groups'] == 188, 'Training fit coverage differs')
    dev = panel['splits']['development']
    require(dev['pairs'] == 256 and dev['selected_source_groups'] == 243, 'Development coverage differs')
    inventory = read('nine_task_root_cause_inventory_v1.json')
    results = {}; previous_selection = None; reference_identities = None
    for regime in REGIMES:
        history = read(f'strength_mechanism_v1_{regime}_history.json')
        objective = read(f'strength_mechanism_v1_{regime}_objective.json')
        model = objective['model']; expected = f'action_strength/lewm/{regime}/s3072'
        require(history['model_id'] == model['id'] == expected and model['training_seed'] == 3072 and
                model['task'] == 'action_strength' and model['family'] == 'lewm' and model['regime'] == regime,
                'Checkpoint identity differs')
        require(history['checkpoint_sha256'] == model['checkpoint_sha256'] and history['checkpoint'] == model['checkpoint'], 'Checkpoint source differs')
        require(history['model_state_hash_before'] == history['model_state_hash_after'] == objective['state_hash_before'], 'Model state differs')
        require(history['no_worldmodel_update'] and history['no_test_read'], 'Unexpected history update/Test access')
        require(history['panel_manifest_sha256'] == objective['selection']['manifest_sha256'] == sources[panel_name], 'Panel manifest bytes differ')
        selection = (objective['selection']['selected_pairs'], objective['selection']['source_groups'])
        if previous_selection is not None: require(selection == previous_selection, 'Gradient panels differ across schemes')
        previous_selection = selection
        readouts = {}
        for name in ('ridge', 'extra_trees'):
            scores = history['readouts'][f'history/{name}']
            for split in ('training', 'development'):
                info = panel['splits'][split]; rows = scores[split]['predictions']; ids = identities(rows)
                expected_ids = {(pair, c) for pair in info['pair_ids'] for c in (0,1)}
                require(ids.keys() == expected_ids and len(rows) == 2*info['pairs'], 'Readout coverage differs')
                for pair in info['pair_ids']:
                    require(ids[(pair,0)]['source_group'] == ids[(pair,1)]['source_group'], 'Classification pair source differs')
                require(len({r['source_group'] for r in rows}) == info['selected_source_groups'], 'Readout source coverage differs')
                require(history['controls'][split]['current_latent_equal_pairs'] == info['pairs'] and
                        history['controls'][split]['current_rgb_equal_pairs'] == info['pairs'] and
                        history['controls'][split]['deterministic_current_only_accuracy_upper_bound'] == .5 and
                        info['actions_equal_pairs'] == info['pairs'], 'Current/action equality control differs')
                measured = float(np.mean([np.mean([r['correct'] for r in rows if r['label'] == c]) for c in (0,1)]))
                require(abs(measured - scores[split]['balanced_accuracy']) < 1e-12, 'Saved readout score differs from predictions')
                require(scores[split]['source_group_bootstrap']['groups'] == info['selected_source_groups'], 'Readout interval groups differ')
                if split == 'development':
                    keys = {(p,c,r['source_group']) for (p,c),r in ids.items()}
                    if reference_identities is not None: require(keys == reference_identities, 'Development identities differ across readouts')
                    reference_identities = keys
            readouts[name] = scores
        training_ids = identities(readouts['ridge']['training']['predictions'])
        for pair,group in zip(*selection):
            require(training_ids[(pair,0)]['source_group'] == group, 'Gradient source differs from Training readout')
        matching = [r for r in inventory['rows'] if (r['task'],r['family'],r['regime']) == ('action_strength','lewm',regime)]
        require(len(matching) == 1 and matching[0]['run_ids'] == [expected], 'Inventory must identify exactly this checkpoint')
        results[regime] = {'scheme': LABELS[regime], 'model_id': expected, 'checkpoint_sha256': model['checkpoint_sha256'],
                           'readouts': readouts, 'original_protocol_response_nre': matching[0]['original_protocol']['nre']['mean'],
                           'gradient': validate_objective(objective,panel)}
    differences = {}
    for left,right in zip(REGIMES,REGIMES[1:]):
        differences[f'{LABELS[right]}-{LABELS[left]}'] = paired_difference(results[left]['readouts']['ridge']['development']['predictions'], results[right]['readouts']['ridge']['development']['predictions'])
    # Per-query predictions remain in the immutable source records, not duplicated here.
    for result in results.values():
        for readout in result['readouts'].values():
            for split in ('training', 'development'):
                readout[split] = {k: v for k, v in readout[split].items() if k != 'predictions'}
    return {'schema':'contextworld.strength_mechanism.v1','scope':'LeWM T1/T2/T3 seed3072; 32k Strength; Training-only reference fit, all Development; posthoc predictor-only visual MSE gradients; no world-model update',
            'source_sha256':sources,'panel_manifest_sha256':sources[panel_name], 'paired_ridge_differences':differences,'rows':results}


def render(summary):
    lines = ['| 方案 | 历史规则识别率 Ridge（%，95% CI） | ExtraTrees（%，95% CI） | 原查询响应 NRE ↓ |',
             '|---|---:|---:|---:|']
    def score(value):
        lo,hi=value['source_group_bootstrap']['balanced_accuracy_95ci']
        return f"{value['balanced_accuracy']*100:.2f} [{lo*100:.2f}, {hi*100:.2f}]"
    for row in summary['rows'].values():
        r=row['readouts'];lines.append(f"| {row['scheme']} | {score(r['ridge']['development'])} | {score(r['extra_trees']['development'])} | {row['original_protocol_response_nre']:.5f} |")
    differences = summary['paired_ridge_differences']
    parts = []
    for contrast in ('T2-T1', 'T3-T2'):
        value = differences[contrast]
        lo, hi = value['ci95_pp']
        parts.append(f"{contrast.replace('-', '−')} 为 {value['difference_pp']:.2f} pp [{lo:.2f}, {hi:.2f}]")
    lines += ['', '配对来源组 bootstrap 的 Ridge 识别率提升（95% CI）：' + '；'.join(parts) + '。']
    lines += ['', '<details>', '<summary>原生视觉损失的局部梯度</summary>', '',
              '| 方案 | Training 查询响应 NRE ↓（能量合并） | 响应损失占比（%） | 响应／整体平均梯度范数比（%） | 响应与整体平均梯度余弦 | 逐查询余弦为负（/16） |',
              '|---|---:|---:|---:|---:|---:|']
    for row in summary['rows'].values():
        g=row['gradient'];lines.append(f"| {row['scheme']} | {g['energy_pooled_query_response_nre']:.5f} | {g['response_loss_share_percent']:.2f} | {g['response_to_visual_gradient_norm_percent']:.2f} | {g['mean_gradient_cosine']:.3f} | {g['negative_pair_cosines']}/16 |")
    lines += ['', '</details>']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    summary=build();summary_path=DATA/'strength_mechanism_v1.json'
    document=ROOT/'docs/ICL_Metric_Study.md';text=document.read_text()
    begin=f'<!-- BEGIN {MARKER} -->';end=f'<!-- END {MARKER} -->'
    require(text.count(begin)==text.count(end)==1, 'Expected exactly one public marker block')
    prefix,rest=text.split(begin,1);_,suffix=rest.split(end,1)
    updated=prefix+begin+'\n'+render(summary)+'\n'+end+suffix
    encoded=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if a.check:
        require(summary_path.exists() and summary_path.read_text()==encoded,'Strength summary is stale')
        require(text==updated,'Strength public table is stale')
    else:
        summary_path.write_text(encoded);document.write_text(updated)
    print('Strength mechanism summary and tables are current')

if __name__=='__main__':main()
