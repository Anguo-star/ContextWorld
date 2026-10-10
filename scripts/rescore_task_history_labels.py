#!/usr/bin/env python3
"""Rescore fixed native history latents after a verified panel-label correction.

This loads no world model and writes a new result/cache/receipt directory. The
original result and latent files remain immutable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import diagnose_task_history_readout as readout


def run(args):
    old_path = args.result
    old = json.loads(old_path.read_text())
    panel_path = args.panel / 'manifest.json'
    manifest = json.loads(panel_path.read_text())
    assert old['task'] == manifest['task']
    assert old['readout'].startswith('StandardScaler(training only) + Ridge(alpha=1')
    assert old['model_state_hash_before'] == old['model_state_hash_after']
    assert old['panel_manifest_sha256'] != readout.sha(panel_path), 'No panel revision to rescore'
    readout.verify_dual()
    output = args.output
    assert not output.exists(), 'Use a new output directory to preserve the original result'
    output.mkdir(parents=True)
    data = {}; cache = {}; checks = {}
    for split in ('training', 'development'):
        info = manifest['splits'][split]
        raw_path = Path(info['path'])
        if not raw_path.is_absolute():
            raw_path = args.panel / raw_path
        assert readout.sha(raw_path) == info['sha256']
        old_cache = Path(old['native_cache'][split]['path'])
        if readout.sha(old_cache) != old['native_cache'][split]['sha256']:
            archived = old_cache.with_name(old_cache.stem + '_uncorrected_labels' + old_cache.suffix)
            assert archived.exists() and readout.sha(archived) == old['native_cache'][split]['sha256']
            old_cache = archived
        with np.load(raw_path, allow_pickle=False) as raw, np.load(old_cache, allow_pickle=False) as native:
            retained = {key: native[key] for key in native.files}
            assert len(retained['history']) == len(raw['labels'])
            assert retained['queryfuture'].shape[0] == len(raw['labels'])
            for key in ('rawactions', 'pair_ids', 'query_ids', 'source_groups', 'modes', 'episode_ids'):
                if key in raw.files and key in retained:
                    assert np.array_equal(raw[key], retained[key]), (split, key)
            old_labels = retained['labels'].copy()
            for key in ('labels', 'conditions', 'physical_target', 'physical_group_labels'):
                if key in raw.files:
                    retained[key] = raw[key]
            checks[split] = dict(raw_npz_path=str(raw_path), raw_npz_sha256=info['sha256'],
                old_native_cache_sha256=old['native_cache'][split]['sha256'],
                labels_changed=int(np.count_nonzero(old_labels != retained['labels'])),
                native_history_and_future_reused=True, unchanged_actions_and_query_identity=True)
        path = output / f'{split}_native.npz'
        np.savez(path, **retained)
        cache[split] = {**old['native_cache'][split], 'path':str(path), 'sha256':readout.sha(path)}
        data[split] = retained
    if old['task'] == 'door':
        for split, item in data.items():
            modes = item['modes'].astype(str)
            assert np.all(item['labels'][np.char.endswith(modes, 'passable')] == 1), split
            assert np.all(item['labels'][np.char.endswith(modes, 'blocked')] == 0), split
    train_x = readout.native_history(data['training']['history'])
    dev_x = readout.native_history(data['development']['history'])
    if old['task'] == 'speed':
        scores = readout.regress(train_x, dev_x, data['training'], data['development'])
    else:
        scores = readout.classify(train_x, dev_x, data['training'], data['development'])
    readouts = {'history/ridge':scores}
    if old['task'] == 'action_delay' and all('physical_group_labels' in data[s] for s in data):
        grouped = {s:{**item, 'labels':item['physical_group_labels']} for s,item in data.items()}
        readouts['history/ridge_six_physical_groups'] = readout.classify(train_x, dev_x, grouped['training'], grouped['development'])
    result = {**old, 'panel_manifest_sha256':readout.sha(panel_path), 'native_cache':cache,
        'readouts':readouts, 'rescored_from':dict(original_result_path=str(old_path),
            original_result_sha256=readout.sha(old_path), method='fixed ridge refit from unchanged native embeddings; no world model loaded')}
    readout.write(output / 'result.json', result)
    receipt = dict(schema='contextworld.native_history_label_rescore.v1', model_id=old['model_id'],
        old_result_path=str(old_path), old_result_sha256=readout.sha(old_path),
        new_result_sha256=readout.sha(output / 'result.json'),
        old_manifest_sha256=old['panel_manifest_sha256'], new_manifest_sha256=readout.sha(panel_path),
        no_world_model_load=True, checks=checks)
    readout.write(output / 'rescore_receipt.json', receipt)
    print(output / 'result.json', scores['development'].get('balanced_accuracy', scores['development'].get('mae')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True, help='Original result.json')
    parser.add_argument('--panel', type=Path, required=True, help='Directory with corrected manifest.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory for corrected result and caches')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
