#!/usr/bin/env python3
"""Build the fixed-target paired Public Test panel after selection is frozen."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

TASKS = {
    'contact_friction': ('pusht-contact-friction', 'hidden_contact_friction', ('low_friction', 'high_friction')),
    'robot_arm_mass': ('reacher-arm-mass', 'hidden_arm_density', ('lighter', 'heavier')),
}
EXPECTED_SOURCE_FILES = 3
SOURCE = Path('/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1-full')
NORMALIZATION = Path('/tmp/cw-cross-task-mechanism-20261010/native')
OUTPUT = Path('/tmp/cw-fixed-target-confirmation-v1/panels')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def frame(value):
    from PIL import Image
    with Image.open(io.BytesIO(value)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def verify_source_payload(source, component):
    """Verify only this component's declared Test Lance files before opening it."""
    source = Path(source)
    registry = json.loads((source / 'task_registry.json').read_text())
    matched = [row for row in registry['components'] if row.get('dataset_id') == component]
    assert len(matched) == 1, f'expected one registry component for {component}'
    item = matched[0]
    payloads = item['public_test_evaluation']['payloads']
    assert len(payloads) == 1, f'expected one Test payload for {component}'
    declared = payloads[0]
    relative = Path(declared['public_path'])
    assert not relative.is_absolute() and '..' not in relative.parts and relative.suffix == '.lance'
    manifest_path = source / 'manifest.jsonl'
    manifest_sha = sha(manifest_path)
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get('role') == 'dataset_payload'
                and row.get('component') == item['component_id'] and row.get('split') == 'test']
    assert len(selected) == declared.get('file_count', EXPECTED_SOURCE_FILES) == EXPECTED_SOURCE_FILES, 'Test payload manifest file count mismatch'
    prefix = relative.as_posix() + '/'
    assert all(row['path'].startswith(prefix) for row in selected), 'Test payload path outside declared Lance directory'
    assert len({row['path'] for row in selected}) == len(selected), 'duplicate Test payload path'
    verified = []
    for row in sorted(selected, key=lambda value: value['path']):
        path = source / row['path']
        assert path.is_file() and path.resolve().is_relative_to((source / relative).resolve())
        expected_bytes = row['bytes']
        assert path.stat().st_size == expected_bytes, f'Test source size mismatch: {path}'
        assert sha(path) == row['sha256'], f'Test source SHA-256 mismatch: {path}'
        verified.append({'path': row['path'], 'bytes': expected_bytes, 'sha256': row['sha256']})
    actual = {path.relative_to(source).as_posix() for path in (source / relative).rglob('*') if path.is_file()}
    assert actual == {row['path'] for row in verified}, 'unlisted files in selected Test Lance payload'
    return source / relative, manifest_sha, verified


def build(task, source, normalization_root, output):
    import lance
    component, parameter, modes = TASKS[task]
    lance_path, source_manifest_sha, verified_files = verify_source_payload(source, component)
    prior_manifest = normalization_root / task / 'manifest.json'
    prior = json.loads(prior_manifest.read_text())
    ds = lance.dataset(lance_path)
    metadata = ds.to_table(filter='step_idx = 0', columns=[
        'episode_idx', 'pair_id', 'hidden_mode', parameter, 'catalog_index']).to_pylist()
    pairs = {}
    for row in metadata:
        pair = pairs.setdefault(str(row['pair_id']), {})
        mode = str(row['hidden_mode'])
        assert mode not in pair, (row['pair_id'], mode)
        pair[mode] = row
    assert len(pairs) == 256, f'expected all 256 Test pairs, got {len(pairs)}'
    assert all(set(pair) == set(modes) for pair in pairs.values())
    records = [pairs[pair_id][mode] for pair_id in sorted(pairs) for mode in modes]
    episode_ids = [int(row['episode_idx']) for row in records]
    assert len(set(episode_ids)) == len(episode_ids)
    rows = {episode_id: {} for episode_id in episode_ids}
    for start in range(0, len(episode_ids), 100):
        batch = episode_ids[start:start + 100]
        expression = 'episode_idx IN (' + ','.join(map(str, batch)) + ')'
        for row in ds.to_table(filter=f'{expression} AND step_idx IN (0,5,10,15)',
                               columns=['episode_idx', 'step_idx', 'pixels']).to_pylist():
            rows[int(row['episode_idx'])].setdefault(int(row['step_idx']), {})['pixels'] = row['pixels']
        for row in ds.to_table(filter=f'{expression} AND step_idx < 15',
                               columns=['episode_idx', 'step_idx', 'action']).to_pylist():
            rows[int(row['episode_idx'])].setdefault(int(row['step_idx']), {})['action'] = row['action']
    history, future, actions = [], [], []
    for record in records:
        episode = rows[int(record['episode_idx'])]
        images = [frame(episode[step]['pixels']) for step in (0, 5, 10, 15)]
        history.append(np.stack(images[:3]))
        future.append(images[3])
        actions.append(np.asarray([episode[step]['action'] for step in range(15)], np.float32))
    history = np.stack(history)
    future = np.stack(future)
    actions = np.stack(actions)
    assert actions.shape[:2] == (512, 15)
    assert np.array_equal(history[::2, -1], history[1::2, -1]), 'matched current pixels differ'
    assert np.array_equal(actions[::2], actions[1::2]), 'matched query actions differ'
    ids = np.asarray([str(row['pair_id']) for row in records])
    groups = np.asarray([f'{task}-catalog:{int(row["catalog_index"][0])}' for row in records])
    assert np.array_equal(groups[::2], groups[1::2])
    labels = np.tile(np.arange(2), 256)
    destination = output / task
    destination.mkdir(parents=True, exist_ok=True)
    panel = destination / 'test.npz'
    np.savez(panel, history_pixels=history, queryfuture_pixels=future, future_pixels=future[:, None],
             rawactions=actions, action_blocks=actions.reshape(512, 3, 5, -1),
             labels=labels, conditions=labels, physical_target=np.asarray([row[parameter][0] for row in records], np.float32),
             pair_ids=ids, query_ids=ids, source_groups=groups,
             modes=np.asarray([str(row['hidden_mode']) for row in records]), episode_ids=np.asarray(episode_ids))
    manifest = {
        'schema': 'contextworld.fixed_target_test_panel.v1', 'task': task,
        'normalization': prior['normalization'],
        'normalization_manifest': str(prior_manifest), 'normalization_manifest_sha256': sha(prior_manifest),
        'source_root': str(source), 'source_lance': str(lance_path), 'source_lance_version': ds.version,
        'source_registry': str(source / 'task_registry.json'), 'source_registry_sha256': sha(source / 'task_registry.json'),
        'source_manifest': str(source / 'manifest.jsonl'), 'source_manifest_sha256': source_manifest_sha,
        'verified_source_files': verified_files,
        'splits': {'test': {'path': panel.name, 'sha256': sha(panel), 'rows': 512,
                            'queries': 256, 'source_group_count': len(set(groups)),
                            'matched_control': {'matched_group_key': 'query_ids', 'current_action_matched': True},
                            'physical_steps': [0, 5, 10, 15], 'action_blocks': 3, 'steps_per_block': 5}}
    }
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(destination / 'manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=TASKS)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--normalization-root', type=Path, default=NORMALIZATION)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.task, args.source, args.normalization_root, args.output)


if __name__ == '__main__':
    main()
