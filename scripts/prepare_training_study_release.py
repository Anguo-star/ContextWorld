#!/usr/bin/env python3
"""Prepare a metadata-only nine-task release plan; never copy or upload payloads."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = ('speed', 'door', 'action_delay', 'action_strength', 'contact_friction',
         'motion_damping', 'robot_arm_mass', 'cube_gripper_carry', 'portal_exit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    study = json.loads((REPO / 'docs/research/data/icl_training_study_v2.json').read_text())
    bundles = {}
    for row in study['rows']:
        if not row.get('training_data_version') or row['regime'] == 'original' or row.get('comparison_variant'):
            continue
        # The three inherited tasks use the base release selection below.
        # Historical whole-bundle hashes also cover older Development layouts;
        # they do not identify a newly expanded Training bundle.
        if row['training_data_version'] == 'ContextWorld-v1':
            continue
        identity = (row['training_data_version'], row['training_dataset_manifest_sha256'])
        if row['task'] in bundles and bundles[row['task']] != identity:
            raise ValueError(f"Inconsistent training data identity: {row['task']}")
        bundles[row['task']] = identity
    inventories = {}
    def read(name, expected=None):
        if name not in inventories:
            path = args.dataset_root / name / 'manifest.jsonl'
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            if len({r['path'] for r in rows}) != len(rows):
                raise ValueError(f'Duplicate paths: {name}')
            inventories[name] = (digest, rows)
        digest, rows = inventories[name]
        if expected and digest != expected:
            raise ValueError(f'Training manifest differs from reported experiment: {name}')
        return rows
    base = read('ContextWorld-v1')
    tests = read('ContextWorld-v1-full')
    tasks, selected = [], []
    for task in TASKS:
        bundle, expected = bundles.get(task, ('ContextWorld-v1', None))
        train = [r for r in read(bundle, expected) if r.get('component') == task and r.get('split') == 'training' and r['role'] == 'dataset_payload']
        dev = [r for r in base if r.get('component') == task and r.get('split') == 'development' and r['role'] == 'dataset_payload']
        test = [r for r in tests if r.get('component') == task and r.get('split') == 'test' and r['role'] == 'dataset_payload']
        if not train or not dev or not test:
            raise ValueError(f'Missing split manifest coverage: {task}')
        # Evaluation always comes from the frozen sources, never the expanded bundle.
        stats = {}
        for split, name, rows in [('training', bundle, train), ('development', 'ContextWorld-v1', dev), ('test', 'ContextWorld-v1-full', test)]:
            stats[split] = {'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}
            selected.extend(dict(r, source_bundle=name) for r in rows)
        tasks.append({'task': task, 'training_source': bundle, 'training_status': 'expanded' if bundle != 'ContextWorld-v1' else 'inherited', 'reported_icl_training_runs': sum(row['task'] == task and row['regime'] in ('scratch', 'joint', 'frozen') and row['measurement_status'] == 'available' and not row.get('comparison_variant') for row in study['rows']), 'splits': stats})
    if len({r['path'] for r in selected}) != len(selected):
        raise ValueError('Destination paths collide')
    report = {
        'schema_version': 'contextworld.release-plan.v2',
        'study_source': 'docs/research/data/icl_training_study_v2.json',
        'status': 'metadata_only_unpublished_candidate',
        'proposed_version': 'expanded-training-v1',
        'payload_bytes_verified': False,
        'inherited_training_identity_verified': False,
        'registry_merge_verified': False,
        'formal_baseline_accepted': False,
        'evaluation_source_policy': 'Frozen Development and Test; expanded Training only',
        'sources': {k: {'manifest_sha256': v[0]} for k, v in sorted(inventories.items())},
        'tasks': tasks,
        'payload_files': len(selected), 'payload_bytes': sum(r['bytes'] for r in selected),
        'remaining_checks': ['Read and hash every selected payload', 'Verify inherited Training payloads against the declared reference identities', 'Merge and validate task registry and normalizer references', 'Load every task and split from assembled package', 'Finalize reference results and acceptance scope', 'Verify licenses and card against assembled package'],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'release_plan.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'proposed_manifest.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(selected, key=lambda r: r['path'])))
    print(f"Prepared {len(tasks)} tasks, {len(selected)} manifest entries; no payload copied or uploaded.")


if __name__ == '__main__':
    main()
