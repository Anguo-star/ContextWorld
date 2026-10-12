#!/usr/bin/env python3
"""Validate package loading, reading sample rows only from Training and Development.

Test checks inspect binding and Lance metadata; no Test rows or scores are read.
"""
import argparse
import json
from pathlib import Path
import sys

import lance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contextworld.benchmarks.bundle_cli import audit_bundle
from contextworld.benchmarks.bundle_development import (
    development_action_normalization, resolve_development_payload,
)
from contextworld.benchmarks.external_model_cli import _public_test_bundle_binding


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.release_root
    registry = json.loads((root / 'task_registry.json').read_text())
    frozen_dev = json.loads((root / 'provenance/ContextWorld-v1/task_registry.json').read_text())
    frozen_test = json.loads((root / 'provenance/ContextWorld-v1-full/task_registry.json').read_text())
    dev_by_task = {c['component_id']: c for c in frozen_dev['components']}
    test_by_task = {c['component_id']: c for c in frozen_test['components']}
    bundle_audit = audit_bundle(root)
    if bundle_audit['status'] != 'passed':
        raise ValueError(f'ContextWorld bundle audit failed: {bundle_audit}')
    sample_rows_read = 0
    test_bindings = 0
    sampled = set()
    checks = []
    for comp in registry['components']:
        task = comp['component_id']
        if comp['development_evaluation'] != dev_by_task[task]['development_evaluation']:
            raise ValueError(f'{task}: frozen Development contract differs')
        if comp['public_test_evaluation'] != test_by_task[task]['public_test_evaluation']:
            raise ValueError(f'{task}: frozen Test contract differs')
        development = resolve_development_payload(root, task=task)
        development_action_normalization(development)
        _public_test_bundle_binding(root, task=task)
        test_bindings += 1
        for payload in comp['payloads']:
            split = payload['split']
            if split not in ('training', 'development', 'test'):
                raise ValueError(f'Unexpected split {split}')
            members = payload['members']
            if payload['payload_kind'] == 'directory':
                entry = root / payload['public_path']
                files = [p for p in entry.rglob('*') if p.is_file()]
                if len(files) != payload['file_count'] or any(p.stat().st_size == 0 for p in files):
                    raise ValueError(f'{comp["component_id"]}/{split}: directory payload coverage mismatch')
                checks.append({'task': comp['component_id'], 'split': split,
                               'payload_id': payload['payload_id'], 'directory_files': len(files),
                               'tables_opened': 0})
                continue
            if not members or len(members) != payload['lance_table_count']:
                raise ValueError(f'{comp["component_id"]}/{split}: Lance table count mismatch')
            rows = 0
            schemas = set()
            for member in members:
                entry = root / member
                if not entry.is_dir():
                    raise ValueError(f'Missing Lance entrypoint: {entry}')
                data_files = list(entry.glob('data/*.lance'))
                if not data_files or any(p.stat().st_size == 0 for p in data_files):
                    raise ValueError(f'Missing or empty Lance data: {entry}')
                ds = lance.dataset(str(entry))
                rows += ds.count_rows()
                schemas.add(str(ds.schema))
            if split in ('training', 'development') and (task, split) not in sampled:
                first = lance.dataset(str(root / members[0]))
                columns = [name for name in ('episode_idx', 'step_idx', 'model_step_idx',
                                             'pixels', 'action', 'action_block')
                           if name in first.schema.names]
                if 'pixels' not in columns or not ({'action', 'action_block'} & set(columns)):
                    raise ValueError(f'{task}/{split}: missing model input columns')
                if first.take([0], columns=columns).num_rows != 1:
                    raise ValueError(f'{task}/{split}: sample row cannot load')
                sample_rows_read += 1
                sampled.add((task, split))
            checks.append({'task': comp['component_id'], 'split': split,
                           'payload_id': payload['payload_id'], 'tables_opened': len(members),
                           'rows': rows, 'schema_variants': len(schemas)})
    if {(c['task'], c['split']) for c in checks} != {(c['component_id'], s) for c in registry['components']
                                                      for s in ('training','development','test')}:
        raise ValueError('Missing task/split Lance load')
    print(json.dumps({'status': 'passed', 'payloads_opened': len(checks),
                      'tables_opened': sum(c['tables_opened'] for c in checks),
                      'task_split_pairs': len({(c['task'], c['split']) for c in checks}),
                      'contextworld_training_views': bundle_audit['training_view_count'],
                      'public_test_bindings': test_bindings,
                      'training_development_sample_rows_read': sample_rows_read,
                      'test_rows_read': 0,
                      'frozen_evaluation_contracts_preserved': True,
                      'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
