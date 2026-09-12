#!/usr/bin/env python3
"""Check all nine native bundle readers locally, without model inference."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lance
import numpy as np

from contextworld.benchmarks.bundle_cli import audit_bundle
from contextworld.benchmarks.bundle_development import resolve_development_payload, development_action_normalization
from contextworld.benchmarks.external_model_cli import _public_test_bundle_binding

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--benchmark-root', type=Path, required=True)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
root = args.benchmark_root.resolve()
registry = json.loads((root / 'task_registry.json').read_text())
sources = root / 'provenance'
dev_registry = json.loads((sources / 'ContextWorld-v1/task_registry.json').read_text())
test_registry = json.loads((sources / 'ContextWorld-v1-full/task_registry.json').read_text())
dev_by_id = {c['component_id']: c for c in dev_registry['components']}
test_by_id = {c['component_id']: c for c in test_registry['components']}
audit = audit_bundle(root)
assert audit['status'] == 'passed', audit
rows = []
for component in registry['components']:
    task = component['component_id']
    assert component['development_evaluation'] == dev_by_id[task]['development_evaluation']
    assert component['public_test_evaluation'] == test_by_id[task]['public_test_evaluation']
    development = resolve_development_payload(root, task=task)
    development_action_normalization(development)
    public = _public_test_bundle_binding(root, task=task)
    probes = []
    for split in ('training', 'development', 'test'):
        payloads = [p for p in component['payloads'] if p['split'] == split]
        assert payloads, (task, split)
        payload = payloads[0]
        members = payload.get('members', [])
        probe = None
        for member in members:
            candidate = root / member
            if candidate.is_dir() and candidate.suffix == '.lance':
                probe = candidate
                break
        if probe is None:
            base = root / payload['public_path']
            probe = base if base.suffix == '.lance' else next(base.rglob('*.lance'), None)
        if probe is not None:
            dataset = lance.dataset(str(probe))
            count = dataset.count_rows()
            assert count > 0
            columns = [k for k in ('episode_idx', 'step_idx', 'model_step_idx', 'pixels', 'action', 'action_block') if k in dataset.schema.names]
            assert 'pixels' in columns and ('action' in columns or (task == 'cube_gripper_carry' and 'action_block' in columns)), (task, split, dataset.schema)
            sample = dataset.take([0], columns=columns)
            assert sample.num_rows == 1
            probes.append({'split': split, 'format': 'lance', 'member': str(probe.relative_to(root)), 'rows': count, 'sample_rows_read': 1})
        else:
            base = root / payload['public_path']
            npz = next(base.rglob('*.npz'), None)
            if npz is not None:
                with np.load(npz, allow_pickle=False) as sample:
                    fields = list(sample.files)
                    assert fields
                    shapes = {k: list(sample[k].shape) for k in fields}
                probes.append({'split': split, 'format': 'npz', 'member': str(npz.relative_to(root)), 'fields': shapes})
            else:
                assert task == 'speed' and split == 'test'
                files = sorted((base / 'catalogs').glob('*.json'))
                files = [p for p in files if p.name != 'build_report.json']
                assert len(files) == 4
                for path in files:
                    assert json.loads(path.read_text())
                probes.append({'split': split, 'format': 'json_catalogs', 'tracks_read': len(files)})
    rows.append({'component': task, 'history_length': development.history_length, 'development_members': len(development.members), 'public_test_files': public['manifest_payload_files'], 'probes': probes})
    print(task, 'loading passed', flush=True)
result = {'status': 'passed', 'scope': 'local_existing_environment_loading_no_model_inference', 'component_count': len(rows), 'split_probe_count': sum(len(row['probes']) for row in rows), 'registry_contracts_preserved': True, 'training_view_count': audit['training_view_count'], 'manifest_file_count': audit['manifest_file_count'], 'components': rows}
if args.output:
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'components'}))
