"""Small, data-free checks for the fixed-target Test split controls."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/evaluate_fixed_target_control.py'
SPEC = importlib.util.spec_from_file_location('evaluate_fixed_target_control', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
BUILDER_SPEC = importlib.util.spec_from_file_location('build_fixed_target_test_panel',
    SCRIPT.with_name('build_fixed_target_test_panel.py'))
BUILDER = importlib.util.module_from_spec(BUILDER_SPEC)
BUILDER_SPEC.loader.exec_module(BUILDER)


def test_default_splits_and_explicit_test_choice():
    assert MODULE.DEFAULT_SPLITS == ('training', 'development')
    assert MODULE.AVAILABLE_SPLITS == ('training', 'development', 'test')


def test_paired_fixed_target_scores_are_finite():
    data = {'labels': np.array([1, 0, 1, 0]),
            'query_ids': np.array(['b', 'a', 'a', 'b']),
            'source_groups': np.array(['g2', 'g1', 'g1', 'g2']),
            'action_blocks': np.zeros((4, 3, 5, 2), np.float32)}
    target = np.array([[2.], [0.], [1.], [0.]], np.float32)
    baseline = {'target': target, 'prediction': target.copy(), 'swapped': target[::-1].copy()}
    candidate = {'target': target.copy(), 'prediction': target + 0.1, 'swapped': target[::-1] + 0.1}
    result = MODULE.evaluate_split(data, baseline, candidate)
    assert result['pairs'] == 2
    assert result['rows'] == 4
    assert result['old']['response_nre'] == 0
    assert result['action_blocks_equal_per_pair']
    assert np.isfinite(result['delta_response_nre']['delta'])


def test_source_payload_verifies_only_selected_component(tmp_path):
    relative = 'artifacts/synthesis/fixture/validation.lance'
    payload = tmp_path / relative / 'data.lance'
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b'fixture')
    extra = [tmp_path / relative / name for name in ('manifest.json', 'index.idx')]
    for path in extra:
        path.write_bytes(path.name.encode())
    registry = {'components': [{'component_id': 'fixture', 'dataset_id': 'pusht-contact-friction',
                 'public_test_evaluation': {'payloads': [{'public_path': relative}]}}]}
    (tmp_path / 'task_registry.json').write_text(json.dumps(registry))
    rows = [{'role': 'dataset_payload', 'component': 'fixture', 'split': 'test',
             'path': path.relative_to(tmp_path).as_posix(), 'bytes': path.stat().st_size,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in [payload, *extra]]
    (tmp_path / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    path, manifest_sha, files = BUILDER.verify_source_payload(tmp_path, 'pusht-contact-friction')
    assert path == tmp_path / relative
    assert manifest_sha == BUILDER.sha(tmp_path / 'manifest.jsonl')
    assert files == sorted([{'path': row['path'], 'bytes': row['bytes'], 'sha256': row['sha256']} for row in rows],
                           key=lambda row: row['path'])
    payload.write_bytes(b'truncated')
    with pytest.raises(AssertionError, match='size mismatch'):
        BUILDER.verify_source_payload(tmp_path, 'pusht-contact-friction')
