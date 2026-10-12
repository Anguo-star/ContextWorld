"""Small failure probes for expanded release packaging; no dataset copy needed."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_expanded_training_release import (
    SPEED_TEST_REL, SPEED_TEST_SOURCE_REL, copy_verified, package_file_paths,
    speed_test_payload_rows,
)


@pytest.mark.parametrize('declared', [
    {'bytes': 7, 'sha256': hashlib.sha256(b'abcdef').hexdigest()},
    {'bytes': 6, 'sha256': hashlib.sha256(b'abcdeg').hexdigest()},
])
def test_copy_rejects_truncated_or_same_size_corrupt_payload(tmp_path, declared):
    source = tmp_path / 'source.lance'
    source.write_bytes(b'abcdef')
    with pytest.raises(ValueError, match='hash/size mismatch'):
        copy_verified(source, tmp_path / 'candidate.lance', declared)


def test_nested_provenance_manifest_is_included(tmp_path):
    (tmp_path / 'manifest.jsonl').write_text('root manifest')
    (tmp_path / 'manifest.sha256').write_text('root sidecar')
    (tmp_path / '.build_in_progress').write_text('incomplete')
    nested = tmp_path / 'provenance' / 'ContextWorld-v1'
    nested.mkdir(parents=True)
    (nested / 'manifest.jsonl').write_text('source manifest')
    (nested / 'manifest.sha256').write_text('source sidecar')
    (nested / 'task_registry.json').write_text('{}')
    (tmp_path / 'VERSION.json').write_text('{}')
    paths = {p.relative_to(tmp_path).as_posix() for p in package_file_paths(tmp_path)}
    assert paths == {'VERSION.json', 'provenance/ContextWorld-v1/manifest.jsonl',
                     'provenance/ContextWorld-v1/manifest.sha256',
                     'provenance/ContextWorld-v1/task_registry.json'}


def test_speed_catalog_closure_checks_all_referenced_payload_hashes(tmp_path):
    frozen = tmp_path / 'ContextWorld-v1-full'
    source = tmp_path / SPEED_TEST_SOURCE_REL
    rows = []
    tracks = ('seen_for_multi', 'unseen_interpolation', 'extrapolation_low',
              'extrapolation_high')
    for track in tracks:
        relative = f'{SPEED_TEST_REL}/payloads/{track}/example.npz'
        payload = source / f'payloads/{track}/example.npz'
        payload.parent.mkdir(parents=True, exist_ok=True)
        blob = track.encode()
        payload.write_bytes(blob)
        digest = hashlib.sha256(blob).hexdigest()
        catalog_rel = f'{SPEED_TEST_REL}/catalogs/{track}.json'
        catalog = frozen / catalog_rel
        catalog.parent.mkdir(parents=True, exist_ok=True)
        catalog.write_text(json.dumps({'track': track, 'bundles': [
            {'payload': relative, 'payload_sha256': digest}]}))
        rows.append({'path': catalog_rel, 'role': 'dataset_payload',
                     'bytes': catalog.stat().st_size,
                     'sha256': hashlib.sha256(catalog.read_bytes()).hexdigest()})
    selected = speed_test_payload_rows(tmp_path, {'root': frozen, 'rows': rows})
    assert len(selected) == 4
    assert sum(row['bytes'] for row in selected) == sum(len(track) for track in tracks)
    assert all(row['split'] == 'test' and row['component'] == 'speed' for row in selected)
    (source / 'payloads/seen_for_multi/example.npz').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='catalog payload hash differs'):
        speed_test_payload_rows(tmp_path, {'root': frozen, 'rows': rows})
