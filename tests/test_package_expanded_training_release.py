"""Small failure probes for expanded release packaging; no dataset copy needed."""
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import package_expanded_training_release as packaging
from package_expanded_training_release import (
    SPEED_TEST_REL, SPEED_TEST_SOURCE_REL, copy_verified, package_file_paths,
    prepare_from, speed_test_payload_rows,
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


def candidate(tmp_path, monkeypatch):
    source = tmp_path / 'candidate'
    source.mkdir()
    rows = []
    files = {
        'artifacts/speed/training/data.lance': b'payload',
        'task_registry.json': json.dumps({'export_id': 'old', 'release_status': 'staging',
                                          'components': [{'component_id': 'speed'}]}).encode(),
        'VERSION.json': json.dumps({'dataset_version': 'old', 'release_status': 'staging',
                                    'baseline_id': 'old', 'baseline_sha256': 'old',
                                    'source_bundles': {'frozen': {'manifest_sha256': 'abc'}}}).encode(),
        'inventory.json': json.dumps({'dataset_version': 'old', 'totals': {'payload_files': 1}}).encode(),
        'reference/obsolete.json': b'old results',
        'README.md': b'old card',
    }
    for rel, blob in files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        row = {'path': rel, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest(),
               'role': 'dataset_payload' if rel.endswith('.lance') else 'release_metadata'}
        if rel.endswith('.lance'):
            row.update(component='speed', split='training', source_bundle='frozen',
                       source_path=rel, source_sha256=row['sha256'])
        rows.append(row)
    data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (source / 'manifest.jsonl').write_bytes(data)
    (source / 'manifest.sha256').write_text(hashlib.sha256(data).hexdigest() + '  manifest.jsonl\n')
    repo = tmp_path / 'repo'
    template = repo / 'docs/templates/ContextWorld_Expanded_Training_Card.md'
    template.parent.mkdir(parents=True)
    template.write_text('Final public card\n')
    for name in ('LICENSE', 'DATA_LICENSE', 'NOTICE'):
        (repo / name).write_text(name + '\n')
    monkeypatch.setattr('package_expanded_training_release.REPO', repo)
    return source, rows


def test_prepare_from_preserves_source_and_payload_contract(tmp_path, monkeypatch):
    source, rows = candidate(tmp_path, monkeypatch)
    before = {p.relative_to(source).as_posix(): p.read_bytes()
              for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'final'
    result = prepare_from(source, output, 2)
    assert result['status'] == 'passed'
    assert before == {p.relative_to(source).as_posix(): p.read_bytes()
                      for p in source.rglob('*') if p.is_file()}
    assert not (output / 'reference').exists()
    assert (output / 'README.md').read_text() == 'Final public card\n'
    assert '*.npz filter=lfs' in (output / '.gitattributes').read_text()
    version = json.loads((output / 'VERSION.json').read_text())
    assert version['dataset_version'] == 'contextworld-benchmark-v1'
    assert version['release_status'] == 'ready_for_upload'
    assert version['source_bundles'] == {'frozen': {'manifest_sha256': 'abc'}}
    assert 'baseline_id' not in version and 'baseline_sha256' not in version
    assert json.loads((output / 'task_registry.json').read_text())['export_id'] == version['dataset_version']
    final_rows = [json.loads(line) for line in (output / 'manifest.jsonl').read_text().splitlines()]
    assert next(row for row in final_rows if row['role'] == 'dataset_payload') == rows[0]
    assert {row['path'] for row in final_rows} | {'manifest.jsonl', 'manifest.sha256'} == {
        p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}


def test_prepare_from_rejects_corrupt_source_and_existing_output(tmp_path, monkeypatch):
    source, _ = candidate(tmp_path, monkeypatch)
    (source / 'artifacts/speed/training/data.lance').write_bytes(b'corrupt!')
    with pytest.raises(ValueError, match='hash mismatch'):
        prepare_from(source, tmp_path / 'final', 2)
    assert not (tmp_path / 'final').exists()
    with pytest.raises(ValueError, match='Output exists'):
        prepare_from(source, source, 2)


def test_prepare_from_copy_fallback_verifies_payload(tmp_path, monkeypatch):
    source, _ = candidate(tmp_path, monkeypatch)
    payload = source / 'artifacts/speed/training/data.lance'

    def link_fails_and_source_changes(src, dst):
        payload.write_bytes(b'corrupt')
        raise OSError('hardlinks unavailable')

    monkeypatch.setattr(packaging.os, 'link', link_fails_and_source_changes)
    output = tmp_path / 'final'
    with pytest.raises(ValueError, match='Payload hash/size mismatch'):
        prepare_from(source, output, 2)
    assert (output / '.build_in_progress').exists()
    assert not (output / 'manifest.jsonl').exists()


@pytest.mark.parametrize('bad_path', ['../escape', '/absolute', 'artifacts/../escape'])
def test_prepare_from_rejects_unsafe_manifest_path(tmp_path, monkeypatch, bad_path):
    source, rows = candidate(tmp_path, monkeypatch)
    rows[0]['path'] = bad_path
    data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    (source / 'manifest.jsonl').write_bytes(data)
    (source / 'manifest.sha256').write_text(hashlib.sha256(data).hexdigest() + '  manifest.jsonl\n')
    with pytest.raises(ValueError, match='unsafe paths'):
        prepare_from(source, tmp_path / 'final', 2)
    assert not (tmp_path / 'final').exists()
