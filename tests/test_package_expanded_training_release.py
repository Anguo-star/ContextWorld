"""Small failure probes for expanded release packaging; no dataset copy needed."""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from package_expanded_training_release import copy_verified, package_file_paths


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
