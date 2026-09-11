"""Selected data must remain unchanged even when its manifest is untouched."""

import hashlib
import importlib.util
import json
from pathlib import Path


def test_selected_bytes_drift_cannot_hide_behind_unchanged_manifest(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/freeze_current_reference_baseline.py"
    spec = importlib.util.spec_from_file_location("selected_data_freeze", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    member = tmp_path / "data.lance"
    member.mkdir()
    actual = member / "data.bin"
    actual.write_bytes(b"original")
    digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"path": "data.lance/data.bin", "sha256": digest}) + "\n")
    data = {
        "package_manifests": [module.identity(manifest, root=tmp_path)],
        "member_identities": [{
            "path": "data.lance", "resolved_path": str(member),
            "sha256": module.canonical_sha256([("data.lance/data.bin", digest)]),
            "status": "verified_from_package_manifest",
        }],
    }
    errors = []
    module._verify_member_identities(data, root=tmp_path, errors=errors)
    assert errors == []
    actual.write_bytes(b"modified")
    module._verify_member_identities(data, root=tmp_path, errors=errors)
    assert any("selected data file SHA drifted" in error for error in errors)
