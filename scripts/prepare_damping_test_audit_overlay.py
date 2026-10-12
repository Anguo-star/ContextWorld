#!/usr/bin/env python3
"""Rebuild the read-only Damping Test provenance overlay from pinned bytes.

Only audit evidence is placed in the overlay. Historical source files are read
from local git object storage and never imported or executed by this builder.
The native scorer remains the current, frozen ContextWorld scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
STABLE_ROOT = ROOT.parent / "stable-worldmodel"
BUNDLE = ROOT.parents[1] / "data/world_model/ContextWorld-v1-full"
RELEASE_CONFIG = ROOT / "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml"
DEFAULT_OVERLAY = Path("/tmp/contextworld-damping-audit-overlay/ContextWorld")
# Exact local git objects whose file contents match the frozen release YAML.
HISTORICAL = {
    "contextworld/benchmarks/adapters.py": (ROOT, "dc09c275336b03102b022614f3911150577a278d"),
    "pyproject.toml": (ROOT, "eaef9056838c68d98ae7820a47ee0d57a4e09299"),
    "../stable-worldmodel/scripts/train/config/lewm.yaml": (STABLE_ROOT, "9603ee6406f3c0e0e90692ef7c2f56ce7e5e64eb"),
    "../stable-worldmodel/scripts/train/config/pldm.yaml": (STABLE_ROOT, "4b6f5d94693631ce64ed1f561dd0bc1d23ca38fa"),
    "../stable-worldmodel/stable_worldmodel/wm/lewm/lewm.py": (STABLE_ROOT, "1986aae61977434d3d55197d2779a5f300deb82f"),
    "../stable-worldmodel/stable_worldmodel/wm/pldm/pldm.py": (STABLE_ROOT, "299caa1fdc1eece1b93a6ba397c882b18f8a3430"),
    "../stable-worldmodel/stable_worldmodel/wm/utils.py": (STABLE_ROOT, "d9cd2884b9528e55445b6f2995ffd68431b03bab"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def place_link(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_file() or destination.is_symlink():
            destination.unlink()
        else:
            raise ValueError(f"refusing to replace directory: {destination}")
    destination.symlink_to(source.resolve())


def git_bytes(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=repo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    args = parser.parse_args()
    overlay = args.overlay.resolve()
    release = yaml.safe_load(RELEASE_CONFIG.read_text())
    sources = []
    for name, spec in release["identity"].items():
        rel = spec["path"]
        target = overlay / rel
        if rel in HISTORICAL:
            repo, commit = HISTORICAL[rel]
            git_path = rel.removeprefix("../stable-worldmodel/")
            payload = git_bytes(repo, commit, git_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                target.unlink()
            target.write_bytes(payload)
            origin = {"kind": "git", "repo": str(repo), "commit": commit, "path": git_path}
        else:
            source = (ROOT / rel).resolve()
            place_link(target, source)
            origin = {"kind": "source", "path": str(source)}
        observed = sha256(target)
        required = spec.get("required_for_release_audit", True)
        if required and observed != spec["sha256"]:
            raise ValueError(f"required identity differs: {name}: {observed}")
        sources.append({"name": name, "path": rel, "expected_sha256": spec["sha256"],
                        "observed_sha256": observed, "required": required, "origin": origin})

    for spec in release["data"]["artifacts"].values():
        rel = spec["path"]
        place_link(overlay / rel, ROOT / rel)
        if sha256(overlay / rel) != spec["sha256"]:
            raise ValueError(f"release artifact differs: {rel}")
    tree_rel = Path(release["data"]["artifact_tree"]["root"])
    source_tree = ROOT / tree_rel
    for source in source_tree.rglob("*"):
        if source.is_file():
            place_link(overlay / tree_rel / source.relative_to(source_tree), source)

    decision = release["reference_results"]["current_decision"]
    place_link(overlay / decision["path"], ROOT / decision["path"])
    if sha256(overlay / decision["path"]) != decision["sha256"]:
        raise ValueError("Development decision receipt differs")

    test_tree = tree_rel / release["data"]["lance_tables"]["validation"]
    source_files = sorted(p.relative_to(source_tree / "validation.lance") for p in
                          (source_tree / "validation.lance").rglob("*") if p.is_file())
    file_matches = []
    for rel in source_files:
        source_hash = sha256(source_tree / "validation.lance" / rel)
        bundle_hash = sha256(BUNDLE / test_tree / rel)
        if source_hash != bundle_hash:
            raise ValueError(f"Test table differs: {rel}")
        file_matches.append({"path": str(rel), "sha256": source_hash})

    sys.path.insert(0, str(ROOT))
    from contextworld.benchmarks.motion_damping_icl_data import audit_motion_damping_icl_release
    audit = audit_motion_damping_icl_release(repo_root=overlay, full=False)
    receipt = {
        "schema": "contextworld.damping-test-audit-overlay.v1",
        "overlay": str(overlay),
        "release_config": str(RELEASE_CONFIG),
        "release_config_sha256": sha256(RELEASE_CONFIG),
        "historical_sources": [row for row in sources if row["origin"]["kind"] == "git"],
        "optional_identity_drift": [row for row in sources if not row["required"] and row["observed_sha256"] != row["expected_sha256"]],
        "decision_sha256": sha256(overlay / decision["path"]),
        "test_validation_bundle_files": file_matches,
        "audit": {
            "passed": audit["passed"],
            "failed_files": [key for key, value in audit["files"].items() if not value["passed"]],
            "failed_data_checks": [key for key, value in audit["data_checks"].items() if not value],
            "artifact_tree": audit["artifact_tree"],
            "causal_contract_passed": audit["causal_data_contract"]["passed"],
            "reference_result_passed": audit["reference_result"]["passed"],
        },
    }
    receipt_path = overlay.parent / "audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    if not audit["passed"]:
        raise SystemExit(f"release audit failed; see {receipt_path}")
    print(f"Overlay: {overlay}\nAudit: passed\nReceipt: {receipt_path}")


if __name__ == "__main__":
    main()
