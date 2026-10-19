"""Verify one public Test payload against a downloaded bundle's manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from contextworld.paths import repository_root
from contextworld.training.stablewm_bundle import resolve_contextworld_bundle


PUBLIC_TEST_POLICY = "public_offline_final_reporting"
PUBLIC_TEST_EVALUATION_SCHEMA_VERSION = "contextworld.public_test_evaluation.v1"
PUBLIC_TEST_EVALUATION_STATUS = "public_final_reporting_only"


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Unsafe bundle path: {value!r}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_test_bundle(
    bundle_root: Path, *, task: str, release_config: Path | str | None = None
) -> dict[str, Any]:
    """Check exact Test files, contents, and the task's pinned scoring config.

    This validates only the selected task, so a Test run need not rehash the
    entire Training and Development package. It never accepts a caller-made
    ``passed`` receipt in place of checking the files itself.
    """

    root = Path(bundle_root).expanduser().resolve(strict=True)
    identity = resolve_contextworld_bundle(root)
    registry = json.loads((root / "task_registry.json").read_text(encoding="utf-8"))
    public = registry.get("public_test")
    if not isinstance(public, dict) or public.get("included") is not True or public.get("policy") != PUBLIC_TEST_POLICY:
        raise ValueError("Bundle does not publish offline Test")
    matches = [c for c in registry.get("components", []) if c.get("component_id") == task]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate ContextWorld component: {task}")
    component = matches[0]
    evaluation = component.get("public_test_evaluation")
    if (not isinstance(evaluation, dict)
            or evaluation.get("schema_version") != PUBLIC_TEST_EVALUATION_SCHEMA_VERSION
            or evaluation.get("status") != PUBLIC_TEST_EVALUATION_STATUS
            or evaluation.get("split") != "test"):
        raise ValueError(f"Invalid public Test contract for {task}")
    artifact_rel = _safe_relative(str(evaluation.get("artifact_root", "")))
    if artifact_rel.parts[0] != "artifacts":
        raise ValueError(f"Unsafe public Test artifact root for {task}")
    artifact_root = root / artifact_rel
    if not artifact_root.is_dir():
        raise ValueError(f"Missing or unsafe public Test artifact root for {task}")
    payloads = evaluation.get("payloads")
    if not isinstance(payloads, list) or not payloads:
        raise ValueError(f"Public Test contract has no payload binding for {task}")
    prefixes = []
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid public Test payload binding for {task}")
        relative = _safe_relative(str(payload.get("public_path", "")))
        if relative != artifact_rel and artifact_rel not in relative.parents:
            raise ValueError(f"Public Test payload escapes artifact root for {task}")
        if not (root / relative).exists():
            raise ValueError(f"Missing public Test payload binding for {task}")
        prefixes.append(relative.as_posix().rstrip("/") + "/")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError(f"Duplicate public Test payload binding for {task}")

    config_rel = _safe_relative(str(component.get("release_config", "")))
    expected_config = str(component.get("release_config_sha256", ""))
    config = Path(release_config) if release_config is not None else repository_root() / config_rel
    if not config.is_file() or _sha256(config) != expected_config:
        raise ValueError(f"Task release config differs from registry: {task}")

    rows = {}
    covered = [False] * len(prefixes)
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("component") != task or row.get("split") != "test":
                continue
            if row.get("role") != "dataset_payload":
                raise ValueError(f"Unexpected Test manifest role for {task}")
            relative = _safe_relative(str(row.get("path", "")))
            name = relative.as_posix()
            if "score_receipts/" in name or name in rows:
                raise ValueError(f"Unsafe or duplicate Test manifest path: {name}")
            hits = [i for i, prefix in enumerate(prefixes) if name.startswith(prefix)]
            if len(hits) != 1:
                raise ValueError(f"Test manifest path outside declared payload: {name}")
            covered[hits[0]] = True
            rows[name] = row
    if not rows or not all(covered):
        raise ValueError(f"Manifest does not bind every public Test payload for {task}")
    found = set()
    for prefix in prefixes:
        payload_root = root / prefix[:-1]
        if payload_root.is_dir() and payload_root.is_symlink():
            raise ValueError(f"Directory symlink in public Test payload: {payload_root}")
        paths = payload_root.rglob("*") if payload_root.is_dir() else (payload_root,)
        for path in paths:
            if path.is_dir() and path.is_symlink():
                raise ValueError(f"Directory symlink in public Test payload: {path}")
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    if found != set(rows):
        raise ValueError(f"Public Test files differ from manifest: {task}")
    total_bytes = 0
    for name, row in rows.items():
        path = root / name
        size = int(row["bytes"])
        digest = str(row["sha256"])
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"Public Test file differs from manifest: {name}")
        total_bytes += size
    return {
        **identity,
        "task": task,
        "artifact_root": artifact_rel.as_posix(),
        "manifest_payload_files": len(rows),
        "test_payload_bytes": total_bytes,
        "test_payload_sha256_verified": True,
        "release_config_sha256": expected_config,
        "selection_policy": evaluation.get("selection_policy"),
    }
