"""Public metadata and integrity CLI for the ``ContextWorld-v1`` bundle.

This command deliberately has no historical-suite fallback.  It reads the
clean Training/Development bundle named by ``CONTEXTWORLD_BENCHMARK_ROOT`` and
never resolves ``CONTEXTWORLD_ARTIFACT_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from contextworld.training.stablewm_bundle import (
    build_contextworld_dataset_uri,
    describe_contextworld_dataset,
    resolve_contextworld_bundle,
    resolve_contextworld_development_payload,
)


def _root(value: str | None) -> Path:
    configured = value or os.environ.get("CONTEXTWORLD_BENCHMARK_ROOT")
    if not configured:
        raise ValueError(
            "Set --benchmark-root or CONTEXTWORLD_BENCHMARK_ROOT to the "
            "ContextWorld-v1 directory"
        )
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError(f"ContextWorld-v1 root must be absolute: {path}")
    return Path(str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(root: Path) -> dict[str, Any]:
    path = root / "VERSION.json"
    if not path.is_file():
        raise ValueError(f"ContextWorld-v1 has no VERSION.json: {root}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ContextWorld-v1 VERSION.json is not an object: {path}")
    return value


def bundle_info(root: Path) -> dict[str, Any]:
    identity = resolve_contextworld_bundle(root)
    version = _version(root)
    components = []
    for component in identity["component_ids"]:
        development = resolve_contextworld_development_payload(
            root, component=component
        )
        components.append(
            {
                "component_id": component,
                "dataset_id": development["dataset_id"],
                "environment": development["environment"],
                "history_length": development["history_length"],
                "action_dimension": development["action_dimension"],
                "development_payload_id": development["payload_id"],
                "development_member_count": len(
                    development["relative_members"]
                ),
            }
        )
    return {
        "schema_version": "contextworld.bundle-info.v1",
        "bundle_root": str(root),
        "dataset_version": version.get("dataset_version"),
        "release_status": version.get("release_status"),
        "public_test_included": version.get("public_test_included"),
        "manifest_sha256": identity["manifest_sha256"],
        "task_registry_sha256": identity["task_registry_sha256"],
        "component_count": len(components),
        "components": components,
    }


def audit_bundle(root: Path, *, full: bool = False) -> dict[str, Any]:
    info = bundle_info(root)
    registry = json.loads((root / "task_registry.json").read_text(encoding="utf-8"))
    training_views = []
    for component in registry["components"]:
        component_id = str(component["component_id"])
        training_payloads = [
            value
            for value in component.get("payloads", [])
            if value.get("split") == "training"
        ]
        if not training_payloads:
            raise ValueError(f"Component has no Training payload: {component_id}")
        for payload in training_payloads:
            uri = build_contextworld_dataset_uri(
                root,
                component=component_id,
                split="training",
                payload_id=str(payload["payload_id"]),
            )
            identity = describe_contextworld_dataset(uri)
            training_views.append(
                {
                    "component_id": component_id,
                    "payload_id": identity["payload_id"],
                    "member_count": identity["member_count"],
                }
            )

    missing: list[str] = []
    size_mismatch: list[str] = []
    hash_mismatch: list[str] = []
    file_count = 0
    with (root / "manifest.jsonl").open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            relative = row.get("path")
            if not isinstance(relative, str) or not relative:
                raise ValueError(f"Invalid manifest path at line {line_number}")
            path = root / relative
            file_count += 1
            if not path.is_file():
                missing.append(relative)
                continue
            observed_size = path.stat().st_size
            declared_size = row.get("bytes", row.get("size", -1))
            if observed_size != int(declared_size):
                size_mismatch.append(relative)
                continue
            if full and _sha256(path) != row.get("sha256"):
                hash_mismatch.append(relative)

    passed = not missing and not size_mismatch and not hash_mismatch
    return {
        "schema_version": "contextworld.bundle-audit.v1",
        "status": "passed" if passed else "failed",
        "full_content_hash_audit": bool(full),
        "public_test_included": info["public_test_included"],
        "component_count": info["component_count"],
        "training_view_count": len(training_views),
        "training_views": training_views,
        "manifest_file_count": file_count,
        "missing": missing,
        "size_mismatch": size_mismatch,
        "hash_mismatch": hash_mismatch,
        "bundle": info,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="contextworld-benchmark",
        description="Inspect or audit a clean ContextWorld-v1 bundle.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("info", "audit"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--benchmark-root",
            help=(
                "Absolute ContextWorld-v1 root; defaults to "
                "CONTEXTWORLD_BENCHMARK_ROOT"
            ),
        )
        if command == "audit":
            child.add_argument(
                "--full",
                action="store_true",
                help="Rehash every distributed payload file (slow).",
            )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        root = _root(args.benchmark_root)
        result = (
            bundle_info(root)
            if args.command == "info"
            else audit_bundle(root, full=bool(args.full))
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ContextWorld-v1 {args.command} failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status", "passed") == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
