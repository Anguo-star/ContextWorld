#!/usr/bin/env python3
"""Runtime-only compatibility entry for frozen native Public Test commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.protocol import _prepare_optional_flash_attention


def _tree_hashes(table: Path) -> dict[str, str]:
    if not table.is_dir():
        raise ValueError(f"missing validation.lance table: {table}")
    files = sorted(path for path in table.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"empty validation.lance table: {table}")
    return {path.relative_to(table).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


def verify_damping_overlay(benchmark_root: Path, overlay_root: Path) -> dict:
    from contextworld.benchmarks.motion_damping_icl_data import (
        audit_motion_damping_icl_release, load_motion_damping_icl_release,
    )
    from contextworld.paths import resolve_contextworld_path

    registry = json.loads((benchmark_root / "task_registry.json").read_text())
    component = next((row for row in registry["components"]
                      if row["component_id"] == "motion_damping"), None)
    if component is None:
        raise ValueError("bundle lacks motion_damping")
    bindings = component["public_test_evaluation"]["payloads"]
    if len(bindings) != 1:
        raise ValueError("expected one damping Test table binding")
    relative = Path(bindings[0]["public_path"])
    if (relative.is_absolute() or ".." in relative.parts or relative.name != "validation.lance"):
        raise ValueError("unsafe damping Test table binding")
    release = load_motion_damping_icl_release()
    overlay_table = (resolve_contextworld_path(release["data"]["artifact_tree"]["root"],
                                                repo_root=overlay_root) /
                     release["data"]["lance_tables"]["validation"])
    bundle_hashes = _tree_hashes(benchmark_root / relative)
    if _tree_hashes(overlay_table) != bundle_hashes:
        raise ValueError("damping overlay validation.lance differs from frozen Test table")
    audit = audit_motion_damping_icl_release(repo_root=overlay_root, full=False)
    if audit.get("passed") is not True:
        raise ValueError("damping overlay release audit failed")
    return {"validation_files": len(bundle_hashes), "audit_passed": True,
            "overlay_root": str(overlay_root)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--damping-audit-root", type=Path)
    options, forwarded = parser.parse_known_args(argv)
    if options.artifact_root is not None and not options.artifact_root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if options.damping_audit_root is not None and not options.damping_audit_root.is_absolute():
        raise ValueError("--damping-audit-root must be absolute")
    _prepare_optional_flash_attention()
    from contextworld.benchmarks import external_model_cli as cli

    args = cli.parse_args(forwarded)
    benchmark_root = Path(args.benchmark_root).resolve()
    overlay = options.damping_audit_root.resolve() if options.damping_audit_root else None
    if args.task == "motion_damping" and overlay is None:
        raise ValueError("motion_damping requires --damping-audit-root")
    overlay_evidence = (verify_damping_overlay(benchmark_root, overlay)
                        if args.task == "motion_damping" else None)
    original_keywords = cli._scorer_keywords
    original_run = cli.run

    def keywords(binding, scorer_args, *, repo_root=cli.ROOT):
        if binding.task == "motion_damping":
            repo_root = overlay
        return original_keywords(binding, scorer_args, repo_root=repo_root)

    def run(scorer_args):
        payload = original_run(scorer_args)
        payload["runtime_compatibility"] = {
            "wrapper": "scripts.run_current_reference_test_compat",
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "artifact_root": str(options.artifact_root.resolve()) if options.artifact_root else None,
            "damping_overlay": overlay_evidence,
        }
        return payload

    cli._scorer_keywords = keywords
    cli.run = run
    previous_artifact_root = os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT")
    if options.artifact_root is not None:
        os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] = str(options.artifact_root.resolve())
    try:
        return cli.main(forwarded)
    finally:
        cli._scorer_keywords = original_keywords
        cli.run = original_run
        if options.artifact_root is not None:
            if previous_artifact_root is None:
                os.environ.pop("CONTEXTWORLD_ARTIFACT_ROOT", None)
            else:
                os.environ["CONTEXTWORLD_ARTIFACT_ROOT"] = previous_artifact_root


if __name__ == "__main__":
    raise SystemExit(main())
