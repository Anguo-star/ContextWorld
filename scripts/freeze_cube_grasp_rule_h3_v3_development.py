#!/usr/bin/env python3
"""Verify and freeze the Development-only Cube History-3 v3 preregistration.

This command reads only the preregistration, its declared implementation
files, the frozen feasibility input, and the explicitly supplied upstream H5
file.  It never opens a Lance table and therefore cannot inspect Development
or Public Test examples.  The output receipt lives outside the preregistration
so the preregistration can record stable dependency hashes without hashing
itself recursively.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / (
    "configs/benchmark/cube_gripper_carry_h3_development_prereg_v3.yaml"
)
PLACEHOLDER = "TO_BE_FROZEN_BEFORE_FIRST_V3_DATA_BUILD"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_placeholder(child) for child in value)
    return isinstance(value, str) and PLACEHOLDER in value


def _resolve_declared_path(value: str, *, artifact_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "artifacts":
        bundled = (ROOT / path).resolve()
        if bundled.exists():
            return bundled
        return artifact_root.joinpath(*path.parts[1:]).resolve()
    return (ROOT / path).resolve()


def _verify_identity_entries(
    entries: Mapping[str, Any], *, artifact_root: Path
) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, entry in entries.items():
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            continue
        path = _resolve_declared_path(str(entry["path"]), artifact_root=artifact_root)
        if not path.is_file():
            raise FileNotFoundError(f"{name}: missing declared file {path}")
        actual = file_sha256(path)
        expected = str(entry["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"{name}: sha256 mismatch for {path}: {actual} != {expected}"
            )
        observed[name] = {
            "path": str(entry["path"]),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
        }
    return observed


def freeze(
    *,
    prereg_path: Path,
    artifact_root: Path,
    source_h5: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt {output}")
    document = yaml.safe_load(prereg_path.read_text(encoding="utf-8"))
    if _contains_placeholder(document):
        raise RuntimeError("Preregistration still contains an identity placeholder")
    if document.get("phase") != "development_only":
        raise RuntimeError("Cube v3 freeze requires phase=development_only")
    if document.get("protocol_id") != (
        "cube_gripper_carry_rule_history3_development_v3"
    ):
        raise RuntimeError("Unexpected Cube v3 protocol_id")
    public = document.get("public_test", {})
    if public.get("access_status") != "closed_not_read_not_scored":
        raise RuntimeError("Public Test is not declared closed")
    if public.get("validation_lance_access_allowed") is not False:
        raise RuntimeError("Public validation access must be false")
    reference = document.get("reference_model_phase", {})
    if reference.get("training_and_scoring_authorized") is not False:
        raise RuntimeError("This freeze must not authorize reference training")

    identity = _verify_identity_entries(
        document.get("identity", {}), artifact_root=artifact_root
    )
    frozen_v2 = _verify_identity_entries(
        document.get("frozen_v2_baseline", {}), artifact_root=artifact_root
    )
    feasibility = document["action_support"]["feasibility_evidence"]
    feasibility_path = _resolve_declared_path(
        str(feasibility["report"]), artifact_root=artifact_root
    )
    feasibility_hash = file_sha256(feasibility_path)
    if feasibility_hash != str(feasibility["sha256"]):
        raise RuntimeError("Frozen action-template feasibility hash mismatch")
    narrative_path = _resolve_declared_path(
        str(feasibility["narrative_report"]), artifact_root=artifact_root
    )
    narrative_hash = file_sha256(narrative_path)
    if narrative_hash != str(feasibility["narrative_sha256"]):
        raise RuntimeError("Frozen action-template narrative hash mismatch")

    source = document["source_and_catalog"]["frozen_source_identity"]
    source_stat = source_h5.stat()
    if source_stat.st_size != int(source["size_bytes"]):
        raise RuntimeError("Upstream Cube H5 size does not match preregistration")
    with h5py.File(source_h5, "r", swmr=True) as handle:
        source_rows = int(handle["action"].shape[0])
        source_episodes = int(handle["ep_len"].shape[0])
    if source_rows != int(source["row_count"]):
        raise RuntimeError("Upstream Cube H5 row count mismatch")
    if source_episodes != int(source["episode_count"]):
        raise RuntimeError("Upstream Cube H5 episode count mismatch")
    source_hash = file_sha256(source_h5)
    if source_hash != str(source["sha256"]):
        raise RuntimeError("Upstream Cube H5 content hash mismatch")

    receipt = {
        "schema_version": 1,
        "protocol_id": document["protocol_id"],
        "status": "frozen_before_first_v3_data_build",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Training_and_Development_data_and_rgb_probe_only",
        "contextworld_git_head": document["identity"][
            "contextworld_git_head_at_prereg_draft"
        ],
        "preregistration": {
            "path": str(prereg_path.relative_to(ROOT)),
            "sha256": file_sha256(prereg_path),
            "size_bytes": prereg_path.stat().st_size,
        },
        "identity": identity,
        "frozen_v2": frozen_v2,
        "source_h5": {
            "symbol": document["source_and_catalog"]["source_symbol"],
            "path_recorded": False,
            "sha256": source_hash,
            "size_bytes": source_stat.st_size,
            "row_count": source_rows,
            "episode_count": source_episodes,
            "action_dataset": source["action_dataset"],
        },
        "feasibility_input": {
            "path": feasibility["report"],
            "sha256": feasibility_hash,
            "size_bytes": feasibility_path.stat().st_size,
            "narrative_path": feasibility["narrative_report"],
            "narrative_sha256": narrative_hash,
            "narrative_size_bytes": narrative_path.stat().st_size,
        },
        "authorized_splits": ["train", "loader_validation"],
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "scored": False,
            "hashed": False,
        },
        "reference_model_training_or_scoring_authorized": False,
        "checks_passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        prereg_path=args.prereg.expanduser().resolve(),
        artifact_root=args.artifact_root.expanduser().resolve(),
        source_h5=args.source_h5.expanduser().resolve(),
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "preregistration_sha256": receipt["preregistration"]["sha256"],
                "checks_passed": receipt["checks_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
