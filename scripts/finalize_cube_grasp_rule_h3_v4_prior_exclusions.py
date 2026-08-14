#!/usr/bin/env python3
"""Bind Cube v4 prior exclusions to the frozen v4 authorization.

The pre-freeze basis receipt records all v3 and coupling-design episodes.  A
final receipt additionally binds the current v4 preregistration and freeze
receipt and embeds the exact prior action/scene/pair/query content sets.  It
may also include completed v4 preformal smoke reports.  A formal v4 build must
use a newly finalized receipt that includes every such smoke, so their source
episodes and content cannot reappear in formal Training or Development.

Only explicitly supplied JSON and identity files are read.  No Lance table or
Public Test path is discovered or opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


V3_PROTOCOL = "cube_gripper_carry_rule_history3_development_v3"
V4_PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
BASIS_STATUS = "frozen_before_first_v4_data_build"
STATUS = "frozen_before_first_v4_build"
ACTIVE_SPLITS = ("train", "loader_validation")
CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excluded_source_episodes_sha256(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("episode IDs must be nonnegative, sorted, and unique")
    payload = b"".join(value.to_bytes(8, "little", signed=True) for value in normalized)
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def basis_episode_ids_sha256(values: Sequence[int]) -> str:
    """Hash the canonical newline-ASCII namespace used by the basis receipt."""

    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("basis episode IDs must be nonnegative, sorted, and unique")
    payload = ("\n".join(str(value) for value in normalized) + "\n").encode(
        "ascii"
    )
    return hashlib.sha256(payload).hexdigest()


def canonical_content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be sorted and unique")
    decoded: list[bytes] = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field_name} contains a non-SHA256 value")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def _read_json_nofollow(path: Path, *, label: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return raw, value


def _verified_file(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": logical_path if logical_path is not None else path.as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _reject_public_path(value: Path | str, *, label: str) -> None:
    forbidden = {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "publictest",
    }
    component = next(
        (part for part in Path(value).parts if part.lower() in forbidden), None
    )
    if component is not None:
        raise RuntimeError(f"{label} contains forbidden Public component {component!r}")


def _extract_build_evidence(
    path: Path, *, protocol: str, role: str, logical_path: str
) -> tuple[set[int], dict[str, set[str]], dict[str, Any]]:
    raw, document = _read_json_nofollow(path, label=role)
    if document.get("protocol") != protocol or document.get("passed") is not True:
        raise RuntimeError(f"{role} is not a passed {protocol} build report")
    if document.get("active_splits") != list(ACTIVE_SPLITS):
        raise RuntimeError(f"{role} active splits mismatch")
    if document.get("public_test_opened") is not False or document.get(
        "public_test_generated"
    ) is not False:
        raise RuntimeError(f"{role} did not keep Public Test closed")
    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(ACTIVE_SPLITS):
        raise RuntimeError(f"{role} split reports are malformed")
    episodes: set[int] = set()
    content = {field: set() for field in CONTENT_FIELDS}
    split_pairs: dict[str, int] = {}
    for split in ACTIVE_SPLITS:
        report = splits[split]
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            raise RuntimeError(f"{role}:{split} did not pass")
        pair_count = int(report.get("pair_count", -1))
        split_pairs[split] = pair_count
        source_values = [int(value) for value in report.get("source_episodes", [])]
        if pair_count <= 0 or len(source_values) != pair_count:
            raise RuntimeError(f"{role}:{split} source count mismatch")
        if episodes.intersection(source_values):
            raise RuntimeError(f"{role} reuses source episodes across splits")
        episodes.update(source_values)
        for field in CONTENT_FIELDS:
            report_field = "query_hashes" if field == "query_pixel_hashes" else field
            values = [str(value) for value in report.get(report_field, [])]
            if len(values) != pair_count or len(set(values)) != pair_count:
                raise RuntimeError(f"{role}:{split}:{field} count/uniqueness mismatch")
            if content[field].intersection(values):
                raise RuntimeError(f"{role} reuses {field} across splits")
            if any(SHA256_PATTERN.fullmatch(value) is None for value in values):
                raise RuntimeError(f"{role}:{split}:{field} contains invalid digest")
            content[field].update(values)
    receipt = {
        "role": role,
        "path": logical_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "protocol": protocol,
        "split_pair_counts": split_pairs,
        "source_episode_count": len(episodes),
        "public_test_opened_or_generated": False,
    }
    return episodes, content, receipt


def _extract_preformal_content_evidence(
    path: Path, *, logical_path: str
) -> tuple[set[int], dict[str, set[str]], dict[str, Any], Mapping[str, Any]]:
    raw, document = _read_json_nofollow(path, label="v4 preformal content receipt")
    if document.get("protocol_id") != V4_PROTOCOL or document.get(
        "status"
    ) != STATUS:
        raise RuntimeError("v4 preformal content receipt identity mismatch")
    if document.get("checks_passed") is not True:
        raise RuntimeError("v4 preformal content receipt did not pass")
    public = document.get("public_test")
    if not isinstance(public, Mapping) or (
        public.get("access_status") != "closed_not_read_not_scored"
        or any(
            public.get(name) is not False
            for name in ("opened", "read", "hashed", "scored")
        )
    ):
        raise RuntimeError("v4 preformal content receipt did not keep Public closed")
    if document.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("v4 preformal content receipt used a reference model")
    reconstruction = document.get("reconstruction_contract")
    expected_reconstruction = {
        "existing_pilot_replayed_not_reselected": True,
        "existing_real_mujoco_tests_replayed": True,
        "lance_opened_or_generated": False,
        "formal_build_attempted": False,
        "coupling_or_probe_recipe_changed": False,
    }
    if not isinstance(reconstruction, Mapping) or {
        key: reconstruction.get(key) for key in expected_reconstruction
    } != expected_reconstruction:
        raise RuntimeError("v4 preformal reconstruction contract mismatch")

    episodes = [int(value) for value in document.get("excluded_source_episodes", [])]
    if episodes != sorted(set(episodes)) or not episodes:
        raise RuntimeError("v4 preformal episode exclusion list is invalid")
    if int(document.get("excluded_source_episode_count", -1)) != len(episodes):
        raise RuntimeError("v4 preformal episode exclusion count mismatch")
    if document.get("excluded_source_episodes_sha256") != (
        excluded_source_episodes_sha256(episodes)
    ):
        raise RuntimeError("v4 preformal episode exclusion digest mismatch")

    raw_content = document.get("prior_content_exclusions")
    if not isinstance(raw_content, Mapping):
        raise RuntimeError("v4 preformal content receipt lacks content sets")
    content: dict[str, set[str]] = {}
    for field in CONTENT_FIELDS:
        entry = raw_content.get(field)
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"v4 preformal receipt lacks {field}")
        values = [str(value) for value in entry.get("values", [])]
        if not values or values != sorted(set(values)):
            raise RuntimeError(f"v4 preformal {field} is not sorted and unique")
        if int(entry.get("count", -1)) != len(values) or entry.get(
            "sha256"
        ) != canonical_content_digest(values, field_name=field):
            raise RuntimeError(f"v4 preformal {field} count/digest mismatch")
        content[field] = set(values)

    inputs = document.get("input_identities")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("v4 preformal receipt lacks input identities")
    artifact = {
        "role": "v4_preformal_smokes_and_pilots",
        "path": logical_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "source_episode_count": len(episodes),
        "content_counts": {field: len(content[field]) for field in CONTENT_FIELDS},
        "public_test_opened_or_generated": False,
        "formal_lance_build_attempted": False,
    }
    return set(episodes), content, artifact, inputs


def finalize(
    *,
    basis_receipt: Path,
    prereg_path: Path,
    freeze_receipt_path: Path,
    formal_v3_build_report: Path,
    formal_v3_logical_path: str,
    archived_v3_smoke_reports: Sequence[tuple[Path, str]],
    coupling_pilot: Path,
    coupling_pilot_logical_path: str,
    exploratory_diagnostic: Path,
    exploratory_diagnostic_logical_path: str,
    v4_preformal_content_receipt: Path,
    v4_preformal_content_logical_path: str,
    v4_preformal_build_reports: Sequence[tuple[Path, str]],
    output: Path,
) -> dict[str, Any]:
    path_inputs: list[tuple[str, Path | str]] = [
        ("basis receipt", basis_receipt),
        ("preregistration", prereg_path),
        ("freeze receipt", freeze_receipt_path),
        ("formal v3 report", formal_v3_build_report),
        ("formal v3 logical path", formal_v3_logical_path),
        ("coupling pilot", coupling_pilot),
        ("coupling pilot logical path", coupling_pilot_logical_path),
        ("exploratory diagnostic", exploratory_diagnostic),
        ("exploratory diagnostic logical path", exploratory_diagnostic_logical_path),
        ("v4 preformal content receipt", v4_preformal_content_receipt),
        ("v4 preformal content logical path", v4_preformal_content_logical_path),
        ("output", output),
    ]
    path_inputs.extend(
        ("archived v3 smoke", value)
        for pair in archived_v3_smoke_reports
        for value in pair
    )
    path_inputs.extend(
        ("v4 preformal build report", value)
        for pair in v4_preformal_build_reports
        for value in pair
    )
    for label, value in path_inputs:
        _reject_public_path(value, label=label)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    basis_raw, basis = _read_json_nofollow(
        basis_receipt, label="episode exclusion basis"
    )
    if (
        basis.get("protocol_id") != V4_PROTOCOL
        or basis.get("status") != BASIS_STATUS
    ):
        raise RuntimeError("Episode exclusion basis identity mismatch")
    if basis.get("checks_passed") is not True:
        raise RuntimeError("Episode exclusion basis did not pass")
    basis_episodes = [int(value) for value in basis.get("excluded_source_episodes", [])]
    if basis_episodes != sorted(set(basis_episodes)) or not basis_episodes:
        raise RuntimeError("Episode exclusion basis list is invalid")
    if int(basis.get("excluded_source_episode_count", -1)) != len(basis_episodes):
        raise RuntimeError("Episode exclusion basis count mismatch")
    basis_episode_digest = basis_episode_ids_sha256(basis_episodes)
    if basis.get("excluded_source_episode_ids_sha256") != basis_episode_digest:
        raise RuntimeError("Episode exclusion basis digest mismatch")
    basis_identity = {
        "sha256": hashlib.sha256(basis_raw).hexdigest(),
        "size_bytes": len(basis_raw),
    }

    prereg = _verified_file(prereg_path, logical_path=(
        "configs/benchmark/cube_gripper_carry_h3_development_prereg_v4.yaml"
    ))
    freeze_raw, freeze_receipt = _read_json_nofollow(
        freeze_receipt_path, label="v4 freeze receipt"
    )
    if freeze_receipt.get("protocol_id") != V4_PROTOCOL:
        raise RuntimeError("v4 freeze receipt protocol mismatch")
    if freeze_receipt.get("status") != STATUS or freeze_receipt.get(
        "checks_passed"
    ) is not True:
        raise RuntimeError("v4 freeze receipt does not authorize a build")
    frozen_prereg = freeze_receipt.get("preregistration")
    if not isinstance(frozen_prereg, Mapping) or frozen_prereg.get(
        "sha256"
    ) != prereg["sha256"]:
        raise RuntimeError("v4 freeze receipt does not bind current prereg")
    frozen_basis = freeze_receipt.get("prior_episode_exclusion_basis")
    if not isinstance(frozen_basis, Mapping):
        raise RuntimeError("v4 freeze receipt does not bind the episode basis")
    expected_frozen_basis = {
        "sha256": basis_identity["sha256"],
        "size_bytes": basis_identity["size_bytes"],
        "excluded_source_episode_count": len(basis_episodes),
        "excluded_source_episode_ids_sha256": basis_episode_digest,
    }
    if {
        key: frozen_basis.get(key) for key in expected_frozen_basis
    } != expected_frozen_basis:
        raise RuntimeError("v4 freeze receipt episode-basis binding mismatch")
    freeze_binding = {
        "path": (
            "artifacts/evaluation/history3/"
            "cube_gripper_carry_h3_development_v4/"
            + freeze_receipt_path.name
        ),
        "sha256": hashlib.sha256(freeze_raw).hexdigest(),
        "size_bytes": len(freeze_raw),
    }

    formal_episodes, content, formal_artifact = _extract_build_evidence(
        formal_v3_build_report,
        protocol=V3_PROTOCOL,
        role="v3_formal",
        logical_path=formal_v3_logical_path,
    )
    if not formal_episodes.issubset(basis_episodes):
        raise RuntimeError("Basis does not cover formal v3 episodes")
    basis_inputs = basis.get("inputs")
    if not isinstance(basis_inputs, Mapping):
        raise RuntimeError("Episode exclusion basis lacks input bindings")
    basis_formal = basis_inputs.get("formal_v3_build_report")
    if not isinstance(basis_formal, Mapping) or basis_formal.get(
        "sha256"
    ) != formal_artifact["sha256"]:
        raise RuntimeError("Formal v3 report differs from the frozen basis input")

    input_artifacts: list[dict[str, Any]] = [formal_artifact]
    smoke_artifacts: list[dict[str, Any]] = []
    for path, logical_path in archived_v3_smoke_reports:
        smoke_episodes, smoke_content, artifact = _extract_build_evidence(
            path,
            protocol=V3_PROTOCOL,
            role="v3_smokes",
            logical_path=logical_path,
        )
        if not smoke_episodes.issubset(formal_episodes):
            raise RuntimeError("A v3 smoke is not a subset of formal v3 episodes")
        # A smoke can use an episode that later appears in formal v3 while
        # still assigning a different simulator seed, task, target, color, or
        # action profile because its candidate-pool split point differs.  The
        # episode basis already covers the source identity, but every distinct
        # smoke content hash must also be excluded explicitly.
        for field in CONTENT_FIELDS:
            content[field].update(smoke_content[field])
        input_artifacts.append(artifact)
        smoke_artifacts.append(artifact)

    basis_smokes = basis_inputs.get("v3_smoke_build_reports")
    if not isinstance(basis_smokes, list) or sorted(
        str(value.get("sha256", ""))
        for value in basis_smokes
        if isinstance(value, Mapping)
    ) != sorted(value["sha256"] for value in smoke_artifacts):
        raise RuntimeError("Archived v3 smokes differ from the frozen basis inputs")

    pilot_binding = _verified_file(
        coupling_pilot, logical_path=coupling_pilot_logical_path
    )
    pilot_binding["role"] = "v4_preformal_smokes_and_pilots"
    input_artifacts.append(pilot_binding)
    diagnostic_binding = _verified_file(
        exploratory_diagnostic, logical_path=exploratory_diagnostic_logical_path
    )
    diagnostic_binding["role"] = "v3_pilots"
    input_artifacts.append(diagnostic_binding)
    basis_pilot = basis_inputs.get("v4_coupling_design_pilot")
    basis_diagnostic = basis_inputs.get("v3_failure_exploratory_diagnostic")
    if not isinstance(basis_pilot, Mapping) or basis_pilot.get(
        "sha256"
    ) != pilot_binding["sha256"]:
        raise RuntimeError("Coupling pilot differs from the frozen basis input")
    if not isinstance(basis_diagnostic, Mapping) or basis_diagnostic.get(
        "sha256"
    ) != diagnostic_binding["sha256"]:
        raise RuntimeError("Exploratory diagnostic differs from the frozen basis input")
    frozen_evidence = freeze_receipt.get("frozen_evidence")
    if not isinstance(frozen_evidence, Mapping):
        raise RuntimeError("v4 freeze receipt lacks frozen evidence bindings")
    if frozen_evidence.get("coupling_pilot", {}).get("sha256") != pilot_binding[
        "sha256"
    ] or frozen_evidence.get("exploratory_diagnostic", {}).get(
        "sha256"
    ) != diagnostic_binding["sha256"]:
        raise RuntimeError("Finalizer evidence differs from the v4 freeze receipt")

    (
        preformal_episodes,
        preformal_content,
        preformal_artifact,
        preformal_inputs,
    ) = _extract_preformal_content_evidence(
        v4_preformal_content_receipt,
        logical_path=v4_preformal_content_logical_path,
    )
    preformal_pilot = preformal_inputs.get("coupling_pilot")
    preformal_formal_v3 = preformal_inputs.get("formal_v3_build_report")
    if not isinstance(preformal_pilot, Mapping) or preformal_pilot.get(
        "sha256"
    ) != pilot_binding["sha256"]:
        raise RuntimeError("preformal receipt coupling-pilot binding mismatch")
    if not isinstance(preformal_formal_v3, Mapping) or preformal_formal_v3.get(
        "sha256"
    ) != formal_artifact["sha256"]:
        raise RuntimeError("preformal receipt formal-v3 binding mismatch")
    frozen_preformal = frozen_evidence.get("v4_preformal_content_receipt")
    if not isinstance(frozen_preformal, Mapping) or frozen_preformal.get(
        "sha256"
    ) != preformal_artifact["sha256"]:
        raise RuntimeError("preformal receipt differs from the v4 freeze receipt")
    input_artifacts.append(preformal_artifact)
    for field in CONTENT_FIELDS:
        content[field].update(preformal_content[field])

    excluded_episodes = set(basis_episodes) | preformal_episodes
    for path, logical_path in v4_preformal_build_reports:
        episodes, smoke_content, artifact = _extract_build_evidence(
            path,
            protocol=V4_PROTOCOL,
            role="v4_preformal_smokes_and_pilots",
            logical_path=logical_path,
        )
        excluded_episodes.update(episodes)
        for field in CONTENT_FIELDS:
            content[field].update(smoke_content[field])
        input_artifacts.append(artifact)

    episodes_sorted = sorted(excluded_episodes)
    content_receipt = {}
    for field in CONTENT_FIELDS:
        values = sorted(content[field])
        content_receipt[field] = {
            "values": values,
            "count": len(values),
            "sha256": canonical_content_digest(values, field_name=field),
        }

    source_h5 = freeze_receipt.get("source_h5")
    if not isinstance(source_h5, Mapping):
        raise RuntimeError("v4 freeze receipt lacks source identity")
    source_receipt = {
        key: source_h5.get(key)
        for key in ("symbol", "sha256", "size_bytes", "row_count", "episode_count")
    }
    preformal_source = preformal_inputs.get("source_h5")
    if not isinstance(preformal_source, Mapping) or {
        key: preformal_source.get(key) for key in ("sha256", "size_bytes")
    } != {
        "sha256": source_receipt["sha256"],
        "size_bytes": source_receipt["size_bytes"],
    }:
        raise RuntimeError("preformal receipt source-H5 binding mismatch")
    if episodes_sorted[-1] >= int(source_receipt["episode_count"]):
        raise RuntimeError("An excluded episode is outside the source H5")

    receipt = {
        "schema_version": 1,
        "protocol_id": V4_PROTOCOL,
        "receipt_id": "cube_gripper_carry_h3_v4_prior_exclusions_final_v1",
        "status": STATUS,
        "checks_passed": True,
        "preregistration": prereg,
        "freeze_receipt": freeze_binding,
        "source_h5": source_receipt,
        "basis_receipt": {
            "path": (
                "artifacts/evaluation/history3/"
                "cube_gripper_carry_h3_development_v4/"
                + basis_receipt.name
            ),
            **basis_identity,
            "basis_episode_count": len(basis_episodes),
            "basis_episode_ids_sha256": basis_episode_digest,
        },
        "coverage": {
            "v3_formal": True,
            "v3_smokes": True,
            "v3_pilots": True,
            "v4_preformal_smokes_and_pilots": True,
        },
        "input_artifacts": input_artifacts,
        "excluded_source_episodes": episodes_sorted,
        "excluded_source_episode_count": len(episodes_sorted),
        "excluded_source_episodes_sha256": excluded_source_episodes_sha256(
            episodes_sorted
        ),
        "prior_content_exclusions": content_receipt,
        "v4_preformal_build_report_count": len(v4_preformal_build_reports),
        "v4_preformal_content_receipt": preformal_artifact,
        "formal_build_requirement": (
            "finalize a new receipt after every v4 preformal smoke; formal "
            "Training/Development must use the newest receipt"
        ),
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def _paired_paths(values: Sequence[Sequence[str]] | None) -> list[tuple[Path, str]]:
    return [(Path(value[0]), value[1]) for value in (values or [])]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis-receipt", type=Path, required=True)
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--formal-v3-build-report", type=Path, required=True)
    parser.add_argument("--formal-v3-logical-path", required=True)
    parser.add_argument(
        "--archived-v3-smoke", nargs=2, action="append", metavar=("PATH", "LOGICAL")
    )
    parser.add_argument("--coupling-pilot", type=Path, required=True)
    parser.add_argument("--coupling-pilot-logical-path", required=True)
    parser.add_argument("--exploratory-diagnostic", type=Path, required=True)
    parser.add_argument("--exploratory-diagnostic-logical-path", required=True)
    parser.add_argument("--v4-preformal-content-receipt", type=Path, required=True)
    parser.add_argument("--v4-preformal-content-logical-path", required=True)
    parser.add_argument(
        "--v4-preformal-build-report",
        nargs=2,
        action="append",
        metavar=("PATH", "LOGICAL"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = finalize(
        basis_receipt=args.basis_receipt.expanduser().resolve(),
        prereg_path=args.prereg.expanduser().resolve(),
        freeze_receipt_path=args.freeze_receipt.expanduser().resolve(),
        formal_v3_build_report=args.formal_v3_build_report.expanduser().resolve(),
        formal_v3_logical_path=args.formal_v3_logical_path,
        archived_v3_smoke_reports=[
            (path.expanduser().resolve(), logical)
            for path, logical in _paired_paths(args.archived_v3_smoke)
        ],
        coupling_pilot=args.coupling_pilot.expanduser().resolve(),
        coupling_pilot_logical_path=args.coupling_pilot_logical_path,
        exploratory_diagnostic=args.exploratory_diagnostic.expanduser().resolve(),
        exploratory_diagnostic_logical_path=args.exploratory_diagnostic_logical_path,
        v4_preformal_content_receipt=(
            args.v4_preformal_content_receipt.expanduser().resolve()
        ),
        v4_preformal_content_logical_path=(
            args.v4_preformal_content_logical_path
        ),
        v4_preformal_build_reports=[
            (path.expanduser().resolve(), logical)
            for path, logical in _paired_paths(args.v4_preformal_build_report)
        ],
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "checks_passed": receipt["checks_passed"],
                "excluded_source_episode_count": receipt[
                    "excluded_source_episode_count"
                ],
                "excluded_source_episodes_sha256": receipt[
                    "excluded_source_episodes_sha256"
                ],
                "v4_preformal_build_report_count": receipt[
                    "v4_preformal_build_report_count"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
