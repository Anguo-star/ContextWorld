#!/usr/bin/env python3
"""Build the self-contained public Action Delay artifact shadow.

The historical experiment tree contains several generations of diagnostics and
machine-specific absolute paths.  This command copies only the frozen training
payload, the 300-query Public Test, and the current six-checkpoint score
receipts into ``ContextWorld/artifacts``.  Binary payload is copied byte for
byte; JSON metadata is rewritten to stable logical paths.

The command is fail-closed and refuses to replace an existing release tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
from typing import Any

from contextworld.evaluation.action_delay_h7_data import directory_sha256
from contextworld.paths import artifact_root, repository_root


TRAINING_ROOTS = (
    "synthesis/action_delay_h7_paired_v1",
    "synthesis/action_delay_h7_core_training_v3",
)
PUBLIC_TEST_ROOT = "evaluation/history7/action_delay_validation_v1"
CURRENT_SCORE_ROOT = "evaluation/history7/action_delay_curriculum_v4"
CURRENT_CEM_ROOT = (
    "evaluation/history7/"
    "action_delay_curriculum_v4_ability_retention_v1"
)
PUBLIC_TEST_METADATA = (
    "catalog.json",
    "build_report.json",
    "audit_report.json",
    "training_exclusion_manifest.json",
)
FORBIDDEN_PUBLIC_TEST_PARTS = {
    "latent_diagnostics",
    "logs",
    "scoring_matrix_report",
    "comparison_summary.json",
}
MODEL_RESULT_GLOB = "h7_action_delay_curriculum_v4_*_formal_s*_validation.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def tree_identity(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    return {
        "root": path.as_posix(),
        "files": len(files),
        "bytes": sum(value.stat().st_size for value in files),
        "sha256": directory_sha256(path),
    }


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _under(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _portable_absolute_path(
    value: str,
    *,
    source_artifacts: Path,
    repo_root: Path,
) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Unsupported Windows absolute path: {value}")
    normalized = Path(os.path.abspath(path.expanduser()))

    relative = _under(normalized, source_artifacts)
    if relative is not None:
        return (Path("artifacts") / relative).as_posix()

    relative = _under(normalized, repo_root)
    if relative is not None:
        return relative.as_posix()

    stable_worldmodel = repo_root.parent / "stable-worldmodel"
    relative = _under(normalized, stable_worldmodel)
    if relative is not None:
        base = Path("../stable-worldmodel")
        return (base / relative).as_posix() if relative.parts else base.as_posix()

    temporary_stable_worldmodel = Path("/tmp/stable-worldmodel-ad2")
    relative = _under(normalized, temporary_stable_worldmodel)
    if relative is not None:
        base = Path("../stable-worldmodel")
        return (base / relative).as_posix() if relative.parts else base.as_posix()

    upstream_h5 = Path(
        "/opt/huawei/explorer-env/dataset/ag_data/data/world_model/"
        "quentinll/lewm-tworooms/tworoom.h5"
    )
    if normalized == upstream_h5:
        return "upstream/lewm-tworooms/tworoom.h5"

    raise ValueError(f"Unknown absolute metadata path: {value}")


def portable_metadata(
    value: Any,
    *,
    source_artifacts: Path,
    repo_root: Path,
) -> tuple[Any, int]:
    replacements = 0

    def rewrite(item: Any) -> Any:
        nonlocal replacements
        if isinstance(item, dict):
            return {key: rewrite(child) for key, child in item.items()}
        if isinstance(item, list):
            return [rewrite(child) for child in item]
        if item == "../../data/world_model/quentinll/lewm-tworooms/tworoom.h5":
            replacements += 1
            return "upstream/lewm-tworooms/tworoom.h5"
        if isinstance(item, str) and _is_absolute(item):
            replacements += 1
            return _portable_absolute_path(
                item,
                source_artifacts=source_artifacts,
                repo_root=repo_root,
            )
        return item

    return rewrite(value), replacements


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sanitize_json_file(
    path: Path,
    *,
    source_artifacts: Path,
    repo_root: Path,
) -> int:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        portable_rows = []
        replacements = 0
        for row in rows:
            portable, changed = portable_metadata(
                row,
                source_artifacts=source_artifacts,
                repo_root=repo_root,
            )
            portable_rows.append(portable)
            replacements += changed
        if replacements:
            path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in portable_rows
                ),
                encoding="utf-8",
            )
        return replacements

    payload = json.loads(path.read_text(encoding="utf-8"))
    portable, replacements = portable_metadata(
        payload,
        source_artifacts=source_artifacts,
        repo_root=repo_root,
    )
    if replacements:
        _write_json(path, portable)
    return replacements


def _sanitize_training_root(
    root: Path,
    *,
    source_artifacts: Path,
    repo_root: Path,
) -> int:
    replacements = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            replacements += sanitize_json_file(
                path,
                source_artifacts=source_artifacts,
                repo_root=repo_root,
            )

    build_path = root / "build_report.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    catalog_path = root / Path(build["artifacts"]["catalog"]).name
    if not catalog_path.is_file():
        catalog_path = root / "catalogs" / Path(
            build["artifacts"]["catalog"]
        ).name
    report_path = root / Path(build["artifacts"]["synthesis_report"]).name
    if not report_path.is_file():
        report_path = root / "reports" / Path(
            build["artifacts"]["synthesis_report"]
        ).name
    build["artifacts"]["catalog_sha256"] = file_sha256(catalog_path)
    build["artifacts"]["synthesis_report_sha256"] = file_sha256(report_path)
    _write_json(build_path, build)
    return replacements


def _copy_public_test(
    *,
    source_artifacts: Path,
    destination: Path,
    repo_root: Path,
) -> dict[str, Any]:
    source = source_artifacts / PUBLIC_TEST_ROOT
    destination.mkdir(parents=True)
    assets = sorted((source / "assets").glob("*.npz"))
    if len(assets) != 300:
        raise RuntimeError(f"Expected 300 Action Delay assets, found {len(assets)}")
    (destination / "assets").mkdir()
    for asset in assets:
        shutil.copy2(asset, destination / "assets" / asset.name)

    replacements = 0
    for name in PUBLIC_TEST_METADATA:
        shutil.copy2(source / name, destination / name)
        replacements += sanitize_json_file(
            destination / name,
            source_artifacts=source_artifacts,
            repo_root=repo_root,
        )

    score_root = source_artifacts / CURRENT_SCORE_ROOT
    receipt_root = destination / "score_receipts"
    result_destination = receipt_root / "model_results"
    result_destination.mkdir(parents=True)
    model_results = sorted((score_root / "model_results").glob(MODEL_RESULT_GLOB))
    if len(model_results) != 6:
        raise RuntimeError(
            f"Expected six current Action Delay model results, found {len(model_results)}"
        )
    core_source = score_root / "core_summary.json"
    core = json.loads(core_source.read_text(encoding="utf-8"))
    # Old three-delay experiments were development diagnostics, not part of
    # the frozen reference result.  Keep the public receipt current-only.
    core.pop("historical_artifacts", None)
    core.pop("reference_comparison", None)
    core["benchmark"] = "contextworld_tworoom_action_delay_icl_history7_v1"
    core["receipt_kind"] = "public_test_core_score_summary"
    core["identity"]["source_core_summary_sha256"] = file_sha256(core_source)
    for family in core.get("by_family", {}).values():
        family.pop("diagnostic_only", None)
    core_models = {
        str(model["label"]): model for model in core.get("models", [])
    }
    for model in core_models.values():
        model.pop("diagnostic_only", None)

    source_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    for source_result in model_results:
        payload = json.loads(source_result.read_text(encoding="utf-8"))
        label = str(payload.get("label"))
        if label in source_payloads or label not in core_models:
            raise RuntimeError(f"Unexpected source score label: {label}")
        source_payloads[label] = (source_result, payload)
    if set(source_payloads) != set(core_models):
        raise RuntimeError("Core summary and source result labels differ")

    result_hashes: dict[str, str] = {}
    compatibility_entries: dict[str, Any] = {}
    label_to_filename: dict[str, str] = {}
    for label in sorted(source_payloads):
        source_result, payload = source_payloads[label]
        core_model = core_models[label]
        family = str(payload["model_family"])
        seed = int(core_model["training_seed"])
        expected_label = (
            f"h7_action_delay_curriculum_v4_{family}_formal_s{seed}"
        )
        source_artifact = core["artifacts"][label]
        source_sha = file_sha256(source_result)
        checks = {
            "label_exact": label == expected_label,
            "model_family_exact": family == core_model["model_family"],
            "source_result_sha256_exact": (
                source_sha == source_artifact["validation_result_sha256"]
            ),
            "checkpoint_sha256_exact": (
                payload["identity"]["checkpoint_sha256"]
                == source_artifact["checkpoint_sha256"]
            ),
            "training_receipt_passed": (
                payload.get("training_receipt", {}).get("passed") is True
            ),
            "score_audit_passed": (
                payload.get("score_audit", {}).get("passed") is True
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Compact score compatibility failed: {label}")
        primary_score = core_model["core_h1"]
        filename = f"{label}_public_test_score.json"
        label_to_filename[label] = filename
        compact = {
            "schema_version": 1,
            "benchmark": "contextworld_tworoom_action_delay_icl_history7_v1",
            "receipt_kind": "public_test_model_score",
            "status": "completed",
            "label": label,
            "model_family": family,
            "training_seed": seed,
            "reference_role": (
                "positive_reference" if family == "pldm" else "model_control"
            ),
            "model": {
                "adapter_id": payload["model"]["adapter_id"],
                "adapter_class": payload["model"]["adapter_class"],
                "model_class": payload["model"]["model_class"],
                "parameters": payload["model"]["parameters"],
                "checkpoint": payload["identity"]["checkpoint"],
                "checkpoint_sha256": payload["identity"]["checkpoint_sha256"],
                "checkpoint_config": payload["identity"]["checkpoint_config"],
                "checkpoint_config_sha256": payload["identity"]
                ["checkpoint_config_sha256"],
                "model_state_sha256_before": payload[
                    "model_state_sha256_before"
                ],
                "model_state_sha256_after": payload[
                    "model_state_sha256_after"
                ],
                "stable_worldmodel_commit": payload["model"]
                ["stable_worldmodel_commit"],
            },
            "public_test": {
                "catalog": payload["identity"]["frozen_catalog"],
                "catalog_sha256": payload["identity"]
                ["frozen_catalog_sha256"],
                "content_manifest_sha256": payload["identity"]
                ["catalog_content_manifest_sha256"],
                "normalizer": payload["identity"]["normalizer"],
                "normalizer_sha256": payload["identity"]["normalizer_sha256"],
                "score_audit": {
                    "queries": payload["score_audit"]["queries"],
                    "model_visible_fields": payload["score_audit"]
                    ["model_visible_fields"],
                    "online_environment_calls": payload["score_audit"]
                    ["online_environment_calls"],
                    "privileged_fields_passed_to_adapter": payload[
                        "score_audit"
                    ]["privileged_fields_passed_to_adapter"],
                    "checks": {
                        key: payload["score_audit"]["checks"][key]
                        for key in (
                            "queries_exact",
                            "no_privileged_fields",
                            "offline_only",
                        )
                    },
                    "passed": payload["score_audit"]["passed"],
                },
            },
            "training_receipt": payload["training_receipt"],
            "primary_score": primary_score,
            "gate": core_model["gate"],
            "source_evidence": {
                "archived_result_sha256": source_sha,
                "primary_score_sha256": canonical_json_sha256(primary_score),
                "compatibility_checks": checks,
            },
            "claim_boundary": {
                "uses_frozen_public_test": True,
                "online_environment_calls": 0,
                "hidden_test_used": False,
                "primary_horizon_action_blocks": 1,
            },
        }
        compact, changed = portable_metadata(
            compact,
            source_artifacts=source_artifacts,
            repo_root=repo_root,
        )
        replacements += changed
        target = result_destination / filename
        _write_json(target, compact)
        compact_sha = file_sha256(target)
        result_hashes[filename] = compact_sha
        compatibility_entries[label] = {
            "archived_result_sha256": source_sha,
            "public_score": (
                Path("artifacts")
                / PUBLIC_TEST_ROOT
                / "score_receipts/model_results"
                / filename
            ).as_posix(),
            "public_score_sha256": compact_sha,
            "checkpoint_sha256": payload["identity"]["checkpoint_sha256"],
            "primary_score_sha256": canonical_json_sha256(primary_score),
            "checks": checks,
        }

    compatibility = {
        "schema_version": 1,
        "benchmark": "contextworld_tworoom_action_delay_icl_history7_v1",
        "receipt_kind": "source_to_public_score_compatibility",
        "status": "passed",
        "archived_source_results_distributed": False,
        "entries": compatibility_entries,
        "checks": {
            "six_current_models_exact": len(compatibility_entries) == 6,
            "all_source_identities_match": all(
                all(entry["checks"].values())
                for entry in compatibility_entries.values()
            ),
            "public_scores_are_compact": True,
            "development_only_fields_excluded": True,
        },
    }
    compatibility_target = receipt_root / "source_compatibility.json"
    _write_json(compatibility_target, compatibility)

    core, changed = portable_metadata(
        core,
        source_artifacts=source_artifacts,
        repo_root=repo_root,
    )
    replacements += changed
    logical_result_root = (
        Path("artifacts") / PUBLIC_TEST_ROOT / "score_receipts/model_results"
    )
    for label, specification in core["artifacts"].items():
        filename = label_to_filename[label]
        specification.pop("validation_result", None)
        specification.pop("validation_result_sha256", None)
        specification["public_test_score"] = (
            logical_result_root / filename
        ).as_posix()
        specification["public_test_score_sha256"] = result_hashes[filename]
    core["identity"]["source_compatibility"] = (
        Path("artifacts")
        / PUBLIC_TEST_ROOT
        / "score_receipts/source_compatibility.json"
    ).as_posix()
    core["identity"]["source_compatibility_sha256"] = file_sha256(
        compatibility_target
    )
    core_target = receipt_root / "core_summary.json"
    _write_json(core_target, core)

    cem_source = source_artifacts / CURRENT_CEM_ROOT
    cem_target = receipt_root / "cem_retention"
    cem_target.mkdir()
    runner_source = cem_source / "runner_report.json"
    runner = json.loads(runner_source.read_text(encoding="utf-8"))
    runner, changed = portable_metadata(
        runner,
        source_artifacts=source_artifacts,
        repo_root=repo_root,
    )
    replacements += changed
    runner_target = cem_target / "runner_report.json"
    _write_json(runner_target, runner)

    final_source = cem_source / "final_summary.json"
    final = json.loads(final_source.read_text(encoding="utf-8"))
    final, changed = portable_metadata(
        final,
        source_artifacts=source_artifacts,
        repo_root=repo_root,
    )
    replacements += changed
    runner_identity = final.get("identity", {}).get("runner_report")
    if isinstance(runner_identity, dict):
        runner_identity["path"] = (
            Path("artifacts")
            / PUBLIC_TEST_ROOT
            / "score_receipts/cem_retention/runner_report.json"
        ).as_posix()
        runner_identity["sha256"] = file_sha256(runner_target)
    final_target = cem_target / "final_summary.json"
    _write_json(final_target, final)

    actual_parts = {
        part
        for path in destination.rglob("*")
        for part in path.relative_to(destination).parts
    }
    forbidden = sorted(actual_parts & FORBIDDEN_PUBLIC_TEST_PARTS)
    if forbidden:
        raise RuntimeError(f"Forbidden Public Test content copied: {forbidden}")
    expected_receipts = {
        "score_receipts/core_summary.json",
        "score_receipts/source_compatibility.json",
        "score_receipts/cem_retention/final_summary.json",
        "score_receipts/cem_retention/runner_report.json",
        *(
            f"score_receipts/model_results/{name}"
            for name in result_hashes
        ),
    }
    observed_receipts = {
        path.relative_to(destination).as_posix()
        for path in receipt_root.rglob("*")
        if path.is_file()
    }
    if observed_receipts != expected_receipts:
        raise RuntimeError("Action Delay score receipt layout is not exact")
    return {
        "metadata_path_replacements": replacements,
        "model_result_sha256": result_hashes,
        "source_compatibility_sha256": file_sha256(compatibility_target),
        "core_summary_sha256": file_sha256(core_target),
        "cem_final_summary_sha256": file_sha256(final_target),
        "cem_runner_report_sha256": file_sha256(runner_target),
    }


def _assert_no_absolute_json_paths(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        values = (
            [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if path.suffix == ".jsonl"
            else [json.loads(path.read_text(encoding="utf-8"))]
        )

        def inspect(item: Any, location: str) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    inspect(child, f"{location}.{key}")
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    inspect(child, f"{location}[{index}]")
            elif isinstance(item, str) and _is_absolute(item):
                raise RuntimeError(
                    f"Non-portable path in {path} at {location}: {item}"
                )

        for index, value in enumerate(values):
            inspect(value, f"$[{index}]")


def build_release(*, repo_root: Path, source_artifacts: Path) -> dict[str, Any]:
    repo_root = repo_root.expanduser().resolve()
    source_artifacts = source_artifacts.expanduser().resolve()
    targets = [
        *(repo_root / "artifacts" / relative for relative in TRAINING_ROOTS),
        repo_root / "artifacts" / PUBLIC_TEST_ROOT,
    ]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to replace Action Delay release trees: "
            + ", ".join(str(path) for path in existing)
        )
    destinations = targets
    for target in destinations:
        target.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_artifact_root": str(source_artifacts),
        "destination_repository": str(repo_root),
        "training": {},
    }
    try:
        for relative, destination in zip(
            TRAINING_ROOTS, destinations[:2], strict=True
        ):
            source = source_artifacts / relative
            print(f"Copying {source} -> {destination}", file=sys.stderr, flush=True)
            shutil.copytree(source, destination, copy_function=shutil.copy2)
            print(f"Sanitizing {destination}", file=sys.stderr, flush=True)
            replacements = _sanitize_training_root(
                destination,
                source_artifacts=source_artifacts,
                repo_root=repo_root,
            )
            _assert_no_absolute_json_paths(destination)
            summary["training"][Path(relative).name] = {
                "metadata_path_replacements": replacements,
                "artifact_tree": tree_identity(destination),
            }

        public_test_destination = destinations[2]
        print("Building current-only Public Test tree", file=sys.stderr, flush=True)
        public_receipts = _copy_public_test(
            source_artifacts=source_artifacts,
            destination=public_test_destination,
            repo_root=repo_root,
        )
        _assert_no_absolute_json_paths(public_test_destination)
        summary["public_test"] = {
            **public_receipts,
            "artifact_tree": tree_identity(public_test_destination),
        }

        for target in targets:
            print(f"Published {target}", file=sys.stderr, flush=True)
    except BaseException:
        for target in targets:
            if target.exists():
                shutil.rmtree(target)
        raise

    for relative, section in zip(
        TRAINING_ROOTS, summary["training"].values(), strict=True
    ):
        section["artifact_tree"]["root"] = (
            Path("artifacts") / relative
        ).as_posix()
    summary["public_test"]["artifact_tree"]["root"] = (
        Path("artifacts") / PUBLIC_TEST_ROOT
    ).as_posix()
    summary["status"] = "passed"
    return summary


def build_public_test_only(
    *, repo_root: Path, source_artifacts: Path
) -> dict[str, Any]:
    """Rebuild only the current-only Public Test after metadata design changes."""

    repo_root = repo_root.expanduser().resolve()
    source_artifacts = source_artifacts.expanduser().resolve()
    destination = repo_root / "artifacts" / PUBLIC_TEST_ROOT
    if destination.exists():
        raise FileExistsError(f"Refusing to replace Public Test: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipts = _copy_public_test(
            source_artifacts=source_artifacts,
            destination=destination,
            repo_root=repo_root,
        )
        _assert_no_absolute_json_paths(destination)
    except BaseException:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    return {
        "schema_version": 1,
        "status": "passed",
        "source_artifact_root": str(source_artifacts),
        "destination_repository": str(repo_root),
        "public_test": {
            **receipts,
            "artifact_tree": {
                **tree_identity(destination),
                "root": (Path("artifacts") / PUBLIC_TEST_ROOT).as_posix(),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repository_root(),
    )
    parser.add_argument(
        "--source-artifact-root",
        type=Path,
        default=artifact_root(repository_root()),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--public-test-only",
        action="store_true",
        help="Build only the cleaned Public Test and current score receipts.",
    )
    args = parser.parse_args()
    builder = build_public_test_only if args.public_test_only else build_release
    summary = builder(
        repo_root=args.repo_root, source_artifacts=args.source_artifact_root
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
