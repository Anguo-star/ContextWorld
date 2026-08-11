from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.evaluation.action_delay_h7_validation import file_sha256
from contextworld.evaluation.action_delay_h7_score import (
    load_h7_validation_assets,
)
from contextworld.evaluation.action_delay_h7_data import directory_sha256
from contextworld.paths import repository_root, resolve_contextworld_path


RELEASE_ID = "contextworld_tworoom_action_delay_icl_history7_v1"
DEFAULT_ACTION_DELAY_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/tworoom_action_delay_icl_release_v1.yaml"
)

_MACHINE_PATH_MARKERS = (
    b"/opt/",
    b"/tmp/",
    b"/home/",
    b"/root/",
    b"../../data/",
    b"\\\\Users\\\\",
)
_FORBIDDEN_PUBLIC_TEST_PARTS = {
    "latent_diagnostics",
    "logs",
    "scoring_matrix_report",
    "comparison_summary.json",
}


def load_action_delay_icl_release(
    path: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported Action Delay release config: {config_path}"
        )
    if payload.get("release_id") != RELEASE_ID:
        raise ValueError(f"Unexpected release id in {config_path}")
    if payload.get("scope", {}).get("sealed_test_included") is not False:
        raise ValueError("Action Delay v1 must not include sealed Test data")
    if str(payload.get("release_status")).startswith("public_test_") and (
        payload.get("scope", {}).get("public_test_included") is not True
    ):
        raise ValueError("Action Delay v1 must include Public Test data")
    if payload.get("release_status") not in {
        "validation_release_candidate",
        "validation_release",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError(
            f"Unsupported release status: {payload.get('release_status')}"
        )
    return {**payload, "_config_path": str(config_path)}


class ActionDelayICLEvalDataset:
    """Hash-checked reader for frozen History=7 Action Delay Public Test."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
        repo_root: Path | None = None,
        eval_seeds: list[int] | tuple[int, ...] | None = None,
        limit_per_seed: int | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_action_delay_icl_release(
            release_config
        )
        evaluation = self.release["evaluation"]
        self.catalog_path = resolve_contextworld_path(
            evaluation["catalog"],
            repo_root=self.repo_root,
        )
        if not self.catalog_path.is_file():
            raise FileNotFoundError(self.catalog_path)
        observed = file_sha256(self.catalog_path)
        if observed != evaluation["catalog_sha256"]:
            raise RuntimeError(
                f"Action Delay catalog hash mismatch: {observed}"
            )
        self.catalog, assets = load_h7_validation_assets(
            self.catalog_path,
            repo_root=self.repo_root,
        )
        if (
            self.catalog["content_manifest_sha256"]
            != evaluation["content_manifest_sha256"]
        ):
            raise RuntimeError(
                "Action Delay catalog content manifest changed"
            )
        official_seeds = tuple(
            int(value) for value in evaluation["eval_seeds"]
        )
        selected_seeds = tuple(
            official_seeds
            if eval_seeds is None
            else (int(value) for value in eval_seeds)
        )
        if (
            len(selected_seeds) != len(set(selected_seeds))
            or not set(selected_seeds).issubset(set(official_seeds))
        ):
            raise ValueError(
                f"Eval seeds must be unique members of {official_seeds}"
            )
        official_limit = int(evaluation["queries_per_eval_seed"])
        selected_limit = (
            official_limit
            if limit_per_seed is None
            else int(limit_per_seed)
        )
        if not 1 <= selected_limit <= official_limit:
            raise ValueError(
                f"limit_per_seed must be in [1,{official_limit}]"
            )
        self.eval_seeds = selected_seeds
        self.limit_per_seed = selected_limit
        self._assets = sorted(
            (
                asset
                for asset in assets
                if int(asset["eval_seed"]) in selected_seeds
                and int(asset["evaluation_index"]) < selected_limit
            ),
            key=lambda row: (
                int(row["eval_seed"]),
                int(row["evaluation_index"]),
                str(row["query_id"]),
            ),
        )
        expected = len(selected_seeds) * selected_limit
        if len(self._assets) != expected:
            raise RuntimeError(
                f"Selected {len(self._assets)} queries, expected {expected}"
            )

    def __len__(self) -> int:
        return len(self._assets)

    @property
    def raw_assets(self) -> list[dict[str, Any]]:
        return list(self._assets)

    @property
    def is_full_protocol(self) -> bool:
        evaluation = self.release["evaluation"]
        return (
            tuple(self.eval_seeds)
            == tuple(int(value) for value in evaluation["eval_seeds"])
            and self.limit_per_seed
            == int(evaluation["queries_per_eval_seed"])
        )

    def describe(self) -> dict[str, Any]:
        return {
            "catalog": str(self.catalog_path),
            "catalog_sha256": file_sha256(self.catalog_path),
            "content_manifest_sha256": self.catalog[
                "content_manifest_sha256"
            ],
            "queries": len(self),
            "eval_seeds": list(self.eval_seeds),
            "queries_per_eval_seed": self.limit_per_seed,
            "delay_values": list(
                self.catalog["protocol"]["delay_values"]
            ),
            "history_tokens": int(
                self.catalog["protocol"]["history_tokens"]
            ),
            "online_environment_calls": 0,
            "full_protocol": self.is_full_protocol,
        }


def _verified_file(
    specification: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(
        specification["path"],
        repo_root=repo_root,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    expected = str(specification["sha256"])
    if observed != expected:
        raise RuntimeError(
            f"Artifact hash mismatch: {path}; {observed} != {expected}"
        )
    return {"path": str(path), "sha256": observed, "passed": True}


def _training_stages(release: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stages = release["training"].get("stages")
    if isinstance(stages, dict) and stages:
        return stages
    # Compatibility with early local release candidates that exposed one
    # training tree directly under ``training``.
    return {
        "training": {
            "artifacts": release["training"]["artifacts"],
            "artifact_tree": release["training"].get("artifact_tree", {}),
        }
    }


def _artifact_tree_audit(
    specification: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
    strict: bool,
) -> dict[str, Any]:
    logical_root = str(specification.get("root", ""))
    resolved = resolve_contextworld_path(logical_root, repo_root=repo_root)
    files = (
        sorted(value for value in resolved.rglob("*") if value.is_file())
        if resolved.is_dir()
        else []
    )
    observed_files = len(files)
    observed_bytes = sum(value.stat().st_size for value in files)
    expected_files = specification.get("files")
    expected_bytes = specification.get("bytes")
    expected_sha256 = specification.get("sha256")
    observed_sha256 = directory_sha256(resolved) if full and files else None
    expected_local = (repo_root / logical_root).resolve()
    repository_local = bool(
        logical_root.startswith("artifacts/")
        and resolved == expected_local
        and resolved.is_dir()
    )
    symlinks = [
        value.relative_to(resolved).as_posix()
        for value in resolved.rglob("*")
        if value.is_symlink()
    ] if resolved.is_dir() else []
    passed = bool(
        resolved.is_dir()
        and (expected_files is None or observed_files == int(expected_files))
        and (expected_bytes is None or observed_bytes == int(expected_bytes))
        and (
            not full
            or expected_sha256 is None
            or observed_sha256 == str(expected_sha256)
        )
        and (not strict or repository_local)
        and (not strict or not symlinks)
    )
    return {
        "logical_root": logical_root,
        "path": str(resolved),
        "repository_local": repository_local,
        "files": observed_files,
        "expected_files": expected_files,
        "bytes": observed_bytes,
        "expected_bytes": expected_bytes,
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "sha256_checked": bool(full and expected_sha256 is not None),
        "symlinks": symlinks,
        "passed": passed,
    }


def _portable_metadata_audit(roots: list[Path]) -> dict[str, Any]:
    checked = 0
    violations: list[dict[str, str]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            checked += 1
            with path.open("rb") as stream:
                tail = b""
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    block = tail + chunk
                    marker = next(
                        (value for value in _MACHINE_PATH_MARKERS if value in block),
                        None,
                    )
                    if marker is not None:
                        violations.append(
                            {
                                "path": str(path),
                                "marker": marker.decode("utf-8", errors="replace"),
                            }
                        )
                        break
                    tail = block[-32:]
    return {
        "json_files_checked": checked,
        "violations": violations,
        "passed": not violations,
    }


def _public_test_layout_audit(
    evaluation_root: Path,
    *,
    expected_model_results: set[str],
) -> dict[str, Any]:
    files = {
        value.relative_to(evaluation_root).as_posix()
        for value in evaluation_root.rglob("*")
        if value.is_file()
    }
    assets = {value for value in files if value.startswith("assets/")}
    expected_receipts = {
        "score_receipts/core_summary.json",
        "score_receipts/source_compatibility.json",
        "score_receipts/cem_retention/final_summary.json",
        "score_receipts/cem_retention/runner_report.json",
        *(f"score_receipts/model_results/{name}" for name in expected_model_results),
    }
    metadata = {
        "catalog.json",
        "build_report.json",
        "audit_report.json",
        "training_exclusion_manifest.json",
    }
    expected = metadata | expected_receipts | assets
    forbidden = sorted(
        relative
        for relative in files
        if any(part in _FORBIDDEN_PUBLIC_TEST_PARTS for part in Path(relative).parts)
    )
    assets_exact = bool(
        len(assets) == 300
        and all(Path(value).suffix == ".npz" for value in assets)
    )
    passed = bool(assets_exact and files == expected and not forbidden)
    return {
        "files": len(files),
        "assets": len(assets),
        "score_receipts": len(files & expected_receipts),
        "expected_score_receipts": len(expected_receipts),
        "unexpected_files": sorted(files - expected),
        "missing_files": sorted(expected - files),
        "forbidden_files": forbidden,
        "passed": passed,
    }


def _reference_result_audit(
    *,
    core_path: Path,
    model_result_paths: list[Path],
    compatibility_path: Path,
    cem_path: Path,
    runner_path: Path,
) -> dict[str, Any]:
    core = json.loads(core_path.read_text(encoding="utf-8"))
    result_rows: list[dict[str, Any]] = []
    for path in model_result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result_rows.append(
            {
                "benchmark": payload.get("benchmark"),
                "receipt_kind": payload.get("receipt_kind"),
                "label": payload.get("label"),
                "model_family": payload.get("model_family"),
                "training_seed": payload.get("training_seed"),
                "status": payload.get("status"),
                "training_receipt": payload.get("training_receipt", {}),
                "score_audit": payload.get("public_test", {}).get(
                    "score_audit", {}
                ),
                "source_evidence": payload.get("source_evidence", {}),
                "primary_score": payload.get("primary_score"),
                "gate": payload.get("gate"),
                "forbidden_development_terms": any(
                    term in json.dumps(payload, sort_keys=True).lower()
                    for term in (
                        "diagnostic",
                        "post_hoc",
                        "validation_checkpoint",
                    )
                ),
            }
        )
    compatibility = json.loads(
        compatibility_path.read_text(encoding="utf-8")
    )
    cem = json.loads(cem_path.read_text(encoding="utf-8"))
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    expected_labels = {
        f"h7_action_delay_curriculum_v4_{family}_formal_s{seed}"
        for family in ("lewm", "pldm")
        for seed in (3072, 4096, 5120)
    }
    observed_labels = {str(row.get("label")) for row in result_rows}
    current_only = bool(
        "historical_artifacts" not in core
        and "reference_comparison" not in core
        and "diagnostic_only" not in json.dumps(core, sort_keys=True)
    )
    current_results_passed = bool(
        len(result_rows) == 6
        and observed_labels == expected_labels
        and all(
            row.get("benchmark")
            == "contextworld_tworoom_action_delay_icl_history7_v1"
            and row.get("receipt_kind") == "public_test_model_score"
            and row.get("status") == "completed"
            and row.get("model_family") in {"lewm", "pldm"}
            and int(row.get("training_seed", 0)) in {3072, 4096, 5120}
            and row.get("training_receipt", {}).get("passed") is True
            and row.get("score_audit", {}).get("passed") is True
            and row.get("source_evidence", {}).get(
                "compatibility_checks"
            )
            and all(
                row["source_evidence"]["compatibility_checks"].values()
            )
            and isinstance(row.get("primary_score"), dict)
            and isinstance(row.get("gate"), dict)
            and row["forbidden_development_terms"] is False
            for row in result_rows
        )
    )
    compatibility_passed = bool(
        compatibility.get("status") == "passed"
        and compatibility.get("receipt_kind")
        == "source_to_public_score_compatibility"
        and compatibility.get("archived_source_results_distributed") is False
        and set(compatibility.get("entries", {})) == expected_labels
        and all(compatibility.get("checks", {}).values())
    )
    core_passed = bool(
        core.get("status") == "completed"
        and core.get("passed") is True
        and core.get("benchmark")
        == "contextworld_tworoom_action_delay_icl_history7_v1"
        and core.get("receipt_kind") == "public_test_core_score_summary"
        and current_only
        and len(core.get("models", [])) == 6
        and core.get("by_family", {}).get("pldm", {}).get("passed_seeds") == 3
        and core.get("by_family", {}).get("lewm", {}).get("passed_seeds") == 0
        and set(core.get("artifacts", {})) == expected_labels
    )
    decision = cem.get("decision", {})
    cem_passed = bool(
        cem.get("status") == "completed"
        and decision.get("target_original_cem_ability_retained") is True
        and int(decision.get("required_target_seeds", 0)) == 3
        and int(decision.get("target_seeds_passed", 0)) == 3
    )
    runner_passed = bool(
        runner.get("status") == "passed"
        and len(runner.get("results", [])) == 84
        and all(row.get("status") == "passed" for row in runner.get("results", []))
    )
    passed = bool(
        core_passed
        and current_results_passed
        and compatibility_passed
        and cem_passed
        and runner_passed
    )
    return {
        "current_only": current_only,
        "model_results": len(result_rows),
        "model_labels": sorted(observed_labels),
        "pldm_training_seeds_passed": core.get("by_family", {})
        .get("pldm", {})
        .get("passed_seeds"),
        "lewm_training_seeds_passed": core.get("by_family", {})
        .get("lewm", {})
        .get("passed_seeds"),
        "cem_target_seeds_passed": decision.get("target_seeds_passed"),
        "cem_runner_results": len(runner.get("results", [])),
        "checks": {
            "core": core_passed,
            "model_results": current_results_passed,
            "source_compatibility": compatibility_passed,
            "cem_retention": cem_passed,
            "cem_runner": runner_passed,
        },
        "passed": passed,
    }


def audit_action_delay_icl_release(
    *,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_action_delay_icl_release(release_config)
    strict_portability = bool(
        release.get("portability", {}).get("repository_local_artifacts")
        is True
    )
    files: dict[str, Any] = {}
    for name, specification in release["identity"].items():
        if isinstance(specification, dict) and {
            "path",
            "sha256",
        }.issubset(specification):
            files[name] = _verified_file(
                specification,
                repo_root=root,
            )
    artifact_sections: list[tuple[str, dict[str, Any]]] = [
        ("evaluation", release["evaluation"]["artifacts"]),
        ("reference", release.get("reference_results", {})),
    ]
    artifact_sections.extend(
        (f"training.{stage_name}", stage["artifacts"])
        for stage_name, stage in _training_stages(release).items()
    )
    for section, values in artifact_sections:
        for name, specification in values.items():
            if isinstance(specification, dict) and {
                "path",
                "sha256",
            }.issubset(specification):
                files[f"{section}.{name}"] = _verified_file(
                    specification,
                    repo_root=root,
                )

    stage_audits: dict[str, Any] = {}
    stage_builds: dict[str, dict[str, Any]] = {}
    tree_audits: dict[str, Any] = {}
    stages = _training_stages(release)
    for stage_name, stage in stages.items():
        prefix = f"training.{stage_name}"
        tree_audits[prefix] = _artifact_tree_audit(
            stage.get("artifact_tree", {}),
            repo_root=root,
            full=full,
            strict=strict_portability,
        )
        build_path = Path(files[f"{prefix}.build_report"]["path"])
        build = json.loads(build_path.read_text(encoding="utf-8"))
        if build.get("passed") is not True:
            raise RuntimeError(
                f"Action Delay {stage_name} data build did not pass"
            )
        stage_builds[stage_name] = build
        manifest_path = Path(files[f"{prefix}.manifest"]["path"])
        manifest_rows = [
            json.loads(line)
            for line in manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        expected_shards = int(build["physical_counts"]["shards"])
        if len(manifest_rows) != expected_shards:
            raise RuntimeError(
                f"Action Delay {stage_name} manifest shard count changed: "
                f"{len(manifest_rows)} != {expected_shards}"
            )
        storage_hashes_checked = 0
        for row in manifest_rows:
            shard = resolve_contextworld_path(
                row["output_path"],
                repo_root=root,
            )
            if not shard.is_dir():
                raise FileNotFoundError(shard)
            if full:
                observed = directory_sha256(shard)
                if observed != row["storage_sha256"]:
                    raise RuntimeError(
                        "Action Delay shard hash mismatch: "
                        f"{stage_name}: {shard}"
                    )
                storage_hashes_checked += 1
        physical = build["physical_counts"]
        stage_audits[stage_name] = {
            "passed": True,
            "query_bundles": int(
                physical.get(
                    "query_bundles",
                    physical.get("query_triplets", 0),
                )
            ),
            "episodes": int(physical["episodes"]),
            "raw_rows_replayed": int(physical["raw_rows_replayed"]),
            "shards_present": len(manifest_rows),
            "storage_hashes_checked": storage_hashes_checked,
        }
    evaluation_tree = _artifact_tree_audit(
        release["evaluation"].get("artifact_tree", {}),
        repo_root=root,
        full=full,
        strict=strict_portability,
    )
    tree_audits["evaluation"] = evaluation_tree
    dataset_audit: dict[str, Any] = {
        "full_payload_arrays_checked": False
    }
    if full:
        dataset = ActionDelayICLEvalDataset(
            release=release,
            repo_root=root,
        )
        if not dataset.is_full_protocol or len(dataset) != 300:
            raise RuntimeError("Action Delay full Eval must have 300 queries")
        dataset_audit = {
            **dataset.describe(),
            "full_payload_arrays_checked": True,
        }
    evaluation_build_path = Path(files["evaluation.build_report"]["path"])
    evaluation_audit_path = Path(files["evaluation.audit_report"]["path"])
    evaluation_build = json.loads(
        evaluation_build_path.read_text(encoding="utf-8")
    )
    evaluation_replay_audit = json.loads(
        evaluation_audit_path.read_text(encoding="utf-8")
    )
    pairing_checks = [
        build["pairing"]["checks"] for build in stage_builds.values()
    ]
    causal_data = audit_causal_data_contract(
        component_id="action_delay",
        evidence_scope=(
            "all 12,160 Training/Development query bundles across both "
            "training stages and all 3,300 Public Test physical rollouts"
        ),
        continuous_environment_trajectory=all(
            build["checks"]["all_shards_pass_raw_physical_replay"]
            for build in stage_builds.values()
        )
        and evaluation_replay_audit["checks"][
            "full_physical_replay_completed"
        ],
        state_installations_after_x0=0,
        query_simulator_recreated=False,
        maximum_query_state_gap=0.0,
        query_state_tolerance=0.0,
        query_pixels_exact=all(
            checks["query_pixels_exact"] for checks in pairing_checks
        )
        and evaluation_build["checks"]["every_physical_family_passed"],
        query_actions_exact=all(
            checks["commanded_actions_exact"] for checks in pairing_checks
        ),
        history_effect_present=all(
            build["pairing"]["passed"] for build in stage_builds.values()
        ),
        true_future_effect_present=(
            int(
                evaluation_build["physical_equivalence"]["horizon1"][
                    "distinct_groups"
                ]
            )
            == 6
        ),
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=all(
            checks["initial_pixels_exact"] for checks in pairing_checks
        ),
        evidence=(
            *(files[f"training.{name}.build_report"]["path"]
              for name in stage_builds),
            str(evaluation_build_path),
            str(evaluation_audit_path),
            files["validation_contract"]["path"],
        ),
    )
    portability_audit: dict[str, Any] = {
        "repository_local_artifacts": False,
        "json_files_checked": 0,
        "violations": [],
        "passed": True,
        "skipped": True,
    }
    layout_audit: dict[str, Any] = {"passed": True, "skipped": True}
    reference_audit: dict[str, Any] = {"passed": True, "skipped": True}
    if strict_portability:
        artifact_roots = [
            Path(row["path"]) for row in tree_audits.values()
        ]
        portability_audit = {
            "repository_local_artifacts": all(
                row["repository_local"] for row in tree_audits.values()
            ),
            **_portable_metadata_audit(artifact_roots),
            "skipped": False,
        }
        portability_audit["passed"] = bool(
            portability_audit["repository_local_artifacts"]
            and not portability_audit["violations"]
        )
        model_result_specifications = {
            name: specification
            for name, specification in release["evaluation"][
                "artifacts"
            ].items()
            if isinstance(specification, dict)
            and str(specification.get("path", "")).startswith(
                str(release["evaluation"]["artifact_tree"]["root"])
                + "/score_receipts/model_results/"
            )
        }
        model_result_paths = [
            Path(files[f"evaluation.{name}"]["path"])
            for name in model_result_specifications
        ]
        expected_model_results = {
            Path(specification["path"]).name
            for specification in model_result_specifications.values()
        }
        layout_audit = {
            **_public_test_layout_audit(
                Path(evaluation_tree["path"]),
                expected_model_results=expected_model_results,
            ),
            "skipped": False,
        }
        reference_audit = {
            **_reference_result_audit(
                core_path=Path(files["reference.core"]["path"]),
                model_result_paths=model_result_paths,
                compatibility_path=Path(
                    files["evaluation.source_compatibility"]["path"]
                ),
                cem_path=Path(files["reference.cem_retention"]["path"]),
                runner_path=Path(files["reference.cem_runner_audit"]["path"]),
            ),
            "skipped": False,
        }
    files_passed = all(row["passed"] for row in files.values())
    trees_passed = all(row["passed"] for row in tree_audits.values())
    training_passed = all(
        row["passed"] for row in stage_audits.values()
    )
    evaluation_passed = bool(
        evaluation_build.get("status") == "passed"
        and evaluation_replay_audit.get("status") == "passed"
        and (
            not full
            or (
                dataset_audit.get("full_payload_arrays_checked") is True
                and int(
                    dataset_audit.get(
                        "queries",
                        dataset_audit.get("query_bundles", 0),
                    )
                )
                == 300
            )
        )
    )
    passed = bool(
        files_passed
        and trees_passed
        and training_passed
        and evaluation_passed
        and causal_data["passed"]
        and portability_audit["passed"]
        and layout_audit["passed"]
        and reference_audit["passed"]
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if passed else "failed",
        "release_config": {
            "path": release["_config_path"],
            "sha256": file_sha256(Path(release["_config_path"])),
        },
        "files": files,
        "artifact_trees": tree_audits,
        "training_data": {
            "passed": training_passed,
            "stages": stage_audits,
            "episodes": sum(
                row["episodes"] for row in stage_audits.values()
            ),
            "raw_rows_replayed": sum(
                row["raw_rows_replayed"]
                for row in stage_audits.values()
            ),
            "shards_present": sum(
                row["shards_present"] for row in stage_audits.values()
            ),
            "storage_hashes_checked": sum(
                row["storage_hashes_checked"]
                for row in stage_audits.values()
            ),
        },
        "evaluation_data": dataset_audit,
        "public_test_layout": layout_audit,
        "reference_results": reference_audit,
        "portability": portability_audit,
        "causal_data_contract": causal_data,
        "full": bool(full),
        "passed": passed,
    }


def action_delay_icl_training_plan(
    recipe: str,
    *,
    training_seed: int,
    release_config: Path | str = DEFAULT_ACTION_DELAY_RELEASE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_action_delay_icl_release(release_config)
    recipes = release["training"]["recipes"]
    if recipe not in recipes:
        raise KeyError(
            f"Unknown recipe {recipe!r}; available={sorted(recipes)}"
        )
    selected = recipes[recipe]
    seed = int(training_seed)
    if seed not in [
        int(value) for value in selected["training_seeds"]
    ]:
        raise ValueError(
            f"Seed {seed} is not frozen for recipe {recipe}"
        )
    family = str(selected["model_family"])
    stages = selected.get("stages")
    if not isinstance(stages, list) or not stages:
        stages = [
            {
                "name": "training",
                "config": selected["config"],
                "config_sha256": selected["config_sha256"],
                "launcher": (
                    "scripts/run_h7_action_delay_full_range_train.sh"
                ),
            }
        ]
    plans = []
    for stage in stages:
        config = resolve_contextworld_path(
            stage["config"],
            repo_root=root,
        )
        observed = file_sha256(config)
        if observed != stage["config_sha256"]:
            raise RuntimeError(f"Training config hash changed: {config}")
        launcher = resolve_contextworld_path(
            stage["launcher"],
            repo_root=root,
        )
        if not launcher.is_file():
            raise FileNotFoundError(launcher)
        plans.append(
            {
                "stage": str(stage["name"]),
                "config": str(config),
                "config_sha256": observed,
                "command": [
                    "bash",
                    str(launcher),
                    family,
                    str(seed),
                ],
            }
        )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "recipe": recipe,
        "training_seed": seed,
        "model_family": family,
        "model_id": selected["model_id"],
        "stages": plans,
        "commands": [stage["command"] for stage in plans],
        "environment": {
            "STABLEWM_REPO": "/path/to/pinned/stable-worldmodel"
        },
    }


__all__ = [
    "ActionDelayICLEvalDataset",
    "DEFAULT_ACTION_DELAY_RELEASE_CONFIG",
    "RELEASE_ID",
    "action_delay_icl_training_plan",
    "audit_action_delay_icl_release",
    "load_action_delay_icl_release",
]
