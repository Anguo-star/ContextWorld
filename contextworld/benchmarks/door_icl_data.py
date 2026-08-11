from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.evaluation.hidden_passage_validation import (
    HISTORY_CONDITIONS,
    TRUE_RULES,
    file_sha256,
    load_validation_assets,
)
from contextworld.paths import repository_root, resolve_contextworld_path


RELEASE_ID = "contextworld_tworoom_door_rule_icl_history3_v1"
DEFAULT_DOOR_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/tworoom_door_icl_release_v1.yaml"
)
_PORTABLE_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
}
_NON_PORTABLE_MARKERS = (
    "/opt/",
    "/tmp/",
    "/home/",
    "/root/",
    "../../data/",
    "\\Users\\",
)
_PUBLIC_TEST_TOP_LEVEL = {
    "build_report.json",
    "catalog.json",
    "payloads",
    "training_exclusion_manifest.json",
}


def door_icl_export_entries(
    release: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return the complete, minimal Door release artifact inventory."""

    entries = [
        (release["training"]["artifact_tree"]["root"], "directory"),
        (release["evaluation"]["artifact_tree"]["root"], "directory"),
        (release["evaluation"]["normalizer"], "file"),
        (release["training"]["initialization"]["checkpoint"], "file"),
        (
            release["training"]["initialization"]["checkpoint_config"],
            "file",
        ),
    ]
    entries.extend(
        (row["root"], "directory")
        for row in release["reference_results"].values()
    )
    return entries


def _tree_fingerprint(
    path: Path,
    *,
    hash_contents: bool,
) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    total_bytes = sum(value.stat().st_size for value in files)
    if not hash_contents:
        return {
            "files": len(files),
            "bytes": total_bytes,
            "sha256": None,
            "full_hash_verified": False,
        }
    digest = hashlib.sha256()
    for value in files:
        relative = value.relative_to(path).as_posix()
        size = value.stat().st_size
        digest.update(
            (
                f"{relative}\0{size}\0{file_sha256(value)}\n"
            ).encode("utf-8")
        )
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "full_hash_verified": True,
    }


def load_door_icl_release(
    path: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Door ICL release config: {config_path}")
    if payload.get("release_id") != RELEASE_ID:
        raise ValueError(f"Unexpected release id in {config_path}")
    status = payload.get("release_status")
    if status not in {
        "validation_release_candidate",
        "validation_release",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError(
            f"Unsupported Door ICL release status: {status!r}"
        )
    if payload.get("scope", {}).get("sealed_test_included") is not False:
        raise ValueError("Door ICL v1 must not include the sealed Test")
    if str(status).startswith("public_test_") and (
        payload.get("scope", {}).get("public_test_included") is not True
    ):
        raise ValueError("Door ICL v1 must include Public Test")
    distribution = payload.get("distribution", {})
    if status in {"validation_release", "public_test_release"} and not (
        distribution.get("code_license_status") == "declared"
        and distribution.get("generated_data_license_status") == "declared"
        and distribution.get("public_download_status") == "configured"
    ):
        raise ValueError(
            "A formal public release requires declared source/data "
            "licenses and configured public artifact downloads"
        )
    return {**payload, "_config_path": str(config_path)}


@dataclass(frozen=True)
class DoorICLEvalExample:
    query_id: str
    static_query_id: str
    eval_seed: int
    evaluation_index: int
    direction: str
    template_id: str
    query_pixels: np.ndarray
    histories: dict[str, np.ndarray]
    action_blocks: dict[str, np.ndarray]
    true_next_pixels: dict[str, np.ndarray]


class DoorICLEvalDataset:
    """Hash-checked reader for the frozen public door Test set."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
        repo_root: Path | None = None,
        eval_seeds: list[int] | tuple[int, ...] | None = None,
        limit_per_seed: int | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_door_icl_release(release_config)
        evaluation = self.release["evaluation"]
        self.catalog_path = resolve_contextworld_path(
            evaluation["catalog"],
            repo_root=self.repo_root,
        )
        if not self.catalog_path.is_file():
            raise FileNotFoundError(self.catalog_path)
        observed_catalog_hash = file_sha256(self.catalog_path)
        if observed_catalog_hash != evaluation["catalog_sha256"]:
            raise RuntimeError(
                "Door Validation catalog hash mismatch: "
                f"{observed_catalog_hash}"
            )
        assets, catalog_audit = load_validation_assets(
            self.catalog_path,
            repo_root=self.repo_root,
        )
        official_seeds = tuple(
            int(value) for value in evaluation["eval_seeds"]
        )
        selected_seeds = tuple(
            int(value)
            for value in (
                official_seeds if eval_seeds is None else eval_seeds
            )
        )
        if len(selected_seeds) != len(set(selected_seeds)):
            raise ValueError("Eval seeds must be unique")
        if not set(selected_seeds).issubset(set(official_seeds)):
            raise ValueError(
                f"Eval seeds must be a subset of {list(official_seeds)}"
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
        self._assets = [
            asset
            for asset in assets
            if int(asset["eval_seed"]) in selected_seeds
            and int(asset["evaluation_index"]) < selected_limit
        ]
        self._assets.sort(
            key=lambda row: (
                int(row["eval_seed"]),
                int(row["evaluation_index"]),
                str(row["query_id"]),
            )
        )
        expected = len(selected_seeds) * selected_limit
        if len(self._assets) != expected:
            raise RuntimeError(
                f"Selected {len(self._assets)} door queries, expected {expected}"
            )
        self.catalog_audit = catalog_audit

    def __len__(self) -> int:
        return len(self._assets)

    def __getitem__(self, index: int) -> DoorICLEvalExample:
        row = self._assets[index]
        return DoorICLEvalExample(
            query_id=str(row["query_id"]),
            static_query_id=str(row["static_query_id"]),
            eval_seed=int(row["eval_seed"]),
            evaluation_index=int(row["evaluation_index"]),
            direction=str(row["direction"]),
            template_id=str(row["template_id"]),
            query_pixels=np.asarray(row["query_pixels"], dtype=np.uint8),
            histories={
                key: np.asarray(value, dtype=np.uint8)
                for key, value in row["histories"].items()
            },
            action_blocks={
                key: np.asarray(value, dtype=np.float32)
                for key, value in row["actions"].items()
            },
            true_next_pixels={
                key: np.asarray(value, dtype=np.uint8)
                for key, value in row["targets"].items()
            },
        )

    def __iter__(self) -> Iterator[DoorICLEvalExample]:
        for index in range(len(self)):
            yield self[index]

    @property
    def raw_assets(self) -> list[dict[str, Any]]:
        return list(self._assets)

    @property
    def is_full_protocol(self) -> bool:
        evaluation = self.release["evaluation"]
        return (
            set(self.eval_seeds)
            == {int(value) for value in evaluation["eval_seeds"]}
            and self.limit_per_seed
            == int(evaluation["queries_per_eval_seed"])
        )

    def describe(self) -> dict[str, Any]:
        return {
            "track": "unseen_door_positions",
            "catalog": str(self.catalog_path),
            "catalog_sha256": self.release["evaluation"]["catalog_sha256"],
            "eval_seeds": list(self.eval_seeds),
            "queries_per_eval_seed": self.limit_per_seed,
            "queries": len(self),
            "history_conditions": list(HISTORY_CONDITIONS),
            "true_next_frame_rules": list(TRUE_RULES),
            "model_predictions": len(self) * len(HISTORY_CONDITIONS),
            "loss_records": (
                len(self) * len(HISTORY_CONDITIONS) * len(TRUE_RULES)
            ),
            "model_visible_fields": ["pixels", "action"],
            "online_environment_calls": 0,
            "full_protocol": self.is_full_protocol,
        }


def _audit_file(
    logical_path: str,
    expected_sha256: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    exists = path.is_file()
    observed = file_sha256(path) if exists else None
    return {
        "logical_path": logical_path,
        "path": str(path),
        "exists": exists,
        "expected_sha256": str(expected_sha256),
        "observed_sha256": observed,
        "passed": bool(exists and observed == expected_sha256),
    }


def _audit_tree(
    logical_path: str,
    expected: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    observed = (
        _tree_fingerprint(path, hash_contents=full)
        if path.is_dir()
        else {
            "files": 0,
            "bytes": 0,
            "sha256": None,
            "full_hash_verified": False,
        }
    )
    passed = bool(
        path.is_dir()
        and observed["files"] == int(expected["files"])
        and observed["bytes"] == int(expected["bytes"])
        and (
            not full or observed["sha256"] == expected["sha256"]
        )
    )
    return {
        "logical_path": logical_path,
        "path": str(path),
        "exists": path.is_dir(),
        "expected": dict(expected),
        "observed": observed,
        "passed": passed,
    }


def _audit_portable_text(
    logical_path: str,
    *,
    kind: str,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(logical_path, repo_root=repo_root)
    candidates = (path,) if kind == "file" else path.rglob("*")
    violations: list[dict[str, Any]] = []
    scanned = 0
    for candidate in candidates:
        if (
            not candidate.is_file()
            or candidate.suffix.lower() not in _PORTABLE_TEXT_SUFFIXES
        ):
            continue
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        markers = [
            marker for marker in _NON_PORTABLE_MARKERS if marker in text
        ]
        if markers:
            relative = (
                candidate.name
                if kind == "file"
                else candidate.relative_to(path).as_posix()
            )
            violations.append({"path": relative, "markers": markers})
    symlinks = (
        []
        if kind == "file"
        else sorted(
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_symlink()
        )
    )
    return {
        "logical_path": logical_path,
        "kind": kind,
        "text_files_scanned": scanned,
        "violations": violations,
        "symlinks": symlinks,
        "passed": bool(path.exists() and not violations and not symlinks),
    }


def _audit_reference_method(
    name: str,
    specification: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
) -> dict[str, Any]:
    root = resolve_contextworld_path(
        specification["root"], repo_root=repo_root
    )
    seeds = [int(value) for value in specification["training_seeds"]]
    expected_files = {str(specification["aggregate"])} | {
        str(specification["result_pattern"]).format(seed=seed)
        for seed in seeds
    }
    observed_files = (
        {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if root.is_dir()
        else set()
    )
    tree = _audit_tree(
        specification["root"],
        specification,
        repo_root=repo_root,
        full=full,
    )
    portability = _audit_portable_text(
        specification["root"], kind="directory", repo_root=repo_root
    )
    aggregate_checks: dict[str, bool] = {}
    aggregate_path = root / str(specification["aggregate"])
    if aggregate_path.is_file():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        result_rows = aggregate.get("result_files", [])
        results_by_seed = {
            int(row["training_seed"]): row
            for row in result_rows
            if isinstance(row, dict) and "training_seed" in row
        }
        result_hashes_match = True
        result_payloads_match = True
        for seed in seeds:
            relative = str(specification["result_pattern"]).format(
                seed=seed
            )
            path = root / relative
            row = results_by_seed.get(seed, {})
            result_hashes_match = bool(
                result_hashes_match
                and path.is_file()
                and row.get("sha256") == file_sha256(path)
            )
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                result_payloads_match = bool(
                    result_payloads_match
                    and payload.get("status") == "completed"
                    and payload.get("model_id")
                    == specification["model_id"]
                    and int(payload.get("training_seed", -1)) == seed
                    and payload.get("summary", {})
                    .get("decision", {})
                    .get("passed")
                    is True
                    and payload.get("portable_metadata_migration", {}).get(
                        "checkpoint_and_scores_unchanged"
                    )
                    is True
                )
        aggregate_checks = {
            "status_completed": aggregate.get("status") == "completed",
            "one_registered_method": {
                row.get("model_id") for row in result_rows
            }
            == {specification["model_id"]},
            "exact_training_seeds": sorted(results_by_seed) == sorted(seeds),
            "required_result_count": aggregate.get(
                "comparison_contract", {}
            ).get("required_result_count")
            == len(seeds),
            "result_hashes_match": result_hashes_match,
            "result_payloads_match": result_payloads_match,
        }
    passed = bool(
        tree["passed"]
        and portability["passed"]
        and observed_files == expected_files
        and bool(aggregate_checks)
        and all(aggregate_checks.values())
    )
    return {
        "name": name,
        "artifact_tree": tree,
        "expected_files": sorted(expected_files),
        "observed_files": sorted(observed_files),
        "exact_layout": observed_files == expected_files,
        "portability": portability,
        "aggregate_checks": aggregate_checks,
        "passed": passed,
    }


def _audit_reference_result(
    name: str,
    specification: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
) -> dict[str, Any]:
    """Audit either a three-seed prediction tree or a CEM result tree."""

    if specification.get("kind") != "result_tree":
        return _audit_reference_method(
            name,
            specification,
            repo_root=repo_root,
            full=full,
        )

    root_logical = str(specification["root"]).rstrip("/")
    tree = _audit_tree(
        root_logical,
        specification,
        repo_root=repo_root,
        full=full,
    )
    files = {
        key: _audit_file(
            f"{root_logical}/{specification[key]}",
            specification[f"{key}_sha256"],
            repo_root=repo_root,
        )
        for key in ("summary", "request", "file_manifest")
    }
    portability = _audit_portable_text(
        root_logical,
        kind="directory",
        repo_root=repo_root,
    )
    return {
        "name": name,
        "kind": "result_tree",
        "artifact_tree": tree,
        "files": files,
        "portability": portability,
        "passed": bool(
            tree["passed"]
            and all(row["passed"] for row in files.values())
            and portability["passed"]
        ),
    }


def _audit_training(
    release: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
) -> dict[str, Any]:
    training = release["training"]
    build_report_audit = _audit_file(
        training["build_report"],
        training["build_report_sha256"],
        repo_root=repo_root,
    )
    build_report_checks: dict[str, Any] = {}
    if build_report_audit["passed"]:
        report = json.loads(
            Path(build_report_audit["path"]).read_text(encoding="utf-8")
        )
        checks = report.get("checks", {})
        build_report_checks = {
            "identity": (
                report.get("benchmark")
                == "tworoom_hidden_passage_history3_training_data_v1"
            ),
            "formal_scale": report.get("scale") == "formal",
            "status_passed": (
                report.get("status") == "passed"
                and report.get("passed") is True
            ),
            "all_declared_checks_passed": (
                isinstance(checks, dict)
                and bool(checks)
                and all(value is True for value in checks.values())
            ),
        }
    groups = {}
    for name, row in training["groups"].items():
        files = {
            key: _audit_file(
                row[key],
                row[f"{key}_sha256"],
                repo_root=repo_root,
            )
            for key in ("catalog", "manifest", "report")
        }
        count_checks: dict[str, bool] = {}
        if files["catalog"]["passed"]:
            catalog = json.loads(
                Path(files["catalog"]["path"]).read_text(encoding="utf-8")
            )
            count_checks = {
                "train_scenarios": (
                    len(catalog["train"]["synthetic"])
                    == int(row["train_scenarios"])
                ),
                "validation_scenarios": (
                    len(catalog["val"]["synthetic"])
                    == int(row["validation_scenarios"])
                ),
                "test_scenarios": (
                    len(catalog["ood_test"]["synthetic"])
                    == int(row["test_scenarios"])
                ),
                "all_scenario_paths_exist": all(
                    resolve_contextworld_path(path, repo_root=repo_root).is_dir()
                    for split in ("train", "val", "ood_test")
                    for path in catalog[split]["synthetic"]
                ),
            }
        groups[str(name)] = {
            "files": files,
            "count_checks": count_checks,
            "passed": bool(
                all(value["passed"] for value in files.values())
                and bool(count_checks)
                and all(count_checks.values())
            ),
        }
    tree = _audit_tree(
        training["artifact_tree"]["root"],
        training["artifact_tree"],
        repo_root=repo_root,
        full=full,
    )
    portability = _audit_portable_text(
        training["artifact_tree"]["root"],
        kind="directory",
        repo_root=repo_root,
    )
    passed = bool(
        build_report_audit["passed"]
        and bool(build_report_checks)
        and all(build_report_checks.values())
        and all(value["passed"] for value in groups.values())
        and tree["passed"]
        and portability["passed"]
    )
    return {
        "build_report": build_report_audit,
        "build_report_checks": build_report_checks,
        "groups": groups,
        "artifact_tree": tree,
        "portability": portability,
        "passed": passed,
    }


def _audit_evaluation(
    release: dict[str, Any],
    *,
    repo_root: Path,
    full: bool,
) -> dict[str, Any]:
    evaluation = release["evaluation"]
    files = {
        key: _audit_file(
            evaluation[key],
            evaluation[f"{key}_sha256"],
            repo_root=repo_root,
        )
        for key in (
            "catalog",
            "build_report",
            "training_exclusion_manifest",
            "normalizer",
            "protocol",
        )
    }
    catalog_checks: dict[str, bool] = {}
    payload_hashes_verified = 0
    if files["catalog"]["passed"]:
        catalog = json.loads(
            Path(files["catalog"]["path"]).read_text(encoding="utf-8")
        )
        payloads_exist = True
        payload_hashes_pass = True
        for bundle in catalog["bundles"]:
            payload = resolve_contextworld_path(
                bundle["payload"],
                repo_root=repo_root,
            )
            payloads_exist = payloads_exist and payload.is_file()
            if full and payload.is_file():
                payload_hashes_verified += 1
                payload_hashes_pass = bool(
                    payload_hashes_pass
                    and file_sha256(payload) == bundle["payload_sha256"]
                )
        summary = catalog.get("summary", {})
        catalog_checks = {
            "frozen_before_scoring": (
                catalog.get("status") == "frozen_before_model_scoring"
            ),
            "content_manifest": (
                catalog.get("content_manifest_sha256")
                == evaluation["content_manifest_sha256"]
            ),
            "query_count": (
                len(catalog["bundles"]) == int(evaluation["queries"])
            ),
            "eval_seeds": (
                list(summary.get("eval_seeds", []))
                == list(evaluation["eval_seeds"])
            ),
            "queries_per_seed": (
                int(summary.get("unique_queries_per_eval_seed", -1))
                == int(evaluation["queries_per_eval_seed"])
            ),
            "all_payloads_exist": payloads_exist,
            "payload_hashes": payload_hashes_pass,
        }
        if full and all(catalog_checks.values()):
            dataset = DoorICLEvalDataset(
                release=release,
                repo_root=repo_root,
            )
            catalog_checks["full_array_audit"] = bool(
                dataset.catalog_audit["passed"]
                and dataset.is_full_protocol
            )
    exclusion_checks: dict[str, bool] = {}
    if files["training_exclusion_manifest"]["passed"]:
        manifest = json.loads(
            Path(
                files["training_exclusion_manifest"]["path"]
            ).read_text(encoding="utf-8")
        )
        exclusion_checks = {
            "query_count": (
                int(manifest.get("query_count", -1))
                == int(evaluation["queries"])
            ),
            "content_manifest": (
                manifest.get("content_manifest_sha256")
                == evaluation["content_manifest_sha256"]
            ),
            "eval_only_door_positions": (
                list(manifest.get("eval_only_door_positions", []))
                == list(evaluation["eval_only_door_positions"])
            ),
        }
    tree = _audit_tree(
        evaluation["artifact_tree"]["root"],
        evaluation["artifact_tree"],
        repo_root=repo_root,
        full=full,
    )
    evaluation_root = resolve_contextworld_path(
        evaluation["artifact_tree"]["root"], repo_root=repo_root
    )
    observed_top_level = (
        {path.name for path in evaluation_root.iterdir()}
        if evaluation_root.is_dir()
        else set()
    )
    layout_checks = {
        "exact_public_test_top_level": (
            observed_top_level == _PUBLIC_TEST_TOP_LEVEL
        ),
        "historical_model_results_absent": not any(
            name in observed_top_level
            for name in (
                "aggregate.json",
                "results",
                "results_summary.json",
                "results_summary.md",
            )
        ),
    }
    portability = _audit_portable_text(
        evaluation["artifact_tree"]["root"],
        kind="directory",
        repo_root=repo_root,
    )
    passed = bool(
        all(value["passed"] for value in files.values())
        and bool(catalog_checks)
        and all(catalog_checks.values())
        and bool(exclusion_checks)
        and all(exclusion_checks.values())
        and tree["passed"]
        and all(layout_checks.values())
        and portability["passed"]
    )
    return {
        "files": files,
        "catalog_checks": catalog_checks,
        "training_exclusion_checks": exclusion_checks,
        "artifact_tree": tree,
        "layout_checks": layout_checks,
        "observed_top_level": sorted(observed_top_level),
        "portability": portability,
        "payload_hashes_verified": payload_hashes_verified,
        "passed": passed,
    }


def audit_door_icl_release(
    *,
    release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Verify the local public train/Validation package without model imports."""

    root = (repo_root or repository_root()).resolve()
    release = load_door_icl_release(release_config)
    code = [
        _audit_file(path, sha256, repo_root=root)
        for path, sha256 in release["runtime"]["contextworld"][
            "source_sha256"
        ].items()
    ]
    training = _audit_training(release, repo_root=root, full=full)
    evaluation = _audit_evaluation(release, repo_root=root, full=full)
    reference_results = {
        name: _audit_reference_result(
            name,
            row,
            repo_root=root,
            full=full,
        )
        for name, row in release["reference_results"].items()
    }
    initialization = {
        key: _audit_file(
            release["training"]["initialization"][key],
            release["training"]["initialization"][f"{key}_sha256"],
            repo_root=root,
        )
        for key in ("checkpoint", "checkpoint_config")
    }
    initialization_portability = _audit_portable_text(
        release["training"]["initialization"]["checkpoint_config"],
        kind="file",
        repo_root=root,
    )
    training_report_path = Path(training["build_report"]["path"])
    evaluation_report_path = Path(
        evaluation["files"]["build_report"]["path"]
    )
    training_report = json.loads(
        training_report_path.read_text(encoding="utf-8")
    )
    evaluation_report = json.loads(
        evaluation_report_path.read_text(encoding="utf-8")
    )
    training_pairs = training_report["pair_and_split_audit"]
    evaluation_checks = evaluation_report["checks"]
    causal_data = audit_causal_data_contract(
        component_id="door",
        evidence_scope=(
            "all 8,960 paired training templates and all 300 Public Test "
            "query bundles"
        ),
        continuous_environment_trajectory=bool(
            training_pairs["passed"]
            and evaluation_checks["all_physics_checks_passed"]
        ),
        state_installations_after_x0=0,
        query_simulator_recreated=False,
        maximum_query_state_gap=0.0,
        query_state_tolerance=0.0,
        query_pixels_exact=bool(
            evaluation_checks["all_physics_checks_passed"]
        ),
        query_actions_exact=bool(
            evaluation_checks["all_physics_checks_passed"]
            and training_pairs["checks"][
                "action_signature_balanced_across_rules"
            ]
        ),
        history_effect_present=bool(
            evaluation_checks["all_physics_checks_passed"]
        ),
        true_future_effect_present=bool(
            evaluation_checks["all_physics_checks_passed"]
        ),
        x0_policy="shared_visible_start",
        x0_static_leakage_check_passed=bool(
            training_pairs["checks"][
                "action_signature_only_accuracy_is_chance"
            ]
        ),
        evidence=(
            str(training_report_path),
            str(evaluation_report_path),
            evaluation["files"]["protocol"]["path"],
        ),
    )
    technical_passed = bool(
        all(value["passed"] for value in code)
        and training["passed"]
        and evaluation["passed"]
        and all(value["passed"] for value in reference_results.values())
        and all(value["passed"] for value in initialization.values())
        and initialization_portability["passed"]
        and causal_data["passed"]
    )
    distribution = release["distribution"]
    public_distribution_ready = bool(
        distribution["code_license_status"] == "declared"
        and distribution["generated_data_license_status"] == "declared"
        and distribution["public_download_status"] == "configured"
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if technical_passed else "failed",
        "release_config": str(release["_config_path"]),
        "artifact_root_override": os.environ.get(
            "CONTEXTWORLD_ARTIFACT_ROOT"
        ),
        "full_content_hash_audit": bool(full),
        "contextworld_code": code,
        "training": training,
        "evaluation": evaluation,
        "causal_data_contract": causal_data,
        "reference_results": reference_results,
        "reference_initialization": initialization,
        "reference_initialization_portability": (
            initialization_portability
        ),
        "technical_release_candidate_passed": technical_passed,
        "public_distribution_ready": public_distribution_ready,
        "distribution_blockers": [
            name
            for name, passed in (
                (
                    "ContextWorld source license declaration",
                    distribution["code_license_status"] == "declared",
                ),
                (
                    "ContextWorld-generated data license declaration",
                    distribution["generated_data_license_status"]
                    == "declared",
                ),
                (
                    "public artifact download URL",
                    distribution["public_download_status"] == "configured",
                ),
            )
            if not passed
        ],
        "sealed_test_included": False,
        "passed": technical_passed,
    }


def door_icl_training_plan(
    recipe: str,
    *,
    training_seed: int,
    release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_door_icl_release(release_config)
    recipes = release["training"]["recipes"]
    if recipe not in recipes:
        raise KeyError(f"Unknown recipe {recipe!r}; available={sorted(recipes)}")
    row = recipes[recipe]
    allowed_seeds = tuple(
        int(value) for value in row["training_seeds"]
    )
    if int(training_seed) not in allowed_seeds:
        raise ValueError(
            f"Training seed must be one of {list(allowed_seeds)}"
        )
    config = resolve_contextworld_path(row["config"], repo_root=root)
    if file_sha256(config) != row["config_sha256"]:
        raise RuntimeError(f"Training config hash mismatch: {config}")
    command = (
        f"TRAINING_SEED={int(training_seed)} "
        "logger_backend=none "
        "bash scripts/run_h3_hidden_passage_train.sh "
        f"{row['shell_variant']} formal"
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed",
        "recipe": recipe,
        "display_name": row["display_name"],
        "model_id": row["model_id"],
        "adapter": row["adapter"],
        "training_seed": int(training_seed),
        "training_config": str(config),
        "training_config_sha256": row["config_sha256"],
        "synthetic_group": "passage_mixed",
        "optimizer_steps": int(row["optimizer_steps"]),
        "effective_global_batch": int(row["effective_global_batch"]),
        "total_logical_draws": int(row["total_logical_draws"]),
        "command": command,
        "requires_upstream_original_h5": True,
        "original_h5": release["training"]["upstream_original_h5"],
        "initialization": release["training"]["initialization"],
    }


def export_door_icl_artifacts(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_DOOR_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
) -> dict[str, Any]:
    """Create a two-entry bundle: one README and one benchmark-data tree.

    The benchmark directory contains machine-readable metadata in addition to
    arrays and tables.  Users only need to read the top-level README.  Exporting
    a bundle does not grant redistribution rights.
    """

    if mode not in {"copy", "symlink"}:
        raise ValueError("Export mode must be 'copy' or 'symlink'")
    root = (repo_root or repository_root()).resolve()
    release = load_door_icl_release(release_config)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    benchmark_root = destination / "benchmark"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    entries = door_icl_export_entries(release)
    portability = [
        _audit_portable_text(
            logical_path,
            kind=kind,
            repo_root=root,
        )
        for logical_path, kind in entries
    ]
    release_portability = _audit_portable_text(
        str(Path(release["_config_path"])),
        kind="file",
        repo_root=root,
    )
    if not (
        all(row["passed"] for row in portability)
        and release_portability["passed"]
    ):
        raise RuntimeError(
            "Door export contains machine-specific paths or symlinks"
        )
    seen: set[str] = set()
    inventory = []
    for logical_path, kind in entries:
        if logical_path in seen:
            continue
        seen.add(logical_path)
        source = resolve_contextworld_path(logical_path, repo_root=root)
        relative = Path(logical_path)
        if relative.parts and relative.parts[0] == "artifacts":
            relative = Path(*relative.parts[1:])
        target = benchmark_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, target)
            inventory.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "bytes": source.stat().st_size,
                    "sha256": file_sha256(source),
                }
            )
        else:
            if not source.is_dir():
                raise FileNotFoundError(source)
            if mode == "symlink":
                target.symlink_to(source, target_is_directory=True)
            else:
                shutil.copytree(source, target)
            inventory.append(
                {
                    "logical_path": logical_path,
                    "kind": kind,
                    "export_mode": mode,
                }
            )

    release_path = benchmark_root / "release.yaml"
    shutil.copy2(
        Path(release["_config_path"]),
        release_path,
    )
    guide = root / "docs/ContextWorld_ICL_Benchmark.md"
    readme_path = destination / "README.md"
    shutil.copy2(guide, readme_path)
    payload = {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed",
        "release_kind": "local_technical_release_candidate",
        "bundle_layout": {
            "readme": "README.md",
            "benchmark_root": "benchmark",
            "release_config": "benchmark/release.yaml",
            "inventory": "benchmark/inventory.json",
        },
        "mode": mode,
        "sealed_test_included": False,
        "redistribution_granted_by_export": False,
        "distribution": release["distribution"],
        "portability_verified": True,
        "entries": inventory,
    }
    inventory_path = benchmark_root / "inventory.json"
    inventory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **payload,
        "destination": str(destination),
        "readme": str(readme_path),
        "benchmark_root": str(benchmark_root),
        "release_config": str(release_path),
        "inventory": str(inventory_path),
    }


__all__ = [
    "DEFAULT_DOOR_RELEASE_CONFIG",
    "DoorICLEvalDataset",
    "DoorICLEvalExample",
    "RELEASE_ID",
    "audit_door_icl_release",
    "door_icl_export_entries",
    "door_icl_training_plan",
    "export_door_icl_artifacts",
    "load_door_icl_release",
]
