from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from contextworld.evaluation.hidden_passage_h3_data import (
    audit_hidden_passage_release_assets,
    lexical_contextworld_path,
    require_lexical_containment,
    require_safe_directory,
    require_safe_regular_file,
    shard_completion_marker_path,
    verify_hidden_passage_shard_completion,
)
from contextworld.paths import resolve_contextworld_path

from .groups import (
    ConcatenatedDataset,
    LogicalGroupDataset,
    ScenarioBalancedDataset,
)
from .episode_split import (
    clip_subset_indices,
    episode_ids_sha256,
    episode_row_indices,
    partition_episode_ids,
)


MODEL_COLUMNS = ("pixels", "action", "proprio")
CATALOG_BY_GROUP = {
    "speed": "speed",
    "speed_single_v2": "speed_single_v2",
    "speed_multi_v2": "speed_multi_v2",
    "synth5_matched": "synth5_matched",
    "door": "door",
    "door_fixed49_v2": "door_fixed49_v2",
    "door_multi_v2": "door_multi_v2",
    "passage_passable": "passage_passable",
    "passage_blocked": "passage_blocked",
    "passage_mixed": "passage_mixed",
    "passage_tiny_overfit": "passage_tiny_overfit",
    "action_delay_single": "action_delay_single",
    "action_delay_multi": "action_delay_multi",
    "speed_door_composition": "speed_door_composition",
}
PASSAGE_GROUPS = (
    "passage_passable",
    "passage_blocked",
    "passage_mixed",
)


class UnbiasedZScoreScaler:
    """Picklable z-score scaler matching historical M_orig training stats."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)

    def transform(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=value.dtype, device=value.device)
            std = torch.as_tensor(self.std, dtype=value.dtype, device=value.device)
            return ((value - mean) / std).float()
        return (np.asarray(value) - self.mean) / self.std

    def inverse_transform(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=value.dtype, device=value.device)
            std = torch.as_tensor(self.std, dtype=value.dtype, device=value.device)
            return value * std + mean
        return np.asarray(value) * self.std + self.mean

    def __call__(self, value):
        return self.transform(value)


@dataclass
class TwoRoomGroupedData:
    train: Any
    val: Any
    metadata: dict[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    """Hash every regular file in a synthetic dataset directory."""

    root = Path(path)
    digest = hashlib.sha256()
    files = []
    for value in root.rglob("*"):
        if value.is_symlink():
            raise ValueError(
                f"Synthetic dataset contains a symlink: {value}"
            )
        if value.is_file():
            files.append(value)
    for value in sorted(
        files,
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = value.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _passage_release_root_for_catalog(catalog_path: Path) -> Path:
    catalog = require_safe_regular_file(catalog_path)
    release_root = require_safe_directory(catalog.parent.parent)
    expected_catalog_root = release_root / "catalogs"
    require_lexical_containment(catalog, expected_catalog_root)
    require_safe_directory(expected_catalog_root)
    return release_root


def _resolve_passage_declared_path(
    value: str | Path,
    *,
    repo_root: Path,
    release_root: Path,
    leaf_kind: str,
    required_subtree: str | None,
) -> Path:
    """Validate lexical containment and lstat before canonicalization."""

    lexical = lexical_contextworld_path(value, repo_root=repo_root)
    subtree = require_safe_directory(
        release_root
        if required_subtree is None
        else release_root / required_subtree
    )
    require_lexical_containment(lexical, subtree)
    if leaf_kind == "directory":
        require_safe_directory(lexical, containment_root=release_root)
    elif leaf_kind == "regular_file":
        require_safe_regular_file(lexical, containment_root=release_root)
    else:
        raise ValueError(f"Unsupported passage leaf kind: {leaf_kind}")
    canonical = lexical.resolve(strict=True)
    canonical_root = release_root.resolve(strict=True)
    require_lexical_containment(canonical, canonical_root)
    return canonical


def _passage_catalog_path(
    value: str | Path,
    *,
    repo_root: Path,
) -> Path:
    lexical = lexical_contextworld_path(value, repo_root=repo_root)
    require_safe_regular_file(lexical)
    release_root = require_safe_directory(lexical.parent.parent)
    require_lexical_containment(lexical, release_root / "catalogs")
    return lexical.resolve(strict=True)


def _load_frozen_normalizer(
    specification: dict[str, Any],
    *,
    repo_root: Path,
    split_metadata: dict[str, Any],
) -> tuple[dict[str, UnbiasedZScoreScaler], dict[str, Any]]:
    """Load one externally frozen normalizer for every compared model."""

    path = resolve_contextworld_path(
        specification["path"],
        repo_root=repo_root,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha256 = str(specification["sha256"])
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Frozen normalizer hash mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_protocol = str(
        specification.get(
            "protocol",
            "tworoom_original_train_s3072_unbiased_zscore_v1",
        )
    )
    expected_scope = str(
        specification.get(
            "statistics_scope",
            "original_9000_train_episodes_only",
        )
    )
    if payload.get("protocol") != expected_protocol:
        raise ValueError(
            f"Frozen normalizer protocol differs in {path}"
        )
    if payload.get("statistics_scope") != expected_scope:
        raise ValueError(
            f"Frozen normalizer statistics scope differs in {path}"
        )
    expected_episode_hash = split_metadata.get(
        "train_episode_ids_sha256"
    )
    if (
        expected_episode_hash is not None
        and payload.get("train_episode_ids_sha256")
        != expected_episode_hash
    ):
        raise ValueError(
            "Frozen normalizer and original training split differ: "
            f"{path}"
        )

    scalers: dict[str, UnbiasedZScoreScaler] = {}
    column_metadata: dict[str, Any] = {}
    for column in ("action", "proprio"):
        try:
            values = payload["columns"][column]
            mean = np.asarray(values["mean"], dtype=np.float64)[None, :]
            std = np.asarray(
                values["std_unbiased"],
                dtype=np.float64,
            )[None, :]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Frozen normalizer has invalid {column!r} statistics: {path}"
            ) from exc
        if (
            mean.shape != (1, 2)
            or std.shape != (1, 2)
            or not np.isfinite(mean).all()
            or not np.isfinite(std).all()
            or not (std > 0).all()
        ):
            raise ValueError(
                f"Frozen normalizer has invalid {column!r} values: {path}"
            )
        scalers[column] = UnbiasedZScoreScaler(mean, std)
        column_metadata[column] = {
            "mean": mean.tolist(),
            "std_unbiased": std.tolist(),
            "valid_rows": int(values["valid_rows"]),
        }

    return scalers, {
        "mode": "frozen_original_training_split",
        "path": str(path),
        "sha256": observed_sha256,
        "protocol": expected_protocol,
        "statistics_scope": expected_scope,
        "train_episode_ids_sha256": payload.get(
            "train_episode_ids_sha256"
        ),
        "source_sha256": payload.get("source_sha256"),
        "columns": column_metadata,
        "passed": True,
    }


def _load_training_exclusion_manifest(
    specification: dict[str, Any],
    *,
    repo_root: Path,
    expected_eval_only_door_positions: list[int],
) -> dict[str, Any]:
    """Verify the sealed validation identities before training can start."""

    path = resolve_contextworld_path(
        specification["path"],
        repo_root=repo_root,
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_sha256 = _sha256(path)
    expected_sha256 = str(specification["sha256"])
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Training exclusion manifest hash mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_content_sha256 = str(specification["content_sha256"])
    if payload.get("content_manifest_sha256") != expected_content_sha256:
        raise ValueError(
            "Training exclusion content hash mismatch: "
            f"expected={expected_content_sha256}, "
            f"observed={payload.get('content_manifest_sha256')}"
        )
    records = payload.get("query_records")
    expected_query_count = int(
        specification.get("query_count", 300)
    )
    if (
        payload.get("schema_version") != 1
        or not isinstance(records, list)
        or payload.get("query_count") != expected_query_count
        or len(records) != expected_query_count
    ):
        raise ValueError(
            f"Training exclusion manifest has invalid query coverage: {path}"
        )
    query_ids = [record.get("query_id") for record in records]
    template_ids = [record.get("template_id") for record in records]
    if (
        any(not isinstance(value, str) or not value for value in query_ids)
        or len(set(query_ids)) != expected_query_count
        or any(
            not isinstance(value, str) or not value
            for value in template_ids
        )
        or len(set(template_ids)) != expected_query_count
    ):
        raise ValueError(
            f"Training exclusion query/template identities are invalid: {path}"
        )
    expected_doors = {
        int(value) for value in expected_eval_only_door_positions
    }
    observed_doors = {
        int(value)
        for value in payload.get("eval_only_door_positions", [])
    }
    if observed_doors != expected_doors:
        raise ValueError(
            "Training exclusion door support differs from benchmark config: "
            f"expected={sorted(expected_doors)}, "
            f"observed={sorted(observed_doors)}"
        )
    return {
        "path": str(path),
        "sha256": observed_sha256,
        "content_sha256": expected_content_sha256,
        "query_count": expected_query_count,
        "unique_query_ids": len(set(query_ids)),
        "unique_template_ids": len(set(template_ids)),
        "eval_only_door_positions": sorted(observed_doors),
        "passed": True,
    }


def _load_formal_passage_build_report(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Bind training to one complete formal data build, not loose files."""

    specification = config["data"].get("formal_build_report")
    if not isinstance(specification, dict):
        raise ValueError(
            "Passage training requires data.formal_build_report"
        )
    catalog_roots = {
        _passage_release_root_for_catalog(
            _passage_catalog_path(value, repo_root=repo_root)
        )
        for group, value in config["data"]["catalogs"].items()
        if group in PASSAGE_GROUPS
    }
    if len(catalog_roots) != 1:
        raise ValueError(
            "Passage training catalogs do not share a release root: "
            f"{sorted(map(str, catalog_roots))}"
        )
    release_root = next(iter(catalog_roots))
    path = _resolve_passage_declared_path(
        specification["path"],
        repo_root=repo_root,
        release_root=release_root,
        leaf_kind="regular_file",
        required_subtree=None,
    )
    observed_sha256 = _sha256(path)
    expected_sha256 = str(specification["sha256"])
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Formal hidden-passage build_report hash mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_benchmark = str(specification["benchmark"])
    expected_scale = str(specification.get("scale", "formal"))
    if (
        payload.get("schema_version") != 1
        or payload.get("benchmark") != expected_benchmark
        or payload.get("scale") != expected_scale
        or payload.get("status") != "passed"
        or payload.get("passed") is not True
    ):
        raise ValueError(
            "Formal hidden-passage build_report identity/status failed: "
            f"path={path}, benchmark={payload.get('benchmark')!r}, "
            f"scale={payload.get('scale')!r}, "
            f"status={payload.get('status')!r}, "
            f"passed={payload.get('passed')!r}"
        )

    required_checks = (
        "all_shards_pass",
        "frozen_validation_exclusion_passes",
        "pair_and_split_audit_passes",
        "three_catalogs_are_same_source",
        "catalog_counts_are_exact",
        "model_columns_are_pixels_and_action_only",
        "catalogs_are_synthetic_only",
        "every_episode_is_exactly_one_h3_clip",
        "no_unreferenced_lance_shards",
        "all_shards_have_valid_completion_markers",
        "no_unreferenced_completion_markers",
    )
    checks = payload.get("checks")
    if (
        not isinstance(checks, dict)
        or any(checks.get(name) is not True for name in required_checks)
    ):
        raise ValueError(
            "Formal hidden-passage build_report checks failed: "
            f"{checks}"
        )

    expected_commit = str(config["stable_worldmodel"]["commit"])
    identity = payload.get("identity")
    if (
        not isinstance(identity, dict)
        or identity.get("stable_worldmodel_commit") != expected_commit
    ):
        raise ValueError(
            "Formal hidden-passage build runtime identity mismatch: "
            f"expected={expected_commit}, observed={identity}"
        )

    exclusion = payload.get("validation_exclusion_audit")
    expected_query_count = int(
        config["data"]["training_exclusion_manifest"].get(
            "query_count",
            300,
        )
    )
    if (
        not isinstance(exclusion, dict)
        or exclusion.get("passed") is not True
        or exclusion.get("selected_query_count") != expected_query_count
        or exclusion.get("selected_query_pixel_hash_overlap") != []
    ):
        raise ValueError(
            "Formal hidden-passage build exclusion/query-overlap gate failed: "
            f"{exclusion}"
        )

    artifacts = payload.get("artifacts_by_group")
    if not isinstance(artifacts, dict) or set(artifacts) != set(
        PASSAGE_GROUPS
    ):
        raise ValueError(
            "Formal hidden-passage build_report must list exactly the three "
            f"active passage groups, got {sorted(artifacts or {})}"
        )
    quality_groups = config["data_quality"]["groups"]
    active_artifacts: dict[str, Any] = {}
    expected_physical_shards = 0
    expected_physical_episodes = 0
    for group in PASSAGE_GROUPS:
        artifact = artifacts[group]
        quality = quality_groups[group]
        expected_counts = {
            "train": {
                "shards": int(quality["exact_train_scenarios"]),
                "episodes": int(quality["exact_train_clips"]),
                "clips": int(quality["exact_train_clips"]),
            },
            "val": {
                "shards": int(quality["exact_validation_scenarios"]),
                "episodes": int(quality["exact_validation_clips"]),
                "clips": int(quality["exact_validation_clips"]),
            },
            "test": {
                "shards": int(quality["exact_test_scenarios"]),
                "episodes": int(quality["exact_test_clips"]),
                "clips": int(quality["exact_test_clips"]),
            },
        }
        if artifact.get("counts") != expected_counts:
            raise ValueError(
                "Formal hidden-passage group counts differ: "
                f"group={group}, expected={expected_counts}, "
                f"observed={artifact.get('counts')}"
            )

        resolved = {}
        observed_hashes = {}
        for name, path_field, hash_field in (
            ("catalog", "catalog", "catalog_sha256"),
            ("manifest", "manifest", "manifest_sha256"),
            (
                "synthesis_report",
                "synthesis_report",
                "synthesis_report_sha256",
            ),
        ):
            artifact_path = _resolve_passage_declared_path(
                artifact[path_field],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="regular_file",
                required_subtree={
                    "catalog": "catalogs",
                    "manifest": "manifests",
                    "synthesis_report": "reports",
                }[name],
            )
            actual_hash = _sha256(artifact_path)
            if actual_hash != artifact.get(hash_field):
                raise ValueError(
                    "Formal hidden-passage active artifact hash mismatch: "
                    f"group={group}, artifact={name}, "
                    f"reported={artifact.get(hash_field)}, "
                    f"observed={actual_hash}"
                )
            resolved[name] = str(artifact_path)
            observed_hashes[name] = actual_hash

        configured_catalog = _passage_catalog_path(
            config["data"]["catalogs"][group],
            repo_root=repo_root,
        )
        if Path(resolved["catalog"]) != configured_catalog:
            raise ValueError(
                "Formal build_report catalog is not the active training "
                f"catalog: group={group}, reported={resolved['catalog']}, "
                f"configured={configured_catalog}"
            )
        for name, quality_key in (
            ("catalog", "required_catalog_sha256"),
            ("manifest", "required_manifest_sha256"),
            (
                "synthesis_report",
                "required_synthesis_report_sha256",
            ),
        ):
            frozen_hash = quality.get(quality_key)
            if (
                frozen_hash is not None
                and str(frozen_hash) != observed_hashes[name]
            ):
                raise ValueError(
                    "Formal build_report artifact differs from the group "
                    f"freeze: group={group}, artifact={name}"
                )
        active_artifacts[group] = {
            name: {
                "path": resolved[name],
                "sha256": observed_hashes[name],
            }
            for name in (
                "catalog",
                "manifest",
                "synthesis_report",
            )
        }
        active_artifacts[group]["counts"] = expected_counts
        if group != "passage_mixed":
            expected_physical_shards += sum(
                row["shards"] for row in expected_counts.values()
            )
            expected_physical_episodes += sum(
                row["episodes"] for row in expected_counts.values()
            )

    rows_per_episode = int(payload["history3"]["rows_per_episode"])
    physical_counts = {
        "shards": expected_physical_shards,
        "episodes": expected_physical_episodes,
        "rows": expected_physical_episodes * rows_per_episode,
    }
    observed_physical_counts = {
        "shards": int(payload.get("physical_shards", -1)),
        "episodes": int(payload.get("physical_episodes", -1)),
        "rows": int(payload.get("physical_rows", -1)),
    }
    if observed_physical_counts != physical_counts:
        raise ValueError(
            "Formal hidden-passage physical counts differ: "
            f"expected={physical_counts}, "
            f"observed={observed_physical_counts}"
        )

    return {
        "required": True,
        "path": str(path),
        "sha256": observed_sha256,
        "benchmark": expected_benchmark,
        "scale": expected_scale,
        "status": "passed",
        "checks": {name: True for name in required_checks},
        "active_artifacts": active_artifacts,
        "physical_counts": physical_counts,
        "passed": True,
    }


def _resolve_catalog_entries(
    entries: list[str], *, repo_root: Path, require_directories: bool = True
) -> list[Path]:
    paths = []
    for entry in entries:
        path = resolve_contextworld_path(entry, repo_root=repo_root)
        if require_directories and not path.is_dir():
            raise FileNotFoundError(path)
        paths.append(path)
    return sorted(paths)


def _catalog_paths(
    catalog_path: Path,
    section: str,
    *,
    repo_root: Path,
) -> list[Path]:
    require_safe_regular_file(catalog_path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    raw = catalog[section]["synthetic"]
    if str(catalog.get("benchmark", "")).startswith(
        "tworoom_hidden_passage_history3"
    ):
        release_root = _passage_release_root_for_catalog(catalog_path)
        return sorted(
            _resolve_passage_declared_path(
                entry,
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="directory",
                required_subtree="tables",
            )
            for entry in raw
        )
    return _resolve_catalog_entries(raw, repo_root=repo_root)


def _validate_complete_synthesis_report(
    synthesis_report: dict[str, Any],
    *,
    report_path: Path,
    catalog_path: Path,
    manifest_path: Path,
    expected_scenario_ids: set[str],
) -> dict[str, Any]:
    """Require a completed, loader-validated synthesis collection."""

    if not isinstance(synthesis_report, dict):
        raise ValueError(f"Invalid synthesis report mapping: {report_path}")
    if synthesis_report.get("passed") is not True:
        raise ValueError(f"Synthesis report did not pass: {report_path}")
    if synthesis_report.get("compile_only") is not False:
        raise ValueError(
            f"Synthesis report is not a completed collection: {report_path}; "
            f"compile_only={synthesis_report.get('compile_only')!r}"
        )
    if synthesis_report.get("preflight_passed") is not True:
        raise ValueError(
            f"Synthesis preflight did not pass: {report_path}"
        )
    loader_compatibility = synthesis_report.get("loader_compatibility")
    if (
        not isinstance(loader_compatibility, dict)
        or loader_compatibility.get("passed") is not True
    ):
        raise ValueError(
            f"Synthesis loader compatibility did not pass: {report_path}"
        )

    declared_paths = {
        "catalog": catalog_path,
        "manifest": manifest_path,
    }
    for field, expected_path in declared_paths.items():
        declared = synthesis_report.get(field)
        if not isinstance(declared, str) or not declared:
            raise ValueError(
                f"Synthesis report is missing {field}: {report_path}"
            )
        if Path(declared).expanduser().resolve() != expected_path.resolve():
            raise ValueError(
                f"Synthesis report {field} mismatch: {report_path}; "
                f"declared={declared}, expected={expected_path}"
            )

    scenario_results = synthesis_report.get("scenarios")
    if not isinstance(scenario_results, list):
        raise ValueError(
            f"Synthesis report is missing scenario results: {report_path}"
        )
    report_scenario_ids = []
    for index, scenario in enumerate(scenario_results):
        if not isinstance(scenario, dict):
            raise ValueError(
                f"Invalid synthesis scenario result at index {index}: "
                f"{report_path}"
            )
        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError(
                f"Synthesis scenario result has no scenario_id at index "
                f"{index}: {report_path}"
            )
        if scenario.get("passed") is not True:
            raise ValueError(
                f"Synthesis scenario did not pass: {scenario_id} in "
                f"{report_path}"
            )
        report_scenario_ids.append(scenario_id)
    if (
        len(report_scenario_ids) != len(set(report_scenario_ids))
        or set(report_scenario_ids) != expected_scenario_ids
    ):
        raise ValueError(
            f"Synthesis report/catalog scenario sets differ: {report_path}; "
            f"expected={len(expected_scenario_ids)}, "
            f"reported={len(report_scenario_ids)}, "
            f"catalog_only={sorted(expected_scenario_ids - set(report_scenario_ids))}, "
            f"report_only={sorted(set(report_scenario_ids) - expected_scenario_ids)}"
        )

    collection_status = synthesis_report.get("collection_status")
    if not isinstance(collection_status, dict):
        raise ValueError(
            f"Synthesis report is missing collection status: {report_path}"
        )
    collection_ids = set(collection_status)
    if collection_ids != expected_scenario_ids:
        raise ValueError(
            f"Synthesis collection/catalog scenario sets differ: "
            f"{report_path}; expected={len(expected_scenario_ids)}, "
            f"reported={len(collection_status)}, "
            f"catalog_only={sorted(expected_scenario_ids - collection_ids)}, "
            f"report_only={sorted(collection_ids - expected_scenario_ids)}"
        )
    incomplete = {
        scenario_id: status
        for scenario_id, status in collection_status.items()
        if status not in {"collected", "reused"}
    }
    if incomplete:
        raise ValueError(
            f"Synthesis scenarios are not collected: {report_path}; "
            f"statuses={incomplete}"
        )

    return {
        "required": True,
        "compile_only": False,
        "preflight_passed": True,
        "loader_compatibility_passed": True,
        "scenario_results": len(report_scenario_ids),
        "collection_status_entries": len(collection_status),
        "collection_status_counts": dict(
            sorted(Counter(collection_status.values()).items())
        ),
        "passed": True,
    }


def _catalog_split_audit(
    catalog_path: Path,
    *,
    swm: Any | None = None,
    repo_root: Path,
    expected_stablewm_commit: str | None,
    require_complete_synthesis_report: bool = False,
    expected_split_scenario_counts: dict[str, int] | None = None,
    factor_support_contract: dict[str, Any] | None = None,
    required_artifact_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    require_safe_regular_file(catalog_path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported catalog schema in {catalog_path}: "
            f"{catalog.get('schema_version')}"
        )

    passage_catalog = str(catalog.get("benchmark", "")).startswith(
        "tworoom_hidden_passage_history3"
    )
    catalog_release_root = (
        _passage_release_root_for_catalog(catalog_path)
        if passage_catalog
        else None
    )
    diagnostic_source_release = catalog.get(
        "diagnostic_source_passage_release_root"
    )
    if diagnostic_source_release is not None:
        if passage_catalog or catalog.get("diagnostic_only") is not True:
            raise ValueError(
                "A diagnostic source passage release is allowed only for "
                "an explicitly diagnostic, non-formal catalog"
            )
        record_release_root = require_safe_directory(
            resolve_contextworld_path(
                diagnostic_source_release,
                repo_root=repo_root,
            )
        )
    else:
        record_release_root = catalog_release_root

    def resolve_entries(entries: list[str]) -> list[Path]:
        if record_release_root is None:
            return _resolve_catalog_entries(entries, repo_root=repo_root)
        return sorted(
            _resolve_passage_declared_path(
                entry,
                repo_root=repo_root,
                release_root=record_release_root,
                leaf_kind="directory",
                required_subtree="tables",
            )
            for entry in entries
        )

    train_paths = resolve_entries(list(catalog["train"]["synthetic"]))
    validation_paths = resolve_entries(
        list(catalog["val"]["synthetic"])
    )
    test_entries = list(catalog.get("ood_test", {}).get("synthetic", []))
    test_paths = resolve_entries(test_entries)

    duplicates = {}
    for name, paths in {
        "train": train_paths,
        "validation": validation_paths,
        "test": test_paths,
    }.items():
        counts = Counter(paths)
        duplicates[name] = sorted(
            str(path) for path, count in counts.items() if count > 1
        )
    if any(duplicates.values()):
        raise ValueError(
            f"Catalog contains duplicate scenario paths in {catalog_path}: "
            f"{duplicates}"
        )

    train = set(train_paths)
    validation = set(validation_paths)
    test = set(test_paths)
    overlaps = {
        "train_validation": sorted(str(path) for path in train & validation),
        "train_test": sorted(str(path) for path in train & test),
        "validation_test": sorted(str(path) for path in validation & test),
    }
    if any(overlaps.values()):
        raise ValueError(
            f"Catalog split paths overlap in {catalog_path}: {overlaps}"
        )

    artifact_root = catalog_path.parent.parent
    manifest_path = artifact_root / "manifests" / f"{catalog_path.stem}.jsonl"
    report_path = artifact_root / "reports" / f"{catalog_path.stem}.json"
    if catalog_release_root is not None:
        require_safe_regular_file(
            manifest_path,
            containment_root=catalog_release_root,
        )
        require_safe_regular_file(
            report_path,
            containment_root=catalog_release_root,
        )
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if not report_path.is_file():
            raise FileNotFoundError(report_path)

    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                record = json.loads(line)
                if record.get("schema_version") != 1:
                    raise ValueError(
                        f"Unsupported manifest schema at "
                        f"{manifest_path}:{line_number}"
                    )
                records.append(record)
    if not records:
        raise ValueError(f"Empty synthesis manifest: {manifest_path}")

    manifest_paths = []
    scenario_ids = []
    fingerprints = []
    commits = set()
    manifest_codecs = set()
    record_by_path = {}
    storage_hashes_verified = 0
    for record in records:
        if record_release_root is not None:
            resolved = _resolve_passage_declared_path(
                record["output_path"],
                repo_root=repo_root,
                release_root=record_release_root,
                leaf_kind="directory",
                required_subtree="tables",
            )
        else:
            resolved = _resolve_catalog_entries(
                [record["output_path"]], repo_root=repo_root
            )[0]
        manifest_paths.append(resolved)
        scenario_ids.append(record["scenario_id"])
        fingerprints.append(record["fingerprint"])
        commits.add(record["stable_worldmodel_commit"])
        manifest_codecs.add(
            json.dumps(record.get("pixel_codec"), sort_keys=True)
        )
        if record.get("collection_status") not in {
            "collected",
            "reused",
        }:
            raise ValueError(
                "Manifest scenario is neither collected nor safely reused: "
                f"{record['scenario_id']}"
            )
        if resolved.stem != record["scenario_id"]:
            raise ValueError(
                f"Scenario id/path mismatch: {record['scenario_id']} vs {resolved}"
            )
        if not record["scenario_id"].endswith(record["fingerprint"][:10]):
            raise ValueError(
                f"Scenario fingerprint/path mismatch: {record['scenario_id']}"
            )
        expected_storage_sha256 = record.get("storage_sha256")
        if expected_storage_sha256 is not None:
            observed_storage_sha256 = _directory_sha256(resolved)
            if observed_storage_sha256 != str(expected_storage_sha256):
                raise ValueError(
                    "Synthetic dataset storage hash mismatch: "
                    f"scenario={record['scenario_id']}, "
                    f"expected={expected_storage_sha256}, "
                    f"observed={observed_storage_sha256}"
                )
            storage_hashes_verified += 1
        record_by_path[resolved] = record

    unique_fields = {
        "output_path": (manifest_paths, len(set(manifest_paths))),
        "scenario_id": (scenario_ids, len(set(scenario_ids))),
        "fingerprint": (fingerprints, len(set(fingerprints))),
    }
    non_unique = {
        name: len(values) - unique_count
        for name, (values, unique_count) in unique_fields.items()
        if len(values) != unique_count
    }
    if non_unique:
        raise ValueError(
            f"Manifest identifiers are not unique in {manifest_path}: "
            f"{non_unique}"
        )

    catalog_union = train | validation | test
    manifest_union = set(manifest_paths)
    if catalog_union != manifest_union:
        raise ValueError(
            f"Catalog/manifest scenario sets differ for {catalog_path}: "
            f"catalog_only={sorted(str(p) for p in catalog_union - manifest_union)}, "
            f"manifest_only={sorted(str(p) for p in manifest_union - catalog_union)}"
        )
    passage_records = [
        record
        for record in records
        if "passage.open" in dict(record.get("factors", {}))
    ]
    if passage_records:
        if swm is None:
            raise ValueError(
                "Passage catalog audit requires Stable-WorldModel to "
                "recompute every shard's logical content"
            )
        logical_shards = [
            _verify_passage_shard_logical_content(
                swm,
                record=record,
                repo_root=repo_root,
                release_root=record_release_root,
            )
            for record in passage_records
        ]
        logical_content_audit = {
            "required": True,
            "shards_verified": len(logical_shards),
            "episodes_verified": sum(
                int(row["episodes_verified"])
                for row in logical_shards
            ),
            "completion_markers_verified": sum(
                row["completion_marker"]["passed"]
                for row in logical_shards
            ),
            "completion_protocol": (
                logical_shards[0]["completion_marker"]["protocol"]
            ),
            "columns": logical_shards[0]["columns"],
            "passed": True,
        }
    else:
        logical_content_audit = {
            "required": False,
            "shards_verified": 0,
            "episodes_verified": 0,
            "passed": True,
        }
    section_contract = {
        "train": (train, "train", "train"),
        "validation": (validation, "val", "validation"),
        "test": (test, "test", "test"),
    }
    for section, (paths, expected_split, regime_prefix) in section_contract.items():
        invalid = []
        for path in paths:
            record = record_by_path[path]
            if (
                record.get("split") != expected_split
                or not str(record.get("regime", "")).startswith(regime_prefix)
            ):
                invalid.append(record["scenario_id"])
        if invalid:
            raise ValueError(
                f"Manifest split/regime mismatch for {section}: {invalid}"
            )

    observed_split_scenario_counts = {
        section: len(paths)
        for section, (paths, _, _) in section_contract.items()
    }
    expected_split_scenario_counts = dict(
        expected_split_scenario_counts or {}
    )
    split_count_mismatches = {
        section: {
            "expected": int(expected),
            "observed": int(observed_split_scenario_counts.get(section, -1)),
        }
        for section, expected in expected_split_scenario_counts.items()
        if observed_split_scenario_counts.get(section) != int(expected)
    }
    if split_count_mismatches:
        raise ValueError(
            f"Catalog exact split counts failed for {catalog_path}: "
            f"{split_count_mismatches}"
        )

    factor_support_audit: dict[str, Any] = {
        "required": False,
        "passed": True,
    }
    if factor_support_contract:
        factor_key = str(factor_support_contract["factor"])
        expected_by_split = {
            str(section): list(values)
            for section, values in factor_support_contract[
                "expected_by_split"
            ].items()
        }
        observed_by_split: dict[str, list[Any]] = {}
        mismatches = {}
        for section, expected_values in expected_by_split.items():
            if section not in section_contract:
                raise ValueError(
                    f"Unknown factor-support split {section!r} for {catalog_path}"
                )
            paths = section_contract[section][0]
            observed_values = []
            for path in paths:
                factors = record_by_path[path].get("factors")
                if not isinstance(factors, dict) or factor_key not in factors:
                    raise ValueError(
                        f"Manifest record is missing factor {factor_key!r}: "
                        f"{record_by_path[path].get('scenario_id')}"
                    )
                observed_values.append(factors[factor_key])
            expected_map = {
                json.dumps(value, sort_keys=True): value
                for value in expected_values
            }
            observed_map = {
                json.dumps(value, sort_keys=True): value
                for value in observed_values
            }
            observed_by_split[section] = [
                observed_map[key] for key in sorted(observed_map)
            ]
            if set(observed_map) != set(expected_map):
                mismatches[section] = {
                    "expected": [expected_map[key] for key in sorted(expected_map)],
                    "observed": observed_by_split[section],
                }
        if mismatches:
            raise ValueError(
                f"Catalog factor support failed for {catalog_path}: "
                f"factor={factor_key}, mismatches={mismatches}"
            )
        factor_support_audit = {
            "required": True,
            "factor": factor_key,
            "expected_by_split": expected_by_split,
            "observed_by_split": observed_by_split,
            "passed": True,
        }

    if expected_stablewm_commit is not None and commits != {
        expected_stablewm_commit
    }:
        raise ValueError(
            f"Synthesis runtime commit mismatch for {catalog_path}: "
            f"observed={sorted(commits)}, expected={expected_stablewm_commit}"
        )
    catalog_codec = catalog.get("pixel_codec")
    expected_codec_set = {json.dumps(catalog_codec, sort_keys=True)}
    if catalog_codec is None or manifest_codecs != expected_codec_set:
        raise ValueError(
            f"Catalog/manifest pixel codec mismatch for {catalog_path}: "
            f"catalog={catalog_codec}, manifest={sorted(manifest_codecs)}"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        synthesis_report = json.load(handle)
    if require_complete_synthesis_report:
        synthesis_report_gate = _validate_complete_synthesis_report(
            synthesis_report,
            report_path=report_path,
            catalog_path=catalog_path,
            manifest_path=manifest_path,
            expected_scenario_ids=set(scenario_ids),
        )
    else:
        if not synthesis_report.get("passed"):
            raise ValueError(f"Synthesis report did not pass: {report_path}")
        synthesis_report_gate = {
            "required": False,
            "passed": True,
        }

    artifact_hashes = {
        "catalog": _sha256(catalog_path),
        "manifest": _sha256(manifest_path),
        "synthesis_report": _sha256(report_path),
    }
    required_artifact_hashes = dict(required_artifact_hashes or {})
    hash_mismatches = {
        name: {"expected": expected, "observed": artifact_hashes.get(name)}
        for name, expected in required_artifact_hashes.items()
        if artifact_hashes.get(name) != expected
    }
    if hash_mismatches:
        raise ValueError(
            f"Frozen synthesis artifact hash mismatch for {catalog_path}: "
            f"{hash_mismatches}"
        )

    return {
        "catalog_sha256": artifact_hashes["catalog"],
        "manifest": str(manifest_path),
        "manifest_sha256": artifact_hashes["manifest"],
        "synthesis_report": str(report_path),
        "synthesis_report_sha256": artifact_hashes["synthesis_report"],
        "required_artifact_hashes": required_artifact_hashes,
        "exact_split_scenario_counts": {
            "expected": expected_split_scenario_counts,
            "observed": observed_split_scenario_counts,
            "passed": True,
        },
        "factor_support": factor_support_audit,
        "logical_content_audit": logical_content_audit,
        "synthesis_report_gate": synthesis_report_gate,
        "stable_worldmodel_commits": sorted(commits),
        "pixel_codec": catalog_codec,
        "train_scenarios": len(train),
        "validation_scenarios": len(validation),
        "test_scenarios": len(test),
        "unique_scenario_ids": len(set(scenario_ids)),
        "unique_fingerprints": len(set(fingerprints)),
        "storage_hashes_verified": storage_hashes_verified,
        "duplicate_paths": duplicates,
        "path_overlap": overlaps,
        "passed": True,
    }


def _catalog_factors_by_path(
    catalog_path: Path, *, repo_root: Path
) -> dict[Path, dict[str, Any]]:
    artifact_root = catalog_path.parent.parent
    manifest_path = artifact_root / "manifests" / f"{catalog_path.stem}.jsonl"
    output = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = _resolve_catalog_entries(
                [record["output_path"]], repo_root=repo_root
            )[0]
            output[path] = dict(record["factors"])
    return output


def _manifest_records_by_seed_group(
    catalog_path: Path, *, repo_root: Path
) -> dict[str, dict[str, Any]]:
    artifact_root = catalog_path.parent.parent
    manifest_path = artifact_root / "manifests" / f"{catalog_path.stem}.jsonl"
    records: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            seed_group = record.get("seed_group")
            if not isinstance(seed_group, str) or not seed_group:
                raise ValueError(
                    f"Missing seed_group at {manifest_path}:{line_number}"
                )
            if seed_group in records:
                raise ValueError(
                    f"Duplicate seed_group {seed_group!r} in {manifest_path}"
                )
            records[seed_group] = record
    if not records:
        raise ValueError(f"Empty synthesis manifest: {manifest_path}")
    return records


def _manifest_records_by_pair(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Index one passage manifest by split and stable physical pair id."""

    artifact_root = catalog_path.parent.parent
    manifest_path = artifact_root / "manifests" / f"{catalog_path.stem}.jsonl"
    require_safe_regular_file(
        manifest_path,
        containment_root=artifact_root,
    )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = record.get("pair_id", record.get("seed_group"))
            split = record.get("split")
            if (
                not isinstance(pair_id, str)
                or not pair_id
                or not isinstance(split, str)
                or not split
            ):
                raise ValueError(
                    "Passage manifest record needs split and pair_id "
                    f"(or seed_group): {manifest_path}:{line_number}"
                )
            identity = (split, pair_id)
            if identity in records:
                raise ValueError(
                    f"Duplicate passage pair {identity!r} in {manifest_path}"
                )
            records[identity] = record
    if not records:
        raise ValueError(f"Empty synthesis manifest: {manifest_path}")
    return records


def _catalog_split_path_sets(
    catalog_path: Path,
    *,
    repo_root: Path,
) -> dict[str, set[Path]]:
    return {
        "train": set(
            _catalog_paths(catalog_path, "train", repo_root=repo_root)
        ),
        "validation": set(
            _catalog_paths(catalog_path, "val", repo_root=repo_root)
        ),
        "test": set(
            _catalog_paths(
                catalog_path,
                "ood_test",
                repo_root=repo_root,
            )
        ),
    }


def _load_passage_episode_sidecar(
    record: dict[str, Any],
    *,
    repo_root: Path,
    release_root: Path,
    strict_release_layout: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_path = record.get("episode_manifest")
    expected_sha256 = record.get("episode_manifest_sha256")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ValueError(
            "Passage manifest record is missing its episode sidecar binding: "
            f"{record.get('scenario_id', record.get('pair_id'))}"
        )
    path = _resolve_passage_declared_path(
        raw_path,
        repo_root=repo_root,
        release_root=release_root,
        leaf_kind="regular_file",
        required_subtree=(
            "episode_manifests" if strict_release_layout else None
        ),
    )
    observed_sha256 = _sha256(path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Passage episode sidecar hash mismatch: "
            f"expected={expected_sha256}, observed={observed_sha256}, "
            f"path={path}"
        )

    episodes: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            episode = json.loads(line)
            template_id = episode.get("template_id")
            if not isinstance(template_id, str) or not template_id:
                raise ValueError(
                    f"Episode sidecar is missing template_id: "
                    f"{path}:{line_number}"
                )
            if template_id in episodes:
                raise ValueError(
                    f"Duplicate episode template {template_id!r}: {path}"
                )
            episodes[template_id] = episode
    expected_count = int(record.get("episode_count", -1))
    if (
        not episodes
        or expected_count != len(episodes)
        or int(record.get("clip_count", -1)) != expected_count
    ):
        raise ValueError(
            "Passage episode sidecar count differs from its shard manifest: "
            f"path={path}, episodes={len(episodes)}, "
            f"expected={expected_count}, "
            f"clips={record.get('clip_count')}"
        )
    return episodes, {
        "path": str(path),
        "sha256": observed_sha256,
        "episodes": len(episodes),
        "passed": True,
    }


def _verify_passage_shard_logical_content(
    swm: Any,
    *,
    record: dict[str, Any],
    repo_root: Path,
    release_root: Path | None = None,
) -> dict[str, Any]:
    """Recompute one shard identity from decoded Lance rows."""

    from contextworld.evaluation.hidden_passage_h3_data import (
        LOGICAL_CONTENT_COLUMNS,
        LOGICAL_CONTENT_HASH_KIND,
        SHARD_COMPLETION_PROTOCOL,
        STORAGE_CONTENT_HASH_KIND,
        logical_episode_content_hashes,
        logical_shard_content_sha256,
        shard_completion_marker_path,
        verify_hidden_passage_shard_completion,
    )

    scenario_id = str(record.get("scenario_id", "unknown"))
    strict_release_layout = release_root is not None
    release_root = (
        require_safe_directory(release_root)
        if release_root is not None
        else require_safe_directory(repo_root)
    )
    if record.get("content_sha256_kind") != LOGICAL_CONTENT_HASH_KIND:
        raise ValueError(
            "Passage shard has an unsupported logical content hash kind: "
            f"scenario={scenario_id}, "
            f"observed={record.get('content_sha256_kind')!r}"
        )
    expected_content_sha256 = record.get("content_sha256")
    if (
        not isinstance(expected_content_sha256, str)
        or len(expected_content_sha256) != 64
    ):
        raise ValueError(
            f"Passage shard has no sealed logical content hash: {scenario_id}"
        )
    table_path = _resolve_passage_declared_path(
        record["output_path"],
        repo_root=repo_root,
        release_root=release_root,
        leaf_kind="directory",
        required_subtree="tables" if strict_release_layout else None,
    )
    if record.get("storage_sha256_kind") != STORAGE_CONTENT_HASH_KIND:
        raise ValueError(
            "Passage shard has an unsupported storage hash kind: "
            f"scenario={scenario_id}, "
            f"observed={record.get('storage_sha256_kind')!r}"
        )
    expected_storage_sha256 = record.get("storage_sha256")
    expected_marker_sha256 = record.get("completion_marker_sha256")
    raw_marker_path = record.get("completion_marker")
    if (
        record.get("completion_protocol") != SHARD_COMPLETION_PROTOCOL
        or not isinstance(expected_storage_sha256, str)
        or len(expected_storage_sha256) != 64
        or not isinstance(expected_marker_sha256, str)
        or len(expected_marker_sha256) != 64
        or not isinstance(raw_marker_path, str)
        or not raw_marker_path
    ):
        raise ValueError(
            "Passage shard has no sealed completion-marker binding: "
            f"{scenario_id}"
        )
    marker_path = _resolve_passage_declared_path(
        raw_marker_path,
        repo_root=repo_root,
        release_root=release_root,
        leaf_kind="regular_file",
        required_subtree="tables" if strict_release_layout else None,
    )
    expected_marker_path = shard_completion_marker_path(table_path)
    if marker_path != expected_marker_path:
        raise ValueError(
            "Passage completion marker is not the shard sibling: "
            f"scenario={scenario_id}, declared={marker_path}, "
            f"expected={expected_marker_path}"
        )
    episodes, _ = _load_passage_episode_sidecar(
        record,
        repo_root=repo_root,
        release_root=release_root,
        strict_release_layout=strict_release_layout,
    )
    completion_audit = verify_hidden_passage_shard_completion(
        table_path=table_path,
        episode_manifest_path=_resolve_passage_declared_path(
            record["episode_manifest"],
            repo_root=repo_root,
            release_root=release_root,
            leaf_kind="regular_file",
            required_subtree=(
                "episode_manifests" if strict_release_layout else None
            ),
        ),
        expected_scenario_id=scenario_id,
        expected_fingerprint=str(record.get("fingerprint", "")),
        expected_content_sha256=str(expected_content_sha256),
        expected_storage_sha256=expected_storage_sha256,
        expected_episode_manifest_sha256=str(
            record["episode_manifest_sha256"]
        ),
        expected_marker_sha256=expected_marker_sha256,
    )
    ordered = sorted(
        episodes.values(),
        key=lambda row: int(row.get("episode_index", -1)),
    )
    expected_indices = list(range(len(ordered)))
    observed_indices = [
        int(row.get("episode_index", -1)) for row in ordered
    ]
    if observed_indices != expected_indices:
        raise ValueError(
            "Passage episode sidecar indices are not contiguous/in order: "
            f"scenario={scenario_id}, observed={observed_indices[:20]}"
        )

    raw = swm.data.LanceDataset(path=table_path)
    rows_per_episode = int(record.get("rows_per_episode", -1))
    if (
        len(raw.lengths) != len(ordered)
        or any(int(length) != rows_per_episode for length in raw.lengths)
        or int(sum(map(int, raw.lengths)))
        != int(record.get("raw_rows", -1))
    ):
        raise ValueError(
            "Passage Lance row/episode counts differ from the sealed "
            f"manifest: scenario={scenario_id}"
        )

    observed_rows = []
    for episode_index, expected_row in enumerate(ordered):
        actual_hashes = logical_episode_content_hashes(
            raw.load_episode(episode_index)
        )
        hash_mismatches = {
            name: {
                "expected": expected_row.get(name),
                "observed": observed,
            }
            for name, observed in actual_hashes.items()
            if expected_row.get(name) != observed
        }
        if hash_mismatches:
            raise ValueError(
                "Passage Lance logical content differs from its episode "
                f"sidecar: scenario={scenario_id}, "
                f"episode={episode_index}, mismatches={hash_mismatches}"
            )
        observed_rows.append(
            {
                "episode_index": episode_index,
                "template_id": expected_row["template_id"],
                "rule": expected_row["rule"],
                **actual_hashes,
            }
        )
    observed_content_sha256 = logical_shard_content_sha256(observed_rows)
    if observed_content_sha256 != expected_content_sha256:
        raise ValueError(
            "Passage Lance logical content hash mismatch: "
            f"scenario={scenario_id}, "
            f"expected={expected_content_sha256}, "
            f"observed={observed_content_sha256}"
        )
    return {
        "scenario_id": scenario_id,
        "path": str(table_path),
        "content_sha256_kind": LOGICAL_CONTENT_HASH_KIND,
        "content_sha256": observed_content_sha256,
        "episodes_verified": len(observed_rows),
        "rows_verified": int(sum(map(int, raw.lengths))),
        "columns": list(LOGICAL_CONTENT_COLUMNS),
        "completion_marker": completion_audit,
        "passed": True,
    }


def _validate_paired_passage_catalogs(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Prove that mixed support is exactly the union of paired single rules."""

    catalog_paths = {
        group: _passage_catalog_path(
            config["data"]["catalogs"][group],
            repo_root=repo_root,
        )
        for group in PASSAGE_GROUPS
    }
    split_paths = {
        group: _catalog_split_path_sets(path, repo_root=repo_root)
        for group, path in catalog_paths.items()
    }
    path_checks: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        passable = split_paths["passage_passable"][split]
        blocked = split_paths["passage_blocked"][split]
        mixed = split_paths["passage_mixed"][split]
        disjoint = not (passable & blocked)
        exact_union = mixed == passable | blocked
        path_checks[split] = {
            "passable_paths": len(passable),
            "blocked_paths": len(blocked),
            "mixed_paths": len(mixed),
            "single_rule_paths_disjoint": disjoint,
            "mixed_is_exact_union": exact_union,
            "passed": disjoint and exact_union,
        }
    if not all(row["passed"] for row in path_checks.values()):
        raise ValueError(
            "Passage mixed catalog is not the exact disjoint union of "
            f"the two single-rule catalogs: {path_checks}"
        )

    records = {
        group: _manifest_records_by_pair(
            catalog_paths[group],
            repo_root=repo_root,
        )
        for group in ("passage_passable", "passage_blocked")
    }
    release_roots = {
        _passage_release_root_for_catalog(path)
        for path in catalog_paths.values()
    }
    if len(release_roots) != 1:
        raise ValueError(
            "Passage catalogs do not share one sealed release root: "
            f"{sorted(map(str, release_roots))}"
        )
    release_root = next(iter(release_roots))
    passable_records = records["passage_passable"]
    blocked_records = records["passage_blocked"]
    if set(passable_records) != set(blocked_records):
        raise ValueError(
            "Passage single-rule manifests have different pair sets: "
            f"passable_only={sorted(set(passable_records) - set(blocked_records))}, "
            f"blocked_only={sorted(set(blocked_records) - set(passable_records))}"
        )

    all_physical_records = [
        record
        for group_records in records.values()
        for record in group_records.values()
    ]
    sealed_release_layout = all(
        all(
            isinstance(record.get(field), str) and bool(record[field])
            for field in (
                "completion_marker",
                "completion_marker_sha256",
                "storage_sha256",
                "episode_manifest",
                "episode_manifest_sha256",
            )
        )
        for record in all_physical_records
    )
    expected_tables: set[Path] = set()
    expected_markers: set[Path] = set()
    expected_sidecars: set[Path] = set()
    if sealed_release_layout:
        for record in all_physical_records:
            table = _resolve_passage_declared_path(
                record["output_path"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="directory",
                required_subtree="tables",
            )
            marker = _resolve_passage_declared_path(
                record["completion_marker"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="regular_file",
                required_subtree="tables",
            )
            sidecar = _resolve_passage_declared_path(
                record["episode_manifest"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="regular_file",
                required_subtree="episode_manifests",
            )
            if marker != shard_completion_marker_path(table):
                raise ValueError(
                    "Passage marker is not the declared table sibling: "
                    f"scenario={record.get('scenario_id')}, marker={marker}, "
                    f"table={table}"
                )
            expected_tables.add(table)
            expected_markers.add(marker)
            expected_sidecars.add(sidecar)
        release_asset_audit = audit_hidden_passage_release_assets(
            release_root=release_root,
            expected_tables=expected_tables,
            expected_markers=expected_markers,
            expected_sidecars=expected_sidecars,
        )
    else:
        release_asset_audit = {
            "required": False,
            "reason": "legacy_unsealed_unit_fixture",
            "passed": True,
        }

    contract = dict(
        config.get("paired_collection_contract", {}).get(
            "passage_rules",
            {},
        )
    )
    paired_fields = tuple(
        contract.get(
            "equal_manifest_fields",
            (
                "split",
                "regime",
                "env_id",
                "env_seed",
                "policy_seed",
                "episodes",
                "task",
                "max_episode_steps",
                "image_shape",
                "reset_constraints",
                "pixel_codec",
                "seed_group",
                "pair_id",
            ),
        )
    )
    mismatches: list[dict[str, Any]] = []
    passage_values: dict[str, Counter] = {
        "passage_passable": Counter(),
        "passage_blocked": Counter(),
    }
    door_positions_by_split: dict[str, set[int]] = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    paired_episode_records = 0
    episode_sidecars_verified = 0
    equal_episode_fields = tuple(
        contract.get(
            "equal_episode_fields",
            (
                "pair_id",
                "action_sha256",
                "initial_pixels_sha256",
                "query_pixels_sha256",
                "model_input_keys",
                "passed",
            ),
        )
    )
    allowed_episode_differences = list(
        contract.get(
            "allowed_episode_differences",
            (
                "rule",
                "passage_open",
                "future_pixels_sha256",
            ),
        )
    )
    for identity in sorted(passable_records):
        left = passable_records[identity]
        right = blocked_records[identity]
        changed = []
        for field in paired_fields:
            if field not in left or field not in right:
                changed.append(f"missing.{field}")
            elif left[field] != right[field]:
                changed.append(field)
        left_factors = dict(left.get("factors", {}))
        right_factors = dict(right.get("factors", {}))
        left_passage = left_factors.pop("passage.open", None)
        right_passage = right_factors.pop("passage.open", None)
        left_door = left_factors.get("door.position")
        if identity[0] in door_positions_by_split:
            try:
                raw_doors = (
                    list(left_door)
                    if isinstance(left_door, (list, tuple))
                    else [left_door]
                )
                normalized_doors = {
                    int(value) for value in raw_doors
                }
                if len(normalized_doors) != 1:
                    raise ValueError(left_door)
                door_positions_by_split[identity[0]].update(
                    normalized_doors
                )
            except (TypeError, ValueError):
                changed.append("door.position")
        passage_values["passage_passable"][str(left_passage)] += 1
        passage_values["passage_blocked"][str(right_passage)] += 1
        if left_factors != right_factors:
            changed.append("factors_except_passage.open")
        if left_passage != 1:
            changed.append("passage_passable_support")
        if right_passage != 0:
            changed.append("passage_blocked_support")
        for optional_field in contract.get(
            "equal_manifest_count_fields",
            (
                "episode_count",
                "clip_count",
                "rows_per_episode",
                "raw_rows",
            ),
        ):
            if (
                optional_field in left
                or optional_field in right
            ) and left.get(optional_field) != right.get(optional_field):
                changed.append(optional_field)

        try:
            left_episodes, _ = _load_passage_episode_sidecar(
                left,
                repo_root=repo_root,
                release_root=release_root,
                strict_release_layout=sealed_release_layout,
            )
            right_episodes, _ = _load_passage_episode_sidecar(
                right,
                repo_root=repo_root,
                release_root=release_root,
                strict_release_layout=sealed_release_layout,
            )
            episode_sidecars_verified += 2
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "Passage episode-level pairing audit failed for "
                f"{identity!r}: {exc}"
            ) from exc
        if set(left_episodes) != set(right_episodes):
            changed.append("episode_template_set")
        else:
            paired_episode_records += len(left_episodes)
            for template_id in sorted(left_episodes):
                left_episode = left_episodes[template_id]
                right_episode = right_episodes[template_id]
                for field in equal_episode_fields:
                    if (
                        field not in left_episode
                        or field not in right_episode
                    ):
                        changed.append(f"episode.missing.{field}")
                    elif left_episode[field] != right_episode[field]:
                        changed.append(f"episode.{field}")
                if set(left_episode) != set(right_episode):
                    changed.append("episode.key_set")
                for field in sorted(
                    set(left_episode) | set(right_episode)
                ):
                    if (
                        left_episode.get(field)
                        != right_episode.get(field)
                        and field not in allowed_episode_differences
                    ):
                        changed.append(f"episode.{field}")
                if (
                    left_episode.get("rule") != "passable"
                    or left_episode.get("passage_open") != 1
                ):
                    changed.append("episode.passable_rule")
                if (
                    right_episode.get("rule") != "blocked"
                    or right_episode.get("passage_open") != 0
                ):
                    changed.append("episode.blocked_rule")
                if left_episode.get("model_input_keys") != [
                    "pixels",
                    "action",
                ]:
                    changed.append("episode.model_input_boundary")
        if changed:
            mismatches.append(
                {
                    "split": identity[0],
                    "pair_id": identity[1],
                    "fields": sorted(set(changed)),
                }
            )
    if mismatches:
        raise ValueError(
            "Paired passage catalogs differ beyond passage.open: "
            f"{mismatches[:20]}"
        )

    expected_eval_doors = {
        int(value)
        for value in config.get("passage_support", {}).get(
            "eval_only_door_positions",
            [],
        )
    }
    train_and_loader_val_doors = (
        door_positions_by_split["train"]
        | door_positions_by_split["val"]
    )
    eval_overlap = train_and_loader_val_doors & expected_eval_doors
    test_catalog_empty = not door_positions_by_split["test"]
    if eval_overlap or not test_catalog_empty:
        raise ValueError(
            "Passage train/validation/test door isolation failed: "
            f"train_or_val_overlap={sorted(eval_overlap)}, "
            "the training catalog ood_test section must be empty, "
            f"observed_test={sorted(door_positions_by_split['test'])}"
        )

    return {
        "required": True,
        "catalogs": {
            key: str(value) for key, value in catalog_paths.items()
        },
        "paired_single_rule_shards": len(passable_records),
        "pair_identity": ["split", "pair_id"],
        "equal_manifest_fields": list(paired_fields),
        "equal_episode_fields": list(equal_episode_fields),
        "allowed_difference": "passage.open",
        "allowed_episode_differences": allowed_episode_differences,
        "episode_sidecars_verified": episode_sidecars_verified,
        "paired_episode_records": paired_episode_records,
        "single_rule_support": {
            group: dict(sorted(values.items()))
            for group, values in passage_values.items()
        },
        "mixed_catalog_composition": "exact_path_union",
        "release_asset_audit": release_asset_audit,
        "split_path_checks": path_checks,
        "door_position_isolation": {
            "expected_eval_only": sorted(expected_eval_doors),
            "observed": {
                "train": sorted(door_positions_by_split["train"]),
                "validation": sorted(door_positions_by_split["val"]),
                "test": sorted(door_positions_by_split["test"]),
            },
            "train_and_loader_validation_exclude_eval_only": (
                not eval_overlap
            ),
            "training_catalog_test_is_empty": test_catalog_empty,
            "validation_assets_are_managed_separately": True,
            "passed": not eval_overlap and test_catalog_empty,
        },
        "fair_draw_contract": (
            "models receive equal logical optimizer draws; mixed raw support "
            "is intentionally the union and therefore twice one single rule"
        ),
        "passed": True,
    }


def revalidate_hidden_passage_training_storage(
    benchmark_config: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Recheck sealed bytes immediately before the first training batch."""

    config = _load_yaml(benchmark_config)
    catalog_paths = {
        group: _passage_catalog_path(
            config["data"]["catalogs"][group],
            repo_root=repo_root,
        )
        for group in PASSAGE_GROUPS
    }
    release_roots = {
        _passage_release_root_for_catalog(path)
        for path in catalog_paths.values()
    }
    if len(release_roots) != 1:
        raise ValueError(
            "Passage catalogs do not share one release root during "
            f"pre-batch revalidation: {sorted(map(str, release_roots))}"
        )
    release_root = next(iter(release_roots))
    records_by_group = {
        group: _manifest_records_by_pair(
            catalog_paths[group],
            repo_root=repo_root,
        )
        for group in ("passage_passable", "passage_blocked")
    }
    physical_records: dict[str, dict[str, Any]] = {}
    expected_tables: set[Path] = set()
    expected_markers: set[Path] = set()
    expected_sidecars: set[Path] = set()
    for records in records_by_group.values():
        for record in records.values():
            scenario_id = str(record["scenario_id"])
            if scenario_id in physical_records:
                raise ValueError(
                    f"Duplicate physical passage scenario: {scenario_id}"
                )
            table = _resolve_passage_declared_path(
                record["output_path"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="directory",
                required_subtree="tables",
            )
            marker = _resolve_passage_declared_path(
                record["completion_marker"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="regular_file",
                required_subtree="tables",
            )
            sidecar = _resolve_passage_declared_path(
                record["episode_manifest"],
                repo_root=repo_root,
                release_root=release_root,
                leaf_kind="regular_file",
                required_subtree="episode_manifests",
            )
            if marker != shard_completion_marker_path(table):
                raise ValueError(
                    "Passage marker is not the table sibling during "
                    f"pre-batch revalidation: {scenario_id}"
                )
            physical_records[scenario_id] = record
            expected_tables.add(table)
            expected_markers.add(marker)
            expected_sidecars.add(sidecar)
    release_assets = audit_hidden_passage_release_assets(
        release_root=release_root,
        expected_tables=expected_tables,
        expected_markers=expected_markers,
        expected_sidecars=expected_sidecars,
    )
    completions = []
    for scenario_id, record in sorted(physical_records.items()):
        table = _resolve_passage_declared_path(
            record["output_path"],
            repo_root=repo_root,
            release_root=release_root,
            leaf_kind="directory",
            required_subtree="tables",
        )
        sidecar = _resolve_passage_declared_path(
            record["episode_manifest"],
            repo_root=repo_root,
            release_root=release_root,
            leaf_kind="regular_file",
            required_subtree="episode_manifests",
        )
        completion = verify_hidden_passage_shard_completion(
            table_path=table,
            episode_manifest_path=sidecar,
            expected_scenario_id=scenario_id,
            expected_fingerprint=str(record["fingerprint"]),
            expected_content_sha256=str(record["content_sha256"]),
            expected_storage_sha256=str(record["storage_sha256"]),
            expected_episode_manifest_sha256=str(
                record["episode_manifest_sha256"]
            ),
            expected_marker_sha256=str(
                record["completion_marker_sha256"]
            ),
        )
        completions.append(completion)
    return {
        "release_root": str(release_root),
        "release_assets": release_assets,
        "physical_shards_verified": len(completions),
        "storage_hashes_verified": len(completions),
        "sidecar_hashes_verified": len(completions),
        "completion_markers_verified": len(completions),
        "passed": True,
    }


def hidden_passage_training_release_root(
    benchmark_config: Path,
    *,
    repo_root: Path,
    model_id: str,
) -> Path | None:
    """Return the one sealed root used by a passage model, if applicable."""

    config = _load_yaml(benchmark_config)
    model_entries = {
        str(entry["model_id"]): entry
        for entry in config.get("models", [])
    }
    entry = model_entries.get(model_id)
    if entry is None:
        return None
    groups = list(entry.get("training_groups", []))
    if not any(group in PASSAGE_GROUPS for group in groups):
        return None
    roots = {
        _passage_release_root_for_catalog(
            _passage_catalog_path(
                config["data"]["catalogs"][group],
                repo_root=repo_root,
            )
        )
        for group in PASSAGE_GROUPS
    }
    if len(roots) != 1:
        raise ValueError(
            f"Passage model {model_id} has multiple release roots: "
            f"{sorted(map(str, roots))}"
        )
    return next(iter(roots))


def _validate_paired_door_catalogs(
    config: dict[str, Any], *, repo_root: Path
) -> dict[str, Any]:
    """Mechanically enforce the fixed-door/multi-door pairing contract."""

    catalog_paths = {
        group: resolve_contextworld_path(
            config["data"]["catalogs"][group], repo_root=repo_root
        )
        for group in ("door_fixed49_v2", "door_multi_v2")
    }
    records = {
        group: _manifest_records_by_seed_group(path, repo_root=repo_root)
        for group, path in catalog_paths.items()
    }
    fixed = records["door_fixed49_v2"]
    multi = records["door_multi_v2"]
    if set(fixed) != set(multi):
        raise ValueError(
            "Paired door catalogs have different seed-group sets: "
            f"fixed_only={sorted(set(fixed) - set(multi))}, "
            f"multi_only={sorted(set(multi) - set(fixed))}"
        )

    paired_fields = (
        "split",
        "regime",
        "env_id",
        "env_seed",
        "policy_seed",
        "episodes",
        "task",
        "max_episode_steps",
        "image_shape",
        "reset_constraints",
        "pixel_codec",
    )
    mismatches: list[dict[str, Any]] = []
    for seed_group in sorted(fixed):
        left = fixed[seed_group]
        right = multi[seed_group]
        changed = [
            field for field in paired_fields if left.get(field) != right.get(field)
        ]
        if changed:
            mismatches.append(
                {"seed_group": seed_group, "fields": changed}
            )
    if mismatches:
        raise ValueError(
            "Paired door catalog contract failed; only door.position may "
            f"differ: {mismatches[:20]}"
        )

    split_counts = Counter(row["split"] for row in fixed.values())
    return {
        "required": True,
        "catalogs": {key: str(value) for key, value in catalog_paths.items()},
        "paired_seed_groups": len(fixed),
        "split_counts": dict(sorted(split_counts.items())),
        "equal_fields": list(paired_fields),
        "allowed_difference": "door.position",
        "passed": True,
    }


def _factor_balanced_group(
    paths: list[Path],
    scenarios: list[Any],
    *,
    factors_by_path: dict[Path, dict[str, Any]],
    factor_key: str,
) -> tuple[ScenarioBalancedDataset, dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    display_values: dict[str, Any] = {}
    for path, scenario in zip(paths, scenarios):
        value = factors_by_path[path][factor_key]
        identity = json.dumps(value, sort_keys=True)
        grouped.setdefault(identity, []).append(scenario)
        display_values[identity] = value
    factor_datasets = [
        ConcatenatedDataset(grouped[identity]) for identity in sorted(grouped)
    ]
    balanced = ScenarioBalancedDataset(factor_datasets)
    return balanced, {
        "strategy": "factor_value_balanced_scenario_proportional",
        "factor_key": factor_key,
        "factor_values": len(grouped),
        "scenarios_per_factor": {
            identity: len(grouped[identity]) for identity in sorted(grouped)
        },
        "raw_clips_per_factor": {
            identity: len(dataset)
            for identity, dataset in zip(sorted(grouped), factor_datasets)
        },
        "display_values": {
            identity: display_values[identity] for identity in sorted(grouped)
        },
    }


def _training_transform(
    original_dataset,
    *,
    img_size: int,
    statistics_rows: np.ndarray | None = None,
    frozen_scalers: dict[str, UnbiasedZScoreScaler] | None = None,
):
    import stable_pretraining as spt
    import torch
    from stable_pretraining import data as dt

    imagenet_stats = dt.dataset_stats.ImageNet
    image = dt.transforms.Compose(
        dt.transforms.ToImage(
            **imagenet_stats, source="pixels", target="pixels"
        ),
        dt.transforms.Resize(img_size, source="pixels", target="pixels"),
    )
    transforms = [image]
    scalers: dict[str, UnbiasedZScoreScaler] = {}
    for column in ("action", "proprio"):
        if frozen_scalers is not None:
            try:
                scaler = frozen_scalers[column]
            except KeyError as exc:
                raise ValueError(
                    f"Frozen normalizer is missing {column!r}"
                ) from exc
        else:
            values_array = np.asarray(
                original_dataset.get_col_data(column)
            )
            if statistics_rows is not None:
                values_array = values_array[statistics_rows]
            values = torch.from_numpy(values_array)
            valid = values[~torch.isnan(values).any(dim=1)]
            scaler = UnbiasedZScoreScaler(
                valid.mean(0, keepdim=True).numpy(),
                valid.std(0, keepdim=True).numpy(),
            )
        scalers[column] = scaler
        transforms.append(
            dt.transforms.WrapTorchTransform(
                scaler, source=column, target=column
            )
        )
    return spt.data.transforms.Compose(*transforms), scalers


def _lance_scenarios(
    swm,
    paths: list[Path],
    *,
    frameskip: int,
    num_steps: int,
    transform,
) -> list[Any]:
    scenarios = []
    for path in paths:
        dataset = swm.data.LanceDataset(
            path=path,
            frameskip=frameskip,
            num_steps=num_steps,
            keys_to_load=list(MODEL_COLUMNS),
            transform=transform,
        )
        if len(dataset) <= 0:
            raise ValueError(
                f"Training scenario {path} has no clips for "
                f"frameskip={frameskip}, num_steps={num_steps}"
            )
        scenarios.append(dataset)
    return scenarios


def build_tworoom_grouped_data(
    swm,
    *,
    repo_root: Path,
    benchmark_config: Path,
    model_id: str,
    epoch_size: int,
    validation_epoch_size: int,
    original_h5: Path | None = None,
    frameskip: int = 5,
    num_steps: int = 4,
    img_size: int = 224,
    seed: int = 3072,
    expected_stablewm_commit: str | None = None,
) -> TwoRoomGroupedData:
    """Build an exact-weight StableWM H5/Lance mixture for one model."""

    import torch

    config = _load_yaml(benchmark_config)
    quality_groups = dict(config.get("data_quality", {}).get("groups", {}))
    try:
        weights = dict(config["training_protocol"]["group_sampling"][model_id])
    except KeyError as exc:
        raise KeyError(f"No group sampling protocol for {model_id}") from exc

    model_entries = {
        entry["model_id"]: entry for entry in config.get("models", [])
    }
    expected_groups = model_entries[model_id]["training_groups"]
    if list(weights) != expected_groups:
        raise ValueError(
            f"{model_id} group order mismatch: weights={list(weights)}, "
            f"model={expected_groups}"
        )
    passage_model = any(
        group in PASSAGE_GROUPS for group in expected_groups
    )

    paired_collection_audit: dict[str, Any] = {
        "required": False,
        "passed": True,
    }
    training_exclusion_audit: dict[str, Any] = {
        "required": False,
        "passed": True,
    }
    formal_build_report_audit: dict[str, Any] = {
        "required": False,
        "passed": True,
    }
    if any(group.startswith("door_") for group in expected_groups):
        paired_collection_audit = _validate_paired_door_catalogs(
            config, repo_root=repo_root
        )
    elif passage_model:
        exclusion_spec = config["data"].get(
            "training_exclusion_manifest"
        )
        if not isinstance(exclusion_spec, dict):
            raise ValueError(
                "Passage training requires data.training_exclusion_manifest"
            )
        training_exclusion_audit = {
            "required": True,
            **_load_training_exclusion_manifest(
                exclusion_spec,
                repo_root=repo_root,
                expected_eval_only_door_positions=list(
                    config["passage_support"][
                        "eval_only_door_positions"
                    ]
                ),
            ),
        }
        formal_build_report_audit = _load_formal_passage_build_report(
            config,
            repo_root=repo_root,
        )
        paired_collection_audit = _validate_paired_passage_catalogs(
            config,
            repo_root=repo_root,
        )
    original_path = (
        Path(original_h5).expanduser().resolve()
        if original_h5 is not None
        else resolve_contextworld_path(
            config["data"]["original_read_only"], repo_root=repo_root
        )
    )
    if not original_path.is_file():
        raise FileNotFoundError(original_path)

    original = swm.data.HDF5Dataset(
        path=original_path,
        frameskip=frameskip,
        num_steps=num_steps,
        keys_to_load=list(MODEL_COLUMNS),
        keys_to_cache=["action", "proprio"],
    )
    split_config = dict(config["data"].get("original_split", {}))
    split_strategy = split_config.get("strategy", "clip_random")
    split_fraction = float(split_config.get("train_fraction", 0.9))
    if split_strategy == "episode_heldout":
        split_seed = int(split_config.get("seed", seed))
        train_episode_ids, heldout_episode_ids = partition_episode_ids(
            len(original.lengths),
            seed=split_seed,
            train_fraction=split_fraction,
        )
        train_indices = clip_subset_indices(original, train_episode_ids)
        heldout_indices = clip_subset_indices(original, heldout_episode_ids)
        statistics_rows = episode_row_indices(
            original.lengths, original.offsets, train_episode_ids
        )
        original_train = torch.utils.data.Subset(original, train_indices)
        original_val = torch.utils.data.Subset(original, heldout_indices)
        split_metadata = {
            "strategy": split_strategy,
            "seed": split_seed,
            "train_fraction": split_fraction,
            "train_episodes": int(len(train_episode_ids)),
            "heldout_episodes": int(len(heldout_episode_ids)),
            "train_episode_ids_sha256": episode_ids_sha256(train_episode_ids),
            "heldout_episode_ids_sha256": episode_ids_sha256(
                heldout_episode_ids
            ),
            "normalization_statistics_scope": "train_episodes_only",
        }
    elif split_strategy == "clip_random":
        generator = torch.Generator().manual_seed(seed)
        original_train, original_val = torch.utils.data.random_split(
            original,
            lengths=[split_fraction, 1.0 - split_fraction],
            generator=generator,
        )
        statistics_rows = None
        split_metadata = {
            "strategy": split_strategy,
            "seed": seed,
            "train_fraction": split_fraction,
            "normalization_statistics_scope": "all_rows_legacy_compatible",
        }
    else:
        raise ValueError(f"Unsupported original split strategy: {split_strategy}")
    frozen_normalizer_spec = config["data"].get("frozen_normalizer")
    frozen_scalers = None
    if frozen_normalizer_spec is not None:
        if not isinstance(frozen_normalizer_spec, dict):
            raise ValueError("data.frozen_normalizer must be a mapping")
        frozen_scalers, normalization_contract = (
            _load_frozen_normalizer(
                frozen_normalizer_spec,
                repo_root=repo_root,
                split_metadata=split_metadata,
            )
        )
    else:
        normalization_contract = {
            "mode": "computed_from_original_training_split",
            "path": None,
            "sha256": None,
            "passed": True,
        }
    transform, scalers = _training_transform(
        original,
        img_size=img_size,
        statistics_rows=statistics_rows,
        frozen_scalers=frozen_scalers,
    )
    original.transform = transform

    # Every controlled model uses the same original-train normalization
    # statistics, but synthetic-only controls must not receive original
    # trajectory samples.
    train_groups: dict[str, Any] = {}
    val_groups: dict[str, Any] = {}
    group_metadata: dict[str, Any] = {}
    normalization_reference = {
        "scenarios": 1,
        "train_clips": len(original_train),
        "val_clips": len(original_val),
        "source": str(original_path),
        "split": split_metadata,
    }
    if "original" in expected_groups:
        train_groups["original"] = original_train
        val_groups["original"] = original_val
        group_metadata["original"] = {
            "scenarios": 1,
            "train_clips": len(original_train),
            "val_clips": len(original_val),
            "source": str(original_path),
            "split": split_metadata,
        }
    for group in expected_groups:
        if group == "original":
            continue
        catalog_key = CATALOG_BY_GROUP[group]
        if group in PASSAGE_GROUPS:
            catalog_path = _passage_catalog_path(
                config["data"]["catalogs"][catalog_key],
                repo_root=repo_root,
            )
        else:
            catalog_path = resolve_contextworld_path(
                config["data"]["catalogs"][catalog_key],
                repo_root=repo_root,
            )
        quality = dict(quality_groups.get(group, {}))
        train_paths = _catalog_paths(
            catalog_path, "train", repo_root=repo_root
        )
        val_paths = _catalog_paths(catalog_path, "val", repo_root=repo_root)
        factor_support_config = quality.get("factor_support_contract")
        factor_support_contract = None
        if factor_support_config:
            if "expected_by_split" in factor_support_config:
                factor_support_contract = {
                    "factor": str(factor_support_config["factor"]),
                    "expected_by_split": {
                        str(section): list(values)
                        for section, values in factor_support_config[
                            "expected_by_split"
                        ].items()
                    },
                }
            else:
                door_support = config["door_support"]
                factor_support_contract = {
                    "factor": str(factor_support_config["factor"]),
                    "expected_by_split": {
                        "train": list(
                            door_support[
                                factor_support_config[
                                    "train_values_from_door_support"
                                ]
                            ]
                        ),
                        "validation": list(
                            door_support[
                                factor_support_config[
                                    "validation_values_from_door_support"
                                ]
                            ]
                        ),
                    },
                }
        exact_split_counts = {
            name: int(quality[key])
            for name, key in (
                ("train", "exact_train_scenarios"),
                ("validation", "exact_validation_scenarios"),
                ("test", "exact_test_scenarios"),
            )
            if key in quality
        }
        required_artifact_hashes = {
            name: str(quality[key])
            for name, key in (
                ("catalog", "required_catalog_sha256"),
                ("manifest", "required_manifest_sha256"),
                (
                    "synthesis_report",
                    "required_synthesis_report_sha256",
                ),
            )
            if key in quality
        }
        split_audit = _catalog_split_audit(
            catalog_path,
            swm=swm,
            repo_root=repo_root,
            expected_stablewm_commit=expected_stablewm_commit,
            require_complete_synthesis_report=bool(
                quality.get(
                    "require_complete_synthesis_report",
                    False,
                )
            ),
            expected_split_scenario_counts=exact_split_counts,
            factor_support_contract=factor_support_contract,
            required_artifact_hashes=required_artifact_hashes,
        )
        train_scenarios = _lance_scenarios(
            swm,
            train_paths,
            frameskip=frameskip,
            num_steps=num_steps,
            transform=transform,
        )
        val_scenarios = _lance_scenarios(
            swm,
            val_paths,
            frameskip=frameskip,
            num_steps=num_steps,
            transform=transform,
        )
        balance_by_factor = quality.get("balance_by_factor")
        if balance_by_factor:
            factors_by_path = _catalog_factors_by_path(
                catalog_path, repo_root=repo_root
            )
            train_groups[group], train_balancing = _factor_balanced_group(
                train_paths,
                train_scenarios,
                factors_by_path=factors_by_path,
                factor_key=str(balance_by_factor),
            )
            val_groups[group], val_balancing = _factor_balanced_group(
                val_paths,
                val_scenarios,
                factors_by_path=factors_by_path,
                factor_key=str(balance_by_factor),
            )
        elif quality.get("sampling_strategy") == "concatenated_raw_clips":
            train_groups[group] = ConcatenatedDataset(train_scenarios)
            val_groups[group] = ConcatenatedDataset(val_scenarios)
            train_balancing = {"strategy": "concatenated_raw_clips"}
            val_balancing = {"strategy": "concatenated_raw_clips"}
        else:
            train_groups[group] = ScenarioBalancedDataset(train_scenarios)
            val_groups[group] = ScenarioBalancedDataset(val_scenarios)
            train_balancing = {"strategy": "scenario_balanced"}
            val_balancing = {"strategy": "scenario_balanced"}
        raw_train_clips = sum(len(value) for value in train_scenarios)
        static_gates = {
            "pixel_codec": (
                not quality.get("required_pixel_codec")
                or split_audit["pixel_codec"]
                == quality["required_pixel_codec"]
            ),
            "raw_train_clips": raw_train_clips
            >= int(quality.get("minimum_raw_train_clips", 0)),
            "train_scenarios": len(train_scenarios)
            >= int(quality.get("minimum_train_scenarios", 0)),
            "validation_scenarios": len(val_scenarios)
            >= int(quality.get("minimum_validation_scenarios", 0)),
        }
        if quality and not all(static_gates.values()):
            raise ValueError(
                f"Data-quality gates failed for {group}: {static_gates}; "
                f"requirements={quality}"
            )
        group_metadata[group] = {
            "scenarios": len(train_scenarios),
            "validation_scenarios": len(val_scenarios),
            "train_clips_raw": raw_train_clips,
            "val_clips_raw": sum(len(value) for value in val_scenarios),
            "minimum_train_scenario_clips": min(
                len(value) for value in train_scenarios
            ),
            "maximum_train_scenario_clips": max(
                len(value) for value in train_scenarios
            ),
            "catalog": str(catalog_path),
            "catalog_split_audit": split_audit,
            "scenario_balanced_virtual_train_clips": len(train_groups[group]),
            "scenario_balanced_virtual_val_clips": len(val_groups[group]),
            "train_balancing": train_balancing,
            "validation_balancing": val_balancing,
            "pixel_codec": split_audit["pixel_codec"],
            "quality_requirements": quality,
            "static_quality_gates": static_gates,
        }

    train = LogicalGroupDataset(
        train_groups, weights, epoch_size=epoch_size
    )
    val = LogicalGroupDataset(
        val_groups, weights, epoch_size=validation_epoch_size
    )
    metadata = {
        "model_id": model_id,
        "frameskip": frameskip,
        "num_steps": num_steps,
        "image_size": img_size,
        "seed": seed,
        "data_split_seed": seed,
        "original_split": split_metadata,
        "epoch_size": epoch_size,
        "validation_epoch_size": validation_epoch_size,
        "group_weights": train.normalized_weights,
        "epoch_group_counts": train.epoch_group_counts(),
        "epoch_group_coverage": train.epoch_group_coverage(),
        "validation_group_counts": val.epoch_group_counts(),
        "validation_group_coverage": val.epoch_group_coverage(),
        "groups": group_metadata,
        "paired_collection_audit": paired_collection_audit,
        "training_exclusion_audit": training_exclusion_audit,
        "formal_build_report_audit": formal_build_report_audit,
        "distributed_passage_audit": {
            "required": passage_model,
            "optimization": "disabled_per_rank_full_audit",
            "process_mode": "full",
            "full_logical_audit_executed_in_this_process": (
                passage_model
            ),
            "rank0_attestation_required": False,
            "rank0_attestation_used": False,
            "passed": True,
        },
        "training_data_scope": {
            "synthetic_only": "original" not in expected_groups,
            "original_samples_included": "original" in expected_groups,
            "original_data_role_when_synthetic_only": (
                "normalizer_and_split_identity_only"
                if "original" not in expected_groups
                else None
            ),
            "model_visible_columns": ["pixels", "action"],
            "diagnostic_only_columns": ["proprio"],
            "hidden_rule_columns_loaded": False,
        },
        "normalization_reference": normalization_reference,
        "normalization_contract": normalization_contract,
        "normalization": {
            name: {
                "mean": scaler.mean.tolist(),
                "std_unbiased": scaler.std.tolist(),
                "source": (
                    normalization_contract["path"]
                    or str(original_path)
                ),
            }
            for name, scaler in scalers.items()
        },
        "factor_columns_exposed_to_model": False,
        "quality_gates_configured": bool(quality_groups),
    }
    return TwoRoomGroupedData(train=train, val=val, metadata=metadata)
