from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
    "speed_door_composition": "speed_door_composition",
}


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
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    raw = catalog[section]["synthetic"]
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
    repo_root: Path,
    expected_stablewm_commit: str | None,
    require_complete_synthesis_report: bool = False,
    expected_split_scenario_counts: dict[str, int] | None = None,
    factor_support_contract: dict[str, Any] | None = None,
    required_artifact_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if catalog.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported catalog schema in {catalog_path}: "
            f"{catalog.get('schema_version')}"
        )

    train_paths = _resolve_catalog_entries(
        list(catalog["train"]["synthetic"]), repo_root=repo_root
    )
    validation_paths = _resolve_catalog_entries(
        list(catalog["val"]["synthetic"]), repo_root=repo_root
    )
    test_entries = list(catalog.get("ood_test", {}).get("synthetic", []))
    test_paths = _resolve_catalog_entries(test_entries, repo_root=repo_root)

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
    for record in records:
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
        if record.get("collection_status") != "collected":
            raise ValueError(
                f"Manifest scenario is not collected: {record['scenario_id']}"
            )
        if resolved.stem != record["scenario_id"]:
            raise ValueError(
                f"Scenario id/path mismatch: {record['scenario_id']} vs {resolved}"
            )
        if not record["scenario_id"].endswith(record["fingerprint"][:10]):
            raise ValueError(
                f"Scenario fingerprint/path mismatch: {record['scenario_id']}"
            )
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
        "synthesis_report_gate": synthesis_report_gate,
        "stable_worldmodel_commits": sorted(commits),
        "pixel_codec": catalog_codec,
        "train_scenarios": len(train),
        "validation_scenarios": len(validation),
        "test_scenarios": len(test),
        "unique_scenario_ids": len(set(scenario_ids)),
        "unique_fingerprints": len(set(fingerprints)),
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
    scalers = {}
    for column in ("action", "proprio"):
        values_array = np.asarray(original_dataset.get_col_data(column))
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

    paired_collection_audit: dict[str, Any] = {
        "required": False,
        "passed": True,
    }
    if any(group.startswith("door_") for group in expected_groups):
        paired_collection_audit = _validate_paired_door_catalogs(
            config, repo_root=repo_root
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
    transform, scalers = _training_transform(
        original, img_size=img_size, statistics_rows=statistics_rows
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
        catalog_path = resolve_contextworld_path(
            config["data"]["catalogs"][catalog_key], repo_root=repo_root
        )
        quality = dict(quality_groups.get(group, {}))
        train_paths = _catalog_paths(
            catalog_path, "train", repo_root=repo_root
        )
        val_paths = _catalog_paths(catalog_path, "val", repo_root=repo_root)
        factor_support_config = quality.get("factor_support_contract")
        factor_support_contract = None
        if factor_support_config:
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
            repo_root=repo_root,
            expected_stablewm_commit=expected_stablewm_commit,
            require_complete_synthesis_report=bool(
                quality.get("require_complete_synthesis_report", False)
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
        "normalization_reference": normalization_reference,
        "normalization": {
            name: {
                "mean": scaler.mean.tolist(),
                "std_unbiased": scaler.std.tolist(),
                "source": str(original_path),
            }
            for name, scaler in scalers.items()
        },
        "factor_columns_exposed_to_model": False,
        "quality_gates_configured": bool(quality_groups),
    }
    return TwoRoomGroupedData(train=train, val=val, metadata=metadata)
