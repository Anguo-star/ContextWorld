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


def _catalog_split_audit(
    catalog_path: Path,
    *,
    repo_root: Path,
    expected_stablewm_commit: str | None,
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
    if not synthesis_report.get("passed"):
        raise ValueError(f"Synthesis report did not pass: {report_path}")

    return {
        "catalog_sha256": _sha256(catalog_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "synthesis_report": str(report_path),
        "synthesis_report_sha256": _sha256(report_path),
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

    original_path = resolve_contextworld_path(
        config["data"]["original_read_only"], repo_root=repo_root
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
        train_paths = _catalog_paths(
            catalog_path, "train", repo_root=repo_root
        )
        val_paths = _catalog_paths(catalog_path, "val", repo_root=repo_root)
        split_audit = _catalog_split_audit(
            catalog_path,
            repo_root=repo_root,
            expected_stablewm_commit=expected_stablewm_commit,
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
        quality = dict(quality_groups.get(group, {}))
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
