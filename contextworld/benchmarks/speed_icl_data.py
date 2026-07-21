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

from contextworld.paths import repository_root, resolve_contextworld_path


DEFAULT_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/tworoom_speed_icl_release_v1.yaml"
)
ORIGINAL_H5_ENV = "CONTEXTWORLD_TWOROOM_H5"
HORIZONS = (1, 2, 3, 5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}:{array.shape}".encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_fingerprint(path: Path, *, hash_contents: bool) -> dict[str, Any]:
    """Fingerprint a directory using sorted relative paths and file hashes."""

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
        file_hash = _sha256(value)
        digest.update(
            f"{relative}\0{size}\0{file_hash}\n".encode("utf-8")
        )
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "full_hash_verified": True,
    }


def load_speed_icl_release(
    path: Path | str = DEFAULT_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Speed ICL release config: {config_path}")
    if payload.get("release_id") != "contextworld_tworoom_speed_icl_history3_v1":
        raise ValueError(f"Unexpected release id in {config_path}")
    return {**payload, "_config_path": str(config_path)}


def resolve_original_h5(
    release: dict[str, Any],
    *,
    repo_root: Path,
    explicit: Path | str | None = None,
) -> Path:
    original = release["training"]["original"]
    value = explicit or os.environ.get(str(original["environment_variable"]))
    if value is not None:
        return Path(value).expanduser().resolve()
    return resolve_contextworld_path(original["local_fallback"], repo_root=repo_root)


@dataclass(frozen=True)
class SpeedICLHistory:
    condition: str
    history_speed: float
    input_pixels: np.ndarray
    raw_action_blocks: np.ndarray


@dataclass(frozen=True)
class SpeedICLEvalBundle:
    query_id: str
    static_query_id: str
    track: str
    reference_speed: float
    matching_condition: str
    action_family: str
    eval_seed: int
    evaluation_index: int
    query_pixels: np.ndarray
    target_pixels: np.ndarray
    future_actions: np.ndarray
    histories: dict[str, SpeedICLHistory]


class SpeedICLEvalDataset:
    """Lazy, hash-checked reader for one public Speed ICL evaluation track."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_RELEASE_CONFIG,
        track: str,
        repo_root: Path | None = None,
        eval_seeds: list[int] | tuple[int, ...] | None = None,
        limit_per_reference_speed_per_seed: int | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_speed_icl_release(release_config)
        tracks = self.release["evaluation"]["tracks"]
        if track not in tracks:
            raise KeyError(
                f"Unknown track {track!r}; available={sorted(tracks)}"
            )
        self.track = str(track)
        self.track_config = dict(tracks[track])
        self.catalog_path = resolve_contextworld_path(
            self.track_config["catalog"], repo_root=self.repo_root
        )
        if not self.catalog_path.is_file():
            raise FileNotFoundError(self.catalog_path)
        observed = _sha256(self.catalog_path)
        if observed != self.track_config["catalog_sha256"]:
            raise RuntimeError(
                f"Catalog hash mismatch for {self.track}: {observed}"
            )
        catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if catalog.get("track") != self.track:
            raise RuntimeError(
                f"Catalog track mismatch: {catalog.get('track')} != {self.track}"
            )
        if not catalog.get("summary", {}).get("passed"):
            raise RuntimeError(f"Catalog integrity status is not passed: {self.track}")
        allowed_seeds = tuple(
            int(value)
            for value in (
                eval_seeds
                if eval_seeds is not None
                else self.release["evaluation"]["eval_seeds"]
            )
        )
        official_seeds = {
            int(value) for value in self.release["evaluation"]["eval_seeds"]
        }
        if not set(allowed_seeds).issubset(official_seeds):
            raise ValueError(
                f"Eval seeds must be a subset of {sorted(official_seeds)}"
            )
        official_limit = int(
            self.release["evaluation"]["queries_per_reference_speed_per_seed"]
        )
        limit = (
            official_limit
            if limit_per_reference_speed_per_seed is None
            else int(limit_per_reference_speed_per_seed)
        )
        if not 1 <= limit <= official_limit:
            raise ValueError(
                f"Query limit must be in [1,{official_limit}], got {limit}"
            )
        self.eval_seeds = allowed_seeds
        self.limit_per_reference_speed_per_seed = limit
        selected = [
            bundle
            for bundle in catalog["bundles"]
            if int(bundle["eval_seed"]) in allowed_seeds
            and int(bundle["evaluation_index"]) < limit
        ]
        self._bundles = sorted(
            selected,
            key=lambda row: (
                float(row["query_factors"]["agent.speed"]),
                int(row["eval_seed"]),
                int(row["evaluation_index"]),
                str(row["query_id"]),
            ),
        )
        expected = (
            len(self.track_config["speeds"])
            * len(self.eval_seeds)
            * self.limit_per_reference_speed_per_seed
        )
        if len(self._bundles) != expected:
            raise RuntimeError(
                f"Selected {len(self._bundles)} bundles, expected {expected}"
            )
        self.conditions = tuple(catalog["summary"]["history_conditions"])

    def __len__(self) -> int:
        return len(self._bundles)

    @property
    def reference_speeds(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.track_config["speeds"])

    @property
    def is_full_protocol(self) -> bool:
        return (
            set(self.eval_seeds)
            == {int(value) for value in self.release["evaluation"]["eval_seeds"]}
            and self.limit_per_reference_speed_per_seed
            == int(
                self.release["evaluation"][
                    "queries_per_reference_speed_per_seed"
                ]
            )
        )

    def _load(self, row: dict[str, Any]) -> SpeedICLEvalBundle:
        payload_path = resolve_contextworld_path(
            row["payload"], repo_root=self.repo_root
        )
        if not payload_path.is_file():
            raise FileNotFoundError(payload_path)
        if _sha256(payload_path) != row["payload_sha256"]:
            raise RuntimeError(f"Payload hash mismatch: {payload_path}")
        with np.load(payload_path, allow_pickle=False) as payload:
            query = np.asarray(payload["query_pixels"], dtype=np.uint8)
            future_actions = np.asarray(
                payload["future_actions"], dtype=np.float32
            )
            targets = np.asarray(payload["future_next_pixels"], dtype=np.uint8)
            if _array_sha256(query) != row["query_pixels_sha256"]:
                raise RuntimeError(f"Query hash mismatch: {payload_path}")
            if _array_sha256(future_actions) != row["future_actions_sha256"]:
                raise RuntimeError(f"Action hash mismatch: {payload_path}")
            for horizon in HORIZONS:
                observed = _array_sha256(targets[horizon - 1])
                expected = row["target_pixels_sha256_by_horizon"][str(horizon)]
                if observed != expected:
                    raise RuntimeError(
                        f"Target hash mismatch h={horizon}: {payload_path}"
                    )
            histories = {}
            for condition in self.conditions:
                prefix = f"context_b2_{condition}"
                context_pixels = np.asarray(
                    payload[f"{prefix}_pixels"], dtype=np.uint8
                )
                context_actions = np.asarray(
                    payload[f"{prefix}_actions"], dtype=np.float32
                )
                context_next = np.asarray(
                    payload[f"{prefix}_next_pixels"], dtype=np.uint8
                )
                if not np.array_equal(context_pixels[1], context_next[0]):
                    raise RuntimeError(f"Broken context continuity: {payload_path}")
                if not np.array_equal(context_next[-1], query):
                    raise RuntimeError(f"Context does not end at query: {payload_path}")
                histories[str(condition)] = SpeedICLHistory(
                    condition=str(condition),
                    history_speed=float(
                        row["conditions"][condition]["factors"]["agent.speed"]
                    ),
                    input_pixels=np.concatenate(
                        [context_pixels, query[None]], axis=0
                    ),
                    raw_action_blocks=np.concatenate(
                        [context_actions, future_actions], axis=0
                    ),
                )
        return SpeedICLEvalBundle(
            query_id=str(row["query_id"]),
            static_query_id=str(row["static_query_id"]),
            track=self.track,
            reference_speed=float(row["query_factors"]["agent.speed"]),
            matching_condition=str(row["matching_condition"]),
            action_family=str(row["query_action_family"]),
            eval_seed=int(row["eval_seed"]),
            evaluation_index=int(row["evaluation_index"]),
            query_pixels=query,
            target_pixels=targets,
            future_actions=future_actions,
            histories=histories,
        )

    def __getitem__(self, index: int) -> SpeedICLEvalBundle:
        return self._load(self._bundles[index])

    def __iter__(self) -> Iterator[SpeedICLEvalBundle]:
        for row in self._bundles:
            yield self._load(row)

    def describe(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "catalog": str(self.catalog_path),
            "catalog_sha256": self.track_config["catalog_sha256"],
            "reference_speeds": list(self.reference_speeds),
            "history_conditions": list(self.conditions),
            "eval_seeds": list(self.eval_seeds),
            "queries_per_reference_speed_per_seed": (
                self.limit_per_reference_speed_per_seed
            ),
            "bundles": len(self),
            "condition_trajectories": len(self) * len(self.conditions),
            "full_protocol": self.is_full_protocol,
            "model_visible_fields": ["pixels", "action"],
        }


def _audit_file(
    value: str,
    expected_sha256: str,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    path = resolve_contextworld_path(value, repo_root=repo_root)
    exists = path.is_file()
    observed = _sha256(path) if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": str(expected_sha256),
        "observed_sha256": observed,
        "passed": exists and observed == expected_sha256,
    }


def audit_speed_icl_release(
    *,
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    original_h5: Path | str | None = None,
    verify_all_eval_payloads: bool = False,
) -> dict[str, Any]:
    """Audit a local release root without importing Stable-WorldModel."""

    root = (repo_root or repository_root()).resolve()
    release = load_speed_icl_release(release_config)
    code_audits = [
        _audit_file(path, expected, repo_root=root)
        for path, expected in release["runtime"]["contextworld"][
            "source_sha256"
        ].items()
    ]
    file_audits = []
    for row in release["training"]["synthetic"].values():
        for name in ("catalog", "manifest", "report"):
            file_audits.append(
                _audit_file(
                    row[name], row[f"{name}_sha256"], repo_root=root
                )
            )
        catalog_path = resolve_contextworld_path(row["catalog"], repo_root=root)
        if catalog_path.is_file():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            observed_counts = {
                "train": len(catalog["train"]["synthetic"]),
                "validation": len(catalog["val"]["synthetic"]),
            }
            expected_counts = {
                "train": int(row["train_scenarios"]),
                "validation": int(row["validation_scenarios"]),
            }
            scenario_paths = [
                resolve_contextworld_path(path, repo_root=root)
                for split in ("train", "val")
                for path in catalog[split]["synthetic"]
            ]
            paths_exist = all(
                path.is_dir() for path in scenario_paths
            )
            data_root = resolve_contextworld_path(
                row["data_root"], repo_root=root
            )
            tree = (
                _tree_fingerprint(
                    data_root, hash_contents=verify_all_eval_payloads
                )
                if data_root.is_dir()
                else {
                    "files": 0,
                    "bytes": 0,
                    "sha256": None,
                    "full_hash_verified": False,
                }
            )
            tree_matches = bool(
                tree["files"] == int(row["data_tree_files"])
                and tree["bytes"] == int(row["data_tree_bytes"])
                and (
                    not verify_all_eval_payloads
                    or tree["sha256"] == row["data_tree_sha256"]
                )
            )
            file_audits.append(
                {
                    "path": str(data_root),
                    "observed_scenario_counts": observed_counts,
                    "expected_scenario_counts": expected_counts,
                    "all_scenario_paths_exist": paths_exist,
                    "observed_tree": tree,
                    "expected_tree": {
                        "files": int(row["data_tree_files"]),
                        "bytes": int(row["data_tree_bytes"]),
                        "sha256": row["data_tree_sha256"],
                    },
                    "passed": bool(
                        observed_counts == expected_counts
                        and paths_exist
                        and tree_matches
                    ),
                }
            )
    evaluation = release["evaluation"]
    for name in ("normalizer", "build_report"):
        file_audits.append(
            _audit_file(
                evaluation[name],
                evaluation[f"{name}_sha256"],
                repo_root=root,
            )
        )
    payload_count = 0
    payload_hashes_verified = 0
    eval_track_audits = {}
    for track, row in evaluation["tracks"].items():
        audit = _audit_file(
            row["catalog"], row["catalog_sha256"], repo_root=root
        )
        catalog_path = Path(audit["path"])
        if audit["passed"]:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            payload_count += len(catalog["bundles"])
            payloads_exist = True
            payload_hashes_pass = True
            for bundle in catalog["bundles"]:
                payload = resolve_contextworld_path(
                    bundle["payload"], repo_root=root
                )
                exists = payload.is_file()
                payloads_exist = payloads_exist and exists
                if verify_all_eval_payloads and exists:
                    payload_hashes_verified += 1
                    payload_hashes_pass = (
                        payload_hashes_pass
                        and _sha256(payload) == bundle["payload_sha256"]
                    )
            audit.update(
                {
                    "bundles": len(catalog["bundles"]),
                    "all_payloads_exist": payloads_exist,
                    "payload_hashes_verified": verify_all_eval_payloads,
                    "all_payload_hashes_pass": (
                        payload_hashes_pass if verify_all_eval_payloads else None
                    ),
                }
            )
            audit["passed"] = bool(
                audit["passed"]
                and payloads_exist
                and (payload_hashes_pass or not verify_all_eval_payloads)
            )
        eval_track_audits[str(track)] = audit
    planning_audits = {}
    planning_payload_hashes_verified = 0
    planning = release.get("planning", {})
    if planning:
        planning_audits["build_report"] = _audit_file(
            planning["build_report"],
            planning["build_report_sha256"],
            repo_root=root,
        )
        for track, row in planning["tracks"].items():
            audit = _audit_file(
                row["catalog"], row["catalog_sha256"], repo_root=root
            )
            if audit["passed"]:
                catalog = json.loads(
                    Path(audit["path"]).read_text(encoding="utf-8")
                )
                payloads_exist = True
                payload_hashes_pass = True
                for bundle in catalog["bundles"]:
                    payload = resolve_contextworld_path(
                        bundle["payload"], repo_root=root
                    )
                    exists = payload.is_file()
                    payloads_exist = payloads_exist and exists
                    if verify_all_eval_payloads and exists:
                        planning_payload_hashes_verified += 1
                        payload_hashes_pass = bool(
                            payload_hashes_pass
                            and _sha256(payload) == bundle["payload_sha256"]
                        )
                audit.update(
                    {
                        "bundles": len(catalog["bundles"]),
                        "all_payloads_exist": payloads_exist,
                        "payload_hashes_verified": verify_all_eval_payloads,
                        "all_payload_hashes_pass": (
                            payload_hashes_pass
                            if verify_all_eval_payloads
                            else None
                        ),
                    }
                )
                audit["passed"] = bool(
                    audit["passed"]
                    and payloads_exist
                    and (
                        payload_hashes_pass or not verify_all_eval_payloads
                    )
                )
            planning_audits[str(track)] = audit
    original_path = resolve_original_h5(
        release, repo_root=root, explicit=original_h5
    )
    original_expected = release["training"]["original"]
    original_exists = original_path.is_file()
    original_size = original_path.stat().st_size if original_exists else None
    original_hash = (
        _sha256(original_path)
        if original_exists and verify_all_eval_payloads
        else None
    )
    original_audit = {
        "path": str(original_path),
        "source": original_expected["source"],
        "license": original_expected["license"],
        "exists": original_exists,
        "expected_bytes": int(original_expected["bytes"]),
        "observed_bytes": original_size,
        "expected_sha256": original_expected["sha256"],
        "observed_sha256": original_hash,
        "full_hash_verified": bool(original_hash is not None),
        "passed": bool(
            original_exists
            and original_size == int(original_expected["bytes"])
            and (
                original_hash is None
                or original_hash == original_expected["sha256"]
            )
        ),
    }
    passed = (
        original_audit["passed"]
        and all(row["passed"] for row in code_audits)
        and all(row["passed"] for row in file_audits)
        and all(row["passed"] for row in eval_track_audits.values())
        and all(row["passed"] for row in planning_audits.values())
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if passed else "failed",
        "release_config": str(Path(release["_config_path"])),
        "artifact_root_override": os.environ.get("CONTEXTWORLD_ARTIFACT_ROOT"),
        "original_h5": original_audit,
        "contextworld_code": code_audits,
        "release_files": file_audits,
        "evaluation_tracks": eval_track_audits,
        "planning_assets": planning_audits,
        "planning_payload_hashes_verified": planning_payload_hashes_verified,
        "eval_payloads": payload_count,
        "payload_hashes_verified": payload_hashes_verified,
        "full_payload_hash_audit": verify_all_eval_payloads,
        "passed": passed,
    }


def build_speed_icl_training_data(
    swm: Any,
    *,
    recipe: str,
    original_h5: Path | str | None = None,
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    epoch_size: int | None = None,
    validation_epoch_size: int | None = None,
    seed: int = 3072,
    expected_stablewm_commit: str | None = None,
):
    """Build the exact Stable-WorldModel training mixture for a release recipe."""

    from contextworld.training.tworoom_data import build_tworoom_grouped_data

    root = (repo_root or repository_root()).resolve()
    release = load_speed_icl_release(release_config)
    recipes = release["training"]["recipes"]
    if recipe not in recipes:
        raise KeyError(f"Unknown recipe {recipe!r}; available={sorted(recipes)}")
    row = recipes[recipe]
    expected_epoch_size = int(row["epoch_size_global"])
    expected_validation_epoch_size = int(row["validation_epoch_size"])
    resolved_epoch_size = int(
        expected_epoch_size if epoch_size is None else epoch_size
    )
    resolved_validation_epoch_size = int(
        expected_validation_epoch_size
        if validation_epoch_size is None
        else validation_epoch_size
    )
    if resolved_epoch_size != expected_epoch_size:
        raise ValueError(
            f"epoch_size is frozen at {expected_epoch_size} for {recipe}"
        )
    if resolved_validation_epoch_size != expected_validation_epoch_size:
        raise ValueError(
            "validation_epoch_size is frozen at "
            f"{expected_validation_epoch_size} for {recipe}"
        )
    allowed_seeds = {int(value) for value in row["training_seeds"]}
    if int(seed) not in allowed_seeds:
        raise ValueError(
            f"Training seed for {recipe} must be one of "
            f"{sorted(allowed_seeds)}"
        )
    pinned_commit = release["runtime"]["stable_worldmodel"]["expected_ref"]
    if expected_stablewm_commit != pinned_commit:
        raise ValueError(
            "Stable-WorldModel commit does not match the frozen release: "
            f"{expected_stablewm_commit!r} != {pinned_commit!r}"
        )
    original = resolve_original_h5(
        release, repo_root=root, explicit=original_h5
    )
    benchmark_config = resolve_contextworld_path(
        row["benchmark_config"], repo_root=root
    )
    return build_tworoom_grouped_data(
        swm,
        repo_root=root,
        benchmark_config=benchmark_config,
        model_id=str(row["model_id"]),
        epoch_size=resolved_epoch_size,
        validation_epoch_size=resolved_validation_epoch_size,
        original_h5=original,
        frameskip=5,
        num_steps=4,
        img_size=224,
        seed=int(seed),
        expected_stablewm_commit=expected_stablewm_commit,
    )


def export_speed_icl_artifacts(
    destination: Path | str,
    *,
    release_config: Path | str = DEFAULT_RELEASE_CONFIG,
    repo_root: Path | None = None,
    mode: str = "copy",
    include_single_speed_control: bool = True,
) -> dict[str, Any]:
    """Materialize the redistributable artifacts under a portable root.

    The upstream original H5 is intentionally not copied; the release config
    records its official source and checksum.  ``symlink`` is useful for a
    zero-copy local validation, while ``copy`` creates an uploadable tree.
    """

    if mode not in {"copy", "symlink"}:
        raise ValueError("Export mode must be 'copy' or 'symlink'")
    root = (repo_root or repository_root()).resolve()
    release = load_speed_icl_release(release_config)
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Export destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []
    synthetic = release["training"]["synthetic"]
    selected = ["multi_speed_target"]
    if include_single_speed_control:
        selected.append("single_speed_control")
    for name in selected:
        row = synthetic[name]
        entries.append((row["data_root"], "directory"))
        for key in ("catalog", "manifest", "report"):
            entries.append((row[key], "file"))
    evaluation = release["evaluation"]
    entries.extend(
        [
            (evaluation["normalizer"], "file"),
            (
                "artifacts/evaluation/history3/speed_multistep_extrap_v5/catalogs",
                "directory",
            ),
            (
                "artifacts/evaluation/history3/speed_multistep_extrap_v5/payloads",
                "directory",
            ),
        ]
    )
    if release.get("planning"):
        entries.extend(
            [
                (
                    "artifacts/evaluation/history3/speed_isolated_v2/catalogs",
                    "directory",
                ),
                (
                    "artifacts/evaluation/history3/speed_isolated_v2/payloads",
                    "directory",
                ),
            ]
        )
    unique_entries = []
    seen = set()
    for logical, kind in entries:
        if logical in seen:
            continue
        seen.add(logical)
        unique_entries.append((logical, kind))

    inventory = []
    tree_contracts = {
        row["data_root"]: {
            "files": int(row["data_tree_files"]),
            "bytes": int(row["data_tree_bytes"]),
            "sha256": row["data_tree_sha256"],
        }
        for row in synthetic.values()
    }
    for logical, kind in unique_entries:
        source = resolve_contextworld_path(logical, repo_root=root)
        relative = Path(logical)
        if relative.parts[0] == "artifacts":
            relative = Path(*relative.parts[1:])
        target = destination / relative
        if kind == "file":
            if not source.is_file():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            inventory.append(
                {
                    "logical_path": logical,
                    "kind": kind,
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
        else:
            if not source.is_dir():
                raise FileNotFoundError(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if mode == "symlink":
                target.symlink_to(source, target_is_directory=True)
            else:
                shutil.copytree(source, target)
            inventory.append(
                {
                    "logical_path": logical,
                    "kind": kind,
                    "export_mode": mode,
                    **(
                        {"expected_tree": tree_contracts[logical]}
                        if logical in tree_contracts
                        else {}
                    ),
                }
            )
    config_target = destination / "release/tworoom_speed_icl_release_v1.yaml"
    config_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(release["_config_path"]), config_target)
    payload = {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed",
        "artifact_root": str(destination),
        "mode": mode,
        "includes_original_h5": False,
        "original_h5_source": release["training"]["original"]["source"],
        "includes_single_speed_control": include_single_speed_control,
        "entries": inventory,
    }
    inventory_path = destination / "release/inventory.json"
    inventory_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**payload, "inventory": str(inventory_path)}


__all__ = [
    "DEFAULT_RELEASE_CONFIG",
    "HORIZONS",
    "ORIGINAL_H5_ENV",
    "SpeedICLEvalBundle",
    "SpeedICLEvalDataset",
    "SpeedICLHistory",
    "audit_speed_icl_release",
    "build_speed_icl_training_data",
    "export_speed_icl_artifacts",
    "load_speed_icl_release",
    "resolve_original_h5",
]
