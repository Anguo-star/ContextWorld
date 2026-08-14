from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import lance
import numpy as np
from PIL import Image
import yaml

from contextworld.benchmarks.causal_data_contract import (
    audit_causal_data_contract,
)
from contextworld.paths import repository_root, resolve_contextworld_path


MOTION_DAMPING_RELEASE_ID = (
    "contextworld_pusht_motion_damping_icl_history3_v1"
)
DEFAULT_MOTION_DAMPING_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml"
)
DAMPING_MODES = ("faster_decay", "no_extra_decay")
DAMPING_VALUES = (0.2, 1.0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_motion_damping_icl_release(
    path: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported Motion Damping release: {config_path}")
    if payload.get("release_id") != MOTION_DAMPING_RELEASE_ID:
        raise ValueError(f"Unexpected Motion Damping release id: {config_path}")
    if payload.get("release_status") not in {
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported Motion Damping release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Motion Damping requires History=3")
    if scope.get("damping_values") != list(DAMPING_VALUES):
        raise ValueError("Motion Damping requires damping values [0.2, 1.0]")
    if scope.get("public_test_included") is not True:
        raise ValueError("Motion Damping must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("Motion Damping has no sealed Test")
    return {**payload, "_config_path": str(config_path)}


@dataclass(frozen=True)
class MotionDampingEvalArrays:
    pair_ids: tuple[str, ...]
    faster_decay_pixels: np.ndarray
    no_extra_decay_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    faster_decay_states: np.ndarray
    no_extra_decay_states: np.ndarray
    faster_decay_physics_states: np.ndarray
    no_extra_decay_physics_states: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)

    # The Stable-WorldModel reference trainer consumes a generic two-endpoint
    # paired array contract.  These aliases avoid duplicating its GPU-heavy
    # training implementation while keeping the public names self-explanatory.
    @property
    def low_pixels(self) -> np.ndarray:
        return self.faster_decay_pixels

    @property
    def high_pixels(self) -> np.ndarray:
        return self.no_extra_decay_pixels

    @property
    def low_states(self) -> np.ndarray:
        return self.faster_decay_states

    @property
    def high_states(self) -> np.ndarray:
        return self.no_extra_decay_states

    @property
    def low_physics_states(self) -> np.ndarray:
        return self.faster_decay_physics_states

    @property
    def high_physics_states(self) -> np.ndarray:
        return self.no_extra_decay_physics_states


def _decode_rgb(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path,
    *,
    expected_pairs: int,
    expected_split: str,
) -> MotionDampingEvalArrays:
    table = lance.dataset(path).to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "action",
            "state",
            "physics_state",
            "pair_id",
            "hidden_mode",
            "hidden_motion_damping",
            "split",
        ]
    )
    episode_indices = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_indices = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    physics = np.asarray(table["physics_state"].to_pylist(), dtype=np.float32)
    pixels = table["pixels"].to_pylist()
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    damping = np.asarray(
        table["hidden_motion_damping"].to_pylist(), dtype=np.float32
    ).reshape(-1)
    splits = table["split"].to_pylist()
    episodes: dict[str, dict[str, tuple[np.ndarray, ...]]] = {}
    unique_episodes = np.unique(episode_indices)
    if len(unique_episodes) != 2 * expected_pairs:
        raise RuntimeError("Unexpected motion-damping episode count")
    expected_value = dict(zip(DAMPING_MODES, DAMPING_VALUES, strict=True))
    for episode in unique_episodes:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(20)):
            raise RuntimeError(f"Episode {episode} is not a complete 20-row clip")
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        damping_values = {float(damping[index]) for index in rows}
        if len(pair_values) != 1 or len(mode_values) != 1:
            raise RuntimeError(f"Episode {episode} changes pair or mode")
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in DAMPING_MODES or split_values != {expected_split}:
            raise RuntimeError(f"Unexpected mode or split for {pair_id}")
        if len(damping_values) != 1 or not np.isclose(
            damping_values.pop(), expected_value[mode], atol=1e-7, rtol=0.0
        ):
            raise RuntimeError(f"Unexpected damping for {pair_id}/{mode}")
        frame_rows = rows[[0, 5, 10, 15]]
        pair = episodes.setdefault(pair_id, {})
        pair[mode] = (
            np.stack([_decode_rgb(pixels[index]) for index in frame_rows]),
            actions[rows].reshape(4, 5, 2),
            states[frame_rows],
            physics[frame_rows],
        )
    if len(episodes) != expected_pairs:
        raise RuntimeError("Unexpected motion-damping pair count")
    ordered_ids = tuple(sorted(episodes))
    faster_pixels, slower_pixels = [], []
    action_blocks, faster_states, slower_states = [], [], []
    faster_physics, slower_physics = [], []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != set(DAMPING_MODES):
            raise RuntimeError(f"Incomplete motion-damping pair {pair_id}")
        faster, slower = (pair[mode] for mode in DAMPING_MODES)
        if not np.array_equal(faster[0][2], slower[0][2]):
            raise RuntimeError(f"Query image differs for {pair_id}")
        if not np.array_equal(faster[1], slower[1]):
            raise RuntimeError(f"Actions differ for {pair_id}")
        if np.array_equal(faster[0][1], slower[0][1]):
            raise RuntimeError(f"History does not reveal damping for {pair_id}")
        if np.array_equal(faster[0][3], slower[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        if not np.array_equal(faster[3][2], slower[3][2]):
            raise RuntimeError(f"Query physics differs for {pair_id}")
        faster_pixels.append(faster[0])
        slower_pixels.append(slower[0])
        action_blocks.append(faster[1])
        faster_states.append(faster[2])
        slower_states.append(slower[2])
        faster_physics.append(faster[3])
        slower_physics.append(slower[3])
    # Strict v3 deliberately permits a mode-specific x0 inside each causal
    # pair. Adjacent forward/reverse twins exchange the two rendered x0
    # images, so their split-level RGB multisets must be exactly equal. This
    # makes a deterministic x0-only label rule no better than 50%. Legacy v1
    # also satisfies this split-level condition because each pair shared x0.
    faster_x0 = Counter(
        hashlib.sha256(value[0].tobytes()).hexdigest()
        for value in faster_pixels
    )
    slower_x0 = Counter(
        hashlib.sha256(value[0].tobytes()).hexdigest()
        for value in slower_pixels
    )
    if faster_x0 != slower_x0:
        raise RuntimeError(
            "Motion-damping x0 RGB hash multisets differ across modes"
        )
    return MotionDampingEvalArrays(
        pair_ids=ordered_ids,
        faster_decay_pixels=np.stack(faster_pixels),
        no_extra_decay_pixels=np.stack(slower_pixels),
        raw_action_blocks=np.stack(action_blocks),
        faster_decay_states=np.stack(faster_states),
        no_extra_decay_states=np.stack(slower_states),
        faster_decay_physics_states=np.stack(faster_physics),
        no_extra_decay_physics_states=np.stack(slower_physics),
    )


class MotionDampingICLEvalDataset:
    """Frozen 256-pair motion-damping Public Test."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_motion_damping_icl_release(release_config)
        self.root = resolve_contextworld_path(
            self.release["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: MotionDampingEvalArrays | None = None

    @property
    def arrays(self) -> MotionDampingEvalArrays:
        if self._arrays is None:
            evaluation = self.release["evaluation"]
            self._arrays = _read_lance_pairs(
                self.root / evaluation["lance_table"],
                expected_pairs=int(evaluation["pair_count"]),
                expected_split="validation",
            )
        return self._arrays

    @property
    def is_full_protocol(self) -> bool:
        return self.arrays.pair_count == int(self.release["evaluation"]["pair_count"])

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "motion_damping_values": list(DAMPING_VALUES),
            "online_environment_calls": 0,
        }


def audit_motion_damping_icl_release(
    *,
    release_config: Path | str = DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_motion_damping_icl_release(release_config)
    files = {}
    for group_name, group in (
        ("identity", release["identity"]),
        ("data", release["data"]["artifacts"]),
    ):
        for name, specification in group.items():
            path = resolve_contextworld_path(specification["path"], repo_root=root)
            observed = file_sha256(path) if path.is_file() else None
            identity_matched = observed == specification["sha256"]
            required_for_release_audit = specification.get(
                "required_for_release_audit", True
            )
            files[f"{group_name}.{name}"] = {
                "path": str(path),
                "exists": path.is_file(),
                "expected_sha256": specification["sha256"],
                "observed_sha256": observed,
                "identity_matched": identity_matched,
                "required_for_release_audit": required_for_release_audit,
                "role": specification.get("role"),
                "drift_policy": specification.get("drift_policy"),
                "passed": bool(
                    identity_matched or not required_for_release_audit
                ),
            }
    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"], repo_root=root
    )
    manifest = json.loads((data_root / "manifest.json").read_text())
    portability_path = resolve_contextworld_path(
        release["data"]["artifacts"]["portability_receipt"]["path"],
        repo_root=root,
    )
    portability_receipt = json.loads(
        portability_path.read_text(encoding="utf-8")
    )
    strict_causal = (
        release["data"]["protocol"]
        == "pusht_motion_damping_history3_strict_causal_release_v3"
    )
    counts = {key: int(value) for key, value in release["data"]["pair_counts"].items()}
    row_counts = {
        split: int(lance.dataset(data_root / release["data"]["lance_tables"][split]).count_rows())
        for split in counts
    }
    specification = release["data"]["artifact_tree"]
    children = [path for path in data_root.rglob("*") if path.is_file()]
    tree_hash = directory_sha256(data_root) if full else None
    tree_passed = bool(
        len(children) == int(specification["files"])
        and sum(path.stat().st_size for path in children) == int(specification["bytes"])
        and (not full or tree_hash == specification["sha256"])
    )
    data_checks = {
        "manifest_passed": manifest.get("passed") is True,
        "cross_split_isolation_passed": manifest["cross_split_audit"]["passed"] is True,
        "pair_counts_match": manifest["pair_counts"] == counts,
        "row_counts_match": all(
            row_counts[split] == 40 * count for split, count in counts.items()
        ),
        "split_audits_passed": all(
            manifest["splits"][split]["passed"] is True for split in counts
        ),
        "query_hashes_unique": all(
            manifest["splits"][split]["query_hash_count"] == counts[split]
            for split in counts
        ),
        "portable_metadata_receipt_passed": bool(
            portability_receipt.get("passed") is True
            and portability_receipt.get("absolute_path_audit", {}).get(
                "passed"
            )
            is True
            and all(
                row.get("identical") is True
                for row in portability_receipt.get(
                    "lance_tables", {}
                ).values()
            )
        ),
    }
    frozen_evaluation = manifest.get(
        "evaluation_tables_reused_byte_for_byte"
    )
    if frozen_evaluation is not None:
        frozen_splits = ("loader_validation", "validation")
        data_checks.update(
            {
                "frozen_evaluation_split_names_match": (
                    frozen_evaluation.get("splits")
                    == list(frozen_splits)
                ),
                "frozen_evaluation_split_receipts_passed": all(
                    manifest["splits"][split]
                    .get("frozen_split_reuse", {})
                    .get("passed")
                    is True
                    for split in frozen_splits
                ),
                "frozen_evaluation_table_hashes_preserved": all(
                    (
                        receipt.get("source_table_sha256")
                        == receipt.get("destination_table_sha256")
                        and receipt.get("model_visible_bytes_preserved")
                        is True
                        and receipt.get("pair_identity_preserved") is True
                    )
                    for receipt in (
                        manifest["splits"][split]["frozen_split_reuse"]
                        for split in frozen_splits
                    )
                ),
            }
        )
    if strict_causal:
        total_pair_count = int(sum(counts.values()))
        total_condition_count = 2 * total_pair_count
        causal = manifest.get("causal_audit", {})
        strict_audit_path = resolve_contextworld_path(
            release["data"]["artifacts"]["strict_causal_audit"]["path"],
            repo_root=root,
        )
        strict_audit = json.loads(strict_audit_path.read_text(encoding="utf-8"))
        identifiability_path = resolve_contextworld_path(
            release["data"]["artifacts"]["history_identifiability_audit"][
                "path"
            ],
            repo_root=root,
        )
        identifiability = json.loads(
            identifiability_path.read_text(encoding="utf-8")
        )
        classifier = causal.get(
            "training_to_public_test_x0_only_geometry_classifier", {}
        )
        data_checks.update(
            {
                "strict_protocol_v3": manifest.get("protocol")
                == "pusht_motion_damping_history3_strict_causal_release_v3",
                "state_installations_after_x0_zero": causal.get(
                    "state_installations_after_x0"
                )
                == 0,
                "query_simulator_not_recreated": causal.get(
                    "query_simulator_recreated"
                )
                is False,
                "query_full_state_within_tolerance": float(
                    causal.get("max_pair_full_state_gap", float("inf"))
                )
                <= float(causal.get("query_full_state_tolerance", 0.0)),
                "query_rgb_exact": causal.get(
                    "max_pair_query_pixel_difference"
                )
                == 0,
                "query_action_exact": float(
                    causal.get("max_pair_query_action_difference", float("inf"))
                )
                == 0.0,
                "continuous_chain_has_no_arbiter": causal.get(
                    "maximum_arbiter_count_from_x0_through_x3"
                )
                == 0,
                "x0_rgb_multisets_balanced": causal.get(
                    "all_split_x0_rgb_hash_multisets_identical_across_modes"
                )
                is True,
                "x0_static_bayes_bound_is_chance": float(
                    causal.get(
                        "maximum_x0_rgb_static_bayes_accuracy_upper_bound",
                        float("inf"),
                    )
                )
                == 0.5,
                "x0_training_to_public_classifier_passed": (
                    classifier.get("passed") is True
                    and float(classifier.get("public_test_accuracy", 1.0))
                    <= float(classifier.get("maximum_allowed_accuracy", 0.0))
                ),
                "strict_clean_replay_audit_passed": (
                    strict_audit.get("passed") is True
                    and strict_audit.get("pair_count") == sum(counts.values())
                    and strict_audit.get("condition_count")
                    == 2 * sum(counts.values())
                    and strict_audit.get("maximum_continuous_arbiter_count") == 0
                    and strict_audit.get(
                        "maximum_clean_start_active_arbiter_count"
                    )
                    == 0
                    and strict_audit.get(
                        "maximum_continuous_vs_clean_x3_full_state_gap"
                    )
                    == 0.0
                    and strict_audit.get(
                        "maximum_continuous_vs_clean_x3_pixel_difference"
                    )
                    == 0
                ),
                "visible_history_identifiability_audit_passed": (
                    identifiability.get("passed") is True
                    and identifiability.get("model_or_checkpoint_loaded")
                    is False
                    and identifiability.get("feature", {}).get("name")
                    == "rgb_only_block_motion_decay_ratio"
                    and identifiability.get("feature", {}).get(
                        "input_feature_columns"
                    )
                    == ["pixels"]
                    and identifiability.get("feature", {}).get(
                        "hidden_mode_used_as_input_feature"
                    )
                    is False
                    and identifiability.get("feature", {}).get(
                        "state_or_physics_used_as_input_feature"
                    )
                    is False
                    and identifiability.get("feature", {})
                    .get("segmentation", {})
                    .get("parameters_fitted_from_training_or_labels")
                    is False
                    and identifiability.get("release", {}).get(
                        "published_manifest_sha256"
                    )
                    == release["data"]["manifest_sha256"]
                    and identifiability.get("release", {}).get(
                        "training_manifest_sha256"
                    )
                    == portability_receipt.get("metadata_sha256", {})
                    .get("manifest.json", {})
                    .get("before")
                    and identifiability.get("release", {}).get(
                        "portability_receipt_sha256"
                    )
                    == release["data"]["artifacts"]
                    ["portability_receipt"]["sha256"]
                    and identifiability.get("threshold", {}).get(
                        "selection_split"
                    )
                    == "Training"
                    and all(
                        identifiability.get("splits", {})
                        .get(split, {})
                        .get("accuracy")
                        == 1.0
                        for split in (
                            "train",
                            "loader_validation",
                            "validation",
                        )
                    )
                    and identifiability.get("public_test_role")
                    == (
                        "data_identifiability_audit_only_not_recipe_or_"
                        "checkpoint_selection"
                    )
                ),
            }
        )
        x0_leakage_passed = bool(
            data_checks["x0_rgb_multisets_balanced"]
            and data_checks["x0_static_bayes_bound_is_chance"]
            and data_checks["x0_training_to_public_classifier_passed"]
        )
        clean_replay_passed = bool(
            data_checks["strict_clean_replay_audit_passed"]
        )
        causal_contract = audit_causal_data_contract(
            component_id=release["release_id"],
            evidence_scope=(
                f"all_{total_pair_count}_pairs_and_"
                f"{total_condition_count}_clean_replays"
            ),
            continuous_environment_trajectory=(
                causal.get("state_installations_after_x0") == 0
                and causal.get("query_simulator_recreated") is False
            ),
            state_installations_after_x0=int(
                causal.get("state_installations_after_x0", -1)
            ),
            query_simulator_recreated=bool(
                causal.get("query_simulator_recreated", True)
            ),
            maximum_query_state_gap=float(
                causal.get("max_pair_full_state_gap", float("inf"))
            ),
            query_state_tolerance=float(
                causal.get("query_full_state_tolerance", 0.0)
            ),
            query_pixels_exact=(
                causal.get("max_pair_query_pixel_difference") == 0
            ),
            query_actions_exact=(
                float(
                    causal.get(
                        "max_pair_query_action_difference", float("inf")
                    )
                )
                == 0.0
            ),
            history_effect_present=(
                float(causal.get("min_history_effect", 0.0)) >= 3.0
                and data_checks[
                    "visible_history_identifiability_audit_passed"
                ]
            ),
            true_future_effect_present=(
                float(causal.get("min_true_future_effect", 0.0)) >= 2.0
            ),
            x0_policy="balanced_visible_start",
            x0_static_leakage_check_passed=x0_leakage_passed,
            solver_cache_check_required=True,
            solver_cache_check_passed=clean_replay_passed,
            evidence=(
                "manifest.causal_audit across all Training, Development, and Public Test pairs",
                (
                    "strict_causal_audit clean replay across all "
                    f"{total_condition_count} conditions"
                ),
                "balanced forward/reverse twin x0 RGB multisets",
                "train-to-Public x0-only visible-geometry classifier",
                "Training-frozen visible block-motion ratio with 100% accuracy on all splits",
            ),
        )
    else:
        causal_contract = {
            "schema_version": 1,
            "component_id": release["release_id"],
            "evidence_scope": "previous_data_revision",
            "passed": False,
        }
    payload = {"decoded_pairs": 0, "passed": True}
    if full:
        dataset = MotionDampingICLEvalDataset(release=release, repo_root=root)
        payload = {
            "decoded_pairs": dataset.arrays.pair_count,
            "query_physics_exactly_paired": bool(
                np.array_equal(
                    dataset.arrays.faster_decay_physics_states[:, 2],
                    dataset.arrays.no_extra_decay_physics_states[:, 2],
                )
            ),
            "x0_rgb_hash_multisets_balanced": True,
            "passed": dataset.is_full_protocol,
        }
        payload["passed"] = bool(
            payload["passed"] and payload["query_physics_exactly_paired"]
        )

    decision_specification = release.get("reference_results", {}).get(
        "current_decision", {}
    )
    decision_path = resolve_contextworld_path(
        decision_specification.get("path", ""), repo_root=root
    )
    decision_sha256 = (
        file_sha256(decision_path) if decision_path.is_file() else None
    )
    decision_file = {
        "path": str(decision_path),
        "exists": decision_path.is_file(),
        "expected_sha256": decision_specification.get("sha256"),
        "observed_sha256": decision_sha256,
        "passed": decision_sha256 == decision_specification.get("sha256"),
    }
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {}
    )
    endpoint = decision.get("reported_endpoint", {})
    decision_metrics = endpoint.get("metrics", {})
    decision_gates = endpoint.get("gates", {})
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_damping_correct_future_rate",
    )
    threshold_keys = {
        name: f"{name}_minimum" for name in metric_names
    }
    registered_thresholds = release["scoring"][
        "hidden_future_prediction"
    ]["gates"]
    computed_gate_passes = {
        name: bool(
            float(decision_metrics.get(name, -np.inf))
            >= float(registered_thresholds[threshold_keys[name]])
        )
        for name in metric_names
    }
    failed_metrics = [
        name for name in metric_names if not computed_gate_passes[name]
    ]
    expected_endpoint = release["training"]["reference_matrix"][
        "reported_endpoint"
    ]
    receipt = endpoint.get("training_receipt", {})
    hash_fields = (
        "runner_sha256",
        "training_report_sha256",
        "checkpoint_sha256",
    )
    result_checks = {
        "status_is_failed_development": (
            decision.get("status") == "failed_development"
        ),
        "current_manifest_is_bound": (
            decision.get("data_release", {}).get(
                "published_manifest_sha256"
            )
            == release["data"]["manifest_sha256"]
            == files["data.manifest"].get("observed_sha256")
        ),
        "training_manifest_is_mapped_by_portability_receipt": bool(
            decision.get("data_release", {}).get(
                "training_manifest_sha256"
            )
            == portability_receipt.get("metadata_sha256", {})
            .get("manifest.json", {})
            .get("before")
            and release["data"]["manifest_sha256"]
            == portability_receipt.get("metadata_sha256", {})
            .get("manifest.json", {})
            .get("after")
            and decision.get("data_release", {}).get(
                "portability_receipt_sha256"
            )
            == release["data"]["artifacts"]["portability_receipt"][
                "sha256"
            ]
            == files["data.portability_receipt"].get("observed_sha256")
            and decision.get("data_release", {}).get(
                "model_visible_lance_tables_unchanged_by_metadata_migration"
            )
            is True
        ),
        "pair_counts_match_release": (
            decision.get("data_release", {}).get("pair_counts")
            == {
                "training": counts["train"],
                "development": counts["loader_validation"],
                "public_test": counts["validation"],
            }
        ),
        "exactly_four_registered_metrics": bool(
            set(decision_metrics) == set(metric_names)
            and all(
                np.isfinite(float(decision_metrics[name]))
                and 0.0 <= float(decision_metrics[name]) <= 1.0
                for name in metric_names
            )
        ),
        "four_gates_match_registered_thresholds": bool(
            set(decision_gates) == set(metric_names)
            and all(
                np.isclose(
                    float(decision_gates[name].get("minimum", np.nan)),
                    float(registered_thresholds[threshold_keys[name]]),
                    atol=0.0,
                    rtol=0.0,
                )
                and decision_gates[name].get("passed")
                is computed_gate_passes[name]
                for name in metric_names
            )
        ),
        "development_gate_failed": bool(
            failed_metrics
            and endpoint.get("passed") is False
            and endpoint.get("failed_metrics") == failed_metrics
        ),
        "reported_endpoint_matches_release": bool(
            endpoint.get("model_family")
            == expected_endpoint["model_family"]
            and receipt.get("runner")
            == release["identity"]["reference_trainer"]["path"]
            and receipt.get("runner_sha256")
            == release["identity"]["reference_trainer"]["sha256"]
            and receipt.get("recipe") == expected_endpoint["recipe"]
            and receipt.get("training_seed")
            == expected_endpoint["training_seed"]
            and receipt.get("optimizer_step")
            == expected_endpoint["optimizer_step"]
            and all(
                isinstance(receipt.get(name), str)
                and len(receipt[name]) == 64
                for name in hash_fields
            )
        ),
        "followup_evaluations_are_closed": bool(
            decision.get("public_model_scoring_opened") is False
            and decision.get("additional_training_seeds_run") is False
            and decision.get("original_task_cem_run") is False
            and decision.get("positive_reference_claim") is False
            and release.get("evaluation", {}).get(
                "model_evaluation_status"
            )
            == "not_opened_after_failed_development"
        ),
    }
    passed = bool(
        all(row["passed"] for row in files.values())
        and all(data_checks.values())
        and causal_contract["passed"]
        and tree_passed
        and payload["passed"]
        and decision_file["passed"]
        and all(result_checks.values())
    )
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if passed else "failed",
        "files": files,
        "data_checks": data_checks,
        "artifact_tree": {
            "path": str(data_root),
            "observed_files": len(children),
            "observed_bytes": sum(path.stat().st_size for path in children),
            "observed_sha256": tree_hash,
            "full_hash_checked": full,
            "passed": tree_passed,
        },
        "payload_audit": payload,
        "counts": counts,
        "row_counts": row_counts,
        "public_test_included": True,
        "sealed_test_included": False,
        "strict_causal_protocol": strict_causal,
        "causal_data_contract": causal_contract,
        "reference_result": {
            "status": decision.get("status"),
            "file": decision_file,
            "checks": result_checks,
            "metrics": decision_metrics,
            "gates": decision_gates,
            "failed_metrics": failed_metrics,
            "public_model_scoring_opened": decision.get(
                "public_model_scoring_opened"
            ),
            "positive_reference_claim": decision.get(
                "positive_reference_claim"
            ),
            "passed": bool(
                decision_file["passed"] and all(result_checks.values())
            ),
        },
        "passed": passed,
    }


__all__ = [
    "DAMPING_MODES",
    "DAMPING_VALUES",
    "DEFAULT_MOTION_DAMPING_RELEASE_CONFIG",
    "MOTION_DAMPING_RELEASE_ID",
    "MotionDampingEvalArrays",
    "MotionDampingICLEvalDataset",
    "audit_motion_damping_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_motion_damping_icl_release",
]
