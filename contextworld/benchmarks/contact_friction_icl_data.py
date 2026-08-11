from __future__ import annotations

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


CONTACT_FRICTION_RELEASE_ID = (
    "contextworld_pusht_contact_friction_icl_history3_v1"
)
DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG = (
    repository_root()
    / "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml"
)
FRICTION_MODES = ("low_friction", "high_friction")
FRICTION_VALUES = (0.05, 0.80)
STRICT_CAUSAL_PROTOCOL = (
    "pusht_contact_friction_history3_strict_continuous_v2"
)
QUERY_FULL_STATE_TOLERANCE = 1e-5
# Lance stores the auxiliary physics-state column as float32.  Near the
# 300-pixel coordinate range, one representable step is about 3.05e-5, so a
# true double-precision gap below 1e-5 may decode as one float32 step.  The
# manifest keeps the hard 1e-5 simulator-state gate; this bound is only for
# checking its serialized diagnostic copy.
SERIALIZED_QUERY_FULL_STATE_TOLERANCE = 5e-5
MINIMUM_CONTACT_FREE_STEPS_BEFORE_QUERY = 3


def _pusht_physics_state_max_abs_gap(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Compare serialized PushT snapshots with circular body angles.

    The 12-D physics snapshot stores the pusher and block angles at indices
    4 and 10.  Those coordinates are periodic, so values near zero and
    ``2*pi`` describe the same physical pose and must not be compared with a
    plain subtraction.
    """

    delta = np.asarray(left, dtype=np.float64) - np.asarray(
        right,
        dtype=np.float64,
    )
    if delta.shape[-1] != 12:
        raise ValueError(
            "PushT physics snapshots must have 12 coordinates, got "
            f"{delta.shape}"
        )
    for index in (4, 10):
        delta[..., index] = (
            delta[..., index] + np.pi
        ) % (2 * np.pi) - np.pi
    return float(np.max(np.abs(delta)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_contact_friction_icl_release(
    path: Path | str = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported Contact Friction release: {config_path}"
        )
    if payload.get("release_id") != CONTACT_FRICTION_RELEASE_ID:
        raise ValueError(
            f"Unexpected Contact Friction release id: {config_path}"
        )
    if payload.get("release_status") not in {
        "validation_release_candidate",
        "validation_release",
        "public_test_release_candidate",
        "public_test_release",
    }:
        raise ValueError("Unsupported Contact Friction release status")
    scope = payload.get("scope", {})
    if scope.get("history_tokens") != 3:
        raise ValueError("Contact Friction v1 requires History=3")
    if str(payload.get("release_status")).startswith("public_test_") and (
        scope.get("public_test_included") is not True
    ):
        raise ValueError("Contact Friction v1 must include Public Test")
    if scope.get("sealed_test_included") is not False:
        raise ValueError("The public release must not include sealed Test")
    if scope.get("friction_values") != list(FRICTION_VALUES):
        raise ValueError(
            "Contact Friction v1 requires effective coefficients "
            f"{list(FRICTION_VALUES)}"
        )
    if payload.get("data", {}).get("protocol") != STRICT_CAUSAL_PROTOCOL:
        raise ValueError(
            "Contact Friction release requires the strict continuous "
            f"causal protocol {STRICT_CAUSAL_PROTOCOL!r}"
        )
    return {**payload, "_config_path": str(config_path)}


@dataclass(frozen=True)
class ContactFrictionEvalArrays:
    pair_ids: tuple[str, ...]
    low_pixels: np.ndarray
    high_pixels: np.ndarray
    raw_action_blocks: np.ndarray
    low_states: np.ndarray
    high_states: np.ndarray
    low_physics_states: np.ndarray
    high_physics_states: np.ndarray

    @property
    def pair_count(self) -> int:
        return len(self.pair_ids)


def _decode_rgb(value: bytes) -> np.ndarray:
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_lance_pairs(
    path: Path,
    *,
    expected_pairs: int,
    expected_split: str,
) -> ContactFrictionEvalArrays:
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
            "hidden_contact_friction",
            "split",
        ]
    )
    episode_indices = np.asarray(
        table["episode_idx"].to_numpy(),
        dtype=np.int64,
    )
    step_indices = np.asarray(
        table["step_idx"].to_numpy(),
        dtype=np.int64,
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    physics = np.asarray(
        table["physics_state"].to_pylist(),
        dtype=np.float32,
    )
    pixel_bytes = table["pixels"].to_pylist()
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    frictions = np.asarray(
        table["hidden_contact_friction"].to_pylist(),
        dtype=np.float32,
    ).reshape(-1)
    splits = table["split"].to_pylist()

    episodes: dict[
        str,
        dict[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
    ] = {}
    unique_episodes = np.unique(episode_indices)
    if len(unique_episodes) != 2 * expected_pairs:
        raise RuntimeError(
            f"Expected {2 * expected_pairs} episodes, got "
            f"{len(unique_episodes)}"
        )
    for episode in unique_episodes:
        rows = np.flatnonzero(episode_indices == episode)
        rows = rows[np.argsort(step_indices[rows])]
        if not np.array_equal(step_indices[rows], np.arange(20)):
            raise RuntimeError(
                f"Episode {episode} is not a complete 20-row clip"
            )
        pair_values = {str(pair_ids[index]) for index in rows}
        mode_values = {str(modes[index]) for index in rows}
        split_values = {str(splits[index]) for index in rows}
        friction_values = {float(frictions[index]) for index in rows}
        if (
            len(pair_values) != 1
            or len(mode_values) != 1
            or split_values != {expected_split}
            or len(friction_values) != 1
        ):
            raise RuntimeError(
                f"Episode {episode} changes pair, mode, split, or friction"
            )
        pair_id = pair_values.pop()
        mode = mode_values.pop()
        if mode not in FRICTION_MODES:
            raise RuntimeError(f"Unexpected friction mode {mode!r}")
        expected_friction = dict(zip(FRICTION_MODES, FRICTION_VALUES))[mode]
        observed_friction = friction_values.pop()
        if not np.isclose(
            observed_friction,
            expected_friction,
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Unexpected friction coefficient for {pair_id}/{mode}"
            )
        frame_rows = rows[[0, 5, 10, 15]]
        pixels = np.stack(
            [_decode_rgb(pixel_bytes[index]) for index in frame_rows]
        )
        action_blocks = actions[rows].reshape(4, 5, 2)
        frame_states = states[frame_rows]
        frame_physics = physics[frame_rows]
        pair = episodes.setdefault(pair_id, {})
        if mode in pair:
            raise RuntimeError(f"Duplicate {pair_id}/{mode}")
        pair[mode] = (
            pixels,
            action_blocks,
            frame_states,
            frame_physics,
        )

    if len(episodes) != expected_pairs:
        raise RuntimeError(
            f"Expected {expected_pairs} pairs, got {len(episodes)}"
        )
    ordered_ids = tuple(sorted(episodes))
    low_pixels: list[np.ndarray] = []
    high_pixels: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    low_states: list[np.ndarray] = []
    high_states: list[np.ndarray] = []
    low_physics: list[np.ndarray] = []
    high_physics: list[np.ndarray] = []
    for pair_id in ordered_ids:
        pair = episodes[pair_id]
        if set(pair) != set(FRICTION_MODES):
            raise RuntimeError(f"Incomplete friction pair {pair_id}")
        low = pair[FRICTION_MODES[0]]
        high = pair[FRICTION_MODES[1]]
        if not np.array_equal(low[0][0], high[0][0]):
            raise RuntimeError(f"Initial image differs for {pair_id}")
        if not np.array_equal(low[0][2], high[0][2]):
            raise RuntimeError(f"Query image differs for {pair_id}")
        if not np.array_equal(low[1], high[1]):
            raise RuntimeError(f"Actions differ for {pair_id}")
        if np.array_equal(low[0][1], high[0][1]):
            raise RuntimeError(f"History does not reveal friction for {pair_id}")
        if np.array_equal(low[0][3], high[0][3]):
            raise RuntimeError(f"True futures do not differ for {pair_id}")
        gap = _pusht_physics_state_max_abs_gap(
            low[3][2],
            high[3][2],
        )
        if gap > SERIALIZED_QUERY_FULL_STATE_TOLERANCE:
            raise RuntimeError(
                f"Query physics gap {gap:.8g} exceeds "
                "the float32 serialization bound "
                f"{SERIALIZED_QUERY_FULL_STATE_TOLERANCE:.8g} for "
                f"{pair_id}"
            )
        low_pixels.append(low[0])
        high_pixels.append(high[0])
        action_blocks.append(low[1])
        low_states.append(low[2])
        high_states.append(high[2])
        low_physics.append(low[3])
        high_physics.append(high[3])
    return ContactFrictionEvalArrays(
        pair_ids=ordered_ids,
        low_pixels=np.stack(low_pixels),
        high_pixels=np.stack(high_pixels),
        raw_action_blocks=np.stack(action_blocks),
        low_states=np.stack(low_states),
        high_states=np.stack(high_states),
        low_physics_states=np.stack(low_physics),
        high_physics_states=np.stack(high_physics),
    )


class ContactFrictionICLEvalDataset:
    """Frozen 256-pair independent Contact Friction Public Test."""

    def __init__(
        self,
        *,
        release: dict[str, Any] | None = None,
        release_config: (
            Path | str
        ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = (repo_root or repository_root()).resolve()
        self.release = release or load_contact_friction_icl_release(
            release_config
        )
        self.root = resolve_contextworld_path(
            self.release["data"]["artifact_tree"]["root"],
            repo_root=self.repo_root,
        )
        self._arrays: ContactFrictionEvalArrays | None = None

    @property
    def arrays(self) -> ContactFrictionEvalArrays:
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
        return self.arrays.pair_count == int(
            self.release["evaluation"]["pair_count"]
        )

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "pair_count": self.arrays.pair_count,
            "condition_count": 2 * self.arrays.pair_count,
            "history_tokens": 3,
            "effective_contact_friction_values": list(FRICTION_VALUES),
            "online_environment_calls": 0,
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
    exists = path.is_file()
    observed = file_sha256(path) if exists else None
    expected = str(specification["sha256"])
    return {
        "path": str(path),
        "exists": exists,
        "expected_sha256": expected,
        "observed_sha256": observed,
        "passed": bool(exists and observed == expected),
    }


def audit_contact_friction_icl_release(
    *,
    release_config: (
        Path | str
    ) = DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
    repo_root: Path | None = None,
    full: bool = False,
) -> dict[str, Any]:
    root = (repo_root or repository_root()).resolve()
    release = load_contact_friction_icl_release(release_config)
    identity_files: dict[str, Any] = {}
    for name, specification in release["identity"].items():
        identity_files[name] = _verified_file(
            specification,
            repo_root=root,
        )
    data_files: dict[str, Any] = {}
    for name, specification in release["data"]["artifacts"].items():
        data_files[name] = _verified_file(
            specification,
            repo_root=root,
        )
    reference_files: dict[str, Any] = {}
    for name, specification in release.get(
        "reference_results",
        {},
    ).items():
        reference_files[name] = _verified_file(
            specification,
            repo_root=root,
        )

    data_root = resolve_contextworld_path(
        release["data"]["artifact_tree"]["root"],
        repo_root=root,
    )
    manifest = json.loads(
        (data_root / "manifest.json").read_text(encoding="utf-8")
    )
    expanded_audit_specification = release["data"]["artifacts"][
        "expanded_release_audit"
    ]
    expanded_audit_path = resolve_contextworld_path(
        expanded_audit_specification["path"],
        repo_root=root,
    )
    expanded_audit = json.loads(
        expanded_audit_path.read_text(encoding="utf-8")
    )
    portability_specification = release["data"]["artifacts"][
        "portability_receipt"
    ]
    portability_path = resolve_contextworld_path(
        portability_specification["path"],
        repo_root=root,
    )
    portability_receipt = json.loads(
        portability_path.read_text(encoding="utf-8")
    )
    expected_counts = {
        key: int(value)
        for key, value in release["data"]["pair_counts"].items()
    }
    row_counts = {
        split: int(
            lance.dataset(
                data_root / release["data"]["lance_tables"][split]
            ).count_rows()
        )
        for split in expected_counts
    }
    causal_chain = manifest.get("causal_chain", {})
    split_audits = manifest.get("splits", {})

    def strict_split_gate(split: str) -> bool:
        audit = split_audits.get(split, {})
        direction_bins = audit.get(
            "initial_offset_direction_bin_counts",
            {},
        )
        order_counts = audit.get("initial_offset_order_counts", {})
        return bool(
            audit.get("passed") is True
            and audit.get("all_paired_x0_pixels_bitwise_identical") is True
            and float(audit.get("max_pair_full_state_gap", np.inf))
            <= QUERY_FULL_STATE_TOLERANCE
            and float(
                audit.get("maximum_natural_query_target_residual", np.inf)
            )
            <= QUERY_FULL_STATE_TOLERANCE
            and float(audit.get("max_pair_query_action_difference", np.inf))
            == 0.0
            and int(audit.get("max_pair_query_pixel_difference", -1)) == 0
            and int(audit.get("minimum_cache_clear_steps_before_query", -1))
            >= MINIMUM_CONTACT_FREE_STEPS_BEFORE_QUERY
            and float(
                audit.get(
                    "maximum_clean_simulator_replay_full_state_gap",
                    np.inf,
                )
            )
            <= QUERY_FULL_STATE_TOLERANCE
            and float(audit.get("minimum_history_gap_px_equivalent", 0.0))
            > 0.0
            and float(audit.get("minimum_future_block_position_gap_px", 0.0))
            > 0.0
            and np.isclose(
                float(audit.get("mode_label_static_x0_accuracy", np.nan)),
                0.5,
                atol=0.0,
                rtol=0.0,
            )
            and len(direction_bins) == 8
            and all(int(value) > 0 for value in direction_bins.values())
            and int(order_counts.get("low_offset_larger", 0)) > 0
            and int(order_counts.get("high_offset_larger", 0)) > 0
            and int(order_counts.get("equal", -1)) == 0
        )

    data_checks = {
        "strict_continuous_protocol": (
            manifest.get("protocol") == STRICT_CAUSAL_PROTOCOL
            and release["data"].get("protocol") == STRICT_CAUSAL_PROTOCOL
        ),
        "no_state_installation_after_x0": (
            causal_chain.get("state_installations_after_x0") == 0
        ),
        "query_simulator_not_recreated": (
            causal_chain.get("query_simulator_recreated") is False
        ),
        "paired_x0_pixels_bitwise_identical": (
            causal_chain.get("paired_x0_pixels_must_be_bitwise_identical")
            is True
        ),
        "clean_replay_is_diagnostic_only": (
            causal_chain.get("clean_simulator_replay_is_diagnostic_only")
            is True
        ),
        "registered_query_tolerance_matches": bool(
            np.isclose(
                float(
                    causal_chain.get(
                        "query_full_state_tolerance",
                        np.nan,
                    )
                ),
                QUERY_FULL_STATE_TOLERANCE,
                atol=0.0,
                rtol=0.0,
            )
        ),
        "registered_contact_free_steps_match": (
            causal_chain.get("minimum_contact_free_steps_before_query")
            == MINIMUM_CONTACT_FREE_STEPS_BEFORE_QUERY
        ),
        "manifest_passed": manifest.get("passed") is True,
        "manifest_sha256_bound": bool(
            release["data"].get("manifest_sha256")
            == release["data"]["artifacts"]["manifest"].get("sha256")
            == data_files["manifest"].get("observed_sha256")
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
                    "lance_tables",
                    {},
                ).values()
            )
        ),
        "cross_split_isolation_passed": (
            manifest.get("cross_split_audit", {}).get("passed") is True
        ),
        "pair_counts_match": (
            manifest.get("pair_counts") == expected_counts
        ),
        "lance_row_counts_match": all(
            row_counts[split] == 2 * pair_count * 20
            for split, pair_count in expected_counts.items()
        ),
        "all_split_audits_passed": all(
            strict_split_gate(split) for split in expected_counts
        ),
        "query_hashes_unique_within_each_split": all(
            manifest["splits"][split]["query_hash_count"]
            == expected_counts[split]
            for split in expected_counts
        ),
        "expanded_release_audit_passed": bool(
            expanded_audit.get("passed") is True
            and expanded_audit.get("manifest_sha256")
            == release["data"]["manifest_sha256"]
            and expanded_audit.get("artifact_tree")
            == {
                "files": int(release["data"]["artifact_tree"]["files"]),
                "bytes": int(release["data"]["artifact_tree"]["bytes"]),
                "sha256": release["data"]["artifact_tree"]["sha256"],
            }
            and all(expanded_audit.get("checks", {}).values())
            and {
                split: int(report.get("pair_count", -1))
                for split, report in expanded_audit.get(
                    "split_reports",
                    {},
                ).items()
            }
            == expected_counts
        ),
    }

    published_pair_count = sum(expected_counts.values())
    x0_static_leakage_passed = bool(
        published_pair_count == 8704
        and all(
            split_audits[split].get(
                "all_paired_x0_pixels_bitwise_identical"
            )
            is True
            and np.isclose(
                float(
                    split_audits[split].get(
                        "mode_label_static_x0_accuracy",
                        np.nan,
                    )
                ),
                0.5,
                atol=0.0,
                rtol=0.0,
            )
            and split_audits[split].get(
                "strict_family_coverage_passed"
            )
            is True
            and len(
                split_audits[split].get(
                    "initial_offset_direction_bin_counts",
                    {},
                )
            )
            == 8
            and all(
                int(value) > 0
                for value in split_audits[split]
                .get("initial_offset_direction_bin_counts", {})
                .values()
            )
            and int(
                split_audits[split]
                .get("initial_offset_order_counts", {})
                .get("low_offset_larger", 0)
            )
            > 0
            and int(
                split_audits[split]
                .get("initial_offset_order_counts", {})
                .get("high_offset_larger", 0)
            )
            > 0
            for split in expected_counts
        )
    )
    causal_data = audit_causal_data_contract(
        component_id="pusht_contact_friction_icl",
        evidence_scope=(
            f"all {published_pair_count:,} Training / Development / "
            "Public Test pairs"
        ),
        continuous_environment_trajectory=bool(
            data_checks["strict_continuous_protocol"]
            and all(
                split_audits[split].get("passed") is True
                for split in expected_counts
            )
        ),
        state_installations_after_x0=max(
            int(
                split_audits[split].get(
                    "state_installations_after_x0",
                    -1,
                )
            )
            for split in expected_counts
        ),
        query_simulator_recreated=any(
            split_audits[split].get("query_simulator_recreated")
            is not False
            for split in expected_counts
        ),
        maximum_query_state_gap=max(
            float(
                split_audits[split].get(
                    "max_pair_full_state_gap",
                    np.inf,
                )
            )
            for split in expected_counts
        ),
        query_state_tolerance=QUERY_FULL_STATE_TOLERANCE,
        query_pixels_exact=all(
            int(
                split_audits[split].get(
                    "max_pair_query_pixel_difference",
                    -1,
                )
            )
            == 0
            for split in expected_counts
        ),
        query_actions_exact=all(
            float(
                split_audits[split].get(
                    "max_pair_query_action_difference",
                    np.inf,
                )
            )
            == 0.0
            for split in expected_counts
        ),
        history_effect_present=min(
            float(
                split_audits[split].get(
                    "minimum_history_gap_px_equivalent",
                    0.0,
                )
            )
            for split in expected_counts
        )
        > 0.0,
        true_future_effect_present=min(
            float(
                split_audits[split].get(
                    "minimum_future_block_position_gap_px",
                    0.0,
                )
            )
            for split in expected_counts
        )
        > 0.0,
        x0_policy="balanced_visible_start",
        x0_static_leakage_check_passed=x0_static_leakage_passed,
        solver_cache_check_required=True,
        solver_cache_check_passed=bool(
            causal_chain.get("clean_simulator_replay_is_diagnostic_only")
            is True
            and min(
                int(
                    split_audits[split].get(
                        "minimum_cache_clear_steps_before_query",
                        -1,
                    )
                )
                for split in expected_counts
            )
            >= MINIMUM_CONTACT_FREE_STEPS_BEFORE_QUERY
            and max(
                float(
                    split_audits[split].get(
                        "maximum_clean_simulator_replay_full_state_gap",
                        np.inf,
                    )
                )
                for split in expected_counts
            )
            <= QUERY_FULL_STATE_TOLERANCE
        ),
        evidence=(
            release["data"]["artifacts"]["manifest"]["path"],
            release["data"]["artifacts"]["build_report"]["path"],
            expanded_audit_specification["path"],
            release["identity"]["causal_physics_contract"]["path"],
        ),
    )

    children = [path for path in data_root.rglob("*") if path.is_file()]
    specification = release["data"]["artifact_tree"]
    observed_hash = directory_sha256(data_root) if full else None
    tree_audit = {
        "path": str(data_root),
        "exists": data_root.is_dir(),
        "expected_files": int(specification["files"]),
        "observed_files": len(children),
        "expected_bytes": int(specification["bytes"]),
        "observed_bytes": sum(path.stat().st_size for path in children),
        "expected_sha256": specification["sha256"],
        "observed_sha256": observed_hash,
        "full_hash_checked": full,
    }
    tree_audit["passed"] = bool(
        tree_audit["exists"]
        and tree_audit["observed_files"] == tree_audit["expected_files"]
        and tree_audit["observed_bytes"] == tree_audit["expected_bytes"]
        and (
            not full
            or tree_audit["observed_sha256"]
            == tree_audit["expected_sha256"]
        )
    )

    payload_audit: dict[str, Any] = {
        "decoded_pairs": 0,
        "passed": True,
    }
    if full:
        dataset = ContactFrictionICLEvalDataset(
            release=release,
            repo_root=root,
        )
        query_physics_gap = _pusht_physics_state_max_abs_gap(
            dataset.arrays.low_physics_states[:, 2],
            dataset.arrays.high_physics_states[:, 2],
        )
        payload_audit = {
            "decoded_pairs": dataset.arrays.pair_count,
            "condition_count": 2 * dataset.arrays.pair_count,
            "role": "data_integrity_only",
            "model_scoring_performed": False,
            "maximum_query_physics_gap": query_physics_gap,
            "serialized_query_physics_within_float32_bound": bool(
                query_physics_gap
                <= SERIALIZED_QUERY_FULL_STATE_TOLERANCE
            ),
            "passed": dataset.is_full_protocol,
        }
        payload_audit["passed"] = bool(
            payload_audit["passed"]
            and payload_audit[
                "serialized_query_physics_within_float32_bound"
            ]
        )

    decision_specification = release.get("reference_results", {}).get(
        "current_decision",
        {},
    )
    decision_path = resolve_contextworld_path(
        decision_specification.get("path", ""),
        repo_root=root,
    )
    development_decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {}
    )
    development_result = development_decision.get(
        "reported_endpoint",
        {},
    )
    metric_names = (
        "correct_future_rate",
        "correct_history_rate",
        "context_switch_rate",
        "worst_friction_correct_future_rate",
    )
    threshold_keys = {
        name: f"{name}_minimum" for name in metric_names
    }
    registered_thresholds = release["scoring"][
        "hidden_future_prediction"
    ]["gates"]
    decision_metrics = development_result.get("metrics", {})
    decision_gates = development_result.get("gates", {})
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
    reference_checks = {
        "status_is_failed_development": (
            development_decision.get("status") == "failed_development"
        ),
        "v3_published_manifest_is_bound": (
            development_decision.get("data_release", {}).get(
                "published_manifest_sha256"
            )
            == release["data"]["manifest_sha256"]
            == data_files["manifest"].get("observed_sha256")
        ),
        "training_manifest_is_mapped_by_portability_receipt": bool(
            development_decision.get("data_release", {}).get(
                "training_manifest_sha256"
            )
            == portability_receipt.get("metadata_sha256", {})
            .get("manifest.json", {})
            .get("before")
            and release["data"]["manifest_sha256"]
            == portability_receipt.get("metadata_sha256", {})
            .get("manifest.json", {})
            .get("after")
            and development_decision.get("data_release", {}).get(
                "portability_receipt_sha256"
            )
            == portability_specification["sha256"]
            == data_files["portability_receipt"].get("observed_sha256")
            and development_decision.get("data_release", {}).get(
                "model_visible_lance_tables_unchanged_by_metadata_migration"
            )
            is True
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
            and development_result.get("passed") is False
            and development_result.get("failed_metrics") == failed_metrics
        ),
        "public_model_scoring_is_closed": bool(
            development_decision.get("public_model_scoring_opened") is False
            and development_decision.get(
                "additional_training_seeds_run"
            )
            is False
            and development_decision.get("original_task_cem_run") is False
            and release.get("evaluation", {}).get(
                "model_evaluation_status"
            )
            == "not_opened_after_failed_development"
        ),
    }

    identity_passed = all(
        value["passed"] for value in identity_files.values()
    )
    data_passed = bool(
        all(value["passed"] for value in data_files.values())
        and all(data_checks.values())
        and tree_audit["passed"]
        and payload_audit["passed"]
        and causal_data["passed"]
    )
    reference_passed = bool(
        all(value["passed"] for value in reference_files.values())
        and all(reference_checks.values())
    )
    passed = bool(identity_passed and data_passed and reference_passed)
    return {
        "schema_version": 1,
        "release_id": release["release_id"],
        "status": "passed" if passed else "failed",
        "full_content_hash_audit": full,
        "identity": {
            "status": "ready" if identity_passed else "failed",
            "files": identity_files,
            "passed": identity_passed,
        },
        "causal_data_contract": causal_data,
        "data_release": {
            "status": "ready" if data_passed else "failed",
            "files": data_files,
            "checks": data_checks,
            "artifact_tree": tree_audit,
            "payload_audit": payload_audit,
            "causal_data_contract": causal_data,
            "pair_counts": expected_counts,
            "row_counts": row_counts,
            "passed": data_passed,
        },
        "reference_result": {
            "status": development_decision.get("status"),
            "integrity_status": (
                "passed" if reference_passed else "failed"
            ),
            "file": reference_files.get("current_decision", {}),
            "checks": reference_checks,
            "metrics": decision_metrics,
            "gates": decision_gates,
            "failed_metrics": failed_metrics,
            "development_gate_passed": development_result.get("passed"),
            "public_model_scoring_opened": development_decision.get(
                "public_model_scoring_opened"
            ),
            "positive_reference_claim": development_decision.get(
                "positive_reference_claim"
            ),
            "passed": reference_passed,
        },
        "sealed_test_included": False,
        "passed": passed,
    }


__all__ = [
    "CONTACT_FRICTION_RELEASE_ID",
    "DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG",
    "ContactFrictionEvalArrays",
    "ContactFrictionICLEvalDataset",
    "audit_contact_friction_icl_release",
    "directory_sha256",
    "file_sha256",
    "load_contact_friction_icl_release",
]
