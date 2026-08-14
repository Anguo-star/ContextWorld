"""Training-safe Cube History-3 action profiles for a prospective v3.

This module leaves the frozen v2 construction untouched.  It reuses v2's
MuJoCo transition implementation while replacing its single out-of-support
action sequence with four balanced anchors and small deterministic
perturbations.  Perturbations live in the nullspace of the three causal
constraints required by the reversible construction:

* the five-step probe sums to zero;
* its final z action is exactly zero, so no mode-dependent applied force is
  retained at the shared query; and
* its discrete displacement moment is one, preserving v2's physical signal.

The public action-profile identifier is solely a SHA256 of the actual float32
``[4, 5, 5]`` action-block bytes.  Split and anchor labels are separate audit
fields and therefore cannot conceal duplicate action content.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
from typing import Any

import numpy as np

from contextworld.evaluation.cube_grasp_rule_h3 import (
    CAPABILITY_NAME,
    GRASP_MODES,
    QUERY_STATE_TOLERANCE,
    CubeGraspRuleCandidate,
    CubeGraspRuleSimulator,
    array_sha256,
    validate_cube_grasp_rule_pair,
)


V3_PROTOCOL = "cube_gripper_carry_rule_history3_development_v3"
V3_ACTION_ANCHORS = (
    "endpoint4",
    "plateau",
    "ramp4",
    "front_hold",
)
V3_ANCHOR_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    "endpoint4": (1.0 / 3.0, 0.0, 0.0, -1.0 / 3.0, 0.0),
    "plateau": (0.25, 0.25, -0.25, -0.25, 0.0),
    "ramp4": (0.3, 0.1, -0.1, -0.3, 0.0),
    "front_hold": (0.2, 0.2, 0.0, -0.4, 0.0),
}
V3_ANCHOR_GRIPPER_ACTIONS = {
    "endpoint4": 0.4,
    "plateau": 0.4,
    "ramp4": 0.5,
    "front_hold": 0.5,
}
V3_PROFILE_SPLIT_SEEDS = {
    "train": 2026081101,
    "loader_validation": 2026081102,
}

# Both suggested vectors have zero sum, zero final entry, and zero moment.
# Unit normalization makes the coefficient bound comparable across bases.
V3_PERTURBATION_BASES = (
    (1.0, 0.0, -3.0, 2.0, 0.0),
    (0.0, 1.0, -2.0, 1.0, 0.0),
)
V3_PERTURBATION_COEFFICIENT_LIMIT = 0.02
V3_CONSTRAINT_GRID_DENOMINATOR = 2**20
V3_CONSTRAINT_TOLERANCE = 1e-7
_MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])


@dataclass(frozen=True)
class CubeGraspRuleV3Profile:
    """One deterministic, content-addressed v3 action profile."""

    action_profile_id: str
    action_anchor_id: str
    split: str
    catalog_index: int
    split_seed: int
    perturbation_coefficients: tuple[float, float]
    probe_profile: tuple[float, float, float, float, float]
    gripper_action: float


@dataclass(frozen=True)
class CubeGraspRuleV3Candidate(CubeGraspRuleCandidate):
    """A v2-compatible physical start plus its deterministic v3 actions."""

    action_profile: CubeGraspRuleV3Profile

    def __post_init__(self) -> None:
        if self.split != self.action_profile.split:
            raise ValueError("Candidate and action profile splits differ")
        if int(self.catalog_index) != int(self.action_profile.catalog_index):
            raise ValueError("Candidate and action profile catalog indices differ")
        audit = validate_v3_action_profile(self.action_profile)
        if not audit["passed"]:
            failed = [name for name, passed in audit["checks"].items() if not passed]
            raise ValueError(
                "Candidate has an invalid v3 action profile: "
                + ", ".join(failed)
            )


def _normalized_perturbation_bases() -> np.ndarray:
    bases = np.asarray(V3_PERTURBATION_BASES, dtype=np.float64)
    return bases / np.linalg.norm(bases, axis=1, keepdims=True)


def _validate_split_and_index(split: str, catalog_index: int) -> tuple[str, int]:
    if split not in V3_PROFILE_SPLIT_SEEDS:
        raise ValueError(
            f"Unknown v3 split {split!r}; expected "
            f"{tuple(V3_PROFILE_SPLIT_SEEDS)}"
        )
    if isinstance(catalog_index, bool) or not isinstance(
        catalog_index, (int, np.integer)
    ):
        raise TypeError("catalog_index must be an integer")
    index = int(catalog_index)
    if index < 0:
        raise ValueError("catalog_index must be non-negative")
    return split, index


def _quantize_constrained_profile(value: np.ndarray) -> np.ndarray:
    """Project a near-feasible profile onto an exact dyadic constraint grid.

    Choosing the first two coordinates fixes the remaining two through the
    integer versions of ``sum(p)=0`` and ``moment(p)=1``.  The grid denominator
    is a power of two and all numerators are well below 2**24, so conversion to
    float32 is exact.  This avoids tiny recovery drift from independently
    rounded floating-point coordinates.
    """

    denominator = int(V3_CONSTRAINT_GRID_DENOMINATOR)
    first = int(np.rint(float(value[0]) * denominator))
    second = int(np.rint(float(value[1]) * denominator))
    third = denominator - 3 * first - 2 * second
    fourth = -first - second - third
    numerators = np.asarray([first, second, third, fourth, 0], dtype=np.int64)
    if np.max(np.abs(numerators)) >= 2**24:
        raise ValueError("Quantized v3 profile exceeds exact float32 integer range")
    return (numerators.astype(np.float64) / denominator).astype(np.float32)


def _derive_profile_components(
    *, split: str, catalog_index: int
) -> tuple[str, int, tuple[float, float], np.ndarray, np.float32]:
    split, index = _validate_split_and_index(split, catalog_index)
    anchor_id = V3_ACTION_ANCHORS[index % len(V3_ACTION_ANCHORS)]
    split_seed = int(V3_PROFILE_SPLIT_SEEDS[split])
    generator = np.random.default_rng(
        np.random.SeedSequence([split_seed, index, 0xC0BE2026])
    )
    coefficients = generator.uniform(
        -V3_PERTURBATION_COEFFICIENT_LIMIT,
        V3_PERTURBATION_COEFFICIENT_LIMIT,
        size=2,
    ).astype(np.float64)
    anchor = np.asarray(V3_ANCHOR_PROFILES[anchor_id], dtype=np.float64)
    requested = anchor + coefficients @ _normalized_perturbation_bases()
    profile = _quantize_constrained_profile(requested)
    gripper = np.float32(V3_ANCHOR_GRIPPER_ACTIONS[anchor_id])
    return (
        anchor_id,
        split_seed,
        (float(coefficients[0]), float(coefficients[1])),
        profile,
        gripper,
    )


def _action_blocks_from_values(
    probe_profile: np.ndarray, gripper_action: np.float32
) -> np.ndarray:
    probe = np.asarray(probe_profile, dtype=np.float32)
    if probe.shape != (5,):
        raise ValueError(f"probe_profile must have shape (5,), got {probe.shape}")
    raw = np.zeros((15, 5), dtype=np.float32)
    raw[:, 2] = np.concatenate((probe, -probe, probe))
    raw[:, 4] = np.float32(gripper_action)
    blocks = np.zeros((4, 5, 5), dtype=np.float32)
    blocks[:3] = raw.reshape(3, 5, 5)
    return blocks


def action_profile_content_sha256(blocks: np.ndarray) -> str:
    """Hash only the actual contiguous float32 action-block bytes."""

    value = np.asarray(blocks)
    if value.shape != (4, 5, 5) or value.dtype != np.float32:
        raise ValueError(
            "Action profile identity requires float32 blocks with shape (4,5,5)"
        )
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_v3_action_profile(
    *, split: str, catalog_index: int
) -> CubeGraspRuleV3Profile:
    """Create one deterministic split-specific constrained action profile."""

    anchor_id, split_seed, coefficients, probe, gripper = (
        _derive_profile_components(split=split, catalog_index=catalog_index)
    )
    blocks = _action_blocks_from_values(probe, gripper)
    profile = CubeGraspRuleV3Profile(
        action_profile_id=action_profile_content_sha256(blocks),
        action_anchor_id=anchor_id,
        split=split,
        catalog_index=int(catalog_index),
        split_seed=split_seed,
        perturbation_coefficients=coefficients,
        probe_profile=tuple(float(value) for value in probe),
        gripper_action=float(gripper),
    )
    audit = validate_v3_action_profile(profile)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("Generated invalid v3 profile: " + ", ".join(failed))
    return profile


def action_blocks(profile: CubeGraspRuleV3Profile) -> np.ndarray:
    """Return the exact float32 action blocks bound to ``profile``."""

    audit = validate_v3_action_profile(profile)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise ValueError("Invalid v3 action profile: " + ", ".join(failed))
    return _action_blocks_from_values(
        np.asarray(profile.probe_profile, dtype=np.float32),
        np.float32(profile.gripper_action),
    )


def validate_v3_action_profile(
    profile: CubeGraspRuleV3Profile,
) -> dict[str, Any]:
    """Audit deterministic derivation, exact recovery, and content identity."""

    if not isinstance(profile, CubeGraspRuleV3Profile):
        raise TypeError("profile must be CubeGraspRuleV3Profile")
    probe = np.asarray(profile.probe_profile, dtype=np.float32)
    coefficients = np.asarray(
        profile.perturbation_coefficients, dtype=np.float64
    )
    shape_valid = probe.shape == (5,)
    finite = bool(
        shape_valid
        and np.isfinite(probe).all()
        and coefficients.shape == (2,)
        and np.isfinite(coefficients).all()
        and np.isfinite(profile.gripper_action)
    )
    profile_sum = float(np.sum(probe.astype(np.float64))) if shape_valid else np.inf
    final_value = float(probe[-1]) if shape_valid else np.inf
    moment = (
        float(np.dot(_MOMENT_WEIGHTS, probe.astype(np.float64)))
        if shape_valid
        else np.inf
    )
    try:
        expected = _derive_profile_components(
            split=profile.split,
            catalog_index=profile.catalog_index,
        )
        (
            expected_anchor,
            expected_seed,
            expected_coefficients,
            expected_probe,
            expected_gripper,
        ) = expected
        deterministic_fields = True
    except (TypeError, ValueError):
        expected_anchor = ""
        expected_seed = -1
        expected_coefficients = (np.nan, np.nan)
        expected_probe = np.full(5, np.nan, dtype=np.float32)
        expected_gripper = np.float32(np.nan)
        deterministic_fields = False

    blocks = (
        _action_blocks_from_values(probe, np.float32(profile.gripper_action))
        if finite
        else np.zeros((4, 5, 5), dtype=np.float32)
    )
    observed_hash = action_profile_content_sha256(blocks)
    checks = {
        "finite_float32_profile": finite,
        "probe_sum_exact_zero": profile_sum == 0.0,
        "probe_sum_within_tolerance": abs(profile_sum) <= V3_CONSTRAINT_TOLERANCE,
        "probe_final_z_exact_zero": final_value == 0.0,
        "probe_moment_exact_one": moment == 1.0,
        "probe_moment_within_tolerance": (
            abs(moment - 1.0) <= V3_CONSTRAINT_TOLERANCE
        ),
        "coefficient_bound_respected": bool(
            coefficients.shape == (2,)
            and np.all(
                np.abs(coefficients)
                <= V3_PERTURBATION_COEFFICIENT_LIMIT
            )
        ),
        "anchor_balanced_by_catalog_index": bool(
            deterministic_fields
            and profile.action_anchor_id == expected_anchor
        ),
        "split_seed_matches": bool(
            deterministic_fields and int(profile.split_seed) == expected_seed
        ),
        "perturbation_coefficients_match": bool(
            deterministic_fields
            and np.array_equal(
                coefficients,
                np.asarray(expected_coefficients, dtype=np.float64),
            )
        ),
        "quantized_profile_matches": bool(
            deterministic_fields and np.array_equal(probe, expected_probe)
        ),
        "gripper_matches_anchor": bool(
            deterministic_fields
            and np.float32(profile.gripper_action) == expected_gripper
        ),
        "action_profile_id_is_content_hash": (
            profile.action_profile_id == observed_hash
        ),
        "action_values_within_environment_bounds": bool(
            finite and np.max(np.abs(blocks)) <= 1.0
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "action_anchor_id": profile.action_anchor_id,
        "action_profile_id": profile.action_profile_id,
        "split": profile.split,
        "catalog_index": int(profile.catalog_index),
        "split_seed": int(profile.split_seed),
        "probe_sum": profile_sum,
        "probe_final_z": final_value,
        "probe_moment": moment,
        "constraint_tolerance": V3_CONSTRAINT_TOLERANCE,
        "content_sha256": observed_hash,
    }


def make_v3_candidate(
    candidate: CubeGraspRuleCandidate,
) -> CubeGraspRuleV3Candidate:
    """Attach the split/index-derived profile to a v2 physical candidate."""

    if not isinstance(candidate, CubeGraspRuleCandidate):
        raise TypeError("candidate must be CubeGraspRuleCandidate")
    if isinstance(candidate, CubeGraspRuleV3Candidate):
        return candidate
    values = {
        field.name: getattr(candidate, field.name)
        for field in fields(CubeGraspRuleCandidate)
    }
    return CubeGraspRuleV3Candidate(
        **values,
        action_profile=make_v3_action_profile(
            split=candidate.split,
            catalog_index=candidate.catalog_index,
        ),
    )


def _maximum_absolute_gap(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape:
        return float("inf")
    if not first.size:
        return 0.0
    delta = np.abs(
        first.astype(np.float64, copy=False)
        - second.astype(np.float64, copy=False)
    )
    return float(np.max(delta)) if np.isfinite(delta).all() else float("inf")


def _fresh_replay_mode_audit(
    *,
    mode: str,
    continuous: dict[str, Any],
    replay: dict[str, Any],
    expected_profile_id: str,
    expected_anchor_id: str,
) -> dict[str, Any]:
    """Compare every saved model-visible and physical replay field."""

    continuous_physical = np.asarray(continuous["physical_state"])
    replay_physical = np.asarray(replay["physical_state"])
    continuous_simulator = np.asarray(continuous["simulator_state"])
    replay_simulator = np.asarray(replay["simulator_state"])
    continuous_pixels = np.asarray(continuous["pixels"])
    replay_pixels = np.asarray(replay["pixels"])
    continuous_actions = np.asarray(continuous["action_blocks"])
    replay_actions = np.asarray(replay["action_blocks"])

    replay_profile_id = action_profile_content_sha256(replay_actions)
    continuous_profile_id = str(
        continuous.get(
            "action_profile_id",
            action_profile_content_sha256(continuous_actions),
        )
    )
    same_pixel_shape = continuous_pixels.shape == replay_pixels.shape
    if same_pixel_shape:
        pixel_delta = continuous_pixels != replay_pixels
        changed_rgb_values = int(np.count_nonzero(pixel_delta))
        changed_pixels = int(np.count_nonzero(np.any(pixel_delta, axis=-1)))
    else:
        changed_rgb_values = max(
            int(continuous_pixels.size), int(replay_pixels.size)
        )
        changed_pixels = changed_rgb_values

    hashes = {
        "continuous": {
            "physical_state": array_sha256(continuous_physical),
            "simulator_state": array_sha256(continuous_simulator),
            "pixels": array_sha256(continuous_pixels),
            "action_blocks": array_sha256(continuous_actions),
        },
        "fresh_replay": {
            "physical_state": array_sha256(replay_physical),
            "simulator_state": array_sha256(replay_simulator),
            "pixels": array_sha256(replay_pixels),
            "action_blocks": array_sha256(replay_actions),
        },
    }
    checks = {
        "physical_state_shape_and_dtype_equal": (
            continuous_physical.shape == replay_physical.shape
            and continuous_physical.dtype == replay_physical.dtype
        ),
        "physical_state_bitwise_equal": np.array_equal(
            continuous_physical, replay_physical
        ),
        "physical_state_hash_equal": (
            hashes["continuous"]["physical_state"]
            == hashes["fresh_replay"]["physical_state"]
        ),
        "simulator_state_shape_and_dtype_equal": (
            continuous_simulator.shape == replay_simulator.shape
            and continuous_simulator.dtype == replay_simulator.dtype
        ),
        "simulator_state_bitwise_equal": np.array_equal(
            continuous_simulator, replay_simulator
        ),
        "simulator_state_hash_equal": (
            hashes["continuous"]["simulator_state"]
            == hashes["fresh_replay"]["simulator_state"]
        ),
        "pixels_shape_and_dtype_equal": (
            same_pixel_shape and continuous_pixels.dtype == replay_pixels.dtype
        ),
        "pixels_bitwise_equal": np.array_equal(
            continuous_pixels, replay_pixels
        ),
        "pixels_hash_equal": (
            hashes["continuous"]["pixels"]
            == hashes["fresh_replay"]["pixels"]
        ),
        "action_blocks_shape_and_dtype_equal": (
            continuous_actions.shape == replay_actions.shape
            and continuous_actions.dtype == replay_actions.dtype
        ),
        "action_blocks_bitwise_equal": np.array_equal(
            continuous_actions, replay_actions
        ),
        "action_blocks_hash_equal": (
            hashes["continuous"]["action_blocks"]
            == hashes["fresh_replay"]["action_blocks"]
        ),
        "action_profile_id_equal": (
            continuous_profile_id
            == replay_profile_id
            == expected_profile_id
        ),
        "action_anchor_id_equal": (
            str(continuous.get("action_anchor_id", expected_anchor_id))
            == expected_anchor_id
        ),
        "hidden_value_equal": (
            float(continuous["hidden_value"])
            == float(replay["hidden_value"])
        ),
        "prequery_residual_equal": (
            float(continuous["prequery_residual"])
            == float(replay["prequery_residual"])
        ),
        "state_installation_count_equal": (
            int(continuous["state_installations_after_x0"])
            == int(replay["state_installations_after_x0"])
            == 0
        ),
        "external_force_update_count_equal": (
            int(continuous["external_force_updates_after_x0"])
            == int(replay["external_force_updates_after_x0"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "mode": mode,
        "checks": checks,
        "maximum_physical_state_gap": _maximum_absolute_gap(
            continuous_physical, replay_physical
        ),
        "maximum_simulator_state_gap": _maximum_absolute_gap(
            continuous_simulator, replay_simulator
        ),
        "changed_rgb_values": changed_rgb_values,
        "changed_pixels": changed_pixels,
        "continuous_action_profile_id": continuous_profile_id,
        "fresh_replay_action_profile_id": replay_profile_id,
        "hashes": hashes,
    }


class CubeGraspRuleV3Simulator(CubeGraspRuleSimulator):
    """Run v3 profiles with unchanged v2 physics and fresh replay audits.

    A caller building many pairs may pass a second, reusable simulator through
    ``replay_simulator``.  It must be a distinct instance with a distinct
    environment.  This is safe across sequential pairs because inherited
    ``_run_mode`` resets and reconstructs the candidate before every mode.
    Do not share either simulator across worker processes or concurrent calls.
    """

    def __init__(self, *, resolution: int = 224) -> None:
        self.resolution = int(resolution)
        super().__init__(resolution=self.resolution)

    def audit_fresh_simulator_replay(
        self,
        candidate: CubeGraspRuleV3Candidate,
        continuous_payload: dict[str, dict[str, Any]],
        *,
        replay_simulator: CubeGraspRuleV3Simulator | None = None,
    ) -> dict[str, Any]:
        """Replay both modes through a distinct simulator via ``_run_mode``."""

        if not isinstance(candidate, CubeGraspRuleV3Candidate):
            raise TypeError(
                "Fresh replay requires CubeGraspRuleV3Candidate"
            )
        if set(continuous_payload) != set(GRASP_MODES):
            raise ValueError(
                "continuous_payload must contain exactly both grasp modes"
            )
        provided_reusable_instance = replay_simulator is not None
        if replay_simulator is None:
            replay_simulator = CubeGraspRuleV3Simulator(
                resolution=self.resolution
            )
            owns_replay_simulator = True
        else:
            owns_replay_simulator = False
            if not isinstance(replay_simulator, CubeGraspRuleV3Simulator):
                raise TypeError(
                    "replay_simulator must be CubeGraspRuleV3Simulator"
                )
        if replay_simulator is self:
            raise ValueError("Fresh replay simulator must be a distinct instance")
        if (
            getattr(self, "env", None) is not None
            and getattr(self, "env", None)
            is getattr(replay_simulator, "env", None)
        ):
            raise ValueError("Fresh replay simulator must not share the primary env")
        if int(replay_simulator.resolution) != int(self.resolution):
            raise ValueError("Fresh replay simulator resolution differs")

        profile = candidate.action_profile
        mode_audits: dict[str, dict[str, Any]] = {}
        try:
            for mode in GRASP_MODES:
                continuous = continuous_payload[mode]
                blocks = np.asarray(
                    continuous["action_blocks"], dtype=np.float32
                )
                replay = replay_simulator._run_mode(
                    candidate,
                    mode=mode,
                    blocks=blocks,
                )
                replay["action_anchor_id"] = profile.action_anchor_id
                replay["action_profile_id"] = (
                    action_profile_content_sha256(replay["action_blocks"])
                )
                mode_audits[mode] = _fresh_replay_mode_audit(
                    mode=mode,
                    continuous=continuous,
                    replay=replay,
                    expected_profile_id=profile.action_profile_id,
                    expected_anchor_id=profile.action_anchor_id,
                )
        finally:
            if owns_replay_simulator:
                replay_simulator.close()

        return {
            "passed": all(value["passed"] for value in mode_audits.values()),
            "method": (
                "same_candidate_mode_and_float32_blocks_via_distinct_"
                "fresh_simulator_run_mode"
            ),
            "independent_simulator_instance": True,
            "provided_reusable_instance": provided_reusable_instance,
            "fresh_candidate_reset_before_each_mode": True,
            "replay_payload_retained_as_training_target": False,
            "maximum_physical_state_gap": max(
                value["maximum_physical_state_gap"]
                for value in mode_audits.values()
            ),
            "maximum_simulator_state_gap": max(
                value["maximum_simulator_state_gap"]
                for value in mode_audits.values()
            ),
            "total_changed_rgb_values": sum(
                value["changed_rgb_values"] for value in mode_audits.values()
            ),
            "total_changed_pixels": sum(
                value["changed_pixels"] for value in mode_audits.values()
            ),
            "modes": mode_audits,
        }

    def build_pair(
        self,
        candidate: CubeGraspRuleV3Candidate,
        *,
        replay_simulator: CubeGraspRuleV3Simulator | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(candidate, CubeGraspRuleV3Candidate):
            raise TypeError(
                "CubeGraspRuleV3Simulator requires CubeGraspRuleV3Candidate"
            )
        profile = candidate.action_profile
        profile_audit = validate_v3_action_profile(profile)
        blocks = action_blocks(profile)
        payload = {
            mode: self._run_mode(candidate, mode=mode, blocks=blocks)
            for mode in GRASP_MODES
        }
        for episode in payload.values():
            episode["action_anchor_id"] = profile.action_anchor_id
            episode["action_profile_id"] = profile.action_profile_id

        audit = validate_cube_grasp_rule_pair(
            payload["cannot_hold"], payload["can_hold"]
        )
        low_profile_id = action_profile_content_sha256(
            payload["cannot_hold"]["action_blocks"]
        )
        high_profile_id = action_profile_content_sha256(
            payload["can_hold"]["action_blocks"]
        )
        audit["checks"].update(
            {
                "v3_profile_constraints_passed": profile_audit["passed"],
                "v3_profile_matches_candidate_split": (
                    profile.split == candidate.split
                ),
                "v3_profile_matches_candidate_catalog_index": (
                    int(profile.catalog_index) == int(candidate.catalog_index)
                ),
                "v3_profile_id_matches_both_action_blocks": (
                    low_profile_id
                    == high_profile_id
                    == profile.action_profile_id
                ),
            }
        )
        audit["v3"] = {
            "protocol": V3_PROTOCOL,
            "action_anchor_id": profile.action_anchor_id,
            "action_profile_id": profile.action_profile_id,
            "profile_constraints": profile_audit,
        }
        audit["passed"] = all(audit["checks"].values())
        if not audit["passed"]:
            return None

        replay_audit = self.audit_fresh_simulator_replay(
            candidate,
            payload,
            replay_simulator=replay_simulator,
        )
        audit["checks"]["fresh_simulator_deterministic_replay_passed"] = (
            replay_audit["passed"]
        )
        audit["v3"]["fresh_simulator_replay"] = replay_audit
        audit["passed"] = all(audit["checks"].values())
        if not audit["passed"]:
            return None
        return {
            "candidate": asdict(candidate),
            "action_profile": {
                **asdict(profile),
                "constraints": profile_audit,
            },
            "audit": audit,
            **payload,
        }


__all__ = [
    "CAPABILITY_NAME",
    "GRASP_MODES",
    "QUERY_STATE_TOLERANCE",
    "CubeGraspRuleCandidate",
    "CubeGraspRuleV3Candidate",
    "CubeGraspRuleV3Profile",
    "CubeGraspRuleV3Simulator",
    "V3_ACTION_ANCHORS",
    "V3_ANCHOR_GRIPPER_ACTIONS",
    "V3_ANCHOR_PROFILES",
    "V3_CONSTRAINT_GRID_DENOMINATOR",
    "V3_CONSTRAINT_TOLERANCE",
    "V3_PERTURBATION_BASES",
    "V3_PERTURBATION_COEFFICIENT_LIMIT",
    "V3_PROFILE_SPLIT_SEEDS",
    "V3_PROTOCOL",
    "action_blocks",
    "action_profile_content_sha256",
    "make_v3_action_profile",
    "make_v3_candidate",
    "validate_v3_action_profile",
]
