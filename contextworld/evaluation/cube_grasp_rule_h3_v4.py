"""Cube History-3 v4 with a stronger, still reversible carry signal.

The sole scientific change from v3 is the ``can_hold`` vertical-force
coupling: 0.40 N rather than 0.30 N per unit normalized z command.  The
capability, History-3 observation contract, four action anchors, gripper
values, perturbation bound, exact ``[p, -p, p]`` recovery constraints, and
fresh-simulator replay gates remain unchanged.

V4 owns its profile and candidate types, split seeds, protocol identifier,
and audit namespace.  It subclasses the proven v3 simulator machinery but
overrides the transition-force method directly; no global v3 constant is
mutated or monkeypatched.
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
    validate_cube_grasp_rule_pair,
)
from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    CubeGraspRuleV3Simulator,
    V3_ACTION_ANCHORS,
    V3_ANCHOR_GRIPPER_ACTIONS,
    V3_ANCHOR_PROFILES,
    V3_CONSTRAINT_GRID_DENOMINATOR,
    V3_CONSTRAINT_TOLERANCE,
    V3_PERTURBATION_BASES,
    V3_PERTURBATION_COEFFICIENT_LIMIT,
    _fresh_replay_mode_audit,
)


V4_PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
V4_VERTICAL_FORCE_COUPLING_N = 0.40
# Formal v4 profiles live in a frozen catalog-index namespace that is
# disjoint from the index 0/1 real-MuJoCo preformal evidence.  The offset is
# four-aligned so moving into the formal namespace does not change anchor
# assignment (catalog_index modulo the four frozen anchors).
V4_FORMAL_CATALOG_INDEX_OFFSET = 1_000_000
if (
    V4_FORMAL_CATALOG_INDEX_OFFSET <= 0
    or V4_FORMAL_CATALOG_INDEX_OFFSET % 4 != 0
):
    raise RuntimeError(
        "V4_FORMAL_CATALOG_INDEX_OFFSET must be positive and four-aligned"
    )
# The original v4 formal attempt consumed the 1,000,000 namespace before its
# Lance commit failed on the artifact NFS mount's atomic rename.  The v4r1
# recovery keeps the scientific v4 protocol unchanged but deterministically
# resamples content in
# a new, four-aligned namespace.  Keeping both constants makes the failed
# attempt reproducible while preventing its action identities from being
# silently reused by the authorized recovery.
V4R1_FORMAL_CATALOG_INDEX_OFFSET = 2_000_000
if (
    V4R1_FORMAL_CATALOG_INDEX_OFFSET <= V4_FORMAL_CATALOG_INDEX_OFFSET
    or V4R1_FORMAL_CATALOG_INDEX_OFFSET % 4 != 0
):
    raise RuntimeError(
        "V4R1_FORMAL_CATALOG_INDEX_OFFSET must follow v4 and be four-aligned"
    )
# Keep the module-local public spelling parallel to the original physics
# module while retaining an explicit versioned spelling for receipts/tests.
VERTICAL_FORCE_COUPLING_N = V4_VERTICAL_FORCE_COUPLING_N

V4_ACTION_ANCHORS = tuple(V3_ACTION_ANCHORS)
V4_ANCHOR_PROFILES = dict(V3_ANCHOR_PROFILES)
V4_ANCHOR_GRIPPER_ACTIONS = dict(V3_ANCHOR_GRIPPER_ACTIONS)
V4_PROFILE_SPLIT_SEEDS = {
    "train": 2026081201,
    "loader_validation": 2026081202,
}
V4_PERTURBATION_BASES = tuple(V3_PERTURBATION_BASES)
V4_PERTURBATION_COEFFICIENT_LIMIT = V3_PERTURBATION_COEFFICIENT_LIMIT
V4_CONSTRAINT_GRID_DENOMINATOR = V3_CONSTRAINT_GRID_DENOMINATOR
V4_CONSTRAINT_TOLERANCE = V3_CONSTRAINT_TOLERANCE
_MOMENT_WEIGHTS = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0])


@dataclass(frozen=True)
class CubeGraspRuleV4Profile:
    """One deterministic, content-addressed v4 action profile."""

    action_profile_id: str
    action_anchor_id: str
    split: str
    catalog_index: int
    split_seed: int
    perturbation_coefficients: tuple[float, float]
    probe_profile: tuple[float, float, float, float, float]
    gripper_action: float


@dataclass(frozen=True)
class CubeGraspRuleV4Candidate(CubeGraspRuleCandidate):
    """A physical Cube start bound to a deterministic v4 action profile."""

    action_profile: CubeGraspRuleV4Profile

    def __post_init__(self) -> None:
        if self.split != self.action_profile.split:
            raise ValueError("Candidate and action profile splits differ")
        if int(self.catalog_index) != int(self.action_profile.catalog_index):
            raise ValueError("Candidate and action profile catalog indices differ")
        audit = validate_v4_action_profile(self.action_profile)
        if not audit["passed"]:
            failed = [name for name, passed in audit["checks"].items() if not passed]
            raise ValueError(
                "Candidate has an invalid v4 action profile: "
                + ", ".join(failed)
            )


def _normalized_perturbation_bases() -> np.ndarray:
    bases = np.asarray(V4_PERTURBATION_BASES, dtype=np.float64)
    return bases / np.linalg.norm(bases, axis=1, keepdims=True)


def _validate_split_and_index(split: str, catalog_index: int) -> tuple[str, int]:
    if split not in V4_PROFILE_SPLIT_SEEDS:
        raise ValueError(
            f"Unknown v4 split {split!r}; expected "
            f"{tuple(V4_PROFILE_SPLIT_SEEDS)}"
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
    """Project a profile onto the exact dyadic recovery grid used by v3."""

    denominator = int(V4_CONSTRAINT_GRID_DENOMINATOR)
    first = int(np.rint(float(value[0]) * denominator))
    second = int(np.rint(float(value[1]) * denominator))
    third = denominator - 3 * first - 2 * second
    fourth = -first - second - third
    numerators = np.asarray([first, second, third, fourth, 0], dtype=np.int64)
    if np.max(np.abs(numerators)) >= 2**24:
        raise ValueError("Quantized v4 profile exceeds exact float32 integer range")
    return (numerators.astype(np.float64) / denominator).astype(np.float32)


def _derive_profile_components(
    *, split: str, catalog_index: int
) -> tuple[str, int, tuple[float, float], np.ndarray, np.float32]:
    split, index = _validate_split_and_index(split, catalog_index)
    anchor_id = V4_ACTION_ANCHORS[index % len(V4_ACTION_ANCHORS)]
    split_seed = int(V4_PROFILE_SPLIT_SEEDS[split])
    generator = np.random.default_rng(
        np.random.SeedSequence([split_seed, index, 0xC0BE2026])
    )
    coefficients = generator.uniform(
        -V4_PERTURBATION_COEFFICIENT_LIMIT,
        V4_PERTURBATION_COEFFICIENT_LIMIT,
        size=2,
    ).astype(np.float64)
    anchor = np.asarray(V4_ANCHOR_PROFILES[anchor_id], dtype=np.float64)
    requested = anchor + coefficients @ _normalized_perturbation_bases()
    profile = _quantize_constrained_profile(requested)
    gripper = np.float32(V4_ANCHOR_GRIPPER_ACTIONS[anchor_id])
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


def make_v4_action_profile(
    *, split: str, catalog_index: int
) -> CubeGraspRuleV4Profile:
    """Create one deterministic, split-specific constrained v4 profile."""

    anchor_id, split_seed, coefficients, probe, gripper = (
        _derive_profile_components(split=split, catalog_index=catalog_index)
    )
    blocks = _action_blocks_from_values(probe, gripper)
    profile = CubeGraspRuleV4Profile(
        action_profile_id=action_profile_content_sha256(blocks),
        action_anchor_id=anchor_id,
        split=split,
        catalog_index=int(catalog_index),
        split_seed=split_seed,
        perturbation_coefficients=coefficients,
        probe_profile=tuple(float(value) for value in probe),
        gripper_action=float(gripper),
    )
    audit = validate_v4_action_profile(profile)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("Generated invalid v4 profile: " + ", ".join(failed))
    return profile


def action_blocks(profile: CubeGraspRuleV4Profile) -> np.ndarray:
    """Return the exact float32 ``[p, -p, p]`` blocks bound to a profile."""

    audit = validate_v4_action_profile(profile)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise ValueError("Invalid v4 action profile: " + ", ".join(failed))
    return _action_blocks_from_values(
        np.asarray(profile.probe_profile, dtype=np.float32),
        np.float32(profile.gripper_action),
    )


def validate_v4_action_profile(
    profile: CubeGraspRuleV4Profile,
) -> dict[str, Any]:
    """Audit v4 derivation, exact recovery constraints, and content identity."""

    if not isinstance(profile, CubeGraspRuleV4Profile):
        raise TypeError("profile must be CubeGraspRuleV4Profile")
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
        "probe_sum_within_tolerance": abs(profile_sum) <= V4_CONSTRAINT_TOLERANCE,
        "probe_final_z_exact_zero": final_value == 0.0,
        "probe_moment_exact_one": moment == 1.0,
        "probe_moment_within_tolerance": (
            abs(moment - 1.0) <= V4_CONSTRAINT_TOLERANCE
        ),
        "coefficient_bound_respected": bool(
            coefficients.shape == (2,)
            and np.all(
                np.abs(coefficients) <= V4_PERTURBATION_COEFFICIENT_LIMIT
            )
        ),
        "anchor_balanced_by_catalog_index": bool(
            deterministic_fields and profile.action_anchor_id == expected_anchor
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
        "constraint_tolerance": V4_CONSTRAINT_TOLERANCE,
        "content_sha256": observed_hash,
    }


def make_v4_candidate(
    candidate: CubeGraspRuleCandidate,
) -> CubeGraspRuleV4Candidate:
    """Attach the v4 split/index-derived profile to a physical candidate."""

    if not isinstance(candidate, CubeGraspRuleCandidate):
        raise TypeError("candidate must be CubeGraspRuleCandidate")
    if isinstance(candidate, CubeGraspRuleV4Candidate):
        return candidate
    values = {
        field.name: getattr(candidate, field.name)
        for field in fields(CubeGraspRuleCandidate)
    }
    return CubeGraspRuleV4Candidate(
        **values,
        action_profile=make_v4_action_profile(
            split=candidate.split,
            catalog_index=candidate.catalog_index,
        ),
    )


class CubeGraspRuleV4Simulator(CubeGraspRuleV3Simulator):
    """Run isolated v4 profiles with a 0.40 N ``can_hold`` coupling."""

    def _apply_transition_force(self, *, mode: str, action_z: float) -> None:
        """Apply gravity compensation plus v4's mode-dependent 0.40 N force."""

        self.base._data.qfrc_applied[:] = 0.0
        gravity_z = float(self.base._model.opt.gravity[2])
        mass = float(self.base._model.body_mass[self.object_body_id])
        object_z_dof = self.object_dof_address + 2
        self.base._data.qfrc_applied[object_z_dof] = -mass * gravity_z
        if mode == "can_hold":
            self.base._data.qfrc_applied[object_z_dof] += (
                V4_VERTICAL_FORCE_COUPLING_N * float(action_z)
            )
        elif mode != "cannot_hold":
            raise ValueError(f"Unknown grasp mode: {mode}")

    def audit_fresh_simulator_replay(
        self,
        candidate: CubeGraspRuleV4Candidate,
        continuous_payload: dict[str, dict[str, Any]],
        *,
        replay_simulator: CubeGraspRuleV4Simulator | None = None,
    ) -> dict[str, Any]:
        """Replay both v4 modes through a distinct v4 simulator instance."""

        if not isinstance(candidate, CubeGraspRuleV4Candidate):
            raise TypeError("Fresh replay requires CubeGraspRuleV4Candidate")
        if set(continuous_payload) != set(GRASP_MODES):
            raise ValueError(
                "continuous_payload must contain exactly both grasp modes"
            )
        provided_reusable_instance = replay_simulator is not None
        if replay_simulator is None:
            replay_simulator = CubeGraspRuleV4Simulator(
                resolution=self.resolution
            )
            owns_replay_simulator = True
        else:
            owns_replay_simulator = False
            if not isinstance(replay_simulator, CubeGraspRuleV4Simulator):
                raise TypeError(
                    "replay_simulator must be CubeGraspRuleV4Simulator"
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
                replay["action_profile_id"] = action_profile_content_sha256(
                    replay["action_blocks"]
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
        candidate: CubeGraspRuleV4Candidate,
        *,
        replay_simulator: CubeGraspRuleV4Simulator | None = None,
    ) -> dict[str, Any] | None:
        """Build and fully audit one v4 counterfactual pair."""

        if not isinstance(candidate, CubeGraspRuleV4Candidate):
            raise TypeError(
                "CubeGraspRuleV4Simulator requires CubeGraspRuleV4Candidate"
            )
        profile = candidate.action_profile
        profile_audit = validate_v4_action_profile(profile)
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
                "v4_profile_constraints_passed": profile_audit["passed"],
                "v4_profile_matches_candidate_split": (
                    profile.split == candidate.split
                ),
                "v4_profile_matches_candidate_catalog_index": (
                    int(profile.catalog_index) == int(candidate.catalog_index)
                ),
                "v4_profile_id_matches_both_action_blocks": (
                    low_profile_id
                    == high_profile_id
                    == profile.action_profile_id
                ),
                "v4_vertical_force_coupling_is_0_40_n": (
                    V4_VERTICAL_FORCE_COUPLING_N == 0.40
                ),
            }
        )
        audit["v4"] = {
            "protocol": V4_PROTOCOL,
            "vertical_force_coupling_n": V4_VERTICAL_FORCE_COUPLING_N,
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
        audit["v4"]["fresh_simulator_replay"] = replay_audit
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
    "VERTICAL_FORCE_COUPLING_N",
    "CubeGraspRuleCandidate",
    "CubeGraspRuleV4Candidate",
    "CubeGraspRuleV4Profile",
    "CubeGraspRuleV4Simulator",
    "V4_ACTION_ANCHORS",
    "V4_ANCHOR_GRIPPER_ACTIONS",
    "V4_ANCHOR_PROFILES",
    "V4_CONSTRAINT_GRID_DENOMINATOR",
    "V4_CONSTRAINT_TOLERANCE",
    "V4_FORMAL_CATALOG_INDEX_OFFSET",
    "V4R1_FORMAL_CATALOG_INDEX_OFFSET",
    "V4_PERTURBATION_BASES",
    "V4_PERTURBATION_COEFFICIENT_LIMIT",
    "V4_PROFILE_SPLIT_SEEDS",
    "V4_PROTOCOL",
    "V4_VERTICAL_FORCE_COUPLING_N",
    "action_blocks",
    "action_profile_content_sha256",
    "make_v4_action_profile",
    "make_v4_candidate",
    "validate_v4_action_profile",
]
