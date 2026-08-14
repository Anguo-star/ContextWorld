#!/usr/bin/env python3
"""Reconstruct and freeze all Cube v4 preformal content identities.

This audit deterministically replays the already-completed 16-scene coupling
pilot and the two real-MuJoCo regression pairs.  It creates no Lance data,
does no model work, and rejects every Public-Test-shaped input or output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.cube_grasp_rule_h3 import (
    GRASP_MODES,
    CubeGraspRuleCandidate,
    CubeGraspRuleSimulator,
    array_sha256,
    validate_cube_grasp_rule_pair,
)
from contextworld.evaluation.cube_grasp_rule_h3_v3 import (
    CubeGraspRuleV3Candidate,
    action_blocks as v3_action_blocks,
    make_v3_candidate,
    validate_v3_action_profile,
)
from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
    V4_FORMAL_CATALOG_INDEX_OFFSET,
    CubeGraspRuleV4Simulator,
    make_v4_candidate,
)
from scripts.build_cube_grasp_rule_h3_v3_data import _eligible_source_rows
from scripts.build_cube_grasp_rule_h3_v4_data import (
    action_profile_content_sha256,
    pair_content_sha256,
    scene_template_content_sha256,
)


PROTOCOL = "cube_gripper_carry_rule_history3_development_v4"
PILOT_SEED = 2026081207
PILOT_COUNT = 16
PILOT_CATALOG_START = 100_000
COUPLINGS = (0.30, 0.40, 0.45, 0.50)
REAL_PAIR_ROWS = (30, 31)
REAL_PAIR_CATALOG_INDICES = (0, 1)
CONTENT_FIELDS = (
    "action_profile_ids",
    "scene_template_content_hashes",
    "pair_content_hashes",
    "query_pixel_hashes",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class CouplingSimulator(CubeGraspRuleSimulator):
    def __init__(self) -> None:
        self.coupling = 0.30
        super().__init__(resolution=224)

    def _apply_transition_force(self, *, mode: str, action_z: float) -> None:
        self.base._data.qfrc_applied[:] = 0.0
        gravity_z = float(self.base._model.opt.gravity[2])
        mass = float(self.base._model.body_mass[self.object_body_id])
        object_z_dof = self.object_dof_address + 2
        self.base._data.qfrc_applied[object_z_dof] = -mass * gravity_z
        if mode == "can_hold":
            self.base._data.qfrc_applied[object_z_dof] += (
                float(self.coupling) * float(action_z)
            )
        elif mode != "cannot_hold":
            raise ValueError(mode)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_content_digest(values: Sequence[str], *, field_name: str) -> str:
    normalized = list(values)
    if normalized != sorted(set(normalized)):
        raise ValueError(f"{field_name} must be sorted and unique")
    decoded = []
    for value in normalized:
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{field_name} contains a non-SHA256 value")
        decoded.append(bytes.fromhex(value))
    return hashlib.sha256(
        b"contextworld-cube-prior-content-exclusions-v1\0"
        + field_name.encode("ascii")
        + b"\0"
        + b"".join(decoded)
    ).hexdigest()


def excluded_source_episodes_sha256(values: Sequence[int]) -> str:
    normalized = [int(value) for value in values]
    if normalized != sorted(set(normalized)) or any(value < 0 for value in normalized):
        raise ValueError("episode IDs must be nonnegative, sorted, and unique")
    payload = np.asarray(normalized, dtype="<i8").tobytes()
    return hashlib.sha256(
        b"contextworld-cube-prior-source-episodes-v1\0" + payload
    ).hexdigest()


def _reject_public_path(path: Path, *, label: str) -> None:
    forbidden = {
        "validation",
        "validation.lance",
        "public",
        "public_test",
        "public-test",
        "publictest",
    }
    component = next(
        (part for part in path.parts if part.lower() in forbidden), None
    )
    if component is not None:
        raise RuntimeError(f"{label} contains Public component {component!r}")


def _verified_file(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    _reject_public_path(path, label=label)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file")
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} expected SHA256 is malformed")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} SHA256 mismatch: {observed} != {expected_sha256}")
    return {"sha256": observed, "size_bytes": path.stat().st_size}


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} root must be an object")
    return value


def _formal_v3_episodes(document: Mapping[str, Any]) -> set[int]:
    if document.get("protocol") != (
        "cube_gripper_carry_rule_history3_development_v3"
    ) or document.get("passed") is not True:
        raise RuntimeError("formal v3 report identity mismatch")
    if document.get("public_test_opened") is not False or document.get(
        "public_test_generated"
    ) is not False:
        raise RuntimeError("formal v3 report did not keep Public closed")
    splits = document.get("splits")
    if not isinstance(splits, Mapping):
        raise RuntimeError("formal v3 report lacks splits")
    values = {
        int(episode)
        for split in ("train", "loader_validation")
        for episode in splits[split]["source_episodes"]
    }
    if len(values) != 2304:
        raise RuntimeError("formal v3 source episode count mismatch")
    return values


def _pilot_candidates(
    source_h5: Path, formal_episodes: set[int]
) -> list[CubeGraspRuleV3Candidate]:
    eligible = [
        item
        for item in _eligible_source_rows(source_h5)
        if int(item[1]) not in formal_episodes
    ]
    generator = np.random.default_rng(PILOT_SEED)
    selected = [
        eligible[int(index)]
        for index in generator.permutation(len(eligible))[:PILOT_COUNT]
    ]
    requested_rows = sorted(row for row, _, _ in selected)
    with h5py.File(source_h5, "r", swmr=True) as handle:
        qpos = np.asarray(handle["qpos"][requested_rows], dtype=np.float64)
        control = np.asarray(handle["control"][requested_rows], dtype=np.float64)
    source_values = {
        row: (qpos[index], control[index])
        for index, row in enumerate(requested_rows)
    }
    candidates = []
    for index, (source_row, source_episode, source_step) in enumerate(selected):
        scene_rng = np.random.default_rng(
            np.random.SeedSequence([PILOT_SEED, index, 0xC0BE])
        )
        source_qpos, source_control = source_values[source_row]
        base = CubeGraspRuleCandidate(
            candidate_id=f"cube-v4-feasibility-{index:03d}",
            split="train",
            catalog_index=PILOT_CATALOG_START + index,
            source_row=int(source_row),
            source_episode=int(source_episode),
            source_step=int(source_step),
            simulator_seed=int(scene_rng.integers(0, 2**31 - 1)),
            task_id=1 + index % 5,
            qpos=tuple(float(value) for value in source_qpos),
            control=tuple(float(value) for value in source_control),
            cube_color=tuple(float(value) for value in scene_rng.uniform(0.18, 0.92, 3)),
            target_position=(
                float(scene_rng.uniform(0.32, 0.53)),
                float(scene_rng.uniform(-0.24, 0.24)),
                0.02,
            ),
        )
        candidates.append(make_v3_candidate(base))
    return candidates


def _pilot_content(
    candidates: Sequence[CubeGraspRuleV3Candidate],
    pilot: Mapping[str, Any],
) -> tuple[set[int], dict[str, set[str]], dict[str, Any]]:
    scope = pilot.get("scope")
    if not isinstance(scope, Mapping) or scope.get(
        "public_test_opened_read_hashed_or_scored"
    ) is not False or scope.get("reference_model_training_or_scoring") is not False:
        raise RuntimeError("coupling pilot scope is contaminated")
    if int(scope.get("new_scene_and_selection_seed", -1)) != PILOT_SEED:
        raise RuntimeError("coupling pilot seed mismatch")
    variants = pilot.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {
        f"coupling_{value:.2f}_n" for value in COUPLINGS
    }:
        raise RuntimeError("coupling pilot variants mismatch")
    content = {field: set() for field in CONTENT_FIELDS}
    episodes = {candidate.source_episode for candidate in candidates}
    simulator = CouplingSimulator()
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    try:
        for coupling in COUPLINGS:
            name = f"coupling_{coupling:.2f}_n"
            frozen_rows = variants[name].get("rows_without_feature_vectors")
            if not isinstance(frozen_rows, list) or len(frozen_rows) != PILOT_COUNT:
                raise RuntimeError(f"{name} row count mismatch")
            expected_rows = {str(row["candidate_id"]): row for row in frozen_rows}
            observed_rows = []
            simulator.coupling = coupling
            for candidate in candidates:
                frozen = expected_rows.get(candidate.candidate_id)
                if frozen is None or int(frozen["source_episode"]) != candidate.source_episode:
                    raise RuntimeError(f"{name} candidate/source identity mismatch")
                if frozen["action_anchor_id"] != candidate.action_profile.action_anchor_id:
                    raise RuntimeError(f"{name} anchor identity mismatch")
                if not validate_v3_action_profile(candidate.action_profile)["passed"]:
                    raise RuntimeError("pilot profile constraints failed")
                blocks = v3_action_blocks(candidate.action_profile)
                payload = {
                    mode: simulator._run_mode(candidate, mode=mode, blocks=blocks)
                    for mode in GRASP_MODES
                }
                audit = validate_cube_grasp_rule_pair(
                    payload["cannot_hold"], payload["can_hold"]
                )
                if not audit["passed"]:
                    raise RuntimeError(f"{name} reconstructed pair failed")
                for key in ("history_height_gap_m", "future_height_gap_m"):
                    observed = audit[
                        "history_cube_height_gap_m"
                        if key.startswith("history")
                        else "future_cube_height_gap_m"
                    ]
                    if not np.isclose(observed, float(frozen[key]), rtol=0.0, atol=1e-15):
                        raise RuntimeError(f"{name} {key} differs from frozen pilot")
                profile_id = action_profile_content_sha256(blocks)
                scene_hash = scene_template_content_sha256(candidate)
                pair_hash = pair_content_sha256(scene_hash, profile_id)
                query_hash = str(audit["hashes"]["query_pixels"])
                content["action_profile_ids"].add(profile_id)
                content["scene_template_content_hashes"].add(scene_hash)
                content["pair_content_hashes"].add(pair_hash)
                content["query_pixel_hashes"].add(query_hash)
                observed_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source_episode": candidate.source_episode,
                        "catalog_index": candidate.catalog_index,
                        "action_anchor_id": candidate.action_profile.action_anchor_id,
                        "action_profile_id": profile_id,
                        "scene_template_content_hash": scene_hash,
                        "pair_content_hash": pair_hash,
                        "query_pixel_hash": query_hash,
                    }
                )
            rows_by_variant[name] = observed_rows
    finally:
        simulator.close()
    return episodes, content, {
        "scene_count": len(candidates),
        "pair_count": len(candidates) * len(COUPLINGS),
        "couplings_n": list(COUPLINGS),
        "rows_by_variant": rows_by_variant,
        "checks_passed": True,
    }


def _real_pair_content(source_h5: Path) -> tuple[set[int], dict[str, set[str]], dict[str, Any]]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        bases = [
            CubeGraspRuleCandidate(
                candidate_id=f"cube-v4-smoke-row{row}",
                split="train",
                catalog_index=index,
                source_row=row,
                source_episode=int(handle["ep_idx"][row]),
                source_step=int(handle["step_idx"][row]),
                simulator_seed=123 + index,
                task_id=1,
                qpos=tuple(float(value) for value in handle["qpos"][row]),
                control=tuple(float(value) for value in handle["control"][row]),
                cube_color=(0.5, 0.4, 0.3),
                target_position=(0.4, 0.0, 0.02),
            )
            for index, row in zip(REAL_PAIR_CATALOG_INDICES, REAL_PAIR_ROWS)
        ]
    content = {field: set() for field in CONTENT_FIELDS}
    episodes = {base.source_episode for base in bases}
    rows = []
    simulator = CubeGraspRuleV4Simulator()
    replay = CubeGraspRuleV4Simulator()
    try:
        for base in bases:
            candidate = make_v4_candidate(base)
            pair = simulator.build_pair(candidate, replay_simulator=replay)
            if pair is None or pair["audit"].get("passed") is not True:
                raise RuntimeError("historical real-MuJoCo pair reconstruction failed")
            profile_id = candidate.action_profile.action_profile_id
            scene_hash = scene_template_content_sha256(candidate)
            pair_hash = pair_content_sha256(scene_hash, profile_id)
            query_hash = str(pair["audit"]["hashes"]["query_pixels"])
            content["action_profile_ids"].add(profile_id)
            content["scene_template_content_hashes"].add(scene_hash)
            content["pair_content_hashes"].add(pair_hash)
            content["query_pixel_hashes"].add(query_hash)
            rows.append(
                {
                    "candidate": asdict(candidate),
                    "action_profile_id": profile_id,
                    "scene_template_content_hash": scene_hash,
                    "pair_content_hash": pair_hash,
                    "query_pixel_hash": query_hash,
                    "history_height_gap_m": pair["audit"]["history_cube_height_gap_m"],
                    "future_height_gap_m": pair["audit"]["future_cube_height_gap_m"],
                    "fresh_replay_passed": pair["audit"]["v4"][
                        "fresh_simulator_replay"
                    ]["passed"],
                }
            )
    finally:
        simulator.close()
        replay.close()
    if any(index >= V4_FORMAL_CATALOG_INDEX_OFFSET for index in REAL_PAIR_CATALOG_INDICES):
        raise RuntimeError("historical real-pair indices overlap formal namespace")
    return episodes, content, {
        "pair_count": len(rows),
        "catalog_indices": list(REAL_PAIR_CATALOG_INDICES),
        "formal_catalog_index_offset": V4_FORMAL_CATALOG_INDEX_OFFSET,
        "catalog_indices_below_formal_offset": True,
        "rows": rows,
        "checks_passed": True,
    }


def freeze(
    *,
    source_h5: Path,
    source_sha256: str,
    pilot_json: Path,
    pilot_sha256: str,
    formal_v3_report: Path,
    formal_v3_sha256: str,
    pilot_runner: Path,
    pilot_runner_sha256: str,
    historical_physics_snapshot: Path,
    historical_physics_sha256: str,
    historical_test_snapshot: Path,
    historical_test_sha256: str,
    output: Path,
) -> dict[str, Any]:
    _reject_public_path(output, label="output")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    inputs = {
        "source_h5": _verified_file(source_h5, source_sha256, label="source H5"),
        "coupling_pilot": _verified_file(pilot_json, pilot_sha256, label="pilot JSON"),
        "formal_v3_build_report": _verified_file(
            formal_v3_report, formal_v3_sha256, label="formal v3 report"
        ),
        "original_pilot_runner": _verified_file(
            pilot_runner, pilot_runner_sha256, label="original pilot runner"
        ),
        "historical_v4_physics_snapshot": _verified_file(
            historical_physics_snapshot,
            historical_physics_sha256,
            label="historical physics snapshot",
        ),
        "historical_v4_test_snapshot": _verified_file(
            historical_test_snapshot,
            historical_test_sha256,
            label="historical test snapshot",
        ),
    }
    pilot = _load_json(pilot_json, label="coupling pilot")
    formal = _load_json(formal_v3_report, label="formal v3 report")
    formal_episodes = _formal_v3_episodes(formal)
    candidates = _pilot_candidates(source_h5, formal_episodes)
    pilot_episodes, pilot_content, pilot_component = _pilot_content(candidates, pilot)
    real_episodes, real_content, real_component = _real_pair_content(source_h5)
    if pilot_episodes & real_episodes:
        raise RuntimeError("pilot and real-pair source episodes overlap")
    episodes = sorted(pilot_episodes | real_episodes)
    content_receipt = {}
    for field in CONTENT_FIELDS:
        values = sorted(pilot_content[field] | real_content[field])
        content_receipt[field] = {
            "values": values,
            "count": len(values),
            "sha256": canonical_content_digest(values, field_name=field),
        }
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "receipt_id": "cube_gripper_carry_h3_v4_preformal_content_v1",
        "status": "frozen_before_first_v4_build",
        "checks_passed": True,
        "input_identities": inputs,
        "reconstruction_contract": {
            "existing_pilot_replayed_not_reselected": True,
            "existing_real_mujoco_tests_replayed": True,
            "lance_opened_or_generated": False,
            "formal_build_attempted": False,
            "coupling_or_probe_recipe_changed": False,
        },
        "components": {
            "coupling_pilot": pilot_component,
            "v4_real_pair_tests": real_component,
        },
        "excluded_source_episodes": episodes,
        "excluded_source_episode_count": len(episodes),
        "excluded_source_episodes_sha256": excluded_source_episodes_sha256(episodes),
        "prior_content_exclusions": content_receipt,
        "public_test": {
            "access_status": "closed_not_read_not_scored",
            "opened": False,
            "read": False,
            "hashed": False,
            "scored": False,
        },
        "reference_model_training_or_scoring": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-h5",
        "pilot-json",
        "formal-v3-report",
        "pilot-runner",
        "historical-physics-snapshot",
        "historical-test-snapshot",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            _reject_public_path(value, label=name)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = freeze(
        source_h5=args.source_h5.expanduser().resolve(),
        source_sha256=args.source_h5_sha256,
        pilot_json=args.pilot_json.expanduser().resolve(),
        pilot_sha256=args.pilot_json_sha256,
        formal_v3_report=args.formal_v3_report.expanduser().resolve(),
        formal_v3_sha256=args.formal_v3_report_sha256,
        pilot_runner=args.pilot_runner.expanduser().resolve(),
        pilot_runner_sha256=args.pilot_runner_sha256,
        historical_physics_snapshot=args.historical_physics_snapshot.expanduser().resolve(),
        historical_physics_sha256=args.historical_physics_snapshot_sha256,
        historical_test_snapshot=args.historical_test_snapshot.expanduser().resolve(),
        historical_test_sha256=args.historical_test_snapshot_sha256,
        output=args.output.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "checks_passed": receipt["checks_passed"],
                "excluded_source_episode_count": receipt[
                    "excluded_source_episode_count"
                ],
                "content_counts": {
                    field: receipt["prior_content_exclusions"][field]["count"]
                    for field in CONTENT_FIELDS
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
