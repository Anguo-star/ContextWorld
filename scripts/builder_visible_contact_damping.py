#!/usr/bin/env python3
"""Build visible PushT futures for frozen contact-friction and motion-damping sources.

The builder replays each exact frozen history from x0, then runs shared action
sequences under both hidden conditions. Candidate slots come from the previous
fixed 11-action bank; each slot receives its own largest safe factor from a
preregistered descending list. Safety checks use every Pymunk shape at every
raw step. Physical future arrays use exact unique float32 action sequences, with
slot-to-unique metadata preserved. No model scores select sources or actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np

TASKS = ("contact_friction", "motion_damping")
# Rendering-identical pins, verified bitwise against the frozen Development tables.
DEFAULT_STABLE_REFS = {
    "contact_friction": "6ab823fdc6921c95089992ed49c39e431e21ca4a",
    "motion_damping": "875e607fc08aa72eacb94d5d178127804134cc06",
}
_SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_REPO = _SCRIPT_DIR.parent if (_SCRIPT_DIR.parent / "contextworld").is_dir() else _SCRIPT_DIR
RELEASE_MANIFEST_RELATIVE = {
    "contact_friction": Path("synthesis/pusht_contact_friction_h3_release_v3/manifest.json"),
    "motion_damping": Path("synthesis/pusht_motion_damping_h3_release_v4/manifest.json"),
}
RELEASE_CONFIGS = {
    "contact_friction": "configs/benchmark/pusht_contact_friction_icl_release_v1.yaml",
    "motion_damping": "configs/benchmark/pusht_motion_damping_icl_release_v1.yaml",
}
DEVELOPMENT_SPLIT = "loader_validation"
EXPECTED_TABLE_SHA256 = {
    "contact_friction": "5b8576f2f8220bf1f2b323799732169837b29b3195a3900f8883f8f3cc62a825",
    "motion_damping": "64d43c931f106c2d53e3c3084e62381d2f2640c9d943e269475f3fb76aaa2de4",
}
QUERY_TOLERANCES = {"contact_friction": 1e-5, "motion_damping": 1e-8}
NATIVE_FUTURE_STATE_TOLERANCE = 5e-5  # float32 serialization of the frozen physics column
CONDITION_NAMES = {
    "contact_friction": ("low_friction", "high_friction"),
    "motion_damping": ("faster_decay", "no_extra_decay"),
}
CONDITION_VALUES = {"contact_friction": (0.05, 0.80), "motion_damping": (0.2, 1.0)}
SCENE_PREFIX = {"contact_friction": "pcf", "motion_damping": "pmd"}
NATIVE_CANDIDATE_INDEX = {"contact_friction": 2, "motion_damping": 0}

HISTORY_TOKENS = 3
RAW_STEPS_PER_BLOCK = 5
FUTURE_BLOCKS = 5
FUTURE_RAW_STEPS = FUTURE_BLOCKS * RAW_STEPS_PER_BLOCK
HISTORY_RAW_STEPS = (HISTORY_TOKENS - 1) * RAW_STEPS_PER_BLOCK
RESOLUTION = 224
JPEG_QUALITY = 95
PHYSICAL_STEPS = [5, 10, 15, 20, 25]
ANGLE_RADIUS_PX = 40.0
FACTOR_VALUES = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0)
GEOMETRY_MIN = 2.0
GEOMETRY_MAX = 510.0
RENDER_EQUIVALENT_WORLD_GAP_THRESHOLD = 2.0 * 512.0 / RESOLUTION
CANDIDATE_NAMES = (
    "amp0.0",
    "amp0.5",
    "amp1.0",
    "amp1.5",
    "amp2.0",
    "amp-1.0",
    "trunc1",
    "trunc2",
    "trunc3",
    "trunc4",
    "counter",
)
AMPLITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, -1.0)
PROBE_MAGNITUDE = 0.4
PROBE_INTERCEPT_AHEAD_PX = 50.0
GOAL_CONDITION_INDEX = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
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


def angle_wrap(delta):
    return (np.asarray(delta, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def snapshot_delta(left, right) -> np.ndarray:
    """Wrapped 12-D snapshot difference (body angles at indices 4 and 10)."""
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if delta.shape[-1] != 12:
        raise ValueError(f"expected 12-D snapshots, got {delta.shape}")
    for index in (4, 10):
        delta[..., index] = angle_wrap(delta[..., index])
    return delta


def physical_cost(states, goal):
    """px-equivalent: agent xy + block xy + 40*wrapped block angle (frozen convention)."""
    delta = snapshot_delta(states, goal)
    squared = np.sum(delta[..., [0, 1, 6, 7]] ** 2, axis=-1)
    return np.sqrt(squared + (ANGLE_RADIUS_PX * delta[..., 10]) ** 2)


def success_predicate(snapshot, goal, symmetry: float) -> bool:
    """Benchmark PushT success: position < 20 px and angle < pi/9 modulo symmetry."""
    snapshot = np.asarray(snapshot, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    pos_diff = np.linalg.norm(
        np.concatenate([snapshot[0:2], snapshot[6:8]])
        - np.concatenate([goal[0:2], goal[6:8]])
    )
    angle_diff = abs(float(angle_wrap(snapshot[10] - goal[10]))) % symmetry
    angle_diff = min(angle_diff, symmetry - angle_diff)
    return bool(pos_diff < 20 and angle_diff < np.pi / 9)


def shape_symmetry(visible_shape_id: int) -> float:
    return np.pi / 2 if int(visible_shape_id) == 4 else 2 * np.pi


def _shape_body_bounds(body, pymunk_module) -> list[float]:
    """Exact conservative world AABB over every Pymunk shape on one body."""
    boxes = []
    for shape in body.shapes:
        if isinstance(shape, pymunk_module.Circle):
            center = body.local_to_world(shape.offset)
            radius = float(shape.radius)
            boxes.append([float(center.x)-radius, float(center.y)-radius,
                          float(center.x)+radius, float(center.y)+radius])
        elif isinstance(shape, pymunk_module.Poly):
            vertices = [body.local_to_world(vertex) for vertex in shape.get_vertices()]
            radius = float(shape.radius)
            boxes.append([min(float(v.x) for v in vertices)-radius,
                          min(float(v.y) for v in vertices)-radius,
                          max(float(v.x) for v in vertices)+radius,
                          max(float(v.y) for v in vertices)+radius])
        elif isinstance(shape, pymunk_module.Segment):
            endpoints = [body.local_to_world(shape.a), body.local_to_world(shape.b)]
            radius = float(shape.radius)
            boxes.append([min(float(v.x) for v in endpoints)-radius,
                          min(float(v.y) for v in endpoints)-radius,
                          max(float(v.x) for v in endpoints)+radius,
                          max(float(v.y) for v in endpoints)+radius])
        else:
            raise TypeError(f"unsupported body shape for geometry audit: {type(shape).__name__}")
    if not boxes:
        raise RuntimeError("dynamic body has no Pymunk shapes")
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def _all_body_geometry_bounds(env, pymunk_module) -> np.ndarray:
    """Return [agent, block] full-shape AABBs as [xmin,ymin,xmax,ymax]."""
    return np.asarray([_shape_body_bounds(env.agent, pymunk_module),
                       _shape_body_bounds(env.block, pymunk_module)], dtype=np.float64)


def _bounds_are_safe(bounds: np.ndarray) -> bool:
    bounds = np.asarray(bounds, dtype=np.float64)
    return bool(np.all(bounds[:, 0] >= GEOMETRY_MIN)
                and np.all(bounds[:, 1] >= GEOMETRY_MIN)
                and np.all(bounds[:, 2] <= GEOMETRY_MAX)
                and np.all(bounds[:, 3] <= GEOMETRY_MAX))


def _snapshot_world_pixel_gap(left: np.ndarray, right: np.ndarray) -> float:
    """512-canvas equivalent from body positions and the block-angle chord."""
    delta = snapshot_delta(left, right)
    position_sq = float(np.sum(delta[[0, 1, 6, 7]] ** 2))
    angle_chord = 2.0 * ANGLE_RADIUS_PX * abs(float(np.sin(delta[10] / 2.0)))
    return float(np.sqrt(position_sq + angle_chord ** 2))


def damping_probe(query_state) -> np.ndarray:
    """Preregistered probe five-step query defined from the shared query state only.

    The frozen motion-damping query action is the zero (coast) block, so
    amplitude scaling of the native query would be degenerate.  The bank
    therefore scales this preregistered probe: direction from the agent query
    position toward a point PROBE_INTERCEPT_AHEAD_PX ahead of the block along
    its shared query velocity (snapshot indices 8:10; falls back to the
    agent->block direction when the query speed is negligible), magnitude
    PROBE_MAGNITUDE.  Amplitude 0 of the probe recovers the frozen coast query
    exactly.  Returns shape (5, 2).
    """
    state = np.asarray(query_state, dtype=np.float64)
    velocity = state[8:10]
    speed = float(np.linalg.norm(velocity))
    target = state[6:8] + (
        PROBE_INTERCEPT_AHEAD_PX * velocity / speed if speed >= 1e-6 else 0.0
    )
    direction = target - state[0:2]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction, norm = np.asarray([1.0, 0.0]), 1.0
    return np.tile(np.clip(PROBE_MAGNITUDE * direction / norm, -1.0, 1.0), (RAW_STEPS_PER_BLOCK, 1))


def build_candidate_bank(native) -> np.ndarray:
    """C=11 fixed bank chosen before any model inference.

    Returns float32 (C, 5 blocks * 5 raw steps, 2).  amp family: the native
    five-step query repeated for all five blocks at amplitudes 0, .5, 1, 1.5,
    2 and -1 (elementwise, clipped to [-1, 1]).  trunc family: native blocks
    1..m followed by zero blocks, m = 1..4.  counter: native block 1 then the
    elementwise counteraction (-native, clipped) for blocks 2..5.
    """
    native = np.asarray(native, dtype=np.float64)
    if native.shape != (RAW_STEPS_PER_BLOCK, 2):
        raise ValueError(f"native query must be {(RAW_STEPS_PER_BLOCK, 2)}, got {native.shape}")
    if np.any(native < -1.0) or np.any(native > 1.0):
        raise ValueError("native query must stay inside [-1, 1]")
    zero_block = np.zeros((RAW_STEPS_PER_BLOCK, 2), dtype=np.float64)
    bank = []
    for amplitude in AMPLITUDES:
        block = np.clip(amplitude * native, -1.0, 1.0)
        bank.append(np.tile(block, (FUTURE_BLOCKS, 1)))
    for keep in (1, 2, 3, 4):
        bank.append(
            np.concatenate(
                [np.tile(native, (keep, 1))] + [zero_block] * (FUTURE_BLOCKS - keep), axis=0
            )
        )
    counter = np.concatenate(
        [native, np.tile(np.clip(-native, -1.0, 1.0), (FUTURE_BLOCKS - 1, 1))], axis=0
    )
    bank.append(counter)
    bank = np.asarray(bank, dtype=np.float32)
    bank[bank == 0.0] = np.float32(0.0)  # canonicalize signed zero before exact-byte deduplication
    assert bank.shape == (len(CANDIDATE_NAMES), FUTURE_RAW_STEPS, 2)
    return bank


def extract_stable_tree(stable_repo: Path, stable_ref: str, destination: Path) -> dict:
    """Materialise stable_worldmodel at stable_ref without editing any repository."""
    marker = destination / "STABLE_REF.txt"
    env_file = destination / "stable_worldmodel" / "envs" / "pusht" / "env.py"
    if marker.exists() and marker.read_text().strip() == stable_ref and env_file.exists():
        return {"tree": str(destination), "extracted": False, "env_py_sha256": file_sha256(env_file)}
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    archive = subprocess.Popen(
        ["git", "-C", str(stable_repo), "archive", stable_ref, "stable_worldmodel"],
        stdout=subprocess.PIPE,
    )
    tar = subprocess.Popen(["tar", "-x", "-C", str(destination)], stdin=archive.stdout)
    archive.stdout.close()
    tar.wait()
    if archive.wait() != 0 or tar.returncode != 0:
        raise RuntimeError(f"git archive of {stable_repo}@{stable_ref} failed")
    marker.write_text(stable_ref + "\n")
    if not env_file.exists():
        raise RuntimeError("extracted tree lacks stable_worldmodel/envs/pusht/env.py")
    return {"tree": str(destination), "extracted": True, "env_py_sha256": file_sha256(env_file)}


def build_scene(job: dict) -> dict:
    """Replay one frozen pair and simulate every candidate future per condition."""
    sys.path[:0] = [job["stable_tree"], str(ROOT_REPO)]
    sys.modules.setdefault("flash_attn", None)
    import pymunk
    from contextworld.benchmarks.contact_friction_icl_data import _decode_rgb
    from contextworld.evaluation import pusht_motion_damping_h3 as damping
    from contextworld.evaluation.pusht_contact_friction_h3 import (
        ENDPOINT_MODES as FRICTION_MODES,
        ContactFrictionTemplate,
        _snapshot_delta,
        _step_and_count_agent_block_contacts,
        body_snapshot,
        make_contact_friction_env,
    )
    from contextworld.synthesis.lance import encode_frame

    task = job["task"]
    with np.load(job["legacy_panel_path"], allow_pickle=False) as legacy:
        legacy_pair_id = str(np.asarray(legacy["pair_id"]).item())
        if legacy_pair_id != str(job["pair_id"]):
            raise RuntimeError(
                f"legacy panel scene mismatch: {legacy_pair_id} != {job['pair_id']}"
            )
        legacy_goal_pixels = np.asarray(legacy["goal_pixels"], dtype=np.uint8).copy()
        legacy_goal_state = np.asarray(legacy["goal_state"], dtype=np.float32).copy()
    if legacy_goal_pixels.shape != (RESOLUTION, RESOLUTION, 3) or legacy_goal_state.shape != (12,):
        raise RuntimeError(f"malformed legacy goal for {job['pair_id']}")
    codec = dict(format="jpeg", quality=JPEG_QUALITY)

    def render_frame(env):
        return _decode_rgb(encode_frame(np.asarray(env.render(), dtype=np.uint8), codec))

    template_data = job["template"]
    base_candidates = np.asarray(job["candidate_actions"], dtype=np.float32).copy()  # (C, 25, 2)
    base_candidates[base_candidates == 0.0] = np.float32(0.0)
    candidates = base_candidates.copy()
    native_query_actions = np.asarray(job["native_query_actions"], dtype=np.float32)
    stored_frames = np.asarray(job["stored_frames"])    # [K, 4, 224, 224, 3] uint8
    stored_physics = np.asarray(job["stored_physics"])  # [K, 4, 12] float32
    history_actions = np.asarray(job["history_blocks"], dtype=np.float32).reshape(
        HISTORY_RAW_STEPS, 2
    )
    C = candidates.shape[0]
    U = C  # replaced by the exact unique-action count after factor selection
    K = stored_frames.shape[0]
    native_slot_index = int(job["native_candidate_index"])
    symmetry = shape_symmetry(job["visible_shape_id"])
    expected_query = np.asarray(job["expected_query_snapshot"], dtype=np.float64)
    query_tolerance = float(job["query_tolerance"])

    history_pixels = stored_frames[:, :HISTORY_TOKENS].copy()
    context_actions = np.tile(
        np.asarray(job["history_blocks"], dtype=np.float32)[None], (K, 1, 1, 1)
    )
    initial_states = np.zeros((K, 12), dtype=np.float32)
    query_states = np.zeros((K, 12), dtype=np.float32)
    query_residuals = np.zeros(K, dtype=np.float64)
    future_pixels = np.zeros((K, U, FUTURE_BLOCKS, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    future_states = np.zeros((K, U, FUTURE_BLOCKS, 12), dtype=np.float32)
    future_step_states = np.zeros((K, U, FUTURE_RAW_STEPS, 12), dtype=np.float32)
    future_geometry_bounds = np.zeros((K, U, FUTURE_RAW_STEPS, 2, 4), dtype=np.float64)
    future_contacts = np.zeros((K, U, FUTURE_RAW_STEPS), dtype=np.int16)
    history_contacts = np.zeros((K, HISTORY_RAW_STEPS), dtype=np.int16)
    bodies_inside = np.ones((K, U), dtype=np.int8)
    render_match = {"x0": [False] * K, "x1": [False] * K, "x2": [False] * K,
                    "native_slot_first_block": [False] * K}
    native_future_state_gap = np.zeros(K, dtype=np.float64)
    native_future_frame_equal = [False] * K
    native_source_state_gap = np.zeros(K, dtype=np.float64)
    native_source_frame_equal = [False] * K
    prefix_pixels_exact = [True] * K
    prefix_state_gap = np.zeros(K, dtype=np.float64)

    def create_env(mode_index: int):
        if task == "contact_friction":
            local_template = ContactFrictionTemplate(**template_data)
            return make_contact_friction_env(
                local_template, mode=FRICTION_MODES[mode_index], resolution=RESOLUTION
            )[0]
        local_template = damping.MotionDampingTemplate(**template_data)
        return damping.make_motion_damping_env(
            local_template, mode=damping.ENDPOINT_MODES[mode_index], resolution=RESOLUTION
        )[0]

    def replay_source_history(mode_index: int):
        fresh = create_env(mode_index)
        checks = [bool(np.array_equal(render_frame(fresh), stored_frames[mode_index, 0]))]
        for step, action in enumerate(history_actions):
            _step_and_count_agent_block_contacts(fresh, action)
            if step + 1 in (RAW_STEPS_PER_BLOCK, HISTORY_RAW_STEPS):
                frame_index = (step + 1) // RAW_STEPS_PER_BLOCK
                checks.append(bool(np.array_equal(
                    render_frame(fresh), stored_frames[mode_index, frame_index]
                )))
        query = body_snapshot(fresh)
        residual = float(np.max(np.abs(_snapshot_delta(query, expected_query))))
        if residual > query_tolerance:
            fresh.close()
            raise RuntimeError(
                f"{job['condition_names'][mode_index]} replayed query residual "
                f"{residual:.3e} exceeds {query_tolerance:.1e}"
            )
        if not all(checks):
            fresh.close()
            raise RuntimeError(f"{job['condition_names'][mode_index]} prefix pixels differ")
        return fresh, query, checks

    factor_search_safe = np.zeros((C, len(FACTOR_VALUES)), dtype=np.bool_)
    factor_search_details = [[None for _ in FACTOR_VALUES] for _ in range(C)]
    candidate_factor = np.zeros(C, dtype=np.float32)
    candidate_safe = np.zeros(C, dtype=np.bool_)
    factor_query_gap_max = np.zeros(K, dtype=np.float64)

    try:
        # Independently test every factor for every candidate. Each true factor
        # cell completes both hidden modes and all 25 raw steps on one replay.
        base_factor_cache = {}
        for candidate_index, base in enumerate(base_candidates):
            base_key = np.ascontiguousarray(base, dtype=np.float32).tobytes()
            if base_key in base_factor_cache:
                safe_row, detail_row = base_factor_cache[base_key]
                factor_search_safe[candidate_index] = safe_row
                factor_search_details[candidate_index] = [dict(item) for item in detail_row]
            else:
                for factor_index, factor in enumerate(FACTOR_VALUES):
                    scaled = np.asarray(base * np.float32(factor), dtype=np.float32)
                    scaled[scaled == 0.0] = np.float32(0.0)
                    condition_safe = []
                    for mode_index in range(K):
                        trial, query, _ = replay_source_history(mode_index)
                        query_gap = float(np.max(np.abs(_snapshot_delta(query, expected_query))))
                        factor_query_gap_max[mode_index] = max(
                            factor_query_gap_max[mode_index], query_gap
                        )
                        is_safe = True
                        try:
                            for raw in range(FUTURE_RAW_STEPS):
                                _step_and_count_agent_block_contacts(trial, scaled[raw])
                                if not _bounds_are_safe(_all_body_geometry_bounds(trial, pymunk)):
                                    is_safe = False
                        finally:
                            trial.close()
                        condition_safe.append(is_safe)
                    factor_search_safe[candidate_index, factor_index] = bool(all(condition_safe))
                    factor_search_details[candidate_index][factor_index] = {
                        "factor": float(factor),
                        "condition_safe": [bool(value) for value in condition_safe],
                        "raw_steps_checked": [FUTURE_RAW_STEPS] * K,
                    }
                base_factor_cache[base_key] = (
                    factor_search_safe[candidate_index].copy(),
                    [dict(item) for item in factor_search_details[candidate_index]],
                )
            safe_indices = np.flatnonzero(factor_search_safe[candidate_index])
            if len(safe_indices):
                candidate_factor[candidate_index] = np.float32(FACTOR_VALUES[int(safe_indices[0])])
                candidate_safe[candidate_index] = True
            else:
                candidate_factor[candidate_index] = np.float32(0.0)
                candidate_safe[candidate_index] = False
        candidates = np.asarray(base_candidates * candidate_factor[:, None, None], dtype=np.float32)
        candidates[candidates == 0.0] = np.float32(0.0)
        if np.any(candidates < -1.0) or np.any(candidates > 1.0):
            raise RuntimeError("scaled candidate actions left [-1,1]")

        # Exact float32 action equality determines the unique bank and equal weighting.
        unique_lookup = {}
        candidate_unique_index = np.zeros(C, dtype=np.int64)
        unique_representative_slots = []
        unique_action_sha256 = []
        for candidate_index, actions in enumerate(candidates):
            key = np.ascontiguousarray(actions, dtype=np.float32).tobytes()
            if key not in unique_lookup:
                unique_lookup[key] = len(unique_lookup)
                unique_representative_slots.append(candidate_index)
                unique_action_sha256.append(hashlib.sha256(key).hexdigest())
            candidate_unique_index[candidate_index] = unique_lookup[key]
        unique_action_count = len(unique_lookup)
        unique_multiplicity = np.bincount(candidate_unique_index, minlength=unique_action_count)
        candidate_slot_weight = np.asarray([
            1.0 / (unique_action_count * unique_multiplicity[index])
            for index in candidate_unique_index
        ], dtype=np.float64)
        unique_candidates = np.asarray(candidates[unique_representative_slots], dtype=np.float32)
        unique_candidate_names = [CANDIDATE_NAMES[index] for index in unique_representative_slots]
        unique_candidate_safe = np.asarray([
            bool(np.all(candidate_safe[candidate_unique_index == index]))
            for index in range(unique_action_count)
        ], dtype=np.bool_)
        U = unique_action_count
        native_index = int(candidate_unique_index[native_slot_index])
        future_pixels = np.zeros((K, U, FUTURE_BLOCKS, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        future_states = np.zeros((K, U, FUTURE_BLOCKS, 12), dtype=np.float32)
        future_step_states = np.zeros((K, U, FUTURE_RAW_STEPS, 12), dtype=np.float32)
        future_geometry_bounds = np.zeros((K, U, FUTURE_RAW_STEPS, 2, 4), dtype=np.float64)
        future_contacts = np.zeros((K, U, FUTURE_RAW_STEPS), dtype=np.int16)
        bodies_inside = np.ones((K, U), dtype=np.int8)

        # Separately replay the original frozen five-step query for row15 identity.
        for mode_index in range(K):
            native_env, _, _ = replay_source_history(mode_index)
            try:
                for action in native_query_actions:
                    _step_and_count_agent_block_contacts(native_env, action)
                native_frame = render_frame(native_env)
                native_state = body_snapshot(native_env)
                native_source_frame_equal[mode_index] = bool(np.array_equal(
                    native_frame, stored_frames[mode_index, 3]
                ))
                native_source_state_gap[mode_index] = float(np.max(np.abs(
                    _snapshot_delta(native_state, stored_physics[mode_index, 3].astype(np.float64))
                )))
            finally:
                native_env.close()

        for mode_index in range(K):
            if task == "contact_friction":
                template = ContactFrictionTemplate(**template_data)
                env, _ = make_contact_friction_env(
                    template, mode=FRICTION_MODES[mode_index], resolution=RESOLUTION
                )
            else:
                template = damping.MotionDampingTemplate(**template_data)
                env, _ = damping.make_motion_damping_env(
                    template, mode=damping.ENDPOINT_MODES[mode_index], resolution=RESOLUTION
                )
            try:
                # x0: capture physics, verify pixels bitwise (render does not step physics).
                initial_states[mode_index] = body_snapshot(env).astype(np.float32)
                render_match["x0"][mode_index] = bool(
                    np.array_equal(render_frame(env), stored_frames[mode_index, 0])
                )
                # Genuine continuous history prefix, identical actions for both conditions.
                for step, action in enumerate(history_actions):
                    history_contacts[mode_index, step] = _step_and_count_agent_block_contacts(
                        env, action
                    )
                    if step + 1 in (RAW_STEPS_PER_BLOCK, HISTORY_RAW_STEPS):
                        key = "x1" if step + 1 == RAW_STEPS_PER_BLOCK else "x2"
                        render_match[key][mode_index] = bool(
                            np.array_equal(
                                render_frame(env),
                                stored_frames[mode_index, (step + 1) // RAW_STEPS_PER_BLOCK],
                            )
                        )
                query = body_snapshot(env)
                query_states[mode_index] = query.astype(np.float32)
                query_residuals[mode_index] = float(
                    np.max(np.abs(_snapshot_delta(query, expected_query)))
                )
                if query_residuals[mode_index] > query_tolerance:
                    raise RuntimeError(
                        f"natural query residual {query_residuals[mode_index]:.3e} exceeds "
                        f"tolerance {query_tolerance:.1e}"
                    )

                def replay_prefix(mode: int):
                    # Fresh simulator per candidate; replay the genuine prefix.
                    env.close()
                    if task == "contact_friction":
                        fresh, _ = make_contact_friction_env(
                            template, mode=FRICTION_MODES[mode], resolution=RESOLUTION
                        )
                    else:
                        fresh, _ = damping.make_motion_damping_env(
                            template, mode=damping.ENDPOINT_MODES[mode], resolution=RESOLUTION
                        )
                    checks = [bool(np.array_equal(render_frame(fresh), stored_frames[mode, 0]))]
                    for step, action in enumerate(history_actions):
                        _step_and_count_agent_block_contacts(fresh, action)
                        if step + 1 in (RAW_STEPS_PER_BLOCK, HISTORY_RAW_STEPS):
                            frame_index = (step + 1) // RAW_STEPS_PER_BLOCK
                            checks.append(bool(np.array_equal(
                                render_frame(fresh), stored_frames[mode, frame_index]
                            )))
                    prefix_pixels_exact[mode] = bool(prefix_pixels_exact[mode] and all(checks))
                    gap = float(np.max(np.abs(_snapshot_delta(body_snapshot(fresh), query))))
                    prefix_state_gap[mode] = max(prefix_state_gap[mode], gap)
                    if gap > 1e-9:
                        raise RuntimeError(f"replayed query state drifted by {gap:.3e}")
                    if not all(checks):
                        raise RuntimeError(f"replayed prefix pixels differ for mode {mode}")
                    return fresh

                for candidate_index in range(U):
                    plan = unique_candidates[candidate_index]
                    if candidate_index:
                        env = replay_prefix(mode_index)
                    inside = True
                    for raw in range(FUTURE_RAW_STEPS):
                        action = np.asarray(plan[raw], dtype=np.float32)
                        future_contacts[mode_index, candidate_index, raw] = (
                            _step_and_count_agent_block_contacts(env, action)
                        )
                        future_step_states[mode_index, candidate_index, raw] = body_snapshot(
                            env
                        ).astype(np.float32)
                        bounds = _all_body_geometry_bounds(env, pymunk)
                        future_geometry_bounds[mode_index, candidate_index, raw] = bounds
                        if not _bounds_are_safe(bounds):
                            inside = False
                        if (raw + 1) % RAW_STEPS_PER_BLOCK == 0:
                            block = raw // RAW_STEPS_PER_BLOCK
                            final = body_snapshot(env)
                            future_states[mode_index, candidate_index, block] = final.astype(
                                np.float32
                            )
                            frame = render_frame(env)
                            future_pixels[mode_index, candidate_index, block] = frame
                            if block == 0 and candidate_index == native_index:
                                # The frozen row15 is the state after history + the
                                # single query block, i.e. the FIRST future block.
                                native_future_state_gap[mode_index] = float(
                                    np.max(
                                        np.abs(
                                            _snapshot_delta(
                                                final,
                                                stored_physics[mode_index, 3].astype(np.float64),
                                            )
                                        )
                                    )
                                )
                                native_future_frame_equal[mode_index] = bool(
                                    np.array_equal(frame, stored_frames[mode_index, 3])
                                )
                                render_match["native_slot_first_block"][mode_index] = (
                                    native_future_frame_equal[mode_index]
                                )
                    bodies_inside[mode_index, candidate_index] = int(inside)
            finally:
                env.close()
    except Exception as error:  # noqa: BLE001 - attempted scenes are retained as findings
        return {
            "index": job["index"],
            "scene_id": job["pair_id"],
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "query_residuals": query_residuals.tolist(),
            "render_match": {k: [bool(x) for x in v] for k, v in render_match.items()},
        }

    failures = []
    if not (all(render_match["x0"]) and all(render_match["x1"]) and all(render_match["x2"])):
        failures.append("history frames do not reproduce bitwise")
    if not all(native_source_frame_equal):
        failures.append("standalone native five-step source query does not reproduce frozen row15 pixels")
    if float(np.max(native_source_state_gap)) > NATIVE_FUTURE_STATE_TOLERANCE:
        failures.append("standalone native five-step source query exceeds frozen row15 state tolerance")
    if task == "motion_damping" and np.any(base_candidates[0] != 0.0):
        failures.append("amplitude-0 base candidate is not the frozen coast query")
    if failures:
        return {
            "index": job["index"],
            "scene_id": job["pair_id"],
            "status": "failed",
            "failures": failures,
            "path": None,
            "query_residuals": query_residuals.tolist(),
            "native_row15_state_gap": native_source_state_gap.tolist(),
            "native_row15_pixels_bitwise_exact": [bool(value) for value in native_source_frame_equal],
            "render_match": {k: [bool(x) for x in v] for k, v in render_match.items()},
            "trajectory_receipt": {
                "replayed_original_history": False,
                "query_state_injected": False,
                "all_candidates_continuous": False,
                "future_raw_steps": FUTURE_RAW_STEPS,
                "factor_search_complete": False,
                "factor_search_factors": list(FACTOR_VALUES),
                "prefix_pixels_exact": [bool(value) for value in prefix_pixels_exact],
                "prefix_state_gap": prefix_state_gap.tolist(),
            },
        }

    prefix_pixels_exact = [bool(prefix_pixels_exact[k] and all(
        render_match[key][k] for key in ("x0", "x1", "x2")
    )) for k in range(K)]
    prefix_state_gap = np.maximum(prefix_state_gap, np.maximum(query_residuals, factor_query_gap_max))

    # The frozen legacy goal stays fixed when future candidate actions are safety-scaled.
    goal_pixels = legacy_goal_pixels
    goal_state = legacy_goal_state.astype(np.float64)
    costs = physical_cost(future_states.astype(np.float64), goal_state)
    step_costs = physical_cost(future_step_states.astype(np.float64), goal_state)
    min_distance = step_costs.min(axis=-1)
    min_step = (step_costs.argmin(axis=-1) + 1).astype(np.int16)
    first_entry = np.zeros((K, U), dtype=np.int16)
    for k in range(K):
        for c in range(U):
            for raw in range(FUTURE_RAW_STEPS):
                if success_predicate(
                    future_step_states[k, c, raw].astype(np.float64), goal_state, symmetry
                ):
                    first_entry[k, c] = raw + 1
                    break

    future_frame_sha256 = np.empty((K, U, FUTURE_BLOCKS), dtype="U64")
    cross_condition_pixels_exact = np.zeros((U, FUTURE_BLOCKS), dtype=np.bool_)
    for mode_index in range(K):
        for unique_index in range(U):
            for block in range(FUTURE_BLOCKS):
                future_frame_sha256[mode_index, unique_index, block] = hashlib.sha256(
                    future_pixels[mode_index, unique_index, block].tobytes()
                ).hexdigest()
    for unique_index in range(U):
        for block in range(FUTURE_BLOCKS):
            cross_condition_pixels_exact[unique_index, block] = bool(np.array_equal(
                future_pixels[0, unique_index, block], future_pixels[1, unique_index, block]
            ))
    condition_gap_world_px = np.zeros((U, FUTURE_RAW_STEPS), dtype=np.float32)
    for unique_index in range(U):
        for raw in range(FUTURE_RAW_STEPS):
            condition_gap_world_px[unique_index, raw] = _snapshot_world_pixel_gap(
                future_step_states[0, unique_index, raw],
                future_step_states[1, unique_index, raw],
            )
    condition_gap_above_threshold = condition_gap_world_px > RENDER_EQUIVALENT_WORLD_GAP_THRESHOLD
    future_block_speed_norm = np.linalg.norm(future_step_states[..., 8:10], axis=-1).astype(np.float32)

    unsafe_slots = np.flatnonzero(~candidate_safe)
    if len(unsafe_slots):
        failures = ["no safe factor for original candidate slots " + ",".join(
            str(int(value)) for value in unsafe_slots
        )]
    else:
        failures = []
    if not bool(bodies_inside.all()):
        failures.append("factor-0 fallback or selected unique trajectory leaves [2,510] full-shape bounds")

    scene_name = f"{job['pair_id']}.npz"
    scene_path = Path(job["output"]) / scene_name
    terminal_costs = costs[..., -1]
    unique_terminal_costs = terminal_costs
    best_per_condition = np.argmin(unique_terminal_costs, axis=1)
    common_mean = int(np.argmin(unique_terminal_costs.mean(axis=0)))
    common_worst = int(np.argmin(unique_terminal_costs.max(axis=0)))
    oracle_costs = unique_terminal_costs[np.arange(K), best_per_condition]
    trajectory_receipt = {
        "replayed_original_history": True,
        "query_state_injected": False,
        "all_candidates_continuous": True,
        "future_raw_steps": FUTURE_RAW_STEPS,
        "factor_search_complete": True,
        "factor_search_factors": list(FACTOR_VALUES),
        "prefix_pixels_exact": [bool(value) for value in prefix_pixels_exact],
        "prefix_state_gap": [float(value) for value in prefix_state_gap],
    }
    receipt = {
        "index": job["index"],
        "scene_id": job["pair_id"],
        "status": "failed" if failures else "passed",
        "failures": failures,
        "path": scene_name,
        "sha256": None,
        "query_residuals": query_residuals.tolist(),
        "query_pair_state_gap": float(
            np.max(np.abs(_snapshot_delta(query_states[0], query_states[1])))
        ),
        "query_tolerance": query_tolerance,
        "native_row15_state_gap": native_source_state_gap.tolist(),
        "native_row15_pixels_bitwise_exact": [bool(v) for v in native_source_frame_equal],
        "selected_native_slot_first_step_state_gap": native_future_state_gap.tolist(),
        "selected_native_slot_first_step_pixels_exact": [bool(v) for v in native_future_frame_equal],
        "render_match": {k: [bool(x) for x in v] for k, v in render_match.items()},
        "history_contact_steps": history_contacts.sum(axis=1).tolist(),
        "future_contact_points_by_unique_action": future_contacts.sum(axis=-1).tolist(),
        "future_contact_raw_steps_by_unique_action": (future_contacts > 0).sum(axis=-1).tolist(),
        "future_block_speed_initial_final_by_unique_action": future_block_speed_norm[:, :, [0, -1]].tolist(),
        "bodies_inside_playfield_all_rollouts": bool(bodies_inside.all()),
        "action_max_abs": float(np.max(np.abs(candidates))),
        "candidate_count": int(U),
        "candidate_slot_count": int(C),
        "candidate_parent_index": list(range(C)),
        "candidate_factor": candidate_factor.tolist(),
        "candidate_safe": candidate_safe.tolist(),
        "factor_values": list(FACTOR_VALUES),
        "factor_search_safe": factor_search_safe.tolist(),
        "factor_search_complete": bool(np.all([
            all(detail is not None for detail in row) for row in factor_search_details
        ])),
        "factor_search_factors": list(FACTOR_VALUES),
        "candidate_slot_to_unique": candidate_unique_index.tolist(),
        "candidate_slot_weight": candidate_slot_weight.tolist(),
        "unique_action_sha256": unique_action_sha256,
        "cross_condition_pixels_exact": cross_condition_pixels_exact.tolist(),
        "cross_condition_frame_sha256": future_frame_sha256.tolist(),
        "condition_gap_world_px_equiv_max_by_unique_action": condition_gap_world_px.max(axis=-1).tolist(),
        "condition_gap_above_2_render_px_by_unique_action": condition_gap_above_threshold.sum(axis=-1).tolist(),
        "trajectory_receipt": trajectory_receipt,
        "goal_source": {
            **job.get("legacy_goal_source", {}),
            "source": "legacy panel goal, copied byte-for-byte",
            "legacy_panel_scene_id": job["pair_id"],
            "legacy_panel_manifest_sha256": job["legacy_panel_manifest_sha256"],
            "goal_state": goal_state.tolist(),
            "definition": "frozen per-scene legacy goal; independent of geometry-safe candidate scaling",
        },
        "decision_audit": {
            "terminal_cost_units": "px_equivalent(agent_xy, block_xy, 40*wrapped_block_angle)",
            "terminal_costs_by_unique_action": terminal_costs.tolist(),
            "best_unique_action_per_condition": best_per_condition.tolist(),
            "best_unique_action_costs": oracle_costs.tolist(),
            "best_common_unique_action_by_mean": common_mean,
            "best_common_unique_action_by_worst_case": common_worst,
            "mean_gap_common_vs_oracle": float(
                unique_terminal_costs[:, common_mean].mean() - oracle_costs.mean()
            ),
            "worst_case_gap_common_vs_oracle": float(
                unique_terminal_costs[:, common_worst].max() - oracle_costs.max()
            ),
            "equal_weighting": "each exact float32 unique action contributes once",
            "note": "fixed-bank diagnostic only; no model score selects scenes, actions, or factors",
        },
        "bootstrap_cluster": job.get("bootstrap_cluster"),
        "source_group": job.get("source_group"),
    }
    np.savez_compressed(
        scene_path,
        pair_id=np.asarray(job["pair_id"]),
        task=np.asarray(task),
        history_pixels=history_pixels,
        context_actions=context_actions,
        candidate_actions=unique_candidates.reshape(U, FUTURE_BLOCKS, RAW_STEPS_PER_BLOCK, 2),
        candidate_names=np.asarray(unique_candidate_names),
        candidate_slot_names=np.asarray(CANDIDATE_NAMES),
        candidate_slot_actions=candidates.astype(np.float32),
        candidate_parent_index=np.arange(C, dtype=np.int64),
        candidate_factor=candidate_factor.astype(np.float32),
        candidate_safe=candidate_safe,
        unique_candidate_safe=unique_candidate_safe,
        factor_values=np.asarray(FACTOR_VALUES, dtype=np.float32),
        factor_search_safe=factor_search_safe,
        candidate_unique_index=candidate_unique_index,
        candidate_slot_to_unique=candidate_unique_index,
        candidate_slot_weight=candidate_slot_weight,
        unique_action_count=np.asarray(U, dtype=np.int64),
        unique_action_sha256=np.asarray(unique_action_sha256),
        candidate_actions_shared_across_conditions=np.asarray(True),
        future_actions_shared_across_conditions=np.asarray(True),
        native_candidate_index=np.asarray(native_index, dtype=np.int64),
        native_candidate_slot_index=np.asarray(native_slot_index, dtype=np.int64),
        future_pixels=future_pixels,
        future_states=future_states,
        future_step_states=future_step_states,
        future_geometry_bounds=future_geometry_bounds,
        future_block_speed_norm=future_block_speed_norm,
        goal_pixels=goal_pixels,
        goal_state=goal_state.astype(np.float32),
        physical_cost=costs.astype(np.float64),
        conditions=np.asarray(job["condition_names"]),
        condition_values=np.asarray(job["condition_values"], dtype=np.float64),
        physical_steps=np.asarray(PHYSICAL_STEPS, dtype=np.int64),
        initial_state=initial_states,
        query_state=query_states,
        query_reference_residual=query_residuals,
        history_contact_counts=history_contacts,
        future_contact_counts=future_contacts,
        bodies_inside_playfield=bodies_inside,
        min_goal_distance=min_distance,
        min_goal_distance_step=min_step,
        first_entry_step=first_entry,
        native_row15_state_gap=native_source_state_gap,
        native_row15_pixels_bitwise_exact=np.asarray(native_source_frame_equal, dtype=np.bool_),
        selected_native_slot_first_step_state_gap=native_future_state_gap,
        prefix_pixels_exact=np.asarray(trajectory_receipt["prefix_pixels_exact"], dtype=np.bool_),
        prefix_state_gap=np.asarray(trajectory_receipt["prefix_state_gap"], dtype=np.float64),
        cross_condition_pixels_exact=cross_condition_pixels_exact,
        future_frame_sha256=future_frame_sha256,
        condition_gap_world_px_equiv=condition_gap_world_px,
        condition_gap_above_2_render_px=condition_gap_above_threshold,
        geometry_bounds_limits=np.asarray([GEOMETRY_MIN, GEOMETRY_MAX], dtype=np.float64),
        render_equivalent_world_gap_threshold=np.asarray(
            RENDER_EQUIVALENT_WORLD_GAP_THRESHOLD, dtype=np.float64
        ),
    )
    receipt["sha256"] = file_sha256(scene_path)
    return receipt


def _pair_reference(task, arrays, index, template):
    """Freeze the benchmark rows and the candidate bank for one pair."""
    from contextworld.evaluation.pusht_contact_friction_h3 import ContactFrictionTemplate
    from contextworld.evaluation.pusht_motion_damping_h3 import MotionDampingTemplate

    pair_id = arrays.pair_ids[index]
    if task == "contact_friction":
        frames = np.stack([arrays.low_pixels[index], arrays.high_pixels[index]])
        physics = np.stack(
            [arrays.low_physics_states[index], arrays.high_physics_states[index]]
        )
        template_obj = ContactFrictionTemplate(**template)
        history_blocks = arrays.raw_action_blocks[index][:2]
        if not np.array_equal(
            np.asarray(template_obj.history_actions, dtype=np.float32),
            history_blocks.reshape(HISTORY_RAW_STEPS, 2),
        ):
            raise RuntimeError(f"stored history actions differ from template for {pair_id}")
        if not np.array_equal(
            np.asarray(template_obj.query_actions, dtype=np.float32),
            arrays.raw_action_blocks[index][2],
        ):
            raise RuntimeError(f"stored query action differs from template for {pair_id}")
        native_query = np.asarray(template_obj.query_actions, dtype=np.float64)
        expected_query = template_obj.canonical_query_snapshot
    else:
        frames = np.stack(
            [arrays.faster_decay_pixels[index], arrays.no_extra_decay_pixels[index]]
        )
        physics = np.stack(
            [
                arrays.faster_decay_physics_states[index],
                arrays.no_extra_decay_physics_states[index],
            ]
        )
        template_obj = MotionDampingTemplate(**template)
        history_blocks = arrays.raw_action_blocks[index][:2]
        if not np.array_equal(
            np.asarray(template_obj.history_actions, dtype=np.float32),
            history_blocks.reshape(HISTORY_RAW_STEPS, 2),
        ):
            raise RuntimeError(f"stored history actions differ from template for {pair_id}")
        if np.any(np.asarray(template_obj.query_actions) != 0.0) or np.any(
            np.asarray(arrays.raw_action_blocks[index][2]) != 0.0
        ):
            raise RuntimeError(f"frozen motion-damping query is not the zero coast for {pair_id}")
        expected_query = template_obj.expected_natural_query_snapshot
        native_query = damping_probe(expected_query)
    return {
        "stored_frames": frames,
        "stored_physics": physics,
        "history_blocks": history_blocks,
        "candidate_actions": build_candidate_bank(native_query),
        "native_query_actions": np.asarray(template_obj.query_actions, dtype=np.float32),
        "expected_query_snapshot": expected_query,
        "visible_shape_id": int(template["visible_shape_id"]),
    }


def _load_legacy_panel_metadata(task: str, manifest_path: Path) -> tuple[dict, Path, dict]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing legacy panel manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != task:
        raise ValueError(f"legacy panel task mismatch: {manifest.get('task')} != {task}")
    panel_root = manifest_path.parent
    receipt_rows = manifest.get("scene_receipts", [])
    if isinstance(receipt_rows, dict):
        receipt_by_id = receipt_rows
    else:
        receipt_by_id = {
            str(row.get("scene_id", row.get("pair_id", row.get("query_id", "")))): row
            for row in receipt_rows if isinstance(row, dict)
        }
    cluster_path = panel_root.parent / "bootstrap_clusters.json"
    cluster_map = {}
    if cluster_path.is_file():
        cluster_file = json.loads(cluster_path.read_text(encoding="utf-8"))
        task_clusters = cluster_file.get(task, {}) if isinstance(cluster_file, dict) else {}
        if isinstance(task_clusters, dict):
            cluster_map = {str(key): str(value) for key, value in task_clusters.items()}

    records = {}
    for row in manifest.get("scenes", []):
        if not isinstance(row, dict):
            continue
        scene_id = str(row.get("scene_id", row.get("pair_id", row.get("query_id", ""))))
        if not scene_id:
            continue
        relative = Path(str(row.get("path") or f"{scene_id}.npz"))
        path = relative if relative.is_absolute() else panel_root / relative
        receipt = receipt_by_id.get(scene_id, {})
        record = {"path": str(path), "scene_id": scene_id}
        if "bootstrap_cluster" in row:
            record["bootstrap_cluster"] = str(row["bootstrap_cluster"])
        elif scene_id in cluster_map:
            record["bootstrap_cluster"] = cluster_map[scene_id]
        if "source_group" in row:
            record["source_group"] = str(row["source_group"])
        if isinstance(receipt, dict) and isinstance(receipt.get("goal_source"), dict):
            record["goal_source"] = receipt["goal_source"]
        if not path.is_file():
            raise FileNotFoundError(f"missing legacy scene file: {path}")
        records[scene_id] = record
    if not records:
        raise ValueError(f"legacy manifest has no scene records: {manifest_path}")
    return manifest, manifest_path, records


def run_task(task, output, limit, stable_repo, stable_ref, bundle, artifacts_root, workers, legacy_panel_manifest) -> dict:
    sys.path[:0] = [str(ROOT_REPO)]
    sys.modules.setdefault("flash_attn", None)
    from contextworld.benchmarks import bundle_development as bd
    from contextworld.benchmarks.contact_friction_icl_data import (
        DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG,
        _read_lance_pairs as read_contact_pairs,
        load_contact_friction_icl_release,
    )
    from contextworld.benchmarks.motion_damping_icl_data import (
        DEFAULT_MOTION_DAMPING_RELEASE_CONFIG,
        _read_lance_pairs as read_damping_pairs,
        load_motion_damping_icl_release,
    )

    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite existing panel: {protocol_path}")

    if task == "contact_friction":
        release = load_contact_friction_icl_release(DEFAULT_CONTACT_FRICTION_RELEASE_CONFIG)
        read_pairs = read_contact_pairs
    else:
        release = load_motion_damping_icl_release(DEFAULT_MOTION_DAMPING_RELEASE_CONFIG)
        read_pairs = read_damping_pairs
    release_config_path = ROOT_REPO / RELEASE_CONFIGS[task]

    payload = bd.resolve_development_payload(bundle, task=task)
    normalization_mean, normalization_std = bd.development_action_normalization(payload)
    table = payload.members[0]
    table_sha = directory_sha256(table)
    if table_sha != EXPECTED_TABLE_SHA256[task]:
        raise RuntimeError(
            f"Development table hash mismatch for {task}: {table_sha} != "
            f"{EXPECTED_TABLE_SHA256[task]}"
        )
    arrays = read_pairs(table, expected_pairs=256, expected_split=DEVELOPMENT_SPLIT)

    release_manifest_path = artifacts_root / RELEASE_MANIFEST_RELATIVE[task]
    release_manifest_sha = file_sha256(release_manifest_path)
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    pair_rows = release_manifest["splits"][DEVELOPMENT_SPLIT]["pairs"]
    pair_ids = tuple(arrays.pair_ids[:limit])
    if tuple(row["template"]["template_id"] for row in pair_rows[:limit]) != pair_ids:
        raise RuntimeError("release manifest pair order differs from the Development table")

    legacy_manifest, legacy_manifest_path, legacy_records = _load_legacy_panel_metadata(
        task, legacy_panel_manifest
    )
    missing_legacy = [pair_id for pair_id in pair_ids if pair_id not in legacy_records]
    if missing_legacy:
        raise RuntimeError(f"legacy panel is missing {len(missing_legacy)} required scenes: {missing_legacy[:5]}")
    legacy_manifest_sha = file_sha256(legacy_manifest_path)

    tree_info = extract_stable_tree(
        stable_repo, stable_ref, output / f".swm_{stable_ref[:12]}"
    )
    if tree_info["tree"] not in sys.path:
        sys.path.insert(0, tree_info["tree"])

    protocol = {
        "schema": "contextworld.pusht_visible_futures.v1",
        "task": task,
        "evaluation_split": "development",
        "development_split_internal": DEVELOPMENT_SPLIT,
        "scene_selection": (
            f"first {limit} frozen Development pairs in registered order; no "
            "model-score-based scene selection"
        ),
        "complete_development": limit == 256,
        "scene_ids": list(pair_ids),
        "history_tokens": HISTORY_TOKENS,
        "action_dimension": 2,
        "raw_steps_per_block": RAW_STEPS_PER_BLOCK,
        "future_blocks": FUTURE_BLOCKS,
        "physical_steps": PHYSICAL_STEPS,
        "candidate_slot_count": len(CANDIDATE_NAMES),
        "candidate_slot_names": list(CANDIDATE_NAMES),
        "native_candidate_slot_index": NATIVE_CANDIDATE_INDEX[task],
        "candidate_count": "per-scene exact unique float32 action count, at most 11",
        "candidate_rules": {
            "base_candidate_names": list(CANDIDATE_NAMES),
            "amplitudes": list(AMPLITUDES),
            "amplitude_family": "native query/probe tiled over five blocks, each base action clipped to [-1,1]",
            "truncation_family": "native blocks 1..m then zero blocks, m in 1..4",
            "counteraction_family": "native block 1 then elementwise -native clipped to [-1,1] for blocks 2..5",
            "native_for_contact_friction": "pair's frozen shared five-step query action",
            "native_for_motion_damping": "preregistered push probe from the shared query state; original coast, truncation and counteraction slots retained",
            "factor_values_descending": list(FACTOR_VALUES),
            "factor_application": "multiply each already-clipped float32 base sequence by one scalar factor",
            "factor_selection": "per original slot/source, largest factor whose full shape bounds stay in [2,510] for both conditions and all 25 raw steps",
            "factor_search_complete": "all seven factors are independently simulated; each cell covers both hidden conditions and 25 raw steps",
            "no_safe_factor": "retain source and slot, use factor 0.0 fallback and mark candidate_safe=false",
            "unique_action_axis": "candidate_actions and every future_* array are exact float32 unique action sequences only",
            "slot_mapping": "11 original slots map to unique candidate_actions via candidate_slot_to_unique; candidate_slot_actions stores the exact per-slot float32 sequence",
            "unique_action_equality": "contiguous float32 action bytes, no tolerance",
            "candidate_weights": "each unique action has equal total weight via candidate_slot_weight metadata",
            "shared_actions": "the same candidate action is used for all hidden conditions",
            "chosen_before_model_inference": True,
        },
        "goal_rule": {
            "definition": "legacy per-scene goal pixels and state copied exactly; held fixed under future action scaling",
            "condition": CONDITION_NAMES[task][GOAL_CONDITION_INDEX],
            "candidate_slot_index": NATIVE_CANDIDATE_INDEX[task],
            "shared_across_conditions": True,
        },
        "condition_names": list(CONDITION_NAMES[task]),
        "condition_values": list(CONDITION_VALUES[task]),
        "hidden_parameter": (
            "pusher-block contact friction coefficient"
            if task == "contact_friction"
            else "pymunk space.damping"
        ),
        "metric": {
            "physical_cost": (
                "sqrt(|agent_xy error|^2 + |block_xy error|^2 + "
                "(40*wrapped_block_angle_error)^2)"
            ),
            "units": "px_equivalent",
            "angle_radius_px": ANGLE_RADIUS_PX,
            "success_predicate": (
                "benchmark PushT: position error < 20 px and angle error < pi/9 modulo shape "
                "symmetry (square pi/2 for motion_damping, full rotation for contact_friction)"
            ),
            "first_entry_step": (
                "first future raw step (1..25) satisfying the success predicate; 0 = never"
            ),
            "min_goal_distance": (
                "minimum px_equivalent goal distance over the 25 future raw steps "
                "(shortcut detector)"
            ),
            "contact_recording": "per-raw-step pusher-block contact points; contact does not filter scenes or factors",
            "motion_damping_free_decay": "per-raw-step contacts and block speed are diagnostics only",
            "same_pixel_counterexample": "SHA256 plus bytewise equality; no RGB tolerance",
            "condition_physics_gap": {
                "units": "512-canvas world-pixel equivalent from xy and block-angle chord",
                "two_render_pixel_threshold": RENDER_EQUIVALENT_WORLD_GAP_THRESHOLD,
                "used_as_gate": False,
            },
        },
        "replay_protocol": {
            "prefix_replay_per_candidate": True,
            "reset_at_query": False,
            "state_installations_after_x0": 0,
            "sanctioned_x0_installation": (
                "motion_damping installs its mode-specific 12-D snapshot at x0 only (single "
                "allowed installation, identical to the frozen release builder)"
            ),
            "render_perturbation": (
                "PushT render() draws without stepping physics; the interleaved render ordering "
                "matches the frozen release builders; bitwise x0/x1/x2 prefix equality and a separate "
                "standalone five-step native row15 replay are recorded"
            ),
            "gates": {
                "history_frames_bitwise_exact": True,
                "standalone_native_row15_pixels_bitwise_exact": True,
                "scaled_native_slot_must_match_row15": False,
                "native_row15_state_tolerance": NATIVE_FUTURE_STATE_TOLERANCE,
                "query_residual_tolerance": QUERY_TOLERANCES[task],
                "replayed_query_pair_gap_tolerance": 1e-9,
            },
        },
        "geometry": {
            "canvas": [512, 512],
            "safe_bounds_inclusive": [GEOMETRY_MIN, GEOMETRY_MAX],
            "full_shapes": "all agent/block Pymunk shapes: Circle center +/- radius; Poly world vertices expanded by shape.radius; Segment world endpoints expanded by radius",
            "future_geometry_bounds_shape": [2, "U", FUTURE_RAW_STEPS, 2, 4],
            "future_geometry_bounds_axes": ["condition", "unique_action", "raw_step", "object(agent,block)", "xmin_ymin_xmax_ymax"],
            "checks_per_factor": "both hidden conditions x all 25 raw steps",
        },
        "normalization": {
            "mean": list(normalization_mean),
            "std": list(normalization_std),
        },
        "rendering": {
            "resolution": RESOLUTION,
            "jpeg_quality": JPEG_QUALITY,
            "codec": "contextworld.synthesis.lance.encode_frame + benchmark _decode_rgb (jpeg q95)",
            "history_pixels_source": (
                "frozen Development lance table (byte-identical benchmark inputs)"
            ),
        },
        "stable_worldmodel": {
            "repo": str(stable_repo),
            "ref": stable_ref,
            "extracted_tree": tree_info["tree"],
            "env_py_sha256": tree_info["env_py_sha256"],
            "note": (
                "the frozen motion-damping Development table was rendered when the PushT square "
                "block scale was 40 (release-pinned ref 875e607); the current checkout uses 30 "
                "and does not reproduce those pixels bitwise. contact_friction uses the T shape "
                "(scale 30 in both refs) and is reproduced by 6ab823fd. pins verified bitwise "
                "against the frozen rows."
            ),
        },
        "sources": {
            "bundle_root": str(bundle),
            "bundle_manifest_sha256": payload.manifest_sha256,
            "bundle_task_registry_sha256": payload.task_registry_sha256,
            "development_member": [m.name for m in payload.members],
            "development_table_sha256": table_sha,
            "expected_table_sha256": EXPECTED_TABLE_SHA256[task],
            "release_config_path": str(release_config_path),
            "release_config_sha256": file_sha256(release_config_path),
            "release_id": release["release_id"],
            "synthesis_manifest_path": str(release_manifest_path),
            "synthesis_manifest_sha256": release_manifest_sha,
            "release_protocol": release_manifest.get("protocol"),
            "legacy_panel_manifest_path": str(legacy_manifest_path),
            "legacy_panel_manifest_sha256": legacy_manifest_sha,
            "legacy_goal_source": "goal_pixels and goal_state are copied per scene from the legacy panel",
        },
        "no_training": True,
        "no_learned_checkpoint_runs": True,
        "attempted_failures_retained": True,
        "source_indices": list(range(limit)),
        "requested_source_count": limit,
        "action_limits": [-1.0, 1.0],
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": file_sha256(Path(__file__).resolve()),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for index, pair_id in enumerate(pair_ids):
        reference = _pair_reference(task, arrays, index, pair_rows[index]["template"])
        legacy_record = legacy_records[pair_id]
        jobs.append(
            {
                "index": index,
                "pair_id": pair_id,
                "task": task,
                "template": pair_rows[index]["template"],
                "output": str(output),
                "stable_tree": tree_info["tree"],
                "condition_names": list(CONDITION_NAMES[task]),
                "condition_values": list(CONDITION_VALUES[task]),
                "query_tolerance": QUERY_TOLERANCES[task],
                "native_candidate_index": NATIVE_CANDIDATE_INDEX[task],
                "legacy_panel_path": legacy_record["path"],
                "legacy_panel_manifest_sha256": legacy_manifest_sha,
                "legacy_goal_source": legacy_record.get("goal_source", {}),
                "bootstrap_cluster": legacy_record.get("bootstrap_cluster"),
                "source_group": legacy_record.get("source_group"),
                **reference,
            }
        )

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    receipts = []
    if workers <= 1:
        for job in jobs:
            receipts.append(build_scene(job))
            print(f"built {len(receipts)}/{len(jobs)} {task} scenes", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            for receipt in pool.map(build_scene, jobs):
                receipts.append(receipt)
                print(f"built {len(receipts)}/{len(jobs)} {task} scenes", flush=True)

    saved = [r for r in receipts if r.get("path")]
    passed = [r for r in receipts if r["status"] == "passed"]
    failed = [r for r in receipts if r["status"] != "passed"]
    manifest = dict(protocol)
    manifest["scenes"] = []
    for r in receipts:
        row = {"scene_id": r["scene_id"], "path": r.get("path"), "sha256": r.get("sha256"), "status": r["status"]}
        if r.get("bootstrap_cluster") is not None:
            row["bootstrap_cluster"] = r["bootstrap_cluster"]
        if r.get("source_group") is not None:
            row["source_group"] = r["source_group"]
        manifest["scenes"].append(row)
    manifest["scene_receipts"] = receipts
    manifest["source_coverage"] = {
        "requested": limit,
        "attempted": len(receipts),
        "produced": len(saved),
        "passed": len(passed),
        "failed_retained": len(failed),
        "unique_query_ids": len({r["scene_id"] for r in receipts}) == len(receipts),
        "source_index_range": [0, limit - 1],
    }
    manifest["summary"] = {
        "attempted": len(receipts),
        "produced": len(saved),
        "passed": len(passed),
        "failed": len(failed),
        "failed_scene_ids": [r["scene_id"] for r in failed],
        "unsafe_candidate_slots": {
            r["scene_id"]: [i for i, safe in enumerate(r.get("candidate_safe", [])) if not safe]
            for r in receipts if any(not safe for safe in r.get("candidate_safe", []))
        },
        "elapsed_seconds": time.monotonic() - started,
        "native_replay_all_bitwise": bool(
            receipts and all(all(r.get("native_row15_pixels_bitwise_exact", [])) for r in receipts)
        ),
        "prefix_all_bitwise": bool(
            receipts and all(all(r.get("trajectory_receipt", {}).get("prefix_pixels_exact", [])) for r in receipts)
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--stable-repo", type=Path, required=True)
    parser.add_argument("--stable-ref", type=str, default=None)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    legacy = parser.add_mutually_exclusive_group(required=True)
    legacy.add_argument("--legacy-panel-root", type=Path)
    legacy.add_argument("--legacy-panel-manifest", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 256:
        raise ValueError("--limit must be within 1..256")
    stable_ref = args.stable_ref or DEFAULT_STABLE_REFS[args.task]
    legacy_panel_manifest = (
        args.legacy_panel_root / "manifest.json"
        if args.legacy_panel_root is not None
        else args.legacy_panel_manifest
    )
    manifest = run_task(
        task=args.task,
        output=args.output,
        limit=args.limit,
        stable_repo=args.stable_repo,
        stable_ref=stable_ref,
        bundle=args.bundle,
        artifacts_root=args.artifacts_root,
        workers=args.workers,
        legacy_panel_manifest=legacy_panel_manifest,
    )
    print(json.dumps({"task": args.task, "output": str(args.output), "summary": manifest["summary"]}, indent=2), flush=True)
    if manifest["summary"]["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
