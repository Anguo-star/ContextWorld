#!/usr/bin/env python3
"""Cube v4r1 Test coherent panel functions from the registered 256-pair source.

Source Lance replay differences are measured and retained; this module does not
claim byte-exact reproduction of the original generation runtime.  A/B current
observations, simulator states, candidate prefixes, and future trajectories
are checked within the fixed coherent rendering runtime.
"""
from __future__ import annotations
import hashlib
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import numpy as np

DEFAULT_CW_REPO = Path("/opt/huawei/explorer-env/dataset/ag_data/code/ContextWorld")
DEFAULT_STABLE_REPO = Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/context_world/upstream/stable-worldmodel-875e607fc08aa72e")
DEFAULT_STABLE_REF = "875e607fc08aa72eacb94d5d178127804134cc06"
DEFAULT_BUNDLE = Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/ContextWorld-v1-full")
DEFAULT_CUBE_H5 = Path("/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/ogbench/cube_single_expert.h5")
CUBE_DATA = "artifacts/synthesis/cube_gripper_carry_rule_h3_v4r1_release_projection_v1/validation.lance"
CUBE_SPLIT = "validation"
CUBE_CATALOG_SEED = 2026081401
CUBE_FORMAL_OFFSET = 3_000_000
JPEG_QUALITY = 95
RAW_STEPS_PER_BLOCK = 5
FUTURE_BLOCKS = 5
PHYSICAL_STEPS = np.asarray([5, 10, 15, 20, 25], dtype=np.int64)
ACTION_BOUNDS = (-1.0, 1.0)
CANDIDATE_AMPLITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, -1.0)
STATE_EXACT_TOL = 1e-6
STATE_TOL = 1e-4
PIXEL_TOL = 16
FIRST_ENTRY_THRESHOLD_M = 0.05

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def jpeg_decode(value: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)

def jpeg_roundtrip(raw: np.ndarray) -> np.ndarray:
    from PIL import Image

    buffer = BytesIO()
    Image.fromarray(np.asarray(raw, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=JPEG_QUALITY
    )
    return jpeg_decode(buffer.getvalue())

def pixel_audit(raw_render: np.ndarray, stored_bytes: bytes) -> dict:
    """Compare a raw render against the frozen JPEG bytes.

    The frozen sources store JPEG quality 95 encodings of raw renders.  If the
    render pipeline is bit-identical, JPEG re-encoding the render reproduces
    the stored pixels exactly.  Otherwise the decoded difference is reported
    and judged against PIXEL_TOL.
    """

    stored = jpeg_decode(stored_bytes)
    reencoded = jpeg_roundtrip(raw_render)
    exact = bool(np.array_equal(reencoded, stored))
    decoded_diff = np.abs(raw_render.astype(np.int16) - stored.astype(np.int16))
    return {
        "jpeg_reencode_exact": exact,
        "decoded_max_abs_diff": int(decoded_diff.max()),
        "decoded_changed_values": int(np.count_nonzero(decoded_diff)),
        "passed": bool(exact or int(decoded_diff.max()) <= PIXEL_TOL),
    }

def state_audit(observed: np.ndarray, stored: np.ndarray) -> dict:
    observed = np.asarray(observed, dtype=np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    diff = np.abs(observed - stored)
    return {
        "max_abs_diff": float(diff.max()),
        "max_abs_diff_within_state_tol": float(diff.max()) <= STATE_TOL,
        "bitwise_float32_equal": bool(
            np.array_equal(
                observed.astype(np.float32), stored.astype(np.float32)
            )
        ),
        "passed": bool(diff.max() <= STATE_TOL),
    }

def clip_actions(values: np.ndarray) -> tuple[np.ndarray, int]:
    clipped = np.clip(values, ACTION_BOUNDS[0], ACTION_BOUNDS[1])
    count = int(np.count_nonzero(clipped != values))
    return clipped.astype(np.float64), count

def build_candidate_bank(native_query: np.ndarray) -> tuple[list, list]:
    """Return (bank, notes).

    Default bank per protocol, adapted for a one-block native query:
      - the native query repeated for all five blocks at amplitudes
        0, 0.5, 1, 1.5, 2, -1 (each block clipped to the native bounds),
      - native 1/2/3/4 blocks followed by zero blocks,
      - native 1 block then counteraction (-native, clipped).
    All candidates share the raw action space and bounds of the native query.
    """

    native = np.asarray(native_query, dtype=np.float64)
    if native.shape != (RAW_STEPS_PER_BLOCK, native.shape[-1]):
        raise ValueError(f"unexpected native query shape {native.shape}")
    bank: list[dict] = []
    notes: list[str] = []
    for amplitude in CANDIDATE_AMPLITUDES:
        block, clipped = clip_actions(amplitude * native)
        plan = np.repeat(block[None], FUTURE_BLOCKS, axis=0)
        bank.append(
            {
                "candidate_id": f"amp_{amplitude:+.1f}_repeat",
                "actions": plan,
                "rule": f"native query actions * {amplitude}, repeated 5 blocks",
                "clipped_values": clipped,
            }
        )
    zero = np.zeros_like(native)
    for persistence in (1, 2, 3, 4):
        plan = np.concatenate(
            [np.repeat(native[None], persistence, axis=0)]
            + [zero[None]] * (FUTURE_BLOCKS - persistence)
        )
        bank.append(
            {
                "candidate_id": f"native_{persistence}_then_zero",
                "actions": plan.astype(np.float64),
                "rule": (
                    f"native query for {persistence} block(s) then zero blocks"
                ),
                "clipped_values": 0,
            }
        )
    counter, clipped = clip_actions(-native)
    plan = np.concatenate([native[None], np.repeat(counter[None], 4, axis=0)])
    bank.append(
        {
            "candidate_id": "native_1_then_counteract",
            "actions": plan.astype(np.float64),
            "rule": "native query block then -native for four blocks",
            "clipped_values": clipped,
        }
    )
    notes.append(
        "native query is one action block for both assigned tasks, so the "
        "'native 1/2/3/4 blocks followed by zero' family is realised by "
        "repeating the single native block; amplitude candidates repeat the "
        "scaled native block on all five future blocks"
    )
    return bank, notes

def extend_bank_with_grip(
    bank: list[dict], native_query: np.ndarray, gripper_column: int = 4
) -> tuple[list[dict], list[str]]:
    """Cube carry task: add discrete grip alternatives to the shared plan.

    The v4r1 carry rule couples the cube transition force to the normalized z
    command (action column 2) only; the gripper column (column 4) is the
    physical gripper command.  Grip alternatives therefore change the effector
    geometry while holding the z command fixed, which probes whether the carry
    response is specific to the rule-relevant channel.
    """

    native = np.asarray(native_query, dtype=np.float64)
    extra: list[dict] = []
    for name, grip in (
        ("grip_open_m1_repeat", -1.0),
        ("grip_neutral_0_repeat", 0.0),
        ("grip_close_p1_repeat", 1.0),
    ):
        block = native.copy()
        block[:, gripper_column] = np.float32(grip)
        plan = np.repeat(block[None], FUTURE_BLOCKS, axis=0)
        extra.append(
            {
                "candidate_id": name,
                "actions": plan,
                "rule": (
                    f"native query with gripper column {gripper_column} "
                    f"replaced by {grip:+.1f}, repeated 5 blocks"
                ),
                "clipped_values": 0,
            }
        )
    notes = [
        "cube carry adaptation: three discrete grip alternatives (open -1.0, "
        "neutral 0.0, close +1.0) replace the gripper command of the native "
        "query; the carry transition force depends on the z command only, so "
        "these isolate gripper-command specificity"
    ]
    return bank + extra, notes

def load_cube_scenes(bundle: Path, limit: int, cube_h5: Path) -> list[dict]:
    import h5py
    import lance

    path = bundle / CUBE_DATA
    columns = [
        "episode_idx",
        "model_step_idx",
        "pixels",
        "action_block",
        "physical_state",
        "hidden_grasp_enabled",
        "pair_id",
        "hidden_mode",
        "split",
        "catalog_index",
        "source_row",
        "source_episode",
        "source_step",
        "action_anchor_id",
        "action_profile_id",
    ]
    table = lance.dataset(str(path)).to_table(columns=columns)
    n = table.num_rows
    pair_ids = table["pair_id"].to_pylist()
    modes = table["hidden_mode"].to_pylist()
    splits = table["split"].to_pylist()
    episode = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step = np.asarray(table["model_step_idx"].to_numpy(), dtype=np.int64)
    catalog = np.asarray(table["catalog_index"].to_numpy(), dtype=np.int64)
    source_row = np.asarray(table["source_row"].to_numpy(), dtype=np.int64)
    source_episode = np.asarray(
        table["source_episode"].to_numpy(), dtype=np.int64
    )
    source_step = np.asarray(table["source_step"].to_numpy(), dtype=np.int64)
    anchor = table["action_anchor_id"].to_pylist()
    profile_id = table["action_profile_id"].to_pylist()
    hidden = np.asarray(
        table["hidden_grasp_enabled"].to_pylist(), dtype=np.float32
    ).reshape(-1)
    pixel_bytes = table["pixels"].to_pylist()
    action_block = np.asarray(
        table["action_block"].to_pylist(), dtype=np.float32
    ).reshape(n, RAW_STEPS_PER_BLOCK, 5)
    physical_state = np.asarray(
        table["physical_state"].to_pylist(), dtype=np.float32
    ).reshape(n, 7)

    unique_pairs = sorted(set(pair_ids))
    if len(unique_pairs) < limit:
        raise RuntimeError(
            f"requested limit {limit} exceeds {len(unique_pairs)} pairs"
        )
    h5_cache: dict[int, tuple[tuple, tuple]] = {}
    scenes = []
    for pair_id in unique_pairs[:limit]:
        pair_rows = np.flatnonzero(np.asarray(pair_ids) == pair_id)
        episodes = {}
        local_index = None
        for mode in ("cannot_hold", "can_hold"):
            mode_rows = [int(r) for r in pair_rows if modes[r] == mode]
            mode_rows.sort(key=lambda r: step[r])
            if [step[r] for r in mode_rows] != [0, 1, 2, 3]:
                raise RuntimeError(f"{pair_id}/{mode}: incomplete 4-frame clip")
            if splits[mode_rows[0]] != CUBE_SPLIT:
                raise RuntimeError(f"{pair_id}: unexpected split")
            expected_hidden = 0.0 if mode == "cannot_hold" else 1.0
            if float(hidden[mode_rows[0]]) != expected_hidden:
                raise RuntimeError(f"{pair_id}/{mode}: unexpected hidden value")
            local_index = int(catalog[mode_rows[0]]) - CUBE_FORMAL_OFFSET
            episodes[mode] = {
                "pixel_bytes": [pixel_bytes[r] for r in mode_rows],
                "blocks": np.stack(
                    [action_block[r] for r in mode_rows]
                ),
                "physical_state": np.stack(
                    [physical_state[r] for r in mode_rows]
                ),
                "anchor": anchor[mode_rows[0]],
                "profile_id": profile_id[mode_rows[0]],
            }
        row = int(source_row[pair_rows[0]])
        if row not in h5_cache:
            with h5py.File(cube_h5, "r", swmr=True) as handle:
                qpos = tuple(
                    float(v)
                    for v in np.asarray(handle["qpos"][row], dtype=np.float64)
                )
                control = tuple(
                    float(v)
                    for v in np.asarray(
                        handle["control"][row], dtype=np.float64
                    )
                )
            h5_cache[row] = (qpos, control)
        qpos, control = h5_cache[row]
        scenes.append(
            {
                "task": "cube_gripper_carry",
                "pair_id": pair_id,
                "local_index": local_index,
                "catalog_index": CUBE_FORMAL_OFFSET + local_index,
                "modes": ("cannot_hold", "can_hold"),
                "condition_values": {"cannot_hold": 0.0, "can_hold": 1.0},
                "source_row": row,
                "source_episode": int(source_episode[pair_rows[0]]),
                "source_step": int(source_step[pair_rows[0]]),
                "anchor": episodes["cannot_hold"]["anchor"],
                "expected_profile_id": episodes["cannot_hold"]["profile_id"],
                "qpos": qpos,
                "control": control,
                "episodes": episodes,
            }
        )
    return scenes

def cube_candidate(scene: dict):
    """Rebuild the registered v4r1 candidate for a Test scene."""

    from contextworld.evaluation.cube_grasp_rule_h3 import (
        CubeGraspRuleCandidate,
    )
    from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
        make_v4_candidate, V4_PROFILE_SPLIT_SEEDS,
    )
    V4_PROFILE_SPLIT_SEEDS[CUBE_SPLIT] = 2026081402  # frozen public profile seed

    local = int(scene["local_index"])
    rng = np.random.default_rng(
        np.random.SeedSequence([CUBE_CATALOG_SEED, local])
    )
    simulator_seed = int(rng.integers(0, 2**31 - 1))
    cube_color = tuple(float(v) for v in rng.uniform(0.18, 0.92, 3))
    target_position = (
        float(rng.uniform(0.32, 0.53)),
        float(rng.uniform(-0.24, 0.24)),
        0.02,
    )
    base = CubeGraspRuleCandidate(
        candidate_id=f"cube-carry-v4r1-public-{local:06d}",
        split=CUBE_SPLIT,
        catalog_index=CUBE_FORMAL_OFFSET + local,
        source_row=int(scene["source_row"]),
        source_episode=int(scene["source_episode"]),
        source_step=int(scene["source_step"]),
        simulator_seed=simulator_seed,
        task_id=1 + local % 5,
        qpos=tuple(float(v) for v in scene["qpos"]),
        control=tuple(float(v) for v in scene["control"]),
        cube_color=cube_color,
        target_position=target_position,
    )
    return make_v4_candidate(base)

def cube_rollout(sim,candidate,mode,plan):
    """Replay two real history blocks, then five candidate blocks; capture AFTER actions."""
    assert plan.shape==(7,5,5),plan.shape
    sim._install_candidate(candidate)
    pixels=[np.asarray(sim.base.render(),np.uint8).copy()];states=[sim._physical_state()];simstates=[sim._simulator_state()];contacts=[int(sim.base._data.ncon)]
    for block in plan:
        for action in block:
            sim._apply_transition_force(mode=mode,action_z=float(action[2]))
            sim.env.step(np.asarray(action,np.float32))
        pixels.append(np.asarray(sim.base.render(),np.uint8).copy());states.append(sim._physical_state());simstates.append(sim._simulator_state());contacts.append(int(sim.base._data.ncon))
    return dict(pixels=np.stack(pixels),physical_state=np.stack(states).astype(np.float64),simulator_state=np.stack(simstates),contacts=np.array(contacts))

def run_cube_scene(scene: dict, sim, audits: dict) -> dict:
    from contextworld.evaluation.cube_grasp_rule_h3_v4 import (
        action_blocks,
        action_profile_content_sha256,
    )

    candidate = cube_candidate(scene)
    print("DEBUG_RUNTIME", np.__version__, np.__file__, type(sim).__module__, candidate.simulator_seed, candidate.qpos[0], flush=True)
    if candidate.candidate_id != scene["pair_id"]:
        raise RuntimeError(
            f"candidate id mismatch {candidate.candidate_id} != "
            f"{scene['pair_id']}"
        )
    blocks = action_blocks(candidate.action_profile)
    if action_profile_content_sha256(blocks) != scene["expected_profile_id"]:
        raise RuntimeError(
            f"{scene['pair_id']}: rebuilt action profile id mismatch"
        )
    official_prefix = {mode: sim._run_mode(candidate, mode=mode, blocks=blocks) for mode in scene["modes"]}
    from contextworld.evaluation.cube_grasp_rule_h3 import validate_cube_grasp_rule_pair
    paired_audit = validate_cube_grasp_rule_pair(official_prefix["cannot_hold"], official_prefix["can_hold"])
    if not paired_audit["passed"]:
        raise RuntimeError(f"{scene['pair_id']}: original paired protocol failed")
    scene["candidate"] = asdict(candidate)
    scene["anchor_rebuilt"] = candidate.action_profile.action_anchor_id

    native_query = scene["episodes"]["cannot_hold"]["blocks"][2].copy()
    for mode in scene["modes"]:
        if not np.array_equal(
            native_query, scene["episodes"][mode]["blocks"][2]
        ):
            raise RuntimeError(f"{scene['pair_id']}: query blocks differ")
    bank, bank_notes = build_candidate_bank(native_query)
    bank, grip_notes = extend_bank_with_grip(bank, native_query)
    scene["bank"] = bank
    scene["bank_notes"] = bank_notes + grip_notes

    modes = scene["modes"]
    k_count = len(modes)
    c_count = len(bank)
    future_pixels = np.zeros(
        (k_count, c_count, FUTURE_BLOCKS, 224, 224, 3), dtype=np.uint8
    )
    future_states = np.zeros(
        (k_count, c_count, FUTURE_BLOCKS, 7), dtype=np.float64
    )
    contacts = np.zeros((k_count, c_count, FUTURE_BLOCKS), dtype=np.int64)
    scene_audits = {"prefix": {}, "native_future": {}}

    goal_episode = scene["episodes"][modes[0]]
    goal_pixels = jpeg_decode(goal_episode["pixel_bytes"][3])
    goal_state = goal_episode["physical_state"][3].astype(np.float64)
    goal_cube = goal_state[2:5]
    goal_eff = goal_state[[0, 1, 6]]

    zero_plan = np.zeros((FUTURE_BLOCKS, 5, 5), np.float32)
    prefix_states = {}
    coherent_histories = []
    for k, mode in enumerate(modes):
        episode = scene["episodes"][mode]
        audit_plan = np.concatenate([blocks[:2], zero_plan]).astype(
            np.float32
        )
        out = official_prefix[mode]
        prefix_states[mode] = out["physical_state"]
        print("PREFIXCHECK", scene["pair_id"], mode, float(np.max(np.abs(prefix_states[mode][2]-episode["physical_state"][2]))), flush=True)
        prefix_audit = []
        for frame in range(3):
            prefix_audit.append(
                {
                    "frame": frame,
                    "pixel": pixel_audit(
                        out["pixels"][frame], episode["pixel_bytes"][frame]
                    ),
                    "state": state_audit(
                        out["physical_state"][frame],
                        episode["physical_state"][frame],
                    ),
                }
            )
        scene_audits["prefix"][mode] = prefix_audit

        mode_history = None
        mode_raw_prefix = None
        mode_prefix_state = None
        for c, plan_spec in enumerate(bank):
            plan = np.concatenate(
                [blocks[:2], plan_spec["actions"].astype(np.float32)]
            )
            out = cube_rollout(sim, candidate, mode, plan)
            candidate_history = np.stack([jpeg_roundtrip(frame) for frame in out["pixels"][:3]])
            if mode_history is None:
                mode_history = candidate_history
                mode_raw_prefix = out["pixels"][:3].copy()
                mode_prefix_state = out["simulator_state"][:3].copy()
            elif not np.array_equal(out["pixels"][:3], mode_raw_prefix) or not np.array_equal(candidate_history, mode_history) or not np.array_equal(out["simulator_state"][:3], mode_prefix_state):
                raise RuntimeError(f"{scene['pair_id']}/{mode}: candidate raw/JPEG prefix pixels or simulator state differs")
            future_pixels[k, c] = out["pixels"][3:]
            future_states[k, c] = out["physical_state"][3:]
            contacts[k, c] = out["contacts"][3:]
        coherent_histories.append(mode_history)
        if k == 0:
            current_simulator_state = mode_prefix_state[2].copy()
            current_raw_pixel = mode_raw_prefix[2].copy()
            current_state_gap = 0.0
        else:
            current_state_gap = float(np.max(np.abs(mode_prefix_state[2] - current_simulator_state)))
            if current_state_gap > 1e-10 or not np.array_equal(mode_raw_prefix[2], current_raw_pixel):
                raise RuntimeError(f"{scene['pair_id']}: A/B current raw pixel or simulator state differs")

    native_index = next(
        i
        for i, plan in enumerate(bank)
        if plan["candidate_id"] == "amp_+1.0_repeat"
    )
    for k, mode in enumerate(modes):
        episode = scene["episodes"][mode]
        scene_audits["native_future"][mode] = {
            "pixel": pixel_audit(
                future_pixels[k, native_index, 0], episode["pixel_bytes"][3]
            ),
            "state": state_audit(
                future_states[k, native_index, 0],
                episode["physical_state"][3],
            ),
        }

    coherent_histories = np.stack(coherent_histories)
    if not np.array_equal(coherent_histories[0, -1], coherent_histories[1, -1]):
        raise RuntimeError(f"{scene['pair_id']}: mismatched current images")
    source_controlled = []
    for k, mode in enumerate(modes):
        src=scene['episodes'][mode]['physical_state']
        source_controlled.append(float(np.max(np.abs(prefix_states[mode][2, [0,1,2,3,4,6]]-src[2, [0,1,2,3,4,6]]))))
        source_controlled.append(float(np.max(np.abs(future_states[k,native_index,0, [0,1,2,3,4,6]]-src[3, [0,1,2,3,4,6]]))))
    scene_audits["paired_audit_passed"] = True
    scene_audits["current_full_simulator_state_max"] = current_state_gap
    scene_audits["current_raw_pixel_equal"] = True
    scene_audits["candidate_raw_prefix_equal"] = True
    scene_audits["source_exact_replay_passed"] = bool(max(source_controlled) <= 1e-6 and all(v["pixel"]["passed"] and v["state"]["passed"] for group in (scene_audits["prefix"], scene_audits["native_future"]) for records in group.values() for v in (records if isinstance(records,list) else [records])))
    scene_audits["source_controlled_position_gaps"] = source_controlled
    scene_audits['coherent_source_controlled_position_max']=max(source_controlled)
    scene_audits['coherent_source_history_pixel_max']=int(np.max(np.abs(coherent_histories.astype(np.int16)-np.stack([[jpeg_decode(scene['episodes'][mode]['pixel_bytes'][f]) for f in range(3)] for mode in modes]).astype(np.int16))))
    future_pixels=np.stack([jpeg_roundtrip(f) for f in future_pixels.reshape(-1,224,224,3)]).reshape(future_pixels.shape)
    goal_pixels = future_pixels[0,native_index,0].copy()
    goal_state = future_states[0,native_index,0].copy()
    goal_cube = goal_state[2:5]
    goal_eff = goal_state[[0,1,6]]
    cube_gap = np.linalg.norm(
        future_states[:, :, :, 2:5] - goal_cube[None, None, None, :], axis=-1
    )
    eff_gap = np.linalg.norm(
        future_states[:, :, :, [0, 1, 6]] - goal_eff[None, None, None, :],
        axis=-1,
    )
    physical_cost = cube_gap + eff_gap
    min_distance = physical_cost.min(axis=-1)
    first_entry = np.where(
        physical_cost <= FIRST_ENTRY_THRESHOLD_M,
        PHYSICAL_STEPS[None, None, :],
        np.iinfo(np.int64).max,
    ).min(axis=-1)
    first_entry = np.where(
        first_entry == np.iinfo(np.int64).max, -1, first_entry
    )

    scene["audits"] = scene_audits
    return {
        "history_pixels": coherent_histories,
        "source_controlled_position_max": np.asarray(max(source_controlled)),
        "source_exact_replay_passed": np.asarray(scene_audits["source_exact_replay_passed"]),
        "source_history_pixel_max": np.asarray(scene_audits["coherent_source_history_pixel_max"]),
        "context_actions": np.stack(
            [
                scene["episodes"][mode]["blocks"][:2]
                for mode in modes
            ]
        ).astype(np.float32),
        "candidate_actions": np.stack(
            [plan["actions"] for plan in bank]
        ).astype(np.float32),
        "future_pixels": future_pixels,
        "future_states": future_states,
        "goal_pixels": goal_pixels,
        "goal_state": goal_state,
        "physical_cost": physical_cost,
        "conditions": np.asarray(list(modes)),
        "physical_steps": PHYSICAL_STEPS,
        "initial_state": prefix_states[modes[0]][0],
        "cube_distance": cube_gap,
        "effector_distance": eff_gap,
        "min_distance": min_distance,
        "first_entry_step": first_entry,
        "mujoco_contacts": contacts,
        "cube_height": future_states[:, :, :, 4].copy(),
        "gripper_opening": future_states[:, :, :, 5].copy(),
        "candidate_ids": np.asarray([plan["candidate_id"] for plan in bank]),
        "goal_cube_pos": goal_cube,
        "goal_effector_pos": goal_eff,
    }

