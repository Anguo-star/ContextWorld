#!/usr/bin/env python3
# Export the Development Action Delay H7 catalog to the 66-table Lance layout.
#
# Reads the Development release built by
# build_tworoom_action_delay_h7_dev_structural_parity_v1.py (300 Test-structured
# queries, 11 delay families per query) and writes one Lance table per
# (eval seed profile, delay) with the member naming the Development reader
# already matches (ad-h7-paired-val-pXXX-dY-hash.lance).  Every episode is a
# 50-raw-step clip laid out exactly like the legacy payload: 6 history blocks,
# 3 future probe blocks, 1 zero-action tail; row r carries the state after r
# raw steps, and the frames at steps 0,5,...,35 are the 7 history tokens plus
# the horizon-1 true future of that delay.  Frames are produced by the same
# extended action-delay environment the frozen builder uses and are verified
# bitwise against the frozen npz assets (all 10 block-boundary frames, the
# raw state trace, and the action commands -- the latter two independently
# re-derived from the catalog's own direction/agent_speed/action_magnitude,
# not merely echoed back), then re-verified after the Lance write.
# Stratification keys ride as per-episode constant dev_ columns.

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation.action_delay_env import (
    ACTION_DELAY_FACTOR,
    make_extended_action_delay_env,
)
from contextworld.evaluation.action_delay_h7_validation import (
    DELAYS,
    QUERIES_PER_SEED,
    file_sha256,
)
from contextworld.evaluation.action_delay_long_history import (
    future_action_blocks,
    history_action_blocks,
)
from contextworld.paths import portable_contextworld_path, resolve_contextworld_path
from contextworld.synthesis.lance import build_lance_writer
from contextworld.synthesis.manifest import write_json
from contextworld.synthesis.stablewm import load_stable_worldmodel

PINNED_STABLEWM = "1b81dcd1a60e3ce27b7cf97f2f71f7da09c6502b"
PIXEL_CODEC = {"format": "png", "compress_level": 1, "lossless": True}
EPISODE_ROWS = 50
HISTORY_TOKENS = 7
# Block-boundary raw steps: 7 history tokens (0,5,...,30) plus the 3 future
# probe boundaries (35,40,45) -- i.e. every one of the 9 protocol blocks'
# ends.  TOKEN_ROWS is the subset the model-facing reader actually consumes
# (history + the horizon-1 true future); BOUNDARY_ROWS is the full set this
# script verifies bitwise against the frozen npz (history_pixels covers the
# first 7, true_future_pixels the last 3).
TOKEN_ROWS = (0, 5, 10, 15, 20, 25, 30, 35)
BOUNDARY_ROWS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
TAIL_BLOCKS = 1
DEV_EVAL_SEEDS = (52, 53, 54, 55, 56, 57)
READER_ID = "tworoom_action_delay_paired_collection_v1"
TEST_CATALOG = (
    ROOT
    / "artifacts/evaluation/history7/action_delay_validation_v1/catalog.json"
)
TEST_CATALOG_SHA256 = (
    "5a3fc1a53c05cc4e6f97f7b7544c48aaaf6033550b70a63899eab35ee46fafca"
)


def _episode_commands(npz: dict[str, np.ndarray]) -> np.ndarray:
    """The 50 raw commands: 9 protocol blocks plus one zero tail block."""

    blocks = np.asarray(npz["action_blocks"], dtype=np.float32)
    if blocks.shape != (9, 5, 2):
        raise RuntimeError(f"Action blocks have shape {blocks.shape}")
    tail = np.zeros((TAIL_BLOCKS, 5, 2), dtype=np.float32)
    return np.concatenate([blocks, tail], axis=0).reshape(-1, 2)


def _rollout_episode(
    query: dict[str, Any],
    delay: int,
    commands: np.ndarray,
    *,
    agent_speed: float,
    maximum_delay_steps: int,
) -> dict[str, np.ndarray]:
    """Per-raw-step rollout; returns the 50-row physical trace (rows 0..49).

    Row r carries the state after r raw steps (row 0 is the reset, before
    any command).  ``commands`` has exactly ``EPISODE_ROWS`` entries (the 45
    frozen protocol commands plus this script's own 5-step zero tail); one
    ``env.step`` is issued per command, matching one raw step to one row.
    The environment's ``info`` dict already carries ``proprio``/``state``/
    ``goal_state``/``distance_to_target`` (the same fields the frozen H7
    catalog builder reads), so this rollout records them directly rather
    than recomputing them by hand.
    """

    template = query["template"]
    query_id = str(query["query_id"])
    if len(commands) != EPISODE_ROWS:
        raise RuntimeError(
            f"{query_id}: expected {EPISODE_ROWS} raw commands, got {len(commands)}"
        )
    env = make_extended_action_delay_env(
        max_delay_steps=maximum_delay_steps,
        render_mode="rgb_array",
    )
    frames: list[np.ndarray] = []
    proprio_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    goal_rows: list[np.ndarray] = []
    distance_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    terminated_rows: list[np.ndarray] = []
    truncated_rows: list[np.ndarray] = []
    render_time_rows: list[np.ndarray] = []

    def _record(
        info: dict[str, Any],
        *,
        reward: float,
        terminated: float,
        truncated: float,
    ) -> None:
        started = time.perf_counter()
        frame = np.asarray(env.render(), dtype=np.uint8).copy()
        render_time_rows.append(
            np.asarray([time.perf_counter() - started], dtype=np.float32)
        )
        frames.append(frame)
        proprio_rows.append(
            np.asarray(info["proprio"], dtype=np.float32).reshape(-1).copy()
        )
        state_rows.append(
            np.asarray(info["state"], dtype=np.float32).reshape(-1).copy()
        )
        goal_rows.append(
            np.asarray(info["goal_state"], dtype=np.float32).reshape(-1).copy()
        )
        distance_rows.append(
            np.asarray([info["distance_to_target"]], dtype=np.float32)
        )
        reward_rows.append(np.asarray([reward], dtype=np.float32))
        terminated_rows.append(np.asarray([terminated], dtype=np.float32))
        truncated_rows.append(np.asarray([truncated], dtype=np.float32))

    try:
        observation, info = env.reset(
            seed=int(template["simulator_seed"]),
            options={
                "variation": (),
                "variation_values": {
                    "agent.speed": np.asarray([float(agent_speed)], dtype=np.float32),
                    ACTION_DELAY_FACTOR: int(delay),
                },
                "state": np.asarray(template["reset_state"], dtype=np.float32),
                "target_state": np.asarray(template["goal_state"], dtype=np.float32),
            },
        )
        # The Lance "observation" column is a per-episode-constant context
        # vector (initial position + goal + door/wall geometry) -- verified
        # against the legacy payload across all 160 of its episodes: obs[0:2]
        # is the *initial* state (not the current row's state), obs[2:4] is
        # goal_state, obs[4:6] are the fixed wall/door geometry, obs[6:10]
        # are zero.  It is captured once here and held constant below.
        initial_observation = np.asarray(observation, dtype=np.float32).reshape(-1).copy()
        _record(info, reward=0.0, terminated=0.0, truncated=0.0)
        for raw_index, command in enumerate(commands):
            observation, reward, terminated, truncated, info = env.step(command)
            if raw_index < 45 and (terminated or truncated):
                raise RuntimeError(
                    f"{query_id}: rollout ended early at delay {delay} (raw "
                    f"command {raw_index}, inside the 45 frozen protocol steps)"
                )
            _record(
                info,
                reward=float(reward),
                terminated=float(bool(terminated)),
                truncated=float(bool(truncated)),
            )
    finally:
        env.close()

    expected_rows = len(commands) + 1
    if len(frames) != expected_rows:
        raise RuntimeError(
            f"{query_id}: rollout produced {len(frames)} rows, expected "
            f"{expected_rows} (reset + {len(commands)} commands)"
        )

    def _clip(rows: list[np.ndarray]) -> np.ndarray:
        return np.stack(rows[:EPISODE_ROWS])

    return {
        "pixels": _clip(frames).astype(np.uint8),
        "observation": np.tile(initial_observation, (EPISODE_ROWS, 1)).astype(np.float32),
        "proprio": _clip(proprio_rows).astype(np.float32),
        "state": _clip(state_rows).astype(np.float32),
        "goal_state": _clip(goal_rows).astype(np.float32),
        "distance_to_target": _clip(distance_rows).astype(np.float32),
        "reward": _clip(reward_rows).astype(np.float32),
        "terminated": _clip(terminated_rows).astype(np.float32),
        "truncated": _clip(truncated_rows).astype(np.float32),
        "render_time": _clip(render_time_rows).astype(np.float32),
    }


def _verify_rollout(
    query: dict[str, Any],
    delay: int,
    npz: dict[str, np.ndarray],
    rollout: dict[str, np.ndarray],
    commands: np.ndarray,
    *,
    action_magnitude: float,
) -> None:
    """Bitwise-verify frames, the raw state trace, and the action commands
    against the frozen npz asset, BEFORE the Lance write.  Raises on the
    first mismatch found; never weakened to let a mismatch pass silently.
    """

    query_id = str(query["query_id"])
    template = query["template"]

    # Action commands: independently re-derive the 9 protocol blocks from
    # the catalog's own direction/action_magnitude (the same builder
    # functions the frozen catalog used), rather than merely comparing the
    # commands this script drove the rollout with back against the same
    # npz array they were copied from.
    expected_blocks = np.concatenate(
        [
            history_action_blocks(
                history_tokens=HISTORY_TOKENS,
                direction=str(query["direction"]),
                action_magnitude=action_magnitude,
            ),
            future_action_blocks(
                direction=str(query["direction"]),
                action_magnitude=action_magnitude,
            ),
        ],
        axis=0,
    ).astype(np.float32)
    if not np.array_equal(expected_blocks, npz["action_blocks"]):
        raise RuntimeError(
            f"{query_id}: frozen action_blocks differ from the independently "
            "recomputed history/future protocol blocks"
        )
    if not np.array_equal(commands[:45], npz["action_blocks"].reshape(-1, 2)):
        raise RuntimeError(
            f"{query_id}: driven raw commands differ from the frozen action_blocks"
        )

    # Block-boundary frames: all 10 (7 history tokens + 3 future horizons).
    expected_frames = np.concatenate(
        [npz["history_pixels"][delay], npz["true_future_pixels"][delay]], axis=0
    )
    observed_frames = rollout["pixels"][list(BOUNDARY_ROWS)]
    if not np.array_equal(observed_frames, expected_frames):
        raise RuntimeError(
            f"{query_id}/d{delay}: block-boundary frames differ from the "
            "frozen npz asset"
        )

    # Raw state trace: row 0 against the frozen initial state, rows 1..45
    # against the frozen 45-raw-step trajectory (audit_raw_states).
    reset_state = np.asarray(template["reset_state"], dtype=np.float32)
    if not np.array_equal(rollout["state"][0], reset_state):
        raise RuntimeError(
            f"{query_id}/d{delay}: row 0 state differs from template.reset_state"
        )
    if not np.array_equal(rollout["state"][0], npz["audit_history_states"][delay][0]):
        raise RuntimeError(
            f"{query_id}/d{delay}: row 0 state differs from the frozen "
            "audit_history_states"
        )
    if not np.array_equal(rollout["state"][1:46], npz["audit_raw_states"][delay]):
        raise RuntimeError(
            f"{query_id}/d{delay}: raw state trace (rows 1-45) differs from "
            "the frozen audit_raw_states"
        )


def _episode_identifier(query_id: str, delay: int) -> float:
    """A deterministic, per-episode-constant, collision-resistant float id.

    The ``id`` column is not part of the frozen reader contract (the paired
    Development reader selects only episode_idx/step_idx/pixels/action, plus
    optionally variation_agent_speed) and is not bit-identical to the
    unrelated legacy value (a different, training-data generation pathway),
    but it must still be constant within an episode and distinct across the
    50 episodes of a table -- matching the legacy convention verified across
    all 160 of its episodes.  blake2b(query_id|delay) truncated to 8 bytes,
    read as an unsigned 64-bit integer, then widened to float64 (the writer
    downcasts to float32, same precision loss the legacy id already has).
    """

    digest = hashlib.blake2b(
        f"{query_id}|delay={delay}".encode("utf-8"), digest_size=8
    ).digest()
    return float(int.from_bytes(digest, byteorder="big", signed=False))


def _table_fingerprint(query_ids: Sequence[str], delay: int) -> str:
    """First 10 hex chars of a sha256 over the table's sorted query ids + delay.

    Deterministic and reproducible from catalog content alone (no pixel data
    needed).  Depending on delay as well as the query set gives every one of
    the 66 tables a distinct fingerprint, mirroring the legacy payload's own
    per-table (not merely per-profile) hash distinctness.
    """

    payload = "\n".join(sorted(query_ids)) + f"\ndelay={delay}\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def _decode_png(blob: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(bytes(blob))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _episode_row_dict(
    query: dict[str, Any],
    delay: int,
    rollout: dict[str, np.ndarray],
    commands: np.ndarray,
    *,
    agent_speed: float,
) -> dict[str, list[Any]]:
    """Build the flat per-step column dict the Lance writer expects.

    Column insertion order fixes the on-disk field order (the pinned
    StableWM LanceWriter orders columns by first appearance): the 16 legacy
    columns first, matching the frozen payload name-for-name and type-for-
    type, then the ``dev_*`` stratification columns.
    """

    rows = EPISODE_ROWS
    query_id = str(query["query_id"])
    episode_id = _episode_identifier(query_id, delay)
    return {
        "proprio": list(rollout["proprio"]),
        "state": list(rollout["state"]),
        "goal_state": list(rollout["goal_state"]),
        "distance_to_target": list(rollout["distance_to_target"]),
        "render_time": list(rollout["render_time"]),
        "pixels": list(rollout["pixels"]),
        "observation": list(rollout["observation"]),
        "reward": list(rollout["reward"]),
        "terminated": list(rollout["terminated"]),
        "truncated": list(rollout["truncated"]),
        "action": list(commands.astype(np.float32)),
        "id": [np.asarray([episode_id], dtype=np.float32) for _ in range(rows)],
        "variation_agent_speed": [
            np.asarray([agent_speed], dtype=np.float32) for _ in range(rows)
        ],
        "variation_action_delay_steps": [
            np.asarray([float(delay)], dtype=np.float32) for _ in range(rows)
        ],
        "dev_eval_seed": [
            np.asarray([float(query["eval_seed"])], dtype=np.float32)
            for _ in range(rows)
        ],
        "dev_room": [str(query["room"])] * rows,
        "dev_direction": [str(query["direction"])] * rows,
        "dev_query_id": [query_id] * rows,
        "dev_delay": [
            np.asarray([float(delay)], dtype=np.float32) for _ in range(rows)
        ],
    }


# ---------------------------------------------------------------------------
# Parallel rollout workers
# ---------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _init_worker(
    repo_root: str,
    configured_repo: str,
    stable_commit: str,
    catalog_path: str,
) -> None:
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    root = Path(repo_root)
    load_stable_worldmodel(root, configured_repo, stable_commit)
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
    protocol = catalog["protocol"]
    _WORKER["root"] = root
    _WORKER["queries_by_id"] = {
        str(row["query_id"]): row for row in catalog["queries"]
    }
    _WORKER["agent_speed"] = float(protocol["agent_speed"])
    _WORKER["action_magnitude"] = float(protocol["action_magnitude"])
    _WORKER["maximum_delay_steps"] = max(int(value) for value in protocol["delay_values"])


def _process_query(query_id: str) -> tuple[str, dict[int, dict[str, list[Any]]]]:
    """Roll out, verify, and build the per-step episode dict for all 11
    delays of one query.  Runs inside a worker process; see :func:`_init_worker`.
    """

    query = _WORKER["queries_by_id"][query_id]
    root = _WORKER["root"]
    agent_speed = _WORKER["agent_speed"]
    action_magnitude = _WORKER["action_magnitude"]
    maximum_delay_steps = _WORKER["maximum_delay_steps"]

    asset_path = resolve_contextworld_path(query["asset"], repo_root=root)
    if file_sha256(asset_path) != query["asset_sha256"]:
        raise RuntimeError(f"{query_id}: asset_sha256 mismatch against the frozen catalog")
    with np.load(asset_path, allow_pickle=False) as bundle:
        npz = {name: np.asarray(bundle[name]).copy() for name in bundle.files}

    commands = _episode_commands(npz)
    episodes: dict[int, dict[str, list[Any]]] = {}
    for delay in DELAYS:
        rollout = _rollout_episode(
            query,
            delay,
            commands,
            agent_speed=agent_speed,
            maximum_delay_steps=maximum_delay_steps,
        )
        _verify_rollout(
            query, delay, npz, rollout, commands, action_magnitude=action_magnitude
        )
        episodes[delay] = _episode_row_dict(
            query, delay, rollout, commands, agent_speed=agent_speed
        )
    return query_id, episodes


# ---------------------------------------------------------------------------
# Post-write verification
# ---------------------------------------------------------------------------


def _verify_table_after_write(
    table_path: Path,
    episode_meta: list[dict[str, Any]],
    queries_by_id: dict[str, dict[str, Any]],
    npz_by_query: dict[str, dict[str, np.ndarray]],
    *,
    agent_speed: float,
    action_magnitude: float,
) -> dict[str, Any]:
    """Re-open the written table and re-verify every episode against the
    frozen npz -- independent of whatever the in-memory rollout produced.
    """

    import lance

    dataset = lance.dataset(table_path)
    table = dataset.to_table(
        columns=[
            "episode_idx",
            "step_idx",
            "pixels",
            "state",
            "goal_state",
            "observation",
            "reward",
            "action",
            "id",
            "variation_agent_speed",
            "variation_action_delay_steps",
            "dev_query_id",
            "dev_delay",
        ]
    )
    episode_idx = np.asarray(table["episode_idx"].to_numpy(), dtype=np.int64)
    step_idx = np.asarray(table["step_idx"].to_numpy(), dtype=np.int64)
    pixels = table["pixels"].to_pylist()
    state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
    goal_state = np.asarray(table["goal_state"].to_pylist(), dtype=np.float32)
    observation = np.asarray(table["observation"].to_pylist(), dtype=np.float32)
    reward = np.asarray(table["reward"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    id_col = np.asarray(
        [value[0] for value in table["id"].to_pylist()], dtype=np.float32
    )
    speed_col = np.asarray(
        [value[0] for value in table["variation_agent_speed"].to_pylist()],
        dtype=np.float32,
    )
    delay_col = np.asarray(
        [value[0] for value in table["variation_action_delay_steps"].to_pylist()],
        dtype=np.float32,
    )
    dev_delay_col = np.asarray(
        [value[0] for value in table["dev_delay"].to_pylist()], dtype=np.float32
    )
    query_id_col = table["dev_query_id"].to_pylist()

    failures: list[str] = []
    distinct_episodes = sorted(int(value) for value in np.unique(episode_idx))
    if len(distinct_episodes) != len(episode_meta):
        failures.append(
            f"expected {len(episode_meta)} distinct episodes, found "
            f"{len(distinct_episodes)}"
        )

    episode_id_values: dict[int, float] = {}

    for meta in episode_meta:
        episode = meta["episode_idx"]
        query_id = meta["query_id"]
        delay = meta["delay"]
        rows = np.flatnonzero(episode_idx == episode)
        rows = rows[np.argsort(step_idx[rows])]
        if not np.array_equal(step_idx[rows], np.arange(EPISODE_ROWS)):
            failures.append(f"{query_id}/ep{episode}: step_idx is not a {EPISODE_ROWS}-row clip")
            continue

        row_query_ids = {query_id_col[int(row)] for row in rows}
        if row_query_ids != {query_id}:
            failures.append(
                f"{query_id}/ep{episode}: dev_query_id column is not the constant "
                f"expected value (found {row_query_ids})"
            )
        if not np.allclose(delay_col[rows], float(delay)):
            failures.append(
                f"{query_id}/ep{episode}: variation_action_delay_steps != {delay}"
            )
        if not np.allclose(dev_delay_col[rows], float(delay)):
            failures.append(f"{query_id}/ep{episode}: dev_delay != {delay}")
        if not np.allclose(speed_col[rows], agent_speed):
            failures.append(
                f"{query_id}/ep{episode}: variation_agent_speed != {agent_speed}"
            )

        query = queries_by_id[query_id]
        npz = npz_by_query[query_id]

        expected_blocks = np.concatenate(
            [
                history_action_blocks(
                    history_tokens=HISTORY_TOKENS,
                    direction=str(query["direction"]),
                    action_magnitude=action_magnitude,
                ),
                future_action_blocks(
                    direction=str(query["direction"]),
                    action_magnitude=action_magnitude,
                ),
            ],
            axis=0,
        ).astype(np.float32)
        expected_commands = np.concatenate(
            [expected_blocks, np.zeros((TAIL_BLOCKS, 5, 2), dtype=np.float32)], axis=0
        ).reshape(-1, 2)
        observed_actions = action[rows]
        if not np.array_equal(observed_actions, expected_commands):
            failures.append(
                f"{query_id}/ep{episode}: action column differs from the "
                "recomputed protocol commands"
            )

        expected_frames = np.concatenate(
            [npz["history_pixels"][delay], npz["true_future_pixels"][delay]], axis=0
        )
        observed_frames = np.stack(
            [_decode_png(pixels[int(row)]) for row in rows[list(BOUNDARY_ROWS)]]
        )
        if not np.array_equal(observed_frames, expected_frames):
            failures.append(
                f"{query_id}/ep{episode}: block-boundary frames differ from the "
                "frozen npz (post-write)"
            )

        observed_state = state[rows]
        reset_state = np.asarray(query["template"]["reset_state"], dtype=np.float32)
        if not np.array_equal(observed_state[0], reset_state):
            failures.append(
                f"{query_id}/ep{episode}: row 0 state differs from template.reset_state "
                "(post-write)"
            )
        if not np.array_equal(observed_state[1:46], npz["audit_raw_states"][delay]):
            failures.append(
                f"{query_id}/ep{episode}: raw state trace (rows 1-45) differs from "
                "audit_raw_states (post-write)"
            )

        observed_reward = reward[rows].reshape(-1)
        if not np.array_equal(observed_reward, np.zeros_like(observed_reward)):
            failures.append(f"{query_id}/ep{episode}: reward is not all-zero (post-write)")

        # observation: per-episode-constant context vector -- initial state
        # (row 0's state) + goal_state, held constant for all 50 rows.
        observed_observation = observation[rows]
        if not np.array_equal(
            observed_observation, np.tile(observed_observation[0], (EPISODE_ROWS, 1))
        ):
            failures.append(
                f"{query_id}/ep{episode}: observation is not constant across the "
                "episode (post-write)"
            )
        if not np.array_equal(observed_observation[0, :2], observed_state[0]):
            failures.append(
                f"{query_id}/ep{episode}: observation[0:2] differs from the initial "
                "state (post-write)"
            )
        if not np.array_equal(observed_observation[0, 2:4], goal_state[rows][0]):
            failures.append(
                f"{query_id}/ep{episode}: observation[2:4] differs from goal_state "
                "(post-write)"
            )

        # id: per-episode constant, matching the independently recomputed
        # blake2b identifier at float32 precision; distinctness across the
        # table's episodes is checked once after this loop.
        observed_id = id_col[rows]
        expected_id = np.float32(_episode_identifier(query_id, delay))
        if not np.array_equal(observed_id, np.full_like(observed_id, expected_id)):
            failures.append(
                f"{query_id}/ep{episode}: id is not the constant recomputed "
                "identifier (post-write)"
            )
        else:
            episode_id_values[episode] = float(expected_id)

    if len(set(episode_id_values.values())) != len(episode_meta):
        failures.append(
            f"episode 'id' values are not pairwise distinct across the table's "
            f"{len(episode_meta)} episodes ({len(set(episode_id_values.values()))} unique, "
            "post-write)"
        )

    return {
        "table": str(table_path),
        "episodes": len(episode_meta),
        "rows": int(len(episode_idx)),
        "failures": failures,
        "passed": not failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    catalog_path = args.catalog.resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("status") != "frozen_before_model_scoring":
        raise ValueError("Development catalog is not frozen")
    protocol = catalog["protocol"]
    if tuple(int(value) for value in protocol["delay_values"]) != DELAYS:
        raise ValueError(
            f"Delay protocol differs from the frozen contract: {protocol['delay_values']}"
        )
    if int(protocol["history_tokens"]) != HISTORY_TOKENS:
        raise ValueError("history_tokens protocol differs from the frozen contract")
    agent_speed = float(protocol["agent_speed"])
    action_magnitude = float(protocol["action_magnitude"])

    queries = list(catalog["queries"])
    if len(queries) != len(DEV_EVAL_SEEDS) * QUERIES_PER_SEED:
        raise ValueError(f"Expected 300 queries, found {len(queries)}")
    queries_by_id = {str(row["query_id"]): row for row in queries}

    eval_seeds = sorted({int(row["eval_seed"]) for row in queries})
    if tuple(eval_seeds) != DEV_EVAL_SEEDS:
        raise ValueError(f"Unexpected eval seeds: {eval_seeds}")
    profile_of_seed = {seed: f"p{index:03d}" for index, seed in enumerate(eval_seeds)}

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in queries:
        by_seed[int(row["eval_seed"])].append(row)
    for seed, rows in by_seed.items():
        if len(rows) != QUERIES_PER_SEED:
            raise ValueError(
                f"eval_seed {seed} has {len(rows)} queries, expected {QUERIES_PER_SEED}"
            )
        indices = sorted(int(row["evaluation_index"]) for row in rows)
        if indices != list(range(QUERIES_PER_SEED)):
            raise ValueError(
                f"eval_seed {seed} evaluation_index is not a 0..{QUERIES_PER_SEED - 1} "
                "permutation"
            )

    # Cheap, informational cross-check against the frozen Public Test
    # catalog: the Development builder already ran (and froze) the full
    # disjointness audit, so this is belt-and-suspenders, not a new gate.
    isolation: dict[str, Any] = {"checked": False}
    if TEST_CATALOG.is_file():
        test_sha = file_sha256(TEST_CATALOG)
        test_catalog = json.loads(TEST_CATALOG.read_text(encoding="utf-8"))
        test_pixel_hashes = {row["query_pixels_sha256"] for row in test_catalog["queries"]}
        dev_pixel_hashes = {row["query_pixels_sha256"] for row in queries}
        overlap = sorted(dev_pixel_hashes & test_pixel_hashes)
        isolation = {
            "checked": True,
            "test_catalog_sha256_matches_frozen_pin": test_sha == TEST_CATALOG_SHA256,
            "query_pixels_sha256_overlap_with_public_test": overlap,
            "passed": test_sha == TEST_CATALOG_SHA256 and not overlap,
        }

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    (output_root / "data").mkdir(parents=True)
    # Lance commits rename directories, which the shared output filesystem
    # rejects; stage tables on local disk and copy them in (the same pattern
    # as the door/speed exporters).
    staging_root = Path(args.staging_root).resolve()
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    print(f"[action-delay-lance] loading {len(queries)} frozen npz assets", flush=True)
    npz_by_query: dict[str, dict[str, np.ndarray]] = {}
    for row in queries:
        query_id = str(row["query_id"])
        asset_path = resolve_contextworld_path(row["asset"], repo_root=ROOT)
        if file_sha256(asset_path) != row["asset_sha256"]:
            raise RuntimeError(f"{query_id}: asset_sha256 mismatch against the frozen catalog")
        with np.load(asset_path, allow_pickle=False) as bundle:
            npz_by_query[query_id] = {
                name: np.asarray(bundle[name]).copy() for name in bundle.files
            }

    swm, stable_repo, commit = load_stable_worldmodel(ROOT, args.stablewm_repo, PINNED_STABLEWM)
    if commit != PINNED_STABLEWM:
        raise RuntimeError(f"StableWM commit mismatch: {commit}")

    query_ids = [str(row["query_id"]) for row in queries]
    episodes_by_query: dict[str, dict[int, dict[str, list[Any]]]] = {}
    if args.workers <= 1:
        _init_worker(str(ROOT), str(stable_repo), PINNED_STABLEWM, str(catalog_path))
        for index, query_id in enumerate(query_ids, start=1):
            _, episodes = _process_query(query_id)
            episodes_by_query[query_id] = episodes
            if index % 20 == 0 or index == len(query_ids):
                print(
                    f"[action-delay-lance] rolled out {index}/{len(query_ids)} queries",
                    flush=True,
                )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
            initializer=_init_worker,
            initargs=(str(ROOT), str(stable_repo), PINNED_STABLEWM, str(catalog_path)),
        ) as executor:
            for index, (query_id, episodes) in enumerate(
                executor.map(_process_query, query_ids, chunksize=1), start=1
            ):
                episodes_by_query[query_id] = episodes
                if index % 20 == 0 or index == len(query_ids):
                    print(
                        f"[action-delay-lance] rolled out {index}/{len(query_ids)} queries",
                        flush=True,
                    )

    # Group into (profile, delay) tables; episode_idx == evaluation_index so
    # the same episode_idx names the same query across all 11 delay tables
    # of a profile (required by the paired-collection reader).
    tables: dict[tuple[str, int], list[tuple[int, str, dict[str, list[Any]]]]] = defaultdict(list)
    for seed, rows in by_seed.items():
        profile = profile_of_seed[seed]
        for row in rows:
            query_id = str(row["query_id"])
            evaluation_index = int(row["evaluation_index"])
            for delay in DELAYS:
                tables[(profile, delay)].append(
                    (evaluation_index, query_id, episodes_by_query[query_id][delay])
                )

    members: list[str] = []
    table_reports: list[dict[str, Any]] = []
    query_id_table_counts: Counter[str] = Counter()
    delay_table_counts: Counter[int] = Counter()
    all_query_ids_seen: set[str] = set()
    total_bytes = 0

    for profile in sorted({profile for profile, _ in tables}):
        for delay in DELAYS:
            rows = sorted(tables[(profile, delay)], key=lambda item: item[0])
            if len(rows) != QUERIES_PER_SEED:
                raise RuntimeError(
                    f"{profile}/d{delay}: expected {QUERIES_PER_SEED} episodes, got {len(rows)}"
                )
            if [index for index, _, _ in rows] != list(range(QUERIES_PER_SEED)):
                raise RuntimeError(f"{profile}/d{delay}: episode indices are not 0..49")
            query_ids_in_table = [query_id for _, query_id, _ in rows]

            # The written "id" column is per-episode constant (asserted by
            # _verify_table_after_write) and must be pairwise distinct across
            # the table's 50 episodes; check at the same float32 precision
            # the writer stores (a float64 blake2b digest could coincide
            # only after that downcast, never before it).
            table_ids_f32 = {
                np.float32(_episode_identifier(query_id, delay)) for query_id in query_ids_in_table
            }
            if len(table_ids_f32) != len(query_ids_in_table):
                raise RuntimeError(
                    f"{profile}/d{delay}: episode 'id' values are not pairwise "
                    f"distinct ({len(table_ids_f32)} unique of {len(query_ids_in_table)})"
                )

            fingerprint = _table_fingerprint(query_ids_in_table, delay)
            table_name = f"ad-h7-paired-val-{profile}-d{delay}-{fingerprint}.lance"
            staged_path = staging_root / table_name
            writer = build_lance_writer(swm, staged_path, pixel_codec=dict(PIXEL_CODEC))
            with writer:
                writer.write_episodes(episode for _, _, episode in rows)

            table_path = output_root / "data" / table_name
            shutil.copytree(staged_path, table_path)
            members.append(str(table_path.relative_to(output_root)))
            total_bytes += sum(
                path.stat().st_size for path in table_path.rglob("*") if path.is_file()
            )

            audit = _verify_table_after_write(
                table_path,
                [
                    {"episode_idx": index, "query_id": query_id, "delay": delay}
                    for index, query_id, _ in rows
                ],
                queries_by_id,
                npz_by_query,
                agent_speed=agent_speed,
                action_magnitude=action_magnitude,
            )
            table_reports.append(audit)
            if not audit["passed"]:
                raise RuntimeError(f"Post-write verification failed for {table_name}: {audit}")

            for query_id in query_ids_in_table:
                query_id_table_counts[query_id] += 1
                all_query_ids_seen.add(query_id)
            delay_table_counts[delay] += 1
            print(f"[action-delay-lance] wrote+verified {table_name}", flush=True)

    if len(members) != 66:
        raise RuntimeError(f"Expected 66 Lance tables, wrote {len(members)}")
    if len(all_query_ids_seen) != 300:
        raise RuntimeError(
            f"Expected 300 distinct dev_query_id across all tables, found "
            f"{len(all_query_ids_seen)}"
        )
    bad_query_counts = {
        qid: n for qid, n in query_id_table_counts.items() if n != len(DELAYS)
    }
    if bad_query_counts:
        raise RuntimeError(
            f"Some query_ids do not appear in exactly {len(DELAYS)} tables: {bad_query_counts}"
        )
    bad_delay_counts = {
        delay: n for delay, n in delay_table_counts.items() if n != len(eval_seeds)
    }
    if bad_delay_counts:
        raise RuntimeError(
            f"Some delays do not appear in exactly {len(eval_seeds)} tables: {bad_delay_counts}"
        )

    coverage = {
        "tables": len(members),
        "profiles": sorted({profile for profile, _ in tables}),
        "profile_eval_seed_mapping": {
            profile: seed for seed, profile in profile_of_seed.items()
        },
        "delays": list(DELAYS),
        "distinct_dev_query_ids": len(all_query_ids_seen),
        "tables_per_query_id": sorted(set(query_id_table_counts.values())),
        "tables_per_delay": sorted(set(delay_table_counts.values())),
        "rows_per_table": 2500,
        "episodes_per_table": QUERIES_PER_SEED,
        "rows_per_episode": EPISODE_ROWS,
    }

    contract = {
        "schema_version": 1,
        "benchmark": catalog["benchmark"],
        "component": "action_delay",
        "payload_kind": "development_structural_parity_v1",
        "reader_id": READER_ID,
        "episode_layout": {
            "rows_per_episode": EPISODE_ROWS,
            "history_blocks": 6,
            "future_probe_blocks": 3,
            "zero_tail_blocks": TAIL_BLOCKS,
            "raw_steps_per_block": 5,
            "model_tokens": len(TOKEN_ROWS),
            "frames_at_steps": list(TOKEN_ROWS),
            "block_boundary_rows_verified": list(BOUNDARY_ROWS),
            "token_semantics": "7 history frames then the horizon-1 true "
            "future for that delay (the reader's frame_steps); this script "
            "additionally verifies the horizon-2/3 boundary frames at rows "
            "40 and 45 bitwise against the frozen npz, though the paired "
            "reader does not consume them",
            "pixel_codec": PIXEL_CODEC,
            "naming_scheme": (
                "ad-h7-paired-val-p<PPP>-d<D>-<hash10>.lance; hash10 is the "
                "first 10 hex chars of sha256(sorted(dev_query_id for the "
                "table) + 'delay=' + D), giving every one of the 66 tables "
                "its own distinct, deterministic fingerprint"
            ),
            "stratification_columns": [
                "dev_eval_seed",
                "dev_room",
                "dev_direction",
                "dev_query_id",
                "dev_delay",
            ],
        },
        "lance_table_count": len(members),
        "members": members,
        "selection": {
            "kind": "test_structured_dev_catalog_exclusion_v1",
            "profiles": 6,
            "profile_eval_seed_mapping": {
                profile: seed for seed, profile in profile_of_seed.items()
            },
            "delay_values": list(DELAYS),
            "reference_condition": 0,
            "contrasts": [value for value in DELAYS if value != 0],
            "unique_queries": len(queries),
            "unique_queries_per_eval_seed": QUERIES_PER_SEED,
            "pairs_per_contrast_per_profile": QUERIES_PER_SEED,
            "selected_pair_count": len(queries) * (len(DELAYS) - 1),
            "eval_seeds": list(eval_seeds),
            "rooms_per_seed": {"left": 25, "right": 25},
            "directions_per_seed": {"up": 25, "down": 25},
            "independent_queries_per_delay": len(queries),
        },
        "source_catalog": {
            "path": portable_contextworld_path(catalog_path, repo_root=ROOT),
            "sha256": file_sha256(catalog_path),
        },
        "stable_worldmodel_commit": PINNED_STABLEWM,
    }

    report = {
        "schema_version": 1,
        "benchmark": catalog["benchmark"],
        "status": "passed",
        "isolation_vs_public_test": isolation,
        "coverage": coverage,
        "table_audits": table_reports,
        "tables": len(members),
        "episodes": sum(audit["episodes"] for audit in table_reports),
        "rows": sum(audit["rows"] for audit in table_reports),
        "total_bytes": total_bytes,
        "runtime_seconds": round(time.monotonic() - started, 1),
    }
    write_json(output_root / "coverage.json", coverage)
    write_json(output_root / "development_registry_contract.json", contract)
    write_json(output_root / "lance_export_report.json", report)
    shutil.rmtree(staging_root, ignore_errors=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=(
            ROOT
            / "artifacts/evaluation/history7/action_delay_dev_structural_parity_v1/catalog.json"
        ),
        help="Development structural-parity catalog.json (Test structure)",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stablewm-repo", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="ProcessPoolExecutor worker count for the 3,300-episode rollout "
        "(300 queries x 11 delays); 1 disables multiprocessing.",
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/tmp/contextworld-action-delay-dev-parity-lance-staging"),
        help="Local-disk staging for Lance writes (the output FS rejects "
        "Lance's directory renames); tables are copied to the output root "
        "after each write commits.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.stablewm_repo is None:
        arguments.stablewm_repo = (
            Path(__file__).resolve().parents[1] / "../stable-worldmodel"
        )
    result = run(arguments)
    print(
        json.dumps(
            {
                "status": result["status"],
                "tables": result["tables"],
                "episodes": result["episodes"],
                "rows": result["rows"],
                "total_bytes": result["total_bytes"],
                "isolation_vs_public_test": result["isolation_vs_public_test"],
                "coverage": result["coverage"],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
