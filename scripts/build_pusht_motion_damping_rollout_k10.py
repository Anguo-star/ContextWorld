#!/usr/bin/env python3
"""Build query-anchored K=10 PushT motion-damping rollout data.

The one-step History-3 release is the immutable source catalogue.  This
builder keeps each source history and its first query-action block unchanged,
then continues the *same* simulator with zero actions through horizon 10.
It writes 65 raw rows per episode, which is exactly one History=3,
``num_preds=10`` clip for the ordinary Stable-WorldModel reader.

This is deliberately a new artifact.  It never rewrites the K=1 release.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Iterator

import lance
import numpy as np
import pyarrow as pa
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = ROOT.parent / "stable-worldmodel"
for source_root in (ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from contextworld.evaluation import pusht_contact_friction_h3 as friction  # noqa: E402
from contextworld.evaluation.pusht_motion_damping_h3 import (  # noqa: E402
    ACTION_BLOCK,
    DAMPING_VALUES,
    ENDPOINT_MODES,
    HISTORY_RAW_STEPS,
    MINIMUM_FUTURE_GAP_PX,
    MINIMUM_HISTORY_GAP_PX,
    QUERY_RAW_STEPS,
    QUERY_REFERENCE_TOLERANCE,
    QUERY_STATE_TOLERANCE,
    MotionDampingTemplate,
    make_catalog_template,
    make_motion_damping_env,
)


PROTOCOL = "pusht_motion_damping_history3_query_anchored_rollout_k10_v1"
SPLITS = ("train", "loader_validation", "validation")
HISTORY_SIZE = 3
MAX_HORIZON = 10
RAW_ROWS_PER_EPISODE = 65
MODEL_FRAME_ROWS = tuple(range(0, 61, ACTION_BLOCK))
HORIZON_FRAME_ROWS = {
    horizon: HISTORY_RAW_STEPS + horizon * ACTION_BLOCK
    for horizon in range(1, MAX_HORIZON + 1)
}
DEFAULT_SOURCE_MANIFEST = (
    ROOT / "artifacts/synthesis/pusht_motion_damping_h3_release_v4/manifest.json"
)
# This is the public K=1 release manifest recorded in the benchmark contract.
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b"
)
DEFAULT_OUTPUT = ROOT / "artifacts/synthesis/pusht_motion_damping_h3_rollout_k10_v1"


LANCE_SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("pixels", pa.binary()),
        pa.field("action", pa.list_(pa.float32(), 2)),
        pa.field("proprio", pa.list_(pa.float32(), 4)),
        pa.field("state", pa.list_(pa.float32(), 7)),
        pa.field("goal_state", pa.list_(pa.float32(), 7)),
        pa.field("physics_state", pa.list_(pa.float32(), 12)),
        pa.field("n_contacts", pa.list_(pa.float32(), 1)),
        pa.field("hidden_motion_damping", pa.list_(pa.float32(), 1)),
    ]
)

# Current Stable-WorldModel readers reject strings repeated on every frame.
# Store all provenance once per episode in the native ``<table>_episodes``
# side table instead.  This also reduces duplicated metadata substantially.
EPISODE_SCHEMA = pa.schema(
    [
        pa.field("episode_idx", pa.int32()),
        pa.field("pair_id", pa.string()),
        pa.field("template_id", pa.string()),
        pa.field("hidden_mode", pa.string()),
        pa.field("split", pa.string()),
        pa.field("synthesis_version", pa.string()),
        pa.field("catalog_index", pa.int32()),
        pa.field("pair_index", pa.int32()),
        pa.field("twin_group_index", pa.int32()),
        pa.field("source_episode_idx", pa.int32()),
    ]
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(child).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_output(path: Path) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def _template_from_dict(raw: dict[str, Any]) -> MotionDampingTemplate:
    """Normalise JSON templates into the frozen evaluation dataclass."""

    return MotionDampingTemplate(
        template_id=str(raw["template_id"]),
        faster_decay_reset_snapshot=tuple(
            float(value) for value in raw["faster_decay_reset_snapshot"]
        ),
        no_extra_decay_reset_snapshot=tuple(
            float(value) for value in raw["no_extra_decay_reset_snapshot"]
        ),
        goal_state=tuple(float(value) for value in raw["goal_state"]),
        history_actions=tuple(
            tuple(float(value) for value in action)
            for action in raw["history_actions"]
        ),
        query_actions=tuple(
            tuple(float(value) for value in action)
            for action in raw["query_actions"]
        ),
        expected_natural_query_snapshot=tuple(
            float(value) for value in raw["expected_natural_query_snapshot"]
        ),
        simulator_seed=int(raw["simulator_seed"]),
        visible_shape_id=int(raw.get("visible_shape_id", 4)),
        visible_shape_name=str(raw.get("visible_shape_name", "square")),
    )


def _rollout_actions(template: MotionDampingTemplate) -> np.ndarray:
    """Return 65 actions: H3 history, query, 9 horizon holds, completion."""

    history = np.asarray(template.history_actions, dtype=np.float64)
    query = np.asarray(template.query_actions, dtype=np.float64)
    if history.shape != (HISTORY_RAW_STEPS, 2):
        raise ValueError(f"invalid history action shape: {history.shape}")
    if query.shape != (QUERY_RAW_STEPS, 2):
        raise ValueError(f"invalid query action shape: {query.shape}")
    holds = np.zeros(((MAX_HORIZON - 1) * ACTION_BLOCK, 2), dtype=np.float64)
    completion = np.zeros((ACTION_BLOCK, 2), dtype=np.float64)
    actions = np.concatenate((history, query, holds, completion), axis=0)
    if actions.shape != (RAW_ROWS_PER_EPISODE, 2):
        raise AssertionError(actions.shape)
    return actions


def _simulate(
    template: MotionDampingTemplate,
    *,
    mode: str,
    resolution: int,
    include_rows: bool,
) -> dict[str, Any]:
    """Simulate x0 through the post-h10 completion block in one environment."""

    actions = _rollout_actions(template)
    expected_query = np.asarray(
        template.expected_natural_query_snapshot, dtype=np.float64
    )
    goal_state = np.asarray(template.goal_state, dtype=np.float32)
    model_states: dict[int, np.ndarray] = {}
    model_pixels: dict[int, np.ndarray] = {}
    rows: dict[str, list[Any]] | None = None
    if include_rows:
        rows = {
            "pixels": [],
            "action": [],
            "proprio": [],
            "state": [],
            "goal_state": [],
            "physics_state": [],
            "n_contacts": [],
            "hidden_motion_damping": [],
        }
    contact_counts: list[int] = []
    arbiter_counts: list[int] = []
    bounds_ok = True
    env, _ = make_motion_damping_env(template, mode=mode, resolution=resolution)
    query_snapshot: np.ndarray | None = None
    try:
        bounds_ok = bounds_ok and friction._bounds_inside_playfield(
            friction._body_shape_bounds(env)
        )
        for raw_step, action in enumerate(actions):
            state = np.asarray(env._get_obs(), dtype=np.float64)
            physics = friction.body_snapshot(env)
            pixels: np.ndarray | None = None
            if include_rows or raw_step == HISTORY_RAW_STEPS:
                # Candidate screening needs RGB only at the paired query row;
                # all horizon checks use exact physics snapshots.  Accepted
                # candidates are replayed once with all 65 frames retained.
                pixels = np.asarray(env.render(), dtype=np.uint8).copy()
            if raw_step in MODEL_FRAME_ROWS:
                model_states[raw_step] = np.asarray(physics, dtype=np.float64).copy()
                if pixels is not None:
                    model_pixels[raw_step] = pixels
            if raw_step == HISTORY_RAW_STEPS:
                query_snapshot = np.asarray(physics, dtype=np.float64).copy()
            if rows is not None:
                if pixels is None:
                    raise AssertionError("row rendering unexpectedly disabled")
                rows["pixels"].append(pixels)
                rows["action"].append(np.asarray(action, dtype=np.float32).copy())
                rows["proprio"].append(
                    np.concatenate((state[:2], state[-2:])).astype(np.float32)
                )
                rows["state"].append(state.astype(np.float32))
                rows["goal_state"].append(goal_state.copy())
                rows["physics_state"].append(physics.astype(np.float32))
                rows["hidden_motion_damping"].append(
                    np.asarray([DAMPING_VALUES[mode]], dtype=np.float32)
                )
            contacts = int(friction._step_and_count_agent_block_contacts(env, action))
            contact_counts.append(contacts)
            arbiter_counts.append(len(env.space._get_arbiters()))
            if rows is not None:
                rows["n_contacts"].append(np.asarray([contacts], dtype=np.float32))
            bounds_ok = bounds_ok and friction._bounds_inside_playfield(
                friction._body_shape_bounds(env)
            )
    finally:
        env.close()
    if query_snapshot is None:
        raise AssertionError("did not reach query boundary")
    if set(model_states) != set(MODEL_FRAME_ROWS):
        raise AssertionError(f"missing model rows: {sorted(model_states)}")
    if HISTORY_RAW_STEPS not in model_pixels:
        raise AssertionError("missing query RGB")
    query_deviation = float(
        np.max(np.abs(friction._snapshot_delta(query_snapshot, expected_query)))
    )
    return {
        "template": asdict(template),
        "mode": mode,
        "raw_actions": actions.astype(np.float32),
        "model_states": model_states,
        "model_pixels": model_pixels,
        "query_snapshot": query_snapshot,
        "query_reference_deviation": query_deviation,
        "contact_counts": contact_counts,
        "arbiter_counts": arbiter_counts,
        "bounds_ok": bool(bounds_ok),
        "rows": rows,
        "state_installations_after_x0": 0,
        "query_simulator_recreated": False,
    }


def _pair_audit(
    faster: dict[str, Any], no_extra: dict[str, Any]
) -> dict[str, Any]:
    """Audit one hidden-mode pair across the full K=10 continuous rollout."""

    query_gap = float(
        np.max(
            np.abs(
                friction._snapshot_delta(
                    faster["model_states"][HISTORY_RAW_STEPS],
                    no_extra["model_states"][HISTORY_RAW_STEPS],
                )
            )
        )
    )
    history_gap = friction._visible_response_gap(
        faster["model_states"][ACTION_BLOCK],
        no_extra["model_states"][ACTION_BLOCK],
        angular_radius_px=60.0,
    )
    gaps = {
        str(horizon): friction._future_gap(
            faster["model_states"][HORIZON_FRAME_ROWS[horizon]],
            no_extra["model_states"][HORIZON_FRAME_ROWS[horizon]],
        )
        for horizon in range(1, MAX_HORIZON + 1)
    }
    # The K=1 release stored its physics sidecar as float32 before deriving
    # the published h1 gap.  Keep a separate like-for-like value for the
    # source-consistency receipt; the normal K=1..10 statistics retain the
    # simulator's float64 state.
    h1_source_compatible_gap = friction._future_gap(
        faster["model_states"][HORIZON_FRAME_ROWS[1]].astype(np.float32).astype(np.float64),
        no_extra["model_states"][HORIZON_FRAME_ROWS[1]].astype(np.float32).astype(np.float64),
    )
    query_pixels_identical = np.array_equal(
        faster["model_pixels"][HISTORY_RAW_STEPS],
        no_extra["model_pixels"][HISTORY_RAW_STEPS],
    )
    actions_identical = np.array_equal(
        faster["raw_actions"], no_extra["raw_actions"]
    )
    all_contacts = faster["contact_counts"] + no_extra["contact_counts"]
    all_arbiters = faster["arbiter_counts"] + no_extra["arbiter_counts"]
    checks = {
        "endpoint_modes": (
            faster["mode"] == ENDPOINT_MODES[0]
            and no_extra["mode"] == ENDPOINT_MODES[1]
        ),
        "actions_identical": actions_identical,
        "query_reference_exact": (
            faster["query_reference_deviation"] <= QUERY_REFERENCE_TOLERANCE
            and no_extra["query_reference_deviation"] <= QUERY_REFERENCE_TOLERANCE
        ),
        "query_full_state_paired": query_gap <= QUERY_STATE_TOLERANCE,
        "query_pixels_identical": query_pixels_identical,
        "history_response_visible": (
            history_gap["px_equivalent"] >= MINIMUM_HISTORY_GAP_PX
        ),
        "h1_response_visible": (
            gaps["1"]["block_position_px"] >= MINIMUM_FUTURE_GAP_PX
        ),
        "all_65_steps_contact_free": all(value == 0 for value in all_contacts),
        "all_65_steps_arbiter_free": all(value == 0 for value in all_arbiters),
        "all_65_steps_in_bounds": faster["bounds_ok"] and no_extra["bounds_ok"],
        "state_installations_after_x0_zero": (
            faster["state_installations_after_x0"] == 0
            and no_extra["state_installations_after_x0"] == 0
        ),
        "query_simulator_not_recreated": (
            not faster["query_simulator_recreated"]
            and not no_extra["query_simulator_recreated"]
        ),
    }
    return {
        "template_id": faster["template"]["template_id"],
        "checks": checks,
        "passed": bool(all(checks.values())),
        "history_gap": history_gap,
        "horizon_gaps": gaps,
        "h1_source_compatible_gap": h1_source_compatible_gap,
        "query_full_state_gap": query_gap,
        "query_reference_deviation": {
            ENDPOINT_MODES[0]: float(faster["query_reference_deviation"]),
            ENDPOINT_MODES[1]: float(no_extra["query_reference_deviation"]),
        },
        "total_contact_steps": int(sum(all_contacts)),
        "maximum_arbiter_count": int(max(all_arbiters, default=0)),
        "bounds_ok": bool(faster["bounds_ok"] and no_extra["bounds_ok"]),
    }


def _visible_x0_key(template: MotionDampingTemplate, mode: str) -> tuple[float, ...]:
    if mode == ENDPOINT_MODES[0]:
        snapshot = template.faster_decay_reset_snapshot
    else:
        snapshot = template.no_extra_decay_reset_snapshot
    # Velocity is not rendered.  These are all RGB-visible geometry fields.
    return tuple(
        np.asarray(
            [
                snapshot[0], snapshot[1], snapshot[6], snapshot[7], snapshot[10],
                template.goal_state[0], template.goal_state[1], template.goal_state[2],
                template.goal_state[3], template.goal_state[4],
            ],
            dtype=np.float64,
        ).round(12)
    )


def _group_preflight(payload: dict[str, Any], resolution: int) -> dict[str, Any]:
    """Picklable physical check for a forward/reverse twin group."""

    try:
        candidates = payload["candidates"]
        templates = [_template_from_dict(row["template"]) for row in candidates]
        if len(templates) != 2:
            raise ValueError("a twin group must contain exactly two templates")
        if [int(row["catalog_index"]) for row in candidates] != [
            int(payload["group_index"]) * 2,
            int(payload["group_index"]) * 2 + 1,
        ]:
            raise ValueError("catalog indices are not an adjacent twin group")
        rendered = []
        audits = []
        for template in templates:
            modes = {
                mode: _simulate(
                    template, mode=mode, resolution=resolution, include_rows=False
                )
                for mode in ENDPOINT_MODES
            }
            rendered.append(modes)
            audits.append(_pair_audit(modes[ENDPOINT_MODES[0]], modes[ENDPOINT_MODES[1]]))
        visible_balance = (
            _visible_x0_key(templates[0], ENDPOINT_MODES[0])
            == _visible_x0_key(templates[1], ENDPOINT_MODES[1])
            and _visible_x0_key(templates[0], ENDPOINT_MODES[1])
            == _visible_x0_key(templates[1], ENDPOINT_MODES[0])
        )
        passed = bool(visible_balance and all(audit["passed"] for audit in audits))
        return {
            **payload,
            "passed": passed,
            "visible_x0_label_balance": visible_balance,
            "template_audits": audits,
            "failure": None,
        }
    except Exception as exc:  # candidate rejection, not a partial artifact
        return {
            **payload,
            "passed": False,
            "visible_x0_label_balance": False,
            "template_audits": [],
            "failure": f"{type(exc).__name__}: {exc}",
        }


def _annotate_rows(
    rollout: dict[str, Any],
    *,
    split: str,
    catalog_index: int,
    pair_index: int,
    twin_group_index: int,
    source_episode_idx: int,
) -> dict[str, list[Any]]:
    rows = rollout["rows"]
    if rows is None:
        raise AssertionError("cannot serialise a preflight rollout")
    result = {name: list(values) for name, values in rows.items()}
    count = len(result["pixels"])
    if count != RAW_ROWS_PER_EPISODE:
        raise AssertionError(f"expected {RAW_ROWS_PER_EPISODE} rows, got {count}")
    template_id = str(rollout["template"]["template_id"])
    result.update(
        {
            "pair_id": [template_id] * count,
            "template_id": [template_id] * count,
            "hidden_mode": [str(rollout["mode"])] * count,
            "split": [split] * count,
            "synthesis_version": [PROTOCOL] * count,
            "catalog_index": [int(catalog_index)] * count,
            "pair_index": [int(pair_index)] * count,
            "twin_group_index": [int(twin_group_index)] * count,
            "source_episode_idx": [int(source_episode_idx)] * count,
        }
    )
    return result


def _fixed(values: Iterable[Any], size: int) -> pa.FixedSizeListArray:
    value = np.asarray(list(values), dtype=np.float32).reshape(-1, size)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(value.reshape(-1), type=pa.float32()), size
    )


def _jpeg(value: np.ndarray, quality: int) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.asarray(value, dtype=np.uint8)).save(
        buffer, format="JPEG", quality=int(quality)
    )
    return buffer.getvalue()


def _record_batch(
    rows: dict[str, list[Any]], *, episode_index: int, jpeg_quality: int
) -> pa.RecordBatch:
    count = len(rows["pixels"])
    arrays: list[pa.Array] = [
        pa.array(np.full(count, episode_index, dtype=np.int32)),
        pa.array(np.arange(count, dtype=np.int32)),
        pa.array(
            rows["pixels"]
            if rows["pixels"] and isinstance(rows["pixels"][0], bytes)
            else [_jpeg(pixel, jpeg_quality) for pixel in rows["pixels"]],
            type=pa.binary(),
        ),
        _fixed(rows["action"], 2),
        _fixed(rows["proprio"], 4),
        _fixed(rows["state"], 7),
        _fixed(rows["goal_state"], 7),
        _fixed(rows["physics_state"], 12),
        _fixed(rows["n_contacts"], 1),
        _fixed(rows["hidden_motion_damping"], 1),
    ]
    return pa.record_batch(arrays, schema=LANCE_SCHEMA)


def _write_lance(
    table: Path, episodes: Iterator[dict[str, list[Any]]], *, jpeg_quality: int
) -> None:
    def batches() -> Iterator[pa.RecordBatch]:
        for episode_index, rows in enumerate(episodes):
            yield _record_batch(rows, episode_index=episode_index, jpeg_quality=jpeg_quality)

    lance.write_dataset(
        pa.RecordBatchReader.from_batches(LANCE_SCHEMA, batches()),
        str(table),
        mode="create",
    )


def _episode_metadata(
    rows: dict[str, list[Any]], *, episode_index: int
) -> dict[str, Any]:
    result: dict[str, Any] = {"episode_idx": int(episode_index)}
    for field in EPISODE_SCHEMA.names[1:]:
        values = rows[field]
        if len(values) != RAW_ROWS_PER_EPISODE or any(
            value != values[0] for value in values[1:]
        ):
            raise RuntimeError(
                f"episode-scoped field {field!r} varies within episode "
                f"{episode_index}"
            )
        result[field] = values[0]
    return result


def _write_episode_lance(table: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty episode metadata table")
    arrays = [
        pa.array([row[field.name] for row in rows], type=field.type)
        for field in EPISODE_SCHEMA
    ]
    lance.write_dataset(
        pa.Table.from_arrays(arrays, schema=EPISODE_SCHEMA),
        str(table),
        mode="create",
    )


def _source_catalog(source: dict[str, Any], split: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for pair in source["splits"][split]["pairs"]:
        index = int(pair["catalog_index"])
        if index in result:
            raise ValueError(f"duplicate source catalog index {split}:{index}")
        result[index] = {
            "template": pair["template"],
            "source_pair_index": int(pair["pair_index"]),
            "source_audit": pair.get("audit"),
        }
    return result


def _candidate_group(
    *,
    split: str,
    group_index: int,
    source_catalog: dict[int, dict[str, Any]],
    catalog_seed: int,
) -> dict[str, Any]:
    candidates = []
    for catalog_index in (2 * group_index, 2 * group_index + 1):
        source = source_catalog.get(catalog_index)
        if source is None:
            template = asdict(
                make_catalog_template(
                    split=split,
                    catalog_index=catalog_index,
                    catalog_seed=catalog_seed,
                )
            )
            source_pair_index: int | None = None
            source_audit = None
        else:
            template = source["template"]
            source_pair_index = int(source["source_pair_index"])
            source_audit = source["source_audit"]
        candidates.append(
            {
                "catalog_index": catalog_index,
                "template": template,
                "source_pair_index": source_pair_index,
                "source_audit": source_audit,
            }
        )
    return {
        "split": split,
        "group_index": int(group_index),
        "candidates": candidates,
    }


def _preflight_many(
    payloads: list[dict[str, Any]],
    *,
    resolution: int,
    workers: int,
    executor: ProcessPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_group_preflight(payload, resolution) for payload in payloads]
    if executor is not None:
        return list(
            executor.map(
                _group_preflight, payloads, [resolution] * len(payloads)
            )
        )
    # ``map`` preserves the supplied order, so parallel preflight cannot alter
    # which frozen candidates are admitted to a split.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        return list(pool.map(_group_preflight, payloads, [resolution] * len(payloads)))


def _select_groups(
    *,
    split: str,
    pair_count: int,
    source_catalog: dict[int, dict[str, Any]],
    catalog_seed: int,
    resolution: int,
    workers: int,
    max_catalog_groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if pair_count % 2:
        raise ValueError(
            "pair counts must be even because forward/reverse twins are accepted together"
        )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    group_index = 0
    chunk_size = max(1, workers * 2)

    def consume(reports: list[dict[str, Any]]) -> bool:
        """Return true once quota is reached; unconsumed valid work is neutral."""

        for report in reports:
            if len(accepted) * 2 >= pair_count:
                return True
            if report["passed"]:
                accepted.append(report)
            else:
                rejected.append(
                    {
                        "group_index": report["group_index"],
                        "catalog_indices": [
                            row["catalog_index"] for row in report["candidates"]
                        ],
                        "template_ids": [
                            row["template"]["template_id"]
                            for row in report["candidates"]
                        ],
                        "visible_x0_label_balance": report[
                            "visible_x0_label_balance"
                        ],
                        "failure": report["failure"],
                        "failed_checks": [
                            {
                                "template_id": audit.get("template_id"),
                                "checks": [
                                    key
                                    for key, value in audit.get("checks", {}).items()
                                    if not value
                                ],
                            }
                            for audit in report["template_audits"]
                        ],
                    }
                )
        return len(accepted) * 2 >= pair_count

    # Starting a Pymunk worker pool is material on a full catalog.  Keep it
    # alive across chunks; map still returns reports in catalog order.
    if workers <= 1:
        executor_context: Any = None
        while len(accepted) * 2 < pair_count:
            if group_index >= max_catalog_groups:
                raise RuntimeError(
                    f"{split}: exhausted {max_catalog_groups} twin groups before "
                    f"accepting {pair_count} templates"
                )
            upper = min(max_catalog_groups, group_index + chunk_size)
            payloads = [
                _candidate_group(
                    split=split,
                    group_index=index,
                    source_catalog=source_catalog,
                    catalog_seed=catalog_seed,
                )
                for index in range(group_index, upper)
            ]
            done = consume(
                _preflight_many(payloads, resolution=resolution, workers=workers)
            )
            group_index = upper
            if done:
                break
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            while len(accepted) * 2 < pair_count:
                if group_index >= max_catalog_groups:
                    raise RuntimeError(
                        f"{split}: exhausted {max_catalog_groups} twin groups before "
                        f"accepting {pair_count} templates"
                    )
                upper = min(max_catalog_groups, group_index + chunk_size)
                payloads = [
                    _candidate_group(
                        split=split,
                        group_index=index,
                        source_catalog=source_catalog,
                        catalog_seed=catalog_seed,
                    )
                    for index in range(group_index, upper)
                ]
                done = consume(
                    _preflight_many(
                        payloads,
                        resolution=resolution,
                        workers=workers,
                        executor=executor,
                    )
                )
                group_index = upper
                if done:
                    break
    return accepted, rejected


def _h1_source_consistency(
    candidate: dict[str, Any], audit: dict[str, Any], modes: dict[str, dict[str, Any]], *, resolution: int
) -> dict[str, Any]:
    source_audit = candidate.get("source_audit")
    observed = float(audit["h1_source_compatible_gap"]["block_position_px"])
    if not source_audit:
        return {
            "source_template": False,
            "h1_gap_block_position_px": observed,
            "pixel_hashes_checked": False,
            "passed": True,
        }
    expected = float(source_audit["future_gap"]["block_position_px"])
    error = abs(expected - observed)
    source_hashes = source_audit.get("hashes", {})
    raw_actions = np.asarray(modes[ENDPOINT_MODES[0]]["raw_actions"], dtype=np.float32)
    observed_action_hash = _array_sha256(raw_actions[:20])
    expected_action_hash = source_hashes.get("raw_actions")
    h1_actions_match = (
        expected_action_hash is None or observed_action_hash == expected_action_hash
    )
    # The public source artifact is rendered at 224px.  A small-resolution
    # smoke build verifies the same physics/actions but cannot have identical
    # RGB bytes by design; a normal full build checks all three K=1 hashes.
    pixel_hashes_checked = int(resolution) == 224
    pixel_hashes_match = True
    pixel_hash_comparison: dict[str, dict[str, str | None]] = {
        "query_pixels": {
            "expected": source_hashes.get("query_pixels"),
            "observed": None,
        },
        "faster_decay_h1_pixels": {
            "expected": source_hashes.get("faster_decay_future_pixels"),
            "observed": None,
        },
        "no_extra_decay_h1_pixels": {
            "expected": source_hashes.get("no_extra_decay_future_pixels"),
            "observed": None,
        },
    }
    if pixel_hashes_checked:
        pixel_hash_comparison["query_pixels"]["observed"] = _array_sha256(
            modes[ENDPOINT_MODES[0]]["model_pixels"][10]
        )
        pixel_hash_comparison["faster_decay_h1_pixels"]["observed"] = _array_sha256(
            modes[ENDPOINT_MODES[0]]["model_pixels"][15]
        )
        pixel_hash_comparison["no_extra_decay_h1_pixels"]["observed"] = _array_sha256(
            modes[ENDPOINT_MODES[1]]["model_pixels"][15]
        )
        pixel_hashes_match = (
            pixel_hash_comparison["query_pixels"]["observed"]
            == pixel_hash_comparison["query_pixels"]["expected"]
            and pixel_hash_comparison["faster_decay_h1_pixels"]["observed"]
            == pixel_hash_comparison["faster_decay_h1_pixels"]["expected"]
            and pixel_hash_comparison["no_extra_decay_h1_pixels"]["observed"]
            == pixel_hash_comparison["no_extra_decay_h1_pixels"]["expected"]
        )
    return {
        "source_template": True,
        "source_h1_gap_block_position_px": expected,
        "h1_gap_block_position_px": observed,
        "absolute_error": error,
        "tolerance": 1e-8,
        "h1_actions_match_source": h1_actions_match,
        "h1_action_hash": {
            "expected": expected_action_hash,
            "observed": observed_action_hash,
        },
        "pixel_hashes_checked": pixel_hashes_checked,
        "pixel_hashes_match_source": pixel_hashes_match,
        "pixel_hash_comparison": pixel_hash_comparison,
        # RGB hashes are recorded even when an active renderer no longer
        # matches the renderer that produced the frozen K=1 release.  They
        # are provenance diagnostics, not a reason to admit an unsafe
        # physical rollout or to reject a physically/action-identical h1.
        "source_render_provenance_match": (
            not pixel_hashes_checked or pixel_hashes_match
        ),
        "source_render_provenance_reason": (
            "not_checked_non_source_resolution"
            if not pixel_hashes_checked
            else (
                None
                if pixel_hashes_match
                else "legacy_renderer_provenance_unavailable"
            )
        ),
        "passed": error <= 1e-8 and h1_actions_match,
    }


def _render_group(
    selected: dict[str, Any], *, resolution: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-run accepted candidates to retain all rows and exact RGB audits."""

    rendered_templates: list[dict[str, Any]] = []
    initial_hashes: dict[str, list[str]] = {mode: [] for mode in ENDPOINT_MODES}
    for candidate, preflight in zip(
        selected["candidates"], selected["template_audits"], strict=True
    ):
        template = _template_from_dict(candidate["template"])
        modes = {
            mode: _simulate(template, mode=mode, resolution=resolution, include_rows=True)
            for mode in ENDPOINT_MODES
        }
        audit = _pair_audit(modes[ENDPOINT_MODES[0]], modes[ENDPOINT_MODES[1]])
        if not audit["passed"]:
            raise RuntimeError(
                f"accepted candidate became invalid: {template.template_id}: "
                f"{[key for key, value in audit['checks'].items() if not value]}"
            )
        # The retained h1 physics must match the immutable source K=1 audit.
        consistency = _h1_source_consistency(
            candidate, audit, modes, resolution=resolution
        )
        if not consistency["passed"]:
            raise RuntimeError(
                f"h1 release consistency failed for {template.template_id}: {consistency}"
            )
        for mode in ENDPOINT_MODES:
            initial_hashes[mode].append(
                _array_sha256(modes[mode]["model_pixels"][0])
            )
        if preflight["template_id"] != audit["template_id"]:
            raise AssertionError("preflight template drift")
        rendered_templates.append(
            {
                "candidate": candidate,
                "modes": modes,
                "audit": audit,
                "h1_consistency": consistency,
            }
        )
    balance = Counter(initial_hashes[ENDPOINT_MODES[0]]) == Counter(
        initial_hashes[ENDPOINT_MODES[1]]
    )
    if not balance:
        raise RuntimeError("forward/reverse x0 RGB label balance failed")
    return rendered_templates, {
        "initial_pixel_hashes": initial_hashes,
        "x0_rgb_label_balanced": balance,
    }


def _render_group_for_write(
    selected: dict[str, Any], resolution: int, jpeg_quality: int
) -> dict[str, Any]:
    """Render one accepted twin group and return compact JPEG rows in order.

    This worker deliberately returns compressed RGB bytes rather than raw
    arrays.  It gives full builds parallel renderer/encoder throughput without
    filling the parent process with roughly 40 MiB of raw pixels per twin.
    """

    rendered, group_audit = _render_group(selected, resolution=resolution)
    pair_index_start = int(selected["output_pair_index_start"])
    episodes: list[dict[str, list[Any]]] = []
    reports: list[dict[str, Any]] = []
    for local_pair_index, rendered_template in enumerate(rendered):
        candidate = rendered_template["candidate"]
        pair_index = pair_index_start + local_pair_index
        reports.append(
            {
                "pair_index": pair_index,
                "catalog_index": int(candidate["catalog_index"]),
                "twin_group_index": int(selected["group_index"]),
                "template_id": candidate["template"]["template_id"],
                "source_pair_index": candidate["source_pair_index"],
                "audit": rendered_template["audit"],
                "h1_consistency": rendered_template["h1_consistency"],
                "query_hash": _array_sha256(
                    rendered_template["modes"][ENDPOINT_MODES[0]]["model_pixels"]
                    [HISTORY_RAW_STEPS]
                ),
            }
        )
        for mode_index, mode in enumerate(ENDPOINT_MODES):
            source_pair_index = candidate["source_pair_index"]
            source_episode_idx = (
                -1
                if source_pair_index is None
                else 2 * int(source_pair_index) + mode_index
            )
            rows = _annotate_rows(
                rendered_template["modes"][mode],
                split=selected["split"],
                catalog_index=int(candidate["catalog_index"]),
                pair_index=pair_index,
                twin_group_index=int(selected["group_index"]),
                source_episode_idx=source_episode_idx,
            )
            rows["pixels"] = [_jpeg(pixel, jpeg_quality) for pixel in rows["pixels"]]
            episodes.append(rows)
    return {
        "episodes": episodes,
        "reports": reports,
        "initial_pixel_hashes": group_audit["initial_pixel_hashes"],
    }


def _render_many_for_write(
    selected: list[dict[str, Any]], *, resolution: int, jpeg_quality: int, workers: int
) -> Iterator[dict[str, Any]]:
    if workers <= 1:
        for group in selected:
            yield _render_group_for_write(group, resolution, jpeg_quality)
        return
    # map keeps catalog-selected output order deterministic while Pymunk and
    # JPEG work take place in independent CPU workers.
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        yield from executor.map(
            _render_group_for_write,
            selected,
            [resolution] * len(selected),
            [jpeg_quality] * len(selected),
        )


def _stats(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if not len(data):
        raise ValueError("cannot compute an empty statistic")
    return {
        "min": float(np.min(data)),
        "mean": float(np.mean(data)),
        "max": float(np.max(data)),
    }


def _build_split(
    *,
    root: Path,
    split: str,
    pair_count: int,
    source_catalog: dict[int, dict[str, Any]],
    catalog_seed: int,
    resolution: int,
    jpeg_quality: int,
    workers: int,
    max_catalog_groups: int,
) -> dict[str, Any]:
    selected, rejected = _select_groups(
        split=split,
        pair_count=pair_count,
        source_catalog=source_catalog,
        catalog_seed=catalog_seed,
        resolution=resolution,
        workers=workers,
        max_catalog_groups=max_catalog_groups,
    )
    for group_offset, group in enumerate(selected):
        # Each accepted twin group contributes its two templates in fixed
        # forward/reverse order, then each template contributes ENDPOINT_MODES.
        group["output_pair_index_start"] = group_offset * 2
    reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    initial_hashes: dict[str, Counter[str]] = {
        mode: Counter() for mode in ENDPOINT_MODES
    }
    episode_metadata: list[dict[str, Any]] = []

    def episodes() -> Iterator[dict[str, list[Any]]]:
        for group_result in _render_many_for_write(
            selected,
            resolution=resolution,
            jpeg_quality=jpeg_quality,
            workers=workers,
        ):
            for mode in ENDPOINT_MODES:
                initial_hashes[mode].update(group_result["initial_pixel_hashes"][mode])
            for report in group_result["reports"]:
                query_hash = report.pop("query_hash")
                if query_hash in query_hashes:
                    raise RuntimeError(
                        f"duplicate query pixels in {split}: {report['template_id']}"
                    )
                query_hashes.add(query_hash)
                reports.append(report)
            for rows in group_result["episodes"]:
                episode_metadata.append(
                    _episode_metadata(rows, episode_index=len(episode_metadata))
                )
                yield rows

    table_path = root / f"{split}.lance"
    _write_lance(table_path, episodes(), jpeg_quality=jpeg_quality)
    episode_table_path = root / f"{split}_episodes.lance"
    _write_episode_lance(episode_table_path, episode_metadata)
    if len(reports) != pair_count:
        raise AssertionError(f"{split}: expected {pair_count} reports, got {len(reports)}")
    if len(episode_metadata) != pair_count * 2:
        raise AssertionError(
            f"{split}: expected {pair_count * 2} episode rows, "
            f"got {len(episode_metadata)}"
        )
    horizon_gaps = {
        str(horizon): _stats(
            report["audit"]["horizon_gaps"][str(horizon)]["block_position_px"]
            for report in reports
        )
        for horizon in range(1, MAX_HORIZON + 1)
    }
    files = [path for path in table_path.rglob("*") if path.is_file()]
    episode_files = [
        path for path in episode_table_path.rglob("*") if path.is_file()
    ]
    accepted_indices = [report["catalog_index"] for report in reports]
    rejected_indices = [
        index for group in rejected for index in group["catalog_indices"]
    ]
    h1_consistency = [report["h1_consistency"] for report in reports]
    all_audits = [report["audit"] for report in reports]
    x0_balanced = initial_hashes[ENDPOINT_MODES[0]] == initial_hashes[ENDPOINT_MODES[1]]
    result = {
        "split": split,
        "pair_count": pair_count,
        "episode_count": pair_count * 2,
        "raw_rows_per_episode": RAW_ROWS_PER_EPISODE,
        "raw_rows": pair_count * 2 * RAW_ROWS_PER_EPISODE,
        "catalog_seed": int(catalog_seed),
        "accepted_catalog_indices": accepted_indices,
        "accepted_catalog_indices_sha256": _array_sha256(
            np.asarray(accepted_indices, dtype=np.int64)
        ),
        "rejected_catalog_indices": rejected_indices,
        "rejected_groups": rejected,
        "table_path": table_path.name,
        "table_files": len(files),
        "table_bytes": sum(path.stat().st_size for path in files),
        "table_sha256": _sha256_tree(table_path),
        "episode_table_path": episode_table_path.name,
        "episode_table_rows": len(episode_metadata),
        "episode_table_files": len(episode_files),
        "episode_table_bytes": sum(path.stat().st_size for path in episode_files),
        "episode_table_sha256": _sha256_tree(episode_table_path),
        "stablewm_native_episode_metadata_layout": True,
        "query_hash_count": len(query_hashes),
        "query_hashes": sorted(query_hashes),
        "horizon_gap_block_position_px": horizon_gaps,
        "h1_source_consistency": {
            "source_templates_checked": sum(
                receipt["source_template"] for receipt in h1_consistency
            ),
            "generated_continuation_templates": sum(
                not receipt["source_template"] for receipt in h1_consistency
            ),
            "maximum_absolute_error": float(
                max(
                    (receipt.get("absolute_error", 0.0) for receipt in h1_consistency),
                    default=0.0,
                )
            ),
            "source_pixel_hashes_checked": sum(
                receipt.get("pixel_hashes_checked", False)
                for receipt in h1_consistency
            ),
            "source_pixel_hashes_matched": all(
                receipt.get("pixel_hashes_match_source", True)
                for receipt in h1_consistency
            ),
            "source_render_provenance_match": all(
                receipt.get("source_render_provenance_match", True)
                for receipt in h1_consistency
            ),
            "passed": all(receipt["passed"] for receipt in h1_consistency),
        },
        "zero_contact_steps": sum(audit["total_contact_steps"] for audit in all_audits)
        == 0,
        "maximum_arbiter_count": max(
            audit["maximum_arbiter_count"] for audit in all_audits
        ),
        "all_bounds_valid": all(audit["bounds_ok"] for audit in all_audits),
        "x0_rgb_hash_multisets_identical_across_modes": x0_balanced,
        "x0_rgb_static_bayes_accuracy_upper_bound": 0.5 if x0_balanced else 1.0,
        "pairs": reports,
        "passed": bool(
            len(reports) == pair_count
            and len(episode_metadata) == pair_count * 2
            and len(query_hashes) == pair_count
            and all(audit["passed"] for audit in all_audits)
            and all(receipt["passed"] for receipt in h1_consistency)
            and x0_balanced
            and max(audit["maximum_arbiter_count"] for audit in all_audits) == 0
            and all(audit["bounds_ok"] for audit in all_audits)
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"{split} did not satisfy the K=10 rollout contract")
    return result


def _cross_split_audit(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    overlaps: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                "query_pixels": len(
                    set(reports[left]["query_hashes"])
                    & set(reports[right]["query_hashes"])
                ),
                "template_ids": len(
                    {row["template_id"] for row in reports[left]["pairs"]}
                    & {row["template_id"] for row in reports[right]["pairs"]}
                ),
            }
    return {
        "overlap_counts": overlaps,
        "passed": all(value == 0 for row in overlaps.values() for value in row.values()),
    }


def _read_source(path: Path) -> tuple[dict[str, Any], str]:
    source_hash = _sha256_file(path)
    if source_hash != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "source manifest does not match the frozen v4 release: "
            f"expected {EXPECTED_SOURCE_MANIFEST_SHA256}, got {source_hash}"
        )
    source = json.loads(path.read_text())
    if not source.get("passed"):
        raise RuntimeError("frozen K=1 source manifest did not pass")
    for split in SPLITS:
        if split not in source.get("splits", {}):
            raise RuntimeError(f"source manifest lacks {split}")
        if split not in source.get("split_catalog_seeds", {}):
            raise RuntimeError(f"source manifest lacks catalog seed for {split}")
    return source, source_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--train-pairs", type=int, default=8192)
    parser.add_argument("--loader-validation-pairs", type=int, default=256)
    parser.add_argument("--validation-pairs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=None,
        help=(
            "Optional local filesystem used while Lance commits its files. "
            "Use this for network mounts that do not implement Lance's atomic "
            "file-commit operation; the completed tree is verified and copied "
            "to --output only after all audits pass."
        ),
    )
    parser.add_argument(
        "--max-catalog-groups",
        type=int,
        default=100_000,
        help="Fail rather than scan indefinitely when long-horizon physics rejects candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = _safe_output(args.output)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.resolution < 16:
        raise ValueError("--resolution must be at least 16")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    counts = {
        "train": int(args.train_pairs),
        "loader_validation": int(args.loader_validation_pairs),
        "validation": int(args.validation_pairs),
    }
    if any(value <= 0 or value % 2 for value in counts.values()):
        raise ValueError("each split needs a positive, even template-pair count")
    source_path = args.source_manifest.expanduser().resolve()
    source, source_hash = _read_source(source_path)
    source_catalogs = {split: _source_catalog(source, split) for split in SPLITS}
    request = {
        "protocol": PROTOCOL,
        "source": {
            "manifest": str(source_path.relative_to(ROOT))
            if source_path.is_relative_to(ROOT)
            else source_path.name,
            "manifest_sha256": source_hash,
            "source_protocol": source["protocol"],
            "source_split_catalog_seeds": source["split_catalog_seeds"],
        },
        "pair_counts": counts,
        "resolution": int(args.resolution),
        "jpeg_quality": int(args.jpeg_quality),
        "workers": int(args.workers),
        "anchored_clip_contract": {
            "start_step": 0,
            "history_size": HISTORY_SIZE,
            "max_prediction_horizon": MAX_HORIZON,
            "raw_rows_per_episode": RAW_ROWS_PER_EPISODE,
            "raw_action_layout": {
                "history": [0, 9],
                "query_h1": [10, 14],
                "zero_holds_h2_to_h10": [15, 59],
                "zero_completion_after_h10": [60, 64],
            },
            "model_frame_rows": list(MODEL_FRAME_ROWS),
            "lance_layout": {
                "frames_table": "<split>.lance",
                "episode_metadata_table": "<split>_episodes.lance",
                "per_step_string_columns": 0,
                "stablewm_native_episode_data_layout": True,
            },
            "direct_unmodified_stablewm_reader": {
                "num_preds_equals_10": "exactly_one_start_step_zero_clip_per_episode",
                "num_preds_less_than_10": (
                    "produces sliding windows; those are not the query-anchored "
                    "K=1..K evaluation contract"
                ),
            },
        },
        "physics_contract": {
            "continuous_simulator": "one simulator from x0 through raw step 65",
            "state_installations_after_x0": 0,
            "query_simulator_recreated": False,
            "all_65_steps_contact_free": True,
            "all_65_steps_arbiter_free": True,
            "all_65_steps_in_bounds": True,
        },
    }
    staging_root = (
        args.staging_root.expanduser().resolve()
        if args.staging_root is not None
        else output.parent
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(staging_root))
    )
    published_output_created = False
    try:
        reports: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            print(f"[rollout-k10] selecting/building {split}: {counts[split]} pairs", flush=True)
            reports[split] = _build_split(
                root=temporary,
                split=split,
                pair_count=counts[split],
                source_catalog=source_catalogs[split],
                catalog_seed=int(source["split_catalog_seeds"][split]),
                resolution=int(args.resolution),
                jpeg_quality=int(args.jpeg_quality),
                workers=int(args.workers),
                max_catalog_groups=int(args.max_catalog_groups),
            )
        cross_split = _cross_split_audit(reports)
        manifest = {
            **request,
            "request_sha256": _canonical_sha256(request),
            "splits": reports,
            "cross_split_audit": cross_split,
            "passed": bool(cross_split["passed"] and all(report["passed"] for report in reports.values())),
        }
        if not manifest["passed"]:
            raise RuntimeError("K=10 artifact failed its final audit")
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        summary = {
            "protocol": PROTOCOL,
            "status": "passed",
            "manifest": "manifest.json",
            "manifest_sha256": _sha256_file(manifest_path),
            "pair_counts": counts,
            "source_manifest_sha256": source_hash,
            "cross_split_audit": cross_split,
            "split_summary": {
                split: {
                    key: reports[split][key]
                    for key in (
                        "pair_count",
                        "episode_count",
                        "raw_rows",
                        "table_bytes",
                        "table_sha256",
                        "episode_table_rows",
                        "episode_table_bytes",
                        "episode_table_sha256",
                        "stablewm_native_episode_metadata_layout",
                        "horizon_gap_block_position_px",
                        "h1_source_consistency",
                        "zero_contact_steps",
                        "maximum_arbiter_count",
                        "all_bounds_valid",
                    )
                }
                for split in SPLITS
            },
        }
        (temporary / "build_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        # ``mkdtemp`` deliberately starts at 0700.  The completed artifact is
        # a portable dataset directory, so make its root traversable before
        # either renaming it locally or copying it to a shared mount.
        temporary.chmod(0o755)
        if temporary.parent == output.parent:
            # ``rename`` is atomic because both paths are on one filesystem.
            os.replace(temporary, output)
        else:
            # Some mounted filesystems allow ordinary copies/renames but do
            # not implement the file-level commit primitive used by Lance.
            # Build Lance locally, copy into a hidden sibling, verify every
            # byte, then expose the completed directory with one rename.
            print(
                f"[rollout-k10] publishing verified tree to {output}", flush=True
            )
            output.mkdir()
            published_output_created = True
            incomplete = output / ".INCOMPLETE"
            incomplete.write_text(
                "ContextWorld K=10 staging copy is not yet verified.\n"
            )
            shutil.copytree(
                temporary,
                output,
                dirs_exist_ok=True,
                copy_function=shutil.copy2,
            )
            # The marker is deliberately outside the staged tree and remains
            # present until copying has finished.  Remove it immediately
            # before comparing the two otherwise-identical trees.
            incomplete.unlink()
            if _sha256_tree(temporary) != _sha256_tree(output):
                raise RuntimeError("staging-to-output copy verification failed")
            shutil.rmtree(temporary)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        if published_output_created:
            shutil.rmtree(output, ignore_errors=True)
        raise
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
