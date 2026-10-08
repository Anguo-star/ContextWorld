"""Build visible, replay-anchored multistep PushT action-strength futures.

Each original candidate slot gets the largest registered magnitude factor whose
entire agent and block geometry stays inside the 512px canvas for both hidden
gains at every one of the 25 raw future steps. The historical pair and source
ordering remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "PYOPENGL_PLATFORM"):
    os.environ.setdefault(_name, "osmesa" if _name.startswith(("MUJOCO", "PYOPENGL")) else "1")

FACTORS = (1.0, 0.75, 0.5, 0.25, 0.125, 0.0625, 0.0)
CONDITIONS = ("low_gain", "high_gain")
CONDITION_SCALES = (60.0, 140.0)
CANVAS_SIZE = 512.0
RAW_FUTURE_STEPS = 25
MODEL_FRAME_STEPS = (4, 9, 14, 19, 24)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def install_import_paths(contextworld_root: str, stable_worldmodel_root: str) -> None:
    for path in (contextworld_root, stable_worldmodel_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.modules.setdefault("flash_attn", None)


def base_candidate_bank(native: Any) -> Any:
    import numpy as np

    native = np.asarray(native, dtype=np.float32)
    if native.shape != (5, 2):
        raise ValueError(f"Expected the original native query block to be (5, 2), got {native.shape}")
    if not np.isfinite(native).all() or (np.abs(native) > 1.0 + 1e-6).any():
        raise ValueError("Original query actions must be finite and within [-1, 1]")

    # Preserve the release's eleven candidate slots and their construction order.
    banks = [np.tile(np.clip(scale * native, -1.0, 1.0), (5, 1, 1)) for scale in (0, 0.5, 1, 1.5, 2, -1)]
    for count in (1, 2, 3, 4):
        banks.append(np.concatenate((np.tile(native, (count, 1, 1)), np.zeros((5 - count, 5, 2), dtype=np.float32))))
    banks.append(np.concatenate((native[None], np.tile(-native, (4, 1, 1)))))
    bank = np.asarray(banks, dtype=np.float32)
    bank[bank == 0.0] = np.float32(0.0)
    if bank.shape != (11, 5, 5, 2):
        raise AssertionError(f"Unexpected legacy candidate bank shape {bank.shape}")
    return bank


def action_key(actions: Any) -> bytes:
    import numpy as np

    canonical = np.array(actions, dtype=np.float32, order="C", copy=True)
    canonical[canonical == 0.0] = np.float32(0.0)
    return np.ascontiguousarray(canonical).tobytes(order="C")


def unique_action_bank(bank: Any) -> tuple[Any, Any, Any]:
    """Deduplicate final scaled slots by exact float32 action bytes."""
    import numpy as np

    unique: list[Any] = []
    key_to_index: dict[bytes, int] = {}
    mapping: list[int] = []
    for actions in bank:
        key = action_key(actions)
        if key not in key_to_index:
            key_to_index[key] = len(unique)
            unique.append(np.ascontiguousarray(actions, dtype=np.float32))
        mapping.append(key_to_index[key])
    unique_bank = np.asarray(unique, dtype=np.float32)
    mapping_array = np.asarray(mapping, dtype=np.int64)
    counts = np.bincount(mapping_array, minlength=len(unique_bank)).astype(np.int64)
    return unique_bank, mapping_array, counts


def one_object_bounds(body: Any, pymunk: Any) -> Any:
    """Return canvas-space bounds for every Pymunk collision shape on a body."""
    import numpy as np

    shape_bounds = []
    for shape in body.shapes:
        radius = float(getattr(shape, "radius", 0.0))
        if isinstance(shape, pymunk.Circle):
            center = body.local_to_world(shape.offset)
            center_x = float(center.x)
            center_y = CANVAS_SIZE - float(center.y)
            shape_bounds.append((center_x - radius, center_y - radius, center_x + radius, center_y + radius))
        elif isinstance(shape, pymunk.Poly):
            vertices = [body.local_to_world(vertex) for vertex in shape.get_vertices()]
            if not vertices:
                raise ValueError("PushT polygon has no vertices")
            points = np.asarray([[float(vertex.x), CANVAS_SIZE - float(vertex.y)] for vertex in vertices], dtype=np.float64)
            shape_bounds.append((
                float(points[:, 0].min()) - radius,
                float(points[:, 1].min()) - radius,
                float(points[:, 0].max()) + radius,
                float(points[:, 1].max()) + radius,
            ))
        else:
            raise TypeError(f"Unsupported PushT collision shape {type(shape).__name__}; refusing an incomplete geometry check")
    if not shape_bounds:
        raise ValueError("PushT body has no collision shapes")
    bounds = np.asarray(shape_bounds, dtype=np.float64)
    return np.asarray((bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()), dtype=np.float64)


def objects_bounds(env: Any) -> Any:
    import numpy as np
    import pymunk

    return np.stack((one_object_bounds(env.agent, pymunk), one_object_bounds(env.block, pymunk))).astype(np.float64)


def within_canvas(bounds: Any, margin: float) -> bool:
    import numpy as np

    return bool(
        np.all(bounds[..., 0] >= margin)
        and np.all(bounds[..., 1] >= margin)
        and np.all(bounds[..., 2] <= CANVAS_SIZE - margin)
        and np.all(bounds[..., 3] <= CANVAS_SIZE - margin)
    )


def render_decoded(env: Any) -> Any:
    import numpy as np
    from contextworld.synthesis.lance import encode_frame
    from contextworld.benchmarks.action_strength_icl_data import _decode_rgb

    return _decode_rgb(encode_frame(np.asarray(env.render()), dict(format="jpeg", quality=95)))


def prefix_checkpoint(env: Any, ref: dict[str, Any], condition_index: int, frame_index: int) -> tuple[int, float]:
    import numpy as np

    rendered = render_decoded(env)
    delta = np.abs(rendered.astype(np.int16) - ref["pixels"][condition_index, frame_index].astype(np.int16))
    pixel_gap = int(delta.max(initial=0))
    observed_state = np.asarray(env._get_obs(), dtype=np.float64)
    expected_state = np.asarray(ref["states"][condition_index, frame_index], dtype=np.float64)
    if observed_state.shape != expected_state.shape:
        raise AssertionError(f"Prefix state shape mismatch: {observed_state.shape} != {expected_state.shape}")
    state_gap = float(np.abs(observed_state - expected_state).max(initial=0.0))
    if pixel_gap != 0:
        raise AssertionError(f"Prefix pixels differ for condition {condition_index}, frame {frame_index}: max delta {pixel_gap}")
    if state_gap > 1e-4:
        raise AssertionError(f"Prefix state differs for condition {condition_index}, frame {frame_index}: max gap {state_gap}")
    return pixel_gap, state_gap


def replay_prefix(env: Any, template: Any, ref: dict[str, Any], condition_index: int) -> dict[str, Any]:
    import numpy as np
    from contextworld.evaluation.pusht_replay_matched_hidden_actuation import _variation_values
    from contextworld.evaluation.pusht_hidden_actuation import _body_snapshot

    env.reset(
        seed=int(template.simulator_seed),
        options=dict(
            variation=(),
            variation_values=_variation_values(template),
            state=template.reset_state,
            goal_state=template.goal_state,
        ),
    )
    pixel_gaps = []
    state_gaps = []
    gap = prefix_checkpoint(env, ref, condition_index, 0)
    pixel_gaps.append(gap[0])
    state_gaps.append(gap[1])
    for raw_index, action in enumerate(np.asarray(ref["actions"][:2], dtype=np.float32).reshape(-1, 2)):
        env.step(action)
        if raw_index in (4, 9):
            frame_index = 1 if raw_index == 4 else 2
            gap = prefix_checkpoint(env, ref, condition_index, frame_index)
            pixel_gaps.append(gap[0])
            state_gaps.append(gap[1])
    return dict(
        query_body=np.asarray(_body_snapshot(env), dtype=np.float64),
        prefix_pixel_gaps=pixel_gaps,
        prefix_state_gaps=state_gaps,
    )


def measure_factor_condition(template: Any, ref: dict[str, Any], condition_index: int, actions: Any, margin: float) -> dict[str, Any]:
    """Replay history and check all 25 raw steps, even after first boundary failure."""
    from stable_worldmodel.envs.pusht.env import PushT
    from contextworld.evaluation.pusht_hidden_actuation import MODE_SCALES

    env = PushT(resolution=224, with_target=True, render_mode="rgb_array")
    env.action_scale = float(MODE_SCALES[CONDITIONS[condition_index]])
    first_failure = None
    all_safe = True
    aggregate_min = [float("inf"), float("inf"), float("inf"), float("inf")]
    aggregate_max = [float("-inf"), float("-inf"), float("-inf"), float("-inf")]
    try:
        prefix = replay_prefix(env, template, ref, condition_index)
        for raw_index, action in enumerate(actions.reshape(RAW_FUTURE_STEPS, 2)):
            env.step(action)
            bounds = objects_bounds(env)
            aggregate_min = [min(a, float(b)) for a, b in zip(aggregate_min, bounds.reshape(-1, 4).min(axis=0))]
            aggregate_max = [max(a, float(b)) for a, b in zip(aggregate_max, bounds.reshape(-1, 4).max(axis=0))]
            step_safe = within_canvas(bounds, margin)
            if not step_safe:
                all_safe = False
                if first_failure is None:
                    first_failure = dict(raw_step=raw_index + 1, bounds=bounds.tolist())
    finally:
        env.close()
    return dict(
        safe=all_safe,
        raw_steps_completed=RAW_FUTURE_STEPS,
        first_failure=first_failure,
        aggregate_min_bounds=aggregate_min,
        aggregate_max_bounds=aggregate_max,
        query_body=prefix["query_body"],
        prefix_pixel_gaps=prefix["prefix_pixel_gaps"],
        prefix_state_gaps=prefix["prefix_state_gaps"],
    )


def simulate_selected_condition(template: Any, ref: dict[str, Any], condition_index: int, actions: Any, margin: float) -> dict[str, Any]:
    """Generate stored pixels, state after every raw step, and all geometry bounds."""
    import numpy as np
    from stable_worldmodel.envs.pusht.env import PushT
    from contextworld.evaluation.pusht_hidden_actuation import MODE_SCALES

    env = PushT(resolution=224, with_target=True, render_mode="rgb_array")
    env.action_scale = float(MODE_SCALES[CONDITIONS[condition_index]])
    try:
        prefix = replay_prefix(env, template, ref, condition_index)
        raw_states = []
        geometry_bounds = []
        future_pixels = []
        for raw_index, action in enumerate(np.asarray(actions, dtype=np.float32).reshape(RAW_FUTURE_STEPS, 2)):
            env.step(action)
            geometry_bounds.append(objects_bounds(env))
            raw_states.append(np.asarray(env._get_obs(), dtype=np.float64).copy())
            if (raw_index + 1) % 5 == 0:
                future_pixels.append(render_decoded(env))
        raw_states_array = np.asarray(raw_states, dtype=np.float64)
        sampled_states = raw_states_array[np.asarray(MODEL_FRAME_STEPS, dtype=np.int64)]
        future_states_array = np.asarray([raw_states_array[step] for step in MODEL_FRAME_STEPS], dtype=np.float64)
        if not np.array_equal(sampled_states, future_states_array):
            raise AssertionError("Five-step future states do not align with raw steps 5,10,15,20,25")
        bounds_array = np.asarray(geometry_bounds, dtype=np.float64)
        return dict(
            safe=bool(within_canvas(bounds_array, margin)),
            raw_steps_completed=RAW_FUTURE_STEPS,
            query_body=prefix["query_body"],
            prefix_pixel_gaps=prefix["prefix_pixel_gaps"],
            prefix_state_gaps=prefix["prefix_state_gaps"],
            raw_states=raw_states_array,
            geometry_bounds=bounds_array,
            future_pixels=np.asarray(future_pixels, dtype=np.uint8),
            future_states=future_states_array,
        )
    finally:
        env.close()


def build_pair(job: tuple[Any, ...]) -> dict[str, Any]:
    import numpy as np
    from contextworld.evaluation.pusht_replay_matched_hidden_actuation import ReplayMatchedHiddenActuationTemplate

    source_index, template_json, ref, output_dir, margin, contextworld_root, stable_worldmodel_root = job
    template = ReplayMatchedHiddenActuationTemplate(**template_json)
    base_bank = base_candidate_bank(ref["actions"][2])
    factor_search_safe = np.zeros((len(base_bank), len(FACTORS)), dtype=np.bool_)
    factor_trials: list[list[dict[str, Any]]] = [[] for _ in range(len(base_bank))]
    geometry_cache: dict[tuple[int, bytes], dict[str, Any]] = {}
    prefix_by_condition: list[dict[str, Any] | None] = [None, None]
    geometry_budget = dict(unique_action_patterns=0, condition_rollouts=0, raw_steps=0)

    # Audit every registered factor for every original slot. Exact action-byte
    # repeats can reuse their deterministic physical safety receipt.
    for factor_index, factor in enumerate(FACTORS):
        scaled_bank = np.ascontiguousarray(base_bank * np.float32(factor), dtype=np.float32)
        scaled_bank[scaled_bank == 0.0] = np.float32(0.0)
        grouped: dict[bytes, list[int]] = {}
        for slot, actions in enumerate(scaled_bank):
            grouped.setdefault(action_key(actions), []).append(slot)
        for key, slots in grouped.items():
            condition_results = []
            actions = scaled_bank[slots[0]]
            for condition_index in range(2):
                cache_key = (condition_index, key)
                if cache_key not in geometry_cache:
                    geometry_cache[cache_key] = measure_factor_condition(template, ref, condition_index, actions, margin)
                    geometry_budget["unique_action_patterns"] += 1
                    geometry_budget["condition_rollouts"] += 1
                    geometry_budget["raw_steps"] += RAW_FUTURE_STEPS
                result = geometry_cache[cache_key]
                if prefix_by_condition[condition_index] is None:
                    prefix_by_condition[condition_index] = dict(
                        pixel_gaps=result["prefix_pixel_gaps"],
                        state_gaps=result["prefix_state_gaps"],
                    )
                condition_results.append(result)
            safe = bool(condition_results[0]["safe"] and condition_results[1]["safe"])
            for slot in slots:
                factor_search_safe[slot, factor_index] = safe
                factor_trials[slot].append(dict(
                    factor=float(factor),
                    factor_safe=safe,
                    condition_safe=[bool(item["safe"]) for item in condition_results],
                    first_outside=[item["first_failure"] for item in condition_results],
                ))

    candidate_safe = factor_search_safe.any(axis=1)
    selected_indices = np.asarray([
        int(np.flatnonzero(factor_search_safe[slot])[0]) if candidate_safe[slot] else len(FACTORS) - 1
        for slot in range(len(base_bank))
    ], dtype=np.int64)
    candidate_factor = np.asarray([FACTORS[index] for index in selected_indices], dtype=np.float32)
    candidate_slot_actions = np.ascontiguousarray(base_bank * candidate_factor[:, None, None, None], dtype=np.float32)
    candidate_slot_actions[candidate_slot_actions == 0.0] = np.float32(0.0)
    unique_bank, slot_to_unique, unique_counts = unique_action_bank(candidate_slot_actions)

    # Store one full trajectory per final unique candidate. Legacy slots use a
    # separate map and never expand the model-facing candidate axis.
    unique_results = []
    full_rollout_budget = dict(unique_candidates=len(unique_bank), condition_rollouts=0, raw_steps=0)
    query_state_gap = 0.0
    for unique_index, actions in enumerate(unique_bank):
        condition_results = []
        for condition_index in range(2):
            result = simulate_selected_condition(template, ref, condition_index, actions, margin)
            full_rollout_budget["condition_rollouts"] += 1
            full_rollout_budget["raw_steps"] += RAW_FUTURE_STEPS
            if result["raw_steps_completed"] != RAW_FUTURE_STEPS:
                raise AssertionError("Selected trajectory did not complete all 25 raw steps")
            condition_results.append(result)
        gap = float(np.abs(condition_results[0]["query_body"] - condition_results[1]["query_body"]).max(initial=0.0))
        query_state_gap = max(query_state_gap, gap)
        if gap >= 1e-5:
            raise AssertionError(f"Query body differs across hidden conditions by {gap}")
        for slot in np.flatnonzero(slot_to_unique == unique_index):
            expected_safe = bool(candidate_safe[slot])
            actual_safe = bool(condition_results[0]["safe"] and condition_results[1]["safe"])
            if expected_safe != actual_safe:
                raise AssertionError(f"Safety search/full trajectory mismatch for slot {slot}: {expected_safe} vs {actual_safe}")
        unique_results.append(condition_results)

    future_pixels = np.stack([
        np.stack([unique_results[unique_index][condition]["future_pixels"] for unique_index in range(len(unique_bank))])
        for condition in range(2)
    ]).astype(np.uint8)
    future_step_states = np.stack([
        np.stack([unique_results[unique_index][condition]["raw_states"] for unique_index in range(len(unique_bank))])
        for condition in range(2)
    ]).astype(np.float64)
    future_geometry_bounds = np.stack([
        np.stack([unique_results[unique_index][condition]["geometry_bounds"] for unique_index in range(len(unique_bank))])
        for condition in range(2)
    ]).astype(np.float64)
    future_states = future_step_states[:, :, np.asarray(MODEL_FRAME_STEPS, dtype=np.int64), :]
    if future_step_states.shape != (2, len(unique_bank), 25, 7) or future_geometry_bounds.shape != (2, len(unique_bank), 25, 2, 4):
        raise AssertionError("Unexpected raw future state or geometry-bound shape")
    if not np.array_equal(future_states, future_step_states[:, :, np.asarray(MODEL_FRAME_STEPS, dtype=np.int64), :]):
        raise AssertionError("future_states do not equal raw states at steps 5,10,15,20,25")

    # Legacy native slot replay remains an exact identity check whenever slot 2
    # retained factor 1.0. Other selected factors deliberately alter that future.
    native_pixels_exact = False
    max_native_state_error = None
    if candidate_factor[2] == np.float32(1.0):
        native_pixel_gaps = []
        native_state_gaps = []
        for condition in range(2):
            native_pixel_gaps.append(int(np.abs(
                future_pixels[condition, int(slot_to_unique[2]), 0].astype(np.int16)
                - ref["pixels"][condition, 3].astype(np.int16)
            ).max(initial=0)))
            native_state_gaps.append(float(np.abs(
                future_states[condition, int(slot_to_unique[2]), 0] - ref["states"][condition, 3]
            ).max(initial=0.0)))
        native_pixels_exact = all(gap == 0 for gap in native_pixel_gaps)
        max_native_state_error = max(native_state_gaps)
        if not native_pixels_exact or max_native_state_error > 1e-4:
            raise AssertionError(f"Legacy native future replay mismatch: pixel gaps={native_pixel_gaps}, state gap={max_native_state_error}")

    goal_state = np.asarray(ref["states"][0, 3], dtype=np.float64)
    delta = np.abs(future_states[..., 4] - goal_state[4]) % (2 * np.pi)
    angle_error = np.minimum(delta, 2 * np.pi - delta)
    physical_cost = np.sqrt(np.sum((future_states[..., :4] - goal_state[:4]) ** 2, axis=-1) + (40.0 * angle_error) ** 2)
    # Unsafe zero-factor fallbacks are retained and explicitly reported.
    for slot in range(11):
        actual_slot_safe = bool(within_canvas(future_geometry_bounds[:, int(slot_to_unique[slot])], margin))
        if actual_slot_safe != bool(candidate_safe[slot]):
            raise AssertionError(f"Selected safety mismatch for candidate slot {slot}")

    unique_weights = np.full(len(unique_bank), 1.0 / len(unique_bank), dtype=np.float32)
    out_dir = Path(output_dir)
    path = out_dir / f"pair_{source_index:04d}.npz"
    np.savez_compressed(
        path,
        history_pixels=np.asarray(ref["pixels"][:, :3], dtype=np.uint8),
        context_actions=np.repeat(np.asarray(ref["actions"][:2], dtype=np.float32)[None], 2, axis=0),
        candidate_actions=unique_bank,
        candidate_slot_actions=candidate_slot_actions,
        candidate_parent_index=np.arange(11, dtype=np.int64),
        candidate_factor=candidate_factor,
        candidate_safe=candidate_safe.astype(np.bool_),
        factor_search_safe=factor_search_safe.astype(np.bool_),
        candidate_slot_to_unique=slot_to_unique.astype(np.int64),
        candidate_unique_index=slot_to_unique.astype(np.int64),
        candidate_unique_weight=unique_weights,
        future_pixels=future_pixels,
        future_states=future_states,
        future_step_states=future_step_states,
        future_geometry_bounds=future_geometry_bounds,
        goal_pixels=np.asarray(ref["pixels"][0, 3], dtype=np.uint8),
        goal_state=goal_state,
        physical_cost=physical_cost,
        conditions=np.asarray(CONDITION_SCALES, dtype=np.float32),
        physical_steps=np.arange(5, 26, 5, dtype=np.int64),
    )

    prefix_pixel_exact = [all(gap == 0 for gap in item["pixel_gaps"]) for item in prefix_by_condition]
    prefix_state_gap = [max(item["state_gaps"]) for item in prefix_by_condition]
    trajectory_receipt = dict(
        replayed_original_history=True,
        query_state_injected=False,
        all_candidates_continuous=True,
        future_raw_steps=RAW_FUTURE_STEPS,
        factor_search_complete=True,
        factor_search_factors=list(FACTORS),
        prefix_pixels_exact=prefix_pixel_exact,
        prefix_state_gap=prefix_state_gap,
        prefix_pixel_max_abs_by_checkpoint=[item["pixel_gaps"] for item in prefix_by_condition],
        prefix_state_gap_by_checkpoint=[item["state_gaps"] for item in prefix_by_condition],
        query_body_gap_across_conditions=query_state_gap,
    )
    slot_records = []
    for slot in range(11):
        slot_records.append(dict(
            slot_index=slot,
            parent_index=slot,
            selected_factor=float(candidate_factor[slot]),
            candidate_safe=bool(candidate_safe[slot]),
            factor_search_safe=factor_search_safe[slot].tolist(),
            selected_action_sha256=hashlib.sha256(action_key(candidate_slot_actions[slot])).hexdigest(),
            unique_index=int(slot_to_unique[slot]),
            unique_multiplicity=int(unique_counts[slot_to_unique[slot]]),
            trials=factor_trials[slot],
        ))
    factor_search_receipt = dict(
        factors=list(FACTORS),
        factor_search_complete=True,
        selected_factor_is_largest_safe_factor_or_zero_fallback=True,
        candidate_safe=candidate_safe.tolist(),
        factor_search_safe=factor_search_safe.tolist(),
        geometry_acceptance="all Pymunk shapes for agent and block inside [margin,512-margin] for both hidden conditions at every one of 25 raw steps",
        margin_px=float(margin),
        geometry_cache_key="exact contiguous float32 action bytes within source pair and hidden condition",
        search_budget=geometry_budget,
        final_trajectory_budget=full_rollout_budget,
        candidates=slot_records,
    )
    scene_failures = [dict(
        scene_id=template.template_id,
        source_index=source_index,
        slot_index=slot,
        parent_index=slot,
        selected_factor=float(candidate_factor[slot]),
        fallback_factor_used=not bool(candidate_safe[slot]),
        factor_search_safe=factor_search_safe[slot].tolist(),
        failure="no_registered_factor_keeps_every_shape_inside_margin_for_both_conditions_and_all_25_raw_steps",
    ) for slot in range(11) if not candidate_safe[slot]]
    return dict(
        ok=True,
        scene_id=template.template_id,
        source_index=source_index,
        path=path.name,
        sha256=sha256_file(path),
        native_pixels_exact=native_pixels_exact,
        max_native_state_error=max_native_state_error,
        query_state_gap=query_state_gap,
        candidate_factors=candidate_factor.tolist(),
        candidate_safe=candidate_safe.tolist(),
        candidate_slot_to_unique=slot_to_unique.tolist(),
        candidate_axis_semantics="candidate_actions/future arrays use unique K axis; legacy 11 slots are in candidate_slot_actions",
        candidate_unique_count=int(len(unique_bank)),
        candidate_unique_multiplicities=unique_counts.tolist(),
        factor_search_receipt=factor_search_receipt,
        trajectory_receipt=trajectory_receipt,
        failures=scene_failures,
    )


def build_pair_worker(job: tuple[Any, ...]) -> dict[str, Any]:
    source_index, template, ref, output_dir, margin, contextworld_root, stable_worldmodel_root = job
    install_import_paths(contextworld_root, stable_worldmodel_root)
    return build_pair((source_index, template, ref, output_dir, margin, contextworld_root, stable_worldmodel_root))


def strength(
    *,
    limit: int,
    workers: int,
    output: Path,
    margin: float,
    contextworld_root: Path,
    stable_worldmodel_root: Path,
    payload_root: Path,
    legacy_panel_manifest: Path | None = None,
    bootstrap_clusters_manifest: Path | None = None,
) -> dict[str, Any]:
    import numpy as np

    if not 1 <= limit <= 256:
        raise ValueError("limit must be between 1 and 256")
    if workers < 1:
        raise ValueError("workers must be positive")
    if margin < 0.0 or margin >= CANVAS_SIZE / 2:
        raise ValueError("margin must lie in [0, 256)")
    contextworld_root = Path(contextworld_root).resolve()
    stable_worldmodel_root = Path(stable_worldmodel_root).resolve()
    payload_root = Path(payload_root).resolve()
    output = Path(output).resolve()
    legacy_panel_manifest = Path(legacy_panel_manifest).resolve()
    for name, path in (
        ("--contextworld-root", contextworld_root),
        ("--stable-worldmodel-root", stable_worldmodel_root),
        ("--payload-root", payload_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{name} must name an existing directory: {path}")
    if not legacy_panel_manifest.is_file():
        raise FileNotFoundError(
            f"--legacy-panel-manifest must name an existing manifest file: {legacy_panel_manifest}"
        )
    if bootstrap_clusters_manifest is not None:
        bootstrap_clusters_manifest = Path(bootstrap_clusters_manifest).resolve()
        if not bootstrap_clusters_manifest.is_file():
            raise FileNotFoundError(
                f"--bootstrap-clusters-manifest must name an existing file: {bootstrap_clusters_manifest}"
            )
    install_import_paths(str(contextworld_root), str(stable_worldmodel_root))

    from contextworld.benchmarks import bundle_development as bd
    from contextworld.benchmarks.action_strength_icl_data import _read_lance_pairs

    payload = bd.resolve_development_payload(payload_root, task="action_strength")
    arrays = _read_lance_pairs(payload.members[0], expected_pairs=256)
    source_manifest = contextworld_root / "artifacts/synthesis/pusht_action_strength_h3_release_v1/manifest.json"
    source_manifest_json = json.loads(source_manifest.read_text(encoding="utf-8"))
    validation_pairs = source_manifest_json["splits"]["validation"]["pairs"]
    validation_ids = [item["template"]["template_id"] for item in validation_pairs]
    templates = {item["template"]["template_id"]: item["template"] for item in validation_pairs}
    pair_ids = list(arrays.pair_ids[:limit])
    if len(pair_ids) != limit or len(validation_ids) < limit:
        raise AssertionError("Development/release validation source does not contain the requested ordered pairs")
    if validation_ids[:limit] != pair_ids:
        raise AssertionError("Development table order differs from release validation order")

    stage = output.with_name(output.name + ".partial")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if stage.exists():
        raise FileExistsError(f"Staging output already exists: {stage}")
    stage.mkdir(parents=True, exist_ok=False)

    jobs = []
    for source_index, pair_id in enumerate(pair_ids):
        ref = dict(
            actions=np.asarray(arrays.raw_action_blocks[source_index]),
            pixels=np.stack((arrays.low_pixels[source_index], arrays.high_pixels[source_index])),
            states=np.stack((arrays.low_states[source_index], arrays.high_states[source_index])),
        )
        jobs.append((
            source_index,
            templates[pair_id],
            ref,
            str(stage),
            float(margin),
            str(contextworld_root),
            str(stable_worldmodel_root),
        ))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(build_pair_worker, jobs))

    scenes = sorted(results, key=lambda item: item["source_index"])
    failures = [failure for scene in scenes for failure in scene["failures"]]
    source_ids = [templates[pair_id]["template_id"] for pair_id in pair_ids]
    unique_counts = [scene["candidate_unique_count"] for scene in scenes]
    bootstrap_metadata_source = dict(available=False)
    legacy_json = json.loads(legacy_panel_manifest.read_text(encoding="utf-8"))
    legacy_by_id = {item["scene_id"]: item for item in legacy_json["scenes"]}
    cluster_by_id = {}
    if bootstrap_clusters_manifest is not None:
        cluster_json = json.loads(bootstrap_clusters_manifest.read_text(encoding="utf-8"))
        cluster_by_id = cluster_json.get("action_strength", {})
    for item in scenes:
        old = legacy_by_id.get(item["scene_id"])
        if old is None:
            raise AssertionError(f"Legacy panel has no source metadata for {item['scene_id']}")
        cluster = old.get("bootstrap_cluster", cluster_by_id.get(item["scene_id"]))
        if item["scene_id"] in cluster_by_id and cluster_by_id[item["scene_id"]] != cluster:
            raise AssertionError(f"Bootstrap cluster differs between legacy sources for {item['scene_id']}")
        group = old.get("source_group", cluster)
        episode = old.get("source_episode")
        episode_source = "legacy_panel_manifest"
        if episode is None and isinstance(cluster, str) and cluster.startswith("episode:"):
            episode = int(cluster.split(":", 1)[1])
            episode_source = "parsed_from_legacy_bootstrap_cluster"
        item.update(bootstrap_cluster=cluster, source_group=group, source_episode=episode)
        item["source_metadata_provenance"] = dict(
            bootstrap_cluster="legacy action_strength panel",
            source_group="legacy source_group field" if old.get("source_group") is not None else "copied bootstrap_cluster label",
            source_episode=episode_source,
        )
    for item in failures:
        old = legacy_by_id.get(item["scene_id"], {})
        cluster = old.get("bootstrap_cluster", cluster_by_id.get(item["scene_id"]))
        episode = old.get("source_episode")
        if episode is None and isinstance(cluster, str) and cluster.startswith("episode:"):
            episode = int(cluster.split(":", 1)[1])
        item.update(bootstrap_cluster=cluster, source_group=old.get("source_group", cluster), source_episode=episode)
    bootstrap_metadata_source = dict(
        available=True,
        legacy_panel_manifest=str(legacy_panel_manifest),
        legacy_panel_manifest_sha256=sha256_file(legacy_panel_manifest),
        bootstrap_clusters_manifest=str(bootstrap_clusters_manifest) if bootstrap_clusters_manifest is not None else None,
        bootstrap_clusters_manifest_sha256=sha256_file(bootstrap_clusters_manifest) if bootstrap_clusters_manifest is not None else None,
        source_group_derivation="legacy source_group if present, otherwise exact legacy bootstrap_cluster label",
        source_episode_derivation="legacy source_episode if present, otherwise integer parsed from legacy episode:<id> cluster label",
    )
    manifest = dict(
        task="action_strength",
        normalization=dict(zip(("mean", "std"), bd.development_action_normalization(payload))),
        scenes=scenes,
        failures=failures,
        selection=f"first{limit} ordered Development pairs; every source pair retained, unsafe candidate fallbacks are explicit failures",
        evaluation_split="development",
        goal_rule="low_gain native five-step future, shared across conditions; unchanged from legacy panel",
        candidates="model-facing candidate_actions/future arrays contain exact unique K action sequences; legacy 11 slots are preserved in candidate_slot_actions with slot-to-unique mapping",
        candidate_base_slots="amplitudes 0,.5,1,1.5,2,-1; native 1..4 blocks then zero; native 1 then reverse 4; original candidate sequences are clipped to [-1,1] before factor multiplication",
        candidate_factor_ladder=list(FACTORS),
        candidate_factor_selected_per_slot=True,
        model_candidate_axis="unique K; duplicate legacy slots cannot increase evaluation weight",
        legacy_candidate_slots=11,
        candidate_actions_shared_across_hidden_conditions=True,
        geometry_acceptance="all Pymunk agent and block shapes lie within [margin,512-margin] in both hidden conditions at each of 25 raw future steps",
        geometry_margin_px=float(margin),
        margin_applies_to_both_hidden_conditions=True,
        candidate_deduplication="after per-slot factor selection, contiguous float32 action bytes with all signed zeros normalized to +0.0; unique action sequences have equal total weight",
        candidate_budget=dict(
            legacy_slots_per_scene=11,
            source_scenes=len(scenes),
            unsafe_candidate_failures=len(failures),
            unique_action_counts=unique_counts,
            physical_rollout_budget="all slot-factor safety combinations audited; identical action bytes within a source pair and condition reuse one geometry receipt; final unique actions get one full stored trajectory per condition",
        ),
        diagnostic_thresholds=dict(
            identical_image="exact bytewise RGB equality",
            physical_difference_threshold_render_px=2.0,
            physical_difference_threshold_world_px=512.0 * 2.0 / 224.0,
            physical_difference_coordinates="agent/block xy and 40-scaled wrapped block angle equivalent",
            used_as_candidate_selection_gate=False,
        ),
        metric="norm(agent_error_xy,block_error_xy,40*wrapped_block_angle_error)",
        source=str(source_manifest),
        source_sha256=sha256_file(source_manifest),
        builder_sha256=sha256_file(Path(__file__)),
        complete_task_dataset=bool(limit == 256 and not failures),
        all_source_pairs_retained=True,
        action_limits=[-1.0, 1.0],
        physical_steps=[5, 10, 15, 20, 25],
        raw_future_steps=RAW_FUTURE_STEPS,
        source_indices=list(range(limit)),
        source_pair_ids=source_ids,
        source_order_matches_release_validation=True,
        bootstrap_metadata_source=bootstrap_metadata_source,
        source_coverage=dict(
            requested=limit,
            produced=len(scenes),
            explicit_unsafe_candidate_failures=len(failures),
            unique_query_ids=len(set(source_ids)) == len(source_ids),
            development_table_pair_count=len(arrays.pair_ids),
            first_source_index=0,
            last_source_index=limit - 1,
        ),
        builder_inputs=dict(
            contextworld_root=str(contextworld_root),
            stable_worldmodel_snapshot=str(stable_worldmodel_root),
            development_payload_root=str(payload_root),
            workers=workers,
        ),
    )
    write_json(stage / "manifest.json", manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(output)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--contextworld-root", type=Path, required=True)
    parser.add_argument("--stable-worldmodel-root", type=Path, required=True)
    parser.add_argument("--payload-root", type=Path, required=True)
    parser.add_argument("--legacy-panel-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-clusters-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = strength(
        limit=args.limit,
        workers=args.workers,
        output=args.output,
        margin=args.margin,
        contextworld_root=args.contextworld_root,
        stable_worldmodel_root=args.stable_worldmodel_root,
        payload_root=args.payload_root,
        legacy_panel_manifest=args.legacy_panel_manifest,
        bootstrap_clusters_manifest=args.bootstrap_clusters_manifest,
    )
    print(json.dumps({
        "output": str(args.output.resolve()),
        "requested": manifest["source_coverage"]["requested"],
        "produced": manifest["source_coverage"]["produced"],
        "unsafe_candidate_failures": manifest["source_coverage"]["explicit_unsafe_candidate_failures"],
        "complete_task_dataset": manifest["complete_task_dataset"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
