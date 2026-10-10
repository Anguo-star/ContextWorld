#!/usr/bin/env python3
"""Check exact image aliases against selected physical geometry in cached panels.

Read-only. Compares every distinct (condition,candidate) pair within each scene.
Full-trajectory checks hash/group then byte-confirm identical five-frame clips;
single-frame checks group by horizon, never across time. Geometry coordinates
are selected task positions, not the full simulator state.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


TASKS = ('action_strength', 'motion_damping', 'robot_arm_mass')
GEOMETRY_EPSILON = 1e-6  # numerical tolerance only; not a scientific threshold
PUSHT_AGENT_MASS = np.array([0, 1, 2, 3], dtype=np.int64)
PUSHT_AGENT_MOTION = np.array([0, 1, 6, 7], dtype=np.int64)


def physical_geometry(task: str, archive: Any) -> np.ndarray:
    """Return selected geometry as [K,C,T,D] in task-native units."""
    if task == 'robot_arm_mass':
        if 'finger_positions' not in archive.files:
            raise KeyError('robot_arm_mass panel is missing finger_positions')
        values = np.asarray(archive['finger_positions'], dtype=np.float64)
        if values.ndim != 4 or values.shape[-1] != 2:
            raise ValueError(f'bad finger_positions shape: {values.shape}')
        return np.ascontiguousarray(values * 1000.0)  # millimetres

    states = np.asarray(archive['future_states'], dtype=np.float64)
    if states.ndim != 4:
        raise ValueError(f'future_states must be [K,C,T,D], got {states.shape}')
    if states.shape[-1] == 7:
        positions = states[..., PUSHT_AGENT_MASS]
        angle = states[..., 4]
    elif states.shape[-1] >= 12:
        positions = states[..., PUSHT_AGENT_MOTION]
        angle = states[..., 10]
    else:
        raise ValueError(f'unrecognized PushT state width: {states.shape[-1]}')
    geometry = np.concatenate(
        [positions, (40.0 * np.sin(angle))[..., None], (40.0 * np.cos(angle))[..., None]],
        axis=-1,
    )
    if not np.isfinite(geometry).all():
        raise ValueError('physical geometry contains non-finite values')
    return np.ascontiguousarray(geometry, dtype=np.float64)


def pixel_groups(frames: np.ndarray) -> dict[bytes, list[int]]:
    """Group row-major clips/frames by SHA-256; equality is confirmed later."""
    groups: dict[bytes, list[int]] = {}
    for i in range(len(frames)):
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(frames[i]))).digest()
        groups.setdefault(digest, []).append(i)
    return groups


def label_for(archive: Any, manifest: dict[str, Any], task: str, k: int, c: int) -> dict[str, Any]:
    conditions = np.asarray(archive['conditions']) if 'conditions' in archive.files else None
    condition = conditions[k].item() if conditions is not None else k
    names = np.asarray(archive['candidate_names']) if 'candidate_names' in archive.files else None
    candidate_name = str(names[c].item()) if names is not None else None
    return {
        'condition_index': int(k),
        'condition_value': condition,
        'candidate_index': int(c),
        'candidate_name': candidate_name,
    }


def pair_example(scene_id: str, left: int, right: int, C: int, archive: Any,
                 manifest: dict[str, Any], task: str,
                 horizon: int | list[int] | None) -> dict[str, Any]:
    return {
        'scene_id': scene_id,
        'left': label_for(archive, manifest, task, left // C, left % C),
        'right': label_for(archive, manifest, task, right // C, right % C),
        'horizon_raw_steps': horizon,
    }


def new_summary() -> dict[str, Any]:
    return {
        'scene_count': 0,
        'pair_denominator': 0,
        'pixel_identical_pair_count': 0,
        'geometry_different_among_pixel_identical_count': 0,
        'max_rms_all_pairs': 0.0,
        'max_rms_all_pair_example': None,
        'max_rms_among_pixel_identical_pairs': 0.0,
        'max_alias_counterexample': None,
        'max_geometry_different_alias_rms': 0.0,
        'max_geometry_different_alias_counterexample': None,
    }


def merge_summary(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in ('scene_count', 'pair_denominator', 'pixel_identical_pair_count',
                'geometry_different_among_pixel_identical_count'):
        dst[key] += src[key]
    if src['max_rms_all_pairs'] > dst['max_rms_all_pairs']:
        dst['max_rms_all_pairs'] = src['max_rms_all_pairs']
        dst['max_rms_all_pair_example'] = src['max_rms_all_pair_example']
    if src['max_rms_among_pixel_identical_pairs'] > dst['max_rms_among_pixel_identical_pairs']:
        dst['max_rms_among_pixel_identical_pairs'] = src['max_rms_among_pixel_identical_pairs']
        dst['max_alias_counterexample'] = src['max_alias_counterexample']
    if src['max_geometry_different_alias_rms'] > dst['max_geometry_different_alias_rms']:
        dst['max_geometry_different_alias_rms'] = src['max_geometry_different_alias_rms']
        dst['max_geometry_different_alias_counterexample'] = src['max_geometry_different_alias_counterexample']


def max_pair_distance(geometry: np.ndarray, temporal: bool) -> tuple[float, np.ndarray]:
    """Pairwise selected-geometry distance matrix for N clips or N single frames."""
    delta = geometry[:, None, ...] - geometry[None, :, ...]
    if temporal:
        # geometry [N,T,D]: sqrt(mean_T sum_D diff^2)
        squared = np.sum(delta * delta, axis=-1)
        distance = np.sqrt(np.mean(squared, axis=-1))
    else:
        # geometry [N,D]: sqrt(sum_D diff^2)
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
    upper = np.triu_indices(len(geometry), k=1)
    return float(np.max(distance[upper], initial=0.0)), distance


def compare_grouped_pairs(
    frames: np.ndarray,
    geometry: np.ndarray,
    groups: dict[bytes, list[int]],
    distance: np.ndarray,
    *,
    scene_id: str,
    C: int,
    archive: Any,
    manifest: dict[str, Any],
    task: str,
    horizon: int | None,
) -> dict[str, Any]:
    upper = np.triu_indices(len(frames), k=1)
    summary = new_summary()
    summary['pair_denominator'] = len(upper[0])
    if len(upper[0]):
        max_pos = int(np.argmax(distance[upper]))
        max_i, max_j = int(upper[0][max_pos]), int(upper[1][max_pos])
        summary['max_rms_all_pairs'] = float(distance[max_i, max_j])
        summary['max_rms_all_pair_example'] = pair_example(
            scene_id, max_i, max_j, C, archive, manifest, task, horizon
        )
    for indices in groups.values():
        for i, j in itertools.combinations(indices, 2):
            # Hash match only narrows candidates; exact array bytes are decisive.
            if not np.array_equal(frames[i], frames[j]):
                continue
            summary['pixel_identical_pair_count'] += 1
            rms = float(distance[i, j])
            if rms > summary['max_rms_among_pixel_identical_pairs']:
                summary['max_rms_among_pixel_identical_pairs'] = rms
                summary['max_alias_counterexample'] = pair_example(
                    scene_id, i, j, C, archive, manifest, task, horizon
                )
            if rms > GEOMETRY_EPSILON:
                summary['geometry_different_among_pixel_identical_count'] += 1
                if rms > summary['max_geometry_different_alias_rms']:
                    summary['max_geometry_different_alias_rms'] = rms
                    summary['max_geometry_different_alias_counterexample'] = pair_example(
                        scene_id, i, j, C, archive, manifest, task, horizon
                    )
    return summary


def summarize_task(panel_root: Path, task: str) -> dict[str, Any]:
    panel = panel_root / task
    manifest = json.loads((panel / 'manifest.json').read_text())
    entries = manifest.get('scenes', [])
    if len(entries) != 256:
        raise ValueError(f'{task}: expected all 256 scenes, found {len(entries)}')

    full_total = new_summary()
    single_totals: dict[int, dict[str, Any]] = {}
    scene_rows: list[dict[str, Any]] = []
    expected_steps: list[int] | None = None
    expected_KC: tuple[int, int] | None = None
    for entry in entries:
        scene_id = str(entry.get('scene_id', Path(entry['path']).stem))
        with np.load(panel / entry['path'], allow_pickle=False) as archive:
            pixels = np.asarray(archive['future_pixels'])
            if pixels.ndim != 6 or pixels.shape[-1] != 3 or pixels.dtype != np.uint8:
                raise ValueError(f'{task}/{scene_id}: invalid future_pixels {pixels.shape}/{pixels.dtype}')
            K, C, T, H, W, _ = pixels.shape
            if (K, C, T) != (2, 11, 5):
                raise ValueError(f'{task}/{scene_id}: expected [2,11,5,...], got {pixels.shape}')
            steps = np.asarray(archive['physical_steps'], dtype=np.int64).reshape(-1)
            if len(steps) != T:
                raise ValueError(f'{task}/{scene_id}: physical_steps length {len(steps)} != T={T}')
            if expected_steps is None:
                expected_steps = steps.tolist()
                expected_KC = (K, C)
                single_totals = {int(h): new_summary() for h in steps}
            elif steps.tolist() != expected_steps or expected_KC != (K, C):
                raise ValueError(f'{task}/{scene_id}: inconsistent conditions/candidates/horizons')

            geometry = physical_geometry(task, archive)
            if geometry.shape[:3] != (K, C, T):
                raise ValueError(f'{task}/{scene_id}: geometry shape mismatch {geometry.shape}')
            n = K * C
            flat_pixels = np.ascontiguousarray(pixels.reshape(n, T, H, W, 3))
            flat_geometry = np.ascontiguousarray(geometry.reshape(n, T, geometry.shape[-1]))
            max_all, full_distances = max_pair_distance(flat_geometry, temporal=True)
            full = compare_grouped_pairs(
                flat_pixels, flat_geometry, pixel_groups(flat_pixels), full_distances,
                scene_id=scene_id, C=C, archive=archive, manifest=manifest, task=task,
                horizon=steps.tolist(),
            )
            full['scene_count'] = 1
            merge_summary(full_total, full)

            one_scene: dict[str, Any] = {
                'scene_id': scene_id,
                'full_trajectory': full,
                'single_frame_by_horizon': {},
            }
            for ti, horizon_value in enumerate(steps):
                h = int(horizon_value)
                one_frames = np.ascontiguousarray(flat_pixels[:, ti])
                one_geometry = np.ascontiguousarray(flat_geometry[:, ti])
                frame_max, frame_distances = max_pair_distance(one_geometry, temporal=False)
                row = compare_grouped_pairs(
                    one_frames, one_geometry, pixel_groups(one_frames), frame_distances,
                    scene_id=scene_id, C=C, archive=archive, manifest=manifest,
                    task=task, horizon=h,
                )
                row['scene_count'] = 1
                merge_summary(single_totals[h], row)
                one_scene['single_frame_by_horizon'][str(h)] = row
            scene_rows.append(one_scene)

    return {
        'task': task,
        'scene_count_expected_and_processed': len(scene_rows),
        'conditions_K': expected_KC[0] if expected_KC else None,
        'candidates_C': expected_KC[1] if expected_KC else None,
        'physical_steps_raw': expected_steps,
        'selected_geometry': (
            'agent_xy + block_xy + 40*sin(block_angle) + 40*cos(block_angle), px'
            if task != 'robot_arm_mass' else 'finger_positions * 1000, mm'
        ),
        'geometry_interpretation': 'Selected task geometry only, not the complete physical state.',
        'geometry_difference_epsilon': GEOMETRY_EPSILON,
        'epsilon_interpretation': 'Floating-point comparison tolerance only; not a scientific or perceptual threshold.',
        'full_trajectory': full_total,
        'single_frame_by_horizon': {str(h): v for h, v in single_totals.items()},
        'all_scenes': scene_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', nargs='*', default=list(TASKS), choices=TASKS)
    args = parser.parse_args()
    results = [summarize_task(args.panel_root, task) for task in args.tasks]
    report = {
        'schema': 'contextworld.pixel_geometry_aliases.v1',
        'read_only': True,
        'training_or_inference': False,
        'simulation_or_cem': False,
        'panel_root': str(args.panel_root),
        'comparison_rule': {
            'full_trajectory': 'Compare all five [224,224,3] frames at the same candidate-condition pair; byte confirmation after SHA grouping.',
            'single_frame': 'Compare pairs only within the same stored horizon; never compare across time.',
            'pair_denominator_per_scene': 'choose(K*C,2)',
            'geometry_distance_full_trajectory': 'sqrt(mean_T(sum_D((x_i-x_j)^2)))',
            'geometry_distance_single_frame': 'sqrt(sum_D((x_i-x_j)^2))',
        },
        'tasks': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    for task in results:
        full = task['full_trajectory']
        print(task['task'], 'scenes', task['scene_count_expected_and_processed'],
              'full pairs', full['pair_denominator'],
              'pixel-identical', full['pixel_identical_pair_count'],
              'geometry-different aliases', full['geometry_different_among_pixel_identical_count'],
              'max alias RMS', full['max_rms_among_pixel_identical_pairs'])
        for h, row in task['single_frame_by_horizon'].items():
            print(' ', h, 'pairs', row['pair_denominator'],
                  'pixel-identical', row['pixel_identical_pair_count'],
                  'geometry-different aliases', row['geometry_different_among_pixel_identical_count'],
                  'max alias RMS', row['max_rms_among_pixel_identical_pairs'])
    print('wrote', args.output)


if __name__ == '__main__':
    main()
