#!/usr/bin/env python3
"""Build audited paired Push-T hidden-actuation History-3 datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np


CONTEXTWORLD_ROOT = Path(__file__).resolve().parents[1]
STABLE_WORLD_MODEL_ROOT = CONTEXTWORLD_ROOT.parent / 'stable-worldmodel'
for source_root in (CONTEXTWORLD_ROOT, STABLE_WORLD_MODEL_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from stable_worldmodel.data import LanceWriter  # noqa: E402
from contextworld.evaluation.pusht_hidden_actuation import (  # noqa: E402
    MODE_SCALES,
    PHYSICS_STATE_COMPONENTS,
    HiddenActuationTemplate,
    array_sha256,
    simulate_hidden_actuation,
    validate_hidden_actuation_pair,
)


PROTOCOL = 'pusht_action_strength_history3_action_coverage_strict_v3'
SPLITS = ('train', 'validation', 'eval')


def canonical_json_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_path(path: Path) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    if path.exists():
        raise FileExistsError(
            f'Output already exists; refusing to overwrite: {path}'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _candidate_template(
    *,
    split: str,
    index: int,
    attempt: int,
    rng: np.random.Generator,
    seed: int,
    query_amplitude_range: tuple[float, float],
) -> HiddenActuationTemplate:
    direction_sign = 1 if int(rng.integers(0, 2)) else -1
    direction = np.asarray([direction_sign, 0.0], dtype=np.float64)
    if direction_sign > 0:
        block_x = float(rng.uniform(190.0, 350.0))
    else:
        block_x = float(rng.uniform(162.0, 322.0))
    # PushT's declared block.start_position support is [100, 400].
    # Keep a small interior margin for numerical safety and probe clearance.
    block_y = float(rng.uniform(105.0, 395.0))
    block = np.asarray([block_x, block_y], dtype=np.float64)
    distance = float(rng.uniform(78.0, 82.0))
    tangent_offset = float(rng.uniform(-4.0, 4.0))
    tangent = np.asarray([0.0, 1.0], dtype=np.float64)
    agent = block - distance * direction + tangent_offset * tangent
    angle = float(0.0 if int(rng.integers(0, 2)) else np.pi)
    probe_sign = 1 if int(rng.integers(0, 2)) else -1
    goal_agent = agent + 60.0 * direction
    goal_block = block + 40.0 * direction
    amplitude_min, amplitude_max = query_amplitude_range
    query_amplitude = (
        float(amplitude_min)
        if amplitude_min == amplitude_max
        else float(rng.uniform(amplitude_min, amplitude_max))
    )
    simulator_seed = int(
        np.random.SeedSequence(
            [seed, index, attempt, {'train': 1, 'validation': 2, 'eval': 3}[split]]
        ).generate_state(1)[0]
    )
    return HiddenActuationTemplate(
        template_id=f'pha-{split}-{index:05d}',
        agent_position=tuple(map(float, agent)),
        block_position=tuple(map(float, block)),
        block_angle=angle,
        contact_direction=(float(direction_sign), 0.0),
        probe_sign=probe_sign,
        goal_agent_position=tuple(map(float, goal_agent)),
        goal_block_position=tuple(map(float, goal_block)),
        goal_block_angle=angle,
        simulator_seed=simulator_seed,
        query_amplitude=query_amplitude,
    )


def _episode(
    rollout: dict[str, Any],
    *,
    split: str,
) -> dict[str, list[Any]]:
    rows = {key: list(value) for key, value in rollout['rows'].items()}
    rows['split'] = [split] * len(rows['pixels'])
    return rows


def _save_eval_payload(
    path: Path,
    *,
    low: dict[str, Any],
    high: dict[str, Any],
) -> dict[str, Any]:
    np.savez_compressed(
        path,
        low_pixels=np.asarray(low['model_pixels'], dtype=np.uint8),
        high_pixels=np.asarray(high['model_pixels'], dtype=np.uint8),
        low_actions=np.asarray(low['action_blocks'], dtype=np.float32),
        high_actions=np.asarray(high['action_blocks'], dtype=np.float32),
        low_states=np.asarray(low['model_states'], dtype=np.float32),
        high_states=np.asarray(high['model_states'], dtype=np.float32),
        low_physics=np.asarray(
            low['model_physics_states'],
            dtype=np.float32,
        ),
        high_physics=np.asarray(
            high['model_physics_states'],
            dtype=np.float32,
        ),
        goal_pixels=np.asarray(low['goal_pixels'], dtype=np.uint8),
        goal_state=np.asarray(
            low['template']['goal_agent_position']
            + low['template']['goal_block_position']
            + (
                low['template']['goal_block_angle'],
                0.0,
                0.0,
            ),
            dtype=np.float32,
        ),
    )
    return {
        'path': path.name,
        'sha256': file_sha256(path),
        'size_bytes': path.stat().st_size,
    }


def _strict_causal_chain_audit(
    pair_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    audits = [row['audit'] for row in pair_reports]
    if not audits:
        raise ValueError('Strict causal audit requires at least one pair')
    summary = {
        'pair_count': len(audits),
        'state_installations_after_x0': int(
            sum(
                int(row['state_installations_after_x0'])
                for row in audits
            )
        ),
        'query_simulator_recreated': bool(
            any(row['query_simulator_recreated'] for row in audits)
        ),
        'max_pair_full_state_gap': float(
            max(row['query_physics_max_abs_gap'] for row in audits)
        ),
        'max_pair_query_pixel_difference': int(
            max(row['pair_query_pixel_difference'] for row in audits)
        ),
        'max_pair_query_action_difference': float(
            max(row['pair_query_action_difference'] for row in audits)
        ),
        'min_history_effect': float(
            min(row['history_effect'] for row in audits)
        ),
        'min_true_future_effect': float(
            min(row['true_future_effect'] for row in audits)
        ),
        'full_state_tolerance': float(
            min(row['query_physics_tolerance'] for row in audits)
        ),
        'full_state_dimensions': int(audits[0]['full_state_dimensions']),
        'full_state_components': audits[0]['full_state_components'],
        'query_pixel_difference_unit': 'different_uint8_channel_values',
        'query_action_difference_unit': 'maximum_absolute_action_value',
        'history_effect_unit': 'agent_position_px_at_x1',
        'true_future_effect_unit': 'block_position_px_at_x3',
    }
    summary['passed'] = (
        summary['state_installations_after_x0'] == 0
        and not summary['query_simulator_recreated']
        and summary['max_pair_full_state_gap']
        <= summary['full_state_tolerance']
        and summary['max_pair_query_pixel_difference'] == 0
        and summary['max_pair_query_action_difference'] == 0.0
    )
    return summary


def build_split(
    *,
    root: Path,
    split: str,
    pair_count: int,
    seed: int,
    resolution: int,
    jpeg_quality: int,
    maximum_attempts_per_pair: int,
    query_amplitude_range: tuple[float, float],
) -> dict[str, Any]:
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split!r}')
    if pair_count <= 0:
        raise ValueError('pair_count must be positive')

    split_seed = int(
        np.random.SeedSequence(
            [seed, {'train': 11, 'validation': 13, 'eval': 17}[split]]
        ).generate_state(1)[0]
    )
    rng = np.random.default_rng(split_seed)
    table_path = root / f'{split}.lance'
    eval_dir = root / 'eval_payloads'
    if split == 'eval':
        eval_dir.mkdir()

    pair_reports: list[dict[str, Any]] = []
    query_hashes: set[str] = set()
    attempts_total = 0

    def episodes() -> Iterator[dict[str, list[Any]]]:
        nonlocal attempts_total
        for index in range(pair_count):
            accepted = False
            last_failure: dict[str, Any] | None = None
            for attempt in range(maximum_attempts_per_pair):
                attempts_total += 1
                template = _candidate_template(
                    split=split,
                    index=index,
                    attempt=attempt,
                    rng=rng,
                    seed=seed,
                    query_amplitude_range=query_amplitude_range,
                )
                try:
                    low = simulate_hidden_actuation(
                        template,
                        mode='low_gain',
                        resolution=resolution,
                    )
                    high = simulate_hidden_actuation(
                        template,
                        mode='high_gain',
                        resolution=resolution,
                    )
                    audit = validate_hidden_actuation_pair(low, high)
                except (AssertionError, RuntimeError, ValueError) as error:
                    last_failure = {
                        'exception': type(error).__name__,
                        'message': str(error),
                    }
                    continue
                query_hash = audit['hashes']['query_pixels']
                if query_hash in query_hashes:
                    last_failure = {'duplicate_query_hash': query_hash}
                    continue
                if not audit['passed']:
                    last_failure = {
                        'failed_checks': [
                            name
                            for name, passed in audit['checks'].items()
                            if not passed
                        ]
                    }
                    continue

                query_hashes.add(query_hash)
                report = {
                    'template': asdict(template),
                    'audit': audit,
                    'eval_payload': None,
                }
                if split == 'eval':
                    payload_path = (
                        eval_dir / f'{template.template_id}.npz'
                    )
                    report['eval_payload'] = _save_eval_payload(
                        payload_path,
                        low=low,
                        high=high,
                    )
                pair_reports.append(report)
                yield _episode(low, split=split)
                yield _episode(high, split=split)
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f'Could not synthesize {split} pair {index} after '
                    f'{maximum_attempts_per_pair} attempts; '
                    f'last_failure={last_failure}'
                )

    with LanceWriter(
        table_path,
        jpeg_quality=jpeg_quality,
        mode='error',
    ) as writer:
        writer.write_episodes(episodes())

    if len(pair_reports) != pair_count:
        raise RuntimeError(
            f'Expected {pair_count} pairs, built {len(pair_reports)}'
        )

    return {
        'split': split,
        'pair_count': pair_count,
        'episode_count': 2 * pair_count,
        'raw_rows': 2 * pair_count * 20,
        'table_path': table_path.name,
        'split_seed': split_seed,
        'attempts_total': attempts_total,
        'acceptance_rate': pair_count / attempts_total,
        'query_amplitude': {
            'requested_minimum': float(query_amplitude_range[0]),
            'requested_maximum': float(query_amplitude_range[1]),
            'accepted_minimum': float(
                min(
                    row['template']['query_amplitude']
                    for row in pair_reports
                )
            ),
            'accepted_maximum': float(
                max(
                    row['template']['query_amplitude']
                    for row in pair_reports
                )
            ),
        },
        'query_hash_count': len(query_hashes),
        'query_hashes': sorted(query_hashes),
        'pairs': pair_reports,
        'strict_causal_chain_audit': _strict_causal_chain_audit(
            pair_reports
        ),
    }


def _cross_split_audit(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hashes = {
        split: set(report['query_hashes'])
        for split, report in reports.items()
    }
    overlaps = {
        f'{left}__{right}': sorted(hashes[left] & hashes[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    return {
        'query_hash_overlap_counts': {
            name: len(values) for name, values in overlaps.items()
        },
        'passed': not any(overlaps.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-pairs', type=int, default=128)
    parser.add_argument('--validation-pairs', type=int, default=32)
    parser.add_argument('--eval-pairs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260729)
    parser.add_argument('--resolution', type=int, default=224)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--maximum-attempts-per-pair', type=int, default=64)
    parser.add_argument('--train-query-amplitude-min', type=float, default=0.4)
    parser.add_argument('--train-query-amplitude-max', type=float, default=0.4)
    parser.add_argument(
        '--validation-query-amplitude-min',
        type=float,
        default=0.4,
    )
    parser.add_argument(
        '--validation-query-amplitude-max',
        type=float,
        default=0.4,
    )
    parser.add_argument('--eval-query-amplitude', type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amplitude_ranges = {
        'train': (
            float(args.train_query_amplitude_min),
            float(args.train_query_amplitude_max),
        ),
        'validation': (
            float(args.validation_query_amplitude_min),
            float(args.validation_query_amplitude_max),
        ),
        'eval': (
            float(args.eval_query_amplitude),
            float(args.eval_query_amplitude),
        ),
    }
    for split, (minimum, maximum) in amplitude_ranges.items():
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError(
                f'{split} query amplitude range must lie in [0, 1], '
                f'got {(minimum, maximum)}'
            )
    output = _safe_output_path(args.output)
    requested = {
        'protocol': PROTOCOL,
        'seed': int(args.seed),
        'resolution': int(args.resolution),
        'jpeg_quality': int(args.jpeg_quality),
        'mode_scales': MODE_SCALES,
        'query_amplitude_ranges': amplitude_ranges,
        'pair_counts': {
            'train': int(args.train_pairs),
            'validation': int(args.validation_pairs),
            'eval': int(args.eval_pairs),
        },
        'strict_causal_contract': {
            'initial_x0_identical_across_modes': True,
            'state_installations_after_x0': 0,
            'query_simulator_recreated': False,
            'x1_to_x2_reached_by_environment_steps_only': True,
            'query_pixels_identical_across_modes': True,
            'query_action_identical_across_modes': True,
            'full_state_tolerance': 1e-5,
            'full_state_dimensions': len(PHYSICS_STATE_COMPONENTS),
            'full_state_components': list(PHYSICS_STATE_COMPONENTS),
        },
    }
    with tempfile.TemporaryDirectory(
        prefix='pusht-hidden-actuation-build-',
        dir='/tmp',
    ) as temporary:
        root = Path(temporary) / output.name
        root.mkdir()
        (root / 'request.json').write_text(
            json.dumps(requested, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

        reports = {
            split: build_split(
                root=root,
                split=split,
                pair_count=requested['pair_counts'][split],
                seed=int(args.seed),
                resolution=int(args.resolution),
                jpeg_quality=int(args.jpeg_quality),
                maximum_attempts_per_pair=int(
                    args.maximum_attempts_per_pair
                ),
                query_amplitude_range=amplitude_ranges[split],
            )
            for split in SPLITS
        }
        cross_split = _cross_split_audit(reports)
        if not cross_split['passed']:
            raise RuntimeError(f'Cross-split audit failed: {cross_split}')
        all_pair_reports = [
            pair
            for split in SPLITS
            for pair in reports[split]['pairs']
        ]
        strict_audit = _strict_causal_chain_audit(all_pair_reports)
        if not strict_audit['passed']:
            raise RuntimeError(
                f'Strict causal-chain audit failed: {strict_audit}'
            )

        manifest = {
            **requested,
            'request_sha256': canonical_json_sha256(requested),
            'splits': reports,
            'cross_split_audit': cross_split,
            'strict_causal_chain_audit': strict_audit,
            'passed': (
                cross_split['passed']
                and strict_audit['passed']
                and all(
                    pair['audit']['passed']
                    for report in reports.values()
                    for pair in report['pairs']
                )
            ),
        }
        manifest_path = root / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        summary = {
            'protocol': PROTOCOL,
            'root': str(output),
            'manifest': manifest_path.name,
            'manifest_sha256': file_sha256(manifest_path),
            'passed': manifest['passed'],
            'pair_counts': requested['pair_counts'],
            'acceptance_rates': {
                split: reports[split]['acceptance_rate']
                for split in SPLITS
            },
            'cross_split_audit': cross_split,
            'strict_causal_chain_audit': strict_audit,
        }
        (root / 'build_report.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        # Lance commits use atomic rename, which is not supported by the
        # repository's NFS mount.  Build on local /tmp and publish only the
        # fully audited immutable tree.
        shutil.copytree(root, output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
