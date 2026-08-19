#!/usr/bin/env python3
"""Search a stronger common-query action for PushT motion damping."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextworld.evaluation import pusht_motion_damping_h3 as damping  # noqa: E402
from contextworld.evaluation import pusht_contact_friction_h3 as friction  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.top_k <= 0:
        raise ValueError("--samples and --top-k must be positive")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    generator = np.random.default_rng(args.seed)
    base = damping.make_base_template()
    candidates: list[dict[str, object]] = []
    # The first three and final two raw actions are tied. This keeps the
    # search reproducible and the eventual benchmark action easy to explain.
    anchors = [
        np.asarray(base.query_actions, dtype=np.float64),
        np.asarray([(0.0, 1.0)] * 5, dtype=np.float64),
        np.asarray([(0.5, 1.0)] * 5, dtype=np.float64),
        np.asarray([(1.0, 1.0)] * 5, dtype=np.float64),
    ]
    for index in range(args.samples + len(anchors)):
        if index < len(anchors):
            actions = anchors[index]
        else:
            first = generator.uniform(-1.0, 1.0, size=2)
            second = generator.uniform(-1.0, 1.0, size=2)
            actions = np.concatenate(
                [np.repeat(first[None], 3, axis=0),
                 np.repeat(second[None], 2, axis=0)],
                axis=0,
            )
        template = replace(
            base,
            query_actions=tuple(map(tuple, actions.tolist())),
        )
        rollouts = {
            mode: damping._simulate_query(
                template,
                mode=mode,
                resolution=96,
                render_pixels=False,
            )
            for mode in damping.ENDPOINT_MODES
        }
        if not all(
            np.any(value["contacts"] > 0)
            and all(
                friction._bounds_inside_playfield(bounds)
                for bounds in value["bounds"]
            )
            for value in rollouts.values()
        ):
            continue
        low, high = (rollouts[mode] for mode in damping.ENDPOINT_MODES)
        gap = friction._future_gap(
            low["future_snapshot"], high["future_snapshot"]
        )
        candidates.append(
            {
                "actions": actions.tolist(),
                "future_gap": gap,
                "contact_steps": {
                    mode: int(np.count_nonzero(value["contacts"]))
                    for mode, value in rollouts.items()
                },
            }
        )
    candidates.sort(
        key=lambda value: float(value["future_gap"]["block_position_px"]),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "seed": args.seed,
        "sample_count": args.samples,
        "valid_count": len(candidates),
        "top": candidates[: args.top_k],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
