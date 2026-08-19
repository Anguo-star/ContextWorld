#!/usr/bin/env python3
"""Audit one complete epoch of motion-damping twin-group batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from contextworld.benchmarks.motion_damping_icl_data import (  # noqa: E402
    _read_lance_pairs,
)
from run_pusht_motion_damping_h3_train import (  # noqa: E402
    CompleteTwinPairedBatchStream,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-table",
        type=Path,
        default=(
            ROOT
            / "artifacts/synthesis/pusht_motion_damping_h3_release_v3/"
            "train.lance"
        ),
    )
    parser.add_argument("--pair-count", type=int, default=2048)
    parser.add_argument("--hidden-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=14321)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = args.train_table.expanduser().resolve()
    arrays = _read_lance_pairs(
        table,
        expected_pairs=int(args.pair_count),
        expected_split="train",
    )
    stream = iter(
        CompleteTwinPairedBatchStream(
            int(args.pair_count),
            batch_size=int(args.hidden_batch_size),
            seed=int(args.seed),
        )
    )
    batches = int(args.pair_count) * 2 // int(args.hidden_batch_size)
    rows = torch.cat([next(stream) for _ in range(batches)])
    groups = rows.reshape(-1, 4)
    expected_rows = torch.arange(2 * int(args.pair_count))
    coverage_exact = torch.equal(torch.sort(rows).values, expected_rows)
    pair_order_preserved = bool(
        torch.all(groups[:, 1] == groups[:, 0] + 1)
        and torch.all(groups[:, 3] == groups[:, 2] + 1)
        and torch.all(groups[:, 0] % 2 == 0)
        and torch.all(groups[:, 2] % 2 == 0)
    )
    first_pairs = groups[:, 0] // 2
    second_pairs = groups[:, 2] // 2
    twins_complete = bool(
        torch.all(first_pairs % 2 == 0)
        and torch.all(second_pairs == first_pairs + 1)
    )
    identifiers_match = all(
        arrays.pair_ids[int(first)].endswith("-forward")
        and arrays.pair_ids[int(second)].endswith("-reverse")
        for first, second in zip(first_pairs, second_pairs, strict=True)
    )
    x0_labels_exchange = all(
        np.array_equal(
            arrays.faster_decay_pixels[int(first), 0],
            arrays.no_extra_decay_pixels[int(second), 0],
        )
        and np.array_equal(
            arrays.no_extra_decay_pixels[int(first), 0],
            arrays.faster_decay_pixels[int(second), 0],
        )
        for first, second in zip(first_pairs, second_pairs, strict=True)
    )
    checks = {
        "one_epoch_condition_row_coverage_exact": bool(coverage_exact),
        "low_high_pair_order_preserved": pair_order_preserved,
        "forward_reverse_twins_complete": twins_complete,
        "pair_identifiers_match_direction": bool(identifiers_match),
        "x0_rgb_labels_exchange_within_every_twin": bool(
            x0_labels_exchange
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "train_table": str(table),
        "train_table_files": sum(path.is_file() for path in table.rglob("*")),
        "pair_count": int(args.pair_count),
        "condition_row_count": 2 * int(args.pair_count),
        "hidden_batch_size": int(args.hidden_batch_size),
        "complete_twin_groups_per_batch": int(args.hidden_batch_size) // 4,
        "batches_per_epoch": batches,
        "seed": int(args.seed),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**payload, "sha256": _sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
