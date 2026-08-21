"""Shared parsing for the public training-seed interface."""

from __future__ import annotations

import argparse
from collections.abc import Mapping


DEFAULT_TRAINING_SEEDS = (3072,)
LEGACY_SEED_VARIABLES = ("CW_SEED", "CW_ALL_SEEDS")


def parse_training_seeds(value: str) -> tuple[int, ...]:
    """Parse a non-empty, comma-separated list of unique non-negative seeds."""

    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers, for example "
            "3072 or 3072,3073,3074"
        )
    try:
        seeds = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers, for example "
            "3072 or 3072,3073,3074"
        ) from exc
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must not contain duplicates")
    return seeds


def reject_legacy_seed_environment(
    parser: argparse.ArgumentParser,
    environment: Mapping[str, str],
) -> None:
    """Fail loudly when a stale cloud template still uses the old switches."""

    configured = [name for name in LEGACY_SEED_VARIABLES if name in environment]
    if configured:
        parser.error(
            f"{', '.join(configured)} has been replaced by CW_SEEDS; use "
            "CW_SEEDS=3072 for one run or "
            "CW_SEEDS=3072,3073,3074 for multiple runs"
        )


__all__ = [
    "DEFAULT_TRAINING_SEEDS",
    "LEGACY_SEED_VARIABLES",
    "parse_training_seeds",
    "reject_legacy_seed_environment",
]
