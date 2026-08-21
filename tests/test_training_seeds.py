"""The public seed list fails before it can waste a training allocation."""

from __future__ import annotations

import argparse

import pytest

from contextworld.training.seeds import parse_training_seeds


def test_one_or_multiple_comma_separated_seeds() -> None:
    assert parse_training_seeds("3072") == (3072,)
    assert parse_training_seeds("3072, 3073,3074") == (3072, 3073, 3074)


@pytest.mark.parametrize("value", ["", "3072,", ",3072", "a", "-1", "7,7"])
def test_invalid_or_duplicate_seeds_are_rejected(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_training_seeds(value)
