from __future__ import annotations

import numpy as np
import pytest

from contextworld.evaluation.natural_history import (
    _clip_sha256,
    _to_hwc_uint8,
    history_token_slice,
)


def test_history_token_slice_ends_at_shared_query() -> None:
    tokens = ["context_1", "context_2", "query"]
    assert tokens[history_token_slice(0)] == ["query"]
    assert tokens[history_token_slice(1)] == ["context_2", "query"]
    assert tokens[history_token_slice(2)] == tokens
    with pytest.raises(ValueError):
        history_token_slice(3)


def test_to_hwc_uint8_preserves_pixels_exactly() -> None:
    chw = np.arange(4 * 3 * 2 * 5, dtype=np.uint8).reshape(4, 3, 2, 5)
    hwc = _to_hwc_uint8(chw)
    assert hwc.shape == (4, 2, 5, 3)
    np.testing.assert_array_equal(hwc, chw.transpose(0, 2, 3, 1))


def test_clip_hash_covers_actions_and_pixels() -> None:
    clip = {
        "pixels": np.zeros((4, 3, 2, 2), dtype=np.uint8),
        "action": np.zeros((4, 10), dtype=np.float32),
    }
    original = _clip_sha256(clip)
    changed_action = {key: value.copy() for key, value in clip.items()}
    changed_action["action"][0, 0] = 1.0
    assert _clip_sha256(changed_action) != original
    changed_pixel = {key: value.copy() for key, value in clip.items()}
    changed_pixel["pixels"][0, 0, 0, 0] = 1
    assert _clip_sha256(changed_pixel) != original
