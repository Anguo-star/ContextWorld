import io

import numpy as np
import pytest
from PIL import Image

from contextworld.synthesis.lance import encode_frame, normalize_pixel_codec


def test_png_codec_roundtrip_is_pixel_exact() -> None:
    rng = np.random.default_rng(17)
    frame = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    codec = {"format": "png", "compress_level": 1}

    blob = encode_frame(frame, codec)
    decoded = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))

    np.testing.assert_array_equal(decoded, frame)
    assert normalize_pixel_codec(codec) == {
        "format": "png",
        "compress_level": 1,
        "lossless": True,
    }


def test_pixel_codec_rejects_unknown_options() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        normalize_pixel_codec({"format": "png", "quality": 95})
