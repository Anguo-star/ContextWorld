from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PIXEL_CODEC = {"format": "jpeg", "quality": 95}


def normalize_pixel_codec(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return a strict, JSON-safe pixel codec specification."""

    raw = dict(DEFAULT_PIXEL_CODEC if value is None else value)
    codec = str(raw.pop("format", "")).lower()
    declared_lossless = raw.pop("lossless", None)
    if codec == "jpeg":
        if declared_lossless not in (None, False):
            raise ValueError("JPEG cannot be declared lossless")
        quality = int(raw.pop("quality", 95))
        if not 1 <= quality <= 100:
            raise ValueError(f"JPEG quality must be in [1, 100], got {quality}")
        normalized = {"format": "jpeg", "quality": quality}
    elif codec == "png":
        if declared_lossless not in (None, True):
            raise ValueError("PNG pixel codec must be declared lossless")
        compress_level = int(raw.pop("compress_level", 6))
        if not 0 <= compress_level <= 9:
            raise ValueError(
                f"PNG compress_level must be in [0, 9], got {compress_level}"
            )
        normalized = {
            "format": "png",
            "compress_level": compress_level,
            "lossless": True,
        }
    else:
        raise ValueError(f"Unsupported pixel codec {codec!r}; use jpeg or png")
    if raw:
        raise ValueError(f"Unexpected pixel codec options: {sorted(raw)}")
    return normalized


def encode_frame(frame: np.ndarray, codec: dict[str, Any]) -> bytes:
    """Encode one RGB frame using the benchmark-declared storage codec."""

    from PIL import Image

    specification = normalize_pixel_codec(codec)
    value = np.asarray(frame)
    if value.ndim == 3 and value.shape[0] in (1, 3, 4):
        value = np.transpose(value, (1, 2, 0))
    if value.shape[-1] == 1:
        value = value.squeeze(-1)
    buffer = io.BytesIO()
    if specification["format"] == "jpeg":
        Image.fromarray(value.astype(np.uint8)).save(
            buffer,
            format="JPEG",
            quality=specification["quality"],
        )
    else:
        Image.fromarray(value.astype(np.uint8)).save(
            buffer,
            format="PNG",
            compress_level=specification["compress_level"],
            optimize=False,
        )
    return buffer.getvalue()


def build_lance_writer(
    swm: Any,
    path: str | Path,
    *,
    pixel_codec: dict[str, Any],
):
    """Build a StableWM-compatible Lance writer with an explicit codec.

    The pinned StableWM writer hard-codes JPEG.  ContextWorld keeps the
    upstream checkout read-only and overrides only its record-batch encoding
    hook; the on-disk Lance schema and reader remain native StableWM.
    """

    import lance as lance_lib
    import pyarrow as pa

    specification = normalize_pixel_codec(pixel_codec)
    base = swm.data.LanceWriter

    class ContextWorldLanceWriter(base):
        def __init__(self, raw_path: str | Path, *, mode: str = "error"):
            super().__init__(raw_path, mode=mode)
            self._raw_path = Path(raw_path)
            self._contextworld_mode = mode

        def __enter__(self):
            if self._raw_path.exists():
                if self._contextworld_mode == "error":
                    raise FileExistsError(self._raw_path)
                if self._contextworld_mode != "overwrite":
                    raise ValueError(
                        "ContextWorld raw Lance writer supports only error "
                        "or overwrite mode"
                    )
            self._raw_path.parent.mkdir(parents=True, exist_ok=True)
            # The upstream writer checks only that this sentinel is non-None.
            # Writing the raw Lance dataset directly avoids a needless
            # LanceDB connection while preserving the native table schema.
            self._db = self
            return self

        def __exit__(self, *exc):
            self._db = None
            self._table = None

        def _consume_episodes(self, episodes) -> None:
            iterator = iter(episodes)
            try:
                first_ep = next(iterator)
            except StopIteration:
                return
            self._init_schema(first_ep)
            self._initialized = True

            def batch_gen():
                yield self._batch_from_episode(first_ep)
                for episode in iterator:
                    yield self._batch_from_episode(episode)

            reader = pa.RecordBatchReader.from_batches(
                self._schema,
                batch_gen(),
            )
            lance_lib.write_dataset(
                reader,
                str(self._raw_path),
                mode=(
                    "overwrite"
                    if self._contextworld_mode == "overwrite"
                    else "create"
                ),
            )

        def _build_batch(self, ep_data: dict, ep_len: int) -> pa.RecordBatch:
            episode_idx = np.full(ep_len, self._ep_idx, dtype=np.int32)
            step_idx = np.arange(ep_len, dtype=np.int32)
            arrays: list[pa.Array] = [
                pa.array(episode_idx, type=pa.int32()),
                pa.array(step_idx, type=pa.int32()),
            ]
            for column, lance_name in self._rename_map.items():
                values = ep_data[column]
                if lance_name in self._image_cols:
                    blobs = [
                        encode_frame(np.asarray(value), specification)
                        for value in values
                    ]
                    arrays.append(pa.array(blobs, type=pa.binary()))
                elif lance_name in self._string_cols:
                    strings = [
                        value.decode()
                        if isinstance(value, (bytes, bytearray))
                        else str(value)
                        for value in values
                    ]
                    arrays.append(pa.array(strings, type=pa.string()))
                else:
                    dimension = self._dims[lance_name]
                    flat = np.asarray(values, dtype=np.float32).reshape(
                        ep_len, dimension
                    )
                    arrays.append(
                        pa.FixedSizeListArray.from_arrays(
                            pa.array(flat.reshape(-1), type=pa.float32()),
                            dimension,
                        )
                    )
            return pa.record_batch(arrays, schema=self._schema)

    return ContextWorldLanceWriter(path, mode="error")


__all__ = [
    "DEFAULT_PIXEL_CODEC",
    "build_lance_writer",
    "encode_frame",
    "normalize_pixel_codec",
]
