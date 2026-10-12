# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-card temporal sampling and object/shared failure boundaries."""

import io
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.video_embed import processing
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, validate_terminal_batch
from cosmos_curator.next.recipes.video_embed.processing import download_clip, sample_indices, sample_video
from cosmos_curator.next.utils.media_reader import (
    MediaObjectError,
    MediaObjectReader,
    MediaSystemicError,
    MediaTransportError,
)


def video_bytes(
    frame_count: int, *, width: int = 64, height: int = 48, codec: str = "mpeg4", container_format: str = "mp4"
) -> bytes:
    """Encode distinguishable nonsquare frames with B frames to exercise display order."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format=container_format) as container:
        stream = container.add_stream(codec, rate=8)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.max_b_frames = 2
        for index in range(frame_count):
            pixels = np.full((height, width, 3), index * 10, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [(1, [0] * 8), (3, [0, 0, 0, 0, 1, 1, 1, 2]), (8, list(range(8))), (15, list(range(0, 15, 2)))],
)
def test_sampling_matches_model_card(frame_count: int, expected: list[int]) -> None:
    """Retain both endpoints and repeated frames for clips shorter than the model window."""
    np.testing.assert_array_equal(sample_indices(frame_count), expected)


def test_bad_video_is_an_object_fault() -> None:
    """Undecodable and empty clips are terminal media failures, not fake vectors."""
    with pytest.raises(MediaObjectError, match="bitstream"):
        sample_video(b"not a video")
    with pytest.raises(MediaObjectError, match="no video frames"):
        sample_indices(0)


def test_truncated_decode_is_an_object_fault(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decoder that drops declared frames cannot silently change the temporal sample."""
    monkeypatch.setattr(processing.VideoDecoder, "decode", lambda *_: iter(()))
    with pytest.raises(MediaObjectError, match="Decoded 0 frames but MP4 declares 3"):
        sample_video(video_bytes(3))


def test_gpu_failures_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unclassified CUDA/decoder failures must not permanently quarantine a valid clip."""

    def fail(*_args: object) -> None:
        msg = "CUDA device unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(processing.VideoDecoder, "decode", fail)
    with pytest.raises(RuntimeError, match="CUDA device unavailable"):
        sample_video(video_bytes(3))


def test_download_size_and_missing_errors(work_rows: list[dict[str, Any]]) -> None:
    """Preserve keyed error rows for unavailable and changed clip objects."""
    row = work_rows[0]
    assert download_clip(row, max_attempts=1)["clip_bytes"] == b"mp4"
    changed = download_clip(row | {"clip_size_bytes": 4}, max_attempts=1)
    assert changed["clip_id"] == row["clip_id"]
    assert changed["error_type"] == "MediaObjectError"
    assert changed["clip_bytes"] is None
    Path(row["clip_uri"]).unlink()
    missing = download_clip(row, max_attempts=1)
    assert missing["error_type"] == "MediaObjectError"


@pytest.mark.parametrize("error", [MediaSystemicError, MediaTransportError])
def test_shared_faults_propagate(
    work_rows: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    error: type[Exception],
) -> None:
    """Credential failures and exhausted transport retries never become terminal clip errors."""

    def fail(*_args: object, **_kwargs: object) -> bytes:
        msg = "storage unavailable"
        raise error(msg)

    monkeypatch.setattr(MediaObjectReader, "read", fail)
    with pytest.raises(error, match="storage unavailable"):
        download_clip(work_rows[0], max_attempts=1)


def test_nullable_vectors_survive_checkpoint_parquet(tmp_path: Path, results: pa.Table, spec: ModelSpec) -> None:
    """Staging supports null vectors while the canonical schema enforces fixed dimension."""
    path = tmp_path / "results.parquet"
    pq.write_table(results, path)
    restored = pq.read_table(path)
    assert restored.schema.equals(results.schema, check_metadata=True)
    assert restored.equals(results)
    canonical = restored.select(spec.fields.names).cast(spec.fields)
    assert canonical[spec.vector_field].null_count == 1
    for batch in canonical.to_batches():
        validate_terminal_batch(batch, spec)
