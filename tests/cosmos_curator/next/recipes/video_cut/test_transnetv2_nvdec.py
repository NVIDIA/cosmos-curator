# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU decoding preserves variable cadence, a nonzero origin, and delayed frames."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from cosmos_curator.next.recipes.video_cut import transnetv2_nvdec
from cosmos_curator.next.recipes.video_cut.transnetv2 import decode_frames
from cosmos_curator.next.recipes.video_cut.transnetv2_nvdec import decode_frames_nvdec

_OFFSETS = [0, 1, 3, 6, 7, 15, 16, 19, 24, 25, 28, 32]


def _write_clip(path: Path, codec: str, b_frames: int) -> Path:
    """Encode distinct frames with spatial detail at a variable cadence and a nonzero origin."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=25)
        stream.width, stream.height, stream.pix_fmt = 128, 96, "yuv420p"
        stream.time_base = Fraction(1, 25)
        stream.codec_context.max_b_frames = b_frames
        for index, offset in enumerate(_OFFSETS):
            pixels = np.full((96, 128, 3), [20 + index * 15, 30, 220 - index * 12], dtype=np.uint8)
            pixels[:, :, 1] = np.roll(np.linspace(0, 255, 128, dtype=np.uint8), index * 8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = 100 + offset, Fraction(1, 25)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return path


@pytest.mark.env("default")
@pytest.mark.parametrize(("codec", "b_frames"), [("libopenh264", 0), ("mpeg4", 2)])
def test_nvdec_preserves_native_timestamps_and_drains_delayed_frames(tmp_path: Path, codec: str, b_frames: int) -> None:
    """All frames survive GPU decode, including frames buffered by B-frame reordering."""
    path = _write_clip(tmp_path / "variable.mp4", codec, b_frames)
    _, cpu_times = decode_frames(path, cpus=1)
    frames, gpu_times = decode_frames_nvdec(path)
    assert frames.shape == (len(_OFFSETS), 27, 48, 3)
    assert frames.dtype == np.uint8
    np.testing.assert_array_equal(gpu_times, np.asarray(_OFFSETS) * 40_000_000)
    np.testing.assert_array_equal(gpu_times, cpu_times)


@pytest.mark.env("default")
@pytest.mark.parametrize(("batch_frames", "batch_bytes"), [(1, 1 << 30), (5, 1 << 30), (32, 3 * 96 * 128 * 3)])
def test_nvdec_resize_batching_does_not_change_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_frames: int, batch_bytes: int
) -> None:
    """Single-frame, partial, and byte-bounded resize batches match one batch holding every frame."""
    path = _write_clip(tmp_path / "variable.mp4", "mpeg4", 2)
    expected, expected_times = decode_frames_nvdec(path)
    monkeypatch.setattr(transnetv2_nvdec, "_RESIZE_BATCH_FRAMES", batch_frames)
    monkeypatch.setattr(transnetv2_nvdec, "_RESIZE_BATCH_BYTES", batch_bytes)
    frames, times = decode_frames_nvdec(path)
    np.testing.assert_array_equal(frames, expected)
    np.testing.assert_array_equal(times, expected_times)
