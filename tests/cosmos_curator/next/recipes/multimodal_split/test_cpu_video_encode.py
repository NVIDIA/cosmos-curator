# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for writing decoded frames to an MP4."""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.next.recipes.multimodal_split.cpu_video_encode import _parse_bitrate, encode_frames_to_path

_WIDTH = 32
_HEIGHT = 16
_ENCODER = "libopenh264"
_BITRATE = "400K"
_GOP = 10


def _frame(value: int) -> npt.NDArray[np.uint8]:
    return np.full((_HEIGHT, _WIDTH, 3), value, dtype=np.uint8)


def _decoded_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(video=0))


def test_frames_are_written_at_the_requested_geometry_and_rate(tmp_path: Path) -> None:
    """The written file is playable and carries what the caller handed over."""
    path = tmp_path / "clip.mp4"

    written = encode_frames_to_path(
        (_frame(index * 8) for index in range(10)),
        path,
        fps=30,
        encoder=_ENCODER,
        bitrate=_BITRATE,
        gop_size=_GOP,
        faststart=False,
    )

    assert written == 10
    assert _decoded_frame_count(path) == 10
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        assert (stream.codec_context.width, stream.codec_context.height) == (_WIDTH, _HEIGHT)
        assert stream.average_rate == 30


def test_each_frame_is_encoded_before_the_next_is_pulled(tmp_path: Path) -> None:
    """Bounding peak memory is the point of interval batching, so the clip is never collected.

    The generator yields one buffer over and over, rewriting it in place each
    time. An encoder that consumes a frame before pulling the next sees four
    different images; one that collects them first holds four references to the
    same buffer and sees the last value four times over.
    """
    buffer = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)

    def frames() -> Iterator[npt.NDArray[np.uint8]]:
        for value in (0, 70, 140, 210):
            buffer[:] = value
            yield buffer

    path = tmp_path / "clip.mp4"
    encode_frames_to_path(frames(), path, fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False)

    with av.open(str(path)) as container:
        means = [float(frame.to_ndarray(format="rgb24").mean()) for frame in container.decode(video=0)]
    assert len(means) == 4
    assert means == sorted(means)
    assert means[-1] - means[0] > 100


def test_a_failure_part_way_through_leaves_no_partial_file(tmp_path: Path) -> None:
    """A clip that cannot be finished must not look like output to whatever reads the tree next."""
    path = tmp_path / "clip.mp4"
    frames = iter([_frame(10), _frame(20), np.full((_HEIGHT * 2, _WIDTH, 3), 30, dtype=np.uint8)])

    with pytest.raises(ValueError, match="shape"):
        encode_frames_to_path(frames, path, fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False)

    assert not path.exists()


def test_a_producer_that_raises_leaves_no_partial_file(tmp_path: Path) -> None:
    """The same holds when the failure comes from upstream decoding rather than from a bad frame."""
    path = tmp_path / "clip.mp4"

    def frames() -> Iterator[npt.NDArray[np.uint8]]:
        yield _frame(10)
        msg = "decode failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="decode failed"):
        encode_frames_to_path(
            frames(), path, fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False
        )

    assert not path.exists()


def test_writing_no_frames_is_rejected(tmp_path: Path) -> None:
    """An empty MP4 is not a clip; a caller reaching here with nothing has already gone wrong."""
    path = tmp_path / "clip.mp4"

    with pytest.raises(ValueError, match="no frames"):
        encode_frames_to_path(
            iter(()), path, fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False
        )

    assert not path.exists()


def test_a_frame_that_changes_shape_mid_clip_is_rejected(tmp_path: Path) -> None:
    """Geometry is fixed by the first frame, so a later mismatch is a defect and not a resize."""
    frames = iter([_frame(10), np.full((_HEIGHT * 2, _WIDTH, 3), 20, dtype=np.uint8)])

    with pytest.raises(ValueError, match="shape"):
        encode_frames_to_path(
            frames, tmp_path / "clip.mp4", fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False
        )


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.int8])
def test_a_frame_that_is_not_uint8_is_rejected(tmp_path: Path, dtype: type) -> None:
    """Pixel depth is not inferable from shape, so a wider frame fails by name here.

    Left through, it reaches PyAV's own converter and fails there instead --
    same outcome, but with a message about a format string rather than about the
    caller's frames.
    """
    frames = iter([np.zeros((_HEIGHT, _WIDTH, 3), dtype=dtype)])

    with pytest.raises(ValueError, match="frames must be uint8"):
        encode_frames_to_path(
            frames, tmp_path / "clip.mp4", fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False
        )


@pytest.mark.parametrize(
    ("bitrate", "expected"), [("400K", 400_000), ("4M", 4_000_000), ("1.5M", 1_500_000), ("900", 900)]
)
def test_a_bitrate_string_converts_to_bits_per_second(bitrate: str, expected: int) -> None:
    """K and M are decimal multipliers here, matching how FFmpeg reads the same strings."""
    assert _parse_bitrate(bitrate) == expected


def test_the_requested_bitrate_reaches_the_encoder(tmp_path: Path) -> None:
    """A bitrate that never leaves the config cannot be tuned by whoever set it.

    The written file's own ``bit_rate`` is what the encoder achieved, not what it
    was asked for, so it cannot be compared to the request. What the request does
    change is how many bytes come out -- given content with enough entropy to
    spend them on, which flat test frames do not have.
    """
    rng = np.random.default_rng(0)
    noise = [rng.integers(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(30)]

    def size_at(bitrate: str) -> int:
        path = tmp_path / f"{bitrate}.mp4"
        encode_frames_to_path(
            iter(noise), path, fps=30, encoder=_ENCODER, bitrate=bitrate, gop_size=_GOP, faststart=False
        )
        return path.stat().st_size

    assert size_at("8M") > size_at("50K") * 2


def test_a_non_rgb_frame_is_rejected(tmp_path: Path) -> None:
    """Channel order is not inferable, so a wrong-shaped array fails rather than being reinterpreted."""
    frames = iter([np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)])

    with pytest.raises(ValueError, match="shape"):
        encode_frames_to_path(
            frames, tmp_path / "clip.mp4", fps=30, encoder=_ENCODER, bitrate=_BITRATE, gop_size=_GOP, faststart=False
        )


def test_gop_size_places_keyframes_where_it_was_asked_to(tmp_path: Path) -> None:
    """Counting keyframes is the only assertion that fails when the setting is ignored.

    libav accepts encoder settings it does not implement and silently drops them
    -- ``crf`` is one, and libopenh264 encodes byte-identical output for every
    value of it. Asserting the file merely encodes would pass in that case, so
    this reads the keyframes back out.
    """
    path = tmp_path / "clip.mp4"

    encode_frames_to_path(
        (_frame(index * 4) for index in range(30)),
        path,
        fps=30,
        encoder=_ENCODER,
        bitrate=_BITRATE,
        gop_size=10,
        faststart=False,
    )

    with av.open(str(path)) as container:
        keyframes = sum(1 for frame in container.decode(video=0) if frame.key_frame)
    assert keyframes == 3


def test_faststart_puts_the_index_before_the_media(tmp_path: Path) -> None:
    """``faststart`` is a muxer option, so the proof is where ``moov`` lands in the bytes.

    Without it the index is written last, and a reader over object storage pays a
    second range request to find it.
    """
    trailing = tmp_path / "trailing.mp4"
    leading = tmp_path / "leading.mp4"
    for path, faststart in ((trailing, False), (leading, True)):
        encode_frames_to_path(
            (_frame(index * 4) for index in range(10)),
            path,
            fps=30,
            encoder=_ENCODER,
            bitrate=_BITRATE,
            gop_size=10,
            faststart=faststart,
        )

    assert trailing.read_bytes().find(b"moov") > trailing.read_bytes().find(b"mdat")
    assert leading.read_bytes().find(b"moov") < leading.read_bytes().find(b"mdat")
