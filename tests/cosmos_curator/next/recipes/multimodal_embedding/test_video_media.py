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

"""Endpoint-inclusive frame sampling and sequential decode validation."""

from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pytest

from cosmos_curator.next.recipes.multimodal_embedding import video_media
from cosmos_curator.next.recipes.multimodal_embedding.video_media import (
    VideoMediaError,
    _stream_timing,
    check_episode_duration,
    decode_selected_frames,
    inspect_video,
    select_frame_indices,
)
from cosmos_curator.next.recipes.multimodal_split.cpu_video_encode import encode_frames_to_path


def _encode_video(path: Path, *, frames: int, gop_size: int = 30) -> None:
    encode_frames_to_path(
        (np.full((16, 32, 3), index % 256, dtype=np.uint8) for index in range(frames)),
        path,
        fps=30,
        encoder="libopenh264",
        bitrate="400K",
        gop_size=gop_size,
        faststart=False,
    )


def test_stream_timeline_requires_metadata_and_uses_stream_duration() -> None:
    """A container-wide or last-frame duration cannot silently choose the sampling bins."""
    stream: Any = SimpleNamespace(start_time=9, duration=60, time_base=Fraction(1, 30), average_rate=30)
    _time_base, start_ns, duration_ns, rate = _stream_timing(stream)
    assert (start_ns, duration_ns, rate) == (300_000_000, 2_000_000_000, 30)

    stream.duration = None
    with pytest.raises(VideoMediaError, match="lacks start_time, duration") as exc_info:
        _stream_timing(stream)
    assert exc_info.value.code == "invalid_timestamps"
    stream.duration = 60
    stream.average_rate = None
    with pytest.raises(VideoMediaError, match="average_rate") as exc_info:
        _stream_timing(stream)
    assert exc_info.value.code == "invalid_timestamps"


def test_duration_mismatch_uses_two_nominal_frames(tmp_path: Path) -> None:
    """A discrepancy at the tolerance is accepted; one nanosecond beyond fails."""
    path = tmp_path / "episode.mp4"
    _encode_video(path, frames=30)
    timeline = inspect_video(path.read_bytes(), frame_count=8)
    assert timeline.duration_ns == 1_000_000_000
    assert timeline.duration_tolerance_ns == 66_666_667
    check_episode_duration(timeline, episode_duration_ns=timeline.duration_ns + timeline.duration_tolerance_ns)
    with pytest.raises(VideoMediaError, match="more than") as exc_info:
        check_episode_duration(timeline, episode_duration_ns=timeline.duration_ns + timeline.duration_tolerance_ns + 1)
    assert exc_info.value.code == "duration_mismatch"
    assert exc_info.value.media_duration_ns == timeline.duration_ns


@pytest.mark.parametrize("threads", [1, 2])
def test_sequential_sampling_handles_b_frames(tmp_path: Path, threads: int) -> None:
    """Selected RGB matches a reference decode in display order, including B-frames."""
    path = tmp_path / "bframes.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width = 64
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 12
        stream.codec_context.max_b_frames = 2
        for index in range(36):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), index * 6, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    timeline = inspect_video(path.read_bytes(), frame_count=8, threads=threads)
    decoded = decode_selected_frames(path.read_bytes(), timeline, threads=threads)
    assert len(decoded.frames) == 8
    with av.open(str(path)) as container:
        sequential = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for selected, sparse in zip(timeline.selected_indices, decoded.frames, strict=True):
        np.testing.assert_array_equal(sparse, sequential[selected])


@pytest.mark.parametrize("frame_count", [1, 4, 8, 15, 300])
def test_endpoint_indices_match_model_card(frame_count: int) -> None:
    """Sampling is exact integer linspace, allowing repeats while including both endpoints."""
    indices = select_frame_indices(frame_count)
    assert indices == tuple(np.linspace(0, frame_count - 1, 8, dtype=np.int64))
    assert indices[0] == 0
    assert indices[-1] == frame_count - 1


def test_short_video_repeats_selected_frames(tmp_path: Path) -> None:
    """Four decoded frames still form the eight-frame model input."""
    path = tmp_path / "short.mp4"
    _encode_video(path, frames=4)
    timeline = inspect_video(path.read_bytes(), frame_count=8)
    decoded = decode_selected_frames(path.read_bytes(), timeline)
    assert decoded.decoded_frame_count == 4
    assert len(decoded.frames) == 8
    assert decoded.frames[0] is decoded.frames[1]


def test_declared_count_mismatch_is_an_item_error(tmp_path: Path) -> None:
    """A truncated or misdeclared stream cannot silently shift the sample indices."""
    path = tmp_path / "clip.mp4"
    _encode_video(path, frames=30)
    timeline = inspect_video(path.read_bytes(), frame_count=8)
    with pytest.raises(VideoMediaError, match="Decoded 30 frames") as exc_info:
        decode_selected_frames(path.read_bytes(), replace(timeline, frame_count=31))
    assert exc_info.value.code == "decode_error"


def test_corrupt_media_is_an_item_error() -> None:
    """Unreadable media and empty samples fail without model weights."""
    with pytest.raises(VideoMediaError) as exc_info:
        inspect_video(b"not an mp4", frame_count=8)
    assert exc_info.value.code == "decode_error"
    with pytest.raises(VideoMediaError, match="no frames"):
        select_frame_indices(0)


def _fake_container_factory(
    *,
    declared: int,
    decoded: int,
    converted: list[int],
    corrupt: int = -1,
    inconsistent: bool = False,
    threads: int = 1,
) -> Callable[..., object]:
    class Container:
        def __init__(self) -> None:
            self.streams = SimpleNamespace(
                video=[
                    SimpleNamespace(
                        frames=declared, start_time=0, duration=decoded, time_base=Fraction(1, 30), average_rate=30
                    )
                ]
            )

        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def decode(self, _stream: object) -> object:
            assert self.streams.video[0].thread_count == threads
            for index in range(decoded):

                def rgb(*, index: int = index, **kwargs: str) -> np.ndarray:
                    assert kwargs["format"] == "rgb24"
                    converted.append(index)
                    return np.zeros((4 if inconsistent and index == 0 else 2, 3, 3), dtype=np.uint8)

                yield SimpleNamespace(is_corrupt=index == corrupt, to_ndarray=rgb)

    return lambda *_args, **_kwargs: Container()


def test_missing_frame_count_reopens_and_retains_only_selected_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count pass creates no RGB arrays; a long sample converts only eight selected frames."""
    converted: list[int] = []
    monkeypatch.setattr(
        video_media.av, "open", _fake_container_factory(declared=0, decoded=300, converted=converted, threads=3)
    )
    timeline = inspect_video(b"payload", frame_count=8, threads=3)
    assert timeline.frame_count == 300
    assert converted == []
    decoded = decode_selected_frames(b"payload", timeline, threads=3)
    assert len(decoded.frames) == 8
    assert converted == list(timeline.selected_indices)
    assert decoded.decoded_frame_count == 300


@pytest.mark.parametrize(("corrupt", "inconsistent"), [(10, False), (-1, True)])
def test_corrupt_frames_and_inconsistent_shapes_fail(
    corrupt: int, monkeypatch: pytest.MonkeyPatch, *, inconsistent: bool
) -> None:
    """Corruption in unselected frames and inconsistent selected RGB shapes are item faults."""
    monkeypatch.setattr(
        video_media.av,
        "open",
        _fake_container_factory(declared=30, decoded=30, converted=[], corrupt=corrupt, inconsistent=inconsistent),
    )
    timeline = inspect_video(b"payload", frame_count=8)
    with pytest.raises(VideoMediaError) as exc_info:
        decode_selected_frames(b"payload", timeline)
    assert exc_info.value.code == "decode_error"
