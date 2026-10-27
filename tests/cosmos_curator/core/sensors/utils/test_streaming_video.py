# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed packet reads and lazy execution of streaming decode plans."""

import gc
import io
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import attrs
import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.utils.video import (
    CpuVideoDecodeConfig,
    StreamingVideoDecoder,
    execute_streaming_decode_plan,
    iter_indexed_video_packets,
    iter_video_frames,
    make_streaming_decode_plan,
)
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as fixtures


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
@pytest.mark.parametrize("clip", [fixtures.CLIP, fixtures.BFRAME_CLIP])
def test_plans_keep_frame_identity_and_start_at_a_preceding_sync_sample(indexer: Mp4Indexer, clip: Path) -> None:
    """B-frame targets stay in presentation order even when index rows are reordered."""
    _, index = index_video(clip, indexer=indexer)
    selected = np.array([103, 103, 104, 108, 108], dtype=np.int64)
    plan = make_streaming_decode_plan(index, selected)
    np.testing.assert_array_equal(plan.frame_indices, [103, 104, 108])
    np.testing.assert_array_equal(plan.counts, [2, 1, 2])
    first_frame_idx = index.pts_sort_order[103]
    assert plan.preceding_keyframe_idx in index.keyframe_indices
    assert 0 <= plan.preceding_keyframe_idx <= first_frame_idx
    assert index.pts_ns[plan.preceding_keyframe_idx] <= index.pts_ns[first_frame_idx]


def test_empty_plan_does_not_open_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty selection does not acquire media or a codec."""
    metadata, index = index_video(fixtures.CLIP)
    opened = Mock(side_effect=AssertionError("unexpected source open"))
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.video.open_data_source", opened)
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index)
    plan = make_streaming_decode_plan(index, np.array([], dtype=np.int64))
    assert plan.preceding_keyframe_idx is None
    assert list(execute_streaming_decode_plan(decoder, plan)) == []
    opened.assert_not_called()


@pytest.mark.parametrize("selected", [[-1], [240], [2, 1]])
def test_invalid_plan_targets_are_rejected(selected: list[int]) -> None:
    """Invalid positions cannot silently select different samples."""
    _, index = index_video(fixtures.CLIP)
    with pytest.raises(ValueError, match="frame_indices"):
        make_streaming_decode_plan(index, np.array(selected, dtype=np.int64))


def test_packet_reader_retains_owned_buffers_and_original_timestamps() -> None:
    """Packet bytes and timing survive after the reader has advanced."""
    metadata, index = index_video(fixtures.BFRAME_CLIP)
    stream = io.BytesIO(fixtures.BFRAME_CLIP.read_bytes())
    start = 100
    packets = list(iter_indexed_video_packets(stream, index, start_decode_index=start))
    gc.collect()
    for frame_idx, packet in packets:
        offset, size = int(index.offset[frame_idx]), int(index.size[frame_idx])
        assert bytes(packet) == stream.getvalue()[offset : offset + size]
        assert packet.opaque == frame_idx
        assert packet.dts == index.dts_ticks[frame_idx]
        assert packet.pts == int(index.dts_ticks[frame_idx]) + int(index.composition_offset_ticks[frame_idx])
        assert packet.time_base == Fraction(metadata.time_base_num, metadata.time_base_den)
    assert not stream.closed


@pytest.mark.parametrize(("offset", "size"), [(-1, 10), (0, 0), (0, -1), (100, 10)])
def test_packet_reader_rejects_invalid_byte_ranges(offset: int, size: int) -> None:
    """Reject corrupt index ranges before passing bytes to FFmpeg."""
    _, index = index_video(fixtures.CLIP)
    index = attrs.evolve(index, offset=np.full(len(index), offset), size=np.full(len(index), size))
    with pytest.raises(ValueError, match="byte range"):
        next(iter_indexed_video_packets(io.BytesIO(b"small"), index))


@pytest.mark.parametrize("clip", [fixtures.CLIP, fixtures.BFRAME_CLIP])
@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_selected_frames_match_container_decode_without_opening_a_container(
    monkeypatch: pytest.MonkeyPatch, clip: Path, indexer: Mp4Indexer
) -> None:
    """Both index backends preserve exact pixels through reordered decoding."""
    metadata, index = index_video(clip, indexer=indexer)
    expected = {i: frame for i, (_, frame) in enumerate(iter_video_frames(clip)) if i in {103, 104, 108, 239}}
    stream = io.BytesIO(clip.read_bytes())
    read_offsets: list[int] = []
    original_read = stream.read

    def read(size: int = -1) -> bytes:
        read_offsets.append(stream.tell())
        return original_read(size)

    monkeypatch.setattr(stream, "read", read)
    monkeypatch.setattr(av, "open", Mock(side_effect=AssertionError("unexpected container open")))
    first = make_streaming_decode_plan(index, np.array([103, 103, 104], dtype=np.int64))
    second = make_streaming_decode_plan(index, np.array([104, 108, 239], dtype=np.int64))
    decoder = StreamingVideoDecoder(stream, metadata, index)
    assert not read_offsets
    rows = [*execute_streaming_decode_plan(decoder, first), *execute_streaming_decode_plan(decoder, second)]
    assert [position for position, _ in rows] == [103, 103, 104, 104, 108, 239]
    for position, pixels in rows:
        np.testing.assert_array_equal(pixels, expected[position])
    assert rows[0][1] is rows[1][1]
    assert rows[2][1] is rows[3][1]
    assert read_offsets[0] == index.offset[first.preceding_keyframe_idx]
    assert min(read_offsets) >= index.offset[first.preceding_keyframe_idx]
    decoder.close()
    assert not stream.closed


def test_pending_frames_survive_plan_boundaries_and_only_targets_are_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A codec batch can span windows without converting skipped frames."""
    metadata, index = index_video(fixtures.CLIP)
    converted: list[int] = []

    def frame(frame_idx: int) -> SimpleNamespace:
        def to_ndarray(*, format: str) -> np.ndarray:  # noqa: A002 -- PyAV keyword
            assert format == "rgb24"
            converted.append(frame_idx)
            return np.full((1, 1, 3), frame_idx, dtype=np.uint8)

        return SimpleNamespace(opaque=frame_idx, to_ndarray=to_ndarray)

    context = SimpleNamespace(decode=Mock(return_value=[frame(i) for i in range(4)]))
    monkeypatch.setattr(av, "CodecContext", SimpleNamespace(create=Mock(return_value=context)))
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index)
    first = make_streaming_decode_plan(index, np.array([1, 1], dtype=np.int64))
    second = make_streaming_decode_plan(index, np.array([1, 3], dtype=np.int64))
    rows = [*execute_streaming_decode_plan(decoder, first), *execute_streaming_decode_plan(decoder, second)]
    assert [position for position, _ in rows] == [1, 1, 1, 3]
    assert converted == [1, 3]
    context.decode.assert_called_once()
    decoder.close()


def test_duplicate_packet_pts_keep_distinct_frame_identities() -> None:
    """Source frame indices distinguish decoded frames with equal packet PTS."""
    metadata, index = index_video(fixtures.CLIP)
    offsets = np.zeros(len(index), dtype=np.int64)
    offsets[0] = index.dts_ticks[1] - index.dts_ticks[0]
    index = attrs.evolve(index, composition_offset_ticks=offsets)
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index)
    rows = list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.arange(3, dtype=np.int64))))
    assert [position for position, _ in rows] == [0, 1, 2]
    assert rows[0][1] is not rows[1][1]
    decoder.close()


def test_owned_source_closes_on_cancellation_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window abandonment preserves the pass; cancellation or final release closes it."""
    metadata, index = index_video(fixtures.CLIP)
    opened: list[io.BufferedReader] = []
    original_open = Path.open

    def open_path(path: Path, mode: str = "rb") -> io.BufferedReader:
        assert mode == "rb"
        stream = original_open(path, "rb")
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index, CpuVideoDecodeConfig(thread_type="SLICE"))
    rows = execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.arange(3, dtype=np.int64)))
    next(rows)
    assert not opened[0].closed
    rows.close()
    assert not opened[0].closed
    decoder.close()
    assert opened[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        next(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([3], dtype=np.int64))))
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index)
    list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([0], dtype=np.int64))))
    del decoder
    gc.collect()
    assert opened[-1].closed


@pytest.mark.parametrize("thread_type", ["SLICE", "AUTO"])
def test_late_start_skips_earlier_gops_and_matches_reordered_container_decode(
    monkeypatch: pytest.MonkeyPatch, thread_type: str
) -> None:
    """Start from a nonzero keyframe, then decode/discard until the first requested B-frame."""
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as sink:
        stream = sink.add_stream("mpeg4", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        stream.codec_context.gop_size = 8
        stream.codec_context.max_b_frames = 2
        for frame_idx in range(32):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), frame_idx * 7, dtype=np.uint8), format="rgb24")
            frame.pts = frame_idx
            sink.mux(stream.encode(frame))
        sink.mux(stream.encode(None))
    source = buffer.getvalue()
    metadata, index = index_video(source)
    targets = np.array([17, 18, 23, 31], dtype=np.int64)
    expected = {i: pixels for i, (_, pixels) in enumerate(iter_video_frames(source)) if i in targets}
    plan = make_streaming_decode_plan(index, targets)
    assert 0 < plan.preceding_keyframe_idx < index.pts_sort_order[targets[0]]
    assert np.any(index.pts_sort_order[1:] < index.pts_sort_order[:-1])
    first_pts = index.pts_ns[index.pts_sort_order[targets[0]]]
    eligible = [
        i for i in index.keyframe_indices if i <= index.pts_sort_order[targets[0]] and index.pts_ns[i] <= first_pts
    ]
    assert plan.preceding_keyframe_idx == eligible[-1]
    reader = io.BytesIO(source)
    offsets: list[int] = []
    original_read = reader.read

    def read(size: int = -1) -> bytes:
        offsets.append(reader.tell())
        return original_read(size)

    monkeypatch.setattr(reader, "read", read)
    monkeypatch.setattr(av, "open", Mock(side_effect=AssertionError("unexpected container open")))
    decoder = StreamingVideoDecoder(reader, metadata, index, CpuVideoDecodeConfig(thread_type=thread_type))
    rows = list(execute_streaming_decode_plan(decoder, plan))
    assert [position for position, _ in rows] == targets.tolist()
    assert offsets[0] == min(offsets) == index.offset[plan.preceding_keyframe_idx]
    for position, pixels in rows:
        np.testing.assert_array_equal(pixels, expected[position])
    decoder.close()


@pytest.mark.parametrize("stream_idx", [0, 1])
def test_supplied_index_selects_the_track_without_container_stream_lookup(stream_idx: int) -> None:
    """Indexed byte ranges and codec metadata select the right video track."""
    source = fixtures.audio_then_two_video_tracks_clip()
    metadata, index = index_video(source, stream_idx=stream_idx)
    targets = np.array([2, 5, len(index) - 1], dtype=np.int64)
    expected = {
        i: pixels for i, (_, pixels) in enumerate(iter_video_frames(source, stream_idx=stream_idx)) if i in targets
    }
    decoder = StreamingVideoDecoder(source, metadata, index)
    for position, pixels in execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, targets)):
        np.testing.assert_array_equal(pixels, expected[position])
    decoder.close()


@pytest.mark.parametrize("failure", ["missing", "identity"])
def test_decode_errors_close_owned_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str) -> None:
    """Missing output or frame identity releases resources while preserving the error."""
    metadata, index = index_video(fixtures.CLIP)
    source = tmp_path / "clip.mp4"
    source.write_bytes(fixtures.CLIP.read_bytes())
    opened: list[io.BufferedReader] = []
    original_open = Path.open

    def open_path(path: Path, mode: str = "rb") -> io.BufferedReader:
        assert mode == "rb"
        stream = original_open(path, "rb")
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    output = [] if failure == "missing" else [SimpleNamespace(opaque=None)]
    context = SimpleNamespace(decode=Mock(return_value=output))
    monkeypatch.setattr(av, "CodecContext", SimpleNamespace(create=Mock(return_value=context)))
    decoder = StreamingVideoDecoder(source, metadata, index)
    with pytest.raises(ValueError, match={"missing": "ended before target", "identity": "identity"}[failure]):
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([0], dtype=np.int64))))
    assert decoder.closed
    assert opened[0].closed
    assert decoder.context is None
    if failure == "missing":
        assert sum(call.args == (None,) for call in context.decode.call_args_list) == 1


def test_short_packet_read_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short source read must not become a truncated decoder packet."""
    _, index = index_video(fixtures.CLIP)
    stream = io.BytesIO(fixtures.CLIP.read_bytes())
    monkeypatch.setattr(stream, "read", lambda _size: b"")
    with pytest.raises(ValueError, match="short read"):
        next(iter_indexed_video_packets(stream, index))


def test_backward_plan_closes_the_session() -> None:
    """Reject a rewind rather than silently serving the wrong frame."""
    metadata, index = index_video(fixtures.CLIP)
    decoder = StreamingVideoDecoder(fixtures.CLIP, metadata, index)
    list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([3], dtype=np.int64))))
    with pytest.raises(RuntimeError, match="presentation order"):
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([2], dtype=np.int64))))
    assert decoder.closed
