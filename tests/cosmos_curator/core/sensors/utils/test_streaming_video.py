# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed packet reads and lazy execution of streaming decode plans."""

import gc
import io
import struct
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import attrs
import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, read_box_header
from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.utils import video
from cosmos_curator.core.sensors.utils.video import (
    CpuVideoDecodeConfig,
    DecoderSetupError,
    StreamingVideoDecoder,
    execute_streaming_decode_plan,
    iter_indexed_video_packets,
    iter_video_frames,
    make_streaming_decode_plan,
    open_video_container,
)
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as fixtures


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
@pytest.mark.parametrize("clip", [fixtures.CLIP, fixtures.BFRAME_CLIP])
def test_plans_keep_frame_identity_and_start_at_a_preceding_sync_sample(indexer: Mp4Indexer, clip: Path) -> None:
    """B-frame targets stay in presentation order even when index rows are reordered."""
    index = index_video(clip, indexer=indexer).frame_index
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
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    opened = Mock(side_effect=AssertionError("unexpected source open"))
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.video.open_data_source", opened)
    decoder = StreamingVideoDecoder(fixtures.CLIP, indexed_video.metadata, index)
    plan = make_streaming_decode_plan(index, np.array([], dtype=np.int64))
    assert plan.preceding_keyframe_idx is None
    assert list(execute_streaming_decode_plan(decoder, plan)) == []
    opened.assert_not_called()


@pytest.mark.parametrize("selected", [[-1], [240], [2, 1]])
def test_invalid_plan_targets_are_rejected(selected: list[int]) -> None:
    """Invalid positions cannot silently select different samples."""
    index = index_video(fixtures.CLIP).frame_index
    with pytest.raises(ValueError, match="frame_indices"):
        make_streaming_decode_plan(index, np.array(selected, dtype=np.int64))


def test_packet_reader_retains_owned_buffers_and_original_timestamps() -> None:
    """Packet bytes and timing survive after the reader has advanced."""
    indexed_video = index_video(fixtures.BFRAME_CLIP)
    index = indexed_video.frame_index
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
        assert packet.time_base == Fraction(indexed_video.metadata.time_base_num, indexed_video.metadata.time_base_den)
    assert not stream.closed


@pytest.mark.parametrize(("offset", "size"), [(-1, 10), (0, 0), (0, -1), (100, 10)])
def test_packet_reader_rejects_invalid_byte_ranges(offset: int, size: int) -> None:
    """Reject corrupt index ranges before passing bytes to FFmpeg."""
    index = index_video(fixtures.CLIP).frame_index
    index = attrs.evolve(index, offset=np.full(len(index), offset), size=np.full(len(index), size))
    with pytest.raises(ValueError, match="byte range"):
        next(iter_indexed_video_packets(io.BytesIO(b"small"), index))


def _record_reads(monkeypatch: pytest.MonkeyPatch, stream: io.BytesIO) -> list[tuple[int, int]]:
    """Record every ``(offset, size)`` read from ``stream``."""
    reads: list[tuple[int, int]] = []
    original_read = stream.read

    def read(size: int = -1) -> bytes:
        offset = stream.tell()
        data = original_read(size)
        reads.append((offset, len(data)))
        return data

    monkeypatch.setattr(stream, "read", read)
    return reads


def _assert_reads_moov_then_packets(
    reads: list[tuple[int, int]], data: bytes, index: Mp4FrameIndex, start_decode_index: int
) -> None:
    """Require top-level box headers and ``moov`` reads, then packets in decode order from ``start_decode_index``."""
    boxes: dict[int, tuple[str, int]] = {}
    position = 0
    while position < len(data):
        box_type, box_size, _ = read_box_header(data, position, len(data))
        boxes[position] = box_type, position + box_size
        position += box_size
    moov_start, moov_end = next((start, end) for start, (box_type, end) in boxes.items() if box_type == "moov")
    first_packet = reads.index((int(index.offset[start_decode_index]), int(index.size[start_decode_index])))
    for offset, size in reads[:first_packet]:
        assert offset in boxes or moov_start <= offset <= offset + size <= moov_end
    packet_rows = range(start_decode_index, start_decode_index + len(reads) - first_packet)
    assert reads[first_packet:] == [(int(index.offset[row]), int(index.size[row])) for row in packet_rows]


@pytest.mark.parametrize("clip", [fixtures.CLIP, fixtures.BFRAME_CLIP])
@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_selected_frames_match_container_decode_reading_only_moov_and_packets(
    monkeypatch: pytest.MonkeyPatch, clip: Path, indexer: Mp4Indexer
) -> None:
    """Both index backends preserve exact pixels through reordered decoding."""
    indexed_video = index_video(clip, indexer=indexer)
    index = indexed_video.frame_index
    expected = {i: frame for i, (_, frame) in enumerate(iter_video_frames(clip)) if i in {103, 104, 108, 239}}
    stream = io.BytesIO(clip.read_bytes())
    reads = _record_reads(monkeypatch, stream)
    container_open = Mock(wraps=av.open)
    monkeypatch.setattr(av, "open", container_open)
    first = make_streaming_decode_plan(index, np.array([103, 103, 104], dtype=np.int64))
    second = make_streaming_decode_plan(index, np.array([104, 108, 239], dtype=np.int64))
    decoder = StreamingVideoDecoder(stream, indexed_video.metadata, index)
    assert not reads
    rows = [*execute_streaming_decode_plan(decoder, first), *execute_streaming_decode_plan(decoder, second)]
    assert [position for position, _ in rows] == [103, 103, 104, 104, 108, 239]
    for position, pixels in rows:
        np.testing.assert_array_equal(pixels, expected[position])
    assert rows[0][1] is rows[1][1]
    assert rows[2][1] is rows[3][1]
    container_open.assert_called_once()
    assert container_open.call_args.args[0] is not stream
    assert first.preceding_keyframe_idx is not None
    _assert_reads_moov_then_packets(reads, stream.getvalue(), index, first.preceding_keyframe_idx)
    decoder.close()
    assert not stream.closed


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_a_supplied_moov_sets_up_the_decoder_reading_only_packets(
    monkeypatch: pytest.MonkeyPatch, indexer: Mp4Indexer
) -> None:
    """With moov supplied, the source is read at packet byte ranges alone; libav sees the recorded ftyp."""
    data = fixtures.BFRAME_CLIP.read_bytes()
    indexed_video = index_video(data, indexer=indexer)
    index = indexed_video.frame_index
    ftyp = indexed_video.metadata.ftyp_bytes
    assert ftyp is not None
    targets = [103, 104, 108]
    expected = {i: frame for i, (_, frame) in enumerate(iter_video_frames(data)) if i in targets}
    stream = io.BytesIO(data)
    reads = _record_reads(monkeypatch, stream)
    container_open = Mock(wraps=av.open)
    monkeypatch.setattr(av, "open", container_open)
    plan = make_streaming_decode_plan(index, np.array(targets, dtype=np.int64))
    decoder = StreamingVideoDecoder(stream, indexed_video.metadata, index, moov_bytes=indexed_video.moov_bytes)

    rows = list(execute_streaming_decode_plan(decoder, plan))

    assert [position for position, _ in rows] == targets
    for position, pixels in rows:
        np.testing.assert_array_equal(pixels, expected[position])
    assert plan.preceding_keyframe_idx is not None
    packet_rows = range(plan.preceding_keyframe_idx, plan.preceding_keyframe_idx + len(reads))
    assert reads == [(int(index.offset[row]), int(index.size[row])) for row in packet_rows]
    boxes = container_open.call_args.args[0]
    boxes.seek(0)
    assert boxes.read(len(ftyp)) == ftyp
    decoder.close()


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_track_ids_libav_reads_as_negative_still_index_and_decode(indexer: Mp4Indexer) -> None:
    """A ``tkhd`` track ID of 2**31 or above is recorded unsigned and still selects its stream."""
    buffer = bytearray(fixtures.BFRAME_CLIP.read_bytes())
    assert buffer.count(b"tkhd") == 1
    tkhd = buffer.index(b"tkhd")
    track_id = 0xFFFFFFF0
    struct.pack_into(">I", buffer, tkhd + 8 + (16 if buffer[tkhd + 4] == 1 else 8), track_id)
    data = bytes(buffer)
    indexed_video = index_video(data, indexer=indexer)
    assert indexed_video.metadata.track_id == track_id
    with open_video_container(io.BytesIO(data), track_id=track_id) as (_, stream):
        assert stream.id < 0
    targets = [0, 103]
    expected = {i: frame for i, (_, frame) in enumerate(iter_video_frames(data)) if i in targets}
    plan = make_streaming_decode_plan(indexed_video.frame_index, np.array(targets, dtype=np.int64))
    for moov_bytes in (indexed_video.moov_bytes, None):
        decoder = StreamingVideoDecoder(data, indexed_video.metadata, indexed_video.frame_index, moov_bytes=moov_bytes)
        rows = list(execute_streaming_decode_plan(decoder, plan))
        decoder.close()
        assert [position for position, _ in rows] == targets
        for position, pixels in rows:
            np.testing.assert_array_equal(pixels, expected[position])


def test_pending_frames_survive_plan_boundaries_and_only_targets_are_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A codec batch can span windows without converting skipped frames."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    converted: list[int] = []

    def frame(frame_idx: int) -> SimpleNamespace:
        def to_ndarray(*, format: str) -> np.ndarray:  # noqa: A002 -- PyAV keyword
            assert format == "rgb24"
            converted.append(frame_idx)
            return np.full((1, 1, 3), frame_idx, dtype=np.uint8)

        return SimpleNamespace(opaque=frame_idx, to_ndarray=to_ndarray)

    context = SimpleNamespace(decode=Mock(return_value=[frame(i) for i in range(4)]))
    monkeypatch.setattr(video, "_open_track_codec_context", Mock(return_value=context))
    decoder = StreamingVideoDecoder(fixtures.CLIP, indexed_video.metadata, index)
    first = make_streaming_decode_plan(index, np.array([1, 1], dtype=np.int64))
    second = make_streaming_decode_plan(index, np.array([1, 3], dtype=np.int64))
    rows = [*execute_streaming_decode_plan(decoder, first), *execute_streaming_decode_plan(decoder, second)]
    assert [position for position, _ in rows] == [1, 1, 1, 3]
    assert converted == [1, 3]
    context.decode.assert_called_once()
    decoder.close()


def test_duplicate_packet_pts_keep_distinct_frame_identities() -> None:
    """Source frame indices distinguish decoded frames with equal packet PTS."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    offsets = np.zeros(len(index), dtype=np.int64)
    offsets[0] = index.dts_ticks[1] - index.dts_ticks[0]
    index = attrs.evolve(index, composition_offset_ticks=offsets)
    decoder = StreamingVideoDecoder(fixtures.CLIP, indexed_video.metadata, index)
    rows = list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.arange(3, dtype=np.int64))))
    assert [position for position, _ in rows] == [0, 1, 2]
    assert rows[0][1] is not rows[1][1]
    decoder.close()


def test_owned_source_closes_on_cancellation_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Window abandonment preserves the pass; cancellation or final release closes it."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    opened: list[io.BufferedReader] = []
    original_open = Path.open

    def open_path(path: Path, mode: str = "rb") -> io.BufferedReader:
        assert mode == "rb"
        stream = original_open(path, "rb")
        opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_path)
    decoder = StreamingVideoDecoder(
        fixtures.CLIP, indexed_video.metadata, index, CpuVideoDecodeConfig(thread_type="SLICE")
    )
    rows = execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.arange(3, dtype=np.int64)))
    next(rows)
    assert not opened[0].closed
    rows.close()
    assert not opened[0].closed
    decoder.close()
    assert opened[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        next(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([3], dtype=np.int64))))
    decoder = StreamingVideoDecoder(fixtures.CLIP, indexed_video.metadata, index)
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
    indexed_video = index_video(source)
    index = indexed_video.frame_index
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
    reads = _record_reads(monkeypatch, reader)
    decoder = StreamingVideoDecoder(
        reader, indexed_video.metadata, index, CpuVideoDecodeConfig(thread_type=thread_type)
    )
    rows = list(execute_streaming_decode_plan(decoder, plan))
    assert [position for position, _ in rows] == targets.tolist()
    _assert_reads_moov_then_packets(reads, source, index, plan.preceding_keyframe_idx)
    for position, pixels in rows:
        np.testing.assert_array_equal(pixels, expected[position])
    decoder.close()


@pytest.mark.parametrize("stream_idx", [0, 1])
def test_supplied_index_selects_the_track_without_container_stream_lookup(stream_idx: int) -> None:
    """Indexed byte ranges and codec metadata select the right video track."""
    source = fixtures.audio_then_two_video_tracks_clip()
    indexed_video = index_video(source, stream_idx=stream_idx)
    index = indexed_video.frame_index
    targets = np.array([2, 5, len(index) - 1], dtype=np.int64)
    expected = {
        i: pixels for i, (_, pixels) in enumerate(iter_video_frames(source, stream_idx=stream_idx)) if i in targets
    }
    decoder = StreamingVideoDecoder(source, indexed_video.metadata, index)
    for position, pixels in execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, targets)):
        np.testing.assert_array_equal(pixels, expected[position])
    decoder.close()


@pytest.mark.parametrize("failure", ["missing", "identity"])
def test_decode_errors_close_owned_media(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str) -> None:
    """Missing output or frame identity releases resources while preserving the error."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
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
    monkeypatch.setattr(video, "_open_track_codec_context", Mock(return_value=context))
    decoder = StreamingVideoDecoder(source, indexed_video.metadata, index)
    with pytest.raises(ValueError, match={"missing": "ended before target", "identity": "identity"}[failure]):
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([0], dtype=np.int64))))
    assert decoder.closed
    assert opened[0].closed
    assert decoder.context is None
    if failure == "missing":
        assert sum(call.args == (None,) for call in context.decode.call_args_list) == 1


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"track_id": 99}, "no video track with ID 99"),
        ({"codec_name": "hevc"}, "names codec 'hevc', but track 1 is 'h264'"),
        ({"codec_name": "not-a-codec"}, "names codec 'not-a-codec'"),
    ],
)
def test_decoder_setup_rejects_metadata_the_moov_does_not_describe(change: dict[str, object], match: str) -> None:
    """The indexed track must exist in the source's moov with the indexed codec."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    decoder = StreamingVideoDecoder(fixtures.CLIP, attrs.evolve(indexed_video.metadata, **change), index)
    with pytest.raises(DecoderSetupError, match=match):
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([0], dtype=np.int64))))
    assert decoder.closed


@pytest.mark.parametrize("failure", ["no moov", "libav"])
def test_decoder_setup_failures_are_typed(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """A missing moov or a libav open error surfaces as a setup error that closes the decoder."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    source = fixtures.CLIP.read_bytes()
    if failure == "no moov":
        moov_start = source.index(fixtures.moov_box(source))
        source = source[: moov_start + 4] + b"free" + source[moov_start + 8 :]
    else:
        error = av.error.InvalidDataError(1094995529, "Invalid data found when processing input")
        monkeypatch.setattr(av, "open", Mock(side_effect=error))
    decoder = StreamingVideoDecoder(source, indexed_video.metadata, index)
    with pytest.raises(DecoderSetupError, match="cannot set up a decoder") as raised:
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([0], dtype=np.int64))))
    assert isinstance(raised.value.__cause__, BoxNotFoundError if failure == "no moov" else av.error.FFmpegError)
    assert decoder.closed


def test_closing_one_decoder_keeps_another_decoders_frame_identities() -> None:
    """Frames freed by one decoder do not erase the identities of another decoder's in-flight packets."""
    indexed_video = index_video(fixtures.BFRAME_CLIP)
    index = indexed_video.frame_index
    finished = StreamingVideoDecoder(fixtures.BFRAME_CLIP, indexed_video.metadata, index)
    running = StreamingVideoDecoder(fixtures.BFRAME_CLIP, indexed_video.metadata, index)
    list(execute_streaming_decode_plan(finished, make_streaming_decode_plan(index, np.arange(10, dtype=np.int64))))
    list(execute_streaming_decode_plan(running, make_streaming_decode_plan(index, np.arange(5, dtype=np.int64))))
    finished.close()
    rows = list(
        execute_streaming_decode_plan(running, make_streaming_decode_plan(index, np.arange(5, 40, dtype=np.int64)))
    )
    assert [position for position, _ in rows] == list(range(5, 40))
    running.close()


def test_short_packet_read_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short source read must not become a truncated decoder packet."""
    index = index_video(fixtures.CLIP).frame_index
    stream = io.BytesIO(fixtures.CLIP.read_bytes())
    monkeypatch.setattr(stream, "read", lambda _size: b"")
    with pytest.raises(ValueError, match="short read"):
        next(iter_indexed_video_packets(stream, index))


def test_backward_plan_closes_the_session() -> None:
    """Reject a rewind rather than silently serving the wrong frame."""
    indexed_video = index_video(fixtures.CLIP)
    index = indexed_video.frame_index
    decoder = StreamingVideoDecoder(fixtures.CLIP, indexed_video.metadata, index)
    list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([3], dtype=np.int64))))
    with pytest.raises(RuntimeError, match="presentation order"):
        list(execute_streaming_decode_plan(decoder, make_streaming_decode_plan(index, np.array([2], dtype=np.int64))))
    assert decoder.closed
