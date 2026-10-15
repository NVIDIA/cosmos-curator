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
"""Tests for indexing a video with PyAV from its extracted boxes, compared with PyAV on the whole file."""

import io
import struct
from unittest.mock import Mock

import attrs
import av
import numpy as np
import pytest
from av.container import InputContainer

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata
from cosmos_curator.core.sensors.data.video import VideoProbe
from cosmos_curator.core.sensors.mp4 import box_probe
from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, find_all_boxes, find_box, find_box_bytes
from cosmos_curator.core.sensors.mp4.box_probe import (
    BoxLayoutUnsupportedError,
    BoxProbeMismatchError,
    BoxProbeReadError,
    UnsupportedTimescaleError,
    boxes_only_stream,
    index_from_boxes,
)
from cosmos_curator.core.sensors.mp4.fragment_table import CompositionOffsetUnsupportedError
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.mp4.moov_track import EmptySampleTableError
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import (
    HeaderIndexUnavailableError,
    IncompletePacketError,
    probe_open_video_stream,
    video_probe,
)
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4

_PER_SAMPLE_FIELDS = ("dts", "composition_offset", "is_keyframe", "offset", "size", "duration", "is_discard")


def _clip_bytes() -> list[tuple[str, bytes]]:
    gop_fragmented = real_mp4.gop_fragmented_clip()
    return [
        *((clip.name, clip.read_bytes()) for clip in real_mp4.REAL_CLIPS),
        ("one moof per frame", real_mp4.fragmented_clip()),
        ("one moof per keyframe interval", gop_fragmented),
        ("negative fragment offsets", real_mp4.lower_fragment_composition_offsets(gop_fragmented, 15000)),
        ("samples in moov beside fragments", real_mp4.moov_and_fragments_clip()),
    ]


_CLIPS = _clip_bytes()


def _boxes(data: bytes) -> tuple[int, bytes, list[tuple[int, bytes]]]:
    return len(data), real_mp4.moov_box(data), real_mp4.moof_rows(data)


def _probe(file_size: int, moov_bytes: bytes, moof_rows: list[tuple[int, bytes]] | None = None) -> VideoProbe:
    return index_from_boxes(file_size, moov_bytes, moof_rows or ()).probe


def _assert_same_samples(actual: VideoProbe, expected: VideoProbe) -> None:
    for name in _PER_SAMPLE_FIELDS:
        actual_values, expected_values = getattr(actual, name), getattr(expected, name)
        if expected_values is None:
            assert actual_values is None, name
        else:
            np.testing.assert_array_equal(actual_values, expected_values, err_msg=name)
    assert actual.time_base == expected.time_base
    assert actual.extradata == expected.extradata


@pytest.mark.parametrize(("name", "data"), _CLIPS, ids=[name for name, _ in _CLIPS])
def test_index_from_boxes_reads_the_samples_a_full_demux_of_the_source_reads(name: str, data: bytes) -> None:
    """The boxes alone index every sample as the whole file does: media bytes never enter the index."""
    del name
    from_source = video_probe(data, index_method=VideoIndexCreationMethod.FULL_DEMUX)

    _assert_same_samples(_probe(*_boxes(data)), from_source)


def test_index_from_boxes_publishes_libavs_timestamps_for_negative_ctts() -> None:
    """Libav lowers every decode time by the most negative ``ctts`` offset; the index keeps libav's values.

    Presentation times are unchanged, so the published offsets are the stored ones raised by that shift.
    """
    ctts_entries = [(1, 1024), (1, -512), (1, 0), (2, 512)]
    moov = bf.synthetic_moov(frame_count=5, keyframe_index=0, ctts_offset=None, ctts_entries=ctts_entries)
    data = bf.synthetic_nonfragmented_mp4(moov, frame_count=5)
    stored_dts = parse_video_track(moov).dts_ticks
    stored_offsets = np.repeat([offset for _count, offset in ctts_entries], [count for count, _ in ctts_entries])

    probe = _probe(len(data), moov)

    assert probe.composition_offset is not None
    np.testing.assert_array_equal(probe.dts, stored_dts - 512)
    np.testing.assert_array_equal(probe.composition_offset, stored_offsets + 512)


def test_index_from_boxes_reads_fragments_the_box_walker_refuses() -> None:
    """B-frame composition offsets in fragments are outside the walker's profile but within libav's."""
    data = real_mp4.gop_fragmented_clip()
    with pytest.raises(CompositionOffsetUnsupportedError):
        real_mp4.walker_track(data)

    probe = _probe(*_boxes(data))

    assert len(real_mp4.moof_rows(data)) < len(probe)
    assert probe.composition_offset is not None


def test_index_from_boxes_indexes_the_video_stream_libav_ranks_best() -> None:
    """With two video tracks, the index and its metadata are those of libav's best stream."""
    data = real_mp4.mjpeg_then_h264_clip()
    with av.open(io.BytesIO(data)) as container:
        best = container.streams.best("video")
        assert best is not None
        position = [stream.index for stream in container.streams.video].index(best.index)
    from_source = video_probe(data, stream_idx=position, index_method=VideoIndexCreationMethod.FULL_DEMUX)

    index = index_from_boxes(*_boxes(data))

    _assert_same_samples(index.probe, from_source)
    assert index.metadata.codec_name == from_source.codec_name


@pytest.mark.parametrize(
    "data",
    [*(clip.read_bytes() for clip in real_mp4.REAL_CLIPS), real_mp4.fragmented_clip()],
    ids=[*(clip.name for clip in real_mp4.REAL_CLIPS), "one moof per frame"],
)
def test_index_from_boxes_metadata_agrees_with_the_box_walker_on_fields_libav_reports(data: bytes) -> None:
    """Decoder setup, time base, counts, rates, duration, and the copied ``elst`` match the walker's."""
    index_metadata = attrs.asdict(index_from_boxes(*_boxes(data)).metadata)
    walker_metadata = attrs.asdict(Mp4Metadata.from_video_track(real_mp4.walker_track(data)))

    reported = {name for name, value in index_metadata.items() if value is not None} - {"encoder_tag"}
    assert {"codec_name", "extradata", "time_base_den", "duration_ns", "bit_rate_bps"} <= reported
    assert {name: index_metadata[name] for name in reported} == {name: walker_metadata[name] for name in reported}


def test_index_from_boxes_duration_takes_libavs_estimate_for_a_final_sample_stored_with_none() -> None:
    """Libav fills in a zero final duration from the previous sample's; the walker keeps the zero."""
    data = bytearray(real_mp4.fragmented_clip())
    moof_start, moof = real_mp4.moof_rows(bytes(data))[-1]
    tfhd_start, _tfhd_end = find_box(moof, ("moof", "traf", "tfhd"), 0, len(moof))
    flags = int.from_bytes(moof[tfhd_start + 1 : tfhd_start + 4], "big")
    assert flags & 0x08  # default-sample-duration-present
    duration_at = tfhd_start + 8 + (8 if flags & 0x01 else 0) + (4 if flags & 0x02 else 0)
    struct.pack_into(">I", data, moof_start + duration_at, 0)
    data = bytes(data)

    index = index_from_boxes(*_boxes(data))
    walker = real_mp4.walker_track(data)

    dts = index.probe.dts
    assert index.probe.duration is not None
    assert index.probe.duration[-1] == dts[-1] - dts[-2] > 0
    assert walker.duration == dts[-1] - dts[0]
    assert index.metadata.duration_ns == int((dts[-1] - dts[0] + dts[-1] - dts[-2]) * index.probe.time_base * 10**9)
    assert index.metadata.duration_ns > Mp4Metadata.from_video_track(walker).duration_ns


@pytest.mark.parametrize("swap", [False, True], ids=["mjpeg trak first", "h264 trak first"])
def test_index_from_boxes_copies_the_edit_list_of_the_trak_it_indexes(*, swap: bool) -> None:
    """``elst`` comes from the trak whose track ID libav's selected stream reports, wherever it sits."""
    data = real_mp4.mjpeg_then_h264_clip()
    moov = real_mp4.moov_box(data)
    traks = find_all_boxes(moov, "trak", 8, len(moov))
    mjpeg_edit_list, h264_edit_list = (
        find_box_bytes(moov, "elst", *find_box(moov, ("edts",), start, end)) for start, end in traks
    )
    assert mjpeg_edit_list != h264_edit_list
    if swap:
        (first_start, first_end), (second_start, second_end) = ((start - 8, end) for start, end in traks)
        moov = (
            moov[:first_start]
            + moov[second_start:second_end]
            + moov[first_end:second_start]
            + moov[first_start:first_end]
            + moov[second_end:]
        )

    index = index_from_boxes(len(data), moov, real_mp4.moof_rows(data))

    assert index.metadata.codec_name == "mjpeg"
    assert index.metadata.elst_bytes == mjpeg_edit_list


def test_index_from_boxes_reports_the_streams_encoder_tag() -> None:
    """Libav exposes the sample entry's compressor name as the stream's ``encoder``."""
    data = real_mp4.CLIP.read_bytes()
    with av.open(io.BytesIO(data)) as container:
        encoder = container.streams.video[0].metadata["encoder"]

    assert index_from_boxes(*_boxes(data)).metadata.encoder_tag == encoder


def test_index_from_boxes_refuses_a_moov_with_no_video_stream() -> None:
    """With no video stream to index, the boxes cannot describe the video."""
    moov = bf.pack_box("moov", bf.mvhd_box())

    with pytest.raises(BoxNotFoundError, match="no video stream"):
        index_from_boxes(len(moov) + 64, moov)


def test_index_from_boxes_refuses_a_track_id_several_traks_share() -> None:
    """Libav's stream cannot be told apart from the other trak, whose edit list would shift its times."""
    data = real_mp4.mjpeg_then_h264_clip()
    moov = bytearray(real_mp4.moov_box(data))
    for trak_start, trak_end in find_all_boxes(moov, "trak", 8, len(moov)):
        tkhd_start, _tkhd_end = find_box(moov, ("tkhd",), trak_start, trak_end)
        struct.pack_into(">I", moov, tkhd_start + (20 if moov[tkhd_start] == 1 else 12), 7)

    with pytest.raises(BoxProbeMismatchError, match="2 traks share that track ID"):
        index_from_boxes(len(data), bytes(moov), real_mp4.moof_rows(data))


def test_trak_boxes_refuses_a_track_id_no_trak_declares() -> None:
    """Another trak's timescale or edit list would misdescribe this stream's ticks."""
    moov = real_mp4.moov_box(real_mp4.CLIP.read_bytes())

    with pytest.raises(BoxProbeMismatchError, match="no trak has that track ID"):
        box_probe._trak_boxes(moov, 99)


@pytest.mark.parametrize("timescale", [0, 0x90000000], ids=["zero", "above int32"])
def test_index_from_boxes_refuses_a_timescale_libav_replaces_with_one(timescale: int) -> None:
    """Libav labels such ticks 1/1 without rescaling them, so no time base describes them."""
    moov = bf.synthetic_moov(frame_count=3, keyframe_index=0, ctts_offset=None, timescale=timescale)
    data = bf.synthetic_nonfragmented_mp4(moov, frame_count=3)

    with pytest.raises(UnsupportedTimescaleError, match=f"mdhd timescale {timescale} is one libav replaces"):
        _probe(len(data), moov)


def test_index_from_boxes_indexes_a_timescale_of_one() -> None:
    """A real timescale of 1 is the 1/1 time base libav also reports for the ones it replaces."""
    moov = bf.synthetic_moov(frame_count=3, keyframe_index=0, ctts_offset=None, timescale=1)
    data = bf.synthetic_nonfragmented_mp4(moov, frame_count=3)

    index = index_from_boxes(len(data), moov)

    assert (index.metadata.time_base_num, index.metadata.time_base_den) == (1, 1)
    np.testing.assert_array_equal(index.probe.dts, parse_video_track(moov).dts_ticks)


def test_boxes_only_stream_places_moov_first_each_moof_at_its_source_offset_and_zeros_between() -> None:
    """Filler box headers span every gap, and the media they hold is zeros."""
    data = real_mp4.fragmented_clip()
    file_size, moov, moofs = _boxes(data)

    laid_out = boxes_only_stream(file_size, moov, moofs).read()

    assert len(laid_out) == file_size
    assert laid_out[: len(moov)] == moov
    assert laid_out[len(moov) + 4 : len(moov) + 8] == b"free"
    for start, moof_bytes in moofs:
        assert laid_out[start : start + len(moof_bytes)] == moof_bytes
        mdat_start = start + len(moof_bytes)
        assert laid_out[mdat_start + 4 : mdat_start + 8] == b"mdat"
    first, second = moofs[0], moofs[1]
    media = laid_out[first[0] + len(first[1]) + 8 : second[0]]
    assert media
    assert not media.strip(b"\0")


def test_boxes_only_stream_seeks_and_reads_across_box_boundaries() -> None:
    """A read spanning zeros and a box returns both, and a read past the end returns nothing."""
    moov = b"\x00\x00\x00\x10moov" + b"m" * 8
    moof = b"\x00\x00\x00\x0cmoof" + b"f" * 4
    stream = boxes_only_stream(64, moov, [(32, moof)])

    stream.seek(28)
    assert stream.read(8) == b"\0\0\0\0" + moof[:4]
    assert stream.seek(-4, 2) == 60
    assert stream.read() == b"\0\0\0\0"
    assert stream.read(1) == b""


@pytest.mark.parametrize(
    ("moof_rows", "match"),
    [
        ([(20, b"\x00\x00\x00\x08moof")], "does not fit before the first moof"),
        ([(24, b"\x00\x00\x00\x08moof"), (28, b"\x00\x00\x00\x08moof")], "overlaps the next box"),
        ([(28, b"\x00\x00\x00\x08moof"), (40, b"\x00\x00\x00\x08moof")], "4-byte gap"),
        ([(48, b"\x00\x00\x00\x0cmoof" + b"f" * 4)], "4-byte gap"),
    ],
)
def test_boxes_only_stream_refuses_boxes_it_cannot_lay_out_as_libav_would_walk_them(
    moof_rows: list[tuple[int, bytes]], match: str
) -> None:
    """Without a box header in every gap, libav would read zeros as a box and walk a different file."""
    moov = b"\x00\x00\x00\x18moov" + b"m" * 16

    with pytest.raises(BoxLayoutUnsupportedError, match=match):
        boxes_only_stream(64, moov, moof_rows)


def test_boxes_only_stream_writes_a_large_size_header_for_a_gap_past_four_gibibytes() -> None:
    """A 32-bit box size cannot span the gap, so the filler box uses the 64-bit form."""
    moov = b"\x00\x00\x00\x08moov"
    stream = boxes_only_stream(8 + (1 << 32) + 16, moov)

    stream.seek(8)
    assert stream.read(16) == struct.pack(">I4sQ", 1, b"free", (1 << 32) + 16)


def test_index_from_boxes_takes_sync_flags_from_the_header_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """A demux of zero-filled media is not trusted for keyframes, so a missing header index is refused."""

    def no_header_index(
        container: InputContainer, stream_idx: int, index_method: VideoIndexCreationMethod
    ) -> VideoProbe:
        if index_method is VideoIndexCreationMethod.FROM_HEADER:
            msg = "stream does not expose header index entries"
            raise HeaderIndexUnavailableError(msg)
        return probe_open_video_stream(container, stream_idx, index_method)

    monkeypatch.setattr(box_probe, "probe_open_video_stream", no_header_index)

    with pytest.raises(BoxProbeMismatchError, match="no header index"):
        _probe(*_boxes(real_mp4.CLIP.read_bytes()))


def test_index_from_boxes_opens_the_boxes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Libav parses every ``moof`` while opening, so stream selection, header index, and demux share one open."""
    counting_open = Mock(wraps=av.open)
    monkeypatch.setattr(av, "open", counting_open)

    index_from_boxes(*_boxes(real_mp4.fragmented_clip()))

    assert counting_open.call_count == 1


@pytest.mark.parametrize(
    "error",
    [
        av.error.InvalidDataError(1094995529, "Invalid data found when processing input"),
        IncompletePacketError("packet 3 has no dts"),
    ],
)
def test_index_from_boxes_reports_a_libav_failure_as_an_unindexable_video(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    """Only the extracted boxes are read, so libav failing on them says the boxes cannot be indexed."""

    def failing_probe(*_args: object, **_kwargs: object) -> VideoProbe:
        raise error

    monkeypatch.setattr(box_probe, "probe_open_video_stream", failing_probe)

    with pytest.raises(BoxProbeReadError, match="cannot index the extracted boxes") as caught:
        _probe(*_boxes(real_mp4.CLIP.read_bytes()))
    assert caught.value.__cause__ is error


def test_index_from_boxes_refuses_a_track_with_no_samples() -> None:
    """No header index exists for an empty track, so emptiness is reported before that check."""
    moov = bf.synthetic_moov(frame_count=0, keyframe_index=None, ctts_offset=None)
    data = bf.synthetic_nonfragmented_mp4(moov, frame_count=0)

    with pytest.raises(EmptySampleTableError):
        _probe(len(data), moov)
