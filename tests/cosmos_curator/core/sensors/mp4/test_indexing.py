# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build sensor indexes directly from MP4 sources without extracted artifacts."""

import io
import struct

import numpy as np
import pytest

from cosmos_curator.core.sensors.mp4.box_primitives import (
    MAX_FTYP_BYTES,
    BoxNotFoundError,
    BoxTooLargeError,
    MalformedBoxHeaderError,
)
from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import video_probe
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
@pytest.mark.parametrize("fragmented", [False, True])
def test_index_video_reads_sources_without_box_artifacts(indexer: Mp4Indexer, *, fragmented: bool) -> None:
    """Both backends produce usable indexes for regular and fragmented videos, with the source's moov."""
    data = real_mp4.fragmented_clip() if fragmented else real_mp4.CLIP.read_bytes()
    video = index_video(data, indexer=indexer)
    expected = video_probe(data, index_method=VideoIndexCreationMethod.FULL_DEMUX)
    assert video.metadata.frame_count == len(video.frame_index) == len(expected)
    assert video.metadata.codec_name == expected.codec_name
    assert (video.metadata.width, video.metadata.height) == (expected.width, expected.height)
    np.testing.assert_array_equal(video.frame_index.offset, expected.offset)
    np.testing.assert_array_equal(video.frame_index.size, expected.size)
    np.testing.assert_array_equal(video.frame_index.dts_ticks, expected.dts)
    assert video.moov_bytes == real_mp4.moov_box(data)


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("with_ftyp", [True, False])
def test_index_video_records_the_leading_ftyp(indexer: Mp4Indexer, *, fragmented: bool, with_ftyp: bool) -> None:
    """Both backends keep the source's ftyp verbatim, and record none for a source without one."""
    data = bytearray(real_mp4.fragmented_clip() if fragmented else real_mp4.CLIP.read_bytes())
    assert data[4:8] == b"ftyp"
    ftyp = bytes(data[: struct.unpack_from(">I", data)[0]])
    if not with_ftyp:
        data[4:8] = b"free"
    assert index_video(bytes(data), indexer=indexer).metadata.ftyp_bytes == (ftyp if with_ftyp else None)


def test_index_video_refuses_an_oversized_ftyp() -> None:
    """An in-bounds ftyp declaring more than the limit is refused rather than read into memory."""
    size = MAX_FTYP_BYTES + 1
    data = struct.pack(">I4s", size, b"ftyp") + bytes(size - 8) + real_mp4.CLIP.read_bytes()

    with pytest.raises(BoxTooLargeError, match=f"declares {size} bytes"):
        index_video(data)


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_index_video_preserves_composition_offsets_and_edits(indexer: Mp4Indexer) -> None:
    """Both backends retain composition offsets and the distinction between media and edited PTS."""
    video = index_video(real_mp4.BFRAME_CLIP, indexer=indexer)
    frame_index = video.frame_index
    assert video.metadata.elst_bytes is not None
    assert frame_index.composition_offset_ns is not None
    assert np.all(np.diff(frame_index.pts_ns[frame_index.pts_sort_order]) > 0)
    assert frame_index.edited_pts_ns[0] != frame_index.pts_ns[0]


def test_index_video_keeps_the_borrowed_stream_open_and_uses_absolute_offsets() -> None:
    """Indexing starts at the file origin even when a borrowed handle was already read."""
    stream = io.BytesIO(real_mp4.CLIP.read_bytes())
    stream.seek(123)
    video = index_video(stream)
    assert not stream.closed
    assert video.metadata.frame_count == len(video.frame_index) == 240


def test_pyav_indexing_selects_the_same_video_stream_as_the_decoder() -> None:
    """The requested stream position takes precedence over libav's preferred stream."""
    data = real_mp4.mjpeg_then_h264_clip()
    first = index_video(data, stream_idx=0)
    second = index_video(data, stream_idx=1)
    assert first.metadata.codec_name == "mjpeg"
    assert len(first.frame_index) == 30
    assert second.metadata.codec_name == "h264"
    assert len(second.frame_index) == 240


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_index_video_rejects_a_source_without_moov(indexer: Mp4Indexer) -> None:
    """A missing required box is reported by both backends."""
    with pytest.raises(BoxNotFoundError, match="moov"):
        index_video(struct.pack(">I4s", 8, b"free"), indexer=indexer)


def test_index_video_rejects_a_box_extending_beyond_the_source() -> None:
    """Source bounds are checked before a declared box size is read."""
    with pytest.raises(MalformedBoxHeaderError, match="past"):
        index_video(struct.pack(">I4s", 1000, b"moov"))


def test_index_video_seeks_past_source_media_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source reads contain box metadata and header probes rather than compressed media."""
    stream = io.BytesIO(real_mp4.CLIP.read_bytes())
    original_read = stream.read
    bytes_read = 0

    def read(size: int = -1) -> bytes:
        nonlocal bytes_read
        data = original_read(size)
        bytes_read += len(data)
        return data

    monkeypatch.setattr(stream, "read", read)
    frame_index = index_video(stream).frame_index
    assert bytes_read < int(frame_index.size.sum())


def test_custom_indexing_refuses_ambiguous_video_stream_selection() -> None:
    """The custom parser cannot silently index a different track from the decoder."""
    with pytest.raises(ValueError, match="exactly one video track"):
        index_video(real_mp4.mjpeg_then_h264_clip(), indexer=Mp4Indexer.CUSTOM)


def test_pyav_indexing_reads_fragmented_bframe_video() -> None:
    """The default backend reads fragment composition offsets outside the custom parser's profile."""
    video = index_video(real_mp4.gop_fragmented_clip())
    assert video.metadata.frame_count == len(video.frame_index) == 240
    assert video.frame_index.composition_offset_ticks is not None
