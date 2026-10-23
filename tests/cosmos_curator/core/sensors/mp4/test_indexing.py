# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build sensor indexes directly from MP4 sources without extracted artifacts."""

import io
import struct

import numpy as np
import pytest

from cosmos_curator.core.sensors.mp4.box_primitives import BoxNotFoundError, MalformedBoxHeaderError
from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.types.types import VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import video_probe
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
@pytest.mark.parametrize("fragmented", [False, True])
def test_index_video_reads_sources_without_box_artifacts(indexer: Mp4Indexer, *, fragmented: bool) -> None:
    """Both backends produce usable indexes for regular and fragmented videos."""
    data = real_mp4.fragmented_clip() if fragmented else real_mp4.CLIP.read_bytes()
    metadata, frame_index = index_video(data, indexer=indexer)
    expected = video_probe(data, index_method=VideoIndexCreationMethod.FULL_DEMUX)
    assert metadata.frame_count == len(frame_index) == len(expected)
    assert metadata.codec_name == expected.codec_name
    assert (metadata.width, metadata.height) == (expected.width, expected.height)
    np.testing.assert_array_equal(frame_index.offset, expected.offset)
    np.testing.assert_array_equal(frame_index.size, expected.size)
    np.testing.assert_array_equal(frame_index.dts_ticks, expected.dts)


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_index_video_preserves_composition_offsets_and_edits(indexer: Mp4Indexer) -> None:
    """Both backends retain composition offsets and the distinction between media and edited PTS."""
    metadata, frame_index = index_video(real_mp4.BFRAME_CLIP, indexer=indexer)
    assert metadata.elst_bytes is not None
    assert frame_index.composition_offset_ns is not None
    assert np.all(np.diff(frame_index.pts_ns[frame_index.pts_sort_order]) > 0)
    assert frame_index.edited_pts_ns[0] != frame_index.pts_ns[0]


def test_index_video_keeps_the_borrowed_stream_open_and_uses_absolute_offsets() -> None:
    """Indexing starts at the file origin even when a borrowed handle was already read."""
    stream = io.BytesIO(real_mp4.CLIP.read_bytes())
    stream.seek(123)
    metadata, frame_index = index_video(stream)
    assert not stream.closed
    assert metadata.frame_count == len(frame_index) == 240


def test_pyav_indexing_selects_the_same_video_stream_as_the_decoder() -> None:
    """The requested stream position takes precedence over libav's preferred stream."""
    data = real_mp4.mjpeg_then_h264_clip()
    first_metadata, first_index = index_video(data, stream_idx=0)
    second_metadata, second_index = index_video(data, stream_idx=1)
    assert first_metadata.codec_name == "mjpeg"
    assert len(first_index) == 30
    assert second_metadata.codec_name == "h264"
    assert len(second_index) == 240


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
    _metadata, frame_index = index_video(stream)
    assert bytes_read < int(frame_index.size.sum())


def test_custom_indexing_refuses_ambiguous_video_stream_selection() -> None:
    """The custom parser cannot silently index a different track from the decoder."""
    with pytest.raises(ValueError, match="exactly one video track"):
        index_video(real_mp4.mjpeg_then_h264_clip(), indexer=Mp4Indexer.CUSTOM)


def test_pyav_indexing_reads_fragmented_bframe_video() -> None:
    """The default backend reads fragment composition offsets outside the custom parser's profile."""
    metadata, frame_index = index_video(real_mp4.gop_fragmented_clip())
    assert metadata.frame_count == len(frame_index) == 240
    assert frame_index.composition_offset_ticks is not None
