# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy metadata and frame-index construction for streaming camera sensors."""

import io
from unittest.mock import Mock

import attrs
import numpy as np
import pytest

from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.sampling.grid import SamplingGrid
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import IndexRecordUnusableError, StreamingCameraSensor
from cosmos_curator.core.sensors.types.types import VideoFrameAlignmentField
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4


@pytest.mark.parametrize("first_property", ["metadata", "frame_index", "timestamps_ns"])
@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_metadata_and_frame_index_are_built_together_once(
    monkeypatch: pytest.MonkeyPatch, first_property: str, indexer: Mp4Indexer
) -> None:
    """Construction does no I/O, and accessing either object populates both, keeping the moov read."""
    video = index_video(real_mp4.CLIP)
    builder = Mock(return_value=video)
    monkeypatch.setattr("cosmos_curator.core.sensors.sensors.streaming_camera_sensor.index_video", builder)
    sensor = StreamingCameraSensor(real_mp4.CLIP, indexer=indexer)
    builder.assert_not_called()
    assert sensor._metadata is None
    assert sensor._frame_index is None
    assert getattr(sensor, first_property) is not None
    assert sensor.metadata is video.metadata
    assert sensor.frame_index is video.frame_index
    assert sensor._moov_bytes is video.moov_bytes
    np.testing.assert_array_equal(sensor.timestamps_ns, video.frame_index.pts_ns[video.frame_index.pts_sort_order])
    builder.assert_called_once_with(real_mp4.CLIP, indexer=indexer, stream_idx=0)


def test_supplied_metadata_and_index_are_used_without_reading_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """A precomputed pair can answer timestamps and metadata without source media."""
    video = index_video(real_mp4.CLIP)
    builder = Mock(side_effect=AssertionError("unexpected indexing"))
    monkeypatch.setattr("cosmos_curator.core.sensors.sensors.streaming_camera_sensor.index_video", builder)
    sensor = StreamingCameraSensor(None, metadata=video.metadata, frame_index=video.frame_index)
    assert sensor.metadata is video.metadata
    assert sensor.frame_index is video.frame_index
    assert len(sensor.timestamps_ns) == video.metadata.frame_count
    builder.assert_not_called()


@pytest.mark.parametrize("provided", ["metadata", "frame_index"])
def test_metadata_and_index_must_be_supplied_together(provided: str) -> None:
    """A half-supplied pair cannot be reconciled by indexing a potentially different source."""
    video = index_video(real_mp4.CLIP)
    with pytest.raises(ValueError, match="supplied together"):
        StreamingCameraSensor(real_mp4.CLIP, **{provided: getattr(video, provided)})


def test_without_precomputed_metadata_a_source_is_required() -> None:
    """An index-only sensor needs its precomputed pair."""
    with pytest.raises(ValueError, match="source is required"):
        StreamingCameraSensor(None)


def test_supplied_metadata_and_index_must_have_the_same_frame_count() -> None:
    """An inconsistent pair cannot provide reliable metadata for the selected timeline."""
    video = index_video(real_mp4.CLIP)
    with pytest.raises(IndexRecordUnusableError, match="count disagrees"):
        StreamingCameraSensor(metadata=attrs.evolve(video.metadata, frame_count=1), frame_index=video.frame_index)


def test_a_borrowed_source_can_be_decoded_after_lazy_indexing() -> None:
    """Indexing and subsequent decode share a borrowed stream without closing it."""
    stream = io.BytesIO(real_mp4.CLIP.read_bytes())
    stream.seek(123)
    sensor = StreamingCameraSensor(stream)
    timestamps = sensor.timestamps_ns[:2]
    start, end = int(timestamps[0]), int(timestamps[-1]) + 1
    spec = SamplingSpec(grid=SamplingGrid(start, end, timestamps, stride_ns=end - start, duration_ns=end - start))
    rows = [
        row
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
        for row in payload.frames()
    ]
    assert [timestamp for _, timestamp, _ in rows] == timestamps.tolist()
    assert not stream.closed


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_lazy_indexing_uses_the_selected_backend_without_box_artifacts(indexer: Mp4Indexer) -> None:
    """The configured backend builds an index from the source on first access."""
    sensor = StreamingCameraSensor(real_mp4.CLIP, indexer=indexer)
    assert len(sensor.frame_index) == sensor.metadata.frame_count == 240
    assert sensor.alignment_field is VideoFrameAlignmentField.PTS_NS
    assert len(sensor.timestamps_ns) == 240


def test_indexing_failure_does_not_leave_a_partial_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsuccessful lazy build retains neither metadata nor an index."""
    builder = Mock(side_effect=OSError("source unavailable"))
    monkeypatch.setattr("cosmos_curator.core.sensors.sensors.streaming_camera_sensor.index_video", builder)
    sensor = StreamingCameraSensor(real_mp4.CLIP)
    with pytest.raises(OSError, match="source unavailable"):
        _ = sensor.metadata
    assert sensor._metadata is None
    assert sensor._frame_index is None


@pytest.mark.parametrize(
    "alignment_field",
    [VideoFrameAlignmentField.EOF_NS, VideoFrameAlignmentField.SOF_NS, VideoFrameAlignmentField.MOE_NS],
)
def test_lazy_indexing_rejects_an_unavailable_alignment_field(alignment_field: VideoFrameAlignmentField) -> None:
    """PyAV timestamps cannot substitute for an unavailable selected sensor timestamp."""
    sensor = StreamingCameraSensor(real_mp4.CLIP, alignment_field=alignment_field)
    assert sensor.metadata.frame_count == 240
    with pytest.raises(IndexRecordUnusableError, match=alignment_field.value):
        _ = sensor.timestamps_ns


def test_default_pts_preserves_media_time_and_edited_pts_is_an_explicit_choice() -> None:
    """The default reports media PTS without silently applying a playback edit."""
    video = index_video(real_mp4.BFRAME_CLIP)
    frame_index = video.frame_index
    media_sensor = StreamingCameraSensor(None, metadata=video.metadata, frame_index=frame_index)
    edited_sensor = StreamingCameraSensor(
        None,
        metadata=video.metadata,
        frame_index=frame_index,
        alignment_field=VideoFrameAlignmentField.EDITED_PTS_NS,
    )
    np.testing.assert_array_equal(media_sensor.timestamps_ns, frame_index.pts_ns[frame_index.pts_sort_order])
    np.testing.assert_array_equal(edited_sensor.timestamps_ns, frame_index.edited_pts_ns[frame_index.pts_sort_order])
    assert media_sensor.timestamps_ns[0] != edited_sensor.timestamps_ns[0]


def test_a_supplied_moov_requires_the_metadata_and_index_it_describes() -> None:
    """A moov alone cannot be paired with an index built later from a potentially different source."""
    with pytest.raises(ValueError, match="moov_bytes requires"):
        StreamingCameraSensor(real_mp4.CLIP, moov_bytes=real_mp4.moov_box(real_mp4.CLIP.read_bytes()))


def test_a_supplied_moov_sets_up_decoding_without_reading_boxes_from_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling hands the supplied moov to the decoder rather than locating one in the source."""
    data = real_mp4.CLIP.read_bytes()
    video = index_video(data)
    walk = Mock(side_effect=AssertionError("unexpected moov read from the source"))
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.video.get_moov_full_walk", walk)
    sensor = StreamingCameraSensor(
        data, metadata=video.metadata, frame_index=video.frame_index, moov_bytes=video.moov_bytes
    )
    timestamps = sensor.timestamps_ns[:2]
    start, end = int(timestamps[0]), int(timestamps[-1]) + 1
    spec = SamplingSpec(grid=SamplingGrid(start, end, timestamps, stride_ns=end - start, duration_ns=end - start))
    rows = [
        row
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
        for row in payload.frames()
    ]
    assert [timestamp for _, timestamp, _ in rows] == timestamps.tolist()
    walk.assert_not_called()


@pytest.mark.parametrize("indexer", list(Mp4Indexer))
def test_lazy_indexing_hands_its_moov_to_the_decoder(monkeypatch: pytest.MonkeyPatch, indexer: Mp4Indexer) -> None:
    """Decoding after a lazy build reuses the moov the build read rather than locating it again."""
    walk = Mock(side_effect=AssertionError("unexpected moov read from the source"))
    monkeypatch.setattr("cosmos_curator.core.sensors.utils.video.get_moov_full_walk", walk)
    sensor = StreamingCameraSensor(real_mp4.CLIP, indexer=indexer)
    timestamps = sensor.timestamps_ns[:2]
    start, end = int(timestamps[0]), int(timestamps[-1]) + 1
    spec = SamplingSpec(grid=SamplingGrid(start, end, timestamps, stride_ns=end - start, duration_ns=end - start))
    rows = [
        row
        for payload in sensor.sample(spec, policy=NearestTimestampPolicy(max_delta_ns=0))
        for row in payload.frames()
    ]
    assert [timestamp for _, timestamp, _ in rows] == timestamps.tolist()
    walk.assert_not_called()
