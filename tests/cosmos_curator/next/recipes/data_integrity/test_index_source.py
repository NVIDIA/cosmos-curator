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

"""Serving a stream's timeline out of an mp4-index manifest row."""

import pathlib
from fractions import Fraction
from typing import Any

import lance  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa
import pytest
from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.data_integrity.index_source import (
    IndexRecordUnusableError,
    Mp4IndexLookup,
    Mp4IndexSensor,
)
from cosmos_curator.next.recipes.mp4_index.contracts import (
    FRAME_INDEX_SCHEMA,
    RECORD_SCHEMA_VERSION,
    VIDEO_MANIFEST_SCHEMA,
    make_video_id,
)

#: Ticks per second. Deliberately not a divisor of 1e9 -- one tick is 81380.2... ns --
#: so the tick-to-nanosecond conversion has to actually happen for these timestamps to
#: come out right, rather than passing under a timescale that divides evenly.
TIME_BASE_DEN = 12288

#: One frame interval at the fixture's nominal 30 fps, in ticks.
FRAME_TICKS = TIME_BASE_DEN // 30

#: What the fixture rows record as the indexed object's size. Lookups pass this as the
#: size observed now; a test about staleness passes something else.
INDEXED_SIZE_BYTES = 1234


def _metadata(**overrides: object) -> dict[str, Any]:
    """One VIDEO_METADATA_SCHEMA row, with the fields these tests vary pulled to the front."""
    row: dict[str, Any] = {
        "codec_name": "h264",
        "extradata": b"\x01\x64\x00\x1f",
        "time_base_num": 1,
        "time_base_den": TIME_BASE_DEN,
        "frame_count": 3,
        "keyframe_count": 1,
        "width": 640,
        "height": 480,
        "avc_profile_idc": 100,
        "avc_constraint_flags": 0,
        "avc_level_idc": 31,
        "nal_length_size": 4,
        "chroma_format_idc": 1,
        "bit_depth_luma_minus8": 0,
        "bit_depth_chroma_minus8": 0,
        "buffer_size_db": None,
        "max_bitrate_bps": None,
        "avg_bitrate_bps": None,
        "movie_creation_time": 0,
        "movie_modification_time": 0,
        "track_creation_time": 0,
        "track_modification_time": 0,
        "track_matrix": [0] * 9,
        "track_display_width": 0,
        "track_display_height": 0,
        "elst_bytes": None,
        "avg_frame_rate_num": 30,
        "avg_frame_rate_den": 1,
        "bit_rate_bps": 1000,
        "duration_ns": 1_000_000_000,
        "encoder_tag": None,
    }
    row.update(overrides)
    return row


def _frames(count: int = 3, *, composition_offset_ticks: int | None = None) -> list[dict[str, Any]]:
    """``count`` evenly spaced frames in decode order, all sharing one composition offset."""
    return [
        {
            "dts_ticks": index * FRAME_TICKS,
            "composition_offset_ticks": composition_offset_ticks,
            "is_keyframe": index == 0,
            "offset": index * 100,
            "size": 100,
        }
        for index in range(count)
    ]


def _index(rows: list[dict[str, Any]], metadata: Mp4Metadata) -> Mp4FrameIndex:
    return Mp4FrameIndex.from_frame_index_table(
        pa.Table.from_pylist(rows, schema=FRAME_INDEX_SCHEMA),
        time_base_num=metadata.time_base_num,
        time_base_den=metadata.time_base_den,
    )


def _sensor(rows: list[dict[str, Any]] | None = None, **metadata_overrides: object) -> Mp4IndexSensor:
    frame_rows = _frames() if rows is None else rows
    metadata = Mp4Metadata.from_row(_metadata(frame_count=len(frame_rows), **metadata_overrides))
    return Mp4IndexSensor(_index(frame_rows, metadata), metadata)


def _expected_ns(tick: int) -> int:
    return tick * 1_000_000_000 // TIME_BASE_DEN


def test_timeline_is_the_decode_timestamps_converted_to_nanoseconds() -> None:
    """The reported timeline is dts_ticks on the track's time base, not raw ticks."""
    sensor = _sensor()

    assert sensor.timestamps_ns.dtype == np.int64
    assert sensor.timestamps_ns.tolist() == [_expected_ns(index * FRAME_TICKS) for index in range(3)]
    assert sensor.start_ns == 0
    assert sensor.end_ns == _expected_ns(2 * FRAME_TICKS)


def test_timeline_from_the_index_is_reported_as_decode_time() -> None:
    """Not merely "a container clock": decode order ascends, so an in-order finding is a tautology."""
    assert _sensor().timestamp_source is TimestampSource.CONTAINER_DTS


def test_scalar_container_facts_come_off_the_metadata_struct() -> None:
    """Codec and nominal rate are read from the row rather than inferred from the frames."""
    sensor = _sensor(codec_name="hevc", avg_frame_rate_num=30000, avg_frame_rate_den=1001)

    assert sensor.codec_name == "hevc"
    assert sensor.avg_frame_rate == Fraction(30000, 1001)


def test_a_declared_rate_of_zero_stays_zero_rather_than_raising() -> None:
    """Fraction(0) is the library's "no usable header rate" sentinel, which makes rate metrics skip."""
    assert _sensor(avg_frame_rate_num=0, avg_frame_rate_den=1).avg_frame_rate == Fraction(0)


def test_a_zero_denominator_degrades_to_the_same_sentinel() -> None:
    """A denominator mp4-index cannot actually emit still skips rather than raising ZeroDivisionError."""
    assert _sensor(avg_frame_rate_den=0).avg_frame_rate == Fraction(0)


def test_an_absent_composition_offset_column_means_no_reordering() -> None:
    """All-null offsets mean the ctts box is absent, so the container signals no reordering."""
    assert _sensor(_frames(composition_offset_ticks=None)).has_bframes is False


def test_an_explicitly_zero_composition_offset_also_means_no_reordering() -> None:
    """A present-but-zero ctts box states there is no reordering, which is not the same as absent."""
    assert _sensor(_frames(composition_offset_ticks=0)).has_bframes is False


def test_a_non_zero_composition_offset_signals_reordering() -> None:
    """A populated ctts offset is the container saying decode order and display order come apart."""
    assert _sensor(_frames(composition_offset_ticks=FRAME_TICKS)).has_bframes is True


def test_composition_offsets_never_move_the_timeline() -> None:
    """Where DTS and PTS do come apart, this serves DTS and does not split the difference.

    No ``elst`` is applied, since an edit list never shifts decode timestamps, so the
    timeline is the same either way and :attr:`timestamp_source` reports which it is.
    """
    delayed = _sensor(_frames(composition_offset_ticks=FRAME_TICKS))

    assert delayed.has_bframes is True
    assert delayed.timestamps_ns.tolist() == _sensor().timestamps_ns.tolist()
    assert delayed.start_ns == 0


def test_a_record_with_no_frames_is_refused() -> None:
    """An empty timeline has no start or end, so it is rejected at construction rather than later."""
    with pytest.raises(ValueError, match="no frames"):
        _sensor([])


def test_a_sensor_timeline_replaces_the_indexed_one_and_says_so() -> None:
    """A recorder's own clock beats the container's, and the index contributes only the scalars."""
    recorded = np.array([5_000, 15_000, 25_000], dtype=np.int64)
    metadata = Mp4Metadata.from_row(_metadata())

    sensor = Mp4IndexSensor(_index(_frames(), metadata), metadata, sensor_timestamps_ns=recorded)

    assert sensor.timestamps_ns.tolist() == recorded.tolist()
    assert sensor.timestamp_source is TimestampSource.SENSOR
    assert sensor.codec_name == "h264"


def test_a_sensor_timeline_of_the_wrong_length_makes_the_record_unusable() -> None:
    """Matched by position, so a disagreeing count means the alignment is unknown, not partial."""
    metadata = Mp4Metadata.from_row(_metadata())

    with pytest.raises(IndexRecordUnusableError, match="2 observations but the mp4-index record has 3"):
        Mp4IndexSensor(
            _index(_frames(), metadata),
            metadata,
            sensor_timestamps_ns=np.array([5_000, 15_000], dtype=np.int64),
        )


def test_streaming_without_a_batch_size_yields_the_whole_timeline_at_once() -> None:
    """batch_size 0 means one batch, matching CameraSensor's contract."""
    batches = list(_sensor().stream_timestamps())

    assert len(batches) == 1
    assert batches[0].tolist() == _sensor().timestamps_ns.tolist()


def test_streaming_splits_into_consecutive_batches_with_a_short_tail() -> None:
    """Batches partition the timeline in order, so streaming and one-shot see the same samples."""
    batches = list(_sensor(_frames(5)).stream_timestamps(batch_size=2))

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert np.concatenate(batches).tolist() == _sensor(_frames(5)).timestamps_ns.tolist()


def test_streaming_rejects_a_negative_batch_size() -> None:
    """Mirrors CameraSensor rather than silently treating a negative size as "all"."""
    with pytest.raises(ValueError, match="batch_size must be non-negative"):
        list(_sensor().stream_timestamps(batch_size=-1))


def _manifest_row(media_location: str, **overrides: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "video_id": make_video_id(media_location),
        "source_session_id": "session-00",
        "camera_key": "front",
        "session_camera_id": "abc123",
        "media_location": media_location,
        "media_size_bytes": INDEXED_SIZE_BYTES,
        "source_width": 640,
        "source_height": 480,
        "source_frame_rate": 30.0,
        "source_frame_count": 3,
        "source_duration_ns": 1_000_000_000,
        "source_video_codec": "h264",
        "metadata": _metadata(),
        "frames": _frames(),
        "moov_location": "s3://bucket/index/moov.bin",
        "moof_table_location": None,
        "container_kind": "nonfragmented",
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "extraction_contract_id": "contract-1",
    }
    row.update(overrides)
    return row


def _write_manifest(path: pathlib.Path, rows: list[dict[str, Any]]) -> str:
    uri = str(path / "manifest.lance")
    lance.write_dataset(pa.Table.from_pylist(rows, schema=VIDEO_MANIFEST_SCHEMA), uri)
    return uri


def test_lookup_returns_none_when_the_manifest_does_not_exist(tmp_path: pathlib.Path) -> None:
    """A supplemental input that has not been built yet is not a reason to refuse the run."""
    assert Mp4IndexLookup.open(str(tmp_path / "absent.lance")) is None


def test_lookup_serves_a_sensor_for_an_indexed_video(tmp_path: pathlib.Path) -> None:
    """The ordinary hit: a row for this media_location becomes a sensor over its frame index."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None
    sensor = lookup.sensor_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES)

    assert sensor is not None
    assert sensor.codec_name == "h264"
    assert sensor.timestamps_ns.tolist() == [_expected_ns(index * FRAME_TICKS) for index in range(3)]


def test_lookup_returns_none_for_a_video_the_manifest_does_not_hold(tmp_path: pathlib.Path) -> None:
    """Only part of a dataset is ever indexed, so a miss is ordinary and the caller opens the media."""
    uri = _write_manifest(tmp_path, [_manifest_row(str(tmp_path / "front.mp4"))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for(str(tmp_path / "rear.mp4"), media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_matches_a_differently_spelled_path_for_the_same_file(tmp_path: pathlib.Path) -> None:
    """Keys are normalised the way mp4-index wrote them, so an equivalent spelling still hits."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for(str(tmp_path / "." / "front.mp4"), media_size_bytes=INDEXED_SIZE_BYTES) is not None


def test_lookup_declines_a_scheme_the_manifest_could_never_hold(tmp_path: pathlib.Path) -> None:
    """An az:// source is not looked up at all, rather than normalised into a confident non-match."""
    uri = _write_manifest(tmp_path, [_manifest_row(str(tmp_path / "front.mp4"))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for("az://container/front.mp4", media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_falls_back_when_a_found_row_cannot_answer(tmp_path: pathlib.Path) -> None:
    """A frame count that disagrees with the sidecar is a miss, not a failure: open the video instead."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert (
        lookup.sensor_for(
            str(video),
            media_size_bytes=INDEXED_SIZE_BYTES,
            sensor_timestamps_ns=np.array([1, 2], dtype=np.int64),
        )
        is None
    )


def test_lookup_refuses_a_manifest_written_at_an_older_record_schema(tmp_path: pathlib.Path) -> None:
    """Reading rows this version cannot interpret is a configuration mistake, reported rather than skipped."""
    uri = _write_manifest(
        tmp_path,
        [_manifest_row(str(tmp_path / "front.mp4"), record_schema_version=RECORD_SCHEMA_VERSION - 1)],
    )

    with pytest.raises(ValueError, match="record_schema_version"):
        Mp4IndexLookup.open(uri)


def test_a_row_with_no_frames_is_unusable_rather_than_a_plain_error() -> None:
    """Typed so the lookup falls back to the media; a bare ValueError would reach the session runner."""
    metadata = Mp4Metadata.from_row(_metadata(frame_count=0))

    with pytest.raises(IndexRecordUnusableError, match="contains no frames"):
        Mp4IndexSensor(_index(_frames(0), metadata), metadata)


def test_lookup_falls_back_when_a_row_holds_no_frames(tmp_path: pathlib.Path) -> None:
    """A row with nothing to read is a stream the index does not cover, not a stream that failed."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(
        tmp_path,
        [_manifest_row(str(video), metadata=_metadata(frame_count=0), frames=_frames(0))],
    )

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_falls_back_when_a_row_contradicts_itself(tmp_path: pathlib.Path) -> None:
    """read_video_record rejects a frame count that disagrees with the metadata; that is still a miss."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video), metadata=_metadata(frame_count=99))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_refuses_a_row_describing_a_video_of_a_different_size(tmp_path: pathlib.Path) -> None:
    """mp4-index never re-indexes a committed location, so a replaced video keeps the old row."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.sensor_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES + 1) is None


def test_lookup_refuses_a_row_it_cannot_check_and_says_so_once(tmp_path: pathlib.Path) -> None:
    """An unreadable size means the guard is unavailable, which is reported once rather than per stream."""
    uri = _write_manifest(
        tmp_path,
        [_manifest_row(str(tmp_path / "front.mp4")), _manifest_row(str(tmp_path / "rear.mp4"))],
    )

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    warnings: list[str] = []
    sink_id = logger.add(warnings.append, level="WARNING")
    try:
        assert lookup.sensor_for(str(tmp_path / "front.mp4"), media_size_bytes=None) is None
        assert lookup.sensor_for(str(tmp_path / "rear.mp4"), media_size_bytes=None) is None
    finally:
        logger.remove(sink_id)

    assert sum("cannot read the size of" in warning for warning in warnings) == 1
