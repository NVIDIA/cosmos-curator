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

"""Reading one video's frame index back out of a manifest row."""

import pathlib
from typing import Any

import lance  # type: ignore[import-untyped]
import pyarrow as pa
import pytest
from loguru import logger

from cosmos_curator.next.recipes.mp4_index.contracts import (
    RECORD_SCHEMA_VERSION,
    VIDEO_MANIFEST_SCHEMA,
    make_video_id,
)
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup

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
        "track_id": 1,
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


def _expected_ns(tick: int) -> int:
    return tick * 1_000_000_000 // TIME_BASE_DEN


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


def test_lookup_serves_the_frame_index_for_an_indexed_video(tmp_path: pathlib.Path) -> None:
    """The ordinary hit: a row for this media_location unpacks into the sensor library's types."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None
    record = lookup.index_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES)

    assert record is not None
    index, metadata = record
    assert metadata.codec_name == "h264"
    assert index.dts_ns.tolist() == [_expected_ns(position * FRAME_TICKS) for position in range(3)]


def test_lookup_returns_none_for_a_video_the_manifest_does_not_hold(tmp_path: pathlib.Path) -> None:
    """Only part of a dataset is ever indexed, so a miss is ordinary and the caller opens the media."""
    uri = _write_manifest(tmp_path, [_manifest_row(str(tmp_path / "front.mp4"))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.index_for(str(tmp_path / "rear.mp4"), media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_matches_a_differently_spelled_path_for_the_same_file(tmp_path: pathlib.Path) -> None:
    """Keys are normalised the way the writer spelled them, so an equivalent spelling still hits."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.index_for(str(tmp_path / "." / "front.mp4"), media_size_bytes=INDEXED_SIZE_BYTES) is not None


def test_lookup_declines_a_scheme_the_manifest_could_never_hold(tmp_path: pathlib.Path) -> None:
    """An az:// source is not looked up at all, rather than normalised into a confident non-match."""
    uri = _write_manifest(tmp_path, [_manifest_row(str(tmp_path / "front.mp4"))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.index_for("az://container/front.mp4", media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_refuses_a_manifest_written_at_an_older_record_schema(tmp_path: pathlib.Path) -> None:
    """Reading rows this version cannot interpret is a configuration mistake, reported rather than skipped."""
    uri = _write_manifest(
        tmp_path,
        [_manifest_row(str(tmp_path / "front.mp4"), record_schema_version=RECORD_SCHEMA_VERSION - 1)],
    )

    with pytest.raises(ValueError, match="record_schema_version"):
        Mp4IndexLookup.open(uri)


def test_lookup_serves_a_row_holding_no_frames_and_leaves_the_refusal_to_the_sensor(
    tmp_path: pathlib.Path,
) -> None:
    """Emptiness is the sensor's judgement to make; the reader only reports what the row holds.

    The recipe turns this into a fallback by catching ``IndexRecordUnusableError`` from
    ``StreamingCameraSensor.from_index``, which is covered in ``test_sources``.
    """
    video = tmp_path / "front.mp4"
    uri = _write_manifest(
        tmp_path,
        [_manifest_row(str(video), metadata=_metadata(frame_count=0), frames=_frames(0))],
    )

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None
    record = lookup.index_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES)

    assert record is not None
    assert len(record[0]) == 0


def test_lookup_falls_back_when_a_row_contradicts_itself(tmp_path: pathlib.Path) -> None:
    """read_video_record rejects a frame count that disagrees with the metadata; that is still a miss."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video), metadata=_metadata(frame_count=99))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.index_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES) is None


def test_lookup_refuses_a_row_describing_a_video_of_a_different_size(tmp_path: pathlib.Path) -> None:
    """A committed location is never re-indexed, so a replaced video keeps the old row."""
    video = tmp_path / "front.mp4"
    uri = _write_manifest(tmp_path, [_manifest_row(str(video))])

    lookup = Mp4IndexLookup.open(uri)
    assert lookup is not None

    assert lookup.index_for(str(video), media_size_bytes=INDEXED_SIZE_BYTES + 1) is None


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
        assert lookup.index_for(str(tmp_path / "front.mp4"), media_size_bytes=None) is None
        assert lookup.index_for(str(tmp_path / "rear.mp4"), media_size_bytes=None) is None
    finally:
        logger.remove(sink_id)

    assert sum("cannot read the size of" in warning for warning in warnings) == 1
