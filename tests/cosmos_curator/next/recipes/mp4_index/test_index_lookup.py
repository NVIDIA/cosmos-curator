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
import struct
from typing import Any
from unittest.mock import Mock

import lance  # type: ignore[import-untyped]
import pyarrow as pa
import pytest
from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import UnsupportedEditListError
from cosmos_curator.core.utils.storage import s3_client, storage_utils
from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix
from cosmos_curator.next.recipes.mp4_index import index_lookup
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


def test_required_lookup_raises_when_the_manifest_does_not_exist(tmp_path: pathlib.Path) -> None:
    """A required index cannot silently fall back to reading source media."""
    with pytest.raises(FileNotFoundError, match="Configured mp4-index manifest"):
        Mp4IndexLookup.open_required(str(tmp_path / "absent.lance"))


def test_required_lookup_uses_the_manifest_writers_profile_endpoint(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reading a recipe's manifest must use the same profile options as writing it."""
    profile = tmp_path / "profiles.ini"
    profile.write_text(
        "[manifest-profile]\n"
        "aws_access_key_id = test-key\n"
        "aws_secret_access_key = test-secret\n"
        "region = us-east-1\n"
        "endpoint_url = https://s3.example.invalid\n"
    )
    monkeypatch.setattr(s3_client, "S3_PROFILE_PATH", profile)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(profile))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "absent-config"))
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
    opener = Mock(return_value=Mock(count_rows=Mock(return_value=0)))
    monkeypatch.setattr(index_lookup, "open_dataset", opener)
    uri = "s3://bucket/index/manifest.lance"

    Mp4IndexLookup.open_required(uri, storage_profile="manifest-profile")

    expected = storage_utils.get_lance_storage_options(uri, profile_name="manifest-profile")
    assert expected is not None
    assert expected["aws_endpoint"] == "https://s3.example.invalid"
    opener.assert_called_once_with(uri, storage_options=expected)

    opener.reset_mock()
    Mp4IndexLookup.open(uri, s3_profile_name="manifest-profile", endpoint_url="https://cli.example.invalid")
    opener.assert_called_once_with(uri, storage_options={**expected, "aws_endpoint": "https://cli.example.invalid"})


def test_required_lookup_preserves_manifest_access_errors(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unreadable manifest retains its original error."""
    monkeypatch.setattr(index_lookup, "open_dataset", Mock(side_effect=PermissionError("access denied")))
    with pytest.raises(PermissionError, match="access denied"):
        Mp4IndexLookup.open_required(str(tmp_path / "manifest.lance"))


def test_required_lookup_normalizes_local_uris_and_reads_the_source_size(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Required reads handle file URIs and validate the current source without opening its media."""
    video = tmp_path / "front camera.mp4"
    video.write_bytes(bytes(INDEXED_SIZE_BYTES))
    manifest_uri = _write_manifest(tmp_path, [_manifest_row(str(video))])
    lookup = Mp4IndexLookup.open_required(pathlib.Path(manifest_uri).as_uri())
    index, metadata = lookup.require_index_for(video.as_uri(), storage_profile="default")
    assert metadata.frame_count == len(index) == 3
    assert index.dts_ns.tolist() == [_expected_ns(position * FRAME_TICKS) for position in range(3)]
    video.write_bytes(bytes(INDEXED_SIZE_BYTES + 1))
    with pytest.raises(ValueError, match="stale"):
        lookup.require_index_for(str(video), storage_profile="default")
    assert "the index is stale" in capfd.readouterr().err


def test_required_lookup_propagates_a_missing_source(tmp_path: pathlib.Path) -> None:
    """A manifest record alone cannot establish that its source is still current."""
    video = tmp_path / "front.mp4"
    lookup = Mp4IndexLookup.open_required(_write_manifest(tmp_path, [_manifest_row(str(video))]))
    with pytest.raises(FileNotFoundError):
        lookup.require_index_for(str(video), storage_profile="default")


@pytest.mark.parametrize("accessible", [False, True])
def test_required_lookup_checks_s3_size_using_the_requested_profile(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, *, accessible: bool
) -> None:
    """S3 validation uses one HEAD request and propagates a source access failure."""
    video = "s3://bucket/session/front.mp4"
    lookup = Mp4IndexLookup.open_required(_write_manifest(tmp_path, [_manifest_row(video)]))
    client = Mock(spec=S3Client)
    client.stat.return_value.size_bytes = INDEXED_SIZE_BYTES
    factory = Mock(return_value=client)
    monkeypatch.setattr(index_lookup, "get_storage_client", factory)
    if accessible:
        index, metadata = lookup.require_index_for(video, storage_profile="video-profile")
        assert metadata.frame_count == len(index) == 3
    else:
        client.stat.side_effect = PermissionError("source access denied")
        with pytest.raises(PermissionError, match="source access denied"):
            lookup.require_index_for(video, storage_profile="video-profile")
    factory.assert_called_once_with(video, profile_name="video-profile")
    client.stat.assert_called_once_with(S3Prefix(video))


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


def test_lookup_returns_none_for_duplicate_manifest_rows(tmp_path: pathlib.Path) -> None:
    """Ambiguous optional indexes fall back to reading media instead of choosing a row."""
    video = str(tmp_path / "front.mp4")
    row = _manifest_row(video)
    lookup = Mp4IndexLookup.open_required(_write_manifest(tmp_path, [row, row]))

    assert lookup.index_for(video, media_size_bytes=INDEXED_SIZE_BYTES) is None


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
    the ``StreamingCameraSensor`` constructor, which is covered in ``test_sources``.
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


@pytest.mark.parametrize("fault", ["missing", "duplicate", "size", "count"])
def test_required_index_raises_instead_of_falling_back(tmp_path: pathlib.Path, fault: str) -> None:
    """Consumers requiring the manifest cannot silently open a different source of timestamps."""
    video = str(tmp_path / "front.mp4")
    row = _manifest_row(video)
    rows = [row]
    if fault == "missing":
        row["media_location"] = str(tmp_path / "other.mp4")
    elif fault == "duplicate":
        rows.append(row)
    elif fault == "count":
        row["metadata"] = _metadata(frame_count=99)
    lookup = Mp4IndexLookup.open(_write_manifest(tmp_path, rows))
    assert lookup is not None
    with pytest.raises(ValueError, match=r"exactly one|stale|Frame count"):
        lookup.index_for(video, media_size_bytes=INDEXED_SIZE_BYTES + (fault == "size"), required=True)


def test_lookup_preserves_edit_list_and_constructs_both_pts_timelines(tmp_path: pathlib.Path) -> None:
    """Lookup retains playback metadata and obtains interpreted timelines from the sensor library."""
    video = str(tmp_path / "front.mp4")
    elst = struct.pack(">I4sIIIihh", 28, b"elst", 0, 1, 1000, FRAME_TICKS, 1, 0)
    row = _manifest_row(video, metadata=_metadata(elst_bytes=elst))
    lookup = Mp4IndexLookup.open(_write_manifest(tmp_path, [row]))
    assert lookup is not None
    record = lookup.index_for(video, media_size_bytes=INDEXED_SIZE_BYTES, required=True)
    assert record is not None
    index, metadata = record
    assert metadata.elst_bytes == elst
    assert index.elst_bytes == metadata.elst_bytes
    assert index.dts_ns[0] == 0
    assert index.pts_ns[0] == 0
    assert index.edited_pts_ns[0] == -_expected_ns(FRAME_TICKS)


@pytest.mark.parametrize("required", [False, True])
def test_lookup_handles_unsupported_edits_at_the_caller_boundary(
    tmp_path: pathlib.Path, capfd: pytest.CaptureFixture[str], *, required: bool
) -> None:
    """Required inputs propagate the original error; optional inputs log and fall back."""
    video = str(tmp_path / "front.mp4")
    row = _manifest_row(video, metadata=_metadata(elst_bytes=b"bad"))
    lookup = Mp4IndexLookup.open(_write_manifest(tmp_path, [row]))
    assert lookup is not None
    if required:
        with pytest.raises(UnsupportedEditListError, match="truncated"):
            lookup.index_for(video, media_size_bytes=INDEXED_SIZE_BYTES, required=True)
    else:
        assert lookup.index_for(video, media_size_bytes=INDEXED_SIZE_BYTES) is None
        assert "unusable; opening the video" in capfd.readouterr().err
