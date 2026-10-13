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
"""Tests for per-video discovery+extraction+indexing, including the caching guard.

Builds real, minimal box layouts (reusing ``_box_fixtures.py``'s
``synthetic_moov``/``synthetic_fragmented_moov``/``pack_moof``/``real_traf`` for a real,
``get_moov_auto``-locatable ``moov``), since this recipe fetches ``moov``/``moof`` from the video
file itself on a cache miss. ``container_kind`` is passed in by every test here, not detected.

The cache-hit/fast-path tests are the real proof the caching guard works: they monkeypatch
``box_locate.get_moov_auto``/``walk_moofs`` to raise if called at all, so a wrongly-triggered
refetch fails the test loudly rather than just being slow.
"""

import pathlib
import struct
import sys
from typing import Any, Self

import av
import pyarrow as pa
import pytest
import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data.mp4 import InvalidTimeBaseError
from cosmos_curator.core.sensors.mp4 import box_locate
from cosmos_curator.core.sensors.mp4.box_primitives import AmbiguousVideoTrackError, find_box
from cosmos_curator.core.sensors.mp4.box_traversal import BoxLocation, UuidHandler
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.fragment_table import CompositionOffsetUnsupportedError
from cosmos_curator.core.sensors.mp4.moov_track import EmptySampleTableError
from cosmos_curator.next.recipes.mp4_index import processing
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    CONTAINER_KIND_FRAGMENTED,
    CONTAINER_KIND_NONFRAGMENTED,
    VIDEO_MANIFEST_SCHEMA,
    VIDEO_METADATA_SCHEMA,
    make_video_id,
    read_video_record,
    write_bytes_sidecar,
)
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf
from tests.cosmos_curator.core.utils.storage.conftest import FakeStorageClient


def _write_synthetic_nonfragmented_mp4(
    path: pathlib.Path,
    *,
    frame_count: int = 1,
    keyframe_index: int | None = 0,
    ctts_offset: int | None = None,
    include_elst: bool = False,
    elst_content: bytes | None = None,
    timescale: int = 12288,
    tkhd_width: int = 640 << 16,
    tkhd_height: int = 480 << 16,
) -> bytes:
    """Write a minimal real ftyp/moov/mdat file and return moov's own bytes (header included)."""
    moov_box = bf.synthetic_moov(
        frame_count=frame_count,
        keyframe_index=keyframe_index,
        ctts_offset=ctts_offset,
        include_elst=include_elst,
        elst_content=elst_content,
        timescale=timescale,
        tkhd_width=tkhd_width,
        tkhd_height=tkhd_height,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bf.synthetic_nonfragmented_mp4(moov_box, frame_count=frame_count))
    return moov_box


def _write_synthetic_fragmented_mp4(
    path: pathlib.Path, *, moof_rows: list[tuple[int, bytes]] | None = None
) -> tuple[bytes, list[tuple[int, bytes]]]:
    """Write a minimal real ftyp/moov/(moof/mdat)* file, return (moov bytes, [(start, moof bytes), ...])."""
    moov_box = bf.synthetic_fragmented_moov()
    rows = (
        moof_rows
        if moof_rows is not None
        else [
            (0, bf.pack_moof(bf.real_traf(decode_time=0, data_offset=32, sample_size=500, is_keyframe=True))),
            (0, bf.pack_moof(bf.real_traf(decode_time=1, data_offset=32, sample_size=400, is_keyframe=False))),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = bf.pack_box("ftyp", b"isom") + moov_box
    written_rows: list[tuple[int, bytes]] = []
    for _, supplied_moof in rows:
        moof_box = supplied_moof
        if moof_rows is None:
            moof = bytearray(moof_box)
            start, _end = find_box(moof, ("moof", "traf", "trun"), 0, len(moof))
            struct.pack_into(">i", moof, start + 8, len(moof) + 8)
            moof_box = bytes(moof)
        written_rows.append((len(buf), moof_box))
        buf += moof_box
        buf += bf.pack_box("mdat", b"video-bytes" * 50)
    path.write_bytes(buf)
    return moov_box, written_rows


def _config_stub(
    tmp_path: pathlib.Path,
    *,
    container_kind: str = CONTAINER_KIND_NONFRAGMENTED,
    session_id_path_depth: int = 2,
    skip_existing_sidecars: bool = True,
    output_path: pathlib.Path | None = None,
) -> ResolvedMp4IndexConfig:
    """Return a resolved config whose output roots land under tmp_path, for constructing a real _ProcessOneVideo."""
    return ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"path_prefix": str(tmp_path), "globs": ["*.mp4"], "container_kind": container_kind},
            "output": {"path": str(output_path if output_path is not None else tmp_path / "output")},
            "execution": {
                "storage_profile": "default",
                "chunk_bytes": 1024,
                "session_id_path_depth": session_id_path_depth,
                "skip_existing_sidecars": skip_existing_sidecars,
            },
        }
    )


def _call(
    tmp_path: pathlib.Path,
    media_path: pathlib.Path,
    *,
    container_kind: str,
    session_id_path_depth: int = 2,
    skip_existing_sidecars: bool = True,
) -> dict[str, object]:
    config = _config_stub(
        tmp_path,
        container_kind=container_kind,
        session_id_path_depth=session_id_path_depth,
        skip_existing_sidecars=skip_existing_sidecars,
        output_path=tmp_path,
    )
    return processing.process_one_video(
        {"video_id": make_video_id(str(media_path)), "media_location": str(media_path)},
        config=config,
    )


@pytest.fixture
def fragmented_mp4(tmp_path: pathlib.Path) -> pathlib.Path:
    """Copy the real 48-frame, two-fragment MP4 into a discoverable source path."""
    fixture = pathlib.Path(__file__).parents[3] / "core/sensors/mp4/data/multi_sample_trun.mp4"
    media_path = tmp_path / "session-a" / "unit-00" / "video.mp4"
    media_path.parent.mkdir(parents=True)
    media_path.write_bytes(fixture.read_bytes())
    return media_path


@pytest.fixture(params=["negative_start", "start_past_eof", "end_past_eof"])
def invalid_fragmented_mp4(fragmented_mp4: pathlib.Path, request: pytest.FixtureRequest) -> pathlib.Path:
    """Corrupt a sample range in the real MP4's second fragment without changing its payload."""
    data = bytearray(fragmented_mp4.read_bytes())
    with fragmented_mp4.open("rb") as handle:
        moofs = list(processing.walk_moofs(len(data), handle, chunk_bytes=1024))
    moof_start, moof_bytes = moofs[-1]
    trun_start, _end = find_box(moof_bytes, ("moof", "traf", "trun"), 0, len(moof_bytes))
    # This fixture has data-offset, first-sample-flags, and a size for each sample.
    assert int.from_bytes(moof_bytes[trun_start + 1 : trun_start + 4], "big") == 0x205
    sample_count = struct.unpack_from(">I", moof_bytes, trun_start + 4)[0]
    if request.param == "end_past_eof":
        struct.pack_into(">I", data, moof_start + trun_start + 16 + 4 * (sample_count - 1), len(data) + 1)
    else:
        offset = -(2**31) if request.param == "negative_start" else 2**31 - 1
        struct.pack_into(">i", data, moof_start + trun_start + 8, offset)
    fragmented_mp4.write_bytes(data)
    return fragmented_mp4


@pytest.mark.parametrize("cached", [False, True])
def test_process_one_video_rejects_invalid_sample_ranges_without_decoding(
    tmp_path: pathlib.Path,
    invalid_fragmented_mp4: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    cached: bool,
) -> None:
    """Out-of-file sample ranges become failed rows with no derived frame index."""

    def captured_warning(message: str, *args: object) -> None:
        sys.stderr.write(message.format(*args) + "\n")

    def unexpected_decode(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Invalid sample ranges must be rejected without decoding")

    monkeypatch.setattr(processing.logger, "warning", captured_warning)
    monkeypatch.setattr(av, "open", unexpected_decode)
    if cached:
        _call(tmp_path, invalid_fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)
        capfd.readouterr()

    result = _call(tmp_path, invalid_fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == "InvalidFrameByteRangeError"
    assert "byte range" in result["error_message"]
    assert result["frames"] is None
    assert result["metadata"] is None
    captured = capfd.readouterr()
    assert "could not be indexed" in captured.err
    assert ("invalid cached boxes" in captured.err) == cached


def test_process_one_video_indexes_real_multi_sample_fragments(
    fragmented_mp4: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """The unmodified fixture still yields a complete 48-frame index."""
    result = _call(tmp_path, fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)

    assert result["status"] == "success"
    frame_table, metadata = read_video_record(processing.result_table(result))
    assert frame_table.num_rows == metadata["frame_count"] == 48


@pytest.mark.parametrize("layout", ["negative_relative_offset", "end_at_eof"])
def test_process_one_video_accepts_valid_sample_range_boundaries(
    fragmented_mp4: pathlib.Path, tmp_path: pathlib.Path, layout: str
) -> None:
    """Valid signed offsets and samples ending exactly at EOF preserve the full index."""
    original = _call(tmp_path, fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)
    original_frames, _metadata = read_video_record(processing.result_table(original))
    data = bytearray(fragmented_mp4.read_bytes())
    if layout == "end_at_eof":
        last = original_frames.to_pylist()[-1]
        del data[last["offset"] + last["size"] :]
    else:
        with fragmented_mp4.open("rb") as handle:
            moof_start, moof_bytes = list(processing.walk_moofs(len(data), handle, chunk_bytes=1024))[-1]
        tfhd_start, _end = find_box(moof_bytes, ("moof", "traf", "tfhd"), 0, len(moof_bytes))
        trun_start, _end = find_box(moof_bytes, ("moof", "traf", "trun"), 0, len(moof_bytes))
        assert int.from_bytes(moof_bytes[tfhd_start + 1 : tfhd_start + 4], "big") & 1
        base = struct.unpack_from(">Q", moof_bytes, tfhd_start + 8)[0]
        relative_offset = struct.unpack_from(">i", moof_bytes, trun_start + 8)[0]
        struct.pack_into(">Q", data, moof_start + tfhd_start + 8, base + relative_offset + 32)
        struct.pack_into(">i", data, moof_start + trun_start + 8, -32)
    fragmented_mp4.write_bytes(data)

    result = _call(tmp_path, fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED, skip_existing_sidecars=False)

    assert result["status"] == "success"
    frames, _metadata = read_video_record(processing.result_table(result))
    assert frames.num_rows == 48
    assert frames.equals(original_frames)


def test_process_one_video_refetches_cached_out_of_file_ranges(
    fragmented_mp4: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Bad ranges in cached boxes are repaired when the source file remains valid."""

    def captured_warning(message: str, *args: object) -> None:
        sys.stderr.write(message.format(*args) + "\n")

    monkeypatch.setattr(processing.logger, "warning", captured_warning)
    first = _call(tmp_path, fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)
    assert first["status"] == "success"
    cache_path = str(tmp_path / "moof" / f"{first['video_id']}.arrow")
    original = processing.read_arrow_sidecar(cache_path, transport_params=None)
    rows = original.to_pylist()
    moof = bytearray(rows[-1]["moof_bytes"])
    start, _end = find_box(moof, ("moof", "traf", "trun"), 0, len(moof))
    struct.pack_into(">i", moof, start + 8, -(2**31))
    rows[-1]["moof_bytes"] = bytes(moof)
    processing.write_arrow_sidecar(rows, original.schema, cache_path)

    result = _call(tmp_path, fragmented_mp4, container_kind=CONTAINER_KIND_FRAGMENTED)

    assert result["status"] == "success"
    frames, _metadata = read_video_record(processing.result_table(result))
    assert frames.num_rows == 48
    assert processing.read_arrow_sidecar(cache_path, transport_params=None).equals(original)
    assert "invalid cached boxes; refetching from the source" in capfd.readouterr().err


def test_make_fetch_range_threads_transport_params_through_unmodified(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm _make_fetch_range doesn't interpret transport_params -- it just threads it to smart_open.open.

    The real local:// round trip through smart_open is already proven by the cache-miss tests
    below (they fetch a real file via _make_fetch_range end to end); this test is only for the
    one thing those don't cover -- that transport_params passes through byte for byte.
    """
    seen: dict[str, Any] = {}

    class _FakeHandle:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def seek(self, offset: int) -> None:
            seen["offset"] = offset

        def read(self, length: int) -> bytes:
            seen["length"] = length
            return b"x" * length

    def fake_open(uri: str, mode: str, transport_params: dict[str, Any] | None) -> _FakeHandle:
        seen["uri"] = uri
        seen["mode"] = mode
        seen["transport_params"] = transport_params
        return _FakeHandle()

    monkeypatch.setattr(smart_open, "open", fake_open)

    fake_client = object()
    fetch_range = processing._make_fetch_range("s3://bucket/key.mp4", {"client": fake_client})
    result = fetch_range(100, 50)

    assert seen["uri"] == "s3://bucket/key.mp4"
    assert seen["mode"] == "rb"
    assert seen["transport_params"] == {"client": fake_client}
    assert seen["offset"] == 100
    assert seen["length"] == 50
    assert result == b"x" * 50


def test_process_one_video_nonfragmented_extracts_and_indexes_on_a_cache_miss(tmp_path: pathlib.Path) -> None:
    """No moov/frame-index/metadata sidecar exists yet -- everything is fetched, extracted, and written."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    assert result["video_id"] == make_video_id(str(media_path))
    assert result["container_kind"] == "nonfragmented"
    assert set(VIDEO_MANIFEST_SCHEMA.names) <= set(result)
    assert "video_sidecar_location" not in result
    assert not (tmp_path / "frames").exists()

    frame_table, metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.num_rows == 1
    for name in ("eof_ns", "sof_ns", "moe_ns"):
        assert frame_table.schema.field(name).type == pa.int64()
        assert frame_table.column(name).null_count == frame_table.num_rows
    assert set(VIDEO_METADATA_SCHEMA.names) <= set(metadata_row)
    assert (tmp_path / "moov" / f"{result['video_id']}.moov").exists()


def test_process_one_video_manifest_summary_columns_match_nested_metadata(tmp_path: pathlib.Path) -> None:
    """Manifest summaries agree with known source values and nested metadata."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, frame_count=1, keyframe_index=0)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    assert result["source_width"] == 640
    assert result["source_height"] == 480
    assert result["source_frame_rate"] == 24.0
    assert result["source_frame_count"] == 1
    assert result["source_duration_ns"] == 41_666_666  # 1 sample * 512 ticks, 1e9 // 12288 ticks/sec
    assert result["source_video_codec"] == "h264"

    _frame_table, metadata_row = read_video_record(processing.result_table(result))
    assert result["source_width"] == metadata_row["width"]
    assert result["source_height"] == metadata_row["height"]
    assert result["source_frame_rate"] == metadata_row["avg_frame_rate_num"] / metadata_row["avg_frame_rate_den"]
    assert result["source_frame_count"] == metadata_row["frame_count"]
    assert result["source_duration_ns"] == metadata_row["duration_ns"]
    assert result["source_video_codec"] == metadata_row["codec_name"]


def test_process_one_video_tolerates_a_zero_timescale_with_zero_rate_and_duration(tmp_path: pathlib.Path) -> None:
    """A malformed zero mdhd timescale still indexes successfully, matching Mp4Metadata's own tolerance."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, frame_count=1, keyframe_index=0, timescale=0)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    assert result["source_frame_rate"] == 0.0
    assert result["source_duration_ns"] == 0


def test_process_one_video_indexes_a_real_unsigned_32_bit_timescale(tmp_path: pathlib.Path) -> None:
    """A legal mdhd timescale above int32's signed max survives, it isn't refused or truncated.

    mdhd's own timescale field is unsigned 32-bit per ISO/IEC 14496-12; 0x90000000
    (2_415_919_104) is a real, legal value above int32's signed max (2_147_483_647).
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, timescale=0x90000000)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    _frame_table, metadata_row = read_video_record(processing.result_table(result))
    assert metadata_row["time_base_den"] == 0x90000000


def test_process_one_video_indexes_a_real_unsigned_32_bit_tkhd_display_size(tmp_path: pathlib.Path) -> None:
    """A legal tkhd display width/height above int32's signed max survives, it isn't refused or truncated.

    tkhd's own width/height fields are unsigned 32-bit (16.16 fixed-point) per ISO/IEC
    14496-12; 0x90000000 is a real, legal value above int32's signed max.
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, tkhd_width=0x90000000, tkhd_height=0x90000000)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    _frame_table, metadata_row = read_video_record(processing.result_table(result))
    assert metadata_row["track_display_width"] == 0x90000000
    assert metadata_row["track_display_height"] == 0x90000000


def test_process_one_video_succeeds_on_an_edit_list_elst_decoding_cannot_apply_to_pts_ns(
    tmp_path: pathlib.Path,
) -> None:
    """A common ffmpeg output_ts_offset shape (empty edit + normal edit) must not fail extraction.

    ``ffmpeg -i in.mp4 -c copy -output_ts_offset 0.5 out.mp4`` writes a two-entry elst: an empty
    edit (``media_time=-1``) followed by a normal one -- a real, common pattern PyAV indexes fine,
    but ``Mp4FrameIndex.from_video_track``'s own eager elst decoding refuses any multi-entry elst.
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    elst_content = bf.edit_list_box(entries=[(0, -1, 1, 0), (1000, 500, 1, 0)])
    _write_synthetic_nonfragmented_mp4(media_path, include_elst=True, elst_content=elst_content)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    assert result["error_type"] is None


def test_process_one_video_fragmented_extracts_and_indexes_on_a_cache_miss(tmp_path: pathlib.Path) -> None:
    """A fragmented video with no cached moov/moof gets both real-fetched, plus a real frame index."""
    media_path = tmp_path / "session-a" / "unit-00" / "extra" / "video-1.mp4"
    _write_synthetic_fragmented_mp4(media_path)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)

    assert result["status"] == "success"
    assert result["container_kind"] == "fragmented"
    frame_table, _metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.column("dts_ticks").to_pylist() == [0, 1]
    assert frame_table.column("size").to_pylist() == [500, 400]
    assert (tmp_path / "moof" / f"{result['video_id']}.arrow").exists()


def test_ensure_moov_cache_hit_checks_existence_correctly_against_a_non_s3_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached moov sidecar on a non-S3 backend (az://) is found and read back, not forced through S3Prefix.

    The old local _exists always built an S3Prefix regardless of the location's own scheme,
    so a non-S3 output root would fail S3Prefix's own bucket-name validation before ever
    reaching the real existence check.
    """
    moov_sidecar_location = "az://test-container/sess-a/unit-00/video-1.moov"
    fake_client = FakeStorageClient({moov_sidecar_location: b"pre-existing-moov-bytes"})
    monkeypatch.setattr(processing, "read_bytes_sidecar", lambda location, **_kwargs: fake_client.objects[location])

    moov_bytes, served_from_cache = processing._ensure_moov(
        "unused-media-location",
        moov_sidecar_location,
        file_size=0,
        skip_existing_sidecars=True,
        output_storage_client=fake_client,
        input_transport_params=None,
        output_transport_params=None,
    )

    assert moov_bytes == b"pre-existing-moov-bytes"
    assert served_from_cache is True


def test_process_one_video_moof_cache_hit_never_rewalks(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-populated moof table is read back, not rewalked -- walk_moofs is never called."""
    media_path = tmp_path / "session-a" / "unit-00" / "extra" / "video-1.mp4"
    moov_box, moof_rows = _write_synthetic_fragmented_mp4(media_path)
    video_id = make_video_id(str(media_path))
    write_bytes_sidecar(moov_box, str(tmp_path / "moov" / f"{video_id}.moov"))
    from cosmos_curator.core.sensors.mp4.moof_schema import MOOF_TABLE_SCHEMA  # noqa: PLC0415
    from cosmos_curator.next.recipes.mp4_index.contracts import write_arrow_sidecar  # noqa: PLC0415

    rows = [{"moof_start": start, "moof_bytes": content} for start, content in moof_rows]
    write_arrow_sidecar(rows, MOOF_TABLE_SCHEMA, str(tmp_path / "moof" / f"{video_id}.arrow"))

    def _fail_if_called(*args: object, **kwargs: object) -> list[tuple[int, bytes]]:  # noqa: ARG001
        msg = "walk_moofs should not be called on a moof cache hit"
        raise AssertionError(msg)

    monkeypatch.setattr(processing, "walk_moofs", _fail_if_called)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)

    assert result["status"] == "success"
    frame_table, _metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.column("dts_ticks").to_pylist() == [0, 1]


@pytest.mark.parametrize("column", ["moof_start", "moof_bytes"])
@pytest.mark.parametrize("corruption", ["missing", "null", "wrong_type"])
def test_process_one_video_rebuilds_malformed_moof_cache(
    tmp_path: pathlib.Path,
    column: str,
    corruption: str,
    capfd: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readable caches with missing, null, or mistyped columns are replaced from the source."""

    # Route expected warnings through the current capture stream, including in -s runs.
    def captured_warning(message: str, *args: object) -> None:
        sys.stderr.write(message.format(*args) + "\n")

    monkeypatch.setattr(processing.logger, "warning", captured_warning)
    from cosmos_curator.next.recipes.mp4_index.contracts import write_arrow_sidecar  # noqa: PLC0415

    media_path = tmp_path / "session-a" / "unit-00" / "extra" / "video-1.mp4"
    _write_synthetic_fragmented_mp4(media_path)
    first = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)
    assert first["status"] == "success"
    cache_path = str(tmp_path / "moof" / f"{first['video_id']}.arrow")
    original = processing.read_arrow_sidecar(cache_path, transport_params=None)
    if corruption == "missing":
        malformed = original.drop([column])
    else:
        values = (
            pa.nulls(original.num_rows, type=original.schema.field(column).type)
            if corruption == "null"
            else pa.array(["invalid"] * original.num_rows)
        )
        malformed = original.set_column(original.schema.get_field_index(column), column, values)
    write_arrow_sidecar(malformed.to_pylist(), malformed.schema, cache_path)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)

    assert result["status"] == "success"
    frames, _metadata = read_video_record(processing.result_table(result))
    assert frames.column("dts_ticks").to_pylist() == [0, 1]
    assert processing.read_arrow_sidecar(cache_path, transport_params=None).equals(original)
    assert "invalid cached boxes; refetching from the source" in capfd.readouterr().err


def test_process_one_video_reuses_cached_moov_without_touching_the_source_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moov sidecar that already exists is read back, not refetched -- the source file is never touched.

    This is the caching guard's one remaining trust-by-existence path (see process_one_video's
    own module docstring for the accepted risk it carries), still worth proving works: a garbage
    media_location that would fail immediately if actually read confirms get_moov_auto is never
    called when the moov sidecar is already present.
    """
    real_media_path = tmp_path / "real" / "video.mp4"
    real_moov_bytes = _write_synthetic_nonfragmented_mp4(real_media_path, frame_count=3, keyframe_index=0)

    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(bytes(real_media_path.stat().st_size))  # never read on a moov cache hit

    video_id = make_video_id(str(media_path))
    write_bytes_sidecar(real_moov_bytes, str(tmp_path / "moov" / f"{video_id}.moov"))

    def _fail_if_called(*args: object, **kwargs: object) -> bytes:  # noqa: ARG001
        msg = "get_moov_auto must not be called when the moov sidecar already exists"
        raise AssertionError(msg)

    monkeypatch.setattr(box_locate, "get_moov_auto", _fail_if_called)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    frame_table, _metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.num_rows == 3  # from the real, cached moov -- not the garbage media file


def test_process_one_video_refetches_when_the_cached_moov_fails_to_parse(tmp_path: pathlib.Path) -> None:
    """A corrupt cached moov sidecar is treated as a cache miss, not a video failure.

    Real scenario this guards against: an interrupted local write leaves a truncated moov
    sidecar that reads back fine (skip_existing_sidecars only checks existence) but fails to
    parse -- indistinguishable, from here, from a genuinely corrupt source. Without this, the
    video is marked "failed" and, with skip_known_failures defaulting on, stays failed on every
    later run even though the real source is perfectly fine.
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    real_moov_bytes = _write_synthetic_nonfragmented_mp4(media_path, frame_count=3, keyframe_index=0)

    video_id = make_video_id(str(media_path))
    moov_sidecar_location = str(tmp_path / "moov" / f"{video_id}.moov")
    write_bytes_sidecar(b"\x00\x00\x00\x08bad!", moov_sidecar_location)  # not a moov box at all

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    frame_table, _metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.num_rows == 3  # refetched from the real source, not the corrupt cache
    with smart_open.open(moov_sidecar_location, "rb") as handle:
        assert handle.read() == real_moov_bytes  # the corrupt cache entry was overwritten too


def test_process_one_video_does_not_retry_a_genuinely_corrupt_source_when_caching_is_off(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """skip_existing_sidecars=False means nothing was read from cache, so a parse failure isn't retried."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("moov", b"not-real-content"))

    calls: list[str] = []
    real_get_moov_auto = box_locate.get_moov_auto

    def _counting_get_moov_auto(*args: Any, **kwargs: Any) -> bytes:  # noqa: ANN401
        calls.append("get_moov_auto")
        return real_get_moov_auto(*args, **kwargs)

    monkeypatch.setattr(box_locate, "get_moov_auto", _counting_get_moov_auto)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED, skip_existing_sidecars=False)

    assert result["status"] == "failed"
    assert len(calls) == 1  # exactly one fetch attempt -- no retry when nothing came from a cache


def test_process_one_video_does_not_retry_a_fresh_parse_failure_with_caching_still_allowed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A video failing to parse on its very first attempt isn't retried just because caching is allowed.

    skip_existing_sidecars defaults to True, but nothing is cached yet on a fresh run -- the
    retry-on-corrupt-cache path exists for a cached sidecar that fails to parse, not for a
    genuinely unparseable source encountered on the first attempt. Retrying here walks a real,
    possibly large source twice for the same failure.
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + bf.pack_box("moov", b"not-real-content"))

    calls: list[str] = []
    real_get_moov_auto = box_locate.get_moov_auto

    def _counting_get_moov_auto(*args: Any, **kwargs: Any) -> bytes:  # noqa: ANN401
        calls.append("get_moov_auto")
        return real_get_moov_auto(*args, **kwargs)

    monkeypatch.setattr(box_locate, "get_moov_auto", _counting_get_moov_auto)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert len(calls) == 1  # exactly one fetch attempt -- nothing was ever cached to retry against


def test_process_one_video_always_rebuilds_frame_index_and_metadata_even_when_already_present(
    tmp_path: pathlib.Path,
) -> None:
    """Legacy metadata/index sidecars do not affect the new Lance record."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, frame_count=3, keyframe_index=0)

    # Legacy derived sidecars must not influence newly extracted canonical records.
    legacy = tmp_path / "frames" / "legacy.arrow"
    legacy.parent.mkdir()
    legacy.write_bytes(b"stale index")

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "success"
    frame_table, metadata_row = read_video_record(processing.result_table(result))
    assert frame_table.num_rows == 3
    assert metadata_row["codec_name"] is not None


def test_process_one_video_sibling_unit_dirs_do_not_share_sidecars(tmp_path: pathlib.Path) -> None:
    """Videos with the same filename in sibling unit directories retain distinct identities and indexes."""
    media_path_a = tmp_path / "session-a" / "unit-00" / "video.mp4"
    media_path_b = tmp_path / "session-a" / "unit-01" / "video.mp4"
    _write_synthetic_nonfragmented_mp4(media_path_a, frame_count=1, keyframe_index=0, ctts_offset=None)
    _write_synthetic_nonfragmented_mp4(media_path_b, frame_count=2, keyframe_index=0, ctts_offset=None)

    result_a = _call(tmp_path, media_path_a, container_kind=CONTAINER_KIND_NONFRAGMENTED)
    result_b = _call(tmp_path, media_path_b, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result_a["status"] == "success"
    assert result_b["status"] == "success"
    assert result_a["video_id"] != result_b["video_id"]

    # The first video's own frame index is still exactly its own -- not clobbered by the second.
    frame_table_a, _metadata_row_a = read_video_record(processing.result_table(result_a))
    frame_table_b, _metadata_row_b = read_video_record(processing.result_table(result_b))
    assert frame_table_a.num_rows == 1
    assert frame_table_b.num_rows == 2


def test_process_one_video_skip_existing_sidecars_false_forces_a_full_rebuild(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turning the caching guard off refetches moov even though a cached sidecar already exists.

    The path/status alone doesn't prove a rebuild happened -- both are deterministic either way.
    Real evidence: get_moov_auto is called on the second run, which mtime comparisons can't
    reliably show (some filesystems/CI runners have mtime resolution coarser than two writes
    apart in the same test).
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, frame_count=1, keyframe_index=0, ctts_offset=None)

    first = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)
    assert first["status"] == "success"

    real_get_moov_auto = box_locate.get_moov_auto
    calls = []

    def _counting_get_moov_auto(*args: Any, **kwargs: Any) -> bytes:  # noqa: ANN401
        calls.append(1)
        return real_get_moov_auto(*args, **kwargs)

    monkeypatch.setattr(box_locate, "get_moov_auto", _counting_get_moov_auto)

    second = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED, skip_existing_sidecars=False)

    assert second["status"] == "success"
    assert second["video_id"] == first["video_id"]
    assert calls == [1]  # moov was genuinely refetched, not read back from its own cached sidecar


def test_process_one_video_nonfragmented_ambiguous_track_fails(tmp_path: pathlib.Path) -> None:
    """A box-format refusal from the non-fragmented path lands as a failed result, same error type."""
    import struct  # noqa: PLC0415

    video_entry = struct.pack(">6sH", b"\x00" * 6, 1) + b"\x00" * 78
    video_stsd = bf.pack_box("stsd", struct.pack(">II", 0, 1) + bf.pack_box("avc1", video_entry))
    video_trak = bf.pack_box(
        "trak", bf.tkhd_box() + bf.pack_box("mdia", bf.pack_box("minf", bf.pack_box("stbl", video_stsd)))
    )
    moov = bf.pack_box("moov", bf.mvhd_box() + video_trak + video_trak)
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    media_path.parent.mkdir(parents=True, exist_ok=True)
    media_path.write_bytes(bf.pack_box("ftyp", b"isom") + moov + bf.pack_box("mdat", b"data"))

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == AmbiguousVideoTrackError.__name__
    assert set(VIDEO_MANIFEST_SCHEMA.names) <= set(result)


def test_process_one_video_fragmented_unproven_shape_fails(tmp_path: pathlib.Path) -> None:
    """A box-format refusal from the fragmented path (fragment_table.py's own) lands as a failed result."""
    media_path = tmp_path / "session-a" / "unit-00" / "extra" / "video-1.mp4"
    _write_synthetic_fragmented_mp4(
        media_path,
        moof_rows=[
            (0, bf.pack_moof(bf.real_traf(decode_time=0, data_offset=32, sample_size=500, composition_offset=100))),
        ],
    )

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)

    assert result["status"] == "failed"
    assert result["error_type"] == CompositionOffsetUnsupportedError.__name__


def test_process_one_video_metadata_construction_failure_is_a_failed_row_not_a_crash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A VideoTrack that parses fine but fails Mp4Metadata's own construction is one failed row, not a crash."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    def _raise_invalid_time_base(video_track: object) -> None:  # noqa: ARG001 -- signature-compatible stand-in
        msg = "time base denominator is 0"
        raise InvalidTimeBaseError(msg)

    monkeypatch.setattr(processing.Mp4Metadata, "from_video_track", _raise_invalid_time_base)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == InvalidTimeBaseError.__name__


def test_process_one_video_arrow_construction_failure_is_a_failed_row_not_a_crash(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arrow construction errors become per-video failures after successful parsing."""
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path)

    def _raise_arrow_invalid(*args: object, **kwargs: object) -> None:  # noqa: ARG001
        msg = "Value too large to fit in C integer type"
        raise pa.ArrowInvalid(msg)

    monkeypatch.setattr(processing, "_frame_index_array", _raise_arrow_invalid)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == pa.ArrowInvalid.__name__


def test_process_one_video_missing_media_location_fails_cleanly(tmp_path: pathlib.Path) -> None:
    """A missing video file is a per-video failure, not a propagated crash.

    Explicit uris entries get no upfront HEAD, so FileNotFoundError (StorageClient.stat's own
    documented contract for "absent," matching pathlib's identical local behavior) joining the
    caught exception tuple is what turns a missing one into a per-video error instead of a
    crashed task. A genuine infrastructure error (permissions, timeout, a ClientError that isn't
    "missing") is untouched by this and still propagates.
    """
    media_path = tmp_path / "session-a" / "unit-00" / "does-not-exist.mp4"

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert result["video_id"] == make_video_id(str(media_path))
    assert result["media_location"] == str(media_path)


def test_process_one_video_nonfragmented_zero_samples_fails(tmp_path: pathlib.Path) -> None:
    """A moov-only video whose sample table parses to zero samples is refused, not reported success.

    Real finding from a real corpus's own validation: a video with no real fragments
    parsed "successfully" via this fallback path with frame_count=0 -- a genuinely fragmented
    moov's own stbl is always empty by design, so an empty result here can't be trusted to mean
    "genuinely non-fragmented."
    """
    media_path = tmp_path / "session-a" / "unit-00" / "video-1.mp4"
    _write_synthetic_nonfragmented_mp4(media_path, frame_count=0, keyframe_index=None, ctts_offset=None)

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_NONFRAGMENTED)

    assert result["status"] == "failed"
    assert result["error_type"] == EmptySampleTableError.__name__
    assert set(VIDEO_MANIFEST_SCHEMA.names) <= set(result)


def test_process_one_video_fragmented_zero_samples_fails(tmp_path: pathlib.Path) -> None:
    """A fragmented run whose file has zero real moof boxes is refused (NoFragmentsFoundError), not reported success."""
    media_path = tmp_path / "session-a" / "unit-00" / "extra" / "video-1.mp4"
    _write_synthetic_fragmented_mp4(media_path, moof_rows=[])

    result = _call(tmp_path, media_path, container_kind=CONTAINER_KIND_FRAGMENTED, session_id_path_depth=3)

    assert result["status"] == "failed"
    assert result["error_type"] == "NoFragmentsFoundError"


def test_process_one_video_instance_lazily_builds_and_caches_a_local_client(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_ProcessOneVideo resolves get_storage_client on the first row, once, and reuses it after."""
    calls: list[str] = []

    def _fake_get_storage_client(target_path: str, *, profile_name: str) -> None:
        calls.append(target_path)
        assert profile_name == "default"

    monkeypatch.setattr(processing, "get_storage_client", _fake_get_storage_client)

    media_path_a = tmp_path / "session-a" / "unit-00" / "camera_a.mp4"
    media_path_b = tmp_path / "session-a" / "unit-00" / "camera_b.mp4"
    _write_synthetic_nonfragmented_mp4(media_path_a)
    _write_synthetic_nonfragmented_mp4(media_path_b)

    instance = processing._ProcessOneVideo(config=_config_stub(tmp_path))
    result_a = instance(
        pa.Table.from_pylist([{"video_id": make_video_id(str(media_path_a)), "media_location": str(media_path_a)}])
    ).to_pylist()[0]
    result_b = instance(
        pa.Table.from_pylist([{"video_id": make_video_id(str(media_path_b)), "media_location": str(media_path_b)}])
    ).to_pylist()[0]

    assert result_a["status"] == "success"
    assert result_b["status"] == "success"
    assert calls == [str(media_path_a)]  # only the first row triggers resolution


def test_process_one_video_instance_resolves_a_separate_client_per_storage_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run mixing local and s3:// media_locations gets the right client for each, not just the first row's."""
    resolved: dict[str, object] = {}

    def _fake_get_storage_client(target_path: str, *, profile_name: str) -> object:
        assert profile_name == "default"
        client = object() if target_path.startswith("s3://") else None
        resolved[target_path] = client
        return client

    monkeypatch.setattr(processing, "get_storage_client", _fake_get_storage_client)

    seen_clients: list[object] = []

    def _fake_process_one_video(
        row: dict[str, Any],
        *,
        input_storage_client: object,
        output_storage_client: object,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        del kwargs, output_storage_client
        seen_clients.append(input_storage_client)
        return {"status": "success", "video_id": row["video_id"], "media_location": row["media_location"]}

    monkeypatch.setattr(processing, "process_one_video", _fake_process_one_video)

    config_stub = ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"uris": ["/local/a.mp4"], "container_kind": "nonfragmented"},
            "output": {"path": "/local"},
        }
    )
    instance = processing._ProcessOneVideo(config=config_stub)
    instance(pa.Table.from_pylist([{"video_id": "a", "media_location": "/local/a.mp4"}])).to_pylist()[0]
    instance(pa.Table.from_pylist([{"video_id": "b", "media_location": "s3://bucket/b.mp4"}])).to_pylist()[0]
    instance(pa.Table.from_pylist([{"video_id": "c", "media_location": "/local/c.mp4"}])).to_pylist()[0]

    # One resolution per scheme, not per row -- the local moov_sidecar_root shares the local
    # media_locations' own scheme, so it's a cache hit and never triggers its own resolution.
    assert sorted(resolved) == ["/local/a.mp4", "s3://bucket/b.mp4"]
    assert seen_clients[0] is None
    assert seen_clients[1] is resolved["s3://bucket/b.mp4"]
    assert seen_clients[1] is not None
    assert seen_clients[2] is None


@pytest.mark.parametrize(
    ("input_location", "output_root"),
    [("s3://bucket/a.mp4", "/local"), ("/local/a.mp4", "s3://bucket/out")],
)
def test_process_one_video_instance_resolves_a_separate_output_client_from_the_input_one(
    monkeypatch: pytest.MonkeyPatch,
    input_location: str,
    output_root: str,
) -> None:
    """Local/S3 inputs and sidecar destinations resolve independent clients in both directions."""
    resolved: dict[str, object] = {}

    def _fake_get_storage_client(target_path: str, *, profile_name: str) -> object:
        assert profile_name == "default"
        client = object() if target_path.startswith("s3://") else None
        resolved[target_path] = client
        return client

    monkeypatch.setattr(processing, "get_storage_client", _fake_get_storage_client)

    seen: dict[str, object] = {}

    def _fake_process_one_video(
        row: dict[str, Any],
        *,
        input_storage_client: object,
        output_storage_client: object,
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        del kwargs
        seen["input_storage_client"] = input_storage_client
        seen["output_storage_client"] = output_storage_client
        return {"status": "success", "video_id": row["video_id"], "media_location": row["media_location"]}

    monkeypatch.setattr(processing, "process_one_video", _fake_process_one_video)

    config_stub = ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"uris": [input_location], "container_kind": "nonfragmented"},
            "output": {"path": output_root},
        }
    )
    instance = processing._ProcessOneVideo(config=config_stub)
    instance(pa.Table.from_pylist([{"video_id": "a", "media_location": input_location}])).to_pylist()[0]

    assert seen["input_storage_client"] is resolved[input_location]
    assert seen["output_storage_client"] is resolved[config_stub.output.moov_sidecar_root]
    assert (seen["input_storage_client"] is not None) == input_location.startswith("s3://")
    assert (seen["output_storage_client"] is not None) == output_root.startswith("s3://")


def test_extension_uses_extracted_bytes_in_the_complete_record(tmp_path: pathlib.Path) -> None:
    """Extension columns and standard samples are assembled together from one extraction."""
    media = tmp_path / "session" / "unit-00" / "camera.mp4"
    moov = _write_synthetic_nonfragmented_mp4(media)
    calls: list[processing.Mp4Extraction] = []

    class Extension:
        uuid_handlers = ()
        frame_schema = pa.schema([pa.field(name, pa.int64()) for name in ("eof_ns", "sof_ns", "moe_ns")])

        def frame_columns(self, extraction: processing.Mp4Extraction) -> pa.RecordBatch:
            calls.append(extraction)
            return pa.RecordBatch.from_arrays(
                [pa.array([123], type=pa.int64()), pa.nulls(1, type=pa.int64()), pa.array([100], type=pa.int64())],
                schema=self.frame_schema,
            )

    result = processing.process_one_video(
        {"video_id": make_video_id(str(media)), "media_location": str(media)},
        config=_config_stub(tmp_path),
        extension_factory=lambda _media_location: Extension(),
    )

    frames, metadata = read_video_record(processing.result_table(result))
    assert result["status"] == "success"
    assert len(calls) == 1
    assert calls[0].media_location == str(media)
    assert calls[0].moov_bytes == moov
    assert calls[0].moof_rows is None
    assert metadata["frame_count"] == len(calls[0].video_track.size) == 1
    assert frames.select(["eof_ns", "sof_ns", "moe_ns"]).to_pylist() == [{"eof_ns": 123, "sof_ns": None, "moe_ns": 100}]


@pytest.mark.parametrize("invalid", ["standard", "unknown", "duplicate", "type", "count", "schema", "nulls"])
@pytest.mark.parametrize("cached", [False, True])
def test_invalid_extension_columns_propagate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, invalid: str, *, cached: bool
) -> None:
    """Extension contract violations fail the attempt without quarantining or refetching the video."""
    media = tmp_path / "session" / "unit-00" / "video.mp4"
    _write_synthetic_nonfragmented_mp4(media)
    row = {"video_id": make_video_id(str(media)), "media_location": str(media)}
    config = _config_stub(tmp_path)
    if cached:
        assert processing.process_one_video(row, config=config)["status"] == "success"

        def unexpected_source_read(*_args: object, **_kwargs: object) -> None:
            pytest.fail("Unexpected source refetch after an extension contract violation")

        monkeypatch.setattr(processing, "_make_fetch_range", unexpected_source_read)

    class Extension:
        uuid_handlers = ()
        frame_schema = pa.schema([pa.field("eof_ns", pa.int64())])

        def frame_columns(self, _extraction: processing.Mp4Extraction) -> pa.RecordBatch:
            name = {"standard": "dts_ticks", "unknown": "custom"}.get(invalid, "eof_ns")
            dtype = pa.string() if invalid == "type" else pa.int64()
            fields = [pa.field(name, dtype, nullable=invalid != "nulls")]
            arrays = [pa.array(["value"] if invalid == "type" else [123], type=dtype)]
            if invalid == "duplicate":
                fields *= 2
                arrays *= 2
            elif invalid == "count":
                arrays = [pa.array([123, 456], type=dtype)]
            elif invalid == "nulls":
                arrays = [pa.nulls(1, type=dtype)]
            schema = pa.schema(fields)
            if invalid != "schema":
                self.frame_schema = schema
            else:
                self.frame_schema = pa.schema([pa.field("sof_ns", pa.int64())])
            return pa.RecordBatch.from_arrays(arrays, schema=schema)

    message = {
        "standard": "supplied more than once",
        "unknown": "Unknown frame column",
        "duplicate": "supplied more than once",
        "type": "incompatible type",
        "count": "rows for",
        "schema": "declared schema",
        "nulls": "non-nullable but contains nulls",
    }[invalid]
    with pytest.raises(ValueError, match=message):
        processing.process_one_video(row, config=config, extension_factory=lambda _media_location: Extension())


@pytest.mark.parametrize(
    "error_type",
    [ValueError, pa.ArrowInvalid, OverflowError, FileNotFoundError, RuntimeError, IndexError, struct.error],
)
def test_uuid_callback_bugs_propagate_without_cache_recovery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    """Handler bugs must not be mistaken for truncated standard MP4 data."""
    media = tmp_path / "session" / "unit-00" / "video.mp4"
    media.parent.mkdir(parents=True)
    identifier = b"1234567890abcdef"
    moov = bf.synthetic_moov(
        frame_count=1,
        keyframe_index=0,
        ctts_offset=None,
        stbl_extra_children=bf.uuid_box(identifier, b"payload"),
    )
    media.write_bytes(bf.synthetic_nonfragmented_mp4(moov, frame_count=1))
    row = {"video_id": make_video_id(str(media)), "media_location": str(media)}
    config = _config_stub(tmp_path)
    assert processing.process_one_video(row, config=config)["status"] == "success"
    error = error_type("callback bug")

    class Extension:
        frame_schema = pa.schema([pa.field("eof_ns", pa.int64())])

        def __init__(self, _media_location: str) -> None:
            self.uuid_handlers = (
                UuidHandler(identifier, frozenset({("moov", "trak", "mdia", "minf", "stbl")}), self.parse),
            )

        def parse(self, _payload: bytes, _box: BoxLocation) -> None:
            raise error

        def frame_columns(self, _extraction: processing.Mp4Extraction) -> pa.RecordBatch:
            pytest.fail("Failed callbacks must not reach record assembly")

    def unexpected_source_read(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unexpected source refetch after a callback bug")

    monkeypatch.setattr(processing, "_make_fetch_range", unexpected_source_read)
    with pytest.raises(error_type) as caught:
        processing.process_one_video(row, config=config, extension_factory=Extension)
    assert caught.value is error


@pytest.mark.parametrize("failure_stage", ["factory", "uuid_handlers"])
@pytest.mark.parametrize("cached", [False, True])
def test_extension_setup_refusal_does_not_refetch(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    failure_stage: str,
    *,
    cached: bool,
) -> None:
    """Setup refusals retain their per-video error without attempting cache recovery."""

    def captured_warning(message: str, *args: object) -> None:
        sys.stderr.write(message.format(*args) + "\n")

    monkeypatch.setattr(processing.logger, "warning", captured_warning)
    media = tmp_path / "session" / "unit-00" / "video.mp4"
    _write_synthetic_nonfragmented_mp4(media)
    row = {"video_id": make_video_id(str(media)), "media_location": str(media)}
    config = _config_stub(tmp_path)
    if cached:
        assert processing.process_one_video(row, config=config)["status"] == "success"

    class SetupRefusalError(UnindexableVideoError):
        pass

    error = SetupRefusalError("extension setup refused this video")
    setup_calls: list[str] = []

    class Extension:
        def __init__(self, _media_location: str) -> None:
            setup_calls.append(_media_location)
            if failure_stage == "factory":
                raise error

        @property
        def uuid_handlers(self) -> tuple[UuidHandler, ...]:
            raise error

    source_reads: list[str] = []
    original_fetch_range = processing._make_fetch_range

    def fetch_range(uri: str, transport_params: dict[str, Any] | None) -> processing.FetchRange:
        source_reads.append(uri)
        if cached or len(source_reads) > 1:
            msg = "source unavailable on unexpected refetch"
            raise FileNotFoundError(msg)
        return original_fetch_range(uri, transport_params)

    monkeypatch.setattr(processing, "_make_fetch_range", fetch_range)
    result = processing.process_one_video(row, config=config, extension_factory=Extension)

    assert result["status"] == "failed"
    assert result["error_type"] == "SetupRefusalError"
    assert result["error_message"] == str(error)
    assert setup_calls == [str(media)]
    assert len(source_reads) == (0 if cached else 1)
    captured = capfd.readouterr()
    assert str(error) in captured.err
    assert "invalid cached boxes" not in captured.err


def test_extension_parse_refusal_refetches_cached_boxes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """A refusal after parsing cached boxes still retries with a fresh extension."""

    def captured_warning(message: str, *args: object) -> None:
        sys.stderr.write(message.format(*args) + "\n")

    monkeypatch.setattr(processing.logger, "warning", captured_warning)
    media = tmp_path / "session" / "unit-00" / "video.mp4"
    _write_synthetic_nonfragmented_mp4(media)
    row = {"video_id": make_video_id(str(media)), "media_location": str(media)}
    config = _config_stub(tmp_path)
    assert processing.process_one_video(row, config=config)["status"] == "success"

    class Extension:
        uuid_handlers = ()
        frame_schema = pa.schema([pa.field("eof_ns", pa.int64())])

        def __init__(self, _media_location: str) -> None:
            instances.append(self)

        def frame_columns(self, _extraction: processing.Mp4Extraction) -> pa.RecordBatch:
            if len(instances) == 1:
                msg = "cached boxes lack extension data"
                raise UnindexableVideoError(msg)
            return pa.RecordBatch.from_arrays([pa.array([123], type=pa.int64())], schema=self.frame_schema)

    instances: list[Extension] = []
    source_reads: list[str] = []
    original_fetch_range = processing._make_fetch_range

    def fetch_range(uri: str, transport_params: dict[str, Any] | None) -> processing.FetchRange:
        source_reads.append(uri)
        return original_fetch_range(uri, transport_params)

    monkeypatch.setattr(processing, "_make_fetch_range", fetch_range)
    result = processing.process_one_video(row, config=config, extension_factory=Extension)

    assert result["status"] == "success"
    assert len(instances) == 2
    assert source_reads == [str(media)]
    assert result["frames"].values.field("eof_ns").to_pylist() == [123]
    assert "invalid cached boxes; refetching from the source" in capfd.readouterr().err


@pytest.mark.parametrize("error_type", [ValueError, pa.ArrowInvalid, OverflowError, FileNotFoundError])
@pytest.mark.parametrize("failure_stage", ["factory", "columns"])
def test_extension_bugs_propagate_without_cache_recovery(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception], failure_stage: str
) -> None:
    """Unexpected extension failures retain their original exception and do not become failed rows."""
    media = tmp_path / "session" / "unit-00" / "video.mp4"
    _write_synthetic_nonfragmented_mp4(media)
    row = {"video_id": make_video_id(str(media)), "media_location": str(media)}
    config = _config_stub(tmp_path)
    assert processing.process_one_video(row, config=config)["status"] == "success"
    error = error_type("extension bug")

    class Extension:
        uuid_handlers = ()
        frame_schema = pa.schema([pa.field("eof_ns", pa.int64())])

        def __init__(self, _media_location: str) -> None:
            if failure_stage == "factory":
                raise error

        def frame_columns(self, _extraction: processing.Mp4Extraction) -> pa.RecordBatch:
            raise error

    def unexpected_source_read(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Unexpected source refetch after an extension bug")

    monkeypatch.setattr(processing, "_make_fetch_range", unexpected_source_read)
    with pytest.raises(error_type) as caught:
        processing.process_one_video(row, config=config, extension_factory=Extension)
    assert caught.value is error
