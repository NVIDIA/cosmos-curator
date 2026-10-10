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
"""Direct tests for the (session, camera) naming key, and sidecar round-trips through real local files."""

import pathlib
from typing import Any, Self

import attrs
import pyarrow as pa
import pytest
import smart_open
from pyarrow import ipc

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.next.recipes.mp4_index import contracts
from cosmos_curator.next.recipes.mp4_index.contracts import (
    FRAME_INDEX_SCHEMA,
    UnresolvableMediaLocationError,
    derive_session_and_camera,
    make_session_camera_id,
    read_video_record,
)
from tests.cosmos_curator.next.recipes.mp4_index._record_fixtures import _FRAME_ROWS, _METADATA_ROW


def test_make_session_camera_id_is_deterministic() -> None:
    """The same (session, camera) always produces the same id."""
    first = make_session_camera_id("session-a", "camera_front")
    second = make_session_camera_id("session-a", "camera_front")

    assert first == second


def test_make_session_camera_id_differs_by_session_or_camera() -> None:
    """A different session or camera produces a different id -- no accidental collisions."""
    base = make_session_camera_id("session-a", "camera_front")

    assert make_session_camera_id("session-b", "camera_front") != base
    assert make_session_camera_id("session-a", "camera_rear") != base


def test_derive_session_and_camera_reads_depth_from_the_end() -> None:
    """source_session_id is the path component depth directories up from the file.

    camera_key is everything after that -- the remaining directory component(s) plus the
    filename stem -- not just the stem alone.
    """
    session, camera = derive_session_and_camera(
        "s3://bucket/session-a/unit-00/camera_front.mp4", session_id_path_depth=2
    )

    assert session == "session-a"
    assert camera == "unit-00/camera_front"


def test_derive_session_and_camera_supports_a_deeper_layout() -> None:
    """A deeper {session}/unit-NN/subdir/camera.mp4 layout works at depth=3."""
    session, camera = derive_session_and_camera(
        "s3://bucket/session-a/unit-10/extra/camera_front.mp4", session_id_path_depth=3
    )

    assert session == "session-a"
    assert camera == "unit-10/extra/camera_front"


def test_derive_session_and_camera_works_for_a_local_path() -> None:
    """A location with no URI scheme is handled the same way as an s3:// one."""
    session, camera = derive_session_and_camera("/data/session-a/unit-00/camera_front.mp4", session_id_path_depth=2)

    assert session == "session-a"
    assert camera == "unit-00/camera_front"


def test_derive_session_and_camera_does_not_collide_across_sibling_directories() -> None:
    """Two videos sharing a session and filename stem, differing only by the directory between them.

    Real, reproduced collision this test guards against: session_id_path_depth=2 previously
    discarded the unit-NN component entirely (neither part of source_session_id nor
    camera_key), so sess-a/unit-00/camera.mp4 and sess-a/unit-01/camera.mp4 -- two
    real, independently discovered videos -- derived the identical sidecar key and would
    silently overwrite one another's sidecars.
    """
    first = derive_session_and_camera("s3://bucket/sess-a/unit-00/camera.mp4", session_id_path_depth=2)
    second = derive_session_and_camera("s3://bucket/sess-a/unit-01/camera.mp4", session_id_path_depth=2)

    assert first != second
    assert make_session_camera_id(*first) != make_session_camera_id(*second)


def test_derive_session_and_camera_raises_when_depth_exceeds_available_components() -> None:
    """A location with fewer directories than session_id_path_depth requires fails obviously."""
    with pytest.raises(UnresolvableMediaLocationError, match="has only"):
        derive_session_and_camera("s3://bucket/camera_front.mp4", session_id_path_depth=2)


def test_derive_session_and_camera_raises_on_no_path_components() -> None:
    """A location with nothing to derive a filename from is refused, not silently mishandled."""
    with pytest.raises(UnresolvableMediaLocationError, match="no path components"):
        derive_session_and_camera("s3://bucket/", session_id_path_depth=1)


def test_video_identity_and_mp4_metadata_schemas_are_disjoint() -> None:
    """The two schemas share no column names -- metadata is nested in the manifest."""
    identity_fields = set(contracts.VIDEO_MANIFEST_SCHEMA.names)
    metadata_fields = {field.name for field in attrs.fields(Mp4Metadata)}

    assert metadata_fields == set(contracts.VIDEO_METADATA_SCHEMA.names)
    assert identity_fields.isdisjoint(metadata_fields)


@pytest.mark.parametrize("composition_offsets", [None, 0, 6])
def test_nested_video_record_round_trips_in_lance(tmp_path: pathlib.Path, composition_offsets: int | None) -> None:
    """Lance preserves nested fields and nulls; a projected scan feeds the existing decoder."""
    import lance  # noqa: PLC0415

    from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex  # noqa: PLC0415
    from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION  # noqa: PLC0415

    frame_rows = [{**row, "composition_offset_ticks": composition_offsets} for row in _FRAME_ROWS]
    schema = pa.schema([contracts.VIDEO_MANIFEST_SCHEMA.field(name) for name in ("metadata", "frames")])
    table = pa.Table.from_pylist([{"metadata": _METADATA_ROW, "frames": frame_rows}], schema=schema)
    uri = str(tmp_path / "index.lance")
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    restored = lance.dataset(uri).to_table(columns=["metadata", "frames"])
    frame_table, metadata_row = contracts.read_video_record(restored)
    assert frame_table.to_pylist() == frame_rows
    assert metadata_row == _METADATA_ROW
    metadata = Mp4Metadata.from_row(metadata_row)
    index = Mp4FrameIndex.from_frame_index_table(
        frame_table, time_base_num=metadata.time_base_num, time_base_den=metadata.time_base_den
    )
    assert index.offset.tolist() == [0, 100, 150]
    assert (index.composition_offset_ticks is None) == (composition_offsets is None)


def test_read_video_record_handles_chunked_sliced_lists_without_copying() -> None:
    """Nonzero list offsets and chunk boundaries preserve the selected video and its buffers."""
    schema = pa.schema([contracts.VIDEO_MANIFEST_SCHEMA.field(name) for name in ("metadata", "frames")])
    table = pa.Table.from_pylist([{"metadata": _METADATA_ROW, "frames": _FRAME_ROWS}] * 3, schema=schema)
    chunked = pa.concat_tables([table.slice(0, 1), table.slice(1, 2)]).slice(1)
    frames, metadata = contracts.read_video_record(chunked, row_index=1)
    assert frames.to_pylist() == _FRAME_ROWS
    assert metadata == _METADATA_ROW
    original = table["frames"].chunk(0).values.field("offset")
    assert frames["offset"].chunk(0).buffers()[1].address == original.buffers()[1].address


def test_read_video_record_refuses_a_mismatched_frame_count() -> None:
    """A structurally valid record with mismatched metadata must fail before decoding."""
    schema = pa.schema([contracts.VIDEO_MANIFEST_SCHEMA.field(name) for name in ("metadata", "frames")])
    table = pa.Table.from_pylist([{"metadata": _METADATA_ROW, "frames": []}], schema=schema)
    with pytest.raises(ValueError, match="Frame count"):
        contracts.read_video_record(table)


def test_write_arrow_sidecar_is_schema_agnostic(tmp_path: pathlib.Path) -> None:
    """write_arrow_sidecar/read_arrow_sidecar round-trip any schema, not just this recipe's own ones."""
    schema = pa.schema([pa.field("session_uuid", pa.string(), nullable=False), pa.field("count", pa.int64())])
    rows: list[dict[str, object]] = [{"session_uuid": "abc", "count": 3}, {"session_uuid": "def", "count": None}]
    path = str(tmp_path / "sidecar.arrow")

    contracts.write_arrow_sidecar(rows, schema, path)
    table = contracts.read_arrow_sidecar(path)

    assert table.to_pylist() == rows


def test_write_bytes_sidecar_round_trips_bytes_verbatim(tmp_path: pathlib.Path) -> None:
    """No schema, no Arrow envelope -- exactly the bytes handed in, back out."""
    path = str(tmp_path / "moov" / "video-1.moov")
    data = b"\x00\x00\x00\x08moov"

    contracts.write_bytes_sidecar(data, path)

    assert pathlib.Path(path).exists()  # write_bytes_sidecar creates parent directories too
    assert contracts.read_bytes_sidecar(path) == data


def test_write_arrow_sidecar_passes_transport_params_through(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transport_params reaches smart_open.open unmodified, so a write reuses the caller's own credentials."""
    seen: dict[str, Any] = {}
    real_open = smart_open.open

    def _recording_open(path: str, mode: str, *, transport_params: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        seen["transport_params"] = transport_params
        return real_open(path, mode, transport_params=transport_params)

    monkeypatch.setattr(smart_open, "open", _recording_open)

    fake_client = object()
    path = str(tmp_path / "sidecar.arrow")

    contracts.write_arrow_sidecar(
        [{"count": 1}], pa.schema([pa.field("count", pa.int64())]), path, transport_params={"client": fake_client}
    )

    assert seen["transport_params"] == {"client": fake_client}


def test_write_bytes_sidecar_leaves_the_prior_local_file_untouched_if_the_write_is_interrupted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash partway through a local write must never leave the real sidecar path truncated.

    Real repro: a truncated moov cache read back later parses as a genuinely malformed box,
    indistinguishable from a corrupt source. Writing to a same-directory temp file and only
    replacing the real path once that write has fully succeeded means an interrupted write
    leaves an orphaned temp file instead -- the real path either keeps its previous complete
    content (this test) or never existed, never a truncated new write.
    """
    path = str(tmp_path / "moov" / "video-1.moov")
    original = b"\x00\x00\x00\x08moov"
    contracts.write_bytes_sidecar(original, path)

    real_smart_open = smart_open.open

    class _InterruptedHandle:
        def __init__(self, inner: Any) -> None:  # noqa: ANN401
            self._inner = inner

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc_info: object) -> bool:
            self._inner.close()
            return False

        def write(self, data: bytes) -> None:
            self._inner.write(data[: len(data) // 2])  # only half reaches disk before "crashing"
            msg = "simulated crash mid-write"
            raise OSError(msg)

    def _fake_open(target: str, mode: str, **kwargs: Any) -> Any:  # noqa: ANN401
        handle = real_smart_open(target, mode, **kwargs)
        return _InterruptedHandle(handle) if mode == "wb" else handle

    monkeypatch.setattr(smart_open, "open", _fake_open)

    with pytest.raises(OSError, match="simulated crash mid-write"):
        contracts.write_bytes_sidecar(b"\x00\x00\x00\x08NEWDATA", path)

    assert contracts.read_bytes_sidecar(path) == original


def test_write_arrow_sidecar_leaves_the_prior_local_file_untouched_if_the_write_is_interrupted(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same atomicity contract as write_bytes_sidecar, one layer up for the Arrow IPC writer."""
    schema = pa.schema([pa.field("count", pa.int64())])
    path = str(tmp_path / "sidecar.arrow")
    contracts.write_arrow_sidecar([{"count": 1}], schema, path)

    def _raising_write_table(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
        msg = "simulated crash mid-write"
        raise OSError(msg)

    monkeypatch.setattr(ipc.RecordBatchFileWriter, "write_table", _raising_write_table)

    with pytest.raises(OSError, match="simulated crash mid-write"):
        contracts.write_arrow_sidecar([{"count": 2}], schema, path)

    assert contracts.read_arrow_sidecar(path).to_pylist() == [{"count": 1}]


def test_write_bytes_sidecar_passes_transport_params_through(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same passthrough contract as write_arrow_sidecar, one layer down."""
    seen: dict[str, Any] = {}
    real_open = smart_open.open

    def _recording_open(path: str, mode: str, *, transport_params: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        seen["transport_params"] = transport_params
        return real_open(path, mode, transport_params=transport_params)

    monkeypatch.setattr(smart_open, "open", _recording_open)

    fake_client = object()
    path = str(tmp_path / "video.moov")

    contracts.write_bytes_sidecar(b"moov-bytes", path, transport_params={"client": fake_client})

    assert seen["transport_params"] == {"client": fake_client}


def test_write_then_read_back_over_an_s3_uri(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain pathlib.Path silently mistreats s3:// as a local path -- this is the regression test.

    Confirmed the hard way: a real pipeline run against s3:// output wrote
    every frame-index sidecar to a local directory literally named ``s3:``
    instead, because pathlib.Path("s3://...") neither raises nor recognizes
    the scheme. smart_open is what makes this actually reach S3 -- proven
    here by asserting ``smart_open.open`` is actually invoked with the
    ``s3://`` URI itself, not silently rewritten. The underlying read/write
    is redirected to a local file so this needs no real bucket or network
    access.
    """
    real_open = smart_open.open
    local_path = tmp_path / "roundtrip.arrow"
    seen_paths: list[str] = []

    def _redirect_to_local(
        path: str,
        mode: str,
        *,
        transport_params: dict[str, Any] | None = None,  # noqa: ARG001
    ) -> Any:  # noqa: ANN401
        seen_paths.append(path)
        return real_open(str(local_path), mode)

    monkeypatch.setattr(smart_open, "open", _redirect_to_local)

    path = "s3://example-bucket/index/_test/test_video_index_s3_roundtrip.arrow"
    contracts.write_arrow_sidecar(_FRAME_ROWS, contracts.FRAME_INDEX_SCHEMA, path)
    frame_table = contracts.read_arrow_sidecar(path)

    assert frame_table.to_pylist() == _FRAME_ROWS
    assert seen_paths == [path, path]


def test_sliced_chunked_records_preserve_timing_buffers() -> None:
    """Unwrap the chosen video, retaining integer precision and independent optional nulls."""
    rows = [
        {
            "dts_ticks": 10,
            "composition_offset_ticks": None,
            "is_keyframe": True,
            "offset": 100,
            "size": 20,
            "eof_ns": 9_007_199_254_740_993,
            "sof_ns": None,
            "moe_ns": 9_007_199_254_740_001,
        }
    ]
    frame_schema = FRAME_INDEX_SCHEMA
    values = pa.StructArray.from_arrays(
        [pa.array([r[f.name] for r in rows], type=f.type) for f in frame_schema],
        fields=list(frame_schema),
    )
    frames = pa.LargeListArray.from_arrays(pa.array([0, 1], type=pa.int64()), values)
    record = pa.table({"frames": frames, "metadata": [{"frame_count": 1}]})
    first = record.set_column(0, "frames", pa.array([[{**rows[0], "eof_ns": 1, "moe_ns": None}]], type=frames.type))
    chunked = pa.concat_tables([first, record])
    assert read_video_record(chunked, row_index=1)[0]["eof_ns"].to_pylist() == [rows[0]["eof_ns"]]
    table = chunked.slice(1, 1)
    timing = read_video_record(table)[0].select(["eof_ns", "sof_ns", "moe_ns"])
    assert timing.schema == pa.schema([FRAME_INDEX_SCHEMA.field(name) for name in ["eof_ns", "sof_ns", "moe_ns"]])
    assert timing["eof_ns"].chunk(0).buffers()[1].address == values.field("eof_ns").buffers()[1].address
    frame_table, metadata = read_video_record(table)
    for field in FRAME_INDEX_SCHEMA:
        output_buffers = frame_table[field.name].chunk(0).buffers()
        stored_buffers = values.field(field.name).buffers()
        assert [buffer.address if buffer is not None else None for buffer in output_buffers] == [
            buffer.address if buffer is not None else None for buffer in stored_buffers
        ]
    assert metadata["frame_count"] == 1
    assert frame_table.schema == frame_schema
    parsed = Mp4FrameIndex.from_frame_index_table(frame_table, time_base_num=1, time_base_den=1_000_000)
    assert parsed.eof_ns is not None
    assert parsed.dts_ns.tolist() == [10_000]
    assert parsed.eof_ns.tolist() == [rows[0]["eof_ns"]]
    assert parsed.sof_ns is None
    assert parsed.moe_ns is not None
    assert parsed.moe_ns.tolist() == [rows[0]["moe_ns"]]


def test_reader_refuses_inconsistent_frame_count() -> None:
    """Do not expose timing from a record whose metadata disagrees with its frame list."""
    table = pa.table(
        {
            "frames": pa.array([_FRAME_ROWS], type=contracts.VIDEO_MANIFEST_SCHEMA.field("frames").type),
            "metadata": [{"frame_count": 2}],
        }
    )
    with pytest.raises(ValueError, match="count"):
        read_video_record(table)


def test_reader_rejects_float_timing() -> None:
    """The reader rejects floating-point timing before a cast can conceal lost precision."""
    frame_schema = pa.schema(
        [
            pa.field(field.name, pa.float64() if field.name == "eof_ns" else field.type, nullable=field.nullable)
            for field in FRAME_INDEX_SCHEMA
        ]
    )
    table = pa.table(
        {
            "frames": pa.array(
                [
                    [
                        {
                            "dts_ticks": 10,
                            "composition_offset_ticks": None,
                            "is_keyframe": True,
                            "offset": 100,
                            "size": 20,
                            "eof_ns": 9_007_199_254_740_992.0,
                            "sof_ns": None,
                            "moe_ns": None,
                        }
                    ]
                ],
                type=pa.large_list(pa.struct(frame_schema)),
            ),
            "metadata": [{"frame_count": 1}],
        }
    )
    with pytest.raises(ValueError, match=r"eof_ns.*int64.*double"):
        read_video_record(table)


@pytest.mark.parametrize("column", FRAME_INDEX_SCHEMA.names)
@pytest.mark.parametrize("stored_type", [pa.float64(), pa.int32(), pa.uint64()])
def test_reader_rejects_incompatible_frame_column_types(column: str, stored_type: pa.DataType) -> None:
    """Every frame field enforces its declared type, including types Arrow could safely cast."""
    fields = [
        pa.field(field.name, stored_type if field.name == column else field.type, nullable=field.nullable)
        for field in FRAME_INDEX_SCHEMA
    ]
    row = {**_FRAME_ROWS[0], column: 100.5 if stored_type == pa.float64() else 100}
    values = pa.StructArray.from_arrays(
        [pa.array([row[field.name]], type=field.type) for field in fields], fields=fields
    )
    frames = pa.LargeListArray.from_arrays(pa.array([0, 1], type=pa.int64()), values)
    table = pa.table({"frames": frames, "metadata": [{"frame_count": 1}]})
    expected = FRAME_INDEX_SCHEMA.field(column).type

    with pytest.raises(ValueError, match=rf"{column}.*{expected}.*{stored_type}"):
        read_video_record(table)


def test_reader_rejects_integer_keyframe_flags() -> None:
    """A nonzero integer must not be accepted as a Boolean keyframe flag."""
    fields = [
        pa.field(field.name, pa.int64() if field.name == "is_keyframe" else field.type, nullable=field.nullable)
        for field in FRAME_INDEX_SCHEMA
    ]
    row = {**_FRAME_ROWS[0], "is_keyframe": 2}
    values = pa.StructArray.from_arrays(
        [pa.array([row[field.name]], type=field.type) for field in fields], fields=fields
    )
    frames = pa.LargeListArray.from_arrays(pa.array([0, 1], type=pa.int64()), values)
    table = pa.table({"frames": frames, "metadata": [{"frame_count": 1}]})

    with pytest.raises(ValueError, match=r"is_keyframe.*bool.*int64"):
        read_video_record(table)
