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

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata
from cosmos_curator.next.recipes.mp4_index import contracts
from cosmos_curator.next.recipes.mp4_index.contracts import (
    UnresolvableMediaLocationError,
    derive_session_and_camera,
    make_session_camera_id,
)

_FRAME_ROWS: list[dict[str, Any]] = [
    {"dts_ticks": 12, "composition_offset_ticks": None, "is_keyframe": True, "offset": 0, "size": 100},
    {"dts_ticks": 422, "composition_offset_ticks": 6, "is_keyframe": False, "offset": 100, "size": 50},
    {"dts_ticks": 832, "composition_offset_ticks": None, "is_keyframe": False, "offset": 150, "size": 42},
]


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
    """The two schemas share no column names -- VIDEO_METADATA_SCHEMA is a sidecar, not joined by video_id."""
    identity_fields = set(contracts.VIDEO_IDENTITY_SCHEMA.names)
    metadata_fields = {field.name for field in attrs.fields(Mp4Metadata)}

    assert metadata_fields == set(contracts.VIDEO_METADATA_SCHEMA.names)
    assert identity_fields.isdisjoint(metadata_fields)


def test_write_then_read_back_matches_what_was_built(tmp_path: pathlib.Path) -> None:
    """Every field survives a real write/read round trip through a local file, nulls included."""
    path = str(tmp_path / "frames" / "video-1.arrow")

    contracts.write_frame_index(_FRAME_ROWS, path)
    table = contracts.read_frame_index(path)

    assert table.to_pylist() == _FRAME_ROWS


def test_write_creates_parent_directories(tmp_path: pathlib.Path) -> None:
    """The caller doesn't need to mkdir first."""
    path = str(tmp_path / "a" / "b" / "c" / "video-1.arrow")

    contracts.write_frame_index(_FRAME_ROWS, path)

    assert pathlib.Path(path).exists()


def test_write_arrow_sidecar_is_schema_agnostic(tmp_path: pathlib.Path) -> None:
    """write_frame_index/read_frame_index are thin FRAME_INDEX_SCHEMA wrappers -- any schema round-trips."""
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
    contracts.write_frame_index(_FRAME_ROWS, path)
    table = contracts.read_frame_index(path)

    assert table.to_pylist() == _FRAME_ROWS
    assert seen_paths == [path, path]
