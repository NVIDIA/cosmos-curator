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
"""Tests for VideoIndexStage: comparison kernel + per-clip actor behavior."""

import json
from collections.abc import Mapping
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.utils.video import _HeaderIndexUnavailableError
from cosmos_curator.pipelines.video.split_comparison import video_index_stage
from cosmos_curator.pipelines.video.split_comparison.clip_discovery import CLIP_ROW_SCHEMA
from cosmos_curator.pipelines.video.split_comparison.config import SplitComparisonConfig, VideoIndexPolicy
from cosmos_curator.pipelines.video.split_comparison.driver import run_video_index_stage
from cosmos_curator.pipelines.video.split_comparison.result_model import ISSUE_SCHEMA
from cosmos_curator.pipelines.video.split_comparison.video_index_stage import _process_row


def _index(*, pts_ns: tuple[int, ...] = (0, 1_000_000_000, 2_000_000_000)) -> VideoIndex:
    n = len(pts_ns)
    is_keyframe = np.zeros(n, dtype=np.bool_)
    is_keyframe[0] = True
    if n > 1:
        is_keyframe[-1] = True
    pts_stream = np.arange(n, dtype=np.int64)
    return VideoIndex(
        offset=np.arange(n, dtype=np.int64) * 10,
        size=np.full(n, 10, dtype=np.int64),
        pts_ns=np.array(pts_ns, dtype=np.int64),
        pts_stream=pts_stream,
        is_keyframe=is_keyframe,
        is_discard=np.zeros(n, dtype=np.bool_),
        kf_pts_ns=np.array(pts_ns, dtype=np.int64)[is_keyframe],
        kf_pts_stream=pts_stream[is_keyframe],
        time_base=Fraction(1, 1),
    )


def _metadata(*, width: int = 16, height: int = 16) -> VideoMetadata:
    return VideoMetadata(
        codec_name="h264",
        codec_max_bframes=0,
        codec_profile="Main",
        container_format="mp4",
        height=height,
        width=width,
        avg_frame_rate=Fraction(30, 1),
        pix_fmt="yuv420p",
        bit_rate_bps=1_000,
    )


def _stub_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the one ``storage_utils`` call ``_process_row`` actually makes (``get_full_path``)."""
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.video_index_stage.storage_utils.get_full_path",
        lambda root, directory, name: Path(root) / directory / name,
    )


def _stub_make_index(
    monkeypatch: pytest.MonkeyPatch,
    values_by_path: Mapping[str, tuple[VideoIndex, VideoMetadata]],
) -> None:
    def fake(
        *,
        data: Any,  # noqa: ANN401 -- positional file-like or path
        **_kwargs: Any,  # noqa: ANN401 -- variadic from make_index_and_metadata signature
    ) -> tuple[VideoIndex, VideoMetadata]:
        source_path = str(data)
        if source_path.endswith("unreadable.mp4"):
            msg = "cannot decode mp4"
            raise ValueError(msg)
        if source_path.endswith("needs-full-demux.mp4"):
            msg = "stream header index is empty; retry with FULL_DEMUX"
            raise _HeaderIndexUnavailableError(msg)
        return values_by_path.get(source_path, (_index(), _metadata()))

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.video_index_stage.make_index_and_metadata",
        fake,
    )


def _setup_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the storage layer + ``_open_video_source`` for direct ``_process_row`` invocation."""
    _stub_storage(monkeypatch)
    # Skip the contextmanager file opening; pass paths through as the data source.
    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.video_index_stage._open_video_source",
        _passthrough_source,
    )


def _drive(
    row: dict[str, Any],
    *,
    policy: VideoIndexPolicy | None = None,
    path_a: str = "/output-a",
    path_b: str = "/output-b",
) -> list[dict[str, Any]]:
    """Test helper: invoke ``_process_row`` with empty params (storage stubs are module-level)."""
    return _process_row(  # type: ignore[arg-type, return-value]
        row,
        path_a=path_a,
        path_b=path_b,
        params_a={},
        params_b={},
        policy=policy or VideoIndexPolicy(),
    )


def _passthrough_source(
    path: Any,  # noqa: ANN401 -- pathlib or smart_open URL passthrough
    *,
    client_params: Any,  # noqa: ANN401, ARG001 -- contract match
) -> Any:  # noqa: ANN401
    @contextmanager
    def _ctx() -> Any:  # noqa: ANN401
        yield path

    return _ctx()


def _details(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["details"]) if row.get("details") else {}


def _row(
    *,
    clip_id: str = "clip-a",
    video_key: str = "video.mp4",
    artifact_kind: str = "clip",
    in_a: bool = True,
    in_b: bool = True,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "video_key": video_key,
        "artifact_kind": artifact_kind,
        "in_a": in_a,
        "in_b": in_b,
    }


@pytest.mark.parametrize(
    ("in_a", "in_b", "expected_codes", "expected_output"),
    [
        # Both sides present + loads succeed -> no issues.
        (True, True, [], None),
        # One-sided clip: present only on A -> one clip_mp4_missing tagged B.
        (True, False, ["clip_mp4_missing"], "b"),
        # Symmetric one-sided: present only on B -> one clip_mp4_missing tagged A.
        (False, True, ["clip_mp4_missing"], "a"),
    ],
)
def test_load_path_emits_codes_per_in_a_in_b(
    monkeypatch: pytest.MonkeyPatch,
    in_a: bool,  # noqa: FBT001 -- parametrize delivers this by name
    in_b: bool,  # noqa: FBT001 -- parametrize delivers this by name
    expected_codes: list[str],
    expected_output: str | None,
) -> None:
    """Happy-path matrix: presence flags on the clip row drive whether issues fire."""
    _setup_stubs(monkeypatch)
    _stub_make_index(monkeypatch, {})

    issues = _drive(_row(in_a=in_a, in_b=in_b))

    assert [issue["code"] for issue in issues] == expected_codes
    if expected_output is not None:
        assert issues[0]["output"] == expected_output


@pytest.mark.parametrize(
    ("clip_id_trigger", "expected_code"),
    [
        # ``_stub_make_index``'s suffix dispatch: ``unreadable.mp4`` -> ValueError -> clip_mp4_unreadable.
        ("unreadable", "clip_mp4_unreadable"),
        # ``needs-full-demux.mp4`` -> _HeaderIndexUnavailableError -> clip_mp4_header_index_unavailable.
        ("needs-full-demux", "clip_mp4_header_index_unavailable"),
    ],
)
def test_make_index_exception_maps_to_issue_code(
    monkeypatch: pytest.MonkeyPatch,
    clip_id_trigger: str,
    expected_code: str,
) -> None:
    """Each known exception out of make_index_and_metadata maps to one issue per output."""
    _setup_stubs(monkeypatch)
    _stub_make_index(monkeypatch, {})

    issues = _drive(_row(clip_id=clip_id_trigger))

    assert [issue["code"] for issue in issues] == [expected_code, expected_code]


def _oserror_chained_with_nosuchkey() -> OSError:
    """Build the OSError-wraps-NoSuchKey chain smart_open produces for a missing S3 key."""
    import botocore.exceptions  # noqa: PLC0415 -- only needed in this helper

    boto_exc = botocore.exceptions.ClientError(
        error_response={"Error": {"Code": "NoSuchKey", "Message": "missing"}},
        operation_name="GetObject",
    )
    wrapped = OSError("unable to access object")
    wrapped.__cause__ = boto_exc
    return wrapped


@pytest.mark.parametrize(
    "exception_factory",
    [
        # Local-fs missing object surfaces as a plain FileNotFoundError.
        pytest.param(lambda: FileNotFoundError("object not found"), id="filenotfound"),
        # smart_open wraps a boto3 NoSuchKey ClientError in an OSError with __cause__ chained.
        pytest.param(_oserror_chained_with_nosuchkey, id="s3-nosuchkey-via-oserror"),
    ],
)
def test_open_raises_missing_object_emits_clip_mp4_missing(
    monkeypatch: pytest.MonkeyPatch,
    exception_factory: Any,  # noqa: ANN401 -- arbitrary exception class / factory
) -> None:
    """Any missing-object error from ``_open_video_source`` flows through to clip_mp4_missing."""
    _stub_storage(monkeypatch)

    @contextmanager
    def _raising_source(_path: Any, *, client_params: Any) -> Any:  # noqa: ANN401, ARG001
        raise exception_factory()
        yield  # pragma: no cover -- unreachable, satisfies the contextmanager contract

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.video_index_stage._open_video_source",
        _raising_source,
    )
    _stub_make_index(monkeypatch, {})

    issues = _drive(_row())

    assert [issue["code"] for issue in issues] == ["clip_mp4_missing", "clip_mp4_missing"]


def test_value_drift_in_one_array_emits_one_index_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When only pts_ns drifts, one clip_mp4_index_mismatch row carries that field."""
    _setup_stubs(monkeypatch)
    _stub_make_index(
        monkeypatch,
        {
            "/output-b/clips/clip-a.mp4": (_index(pts_ns=(0, 1_000_000_000, 3_000_000_000)), _metadata()),
        },
    )

    issues = _drive(_row())

    # pts_ns triggers a value mismatch; kf_pts_ns is a derived array so it ALSO drifts.
    code_field_pairs = sorted(
        (issue["code"], issue["field"]) for issue in issues if issue["code"] == "clip_mp4_index_mismatch"
    )
    assert code_field_pairs == [
        ("clip_mp4_index_mismatch", "kf_pts_ns"),
        ("clip_mp4_index_mismatch", "pts_ns"),
    ]


def test_metadata_value_drift_emits_clip_mp4_metadata_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scalar metadata field difference produces a clip_mp4_metadata_mismatch row."""
    _setup_stubs(monkeypatch)
    _stub_make_index(
        monkeypatch,
        {
            "/output-a/clips/clip-a.mp4": (_index(), _metadata(width=16)),
            "/output-b/clips/clip-a.mp4": (_index(), _metadata(width=32)),
        },
    )

    issues = _drive(_row())
    metadata_issues = [issue for issue in issues if issue["code"] == "clip_mp4_metadata_mismatch"]

    assert len(metadata_issues) == 1
    assert metadata_issues[0]["field"] == "width"
    assert _details(metadata_issues[0]) == {"artifact_kind": "clip", "a": "16", "b": "32"}


def test_one_sided_load_failure_does_not_run_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    """If one output can't be loaded, the comparison is skipped (no mismatch issues emitted)."""
    _stub_storage(monkeypatch)

    @contextmanager
    def _selective_source(path: Any, *, client_params: Any) -> Any:  # noqa: ANN401, ARG001
        if str(path).startswith("/output-b"):
            msg = "output B object not found"
            raise FileNotFoundError(msg)
        yield path

    monkeypatch.setattr(
        "cosmos_curator.pipelines.video.split_comparison.video_index_stage._open_video_source",
        _selective_source,
    )
    _stub_make_index(monkeypatch, {})

    issues = _drive(_row())
    codes = [issue["code"] for issue in issues]

    assert "clip_mp4_missing" in codes
    assert not any(code in {"clip_mp4_index_mismatch", "clip_mp4_metadata_mismatch"} for code in codes)


def test_video_index_scalar_fields_lists_non_array_attributes() -> None:
    """``VideoIndex.scalar_fields()`` should cover non-array init attributes (today: time_base)."""
    assert set(VideoIndex.scalar_fields()) == {"time_base"}
    # Belt-and-suspenders: no packet array sneaks in.
    assert "pts_ns" not in VideoIndex.scalar_fields()


def test_video_index_packet_array_fields_lists_per_packet_arrays() -> None:
    """``VideoIndex.packet_array_fields()`` should cover every per-packet array attribute.

    Regression guard: if VideoIndex grows or shrinks a packet-level numpy array
    field, ``packet_array_fields()`` picks it up automatically; this test fires
    loudly if someone changes the attrs schema without updating the audit
    comparator's expected coverage.
    """
    expected = {
        "offset",
        "size",
        "pts_ns",
        "pts_stream",
        "is_keyframe",
        "is_discard",
        "kf_pts_ns",
        "kf_pts_stream",
    }
    assert set(VideoIndex.packet_array_fields()) == expected
    # Scalar ``time_base`` is handled separately by the audit comparator.
    assert "time_base" not in VideoIndex.packet_array_fields()


@pytest.mark.parametrize(
    ("a", "b", "policy", "equal"),
    [
        # Strict by default (int_tolerance=0): any difference flags the field.
        (np.array([1, 2, 3], dtype=np.int64), np.array([1, 2, 4], dtype=np.int64), VideoIndexPolicy(), False),
        # Same diff within int_tolerance=1 -> equal.
        (
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2, 4], dtype=np.int64),
            VideoIndexPolicy(int_tolerance=1),
            True,
        ),
        # Shape mismatch never matches, even with generous tolerance.
        (
            np.array([1, 2], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.int64),
            VideoIndexPolicy(int_tolerance=100),
            False,
        ),
    ],
)
def test_arrays_equal_under_policy(
    a: np.ndarray,
    b: np.ndarray,
    policy: VideoIndexPolicy,
    equal: bool,  # noqa: FBT001 -- parametrize delivers this by name
) -> None:
    """Strict-by-default + tolerance + shape-mismatch in one table."""
    assert video_index_stage._arrays_equal_under_policy(a, b, policy=policy) is equal


def test_run_video_index_stage_returns_empty_table_when_no_clips() -> None:
    """The driver short-circuits to empty_issues() for a zero-row clip table."""
    empty = pa.Table.from_pylist([], schema=CLIP_ROW_SCHEMA)

    out = run_video_index_stage(empty, config=SplitComparisonConfig(output_a="/a", output_b="/b"))

    assert out.num_rows == 0
    assert out.schema == ISSUE_SCHEMA
