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
"""Stage 2: per-clip MP4 ``VideoIndex`` + ``VideoMetadata`` comparison.

The :class:`VideoIndexStage` actor loads both outputs' MP4, builds each
side's ``VideoIndex`` / ``VideoMetadata``, and emits one issue per divergent
field (flattened at the source). Ray Data wiring (pool sizing, runtime env)
lives in :mod:`...driver`.
"""

import pathlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, cast

if TYPE_CHECKING:
    import io

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import smart_open  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data.video import VideoIndex, VideoMetadata
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import _HeaderIndexUnavailableError, make_index_and_metadata
from cosmos_curator.core.utils.storage import storage_utils
from cosmos_curator.core.utils.storage.storage_client import StoragePrefix
from cosmos_curator.pipelines.video.split_comparison.config import VideoIndexPolicy
from cosmos_curator.pipelines.video.split_comparison.result_model import (
    ISSUE_SCHEMA,
    Issue,
    make_issue,
)

_FEATURE = "video_indexes"
_HEAD_LIMIT = 10
_INT64_BYTES = 8


class VideoIndexStage:
    """Compares clip-MP4 ``VideoIndex`` + ``VideoMetadata``."""

    # Pixi env this actor runs in.
    conda_env_name: ClassVar[str] = "default"

    def __init__(
        self,
        output_a: str,
        output_b: str,
        profile_name: str,
        policy: VideoIndexPolicy,
    ) -> None:
        """Resolve per-output ``smart_open`` params once for the actor's lifetime.

        No per-clip ``StorageClient`` cache -- ``_load_output`` detects missing
        objects from the exception itself.
        """
        self._output_a = output_a
        self._output_b = output_b
        self._policy = policy
        self._params_a: Mapping[str, Any] = storage_utils.get_smart_open_params(output_a, profile_name=profile_name)
        self._params_b: Mapping[str, Any] = storage_utils.get_smart_open_params(output_b, profile_name=profile_name)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Dispatch each row in the batch through :func:`_process_row`; concatenate the results."""
        rows: list[Issue] = []
        for row in batch.to_pylist():
            rows.extend(
                _process_row(
                    row,
                    path_a=self._output_a,
                    path_b=self._output_b,
                    params_a=self._params_a,
                    params_b=self._params_b,
                    policy=self._policy,
                ),
            )
        return pa.Table.from_pylist(rows, schema=ISSUE_SCHEMA)


class _Output:
    """Per-output load result: index/metadata or issues describing the failure."""

    __slots__ = ("index", "issues", "metadata")

    def __init__(
        self,
        index: VideoIndex | None,
        metadata: VideoMetadata | None,
        issues: list[Issue],
    ) -> None:
        self.index = index
        self.metadata = metadata
        self.issues = issues


def _absent_output() -> _Output:
    return _Output(index=None, metadata=None, issues=[])


def _load_output(  # noqa: PLR0913 -- per-output load takes its identity + storage params as inputs
    *,
    path: StoragePrefix | pathlib.Path,
    params: Mapping[str, Any],
    video_key: str,
    clip_id: str,
    artifact_kind: str,
    output: str,
) -> _Output:
    """Load one output's MP4 and build its ``VideoIndex`` / ``VideoMetadata``.

    Pure I/O -- no state. Missing objects (``FileNotFoundError`` /
    ``NoSuchKey`` wrapped in ``OSError``) are detected from the exception
    via :func:`storage_utils.is_missing_object_error`; no existence probe.
    """
    try:
        with _open_video_source(path, client_params=params) as source:
            index, metadata = make_index_and_metadata(
                data=source,
                index_method=VideoIndexCreationMethod.FROM_HEADER,
                allow_header_fallback=False,
            )
        return _Output(index=index, metadata=metadata, issues=[])
    except _HeaderIndexUnavailableError as exc:
        return _Output(
            index=None,
            metadata=None,
            issues=[
                make_issue(
                    code="clip_mp4_header_index_unavailable",
                    message="Clip MP4 header index unavailable",
                    feature=_FEATURE,
                    video=video_key,
                    clip=clip_id,
                    output=output,
                    details={
                        "artifact_kind": artifact_kind,
                        "path": str(path),
                        "error": str(exc),
                    },
                ),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- map any read failure to an issue; never propagate
        if storage_utils.is_missing_object_error(exc):
            return _Output(
                index=None,
                metadata=None,
                issues=[_clip_mp4_missing_issue(video_key, clip_id, artifact_kind, output, path)],
            )
        return _Output(
            index=None,
            metadata=None,
            issues=[
                make_issue(
                    code="clip_mp4_unreadable",
                    message="Clip MP4 could not be indexed",
                    feature=_FEATURE,
                    video=video_key,
                    clip=clip_id,
                    output=output,
                    details={
                        "artifact_kind": artifact_kind,
                        "path": str(path),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                ),
            ],
        )


def _process_row(  # noqa: PLR0913 -- routes a row through the actor-bound state
    row: Mapping[str, Any],
    *,
    path_a: str,
    path_b: str,
    params_a: Mapping[str, Any],
    params_b: Mapping[str, Any],
    policy: VideoIndexPolicy,
) -> list[Issue]:
    """Load both outputs of one clip MP4 and run the index / metadata comparisons.

    Free function (not a method) so the per-row logic is testable without
    constructing a stage actor.
    """
    clip_id = row["clip_id"]
    video_key = row["video_key"]
    artifact_kind = row["artifact_kind"]
    in_a, in_b = row["in_a"], row["in_b"]

    output_a = (
        _load_output(
            path=_clip_mp4_path(path_a, clip_id, artifact_kind),
            params=params_a,
            video_key=video_key,
            clip_id=clip_id,
            artifact_kind=artifact_kind,
            output="a",
        )
        if in_a
        else _absent_output()
    )
    output_b = (
        _load_output(
            path=_clip_mp4_path(path_b, clip_id, artifact_kind),
            params=params_b,
            video_key=video_key,
            clip_id=clip_id,
            artifact_kind=artifact_kind,
            output="b",
        )
        if in_b
        else _absent_output()
    )

    issues: list[Issue] = []
    issues.extend(output_a.issues)
    issues.extend(output_b.issues)
    # One-sided clip artifacts are reported and we don't attempt a compare.
    if in_a != in_b:
        issues.append(
            make_issue(
                code="clip_mp4_missing",
                message="Clip MP4 present on only one output",
                feature=_FEATURE,
                video=video_key,
                clip=clip_id,
                output="b" if in_a else "a",
                details={"artifact_kind": artifact_kind},
            ),
        )
        return issues
    if output_a.index is None or output_b.index is None or output_a.metadata is None or output_b.metadata is None:
        # Load failure on at least one output already emitted an issue.
        return issues
    issues.extend(
        _index_field_issues(
            video_key=video_key,
            clip_id=clip_id,
            artifact_kind=artifact_kind,
            index_a=output_a.index,
            index_b=output_b.index,
            policy=policy,
        ),
    )
    issues.extend(
        _video_metadata_field_issues(
            video_key=video_key,
            clip_id=clip_id,
            artifact_kind=artifact_kind,
            metadata_a=output_a.metadata,
            metadata_b=output_b.metadata,
            policy=policy,
        ),
    )
    return issues


def _clip_mp4_path(output_root: str, clip_id: str, artifact_kind: str) -> StoragePrefix | pathlib.Path:
    directory = "filtered_clips" if artifact_kind == "filtered_clip" else "clips"
    return storage_utils.get_full_path(output_root, directory, f"{clip_id}.mp4")


def _clip_mp4_missing_issue(
    video_key: str,
    clip_id: str,
    artifact_kind: str,
    output: str,
    path: StoragePrefix | pathlib.Path,
) -> Issue:
    return make_issue(
        code="clip_mp4_missing",
        message=f"Clip MP4 missing on output {output.upper()}",
        feature=_FEATURE,
        video=video_key,
        clip=clip_id,
        output=output,
        details={"artifact_kind": artifact_kind, "path": str(path)},
    )


@contextmanager
def _open_video_source(
    path: StoragePrefix | pathlib.Path,
    *,
    client_params: Mapping[str, Any],
) -> Iterator[DataSource]:
    if isinstance(path, pathlib.Path):
        yield path
        return
    with smart_open.open(str(path), "rb", **client_params) as stream:
        yield cast("io.BufferedIOBase", stream)


def _index_field_issues(  # noqa: PLR0913 -- comparator carries the cross-clip context per call
    *,
    video_key: str,
    clip_id: str,
    artifact_kind: str,
    index_a: VideoIndex,
    index_b: VideoIndex,
    policy: VideoIndexPolicy,
) -> list[Issue]:
    issues: list[Issue] = []
    for field in VideoIndex.packet_array_fields():
        raw_a = getattr(index_a, field)
        raw_b = getattr(index_b, field)
        # Guard against future VideoIndex evolution: if a field returned by
        # ``VideoIndex.packet_array_fields()`` turns out to not be a numpy
        # array, skip it instead of crashing. Future-loud alternative: raise.
        if not (isinstance(raw_a, np.ndarray) and isinstance(raw_b, np.ndarray)):
            continue
        array_a = cast("npt.NDArray[Any]", raw_a)
        array_b = cast("npt.NDArray[Any]", raw_b)
        if array_a.dtype != array_b.dtype:
            issues.append(
                make_issue(
                    code="clip_mp4_index_dtype_mismatch",
                    message=f"Clip MP4 VideoIndex array {field!r} dtypes differ",
                    feature=_FEATURE,
                    video=video_key,
                    clip=clip_id,
                    field=field,
                    details={
                        "artifact_kind": artifact_kind,
                        "a": _array_summary(array_a),
                        "b": _array_summary(array_b),
                    },
                ),
            )
            continue
        if _arrays_equal_under_policy(array_a, array_b, policy=policy):
            continue
        issues.append(
            make_issue(
                code="clip_mp4_index_mismatch",
                message=f"Clip MP4 VideoIndex array {field!r} differs",
                feature=_FEATURE,
                video=video_key,
                clip=clip_id,
                field=field,
                details={
                    "artifact_kind": artifact_kind,
                    "a": _array_summary(array_a),
                    "b": _array_summary(array_b),
                },
            ),
        )
    for field in VideoIndex.scalar_fields():
        value_a = getattr(index_a, field)
        value_b = getattr(index_b, field)
        if value_a == value_b:
            continue
        issues.append(
            make_issue(
                code="clip_mp4_index_mismatch",
                message=f"Clip MP4 VideoIndex {field} differs",
                feature=_FEATURE,
                video=video_key,
                clip=clip_id,
                field=field,
                details={
                    "artifact_kind": artifact_kind,
                    "a": str(value_a),
                    "b": str(value_b),
                },
            ),
        )
    return issues


def _video_metadata_field_issues(  # noqa: PLR0913 -- comparator carries the cross-clip context per call
    *,
    video_key: str,
    clip_id: str,
    artifact_kind: str,
    metadata_a: VideoMetadata,
    metadata_b: VideoMetadata,
    policy: VideoIndexPolicy,
) -> list[Issue]:
    string_a = metadata_a.to_string_dict()
    string_b = metadata_b.to_string_dict()
    issues: list[Issue] = []
    for field in policy.compare_metadata_fields:
        value_a = string_a.get(field)
        value_b = string_b.get(field)
        if value_a == value_b:
            continue
        issues.append(
            make_issue(
                code="clip_mp4_metadata_mismatch",
                message=f"Clip MP4 VideoMetadata field {field!r} differs",
                feature=_FEATURE,
                video=video_key,
                clip=clip_id,
                field=field,
                details={"artifact_kind": artifact_kind, "a": value_a, "b": value_b},
            ),
        )
    return issues


def _arrays_equal_under_policy(
    array_a: npt.NDArray[Any],
    array_b: npt.NDArray[Any],
    *,
    policy: VideoIndexPolicy,
) -> bool:
    if array_a.shape != array_b.shape:
        return False
    if np.issubdtype(array_a.dtype, np.bool_):
        return bool(np.array_equal(array_a, array_b))
    if np.issubdtype(array_a.dtype, np.integer):
        return _integer_arrays_equal(array_a, array_b, tolerance=policy.int_tolerance)
    if np.issubdtype(array_a.dtype, np.floating):
        return bool(
            np.allclose(
                array_a,
                array_b,
                rtol=policy.float_rtol,
                atol=policy.float_atol,
                equal_nan=True,
            ),
        )
    return bool(np.array_equal(array_a, array_b))


def _integer_arrays_equal(
    array_a: npt.NDArray[Any],
    array_b: npt.NDArray[Any],
    *,
    tolerance: int,
) -> bool:
    if tolerance == 0:
        return bool(np.array_equal(array_a, array_b))
    if array_a.dtype.itemsize >= _INT64_BYTES:
        # 64-bit ints can't be safely widened to int64 for differencing; fall back to
        # per-element Python ints (slower but correct at any magnitude).
        return all(
            abs(int(va) - int(vb)) <= tolerance
            for va, vb in zip(array_a.ravel().tolist(), array_b.ravel().tolist(), strict=True)
        )
    diff = np.abs(array_a.astype(np.int64) - array_b.astype(np.int64))
    return bool(np.all(diff <= tolerance))


def _array_summary(values: npt.NDArray[Any]) -> dict[str, Any]:
    head = values.ravel()[:_HEAD_LIMIT].tolist()
    return {
        "length": int(values.shape[0]) if values.ndim else 1,
        "dtype": str(values.dtype),
        "head": head,
    }
