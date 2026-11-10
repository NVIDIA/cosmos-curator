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

"""Turn a URI into an open sensor and run the metrics over it.

The IO wrapper around the per-stream engine
(:mod:`cosmos_curator.core.sensors.data_integrity.engine`): the engine takes an
already-open sensor and performs no IO, so resolving a local path or cloud URI into
a stream, and constructing the sensor on it, happens here. That split is what keeps
the sensor library backend-agnostic -- it never accepts a URI and imports no cloud
client (see :mod:`cosmos_curator.core.sensors.utils.io`) -- and it is why these two
functions sit in the recipe rather than beside the metrics they call.

Both entry points build on :func:`run_checks` through the concurrent session runner
(:mod:`.session_runner`): the ``di`` tool (:mod:`.session_cli`) and the Ray Data
pipeline (:mod:`.pipeline`).

Which clock a stream is measured on follows ``checks.camera_alignment_fields``, an
ordered preference shared with ``multimodal-split``
(:mod:`cosmos_curator.next.recipes.mp4_index.timestamp_field_selection`). Given an ``mp4-index``
manifest, a stream the manifest covers takes its timeline from the index instead of
being opened (:mod:`cosmos_curator.next.recipes.mp4_index.index_lookup`). Every other
stream can use explicitly selected container PTS, whether the manifest has no row for
it or the row it has cannot be used. Recorded EOF/SOF/MOE clocks require an index.
"""

import pathlib
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import BinaryIO, cast, get_args

from loguru import logger

from cosmos_curator.core.sensors.data.mp4 import Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.data_integrity.engine import run_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, ResolvedConfig, VideoInfo
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.sensors.streaming_camera_sensor import (
    IndexRecordUnusableError,
    StreamingCameraSensor,
)
from cosmos_curator.core.sensors.types.types import DataSource, VideoFrameAlignmentField
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import open_storage_source
from cosmos_curator.next.recipes.data_integrity import storage_io
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup
from cosmos_curator.next.recipes.mp4_index.timestamp_field_selection import (
    CameraAlignmentField,
    first_complete_alignment_field,
)

#: The camera clock preference when a caller names none, matching the config default.
DEFAULT_ALIGNMENT_FIELDS: tuple[CameraAlignmentField, ...] = ("eof_ns",)
_CAMERA_ALIGNMENT_FIELDS: frozenset[str] = frozenset(get_args(CameraAlignmentField))


def _as_data_source(stream: BinaryIO) -> DataSource:
    """Cast a ``BinaryIO`` from :func:`open_source` to a ``DataSource``.

    ``smart_open``'s S3 / Azure readers and a plain ``Path.open("rb")`` handle are all
    seekable ``io.BufferedIOBase`` subclasses, so they satisfy the ``DataSource`` union
    at runtime even though static typing only sees ``BinaryIO`` (mirrors
    ``check_video_index``).
    """
    return cast("DataSource", stream)


@contextmanager
def open_source(
    source: str,
    *,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> Generator[BinaryIO]:
    """Yield a fresh readable stream for ``source``, cloud URI or local path alike.

    ``stream_wrapper``, when given, wraps that stream before it is yielded (a
    byte-counting reader for download progress, a cancellable one for Ctrl-C).

    Local paths are opened here rather than handed to the sensor as a :class:`Path` so
    that they get the same wrappers: the sensor library opens a ``Path`` into a Python
    handle anyway (``open_file``), so libav sees the same callbacks either way and this
    costs nothing measurable, while a path on a shared filesystem can be every bit as
    slow to read as a cloud object. The trade is that the sensor now borrows one
    stateful handle instead of being able to re-open the file, which matches what every
    cloud source has always given it.
    """
    if is_remote_path(source):
        with open_storage_source(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        ) as remote_stream:
            yield stream_wrapper(remote_stream) if stream_wrapper is not None else remote_stream
    else:
        with pathlib.Path(source).open("rb") as local_stream:
            yield stream_wrapper(local_stream) if stream_wrapper is not None else local_stream


def error_chain(exc: BaseException) -> Generator[BaseException]:
    """Yield *exc* and the errors it was raised from, including transport wrappers.

    ``smart_open`` reports every failed S3 call as a plain ``IOError``, hiding the
    code and status on the ``botocore`` error it attaches as ``backend_error`` (and
    chains as ``__cause__``). So a 404, a 403 and a 500 all arrive as the same
    exception type, and telling them apart means looking past the wrapper.

    The walk under :func:`error_code` and :func:`http_status`, which are what the rest
    of the recipe reads an error's shape through. All three live here rather than
    beside :func:`~.session_runner.is_infrastructure_error`, their other caller,
    because this is the module that opens sources and so the module the wrapper comes
    from; ``session_runner`` already imports from here, and the reverse would be a
    cycle.
    """
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.append(current.__cause__)
        backend_error = getattr(current, "backend_error", None)
        pending.append(backend_error if isinstance(backend_error, BaseException) else None)


def error_code(exc: BaseException) -> str | None:
    """Read the backend's error code off an exception shaped like ``botocore``'s ``ClientError``.

    Searched through the exception chain rather than read off ``exc`` alone, because
    the reads on this path arrive through ``smart_open``, which replaces every failed
    S3 call with a plain ``IOError`` carrying none of this (see :func:`error_chain`).
    Read only from the top, an ``AccessDenied`` on an object fetch looks like nothing
    at all, and the stream it belongs to gets a manufactured data-quality error
    instead of being counted unreachable.

    Public, and sited here rather than beside its other caller, for the same reason as
    :func:`error_chain`. Absence and unreachability are different questions, but they
    are answered off the same two attributes, and reading those twice is how the
    answers come to disagree -- one saying a denied object is simply not there.
    """
    for error in error_chain(exc):
        response = getattr(error, "response", None)
        if not isinstance(response, Mapping):
            continue
        detail = response.get("Error")
        if not isinstance(detail, Mapping):
            continue
        code = detail.get("Code")
        if isinstance(code, str):
            return code
    return None


def http_status(exc: BaseException) -> int | None:
    """Read the HTTP status of a failed cloud call, from either SDK's shape for it.

    Chain-searched, and public, for the same reasons as :func:`error_code`.
    """
    for error in error_chain(exc):
        response = getattr(error, "response", None)
        if isinstance(response, Mapping):
            metadata = response.get("ResponseMetadata")
            if isinstance(metadata, Mapping):
                status = metadata.get("HTTPStatusCode")
                if isinstance(status, int):
                    return status
        # azure.core.exceptions.HttpResponseError carries it directly.
        status = getattr(error, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def observed_size_bytes(
    source: str,
    *,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> int | None:
    """Return how many bytes sit at *source* now, or ``None`` if the size cannot be read.

    Costs one metadata lookup: a ``HEAD`` for a cloud object, an ``os.stat`` for a file.
    Both are best effort, and the caller opens the video when the answer is ``None``.
    """
    if is_remote_path(source):
        stat = storage_io.object_stat(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
        )
        return None if stat is None else stat.size_bytes
    try:
        return pathlib.Path(source).stat().st_size
    except OSError:
        return None


def _indexed_sensor(
    source: str,
    record: tuple[Mp4FrameIndex, Mp4Metadata],
    alignment_fields: Sequence[str],
) -> StreamingCameraSensor | None:
    """Build a sensor over *record* on the first complete field, or ``None`` to open the video.

    A row with none of *alignment_fields* is a finding about this stream, so it raises
    rather than falling back to the media: the row is current, and opening the video
    could not supply a sensor clock the index lacks. A row that cannot describe this
    stream at all is expected rather than exceptional -- an index whose frame count
    disagrees with its own metadata -- so it is logged at debug and answered by reading
    the media instead, exactly as if the manifest had no row.

    The chosen timeline is not required to ascend, unlike ``multimodal-split``'s use of
    the same rule: a recorder clock that steps backwards is what the ordering metric is
    here to report.

    Raises:
        ValueError: If the row populates none of *alignment_fields* for every frame.

    """
    index, metadata = record
    alignment_field = first_complete_alignment_field(index, alignment_fields)
    if alignment_field is None:
        msg = (
            f"mp4-index record for {source} has no complete camera timestamp column from "
            f"{list(alignment_fields)}; set checks.camera_alignment_fields to include one it has"
        )
        raise ValueError(msg)
    try:
        return StreamingCameraSensor(metadata=metadata, frame_index=index, alignment_field=alignment_field)
    except IndexRecordUnusableError:
        logger.opt(exception=True).debug("mp4-index row for {} cannot answer; opening the video", source)
        return None


def _validate_alignment_fields(alignment_fields: Sequence[str]) -> tuple[str, ...]:
    """Return *alignment_fields* as a tuple, rejecting an empty list or an unknown field."""
    fields = tuple(alignment_fields)
    if not fields:
        msg = "alignment_fields must name at least one camera timestamp field"
        raise ValueError(msg)
    unknown = [field for field in fields if field not in _CAMERA_ALIGNMENT_FIELDS]
    if unknown:
        msg = f"unknown camera alignment field(s) {unknown}; expected some of {sorted(_CAMERA_ALIGNMENT_FIELDS)}"
        raise ValueError(msg)
    return fields


def _measure_media(  # noqa: PLR0913 -- the source, its clock preference, and the open/metric settings
    source: str,
    *,
    alignment_fields: tuple[str, ...],
    expected_hz: float | None,
    thresholds: Thresholds,
    stream_idx: int,
    batch_size: int,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None,
    stats: dict[str, float] | None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Measure an unindexed stream using an explicitly selected container PTS clock.

    Recorded EOF/SOF/MOE clocks require an index. Legacy sidecars are never probed.
    """
    for field in alignment_fields:
        if field not in ("pts_ns", "edited_pts_ns"):
            continue
        with open_source(
            source,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            stream_wrapper=stream_wrapper,
        ) as src:
            t0 = time.perf_counter()
            sensor: CameraSensor | StreamingCameraSensor
            if field == VideoFrameAlignmentField.PTS_NS.value:
                sensor = StreamingCameraSensor(
                    _as_data_source(src), stream_idx=stream_idx, alignment_field=VideoFrameAlignmentField.PTS_NS
                )
            else:
                sensor = CameraSensor(_as_data_source(src), stream_idx=stream_idx)
            if stats is not None:
                stats["sensor_init_ms"] = (time.perf_counter() - t0) * 1000
            return run_metrics(
                sensor, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size, stats=stats
            )
    msg = (
        f"no requested camera timestamps for {source} from "
        f"{list(alignment_fields)}; add edited_pts_ns to checks.camera_alignment_fields to measure container PTS, "
        f"EOF/SOF/MOE require input.mp4_index_uri"
    )
    raise ValueError(msg)


def run_checks(  # noqa: PLR0913
    source: str,
    *,
    expected_hz: float | None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    stream_idx: int = 0,
    batch_size: int = 0,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    stats: dict[str, float] | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
    index_lookup: Mp4IndexLookup | None = None,
    alignment_fields: Sequence[str] = DEFAULT_ALIGNMENT_FIELDS,
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Open ``source``, build a sensor over it, and run every metric on it.

    The I/O wrapper around :func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics`:
    it resolves the local path or cloud stream, constructs the sensor, and delegates
    the compute. Any open / decode failure propagates as an exception for the caller
    to classify (``di`` records a per-stream ERROR; the pipeline additionally sorts
    unreachable from unreadable).

    Also where the stream's clock is chosen, from *alignment_fields* in order, so both
    entry points inherit one rule -- the one ``multimodal-split`` applies to cameras. A
    stream with a current ``mp4-index`` row is measured from that row alone, and never
    opens the video; any other stream can use explicitly selected container PTS (see
    :func:`_measure_media`). A stream for which no listed field is available raises, so
    it is recorded as an error rather than silently measured on another clock.

    Args:
        source: local path, ``s3://`` URI, or ``az://`` URI.
        expected_hz: expected sample rate for the rate-dependent metrics.
        thresholds: pass/fail policy (see :class:`Thresholds`).
        stream_idx: which video stream to open (default 0). Only 0 can use the index,
            since a manifest row is keyed by file rather than by track.
        batch_size: timestamps per metric update; 0 feeds the whole array at once.
        s3_profile_name: optional AWS profile forwarded to ``open_storage_source``.
        azure_profile_name: Azure profile forwarded to ``open_storage_source``.
        endpoint_url: optional S3 endpoint override for S3-compatible stores.
        stats: optional out-parameter; ``sensor_init_ms`` is recorded here in
            addition to the ``stream_ms`` / ``evaluate_ms`` from ``run_metrics``.
        stream_wrapper: optional wrapper applied to the opened video (e.g. a byte-counting
            reader for progress), for local paths as well as cloud URIs.
        index_lookup: optional opened ``mp4-index`` manifest. When it holds a usable row
            for ``source``, the timeline comes from that row instead of from the video;
            otherwise explicitly selected container PTS can be read from the video.
            Costs one metadata lookup per stream to confirm the row still describes
            the bytes there. Consulted only for
            ``stream_idx`` 0, since a manifest row is keyed by file, not by track.
        alignment_fields: camera timestamp fields in preference order (see
            :data:`~cosmos_curator.next.recipes.mp4_index.timestamp_field_selection.CameraAlignmentField`).

    Returns:
        The tuple returned by ``run_metrics``.

    Raises:
        ValueError: If no field in ``alignment_fields`` is available for the stream,
            or if ``alignment_fields`` is empty or names an unknown field.

    """
    fields = _validate_alignment_fields(alignment_fields)
    # A manifest row is keyed by file, not by track, so it only describes the stream
    # the indexer chose. Any other stream_idx is opened instead.
    if index_lookup is not None and stream_idx == 0:
        t0 = time.perf_counter()
        record = index_lookup.index_for(
            source,
            media_size_bytes=observed_size_bytes(
                source,
                s3_profile_name=s3_profile_name,
                azure_profile_name=azure_profile_name,
                endpoint_url=endpoint_url,
            ),
        )
        indexed = None if record is None else _indexed_sensor(source, record, fields)
        if indexed is not None:
            if stats is not None:
                stats["sensor_init_ms"] = (time.perf_counter() - t0) * 1000
            return run_metrics(
                indexed, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size, stats=stats
            )
    return _measure_media(
        source,
        alignment_fields=fields,
        expected_hz=expected_hz,
        thresholds=thresholds,
        stream_idx=stream_idx,
        batch_size=batch_size,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stats=stats,
        stream_wrapper=stream_wrapper,
    )
