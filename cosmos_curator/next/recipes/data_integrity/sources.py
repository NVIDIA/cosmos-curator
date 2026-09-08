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

Both entry points build on :func:`run_checks`: the single-video ``di-check`` CLI
(:mod:`.cli`) and the concurrent session runner (:mod:`.session_runner`).
"""

import pathlib
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import BinaryIO, cast

import numpy as np
import numpy.typing as npt
from loguru import logger

from cosmos_curator.core.sensors.data_integrity.engine import run_metrics
from cosmos_curator.core.sensors.data_integrity.instruments import DEFAULT_THRESHOLDS, Thresholds
from cosmos_curator.core.sensors.data_integrity.results import CheckResult, ResolvedConfig, VideoInfo
from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.timestamps import read_timestamps_ns
from cosmos_curator.core.utils.storage.storage_utils import is_remote_path
from cosmos_curator.core.utils.storage_cli import open_storage_source


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


#: What a recorder's capture timeline is called, relative to the media it describes.
#: Spelled the same way ``multimodal_split`` spells it (see that recipe's ``uris.py``),
#: because that is the artifact this reads on the way *in*; not imported from there,
#: since one recipe reaching into another's naming would couple them over a constant.
#: What that recipe writes on the way *out* wears the same name and is not the same
#: thing -- see :data:`_DERIVED_TIMELINE_MARKER_SUFFIX`.
CAPTURE_TIMELINE_SUFFIX = ".timestamps.npy"

#: The artifact whose presence means the timeline beside it is a resampling grid rather
#: than a recording's own clock.
#:
#: ``multimodal_split`` cuts a session into clips by resampling every camera onto one
#: evenly spaced grid at the configured output rate, and writes *that grid* to
#: :data:`CAPTURE_TIMELINE_SUFFIX` -- deliberately, so a later stage reads a clip the
#: way it read a session. The instants a camera actually observed go beside it under
#: this name instead. Read as a capture timeline, the grid answers the session grain's
#: question before it is asked: every camera is on the same lattice, so spread is zero
#: and overlap total, however far out of sync the rig behind the clips was. So a clip
#: is measured on its container's PTS and reported as such, which is the outcome for
#: any other source that cannot say when its frames happened.
#:
#: Absence of a marker is not proof of a recorder, only of nothing claiming otherwise.
#: This recipe is meant for sessions as recorded, and a derived tree is out of contract
#: rather than supported by this; the check is here because being wrong quietly is
#: worse than the one request it costs.
_DERIVED_TIMELINE_MARKER_SUFFIX = ".observations.npy"

#: S3 / Azure error codes that mean the object is not there. Distinct from
#: :data:`~.session_runner._INFRASTRUCTURE_S3_CODES`, which answers the opposite
#: question -- whether the request failed for reasons of our own access -- and never
#: needed to name the 404 family because a missing *video* is a fact about the dataset
#: rather than an absence to be tolerated. A missing capture timeline is the common
#: case, so this path has to recognise it precisely enough not to swallow a 403.
_NOT_FOUND_CODES = frozenset(
    {"404", "NoSuchKey", "NoSuchBucket", "BlobNotFound", "ContainerNotFound", "ResourceNotFound"}
)
_NOT_FOUND_STATUS = 404


def capture_timeline_uri(source: str) -> str:
    """Return where a stream's capture timeline would sit, whether or not one does."""
    return f"{source}{CAPTURE_TIMELINE_SUFFIX}"


def derived_timeline_marker_uri(source: str) -> str:
    """Return where the marker disowning a stream's timeline would sit, if one does."""
    return f"{source}{_DERIVED_TIMELINE_MARKER_SUFFIX}"


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


def _is_not_found(exc: BaseException) -> bool:
    """Report whether ``exc`` says the object is absent, rather than unreachable.

    Only absence may be treated as "this stream has no capture timeline". Anything
    else -- a denial, a 5xx, an expired token -- has to keep travelling, because
    reading it as absence would quietly measure a whole rig against the container's
    metronome and record that as the recording's own timeline. Which is why the code
    and the status come from the same readers
    :func:`~.session_runner.is_infrastructure_error` uses: whatever a chain is taken
    to say, both questions have to be answered off the same reading of it.
    """
    if any(isinstance(error, FileNotFoundError) for error in error_chain(exc)):
        return True
    return error_code(exc) in _NOT_FOUND_CODES or http_status(exc) == _NOT_FOUND_STATUS


def _timeline_is_disowned(
    source: str,
    *,
    s3_profile_name: str | None,
    azure_profile_name: str,
    endpoint_url: str | None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None,
) -> bool:
    """Report whether a marker sits beside ``source`` disowning its timeline.

    Existence is the whole signal; the contents are never read, and could not be used
    if they were. The marker holds one observation per output frame, which repeats
    wherever the grid asked for more frames than the camera captured -- a dropped
    exposure -- so it is non-decreasing where a timeline must strictly ascend, and
    :class:`~cosmos_curator.core.sensors.sensors.camera_sensor.CameraSensor` would
    reject it. What it says instead is that something upstream distinguished the two
    clocks and wrote the honest one down separately.

    One byte is read because a stream that opens is not yet a stream that exists on
    every backend, and a probe that cannot tell absence from a lazy handle would
    disown every timeline it met.
    """
    try:
        with open_source(
            derived_timeline_marker_uri(source),
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            stream_wrapper=stream_wrapper,
        ) as stream:
            stream.read(1)
    except OSError as exc:
        if _is_not_found(exc):
            return False
        raise
    return True


def read_capture_timeline(
    source: str,
    *,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
    stream_wrapper: Callable[[BinaryIO], BinaryIO] | None = None,
) -> npt.NDArray[np.int64] | None:
    """Read the capture timeline recorded beside ``source``, or ``None`` if there is none.

    The recipe half of preferring capture times over container PTS: the sensor library
    accepts a timeline but never a URI, so finding the artifact is this layer's job (see
    :class:`~cosmos_curator.core.sensors.sensors.camera_sensor.CameraSensor`).

    Four outcomes, and the difference between them matters:

    * absent -- ``None``, and the caller measures the container's PTS. The ordinary
      case for any source whose recorder wrote no timeline.
    * present but disowned -- ``None`` as well, and logged. A timeline the tree itself
      says is a resampling grid is worth less than no timeline at all, because it
      would be believed; see :data:`_DERIVED_TIMELINE_MARKER_SUFFIX`.
    * present but not a timeline -- ``ValueError`` from
      :func:`~cosmos_curator.core.sensors.utils.timestamps.read_timestamps_ns`, which
      the session runner records as that stream's error. An artifact that fails its
      own reader is a defect worth reporting, not one to fall back from.
    * unreachable -- propagated untouched, for
      :func:`~.session_runner.is_infrastructure_error` to classify as it does any
      other failed fetch.

    Args:
        source: the stream's local path or cloud URI. The timeline is looked for beside
            it, under the same name plus :data:`CAPTURE_TIMELINE_SUFFIX`.
        s3_profile_name: AWS profile forwarded to ``open_storage_source``.
        azure_profile_name: Azure profile forwarded to ``open_storage_source``.
        endpoint_url: optional S3 endpoint override.
        stream_wrapper: optional wrapper applied to the timeline's stream, on the
            same terms as the video's. A cancellable reader that covered only the
            video would leave a hung fetch of this artifact uninterruptible, and it
            is fetched first.

    Raises:
        ValueError: If the artifact exists but does not hold one ascending int64
            ``.npy`` timeline.

    """
    uri = capture_timeline_uri(source)
    try:
        with open_source(
            uri,
            s3_profile_name=s3_profile_name,
            azure_profile_name=azure_profile_name,
            endpoint_url=endpoint_url,
            stream_wrapper=stream_wrapper,
        ) as stream:
            timeline_ns = read_timestamps_ns(stream)
    except OSError as exc:
        # Narrower than it looks: read_timestamps_ns raises ValueError for a bad
        # artifact and documents OSError as meaning the bytes were never seen, so
        # what reaches here is a failed fetch and nothing about the timeline itself.
        if _is_not_found(exc):
            return None
        raise
    # Only once a timeline is in hand, so the ordinary source -- no sidecar at all --
    # pays nothing for a question that was already answered.
    if _timeline_is_disowned(
        source,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stream_wrapper=stream_wrapper,
    ):
        # Logged rather than recorded as the stream's error: nothing here is broken,
        # and the stored row says what it always says -- which clock this was measured
        # on. The line is what tells an operator why a tree full of timelines reports
        # container PTS anyway.
        logger.warning(
            f"{uri} is a resampling grid, not a recording's clock "
            f"({derived_timeline_marker_uri(source)} sits beside it); "
            f"measuring container PTS instead"
        )
        return None
    return timeline_ns


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
) -> tuple[list[CheckResult], VideoInfo, ResolvedConfig]:
    """Open ``source``, build a :class:`CameraSensor`, and run every metric on it.

    The I/O wrapper around :func:`~cosmos_curator.core.sensors.data_integrity.engine.run_metrics`:
    it resolves the local path or cloud stream, constructs the sensor, and delegates
    the compute. Any open / decode failure propagates as an exception for the caller
    to classify (the single-video CLI maps it to exit code 2; the session tool records
    a per-stream ERROR).

    Also where a capture timeline is preferred over the container's PTS, when the source
    has one beside it (:func:`read_capture_timeline`). Done here rather than at either
    entry point so both inherit it, and unconditionally rather than behind a flag: a
    recorder's own timestamps are the better answer wherever they exist, and which clock
    a stream was measured on is reported rather than chosen. The two exceptions are both
    cases where the artifact's name stops saying whose clock it is: a non-zero
    ``stream_idx``, and a timeline the tree marks as a resampling grid.

    Args:
        source: local path, ``s3://`` URI, or ``az://`` URI.
        expected_hz: expected sample rate for the rate-dependent metrics.
        thresholds: pass/fail policy (see :class:`Thresholds`).
        stream_idx: which video stream to open (default 0). Only 0 may be paired with
            a capture timeline, since the sidecar names the file rather than a stream.
        batch_size: timestamps per metric update; 0 feeds the whole array at once.
        s3_profile_name: optional AWS profile forwarded to ``open_storage_source``.
        azure_profile_name: Azure profile forwarded to ``open_storage_source``.
        endpoint_url: optional S3 endpoint override for S3-compatible stores.
        stats: optional out-parameter; ``sensor_init_ms`` is recorded here in
            addition to the ``stream_ms`` / ``evaluate_ms`` from ``run_metrics``.
        stream_wrapper: optional wrapper applied to every stream this opens -- the
            capture timeline as well as the video (e.g. a byte-counting reader for
            progress), for local paths as well as cloud URIs.

    Returns:
        The tuple returned by ``run_metrics``.

    Raises:
        ValueError: If ``stream_idx`` is not 0 and a capture timeline sits beside
            ``source``; see :func:`read_capture_timeline` for why it cannot be
            attributed to a particular stream.

    """
    # Before the video is opened, so a source whose timeline is unreachable fails
    # without paying to decode it -- and so a corrupt timeline is reported as the
    # stream's error rather than being overtaken by whatever the video says.
    capture_timestamps_ns = read_capture_timeline(
        source,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stream_wrapper=stream_wrapper,
    )
    if stream_idx != 0 and capture_timestamps_ns is not None:
        # The artifact is named for the file, not for a stream within it, so which
        # stream it describes is simply not recorded. Refused rather than guessed:
        # applying it to another stream would either silently put that stream on a
        # clock measured elsewhere, or fail on a frame count that says nothing about
        # the recording. A stream with no timeline beside it is unaffected.
        msg = (
            f"{capture_timeline_uri(source)} does not record which video stream it describes, "
            f"so it cannot be applied to stream_idx {stream_idx}; only stream 0 is defined"
        )
        raise ValueError(msg)
    with open_source(
        source,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
        stream_wrapper=stream_wrapper,
    ) as src:
        t0 = time.perf_counter()
        sensor = CameraSensor(
            _as_data_source(src),
            stream_idx=stream_idx,
            capture_timestamps_ns=capture_timestamps_ns,
        )
        if stats is not None:
            stats["sensor_init_ms"] = (time.perf_counter() - t0) * 1000
        return run_metrics(sensor, expected_hz=expected_hz, thresholds=thresholds, batch_size=batch_size, stats=stats)
