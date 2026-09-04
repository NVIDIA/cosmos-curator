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

"""Reading a session's artifacts, whatever backend holds them.

The sensor library accepts no URIs and imports no cloud client by design, so
resolving one to an open stream is this recipe's job. ``smart_open`` covers a
local path, a ``file://`` URI and an ``s3://`` URI alike, which is why nothing
here inspects the scheme.
"""

import fnmatch
import io
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, BinaryIO, cast

import numpy as np
import numpy.typing as npt
import smart_open

from cosmos_curator.core.sensors.utils.timestamps import read_timestamps_ns
from cosmos_curator.core.utils.storage.s3_client import S3Client
from cosmos_curator.core.utils.storage.storage_utils import (
    get_files_relative,
    get_storage_client,
    is_remote_path,
)
from cosmos_curator.next.recipes.multimodal_split.reports import errors_prefix
from cosmos_curator.next.recipes.multimodal_split.uris import resolve_sensor_uri, slashed, timeline_uri


@contextmanager
def open_sensor_source(
    sensor_uri: str, *, storage_profile: str = "default", read_buffer_bytes: int | None = None
) -> Iterator[io.BufferedIOBase]:
    """Open a camera artifact as the seekable stream the sensor library takes.

    The sensor library accepts no URIs and imports no cloud client by design, so
    resolving one to bytes is this caller's job. ``smart_open`` returns a seekable
    stream for a local path, a ``file://`` URI, and an ``s3://`` URI alike, which
    is why nothing here inspects the scheme -- a recipe that branched on it would
    be re-deciding something the storage layer already knows.

    The stream is handed over as the ``io.BufferedIOBase`` arm of ``DataSource``
    even for a local path, rather than as a ``Path``. That arm is caller-owned:
    the library positions the stream but neither closes it nor restores its
    position, so the caller must hold it open for as long as it samples and close
    it afterwards. Passing a stream throughout also means this recipe survives the
    ``Path`` arm being removed.

    Credentials come from the same storage profile the written clips use, so a
    run cannot read with one identity and write with another.
    """
    client = get_storage_client(sensor_uri, profile_name=storage_profile)
    # ``None`` means the storage layer recognized no remote backend, which is the
    # only signal needed: transport params configure a client or there is none.
    if client is None:
        transport_params: dict[str, Any] = {}
    elif isinstance(client, S3Client):
        transport_params = {"client": client.s3}
        # S3 only. A local path goes through the builtin open, which takes no
        # such parameter and buffers by block. This is not the size of a request:
        # ``range_chunk_size`` is what would split the object into ranged GETs and
        # it defaults to None, so smart_open issues one GET and streams the body
        # through a buffer this size. What it bounds is memory held per open
        # stream, of which a rig has one per sensor, and how far a forward seek
        # can reach without reopening the body.
        if read_buffer_bytes is not None:
            transport_params["buffer_size"] = read_buffer_bytes
    else:
        # The config restricts locations to s3 and file, so nothing else can
        # arrive here today. Naming it beats an AttributeError on some other
        # client's internals if that ever widens.
        msg = f"multimodal-split can read a camera over S3 or a local path, not {type(client).__name__}"
        raise NotImplementedError(msg)

    with smart_open.open(sensor_uri, "rb", transport_params=transport_params) as stream:
        # smart_open types its return as BinaryIO; every backend reachable here
        # yields a seekable buffered reader, which is DataSource's third arm.
        yield cast("io.BufferedIOBase", stream)


def load_timeline_ns(
    sensor_uri: str, *, storage_profile: str = "default", read_buffer_bytes: int | None = None
) -> npt.NDArray[np.int64]:
    """Load the camera's capture timestamps, in nanoseconds.

    The camera's own timestamps, not a rate: real capture deltas on 30 fps
    footage measure 33330 to 33361 microseconds, so a timeline derived from a
    nominal rate drifts against the camera it describes.
    """
    with open_sensor_source(
        timeline_uri(sensor_uri), storage_profile=storage_profile, read_buffer_bytes=read_buffer_bytes
    ) as stream:
        # open_sensor_source is typed to DataSource's BufferedIOBase arm; np.load
        # wants the narrower BinaryIO, and every stream reachable here satisfies it.
        return read_timestamps_ns(cast("BinaryIO", stream))


def resolve_sensor_pattern(
    session_uri: str,
    uri_pattern: str,
    *,
    storage_profile: str = "default",
    session_paths: list[str] | None = None,
) -> str:
    """Resolve one camera's pattern against a session, wildcards and all.

    A pattern naming its path exactly is joined without asking storage anything.
    One with a wildcard segment is matched against the session's contents, which
    matched against the session's contents -- a rig spreads twelve cameras over
    five recorders, and which recorder holds which is the rig's business rather
    than the config's.

    Pass ``session_paths`` to resolve a whole rig against one listing. Without
    it every camera lists the session again, which is twelve recursive listings
    of the same prefix for one twelve-camera session.

    Exactly one match, or neither answer is safe: none means the camera is not in
    this session, and several mean the pattern describes the rig rather than a
    camera in it. Picking one would silently drop a view from every clip.

    Args:
        session_uri: The session to resolve within.
        uri_pattern: A session-relative path, optionally with ``*`` segments.
        storage_profile: Profile name used to build storage clients.
        session_paths: A listing of the session to match against, so a whole rig
            resolves from one listing instead of one per camera.

    Returns:
        The session-relative path that matched, joined onto *session_uri*.

    Raises:
        FileNotFoundError: If nothing in the session matches.
        ValueError: If more than one thing does.

    """
    if "*" not in uri_pattern:
        return resolve_sensor_uri(session_uri, uri_pattern)
    listing = (
        session_paths if session_paths is not None else list_session_paths(session_uri, storage_profile=storage_profile)
    )
    matches = sorted(match_session_paths(listing, uri_pattern))
    if not matches:
        msg = f"no artifact in {session_uri} matches {uri_pattern!r}"
        raise FileNotFoundError(msg)
    if len(matches) > 1:
        msg = (
            f"{uri_pattern!r} matches {len(matches)} artifacts in {session_uri}, so it names no single "
            f"camera: {', '.join(matches)}"
        )
        raise ValueError(msg)
    return resolve_sensor_uri(session_uri, matches[0])


def list_error_reports(output_path: str, *, storage_profile: str = "default") -> list[str]:
    """Return every failure report a run wrote, by one listing of the errors tree.

    Empty when nothing failed, which is the common case and answers in a single
    request: reports are written only where there was something to report, so
    asking each session for one instead is a request per session that misses
    whenever the session was fine.

    A tree that was never created is not a fault. Nothing failed, so there is
    nothing to list, and the backends disagree about whether an absent prefix is
    an error or an empty answer.
    """
    prefix = errors_prefix(output_path)
    try:
        names = list_session_paths(prefix, storage_profile=storage_profile)
    except (FileNotFoundError, NotADirectoryError):
        return []
    return [f"{prefix.rstrip('/')}/{name}" for name in names if name.endswith(".json")]


def list_session_paths(session_uri: str, *, storage_profile: str = "default") -> list[str]:
    """Return every session-relative file path, whatever backend holds the session.

    One listing serves every camera in the rig. Resolving twelve wildcard
    patterns by listing twelve times is twelve recursive listings of the same
    session, which at a thousand concurrent sessions is the difference between
    one request and twelve.
    """
    # The listing itself is backend-agnostic; only the identity it reads under is
    # not. ``get_storage_client`` answers ``None`` for a local path and for a
    # remote one it could not build a client for, so the scheme is what says
    # which of those this is -- and a remote path with no client has to say so,
    # or the walk finds nothing and a rig reads as missing every sensor.
    client = None
    if is_remote_path(session_uri):
        client = get_storage_client(session_uri, profile_name=storage_profile)
        if client is None:
            msg = f"Could not create a storage client for {session_uri} under profile {storage_profile!r}"
            raise NotImplementedError(msg)
    # Passed explicitly rather than resolved inside: left to itself it builds one
    # on the default profile, and a run must not read with one identity and write
    # with another.
    return get_files_relative(slashed(session_uri), client)


def match_session_paths(paths: list[str], uri_pattern: str) -> list[str]:
    """Return the paths matching *uri_pattern*, one path segment at a time."""
    return [path for path in paths if _matches_by_segment(path, uri_pattern)]


def _matches_by_segment(path: str, uri_pattern: str) -> bool:
    """Match a session-relative path against a pattern one path segment at a time.

    ``fnmatch`` alone would not do: its ``*`` spans separators, so ``*/front.mp4``
    would also match ``recorder-00/sub/front.mp4``. The local branch globs, where
    ``*`` is exactly one segment, and the two backends have to agree -- otherwise
    the same config finds a camera on S3 and reports it missing on a local copy
    of the same session.
    """
    segments = path.split("/")
    wanted = uri_pattern.split("/")
    if len(segments) != len(wanted):
        return False
    return all(fnmatch.fnmatchcase(segment, expected) for segment, expected in zip(segments, wanted, strict=True))
