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

"""Driver-side candidate session discovery for Curator Next ``multimodal-split``.

Discovery turns the configured input location into a deterministic PyArrow table
of candidate sessions for the first Ray Data stage. It deliberately stops at the
session boundary: it does not resolve camera, IMU, GPS, or calibration artifact
paths, check that any artifact exists, or decode media. Those belong to the
splitting stage.

The production output is a ``pyarrow.Table`` rather than a sequence of Python
session objects because a large run may discover millions of sessions, and one
Python object per session would add avoidable Ray Data serialization, copying,
and garbage-collection overhead.

A ``session_uri`` is the configured prefix with a session ID appended: an
``s3://`` URI for remote input and an absolute filesystem path for local input.
Both forms are accepted directly by the storage helpers and the sensor library,
so the splitting stage can join artifact patterns onto them without converting
anything first.
"""

import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pyarrow as pa
import pyarrow.compute as pc
from botocore.exceptions import ClientError

from cosmos_curator.core.utils.storage.s3_client import S3Client, S3Prefix, is_s3path
from cosmos_curator.core.utils.storage.storage_utils import (
    get_storage_client,
    list_child_directories,
    read_text,
)
from cosmos_curator.next.recipes.multimodal_split.config import MultimodalSplitInputConfig

CANDIDATE_SESSION_SCHEMA = pa.schema(
    [
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("session_uri", pa.large_string(), nullable=False),
    ]
)

_RESERVED_SESSION_IDS = frozenset({".", ".."})
_UTF8_BOM = "\ufeff"
_URI_SEPARATOR = pa.scalar("/", type=pa.large_string())

# ``head_object`` reports a missing key as the bare status code because a HEAD
# response carries no body to parse a code from; S3-compatible stores are less
# consistent, so the documented key-level codes are accepted too.
_MISSING_OBJECT_CODES = frozenset({"404", "NotFound", "NoSuchKey"})
_FORBIDDEN_OBJECT_CODES = frozenset({"403", "AccessDenied", "AllAccessDisabled"})
_HTTP_NOT_FOUND = 404
_HTTP_FORBIDDEN = 403


def discover_candidate_sessions(
    config: MultimodalSplitInputConfig,
    *,
    storage_profile: str = "default",
) -> pa.Table:
    """Enumerate candidate sessions as a table of ``CANDIDATE_SESSION_SCHEMA`` rows.

    With ``input_path_prefix`` alone, the immediate children of the prefix are the
    candidate sessions. When ``session_id_list_path`` is also set, the session IDs
    are read from that file instead and joined to the prefix without listing it.

    In both modes the IDs are deduplicated and sorted before ``limit`` is applied,
    so a given configuration and input always produce the same rows in the same
    order.

    In session-ID-list mode the prefix is only a join base and is never listed or
    checked for existence, so a prefix that does not exist yields rows rather than
    an error. Confirming that a session is real is the splitting stage's job.

    A candidate is not guaranteed to be a distinct recording. Local listing
    follows symlinks, so a ``latest -> session-a`` convention produces two
    candidates for one recording; see ``_list_local_child_session_ids``. A caller
    that must not process a recording twice should deduplicate by resolved target
    rather than assuming session IDs are one-to-one with recordings.

    The returned table becomes a single Ray Data block, because ``from_arrow``
    maps one table to one block regardless of how the table is chunked. A caller
    wanting parallelism across a large selection should repartition it.

    Args:
        config: The resolved candidate session selection.
        storage_profile: Profile name used to build S3 clients.

    Returns:
        A table with one row per candidate session.

    Raises:
        FileNotFoundError: If a listed local prefix does not exist, a listed S3
            prefix holds no objects, or the session ID list does not exist.
        NotADirectoryError: If a local prefix names a file instead of a directory.
        ValueError: If a session ID is not a single immediate child name or is not
            valid UTF-8.

    """
    if config.session_id_list_path is not None:
        session_ids = _read_session_id_list(config.session_id_list_path, storage_profile=storage_profile)
    else:
        session_ids = _list_child_session_ids(config.input_path_prefix, storage_profile=storage_profile)

    ordered = sorted({_validate_session_id(session_id) for session_id in session_ids})
    if config.limit is not None:
        ordered = ordered[: config.limit]
    return _build_candidate_session_table(ordered, config.input_path_prefix)


def _build_candidate_session_table(session_ids: list[str], input_path_prefix: str) -> pa.Table:
    """Join canonical session IDs onto the prefix without a Python object per session."""
    base = pa.scalar(_session_uri_base(input_path_prefix), type=pa.large_string())
    ids = pa.array(session_ids, type=pa.large_string())
    uris = pc.binary_join_element_wise(base, ids, _URI_SEPARATOR)
    return pa.Table.from_arrays([ids, uris], schema=CANDIDATE_SESSION_SCHEMA)


def _session_uri_base(input_path_prefix: str) -> str:
    """Return the location that session IDs are appended to.

    Local prefixes stay filesystem paths rather than becoming ``file://`` URIs so
    that a ``session_uri`` can be handed straight to the sensor library, which
    parses no URI scheme; emitting one would turn a forgotten conversion
    downstream into a decode failure on media that is present on disk. They are
    resolved to absolute paths because Ray workers do not share the driver's
    working directory.

    Trailing slashes are dropped so the join adds exactly one separator. Both a
    bucket root (``s3://bucket/``) and a filesystem root (``/``) reduce correctly.
    """
    if is_s3path(input_path_prefix):
        base = S3Prefix(input_path_prefix).path
    else:
        # ``abspath`` rather than ``resolve``: absolute is required because Ray
        # workers do not share the driver's working directory, but resolving
        # symlinks is not. A recording root is frequently a symlink or autofs
        # mount, and the workers know it by that name, not by its target.
        base = os.path.abspath(_local_path(input_path_prefix))  # noqa: PTH100
    return base.removesuffix("/")


def _validate_session_id(session_id: str) -> str:
    """Reject anything that is not a single immediate child name.

    Every current caller already drops empty entries, but an empty ID would join
    to a bare trailing separator that names the prefix itself rather than a
    session, so the validator rejects it on its own terms.
    """
    if not session_id or "/" in session_id or session_id in _RESERVED_SESSION_IDS:
        msg = f"Invalid session ID {session_id!r}: a session ID must name one immediate child of the input prefix"
        raise ValueError(msg)
    try:
        # POSIX directory names are bytes, so a non-UTF-8 name reaches us through
        # surrogateescape. Arrow rejects surrogates, and its own error names
        # neither the entry nor the prefix it came from.
        session_id.encode("utf-8")
    except UnicodeEncodeError as exc:
        msg = f"Session ID {session_id!r} is not valid UTF-8 and cannot be recorded"
        raise ValueError(msg) from exc
    return session_id


def _list_child_session_ids(input_path_prefix: str, *, storage_profile: str) -> list[str]:
    """List the immediate children of the prefix, without descending into them."""
    if is_s3path(input_path_prefix):
        return _list_s3_child_session_ids(input_path_prefix, storage_profile=storage_profile)
    return _list_local_child_session_ids(input_path_prefix)


def _list_local_child_session_ids(input_path_prefix: str) -> list[str]:
    """List the immediate child directories of a local prefix.

    The listing itself, including its symlink semantics, is
    :func:`list_child_directories`. This wrapper only restates its errors in
    terms of the config field the prefix came from.
    """
    try:
        return list_child_directories(_local_path(input_path_prefix))
    except FileNotFoundError as exc:
        msg = f"Input path prefix does not exist: {input_path_prefix}"
        raise FileNotFoundError(msg) from exc
    except NotADirectoryError as exc:
        msg = f"Input path prefix is not a directory: {input_path_prefix}"
        raise NotADirectoryError(msg) from exc


def _list_s3_child_session_ids(input_path_prefix: str, *, storage_profile: str) -> list[str]:
    """List the immediate child prefixes of an S3 prefix.

    The delimited listing itself, including why a prefix holding no keys is an
    error rather than an empty result, is
    :meth:`S3Client.list_child_prefixes`. This wrapper only restates that error
    in terms of the config field the prefix came from.
    """
    prefix = S3Prefix(input_path_prefix)
    client = _require_s3_client(prefix.path, storage_profile=storage_profile)
    try:
        return client.list_child_prefixes(prefix)
    except FileNotFoundError as exc:
        msg = f"Input path prefix does not exist or contains no objects: {input_path_prefix}"
        raise FileNotFoundError(msg) from exc


def _remote_object_exists(client: S3Client, prefix: S3Prefix, location: str) -> bool:
    """Probe an object, translating the store's refusals into a usable error.

    ``S3Client.object_exists`` maps only the literal code ``"404"`` to ``False``
    and re-raises everything else, so a permissions problem or an S3-compatible
    store that answers ``NoSuchKey`` to a HEAD would surface as a raw
    ``ClientError`` instead of the intended "list does not exist" message.

    A 403 is reported as a permissions error rather than a missing object, but
    the message says both: without ``s3:ListBucket`` on the bucket, AWS answers
    403 for an object that merely does not exist, so the two are genuinely
    indistinguishable from the caller's side.
    """
    try:
        return client.object_exists(prefix)
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = str(error.get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in _MISSING_OBJECT_CODES or status == _HTTP_NOT_FOUND:
            return False
        if code in _FORBIDDEN_OBJECT_CODES or status == _HTTP_FORBIDDEN:
            msg = (
                f"Access denied reading {location}. The credentials may lack s3:GetObject on it, "
                f"or it may not exist and the credentials lack s3:ListBucket on the bucket, which "
                f"makes a missing object indistinguishable from a forbidden one."
            )
            raise PermissionError(msg) from exc
        raise


def _require_s3_client(location: str, *, storage_profile: str) -> S3Client:
    """Return the client for a location the caller has already read as remote.

    Callers check the scheme before reaching this, and that order matters:
    ``get_storage_client`` answers ``None`` both for a local path and for a
    remote one it could not build a client for. Dispatching on the client alone
    would collapse those, turning an unreachable bucket into a walk of a
    directory that was never local -- which finds nothing, and reports a
    recording that is absent rather than a credential that is.
    """
    client = get_storage_client(location, profile_name=storage_profile)
    if not isinstance(client, S3Client):
        msg = f"Could not create an S3 client for {location}"
        raise TypeError(msg)
    return client


def _read_session_id_list(session_id_list_path: str, *, storage_profile: str) -> list[str]:
    """Read a newline-delimited UTF-8 file of session IDs."""
    if is_s3path(session_id_list_path):
        prefix = S3Prefix(session_id_list_path)
        client = _require_s3_client(prefix.path, storage_profile=storage_profile)
        # ``read_text`` retries with backoff, which is right for a transient read
        # fault but turns a mistyped list path into a multi-minute stall. A single
        # HEAD keeps that failure as fast and as clear as the local one.
        if not _remote_object_exists(client, prefix, session_id_list_path):
            msg = f"Session ID list does not exist: {session_id_list_path}"
            raise FileNotFoundError(msg)
        return _parse_session_id_list(read_text(prefix, client=client))

    path = _local_path(session_id_list_path)
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        msg = f"Session ID list does not exist: {session_id_list_path}"
        raise FileNotFoundError(msg) from exc
    return _parse_session_id_list(contents)


def _parse_session_id_list(contents: str) -> list[str]:
    r"""Split list contents into session IDs, dropping blank lines and padding.

    A UTF-8 BOM is removed explicitly because it is not whitespace: leaving it in
    place would turn the first entry into a session ID that silently matches
    nothing, and list files exported by Windows tooling routinely carry one.

    Splitting on ``\n`` rather than with ``str.splitlines`` keeps the format
    exactly "newline-delimited". ``splitlines`` also breaks on ``\v``, ``\f``,
    ``\x1c``-``\x1e``, ``\x85``, ``U+2028``, and ``U+2029``, all of which are
    legal in POSIX filenames and S3 keys, so it would split one real session ID
    into two that match nothing. Trailing ``\r`` is removed by the strip below.
    """
    return [stripped for line in contents.removeprefix(_UTF8_BOM).split("\n") if (stripped := line.strip())]


def _local_path(location: str) -> Path:
    """Return the filesystem path for a local path or ``file://`` URI."""
    if location.startswith("file://"):
        return _file_uri_to_path(location)
    return Path(location).expanduser()


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"} or not parsed.path:
        msg = f"Unsupported local file URI: {uri}"
        raise ValueError(msg)
    if parsed.query or parsed.fragment:
        # The config validator rejects these, but only for what it was handed.
        # Without the check here, an unencoded '#' or '?' reaching this function by
        # any other route silently truncates the path -- ``file:///data/my#dir``
        # would list ``/data/my`` -- rather than failing.
        msg = f"Local file URI {uri} contains an unencoded '?' or '#'; percent-encode them as %3F and %23"
        raise ValueError(msg)
    return Path(unquote(parsed.path))
