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

"""Pure URI parsing: which locations are local, and what filesystem path each names.

Answers questions about a location's *text*; moving its bytes is
:mod:`cosmos_curator.next.utils.storage`, which imports these helpers back.

The split is load-bearing, not tidiness. ``storage`` imports ``botocore`` and,
transitively, ``azure.identity``; a client-only install declares neither, so a
config model that resolves a local path through ``storage`` fails at import time
and takes ``pipeline validate`` / ``pipeline render`` down with it. Keep every
import here in the standard library, and route a config or CLI caller that needs
a path to this module rather than to its byte-moving neighbour.
"""

from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlsplit

# Schemes that name an object on this machine's filesystem, whether written as a
# bare path or a file:// URI. Names the set that has an atomic create-if-absent
# primitive in ``storage``, which is narrower than the set of URIs a reader may
# open.
LOCAL_SCHEMES: Final = frozenset({"", "file"})


def local_path(location: str) -> Path:
    """Return the filesystem path a non-S3 location names, whether URI or path."""
    if location.startswith("file://"):
        return _file_uri_to_path(location)
    return Path(location)


def local_media_path(uri: str) -> Path:
    """Return the filesystem path for a bare local path or a ``file://`` URI.

    Public because the media reader resolves its local branch through the same
    rule; a second copy of it there could drift from what the writers accept.

    A non-``file`` scheme is refused rather than coerced: falling through to
    ``Path(uri)`` would turn an unsupported URI (``http://``, ``s3://``, a
    typo like ``s33://``) into a nonsense local path and read the wrong bytes
    silently. Only an empty scheme (a bare path) and the canonical ``file://``
    form are honoured; one-slash ``file:/`` input is malformed, as is a
    ``file://`` URI carrying an unencoded ``?`` or ``#``.
    """
    scheme = uri_scheme(uri)
    if scheme == "file":
        return _file_uri_to_path(uri)
    if scheme:
        msg = f"unsupported local media URI scheme {scheme!r} ({uri}); supported forms are bare local paths and file://"
        raise ValueError(msg)
    return Path(uri)


def uri_scheme(uri: str) -> str:
    """Return the lowercase scheme a URI claims, or ``""`` for a bare local path.

    Raises:
        ValueError: If a scheme precedes ``://`` but is not a legal scheme.

    """
    # Splitting the ``scheme://`` delimiter before parsing preserves
    # parser-hostile POSIX names such as ``//host[0]/share`` while urlsplit
    # remains the authority on URI syntax: urlsplit validates an authority it has
    # begun reading and rejects a bracket there, so handed the whole location it
    # fails identically for that bare path and for ``s3://host[0]/key``. It must
    # not be asked, because the two answers differ - reading the failure as "not
    # a URI, so a local path" would coerce the S3 key onto local disk, the silent
    # misread this helper refuses schemes to prevent. Same rule and same order as
    # ``next.utils.arrow_utils._validated_filesystem_scheme``.
    raw_scheme, separator, _ = uri.partition("://")
    if not separator:
        return _claimed_uri_scheme(uri)
    # Parsing the isolated token rather than prefix-matching it keeps one
    # definition of scheme syntax. The equality check is what refuses a token
    # urlsplit had to repair to parse: it strips whitespace, so " s3" and "s\t3"
    # both yield "s3", and accepting that would route a remote URI to local disk.
    scheme = urlsplit(f"{raw_scheme}:").scheme.lower()
    if not scheme or scheme != raw_scheme.lower():
        msg = f"malformed URI {uri!r}; supported forms are bare local paths and file://"
        raise ValueError(msg)
    return scheme


def _claimed_uri_scheme(uri: str) -> str:
    """Return the scheme a delimiter-less location claims, or ``""`` for a bare path."""
    try:
        return urlsplit(uri).scheme.lower()
    except ValueError:
        # Reached only without a ``://`` delimiter, so a rejected authority means
        # a ``//`` came first with no scheme before it - a bare path. Anything
        # claiming a scheme was parsed by the canonical branch above.
        return ""


def _file_uri_to_path(uri: str) -> Path:
    try:
        parsed = urlsplit(uri)
    except ValueError as error:
        msg = f"Unsupported local file URI: {uri}"
        raise ValueError(msg) from error
    # The authority is a host name, so it is case-insensitive: comparing it as
    # written would refuse ``file://LOCALHOST/x``, which the shared Arrow gate
    # resolves to this same local path. ``urlsplit`` already folds the scheme.
    #
    # ``?`` and ``#`` are URI delimiters, so urlsplit drops everything from the
    # first one onwards out of ``path``. Both are legal characters in a POSIX
    # name, so an unencoded one does not fail - it silently names a SHORTER path
    # than was written, and one that may well exist: ``file:///data/runs#2``
    # reads ``/data/runs``. Refusing loses no expressible name, because %3F and
    # %23 survive into ``path`` and are decoded below, and a bare local path
    # carries either character literally. Same rule as the shared Arrow gate and
    # as ``multimodal_split.config``.
    if (
        parsed.scheme != "file"
        or not uri.lower().startswith("file://")
        or parsed.netloc.lower() not in {"", "localhost"}
        or not parsed.path
        or parsed.query
        or parsed.fragment
    ):
        msg = f"Unsupported local file URI: {uri}"
        raise ValueError(msg)
    return Path(unquote(parsed.path))
