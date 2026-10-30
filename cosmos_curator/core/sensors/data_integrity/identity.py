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

"""Stable identities for the streams and sessions a data-integrity run records.

A *stream* is not always a file: one MCAP holds many topics, each its own
timeline. So a stream is addressed by a file plus a **selector**
(``selector_type`` / ``selector_value``), and :func:`stream_id` hashes that triple
into the key every store dataset de-duplicates on.

The id is derived from the *absolute* source rather than from a session root plus
a relative key. The two carry the same information -- a root plus a relative key
reconstructs the source -- but the absolute form is invariant to **how the stream
was discovered**: checking one video directly and then checking the session that
contains it produces one identity instead of two rows for the same bytes.

Normalisation is deliberately narrow, and touches no filesystem: lowercase the URI
scheme, drop trailing slashes, and make local paths absolute. Nothing inside a cloud
object key is rewritten, because a key is an opaque string rather than a path.
Paths and S3 keys are case-sensitive, so case is never folded below the scheme, and
symlinks are left alone -- resolving them would make an id depend on the mount layout
of whichever machine happened to run the check.
"""

import hashlib
import os
import posixpath

# Storage backend a source lives on, recorded so a future session table can group by
# it without re-parsing URIs.
NAMESPACE_LOCAL = "local"
NAMESPACE_S3 = "s3"
NAMESPACE_AZURE = "az"

# How a stream is addressed *within* its file.
SELECTOR_VIDEO_STREAM = "video_stream"
SELECTOR_MCAP_TOPIC = "mcap_topic"

#: Default selector for the one-video-per-file sources handled today.
DEFAULT_SELECTOR_TYPE = SELECTOR_VIDEO_STREAM
DEFAULT_SELECTOR_VALUE = "0"

# 128 bits of a sha256, hex encoded. Short enough to eyeball in a report, wide
# enough that a collision is not a thing anyone needs to think about.
_ID_HEX_CHARS = 32

# Joins the parts of a hashed tuple. A NUL can't appear in a path or a selector, so
# no combination of parts can be re-split into a different tuple with the same digest.
_FIELD_SEPARATOR = "\0"

_CLOUD_SCHEMES = (NAMESPACE_S3, NAMESPACE_AZURE)


def _digest(*parts: str) -> str:
    """Hash ``parts`` into a short stable hex id.

    sha256 rather than :func:`hash`, whose randomised seed makes it differ between
    interpreter runs and so useless as a persisted key.
    """
    payload = _FIELD_SEPARATOR.join(parts).encode()
    return hashlib.sha256(payload).hexdigest()[:_ID_HEX_CHARS]


def _split_scheme(source: str) -> tuple[str, str]:
    """Split ``source`` into ``(scheme, remainder)``; scheme is ``""`` for a local path."""
    scheme, separator, remainder = source.partition("://")
    if not separator:
        return "", source
    return scheme.lower(), remainder


def locator_namespace(source: str) -> str:
    """Which storage backend ``source`` lives on: ``local``, ``s3``, or ``az``.

    An unrecognised scheme reports as its own lowercased scheme rather than raising:
    this is a descriptive column, and a store write is not the place to re-litigate a
    URI that the reader already accepted.
    """
    scheme, _ = _split_scheme(source)
    if not scheme:
        return NAMESPACE_LOCAL
    return scheme


def normalize_source(source: str) -> str:
    """Canonicalise a path or URI so equivalent spellings hash to one id.

    Local paths become absolute (against the current directory, like any other
    relative path a CLI is handed) with ``.`` / ``..`` segments folded. Cloud URIs
    keep their bucket and key verbatim apart from a lowercased scheme and a dropped
    trailing slash.
    """
    scheme, remainder = _split_scheme(source)
    if not scheme:
        # abspath, not Path.resolve: resolve would follow symlinks, so the same file
        # would hash differently depending on which link a caller happened to hand us.
        return os.path.abspath(source)  # noqa: PTH100
    # Nothing inside the key is folded: an object key is an opaque string rather than
    # a path, so "a/../b.mp4", "a//b.mp4" and "b.mp4" name three different objects and
    # normalising between them would merge them into one row. The trailing slash is
    # the one exception, because a prefix is written both ways by hand.
    return f"{scheme}://{remainder.rstrip('/')}"


def stream_id(
    source: str,
    *,
    selector_type: str = DEFAULT_SELECTOR_TYPE,
    selector_value: str = DEFAULT_SELECTOR_VALUE,
) -> str:
    """Derive the dedup key for one stream: a hash of its normalized source plus selector.

    Args:
        source: local path, ``s3://`` URI, or ``az://`` URI of the containing file.
        selector_type: how the stream is addressed inside that file
            (:data:`SELECTOR_VIDEO_STREAM` / :data:`SELECTOR_MCAP_TOPIC`).
        selector_value: the selector itself -- a video stream index as a string, or
            an MCAP topic name.

    """
    return _digest(normalize_source(source), selector_type, selector_value)


def session_id(session_path: str | None) -> str | None:
    """Derive the id of the session a run covered, or ``None`` for a single-stream run.

    Namespaced by backend so the same prefix under two backends stays distinct.
    """
    if session_path is None:
        return None
    return _digest(locator_namespace(session_path), normalize_source(session_path))


def relative_key(session_path: str | None, source: str) -> str | None:
    """Where ``source`` sits inside ``session_path``, as a normalized relative path.

    ``None`` when there is no session, or when ``source`` turns out not to live under
    it -- an escaping ``../`` result would describe a relationship that doesn't hold,
    which is worse than admitting we don't have one.

    ``relpath`` folds any ``.`` / ``..`` the key itself contains, unlike
    :func:`normalize_source`. That is fine here and wrong there: this is a descriptive
    column, while an id must keep two distinct keys distinct.
    """
    if session_path is None:
        return None
    root = normalize_source(session_path)
    normalized = normalize_source(source)
    scheme, _ = _split_scheme(normalized)
    # posixpath for cloud URIs (always "/"), os.path for local paths (platform aware).
    key = posixpath.relpath(normalized, root) if scheme else os.path.relpath(normalized, root)
    if key.startswith(os.pardir):
        return None
    return key
