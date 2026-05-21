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
"""Input/output utilities for the sensor library."""

import io
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Literal, cast

from cosmos_curator.core.sensors.types.types import DataSource


def open_file(src: DataSource, mode: Literal["rb", "wb"] = "rb") -> BinaryIO:
    """Convert *src* to a readable or writable file-like object.

    The sensor library is backend-agnostic: it does not accept URIs and does
    not import any cloud client. Callers that want to feed data from a remote
    backend wrap the remote stream as a ``BinaryIO`` (e.g. via
    ``smart_open.open(uri, "rb")``) and pass that file-like in as
    ``io.BufferedIOBase``.

    Arguments:
        src: Local path, raw bytes, or an existing buffered binary stream.
        mode: File open mode (default ``"rb"``).  Use ``"wb"`` for binary writes.
            Must be a read or write mode (e.g. ``"rb"``, ``"wb"``).

    Returns:
        A ``BinaryIO`` object. For ``Path`` and ``bytes`` inputs the library
        owns the returned handle; callers are responsible for closing it
        (typically via a ``with`` statement). For
        :class:`io.BufferedIOBase` inputs the caller retains ownership and
        the stream is returned unchanged — the library performs no
        ``seek(0)`` on entry and does not restore the caller's position on
        exit. The stream must be readable and seekable; concurrent use by
        multiple readers is unsupported.

    """
    src_obj: object = src
    match src_obj:
        case Path() as path:
            return path.open(mode)
        case bytes() as src_bytes:
            return io.BytesIO(src_bytes)
        case io.BufferedIOBase() as src_stream:
            # ``io.BytesIO``, ``io.BufferedReader``, etc. ``typing.BinaryIO`` is not an isinstance
            # target at runtime, so we use the stdlib binary buffered base class. The
            # ``cast`` is structural: ``BufferedIOBase`` already supplies every method on
            # the ``BinaryIO`` Protocol surface used by PyAV / PIL.
            if not src_stream.readable():
                msg = "buffered binary streams must be readable"
                raise ValueError(msg)
            if not src_stream.seekable():
                msg = "buffered binary streams must be seekable"
                raise ValueError(msg)
            return cast("BinaryIO", src_stream)
        case _:
            error_msg = f"Invalid src type: {type(src)}"
            raise ValueError(error_msg)


@contextmanager
def open_data_source(src: DataSource, mode: Literal["rb", "wb"] = "rb") -> Generator[BinaryIO, None, None]:
    """Yield a binary stream for ``src``, owning or borrowing as appropriate.

    For ``Path`` and ``bytes`` inputs, delegates to :func:`open_file` and
    closes the created stream on exit.

    For caller-owned :class:`io.BufferedIOBase` inputs, ownership is retained
    by the caller. The stream must be readable and seekable. The library
    uses it as-is: no ``seek(0)`` on entry, no position restore on exit. The
    underlying decoder (PyAV / PIL) performs absolute seeks within the
    stream as it pleases. Concurrent use of the same stream by multiple
    readers (including reuse across overlapping sensor calls) is
    unsupported.
    """
    if isinstance(src, io.BufferedIOBase):
        if not src.readable():
            msg = "buffered binary streams must be readable"
            raise ValueError(msg)
        if not src.seekable():
            msg = "buffered binary streams must be seekable"
            raise ValueError(msg)
        yield cast("BinaryIO", src)
        return

    with open_file(src, mode=mode) as stream:
        yield stream
