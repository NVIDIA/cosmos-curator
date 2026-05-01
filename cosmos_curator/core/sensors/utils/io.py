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
from typing import Any, BinaryIO, Literal

import numpy as np
import smart_open  # type: ignore[import-untyped]
from smart_open.utils import FileLikeProxy  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.types.types import DataSource


def open_file(
    src: DataSource, mode: Literal["rb", "wb"] = "rb", client_params: dict[str, Any] | None = None
) -> BinaryIO | FileLikeProxy:
    """Convert *src* to a readable or writable file-like object.

    Return type is :class:`~typing.BinaryIO` or :class:`~smart_open.utils.FileLikeProxy`.
    All current return paths are compatible with ``with`` statements, so callers
    may use ``with open_file(...) as stream: ...`` when they own the returned
    handle and want it closed promptly.

    If *src* is an Azure or S3 URI, pass *client_params* for the client.

    If *src* is already a caller-owned buffered binary stream, this function returns it
    unchanged. Callers that use ``with`` and need borrow-only semantics should
    use :func:`open_data_source` instead. In general, prefer
    :func:`open_data_source` when ownership or automatic cleanup semantics matter.

    Arguments:
        src: Local path, URI, raw bytes, or an existing buffered binary stream.
        mode: File open mode (default ``"rb"``).  Use ``"wb"`` for binary writes.
            Must be a read or write mode (e.g. ``"rb"``, ``"wb"``).
        client_params: Extra arguments for ``smart_open`` when *src* is a cloud URI.

    Returns:
        A BinaryIO or FileLikeProxy object. Callers are responsible for closing
        the returned handle when appropriate.

    """
    src_obj: object = src
    match src_obj:
        case str() as src_str:
            if src_str.startswith(("s3://", "az://")):
                if client_params is None:
                    msg = "client_params is required when src is an s3 or azure path"
                    raise ValueError(msg)
                return smart_open.open(src_str, mode, **client_params)
            if "://" in src_str:
                msg = f"unsupported URI scheme in {src_str!r}; only s3:// and az:// are supported"
                raise ValueError(msg)
            return Path(src_str).open(mode)
        case Path() as path:
            return path.open(mode)
        case bytes() as src_bytes:
            return io.BytesIO(src_bytes)
        case np.ndarray() as src_array:
            if src_array.dtype != np.uint8:
                msg = f"ndarray data sources must have dtype uint8, got {src_array.dtype}"
                raise ValueError(msg)
            return io.BytesIO(np.ascontiguousarray(src_array).tobytes())
        case io.BufferedIOBase() as src_stream:
            # ``io.BytesIO``, ``io.BufferedReader``, etc. ``typing.BinaryIO`` is not an isinstance
            # target at runtime, so we use the stdlib binary buffered base class.
            return src_stream
        case _:
            error_msg = f"Invalid src type: {type(src)}"
            raise ValueError(error_msg)


@contextmanager
def open_data_source(
    src: DataSource, mode: Literal["rb", "wb"] = "rb", client_params: dict[str, Any] | None = None
) -> Generator[BinaryIO | FileLikeProxy, None, None]:
    """Open owned sources, or temporarily borrow caller-owned binary streams.

    For ``Path``/``str``/``bytes``/``ndarray`` inputs, this delegates to
    :func:`open_file` and closes the created stream on exit.

    For caller-owned buffered binary streams, ownership is retained by the caller.
    Seekable streams are rewound on entry and restored to their original
    position on exit so repeated opens behave like a fresh read. Non-seekable
    borrowed streams are unsupported because sensor code may reopen the source
    multiple times across indexing, metadata loading, and sampling.
    """
    if isinstance(src, io.BufferedIOBase):
        if not src.seekable():
            msg = "borrowed binary streams must be seekable"
            raise ValueError(msg)
        position = src.tell()
        src.seek(0)
        try:
            yield src
        finally:
            src.seek(position)
        return

    with open_file(src, mode=mode, client_params=client_params) as stream:
        yield stream
