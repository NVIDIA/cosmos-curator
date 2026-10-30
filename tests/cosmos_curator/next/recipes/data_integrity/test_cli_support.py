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

"""Unit tests for the cancellation plumbing behind the ``di`` tool."""

import io
import threading

import pytest

from cosmos_curator.core.sensors.utils.io import open_data_source
from cosmos_curator.next.recipes.data_integrity.cli_support import cancellable_reader


class _NotSeekable(io.BufferedIOBase):
    """A readable but unseekable stream, which the sensor library refuses to accept."""

    def read(self, size: int | None = -1) -> bytes:
        return b"x" * (16 if size in (None, -1) else int(size))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def test_the_wrapper_reports_its_streams_own_capabilities() -> None:
    """Capabilities are delegated, not asserted, so the sensor's guard still sees the truth.

    The sensor library rejects a stream that is not readable and seekable. A wrapper
    claiming both would smuggle an unusable stream past that check and turn a clear
    up-front error into an opaque failure inside libav.
    """
    assert not cancellable_reader(_NotSeekable(), threading.Event()).seekable()  # type: ignore[arg-type]

    seekable = cancellable_reader(io.BytesIO(b"x"), threading.Event())
    assert seekable.seekable()
    assert seekable.readable()


def test_an_unseekable_stream_is_still_rejected_up_front() -> None:
    """The wrapper must not let a stream the sensor cannot use reach libav."""
    wrapped = cancellable_reader(_NotSeekable(), threading.Event())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seekable"), open_data_source(wrapped):  # type: ignore[arg-type]
        pass
