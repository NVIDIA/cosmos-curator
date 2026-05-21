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
"""Test input/output utilities for the sensor library."""

import io
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import pytest

from cosmos_curator.core.sensors.utils.io import open_data_source, open_file


@pytest.mark.parametrize(
    ("input_data", "expected_type", "raises"),
    [
        (
            Path("dummy.mp4"),
            io.BufferedReader,
            nullcontext(),
        ),
        (b"video data", io.BytesIO, nullcontext()),
        (io.BytesIO(b"stream data"), io.BytesIO, nullcontext()),
        # Error cases: ``str`` URIs / ``int`` / ``list`` are no longer in DataSource.
        (
            "dummy.mp4",
            None,
            pytest.raises(ValueError),  # noqa: PT011
        ),
        (
            123,
            None,
            pytest.raises(ValueError),  # noqa: PT011
        ),
        (
            [],
            None,
            pytest.raises(ValueError),  # noqa: PT011
        ),
    ],
)
def test_open_file(
    input_data: Path | bytes | io.BytesIO | str | int | list[Any],
    expected_type: type | tuple[type, ...] | None,
    raises: AbstractContextManager[Any],
    tmp_path: Path,
) -> None:
    """Open the supported ``DataSource`` shapes and reject everything else."""
    if isinstance(input_data, Path):
        test_file = tmp_path / "dummy.mp4"
        test_file.write_bytes(b"test data")
        input_data = test_file

    with raises:
        result = open_file(input_data)  # type: ignore[arg-type]
        if expected_type is not None:
            assert isinstance(result, expected_type)
        if isinstance(result, io.BufferedReader):
            assert Path(result.name).exists()
        assert result.readable()
        data = result.read(1)
        assert len(data) == 1
        result.seek(0)
        result.close()


def test_open_file_rejects_non_readable_buffered_stream() -> None:
    """``open_file`` should reject a ``BufferedIOBase`` that is not readable."""

    class _NonReadable(io.BytesIO):
        def readable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="buffered binary streams must be readable"):
        open_file(_NonReadable(b"abcdef"))


def test_open_file_rejects_non_seekable_buffered_stream() -> None:
    """``open_file`` should reject a ``BufferedIOBase`` that is not seekable."""

    class _NonSeekable(io.BytesIO):
        def seekable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="buffered binary streams must be seekable"):
        open_file(_NonSeekable(b"abcdef"))


def test_open_data_source_yields_buffered_stream_without_seek_or_restore() -> None:
    """Borrowed buffered streams must be used as-is.

    The library does not ``seek(0)`` on entry and does not restore the
    caller's position on exit. The caller observes the stream wherever the
    library (and its inner consumers) left it.
    """
    stream = io.BytesIO(b"abcdef")
    stream.seek(3)

    with open_data_source(stream, mode="rb") as opened:
        assert opened is stream
        # No seek(0) on entry — the caller's position is preserved.
        assert opened.tell() == 3
        assert opened.read(2) == b"de"
        # Position drifts as the consumer reads.
        assert opened.tell() == 5

    # No restore on exit — the stream's position is wherever the library
    # left it (after the consumer's last read).
    assert stream.tell() == 5
    assert not stream.closed


def test_open_data_source_rejects_non_seekable_borrowed_stream() -> None:
    """Borrowed binary streams must be seekable because PyAV / PIL do absolute seeks."""

    class _NonSeekableBytesIO(io.BytesIO):
        def seekable(self) -> bool:
            return False

    stream = _NonSeekableBytesIO(b"abcdef")

    with (
        pytest.raises(ValueError, match="buffered binary streams must be seekable"),
        open_data_source(stream, mode="rb"),
    ):
        pass


def test_open_data_source_rejects_non_readable_borrowed_stream() -> None:
    """Borrowed binary streams must be readable."""

    class _NonReadableBytesIO(io.BytesIO):
        def readable(self) -> bool:
            return False

    stream = _NonReadableBytesIO(b"abcdef")

    with (
        pytest.raises(ValueError, match="buffered binary streams must be readable"),
        open_data_source(stream, mode="rb"),
    ):
        pass


def test_open_data_source_opens_and_closes_owned_sources() -> None:
    """Owned sources should be opened through open_file and closed on context exit."""
    with open_data_source(b"abcdef", mode="rb") as opened:
        assert opened.read() == b"abcdef"
        assert not opened.closed

    assert opened.closed
