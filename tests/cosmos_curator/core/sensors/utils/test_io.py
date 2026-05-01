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

import numpy as np
import pytest

from cosmos_curator.core.sensors.utils.io import open_data_source, open_file


@pytest.mark.parametrize(
    ("input_data", "expected_type", "raises"),
    [
        # Path and str open the file for reading (BinaryIO / BufferedReader)
        (
            Path("dummy.mp4"),
            io.BufferedReader,
            nullcontext(),
        ),
        (
            "dummy.mp4",
            io.BufferedReader,
            nullcontext(),
        ),
        (b"video data", io.BytesIO, nullcontext()),  # bytes input
        (io.BytesIO(b"stream data"), io.BytesIO, nullcontext()),  # existing BytesIO
        # Error cases
        (
            123,
            None,
            pytest.raises(ValueError),  # noqa: PT011, Integer input should raise ValueError
        ),
        (
            [],
            None,
            pytest.raises(ValueError),  # noqa: PT011, List input should raise ValueError
        ),
    ],
)
def test_open_file(
    input_data: Path | str | bytes | io.BytesIO | io.BufferedReader | int | list[Any],
    expected_type: type | tuple[type, ...] | None,
    raises: AbstractContextManager[Any],
    tmp_path: Path,
) -> None:
    """Test the _make_video_stream function with various input types.

    Args:
        input_data: The input data to test
        expected_type: The expected type of the returned stream
        raises: Either nullcontext() for success cases or pytest.raises() for error cases
        tmp_path: Pytest fixture providing a temporary directory

    """
    if isinstance(input_data, (Path, str)):
        # Create a temporary file for Path test case
        test_file = tmp_path / "dummy.mp4"
        test_file.write_bytes(b"test data")
        # Cast input data back to the the original type so that
        # make_video_stream is properly exercised
        input_data = input_data.__class__(test_file)

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


def test_open_file_raises_on_unsupported_uri_scheme() -> None:
    """Unsupported URI schemes should raise a clear ValueError instead of looking like local paths."""
    with pytest.raises(ValueError, match="unsupported URI scheme"):
        open_file("gs://bucket/video.mp4")


def test_open_file_uses_smart_open_for_supported_cloud_uris(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supported cloud URIs should delegate to smart_open with caller-provided client params."""
    calls: list[tuple[str, str, dict[str, Any]]] = []
    fake_stream = io.BytesIO(b"cloud-bytes")

    def fake_smart_open(path: str, mode: str, **client_params: object) -> io.BytesIO:
        calls.append((path, mode, client_params))
        fake_stream.seek(0)
        return fake_stream

    monkeypatch.setattr("cosmos_curator.core.sensors.utils.io.smart_open.open", fake_smart_open)

    result = open_file("s3://bucket/video.mp4", mode="rb", client_params={"transport_params": {"client": object()}})

    assert result is fake_stream
    assert calls[0][0] == "s3://bucket/video.mp4"
    assert calls[0][1] == "rb"
    assert "transport_params" in calls[0][2]


@pytest.mark.parametrize("uri", ["s3://bucket/video.mp4", "az://container/video.mp4"])
def test_open_file_requires_client_params_for_supported_cloud_uris(uri: str) -> None:
    """Supported cloud URIs should fail clearly when client_params are omitted."""
    with pytest.raises(ValueError, match="client_params is required"):
        open_file(uri)


def test_open_file_rejects_ndarray_with_non_uint8_dtype() -> None:
    """Ndarray data sources should be accepted only when their bytes are already uint8."""
    with pytest.raises(ValueError, match="ndarray data sources must have dtype uint8"):
        open_file(np.array([1, 2, 3], dtype=np.int16))


def test_open_file_accepts_uint8_ndarray_data_sources() -> None:
    """Uint8 ndarray data sources should be copied into a BytesIO stream without dtype coercion."""
    array = np.array([1, 2, 3, 4], dtype=np.uint8)

    result = open_file(array)

    assert isinstance(result, io.BytesIO)
    assert result.read() == b"\x01\x02\x03\x04"


def test_open_data_source_rewinds_and_restores_seekable_borrowed_stream() -> None:
    """Borrowed seekable streams should be rewound on entry and restored on exit."""
    stream = io.BytesIO(b"abcdef")
    stream.seek(3)

    with open_data_source(stream, mode="rb") as opened:
        assert opened is stream
        assert opened.tell() == 0
        assert opened.read(2) == b"ab"

    assert stream.tell() == 3
    assert not stream.closed


def test_open_data_source_restores_seekable_borrowed_stream_after_exception() -> None:
    """Borrowed seekable streams should restore their original position even if the caller raises."""
    stream = io.BytesIO(b"abcdef")
    stream.seek(4)

    def _raise_inside_context() -> None:
        with open_data_source(stream, mode="rb") as opened:
            assert opened.tell() == 0
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="boom"):
        _raise_inside_context()

    assert stream.tell() == 4


def test_open_data_source_rejects_nonseekable_borrowed_stream() -> None:
    """Borrowed binary streams must be seekable because the library may reopen them."""

    class _NonSeekableBytesIO(io.BytesIO):
        def seekable(self) -> bool:
            return False

    stream = _NonSeekableBytesIO(b"abcdef")

    with (
        pytest.raises(ValueError, match="borrowed binary streams must be seekable"),
        open_data_source(stream, mode="rb"),
    ):
        pass


def test_open_data_source_opens_and_closes_owned_sources() -> None:
    """Owned sources should be opened through open_file and closed on context exit."""
    with open_data_source(b"abcdef", mode="rb") as opened:
        assert opened.read() == b"abcdef"
        assert not opened.closed

    assert opened.closed
