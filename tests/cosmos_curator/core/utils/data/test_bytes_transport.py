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

"""Unit tests for bytes_transport boundary conversions and buffer-protocol claims."""

import io
import subprocess

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy, numpy_to_bytes

RAW_BYTES = b"hello world test data for buffer protocol verification"


@pytest.fixture
def uint8_array() -> npt.NDArray[np.uint8]:
    """C-contiguous uint8 array created from raw bytes."""
    return bytes_to_numpy(RAW_BYTES)


class TestBytesToNumpy:
    """Round-trip correctness for bytes <-> numpy conversions."""

    def test_roundtrip_with_copy(self) -> None:
        """Copied array preserves dtype, layout, and content."""
        arr = bytes_to_numpy(RAW_BYTES, copy=True)
        assert arr.dtype == np.uint8
        assert arr.flags.c_contiguous
        assert bytes(arr) == RAW_BYTES

    def test_roundtrip_without_copy(self) -> None:
        """View array preserves dtype and content but is read-only."""
        arr = bytes_to_numpy(RAW_BYTES, copy=False)
        assert arr.dtype == np.uint8
        assert not arr.flags.writeable
        assert bytes(arr) == RAW_BYTES

    def test_copy_true_owns_memory(self) -> None:
        """copy=True array is independent of the source bytes."""
        arr = bytes_to_numpy(RAW_BYTES, copy=True)
        assert arr.flags.owndata

    def test_copy_false_is_view(self) -> None:
        """copy=False array shares memory with the source bytes."""
        arr = bytes_to_numpy(RAW_BYTES, copy=False)
        assert not arr.flags.owndata


class TestNumpyToBytes:
    """numpy_to_bytes conversion and contiguity guard."""

    def test_contiguous_roundtrip(self, uint8_array: npt.NDArray[np.uint8]) -> None:
        """Contiguous array round-trips to identical bytes."""
        result = numpy_to_bytes(uint8_array)
        assert result == RAW_BYTES

    def test_non_contiguous_auto_corrects(self) -> None:
        """Non-contiguous arrays are auto-corrected with a warning log."""
        arr = np.arange(20, dtype=np.uint8).reshape(4, 5)
        col_view = arr[:, 0]
        assert not col_view.flags.c_contiguous
        result = numpy_to_bytes(col_view)
        np.testing.assert_array_equal(np.frombuffer(result, dtype=np.uint8), col_view)


class TestBufferProtocolCompatibility:
    """Verify buffer-protocol claims from the bytes_transport docstring.

    These tests ensure that numpy uint8 arrays work correctly with stdlib
    APIs that accept buffer-protocol objects, validating the docstring's
    guidance to "prefer passing the array directly over calling
    numpy_to_bytes()".
    """

    def test_io_bytesio_accepts_numpy(self, uint8_array: npt.NDArray[np.uint8]) -> None:
        """io.BytesIO(array) works via buffer protocol."""
        bio = io.BytesIO(uint8_array)
        assert bio.read() == RAW_BYTES

    def test_file_write_accepts_numpy(
        self, uint8_array: npt.NDArray[np.uint8], tmp_path: pytest.TempPathFactory
    ) -> None:
        """file.write(array) works via buffer protocol."""
        path = tmp_path / "test.bin"  # type: ignore[operator]
        with path.open("wb") as f:
            f.write(uint8_array)
        assert path.read_bytes() == RAW_BYTES

    def test_path_write_bytes_accepts_numpy(
        self, uint8_array: npt.NDArray[np.uint8], tmp_path: pytest.TempPathFactory
    ) -> None:
        """Path.write_bytes(array) works via buffer protocol."""
        path = tmp_path / "test.bin"  # type: ignore[operator]
        path.write_bytes(uint8_array)
        assert path.read_bytes() == RAW_BYTES

    def test_subprocess_run_rejects_numpy(self, uint8_array: npt.NDArray[np.uint8]) -> None:
        """subprocess.run(input=array) does NOT work -- CPython bug.

        CPython's subprocess._communicate() checks ``if not input:``
        which triggers numpy's ambiguous truth-value error for
        multi-element arrays.  This test documents the limitation and
        ensures the docstring warning remains accurate.
        """
        with pytest.raises(ValueError, match="truth value of an array"):
            subprocess.run(["cat"], input=uint8_array, capture_output=True, check=True)  # noqa: S607

    def test_subprocess_run_works_with_bytes_conversion(self, uint8_array: npt.NDArray[np.uint8]) -> None:
        """bytes(array) conversion makes it safe for subprocess.run(input=...)."""
        result = subprocess.run(["cat"], input=bytes(uint8_array), capture_output=True, check=True)  # noqa: S607
        assert result.stdout == RAW_BYTES
