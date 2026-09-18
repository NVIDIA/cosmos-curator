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
"""Unit tests for the timestamp-timeline ``.npy`` reader and writer."""

import io
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.utils.timestamps import read_timestamps_ns, write_timestamps_ns

# Three consecutive sensor timestamps from real 30 fps recorder output.
_SAMPLE_NS = np.array([10459540803000, 10459574136000, 10459607470000], dtype=np.int64)


def _npy_bytes(array: npt.NDArray[Any]) -> io.BytesIO:
    """Serialize *array* with plain numpy, bypassing the writer's validation."""
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    buffer.seek(0)
    return buffer


def test_round_trip_preserves_the_array_exactly() -> None:
    """A written artifact reads back as the identical array."""
    buffer = io.BytesIO()
    write_timestamps_ns(_SAMPLE_NS, buffer)
    buffer.seek(0)

    loaded = read_timestamps_ns(buffer)

    assert loaded.dtype == np.int64
    np.testing.assert_array_equal(loaded, _SAMPLE_NS)


def test_read_rejects_a_non_int64_array() -> None:
    """A float array is not this artifact, whatever its values look like."""
    with pytest.raises(ValueError, match="signed 64-bit"):
        read_timestamps_ns(_npy_bytes(_SAMPLE_NS.astype(np.float64)))


def test_read_rejects_a_multidimensional_array() -> None:
    """A timestamp timeline is 1-D; anything else is a different file."""
    with pytest.raises(ValueError, match="1-D"):
        read_timestamps_ns(_npy_bytes(_SAMPLE_NS.reshape(1, 3)))


def test_read_rejects_non_ascending_timestamps() -> None:
    """A timeline that runs backwards cannot be aligned against."""
    with pytest.raises(ValueError, match="ascending"):
        read_timestamps_ns(_npy_bytes(_SAMPLE_NS[::-1].copy()))


def test_a_big_endian_int64_timeline_is_accepted() -> None:
    """``.npy`` preserves byte order, and a big-endian int64 is still an int64.

    This module reads a generic artifact it did not necessarily write, so
    rejecting ``>i8`` would refuse a file that is correct by every property the
    format actually cares about.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.array([100, 200, 300], dtype=">i8"), allow_pickle=False)
    buffer.seek(0)

    timestamps_ns = read_timestamps_ns(buffer)

    # Native order on the way out, whatever the file held: without this a
    # regression that drops the conversion hands callers a ``>i8`` unnoticed.
    assert timestamps_ns.dtype == np.int64
    assert timestamps_ns.tolist() == [100, 200, 300]


def test_an_empty_timeline_is_rejected() -> None:
    """A timeline with no instants is not a timeline.

    ``require_strictly_increasing`` passes vacuously on an empty array, so
    without this the artifact round-trips cleanly and fails later at whatever
    asks it for a first timestamp. The producer that exists today rejects empty
    input, but the contract belongs to the artifact rather than to one producer.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.array([], dtype=np.int64), allow_pickle=False)
    buffer.seek(0)

    with pytest.raises(ValueError, match="at least one"):
        read_timestamps_ns(buffer)

    with pytest.raises(ValueError, match="at least one"):
        write_timestamps_ns(np.array([], dtype=np.int64), io.BytesIO())


def test_read_rejects_duplicate_timestamps() -> None:
    """Two observations cannot share an instant; a duplicate is not a reversal."""
    with pytest.raises(ValueError, match="ascending"):
        read_timestamps_ns(_npy_bytes(np.array([100, 100, 200], dtype=np.int64)))


def test_write_rejects_duplicate_timestamps() -> None:
    """The producer discovers a duplicated instant, not whoever reads it back."""
    with pytest.raises(ValueError, match="ascending"):
        write_timestamps_ns(np.array([100, 100, 200], dtype=np.int64), io.BytesIO())


def test_a_rejected_write_leaves_the_stream_untouched() -> None:
    """The producer discovers a bad timeline, and no bytes reach the stream.

    Validation runs before ``np.save``: writing first and validating second
    would leave a partial artifact in the caller's stream, which is exactly the
    failure validating on write exists to prevent.
    """
    buffer = io.BytesIO()

    with pytest.raises(ValueError, match="ascending"):
        write_timestamps_ns(_SAMPLE_NS[::-1].copy(), buffer)

    assert buffer.getvalue() == b""


def test_read_rejects_a_truncated_artifact() -> None:
    """A failed or partial upload is a bad artifact, not an unhandled ``EOFError``."""
    with pytest.raises(ValueError, match="not a valid"):
        read_timestamps_ns(io.BytesIO(b""))


def test_read_rejects_an_npz_archive() -> None:
    """``np.savez`` output loads as a lazy archive, never as this artifact."""
    buffer = io.BytesIO()
    np.savez(buffer, timestamps_ns=_SAMPLE_NS)
    buffer.seek(0)

    with pytest.raises(ValueError, match="single array"):
        read_timestamps_ns(buffer)


def test_read_rejects_an_object_array() -> None:
    """``allow_pickle=False`` keeps a pickled payload from executing on load."""
    buffer = io.BytesIO()
    np.save(buffer, np.array([object()], dtype=object), allow_pickle=True)
    buffer.seek(0)

    with pytest.raises(ValueError, match="not a valid"):
        read_timestamps_ns(buffer)


def test_read_rejects_a_non_seekable_stream() -> None:
    """``np.load`` sniffs the header by seeking, so a pipe cannot be read."""

    class _Unseekable(io.BytesIO):
        """A readable stream carrying a real artifact that cannot seek back."""

        def seekable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="seekable"):
        read_timestamps_ns(_Unseekable(_npy_bytes(_SAMPLE_NS).getvalue()))


def test_read_lets_a_transient_io_error_through() -> None:
    """A dropped connection is retryable; a corrupt artifact is not.

    Collapsing both into ``ValueError`` would let a caller that skips or deletes
    on a bad artifact silently discard a file that was fine and would have read
    on the next attempt.
    """

    class _DroppedConnection(io.BytesIO):
        def read(self, size: int | None = -1, /) -> bytes:  # noqa: ARG002
            raise ConnectionResetError(104, "Connection reset by peer")

    with pytest.raises(ConnectionResetError):
        read_timestamps_ns(_DroppedConnection(_npy_bytes(_SAMPLE_NS).getvalue()))


def test_write_rejects_a_non_int64_array() -> None:
    """The write path checks dtype itself, not only by way of the reader."""
    with pytest.raises(ValueError, match="signed 64-bit"):
        write_timestamps_ns(_SAMPLE_NS.astype(np.float64), io.BytesIO())  # type: ignore[arg-type]


def test_read_rejects_bytes_after_the_array() -> None:
    """A timeline is one array and nothing else, so anything following it is not one.

    ``np.load`` stops at the end of the first array and never looks further, so a
    file with a second array appended, or with trailing bytes from a partial
    overwrite, reads as valid and returns only its first half.
    """
    stream = io.BytesIO()
    write_timestamps_ns(np.array([1, 2, 3], dtype=np.int64), stream)
    np.save(stream, np.array([4, 5, 6], dtype=np.int64))
    stream.seek(0)

    with pytest.raises(ValueError, match="trailing"):
        read_timestamps_ns(stream)


def test_read_rejects_trailing_junk() -> None:
    """Truncated overwrites leave bytes that are not an array at all."""
    stream = io.BytesIO()
    write_timestamps_ns(np.array([1, 2, 3], dtype=np.int64), stream)
    stream.write(b"\x00\x01\x02")
    stream.seek(0)

    with pytest.raises(ValueError, match="trailing"):
        read_timestamps_ns(stream)
