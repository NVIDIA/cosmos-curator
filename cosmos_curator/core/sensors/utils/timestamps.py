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
"""A timestamp timeline: a 1-D ``int64`` nanosecond ``.npy``.

Both sensor timestamps and generated alignment timestamps reduce to an ascending
sequence of instants. This module owns only their shared representation: one
strictly ascending nanosecond value per row in a plain ``.npy``. The caller owns
the timeline's semantics and clock provenance.

Integer nanoseconds because a ``float64`` carries only 53 bits of mantissa. At
epoch-nanosecond magnitudes consecutive ``float64`` values are 256 ns apart, so
two instants closer together than that land on the same float, and the strictly
ascending rule this format rests on stops holding. ``.npy`` because it is self-describing enough that a reader can
reject a wrongly-typed file rather than reinterpret its bytes.

This module owns only that generic form, and deliberately knows nothing about
which sensor produced it. Deriving one from a recorder's own sidecar format is
the device adapter's job, not the library's.
"""

from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt

from cosmos_curator.core.sensors.utils.validation import require_strictly_increasing

_INT64_BYTES = 8


def _require_timestamps_ns(timestamps_ns: npt.NDArray[Any]) -> None:
    """Raise unless *timestamps_ns* is a 1-D, strictly ascending ``int64`` array."""
    if timestamps_ns.ndim != 1:
        msg = f"timestamps_ns must be 1-D, got ndim={timestamps_ns.ndim}"
        raise ValueError(msg)
    # Compared by kind and width rather than against np.int64, because ``.npy``
    # preserves byte order: a big-endian ``>i8`` timeline is correct by every
    # property this format cares about, and equality would refuse it.
    if timestamps_ns.dtype.kind != "i" or timestamps_ns.dtype.itemsize != _INT64_BYTES:
        msg = f"timestamps_ns must be a signed 64-bit integer array, got dtype {timestamps_ns.dtype}"
        raise ValueError(msg)
    if timestamps_ns.size == 0:
        # require_strictly_increasing passes vacuously on an empty array, so
        # without this an empty artifact round-trips and fails later, at whatever
        # asks it for a first timestamp.
        msg = "timestamps_ns must contain at least one entry"
        raise ValueError(msg)
    require_strictly_increasing("timestamps_ns", timestamps_ns)


def _load_single_array(stream: BinaryIO) -> npt.NDArray[Any]:
    """Load one ``.npy`` array from *stream*, raising ``ValueError`` if it is malformed.

    ``OSError`` is left to propagate rather than folded in: a reset connection or
    a failing disk means the bytes were never seen, so the file usually reads on
    a retry, and a caller that discards bad artifacts would throw away a good one.
    """
    try:
        loaded = np.load(stream, allow_pickle=False)
    except (EOFError, ValueError) as exc:
        msg = f"stream is not a valid .npy timeline: {exc}"
        raise ValueError(msg) from exc
    if isinstance(loaded, np.ndarray):
        # np.load stops at the end of the first array, so a second array appended
        # to the file, or the tail of a longer timeline a shorter one was written
        # over, reads as valid and returns only what came first.
        if stream.read(1):
            msg = "stream has trailing bytes after the timeline: a timeline holds one array and nothing else"
            raise ValueError(msg)
        return loaded
    # np.load returns np.savez output rather than raising over it. Closing
    # releases the ZipFile it opened; the caller's own handle stays open, because
    # numpy sets own_fid=False when handed a stream.
    loaded.close()
    msg = f"a timeline holds a single array, got {type(loaded).__name__} (a .npz archive)"
    raise ValueError(msg)


def read_timestamps_ns(stream: BinaryIO) -> npt.NDArray[np.int64]:
    """Read a timestamp timeline from an open binary *stream*.

    Takes an already-open stream rather than a path or a URI: callers own their
    transport, so a local file, an object pulled from a bucket through
    ``smart_open``, and bytes already in memory all arrive here identically.
    ``smart_open.open`` returns a seekable stream for a local path, ``file://``
    and ``s3://`` alike, so one call covers every backend and no caller needs to
    branch on scheme.

    Stream only, with no ``Path`` arm. The sensor entry points still take a
    ``DataSource`` that admits ``Path``, but that arm is being removed from them;
    this is the shape they are moving to, not a narrowing to be reconciled.

    The stream must be seekable, because ``.npy`` is identified by sniffing a
    magic prefix and seeking back over it.

    Raises:
        ValueError: If *stream* is not seekable, does not hold a single readable
            ``.npy`` array, or holds an array that is not 1-D signed 64-bit in
            strictly ascending order -- a shape or dtype mismatch means the file
            is not this artifact, and a non-ascending array cannot be a timeline.
        OSError: If the stream itself fails while being read. Distinct from
            ``ValueError`` on purpose: this one is worth retrying, and says
            nothing about whether the artifact is any good.

    """
    if not stream.seekable():
        # Mirrors the sensor library's stream contract in ``utils.io``; without
        # it numpy reports a pipe as "No data left in file", which sends the
        # caller looking for a truncated upload that does not exist.
        msg = "buffered binary streams must be seekable"
        raise ValueError(msg)
    timestamps_ns = _load_single_array(stream)
    _require_timestamps_ns(timestamps_ns)
    # Hand back native byte order regardless of how the file was written, so a
    # caller never has to think about it.
    return np.asarray(timestamps_ns, dtype=np.int64)


def write_timestamps_ns(timestamps_ns: npt.NDArray[np.int64], stream: BinaryIO) -> None:
    """Write a timestamp timeline to an open binary *stream*.

    Validated before writing rather than only on read: an artifact that fails
    its own reader is worse than no artifact, because it is discovered by
    whoever consumes it rather than by whoever produced it. Validation therefore
    runs before ``np.save``, so a rejected timeline leaves *stream* untouched
    rather than holding a partial artifact.

    Raises:
        ValueError: If *timestamps_ns* is not 1-D signed 64-bit in strictly
            ascending order.

    """
    _require_timestamps_ns(timestamps_ns)
    np.save(stream, timestamps_ns, allow_pickle=False)
