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
"""Tests for walking a fragmented MP4's top-level boxes and capturing every moof verbatim.

Focus is on the never-parse contract (moof bytes come back byte-for-byte unread, skipping past
mdat/moov/free costs nothing but header parsing), and a read failure surfaces to the consumer
rather than hanging or being swallowed.
"""

import io
import math
import struct
import tracemalloc

import pytest

from cosmos_curator.core.sensors.mp4.moof_walk import _consume_skip, walk_moofs
from tests.cosmos_curator.core.sensors.mp4._box_fixtures import pack_box, pack_moof, pack_traf


class _CountingReader:
    """A ready-to-read in-memory handle, counting ``read()`` calls."""

    def __init__(self, buf: bytes) -> None:
        self._stream = io.BytesIO(buf)
        self.read_call_count = 0

    def read(self, size: int | None = None) -> bytes:
        self.read_call_count += 1
        return self._stream.read(size)


def _build_fragmented_buffer() -> tuple[bytes, list[tuple[int, bytes]]]:
    """Build a synthetic fragmented MP4: ftyp + moov + (moof + mdat) x3, one moof with a uuid sibling, a free box.

    Returns the whole buffer plus the expected ``(moof_start, moof_bytes)`` pairs, computed
    independently of ``walk_moofs`` itself as the buffer is assembled.
    """
    buf = b""
    expected: list[tuple[int, bytes]] = []

    buf += pack_box("ftyp", b"isom")
    buf += pack_box("moov", b"placeholder-not-a-real-moov")

    moof1 = pack_moof()
    expected.append((len(buf), moof1))
    buf += moof1
    buf += pack_box("mdat", b"video-bytes-1" * 50)

    buf += pack_box("free", b"padding")

    moof2 = pack_moof(siblings=pack_box("uuid", b"0123456789abcdef" + b"extension-content"))
    expected.append((len(buf), moof2))
    buf += moof2
    buf += pack_box("mdat", b"video-bytes-2" * 50)

    moof3 = pack_moof(pack_traf(tfhd=b"garbage-\x00\x01\x02", trun=b"\xff\xff\xff"))
    expected.append((len(buf), moof3))
    buf += moof3
    buf += pack_box("mdat", b"video-bytes-3" * 50)

    return buf, expected


def test_walk_moofs_yields_every_moof_byte_for_byte() -> None:
    """Every moof box comes back verbatim, at its correct absolute offset, in file order."""
    buf, expected = _build_fragmented_buffer()
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=len(buf)))

    assert result == expected


def test_walk_moofs_reassembles_a_box_split_across_a_chunk_boundary() -> None:
    """A small chunk_bytes forces the carry-over path, but the yielded moofs are unchanged."""
    buf, expected = _build_fragmented_buffer()
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=10))

    assert result == expected
    assert reader.read_call_count > 1  # confirms the small chunk_bytes actually forced multiple reads


def test_walk_moofs_carries_an_extended_size_header_split_across_a_chunk_boundary() -> None:
    """A 64-bit extended-size box header straddling a chunk boundary is carried over, not a crash.

    Without tracking how many header bytes a box actually needs before deciding whether it's
    fully buffered, a box using the extended (size32 == 1) form whose header lands 8-15 bytes
    into a chunk -- enough for the standard 8-byte check to pass, not enough for the 8 more bytes
    the 64-bit size field itself needs -- crashes with a raw struct.error instead of being carried
    into the next chunk like every other partially-buffered box.
    """
    prefix = pack_box("ftyp", b"isom") + pack_box("moov", b"placeholder")
    content = b"x" * 100
    mdat_ext = struct.pack(">I", 1) + b"mdat" + struct.pack(">Q", 16 + len(content)) + content
    moof = pack_moof()
    buf = prefix + mdat_ext + moof

    # Chunk boundary lands exactly 8 bytes into mdat_ext's 16-byte header -- after size32+type,
    # before the size64 field is fully buffered. The straddle this test exists to force.
    chunk_bytes = len(prefix) + 8
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=chunk_bytes))

    assert result == [(len(prefix) + len(mdat_ext), moof)]


def test_walk_moofs_does_not_mistake_a_to_end_mdat_payload_for_a_sibling_box() -> None:
    """A size-0 ('to end of file') mdat spans to true EOF, not just to the next chunk boundary.

    Demonstrates the real bug this test guards against: without tracking the true file end (only
    ever using len(buf), the in-progress chunk accumulation buffer, as read_box_header's own
    'end'), a size-0 box gets treated as ending at whatever chunk boundary happens to land next --
    and whatever follows in the next chunk, here deliberately a real moof-shaped blob sitting
    inside mdat's own opaque payload, gets misparsed as a sibling box instead of staying part of
    mdat's own content.
    """
    prefix = pack_box("ftyp", b"isom") + pack_box("moov", b"placeholder")
    leading_payload = b"y" * 40
    fake_moof = pack_moof()
    trailing_payload = b"z" * 40
    mdat0_header = struct.pack(">I", 0) + b"mdat"
    buf = prefix + mdat0_header + leading_payload + fake_moof + trailing_payload

    # Chunk boundary lands exactly where the fake moof-shaped bytes begin -- the split this test
    # exists to force.
    chunk_bytes = len(prefix) + len(mdat0_header) + len(leading_payload)
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=chunk_bytes))

    assert result == []


def test_walk_moofs_costs_no_extra_read_beyond_the_sequential_chunk_sweep_plus_one_eof_probe() -> None:
    """Skipping mdat/moov/free costs nothing but header parsing, plus exactly one read to confirm EOF.

    No per-box targeted read happens for a box this walker isn't collecting, unlike
    box_locate.py's own per-box header-probing style -- the read count is exactly the number of
    sequential chunk_bytes-sized strides needed to cover the file, plus one final read (returning
    empty) that confirms true end of stream -- this walker trusts the stream's own EOF signal
    over the caller-supplied file size, which could be stale.
    """
    buf, _ = _build_fragmented_buffer()
    chunk_bytes = 300  # smaller than the buffer, so the sweep actually spans several chunks
    reader = _CountingReader(buf)

    list(walk_moofs(len(buf), reader, chunk_bytes=chunk_bytes))

    assert reader.read_call_count == math.ceil(len(buf) / chunk_bytes) + 1


def test_walk_moofs_stops_cleanly_on_a_truncated_trailing_box() -> None:
    """A box header (or declared content) cut off at EOF is dropped silently, not an error."""
    buf, expected = _build_fragmented_buffer()
    truncated = buf + pack_box("free", b"padding-that-gets-cut-off")[:5]
    reader = _CountingReader(truncated)

    result = list(walk_moofs(len(truncated), reader, chunk_bytes=len(truncated)))

    assert result == expected


def test_consume_skip_passes_fetched_through_unchanged_when_nothing_to_skip() -> None:
    """skip_remaining=0 is a no-op -- fetched comes back untouched, skip_remaining stays 0."""
    fetched, skip_remaining = _consume_skip(b"abcdef", 0)

    assert fetched == b"abcdef"
    assert skip_remaining == 0


def test_consume_skip_drops_a_prefix_shorter_than_fetched() -> None:
    """A skip that finishes mid-chunk drops exactly that many bytes and reports no remainder."""
    fetched, skip_remaining = _consume_skip(b"abcdef", 4)

    assert fetched == b"ef"
    assert skip_remaining == 0


def test_consume_skip_consumes_the_whole_chunk_when_skip_remaining_is_larger() -> None:
    """A skip that outlasts the chunk drops everything and carries the remainder forward."""
    fetched, skip_remaining = _consume_skip(b"abcdef", 10)

    assert fetched == b""
    assert skip_remaining == 4


def test_walk_moofs_skips_a_non_moof_box_spanning_many_chunks_without_buffering_it() -> None:
    """A single mdat far bigger than chunk_bytes is skipped a chunk at a time, not carried in leftover.

    Its payload spans ~50 chunk-sized reads. If it were still carried into ``leftover`` the
    way a ``moof`` legitimately is, this would just be a slower version of the old test --
    the real assertion here is that the moof right after it is still found correctly, and
    that no extra reads happen beyond the exact sequential chunk sweep (the same invariant
    ``test_walk_moofs_costs_no_extra_read_beyond_the_sequential_chunk_sweep_plus_one_eof_probe``
    checks), which would only hold if the skip path is actually being taken.
    """
    chunk_bytes = 16
    prefix = pack_box("ftyp", b"isom") + pack_box("moov", b"placeholder")
    big_mdat = pack_box("mdat", b"x" * (chunk_bytes * 50))
    moof = pack_moof()
    buf = prefix + big_mdat + moof
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=chunk_bytes))

    assert result == [(len(prefix) + len(big_mdat), moof)]
    assert reader.read_call_count == math.ceil(len(buf) / chunk_bytes) + 1


def test_walk_moofs_peak_memory_is_bounded_by_chunk_bytes_not_by_a_large_non_moof_box() -> None:
    """A 2 MiB mdat with a 4 KiB chunk_bytes never drives peak traced memory anywhere near 2 MiB.

    The real regression this guards against: carrying a box this walker never reads into
    ``leftover`` (instead of skipping it) makes memory scale with that box's own size. Measured
    directly (not inferred from read counts, which stay identical either way): the pre-fix
    implementation peaks around 4.2 MiB walking this exact buffer, ~2x the mdat size (the
    accumulating buffer plus its own repeated copies); this one peaks under 50 KiB.
    """
    chunk_bytes = 4096
    mdat_payload_size = 2 * 1024 * 1024
    prefix = pack_box("ftyp", b"isom") + pack_box("moov", b"placeholder")
    big_mdat = pack_box("mdat", b"x" * mdat_payload_size)
    moof = pack_moof()
    buf = prefix + big_mdat + moof
    reader = _CountingReader(buf)

    tracemalloc.start()
    try:
        result = list(walk_moofs(len(buf), reader, chunk_bytes=chunk_bytes))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result == [(len(prefix) + len(big_mdat), moof)]
    assert peak < 50_000  # bounded by chunk_bytes (4096) plus small fixed overhead, not the 2 MiB mdat


def test_walk_moofs_yields_nothing_for_a_non_fragmented_buffer() -> None:
    """A plain ftyp/moov/mdat file with zero moof boxes yields an empty sequence, not an error."""
    buf = pack_box("ftyp", b"isom") + pack_box("moov", b"placeholder") + pack_box("mdat", b"data")
    reader = _CountingReader(buf)

    result = list(walk_moofs(len(buf), reader, chunk_bytes=len(buf)))

    assert result == []


class _FailingStream:
    """A stream whose very first read raises -- simulates a real transport failure mid-walk."""

    def read(self, size: int = -1) -> bytes:  # noqa: ARG002 -- size is part of the file-like contract, unused here
        msg = "simulated read failure"
        raise RuntimeError(msg)


def test_walk_moofs_propagates_a_read_failure() -> None:
    """A stream read failure surfaces to the consumer, not silently swallowed or hung."""
    with pytest.raises(RuntimeError, match="simulated read failure"):
        list(walk_moofs(1000, _FailingStream(), chunk_bytes=100))
