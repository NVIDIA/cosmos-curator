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
"""Tests for :mod:`cosmos_curator.core.utils.storage.block_cache_reader`."""

import io
import pathlib
import random
from typing import BinaryIO, cast

import attrs
import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor
from cosmos_curator.core.sensors.types.types import DataSource, VideoIndexCreationMethod
from cosmos_curator.core.sensors.utils.video import make_index_and_metadata
from cosmos_curator.core.utils.storage.block_cache_reader import BlockCacheReader

_CLIP = pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data" / "test_clip_10s.mp4"


class _CountingStream(io.BufferedIOBase):
    """In-memory object that records what reaches it.

    A seek that moves the position stands for one ``GetObject`` -- the cost model of a
    cloud reader that reopens its response on every reposition -- on top of the one
    request that opened it. ``clamp_seeks`` stops a seek at the end of the object, as
    ``smart_open`` does, instead of past it, as ``BytesIO`` does. The next ``fail_reads``
    reads raise ``error``.
    """

    def __init__(self, data: bytes, *, fail_reads: int = 0, clamp_seeks: bool = False) -> None:
        super().__init__()
        self._data = io.BytesIO(data)
        self._size = len(data)
        self.fail_reads = fail_reads
        self._clamp_seeks = clamp_seeks
        self.error = OSError("connection reset")
        self.name = "object.mp4"
        self.seek_calls: list[tuple[int, int]] = []
        self.reads = 0
        self.bytes_read = 0
        self.gets = 1

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._data.tell()

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.seek_calls.append((offset, whence))
        before = self._data.tell()
        position = self._data.seek(offset, whence)
        if self._clamp_seeks:
            position = self._data.seek(min(position, self._size))
        self.gets += position != before
        return position

    def read(self, size: int | None = -1) -> bytes:
        self.reads += 1
        if self.fail_reads:
            self.fail_reads -= 1
            self._data.read(1)  # a failed transfer can leave the position anywhere
            raise self.error
        data = self._data.read(size)
        self.bytes_read += len(data)
        return data

    read1 = read


def _reader(stream: _CountingStream, **kwargs: int) -> BlockCacheReader:
    return BlockCacheReader(cast("BinaryIO", stream), **kwargs)


@pytest.mark.parametrize(
    ("size", "block_size", "max_blocks"),
    [
        pytest.param(10_007, 64, 3, id="ragged-last-block"),
        pytest.param(3_200, 64, 2, id="eof-on-a-block-boundary"),
        pytest.param(100, 4096, 4, id="smaller-than-one-block"),
        pytest.param(0, 64, 1, id="empty-object"),
    ],
)
@pytest.mark.parametrize(
    "skip_threshold",
    [
        pytest.param(0, id="always-seek"),
        pytest.param(200, id="read-through-short-gaps"),
        pytest.param(1 << 20, id="read-through-every-gap"),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_access_matches_bytesio(
    size: int, block_size: int, max_blocks: int, skip_threshold: int, seed: int
) -> None:
    """Every read, seek and tell must agree with an in-memory file over the same bytes."""
    rng = random.Random(seed)  # noqa: S311 (deterministic test fixture, not cryptography)
    data = rng.randbytes(size)
    expected = io.BytesIO(data)
    reader = _reader(_CountingStream(data), block_size=block_size, max_blocks=max_blocks, skip_threshold=skip_threshold)

    for _ in range(2_000):
        op = rng.choice(["seek", "read", "read_all", "read1", "readinto", "tell"])
        n = rng.choice([0, 1, block_size - 1, block_size, block_size + 1, rng.randrange(1, 4 * block_size)])
        if op == "seek":
            whence = rng.choice([io.SEEK_SET, io.SEEK_CUR, io.SEEK_END])
            base = (0, expected.tell(), size)[whence]
            offset = rng.randrange(0, size + 2 * block_size) - base
            assert reader.seek(offset, whence) == expected.seek(offset, whence)
        elif op == "read":
            assert reader.read(n) == expected.read(n)
        elif op == "read_all":
            assert reader.read(rng.choice([-1, None])) == expected.read()
        elif op == "read1":
            got = reader.read1(n)
            assert got == expected.read(len(got))
            assert got or n == 0 or expected.tell() >= size
        elif op == "readinto":
            buffer, want = bytearray(n), bytearray(n)
            assert reader.readinto(buffer) == expected.readinto(want)
            assert buffer == want
        assert reader.tell() == expected.tell()


def test_size_probe_ping_pong_touches_the_wrapped_stream_once() -> None:
    """Libav's ``seek(-1, SEEK_END)`` and seek-back per fragment must not reach the network."""
    size = 16_384
    stream = _CountingStream(bytes(size))
    reader = _reader(stream, block_size=1024)
    reader.read(16)

    for position in range(0, size, 100):
        assert reader.seek(-1, io.SEEK_END) == size - 1
        assert reader.seek(position) == position

    assert stream.seek_calls == [(0, io.SEEK_END)]


def test_size_learned_from_a_short_read_makes_size_probes_free() -> None:
    """An object that ends inside a fetched block needs no separate size lookup."""
    stream = _CountingStream(b"x" * 100)
    reader = _reader(stream, block_size=1024)
    reader.read(1)

    for _ in range(1_000):
        assert reader.seek(-1, io.SEEK_END) == 99
        reader.seek(0)

    assert stream.seek_calls == []


def test_sequential_misses_stream_from_one_request() -> None:
    """A scan across many blocks must not reposition the wrapped stream between them."""
    stream = _CountingStream(b"y" * 64 * 50)
    reader = _reader(stream, block_size=64, max_blocks=2)

    while reader.read(10):
        pass

    assert stream.seek_calls == []
    assert stream.gets == 1


def test_a_forward_gap_within_the_threshold_is_read_through() -> None:
    """Skipping a short stretch reuses the open response instead of paying a round trip."""
    data = bytes(range(256)) * 4
    stream = _CountingStream(data)
    reader = _reader(stream, block_size=64, skip_threshold=192)
    reader.read(1)

    reader.seek(256)
    assert reader.read(4) == data[256:260]

    assert stream.seek_calls == []
    assert stream.bytes_read == 320


@pytest.mark.parametrize(
    ("skip_threshold", "target"),
    [pytest.param(128, 256, id="gap-wider-than-threshold"), pytest.param(0, 128, id="read-through-disabled")],
)
def test_a_forward_gap_beyond_the_threshold_is_sought_over(skip_threshold: int, target: int) -> None:
    """A long stretch -- an ``mdat`` between ``moof`` boxes -- is skipped rather than downloaded."""
    data = bytes(range(256)) * 4
    stream = _CountingStream(data)
    reader = _reader(stream, block_size=64, skip_threshold=skip_threshold)
    reader.read(1)

    reader.seek(target)
    assert reader.read(4) == data[target : target + 4]

    assert stream.seek_calls == [(target, io.SEEK_SET)]
    assert stream.bytes_read == 128


def test_a_backward_jump_seeks_whatever_the_threshold() -> None:
    """Only a forward gap can be read through."""
    data = bytes(range(256))
    stream = _CountingStream(data)
    reader = _reader(stream, block_size=64, max_blocks=1, skip_threshold=1 << 20)
    reader.seek(128)
    reader.read(1)

    reader.seek(0)
    assert reader.read(4) == data[:4]

    assert stream.seek_calls == [(0, io.SEEK_SET)]


def test_reading_through_to_the_end_learns_the_size() -> None:
    """A read-through that runs out of object ends the read and makes size probes free."""
    stream = _CountingStream(b"z" * 100)
    reader = _reader(stream, block_size=64, skip_threshold=1024)

    reader.seek(200)
    assert reader.read(4) == b""
    assert reader.seek(-1, io.SEEK_END) == 99
    assert reader.read() == b"z"

    assert stream.seek_calls == [(64, io.SEEK_SET)]


def test_reading_to_an_end_on_a_block_boundary_learns_the_size() -> None:
    """An empty fetch after the last full block ends the object as surely as a short one."""
    data = bytes(range(256))
    stream = _CountingStream(data)
    reader = _reader(stream, block_size=64)
    assert reader.read() == data
    reads = stream.reads

    assert reader.read(1) == b""
    assert reader.read() == b""
    assert reader.seek(-1, io.SEEK_END) == 255
    assert reader.read() == data[-1:]
    reader.seek(1_000)
    assert reader.readinto(bytearray(4)) == 0

    assert stream.seek_calls == []
    assert stream.reads == reads


@pytest.mark.parametrize(
    "clamp_seeks", [pytest.param(True, id="seek-stops-at-end"), pytest.param(False, id="seek-lands-past-end")]
)
def test_a_seek_past_the_end_learns_the_true_size(*, clamp_seeks: bool) -> None:
    """Where the wrapped stream lands after seeking past the end is not taken as the size."""
    data = bytes(range(256))
    stream = _CountingStream(data, clamp_seeks=clamp_seeks)
    reader = _reader(stream, block_size=64, skip_threshold=64)

    for position in (512, 576):
        reader.seek(position)
        assert reader.read(4) == b""
    assert reader.seek(-1, io.SEEK_END) == 255
    assert reader.read() == data[-1:]


def test_the_least_recently_used_block_is_evicted() -> None:
    """A re-touched block survives; the one untouched longest is fetched again."""
    stream = _CountingStream(bytes(64 * 4))
    reader = _reader(stream, block_size=64, max_blocks=2)
    for block in (0, 1, 0, 2):
        reader.seek(block * 64)
        reader.read(1)
    reads = stream.reads

    reader.seek(0)
    reader.read(1)
    assert stream.reads == reads

    reader.seek(64)
    reader.read(1)
    assert stream.reads == reads + 1


def test_a_failed_fetch_propagates_unchanged_and_a_retry_re_seeks() -> None:
    """The wrapped stream's error keeps its identity, and its unknown position is not trusted."""
    data = bytes(range(200))
    stream = _CountingStream(data, fail_reads=1)
    reader = _reader(stream, block_size=64)

    with pytest.raises(OSError, match="connection reset") as excinfo:
        reader.read(10)

    assert excinfo.value is stream.error
    assert reader.read(10) == data[:10]
    assert stream.seek_calls == [(0, io.SEEK_SET)]


def test_a_failed_read_through_propagates_unchanged_and_a_retry_re_seeks() -> None:
    """A gap that fails part-way leaves the wrapped stream's position unknown, as a failed fetch does."""
    data = bytes(range(256))
    stream = _CountingStream(data)
    reader = _reader(stream, block_size=64, skip_threshold=128)
    reader.read(1)
    stream.fail_reads = 1

    reader.seek(192)
    with pytest.raises(OSError, match="connection reset") as excinfo:
        reader.read(4)

    assert excinfo.value is stream.error
    assert reader.read(4) == data[192:196]
    assert stream.seek_calls == [(192, io.SEEK_SET)]


@pytest.mark.parametrize(
    ("offset", "whence", "match"),
    [
        pytest.param(-1, io.SEEK_SET, "negative seek position -1", id="before-start"),
        pytest.param(-11, io.SEEK_END, "negative seek position -1", id="before-start-from-end"),
        pytest.param(0, 3, "invalid whence", id="bad-whence"),
    ],
)
def test_invalid_seeks_raise_and_leave_the_position_alone(offset: int, whence: int, match: str) -> None:
    """Out-of-range targets are refused, as a regular file refuses them."""
    reader = _reader(_CountingStream(b"0123456789"))
    reader.seek(4)

    with pytest.raises(ValueError, match=match):
        reader.seek(offset, whence)

    assert reader.tell() == 4


def test_reading_past_the_end_returns_empty() -> None:
    """A position beyond the object reads as end of file."""
    reader = _reader(_CountingStream(b"0123456789"))

    assert reader.seek(1_000) == 1_000
    assert reader.read(5) == b""
    assert reader.read() == b""
    assert reader.read1(5) == b""
    assert reader.readinto(bytearray(5)) == 0


def test_close_keeps_the_wrapped_stream_open() -> None:
    """The wrapped stream belongs to whoever opened it."""
    stream = _CountingStream(b"abc")
    with _reader(stream) as reader:
        assert reader.name == "object.mp4"
        assert reader.readable()
        assert reader.seekable()

    assert reader.closed
    assert not stream.closed
    with pytest.raises(ValueError, match="closed file"):
        reader.read()


@pytest.mark.parametrize(("block_size", "max_blocks"), [(0, 4), (1024, 0)])
def test_non_positive_geometry_is_rejected(block_size: int, max_blocks: int) -> None:
    """A zero-sized block or cache could never serve a read."""
    with pytest.raises(ValueError, match="must be positive"):
        _reader(_CountingStream(b""), block_size=block_size, max_blocks=max_blocks)


def test_a_negative_skip_threshold_is_rejected() -> None:
    """No gap is shorter than zero bytes."""
    with pytest.raises(ValueError, match="must be non-negative"):
        _reader(_CountingStream(b""), skip_threshold=-1)


def _remux(destination: pathlib.Path, options: dict[str, str]) -> bytes:
    with av.open(str(_CLIP)) as source, av.open(str(destination), mode="w", options=options) as sink:
        stream = source.streams.video[0]
        output = sink.add_stream_from_template(stream)
        for packet in source.demux(stream):
            if packet.dts is None:
                continue
            packet.stream = output
            sink.mux(packet)
    return destination.read_bytes()


@pytest.fixture(scope="module")
def fragmented_mp4(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """Remux the reference clip into one ``moof``/``mdat`` fragment per frame."""
    return _remux(
        tmp_path_factory.mktemp("fmp4") / "fragmented.mp4",
        {"movflags": "frag_every_frame+empty_moov+default_base_moof"},
    )


@pytest.mark.parametrize(
    ("block_size", "max_blocks"),
    [pytest.param(None, None, id="defaults"), pytest.param(16 * 1024, 2, id="small-blocks")],
)
def test_indexing_a_fragmented_mp4_costs_a_bounded_number_of_requests(
    fragmented_mp4: bytes,
    block_size: int | None,
    max_blocks: int | None,
) -> None:
    """Libav's per-fragment size probes must not each become a request.

    Indexing a fragmented MP4 walks every fragment and asks for the file size several
    times per fragment; straight onto a request-per-seek stream that is thousands of
    requests for a 240-frame clip.
    """
    direct = _CountingStream(fragmented_mp4)
    expected = CameraSensor(direct).video_index

    cached = _CountingStream(fragmented_mp4)
    geometry = {} if block_size is None or max_blocks is None else {"block_size": block_size, "max_blocks": max_blocks}
    actual = CameraSensor(_reader(cached, **geometry)).video_index

    for want, got in zip(attrs.astuple(expected, recurse=False), attrs.astuple(actual, recurse=False), strict=True):
        np.testing.assert_array_equal(got, want)
    assert direct.gets > 1_000
    assert cached.gets <= 5


def test_indexing_coarse_fragments_skips_their_media_data(tmp_path: pathlib.Path) -> None:
    """Fragments wider than the skip threshold must be sought over, not downloaded.

    Two-second fragments of the reference clip are ~350 KB, so the ``mdat`` between one
    ``moof`` and the next is a gap the default threshold refuses to read through. The
    header index is built directly because ``AUTO``'s composition probe would demux this
    short clip end to end.
    """
    data = _remux(
        tmp_path / "coarse.mp4",
        {"movflags": "empty_moov+default_base_moof", "frag_duration": "2000000"},
    )
    expected, _ = make_index_and_metadata(
        cast("DataSource", _CountingStream(data)), index_method=VideoIndexCreationMethod.FROM_HEADER
    )

    cached = _CountingStream(data)
    actual, _ = make_index_and_metadata(
        cast("DataSource", _reader(cached)), index_method=VideoIndexCreationMethod.FROM_HEADER
    )

    assert actual == expected
    assert cached.bytes_read < 0.75 * len(data)
    assert cached.gets <= 10
