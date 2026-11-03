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
"""Tests for laying out extracted boxes as a file libav walks the way it walks the source."""

import io
import struct

import av
import pytest

from cosmos_curator.core.sensors.mp4.box_layout import BoxLayoutUnsupportedError, boxes_only_stream
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as real_mp4


def test_boxes_only_stream_places_moov_first_each_moof_at_its_source_offset_and_zeros_between() -> None:
    """Filler box headers span every gap, and the media they hold is zeros."""
    data = real_mp4.fragmented_clip()
    file_size, moov, moofs = len(data), real_mp4.moov_box(data), real_mp4.moof_rows(data)

    laid_out = boxes_only_stream(file_size, moov, moofs).read()

    assert len(laid_out) == file_size
    assert laid_out[: len(moov)] == moov
    assert laid_out[len(moov) + 4 : len(moov) + 8] == b"free"
    for start, moof_bytes in moofs:
        assert laid_out[start : start + len(moof_bytes)] == moof_bytes
        mdat_start = start + len(moof_bytes)
        assert laid_out[mdat_start + 4 : mdat_start + 8] == b"mdat"
    first, second = moofs[0], moofs[1]
    media = laid_out[first[0] + len(first[1]) + 8 : second[0]]
    assert media
    assert not media.strip(b"\0")


def test_boxes_only_stream_seeks_and_reads_across_box_boundaries() -> None:
    """A read spanning zeros and a box returns both, and a read past the end returns nothing."""
    moov = b"\x00\x00\x00\x10moov" + b"m" * 8
    moof = b"\x00\x00\x00\x0cmoof" + b"f" * 4
    stream = boxes_only_stream(64, moov, [(32, moof)])

    stream.seek(28)
    assert stream.read(8) == b"\0\0\0\0" + moof[:4]
    assert stream.seek(-4, 2) == 60
    assert stream.read() == b"\0\0\0\0"
    assert stream.read(1) == b""


@pytest.mark.parametrize(
    ("moof_rows", "match"),
    [
        ([(20, b"\x00\x00\x00\x08moof")], "moov \\(24 bytes\\) cannot fit before the first moof"),
        ([(24, b"\x00\x00\x00\x08moof"), (28, b"\x00\x00\x00\x08moof")], "overlaps the next box"),
        ([(28, b"\x00\x00\x00\x08moof"), (40, b"\x00\x00\x00\x08moof")], "4-byte gap"),
        ([(48, b"\x00\x00\x00\x0cmoof" + b"f" * 4)], "4-byte gap"),
    ],
)
def test_boxes_only_stream_refuses_boxes_it_cannot_lay_out_as_libav_would_walk_them(
    moof_rows: list[tuple[int, bytes]], match: str
) -> None:
    """Without a box header in every gap, libav would read zeros as a box and walk a different file."""
    moov = b"\x00\x00\x00\x18moov" + b"m" * 16

    with pytest.raises(BoxLayoutUnsupportedError, match=match):
        boxes_only_stream(64, moov, moof_rows)


def test_boxes_only_stream_writes_a_large_size_header_for_a_gap_past_four_gibibytes() -> None:
    """A 32-bit box size cannot span the gap, so the filler box uses the 64-bit form."""
    moov = b"\x00\x00\x00\x08moov"
    stream = boxes_only_stream(8 + (1 << 32) + 16, moov)

    stream.seek(8)
    assert stream.read(16) == struct.pack(">I4sQ", 1, b"free", (1 << 32) + 16)


@pytest.mark.parametrize("with_ftyp", [False, True])
def test_boxes_only_stream_lays_ftyp_ahead_of_moov_so_libav_reads_the_source_brand(*, with_ftyp: bool) -> None:
    """With ftyp, the leading boxes are the source's own bytes; without it, libav reads the file as QuickTime."""
    data = real_mp4.fragmented_clip()
    moov, moofs = real_mp4.moov_box(data), real_mp4.moof_rows(data)
    ftyp = data[: struct.unpack_from(">I", data)[0]]
    assert ftyp[4:8] == b"ftyp"

    laid_out = boxes_only_stream(len(data), moov, moofs, ftyp_bytes=ftyp if with_ftyp else None).read()

    leading = ftyp + moov if with_ftyp else moov
    assert laid_out[: len(leading)] == leading
    with av.open(io.BytesIO(data)) as source, av.open(io.BytesIO(laid_out)) as boxes:
        assert source.metadata["major_brand"]
        assert boxes.metadata.get("major_brand") == (source.metadata["major_brand"] if with_ftyp else None)


def test_boxes_only_stream_refuses_ftyp_and_moov_that_overlap_the_first_moof() -> None:
    """Ftyp and moov together must end at or before the first moof, as they do in the source."""
    moov = b"\x00\x00\x00\x10moov" + b"m" * 8
    ftyp = b"\x00\x00\x00\x10ftypisom\x00\x00\x02\x00"

    with pytest.raises(BoxLayoutUnsupportedError, match=r"ftyp and moov \(32 bytes\) cannot fit"):
        boxes_only_stream(64, moov, [(24, b"\x00\x00\x00\x08moof")], ftyp_bytes=ftyp)
    assert boxes_only_stream(64, moov, [(32, b"\x00\x00\x00\x08moof")], ftyp_bytes=ftyp).read(32) == ftyp + moov
