# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed decoding matches a container open of the same file, including container-only stream properties."""

import io
import struct
from collections.abc import Callable
from itertools import islice
from pathlib import Path

import av
import numpy as np
import numpy.typing as npt
import pytest

from cosmos_curator.core.sensors.mp4.box_primitives import find_box, iter_child_boxes, read_box_header
from cosmos_curator.core.sensors.mp4.indexing import Mp4Indexer, index_video
from cosmos_curator.core.sensors.utils import video
from cosmos_curator.core.sensors.utils.video import (
    StreamingVideoDecoder,
    execute_streaming_decode_plan,
    make_streaming_decode_plan,
)
from tests.cosmos_curator.core.sensors.mp4 import _real_fixtures as fixtures

type DecodedFrame = tuple[int | None, npt.NDArray[np.uint8], list[tuple[str, bytes]]]

_PACKETS = 12
_FULL_RANGE_COLR = b"nclx" + struct.pack(">HHHB", 2, 2, 2, 0x80)
_MASTERING_DISPLAY = struct.pack(">8H2I", 13250, 34500, 7500, 3000, 34000, 16000, 15635, 16450, 10_000_000, 50)
_CONTENT_LIGHT_LEVEL = struct.pack(">HH", 1000, 400)
_ROTATE_90 = (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30)
#: Full range is a frame attribute rather than side data; this names it among the reported properties.
_FULL_RANGE = "FULL_RANGE"


def _remux(clip: Path, **muxer_options: str) -> bytes:
    """Remux the first H.264 packets of *clip* to MP4, writing no color, HDR, or rotation boxes."""
    buffer = io.BytesIO()
    with av.open(str(clip)) as source, av.open(buffer, "w", format="mp4", options=muxer_options) as sink:
        selected = source.streams.video[0]
        output = sink.add_stream_from_template(selected)
        for packet in islice((p for p in source.demux(selected) if p.dts is not None), _PACKETS):
            packet.stream = output
            sink.mux(packet)
    return buffer.getvalue()


def _with_sample_entry_box(data: bytes, box_type: bytes, payload: bytes) -> bytes:
    """Append a box to the first video sample entry, growing every enclosing box.

    Offsets past ``moov`` must be relative, as with a trailing ``moov`` or ``default-base-is-moof`` fragments.
    """
    headers = []
    start, end = 0, len(data)
    for name in ("moov", "trak", "mdia", "minf", "stbl", "stsd"):
        start, end = find_box(data, (name,), start, end)
        headers.append(start - 8)
    entry_start = start + 8
    headers.append(entry_start)
    _, entry_size, _ = read_box_header(data, entry_start, end)
    box = struct.pack(">I4s", len(payload) + 8, box_type) + payload
    patched = bytearray(data[: entry_start + entry_size] + box + data[entry_start + entry_size :])
    for header in headers:
        struct.pack_into(">I", patched, header, struct.unpack_from(">I", data, header)[0] + len(box))
    return bytes(patched)


def _fragmented(clip: Path) -> bytes:
    """Remux *clip* as several ``sidx``-indexed fragments, with no ``mfra`` for a ``moov`` edit to invalidate."""
    data = _remux(clip, movflags="frag_keyframe+empty_moov+default_base_moof+dash+skip_trailer", frag_duration="250000")
    top_level = [box_type for box_type, _, _ in iter_child_boxes(data, 0, len(data))]
    assert top_level.count("moof") > 1
    assert top_level.count("sidx") == top_level.count("moof")
    assert "mfra" not in top_level
    return data


def _rotated(data: bytes) -> bytes:
    """Set a 90-degree display matrix in the version-0 ``tkhd``."""
    patched = bytearray(data)
    tkhd_start, _ = find_box(patched, ("moov", "trak", "tkhd"), 0, len(patched))
    assert patched[tkhd_start] == 0
    struct.pack_into(">9i", patched, tkhd_start + 40, *_ROTATE_90)
    return bytes(patched)


def _full_range(data: bytes) -> bytes:
    return _with_sample_entry_box(data, b"colr", _FULL_RANGE_COLR)


def _hdr(data: bytes) -> bytes:
    return _with_sample_entry_box(
        _with_sample_entry_box(data, b"mdcv", _MASTERING_DISPLAY), b"clli", _CONTENT_LIGHT_LEVEL
    )


def _uncompressed_rgb_mov() -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="mov") as sink:
        stream = sink.add_stream("rawvideo", rate=24)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "rgb24"
        pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        for frame_idx in range(6):
            frame = av.VideoFrame.from_ndarray(pixels + frame_idx, format="rgb24")
            frame.pts = frame_idx
            sink.mux(stream.encode(frame))
        sink.mux(stream.encode(None))
    return buffer.getvalue()


_BOTH = tuple(Mp4Indexer)
_PYAV = (Mp4Indexer.PYAV,)
#: ``(fixture, indexers that accept it, properties the container open must report)``
_CASES: dict[str, tuple[Callable[[], bytes], tuple[Mp4Indexer, ...], frozenset[str]]] = {
    "full_range": (lambda: _full_range(_remux(fixtures.CLIP)), _BOTH, frozenset({_FULL_RANGE})),
    "rotation": (lambda: _rotated(_remux(fixtures.CLIP)), _BOTH, frozenset({"DISPLAYMATRIX"})),
    "hdr": (
        lambda: _hdr(_remux(fixtures.CLIP)),
        _BOTH,
        frozenset({"MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL"}),
    ),
    "bframes": (
        lambda: _rotated(_full_range(_remux(fixtures.BFRAME_CLIP))),
        _BOTH,
        frozenset({_FULL_RANGE, "DISPLAYMATRIX"}),
    ),
    "uncompressed_rgb": (_uncompressed_rgb_mov, _PYAV, frozenset()),
    "fragmented": (
        lambda: _rotated(_hdr(_full_range(_fragmented(fixtures.CLIP)))),
        _BOTH,
        frozenset({_FULL_RANGE, "DISPLAYMATRIX", "MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL"}),
    ),
    "fragmented_bframes": (
        lambda: _rotated(_hdr(_full_range(_fragmented(fixtures.BFRAME_CLIP)))),
        _PYAV,
        frozenset({_FULL_RANGE, "DISPLAYMATRIX", "MASTERING_DISPLAY_METADATA", "CONTENT_LIGHT_LEVEL"}),
    ),
}


def _decoded(frame: av.VideoFrame, pixels: npt.NDArray[np.uint8]) -> DecodedFrame:
    return frame.pts, pixels, [(item.type.name, bytes(item)) for item in frame.side_data]


def _properties(frame: av.VideoFrame) -> set[str]:
    return {item.type.name for item in frame.side_data} | ({_FULL_RANGE} if frame.color_range == 2 else set())


def _decode_container(data: bytes) -> tuple[list[DecodedFrame], set[str]]:
    """Decode every frame with ``av.open()`` on the whole file, leaving timestamps unedited as indexes do."""
    with av.open(io.BytesIO(data), options={"ignore_editlist": "1"}) as container:
        frames = list(container.decode(video=0))
    return [_decoded(frame, frame.to_ndarray(format="rgb24")) for frame in frames], _properties(frames[0])


def _decode_indexed(
    monkeypatch: pytest.MonkeyPatch, data: bytes, indexer: Mp4Indexer, *, supply_moov: bool
) -> list[DecodedFrame]:
    """Decode every frame from the index, in presentation order, keeping each output frame's annotations.

    With *supply_moov*, the decoder is given the moov rather than reading it from the source.
    """
    indexed_video = index_video(data, indexer=indexer)
    index = indexed_video.frame_index
    frames: dict[int, av.VideoFrame] = {}
    next_frame = video._next_streaming_frame

    def record(state: StreamingVideoDecoder) -> av.VideoFrame | None:
        frame = next_frame(state)
        if frame is not None:
            frames[frame.opaque] = frame
        return frame

    monkeypatch.setattr(video, "_next_streaming_frame", record)
    moov_bytes = indexed_video.moov_bytes if supply_moov else None
    decoder = StreamingVideoDecoder(data, indexed_video.metadata, index, moov_bytes=moov_bytes)
    plan = make_streaming_decode_plan(index, np.arange(len(index), dtype=np.int64))
    rows = [
        _decoded(frames[int(index.pts_sort_order[position])], pixels)
        for position, pixels in execute_streaming_decode_plan(decoder, plan)
    ]
    decoder.close()
    return rows


@pytest.mark.parametrize("supply_moov", [True, False], ids=["supplied_moov", "source_moov"])
@pytest.mark.parametrize(
    ("case", "indexer"),
    [
        pytest.param(case, indexer, id=f"{case}-{indexer.value}")
        for case, (_, idx, _) in _CASES.items()
        for indexer in idx
    ],
)
def test_indexed_decode_matches_a_container_open(
    monkeypatch: pytest.MonkeyPatch, case: str, indexer: Mp4Indexer, *, supply_moov: bool
) -> None:
    """Every frame has the container decode's PTS, RGB pixels, and side data, set up from the moov alone."""
    build, _, properties = _CASES[case]
    data = build()
    expected, reported = _decode_container(data)
    assert properties <= reported

    actual = _decode_indexed(monkeypatch, data, indexer, supply_moov=supply_moov)

    assert len(actual) == len(expected)
    for (pts, pixels, side_data), (expected_pts, expected_pixels, expected_side_data) in zip(
        actual, expected, strict=True
    ):
        assert pts == expected_pts
        np.testing.assert_array_equal(pixels, expected_pixels)
        assert side_data == expected_side_data
