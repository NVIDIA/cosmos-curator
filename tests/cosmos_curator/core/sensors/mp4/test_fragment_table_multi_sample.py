# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Multi-sample fragment indexing against real packets and controlled box layouts.

The checked-in ``data/multi_sample_trun.mp4`` fixture was remuxed from the existing
H.264 fixture with FFmpeg 8.1.2 (the Pixi LGPL build), without re-encoding.
Regenerate it from the repository root::

    pixi run ffmpeg -hide_banner -loglevel error -y \
      -i tests/cosmos_curator/pipelines/video/data/test_clip_10s.mp4 \
      -t 1 -map 0:v:0 -c:v copy /tmp/short.mp4
    pixi run ffmpeg -hide_banner -loglevel error -y \
      -stream_loop 1 -i /tmp/short.mp4 -map 0:v:0 -c:v copy \
      -movflags frag_keyframe+empty_moov \
      tests/cosmos_curator/core/sensors/mp4/data/multi_sample_trun.mp4
"""

import struct
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import pytest

from cosmos_curator.core.sensors.mp4 import fragment_table as ft
from cosmos_curator.core.sensors.mp4.box_primitives import MalformedBoxHeaderError, find_box, read_box_header
from tests.cosmos_curator.core.sensors.mp4 import _box_fixtures as bf


def _fragment_boxes(data: bytes) -> tuple[bytes, list[tuple[int, bytes]], list[int]]:
    moov = b""
    moofs = []
    sample_counts = []
    offset = 0
    while offset < len(data):
        kind, size, header = read_box_header(data, offset, len(data))
        if kind == "moov":
            moov = data[offset : offset + size]
        elif kind == "moof":
            moofs.append((offset, data[offset : offset + size]))
            start, _end = find_box(data, ("traf", "trun"), offset + header, offset + size)
            sample_counts.append(struct.unpack_from(">I", data, start + 4)[0])
        offset += size
    return moov, moofs, sample_counts


def _remux_fixture(movflags: str) -> bytes:
    path = Path(__file__).parent / "data" / "multi_sample_trun.mp4"
    buffer = BytesIO()
    with (
        av.open(str(path)) as source,
        av.open(buffer, "w", format="mp4", options={"movflags": movflags}) as output,
    ):
        stream = source.streams.video[0]
        target = output.add_stream_from_template(stream)
        for packet in source.demux(stream):
            if packet.size:
                packet.stream = target
                output.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "movflags",
    [
        None,
        "frag_keyframe+empty_moov",
        "frag_keyframe+empty_moov+default_base_moof",
        "frag_every_frame+empty_moov",
        "frag_every_frame+empty_moov+default_base_moof",
        "frag_keyframe+delay_moov",
        "frag_keyframe+empty_moov+negative_cts_offsets",
    ],
)
def test_ffmpeg_multi_sample_fragments_match_pyav(movflags: str | None) -> None:
    """Index real GOP fragments exactly as PyAV demuxes them."""
    path = Path(__file__).parent / "data" / "multi_sample_trun.mp4"
    data = path.read_bytes() if movflags is None else _remux_fixture(movflags)
    moov, moofs, sample_counts = _fragment_boxes(data)
    assert sample_counts == ([1] * 48 if movflags and "frag_every_frame" in movflags else [24, 24])

    track = ft.parse_fragment_table(moov, moofs)
    with av.open(BytesIO(data)) as container:
        stream = container.streams.video[0]
        assert stream.time_base is not None
        assert stream.time_base * track.timescale == 1
        packets = [packet for packet in container.demux(stream) if packet.size]
        assert len(packets) == 48
        assert all(packet.pts == packet.dts for packet in packets)
        np.testing.assert_array_equal(track.dts_ticks, [packet.dts for packet in packets])
        np.testing.assert_array_equal(track.offset, [packet.pos for packet in packets])
        np.testing.assert_array_equal(track.size, [packet.size for packet in packets])
        np.testing.assert_array_equal(track.is_keyframe, [packet.is_keyframe for packet in packets])
        assert track.duration == packets[-1].dts - packets[0].dts + packets[-1].duration
        for index, packet in enumerate(packets):
            start = int(track.offset[index])
            assert data[start : start + int(track.size[index])] == bytes(packet)

    with av.open(BytesIO(data)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 48


def test_fragmented_file_with_initial_samples_is_refused() -> None:
    """A mixed moov/moof file must not return a successful but partial index."""
    data = _remux_fixture("frag_keyframe")
    moov, moofs, sample_counts = _fragment_boxes(data)
    start, _end = find_box(moov, ("moov", "trak", "mdia", "minf", "stbl", "stsz"), 0, len(moov))
    assert struct.unpack_from(">I", moov, start + 8)[0] == 24
    assert sample_counts == [24]
    with av.open(BytesIO(data)) as container:
        packets = [packet for packet in container.demux(video=0) if packet.size]
        assert len(packets) == 48
        assert sum(packet.duration for packet in packets) * container.streams.video[0].time_base == 2
    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="initial samples"):
        ft.parse_fragment_table(moov, moofs)


@pytest.mark.parametrize("name", ["stts", "stsz", "stz2", "stsc", "stco", "co64"])
@pytest.mark.parametrize("count", [0, 1])
def test_initial_sample_tables_must_be_empty(name: str, count: int) -> None:
    """Allow absent or empty initial tables, but refuse populated ones."""
    entries = {
        "stts": struct.pack(">II", 1, 10),
        "stsc": struct.pack(">III", 1, 1, 1),
        "stco": struct.pack(">I", 1000),
        "co64": struct.pack(">Q", 1000),
        "stsz": b"",
        "stz2": b"\x14",
    }
    prefix = struct.pack(">I", 20) if name == "stsz" else b"\x00\x00\x00\x08" if name == "stz2" else b""
    table = bf.pack_full_box(name, 0, prefix + struct.pack(">I", count) + (entries[name] if count else b""))
    moov = bf.synthetic_fragmented_moov(stbl_extra_children=table)
    fragments = [_fragment(_trun())]
    if count:
        with pytest.raises(ft.FragmentLayoutUnsupportedError, match="initial samples"):
            ft.parse_fragment_table(moov, fragments)
    else:
        assert ft.parse_fragment_table(moov, fragments).size.tolist() == [20, 20]


def test_fragment_track_id_must_match_the_video_track() -> None:
    """Do not index another track's samples using the selected video's metadata."""
    fragment = _fragment(_trun(), tfhd=_tfhd(track_id=2))
    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="track ID"):
        ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment])


@pytest.mark.parametrize("version", [0, 1, 2])
def test_decode_time_versions_are_explicitly_supported(version: int) -> None:
    """An unknown version must not reinterpret a 64-bit timestamp as 32-bit."""
    decode_time = 100 if version == 0 else 2**32 + 100
    tfdt = bf.pack_full_box("tfdt", version, struct.pack(">I" if version == 0 else ">Q", decode_time))
    fragment = (1000, bf.pack_moof(bf.pack_box("traf", _tfhd() + tfdt + _trun())))
    if version == 2:
        with pytest.raises(ft.FragmentLayoutUnsupportedError, match=r"tfdt.*unsupported version"):
            ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment])
    else:
        track = ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment])
        assert track.dts_ticks.tolist() == [decode_time, decode_time + 10]


def _trun(
    *,
    count: int = 2,
    durations: list[int] | None = None,
    sizes: list[int] | None = None,
    sample_flags: list[int] | None = None,
    composition_offsets: list[int] | None = None,
    first_sample_flags: int | None = None,
    data_offset: int | None = 32,
    version: int = 0,
) -> bytes:
    fields = (
        (bf._TRUN_SAMPLE_DURATION_PRESENT, durations),
        (bf._TRUN_SAMPLE_SIZE_PRESENT, sizes),
        (bf._TRUN_SAMPLE_FLAGS_PRESENT, sample_flags),
        (bf._TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT, composition_offsets),
    )
    flags = sum(flag for flag, values in fields if values is not None)
    payload = struct.pack(">I", count)
    if data_offset is not None:
        flags |= bf._TRUN_DATA_OFFSET_PRESENT
        payload += struct.pack(">i", data_offset)
    if first_sample_flags is not None:
        flags |= 0x000004
        payload += struct.pack(">I", first_sample_flags)
    for index in range(count):
        for flag, values in fields:
            if values is not None:
                fmt = ">i" if flag == bf._TRUN_SAMPLE_COMPOSITION_TIME_OFFSETS_PRESENT and version == 1 else ">I"
                payload += struct.pack(fmt, values[index])
    return bf.pack_box("trun", bytes([version]) + flags.to_bytes(3, "big") + payload)


def _tfhd(
    *,
    track_id: int = 1,
    duration: int | None = 10,
    size: int | None = 20,
    sample_flags: int | None = bf._SAMPLE_IS_NON_SYNC_SAMPLE_BIT,
    base_offset: int | None = None,
) -> bytes:
    flags = 0x020000 if base_offset is None else 0x000001
    payload = struct.pack(">I", track_id)
    if base_offset is not None:
        payload += struct.pack(">Q", base_offset)
    for flag, value in ((0x000008, duration), (0x000010, size), (0x000020, sample_flags)):
        if value is not None:
            flags |= flag
            payload += struct.pack(">I", value)
    return bf.pack_box("tfhd", b"\x00" + flags.to_bytes(3, "big") + payload)


def _fragment(trun: bytes, *, tfhd: bytes | None = None, dts: int = 100, offset: int = 1000) -> tuple[int, bytes]:
    return offset, bf.pack_moof(
        bf.pack_box("traf", (tfhd if tfhd is not None else _tfhd()) + bf.real_tfdt(base_media_decode_time=dts) + trun)
    )


def test_variable_sample_columns_and_fragment_boundaries() -> None:
    """Accumulate per-sample values and restart from each fragment's tfdt."""
    moov = bf.synthetic_fragmented_moov()
    fragments = [
        _fragment(_trun(count=3, durations=[3, 5, 7], sizes=[30, 40, 50], sample_flags=[0, 0x10000, 0x10000])),
        _fragment(_trun(durations=[2, 4], sizes=[60, 70], sample_flags=[0, 0x10000]), offset=2000, dts=120),
    ]
    track = ft.parse_fragment_table(moov, fragments)
    assert track.dts_ticks.tolist() == [100, 103, 108, 120, 122]
    assert track.offset.tolist() == [1032, 1062, 1102, 2032, 2092]
    assert track.size.tolist() == [30, 40, 50, 60, 70]
    assert track.is_keyframe.tolist() == [True, False, False, True, False]
    assert track.duration == 26


@pytest.mark.parametrize("source", ["trun", "tfhd", "trex"])
def test_sample_defaults_follow_precedence(source: str) -> None:
    """Resolve duration, size, and flags from the closest available source."""
    moov = bf.synthetic_fragmented_moov(include_trex=False)
    start, end = find_box(moov, ("moov",), 0, len(moov))
    moov = bf.pack_box(
        "moov", moov[start:end] + bf.trex_box(default_sample_duration=7, default_sample_size=9, default_sample_flags=0)
    )
    trun = _trun(durations=[3, 4], sizes=[5, 6], sample_flags=[0x10000, 0]) if source == "trun" else _trun()
    tfhd = _tfhd() if source != "trex" else _tfhd(duration=None, size=None, sample_flags=None)
    track = ft.parse_fragment_table(moov, [_fragment(trun, tfhd=tfhd)])
    if source == "trun":
        assert track.dts_ticks.tolist() == [100, 103]
        assert track.size.tolist() == [5, 6]
        assert track.is_keyframe.tolist() == [False, True]
        assert track.duration == 7
    elif source == "tfhd":
        assert track.dts_ticks.tolist() == [100, 110]
        assert track.size.tolist() == [20, 20]
        assert track.is_keyframe.tolist() == [False, False]
        assert track.duration == 20
    else:
        assert track.dts_ticks.tolist() == [100, 107]
        assert track.size.tolist() == [9, 9]
        assert track.is_keyframe.tolist() == [True, True]
        assert track.duration == 14


@pytest.mark.parametrize("use_trex", [False, True])
def test_first_sample_flags_only_override_the_first_sample(*, use_trex: bool) -> None:
    """Later samples retain the tfhd or trex flags."""
    moov = bf.synthetic_fragmented_moov(include_trex=False)
    start, end = find_box(moov, ("moov",), 0, len(moov))
    moov = bf.pack_box("moov", moov[start:end] + bf.trex_box(default_sample_flags=0x10000))
    track = ft.parse_fragment_table(
        moov, [_fragment(_trun(count=3, first_sample_flags=0), tfhd=_tfhd(sample_flags=None if use_trex else 0x10000))]
    )
    assert track.is_keyframe.tolist() == [True, False, False]


@pytest.mark.parametrize("base_offset", [None, 5000])
def test_run_byte_offsets_use_the_selected_base(base_offset: int | None) -> None:
    """Apply signed data offsets to explicit or moof-relative bases."""
    track = ft.parse_fragment_table(
        bf.synthetic_fragmented_moov(),
        [_fragment(_trun(sizes=[11, 13], data_offset=-32), tfhd=_tfhd(base_offset=base_offset))],
    )
    base = 1000 if base_offset is None else base_offset
    assert track.offset.tolist() == [base - 32, base - 21]


def test_missing_duration_cannot_be_estimated_within_a_run() -> None:
    """Missing interior durations leave subsequent DTS values unknowable."""
    with pytest.raises(ft.FragmentLayoutUnsupportedError, match="sample duration unresolved"):
        ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [_fragment(_trun(), tfhd=_tfhd(duration=None))])


@pytest.mark.parametrize(("field", "value"), [("duration", 0), ("size", 0), ("sample_flags", 0)])
def test_explicit_zero_defaults_are_preserved(field: str, value: int) -> None:
    """An explicit zero takes precedence over trex defaults."""
    track = ft.parse_fragment_table(
        bf.synthetic_fragmented_moov(trex_sample_duration=7), [_fragment(_trun(), tfhd=_tfhd(**{field: value}))]
    )
    if field == "duration":
        assert track.dts_ticks.tolist() == [100, 100]
        assert track.duration == 0
    elif field == "size":
        assert track.size.tolist() == [0, 0]
    else:
        assert track.is_keyframe.tolist() == [True, True]


@pytest.mark.parametrize(("version", "offsets"), [(0, [0, 0]), (1, [0, 0]), (0, [0, 3]), (1, [0, -3])])
def test_composition_offsets_in_later_samples(version: int, offsets: list[int]) -> None:
    """Inspect every offset, accepting zero and rejecting reordering."""
    fragment = _fragment(_trun(composition_offsets=offsets, version=version))
    if any(offsets):
        with pytest.raises(ft.CompositionOffsetUnsupportedError):
            ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment])
    else:
        assert ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment]).composition_offsets is None


def test_truncated_entry_cannot_read_into_the_next_box() -> None:
    """A sibling box cannot supply bytes missing from the trun."""
    trun = _trun(sizes=[11, 13])
    truncated = bf.pack_box("trun", trun[8:-4])
    fragment = _fragment(truncated + bf.pack_box("free", b"\x00" * 16))
    with pytest.raises(MalformedBoxHeaderError, match="truncated sample entries"):
        ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [fragment])


@pytest.mark.parametrize("kind", ["traf", "trun", "trak"])
def test_multiple_tracks_or_runs_are_refused(kind: str) -> None:
    """Refuse layouts that would otherwise silently omit sample data."""
    moov = bf.synthetic_fragmented_moov()
    offset, fragment = _fragment(_trun())
    if kind == "trak":
        start, end = find_box(moov, ("moov", "trak"), 0, len(moov))
        moov = bf.pack_box("moov", moov[8:] + bf.pack_box("trak", moov[start:end]))
    elif kind == "traf":
        start, end = find_box(fragment, ("traf",), 8, len(fragment))
        fragment = bf.pack_moof(fragment[8:] + bf.pack_box("traf", fragment[start:end]))
    else:
        start, end = find_box(fragment, ("traf",), 8, len(fragment))
        fragment = bf.pack_moof(bf.pack_box("traf", fragment[start:end] + _trun()))
    with pytest.raises(ft.FragmentLayoutUnsupportedError, match=f"exactly one {kind}"):
        ft.parse_fragment_table(moov, [(offset, fragment)])


@pytest.mark.parametrize(
    "kwargs", [{"count": 0}, {"version": 2}, {"data_offset": None}, {"sample_flags": [0, 0], "first_sample_flags": 0}]
)
def test_unsupported_run_layouts_are_refused(kwargs: dict[str, object]) -> None:
    """Reject empty, unknown-version, ambiguous-flag, and implicit-offset runs."""
    with pytest.raises(ft.FragmentLayoutUnsupportedError):
        ft.parse_fragment_table(bf.synthetic_fragmented_moov(), [_fragment(_trun(**kwargs))])
