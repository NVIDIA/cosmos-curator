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
"""Real MP4 clips, and the box walker's reading of them, for tests comparing it against PyAV.

Unlike ``_box_fixtures.py``'s hand-packed boxes, these carry decodable media, so PyAV opens them.
"""

import functools
import io
import pathlib
import struct

import av
import numpy as np

from cosmos_curator.core.sensors.mp4 import box_locate
from cosmos_curator.core.sensors.mp4.box_primitives import find_box
from cosmos_curator.core.sensors.mp4.fragment_table import parse_fragment_table
from cosmos_curator.core.sensors.mp4.moof_walk import walk_moofs
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.mp4.moov_track import VideoTrack

REAL_CLIP_DIR = pathlib.Path(__file__).resolve().parents[3] / "pipelines" / "video" / "data"
CLIP = REAL_CLIP_DIR / "test_clip_10s.mp4"
#: Carries B-frames, a ``ctts``, and an edit list that FFmpeg applies to dts unless told not to.
BFRAME_CLIP = REAL_CLIP_DIR / "test_clip_10s_bframes.mp4"
REAL_CLIPS = (CLIP, BFRAME_CLIP, REAL_CLIP_DIR / "test_video_30s.mp4")


@functools.cache
def fragmented_clip() -> bytes:
    """Remux :data:`CLIP` into one ``moof``/``mdat`` per frame, the one layout ``fragment_table.py`` accepts."""
    return _remux(CLIP, "frag_every_frame+empty_moov+default_base_moof")


@functools.cache
def gop_fragmented_clip() -> bytes:
    """Remux :data:`BFRAME_CLIP` into one ``moof`` per keyframe interval, whose ``trun`` carries many samples.

    ``fragment_table.py`` refuses its B-frame composition offsets; libav reads them.
    """
    return _remux(BFRAME_CLIP, "frag_keyframe+empty_moov+default_base_moof")


@functools.cache
def moov_and_fragments_clip() -> bytes:
    """Remux :data:`BFRAME_CLIP` with its first second in ``moov`` and each later second in a fragment.

    ``fragment_table.py`` refuses initial ``moov`` samples beside fragments; libav reads both.
    """
    return _remux(BFRAME_CLIP, "default_base_moof", frag_duration="1000000")


def lower_fragment_composition_offsets(data: bytes, by: int) -> bytes:
    """Return fragmented *data* with each ``trun``'s composition offsets lowered by *by*, as signed version 1.

    Every ``moof`` must hold one ``traf`` with one ``trun`` storing composition offsets.
    """
    patched = bytearray(data)
    for moof_start, moof_bytes in moof_rows(data):
        trun_start = moof_start + find_box(moof_bytes, ("moof", "traf", "trun"), 0, len(moof_bytes))[0]
        flags = int.from_bytes(patched[trun_start + 1 : trun_start + 4], "big")
        sample_count = struct.unpack_from(">I", patched, trun_start + 4)[0]
        field_count = sum(bool(flags & field) for field in (0x100, 0x200, 0x400, 0x800))
        entry = trun_start + 8 + 4 * sum(bool(flags & field) for field in (0x1, 0x4)) + 4 * (field_count - 1)
        for _ in range(sample_count):
            struct.pack_into(">i", patched, entry, struct.unpack_from(">i", patched, entry)[0] - by)
            entry += 4 * field_count
        patched[trun_start] = 1
    return bytes(patched)


@functools.cache
def mjpeg_then_h264_clip() -> bytes:
    """Mux an MJPEG track ahead of :data:`CLIP`'s H.264 track, so libav lists the MJPEG stream first.

    The box walker selects the H.264 track, the only one it can index.
    """
    buffer = io.BytesIO()
    with av.open(str(CLIP)) as source, av.open(buffer, mode="w", format="mp4") as sink:
        h264_source = source.streams.video[0]
        mjpeg = sink.add_stream("mjpeg", rate=30)
        mjpeg.width, mjpeg.height, mjpeg.pix_fmt = 64, 48, "yuvj420p"
        h264 = sink.add_stream_from_template(h264_source)
        black = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24")
        for index in range(30):
            frame = black.reformat(format="yuvj420p")
            frame.pts = index
            sink.mux(mjpeg.encode(frame))
        sink.mux(mjpeg.encode(None))
        for packet in source.demux(h264_source):
            if packet.dts is None:
                continue
            packet.stream = h264
            sink.mux(packet)
    return buffer.getvalue()


def _remux(clip: pathlib.Path, movflags: str, **muxer_options: str) -> bytes:
    buffer = io.BytesIO()
    options = {"movflags": movflags, **muxer_options}
    with av.open(str(clip)) as source, av.open(buffer, mode="w", format="mp4", options=options) as sink:
        stream = source.streams.video[0]
        output = sink.add_stream_from_template(stream)
        for packet in source.demux(stream):
            if packet.dts is None:
                continue
            packet.stream = output
            sink.mux(packet)
    return buffer.getvalue()


def moov_box(data: bytes) -> bytes:
    """Return *data*'s whole ``moov`` box, header included."""
    return box_locate.get_moov_auto(
        len(data), lambda offset, length: data[offset : offset + length], source_name="fixture"
    )


def moof_rows(data: bytes) -> list[tuple[int, bytes]]:
    """Return *data*'s ``(moof_start, moof_bytes)`` pairs, in file order; empty if nonfragmented."""
    return list(walk_moofs(len(data), io.BytesIO(data), chunk_bytes=1 << 20))


def walker_track(data: bytes) -> VideoTrack:
    """Parse *data* with the box walker, as fragmented exactly when it carries a ``moof``."""
    rows = moof_rows(data)
    return parse_fragment_table(moov_box(data), rows) if rows else parse_video_track(moov_box(data))
