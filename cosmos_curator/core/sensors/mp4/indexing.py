# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build MP4 metadata and frame indexes directly from video sources."""

import enum
import os
import struct
from typing import BinaryIO

from cosmos_curator.core.sensors.data.mp4 import IndexedVideo, Mp4FrameIndex, Mp4Metadata
from cosmos_curator.core.sensors.mp4.box_primitives import (
    EXTENDED_HEADER_BYTES,
    BoxNotFoundError,
    MalformedBoxHeaderError,
    find_all_boxes,
    find_box,
    read_box_header,
    require_ftyp_size,
)
from cosmos_curator.core.sensors.mp4.box_probe import check_frame_byte_ranges, index_from_boxes
from cosmos_curator.core.sensors.mp4.fragment_table import parse_fragment_table
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.types.types import DataSource
from cosmos_curator.core.sensors.utils.io import open_data_source


class Mp4Indexer(enum.Enum):
    """Backend used to build an MP4 frame index."""

    PYAV = "pyav"
    CUSTOM = "custom"


def _read_whole_box(stream: BinaryIO, box_type: str, position: int, box_size: int) -> bytes:
    """Read the *box_size*-byte box at *position*, refusing an oversized ``ftyp`` before reading it."""
    if box_type == "ftyp":
        require_ftyp_size(box_size, position)
    stream.seek(position)
    content = stream.read(box_size)
    if len(content) != box_size:
        msg = f"truncated {box_type} box at offset {position}"
        raise MalformedBoxHeaderError(msg)
    return content


def _read_boxes(stream: BinaryIO) -> tuple[int, bytes | None, bytes, list[tuple[int, bytes]]]:
    """Read the first ftyp ahead of moov, then moov and moof boxes, seeking past media payloads."""
    file_size = stream.seek(0, os.SEEK_END)
    ftyp: bytes | None = None
    moov: bytes | None = None
    moofs: list[tuple[int, bytes]] = []
    position = 0
    while position < file_size:
        stream.seek(position)
        header = stream.read(EXTENDED_HEADER_BYTES)
        try:
            box_type, box_size, _ = read_box_header(header, 0, file_size - position)
        except struct.error as error:
            msg = f"truncated MP4 box header at offset {position}"
            raise MalformedBoxHeaderError(msg) from error
        if position + box_size > file_size:
            msg = f"box {box_type!r} at offset {position} extends past the source end"
            raise MalformedBoxHeaderError(msg)
        if box_type in ("moov", "moof") or (box_type == "ftyp" and ftyp is None and moov is None):
            content = _read_whole_box(stream, box_type, position, box_size)
            if box_type == "ftyp":
                ftyp = content
            elif box_type == "moov":
                if moov is not None:
                    msg = "MP4 contains multiple moov boxes"
                    raise MalformedBoxHeaderError(msg)
                moov = content
            else:
                moofs.append((position, content))
        position += box_size
    if moov is None:
        msg = "no top-level moov box found in the source"
        raise BoxNotFoundError(msg)
    return file_size, ftyp, moov, moofs


def _require_single_video_track(moov: bytes, stream_idx: int) -> None:
    """Require the custom parser's track selection to match the decoder's."""
    _, _, header_len = read_box_header(moov, 0, len(moov))
    video_tracks = 0
    for start, end in find_all_boxes(moov, "trak", header_len, len(moov)):
        hdlr_start, _ = find_box(moov, ("mdia", "hdlr"), start, end)
        video_tracks += moov[hdlr_start + 8 : hdlr_start + 12] == b"vide"
    if stream_idx != 0 or video_tracks != 1:
        msg = "custom MP4 indexing requires exactly one video track and stream_idx=0"
        raise ValueError(msg)


def index_video(source: DataSource, *, indexer: Mp4Indexer = Mp4Indexer.PYAV, stream_idx: int = 0) -> IndexedVideo:
    """Build metadata and a frame index without requiring extracted box artifacts.

    Also returns the source's ``moov`` box, header included. Indexed decoding sets up its
    decoder from that ``moov``, so passing it on saves reading it again.

    Raises:
        TypeError: The indexer is not an Mp4Indexer.
        ValueError: Custom indexing is requested for multiple video tracks or a nonzero stream index.

    """
    indexer_value: object = indexer
    if not isinstance(indexer_value, Mp4Indexer):
        msg = f"Unknown MP4 indexer: {indexer!r}"
        raise TypeError(msg)
    with open_data_source(source) as stream:
        file_size, ftyp, moov, moofs = _read_boxes(stream)
    if indexer is Mp4Indexer.PYAV:
        result = index_from_boxes(file_size, moov, moofs, ftyp_bytes=ftyp, stream_idx=stream_idx)
        frame_index = Mp4FrameIndex.from_video_probe(result.probe, elst_bytes=result.metadata.elst_bytes)
        return IndexedVideo(frame_index, result.metadata, moov)
    _require_single_video_track(moov, stream_idx)
    track = parse_fragment_table(moov, moofs) if moofs else parse_video_track(moov)
    check_frame_byte_ranges(track.offset, track.size, file_size)
    return IndexedVideo(
        Mp4FrameIndex.from_video_track(track), Mp4Metadata.from_video_track(track, ftyp_bytes=ftyp), moov
    )
