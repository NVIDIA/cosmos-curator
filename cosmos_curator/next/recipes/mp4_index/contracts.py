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

"""Constants, identities, and byte-range/frame-index contracts for the ``mp4-index`` recipe.

``FRAME_INDEX_SCHEMA``/``VIDEO_METADATA_SCHEMA`` rows have typed equivalents --
``Mp4FrameIndex``/``Mp4Metadata`` -- in ``cosmos_curator.core.sensors.data.mp4``.
"""

import contextlib
import os
import pathlib
import posixpath
import uuid
from collections.abc import Iterator
from typing import IO, Any, Final, Literal
from urllib.parse import urlsplit

import pyarrow as pa
import smart_open
from pyarrow import ipc

from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.local_uri import local_media_path, uri_scheme

KIND = "mp4-index"

CONTAINER_KIND_FRAGMENTED = "fragmented"
CONTAINER_KIND_NONFRAGMENTED = "nonfragmented"

#: VIDEO_MANIFEST_SCHEMA's row-shape version -- extraction_contract_id versions behavior, not shape.
RECORD_SCHEMA_VERSION = 7

#: Which indexer built a video's frames and metadata, part of ``extraction_contract_id``.
Indexer = Literal["pyav", "box_walker"]
INDEXER_PYAV: Final[Indexer] = "pyav"
#: The box walker, which indexes a video whenever an extension decodes its UUID boxes.
INDEXER_BOX_WALKER: Final[Indexer] = "box_walker"


def make_video_id(media_location: str) -> str:
    """Identify a video by where it is, not a name a directory happens to have.

    Two different videos under two different prefixes never collide just by sharing a naming
    convention, and a video that moves to another location is treated as a different video.
    """
    return canonical_digest(media_location.rstrip("/"))


class UnresolvableMediaLocationError(ValueError):
    """media_location's own path structure has too few components to derive an identity."""


def make_session_camera_id(source_session_id: str, camera_key: str) -> str:
    """Identify a logical session/camera for manifest lookups."""
    return canonical_digest([source_session_id, camera_key])


def derive_session_and_camera(media_location: str, *, session_id_path_depth: int) -> tuple[str, str]:
    """Derive ``(source_session_id, camera_key)`` from *media_location*'s own path structure.

    ``camera_key`` is everything after ``source_session_id`` -- every remaining directory
    component plus the filename stem, not the stem alone: two videos can share a session and
    filename stem while differing only by an intermediate directory (``unit-00``/``unit-01``,
    a real independently-discovered pair), and the stem alone would conflate their logical identities.
    """
    path = urlsplit(media_location).path if "://" in media_location else media_location
    parts = [part for part in path.split("/") if part]
    if not parts:
        msg = f"media_location {media_location!r} has no path components to derive a filename from"
        raise UnresolvableMediaLocationError(msg)
    filename, dir_parts = parts[-1], parts[:-1]
    if session_id_path_depth > len(dir_parts):
        msg = (
            f"media_location {media_location!r} has only {len(dir_parts)} directory component(s), "
            f"but execution.session_id_path_depth={session_id_path_depth} needs at least that many"
        )
        raise UnresolvableMediaLocationError(msg)
    session_index = len(dir_parts) - session_id_path_depth
    source_session_id = dir_parts[session_index]
    camera_key = "/".join([*dir_parts[session_index + 1 :], posixpath.splitext(filename)[0]])
    return source_session_id, camera_key


@contextlib.contextmanager
def _atomic_local_write(local_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield a same-directory temp path; replace *local_path* with it on clean exit, discard it otherwise.

    ``smart_open``'s ``s3://`` backend is crash-safe for free (a multipart upload only completes
    on ``close()``); a local write isn't -- writing straight to *local_path* and getting killed
    partway leaves a truncated file at the path a later reader trusts as complete, indistinguishable
    from a genuinely corrupt source. Renaming a same-directory temp file on success instead leaves
    an orphaned temp file on failure, never a partial *local_path* -- same directory, not a shared
    system temp dir, so the rename is same-filesystem (atomic), not an interruptible cross-filesystem copy.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(f".{local_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        yield temp_path
        temp_path.replace(local_path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def _open_sidecar_for_write(path: str, *, transport_params: dict[str, Any] | None = None) -> Iterator[IO[bytes]]:
    """Open *path* for a single atomic write, local or ``s3://``.

    A malformed URI that claims to be local (``file:/`` single-slash, an unencoded ``#``) still
    raises -- only a genuinely different scheme routes through ``smart_open`` directly; conflating
    the two would treat a broken local path as remote instead of failing loudly. The remote branch
    goes through ``smart_open``, not ``pathlib.Path``/plain ``open()``: ``pathlib.Path("s3://bucket/key")``
    doesn't raise or recognize the scheme, it silently treats the whole URI as a literal local
    relative path.
    """
    if uri_scheme(path) in ("", "file"):
        with (
            _atomic_local_write(local_media_path(path)) as temp_path,
            smart_open.open(str(temp_path), "wb", transport_params=transport_params) as handle,
        ):
            yield handle
        return
    with smart_open.open(path, "wb", transport_params=transport_params) as handle:
        yield handle


#: One struct per sample, in decode order. Ticks use the video's metadata time base,
#: without edit-list shifts. Upstream-provided EOF/SOF/MOE are expressed in nanoseconds
#: and are all-null for ordinary MP4s. Composition offsets are null
#: when absent; explicit offsets, including zero, remain signed integers.
FRAME_INDEX_SCHEMA = pa.schema(
    [
        pa.field("dts_ticks", pa.int64(), nullable=False),
        pa.field("composition_offset_ticks", pa.int64(), nullable=True),
        pa.field("is_keyframe", pa.bool_(), nullable=False),
        pa.field("offset", pa.int64(), nullable=False),  # packet data start
        pa.field("size", pa.int64(), nullable=False),
        pa.field("eof_ns", pa.int64(), nullable=True),
        pa.field("sof_ns", pa.int64(), nullable=True),
        pa.field("moe_ns", pa.int64(), nullable=True),
    ]
)


#: Stream metadata embedded as a struct in each canonical video record.
VIDEO_METADATA_SCHEMA = pa.schema(
    [
        pa.field("codec_name", pa.string(), nullable=False),
        pa.field("extradata", pa.binary(), nullable=False),
        pa.field("time_base_num", pa.int32(), nullable=False),
        # mdhd's own timescale is unsigned 32-bit -- int64, not int32, so a real value above
        # int32's signed max (observed: 0x90000000) still fits.
        pa.field("time_base_den", pa.int64(), nullable=False),
        pa.field("frame_count", pa.int64(), nullable=False),
        pa.field("keyframe_count", pa.int64(), nullable=False),
        pa.field("width", pa.int32(), nullable=False),
        pa.field("height", pa.int32(), nullable=False),
        # avcC fields, also in extradata: null when PyAV indexed the video
        pa.field("avc_profile_idc", pa.int32(), nullable=True),
        pa.field("avc_constraint_flags", pa.int32(), nullable=True),
        pa.field("avc_level_idc", pa.int32(), nullable=True),
        pa.field("nal_length_size", pa.int32(), nullable=True),
        pa.field("chroma_format_idc", pa.int32(), nullable=True),
        pa.field("bit_depth_luma_minus8", pa.int32(), nullable=True),
        pa.field("bit_depth_chroma_minus8", pa.int32(), nullable=True),
        # btrt -- the encoder's own claimed bitrate, distinct from this schema's own measured bit_rate_bps;
        # null when PyAV indexed the video
        pa.field("buffer_size_db", pa.int64(), nullable=True),
        pa.field("max_bitrate_bps", pa.int64(), nullable=True),
        pa.field("avg_bitrate_bps", pa.int64(), nullable=True),
        # mvhd/tkhd fields libav does not report: null when PyAV indexed the video
        pa.field("movie_creation_time", pa.int64(), nullable=True),
        pa.field("movie_modification_time", pa.int64(), nullable=True),
        pa.field("track_creation_time", pa.int64(), nullable=True),
        pa.field("track_modification_time", pa.int64(), nullable=True),
        pa.field("track_matrix", pa.list_(pa.int32()), nullable=True),
        # tkhd's own width/height are unsigned 32-bit (16.16 fixed-point) -- int64, not int32,
        # for the same reason as time_base_den above.
        pa.field("track_display_width", pa.int64(), nullable=True),
        pa.field("track_display_height", pa.int64(), nullable=True),
        # edts/elst box, verbatim, header included; null when the track has no edit list --
        # query presence directly (IS NOT NULL), no separate bool
        pa.field("elst_bytes", pa.binary(), nullable=True),
        pa.field("avg_frame_rate_num", pa.int32(), nullable=False),
        pa.field("avg_frame_rate_den", pa.int32(), nullable=False),
        pa.field("bit_rate_bps", pa.int64(), nullable=False),
        pa.field("duration_ns", pa.int64(), nullable=False),
        pa.field("encoder_tag", pa.string(), nullable=True),
    ]
)


#: One complete video record: source identity, provenance, metadata, and frame index.
VIDEO_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("source_session_id", pa.large_string(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
        # Hash of (source_session_id, camera_key) -- stored so a scan can filter on it via a BTREE index.
        pa.field("session_camera_id", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("source_width", pa.int32(), nullable=False),
        pa.field("source_height", pa.int32(), nullable=False),
        pa.field("source_frame_rate", pa.float64(), nullable=False),
        pa.field("source_frame_count", pa.int64(), nullable=False),
        pa.field("source_duration_ns", pa.int64(), nullable=False),
        pa.field("source_video_codec", pa.string(), nullable=False),
        pa.field("metadata", pa.struct(VIDEO_METADATA_SCHEMA), nullable=False),
        pa.field(
            "frames", pa.large_list(pa.field("item", pa.struct(FRAME_INDEX_SCHEMA), nullable=False)), nullable=False
        ),
        pa.field("moov_location", pa.large_string(), nullable=False),
        pa.field("moof_table_location", pa.large_string(), nullable=True),
        pa.field("container_kind", pa.string(), nullable=False),
        pa.field("record_schema_version", pa.int32(), nullable=False),
        pa.field("extraction_contract_id", pa.large_string(), nullable=False),
    ]
)


#: One row per failed attempt -- keyed by the same video_id as VIDEO_MANIFEST_SCHEMA but never
#: unique, since a video can be retried across runs. Append-only, so failure history stays
#: queryable across runs without log-scraping. No reconciliation with VIDEO_MANIFEST_SCHEMA: a row
#: here records one failed attempt, not the video's current status.
ERROR_SCHEMA = pa.schema(
    [
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        # Producer-defined category (e.g. an exception class name) -- a stable,
        # queryable value. error_message is free text, for a human reading one row.
        pa.field("error_type", pa.string(), nullable=False),
        pa.field("error_message", pa.large_string(), nullable=False),
        pa.field("attempted_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)


def write_arrow_sidecar(
    rows: list[dict[str, object]], schema: pa.Schema, path: str, *, transport_params: dict[str, Any] | None = None
) -> None:
    """Write *rows* to *path* as a single Arrow IPC file under *schema*, local or ``s3://``."""
    table = pa.Table.from_pylist(rows, schema=schema)
    with (
        _open_sidecar_for_write(path, transport_params=transport_params) as handle,
        ipc.new_file(handle, table.schema) as writer,
    ):
        writer.write_table(table)


def read_arrow_sidecar(path: str, *, transport_params: dict[str, Any] | None = None) -> pa.Table:
    """Read one Arrow IPC sidecar back from *path*, whole -- these files are never scanned partially."""
    with smart_open.open(path, "rb", transport_params=transport_params) as handle:
        data = handle.read()
    with ipc.open_file(pa.BufferReader(data)) as reader:
        return reader.read_all()


def write_bytes_sidecar(data: bytes, path: str, *, transport_params: dict[str, Any] | None = None) -> None:
    """Write *data* verbatim to *path*, local or ``s3://`` -- no envelope, no schema."""
    with _open_sidecar_for_write(path, transport_params=transport_params) as handle:
        handle.write(data)


def read_bytes_sidecar(path: str, *, transport_params: dict[str, Any] | None = None) -> bytes:
    """Read one bytes sidecar back from *path*, whole -- these files are never scanned partially."""
    with smart_open.open(path, "rb", transport_params=transport_params) as handle:
        return handle.read()


def read_video_record(table: pa.Table, *, row_index: int = 0) -> tuple[pa.Table, dict[str, Any]]:
    """Unwrap one video record's frame columns without copying their Arrow value buffers.

    Accepts projected or chunked Arrow tables, including slices of a Lance scan.
    Metadata alone is converted to Python; frame values remain Arrow-native. Frame
    columns must match their declared types; incompatible stored types are never cast.
    """
    metadata = table.column("metadata")[row_index]
    frames = table.column("frames")[row_index]
    if not metadata.is_valid or not frames.is_valid:
        msg = "Video metadata and frames must be present"
        raise ValueError(msg)
    values = frames.values
    if values.null_count:
        msg = "Frame records must not be null"
        raise ValueError(msg)
    frame_fields = [values.type.field(name) for name in FRAME_INDEX_SCHEMA.names]
    # Validate before constructing the table: Arrow could otherwise cast compatible
    # types, concealing an incompatible stored representation or lost precision.
    for field, expected in zip(frame_fields, FRAME_INDEX_SCHEMA, strict=True):
        if field.type != expected.type:
            msg = f"Frame column {field.name!r} must have type {expected.type}, got {field.type}"
            raise ValueError(msg)
    frame_table = pa.Table.from_arrays(
        [values.field(field.name) for field in frame_fields], schema=pa.schema(frame_fields)
    )
    metadata_row = metadata.as_py()
    if len(frame_table) != metadata_row["frame_count"]:
        msg = "Frame count disagrees with video metadata"
        raise ValueError(msg)
    return frame_table, metadata_row
