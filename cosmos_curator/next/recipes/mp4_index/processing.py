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

"""Discover, extract, and index one video in a single pass."""

import pathlib
from typing import Any
from urllib.parse import urlsplit

import attrs
import pyarrow as pa
import smart_open  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata, video_track_sample_arrays
from cosmos_curator.core.sensors.mp4 import box_locate
from cosmos_curator.core.sensors.mp4.byte_range import FetchRange
from cosmos_curator.core.sensors.mp4.errors import UnindexableVideoError
from cosmos_curator.core.sensors.mp4.fragment_table import parse_fragment_table
from cosmos_curator.core.sensors.mp4.moof_schema import MOOF_TABLE_SCHEMA
from cosmos_curator.core.sensors.mp4.moof_walk import NoFragmentsFoundError, walk_moofs
from cosmos_curator.core.sensors.mp4.moov import parse_video_track
from cosmos_curator.core.sensors.mp4.moov_track import VideoTrack
from cosmos_curator.core.utils.storage.s3_client import S3Prefix
from cosmos_curator.core.utils.storage.storage_client import StorageClient
from cosmos_curator.core.utils.storage.storage_utils import (
    get_smart_open_client_params,
    get_storage_client,
    path_exists,
)
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    CONTAINER_KIND_FRAGMENTED,
    RECORD_SCHEMA_VERSION,
    VIDEO_IDENTITY_SCHEMA,
    UnresolvableMediaLocationError,
    derive_session_and_camera,
    make_session_camera_id,
    read_arrow_sidecar,
    read_bytes_sidecar,
    write_arrow_sidecar,
    write_bytes_sidecar,
    write_frame_index,
    write_video_metadata,
)

#: Every VIDEO_IDENTITY_SCHEMA field a failed video carries as None.
_VIDEO_ONLY_FIELDS = tuple(set(VIDEO_IDENTITY_SCHEMA.names) - {"video_id", "media_location"})


def _file_size(media_location: str, storage_client: StorageClient | None) -> int:
    """Return *media_location*'s byte size, local or ``s3://``."""
    if storage_client is not None:
        size_bytes = storage_client.stat(S3Prefix(media_location)).size_bytes
        if size_bytes is None:
            msg = f"{media_location}: storage backend reported no size for an object that exists"
            raise UnindexableVideoError(msg)
        return size_bytes
    return pathlib.Path(media_location).stat().st_size


def _frame_index_rows(video_track: VideoTrack) -> list[dict[str, object]]:
    """Serialize a ``VideoTrack`` into ``FRAME_INDEX_SCHEMA`` row dicts, one per sample.

    Built from :func:`video_track_sample_arrays` directly, not through
    ``Mp4FrameIndex.from_video_track`` -- nothing here depends on ``pts_ns`` or the edit list,
    so extraction should not fail over an ``elst`` shape only a later sidecar reader needs.
    """
    dts_ticks, composition_offset_ticks_array, is_keyframe_array = video_track_sample_arrays(video_track)
    dts_ticks = dts_ticks.tolist()
    composition_offset_ticks = (
        [None] * len(dts_ticks) if composition_offset_ticks_array is None else composition_offset_ticks_array.tolist()
    )
    is_keyframe = is_keyframe_array.tolist()
    offset = video_track.offset.tolist()
    size = video_track.size.tolist()
    return [
        {
            "dts_ticks": dts_ticks[i],
            "composition_offset_ticks": composition_offset_ticks[i],
            "is_keyframe": is_keyframe[i],
            "offset": offset[i],
            "size": size[i],
        }
        for i in range(len(dts_ticks))
    ]


def _moof_rows(moof_table_location: str, *, transport_params: dict[str, Any] | None) -> list[tuple[int, bytes]]:
    """Read a published ``MOOF_TABLE_SCHEMA`` sidecar back into ``(moof_start, moof_bytes)`` pairs, in file order."""
    table = read_arrow_sidecar(moof_table_location, transport_params=transport_params)
    starts = table.column("moof_start").to_pylist()
    contents = table.column("moof_bytes").to_pylist()
    return list(zip(starts, contents, strict=True))


def _make_fetch_range(uri: str, transport_params: dict[str, Any] | None) -> FetchRange:
    """Build a :data:`FetchRange` over *uri* using ``smart_open`` -- already a normal dependency at this layer."""

    def fetch_range(offset: int, length: int) -> bytes:
        with smart_open.open(uri, "rb", transport_params=transport_params) as handle:
            handle.seek(offset)
            return handle.read(length)

    return fetch_range


def _ensure_moov(  # noqa: PLR0913 -- one caller-supplied identity/location/tuning field per arg; all independent
    media_location: str,
    moov_sidecar_location: str,
    *,
    file_size: int,
    skip_existing_sidecars: bool,
    output_storage_client: StorageClient | None,
    input_transport_params: dict[str, Any] | None,
    output_transport_params: dict[str, Any] | None,
) -> tuple[bytes, bool]:
    """Return (this video's ``moov`` bytes, whether they came from an already-extracted sidecar)."""
    if skip_existing_sidecars and path_exists(moov_sidecar_location, output_storage_client):
        return read_bytes_sidecar(moov_sidecar_location, transport_params=output_transport_params), True
    fetch_range = _make_fetch_range(media_location, input_transport_params)
    moov_bytes = box_locate.get_moov_auto(file_size, fetch_range, source_name=media_location)
    write_bytes_sidecar(moov_bytes, moov_sidecar_location, transport_params=output_transport_params)
    return moov_bytes, False


def _ensure_moof(  # noqa: PLR0913 -- one caller-supplied identity/location/tuning field per arg; all independent
    media_location: str,
    moof_table_location: str,
    *,
    file_size: int,
    chunk_bytes: int,
    skip_existing_sidecars: bool,
    output_storage_client: StorageClient | None,
    input_transport_params: dict[str, Any] | None,
    output_transport_params: dict[str, Any] | None,
) -> tuple[list[tuple[int, bytes]], bool]:
    """Return (this video's ``(moof_start, moof_bytes)`` pairs, whether they came from an already-extracted sidecar)."""
    if skip_existing_sidecars and path_exists(moof_table_location, output_storage_client):
        return _moof_rows(moof_table_location, transport_params=output_transport_params), True
    with smart_open.open(media_location, "rb", transport_params=input_transport_params) as handle:
        moof_pairs = list(walk_moofs(file_size, handle, chunk_bytes=chunk_bytes))
    if not moof_pairs:
        msg = f"{media_location}: input.container_kind=fragmented but no moof boxes found"
        raise NoFragmentsFoundError(msg)
    moof_rows = [{"moof_start": start, "moof_bytes": moof_bytes} for start, moof_bytes in moof_pairs]
    write_arrow_sidecar(moof_rows, MOOF_TABLE_SCHEMA, moof_table_location, transport_params=output_transport_params)
    return moof_pairs, False


def _extract_video_track(  # noqa: PLR0913 -- one caller-supplied identity/location/tuning field per arg; all independent
    media_location: str,
    moov_sidecar_location: str,
    moof_table_location: str,
    *,
    container_kind: str,
    file_size: int,
    chunk_bytes: int,
    skip_existing_sidecars: bool,
    output_storage_client: StorageClient | None,
    input_transport_params: dict[str, Any] | None,
    output_transport_params: dict[str, Any] | None,
) -> VideoTrack:
    """Fetch (with caching) and parse one video's ``moov``/``moof`` into a :class:`VideoTrack`.

    A parse failure against bytes actually served from a cache retries once with caching off --
    a cached sidecar that fails to parse could be stale/corrupt rather than a genuine source
    problem. A parse failure on a fresh fetch never retries: nothing was cached to blame, so a
    second attempt would just walk the same source again for the same result.

    Raises:
        UnindexableVideoError: the fetched content doesn't parse, or parses to zero samples
            (``EmptySampleTableError``).

    """
    moov_bytes, served_from_cache = _ensure_moov(
        media_location,
        moov_sidecar_location,
        file_size=file_size,
        skip_existing_sidecars=skip_existing_sidecars,
        output_storage_client=output_storage_client,
        input_transport_params=input_transport_params,
        output_transport_params=output_transport_params,
    )
    try:
        if container_kind == CONTAINER_KIND_FRAGMENTED:
            moof_pairs, moof_from_cache = _ensure_moof(
                media_location,
                moof_table_location,
                file_size=file_size,
                chunk_bytes=chunk_bytes,
                skip_existing_sidecars=skip_existing_sidecars,
                output_storage_client=output_storage_client,
                input_transport_params=input_transport_params,
                output_transport_params=output_transport_params,
            )
            served_from_cache = served_from_cache or moof_from_cache
            return parse_fragment_table(moov_bytes, moof_pairs)
        return parse_video_track(moov_bytes)
    except UnindexableVideoError:
        if not served_from_cache:
            raise
        logger.warning(
            "mp4-index: {} failed to parse from its cached sidecar -- treating as a cache miss "
            "and refetching from the source",
            media_location,
        )
        moov_bytes, _ = _ensure_moov(
            media_location,
            moov_sidecar_location,
            file_size=file_size,
            skip_existing_sidecars=False,
            output_storage_client=output_storage_client,
            input_transport_params=input_transport_params,
            output_transport_params=output_transport_params,
        )
        if container_kind == CONTAINER_KIND_FRAGMENTED:
            moof_pairs, _ = _ensure_moof(
                media_location,
                moof_table_location,
                file_size=file_size,
                chunk_bytes=chunk_bytes,
                skip_existing_sidecars=False,
                output_storage_client=output_storage_client,
                input_transport_params=input_transport_params,
                output_transport_params=output_transport_params,
            )
            return parse_fragment_table(moov_bytes, moof_pairs)
        return parse_video_track(moov_bytes)


def process_one_video(
    row: dict[str, Any],
    *,
    config: ResolvedMp4IndexConfig,
    input_storage_client: StorageClient | None = None,
    output_storage_client: StorageClient | None = None,
) -> dict[str, Any]:
    """Discover, extract (with caching), and index one video, publishing its frame-index + metadata sidecars.

    Returns every ``VIDEO_IDENTITY_SCHEMA`` field on success (``None`` on failure) plus
    ``status``/``error_type``/``error_message`` bookkeeping.

    Args:
        row: ``{"video_id": ..., "media_location": ...}`` -- ``video_id`` is read back here, not
            recomputed; ``pipeline.py``'s own ``discover_rows`` computes it once, upfront.
        config: the run's own resolved config.
        input_storage_client: used for every read of *media_location*. ``None`` treats it as local.
        output_storage_client: used for every sidecar existence check/read/write. ``None`` treats
            every sidecar location as local.

    Known per-video errors are caught into a ``"failed"`` result rather than failing the whole
    Ray Data task; anything else propagates.

    """
    container_kind = config.input.container_kind
    chunk_bytes = config.execution.chunk_bytes
    session_id_path_depth = config.execution.session_id_path_depth
    frame_index_root = config.output.frame_index_root
    metadata_root = config.output.metadata_root
    moov_sidecar_root = config.output.moov_sidecar_root
    moof_table_root = config.output.moof_table_root
    skip_existing_sidecars = config.execution.skip_existing_sidecars
    extraction_contract_id = config.extraction_contract_id

    media_location = row["media_location"]
    video_id = row["video_id"]
    input_transport_params = (
        get_smart_open_client_params(input_storage_client)["transport_params"]
        if input_storage_client is not None
        else None
    )
    output_transport_params = (
        get_smart_open_client_params(output_storage_client)["transport_params"]
        if output_storage_client is not None
        else None
    )

    try:
        source_session_id, camera_key = derive_session_and_camera(
            media_location, session_id_path_depth=session_id_path_depth
        )
        session_camera_id = make_session_camera_id(source_session_id, camera_key)
        frame_index_location = f"{frame_index_root.rstrip('/')}/{session_camera_id}.arrow"
        metadata_location = f"{metadata_root.rstrip('/')}/{session_camera_id}.arrow"

        file_size = _file_size(media_location, input_storage_client)
        moov_sidecar_location = f"{moov_sidecar_root.rstrip('/')}/{video_id}.moov"
        moof_table_location = f"{moof_table_root.rstrip('/')}/{video_id}.arrow"
        video_track = _extract_video_track(
            media_location,
            moov_sidecar_location,
            moof_table_location,
            container_kind=container_kind,
            file_size=file_size,
            chunk_bytes=chunk_bytes,
            skip_existing_sidecars=skip_existing_sidecars,
            output_storage_client=output_storage_client,
            input_transport_params=input_transport_params,
            output_transport_params=output_transport_params,
        )
        metadata = Mp4Metadata.from_video_track(video_track)

        frame_rows = _frame_index_rows(video_track)
        write_frame_index(frame_rows, frame_index_location, transport_params=output_transport_params)
        write_video_metadata(attrs.asdict(metadata), metadata_location, transport_params=output_transport_params)
    except (
        UnindexableVideoError,
        UnresolvableMediaLocationError,
        FileNotFoundError,
        pa.ArrowInvalid,
        OverflowError,
    ) as exc:
        logger.warning("mp4-index: {} could not be indexed: {}", media_location, exc)
        return {
            "status": "failed",
            "video_id": video_id,
            "media_location": media_location,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            **dict.fromkeys(_VIDEO_ONLY_FIELDS),
        }

    return {
        "status": "success",
        "video_id": video_id,
        "media_location": media_location,
        "source_session_id": source_session_id,
        "camera_key": camera_key,
        "session_camera_id": session_camera_id,
        "media_size_bytes": file_size,
        "frame_index_location": frame_index_location,
        "metadata_location": metadata_location,
        "moov_location": moov_sidecar_location,
        "moof_table_location": moof_table_location if container_kind == CONTAINER_KIND_FRAGMENTED else None,
        "container_kind": container_kind,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "extraction_contract_id": extraction_contract_id,
        "error_type": None,
        "error_message": None,
    }


class _ProcessOneVideo:
    """Ray Data actor UDF wrapping :func:`process_one_video`.

    ``_storage_clients`` caches one client per URI scheme, not a single client: nothing in
    ``input.uris`` stops one run from mixing local and ``s3://``/``az://`` locations.
    """

    def __init__(self, *, config: ResolvedMp4IndexConfig) -> None:
        self._config = config
        self._storage_clients: dict[str, StorageClient | None] = {}

    def _client_for(self, location: str) -> StorageClient | None:
        scheme = urlsplit(location).scheme
        if scheme not in self._storage_clients:
            self._storage_clients[scheme] = get_storage_client(
                location, profile_name=self._config.execution.storage_profile
            )
        return self._storage_clients[scheme]

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        return process_one_video(
            row,
            config=self._config,
            input_storage_client=self._client_for(row["media_location"]),
            output_storage_client=self._client_for(self._config.output.moov_sidecar_root),
        )
