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
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import urlsplit

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import smart_open  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-untyped]

from cosmos_curator.core.sensors.data.mp4 import Mp4Metadata, video_track_sample_arrays
from cosmos_curator.core.sensors.mp4 import box_locate
from cosmos_curator.core.sensors.mp4.box_probe import check_frame_byte_ranges, index_from_boxes
from cosmos_curator.core.sensors.mp4.box_traversal import UuidCallbackError, UuidHandler
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
from cosmos_curator.next.recipes.mp4_index.config import Mp4IndexRunConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    CONTAINER_KIND_FRAGMENTED,
    FRAME_INDEX_SCHEMA,
    INDEXER_BOX_WALKER,
    INDEXER_PYAV,
    RECORD_SCHEMA_VERSION,
    VIDEO_MANIFEST_SCHEMA,
    VIDEO_METADATA_SCHEMA,
    Indexer,
    UnresolvableMediaLocationError,
    derive_session_and_camera,
    make_session_camera_id,
    read_arrow_sidecar,
    read_bytes_sidecar,
    write_arrow_sidecar,
    write_bytes_sidecar,
)

#: Every VIDEO_MANIFEST_SCHEMA field a failed video carries as None.
_VIDEO_ONLY_FIELDS = tuple(set(VIDEO_MANIFEST_SCHEMA.names) - {"video_id", "media_location"})


def failed_result(row: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Retain source fields and clear derived fields for a failed video."""
    return {
        "status": "failed",
        **row,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **dict.fromkeys(_VIDEO_ONLY_FIELDS),
    }


def _file_size(media_location: str, storage_client: StorageClient | None) -> int:
    """Return *media_location*'s byte size, local or ``s3://``."""
    if storage_client is not None:
        size_bytes = storage_client.stat(S3Prefix(media_location)).size_bytes
        if size_bytes is None:
            msg = f"{media_location}: storage backend reported no size for an object that exists"
            raise UnindexableVideoError(msg)
        return size_bytes
    return pathlib.Path(media_location).stat().st_size


def _standard_frame_columns(
    dts: npt.NDArray[np.int64],
    composition_offsets: npt.NDArray[np.int64] | None,
    keyframes: npt.NDArray[np.bool_],
    offset: npt.NDArray[np.int64],
    size: npt.NDArray[np.int64],
) -> pa.RecordBatch:
    """Build the frame columns owned by standard MP4 indexing."""
    arrays = {
        "dts_ticks": pa.array(dts, type=pa.int64()),
        "composition_offset_ticks": pa.nulls(len(dts), type=pa.int64())
        if composition_offsets is None
        else pa.array(composition_offsets),
        "is_keyframe": pa.array(keyframes),
        "offset": pa.array(offset),
        "size": pa.array(size),
    }
    return pa.RecordBatch.from_arrays(
        list(arrays.values()), schema=pa.schema([FRAME_INDEX_SCHEMA.field(name) for name in arrays])
    )


def _validate_frame_column(field: pa.Field, column: pa.Array) -> None:
    if field.name not in FRAME_INDEX_SCHEMA.names:
        msg = f"Unknown frame column {field.name!r}"
        raise ValueError(msg)
    expected = FRAME_INDEX_SCHEMA.field(field.name)
    if field.type != expected.type:
        msg = f"Frame column {field.name!r} has an incompatible type"
        raise ValueError(msg)
    if column.null_count and (not field.nullable or not expected.nullable):
        msg = f"Frame column {field.name!r} is non-nullable but contains nulls"
        raise ValueError(msg)


def _frame_index_array(standard: pa.RecordBatch, extension: pa.RecordBatch | None = None) -> pa.LargeListArray:
    """Merge disjoint frame batches and fill omitted nullable fields."""
    arrays: dict[str, pa.Array] = {}
    for batch in (standard, extension):
        if batch is None:
            continue
        if batch.num_rows != standard.num_rows:
            msg = f"Frame batch has {batch.num_rows} rows for {standard.num_rows} frames"
            raise ValueError(msg)
        for field, column in zip(batch.schema, batch.columns, strict=True):
            if field.name in arrays:
                msg = f"Frame column {field.name!r} is supplied more than once"
                raise ValueError(msg)
            _validate_frame_column(field, column)
            arrays[field.name] = column
    for field in FRAME_INDEX_SCHEMA:
        if field.name not in arrays:
            if not field.nullable:
                msg = f"Required frame column {field.name!r} is missing"
                raise ValueError(msg)
            arrays[field.name] = pa.nulls(standard.num_rows, type=field.type)
    values = pa.StructArray.from_arrays(
        [arrays[field.name] for field in FRAME_INDEX_SCHEMA], fields=list(FRAME_INDEX_SCHEMA)
    )
    return pa.LargeListArray.from_arrays(
        pa.array([0, standard.num_rows], type=pa.int64()), values, type=VIDEO_MANIFEST_SCHEMA.field("frames").type
    )


# Failure rows have absent derived fields; canonical successful rows retain their non-null schema.
_PROCESS_RESULT_SCHEMA = pa.schema(
    [pa.field(field.name, field.type) for field in VIDEO_MANIFEST_SCHEMA]
    + [pa.field(name, pa.string()) for name in ("status", "error_type", "error_message")]
)


def result_table(result: dict[str, Any], *, schema: pa.Schema = VIDEO_MANIFEST_SCHEMA) -> pa.Table:
    """Wrap an extraction result, retaining its nested Arrow arrays."""
    result_schema = (
        _PROCESS_RESULT_SCHEMA
        if schema == VIDEO_MANIFEST_SCHEMA
        else pa.schema(
            [pa.field(field.name, field.type) for field in schema]
            + [pa.field(name, pa.string()) for name in ("status", "error_type", "error_message")]
        )
    )
    return pa.Table.from_arrays(
        [
            result[field.name]
            if isinstance(result.get(field.name), pa.Array)
            else pa.array([result.get(field.name)], type=field.type)
            for field in result_schema
        ],
        schema=result_schema,
    )


class _CachedBoxReadError(Exception):
    """A selected moov/moof sidecar could not be read or deserialized."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class _ExtensionSetupError(Exception):
    """Keep setup refusals outside cache recovery while retaining per-video handling."""

    def __init__(self, error: UnindexableVideoError) -> None:
        super().__init__(str(error))
        self.error = error


class _ExtensionContractError(Exception):
    """Keep extension bugs outside per-video failure and cache-recovery handling."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


def _moof_rows(moof_table_location: str, *, transport_params: dict[str, Any] | None) -> list[tuple[int, bytes]]:
    """Read a published ``MOOF_TABLE_SCHEMA`` sidecar back into ``(moof_start, moof_bytes)`` pairs, in file order."""
    table = read_arrow_sidecar(moof_table_location, transport_params=transport_params)
    # Validate cached values before handing them to the byte parser. Arrow IPC
    # can be readable while its columns violate the sidecar's required shape.
    for field in MOOF_TABLE_SCHEMA:
        column = table.column(field.name)
        if column.type != field.type or column.null_count:
            msg = f"Cached moof column {field.name!r} must contain non-null {field.type} values"
            raise pa.ArrowInvalid(msg)
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
        try:
            return read_bytes_sidecar(moov_sidecar_location, transport_params=output_transport_params), True
        except FileNotFoundError as exc:
            raise _CachedBoxReadError(exc) from exc
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
        try:
            return _moof_rows(moof_table_location, transport_params=output_transport_params), True
        except (FileNotFoundError, pa.ArrowInvalid, KeyError) as exc:
            raise _CachedBoxReadError(exc) from exc
    with smart_open.open(media_location, "rb", transport_params=input_transport_params) as handle:
        moof_pairs = list(walk_moofs(file_size, handle, chunk_bytes=chunk_bytes))
    if not moof_pairs:
        msg = f"{media_location}: input.container_kind=fragmented but no moof boxes found"
        raise NoFragmentsFoundError(msg)
    moof_rows = [{"moof_start": start, "moof_bytes": moof_bytes} for start, moof_bytes in moof_pairs]
    write_arrow_sidecar(moof_rows, MOOF_TABLE_SCHEMA, moof_table_location, transport_params=output_transport_params)
    return moof_pairs, False


@attrs.define(frozen=True)
class Mp4Extraction:
    """A parsed track and the moov/moof bytes used to build it."""

    media_location: str
    video_track: VideoTrack
    moov_bytes: bytes
    moof_rows: list[tuple[int, bytes]] | None = None


class Mp4IndexExtension(Protocol):
    """Register UUID decoders and contribute columns within the existing frame schema.

    PyAV does not read UUID boxes, so a run with an extension indexes with the box walker,
    which runs the decoders while it lists the track's samples.

    Report malformed source data through explicit ``UnindexableVideoError``
    subclasses. Contract violations and other unexpected errors fail the run.
    """

    @property
    def uuid_handlers(self) -> tuple[UuidHandler, ...]:
        """Return handlers local to this parsing attempt."""
        ...

    @property
    def frame_schema(self) -> pa.Schema:
        """Declare the columns produced after standard track parsing."""
        ...

    def frame_columns(self, extraction: Mp4Extraction) -> pa.RecordBatch:
        """Align collected box records with the track's decode-order frames."""
        ...


Mp4ExtensionFactory = Callable[[str], Mp4IndexExtension]


def _extension_frame_columns(extension: Mp4IndexExtension, extraction: Mp4Extraction) -> pa.RecordBatch:
    columns = extension.frame_columns(extraction)
    if not columns.schema.equals(extension.frame_schema):
        msg = "Extension frame columns do not match their declared schema"
        raise ValueError(msg)
    return columns


def indexer_for(extension_factory: Mp4ExtensionFactory | None) -> Indexer:
    """Return the indexer a run uses: the box walker exactly when an extension decodes UUID boxes."""
    return INDEXER_PYAV if extension_factory is None else INDEXER_BOX_WALKER


def _parse_mp4(
    media_location: str,
    moov_bytes: bytes,
    moof_rows: list[tuple[int, bytes]] | None,
    extension_factory: Mp4ExtensionFactory | None,
    *,
    file_size: int,
) -> tuple[Mp4Metadata, pa.LargeListArray]:
    if extension_factory is None:
        index = index_from_boxes(file_size, moov_bytes, moof_rows or ())
        probe = index.probe
        standard = _standard_frame_columns(
            probe.dts, probe.composition_offset, probe.is_keyframe, probe.offset, probe.size
        )
        return index.metadata, _frame_index_array(standard)
    try:
        extension = extension_factory(media_location)
        handlers = extension.uuid_handlers
    except UnindexableVideoError as exc:
        raise _ExtensionSetupError(exc) from exc
    except Exception as exc:
        raise _ExtensionContractError(exc) from exc
    try:
        track = (
            parse_video_track(moov_bytes, uuid_handlers=handlers)
            if moof_rows is None
            else parse_fragment_table(moov_bytes, moof_rows, uuid_handlers=handlers)
        )
    except UuidCallbackError as exc:
        if isinstance(exc.error, UnindexableVideoError):
            raise exc.error from exc
        raise _ExtensionContractError(exc.error) from exc
    check_frame_byte_ranges(track.offset, track.size, file_size)
    extraction = Mp4Extraction(media_location, track, moov_bytes, moof_rows)
    standard = _standard_frame_columns(*video_track_sample_arrays(track), track.offset, track.size)
    try:
        columns = _extension_frame_columns(extension, extraction)
        frames = _frame_index_array(standard, columns)
    except UnindexableVideoError:
        raise
    except Exception as exc:
        raise _ExtensionContractError(exc) from exc
    return Mp4Metadata.from_video_track(track), frames


def _extract_mp4(  # noqa: PLR0913 -- source, sidecars, transport, and parsing settings
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
    extension_factory: Mp4ExtensionFactory | None = None,
) -> tuple[Mp4Metadata, pa.LargeListArray]:
    """Parse standard and extension fields, refetching invalid cached boxes once."""
    for attempt in range(2):
        use_cache = skip_existing_sidecars and attempt == 0
        moov_from_cache = moof_from_cache = parsing = False
        try:
            moov_bytes, moov_from_cache = _ensure_moov(
                media_location,
                moov_sidecar_location,
                file_size=file_size,
                skip_existing_sidecars=use_cache,
                output_storage_client=output_storage_client,
                input_transport_params=input_transport_params,
                output_transport_params=output_transport_params,
            )
            moof_rows = None
            if container_kind == CONTAINER_KIND_FRAGMENTED:
                moof_rows, moof_from_cache = _ensure_moof(
                    media_location,
                    moof_table_location,
                    file_size=file_size,
                    chunk_bytes=chunk_bytes,
                    skip_existing_sidecars=use_cache,
                    output_storage_client=output_storage_client,
                    input_transport_params=input_transport_params,
                    output_transport_params=output_transport_params,
                )
            parsing = True
            return _parse_mp4(media_location, moov_bytes, moof_rows, extension_factory, file_size=file_size)
        except _ExtensionSetupError as exc:
            raise exc.error from exc
        except (UnindexableVideoError, pa.ArrowInvalid, OverflowError, _CachedBoxReadError) as exc:
            # A failure fetching boxes from the source is the source's own, whatever else came from cache.
            cache_at_fault = isinstance(exc, _CachedBoxReadError) or (parsing and (moov_from_cache or moof_from_cache))
            if attempt == 0 and cache_at_fault:
                logger.warning("mp4-index: {} has invalid cached boxes; refetching from the source", media_location)
                continue
            if isinstance(exc, _CachedBoxReadError):
                raise exc.error from exc
            raise
    msg = "MP4 extraction exhausted its retry without a result"
    raise RuntimeError(msg)


def process_one_video(
    row: dict[str, Any],
    *,
    config: Mp4IndexRunConfig,
    input_storage_client: StorageClient | None = None,
    output_storage_client: StorageClient | None = None,
    extension_factory: Mp4ExtensionFactory | None = None,
) -> dict[str, Any]:
    """Extract or reuse moov/moof sidecars and build one complete video record.

    PyAV indexes the frames from those boxes alone, so a video whose sidecars are cached is
    indexed without reading its media. With *extension_factory*, the box walker indexes
    instead, so the extension's UUID decoders run.

    Returns manifest fields plus ``status``, ``error_type``, and ``error_message``.
    ``metadata`` and ``frames`` are single-row Arrow arrays whose buffers are retained
    by ``result_table``. On failure, video-specific manifest fields are ``None``.

    Args:
        row: Discovered ``video_id`` and ``media_location``.
        config: Resolved recipe configuration.
        input_storage_client: Source-video reads; ``None`` for local files.
        output_storage_client: Moov/moof sidecar reads and writes; ``None`` for local files.
        extension_factory: Optional factory for an extension local to each parsing attempt.

    Known per-video errors produce a ``"failed"`` result; unexpected errors propagate.

    """
    container_kind = config.input.container_kind
    chunk_bytes = config.execution.chunk_bytes
    session_id_path_depth = config.execution.session_id_path_depth
    moov_sidecar_root = config.output.moov_sidecar_root
    moof_table_root = config.output.moof_table_root
    skip_existing_sidecars = config.execution.skip_existing_sidecars
    extraction_contract_id = config.extraction_contract_id(indexer=indexer_for(extension_factory))

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

        file_size = _file_size(media_location, input_storage_client)
        moov_sidecar_location = f"{moov_sidecar_root.rstrip('/')}/{video_id}.moov"
        moof_table_location = f"{moof_table_root.rstrip('/')}/{video_id}.arrow"
        metadata, frames = _extract_mp4(
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
            extension_factory=extension_factory,
        )
        metadata_array = pa.array([attrs.asdict(metadata)], type=pa.struct(VIDEO_METADATA_SCHEMA))
    except _ExtensionContractError as exc:
        raise exc.error from exc
    except (
        UnindexableVideoError,
        UnresolvableMediaLocationError,
        FileNotFoundError,
        pa.ArrowInvalid,
        OverflowError,
    ) as exc:
        logger.warning("mp4-index: {} could not be indexed: {}", media_location, exc)
        return failed_result({"video_id": video_id, "media_location": media_location}, exc)

    return {
        "status": "success",
        "video_id": video_id,
        "media_location": media_location,
        "source_session_id": source_session_id,
        "camera_key": camera_key,
        "session_camera_id": session_camera_id,
        "media_size_bytes": file_size,
        "source_width": metadata.width,
        "source_height": metadata.height,
        "source_frame_rate": metadata.avg_frame_rate_num / metadata.avg_frame_rate_den,
        "source_frame_count": metadata.frame_count,
        "source_duration_ns": metadata.duration_ns,
        "source_video_codec": metadata.codec_name,
        "metadata": metadata_array,
        "frames": frames,
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

    def __init__(
        self,
        *,
        config: Mp4IndexRunConfig,
        extension_factory: Mp4ExtensionFactory | None = None,
    ) -> None:
        self._config = config
        self._extension_factory = extension_factory
        self._storage_clients: dict[str, StorageClient | None] = {}

    def _client_for(self, location: str) -> StorageClient | None:
        scheme = urlsplit(location).scheme
        if scheme not in self._storage_clients:
            self._storage_clients[scheme] = get_storage_client(
                location, profile_name=self._config.execution.storage_profile
            )
        return self._storage_clients[scheme]

    def __call__(self, batch: pa.Table) -> pa.Table:
        # Input contains only source references; never materialize output frame lists in Python.
        results = [
            result_table(
                process_one_video(
                    row,
                    config=self._config,
                    input_storage_client=self._client_for(row["media_location"]),
                    output_storage_client=self._client_for(self._config.output.moov_sidecar_root),
                    extension_factory=self._extension_factory,
                )
            )
            for row in batch.to_pylist()
        ]
        return pa.concat_tables(results)
