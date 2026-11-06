# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output MP4 indexes and episode clocks, committed before episode publication.

Each camera's one-row manifest extends mp4-index's schema with presentation-order timing provenance.
Media and moov are durable before the complete index manifest is created. Published artifacts
and source recordings must remain immutable. Incomplete attempts can be replayed.
"""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from lance.fragment import write_fragments

from cosmos_curator.core.sensors.data.streaming_camera_data import StreamingCameraData
from cosmos_curator.core.sensors.sampling.grid import make_ts_grid
from cosmos_curator.core.sensors.utils.validation import require_nondecreasing, require_strictly_increasing
from cosmos_curator.core.sensors.utils.video import (
    StreamingVideoDecoder,
    execute_streaming_decode_plan,
    make_streaming_decode_plan,
)
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.mp4_index.config import ResolvedMp4IndexConfig
from cosmos_curator.next.recipes.mp4_index.contracts import (
    VIDEO_MANIFEST_SCHEMA,
    make_session_camera_id,
    make_video_id,
)
from cosmos_curator.next.recipes.mp4_index.index_lookup import Mp4IndexLookup
from cosmos_curator.next.recipes.mp4_index.processing import process_one_video, result_table
from cosmos_curator.next.recipes.multimodal_split.records import resolve_camera_media
from cosmos_curator.next.recipes.multimodal_split.storage import open_sensor_source
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal, validate_table
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION, open_dataset
from cosmos_curator.next.utils.storage import write_if_absent
from cosmos_curator.next.utils.uris import normalize_location

OUTPUT_INDEX_SCHEMA = pa.schema(
    [
        *VIDEO_MANIFEST_SCHEMA,
        pa.field(
            "alignment_timestamps_ns", pa.large_list(pa.field("item", pa.int64(), nullable=False)), nullable=False
        ),
        pa.field("source_timestamps_ns", pa.large_list(pa.field("item", pa.int64(), nullable=False)), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
        pa.field("moov_sha256", pa.string(), nullable=False),
        pa.field("split_contract_digest", pa.string(), nullable=False),
    ]
)


def output_index_uri(media_location: str) -> str:
    """Return this camera's one-row manifest without cross-camera commit contention."""
    return f"{media_location}.index.lance"


def _digest(table: pa.Table) -> str:
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_stream(buffer, OUTPUT_INDEX_SCHEMA) as writer:
        writer.write_table(table.select(OUTPUT_INDEX_SCHEMA.names).cast(OUTPUT_INDEX_SCHEMA).combine_chunks())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _payload_sha256(location: str, storage_profile: str) -> str:
    with open_sensor_source(location, storage_profile=storage_profile, read_buffer_bytes=1024 * 1024) as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def find_output_index(media_location: str, *, uri: str, storage_profile: str) -> pa.Table | None:
    """Find exactly one committed output receipt, or no receipt for an unfinished attempt."""
    dataset = open_dataset(uri, storage_options=get_lance_storage_options(uri, profile_name=storage_profile))
    if dataset is None:
        return None
    validate_table(dataset, uri=uri, schema=OUTPUT_INDEX_SCHEMA, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    table = dataset.to_table(
        filter=f"video_id = {sql_string_literal(make_video_id(normalize_location(media_location)))}"
    )
    if len(table) > 1:
        msg = f"Duplicate output indexes for {media_location}"
        raise ValueError(msg)
    return table if len(table) else None


def index_descriptor(table: pa.Table, *, uri: str) -> dict[str, Any]:
    """Bind a camera receipt to its MP4 and exact index contents."""
    return {
        "media_size_bytes": table["media_size_bytes"][0].as_py(),
        "media_sha256": table["media_sha256"][0].as_py(),
        "mp4_index_uri": uri,
        "index_sha256": _digest(table),
    }


def validate_output_index(table: pa.Table, *, uri: str, storage_profile: str) -> None:
    """Reject incomplete payloads and timing before a receipt is reused or read."""
    row = table.to_pylist()[0]
    count = row["metadata"]["frame_count"]
    alignment = np.asarray(row["alignment_timestamps_ns"], dtype=np.int64)
    source = np.asarray(row["source_timestamps_ns"], dtype=np.int64)
    if not count or len(alignment) != count or len(source) != count:
        msg = "Output index timing arrays must describe every encoded frame"
        raise ValueError(msg)
    require_strictly_increasing("alignment_timestamps_ns", alignment)
    require_nondecreasing("source_timestamps_ns", source)
    Mp4IndexLookup.open_required(uri, storage_profile=storage_profile).require_indexed_video_for(
        row["media_location"], storage_profile=storage_profile
    )


def publish_output_index(  # noqa: PLR0913 -- one encoded output and its two clocks/identity
    local: Path,
    *,
    destination: str,
    uri: str,
    session_id: str,
    camera_key: str,
    alignment_timestamps_ns: npt.NDArray[np.int64],
    source_timestamps_ns: npt.NDArray[np.int64],
    split_contract_digest: str,
    storage_profile: str,
) -> dict[str, Any]:
    """Index the newly encoded file, then commit media, moov, and the canonical receipt."""
    config = ResolvedMp4IndexConfig.model_validate(
        {
            "schema_version": 1,
            "kind": "mp4-index",
            "input": {"uris": [str(local)], "container_kind": "nonfragmented"},
            "output": {"path": str(local.parent / "index")},
            "execution": {"skip_existing_sidecars": False},
        }
    )
    result = process_one_video({"media_location": str(local), "video_id": make_video_id(str(local))}, config=config)
    if result["status"] != "success":
        msg = f"Could not index output MP4: {result['error_message']}"
        raise ValueError(msg)
    moov = Path(result["moov_location"])
    media_sha256 = _payload_sha256(str(local), storage_profile)
    moov_sha256 = _payload_sha256(str(moov), storage_profile)
    location = normalize_location(destination)
    moov_location = f"{destination}.moov"
    result.update(
        video_id=make_video_id(location),
        media_location=location,
        source_session_id=session_id,
        camera_key=camera_key,
        session_camera_id=make_session_camera_id(session_id, camera_key),
        moov_location=moov_location,
        alignment_timestamps_ns=alignment_timestamps_ns,
        source_timestamps_ns=source_timestamps_ns,
        media_sha256=media_sha256,
        moov_sha256=moov_sha256,
        split_contract_digest=split_contract_digest,
    )
    table = result_table(result, schema=OUTPUT_INDEX_SCHEMA)
    count = result["source_frame_count"]
    if len(alignment_timestamps_ns) != count or len(source_timestamps_ns) != count:
        msg = "Output index frame count differs from the sampled window"
        raise ValueError(msg)
    require_strictly_increasing("alignment_timestamps_ns", alignment_timestamps_ns)
    require_nondecreasing("source_timestamps_ns", source_timestamps_ns)
    existing = find_output_index(destination, uri=uri, storage_profile=storage_profile)
    if existing is not None and _digest(existing) != _digest(table):
        msg = f"Output index conflicts with the committed receipt for {destination}"
        raise ValueError(msg)
    for target, payload, expected_sha256 in (
        (moov_location, moov, moov_sha256),
        (destination, local, media_sha256),
    ):
        if (
            not write_if_absent(target, payload, storage_profile=storage_profile)
            and _payload_sha256(target, storage_profile) != expected_sha256
        ):
            msg = f"Output payload already exists with different bytes: {target}"
            raise ValueError(msg)
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    try:
        # This only stages data files; publication below is create-only.
        fragments = write_fragments(
            table.select(OUTPUT_INDEX_SCHEMA.names).cast(OUTPUT_INDEX_SCHEMA),
            uri,
            schema=OUTPUT_INDEX_SCHEMA,
            mode="create",
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
            storage_options=storage_options,
        )
        transaction = lance.Transaction(
            read_version=0,
            operation=lance.LanceOperation.Overwrite(OUTPUT_INDEX_SCHEMA, fragments),
            transaction_properties={"kind": "multimodal-split", "operation": "publish-camera-index"},
        )
        # Never rebase this create-only transaction over another publisher's receipt.
        lance.LanceDataset.commit(
            uri,
            transaction,
            storage_options=storage_options,
            enable_v2_manifest_paths=True,
            max_retries=0,
        )
    except OSError:
        # A competing create or a lost commit response may have published the receipt.
        dataset = open_dataset(uri, storage_options=storage_options)
        if dataset is None:
            raise
        if dataset.count_rows() == 0:
            msg = f"Output index {uri} is empty from an older writer; use a fresh output directory"
            raise ValueError(msg) from None
    committed = find_output_index(destination, uri=uri, storage_profile=storage_profile)
    if committed is None or _digest(committed) != _digest(table):
        msg = f"Output index conflicts with the committed receipt for {destination}"
        raise ValueError(msg)
    return index_descriptor(committed, uri=uri)


def require_episode_index(episode: dict[str, Any], camera_key: str, *, storage_profile: str = "default") -> pa.Table:
    """Require the episode's exact index, clocks, and complete media payloads."""
    camera = resolve_camera_media(episode, camera_key)
    uri = str(camera["mp4_index_uri"])
    table = find_output_index(camera["media_location"], uri=uri, storage_profile=storage_profile)
    if table is None:
        msg = f"Missing output index for {camera['media_location']}"
        raise ValueError(msg)
    descriptor = index_descriptor(table, uri=uri)
    if descriptor != {name: camera[name] for name in descriptor}:
        msg = f"Changed output index for {camera['media_location']}"
        raise ValueError(msg)
    _, _, grid = make_ts_grid(
        episode["start_ns"],
        exclusive_end_ns=episode["exclusive_end_ns"],
        sample_rate_hz=episode["sampling_fps"],
    )
    if table["alignment_timestamps_ns"][0].as_py() != grid.tolist():
        msg = "Output index alignment timeline differs from its episode grid"
        raise ValueError(msg)
    if table["split_contract_digest"][0].as_py() != episode["split_contract_digest"]:
        msg = "Output index split contract differs from its episode"
        raise ValueError(msg)
    validate_output_index(table, uri=uri, storage_profile=storage_profile)
    return table


@contextmanager
def open_episode_camera(
    episode: dict[str, Any], camera_key: str, *, storage_profile: str = "default"
) -> Iterator[StreamingCameraData]:
    """Read indexed episode frames with their alignment and original source timestamps."""
    camera = resolve_camera_media(episode, camera_key)
    uri = str(camera["mp4_index_uri"])
    table = require_episode_index(episode, camera_key, storage_profile=storage_profile)
    row = table.to_pylist()[0]
    alignment = np.asarray(row["alignment_timestamps_ns"], dtype=np.int64)
    source = np.asarray(row["source_timestamps_ns"], dtype=np.int64)
    indexed = Mp4IndexLookup.open_required(uri, storage_profile=storage_profile).require_indexed_video_for(
        camera["media_location"], storage_profile=storage_profile
    )
    with open_sensor_source(camera["media_location"], storage_profile=storage_profile) as stream:
        decoder = StreamingVideoDecoder(stream, indexed.metadata, indexed.frame_index, moov_bytes=indexed.moov_bytes)
        plan = make_streaming_decode_plan(indexed.frame_index, np.arange(len(alignment), dtype=np.int64))
        frames = execute_streaming_decode_plan(decoder, plan)
        try:
            yield StreamingCameraData(
                alignment,
                source,
                walk=((int(alignment[position]), int(source[position]), pixels) for position, pixels in frames),
            )
        finally:
            frames.close()
            decoder.close()
