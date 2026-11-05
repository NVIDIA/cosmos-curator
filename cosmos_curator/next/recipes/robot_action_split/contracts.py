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

"""Chunk work items, chunk outcomes, and canonical clip rows for ``robot-action-split``.

A work item is one source chunk: one view's chunk MP4 and the data parquet holding
its episodes, with the chunk's discovered spans carried as JSON. Processing stages
one terminal outcome per chunk holding every clip it owes, so a checkpointed chunk
never has partial clips. ``expand_chunks`` flattens outcomes into one row per clip.
A clip that could not be produced keeps its row with null ``clip_uri`` and
``action_data_uri`` and a non-null ``error_stage``. A chunk that failed as a whole
also sets ``error_stage`` on its outcome, which publication counts as a terminal
error, and every one of its clips carries that error.
"""

from collections.abc import Iterator

import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.utils.identity import canonical_digest
from cosmos_curator.next.utils.storage import artifact_uri

# Versions the generated MP4 byte contract and its matching clip identity.
# Bump when non-configurable cut semantics change such that the same span_group_id
# would produce different MP4 bytes (e.g. smart-cut -> full-reencode switch).
# Metadata-only Lance or sidecar field changes do not move this version.
MEDIA_CONTRACT_VERSION = 1

# Bump when the action binary format or field layout changes in a way that would
# produce different bytes for the same span/dataset (e.g. ACT2 header version bump,
# field reordering, or dtype change).
ACTION_CONTRACT_VERSION = 1

# Bounds the clip rows built at once from one block of chunk outcomes.
EXPANSION_ROWS = 100_000
# Bounds one terminal error's message as it is staged and reported.
_ERROR_MESSAGE_CHARS = 8192

INPUT_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string(), nullable=False),
        pa.field("chunk_mp4_uri", pa.string(), nullable=False),
        pa.field("data_parquet_uri", pa.string(), nullable=False),
        # The chunk's discovered spans as canonical JSON: the append input schema
        # holds only scalars, and JSON round-trips the float geometry exactly.
        pa.field("spans_json", pa.large_string(), nullable=False),
    ]
)
_CHUNK_FIELDS = [INPUT_SCHEMA.field(name) for name in ("chunk_id", "chunk_mp4_uri", "data_parquet_uri")]
_ERROR_FIELDS = [pa.field("error_stage", pa.string()), pa.field("error_message", pa.string())]
CLIP_FIELDS = [
    pa.field("clip_id", pa.string(), nullable=False),
    pa.field("span_group_id", pa.string(), nullable=False),
    pa.field("view_name", pa.string(), nullable=False),
    pa.field("source_id", pa.string(), nullable=False),
    pa.field("source_dataset", pa.string(), nullable=False),
    pa.field("episode_id", pa.string(), nullable=False),
    pa.field("episode_index", pa.int32(), nullable=False),
    pa.field("subtask_index", pa.int32(), nullable=False),
    pa.field("subtask_name", pa.string(), nullable=False),
    pa.field("task_index", pa.int32(), nullable=False),
    pa.field("task_name", pa.string(), nullable=False),
    pa.field("task_description", pa.large_string()),
    pa.field("frame_start", pa.int32(), nullable=False),
    pa.field("frame_end", pa.int32(), nullable=False),
    pa.field("start_ns", pa.int64(), nullable=False),
    pa.field("end_ns", pa.int64(), nullable=False),
    pa.field("native_fps", pa.float64(), nullable=False),
    pa.field("episode_from_timestamp", pa.float64(), nullable=False),
    # Null exactly when the clip has a terminal error.
    pa.field("clip_uri", pa.large_string()),
    pa.field("action_data_uri", pa.large_string()),
    # Genuinely optional even for a successful clip: absent when the source
    # has no camera-trajectory data to describe, not a failure signal.
    pa.field("camera_motion_annotation", pa.large_string()),
    *_ERROR_FIELDS,
]
_CLIP_NAMES = frozenset(field.name for field in CLIP_FIELDS)
_REQUIRED_CLIP_NAMES = tuple(field.name for field in CLIP_FIELDS if not field.nullable)
CHUNK_SCHEMA = pa.schema(
    [
        *_CHUNK_FIELDS,
        pa.field("clips", pa.list_(pa.struct(CLIP_FIELDS)), nullable=False),
        # Non-null when the chunk failed as a whole, for example a missing source;
        # every clip then carries the same error.
        *_ERROR_FIELDS,
    ]
)
# The canonical clip table: one row per clip, successful or not, carrying its chunk.
CLIP_SCHEMA = pa.schema([*CLIP_FIELDS[:-2], *_CHUNK_FIELDS, *_ERROR_FIELDS])


def make_chunk_id(chunk_mp4_uri: str, data_parquet_uri: str) -> str:
    """Identify one view's chunk MP4 with the data parquet holding its episodes."""
    return canonical_digest({"chunk_mp4_uri": chunk_mp4_uri, "data_parquet_uri": data_parquet_uri})


def error_details(error_stage: str, error_message: str) -> dict[str, str]:
    """Bound one terminal error's description as it is staged and reported."""
    return {"error_stage": error_stage, "error_message": error_message[:_ERROR_MESSAGE_CHARS]}


def _require(condition: pa.Array | pa.ChunkedArray | pa.Scalar, message: str) -> None:
    if not pc.all(condition).as_py():
        raise ValueError(message)


def validate_chunk_results(batch: pa.RecordBatch, *, media_root: str) -> None:
    """Accept only terminal chunk outcomes whose clips are each finished or failed.

    Every chunk owes at least one clip. A finished clip carries both payload URIs,
    beneath ``media_root`` where processing writes them; a failed clip names its
    stage and message and carries neither payload. A chunk that failed as a whole
    fails every clip. Operational failures never reach a result.
    """
    if not batch.num_rows:
        return
    clips_column = batch.column("clips")
    chunk_failed = pc.is_valid(batch.column("error_stage"))
    _require(pc.equal(chunk_failed, pc.is_valid(batch.column("error_message"))), "Chunk errors need a message")
    _require(pc.greater(pc.list_value_length(clips_column), 0), "Chunk outcomes must list at least one clip")

    clips = pc.list_flatten(clips_column)
    clip_ids = pc.struct_field(clips, "clip_id")
    if pc.count_distinct(clip_ids).as_py() != len(clip_ids):
        msg = "Chunk outcomes must not repeat a clip_id"
        raise ValueError(msg)
    for name in _REQUIRED_CLIP_NAMES:
        if pc.struct_field(clips, name).null_count:
            msg = f"Clip outcomes require {name}"
            raise ValueError(msg)
    clip_failed = pc.is_valid(pc.struct_field(clips, "error_stage"))
    _require(pc.equal(clip_failed, pc.is_valid(pc.struct_field(clips, "error_message"))), "Clip errors need a message")
    payloads = [pc.is_valid(pc.struct_field(clips, name)) for name in ("clip_uri", "action_data_uri")]
    finished = pc.and_(*payloads)
    no_outputs = pc.and_(
        pc.and_(pc.invert(payloads[0]), pc.invert(payloads[1])),
        pc.is_null(pc.struct_field(clips, "camera_motion_annotation")),
    )
    _require(
        pc.if_else(clip_failed, no_outputs, finished),
        "Clips must be finished with both payloads or described terminal errors without payloads",
    )
    root = artifact_uri(media_root)
    expected_clip_uris = pc.binary_join_element_wise(
        f"{root}/video/", pc.struct_field(clips, "view_name"), "/", clip_ids, ".mp4", ""
    )
    _require(
        # Failed clips carry no payload URI, so a null comparison passes.
        pc.fill_null(
            pc.equal(pc.cast(pc.struct_field(clips, "clip_uri"), pa.string()), expected_clip_uris), fill_value=True
        ),
        f"Finished clips must be written to {root}/video/<view_name>/<clip_id>.mp4",
    )
    _require(
        pc.fill_null(pc.starts_with(pc.struct_field(clips, "action_data_uri"), f"{root}/action/"), fill_value=True),
        f"Finished clips must reference action payloads beneath {root}/action/",
    )
    _require(
        pc.greater(pc.struct_field(clips, "frame_end"), pc.struct_field(clips, "frame_start")),
        "Clips must span at least one frame",
    )
    failed_with_chunk = pc.take(chunk_failed, pc.list_parent_indices(clips_column))
    _require(pc.or_(pc.invert(failed_with_chunk), clip_failed), "A failed chunk must fail every clip")


def expand_chunks(chunks: pa.Table, *, max_rows: int = EXPANSION_ROWS) -> Iterator[pa.Table]:
    """Flatten chunk outcomes into one canonical row per clip, including failed clips.

    Every chunk lists at least one clip, so every chunk remains present in
    current Lance rows after history and checkpoints are pruned.
    """
    clips = pc.list_flatten(chunks["clips"])
    parents = pc.list_parent_indices(chunks["clips"])
    for offset in range(0, len(clips), max_rows):
        items = clips.slice(offset, max_rows)
        indices = parents.slice(offset, max_rows)
        yield pa.Table.from_arrays(
            [
                pc.struct_field(items, field.name)
                if field.name in _CLIP_NAMES
                else pc.take(chunks[field.name], indices)
                for field in CLIP_SCHEMA
            ],
            schema=CLIP_SCHEMA,
        )
