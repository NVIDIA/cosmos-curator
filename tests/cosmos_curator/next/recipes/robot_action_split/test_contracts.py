# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk work items, terminal chunk outcomes, and their expansion into clip rows."""

from collections.abc import Callable
from typing import Any

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_split.bindings import resolve_bindings
from cosmos_curator.next.recipes.robot_action_split.config import ResolvedRobotActionSplitConfig
from cosmos_curator.next.recipes.robot_action_split.contracts import (
    CHUNK_SCHEMA,
    CLIP_SCHEMA,
    INPUT_SCHEMA,
    expand_chunks,
    make_chunk_id,
    validate_chunk_results,
)
from cosmos_curator.next.recipes.robot_action_split.discovery import (
    ChunkSpanBatch,
    SpanWorkItem,
    decode_chunk,
    items_table,
)

_SPAN: dict[str, Any] = {
    "source_id": "source-0",
    "span_group_id": "span-0",
    "clip_id": "clip-0",
    "view_name": "observation.images.main",
    "chunk_mp4_uri": "s3://bucket/videos/chunk-000/file-000.mp4",
    "data_parquet_uri": "s3://bucket/data/chunk-000/file-000.parquet",
    "episode_index": 3,
    "episode_frame_base": 120,
    "frame_start": 130,
    "frame_end": 250,
    "native_fps": 29.97002997002997,
    "episode_from_timestamp": 4.004004004004004,
    "subtask_index": 1,
    "subtask_name": "open lid",
    "task_index": 0,
    "task_name": "make coffee",
    "task_description": "  Prepare coffee.\n",
    "subtask_label_resolved": True,
    "episode_id": "ep_003",
    "camera_intrinsics": [612.3456789012345, 0.0, 320.1, 0.0, 611.9, 240.7, 0.0, 0.0, 1.0],
}


def test_spec_binds_chunks_to_clip_rows(config: ResolvedRobotActionSplitConfig) -> None:
    """The chunk is the item; clip rows carry its ID, and the split settings form the contract."""
    spec = resolve_bindings(config).spec
    assert (spec.owner, spec.id_column, spec.error_column) == ("robot-action-split", "chunk_id", "error_stage")
    assert (spec.input_fields, spec.result_fields, spec.output_fields) == (INPUT_SCHEMA, CHUNK_SCHEMA, CLIP_SCHEMA)
    assert spec.result_contract == config.split_contract


def test_clip_rows_keep_the_identity_and_label_columns_consumers_read() -> None:
    """Captioning, embeddings, and curation read these columns by name, with these nullabilities."""
    assert not CLIP_SCHEMA.field("clip_id").nullable
    assert CLIP_SCHEMA.field("clip_id").type == pa.string()
    for name in ("task_name", "subtask_name", "source_dataset"):
        assert not CLIP_SCHEMA.field(name).nullable
    for name in ("clip_uri", "action_data_uri", "error_stage", "error_message"):
        assert CLIP_SCHEMA.field(name).nullable


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"camera_intrinsics": None, "task_description": None},
        {"native_fps": 30.0, "episode_from_timestamp": 0.0},
    ],
)
def test_work_items_round_trip_span_geometry_exactly(changes: dict[str, Any]) -> None:
    """Saved selections rebuild the same spans, including float frame rates and offsets."""
    span = SpanWorkItem(**(_SPAN | changes))
    batch = ChunkSpanBatch(chunk_mp4_uri=span.chunk_mp4_uri, data_parquet_uri=span.data_parquet_uri, items=[span])
    items = items_table([batch])

    assert items.schema.equals(INPUT_SCHEMA)
    (item,) = items.to_pylist()
    assert item["chunk_id"] == make_chunk_id(span.chunk_mp4_uri, span.data_parquet_uri)
    (decoded,) = decode_chunk(item).items
    assert decoded == span
    assert (decoded.start_ns, decoded.end_ns) == (span.start_ns, span.end_ns)


def test_chunk_identity_names_both_source_files() -> None:
    """A view's chunk MP4 and its data parquet together identify one work item."""
    chunk_id = make_chunk_id("s3://b/videos/v/chunk-000/file-000.mp4", "s3://b/data/chunk-000/file-000.parquet")
    assert chunk_id != make_chunk_id("s3://b/videos/v/chunk-000/file-001.mp4", "s3://b/data/chunk-000/file-000.parquet")
    assert chunk_id != make_chunk_id("s3://b/videos/v/chunk-000/file-000.mp4", "s3://b/data/chunk-000/file-001.parquet")


def test_valid_outcomes_pass_and_expand_to_one_row_per_clip(
    results: pa.Table, config: ResolvedRobotActionSplitConfig
) -> None:
    """Finished, partly failed, and failed chunks all expand to one canonical row per clip."""
    for batch in results.to_batches():
        validate_chunk_results(batch, media_root=config.output.media_root)
    rows = pa.concat_tables(expand_chunks(results, max_rows=4))

    assert rows.schema.equals(CLIP_SCHEMA)
    clips = [clip for outcome in results.to_pylist() for clip in outcome["clips"]]
    assert sorted(rows["clip_id"].to_pylist()) == sorted(clip["clip_id"] for clip in clips)
    failed = rows.filter(rows["error_stage"].is_valid()).to_pylist()
    assert sorted(row["error_stage"] for row in failed) == ["chunk-stage", "chunk-stage", "cut"]
    assert all(row["clip_uri"] is None and row["action_data_uri"] is None for row in failed)
    # Expansion pairs each clip with its own chunk, across the max_rows boundary too.
    owed = {outcome["chunk_id"]: {clip["clip_id"] for clip in outcome["clips"]} for outcome in results.to_pylist()}
    assert all(row["clip_id"] in owed[row["chunk_id"]] for row in rows.to_pylist())
    assert all(f"/{row['view_name']}/" in row["chunk_mp4_uri"] for row in rows.to_pylist())


def _with_first_clip(results: pa.Table, change: Callable[[dict[str, Any]], None]) -> pa.RecordBatch:
    outcomes = results.to_pylist()
    first = next(outcome for outcome in outcomes if outcome["error_stage"] is None)
    change(first["clips"][0])
    return pa.RecordBatch.from_pylist(outcomes, schema=CHUNK_SCHEMA)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda clip: clip.update(clip_uri=None), "finished with both payloads"),
        (lambda clip: clip.update(error_stage="cut"), "Clip errors need a message"),
        (lambda clip: clip.update(error_stage="cut", error_message="bad"), "without payloads"),
        (lambda clip: clip.update(frame_end=clip["frame_start"]), "at least one frame"),
        (lambda clip: clip.update(task_name=None), "require task_name"),
        # Payloads outside the bound media_root would publish rows the contract does not describe.
        (lambda clip: clip.update(clip_uri=clip["clip_uri"].replace("/media/", "/elsewhere/")), "written to"),
        (lambda clip: clip.update(clip_uri=clip["clip_uri"].replace(".mp4", "-other.mp4")), "written to"),
        (lambda clip: clip.update(action_data_uri="s3://bucket/action/span.pickle"), "beneath"),
    ],
)
def test_malformed_clip_outcomes_are_rejected(
    results: pa.Table, config: ResolvedRobotActionSplitConfig, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    """Only finished clips or described terminal errors can be checkpointed."""
    with pytest.raises(ValueError, match=message):
        validate_chunk_results(_with_first_clip(results, change), media_root=config.output.media_root)


def test_a_failed_chunk_must_fail_every_clip(results: pa.Table, config: ResolvedRobotActionSplitConfig) -> None:
    """A chunk-level error leaves no clip looking finished."""
    outcomes = results.to_pylist()
    finished = next(outcome for outcome in outcomes if outcome["error_stage"] is None)
    finished.update(error_stage="chunk-stage", error_message="missing")
    with pytest.raises(ValueError, match="must fail every clip"):
        validate_chunk_results(
            pa.RecordBatch.from_pylist(outcomes, schema=CHUNK_SCHEMA), media_root=config.output.media_root
        )


def test_a_chunk_must_owe_a_clip(results: pa.Table, config: ResolvedRobotActionSplitConfig) -> None:
    """An outcome without clips would leave its chunk without a canonical row."""
    outcomes = results.to_pylist()
    outcomes[0]["clips"] = []
    with pytest.raises(ValueError, match="at least one clip"):
        validate_chunk_results(
            pa.RecordBatch.from_pylist(outcomes, schema=CHUNK_SCHEMA), media_root=config.output.media_root
        )


def test_a_chunk_must_not_repeat_a_clip(results: pa.Table, config: ResolvedRobotActionSplitConfig) -> None:
    """Two outcomes for one clip would publish two rows that consumers join on one key."""
    outcomes = results.to_pylist()
    finished = next(outcome for outcome in outcomes if outcome["error_stage"] is None)
    finished["clips"].append(finished["clips"][0])
    with pytest.raises(ValueError, match="must not repeat a clip_id"):
        validate_chunk_results(
            pa.RecordBatch.from_pylist(outcomes, schema=CHUNK_SCHEMA), media_root=config.output.media_root
        )


# A saved selection's spans, as this format wrote them. ``spans_json`` is persisted in recovery
# workspaces, so a SpanWorkItem change that cannot decode this breaks resuming in-flight runs.
_SAVED_SPANS_JSON = (
    '[{"camera_intrinsics":null,"clip_id":"clip-0","episode_frame_base":0,"episode_from_timestamp":0.0,'
    '"episode_id":"ep_000","episode_index":0,"frame_end":100,"frame_start":0,"native_fps":24.0,'
    '"source_id":"source-0","span_group_id":"span-0","subtask_index":0,"subtask_label_resolved":true,'
    '"subtask_name":"pick up coffee pod","task_description":null,"task_index":0,"task_name":"make coffee",'
    '"view_name":"observation.images.main"}]'
)


def test_saved_selections_keep_decoding() -> None:
    """The persisted span format decodes, and a field a later version stopped writing is ignored."""
    item = {
        "chunk_id": "chunk-0",
        "chunk_mp4_uri": "s3://b/videos/chunk-000/file-000.mp4",
        "data_parquet_uri": "s3://b/data/chunk-000/file-000.parquet",
        "spans_json": _SAVED_SPANS_JSON,
    }
    (span,) = decode_chunk(item).items
    assert (span.clip_id, span.frame_end, span.chunk_mp4_uri) == ("clip-0", 100, item["chunk_mp4_uri"])

    retired = _SAVED_SPANS_JSON.replace('"camera_intrinsics":null,', '"camera_intrinsics":null,"retired_field":1,')
    assert decode_chunk(item | {"spans_json": retired}).items == [span]
