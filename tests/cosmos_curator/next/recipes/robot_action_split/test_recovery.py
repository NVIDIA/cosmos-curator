# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for driver-side reconciliation of discovered spans with committed clips."""

from pathlib import Path
from types import SimpleNamespace

import lance
import pytest

from cosmos_curator.next.recipes.robot_action_split import recovery
from cosmos_curator.next.recipes.robot_action_split.contracts import (
    CLIP_RECORD_SCHEMA_VERSION,
    MEDIA_CONTRACT_VERSION,
)
from cosmos_curator.next.recipes.robot_action_split.discovery import ChunkSpanBatch, SpanWorkItem
from cosmos_curator.next.recipes.robot_action_split.lance_sink import (
    append_clip_fragment,
    open_or_create_clip_table,
    write_clip_fragment,
)
from cosmos_curator.next.recipes.robot_action_split.records import clip_table
from cosmos_curator.next.recipes.robot_action_split.recovery import reconcile_batches

_CHUNK_URI = "s3://example-bucket/dataset/videos/observation.images.main/chunk-000/file-000.mp4"
_PARQUET_URI = "s3://example-bucket/dataset/data/chunk-000/file-000.parquet"


def _item(
    clip_id: str,
    *,
    span_group_id: str | None = None,
    view_name: str = "observation.images.main",
    task_description: str | None = "A person prepares coffee.",
) -> SpanWorkItem:
    return SpanWorkItem(
        source_id="source-0",
        span_group_id=span_group_id or f"span-{clip_id}",
        clip_id=clip_id,
        view_name=view_name,
        chunk_mp4_uri=_CHUNK_URI,
        data_parquet_uri=_PARQUET_URI,
        episode_index=0,
        episode_frame_base=0,
        frame_start=0,
        frame_end=100,
        native_fps=24.0,
        episode_from_timestamp=0.0,
        subtask_index=0,
        subtask_name="pick up coffee pod",
        subtask_label_resolved=True,
        task_index=0,
        task_name="make coffee",
        task_description=task_description,
        episode_id="ep_000",
        camera_intrinsics=None,
    )


def _outcome_row(item: SpanWorkItem) -> dict:
    return {
        "clip_id": item.clip_id,
        "span_group_id": item.span_group_id,
        "view_name": item.view_name,
        "source_id": item.source_id,
        "source_dataset": "test_dataset",
        "episode_id": item.episode_id,
        "episode_index": item.episode_index,
        "subtask_index": item.subtask_index,
        "subtask_name": item.subtask_name,
        "task_index": item.task_index,
        "task_name": item.task_name,
        "task_description": item.task_description,
        "frame_start": item.frame_start,
        "frame_end": item.frame_end,
        "start_ns": item.start_ns,
        "end_ns": item.end_ns,
        "native_fps": item.native_fps,
        "episode_from_timestamp": item.episode_from_timestamp,
        "clip_uri": f"s3://example-bucket/clips/{item.clip_id}.mp4",
        "action_data_uri": f"s3://example-bucket/action/{item.span_group_id}.bin",
        "camera_motion_annotation": None,
        "status": "success",
    }


def _commit(uri: str, *items: SpanWorkItem) -> None:
    open_or_create_clip_table(uri=uri, storage_profile="default")
    table = clip_table(
        [_outcome_row(item) for item in items],
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    candidate = write_clip_fragment(table, uri=uri, storage_profile="default")
    assert candidate is not None
    append_clip_fragment(candidate, uri=uri, storage_profile="default", attempts=1)


def test_unknown_batch_is_returned_unchanged(tmp_path: Path) -> None:
    """A batch with no committed clips is untouched and counted as unknown."""
    uri = str(tmp_path / "clips.lance")
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[_item("a"), _item("b")])

    reconciled = reconcile_batches([batch], dataset=dataset)

    assert reconciled.unknown_batches == 1
    assert reconciled.complete_batches == 0
    assert reconciled.partial_batches == 0
    assert reconciled.committed_clip_rows == 0
    assert len(reconciled.batches) == 1
    assert [item.clip_id for item in reconciled.batches[0].items] == ["a", "b"]


def test_fully_committed_batch_is_dropped(tmp_path: Path) -> None:
    """A batch whose every clip is already committed is skipped entirely."""
    uri = str(tmp_path / "clips.lance")
    item_a, item_b = _item("a"), _item("b")
    _commit(uri, item_a, item_b)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[item_a, item_b])

    reconciled = reconcile_batches([batch], dataset=dataset)

    assert reconciled.complete_batches == 1
    assert reconciled.partial_batches == 0
    assert reconciled.unknown_batches == 0
    assert reconciled.committed_clip_rows == 2
    assert reconciled.batches == ()


def test_partially_committed_batch_keeps_only_missing_items(tmp_path: Path) -> None:
    """A batch with one committed view and one missing view processes only the gap."""
    uri = str(tmp_path / "clips.lance")
    committed_item = _item("a", span_group_id="span-shared", view_name="observation.images.main")
    missing_item = _item("b", span_group_id="span-shared", view_name="observation.images.wrist")
    _commit(uri, committed_item)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(
        chunk_mp4_uri=_CHUNK_URI,
        data_parquet_uri=_PARQUET_URI,
        items=[committed_item, missing_item],
    )

    reconciled = reconcile_batches([batch], dataset=dataset)

    assert reconciled.partial_batches == 1
    assert reconciled.complete_batches == 0
    assert reconciled.unknown_batches == 0
    assert len(reconciled.batches) == 1
    assert [item.clip_id for item in reconciled.batches[0].items] == ["b"]


def test_multiple_batches_are_classified_independently(tmp_path: Path) -> None:
    """Complete, partial, and unknown batches in one run are each classified correctly."""
    uri = str(tmp_path / "clips.lance")
    complete_item = _item("complete")
    partial_committed = _item("partial-committed", span_group_id="span-partial")
    partial_missing = _item("partial-missing", span_group_id="span-partial", view_name="observation.images.wrist")
    unknown_item = _item("unknown")
    _commit(uri, complete_item, partial_committed)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")

    batches = [
        ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[complete_item]),
        ChunkSpanBatch(
            chunk_mp4_uri=_CHUNK_URI,
            data_parquet_uri=_PARQUET_URI,
            items=[partial_committed, partial_missing],
        ),
        ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[unknown_item]),
    ]

    reconciled = reconcile_batches(batches, dataset=dataset)

    assert reconciled.complete_batches == 1
    assert reconciled.partial_batches == 1
    assert reconciled.unknown_batches == 1
    assert reconciled.committed_clip_rows == 2
    remaining_clip_ids = {item.clip_id for batch in reconciled.batches for item in batch.items}
    assert remaining_clip_ids == {"partial-missing", "unknown"}


def test_reconciliation_rejects_contract_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A committed clip whose stored media_contract_version no longer matches fails loudly."""
    uri = str(tmp_path / "clips.lance")
    item = _item("a")
    _commit(uri, item)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[item])

    import cosmos_curator.next.recipes.robot_action_split.recovery as recovery_mod  # noqa: PLC0415

    monkeypatch.setattr(recovery_mod, "MEDIA_CONTRACT_VERSION", MEDIA_CONTRACT_VERSION + 1)

    with pytest.raises(ValueError, match="media_contract_version"):
        reconcile_batches([batch], dataset=dataset)


def test_reconciliation_rejects_v3_record_schema_rows(tmp_path: Path) -> None:
    """Verify a schema-v3 row cannot silently reconcile under the current schema."""
    uri = str(tmp_path / "clips.lance")
    item = _item("v3-row")
    open_or_create_clip_table(uri=uri, storage_profile="default")
    stale_table = clip_table(
        [_outcome_row(item)],
        record_schema_version=3,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    stale_candidate = write_clip_fragment(stale_table, uri=uri, storage_profile="default")
    assert stale_candidate is not None
    append_clip_fragment(stale_candidate, uri=uri, storage_profile="default", attempts=1)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[item])

    with pytest.raises(ValueError, match="record_schema_version"):
        reconcile_batches([batch], dataset=dataset)


def test_reconciliation_rejects_contract_drift_with_disjoint_clip_ids(tmp_path: Path) -> None:
    """A stale-contract row is caught even when its clip_id has no overlap with this run's set.

    ``make_clip_id`` bakes ``media_contract_version`` into the digest itself, so
    a real stale-contract row's clip_id never appears in a fresh run's
    discovered set at all — the version check must not depend on first
    filtering by clip_id, or it silently never sees this row.
    """
    uri = str(tmp_path / "clips.lance")
    stale_item = _item("stale-clip-id")
    open_or_create_clip_table(uri=uri, storage_profile="default")
    stale_table = clip_table(
        [_outcome_row(stale_item)],
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION + 1,
    )
    stale_candidate = write_clip_fragment(stale_table, uri=uri, storage_profile="default")
    assert stale_candidate is not None
    append_clip_fragment(stale_candidate, uri=uri, storage_profile="default", attempts=1)
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")

    # A fresh run's discovered clip_id never collides with the stale row's, since
    # a real make_clip_id would fold the current (different) contract version in.
    fresh_item = _item("fresh-clip-id")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[fresh_item])

    with pytest.raises(ValueError, match="media_contract_version"):
        reconcile_batches([batch], dataset=dataset)


def test_duplicate_discovered_clip_id_is_rejected(tmp_path: Path) -> None:
    """Two freshly discovered items sharing a clip_id fail before either is cut."""
    uri = str(tmp_path / "clips.lance")
    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch_a = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[_item("dup")])
    batch_b = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[_item("dup")])

    with pytest.raises(ValueError, match="duplicate clip_id"):
        reconcile_batches([batch_a, batch_b], dataset=dataset)


def test_duplicate_committed_clip_id_is_rejected(tmp_path: Path) -> None:
    """Two committed rows sharing a clip_id violate the canonical uniqueness invariant."""
    uri = str(tmp_path / "clips.lance")
    item = _item("a")
    _commit(uri, item)
    # Force a second, independent commit of the same clip_id by bypassing the
    # idempotent-append presence check (writing a second fragment directly).
    table = clip_table(
        [_outcome_row(item)],
        record_schema_version=CLIP_RECORD_SCHEMA_VERSION,
        media_contract_version=MEDIA_CONTRACT_VERSION,
    )
    import lance  # noqa: PLC0415

    from cosmos_curator.next.recipes.robot_action_split.records import CLIP_SCHEMA  # noqa: PLC0415

    dataset = lance.dataset(uri)
    fragments = lance.fragment.write_fragments(table, uri, schema=CLIP_SCHEMA, mode="append")
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Append(fragments),
        transaction_properties={"kind": "test", "operation": "force-duplicate"},
    )
    lance.LanceDataset.commit(uri, transaction)

    dataset = open_or_create_clip_table(uri=uri, storage_profile="default")
    batch = ChunkSpanBatch(chunk_mp4_uri=_CHUNK_URI, data_parquet_uri=_PARQUET_URI, items=[item])

    with pytest.raises(ValueError, match="duplicate clip_id"):
        reconcile_batches([batch], dataset=dataset)


def test_reconciliation_does_not_push_discovered_ids_into_a_filter(tmp_path: Path) -> None:
    """The scan must be bounded by the table, not by the discovered id list.

    Pushing ``clip_id IN (...)`` sent one 64-char sha256 literal per discovered clip to Lance's SQL
    parser. At batch-1 scale (1.2M spans) that filter is ~81MB of SQL and the parser grew past 126GB
    RSS before being killed, so a resumed run could never reconcile -- the failure only appears once
    the run is large enough to matter.
    """
    uri = str(tmp_path / "clips.lance")
    committed = _item("committed-clip")
    _commit(uri, committed)

    dataset = lance.dataset(uri)
    scanner_calls: list[dict] = []
    original_scanner = dataset.scanner

    def recording_scanner(**kwargs: object) -> object:
        scanner_calls.append(kwargs)
        return original_scanner(**kwargs)

    monkeypatched = SimpleNamespace(scanner=recording_scanner, count_rows=dataset.count_rows)
    scanned = recovery._scan_committed(("committed-clip", "absent-clip"), dataset=monkeypatched)

    assert scanned.keys() == {"committed-clip"}
    assert scanner_calls, "expected _scan_committed to scan the table"
    assert all("filter" not in call for call in scanner_calls), (
        f"discovered ids must not be pushed into a filter, got: {scanner_calls}"
    )
