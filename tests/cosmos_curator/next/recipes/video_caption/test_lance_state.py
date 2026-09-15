# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for field registration and finite-attempt fragment classification."""

from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.contracts import CaptionModelSpec, terminal_metadata
from cosmos_curator.next.recipes.video_caption.lance_state import (
    CaptionAttempt,
    FragmentCaptionState,
    capture_attempt,
    classify_fragment,
    ensure_caption_fields,
    leaf_field_ids,
    validate_caption_input,
)


def _registered(
    factory: Callable[..., tuple[str, lance.LanceDataset]],
    bindings: CaptionRunBindings,
    *,
    count: int = 2,
    rows_per_fragment: int = 2,
) -> tuple[str, lance.LanceDataset]:
    uri, _ = factory(count=count, rows_per_fragment=rows_per_fragment)
    return uri, ensure_caption_fields(uri, storage_options=None, bindings=bindings, attempts=3)


def _commit_terminal_rows(
    dataset: lance.LanceDataset,
    spec: CaptionModelSpec,
    digest: str,
    *,
    rows: int,
) -> lance.LanceDataset:
    fragment = dataset.get_fragment(0)
    assert fragment is not None
    values = [
        {
            "clip_id": f"clip-{index}",
            spec.caption_field_name: f"caption {index}",
            spec.metadata_field_name: terminal_metadata(
                spec,
                digest,
                status="success",
                prompt_token_count=4,
                generated_token_count=8,
                error_type=None,
                error_message=None,
            ),
        }
        for index in range(rows)
    ]
    schema = pa.schema([pa.field("clip_id", pa.string(), nullable=False), spec.caption_field, spec.metadata_field])
    table = pa.Table.from_pylist(values, schema=schema)
    metadata, field_ids = fragment.update_columns(
        pa.RecordBatchReader.from_batches(schema, table.to_batches()),
        left_on="clip_id",
        right_on="clip_id",
    )
    transaction = lance.Transaction(
        read_version=dataset.version,
        operation=lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=field_ids),
    )
    return lance.LanceDataset.commit(dataset.uri, transaction, max_retries=0)


def test_registration_adds_both_fields_in_one_version_and_is_idempotent(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Registration is one atomic schema version and an exact retry is a no-op."""
    uri, initial = clip_dataset_factory()

    registered = ensure_caption_fields(uri, storage_options=None, bindings=caption_bindings, attempts=3)
    reopened = ensure_caption_fields(uri, storage_options=None, bindings=caption_bindings, attempts=3)

    assert registered.version == initial.version + 1
    assert reopened.version == registered.version
    for expected in caption_spec.fields:
        assert registered.schema.field(expected.name).equals(expected, check_metadata=True)
    expected_ids = leaf_field_ids(registered, (caption_spec.caption_field_name, caption_spec.metadata_field_name))
    assert len(expected_ids) == 11


def test_partial_or_conflicting_field_sets_fail(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
) -> None:
    """A field set cannot be partial or reuse owned names with different types."""
    partial_uri, partial = clip_dataset_factory()
    partial.add_columns(pa.schema([caption_spec.caption_field]))
    with pytest.raises(ValueError, match="partially present"):
        ensure_caption_fields(partial_uri, storage_options=None, bindings=caption_bindings, attempts=1)

    conflict_uri, conflict = clip_dataset_factory()
    conflict.add_columns(
        pa.schema(
            [
                pa.field(caption_spec.caption_field_name, pa.string()),
                caption_spec.metadata_field,
            ]
        )
    )
    with pytest.raises(ValueError, match="incompatible"):
        ensure_caption_fields(conflict_uri, storage_options=None, bindings=caption_bindings, attempts=1)


def test_capture_attempt_classifies_pending_and_complete_fragments(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Each fragment in a pinned snapshot is exactly pending or complete."""
    _, dataset = _registered(clip_dataset_factory, caption_bindings, count=3, rows_per_fragment=2)
    dataset = _commit_terminal_rows(dataset, caption_spec, caption_digest, rows=2)

    attempt = capture_attempt(dataset, bindings=caption_bindings)

    assert attempt == CaptionAttempt(version=dataset.version, pending_fragment_ids=(1,), complete_fragment_ids=(0,))
    fragment = dataset.get_fragment(0)
    assert fragment is not None
    assert classify_fragment(fragment, bindings=caption_bindings) is FragmentCaptionState.COMPLETE


def test_pending_fragment_classification_reads_only_result_columns(
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Pending classification does not read or project media input columns."""
    table = pa.Table.from_pylist(
        [{caption_spec.caption_field_name: None, caption_spec.metadata_field_name: None}],
        schema=caption_spec.fields,
    )
    scans = 0

    def scan(*, columns: list[str]) -> SimpleNamespace:
        nonlocal scans
        scans += 1
        assert columns == [caption_spec.caption_field_name, caption_spec.metadata_field_name]
        return SimpleNamespace(to_batches=table.to_batches)

    def unexpected_projection(_batch: pa.Table) -> pa.Table:
        pytest.fail("pending fragment input must not be projected")

    fragment = SimpleNamespace(fragment_id=0, scanner=scan)
    bindings = replace(
        caption_bindings,
        input=replace(caption_bindings.input, project_batch=unexpected_projection),
    )

    assert classify_fragment(fragment, bindings=bindings) is FragmentCaptionState.PENDING  # type: ignore[arg-type]
    assert scans == 1


def test_complete_fragment_classification_reads_only_result_columns(
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Clip completion is validated in one result-only scan without projection."""
    metadata = terminal_metadata(
        caption_spec,
        caption_digest,
        status="success",
        prompt_token_count=4,
        generated_token_count=8,
        error_type=None,
        error_message=None,
    )
    table = pa.Table.from_pylist(
        [{caption_spec.caption_field_name: "caption", caption_spec.metadata_field_name: metadata}],
        schema=caption_spec.fields,
    )
    scans = 0

    def scan(*, columns: list[str]) -> SimpleNamespace:
        nonlocal scans
        scans += 1
        assert columns == [caption_spec.caption_field_name, caption_spec.metadata_field_name]
        return SimpleNamespace(to_batches=table.to_batches)

    def unexpected_projection(_batch: pa.Table) -> pa.Table:
        pytest.fail("complete clip fragment input must not be projected")

    fragment = SimpleNamespace(fragment_id=0, scanner=scan)
    bindings = replace(
        caption_bindings,
        input=replace(caption_bindings.input, project_batch=unexpected_projection),
    )

    assert bindings.publication.validate_complete_fragment is None
    assert classify_fragment(fragment, bindings=bindings) is FragmentCaptionState.COMPLETE  # type: ignore[arg-type]
    assert scans == 1


def test_fragment_with_mixed_pending_and_terminal_rows_fails(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_digest: str,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Partial fragment completion violates fragment-atomic ownership."""
    del clip_dataset_factory
    metadata = terminal_metadata(
        caption_spec,
        caption_digest,
        status="success",
        prompt_token_count=4,
        generated_token_count=8,
        error_type=None,
        error_message=None,
    )
    table = pa.Table.from_pylist(
        [
            {
                "clip_id": "clip-0",
                "clip_uri": "/clip-0.mp4",
                "clip_size_bytes": 1,
                "_rowaddr": 0,
                caption_spec.caption_field_name: "caption",
                caption_spec.metadata_field_name: metadata,
            },
            {
                "clip_id": "clip-1",
                "clip_uri": "/clip-1.mp4",
                "clip_size_bytes": 1,
                "_rowaddr": 1,
                caption_spec.caption_field_name: None,
                caption_spec.metadata_field_name: None,
            },
        ],
        schema=pa.schema(
            [
                *caption_bindings.input.source_schema,
                pa.field("_rowaddr", pa.uint64(), nullable=False),
                *caption_spec.fields,
            ]
        ),
    )

    def to_batches() -> list[pa.RecordBatch]:
        return table.to_batches()

    scanner = SimpleNamespace(to_batches=to_batches)

    def scan(**_kwargs: object) -> SimpleNamespace:
        return scanner

    fragment = SimpleNamespace(fragment_id=0, scanner=scan)

    with pytest.raises(ValueError, match="mixes pending and terminal"):
        classify_fragment(fragment, bindings=caption_bindings)  # type: ignore[arg-type]


def test_fragment_with_stale_contract_digest_fails(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
    caption_spec: CaptionModelSpec,
    caption_bindings: CaptionRunBindings,
) -> None:
    """Terminal metadata from another result contract is incompatible."""
    _, dataset = _registered(clip_dataset_factory, caption_bindings)
    dataset = _commit_terminal_rows(dataset, caption_spec, "0" * 64, rows=2)
    fragment = dataset.get_fragment(0)
    assert fragment is not None

    with pytest.raises(ValueError, match="contract_digest"):
        classify_fragment(fragment, bindings=caption_bindings)


def test_input_validation_rejects_duplicate_clip_ids(
    clip_dataset_factory: Callable[..., tuple[str, lance.LanceDataset]],
) -> None:
    """Publication keys must be globally unique before caption registration."""
    uri, dataset = clip_dataset_factory(clip_ids=["duplicate", "duplicate"])

    with pytest.raises(ValueError, match="not globally unique"):
        validate_caption_input(dataset, uri=uri)
