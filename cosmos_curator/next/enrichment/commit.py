# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed, full-fragment enrichment published in one Lance Update.

One logical writer owns a selected fragment and result field set. Selected input
rows and their file layout are immutable; other writers may append
fragments or enrich disjoint columns or fragments. Workers prepare column files
from validated Parquet results. The driver reopens latest, checks ownership,
and rebuilds stale descriptors before committing.
Only compact descriptors reach the driver; media and inference remain recipe-owned.

A receipt shares the result transaction and resolves ambiguous commit responses.
Canonical verification must succeed before the caller removes its recovery workspace.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Self

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata
from loguru import logger

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.enrichment.contracts import EnrichmentBindings, EnrichmentSpec, canonical_results
from cosmos_curator.next.enrichment.inputs import (
    PUBLICATION_RECEIPT_KEY,
    field_ids,
    file_layout_digest,
    fragment_complete,
    result_files_match,
    selected_fragments,
    validate_fields,
)
from cosmos_curator.next.enrichment.transactions import find_transaction

PREPARED_SCHEMA = pa.schema(
    [
        ("fragment_id", pa.int64()),
        ("fragment", pa.large_string()),
        ("source_fragment", pa.large_string()),
        ("fields_modified", pa.list_(pa.int32())),
    ]
)


@attrs.frozen
class PreparedFragment:
    """Decoded worker descriptor for one complete fragment's owned column files."""

    fragment_id: int
    fragment: FragmentMetadata
    source_fragment: FragmentMetadata
    fields_modified: tuple[int, ...]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Self:
        """Decode an Arrow control row once at the worker-to-driver boundary."""
        return cls(
            fragment_id=row["fragment_id"],
            fragment=FragmentMetadata.from_json(row["fragment"]),
            source_fragment=FragmentMetadata.from_json(row["source_fragment"]),
            fields_modified=tuple(row["fields_modified"]),
        )


def prepare_fragment(
    batch: pa.Table,
    *,
    bindings: EnrichmentBindings,
    version: int,
) -> pa.Table:
    """Validate every address, identity, and value before staging owned columns."""
    spec = bindings.spec
    if not batch.schema.equals(spec.staged_schema, check_metadata=True):
        msg = "Invalid staged enrichment schema"
        raise ValueError(msg)
    if batch["fragment_id"].null_count or pc.count_distinct(batch["fragment_id"]).as_py() != 1:
        msg = "Publication requires one complete fragment per group"
        raise ValueError(msg)
    fragment_id = int(batch["fragment_id"][0].as_py())
    dataset = lance.dataset(bindings.uri, version=version, storage_options=bindings.storage_options)
    validate_fields(dataset, spec.output_fields)
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None or fragment.metadata.num_deletions:
        msg = f"Staged fragment {fragment_id} disappeared or has deleted rows"
        raise ValueError(msg)
    ordered = batch.take(pc.sort_indices(batch, sort_keys=[("row_offset", "ascending")]))
    expected = fragment.to_table(columns=[spec.id_column])
    expected_ids = expected[spec.id_column]
    if expected_ids.null_count or pc.count_distinct(expected_ids).as_py() != expected.num_rows:
        msg = f"Input {spec.id_column} values must be non-null and unique within fragment {fragment_id}"
        raise ValueError(msg)
    if (
        ordered.num_rows != expected.num_rows
        or not ordered["row_offset"].combine_chunks().equals(pa.array(range(expected.num_rows), type=pa.int64()))
        or not ordered[spec.id_column].equals(expected_ids)
    ):
        msg = f"Missing, duplicate, or misaligned {spec.id_column} results for fragment {fragment_id}"
        raise ValueError(msg)
    try:
        values = canonical_results(ordered, spec)
    except ValueError as exc:
        msg = f"Staged results for fragment {fragment_id} cannot be published: {exc}"
        raise ValueError(msg) from exc
    metadata, modified = fragment.update_columns(values, left_on=spec.id_column, right_on=spec.id_column)
    return pa.Table.from_pylist(
        [
            {
                "fragment_id": fragment_id,
                "fragment": json.dumps(metadata.to_json()),
                "source_fragment": json.dumps(fragment.metadata.to_json()),
                "fields_modified": modified,
            }
        ],
        schema=PREPARED_SCHEMA,
    )


def verify_inputs(
    *, original: lance.LanceDataset, current: lance.LanceDataset, spec: EnrichmentSpec, selected: Sequence[int]
) -> None:
    """Allow disjoint enrichment and later appends while protecting selected inputs."""
    validate_fields(current, spec.input_fields)
    validate_fields(current, spec.output_fields)
    protected = [*spec.input_fields.names, *spec.output_fields.names]
    if field_ids(original, protected) != field_ids(current, protected):
        msg = "Input or result field identities changed during publication"
        raise ValueError(msg)
    input_ids = set(field_ids(original, spec.input_fields.names).values())
    for before in selected_fragments(original, selected):
        after = current.get_fragment(before.fragment_id)
        before_layout = file_layout_digest(before.metadata, input_ids)
        if after is None or file_layout_digest(after.metadata, input_ids) != before_layout:
            msg = f"Input rows or file layout changed for fragment {before.fragment_id}"
            raise ValueError(msg)


def verify_completed(
    *, original: lance.LanceDataset, current: lance.LanceDataset, spec: EnrichmentSpec, selected: Sequence[int]
) -> None:
    """Verify the finite selection, leaving concurrent appends for the next invocation."""
    verify_inputs(original=original, current=current, spec=spec, selected=selected)
    for fragment_id in selected:
        fragment = current.get_fragment(fragment_id)
        if fragment is None or not fragment_complete(fragment, spec):
            msg = f"Selected result fragment {fragment_id} remains pending"
            raise ValueError(msg)


def verify_result_files(
    dataset: lance.LanceDataset, spec: EnrichmentSpec, prepared: Sequence[FragmentMetadata]
) -> None:
    """Prove canonical output uses the immutable files validated during preparation."""
    validate_fields(dataset, spec.input_fields)
    validate_fields(dataset, spec.output_fields)
    owned = set(field_ids(dataset, spec.output_fields.names).values())
    for metadata in prepared:
        fragment = dataset.get_fragment(metadata.id)
        if fragment is None or not result_files_match(fragment, metadata, owned):
            msg = f"Published result files differ from the publication evidence for fragment {metadata.id}"
            raise ValueError(msg)


def _verify_publication(
    current: lance.LanceDataset,
    original: lance.LanceDataset,
    spec: EnrichmentSpec,
    candidates: list[PreparedFragment],
) -> None:
    """Require canonical results to retain the owned files prepared by this attempt."""
    selected = tuple(candidate.fragment_id for candidate in candidates)
    verify_inputs(original=original, current=current, spec=spec, selected=selected)
    verify_result_files(current, spec, [candidate.fragment for candidate in candidates])


def _validate_candidates(candidates: list[PreparedFragment], expected: set[int], modified: set[int]) -> None:
    ids = [candidate.fragment_id for candidate in candidates]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        msg = "Prepared result fragments do not exactly cover the selected input"
        raise ValueError(msg)
    for candidate in candidates:
        if candidate.fragment.id != candidate.fragment_id or set(candidate.fields_modified) != modified:
            msg = "Prepared descriptor modifies an unexpected fragment or field"
            raise ValueError(msg)


def _refresh_candidates(  # noqa: PLR0913 -- selection and current snapshot guards
    candidates: list[PreparedFragment],
    original: lance.LanceDataset,
    current: lance.LanceDataset,
    spec: EnrichmentSpec,
    expected: set[int],
    prepare: Callable[[int, tuple[int, ...]], list[PreparedFragment]],
) -> list[PreparedFragment]:
    """Rebuild stale column files and prove the descriptors preserve every other field."""
    modified = set(field_ids(original, spec.output_fields.names).values())
    for before in selected_fragments(original, tuple(expected)):
        after = current.get_fragment(before.fragment_id)
        assert after is not None
        if file_layout_digest(before.metadata, modified) != file_layout_digest(after.metadata, modified):
            msg = "Result fields changed during publication; multiple writers for one field set are unsupported"
            raise ValueError(msg)
    current_fragments = {
        int(fragment.fragment_id): fragment for fragment in selected_fragments(current, tuple(expected))
    }
    stale = {
        candidate.fragment_id
        for candidate in candidates
        if candidate.source_fragment.to_json() != current_fragments[candidate.fragment_id].metadata.to_json()
    }
    if stale:
        refreshed = prepare(int(current.version), tuple(sorted(stale)))
        _validate_candidates(refreshed, stale, modified)
        replacements = {candidate.fragment_id: candidate for candidate in refreshed}
        candidates = [replacements.get(candidate.fragment_id, candidate) for candidate in candidates]
    _validate_candidates(candidates, expected, modified)
    unmodified = set(field_ids(current, current.schema.names).values()) - modified
    for candidate in candidates:
        source = current_fragments[candidate.fragment_id].metadata
        if candidate.source_fragment.to_json() != source.to_json():
            msg = "Prepared descriptor does not preserve the current fragment files"
            raise ValueError(msg)
        if file_layout_digest(source, unmodified) != file_layout_digest(candidate.fragment, unmodified):
            msg = "Prepared descriptor changes physical rows or unowned columns"
            raise ValueError(msg)
    return candidates


def commit_fragments(  # noqa: PLR0913 -- explicit transaction and recovery inputs
    prepare: Callable[[int, tuple[int, ...]], list[PreparedFragment]],
    *,
    dataset: lance.LanceDataset,
    bindings: EnrichmentBindings,
    selected: Sequence[int],
    receipt: str,
    transaction_properties: Mapping[str, str],
) -> lance.LanceDataset:
    """Publish once, rebuilding stale descriptors without repeating expensive work.

    Call ``prepare`` once for the pinned selection, then only for stale fragments
    at their latest version. Retain those candidates across commit retries.
    Only this publication's receipt proves an ambiguous success. Terminal values
    without it cannot authorize overwriting or accepting a competing writer's work.

    Preparation must validate staged coverage and terminal values before returning
    descriptors, as ``prepare_fragment`` does. Publication verifies the exact
    immutable files, without scanning their values again on the driver. Recipes
    normally call ``publish_enrichment`` to bind this primitive to validated input
    and a recovery attempt.
    """
    spec = bindings.spec
    storage_options = bindings.storage_options
    expected = set(selected)
    selected_fragments(dataset, selected)
    modified = set(field_ids(dataset, spec.output_fields.names).values())
    candidates = prepare(int(dataset.version), tuple(selected))
    _validate_candidates(candidates, expected, modified)

    def commit() -> lance.LanceDataset:
        nonlocal candidates
        current = lance.dataset(dataset.uri, storage_options=storage_options)
        verify_inputs(original=dataset, current=current, spec=spec, selected=selected)
        if (
            find_transaction(
                current,
                property_key=PUBLICATION_RECEIPT_KEY,
                property_value=receipt,
                after_version=int(dataset.version),
            )
            is not None
        ):
            _verify_publication(current, dataset, spec, candidates)
            return current
        candidates = _refresh_candidates(candidates, dataset, current, spec, expected, prepare)
        transaction = lance.Transaction(
            read_version=int(current.version),
            operation=lance.LanceOperation.Update(
                updated_fragments=[candidate.fragment for candidate in candidates],
                fields_modified=sorted(modified),
            ),
            transaction_properties={**transaction_properties, PUBLICATION_RECEIPT_KEY: receipt},
        )
        try:
            published = lance.LanceDataset.commit(
                dataset.uri, transaction, storage_options=storage_options, max_retries=0
            )
        except Exception as exc:
            latest = lance.dataset(dataset.uri, storage_options=storage_options)
            if (
                find_transaction(
                    latest,
                    property_key=PUBLICATION_RECEIPT_KEY,
                    property_value=receipt,
                    after_version=int(dataset.version),
                )
                is not None
            ):
                _verify_publication(latest, dataset, spec, candidates)
                logger.warning(
                    "Lance enrichment commit for {} raised {!r}; "
                    "publication verified from its durable receipt and canonical state at version {}",
                    dataset.uri,
                    exc,
                    latest.version,
                )
                return latest
            raise
        _verify_publication(published, dataset, spec, candidates)
        return published

    return do_with_retries(
        commit, exceptions_to_retry=(OSError, RuntimeError), max_attempts=bindings.execution.commit_attempts
    )
