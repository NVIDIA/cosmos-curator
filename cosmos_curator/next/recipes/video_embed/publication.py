# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare result column files on workers and commit all fragments together."""

import json
from collections.abc import Callable
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.recipes.video_embed.config import ResolvedVideoEmbedConfig
from cosmos_curator.next.recipes.video_embed.contracts import ModelSpec, validate_terminal_row
from cosmos_curator.next.recipes.video_embed.state import (
    Workspace,
    binding,
    field_ids,
    fragment_complete,
    storage_options,
    validate_fields,
)
from cosmos_curator.next.recipes.video_transcode.contracts import CLIP_SCHEMA

PUBLICATION_KEY = "video_embed.publication"
PREPARED_SCHEMA = pa.schema(
    [
        ("fragment_id", pa.int64()),
        ("fragment", pa.large_string()),
        ("source_fragment", pa.large_string()),
        ("fields_modified", pa.list_(pa.int32())),
    ]
)


def prepare_fragment(batch: pa.Table, *, config: ResolvedVideoEmbedConfig, spec: ModelSpec, version: int) -> pa.Table:
    """Prove exact clip coverage, then write only vector and metadata columns."""
    if not batch.schema.equals(spec.result_schema, check_metadata=True):
        msg = "Invalid staged video-embed schema"
        raise ValueError(msg)
    ids = pc.unique(batch["fragment_id"]).to_pylist()
    if len(ids) != 1 or ids[0] is None:
        msg = "Publication requires one complete fragment per group"
        raise ValueError(msg)
    dataset = lance.dataset(config.input.clips_lance_uri, version=version, storage_options=storage_options(config))
    validate_fields(dataset, spec)
    fragment = dataset.get_fragment(ids[0])
    if fragment is None or fragment.metadata.num_deletions:
        msg = f"Staged fragment {ids[0]} disappeared or has deleted rows"
        raise ValueError(msg)
    ordered = batch.take(pc.sort_indices(batch, sort_keys=[("row_offset", "ascending")]))
    expected = fragment.to_table(columns=["clip_id"])
    if (
        ordered.num_rows != expected.num_rows
        or ordered["row_offset"].to_pylist() != list(range(expected.num_rows))
        or not ordered["clip_id"].equals(expected["clip_id"])
    ):
        msg = f"Missing, duplicate, or misaligned clip results for fragment {ids[0]}"
        raise ValueError(msg)
    for row in ordered.to_pylist():
        validate_terminal_row(row, spec)
    values = ordered.select(["clip_id", *spec.fields.names]).cast(pa.schema([("clip_id", pa.string()), *spec.fields]))
    metadata, fields_modified = fragment.update_columns(values, left_on="clip_id", right_on="clip_id")
    return pa.Table.from_pylist(
        [
            {
                "fragment_id": ids[0],
                "fragment": json.dumps(metadata.to_json()),
                "source_fragment": json.dumps(fragment.metadata.to_json()),
                "fields_modified": fields_modified,
            }
        ],
        schema=PREPARED_SCHEMA,
    )


def _verify_inputs(original: lance.LanceDataset, current: lance.LanceDataset, spec: ModelSpec) -> None:
    validate_fields(current, spec)
    for field in CLIP_SCHEMA:
        if field.name not in current.schema.names or not field.equals(
            current.schema.field(field.name), check_metadata=True
        ):
            msg = f"Input clip schema changed during publication: {field.name}"
            raise ValueError(msg)
    protected = [*CLIP_SCHEMA.names, *spec.fields.names]
    if field_ids(original, protected) != field_ids(current, protected):
        msg = "Input or embedding field identities changed during publication"
        raise ValueError(msg)
    input_ids = set(field_ids(original, CLIP_SCHEMA.names).values())
    for before in original.get_fragments():
        after = current.get_fragment(before.fragment_id)
        if after is None or binding(before, input_ids) != binding(after, input_ids):
            msg = f"Clip rows or physical bindings changed for fragment {before.fragment_id}"
            raise ValueError(msg)


def _validate_candidates(candidates: list[dict[str, Any]], expected: set[int], fields: set[int]) -> None:
    ids = [candidate["fragment_id"] for candidate in candidates]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        msg = "Prepared embedding fragments do not exactly cover the selected input"
        raise ValueError(msg)
    for candidate in candidates:
        metadata = FragmentMetadata.from_json(candidate["fragment"])
        if metadata.id != candidate["fragment_id"] or set(candidate["fields_modified"]) != fields:
            msg = "Prepared embedding descriptor modifies an unexpected fragment or field"
            raise ValueError(msg)


def verify_completed(dataset: lance.LanceDataset, original: lance.LanceDataset, spec: ModelSpec) -> None:
    """Verify the finite selection, allowing later producer appends to remain pending."""
    _verify_inputs(original, dataset, spec)
    for before in original.get_fragments():
        fragment = dataset.get_fragment(before.fragment_id)
        if fragment is None or not fragment_complete(fragment, spec):
            msg = f"Selected embedding fragment {before.fragment_id} remains pending"
            raise ValueError(msg)


def _has_receipt(dataset: lance.LanceDataset, *, after_version: int, scope: str) -> bool:
    for version in range(int(dataset.version), after_version, -1):
        transaction = dataset.read_transaction(version)
        if transaction is not None and (transaction.transaction_properties or {}).get(PUBLICATION_KEY) == scope:
            return True
    return False


def commit_fragments(  # noqa: C901, PLR0913 -- one atomic transaction and its reconciliation paths
    candidates: list[dict[str, Any]],
    *,
    config: ResolvedVideoEmbedConfig,
    dataset: lance.LanceDataset,
    spec: ModelSpec,
    workspace: Workspace,
    pending: tuple[int, ...],
    reprepare: Callable[[int], list[dict[str, Any]]],
) -> lance.LanceDataset:
    """Publish the entire result in one Update, rebuilding stale descriptors only.

    Disjoint caption commits may change file bindings in the same fragments.
    Reprepare from durable Parquet at latest before using its read version. A
    unique workspace receipt resolves a lost commit response without inference.
    The pending selection belongs to the original input snapshot.
    One logical writer owns this variant's field set.
    """
    expected = set(pending)
    modified = set(field_ids(dataset, spec.fields.names).values())
    _validate_candidates(candidates, expected, modified)

    def commit() -> lance.LanceDataset:
        nonlocal candidates
        current = lance.dataset(config.input.clips_lance_uri, storage_options=storage_options(config))
        _verify_inputs(dataset, current, spec)
        current_fragments = {int(fragment.fragment_id): fragment for fragment in current.get_fragments()}
        if _has_receipt(current, after_version=int(dataset.version), scope=workspace.scope):
            verify_completed(current, dataset, spec)
            return current
        if not expected:
            verify_completed(current, dataset, spec)
            return current
        owned = set(field_ids(dataset, spec.fields.names).values())
        for before in dataset.get_fragments():
            after = current.get_fragment(before.fragment_id)
            assert after is not None
            if binding(before, owned) != binding(after, owned):
                msg = "Embedding fields changed during publication; multiple writers for one field set are unsupported"
                raise ValueError(msg)
        if any(
            json.loads(candidate["source_fragment"]) != current_fragments[candidate["fragment_id"]].metadata.to_json()
            for candidate in candidates
        ):
            candidates = reprepare(int(current.version))
        _validate_candidates(candidates, expected, modified)
        for candidate in candidates:
            if (
                json.loads(candidate["source_fragment"])
                != current_fragments[candidate["fragment_id"]].metadata.to_json()
            ):
                msg = "Prepared embedding descriptor does not preserve the current fragment files"
                raise ValueError(msg)
        transaction = lance.Transaction(
            read_version=int(current.version),
            operation=lance.LanceOperation.Update(
                updated_fragments=[FragmentMetadata.from_json(candidate["fragment"]) for candidate in candidates],
                fields_modified=sorted(modified),
            ),
            transaction_properties={
                "kind": "video-embed",
                "field": spec.vector_field,
                "contract_digest": spec.digest,
                PUBLICATION_KEY: workspace.scope,
            },
        )
        try:
            published = lance.LanceDataset.commit(
                config.input.clips_lance_uri, transaction, storage_options=storage_options(config), max_retries=0
            )
        except Exception:
            latest = lance.dataset(config.input.clips_lance_uri, storage_options=storage_options(config))
            if _has_receipt(latest, after_version=int(dataset.version), scope=workspace.scope):
                verify_completed(latest, dataset, spec)
                return latest
            raise
        verify_completed(published, dataset, spec)
        return published

    return do_with_retries(
        commit, exceptions_to_retry=(OSError, RuntimeError), max_attempts=config.execution.commit_attempts
    )
