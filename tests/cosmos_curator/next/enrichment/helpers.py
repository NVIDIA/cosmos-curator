# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared publication with real Arrow/Lance data and no Ray execution."""

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any
from unittest.mock import patch

import lance
import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.enrichment import commit, contracts, execution
from cosmos_curator.next.enrichment.attempts import EnrichmentAttempt
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig


def validate_items(batch: pa.RecordBatch) -> None:
    """Require a non-null result and successful status on every terminal item row."""
    valid = pc.and_(
        pc.is_valid(batch.column("result")), pc.equal(pc.struct_field(batch.column("metadata"), "status"), "success")
    )
    if not pc.all(pc.fill_null(valid, fill_value=False)).as_py():
        msg = "Invalid test result"
        raise ValueError(msg)


def make_bindings(
    uri: str,
    spec: contracts.EnrichmentSpec,
    *,
    staging_root: str = "staging",
    result_contract: dict[str, Any] | None = None,
    commit_attempts: int = 1,
) -> contracts.EnrichmentBindings:
    """Bind local test data to a small result contract and default execution settings."""
    return contracts.EnrichmentBindings(
        uri=uri,
        spec=spec,
        staging_root=staging_root,
        result_contract={} if result_contract is None else result_contract,
        execution=EnrichmentExecutionConfig(commit_attempts=commit_attempts),
    )


def prepare_items(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, selected: tuple[int, ...] | None = None
) -> list[commit.PreparedFragment]:
    """Prepare successful results for the shared item fixtures' fragments."""
    candidates = []
    for fragment in dataset.get_fragments():
        if selected is not None and fragment.fragment_id not in selected:
            continue
        rows = [
            row
            | {
                "fragment_id": fragment.fragment_id,
                "row_offset": offset,
                "result": 1,
                "metadata": {"status": "success"},
            }
            for offset, row in enumerate(fragment.to_table(columns=["item_id"]).to_pylist())
        ]
        candidates.extend(
            commit.PreparedFragment.from_row(row)
            for row in commit.prepare_fragment(
                pa.Table.from_pylist(rows, schema=spec.staged_schema),
                version=int(dataset.version),
                bindings=make_bindings(dataset.uri, spec),
            ).to_pylist()
        )
    return candidates


def commit_items(
    prepare: Callable[[int, tuple[int, ...]], list[commit.PreparedFragment]],
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    *,
    selected: tuple[int, ...] = (0, 1),
) -> lance.LanceDataset:
    """Commit the shared item fixtures' selection with its own recovery receipt."""
    bindings = make_bindings(dataset.uri, spec)
    return commit.commit_fragments(
        prepare,
        dataset=dataset,
        selected=selected,
        receipt=contracts.resolve_workspace(dataset, bindings, selected=selected).manifest_digest,
        transaction_properties={},
        bindings=bindings,
    )


def prepare_results(
    results: pa.Table,
    _results_uri: str,
    *,
    selected: tuple[int, ...],
    prepare_fragment: Callable[[pa.Table], pa.Table],
    **_kwargs: object,
) -> list[commit.PreparedFragment]:
    """Replace Ray's grouping and collection while retaining the production worker callable."""
    candidates = []
    for fragment_id in pc.unique(results["fragment_id"]).to_pylist():
        if fragment_id in selected:
            group = results.filter(pc.equal(results["fragment_id"], fragment_id))
            candidates.extend(commit.PreparedFragment.from_row(row) for row in prepare_fragment(group).to_pylist())
    return candidates


def attempt_all(enrichment_input: contracts.EnrichmentInput) -> EnrichmentAttempt:
    """Select every pending fragment without discovering or writing recovery state."""
    dataset, bindings = enrichment_input.dataset, enrichment_input.bindings
    pending = contracts.pending_fragment_ids(dataset, bindings)
    workspace = contracts.resolve_workspace(dataset, bindings, selected=pending) if pending else None
    return EnrichmentAttempt(enrichment_input, pending, pending, workspace, cleanup_complete=True)


def publish_attempt(
    results: pa.Table,
    attempt: EnrichmentAttempt,
    *,
    properties: Callable[[EnrichmentAttempt], Mapping[str, str]] | None = None,
) -> execution.EnrichmentOutcome:
    """Publish staged results through the recipes' adapter, preparing fragments locally instead of with Ray."""
    with patch.object(execution, "prepare_enrichment_fragments", partial(prepare_results, results)):
        return execution.publish_enrichment(
            attempt, transaction_properties=None if properties is None else properties(attempt)
        )


def publish_results(
    results: pa.Table,
    enrichment_input: contracts.EnrichmentInput,
    *,
    properties: Callable[[EnrichmentAttempt], Mapping[str, str]] | None = None,
) -> lance.LanceDataset:
    """Publish every pending fragment of a validated input and return the published table."""
    return publish_attempt(results, attempt_all(enrichment_input), properties=properties).dataset
