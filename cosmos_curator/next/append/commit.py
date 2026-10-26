# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage packed Lance fragments on workers and append one selection's outcomes atomically.

One producer appends to a table bound to its result contract; other writers may
enrich existing fragments or add nullable fields while fragments are staged.
Workers write each packed block of rows as one uncommitted fragment and return
compact descriptors. The driver appends every descriptor in one ``Append``.
Canonical files resolve ambiguous commit responses; current item IDs let a
fresh driver recognize publication. Verification precedes workspace cleanup.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata, write_fragments
from loguru import logger

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.append.contracts import AppendBindings
from cosmos_curator.next.append.inputs import validate_target
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

DESCRIPTOR_SCHEMA = pa.schema([pa.field("fragment", pa.large_string(), nullable=False)])


def stage_fragment(rows: pa.Table, *, bindings: AppendBindings) -> pa.Table:
    """Write one packed block of appended rows as one uncommitted fragment and return its descriptor.

    Uncommitted files are invisible to readers; a failed publication leaves them
    for Lance cleanup.
    """
    if not rows.num_rows:
        return DESCRIPTOR_SCHEMA.empty_table()
    spec = bindings.spec
    if not rows.schema.equals(spec.output_fields):
        msg = "Appended rows do not match the producer's output fields"
        raise ValueError(msg)
    if rows.num_rows > bindings.execution.rows_per_fragment:
        msg = "A packed block of appended rows exceeds the configured fragment size"
        raise ValueError(msg)
    # ``overwrite`` affects only uncommitted fragment ID allocation; the driver
    # publishes the descriptor with ``Append``.
    fragments = write_fragments(
        rows,
        bindings.uri,
        schema=spec.output_fields,
        mode="overwrite",
        max_rows_per_file=rows.num_rows,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=bindings.storage_options,
    )
    if len(fragments) != 1 or fragments[0].num_rows != rows.num_rows:
        msg = "One packed block must stage exactly one complete Lance fragment"
        raise RuntimeError(msg)
    return pa.Table.from_pylist([{"fragment": json.dumps(fragments[0].to_json())}], schema=DESCRIPTOR_SCHEMA)


def _file_identity(data_file: Mapping[str, Any]) -> tuple[object, ...]:
    return (data_file["path"], tuple(data_file["fields"]), tuple(data_file.get("column_indices") or ()))


def verify_appended(dataset: lance.LanceDataset, fragments: Sequence[FragmentMetadata]) -> None:
    """Prove the canonical table still holds each appended fragment's rows in the same files.

    Enrichment may add column files to these fragments; the appended files,
    physical rows, and absence of deletions must remain unchanged.
    """
    holders: dict[str, dict[str, Any]] = {}
    for fragment in dataset.get_fragments():
        metadata = fragment.metadata.to_json()
        for data_file in metadata["files"]:
            holders[data_file["path"]] = metadata
    for appended in fragments:
        expected = appended.to_json()
        files = {_file_identity(data_file) for data_file in expected["files"]}
        holder = holders.get(expected["files"][0]["path"]) if expected["files"] else None
        if (
            holder is None
            or holder["physical_rows"] != expected["physical_rows"]
            or holder.get("deletion_file") is not None
            or not files <= {_file_identity(data_file) for data_file in holder["files"]}
        ):
            msg = f"Appended rows are missing or changed in {dataset.uri}; the publication's files are not canonical"
            raise ValueError(msg)


def commit_append(
    fragments: Sequence[FragmentMetadata],
    *,
    selected: Sequence[str],
    dataset: lance.LanceDataset,
    bindings: AppendBindings,
) -> lance.LanceDataset:
    """Atomically append complete item outcomes; reconcile lost responses from current files.

    Every selected item contributes a row, even an empty/error outcome. A fresh
    driver can therefore recognize publication from current item IDs. Within a
    commit retry, the staged file paths prove exactly which append succeeded,
    without transaction history or a second completion write.
    """
    spec = bindings.spec
    storage_options = bindings.storage_options
    original_ids = {fragment.fragment_id for fragment in dataset.get_fragments()}
    paths = {data_file["path"] for fragment in fragments for data_file in fragment.to_json()["files"]}
    if not paths or not selected:
        msg = "An append requires selected items and nonempty outcome fragments"
        raise ValueError(msg)

    def published_since(current: lance.LanceDataset) -> bool:
        added = [fragment for fragment in current.get_fragments() if fragment.fragment_id not in original_ids]
        current_paths = {data_file["path"] for fragment in added for data_file in fragment.metadata.to_json()["files"]}
        if paths & current_paths:
            verify_appended(current, fragments)
            return True
        if added:
            predicate = pc.field(spec.id_column).isin(selected)
            scanner = current.scanner(columns=[spec.id_column], fragments=added, filter=predicate)
            if any(batch.num_rows for batch in scanner.to_batches()):
                msg = "Another publication appended selected items; one producer must own the table"
                raise ValueError(msg)
        return False

    def commit() -> lance.LanceDataset:
        current = lance.dataset(dataset.uri, storage_options=storage_options)
        validate_target(current, spec)
        if published_since(current):
            return current
        transaction = lance.Transaction(
            read_version=int(current.version),
            operation=lance.LanceOperation.Append(list(fragments)),
            transaction_properties={KIND_KEY: spec.owner, CONTRACT_DIGEST_KEY: spec.contract_digest},
        )
        try:
            published = lance.LanceDataset.commit(
                dataset.uri, transaction, storage_options=storage_options, max_retries=0
            )
        except Exception as exc:
            latest = lance.dataset(dataset.uri, storage_options=storage_options)
            if published_since(latest):
                logger.warning(
                    "Lance append commit for {} raised {!r}; publication verified from canonical files at version {}",
                    dataset.uri,
                    exc,
                    latest.version,
                )
                return latest
            raise
        verify_appended(published, fragments)
        logger.info(
            "Appended {} row(s) in {} fragment(s) for {} item(s) to {} at version {}",
            sum(fragment.num_rows for fragment in fragments),
            len(fragments),
            len(selected),
            dataset.uri,
            published.version,
        )
        return published

    return do_with_retries(
        commit, exceptions_to_retry=(OSError, RuntimeError), max_attempts=bindings.execution.commit_attempts
    )
