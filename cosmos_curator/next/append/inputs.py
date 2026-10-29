# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open a producer's append target, validate new work items, and index earlier publications.

``open_target`` creates the producer's table or validates its schema and
contract binding, and pins it before recovery or discovery.
A table is bound to one producer, item ID column, and result contract.
Every completed item has at least one canonical row, including empty and error
outcomes. ``published_item_ids`` projects candidate IDs from current rows without
reading history. Recovery identity binds the selected items and result contract,
never the Lance version or execution tuning.
"""

from collections.abc import Sequence

import lance
import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.append.contracts import AppendBindings, AppendSpec
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY
from cosmos_curator.next.core.recovery_workspace import RecoveryWorkspace
from cosmos_curator.next.utils.identity import canonical_digest, canonical_json
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION, open_dataset

CONTRACT_METADATA = b"cosmos_curator.append.contract"


def target_schema(spec: AppendSpec) -> pa.Schema:
    """Bind the appended fields to their producer, item ID column, and result contract."""
    binding = canonical_json(
        {"format_version": 2, "owner": spec.owner, "id_column": spec.id_column, "result_contract": spec.result_contract}
    )
    return spec.output_fields.with_metadata({CONTRACT_METADATA: binding.encode()})


def validate_target(dataset: lance.LanceDataset, spec: AppendSpec) -> None:
    """Require this producer's contract binding, its fields first and unchanged, and nullable extensions.

    Staged fragments hold only the producer's fields, which Lance numbers by
    position when it creates the table; other writers may add nullable fields
    after them, which appended rows read as null.
    """
    if dataset.data_storage_version != LANCE_DATA_STORAGE_VERSION:
        msg = f"Lance table {dataset.uri} uses data storage version {dataset.data_storage_version}"
        raise ValueError(msg)
    binding = target_schema(spec).metadata[CONTRACT_METADATA]
    if (dataset.schema.metadata or {}).get(CONTRACT_METADATA) != binding:
        msg = (
            f"Lance table {dataset.uri} has a different producer, item ID column, or result contract; "
            "use a new output location"
        )
        raise ValueError(msg)
    owned = spec.output_fields
    leading = dataset.schema.names[: len(owned)]
    if leading != owned.names or any(
        not field.equals(dataset.schema.field(field.name), check_metadata=True) for field in owned
    ):
        msg = f"Lance table {dataset.uri} does not begin with this producer's fields"
        raise ValueError(msg)
    required = [field.name for field in dataset.schema if field.name not in owned.names and not field.nullable]
    if required:
        msg = f"Lance table {dataset.uri} has non-nullable fields that appended rows cannot fill: {required}"
        raise ValueError(msg)


def open_target(bindings: AppendBindings) -> lance.LanceDataset:
    """Create the empty table with its contract binding, or validate the existing one, and pin it."""
    spec = bindings.spec
    storage_options = bindings.storage_options
    schema = target_schema(spec)

    def open_or_create() -> lance.LanceDataset:
        dataset = open_dataset(bindings.uri, storage_options=storage_options)
        if dataset is None:
            try:
                dataset = lance.write_dataset(
                    schema.empty_table(),
                    bindings.uri,
                    schema=schema,
                    mode="create",
                    data_storage_version=LANCE_DATA_STORAGE_VERSION,
                    storage_options=storage_options,
                    enable_v2_manifest_paths=True,
                    transaction_properties={KIND_KEY: spec.owner, CONTRACT_DIGEST_KEY: spec.contract_digest},
                )
            except OSError:
                # A lost response or a concurrent creator may have created the table.
                dataset = open_dataset(bindings.uri, storage_options=storage_options)
                if dataset is None:
                    raise
        validate_target(dataset, spec)
        return dataset

    return do_with_retries(
        open_or_create, exceptions_to_retry=(OSError, RuntimeError), max_attempts=bindings.execution.commit_attempts
    )


def validate_items(items: pa.Table, spec: AppendSpec) -> pa.Table:
    """Require the declared schema, non-null required fields, and unique IDs; return the items in ID order."""
    if not items.schema.equals(spec.input_fields):
        msg = "Work items do not match the append's input fields"
        raise ValueError(msg)
    null_fields = [field.name for field in spec.input_fields if not field.nullable and items[field.name].null_count]
    if null_fields:
        msg = f"Work item fields must be non-null: {null_fields}"
        raise ValueError(msg)
    ids = items[spec.id_column]
    if pc.count_distinct(ids).as_py() != items.num_rows:
        msg = f"Work item {spec.id_column} values must be unique"
        raise ValueError(msg)
    return items.sort_by(spec.id_column).combine_chunks()


def published_item_ids(dataset: lance.LanceDataset, bindings: AppendBindings, item_ids: Sequence[str]) -> set[str]:
    """Look up only candidate IDs in current rows, including empty and error outcomes.

    The predicate permits Lance to use a scalar index when present. Only the ID
    column is read, in batches; neither output values nor history are loaded.
    """
    if not item_ids:
        return set()
    id_column = bindings.spec.id_column
    published: set[str] = set()
    # One scan, including when the table has no scalar index. Chunking the
    # predicate would rescan the table once per chunk at large discovery sizes.
    predicate = pc.field(id_column).isin(item_ids)
    for batch in dataset.scanner(columns=[id_column], filter=predicate).to_batches():
        published.update(pc.unique(batch[id_column]).to_pylist())
    return published


def _namespace_manifest(dataset: lance.LanceDataset, spec: AppendSpec) -> dict[str, object]:
    # Persisted manifest keys: renaming one changes every namespace and orphans saved work.
    return {
        "workspace_version": 5,
        "output_uri": dataset.uri,
        "owner": spec.owner,
        "id_column": spec.id_column,
        "result_contract": spec.result_contract,
        "input_schema": spec.input_fields.serialize().to_pybytes().hex(),
        "result_schema": spec.result_fields.serialize().to_pybytes().hex(),
        "output_schema": spec.output_fields.serialize().to_pybytes().hex(),
        "error_column": spec.error_column,
    }


def workspace_root(dataset: lance.LanceDataset, bindings: AppendBindings) -> str:
    """Locate the single workspace for this table and result contract."""
    namespace = canonical_digest(_namespace_manifest(dataset, bindings.spec))
    return f"{bindings.staging_root.rstrip('/')}/{namespace}"


def resolve_workspace(
    dataset: lance.LanceDataset, bindings: AppendBindings, *, selection: pa.Table
) -> RecoveryWorkspace:
    """Record the selected items in the table and contract's fixed workspace.

    The manifest saves the selected items, so recovery resumes them even when
    discovery later finds different items. Each selection reuses the same path
    after verified publication and cleanup. Table versions and execution tuning
    do not define compatibility.
    """
    items = selection.to_pylist()
    manifest = _namespace_manifest(dataset, bindings.spec) | {"selected_items": items}
    return RecoveryWorkspace(workspace_root(dataset, bindings), manifest)
