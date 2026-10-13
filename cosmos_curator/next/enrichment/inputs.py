# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open a recipe's pinned enrichment input, identify its physical layout, and select pending work.

``open_enrichment_input`` registers the spec's output fields, pins the latest
table, and validates it or reuses evidence an earlier publication recorded.
Recovery identity binds the selected physical fragments and the result contract,
never the Lance version or execution tuning. Publication receipts let startup
skip value scans for fragments whose owned-column files were already validated;
``receipt_fragments`` and ``result_files_match`` define what a receipt proves,
for pending selection and workspace recovery alike.
"""

from collections.abc import Sequence
from typing import Any

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
from lance.fragment import FragmentMetadata, LanceFragment

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.next.enrichment.contracts import VALIDATION_BATCH_SIZE, EnrichmentBindings, EnrichmentSpec
from cosmos_curator.next.enrichment.transactions import find_transaction, iter_transactions
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace
from cosmos_curator.next.utils.identity import canonical_digest

PUBLICATION_RECEIPT_KEY = "cosmos_curator.enrichment.publication"
VALIDATED_INPUT_KEY = "cosmos_curator.enrichment.validated_input"
# Bump when shared input validation changes, so earlier evidence is not reused.
_VALIDATION_VERSION = 1


def validate_fields(dataset: lance.LanceDataset, schema: pa.Schema) -> None:
    """Require exact types, nullability, and ownership metadata for named fields."""
    for field in schema:
        if field.name not in dataset.schema.names or not field.equals(
            dataset.schema.field(field.name), check_metadata=True
        ):
            msg = f"Missing or incompatible Lance field: {field.name}"
            raise ValueError(msg)


def register_fields(bindings: EnrichmentBindings) -> lance.LanceDataset:
    """Register the complete nullable field set, retrying disjoint schema races."""
    spec = bindings.spec
    storage_options = bindings.storage_options

    def register() -> lance.LanceDataset:
        dataset = lance.dataset(bindings.uri, storage_options=storage_options)
        validate_fields(dataset, spec.input_fields)
        if not set(spec.output_fields.names) & set(dataset.schema.names):
            dataset.add_columns(spec.output_fields)
            dataset = lance.dataset(bindings.uri, storage_options=storage_options)
        validate_fields(dataset, spec.output_fields)
        return dataset

    return do_with_retries(
        register, exceptions_to_retry=(OSError, RuntimeError), max_attempts=bindings.execution.commit_attempts
    )


@attrs.frozen
class EnrichmentInput:
    """A pinned dataset and the evidence obtained by validating that exact input.

    Create through ``open_enrichment_input``. Keep this value when publication
    refreshes descriptors against newer versions; those versions were not the
    snapshot examined by input validation.
    """

    dataset: lance.LanceDataset
    bindings: EnrichmentBindings
    fingerprint: str


def open_enrichment_input(bindings: EnrichmentBindings) -> EnrichmentInput:
    """Register fields, pin input, and validate it or reuse matching durable proof.

    Validation requires globally unique, non-null IDs across the complete
    snapshot, then runs the spec's optional ``validate_input`` for domain
    readiness. Only successful validation or existing proof produces evidence
    that publication can record.
    """
    dataset = register_fields(bindings)
    fingerprint = input_fingerprint(dataset, bindings)
    if not input_is_validated(dataset, fingerprint=fingerprint):
        _require_unique_ids(dataset, bindings.spec.id_column)
        if bindings.spec.validate_input is not None:
            bindings.spec.validate_input(dataset)
    return EnrichmentInput(dataset, bindings, fingerprint)


def _require_unique_ids(dataset: lance.LanceDataset, id_column: str) -> None:
    """Checkpoints key on IDs, so an ID shared by two rows would silently skip one."""
    column = id_column.replace('"', '""')
    query = f'SELECT COUNT(*) AS n, COUNT(DISTINCT "{column}") AS ids FROM dataset'  # noqa: S608 -- recipe column
    stats = next(iter(dataset.sql(query).build().to_batch_records())).to_pylist()[0]
    if stats["n"] != stats["ids"]:
        msg = f"Input {id_column} values must be non-null and globally unique"
        raise ValueError(msg)


def input_fingerprint(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> str:
    """Identify the complete pinned input under this recipe's validation contract.

    Shared publication records this for the snapshot that ``open_enrichment_input``
    validated. Appends and input rewrites invalidate it; disjoint outputs do not.
    """
    selected = tuple(int(fragment.fragment_id) for fragment in dataset.get_fragments())
    return canonical_digest(
        {
            "validation_version": _VALIDATION_VERSION,
            "manifest_digest": resolve_workspace(dataset, bindings, selected=selected).manifest_digest,
            "input_schema": bindings.spec.input_fields.serialize().to_pybytes().hex(),
        }
    )


def input_is_validated(dataset: lance.LanceDataset, *, fingerprint: str) -> bool:
    """Reuse a recipe's input validation only for the same contract and complete input."""
    transaction = find_transaction(dataset, property_key=VALIDATED_INPUT_KEY, property_value=fingerprint)
    return transaction is not None and PUBLICATION_RECEIPT_KEY in (transaction.transaction_properties or {})


def field_ids(dataset: lance.LanceDataset, names: Sequence[str]) -> dict[str, int]:
    """Map field paths to Lance leaf IDs, including nested result metadata."""
    result: dict[str, int] = {}

    def visit(field: Any, path: str) -> None:  # noqa: ANN401 -- LanceField has no public type stub
        children = field.children()
        if children:
            for child in children:
                visit(child, f"{path}.{child.name()}")
        else:
            result[path] = int(field.id())

    for field in dataset.lance_schema.fields():
        if field.name() in names:
            visit(field, field.name())
    return result


def file_layout_digest(metadata: FragmentMetadata, protected_ids: set[int]) -> str:
    """Digest a fragment's physical rows and the files and column positions holding the protected fields."""
    fragment_json = metadata.to_json()
    files = []
    for data_file in fragment_json["files"]:
        fields = data_file["fields"]
        positions = data_file.get("column_indices", list(range(len(fields))))
        selected = [(field, pos) for field, pos in zip(fields, positions, strict=True) if field in protected_ids]
        if selected:
            files.append({**data_file, "fields": selected, "column_indices": None})
    return canonical_digest({key: value for key, value in fragment_json.items() if key != "files"} | {"files": files})


def selected_fragments(dataset: lance.LanceDataset, selected: Sequence[int]) -> tuple[LanceFragment, ...]:
    """Resolve a unique selection to whole, undeleted fragments in ascending ID order."""
    if len(selected) != len(set(selected)):
        msg = "Selected fragment IDs must be unique"
        raise ValueError(msg)
    fragments = []
    for fragment_id in sorted(selected):
        fragment = dataset.get_fragment(fragment_id)
        if fragment is None or fragment.metadata.num_deletions:
            msg = f"Selected fragment {fragment_id} is absent or has deleted rows"
            raise ValueError(msg)
        fragments.append(fragment)
    return tuple(fragments)


def _namespace_manifest(dataset: lance.LanceDataset, spec: EnrichmentSpec) -> dict[str, Any]:
    # Persisted manifest keys: renaming one changes every namespace and orphans saved work.
    return {
        "workspace_version": 3,
        "input_uri": dataset.uri,
        "input_fields": field_ids(dataset, spec.input_fields.names),
        "result_contract": spec.result_contract,
        "result_schema": spec.staged_schema.serialize().to_pybytes().hex(),
    }


def workspace_prefix(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> str:
    """Locate this dataset and result contract's selection workspaces."""
    namespace = canonical_digest(_namespace_manifest(dataset, bindings.spec))
    return f"{bindings.staging_root.rstrip('/')}/{namespace}"


def resolve_workspace(
    dataset: lance.LanceDataset,
    bindings: EnrichmentBindings,
    *,
    selected: Sequence[int],
) -> RecoveryWorkspace:
    """Bind recovery to selected inputs, excluding unrelated fragments and tuning.

    Appends and changes outside the selection retain its workspace. The selected
    input file layout is immutable; a changed selection or layout resolves to
    another workspace.
    The input version is pinned per attempt and is never a recovery key.
    """
    manifest = _namespace_manifest(dataset, bindings.spec)
    input_ids = set(field_ids(dataset, bindings.spec.input_fields.names).values())
    fragments = [
        {"fragment_id": fragment.fragment_id, "file_layout": file_layout_digest(fragment.metadata, input_ids)}
        for fragment in selected_fragments(dataset, selected)
    ]
    manifest["input_fragments"] = fragments
    prefix = workspace_prefix(dataset, bindings)
    return RecoveryWorkspace(f"{prefix}/{canonical_digest(fragments)}", manifest)


def receipt_fragments(transaction: lance.Transaction, owned: set[int]) -> dict[int, FragmentMetadata] | None:
    """Decode a receipt's column update of exactly the owned fields over distinct fragments.

    Startup selection and workspace recovery share this reading of a receipt. Any
    other operation, field group, or fragment list proves nothing.
    """
    operation = transaction.operation
    if not isinstance(operation, lance.LanceOperation.Update) or set(operation.fields_modified) != owned:
        return None
    prepared = {int(fragment.id): fragment for fragment in operation.updated_fragments}
    if not prepared or len(prepared) != len(operation.updated_fragments):
        return None
    return prepared


def result_files_match(fragment: LanceFragment, prepared: FragmentMetadata, owned: set[int]) -> bool:
    """Report whether a canonical fragment still holds the owned-column files prepared and validated for it."""
    return file_layout_digest(prepared, owned) == file_layout_digest(fragment.metadata, owned)


def _published_fragment_ids(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> set[int]:
    """Recognize validated outputs from matching receipts and unchanged input/owned files."""
    fragments = {int(fragment.fragment_id): fragment for fragment in dataset.get_fragments()}
    owned = set(field_ids(dataset, bindings.spec.output_fields.names).values())
    published: set[int] = set()
    for transaction in iter_transactions(dataset, property_key=PUBLICATION_RECEIPT_KEY):
        prepared = receipt_fragments(transaction, owned)
        if prepared is None or any(
            fragment_id not in fragments or fragments[fragment_id].metadata.num_deletions for fragment_id in prepared
        ):
            continue
        workspace = resolve_workspace(dataset, bindings, selected=tuple(prepared))
        if (transaction.transaction_properties or {})[PUBLICATION_RECEIPT_KEY] != workspace.manifest_digest:
            continue
        published.update(
            fragment_id
            for fragment_id, metadata in prepared.items()
            if result_files_match(fragments[fragment_id], metadata, owned)
        )
        if len(published) == len(fragments):
            break
    return published


def fragment_complete(fragment: LanceFragment, spec: EnrichmentSpec) -> bool:
    """Require an entirely pending or entirely valid terminal physical fragment."""
    has_complete = has_pending = False
    for batch in fragment.scanner(
        columns=[spec.id_column, *spec.output_fields.names], batch_size=VALIDATION_BATCH_SIZE
    ).to_batches():
        complete = pc.is_valid(batch.column(spec.output_fields.names[0]))
        for name in spec.output_fields.names[1:]:
            complete = pc.or_(complete, pc.is_valid(batch.column(name)))
        if pc.any(complete).as_py():
            spec.validate_batch(batch.filter(complete))
            has_complete = True
        if pc.any(pc.invert(complete)).as_py():
            has_pending = True
        if has_complete and has_pending:
            msg = f"Fragment {fragment.fragment_id} mixes pending and terminal results"
            raise ValueError(msg)
    return not has_pending


def pending_fragment_ids(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> tuple[int, ...]:
    """Select pending work, reading values only where publication evidence is unavailable."""
    spec = bindings.spec
    validate_fields(dataset, spec.input_fields)
    validate_fields(dataset, spec.output_fields)
    published = _published_fragment_ids(dataset, bindings)
    return tuple(
        int(fragment.fragment_id)
        for fragment in dataset.get_fragments()
        if fragment.fragment_id not in published and not fragment_complete(fragment, spec)
    )
