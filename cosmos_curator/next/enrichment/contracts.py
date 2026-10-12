# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe enrichment contracts, validated input, pending selection, and recovery identity.

Recipes describe their input, owned output fields, and terminal validation in an
``EnrichmentSpec``, which derives the staged result schema, then resolve one
``EnrichmentBindings`` value per invocation; ``run_enrichment`` opens their input
with ``open_enrichment_input``. Recovery identity binds the input's selected physical
fragments and the result contract, never the Lance version or execution tuning.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from lance.fragment import FragmentMetadata, LanceFragment

from cosmos_curator.core.utils.misc.retry_utils import do_with_retries
from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.lance_datasource import ROW_ADDRESS_FIELDS
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig
from cosmos_curator.next.enrichment.transactions import find_transaction, iter_transactions
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.identity import canonical_digest, json_snapshot

PUBLICATION_RECEIPT_KEY = "cosmos_curator.enrichment.publication"
VALIDATED_INPUT_KEY = "cosmos_curator.enrichment.validated_input"
# Bump when shared input validation changes, so earlier evidence is not reused.
_VALIDATION_VERSION = 1
# Free-form provenance under the keys other Curator Next enrichment commits use:
# the publishing recipe, the field set it fills, and its result contract digest.
KIND_KEY = "kind"
FIELD_KEY = "field"
CONTRACT_DIGEST_KEY = "contract_digest"
OWNER_METADATA = b"cosmos_curator.owner"
FIELD_SET_METADATA = b"cosmos_curator.field_set"
VALIDATION_BATCH_SIZE = 1_024


def owned_fields(fields: pa.Schema, *, owner: str, field_set: str) -> pa.Schema:
    """Declare output fields as registered: nullable while pending and marked with their owner.

    Registration compares field metadata exactly, so another recipe or field set
    cannot claim the same columns. ``owner`` is the recipe kind, which
    publication also records as the commit's ``kind``.
    """
    ownership = {OWNER_METADATA: owner.encode(), FIELD_SET_METADATA: field_set.encode()}
    return pa.schema(
        [field.with_nullable(nullable=True).with_metadata({**(field.metadata or {}), **ownership}) for field in fields]
    )


def _ownership(fields: pa.Schema) -> tuple[str, str]:
    """Return the one owner and field set that every output field declares."""
    declared = {
        ((field.metadata or {}).get(OWNER_METADATA), (field.metadata or {}).get(FIELD_SET_METADATA)) for field in fields
    }
    if len(declared) == 1:
        ((owner, field_set),) = declared
        if owner is not None and field_set is not None:
            return owner.decode(), field_set.decode()
    msg = "Enrichment output fields require one owner and field set; declare them with owned_fields"
    raise ValueError(msg)


def _parquet_stable(data_type: pa.DataType) -> pa.DataType:
    """Stage list types as Parquet reads them back, including lists nested in structs.

    Parquet names every list child ``element``. Arrow cannot read back a null
    ``fixed_size_list``, so fixed-size lists stage as variable lists; publication
    casts staged values to their canonical types and rejects rows that do not convert.
    """
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type) or pa.types.is_fixed_size_list(data_type):
        value = data_type.value_field
        element = value.with_name("element").with_type(_parquet_stable(value.type))
        return pa.large_list(element) if pa.types.is_large_list(data_type) else pa.list_(element)
    if pa.types.is_struct(data_type):
        return pa.struct([field.with_type(_parquet_stable(field.type)) for field in data_type])
    return data_type


def _staged_result_schema(id_field: pa.Field, output_fields: pa.Schema) -> pa.Schema:
    """Return the checkpointed row schema: ``ROW_ADDRESS_FIELDS``, the ID, and Parquet-stable outputs.

    Ray checkpoints a result's ID once its Parquet file is written, so a staged
    schema that changed in that round trip would leave checkpointed results that
    publication rejects. Reject any output type Parquet would not read back
    unchanged, before processing begins.
    """
    schema = pa.schema(
        [*ROW_ADDRESS_FIELDS, id_field, *(field.with_type(_parquet_stable(field.type)) for field in output_fields)]
    )
    sink = pa.BufferOutputStream()
    pq.write_table(schema.empty_table(), sink)
    restored = pq.read_schema(pa.BufferReader(sink.getvalue()))
    for field, read_back in zip(schema, restored, strict=True):
        if not field.equals(read_back, check_metadata=True):
            msg = f"Staged field {field.name!r} of type {field.type} reads back from Parquet as {read_back.type}"
            raise ValueError(msg)
    return schema


@attrs.frozen
class EnrichmentSpec:
    """Recipe-owned input, output fields, readiness, and terminal validation: the complete data contract.

    ``input_fields`` is the upstream schema the recipe requires. Registration
    validates it, and recovery protects the file layout of every listed field;
    processing reads a subset through ``read_schema``. ``output_fields`` are the
    nullable fields this writer owns, declared with ``owned_fields``.

    ``validate_batch`` receives Arrow record batches of at most
    ``VALIDATION_BATCH_SIZE`` rows, holding ``id_column`` and the output fields in
    their canonical types. Every row claims to be terminal; reject any that is not.
    The optional ``validate_input`` checks domain readiness of the complete pinned
    table after the shared ID check. Validation evidence is keyed by the input and
    result contract, not by this code: when a recipe tightens ``validate_input``,
    it must change its result contract so earlier evidence is not reused.

    The derived ``staged_schema`` is the only definition of checkpointed Parquet
    rows: ``ROW_ADDRESS_FIELDS``, the ID as declared in the input, and the output
    fields as Parquet reads them back. Processing receives it from ``ProcessingScope``.
    """

    id_column: str
    input_fields: pa.Schema
    output_fields: pa.Schema
    validate_batch: Callable[[pa.RecordBatch], None]
    validate_input: Callable[[lance.LanceDataset], None] | None = None
    staged_schema: pa.Schema = attrs.field(init=False, eq=False)

    def __attrs_post_init__(self) -> None:
        """Reject malformed publication contracts and derive staged rows before any processing begins."""
        for schema in (self.input_fields, self.output_fields):
            if len(schema.names) != len(set(schema.names)):
                msg = "Enrichment schemas require unique field names"
                raise ValueError(msg)
        if self.id_column not in self.input_fields.names:
            msg = "The enrichment ID must be an input field"
            raise ValueError(msg)
        if not self.output_fields or any(not field.nullable for field in self.output_fields):
            msg = "Enrichment output fields must be nonempty and nullable while pending"
            raise ValueError(msg)
        if set(self.input_fields.names) & set(self.output_fields.names):
            msg = "Enrichment input and output fields must be disjoint"
            raise ValueError(msg)
        if {field.name for field in ROW_ADDRESS_FIELDS} & {*self.input_fields.names, *self.output_fields.names}:
            msg = "Enrichment input and output names must not overlap reserved row addresses"
            raise ValueError(msg)
        _ownership(self.output_fields)
        # Frozen attrs classes assign derived fields once, through object.__setattr__.
        staged = _staged_result_schema(self.input_fields.field(self.id_column), self.output_fields)
        object.__setattr__(self, "staged_schema", staged)

    @property
    def owner(self) -> str:
        """Return the recipe that owns the output fields."""
        return _ownership(self.output_fields)[0]

    @property
    def field_set(self) -> str:
        """Return the field set the output fields form."""
        return _ownership(self.output_fields)[1]

    def read_schema(self, columns: Sequence[str]) -> pa.Schema:
        """Project the ID, the named input columns, and ``ROW_ADDRESS_FIELDS`` for processing.

        Ray filters checkpointed IDs on the read operator, so the ID is always
        included. Only declared input fields can be read, so recovery protects
        every value the plan uses.
        """
        names = [self.id_column, *(name for name in columns if name != self.id_column)]
        if len(names) != len(set(names)):
            msg = "Read columns must be unique"
            raise ValueError(msg)
        undeclared = [name for name in names if name not in self.input_fields.names]
        if undeclared:
            msg = f"Read columns must be declared enrichment input fields: {undeclared}"
            raise ValueError(msg)
        return pa.schema([*(self.input_fields.field(name) for name in names), *ROW_ADDRESS_FIELDS])


def canonical_results(staged: pa.Table, spec: EnrichmentSpec) -> pa.Table:
    """Validate staged rows as terminal results and return their canonical ID and output columns.

    Processing applies this before Ray checkpoints a result's ID, so every
    checkpointed result can be published. Publication applies it again.
    """
    if not staged.schema.equals(spec.staged_schema, check_metadata=True):
        msg = "Staged results do not match the enrichment's staged schema"
        raise ValueError(msg)
    try:
        canonical = staged.select([spec.id_column, *spec.output_fields.names]).cast(
            pa.schema([spec.input_fields.field(spec.id_column), *spec.output_fields])
        )
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        msg = "Staged results do not convert to the canonical output types"
        raise ValueError(msg) from exc
    # Terminal validation sees the same canonical columns as pending-fragment scans.
    for batch in canonical.to_batches(max_chunksize=VALIDATION_BATCH_SIZE):
        spec.validate_batch(batch)
    return canonical


@attrs.frozen
class EnrichmentBindings:
    """Resolved recipe settings shared by input, recovery, and publication.

    Only ``spec`` and the snapshotted ``result_contract`` contribute to recovery
    compatibility alongside the pinned input's identity. ``execution`` holds the
    recipe's shared tuning, which never enters the manifest; storage credentials
    are resolved at runtime from its profile.
    """

    uri: str
    spec: EnrichmentSpec
    staging_root: str
    _result_contract_json: str = attrs.field(alias="result_contract", converter=json_snapshot)
    execution: EnrichmentExecutionConfig

    @property
    def result_contract(self) -> dict[str, Any]:
        """Return an independent copy of the result-defining settings."""
        return dict(json.loads(self._result_contract_json))

    @property
    def contract_digest(self) -> str:
        """Identify the result-defining settings in publication provenance and run summaries."""
        return canonical_digest(self.result_contract)

    @property
    def storage_options(self) -> dict[str, str] | None:
        """Resolve the current runtime credentials without retaining them in bindings."""
        return get_lance_storage_options(self.uri, profile_name=self.execution.storage_profile)

    def storage_env_vars(self, *locations: str) -> dict[str, str]:
        """Return the Ray worker storage environment for the table, staging root, and the recipe's locations.

        Pass every other location the recipe's workers read or write, such as media roots.
        """
        return s3_compatibility_env_vars(
            self.uri, self.staging_root, *locations, storage_profile=self.execution.storage_profile
        )


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


def _namespace_manifest(
    dataset: lance.LanceDataset, spec: EnrichmentSpec, result_contract: Mapping[str, Any]
) -> dict[str, Any]:
    # Persisted manifest keys: renaming one changes every namespace and orphans saved work.
    return {
        "workspace_version": 3,
        "input_uri": dataset.uri,
        "input_fields": field_ids(dataset, spec.input_fields.names),
        "result_contract": dict(result_contract),
        "result_schema": spec.staged_schema.serialize().to_pybytes().hex(),
    }


def workspace_prefix(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> str:
    """Locate this dataset and result contract's selection workspaces."""
    namespace = canonical_digest(_namespace_manifest(dataset, bindings.spec, bindings.result_contract))
    return f"{bindings.staging_root.rstrip('/')}/{namespace}"


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
    manifest = _namespace_manifest(dataset, bindings.spec, bindings.result_contract)
    input_ids = set(field_ids(dataset, bindings.spec.input_fields.names).values())
    fragments = [
        {"fragment_id": fragment.fragment_id, "file_layout": file_layout_digest(fragment.metadata, input_ids)}
        for fragment in selected_fragments(dataset, selected)
    ]
    manifest["input_fragments"] = fragments
    prefix = workspace_prefix(dataset, bindings)
    return RecoveryWorkspace(f"{prefix}/{canonical_digest(fragments)}", manifest)


def _published_fragment_ids(dataset: lance.LanceDataset, bindings: EnrichmentBindings) -> set[int]:
    """Recognize validated outputs from matching receipts and unchanged input/owned files."""
    fragments = {int(fragment.fragment_id): fragment for fragment in dataset.get_fragments()}
    owned = set(field_ids(dataset, bindings.spec.output_fields.names).values())
    published: set[int] = set()
    for transaction in iter_transactions(dataset, property_key=PUBLICATION_RECEIPT_KEY):
        operation = transaction.operation
        if not isinstance(operation, lance.LanceOperation.Update) or set(operation.fields_modified) != owned:
            continue
        prepared = {int(fragment.id): fragment for fragment in operation.updated_fragments}
        if len(prepared) != len(operation.updated_fragments) or not prepared:
            continue
        if any(
            fragment_id not in fragments or fragments[fragment_id].metadata.num_deletions for fragment_id in prepared
        ):
            continue
        workspace = resolve_workspace(dataset, bindings, selected=tuple(prepared))
        if (transaction.transaction_properties or {})[PUBLICATION_RECEIPT_KEY] != workspace.manifest_digest:
            continue
        published.update(
            fragment_id
            for fragment_id, metadata in prepared.items()
            if file_layout_digest(metadata, owned) == file_layout_digest(fragments[fragment_id].metadata, owned)
        )
        if len(published) == len(fragments):
            break
    return published


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
