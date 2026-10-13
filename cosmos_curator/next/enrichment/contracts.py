# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-facing enrichment declarations: owned output fields, the data contract, and its bindings.

Recipes declare their outputs with ``owned_fields`` and describe their input,
result contract, and validation in an ``EnrichmentSpec``, which derives the
staged result schema. ``EnrichmentBindings`` binds the spec to one invocation's
table, staging root, and execution settings; ``run_enrichment`` does the rest.
"""

import json
from collections.abc import Callable, Sequence
from typing import Any

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.lance_datasource import ROW_ADDRESS_FIELDS
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars
from cosmos_curator.next.utils.identity import canonical_digest, json_snapshot

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
    """Recipe-owned input, output fields, result contract, and validation: the complete data contract.

    ``input_fields`` is the upstream schema the recipe requires. Registration
    validates it, and recovery protects the file layout of every listed field;
    processing reads a subset through ``read_schema``. ``output_fields`` are the
    nullable fields this writer owns, declared with ``owned_fields``.
    ``result_contract`` holds only result-defining settings, snapshotted as JSON;
    it scopes recovery compatibility and publication provenance.

    ``validate_batch`` receives Arrow record batches of at most
    ``VALIDATION_BATCH_SIZE`` rows, holding ``id_column`` and the output fields in
    their canonical types. Every row claims to be terminal; reject any that is not.
    The optional ``validate_input`` checks domain readiness of the complete pinned
    table after the shared ID check. Both validators must depend only on the result
    contract: validation evidence is keyed by the input and result contract, not by
    validator code, so a recipe that changes what either accepts must change its
    result contract so earlier evidence is not reused.

    The derived ``staged_schema`` is the only definition of checkpointed Parquet
    rows: ``ROW_ADDRESS_FIELDS``, the ID as declared in the input, and the output
    fields as Parquet reads them back. Processing receives it from ``ProcessingScope``.
    """

    id_column: str
    input_fields: pa.Schema
    output_fields: pa.Schema
    _result_contract_json: str = attrs.field(alias="result_contract", converter=json_snapshot)
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

    @property
    def result_contract(self) -> dict[str, Any]:
        """Return an independent copy of the result-defining settings."""
        return dict(json.loads(self._result_contract_json))

    @property
    def contract_digest(self) -> str:
        """Identify the result-defining settings in publication provenance and run summaries."""
        return canonical_digest(self.result_contract)

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
    """Bind a spec to one invocation's table, staging root, and execution settings.

    The spec says what is produced; the bindings say where and how. Only ``spec``
    contributes to recovery compatibility alongside the pinned input's identity.
    ``execution`` holds the recipe's shared tuning, which never enters the
    manifest; storage credentials are resolved at runtime from its profile.
    """

    uri: str
    spec: EnrichmentSpec
    staging_root: str
    execution: EnrichmentExecutionConfig

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
