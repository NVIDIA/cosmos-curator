# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-facing append declarations: the data contract and its bindings.

Recipes describe their work items, staged item results, appended rows, result
contract, validation, and expansion in an ``AppendSpec``, which derives the
staged result schema. ``AppendBindings`` binds the spec to one invocation's
table, staging root, and execution settings; ``run_append`` does the rest.
"""

import json
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import attrs
import pyarrow as pa
import pyarrow.compute as pc

from cosmos_curator.next.append.config import AppendExecutionConfig
from cosmos_curator.next.core.ray_checkpoints import VALIDATION_BATCH_SIZE, parquet_stable_schema
from cosmos_curator.next.core.recovery_bindings import RecoveryBindings
from cosmos_curator.next.utils.identity import canonical_digest, json_snapshot

# Workspace manifests save selected items as JSON scalars.
_JSON_SCALARS = (pa.types.is_string, pa.types.is_large_string, pa.types.is_integer, pa.types.is_boolean)


def _json_scalar(field: pa.Field) -> bool:
    return any(accepts(field.type) for accepts in _JSON_SCALARS)


@attrs.frozen
class AppendSpec:
    """Recipe-owned work items, item results, appended rows, and result contract: the complete data contract.

    ``id_column`` names one work item. Checkpoints key on it and every appended
    row carries it, so it is a non-null
    string in all three schemas. ``input_fields`` is the schema of the discovered
    work items, which processing reads; recovery saves a selection's items as
    JSON, so their fields are strings, integers, or booleans. ``result_fields``
    holds one terminal result per item, staged and checkpointed before
    publication. ``output_fields`` is the schema of the rows this producer
    appends; the producer owns the table they form. ``result_contract`` holds only
    result-defining settings, snapshotted as JSON; it binds the table, scopes
    recovery compatibility, and identifies provenance. ``owner`` is the recipe
    kind, which publication records as ``kind``.

    ``validate_batch`` receives Arrow record batches of at most
    ``VALIDATION_BATCH_SIZE`` item results in their canonical types. Every row
    claims to be terminal; reject any that is not. ``expand`` turns a table of
    validated item results into the rows to append, in ``output_fields``. Every
    item must contribute at least one row, including empty outcomes and terminal
    errors. ``expand`` may yield several tables to bound memory. Validation and
    expansion must depend only on the result contract: tables and workspaces are
    keyed by it, not by code, so changes require a result-contract version change.

    ``error_column`` optionally names a nullable string field shared by results
    and appended rows whose value marks a terminal error, such as a missing
    source. Publication counts these errors; the recipe's expansion preserves
    their diagnostics in canonical rows. Which failures are terminal remains
    the recipe's decision; retryable failures must propagate without a result.

    Processing emits canonical ``result_fields`` rows, also exposed as
    ``ProcessingScope.result_schema``. The shared writer converts them to the
    derived ``staged_schema`` for Parquet; recipes do not manage storage types.
    Recovery identity includes both schemas, the input and output schemas, and
    the declared ID and error column settings automatically. The table binding
    also names ``id_column``, so current rows keep one meaning for their IDs.
    """

    owner: str
    id_column: str
    input_fields: pa.Schema
    result_fields: pa.Schema
    output_fields: pa.Schema
    _result_contract_json: str = attrs.field(alias="result_contract", converter=json_snapshot)
    validate_batch: Callable[[pa.RecordBatch], None]
    expand: Callable[[pa.Table], Iterable[pa.Table]]
    error_column: str | None = None
    staged_schema: pa.Schema = attrs.field(init=False, eq=False)

    def __attrs_post_init__(self) -> None:
        """Reject malformed append contracts and derive staged rows before any processing begins."""
        if not self.owner:
            msg = "Append publication requires an owner"
            raise ValueError(msg)
        schemas = (self.input_fields, self.result_fields, self.output_fields)
        for schema in schemas:
            if len(schema.names) != len(set(schema.names)):
                msg = "Append schemas require unique field names"
                raise ValueError(msg)
            if self.id_column not in schema.names:
                msg = "Work items, item results, and appended rows must all carry the item ID"
                raise ValueError(msg)
        id_field = self.input_fields.field(self.id_column)
        if (
            not (pa.types.is_string(id_field.type) or pa.types.is_large_string(id_field.type))
            or id_field.nullable
            or any(not schema.field(self.id_column).equals(id_field) for schema in schemas)
        ):
            msg = "The item ID must be the same non-null string field in every append schema"
            raise ValueError(msg)
        if any(not _json_scalar(field) for field in self.input_fields):
            msg = "Work item fields must be strings, integers, or booleans"
            raise ValueError(msg)
        self._validate_errors()
        # Frozen attrs classes assign derived fields once, through object.__setattr__.
        object.__setattr__(self, "staged_schema", parquet_stable_schema(self.result_fields))

    def _validate_errors(self) -> None:
        if self.error_column is None:
            return
        names = self.result_fields.names
        if self.error_column == self.id_column or self.error_column not in names:
            msg = "The error column must be an item result field other than the item ID"
            raise ValueError(msg)
        error_field = self.result_fields.field(self.error_column)
        if not error_field.nullable or not (
            pa.types.is_string(error_field.type) or pa.types.is_large_string(error_field.type)
        ):
            msg = "The error column must be a nullable string; null marks a result without a terminal error"
            raise ValueError(msg)
        if self.error_column not in self.output_fields.names or not self.output_fields.field(self.error_column).equals(
            error_field
        ):
            msg = "The error column must also be preserved in the appended rows"
            raise ValueError(msg)

    @property
    def result_contract(self) -> dict[str, Any]:
        """Return an independent copy of the result-defining settings."""
        return dict(json.loads(self._result_contract_json))

    @property
    def contract_digest(self) -> str:
        """Identify the result-defining settings in the table binding, provenance, and run summaries."""
        return canonical_digest(self.result_contract)


def canonical_results(staged: pa.Table, spec: AppendSpec) -> pa.Table:
    """Validate staged rows as terminal item results and return them in their canonical types.

    Processing applies this before Ray checkpoints an item's ID, so every
    checkpointed result can be published. Publication applies it again.
    """
    if not staged.schema.equals(spec.staged_schema, check_metadata=True):
        msg = "Staged results do not match the append's staged schema"
        raise ValueError(msg)
    try:
        canonical = staged.cast(spec.result_fields)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        msg = "Staged results do not convert to the canonical result types"
        raise ValueError(msg) from exc
    for batch in canonical.to_batches(max_chunksize=VALIDATION_BATCH_SIZE):
        spec.validate_batch(batch)
    return canonical


def expanded_rows(results: pa.Table, spec: AppendSpec) -> Iterator[pa.Table]:
    """Expand validated results and require at least one durable row for every item.

    Empty results and terminal errors must survive in the current table. Only
    an empty input may expand to no rows; yield a typed empty table in that case.
    """
    item_ids = results[spec.id_column].combine_chunks()
    covered: set[str] = set()
    expanded = False
    for rows in spec.expand(results):
        try:
            canonical = rows.select(spec.output_fields.names).cast(spec.output_fields)
        except (KeyError, pa.ArrowInvalid, pa.ArrowTypeError) as exc:
            msg = "Expanded rows do not match the producer's output fields"
            raise ValueError(msg) from exc
        if canonical.num_columns != rows.num_columns:
            msg = "Expanded rows contain fields the producer does not own"
            raise ValueError(msg)
        named = pc.is_in(canonical[spec.id_column], value_set=item_ids)
        if canonical.num_rows and not pc.all(named).as_py():
            msg = f"Expanded rows name a {spec.id_column} outside the expanded item results"
            raise ValueError(msg)
        expanded = True
        covered.update(pc.unique(canonical[spec.id_column]).to_pylist())
        yield canonical
    if covered != set(item_ids.to_pylist()):
        msg = "Every selected item must expand to at least one row, including empty outcomes and terminal errors"
        raise ValueError(msg)
    if not expanded:
        yield spec.output_fields.empty_table()


@attrs.frozen
class AppendBindings(RecoveryBindings):
    """Bind a spec to one invocation's table, staging root, and execution settings.

    The spec says what is produced; the bindings say where and how. Only ``spec``
    and the table's identity contribute to recovery compatibility. ``execution``
    holds the recipe's shared tuning, which never enters the manifest; storage
    credentials are resolved at runtime from its profile.
    """

    uri: str
    spec: AppendSpec
    staging_root: str
    execution: AppendExecutionConfig
