# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The target table's contract binding, work item validation, published items, and recovery identity."""

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.append import contracts, inputs
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY
from cosmos_curator.next.utils.identity import canonical_digest, canonical_json
from tests.cosmos_curator.next.append.helpers import ROW_FIELDS, make_items, publish_attempt, select


def _create(bindings: contracts.AppendBindings, schema: pa.Schema, **kwargs: object) -> lance.LanceDataset:
    return lance.write_dataset(
        schema.empty_table(), bindings.uri, schema=schema, **{"data_storage_version": "2.2", **kwargs}
    )


def test_target_is_created_once_with_its_contract_binding(bindings: contracts.AppendBindings) -> None:
    """Creation commits the empty producer schema and provenance; reopening commits nothing."""
    dataset = inputs.open_target(bindings)
    assert dataset.count_rows() == 0
    assert dataset.schema.equals(inputs.target_schema(bindings.spec), check_metadata=True)
    binding = json.loads(dataset.schema.metadata[inputs.CONTRACT_METADATA])
    assert binding == {
        "format_version": 2,
        "owner": "item-recipe",
        "id_column": "item_id",
        "result_contract": {"version": 1},
    }
    properties = dataset.read_transaction(dataset.version).transaction_properties
    assert properties[KIND_KEY] == "item-recipe"
    assert properties[CONTRACT_DIGEST_KEY] == canonical_digest({"version": 1})
    assert inputs.open_target(bindings).version == dataset.version


@pytest.mark.parametrize("change", ["contract", "owner"])
def test_target_belongs_to_one_producer_contract(bindings: contracts.AppendBindings, change: str) -> None:
    """Another result contract or producer needs a new output location."""
    inputs.open_target(bindings)
    spec = bindings.spec
    other = (
        attrs.evolve(spec, result_contract={"version": 2}) if change == "contract" else attrs.evolve(spec, owner="x")
    )
    with pytest.raises(ValueError, match="different producer, item ID column, or result contract"):
        inputs.open_target(attrs.evolve(bindings, spec=other))


def test_item_id_column_binds_the_table_and_workspace(bindings: contracts.AppendBindings, items: pa.Table) -> None:
    """Choosing another string column cannot reinterpret canonical item IDs or checkpoints."""
    spec = bindings.spec
    uri = spec.input_fields.field("uri")
    spec = attrs.evolve(
        spec, result_fields=spec.result_fields.append(uri), output_fields=spec.output_fields.append(uri)
    )
    bindings = attrs.evolve(bindings, spec=spec)
    items = inputs.validate_items(items, spec)
    dataset = inputs.open_target(bindings)
    changed = attrs.evolve(bindings, spec=attrs.evolve(spec, id_column="uri"))
    assert inputs.resolve_workspace(dataset, changed, selection=items) != inputs.resolve_workspace(
        dataset, bindings, selection=items
    )
    with pytest.raises(ValueError, match="different producer, item ID column, or result contract"):
        inputs.open_target(changed)


@pytest.mark.parametrize("missing", ["id_column", "format_version"])
def test_target_rejects_an_earlier_binding(
    bindings: contracts.AppendBindings,
    missing: str,
) -> None:
    """Earlier development bindings cannot prove current item completion semantics."""
    schema = inputs.target_schema(bindings.spec)
    binding = json.loads(schema.metadata[inputs.CONTRACT_METADATA])
    del binding[missing]
    _create(bindings, schema.with_metadata({inputs.CONTRACT_METADATA: canonical_json(binding).encode()}))
    with pytest.raises(ValueError, match=r"item ID column.*use a new output location"):
        inputs.open_target(bindings)


@pytest.mark.parametrize(
    "fields",
    [
        pa.schema([ROW_FIELDS.field("row_id"), ROW_FIELDS.field("item_id"), ROW_FIELDS.field("value")]),
        ROW_FIELDS.remove(2),
        ROW_FIELDS.set(2, pa.field("value", pa.int32(), nullable=False)),
        pa.schema([pa.field("other", pa.string()), *ROW_FIELDS]),
    ],
    ids=["reordered", "missing", "retyped", "preceded"],
)
def test_target_must_begin_with_the_producer_fields(bindings: contracts.AppendBindings, fields: pa.Schema) -> None:
    """Staged fragments hold exactly the producer's fields, which Lance numbers by position."""
    metadata = inputs.target_schema(bindings.spec).metadata
    _create(bindings, fields.with_metadata(metadata))
    with pytest.raises(ValueError, match="does not begin with this producer's fields"):
        inputs.open_target(bindings)


@pytest.mark.parametrize("nullable", [True, False])
def test_target_accepts_only_nullable_extensions(bindings: contracts.AppendBindings, *, nullable: bool) -> None:
    """Appended rows read other writers' fields as null, so those fields must accept null."""
    schema = inputs.target_schema(bindings.spec)
    _create(bindings, schema.append(pa.field("caption", pa.string(), nullable=nullable)))
    if nullable:
        assert inputs.open_target(bindings).schema.names[-1] == "caption"
    else:
        with pytest.raises(ValueError, match=r"non-nullable fields that appended rows cannot fill: \['caption'\]"):
            inputs.open_target(bindings)


def test_target_requires_the_shared_storage_version(bindings: contracts.AppendBindings) -> None:
    """Every Curator Next table shares one fragment format."""
    _create(bindings, inputs.target_schema(bindings.spec), data_storage_version="2.1")
    with pytest.raises(ValueError, match=r"data storage version 2\.1"):
        inputs.open_target(bindings)


@pytest.mark.parametrize(
    ("discovered", "message"),
    [
        (make_items(["a", "a"]), "must be unique"),
        (pa.table({"item_id": ["a", None], "uri": ["a", "b"]}, schema=make_items([]).schema), "must be non-null"),
        (pa.table({"item_id": ["a", "b"], "uri": ["a", None]}, schema=make_items([]).schema), "must be non-null"),
        (pa.table({"item_id": ["a"], "uri": ["a"]}), "do not match the append's input fields"),
    ],
    ids=["duplicate", "null-id", "null-uri", "schema"],
)
def test_invalid_items_fail_before_a_selection_is_saved(
    bindings: contracts.AppendBindings, discovered: pa.Table, message: str
) -> None:
    """Invalid required fields or IDs fail before saving a selection that recovery could not accept."""
    with pytest.raises(ValueError, match=message):
        select(bindings, discovered)
    assert lance.dataset(bindings.uri).count_rows() == 0
    assert not Path(bindings.staging_root).exists()


def test_items_are_kept_in_id_order(bindings: contracts.AppendBindings, items: pa.Table) -> None:
    """Selections, limits, and workspace identity do not depend on discovery order."""
    validated = inputs.validate_items(items, bindings.spec)
    assert validated["item_id"].to_pylist() == ["a", "b", "c"]
    assert validated.column("item_id").num_chunks == 1


def test_recovery_identity_binds_the_selection_and_contract(
    bindings: contracts.AppendBindings, items: pa.Table, tmp_path: Path
) -> None:
    """Selections reuse one path with distinct manifests; another contract or table has its own path."""
    items = inputs.validate_items(items, bindings.spec)
    dataset = inputs.open_target(bindings)
    workspace = inputs.resolve_workspace(dataset, bindings, selection=items)
    tuned = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"rows_per_checkpoint": 7}))
    assert inputs.resolve_workspace(dataset, tuned, selection=items) == workspace
    assert workspace.root == inputs.workspace_root(dataset, bindings)
    assert workspace.manifest["selected_items"] == items.to_pylist()
    subset = inputs.resolve_workspace(dataset, bindings, selection=items.slice(0, 2))
    assert subset.root == workspace.root
    assert subset.manifest_digest != workspace.manifest_digest
    assert subset != workspace
    recontracted = attrs.evolve(bindings, spec=attrs.evolve(bindings.spec, result_contract={"version": 2}))
    assert inputs.workspace_root(dataset, recontracted) != inputs.workspace_root(dataset, bindings)
    other = inputs.open_target(attrs.evolve(bindings, uri=str(tmp_path / "other.lance")))
    assert inputs.workspace_root(other, bindings) != inputs.workspace_root(dataset, bindings)


def test_current_rows_cover_every_item_after_pruning(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty/error outcomes survive pruning and discovery only returns candidate IDs."""
    published = publish_attempt(select(bindings, items), results).dataset
    published.add_columns(pa.schema([pa.field("extra", pa.string())]))
    published.cleanup_old_versions(older_than=timedelta(seconds=0))
    monkeypatch.setattr(lance.LanceDataset, "read_transaction", Mock(side_effect=AssertionError("History read")))
    current = lance.dataset(bindings.uri)
    assert inputs.published_item_ids(current, bindings, ["a", "b", "c", "absent"]) == {"a", "b", "c"}
    assert inputs.published_item_ids(current, bindings, ["b", "absent"]) == {"b"}
    assert select(bindings, items).selected == ()
    assert select(bindings, make_items(["a", "b", "c", "d"])).selected == ("d",)
