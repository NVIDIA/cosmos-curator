# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe-facing enrichment declarations: specs, owned fields, staged schemas, and bindings."""

from pathlib import Path

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.enrichment import contracts
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.enrichment.helpers import make_bindings


@pytest.mark.parametrize(
    "fault",
    [
        "missing_input_id",
        "duplicate_name",
        "empty_outputs",
        "required_output",
        "unowned_output",
        "mixed_owners",
        "overlapping_output",
        "reserved_name",
        "unstable_staging",
    ],
)
def test_invalid_contracts_fail_at_construction(spec: contracts.EnrichmentSpec, fault: str) -> None:
    """Reject malformed recipe contracts before a dataset, worker, or checkpoint is needed."""
    result = spec.output_fields.field("result")
    invalid = {
        "missing_input_id": ({"input_fields": pa.schema([spec.input_fields.field("uri")])}, "ID must be an input"),
        "duplicate_name": ({"output_fields": spec.output_fields.append(result)}, "unique field names"),
        "empty_outputs": ({"output_fields": pa.schema([])}, "nonempty and nullable"),
        "required_output": (
            {
                "output_fields": spec.output_fields.set(
                    0, spec.output_fields.field("result").with_nullable(nullable=False)
                )
            },
            "nonempty and nullable",
        ),
        "unowned_output": (
            {"output_fields": pa.schema([field.with_metadata({}) for field in spec.output_fields])},
            "one owner and field set",
        ),
        "mixed_owners": (
            {
                "output_fields": spec.output_fields.set(
                    0,
                    spec.output_fields.field("result").with_metadata(
                        {contracts.OWNER_METADATA: b"item-recipe", contracts.FIELD_SET_METADATA: b"other"}
                    ),
                )
            },
            "one owner and field set",
        ),
        "overlapping_output": (
            {"output_fields": spec.output_fields.append(spec.input_fields.field("uri"))},
            "must be disjoint",
        ),
        "reserved_name": (
            {"input_fields": spec.input_fields.append(pa.field("fragment_id", pa.int64()))},
            "reserved row addresses",
        ),
        # Checkpointed rows must read back unchanged; Parquet stores date64 as date32.
        "unstable_staging": (
            {"output_fields": spec.output_fields.append(result.with_name("day").with_type(pa.date64()))},
            "'day' of type date64.* reads back from Parquet as date32",
        ),
    }
    changes, message = invalid[fault]
    with pytest.raises(ValueError, match=message):
        attrs.evolve(spec, **changes)


def test_staged_lists_read_back_from_parquet_and_cast_to_canonical_types(tmp_path: Path) -> None:
    """Fixed-size and nested lists stage as Parquet reads them back, including null rows."""
    items = pa.schema([pa.field("item_id", pa.string(), nullable=False)])
    labels = pa.struct([("tags", pa.list_(pa.string())), ("boxes", pa.large_list(pa.list_(pa.int32(), 4)))])
    fields = contracts.owned_fields(
        pa.schema([("vector", pa.list_(pa.float32(), 2)), ("labels", labels)]), owner="item-recipe", field_set="vector"
    )
    spec = contracts.EnrichmentSpec(
        id_column="item_id",
        input_fields=items,
        output_fields=fields,
        result_contract={},
        validate_batch=lambda _batch: None,
    )

    def element(value_type: pa.DataType) -> pa.Field:
        return pa.field("element", value_type)

    assert spec.staged_schema.names == ["fragment_id", "row_offset", "item_id", "vector", "labels"]
    assert spec.staged_schema.field("vector").type == pa.list_(element(pa.float32()))
    assert spec.staged_schema.field("labels").type == pa.struct(
        [("tags", pa.list_(element(pa.string()))), ("boxes", pa.large_list(element(pa.list_(element(pa.int32())))))]
    )
    staged = pa.Table.from_pylist(
        [
            {
                "fragment_id": 0,
                "row_offset": 0,
                "item_id": "a",
                "vector": [1.0, 0.0],
                "labels": {"tags": ["car"], "boxes": [[0, 0, 4, 4]]},
            },
            {"fragment_id": 0, "row_offset": 1, "item_id": "b", "vector": None, "labels": None},
        ],
        schema=spec.staged_schema,
    )
    path = tmp_path / "staged.parquet"
    pq.write_table(staged, path)
    restored = pq.read_table(path)

    assert restored.schema.equals(spec.staged_schema, check_metadata=True)
    canonical = contracts.canonical_results(restored, spec)
    assert canonical.schema.equals(pa.schema([items.field("item_id"), *fields]), check_metadata=True)
    assert canonical["vector"].to_pylist() == [[1.0, 0.0], None]


def test_owned_fields_mark_registrable_outputs(spec: contracts.EnrichmentSpec) -> None:
    """Declared outputs become nullable, keep their own metadata, and name one owner and field set."""
    fields = contracts.owned_fields(
        pa.schema([pa.field("score", pa.float64(), nullable=False, metadata={b"unit": b"ratio"})]),
        owner="scorer",
        field_set="score_v1",
    )
    assert fields.field("score").nullable
    assert fields.field("score").metadata == {
        b"unit": b"ratio",
        contracts.OWNER_METADATA: b"scorer",
        contracts.FIELD_SET_METADATA: b"score_v1",
    }
    assert (spec.owner, spec.field_set) == ("item-recipe", "result")


def test_storage_environment_covers_the_table_staging_root_and_recipe_locations(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recipes add only their own locations; the bindings always include the table and staging root."""
    calls = []

    def environment(*locations: str, storage_profile: str) -> dict[str, str]:
        calls.append((locations, storage_profile))
        return {"AWS_EXAMPLE": "value"}

    monkeypatch.setattr(contracts, "s3_compatibility_env_vars", environment)
    bindings = make_bindings(dataset.uri, spec, staging_root=str(tmp_path / "staging"))
    assert bindings.storage_env_vars("s3://media") == {"AWS_EXAMPLE": "value"}
    assert calls == [((dataset.uri, str(tmp_path / "staging"), "s3://media"), "default")]


def test_spec_snapshots_its_result_contract(spec: contracts.EnrichmentSpec) -> None:
    """Caller mutations cannot change the contract that defines results, provenance, and recovery."""
    contract = {"model": {"revision": "v1"}}
    snapshotted = attrs.evolve(spec, result_contract=contract)
    contract["model"]["revision"] = "v2"
    snapshotted.result_contract["model"]["revision"] = "v3"
    assert snapshotted.result_contract == {"model": {"revision": "v1"}}
    assert snapshotted.contract_digest == canonical_digest({"model": {"revision": "v1"}})
    assert attrs.evolve(snapshotted, validate_input=None).result_contract == snapshotted.result_contract
