# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append contracts reject malformed declarations and every result publication could not accept."""

from collections.abc import Iterator
from unittest.mock import Mock

import attrs
import pyarrow as pa
import pytest

from cosmos_curator.next.append import contracts
from tests.cosmos_curator.next.append.helpers import ITEM_FIELDS, RESULT_FIELDS, ROW_FIELDS, make_results


def test_item_results_stage_in_their_canonical_types(spec: contracts.AppendSpec) -> None:
    """Staged rows keep the canonical result schema; a converted storage representation is rejected."""
    results = make_results(spec, {"a": [1]})
    assert contracts.canonical_results(results, spec).equals(results)
    converted = results.cast(RESULT_FIELDS.set(1, pa.field("values", pa.list_(pa.field("element", pa.int64())))))
    with pytest.raises(ValueError, match="result schema"):
        contracts.canonical_results(converted, spec)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"owner": ""}, "requires an owner"),
        ({"id_column": "missing"}, "must all carry the item ID"),
        ({"output_fields": ROW_FIELDS.remove(0)}, "must all carry the item ID"),
        ({"input_fields": ITEM_FIELDS.append(pa.field("uri", pa.string()))}, "unique field names"),
        ({"id_column": "uri"}, "must all carry the item ID"),
        (
            {"result_fields": RESULT_FIELDS.set(0, pa.field("item_id", pa.large_string(), nullable=False))},
            "same non-null string field",
        ),
        (
            {
                "input_fields": ITEM_FIELDS.set(0, pa.field("item_id", pa.int64(), nullable=False)),
                "result_fields": RESULT_FIELDS.set(0, pa.field("item_id", pa.int64(), nullable=False)),
                "output_fields": ROW_FIELDS.set(0, pa.field("item_id", pa.int64(), nullable=False)),
            },
            "same non-null string field",
        ),
        ({"input_fields": ITEM_FIELDS.append(pa.field("score", pa.float64()))}, "strings, integers, or booleans"),
        ({"error_column": "missing"}, "error column must be an item result field"),
        ({"error_column": "item_id"}, "error column must be an item result field"),
        ({"error_column": "values"}, "nullable string"),
        (
            {"result_fields": RESULT_FIELDS.set(2, pa.field("error", pa.string(), nullable=False))},
            "nullable string",
        ),
    ],
    ids=[
        "owner",
        "missing-id",
        "rows-without-id",
        "duplicate-name",
        "id-not-in-results",
        "id-type-differs",
        "non-string-id",
        "item-float",
        "error-not-a-result",
        "error-is-the-id",
        "error-not-a-string",
        "error-not-nullable",
    ],
)
def test_malformed_contracts_fail_before_processing(
    spec: contracts.AppendSpec, change: dict[str, object], message: str
) -> None:
    """Every checkpoint and appended row must name the same kind of item."""
    with pytest.raises(ValueError, match=message):
        attrs.evolve(spec, **change)


def test_result_contract_is_an_immutable_snapshot(spec: contracts.AppendSpec) -> None:
    """Caller mutations cannot change the identity of an existing contract."""
    contract = {"version": 1, "settings": {"size": [2, 3]}}
    snapshot = attrs.evolve(spec, result_contract=contract)
    digest = snapshot.contract_digest
    contract["settings"]["size"].append(4)  # type: ignore[index]
    snapshot.result_contract["settings"]["size"].append(5)
    assert snapshot.result_contract == {"version": 1, "settings": {"size": [2, 3]}}
    assert snapshot.contract_digest == digest


def test_canonical_results_validate_bounded_batches(spec: contracts.AppendSpec) -> None:
    """Terminal validation sees canonical types in batches of at most 1,024 item results."""
    validator = Mock()
    checked = attrs.evolve(spec, validate_batch=validator)
    staged = make_results(spec, {f"item-{index}": [index] for index in range(2_049)})
    contracts.canonical_results(staged, checked)
    sizes = [call.args[0].num_rows for call in validator.call_args_list]
    assert sizes == [1_024, 1_024, 1]
    assert all(call.args[0].schema.equals(RESULT_FIELDS) for call in validator.call_args_list)


@pytest.mark.parametrize("fault", ["schema", "invalid"])
def test_canonical_results_reject_unpublishable_rows(spec: contracts.AppendSpec, fault: str) -> None:
    """A row publication would reject fails processing before Ray can checkpoint its item."""
    staged = make_results(spec, {"a": [1]})
    if fault == "schema":
        with pytest.raises(ValueError, match="result schema"):
            contracts.canonical_results(staged.drop_columns(["values"]), spec)
    else:
        rows = staged.to_pylist()
        rows[0]["error"] = "unreadable a"
        with pytest.raises(ValueError, match="Invalid test result for a"):
            contracts.canonical_results(pa.Table.from_pylist(rows, schema=staged.schema), spec)


def _expansion(*tables: pa.Table) -> object:
    def expand(_results: pa.Table) -> Iterator[pa.Table]:
        yield from tables

    return expand


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (pa.table({"item_id": ["a"], "value": [1]}), "do not match the producer's output fields"),
        (
            pa.table({"item_id": ["a"], "row_id": ["a:0"], "value": [1], "extra": [1]}),
            "fields the producer does not own",
        ),
        (pa.table({"item_id": ["x"], "row_id": ["x:0"], "value": [1]}), "outside the expanded item results"),
        (pa.table({"item_id": ["a"], "row_id": ["a:0"], "value": ["one"]}), "do not match the producer's output"),
    ],
    ids=["missing-field", "extra-field", "foreign-item", "wrong-type"],
)
def test_expansion_rejects_rows_the_producer_cannot_append(
    spec: contracts.AppendSpec, rows: pa.Table, message: str
) -> None:
    """Every appended row has the producer's fields and names one of the items being expanded."""
    results = make_results(spec, {"a": [1]})
    with pytest.raises(ValueError, match=message):
        list(
            contracts.expanded_rows(
                results,
                attrs.evolve(
                    spec, expand=_expansion(rows.append_column("error", pa.nulls(rows.num_rows, type=pa.string())))
                ),
            )
        )


def test_empty_and_error_results_must_leave_durable_rows(spec: contracts.AppendSpec) -> None:
    """Both outcomes retain their item IDs and error values in the canonical table."""
    results = make_results(spec, {"a": [], "b": None})
    rows = pa.concat_tables(list(contracts.expanded_rows(results, spec)))
    assert rows["item_id"].to_pylist() == ["a", "b"]
    assert rows["error"].to_pylist() == [None, "unreadable b"]
    with pytest.raises(ValueError, match="Every selected item must expand"):
        list(contracts.expanded_rows(results, attrs.evolve(spec, expand=_expansion(ROW_FIELDS.empty_table()))))
