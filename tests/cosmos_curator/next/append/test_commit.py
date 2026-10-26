# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One append per selection: canonical files prove success after an ambiguous response."""

import sys
from typing import Any

import attrs
import lance
import pyarrow as pa
import pytest
from lance.fragment import FragmentMetadata

from cosmos_curator.next.append import commit, contracts, inputs
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY
from tests.cosmos_curator.next.append.helpers import ROW_FIELDS, make_items, make_results, publish_attempt, select


def _rows(*item_values: tuple[str, int]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"item_id": item, "row_id": f"{item}:{value}", "value": value} for item, value in item_values],
        schema=ROW_FIELDS,
    )


def _stage(bindings: contracts.AppendBindings, rows: pa.Table) -> list[FragmentMetadata]:
    return [
        FragmentMetadata.from_json(fragment)
        for fragment in commit.stage_fragment(rows, bindings=bindings)["fragment"].to_pylist()
    ]


def _commit(
    bindings: contracts.AppendBindings,
    dataset: lance.LanceDataset,
    fragments: list[FragmentMetadata],
    *,
    selected: tuple[str, ...] = ("a",),
) -> lance.LanceDataset:
    return commit.commit_append(fragments, selected=selected, dataset=dataset, bindings=bindings)


@pytest.fixture
def dataset(bindings: contracts.AppendBindings) -> lance.LanceDataset:
    """Create the empty producer table."""
    return inputs.open_target(bindings)


def test_stage_fragment_writes_one_uncommitted_fragment(
    bindings: contracts.AppendBindings, dataset: lance.LanceDataset
) -> None:
    """Workers return compact descriptors; nothing is visible before the driver commits."""
    assert commit.stage_fragment(ROW_FIELDS.empty_table(), bindings=bindings).num_rows == 0
    (fragment,) = _stage(bindings, _rows(("a", 1), ("b", 2)))
    assert fragment.num_rows == 2
    assert lance.dataset(dataset.uri).count_rows() == 0
    with pytest.raises(ValueError, match="exceeds the configured fragment size"):
        commit.stage_fragment(_rows(("a", 1), ("a", 2), ("a", 3)), bindings=bindings)
    with pytest.raises(ValueError, match="do not match the producer's output fields"):
        commit.stage_fragment(_rows(("a", 1)).drop_columns(["value"]), bindings=bindings)


def test_one_commit_appends_all_outcomes(bindings: contracts.AppendBindings, dataset: lance.LanceDataset) -> None:
    """All selected items contribute rows in one transaction with only compact provenance."""
    fragments = [*_stage(bindings, _rows(("a", 1), ("a", 2))), *_stage(bindings, _rows(("b", 3)))]
    published = _commit(bindings, dataset, fragments, selected=("b", "a"))
    assert published.version == dataset.version + 1
    assert published.count_rows() == 3
    assert [fragment.count_rows() for fragment in published.get_fragments()] == [2, 1]
    properties = published.read_transaction(published.version).transaction_properties
    assert "cosmos_curator.append.publication" not in properties
    assert (properties[KIND_KEY], properties[CONTRACT_DIGEST_KEY]) == ("item-recipe", bindings.spec.contract_digest)


@pytest.mark.parametrize("append_after_commit", [False, True])
def test_lost_commit_response_is_reconciled_without_a_duplicate(
    bindings: contracts.AppendBindings,
    dataset: lance.LanceDataset,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
    capfd: pytest.CaptureFixture[str],
    *,
    append_after_commit: bool,
) -> None:
    """Lance would rebase a retried append over its own success; canonical files prevent a second copy."""
    retrying = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"commit_attempts": 3}))
    fragments = _stage(bindings, _rows(("a", 1)))
    lance_commit = lance.LanceDataset.commit
    calls = []

    def lose_response(*args: object, **kwargs: object) -> None:
        calls.append(1)
        lance_commit(*args, **kwargs)
        if append_after_commit:
            lance.write_dataset(_rows(("z", 9)), dataset.uri, mode="append", data_storage_version="2.2")
        message = "lost response"
        raise OSError(message)

    monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
    sink = commit.logger.add(sys.stderr, level="WARNING")
    try:
        published = _commit(retrying, dataset, fragments)
    finally:
        commit.logger.remove(sink)
    assert len(calls) == 1
    assert published.to_table(columns=["row_id"])["row_id"].to_pylist() == (
        ["a:1", "z:9"] if append_after_commit else ["a:1"]
    )
    (warning,) = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert "OSError('lost response')" in warning
    assert f"verified from canonical files at version {published.version}" in warning
    assert warning in capfd.readouterr().err


def test_failed_commit_without_published_files_is_retried(
    bindings: contracts.AppendBindings, dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit that did not happen is attempted again, from the latest version."""
    retrying = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"commit_attempts": 2}))
    lance_commit = lance.LanceDataset.commit
    calls = []

    def fail_once(*args: object, **kwargs: object) -> lance.LanceDataset:
        calls.append(1)
        if len(calls) == 1:
            message = "transient"
            raise OSError(message)
        return lance_commit(*args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "commit", fail_once)
    published = _commit(retrying, dataset, _stage(bindings, _rows(("a", 1))))
    assert (len(calls), published.count_rows()) == (2, 1)


def test_concurrent_enrichment_is_preserved(
    bindings: contracts.AppendBindings, results: pa.Table, items: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another writer may add a nullable field and fill it on existing rows while new rows are staged."""
    first = publish_attempt(select(bindings, items), results).dataset
    first.add_columns(pa.schema([pa.field("caption", pa.string())]))
    later = make_items(["a", "b", "c", "d"])
    attempt = select(bindings, later)
    lance_commit = lance.LanceDataset.commit
    raced = False

    def enrich_first(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal raced
        if not raced:
            raced = True
            current = lance.dataset(bindings.uri)
            fragment = current.get_fragments()[0]
            values = fragment.to_table(columns=["row_id"]).append_column("caption", pa.array(["keep", "keep"]))
            metadata, modified = fragment.update_columns(values, left_on="row_id", right_on="row_id")
            lance_commit(
                bindings.uri,
                lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
                read_version=current.version,
            )
        return lance_commit(*args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "commit", enrich_first)
    published = publish_attempt(attempt, make_results(bindings.spec, {"d": [4]})).dataset
    table = published.to_table(columns=["row_id", "caption"], filter="value IS NOT NULL").sort_by("row_id")
    assert table.to_pylist() == [
        {"row_id": "a:0", "caption": "keep"},
        {"row_id": "a:1", "caption": "keep"},
        {"row_id": "d:0", "caption": None},
    ]


def test_another_publication_of_selected_items_is_rejected(
    bindings: contracts.AppendBindings, dataset: lance.LanceDataset
) -> None:
    """One producer owns the table; a competing append of the same items stops the commit."""
    _commit(bindings, dataset, _stage(bindings, _rows(("b", 2))), selected=("b",))
    with pytest.raises(ValueError, match="Another publication appended selected items"):
        _commit(bindings, dataset, _stage(bindings, _rows(("a", 1), ("b", 2))), selected=("a", "b"))
    assert lance.dataset(dataset.uri).count_rows() == 1


def test_rebound_table_is_rejected_before_committing(
    bindings: contracts.AppendBindings, dataset: lance.LanceDataset
) -> None:
    """Each attempt rereads the table and requires the producer's contract binding."""
    fragments = _stage(bindings, _rows(("a", 1)))
    lance.write_dataset(
        ROW_FIELDS.empty_table(), dataset.uri, schema=ROW_FIELDS, mode="overwrite", data_storage_version="2.2"
    )
    with pytest.raises(ValueError, match="different producer, item ID column, or result contract"):
        _commit(bindings, dataset, fragments)


def test_verification_requires_the_appended_files(
    bindings: contracts.AppendBindings, dataset: lance.LanceDataset
) -> None:
    """Deleted rows or missing files cannot prove a publication, even after commit."""
    fragments = _stage(bindings, _rows(("a", 1), ("a", 2)))
    published = _commit(bindings, dataset, fragments)
    commit.verify_appended(published, fragments)
    published.add_columns({"doubled": "value * 2"})
    commit.verify_appended(lance.dataset(dataset.uri), fragments)
    lance.dataset(dataset.uri).delete("row_id = 'a:2'")
    with pytest.raises(ValueError, match="Appended rows are missing or changed"):
        commit.verify_appended(lance.dataset(dataset.uri), fragments)
    with pytest.raises(ValueError, match="Appended rows are missing or changed"):
        commit.verify_appended(published, _stage(bindings, _rows(("b", 1))))
