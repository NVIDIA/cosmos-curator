# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared publication invariants using real Lance files and no Ray execution."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import commit, contracts
from tests.cosmos_curator.next.enrichment.helpers import commit_items, make_bindings, prepare_items, validate_items


@pytest.mark.parametrize("fragments", [1, 2])
def test_competing_terminal_results_require_this_publications_receipt(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, fragments: int
) -> None:
    """Even valid terminal values cannot stand in for this transaction's receipt."""
    candidates = prepare_items(dataset, spec)
    competing = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(
            updated_fragments=[row.fragment for row in candidates[:fragments]],
            fields_modified=list(candidates[0].fields_modified),
        ),
        read_version=dataset.version,
    )
    with pytest.raises(ValueError, match="multiple writers"):
        commit_items(lambda *_: candidates, dataset, spec)
    assert lance.dataset(dataset.uri).version == competing.version


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "wrong_fragment", "wrong_fields", "drop_input"])
def test_invalid_descriptors_cannot_publish(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, corruption: str
) -> None:
    """Validate worker control records independently of their staged row validation."""
    candidates = prepare_items(dataset, spec)
    if corruption == "missing":
        candidates.pop()
    elif corruption == "duplicate":
        candidates.append(candidates[0])
    elif corruption == "wrong_fields":
        candidates[0] = attrs.evolve(
            candidates[0], fields_modified=tuple(contracts.field_ids(dataset, ["item_id"]).values())
        )
    else:
        metadata = candidates[0].fragment.to_json()
        if corruption == "wrong_fragment":
            metadata["id"] = 100
        else:
            metadata["files"].pop(0)
        candidates[0] = attrs.evolve(candidates[0], fragment=lance.FragmentMetadata.from_json(json.dumps(metadata)))
    with pytest.raises(ValueError, match=r"exactly cover|unexpected fragment or field|unowned columns"):
        commit_items(lambda *_: candidates, dataset, spec)
    assert lance.dataset(dataset.uri).version == dataset.version
    assert contracts.pending_fragment_ids(dataset, make_bindings(dataset.uri, spec)) == (0, 1)


@pytest.mark.parametrize("mutation", ["delete_input", "replace_result"])
def test_receipt_replay_verifies_canonical_results(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, mutation: str
) -> None:
    """A receipt never hides a later mutation of the selected input or result files."""
    candidates = prepare_items(dataset, spec)
    published = commit_items(lambda *_: candidates, dataset, spec)
    assert commit_items(lambda *_: candidates, dataset, spec).version == published.version
    if mutation == "delete_input":
        published.delete("item_id = 'a'")
    else:
        fragment = published.get_fragment(0)
        values = fragment.to_table(columns=["item_id"]).append_column("result", pa.array([2], type=pa.int64()))
        metadata, modified = fragment.update_columns(values, left_on="item_id", right_on="item_id")
        lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=published.version,
        )
    with pytest.raises(ValueError, match=r"file layout changed|Published result files differ"):
        commit_items(lambda *_: candidates, dataset, spec)


@pytest.mark.parametrize("committed", [False, True])
def test_commit_retry_retains_prepared_files(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    committed: bool,
) -> None:
    """Retry even an unreadable receipt without losing the prepared publication identity."""
    prepare = Mock(
        side_effect=lambda version, selected: prepare_items(lance.dataset(dataset.uri, version=version), spec, selected)
    )
    lance_commit = lance.LanceDataset.commit
    commit_calls = 0

    def interrupted_commit(*args: object, **kwargs: object) -> lance.LanceDataset:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            if committed:
                lance_commit(*args, **kwargs)
            msg = "commit response unavailable"
            raise OSError(msg)
        return lance_commit(*args, **kwargs)

    find_transaction = commit.find_transaction
    receipt_calls = 0

    def interrupted_receipt(*args: object, **kwargs: object) -> lance.Transaction | None:
        nonlocal receipt_calls
        receipt_calls += 1
        if receipt_calls == 2:
            msg = "receipt lookup unavailable"
            raise OSError(msg)
        return find_transaction(*args, **kwargs)

    monkeypatch.setattr(lance.LanceDataset, "commit", interrupted_commit)
    monkeypatch.setattr(commit, "find_transaction", interrupted_receipt)
    sink = commit.logger.add(sys.stderr, level="WARNING")
    try:
        published = commit.commit_fragments(
            prepare,
            dataset=dataset,
            bindings=make_bindings(dataset.uri, spec, commit_attempts=2),
            selected=(0, 1),
            receipt="this-publication",
            transaction_properties={"kind": "test"},
        )
    finally:
        commit.logger.remove(sink)

    prepare.assert_called_once_with(int(dataset.version), (0, 1))
    assert commit_calls == (1 if committed else 2)
    assert receipt_calls == 3
    assert published.version == dataset.version + 1
    assert contracts.pending_fragment_ids(published, make_bindings(published.uri, spec)) == ()
    captured = capfd.readouterr()
    assert "receipt lookup unavailable" in captured.err
    assert "Retrying" in captured.err


def test_publication_leaves_unselected_fragments_pending(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec
) -> None:
    """One job can complete its selection while the rest of the table is unfinished."""
    candidates = prepare_items(dataset, spec, (0,))
    published = commit_items(lambda *_: candidates, dataset, spec, selected=(0,))
    commit.verify_completed(original=dataset, current=published, spec=spec, selected=(0,))
    assert contracts.pending_fragment_ids(published, make_bindings(published.uri, spec)) == (1,)
    assert commit_items(lambda *_: candidates, dataset, spec, selected=(0,)).version == published.version


@pytest.mark.parametrize("mutation", ["input", "result", "delete"])
def test_publication_ignores_unselected_fragment_changes(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, mutation: str
) -> None:
    """Other jobs' input and owned files do not enter this selection's guards."""
    root = str(tmp_path / "staging")
    workspace = contracts.resolve_workspace(
        dataset, make_bindings(dataset.uri, spec, staging_root=root, result_contract={}), selected=(0,)
    )
    candidates = prepare_items(dataset, spec, (0,))
    current = lance.dataset(dataset.uri)
    if mutation == "delete":
        current.delete("item_id = 'b'")
    elif mutation == "input":
        fragment = current.get_fragment(1)
        values = fragment.to_table(columns=["item_id"]).append_column("uri", pa.array(["changed.mp4"]))
        metadata, modified = fragment.update_columns(values, left_on="item_id", right_on="item_id")
        current = lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=current.version,
        )
    else:
        prepared = prepare_items(current, spec, (1,))[0]
        current = lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(
                updated_fragments=[prepared.fragment],
                fields_modified=list(prepared.fields_modified),
            ),
            read_version=current.version,
        )
    assert (
        contracts.resolve_workspace(
            current, make_bindings(current.uri, spec, staging_root=root, result_contract={}), selected=(0,)
        )
        == workspace
    )
    published = commit_items(lambda *_: candidates, dataset, spec, selected=(0,))
    commit.verify_completed(original=dataset, current=published, spec=spec, selected=(0,))
    if mutation == "delete":
        assert published.count_rows() == 1
    elif mutation == "input":
        assert published.get_fragment(1).to_table(columns=["uri"]).column(0).to_pylist() == ["changed.mp4"]
    else:
        assert contracts.fragment_complete(published.get_fragment(1), spec)


def test_only_stale_fragments_are_reprepared(dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec) -> None:
    """A disjoint update on one fragment does not rewrite another fragment's result files."""
    candidates = prepare_items(dataset, spec)
    current = lance.dataset(dataset.uri)
    current.add_columns(pa.schema([("unrelated", pa.string())]))
    fragment = current.get_fragment(0)
    values = fragment.to_table(columns=["item_id"]).append_column("unrelated", pa.array(["keep"]))
    metadata, modified = fragment.update_columns(values, left_on="item_id", right_on="item_id")
    current = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
        read_version=current.version,
    )
    calls = []

    def prepare(version: int, selected: tuple[int, ...]) -> list[commit.PreparedFragment]:
        calls.append((version, selected))
        if version == int(dataset.version):
            return candidates
        return prepare_items(lance.dataset(dataset.uri, version=version), spec, selected)

    published = commit_items(prepare, dataset, spec)
    assert calls == [(int(dataset.version), (0, 1)), (int(current.version), (0,))]
    owned = set(contracts.field_ids(dataset, spec.output_fields.names).values())
    assert contracts.file_layout_digest(published.get_fragment(1).metadata, owned) == contracts.file_layout_digest(
        candidates[1].fragment, owned
    )
    assert published.get_fragment(0).to_table(columns=["unrelated"]).column(0).to_pylist() == ["keep"]


@pytest.mark.parametrize("coverage", [(), (0, 1), (1,)])
def test_reprepared_descriptors_must_exactly_cover_stale_fragments(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, coverage: tuple[int, ...]
) -> None:
    """Refreshing one descriptor cannot silently drop or replace the rest of the job."""
    candidates = prepare_items(dataset, spec)
    current = lance.dataset(dataset.uri)
    current.add_columns(pa.schema([("unrelated", pa.string())]))
    fragment = current.get_fragment(0)
    values = fragment.to_table(columns=["item_id"]).append_column("unrelated", pa.array(["keep"]))
    metadata, modified = fragment.update_columns(values, left_on="item_id", right_on="item_id")
    current = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
        read_version=current.version,
    )

    def prepare(version: int, selected: tuple[int, ...]) -> list[commit.PreparedFragment]:
        if version == int(dataset.version):
            assert selected == (0, 1)
            return candidates
        assert selected == (0,)
        return prepare_items(lance.dataset(dataset.uri, version=version), spec, coverage)

    with pytest.raises(ValueError, match="exactly cover"):
        commit_items(prepare, dataset, spec)
    assert lance.dataset(dataset.uri).version == current.version


def test_terminal_validation_uses_bounded_arrow_batches(
    tmp_path: Path, spec: contracts.EnrichmentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Workers validate bounded batches; publication proves their files without another row scan."""
    rows = 2_049
    uri = str(tmp_path / "large.lance")
    lance.write_dataset(
        pa.table({"item_id": [str(i) for i in range(rows)], "uri": ["video.mp4"] * rows}, schema=spec.input_fields),
        uri,
        data_storage_version="2.2",
    )
    sizes = []

    def validate(batch: pa.RecordBatch) -> None:
        assert isinstance(batch, pa.RecordBatch)
        sizes.append(batch.num_rows)
        validate_items(batch)

    bounded_spec = attrs.evolve(spec, validate_batch=validate)
    dataset = contracts.register_fields(make_bindings(uri, bounded_spec))
    candidates = prepare_items(dataset, bounded_spec)
    assert sizes == [1_024, 1_024, 1]
    sizes.clear()
    monkeypatch.setattr(lance.fragment.LanceFragment, "scanner", Mock(side_effect=AssertionError("Driver row scan")))
    commit_items(lambda *_: candidates, dataset, bounded_spec, selected=(0,))
    assert sizes == []


@pytest.mark.parametrize("corruption", ["duplicate_offset", "null_offset", "wrong_id", "invalid_value"])
def test_staged_coverage_and_values_are_validated(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, corruption: str
) -> None:
    """Arrow coverage checks reject misaligned addresses and nonterminal values before commit."""
    row = {"fragment_id": 0, "row_offset": 0, "item_id": "a", "result": 1, "metadata": {"status": "success"}}
    if corruption == "null_offset":
        row["row_offset"] = None
    elif corruption == "wrong_id":
        row["item_id"] = "b"
    elif corruption == "invalid_value":
        row["result"] = None
    rows = [row, row] if corruption == "duplicate_offset" else [row]
    with pytest.raises(ValueError, match=r"misaligned|Invalid test result"):
        commit.prepare_fragment(
            pa.Table.from_pylist(rows, schema=spec.staged_schema),
            version=int(dataset.version),
            bindings=make_bindings(dataset.uri, spec),
        )


def test_duplicate_input_ids_are_rejected_before_join(tmp_path: Path, spec: contracts.EnrichmentSpec) -> None:
    """Duplicate IDs cannot make the fragment update join choose an arbitrary result."""
    uri = str(tmp_path / "duplicates.lance")
    lance.write_dataset(
        pa.table({"item_id": ["a", "a"], "uri": ["a.mp4", "a.mp4"]}, schema=spec.input_fields),
        uri,
        data_storage_version="2.2",
    )
    dataset = contracts.register_fields(make_bindings(uri, spec))
    with pytest.raises(ValueError, match="non-null and unique"):
        prepare_items(dataset, spec)


def test_terminal_validation_sees_canonical_columns(tmp_path: Path) -> None:
    """Publication and pending scans give validators the same canonical columns and types."""
    items = pa.schema([pa.field("item_id", pa.string(), nullable=False)])
    vector = pa.field("vector", pa.list_(pa.float32(), 2))
    seen = []

    def validate(batch: pa.RecordBatch) -> None:
        seen.append([(field.name, field.type) for field in batch.schema])
        if batch.column("vector").null_count:
            msg = "Missing vector"
            raise ValueError(msg)

    spec = contracts.EnrichmentSpec(
        id_column="item_id",
        input_fields=items,
        output_fields=contracts.owned_fields(pa.schema([vector]), owner="item-recipe", field_set="vector"),
        validate_batch=validate,
    )
    uri = str(tmp_path / "vectors.lance")
    lance.write_dataset(pa.table({"item_id": ["a", "b"]}, schema=items), uri, data_storage_version="2.2")
    dataset = contracts.register_fields(make_bindings(uri, spec))
    staged = pa.table(
        {"fragment_id": [0, 0], "row_offset": [0, 1], "item_id": ["a", "b"], "vector": [[1.0, 0.0], [0.0, 1.0]]},
        schema=spec.staged_schema,
    )
    bindings = make_bindings(uri, spec)
    candidates = [
        commit.PreparedFragment.from_row(row)
        for row in commit.prepare_fragment(staged, bindings=bindings, version=int(dataset.version)).to_pylist()
    ]
    published = commit_items(lambda *_: candidates, dataset, spec, selected=(0,))
    assert contracts.fragment_complete(published.get_fragment(0), spec)
    canonical = [("item_id", pa.string()), ("vector", vector.type)]
    assert seen == [canonical, canonical]

    staged_vector = spec.staged_schema.field("vector")
    short = staged.set_column(3, staged_vector, pa.array([[1.0], [0.0, 1.0]], staged_vector.type))
    with pytest.raises(ValueError, match="do not convert to the canonical output types"):
        commit.prepare_fragment(short, bindings=bindings, version=int(dataset.version))
