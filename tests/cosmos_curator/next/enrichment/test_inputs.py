# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pending selection, publication evidence, and recovery identity with real Lance files."""

from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import contracts, inputs
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig
from tests.cosmos_curator.next.enrichment.helpers import commit_items, make_bindings, prepare_items


def test_workspace_tracks_input_file_layout_only(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Only the selected input file layout and the result contract define recovery compatibility."""
    root = str(tmp_path / "staging")
    workspace = inputs.resolve_workspace(
        dataset, make_bindings(dataset.uri, spec, staging_root=root, result_contract={"version": 1}), selected=(0, 1)
    )
    assert workspace.manifest["workspace_version"] == 3
    assert [fragment["fragment_id"] for fragment in workspace.manifest["input_fragments"]] == [0, 1]
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
    assert (
        inputs.resolve_workspace(
            current,
            make_bindings(current.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(1, 0),
        )
        == workspace
    )
    published = commit_items(lambda *_: prepare_items(current, spec), current, spec)
    assert (
        inputs.resolve_workspace(
            published,
            make_bindings(published.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )
        == workspace
    )
    assert published.get_fragment(0).to_table(columns=["unrelated"]).column(0).to_pylist() == ["keep"]
    assert (
        inputs.resolve_workspace(
            published,
            make_bindings(published.uri, spec, staging_root=root, result_contract={"version": 2}),
            selected=(0, 1),
        )
        != workspace
    )
    appended = lance.write_dataset(
        pa.Table.from_pylist([{"item_id": "c", "uri": "c.mp4"}], schema=spec.input_fields),
        dataset.uri,
        mode="append",
        data_storage_version="2.2",
    )
    assert (
        inputs.resolve_workspace(
            appended,
            make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )
        == workspace
    )
    assert (
        inputs.resolve_workspace(
            appended,
            make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1, 2),
        )
        != workspace
    )
    assert inputs.workspace_prefix(
        appended, make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1})
    ) == str(Path(workspace.root).parent)
    published.delete("item_id = 'a'")
    with pytest.raises(ValueError, match="deleted rows"):
        inputs.resolve_workspace(
            published,
            make_bindings(published.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )


def test_recovery_identity_ignores_caller_mutations_and_runtime_settings(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec snapshots its result contract, and runtime tuning cannot change a resolved recovery identity."""
    contract = {"model": {"revision": "v1"}}
    bindings = make_bindings(dataset.uri, spec, staging_root=str(tmp_path / "staging"), result_contract=contract)
    workspace = inputs.resolve_workspace(dataset, bindings, selected=(0, 1))
    contract["model"]["revision"] = "v2"
    bindings.spec.result_contract["model"]["revision"] = "v3"
    assert bindings.spec.result_contract == {"model": {"revision": "v1"}}
    assert inputs.resolve_workspace(dataset, bindings, selected=(0, 1)) == workspace

    def unexpected_credentials(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Recovery identity must not resolve runtime credentials")

    monkeypatch.setattr(contracts, "get_lance_storage_options", unexpected_credentials)
    tuned = attrs.evolve(
        bindings, execution=EnrichmentExecutionConfig(storage_profile="another-profile", commit_attempts=9)
    )
    assert inputs.resolve_workspace(dataset, tuned, selected=(0, 1)) == workspace


def test_receipts_skip_completed_fragments_across_selections_and_disjoint_writes(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the appended fragment is read after separate publications and unrelated column writes."""
    current = dataset
    for fragment_id in (0, 1):
        candidates = prepare_items(current, spec, (fragment_id,))
        current = commit_items(lambda *_, prepared=candidates: prepared, current, spec, selected=(fragment_id,))
    current.add_columns({"unrelated": "'keep'"})
    current = lance.write_dataset(
        pa.Table.from_pylist([{"item_id": "c", "uri": "c.mp4", "unrelated": "keep"}], schema=current.schema),
        dataset.uri,
        mode="append",
        data_storage_version="2.2",
    )
    validate = Mock(wraps=inputs.fragment_complete)
    monkeypatch.setattr(inputs, "fragment_complete", validate)

    assert inputs.pending_fragment_ids(current, make_bindings(current.uri, spec)) == (2,)
    validate.assert_called_once()
    assert validate.call_args.args[0].fragment_id == 2


@pytest.mark.parametrize("receipt", ["absent", "pruned"])
def test_missing_publication_evidence_falls_back_to_values(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    monkeypatch: pytest.MonkeyPatch,
    receipt: str,
) -> None:
    """Valid completed fragments remain usable without retained receipts, but must be scanned."""
    candidates = prepare_items(dataset, spec)
    if receipt == "absent":
        current = lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(
                updated_fragments=[candidate.fragment for candidate in candidates],
                fields_modified=list(candidates[0].fields_modified),
            ),
            read_version=dataset.version,
        )
    else:
        published = commit_items(lambda *_: candidates, dataset, spec)
        publication_version = published.version
        published.add_columns({"unrelated": "'keep'"})
        current = lance.dataset(dataset.uri)
        current.cleanup_old_versions(versions=[publication_version])
    validate = Mock(wraps=inputs.fragment_complete)
    monkeypatch.setattr(inputs, "fragment_complete", validate)

    assert inputs.pending_fragment_ids(current, make_bindings(current.uri, spec)) == ()
    assert [call.args[0].fragment_id for call in validate.call_args_list] == [0, 1]


@pytest.mark.parametrize("field", ["uri", "result"])
def test_changed_files_invalidate_publication_evidence(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """Receipts cannot suppress validation after input or owned result files change."""
    candidates = prepare_items(dataset, spec)
    published = commit_items(lambda *_: candidates, dataset, spec)
    fragment = published.get_fragment(0)
    values = fragment.to_table(columns=["item_id"]).append_column(
        field, pa.array([None], type=published.schema.field(field).type)
    )
    metadata, modified = fragment.update_columns(values, left_on="item_id", right_on="item_id")
    current = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
        read_version=published.version,
    )
    validate = Mock(wraps=inputs.fragment_complete)
    monkeypatch.setattr(inputs, "fragment_complete", validate)
    bindings = make_bindings(current.uri, spec)

    if field == "result":
        with pytest.raises(ValueError, match="Invalid test result"):
            inputs.pending_fragment_ids(current, bindings)
    else:
        assert inputs.pending_fragment_ids(current, bindings) == ()
    assert validate.call_args_list[0].args[0].fragment_id == 0


@pytest.mark.parametrize("selected", [(0, 0), (42,)])
def test_invalid_selections_are_rejected(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, selected: tuple[int, ...]
) -> None:
    """A frozen job selection cannot alias or refer to absent physical fragments."""
    with pytest.raises(ValueError, match=r"must be unique|is absent"):
        inputs.resolve_workspace(
            dataset, make_bindings(dataset.uri, spec, staging_root=str(tmp_path), result_contract={}), selected=selected
        )


def test_mixed_pending_and_terminal_rows_are_rejected(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec
) -> None:
    """Arrow batch classification preserves the whole-fragment terminal boundary."""
    row = {"item_id": "c", "uri": "c.mp4", "result": 1, "metadata": {"status": "success"}}
    mixed = lance.write_dataset(
        pa.Table.from_pylist([row, {"item_id": "d", "uri": "d.mp4"}], schema=dataset.schema),
        dataset.uri,
        mode="append",
        data_storage_version="2.2",
    )
    with pytest.raises(ValueError, match="mixes pending and terminal"):
        inputs.fragment_complete(mixed.get_fragments()[-1], spec)


@pytest.mark.parametrize("fault", [None, "append", "other_fields", "repeated_fragment", "no_fragments"])
def test_receipts_prove_only_owned_column_updates_of_distinct_fragments(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, fault: str | None
) -> None:
    """Pending selection and workspace recovery read receipts the same way; anything else proves nothing."""
    owned = set(inputs.field_ids(dataset, spec.output_fields.names).values())
    fragments = [candidate.fragment for candidate in prepare_items(dataset, spec)]
    fields = sorted(owned)
    if fault == "other_fields":
        fields = sorted(inputs.field_ids(dataset, ["uri"]).values())
    elif fault == "repeated_fragment":
        fragments.append(fragments[0])
    elif fault == "no_fragments":
        fragments = []
    operation = (
        lance.LanceOperation.Append(fragments)
        if fault == "append"
        else lance.LanceOperation.Update(updated_fragments=fragments, fields_modified=fields)
    )
    prepared = inputs.receipt_fragments(
        lance.Transaction(read_version=int(dataset.version), operation=operation), owned
    )
    if fault is None:
        assert prepared == {0: fragments[0], 1: fragments[1]}
    else:
        assert prepared is None
