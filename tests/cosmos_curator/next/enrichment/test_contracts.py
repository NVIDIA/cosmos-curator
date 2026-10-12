# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe contracts, pending selection, and recovery identity with real Lance files."""

from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.enrichment import contracts
from cosmos_curator.next.enrichment.config import EnrichmentExecutionConfig
from tests.cosmos_curator.next.enrichment.helpers import commit_items, make_bindings, prepare_items


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
        id_column="item_id", input_fields=items, output_fields=fields, validate_batch=lambda _batch: None
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


def test_workspace_tracks_input_file_layout_only(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Only the selected input file layout and the result contract define recovery compatibility."""
    root = str(tmp_path / "staging")
    workspace = contracts.resolve_workspace(
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
        contracts.resolve_workspace(
            current,
            make_bindings(current.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(1, 0),
        )
        == workspace
    )
    published = commit_items(lambda *_: prepare_items(current, spec), current, spec)
    assert (
        contracts.resolve_workspace(
            published,
            make_bindings(published.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )
        == workspace
    )
    assert published.get_fragment(0).to_table(columns=["unrelated"]).column(0).to_pylist() == ["keep"]
    assert (
        contracts.resolve_workspace(
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
        contracts.resolve_workspace(
            appended,
            make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )
        == workspace
    )
    assert (
        contracts.resolve_workspace(
            appended,
            make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1, 2),
        )
        != workspace
    )
    assert contracts.workspace_prefix(
        appended, make_bindings(appended.uri, spec, staging_root=root, result_contract={"version": 1})
    ) == str(Path(workspace.root).parent)
    published.delete("item_id = 'a'")
    with pytest.raises(ValueError, match="deleted rows"):
        contracts.resolve_workspace(
            published,
            make_bindings(published.uri, spec, staging_root=root, result_contract={"version": 1}),
            selected=(0, 1),
        )


def test_bindings_snapshot_contract_and_exclude_runtime_settings(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller mutations and runtime tuning cannot change a resolved recovery identity."""
    contract = {"model": {"revision": "v1"}}
    bindings = make_bindings(dataset.uri, spec, staging_root=str(tmp_path / "staging"), result_contract=contract)
    workspace = contracts.resolve_workspace(dataset, bindings, selected=(0, 1))
    contract["model"]["revision"] = "v2"
    bindings.result_contract["model"]["revision"] = "v3"
    assert bindings.result_contract == {"model": {"revision": "v1"}}
    assert contracts.resolve_workspace(dataset, bindings, selected=(0, 1)) == workspace

    def unexpected_credentials(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Recovery identity must not resolve runtime credentials")

    monkeypatch.setattr(contracts, "get_lance_storage_options", unexpected_credentials)
    tuned = attrs.evolve(
        bindings, execution=EnrichmentExecutionConfig(storage_profile="another-profile", commit_attempts=9)
    )
    assert contracts.resolve_workspace(dataset, tuned, selected=(0, 1)) == workspace


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
    validate = Mock(wraps=contracts.fragment_complete)
    monkeypatch.setattr(contracts, "fragment_complete", validate)

    assert contracts.pending_fragment_ids(current, make_bindings(current.uri, spec)) == (2,)
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
    validate = Mock(wraps=contracts.fragment_complete)
    monkeypatch.setattr(contracts, "fragment_complete", validate)

    assert contracts.pending_fragment_ids(current, make_bindings(current.uri, spec)) == ()
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
    validate = Mock(wraps=contracts.fragment_complete)
    monkeypatch.setattr(contracts, "fragment_complete", validate)
    bindings = make_bindings(current.uri, spec)

    if field == "result":
        with pytest.raises(ValueError, match="Invalid test result"):
            contracts.pending_fragment_ids(current, bindings)
    else:
        assert contracts.pending_fragment_ids(current, bindings) == ()
    assert validate.call_args_list[0].args[0].fragment_id == 0


@pytest.mark.parametrize("selected", [(0, 0), (42,)])
def test_invalid_selections_are_rejected(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, selected: tuple[int, ...]
) -> None:
    """A frozen job selection cannot alias or refer to absent physical fragments."""
    with pytest.raises(ValueError, match=r"must be unique|is absent"):
        contracts.resolve_workspace(
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
        contracts.fragment_complete(mixed.get_fragments()[-1], spec)
