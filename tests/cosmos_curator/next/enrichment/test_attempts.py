# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved selections resume first, and current values release published workspaces, with real Lance metadata."""

import json
from pathlib import Path

import attrs
import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.core.recovery_workspace import RecoveryStateError, RecoveryWorkspace
from cosmos_curator.next.enrichment import commit, contracts, inputs
from cosmos_curator.next.enrichment.attempts import EnrichmentAttempt, reconcile_and_select_attempt
from tests.cosmos_curator.next.enrichment.helpers import make_bindings


def _validate(batch: pa.RecordBatch) -> None:
    if batch.column("result").null_count:
        msg = "Missing terminal result"
        raise ValueError(msg)


@pytest.fixture
def spec() -> contracts.EnrichmentSpec:
    """Keep outcomes small while exercising exact coverage and file ownership."""
    items = pa.schema([pa.field("item_id", pa.string(), nullable=False)])
    fields = contracts.owned_fields(pa.schema([("result", pa.int64())]), owner="item-recipe", field_set="result")
    return contracts.EnrichmentSpec(
        id_column="item_id",
        input_fields=items,
        output_fields=fields,
        result_contract={},
        validate_batch=_validate,
    )


@pytest.fixture
def dataset(tmp_path: Path, spec: contracts.EnrichmentSpec) -> lance.LanceDataset:
    """Create two independent pending fragments."""
    uri = str(tmp_path / "input.lance")
    lance.write_dataset(
        pa.Table.from_pylist([{"item_id": "a"}, {"item_id": "b"}], schema=spec.input_fields),
        uri,
        max_rows_per_file=1,
        data_storage_version="2.2",
    )
    return inputs.register_fields(make_bindings(uri, spec))


def _select(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    root: Path,
    *,
    limit: int | None = None,
) -> EnrichmentAttempt:
    bindings = make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1})
    bindings = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"max_fragments": limit}))
    enrichment_input = inputs.EnrichmentInput(dataset, bindings, inputs.input_fingerprint(dataset, bindings))
    return reconcile_and_select_attempt(enrichment_input)


def _publish(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    workspace: RecoveryWorkspace,
    selected: tuple[int, ...],
    *,
    matching_receipt: bool = True,
) -> lance.LanceDataset:
    candidates: list[commit.PreparedFragment] = []
    for fragment_id in selected:
        fragment = dataset.get_fragment(fragment_id)
        rows = [
            row | {"fragment_id": fragment_id, "row_offset": offset, "result": 1}
            for offset, row in enumerate(fragment.to_table(columns=["item_id"]).to_pylist())
        ]
        candidates.extend(
            commit.PreparedFragment.from_row(row)
            for row in commit.prepare_fragment(
                pa.Table.from_pylist(rows, schema=spec.staged_schema),
                version=int(dataset.version),
                bindings=make_bindings(dataset.uri, spec),
            ).to_pylist()
        )
    return commit.commit_fragments(
        lambda *_: candidates,
        dataset=dataset,
        selected=selected,
        receipt=workspace.manifest_digest if matching_receipt else "another-publication",
        transaction_properties={"kind": "test"},
        bindings=make_bindings(dataset.uri, spec, result_contract=workspace.manifest["result_contract"]),
    )


@pytest.mark.parametrize("limit", [None, 1, 10])
def test_retry_keeps_selection_after_appends_and_limit_changes(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, limit: int | None
) -> None:
    """Sorting new pending IDs cannot replace the membership frozen before a crash."""
    attempt = _select(dataset, spec, tmp_path / "staging", limit=limit)
    assert attempt.workspace is not None
    assert Path(attempt.workspace.root, "workspace.json").exists()
    attempt.workspace.record_completion(storage_profile="default")
    current = lance.write_dataset(
        pa.Table.from_pylist([{"item_id": "c"}], schema=spec.input_fields),
        dataset.uri,
        mode="append",
        data_storage_version="2.2",
    )
    current.add_columns(pa.schema([("unrelated", pa.string())]))
    resumed = _select(current, spec, tmp_path / "staging", limit=1 if limit is None else None)
    assert (resumed.selected, resumed.workspace) == (attempt.selected, attempt.workspace)
    assert resumed.pending == (*attempt.pending, 2)
    assert resumed.processing_complete()


@pytest.mark.parametrize("matching_receipt", [True, False])
def test_published_workspace_is_cleaned_before_selecting_new_work(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, *, matching_receipt: bool
) -> None:
    """Current values prove publication, with or without a matching receipt; other contracts keep their workspace."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    attempt.workspace.record_completion(storage_profile="default")
    other = inputs.resolve_workspace(
        dataset, make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 2}), selected=(1,)
    )
    other.ensure(storage_profile="default")
    published = _publish(dataset, spec, attempt.workspace, attempt.selected, matching_receipt=matching_receipt)
    next_attempt = _select(published, spec, root)
    assert next_attempt.selected == (1,)
    assert next_attempt.workspace is not None
    assert next_attempt.workspace.root == attempt.workspace.root
    assert RecoveryWorkspace.load(attempt.workspace.root, storage_profile="default") == next_attempt.workspace
    assert not next_attempt.processing_complete()
    assert Path(other.root, "workspace.json").exists()


def test_pruned_receipt_still_releases_the_completed_workspace(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """A gap in retained Lance history leaves current values to prove publication."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    publication_version = published.version
    published.add_columns(pa.schema([("unrelated", pa.string())]))
    published.cleanup_old_versions(versions=[publication_version])
    assert publication_version not in {ref["version"] for ref in published.version_refs()}
    next_attempt = _select(published, spec, root)
    assert (next_attempt.selected, next_attempt.workspace) == ((), None)
    assert not Path(attempt.workspace.root).exists()


def test_changed_selected_inputs_preserve_the_workspace(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Completion never releases a workspace whose selected input files changed."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    metadata, modified = published.get_fragment(0).update_columns(
        pa.table({"result": [1], "item_id": ["changed"]}), left_on="result", right_on="result"
    )
    published = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
        read_version=published.version,
    )
    with pytest.raises(RecoveryStateError, match="current input file layout"):
        _select(published, spec, root)
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_cleanup_still_requires_compatible_schema(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Publication does not authorize cleanup after schema changes that leave files intact."""
    root = tmp_path / "staging"
    bindings = make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1})
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    published.alter_columns({"path": "item_id", "nullable": True})

    with pytest.raises(ValueError, match="Missing or incompatible Lance field: item_id"):
        reconcile_and_select_attempt(inputs.EnrichmentInput(published, bindings, "unused"))
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_receipt_read_failure_prevents_cleanup(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable publication evidence remains fatal rather than a reason to clean up."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)

    def unavailable_history(*_args: object, **_kwargs: object) -> None:
        msg = "Receipt storage unavailable"
        raise OSError(msg)

    monkeypatch.setattr(inputs, "iter_transactions", unavailable_history)
    with pytest.raises(OSError, match="Receipt storage unavailable"):
        _select(published, spec, root)
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_incompatible_manifest_is_preserved(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Recovery validates the saved manifest before altering its workspace."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    manifest = attempt.workspace.manifest
    manifest["result_contract"] = {"version": "wrong"}
    path = Path(attempt.workspace.root, "workspace.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RecoveryStateError, match=f"workspace does not match.*{attempt.workspace.root}"):
        _select(dataset, spec, root)
    assert json.loads(path.read_text())["result_contract"] == {"version": "wrong"}


@pytest.mark.parametrize(("fragments", "message"), [([], "Invalid fragment"), ([1, 0], "not sorted and unique")])
def test_invalid_saved_selections_are_preserved(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, fragments: list[int], message: str
) -> None:
    """A saved selection that is empty or out of order fails instead of resuming or being replaced."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    manifest = attempt.workspace.manifest
    by_id = {fragment["fragment_id"]: fragment for fragment in manifest["input_fragments"]}
    manifest["input_fragments"] = [by_id[fragment_id] for fragment_id in fragments]
    path = Path(attempt.workspace.root, "workspace.json")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RecoveryStateError, match=message):
        _select(dataset, spec, root)
    assert json.loads(path.read_text())["input_fragments"] == manifest["input_fragments"]


def test_another_selection_cannot_replace_the_active_manifest(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Every selection claims the same workspace, so a different manifest cannot create a second one."""
    root = tmp_path / "staging"
    first = _select(dataset, spec, root, limit=1)
    assert first.workspace is not None
    other = inputs.resolve_workspace(
        dataset, make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1}), selected=(1,)
    )
    assert other.root == first.workspace.root
    with pytest.raises(RecoveryStateError, match="workspace does not match"):
        other.ensure(storage_profile="default")
    assert _select(dataset, spec, root).workspace == first.workspace


def test_partially_published_selection_is_preserved(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Independent publication of part of an atomic selection cannot claim the rest."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, (0,))
    with pytest.raises(RecoveryStateError, match="partially published"):
        _select(published, spec, root)
    assert Path(attempt.workspace.root).exists()


def test_cleanup_failure_blocks_new_selections(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published selection's workspace must be removed before the next selection can reuse it."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    monkeypatch.setattr(RecoveryWorkspace, "cleanup", lambda _self, **_kwargs: False)
    with pytest.raises(OSError, match="until workspace cleanup succeeds"):
        _select(published, spec, root)
    assert RecoveryWorkspace.load(attempt.workspace.root, storage_profile="default") == attempt.workspace
    monkeypatch.undo()
    assert _select(published, spec, root).selected == (1,)


@pytest.mark.parametrize("manifest_removed", [False, True])
def test_interrupted_cleanup_cannot_leak_state_into_the_next_selection(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, *, manifest_removed: bool
) -> None:
    """A fixed workspace is reused only after earlier checkpoints and markers are removed."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    workspace = attempt.workspace
    assert workspace is not None
    workspace.record_completion(storage_profile="default")
    Path(workspace.checkpoints).mkdir()
    Path(workspace.checkpoints, "old-checkpoint").write_text("a", encoding="utf-8")
    published = _publish(dataset, spec, workspace, attempt.selected)
    if manifest_removed:
        Path(workspace.root, "workspace.json").unlink()
    next_attempt = _select(published, spec, root, limit=1)
    assert next_attempt.selected == (1,)
    assert next_attempt.workspace is not None
    assert next_attempt.workspace.root == workspace.root
    assert not Path(workspace.checkpoints).exists()
    assert not next_attempt.processing_complete()


def test_empty_input_selects_nothing_and_creates_no_workspace(spec: contracts.EnrichmentSpec, tmp_path: Path) -> None:
    """An empty table has nothing to process or publish, so it leaves no recovery state."""
    uri = str(tmp_path / "empty.lance")
    lance.write_dataset(pa.Table.from_pylist([], schema=spec.input_fields), uri, data_storage_version="2.2")
    dataset = inputs.register_fields(make_bindings(uri, spec))
    attempt = _select(dataset, spec, tmp_path / "staging")
    assert (attempt.pending, attempt.selected, attempt.workspace) == ((), (), None)
    assert not (tmp_path / "staging").exists()


def test_counts_describe_the_pinned_selection(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Shared run counts separate the frozen selection from pending work left for later."""
    attempt = _select(dataset, spec, tmp_path / "staging", limit=1)
    assert attempt.counts() == {
        "input_rows": 2,
        "selected_rows": 1,
        "remaining_rows": 1,
        "selected_fragments": 1,
        "remaining_fragments": 1,
    }
