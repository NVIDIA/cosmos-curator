# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection recovery and receipt-authorized cleanup with real Lance metadata."""

import json
import sys
from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.enrichment import attempts, commit, contracts
from cosmos_curator.next.enrichment.attempts import EnrichmentAttempt, reconcile_and_select_attempt
from cosmos_curator.next.enrichment.workspace import RecoveryStateError, RecoveryWorkspace
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
    return contracts.register_fields(make_bindings(uri, spec))


def _select(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    root: Path,
    *,
    limit: int | None = None,
) -> EnrichmentAttempt:
    bindings = make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1})
    enrichment_input = contracts.EnrichmentInput(dataset, bindings, contracts.input_fingerprint(dataset, bindings))
    return reconcile_and_select_attempt(enrichment_input, max_fragments=limit)


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


def test_published_workspace_is_verified_and_cleaned_before_selecting_new_work(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Publication receipts retain the cleanup path after a batch becomes complete."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    other = contracts.resolve_workspace(
        dataset, make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 2}), selected=(1,)
    )
    other.ensure(storage_profile="default")
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    next_attempt = _select(published, spec, root)
    assert next_attempt.selected == (1,)
    assert next_attempt.cleanup_complete
    assert not Path(attempt.workspace.root).exists()
    assert Path(other.root).exists()


@pytest.mark.parametrize("resume", [False, True])
def test_missing_receipt_preserves_completed_workspace_and_continues(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    *,
    resume: bool,
) -> None:
    """Missing proof prevents cleanup without treating terminal work as unfinished."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    attempt.workspace.record_completion(storage_profile="default")
    published = _publish(dataset, spec, attempt.workspace, attempt.selected, matching_receipt=False)
    next_workspace = contracts.resolve_workspace(
        published,
        make_bindings(published.uri, spec, staging_root=str(root), result_contract={"version": 1}),
        selected=(1,),
    )
    if resume:
        next_workspace.ensure(storage_profile="default")
    sink = attempts.logger.add(sys.stderr, level="WARNING")
    try:
        next_attempt = _select(published, spec, root)
    finally:
        attempts.logger.remove(sink)
    assert next_attempt.selected == (1,)
    assert next_attempt.workspace == next_workspace
    assert not next_attempt.cleanup_complete
    assert Path(attempt.workspace.root, "workspace.json").exists()
    assert attempt.workspace.processing_complete(storage_profile="default")
    warning = capfd.readouterr().err
    assert "no matching publication receipt" in warning
    assert attempt.workspace.root in warning


def test_pruned_receipt_preserves_completed_workspace_without_blocking(
    dataset: lance.LanceDataset,
    spec: contracts.EnrichmentSpec,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """A real gap in retained Lance history is absent evidence, not a read failure."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    publication_version = published.version
    published.add_columns(pa.schema([("unrelated", pa.string())]))
    published.cleanup_old_versions(versions=[publication_version])
    assert publication_version not in {ref["version"] for ref in published.version_refs()}
    sink = attempts.logger.add(sys.stderr, level="WARNING")
    try:
        next_attempt = _select(published, spec, root)
    finally:
        attempts.logger.remove(sink)
    assert next_attempt.selected == ()
    assert next_attempt.workspace is None
    assert not next_attempt.cleanup_complete
    assert Path(attempt.workspace.root, "workspace.json").exists()
    warning = capfd.readouterr().err
    assert "no matching publication receipt" in warning
    assert attempt.workspace.root in warning


@pytest.mark.parametrize("fault", ["replaced_results", "changed_inputs"])
def test_unproven_publication_never_authorizes_cleanup(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, fault: str
) -> None:
    """A manifest or valid terminal values alone cannot prove this selection was published."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    fragment = published.get_fragment(0)
    field = "result" if fault == "replaced_results" else "item_id"
    values = {"item_id": ["a"], "result": [2]} if field == "result" else {"result": [1], "item_id": ["changed"]}
    join = "item_id" if field == "result" else "result"
    metadata, modified = fragment.update_columns(pa.table(values), left_on=join, right_on=join)
    published = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
        read_version=published.version,
    )
    with pytest.raises(ValueError, match=r"result files differ|current input file layout"):
        _select(published, spec, root)
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_receipted_cleanup_still_requires_compatible_schema(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """A receipt does not authorize cleanup after schema changes that leave files intact."""
    root = tmp_path / "staging"
    bindings = make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1})
    attempt = _select(dataset, spec, root)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    published.alter_columns({"path": "item_id", "nullable": True})

    with pytest.raises(ValueError, match="Missing or incompatible Lance field: item_id"):
        reconcile_and_select_attempt(contracts.EnrichmentInput(published, bindings, "unused"))
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_receipt_read_failure_prevents_all_cleanup(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable evidence remains fatal even after another workspace was verified."""
    root = tmp_path / "staging"
    workspaces = [
        contracts.resolve_workspace(
            dataset,
            make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1}),
            selected=(fragment_id,),
        )
        for fragment_id in (0, 1)
    ]
    for fragment_id, workspace in enumerate(workspaces):
        workspace.ensure(storage_profile="default")
        dataset = _publish(dataset, spec, workspace, (fragment_id,))
    find_transaction = attempts.find_transaction

    def unavailable_receipt(
        dataset: lance.LanceDataset, *, property_key: str, property_value: str | None = None, after_version: int = 0
    ) -> lance.Transaction | None:
        if property_value == workspaces[1].manifest_digest:
            msg = "Receipt storage unavailable"
            raise OSError(msg)
        return find_transaction(
            dataset, property_key=property_key, property_value=property_value, after_version=after_version
        )

    monkeypatch.setattr(RecoveryWorkspace, "discover", lambda *_args, **_kwargs: workspaces)
    monkeypatch.setattr(attempts, "find_transaction", unavailable_receipt)
    with pytest.raises(OSError, match="Receipt storage unavailable"):
        _select(dataset, spec, root)
    assert all(Path(workspace.root, "workspace.json").exists() for workspace in workspaces)


def test_incompatible_manifest_is_preserved(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """Discovery validates contracts before altering any workspace under its prefix."""
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


def test_multiple_unfinished_selections_require_explicit_resolution(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> None:
    """The recovery selector does not silently merge independently owned checkpoints."""
    root = tmp_path / "staging"
    workspaces = [
        contracts.resolve_workspace(
            dataset,
            make_bindings(dataset.uri, spec, staging_root=str(root), result_contract={"version": 1}),
            selected=(fragment_id,),
        )
        for fragment_id in (0, 1)
    ]
    for workspace in workspaces:
        workspace.ensure(storage_profile="default")
    with pytest.raises(RecoveryStateError, match="Multiple unfinished") as error:
        _select(dataset, spec, root)
    assert all(workspace.root in str(error.value) for workspace in workspaces)
    assert all(Path(workspace.root).exists() for workspace in workspaces)


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


def test_cleanup_failure_does_not_discard_new_pending_selection(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort cleanup failures remain visible while new pending work is selected."""
    root = tmp_path / "staging"
    attempt = _select(dataset, spec, root, limit=1)
    assert attempt.workspace is not None
    published = _publish(dataset, spec, attempt.workspace, attempt.selected)
    monkeypatch.setattr(RecoveryWorkspace, "cleanup", lambda _self, **_kwargs: False)
    next_attempt = _select(published, spec, root)
    assert next_attempt.selected == (1,)
    assert not next_attempt.cleanup_complete
    assert Path(attempt.workspace.root).exists()


def test_empty_input_selects_nothing_and_creates_no_workspace(spec: contracts.EnrichmentSpec, tmp_path: Path) -> None:
    """An empty table has nothing to process or publish, so it leaves no recovery state."""
    uri = str(tmp_path / "empty.lance")
    lance.write_dataset(pa.Table.from_pylist([], schema=spec.input_fields), uri, data_storage_version="2.2")
    dataset = contracts.register_fields(make_bindings(uri, spec))
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
