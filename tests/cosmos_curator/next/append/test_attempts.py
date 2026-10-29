# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Saved selections resume before new items, and only proven publications release their workspaces."""

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pytest
from pydantic import ValidationError

from cosmos_curator.next.append import attempts, contracts, inputs
from cosmos_curator.next.append.config import AppendExecutionConfig
from cosmos_curator.next.core import recovery_workspace
from cosmos_curator.next.core.recovery_workspace import RecoveryStateError, RecoveryWorkspace
from tests.cosmos_curator.next.append.helpers import make_items, make_results, publish_attempt, select, stage_results


def test_new_selection_holds_every_pending_item(bindings: contracts.AppendBindings, items: pa.Table) -> None:
    """A durable manifest records the selected items in ID order before processing starts."""
    attempt = select(bindings, items)
    assert attempt.selected == ("a", "b", "c")
    assert attempt.workspace is not None
    manifest = json.loads(Path(attempt.workspace.root, "workspace.json").read_text(encoding="utf-8"))
    assert manifest["selected_items"] == [{"item_id": item, "uri": f"{item}.bin"} for item in ("a", "b", "c")]
    assert attempt.counts() == {"input_items": 3, "selected_items": 3, "remaining_items": 0}
    assert RecoveryWorkspace.load(attempt.workspace.root, storage_profile="default") == attempt.workspace


def test_limit_bounds_only_a_new_selection(bindings: contracts.AppendBindings, items: pa.Table) -> None:
    """``execution.max_items`` is positive, and a smaller or larger limit never changes a saved selection."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        AppendExecutionConfig(max_items=0)
    limited = select(bindings, items, max_items=2)
    assert limited.selected == ("a", "b")
    assert limited.counts() == {"input_items": 3, "selected_items": 2, "remaining_items": 1}
    assert select(bindings, items, max_items=1).selected == ("a", "b")
    assert select(bindings, items).workspace == limited.workspace


def test_unfinished_selection_resumes_before_new_items(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table
) -> None:
    """Newly discovered items wait while the saved selection finishes, including items no longer discovered."""
    saved = select(bindings, items)
    discovered = make_items(["b", "d"])
    resumed = select(bindings, discovered)
    assert (resumed.selected, resumed.workspace) == (saved.selected, saved.workspace)
    assert resumed.counts() == {"input_items": None, "selected_items": 3, "remaining_items": None}
    publish_attempt(resumed, results)
    later = select(bindings, discovered)
    assert later.selected == ("d",)
    assert later.counts() == {"input_items": 2, "selected_items": 1, "remaining_items": 0}
    assert saved.workspace is not None
    assert later.workspace is not None
    assert later.workspace.root == saved.workspace.root
    assert not later.processing_complete()


def test_canonical_result_schema_isolates_saved_work(bindings: contracts.AppendBindings) -> None:
    """A saved selection never resumes under a different canonical list width."""
    spec = bindings.spec
    values_index = spec.result_fields.get_field_index("values")
    values_field = spec.result_fields.field("values")
    spec = attrs.evolve(
        spec,
        result_fields=spec.result_fields.set(values_index, values_field.with_type(pa.list_(pa.int64(), 2))),
    )
    bindings = attrs.evolve(bindings, spec=spec)
    items = make_items(["a"])
    saved = select(bindings, items)
    stage_results(saved, make_results(spec, {"a": [1, 2]}))
    changed_spec = attrs.evolve(
        spec,
        result_fields=spec.result_fields.set(values_index, values_field.with_type(pa.list_(pa.int64(), 3))),
    )
    changed = select(attrs.evolve(bindings, spec=changed_spec), items)
    assert changed.selected == saved.selected
    assert changed.workspace != saved.workspace
    assert not changed.processing_complete()
    assert saved.processing_complete()


def test_completed_workspace_is_verified_and_cleaned(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A workspace left after commit is released by complete canonical outcome coverage."""
    outcome = publish_attempt(select(bindings, items), results)
    workspace = outcome.attempt.workspace
    assert workspace is not None
    workspace.ensure(storage_profile="default")
    monkeypatch.setattr(RecoveryWorkspace, "cleanup", lambda _self, **_kwargs: False)
    discover = Mock(side_effect=AssertionError("Discovery must wait for cleanup"))
    with pytest.raises(OSError, match="until workspace cleanup succeeds"):
        attempts.reconcile_and_select_attempt(bindings, discover)
    discover.assert_not_called()
    assert RecoveryWorkspace.load(workspace.root, storage_profile="default") == workspace
    monkeypatch.undo()
    rerun = select(bindings, items)
    assert (rerun.selected, rerun.workspace) == ((), None)
    assert not Path(workspace.root).exists()


def test_partially_published_selection_is_rejected(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, tmp_path: Path
) -> None:
    """A partial outcome set contradicts atomic publication and rejects overlapping writers."""
    stale = select(bindings, items)
    assert stale.workspace is not None
    elsewhere = attrs.evolve(bindings, staging_root=str(tmp_path / "elsewhere"))
    publish_attempt(select(elsewhere, make_items(["a"])), results)
    with pytest.raises(RecoveryStateError, match="partially published"):
        select(bindings, items)
    assert Path(stale.workspace.root, "workspace.json").exists()


def test_another_selection_cannot_replace_the_active_manifest(
    bindings: contracts.AppendBindings, items: pa.Table
) -> None:
    """Every selection claims the same workspace, so a different manifest cannot create a second one."""
    first = select(bindings, items, max_items=1)
    other = inputs.resolve_workspace(
        inputs.open_target(bindings), bindings, selection=inputs.validate_items(items, bindings.spec).slice(1)
    )
    assert first.workspace is not None
    assert other.root == first.workspace.root
    with pytest.raises(RecoveryStateError, match="workspace does not match"):
        other.ensure(storage_profile="default")
    assert select(bindings, items).workspace == first.workspace


@pytest.mark.parametrize("manifest_removed", [False, True])
def test_partial_cleanup_cannot_leak_results_into_the_next_selection(
    bindings: contracts.AppendBindings,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    manifest_removed: bool,
) -> None:
    """A fixed workspace is reused only after all prior results and checkpoints have been removed."""
    items = make_items(["a", "b"])
    first = select(bindings, items, max_items=1)
    workspace = first.workspace
    assert workspace is not None
    Path(workspace.checkpoints).mkdir()
    Path(workspace.checkpoints, "old-checkpoint").write_text("a", encoding="utf-8")

    def interrupt_cleanup(*_args: object, **_kwargs: object) -> None:
        if manifest_removed:
            Path(workspace.root, "workspace.json").unlink(missing_ok=True)
        msg = "cleanup interrupted"
        raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(recovery_workspace, "remove_prefix", interrupt_cleanup)
        sink = recovery_workspace.logger.add(sys.stderr, level="WARNING")
        try:
            published = publish_attempt(first, make_results(bindings.spec, {"a": [1]}))
            assert not published.cleanup_complete
            with pytest.raises(OSError, match="cleanup"):
                select(bindings, items, max_items=1)
        finally:
            recovery_workspace.logger.remove(sink)
        assert "workspace cleanup failed" in capfd.readouterr().err
    assert Path(workspace.checkpoints, "old-checkpoint").exists()
    assert list(Path(workspace.results).glob("*.arrow"))
    assert lance.dataset(bindings.uri).count_rows() == 1

    second = select(bindings, items, max_items=1)
    assert second.selected == ("b",)
    assert second.workspace is not None
    assert second.workspace.root == workspace.root
    assert not Path(workspace.results).exists()
    assert not Path(workspace.checkpoints).exists()
    assert not second.processing_complete()
    published = publish_attempt(second, make_results(bindings.spec, {"b": [2]}))
    assert published.dataset.to_table(columns=["item_id", "value"]).sort_by("item_id").to_pylist() == [
        {"item_id": "a", "value": 1},
        {"item_id": "b", "value": 2},
    ]


def _rewrite_manifest(attempt: attempts.AppendAttempt, **changes: Any) -> None:  # noqa: ANN401
    assert attempt.workspace is not None
    path = Path(attempt.workspace.root, "workspace.json")
    path.write_text(json.dumps({**attempt.workspace.manifest, **changes}), encoding="utf-8")


@pytest.mark.parametrize(
    ("selected_items", "message"),
    [
        ("all", "Invalid item selection"),
        ([], "Invalid item selection"),
        ([{"item_id": "a"}], "Invalid item selection"),
        ([{"item_id": 1, "uri": "a.bin"}], "Invalid item selection"),
        ([{"item_id": None, "uri": "a.bin"}], "Invalid item selection"),
        ([{"item_id": "a", "uri": None}], "Invalid item selection"),
        (
            [
                {"item_id": "c", "uri": "c.bin"},
                {"item_id": "b", "uri": "b.bin"},
                {"item_id": "a", "uri": "a.bin"},
            ],
            "does not match its recorded selection",
        ),
        ([{"item_id": "a", "uri": "a.bin"}, {"item_id": "a", "uri": "a.bin"}], "Invalid item selection"),
    ],
    ids=["not-a-list", "empty", "missing-field", "wrong-type", "null-id", "null-uri", "unsorted", "duplicate"],
)
def test_invalid_saved_selections_are_preserved(
    bindings: contracts.AppendBindings, items: pa.Table, selected_items: object, message: str
) -> None:
    """A manifest whose selection or path cannot be trusted fails without changing durable state."""
    attempt = select(bindings, items)
    _rewrite_manifest(attempt, selected_items=selected_items)
    with pytest.raises(RecoveryStateError, match=message):
        select(bindings, items)
    assert attempt.workspace is not None
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_pruning_releases_a_completed_workspace_without_history(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current outcomes prove completion even when cleanup and old history are gone."""
    outcome = publish_attempt(select(bindings, items), results)
    workspace = outcome.attempt.workspace
    assert workspace is not None
    workspace.ensure(storage_profile="default")
    outcome.dataset.add_columns(pa.schema([pa.field("extra", pa.string())]))
    outcome.dataset.cleanup_old_versions(older_than=timedelta(seconds=0))
    monkeypatch.setattr(lance.LanceDataset, "read_transaction", Mock(side_effect=AssertionError("History read")))
    rerun = select(bindings, items)
    assert rerun.selected == ()
    assert not Path(workspace.root).exists()
