# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed processing and append publication stay bound to one selection."""

from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pytest
from ray.data.context import DataContext

from cosmos_curator.next.append import commit, contracts, execution, inputs
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY
from cosmos_curator.next.core.ray_checkpoints import STAGED_FILE_EXTENSION, StagedResultsDatasink, write_staged_table
from cosmos_curator.next.core.recovery_workspace import RecoveryWorkspace
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.append.helpers import (
    make_items,
    make_results,
    publish_attempt,
    select,
    stage_locally,
    stage_results,
)


@pytest.fixture(autouse=True)
def _no_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test without an installed Ray Data checkpoint."""
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)


@pytest.mark.parametrize("parallelism", [1, 2, 3, 5])
def test_selection_datasource_reads_every_item_once(items: pa.Table, parallelism: int) -> None:
    """Balanced read tasks cover the selection exactly, never more tasks than items."""
    datasource = execution.SelectionDatasource(items)
    tasks = datasource.get_read_tasks(parallelism)
    assert len(tasks) == min(parallelism, 3)
    assert max(task.metadata.num_rows for task in tasks) - min(task.metadata.num_rows for task in tasks) <= 1
    read = pa.concat_tables([table for task in tasks for table in task()])
    assert read.equals(items)
    assert datasource.estimate_inmemory_data_size() == items.nbytes
    with pytest.raises(ValueError, match="must be positive"):
        datasource.get_read_tasks(0)


@pytest.mark.parametrize("fail_write", [False, True])
def test_processing_scope_owns_checkpoint_and_sink(
    bindings: contracts.AppendBindings, items: pa.Table, monkeypatch: pytest.MonkeyPatch, *, fail_write: bool
) -> None:
    """The item ID keys the checkpoint, the sink shares its filesystem, and only success records completion."""
    tuned = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"rows_per_checkpoint": 5}))
    attempt = select(tuned, items, max_items=2)
    assert attempt.workspace is not None
    plan = Mock()
    plan.map_batches.return_value = plan
    plan.repartition.return_value = plan
    if fail_write:
        plan.write_datasink.side_effect = RuntimeError("injected write failure")
    reads = []

    def read(datasource: execution.SelectionDatasource, **kwargs: object) -> Mock:
        reads.append((datasource, kwargs, DataContext.get_current().checkpoint_config))
        return plan

    monkeypatch.setattr(execution.ray.data, "read_datasource", read)
    with ExitStack() as stack:
        if fail_write:
            stack.enter_context(pytest.raises(RuntimeError, match="injected write failure"))
        with execution.checkpointed_processing(attempt) as processing:
            assert processing.result_schema is bindings.spec.result_fields
            processing.write(processing.read())

    ((datasource, kwargs, checkpoint),) = reads
    assert kwargs == {"override_num_blocks": 2, "num_cpus": None}
    rows = [row for task in datasource.get_read_tasks(4) for table in task() for row in table.to_pylist()]
    assert [row["item_id"] for row in rows] == ["a", "b"]
    assert checkpoint.id_column == "item_id"
    assert checkpoint.checkpoint_path == attempt.workspace.checkpoints
    plan.repartition.assert_called_once_with(target_num_rows_per_block=5, strict=True)
    ((datasink,), _) = plan.write_datasink.call_args
    assert isinstance(datasink, StagedResultsDatasink)
    assert checkpoint.filesystem is datasink.filesystem.unwrap()
    assert datasink.unresolved_path == attempt.workspace.results
    assert datasink.min_rows_per_write == 5
    assert DataContext.get_current().checkpoint_config is None
    assert attempt.processing_complete() is not fail_write


def test_processing_reads_one_block_per_selected_item(
    bindings: contracts.AppendBindings, items: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The block count follows the selection, never the CPUs present when the plan starts.

    A read may reserve its first stage's CPUs so Ray fuses the two.
    """
    attempt = select(bindings, items, max_items=2)
    reads: list[dict[str, object]] = []

    def read(_datasource: execution.SelectionDatasource, **kwargs: object) -> Mock:
        reads.append(kwargs)
        return Mock()

    monkeypatch.setattr(execution.ray.data, "read_datasource", read)
    with execution.checkpointed_processing(attempt) as processing:
        processing.read()
        processing.read(num_cpus=0.25)
    assert reads == [
        {"override_num_blocks": 2, "num_cpus": None},
        {"override_num_blocks": 2, "num_cpus": 0.25},
    ]


def test_processing_validates_results_before_ray_checkpoints_them(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table
) -> None:
    """A result publication would reject fails before the sink can checkpoint its item."""
    attempt = select(bindings, items)
    plan = Mock()
    plan.map_batches.return_value = plan
    plan.repartition.return_value = plan
    with execution.checkpointed_processing(attempt) as processing:
        processing.write(plan)
    ((validate,), options) = plan.map_batches.call_args
    # Default remote args after the strict repartition let Ray fuse validation into each group's write task.
    assert options == {"batch_format": "pyarrow", "batch_size": None, "zero_copy_batch": True}
    assert [call[0] for call in plan.method_calls] == ["repartition", "map_batches", "write_datasink"]
    assert validate(results).equals(results, check_metadata=True)
    assert validate(results.slice(0, 0)).num_rows == 0
    error_with_values = [{"item_id": "a", "values": [1], "error": "unreadable a"}]
    with pytest.raises(ValueError, match="Invalid test result"):
        validate(pa.Table.from_pylist(error_with_values, schema=bindings.spec.result_fields))
    with pytest.raises(ValueError, match="canonical result schema"):
        validate(results.drop_columns(["values"]))


def test_processing_requires_selected_items(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table
) -> None:
    """A completed table has no checkpoint scope to own."""
    publish_attempt(select(bindings, items), results)
    attempt = select(bindings, items)
    with ExitStack() as stack, pytest.raises(ValueError, match="selected items"):
        stack.enter_context(execution.checkpointed_processing(attempt))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra"])
def test_incomplete_results_never_publish(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, fault: str
) -> None:
    """Every selected item needs exactly one result before any fragment is staged or committed."""
    attempt = select(bindings, items)
    assert attempt.workspace is not None
    if fault == "missing":
        staged = results.filter(pc.not_equal(results["item_id"], "b"))
    elif fault == "duplicate":
        staged = pa.concat_tables([results, results.slice(0, 1)])
    else:
        staged = pa.concat_tables([results, make_results(bindings.spec, {"x": [1]})])
    Path(attempt.workspace.results).mkdir(parents=True)
    write_staged_table(staged, str(Path(attempt.workspace.results, f"staged.{STAGED_FILE_EXTENSION}")))
    with pytest.raises(ValueError, match="do not exactly cover the selected items"):
        publish_attempt(attempt)
    assert lance.dataset(bindings.uri).version == attempt.dataset.version
    assert Path(attempt.workspace.root, "workspace.json").exists()


@pytest.mark.parametrize("duplicated", [False, True])
def test_coverage_spans_every_staged_file(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, *, duplicated: bool
) -> None:
    """Results staged across files cover each item once; an item staged in two files never publishes."""
    attempt = select(bindings, items)
    for index in range(results.num_rows):
        stage_results(attempt, results.slice(index, 1))
    if not duplicated:
        outcome = publish_attempt(attempt)
        assert (outcome.appended_rows, outcome.errors) == (4, 1)
        return
    stage_results(attempt, results.slice(0, 1))
    with pytest.raises(ValueError, match="do not exactly cover the selected items"):
        publish_attempt(attempt)
    assert lance.dataset(bindings.uri).version == attempt.dataset.version


@pytest.mark.parametrize("cleaned", [True, False])
def test_publication_records_provenance_and_removes_the_workspace(
    bindings: contracts.AppendBindings,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    cleaned: bool,
) -> None:
    """Rows pack across items; cleanup follows verification and its failure is reported, not undone."""
    attempt = select(bindings, make_items(["a", "b", "c", "d"]))
    assert attempt.workspace is not None
    if not cleaned:
        monkeypatch.setattr(RecoveryWorkspace, "cleanup", lambda _self, **_kwargs: False)
    staged = pa.concat_tables([results, make_results(bindings.spec, {"d": [4]})])
    outcome = publish_attempt(attempt, staged)
    assert [fragment.count_rows() for fragment in outcome.dataset.get_fragments()] == [2, 2, 1]
    properties = outcome.dataset.read_transaction(outcome.dataset.version).transaction_properties
    assert properties == {
        KIND_KEY: "item-recipe",
        CONTRACT_DIGEST_KEY: canonical_digest({"version": 1}),
    }
    assert Path(attempt.workspace.root).exists() is not cleaned
    assert outcome.summary() == {
        "input_items": 4,
        "selected_items": 4,
        "remaining_items": 0,
        "appended_rows": 5,
        "errors": 1,
        "cleanup_complete": cleaned,
        "contract_digest": canonical_digest({"version": 1}),
    }
    assert execution.summary_message("item-recipe", outcome.summary()) == (
        "item-recipe: published 4 item(s) as 5 row(s) with 1 terminal error(s); 0 item(s) pending"
    )


def test_nothing_selected_publishes_nothing(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A table with every discovered item published reports its pinned version without committing."""
    publish_attempt(select(bindings, items), results)
    attempt = select(bindings, items)
    assert (attempt.selected, attempt.workspace) == ((), None)
    monkeypatch.setattr(execution, "stage_append_fragments", Mock(side_effect=AssertionError("Staged")))
    monkeypatch.setattr(commit, "commit_append", Mock(side_effect=AssertionError("Committed")))
    outcome = execution.publish_append(attempt)
    assert outcome.dataset.version == attempt.dataset.version
    assert (outcome.appended_rows, outcome.summary()["selected_items"]) == (0, 0)


@pytest.mark.parametrize("marker", [False, True])
def test_run_append_starts_ray_whenever_items_are_selected(
    bindings: contracts.AppendBindings,
    items: pa.Table,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    marker: bool,
) -> None:
    """A processing marker skips the plan but not Ray, which publication needs; a finished run needs neither."""
    events: list[object] = []
    if marker:
        stage_results(select(bindings, items), results)

    def process(processing: execution.ProcessingScope) -> None:
        checkpoint = DataContext.get_current().checkpoint_config
        events.append(("process", checkpoint and checkpoint.id_column))
        stage_results(processing._attempt, results)

    def stage(*args: object, **kwargs: object) -> list[object]:
        events.append("stage")
        return stage_locally(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(execution, "stage_append_fragments", stage)
    outcome = execution.run_append(
        bindings,
        lambda: items,
        start_ray=lambda _bindings: events.append("ray"),
        process=process,
    )
    processed = [] if marker else [("process", "item_id")]
    assert events == ["ray", *processed, "stage"]
    assert (outcome.summary()["appended_rows"], outcome.summary()["errors"]) == (4, 1)
    assert DataContext.get_current().checkpoint_config is None

    rerun = execution.run_append(
        bindings,
        lambda: items,
        start_ray=Mock(side_effect=AssertionError("Ray must not start")),
        process=Mock(side_effect=AssertionError("Processing must not run")),
    )
    assert (rerun.summary()["selected_items"], rerun.summary()["errors"]) == (0, 0)


def test_run_append_limits_only_a_new_selection(
    bindings: contracts.AppendBindings, items: pa.Table, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An item limit bounds new work and leaves the rest pending for a later run."""
    monkeypatch.setattr(execution, "stage_append_fragments", stage_locally)
    process = partial(lambda processing, staged: stage_results(processing._attempt, staged), staged=results)
    bindings = attrs.evolve(bindings, execution=bindings.execution.model_copy(update={"max_items": 2}))
    outcome = execution.run_append(bindings, lambda: items, start_ray=lambda _bindings: None, process=process)
    assert (outcome.summary()["selected_items"], outcome.summary()["remaining_items"]) == (2, 1)
    assert inputs.published_item_ids(outcome.dataset, bindings, items["item_id"].to_pylist()) == {"a", "b"}


@pytest.mark.parametrize("complete", [False, True])
def test_resume_never_discovers_sources(
    bindings: contracts.AppendBindings,
    items: pa.Table,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool,
) -> None:
    """Processing and publication recovery use saved items even when source listing is unavailable."""
    saved = select(bindings, items)
    if complete:
        stage_results(saved, results)
    discover = Mock(side_effect=OSError("Source listing is unavailable"))
    process = Mock(side_effect=lambda processing: stage_results(processing._attempt, results))
    monkeypatch.setattr(execution, "stage_append_fragments", stage_locally)
    outcome = execution.run_append(bindings, discover, start_ray=lambda _: None, process=process)
    discover.assert_not_called()
    assert process.call_count == (0 if complete else 1)
    assert outcome.attempt.selected == saved.selected
    assert outcome.attempt.counts() == {"input_items": None, "selected_items": 3, "remaining_items": None}
    assert outcome.dataset.count_rows() == 4
    assert "discovery deferred; rerun to check for pending items" in execution.summary_message(
        "items", outcome.summary()
    )

    discover.side_effect = None
    discover.return_value = make_items(["a", "b", "c", "d"])
    staged = make_results(bindings.spec, {"d": [4]})
    next_run = execution.run_append(
        bindings,
        discover,
        start_ray=lambda _: None,
        process=lambda processing: stage_results(processing._attempt, staged),
    )
    discover.assert_called_once()
    assert next_run.attempt.counts() == {"input_items": 4, "selected_items": 1, "remaining_items": 0}


def test_staging_never_starts_a_default_ray_runtime(
    bindings: contracts.AppendBindings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expanding and staging fragments requires the recipe's Ray runtime."""
    monkeypatch.setattr(execution.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(execution, "read_staged_results", Mock(side_effect=AssertionError("Read staged results")))
    with pytest.raises(RuntimeError, match="Start Ray with the recipe's runtime"):
        execution.stage_append_fragments(["results.arrow"], filesystem=pafs.LocalFileSystem(), bindings=bindings)
