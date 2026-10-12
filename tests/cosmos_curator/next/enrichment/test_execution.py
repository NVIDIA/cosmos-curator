# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpointed processing and publication stay bound to one validated attempt."""

from collections.abc import Callable
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pytest
from ray.data.context import DataContext

from cosmos_curator.next.core.lance_datasource import LanceFragmentDatasource
from cosmos_curator.next.enrichment import commit, contracts, execution
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.enrichment.helpers import (
    attempt_all,
    make_bindings,
    prepare_results,
    publish_attempt,
    publish_results,
)


def _readiness(dataset: lance.LanceDataset) -> None:
    """Stand in for a recipe's domain readiness check."""
    if dataset.to_table(columns=["uri"])["uri"].null_count:
        msg = "Items require URIs"
        raise ValueError(msg)


@pytest.fixture
def enrichment_input(
    dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, tmp_path: Path
) -> contracts.EnrichmentInput:
    """Open the shared item table through the public API, retaining its validation proof."""
    bindings = make_bindings(
        dataset.uri,
        attrs.evolve(spec, validate_input=_readiness),
        staging_root=str(tmp_path / "staging"),
        result_contract={"version": 1},
        commit_attempts=2,
    )
    return contracts.open_enrichment_input(bindings)


def _with_readiness(
    bindings: contracts.EnrichmentBindings, validate_input: Callable[[lance.LanceDataset], None]
) -> contracts.EnrichmentBindings:
    return attrs.evolve(bindings, spec=attrs.evolve(bindings.spec, validate_input=validate_input))


@pytest.fixture
def results(enrichment_input: contracts.EnrichmentInput) -> pa.Table:
    """Supply valid results for both pending fragments."""
    return pa.Table.from_pylist(
        [
            {"fragment_id": 0, "row_offset": 0, "item_id": "a", "result": 1, "metadata": {"status": "success"}},
            {"fragment_id": 1, "row_offset": 0, "item_id": "b", "result": 2, "metadata": {"status": "success"}},
        ],
        schema=enrichment_input.bindings.spec.staged_schema,
    )


def _append(enrichment_input: contracts.EnrichmentInput, item_id: str, uri: str | None) -> lance.LanceDataset:
    rows = pa.table({"item_id": [item_id], "uri": [uri]}, schema=enrichment_input.bindings.spec.input_fields)
    return lance.write_dataset(rows, enrichment_input.dataset.uri, mode="append", data_storage_version="2.2")


def test_only_successful_validation_can_be_reused(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table
) -> None:
    """Opening is insufficient evidence; only validated publication makes a later scan unnecessary."""
    validator = Mock(wraps=_readiness)
    bindings = _with_readiness(enrichment_input.bindings, validator)
    contracts.open_enrichment_input(bindings)
    validator.assert_called_once()
    published = publish_results(results, enrichment_input)
    assert published.read_transaction(published.version).transaction_properties[contracts.VALIDATED_INPUT_KEY] == (
        enrichment_input.fingerprint
    )
    validator.reset_mock()
    reopened = contracts.open_enrichment_input(bindings)
    validator.assert_not_called()
    assert reopened.fingerprint == enrichment_input.fingerprint

    _append(enrichment_input, "c", None)
    with pytest.raises(ValueError, match="Items require URIs"):
        contracts.open_enrichment_input(bindings)
    validator.assert_called_once()


def test_shared_validation_rejects_duplicate_ids_before_recipe_checks(
    enrichment_input: contracts.EnrichmentInput,
) -> None:
    """Checkpoints key on IDs, so uniqueness is checked for every recipe across the whole snapshot."""
    _append(enrichment_input, "a", "duplicate.mp4")
    validator = Mock(wraps=_readiness)
    with pytest.raises(ValueError, match="item_id values must be non-null and globally unique"):
        contracts.open_enrichment_input(_with_readiness(enrichment_input.bindings, validator))
    validator.assert_not_called()


def test_refresh_does_not_certify_concurrently_appended_input(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descriptor refresh can use a new version without changing the validated snapshot's proof."""
    original_prepare = commit.prepare_fragment
    raced = False
    versions = []

    def prepare(batch: pa.Table, *, bindings: contracts.EnrichmentBindings, version: int) -> pa.Table:
        nonlocal raced
        versions.append(version)
        candidate = original_prepare(batch, bindings=bindings, version=version)
        if not raced:
            raced = True
            _append(enrichment_input, "a", "a.mp4").add_columns({"unrelated": "'keep'"})
        return candidate

    monkeypatch.setattr(commit, "prepare_fragment", prepare)
    published = publish_results(results, enrichment_input)
    assert max(versions) > enrichment_input.dataset.version
    transaction = published.read_transaction(published.version)
    assert transaction.transaction_properties[contracts.VALIDATED_INPUT_KEY] == enrichment_input.fingerprint
    assert contracts.input_fingerprint(published, enrichment_input.bindings) != enrichment_input.fingerprint
    assert published.to_table(columns=["unrelated"])["unrelated"].to_pylist() == ["keep"] * 3
    with pytest.raises(ValueError, match="globally unique"):
        contracts.open_enrichment_input(enrichment_input.bindings)


@pytest.mark.parametrize("fault", ["selection", "contract", "snapshot", "missing_workspace"])
def test_mismatched_attempts_fail_before_preparing_files(
    enrichment_input: contracts.EnrichmentInput, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """The adapter cannot bind one selection's results to another input or recovery identity."""
    attempt = attempt_all(enrichment_input)
    if fault == "selection":
        attempt = attrs.evolve(attempt, selected=(0,))
    elif fault == "contract":
        attempt = attrs.evolve(
            attempt,
            workspace=contracts.resolve_workspace(
                enrichment_input.dataset,
                attrs.evolve(enrichment_input.bindings, result_contract={"version": 2}),
                selected=attempt.selected,
            ),
        )
    elif fault == "snapshot":
        attempt = attrs.evolve(
            attempt, input=attrs.evolve(enrichment_input, dataset=_append(enrichment_input, "c", "c.mp4"))
        )
    else:
        attempt = attrs.evolve(attempt, workspace=None)
    prepare = Mock(side_effect=AssertionError("Must reject before preparing files"))
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", prepare)
    with pytest.raises(ValueError, match=r"validated input|validated snapshot"):
        execution.publish_enrichment(attempt)
    prepare.assert_not_called()


@pytest.mark.parametrize(
    "key",
    [
        contracts.KIND_KEY,
        contracts.FIELD_KEY,
        contracts.CONTRACT_DIGEST_KEY,
        contracts.PUBLICATION_RECEIPT_KEY,
        contracts.VALIDATED_INPUT_KEY,
    ],
)
def test_recipe_properties_cannot_replace_shared_proof(enrichment_input: contracts.EnrichmentInput, key: str) -> None:
    """Recipes add their own records; the adapter supplies provenance, the recovery receipt, and evidence."""
    with pytest.raises(ValueError, match="cannot replace the shared provenance"):
        execution.publish_enrichment(attempt_all(enrichment_input), transaction_properties={key: "recipe-value"})


def test_publication_records_shared_provenance(enrichment_input: contracts.EnrichmentInput, results: pa.Table) -> None:
    """Every enrichment commit names its recipe, field set, and result contract without recipe properties."""
    published = publish_results(results, enrichment_input)
    properties = published.read_transaction(published.version).transaction_properties
    assert properties[contracts.KIND_KEY] == "item-recipe"
    assert properties[contracts.FIELD_KEY] == "result"
    assert properties[contracts.CONTRACT_DIGEST_KEY] == canonical_digest({"version": 1})


@pytest.mark.parametrize("cleaned", [True, False])
def test_publication_removes_the_verified_workspace(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch, *, cleaned: bool
) -> None:
    """Cleanup follows canonical verification; its failure is reported without undoing publication."""
    attempt = reconcile_and_select_attempt(enrichment_input)
    assert attempt.workspace is not None
    if not cleaned:
        monkeypatch.setattr(RecoveryWorkspace, "cleanup", lambda _self, **_kwargs: False)
    outcome = publish_attempt(results, attempt)
    assert outcome.dataset.version == enrichment_input.dataset.version + 1
    assert Path(attempt.workspace.root).exists() is not cleaned
    assert outcome.summary() == {
        "input_rows": 2,
        "selected_rows": 2,
        "remaining_rows": 0,
        "selected_fragments": 2,
        "remaining_fragments": 0,
        "cleanup_complete": cleaned,
        "field_set": "result",
        "contract_digest": canonical_digest({"version": 1}),
    }


def test_failed_publication_retains_the_workspace(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table
) -> None:
    """Incomplete staged results fail before cleanup can discard recovery state."""
    attempt = reconcile_and_select_attempt(enrichment_input)
    assert attempt.workspace is not None
    with pytest.raises(ValueError, match="exactly cover"):
        publish_attempt(results.slice(0, 1), attempt)
    assert Path(attempt.workspace.root, "workspace.json").exists()


def test_nothing_selected_publishes_nothing(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed input reports its pinned version without preparing or committing."""
    publish_results(results, enrichment_input)
    attempt = reconcile_and_select_attempt(contracts.open_enrichment_input(enrichment_input.bindings))
    assert (attempt.selected, attempt.workspace) == ((), None)
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", Mock(side_effect=AssertionError("Prepared")))
    monkeypatch.setattr(commit, "commit_fragments", Mock(side_effect=AssertionError("Committed")))
    outcome = execution.publish_enrichment(attempt)
    assert outcome.dataset.version == attempt.input.dataset.version
    assert outcome.summary()["selected_rows"] == 0


@pytest.mark.parametrize("fail_write", [False, True])
@pytest.mark.parametrize("include_uri", [False, True])
def test_processing_scope_owns_checkpoint_and_sink(
    enrichment_input: contracts.EnrichmentInput, monkeypatch: pytest.MonkeyPatch, *, fail_write: bool, include_uri: bool
) -> None:
    """The spec's ID keys the checkpoint, the sink shares its filesystem, and only success records completion."""
    bindings = enrichment_input.bindings
    tuned = bindings.execution.model_copy(update={"lance_read_batch_size": 7, "parquet_rows_per_file": 5})
    attempt = reconcile_and_select_attempt(contracts.open_enrichment_input(attrs.evolve(bindings, execution=tuned)))
    assert attempt.workspace is not None
    context = DataContext.get_current()
    monkeypatch.setattr(context, "checkpoint_config", None)
    plan = Mock()
    plan.map_batches.return_value = plan
    plan.repartition.return_value = plan
    if fail_write:
        plan.write_parquet.side_effect = RuntimeError("injected write failure")
    reads = []

    def read(datasource: LanceFragmentDatasource, **kwargs: object) -> Mock:
        reads.append((datasource, kwargs, context.checkpoint_config))
        return plan

    monkeypatch.setattr(execution.ray.data, "read_datasource", read)
    columns = ["uri"] if include_uri else []
    with ExitStack() as stack:
        if fail_write:
            stack.enter_context(pytest.raises(RuntimeError, match="injected write failure"))
        with execution.checkpointed_processing(attempt) as processing:
            assert processing.staged_schema is attempt.input.bindings.spec.staged_schema
            processing.write(processing.read(columns, override_num_blocks=3))

    ((datasource, kwargs, checkpoint),) = reads
    assert kwargs == {"override_num_blocks": 3}
    assert datasource._batch_size == 7
    assert datasource._schema.names == ["item_id", *columns, "fragment_id", "row_offset"]
    rows = [row for task in datasource.get_read_tasks(2) for batch in task() for row in batch.to_pylist()]
    assert [row["item_id"] for row in rows] == ["a", "b"]
    if include_uri:
        assert [row["uri"] for row in rows] == enrichment_input.dataset.to_table(columns=["uri"])["uri"].to_pylist()
    assert checkpoint.id_column == "item_id"
    assert checkpoint.checkpoint_path == attempt.workspace.checkpoints
    assert checkpoint.filesystem is plan.write_parquet.call_args.kwargs["filesystem"]
    plan.repartition.assert_called_once_with(target_num_rows_per_block=5, strict=True)
    assert plan.write_parquet.call_args.args == (attempt.workspace.results,)
    assert plan.write_parquet.call_args.kwargs["min_rows_per_file"] == 5
    assert plan.write_parquet.call_args.kwargs["max_rows_per_file"] == 5
    assert context.checkpoint_config is None
    assert attempt.processing_complete() is not fail_write


def test_processing_validates_results_before_ray_checkpoints_them(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that publication would reject fails before the sink can checkpoint its ID."""
    attempt = reconcile_and_select_attempt(enrichment_input)
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)
    plan = Mock()
    plan.map_batches.return_value = plan
    plan.repartition.return_value = plan
    with execution.checkpointed_processing(attempt) as processing:
        processing.write(plan)

    ((validate,), options) = plan.map_batches.call_args
    assert (options["batch_format"], options["batch_size"]) == ("pyarrow", None)
    assert plan.method_calls[0][0] == "map_batches"
    assert validate(results) is results
    assert validate(results.slice(0, 0)).num_rows == 0
    rows = results.to_pylist()
    rows[0]["result"] = None
    with pytest.raises(ValueError, match="Invalid test result"):
        validate(pa.Table.from_pylist(rows, schema=results.schema))
    with pytest.raises(ValueError, match="staged schema"):
        validate(results.drop_columns(["metadata"]))


@pytest.mark.parametrize("marker", [False, True])
def test_run_enrichment_starts_ray_whenever_fragments_are_selected(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch, *, marker: bool
) -> None:
    """A processing marker skips the plan but not Ray, which publication needs; a finished input needs neither."""
    events: list[object] = []
    if marker:
        attempt = reconcile_and_select_attempt(enrichment_input)
        assert attempt.workspace is not None
        attempt.workspace.record_completion(storage_profile="default")
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)

    def process(_processing: execution.ProcessingScope) -> None:
        checkpoint = DataContext.get_current().checkpoint_config
        events.append(("process", checkpoint and checkpoint.id_column))

    def prepare(*args: object, **kwargs: object) -> list[commit.PreparedFragment]:
        events.append("prepare")
        return prepare_results(results, *args, **kwargs)

    monkeypatch.setattr(execution, "prepare_enrichment_fragments", prepare)
    outcome = execution.run_enrichment(
        enrichment_input.bindings,
        start_ray=lambda _bindings: events.append("ray"),
        process=process,
    )
    assert events == (["ray", "prepare"] if marker else ["ray", ("process", "item_id"), "prepare"])
    assert outcome.summary()["selected_fragments"] == 2
    assert DataContext.get_current().checkpoint_config is None

    rerun = execution.run_enrichment(
        enrichment_input.bindings,
        start_ray=Mock(side_effect=AssertionError("Ray must not start")),
        process=Mock(side_effect=AssertionError("Processing must not run")),
    )
    assert rerun.summary()["selected_fragments"] == 0


def test_run_enrichment_limits_only_a_new_selection(
    enrichment_input: contracts.EnrichmentInput, results: pa.Table, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fragment limit bounds new work and leaves the rest pending for a later run."""
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", partial(prepare_results, results))
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)
    outcome = execution.run_enrichment(
        enrichment_input.bindings, start_ray=lambda _bindings: None, process=lambda _processing: None, max_fragments=1
    )
    assert (outcome.summary()["selected_fragments"], outcome.summary()["remaining_fragments"]) == (1, 1)
    assert contracts.pending_fragment_ids(outcome.dataset, enrichment_input.bindings) == (1,)


def test_preparation_never_starts_a_default_ray_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparing selected fragments requires the recipe's Ray runtime."""
    monkeypatch.setattr(execution.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(execution.ray.data, "read_parquet", Mock(side_effect=AssertionError("Read staged results")))
    options = {"filesystem": pafs.LocalFileSystem(), "prepare_fragment": Mock(), "memory_bytes": 1}
    with pytest.raises(RuntimeError, match="Start Ray with the recipe's runtime"):
        execution.prepare_enrichment_fragments(str(tmp_path), selected=(0,), **options)


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        (["untracked"], r"declared enrichment input fields: \['untracked'\]"),
        (["result"], r"declared enrichment input fields: \['result'\]"),
        (["fragment_id"], r"declared enrichment input fields: \['fragment_id'\]"),
        (["uri", "uri"], "must be unique"),
    ],
    ids=["undeclared", "owned_output", "row_address", "duplicate"],
)
def test_processing_reads_only_declared_input_columns(
    enrichment_input: contracts.EnrichmentInput, columns: list[str], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every projected value belongs to the input contract that recovery protects."""
    dataset = lance.dataset(enrichment_input.dataset.uri)
    dataset.add_columns(pa.schema([pa.field("untracked", pa.string())]))
    enrichment_input = contracts.open_enrichment_input(enrichment_input.bindings)
    attempt = reconcile_and_select_attempt(enrichment_input)
    monkeypatch.setattr(DataContext.get_current(), "checkpoint_config", None)
    read = Mock(side_effect=AssertionError("Invalid projections must fail before Ray reads"))
    monkeypatch.setattr(execution.ray.data, "read_datasource", read)
    with (
        execution.checkpointed_processing(attempt) as processing,
        pytest.raises(ValueError, match=message),
    ):
        processing.read(columns)
    read.assert_not_called()
    assert DataContext.get_current().checkpoint_config is None
    assert not attempt.processing_complete()


def test_processing_requires_selected_work(enrichment_input: contracts.EnrichmentInput, results: pa.Table) -> None:
    """A completed input has no checkpoint scope to own."""
    publish_results(results, enrichment_input)
    attempt = reconcile_and_select_attempt(contracts.open_enrichment_input(enrichment_input.bindings))
    with ExitStack() as stack, pytest.raises(ValueError, match="selected fragments"):
        stack.enter_context(execution.checkpointed_processing(attempt))
