# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared recovery behavior that every checkpointed append recipe must satisfy.

A recipe adopts the suite by subclassing ``AppendConformance`` as ``Test<Recipe>Conformance``
in its ``test_conformance.py``. Its conftest supplies the ``config`` and ``results`` fixtures
and a ``recipe`` fixture returning the recipe's module-level ``RecipeUnderTest``, which the
fresh-cluster scenario in ``fresh_cluster.py`` reuses. ``config`` is the recipe's resolved
config, whose ``kind`` the spec's owner must name; ``recipe.discover`` finds its work items
without external services. ``results`` must hold one valid canonical result in the spec's
``result_fields`` for each of at least three discovered items: at least two that append
rows and one that appends none.
When the spec declares an ``error_column``, at least one result must be a terminal error.
Recipe-specific summary keys are unrestricted. The recipe's ``run_config`` must run through
its pipeline module's ``run_append``; the suite replaces the ``start_ray`` and ``process``
callbacks passed there, stages ``results`` into the selection's workspace as processing
would, and stages fragments locally instead of with Ray. It uses real Lance and staged
result files. Recipe test modules keep only the checks specific to their own contract.
"""

import json
import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from cosmos_curator.next.append import contracts, execution, inputs
from cosmos_curator.next.append.attempts import reconcile_and_select_attempt
from cosmos_curator.next.core import recovery_workspace
from cosmos_curator.next.core.lance_transactions import CONTRACT_DIGEST_KEY, KIND_KEY, iter_transactions
from cosmos_curator.next.core.recovery_workspace import RecoveryStateError
from tests.cosmos_curator.next.append.helpers import Runs, replace_ray, stage_results


@attrs.frozen
class RecipeUnderTest:
    """How the conformance suite and fresh-cluster scenario drive one append recipe."""

    pipeline: ModuleType
    """Recipe pipeline module whose ``run_config`` calls its imported ``run_append``."""
    resolve_bindings: Callable[[Any], contracts.AppendBindings]
    discover: Callable[[Any], pa.Table]
    """Return the work items the recipe's ``run_config`` discovers for a config."""
    retune: Callable[[Any], Any]
    """Return the config with different execution tuning and the same results.

    The tuning must stay schedulable on the fresh-cluster scenario's small local cluster.
    """
    recontract: Callable[[Any], Any]
    """Return the config with a different result contract for the same table."""


_OTHER_WRITER = pa.schema([pa.field("other_writer", pa.string())])


def _replace_ray(recipe: RecipeUnderTest, monkeypatch: pytest.MonkeyPatch, *, results: pa.Table | None) -> Runs:
    return replace_ray(recipe.pipeline, monkeypatch, results=results)


def _forbid_ray(recipe: RecipeUnderTest, monkeypatch: pytest.MonkeyPatch) -> Runs:
    return replace_ray(
        recipe.pipeline, monkeypatch, results=None, start_ray=Mock(side_effect=AssertionError("Ray must not start"))
    )


def _sorted_ids(items: pa.Table, spec: contracts.AppendSpec) -> list[str]:
    return sorted(items[spec.id_column].to_pylist())


def _expected_rows(spec: contracts.AppendSpec, results: pa.Table, item_ids: list[str]) -> list[dict[str, Any]]:
    selected = results.filter(pc.is_in(results[spec.id_column], value_set=pa.array(item_ids, type=pa.string())))
    tables = contracts.expanded_rows(selected, spec)
    return sorted((row for table in tables for row in table.to_pylist()), key=json.dumps)


def _appended_rows(dataset: lance.LanceDataset, spec: contracts.AppendSpec) -> list[dict[str, Any]]:
    return sorted(dataset.to_table(columns=spec.output_fields.names).to_pylist(), key=json.dumps)


def _items_with_rows(spec: contracts.AppendSpec, results: pa.Table, item_ids: list[str]) -> list[str]:
    return sorted({row[spec.id_column] for row in _expected_rows(spec, results, item_ids)})


def _appends(dataset: lance.LanceDataset) -> list[lance.Transaction]:
    return [
        transaction
        for transaction in iter_transactions(dataset, property_key=KIND_KEY)
        if isinstance(transaction.operation, lance.LanceOperation.Append)
    ]


def _expected_errors(spec: contracts.AppendSpec, results: pa.Table, item_ids: list[str]) -> list[dict[str, Any]]:
    if spec.error_column is None:
        return []
    selected = results.filter(pc.is_in(results[spec.id_column], value_set=pa.array(item_ids, type=pa.string())))
    errors = selected.filter(pc.is_valid(selected[spec.error_column]))
    return sorted(errors.select([spec.id_column, spec.error_column]).to_pylist(), key=json.dumps)


class AppendConformance:
    """Shared publication and recovery checks, collected once per recipe subclass."""

    @staticmethod
    def _split(recipe: RecipeUnderTest, config: object, results: pa.Table) -> tuple[pa.Table, pa.Table]:
        """Hold out the last item with rows, so the earlier items include both kinds."""
        spec = recipe.resolve_bindings(config).spec
        items = recipe.discover(config)
        item_ids = _sorted_ids(items, spec)
        with_rows = _items_with_rows(spec, results, item_ids)
        assert len(item_ids) >= 3, "The suite needs at least three discovered items"
        assert len(with_rows) >= 2, "The suite needs at least two items that append rows"
        held_out = pc.equal(items[spec.id_column], with_rows[-1])
        return items.filter(pc.invert(held_out)), items.filter(held_out)

    def test_owner_is_the_recipe_kind(self, recipe: RecipeUnderTest, config: object) -> None:
        """Publication provenance and the table binding name the recipe by its config ``kind``."""
        assert recipe.resolve_bindings(config).spec.owner == config.kind

    def test_one_commit_appends_every_discovered_item(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful, empty, and error outcomes become visible in one atomic append."""
        bindings = recipe.resolve_bindings(config)
        spec = bindings.spec
        assert results.schema.equals(spec.result_fields, check_metadata=True)
        item_ids = _sorted_ids(recipe.discover(config), spec)
        runs = _replace_ray(recipe, monkeypatch, results=results)
        summary = recipe.pipeline.run_config(config)
        ((_, outcome),) = runs.calls
        published = outcome.dataset
        (transaction,) = _appends(published)
        properties = transaction.transaction_properties
        assert published.read_transaction(published.version).transaction_properties == properties
        assert "cosmos_curator.append.publication" not in properties
        assert properties[KIND_KEY] == spec.owner
        assert properties[CONTRACT_DIGEST_KEY] == spec.contract_digest
        expected = _expected_rows(spec, results, item_ids)
        assert _appended_rows(published, spec) == expected
        errors = len(_expected_errors(spec, results, item_ids))
        assert {name: summary[name] for name in outcome.summary()} == {
            "input_items": len(item_ids),
            "selected_items": len(item_ids),
            "remaining_items": 0,
            "appended_rows": len(expected),
            "errors": errors,
            "cleanup_complete": True,
            "contract_digest": spec.contract_digest,
        }
        assert outcome.attempt.workspace is not None
        assert not Path(outcome.attempt.workspace.root).exists()

    def test_terminal_errors_remain_in_current_rows(
        self, recipe: RecipeUnderTest, config: object, results: pa.Table, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Diagnostics and completion survive cleanup, history pruning, and unchanged reruns."""
        bindings = recipe.resolve_bindings(config)
        spec = bindings.spec
        if spec.error_column is None:
            pytest.skip("The recipe declares no terminal errors")
        expected = _expected_errors(spec, results, _sorted_ids(recipe.discover(config), spec))
        assert expected
        _replace_ray(recipe, monkeypatch, results=results)
        summary = recipe.pipeline.run_config(config)
        dataset = lance.dataset(bindings.uri)
        dataset.add_columns(pa.schema([pa.field("extra", pa.string())]))
        dataset.cleanup_old_versions(older_than=timedelta(seconds=0))
        errors = lance.dataset(bindings.uri).to_table(
            columns=[spec.id_column, spec.error_column], filter=pc.field(spec.error_column).is_valid()
        )
        assert sorted(errors.to_pylist(), key=json.dumps) == expected
        assert summary["errors"] == len(expected)
        rerun = recipe.pipeline.run_config(config)
        assert (rerun["selected_items"], rerun["errors"]) == (0, 0)

    def test_published_items_are_never_processed_again(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty and error outcomes are recognized from current rows; a rerun starts no Ray and commits nothing."""
        _replace_ray(recipe, monkeypatch, results=results)
        recipe.pipeline.run_config(config)
        version = lance.dataset(recipe.resolve_bindings(config).uri).version
        monkeypatch.undo()
        _forbid_ray(recipe, monkeypatch)
        summary = recipe.pipeline.run_config(config)
        assert (summary["selected_items"], summary["appended_rows"], summary["cleanup_complete"]) == (0, 0, True)
        assert lance.dataset(recipe.resolve_bindings(config).uri).version == version

    def test_new_items_append_without_changing_published_fragments(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Later discoveries append new fragments; earlier files and another writer's values are preserved."""
        bindings = recipe.resolve_bindings(config)
        spec = bindings.spec
        earlier, later = self._split(recipe, config, results)
        runs = _replace_ray(recipe, monkeypatch, results=results)
        with runs.discovering(earlier):
            recipe.pipeline.run_config(config)

        @lance.batch_udf(output_schema=_OTHER_WRITER)
        def other_writer(batch: pa.RecordBatch) -> pa.RecordBatch:
            return pa.RecordBatch.from_pylist([{"other_writer": "keep"}] * batch.num_rows, schema=_OTHER_WRITER)

        lance.dataset(bindings.uri).add_columns(other_writer, read_columns=[spec.id_column])
        enriched = lance.dataset(bindings.uri)
        existing = [fragment.metadata.to_json()["files"] for fragment in enriched.get_fragments()]
        summary = recipe.pipeline.run_config(config)
        assert (summary["selected_items"], summary["remaining_items"]) == (later.num_rows, 0)
        published = lance.dataset(bindings.uri)
        files = [fragment.metadata.to_json()["files"] for fragment in published.get_fragments()]
        assert files[: len(existing)] == existing
        values = published.to_table(columns=[spec.id_column, "other_writer"])
        added = pc.is_in(values[spec.id_column], value_set=later[spec.id_column].combine_chunks())
        assert values.filter(pc.invert(added))["other_writer"].to_pylist() == ["keep"] * enriched.count_rows()
        appended = values.filter(added)["other_writer"]
        assert len(appended) > 0
        assert appended.null_count == len(appended)
        assert _appended_rows(published, spec) == _expected_rows(
            spec, results, _sorted_ids(recipe.discover(config), spec)
        )

    def test_unfinished_selection_resumes_before_new_items(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A saved selection finishes without discovery; later invocations discover and publish new items."""
        bindings = recipe.resolve_bindings(config)
        earlier, later = self._split(recipe, config, results)
        saved = reconcile_and_select_attempt(bindings, lambda: earlier)
        _replace_ray(recipe, monkeypatch, results=results)
        first = recipe.pipeline.run_config(config)
        assert (first["input_items"], first["selected_items"], first["remaining_items"]) == (
            None,
            earlier.num_rows,
            None,
        )
        assert saved.workspace is not None
        assert not Path(saved.workspace.root).exists()
        second = recipe.pipeline.run_config(config)
        assert (second["selected_items"], second["remaining_items"]) == (later.num_rows, 0)
        published = lance.dataset(bindings.uri)
        assert inputs.published_item_ids(
            published, bindings, _sorted_ids(recipe.discover(config), bindings.spec)
        ) >= set(_sorted_ids(recipe.discover(config), bindings.spec))

    def test_lost_commit_response_is_reconciled(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """One warning reports the lost response while canonical files prevent a second append."""
        lance_commit = lance.LanceDataset.commit
        calls = []

        def lose_response(*args: object, **kwargs: object) -> None:
            calls.append(1)
            lance_commit(*args, **kwargs)
            message = "lost response"
            raise OSError(message)

        _replace_ray(recipe, monkeypatch, results=results)
        monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
        sink = execution.commit.logger.add(sys.stderr, level="WARNING")
        try:
            recipe.pipeline.run_config(config)
        finally:
            execution.commit.logger.remove(sink)
        bindings = recipe.resolve_bindings(config)
        published = lance.dataset(bindings.uri)
        assert len(calls) == 1
        assert len(_appends(published)) == 1
        assert _appended_rows(published, bindings.spec) == _expected_rows(
            bindings.spec, results, _sorted_ids(recipe.discover(config), bindings.spec)
        )
        (warning,) = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
        assert "OSError('lost response')" in warning
        assert warning in capfd.readouterr().err

    @pytest.mark.parametrize("fault", ["missing", "invalid_schema"])
    def test_incomplete_results_never_publish(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        fault: str,
    ) -> None:
        """A missing or malformed result fails publication and retains the workspace."""
        bindings = recipe.resolve_bindings(config)
        id_column = bindings.spec.id_column
        if fault == "missing":
            staged = results.filter(pc.not_equal(results[id_column], results[id_column][0]))
            message = "do not exactly cover the selected items"
        else:
            staged = results.drop_columns([name for name in results.schema.names if name != id_column][:1])
            message = "result schema"
        attempt = reconcile_and_select_attempt(bindings, lambda: recipe.discover(config))
        stage_results(attempt, staged)
        _replace_ray(recipe, monkeypatch, results=None)
        with pytest.raises(ValueError, match=message):
            recipe.pipeline.run_config(config)
        assert _appends(lance.dataset(bindings.uri)) == []
        assert attempt.workspace is not None
        assert Path(attempt.workspace.root, "workspace.json").exists()

    def test_execution_tuning_keeps_the_workspace(self, recipe: RecipeUnderTest, config: object) -> None:
        """Changing tuning reuses durable results; a different result contract does not."""
        bindings = recipe.resolve_bindings(config)
        items = inputs.validate_items(recipe.discover(config), bindings.spec)
        dataset = inputs.open_target(bindings)
        workspace = inputs.resolve_workspace(dataset, bindings, selection=items)
        tuned_config = recipe.retune(config)
        tuned = recipe.resolve_bindings(tuned_config)
        assert recipe.discover(tuned_config).equals(recipe.discover(config))
        assert inputs.resolve_workspace(dataset, tuned, selection=items) == workspace
        changed = recipe.resolve_bindings(recipe.recontract(config))
        assert inputs.workspace_root(dataset, changed) != inputs.workspace_root(dataset, bindings)

    def test_table_belongs_to_one_result_contract(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A different result contract needs a new output location, so rows never mix contracts."""
        _replace_ray(recipe, monkeypatch, results=results)
        recipe.pipeline.run_config(config)
        with pytest.raises(ValueError, match="different producer, item ID column, or result contract"):
            recipe.pipeline.run_config(recipe.recontract(config))

    def test_rerun_verifies_publication_and_cleans_without_history(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A workspace left after commit is cleaned from current rows without Ray or retained history."""
        runs = _replace_ray(recipe, monkeypatch, results=results)
        recipe.pipeline.run_config(config)
        ((_, outcome),) = runs.calls
        workspace = outcome.attempt.workspace
        assert workspace is not None
        workspace.ensure(storage_profile=recipe.resolve_bindings(config).execution.storage_profile)
        monkeypatch.undo()
        _forbid_ray(recipe, monkeypatch)
        monkeypatch.setattr(lance.LanceDataset, "read_transaction", Mock(side_effect=AssertionError("History read")))
        summary = recipe.pipeline.run_config(config)
        assert (summary["selected_items"], summary["cleanup_complete"]) == (0, True)
        assert not Path(workspace.root).exists()

    @pytest.mark.parametrize("incomplete", [False, True])
    def test_processing_marker_skips_processing_but_not_validation(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        *,
        incomplete: bool,
    ) -> None:
        """Publication-only recovery still refuses missing results before exposing any of them."""
        bindings = recipe.resolve_bindings(config)
        attempt = reconcile_and_select_attempt(bindings, lambda: recipe.discover(config))
        assert attempt.workspace is not None
        stage_results(attempt, results.slice(1) if incomplete else results)
        _replace_ray(recipe, monkeypatch, results=None)
        if incomplete:
            with pytest.raises(ValueError, match="do not exactly cover the selected items"):
                recipe.pipeline.run_config(config)
            assert _appends(lance.dataset(bindings.uri)) == []
            assert Path(attempt.workspace.completion).exists()
        else:
            summary = recipe.pipeline.run_config(config)
            assert summary["selected_items"] == attempt.selection.num_rows
            assert not Path(attempt.workspace.root).exists()

    def test_incompatible_workspace_is_preserved(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unreadable saved selection is never discarded or replaced by new work."""
        bindings = recipe.resolve_bindings(config)
        attempt = reconcile_and_select_attempt(bindings, lambda: recipe.discover(config))
        assert attempt.workspace is not None
        manifest = Path(attempt.workspace.root, "workspace.json")
        manifest.write_text(json.dumps({"incompatible": True}), encoding="utf-8")
        _replace_ray(recipe, monkeypatch, results=results)
        with pytest.raises(RecoveryStateError, match="Invalid item selection in recovery workspace"):
            recipe.pipeline.run_config(config)
        assert manifest.exists()

    def test_cleanup_failure_keeps_published_results(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Publication survives cleanup failure, but another invocation waits for the workspace to clear."""
        _replace_ray(recipe, monkeypatch, results=results)

        def fail_cleanup(uri: str, **_kwargs: object) -> None:
            # Removing a workspace fails; a new selection's absent workspace has nothing to remove.
            if Path(uri).exists():
                message = "cleanup unavailable"
                raise OSError(message)

        monkeypatch.setattr(recovery_workspace, "remove_prefix", fail_cleanup)
        sink = recovery_workspace.logger.add(sys.stderr, level="WARNING")
        try:
            first = recipe.pipeline.run_config(config)
            with pytest.raises(OSError, match="until workspace cleanup succeeds"):
                recipe.pipeline.run_config(config)
        finally:
            recovery_workspace.logger.remove(sink)
        assert (first["selected_items"] > 0, first["cleanup_complete"]) == (True, False)
        assert "workspace cleanup failed" in capfd.readouterr().err

    def test_pruned_history_never_duplicates_rows(
        self,
        recipe: RecipeUnderTest,
        config: object,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All outcomes still identify completed items after history pruning."""
        bindings = recipe.resolve_bindings(config)
        spec = bindings.spec
        earlier, _ = self._split(recipe, config, results)
        runs = _replace_ray(recipe, monkeypatch, results=results)
        with runs.discovering(earlier):
            recipe.pipeline.run_config(config)
        recipe.pipeline.run_config(config)
        lance.dataset(bindings.uri).cleanup_old_versions(older_than=timedelta(seconds=0))
        item_ids = _sorted_ids(recipe.discover(config), spec)
        rerun = recipe.pipeline.run_config(config)
        assert (rerun["selected_items"], rerun["appended_rows"]) == (0, 0)
        assert _appended_rows(lance.dataset(bindings.uri), spec) == _expected_rows(spec, results, item_ids)
