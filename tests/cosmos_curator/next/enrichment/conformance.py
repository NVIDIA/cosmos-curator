# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared recovery behavior that every checkpointed enrichment recipe must satisfy.

A recipe adopts the suite by subclassing ``EnrichmentConformance`` as
``Test<Recipe>Conformance`` in its ``test_conformance.py``. Its conftest supplies the
``config``, ``enrichment_input``, ``dataset``, and ``results`` fixtures and a ``recipe``
fixture returning the recipe's module-level ``RecipeUnderTest``, which the fresh-cluster
scenario in ``fresh_cluster.py`` reuses. ``config`` is the recipe's resolved config,
whose ``kind`` the owned fields must name. ``results`` must hold one valid staged row
for every input row, spread across at least two fragments with at least two rows each.
Fragment IDs and recipe-specific summary keys are unrestricted. The recipe's
``run_config`` must run through its pipeline module's ``run_enrichment``; the suite
replaces the ``start_ray`` and ``process`` callbacks passed there, checks the recipe's
own ``transaction_properties``, and prepares fragments locally instead of starting Ray.
It uses real Lance files. Recipe test modules keep only the checks specific to their
own contract.
"""

import json
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import attrs
import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from cosmos_curator.next.enrichment import commit, contracts, execution, inputs
from cosmos_curator.next.enrichment import workspace as recovery_workspace
from cosmos_curator.next.enrichment.attempts import reconcile_and_select_attempt
from cosmos_curator.next.enrichment.workspace import RecoveryStateError
from cosmos_curator.next.utils.identity import canonical_digest
from tests.cosmos_curator.next.enrichment.helpers import attempt_all, prepare_results, publish_results


@attrs.frozen
class RecipeUnderTest:
    """How the conformance suite and fresh-cluster scenario drive one recipe."""

    pipeline: ModuleType
    """Recipe pipeline module whose ``run_config`` calls its imported ``run_enrichment``."""
    resolve_bindings: Callable[[Any], contracts.EnrichmentBindings]
    retune: Callable[[Any], Any]
    """Return the config with different execution tuning and the same results.

    The tuning must stay schedulable on the fresh-cluster scenario's small local cluster.
    """
    recontract: Callable[[Any], Any]
    """Return the config with a different result contract."""


type _Runs = list[tuple[dict[str, Any], execution.EnrichmentOutcome]]


def _replace_ray(
    recipe: RecipeUnderTest,
    monkeypatch: pytest.MonkeyPatch,
    *,
    start_ray: Callable[[contracts.EnrichmentBindings], None],
    process: Callable[[execution.ProcessingScope], None],
) -> _Runs:
    """Keep the recipe's run sequence while replacing its Ray setup and processing plan.

    Record each call's remaining recipe arguments and outcome, so checks use the
    callbacks the recipe actually passes.
    """
    runs: _Runs = []

    def run_enrichment(
        bindings: contracts.EnrichmentBindings,
        **kwargs: Any,  # noqa: ANN401
    ) -> execution.EnrichmentOutcome:
        outcome = execution.run_enrichment(bindings, **{**kwargs, "start_ray": start_ray, "process": process})
        runs.append((kwargs, outcome))
        return outcome

    monkeypatch.setattr(recipe.pipeline, "run_enrichment", run_enrichment)
    return runs


def _forbid_ray(recipe: RecipeUnderTest, monkeypatch: pytest.MonkeyPatch) -> _Runs:
    return _replace_ray(
        recipe,
        monkeypatch,
        start_ray=Mock(side_effect=AssertionError("Ray must not start")),
        process=Mock(side_effect=AssertionError("Processing must not run")),
    )


def _skip_processing(
    recipe: RecipeUnderTest,
    results: pa.Table,
    monkeypatch: pytest.MonkeyPatch,
    *,
    process: Callable[[execution.ProcessingScope], None] = lambda _processing: None,
) -> _Runs:
    """Treat ``results`` as already staged: no Ray setup, processing, or Ray preparation."""
    monkeypatch.setattr(execution, "prepare_enrichment_fragments", partial(prepare_results, results))
    return _replace_ray(recipe, monkeypatch, start_ray=lambda _bindings: None, process=process)


def _append_copy(dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, *, new_id: bool) -> lance.LanceDataset:
    """Append a copy of the first input row as a new fragment, optionally reusing its ID."""
    row = dataset.to_table(columns=spec.input_fields.names).slice(0, 1).to_pylist()[0]
    if new_id:
        row[spec.id_column] = f"{row[spec.id_column]}-appended"
    return lance.write_dataset(
        pa.Table.from_pylist([row], schema=spec.input_fields), dataset.uri, mode="append", data_storage_version="2.2"
    )


class EnrichmentConformance:
    """Shared publication and recovery checks, collected once per recipe subclass."""

    @staticmethod
    def _update_fragment(
        dataset: lance.LanceDataset, spec: contracts.EnrichmentSpec, column: str, values: pa.Array
    ) -> lance.LanceDataset:
        current = lance.dataset(dataset.uri)
        fragment = current.get_fragments()[0]
        rows = fragment.to_table(columns=[spec.id_column]).append_column(column, values)
        metadata, modified = fragment.update_columns(rows, left_on=spec.id_column, right_on=spec.id_column)
        return lance.LanceDataset.commit(
            dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=current.version,
        )

    def test_owner_is_the_recipe_kind(self, recipe: RecipeUnderTest, config: Any) -> None:  # noqa: ANN401
        """Publication provenance names the recipe by the ``kind`` its config declares."""
        assert recipe.resolve_bindings(config).spec.owner == config.kind

    def test_registration_adds_nullable_fields_without_data(
        self, enrichment_input: inputs.EnrichmentInput, dataset: lance.LanceDataset
    ) -> None:
        """Registration is metadata only, and reopening a registered table commits nothing."""
        spec = enrichment_input.bindings.spec
        assert set(spec.output_fields.names) <= set(dataset.schema.names)
        output_ids = set(inputs.field_ids(dataset, spec.output_fields.names).values())
        assert all(
            not output_ids.intersection(data_file["fields"])
            for fragment in dataset.get_fragments()
            for data_file in fragment.metadata.to_json()["files"]
        )
        assert inputs.open_enrichment_input(enrichment_input.bindings).dataset.version == dataset.version

    def test_one_commit_publishes_every_selected_fragment(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rows, fragment IDs, and base files survive; provenance and recipe records share the receipt's commit."""
        dataset = enrichment_input.dataset
        bindings = enrichment_input.bindings
        runs = _skip_processing(recipe, results, monkeypatch)
        summary = recipe.pipeline.run_config(config)
        ((recipe_arguments, outcome),) = runs
        published = outcome.dataset
        assert summary["field_set"] == bindings.spec.field_set
        assert summary["contract_digest"] == canonical_digest(bindings.spec.result_contract)
        assert published.version == dataset.version + 1
        assert inputs.pending_fragment_ids(published, enrichment_input.bindings) == ()
        for before, after in zip(dataset.get_fragments(), published.get_fragments(), strict=True):
            assert before.fragment_id == after.fragment_id
            original_files = before.metadata.to_json()["files"]
            assert original_files == after.metadata.to_json()["files"][: len(original_files)]
            assert after.metadata.num_deletions == 0
        properties = published.read_transaction(published.version).transaction_properties
        recipe_properties = recipe_arguments.get("transaction_properties")
        if recipe_properties is not None:
            assert recipe_properties(outcome.attempt).items() <= properties.items()
        assert properties[execution.KIND_KEY] == bindings.spec.owner
        assert properties[execution.FIELD_KEY] == bindings.spec.field_set
        assert properties[execution.CONTRACT_DIGEST_KEY] == canonical_digest(bindings.spec.result_contract)
        assert properties[inputs.VALIDATED_INPUT_KEY] == enrichment_input.fingerprint
        assert published.to_table(columns=bindings.spec.input_fields.names).equals(
            dataset.to_table(columns=bindings.spec.input_fields.names)
        )

    @pytest.mark.parametrize("fault", ["missing", "duplicate", "wrong_id", "wrong_offset"])
    def test_misaligned_results_never_publish(
        self, enrichment_input: inputs.EnrichmentInput, results: pa.Table, fault: str
    ) -> None:
        """Every row address and ID is checked before any result becomes visible."""
        id_column = enrichment_input.bindings.spec.id_column
        rows = results.to_pylist()
        if fault == "missing":
            rows.pop(0)
        elif fault == "duplicate":
            rows.append(rows[0])
        elif fault == "wrong_id":
            rows[0][id_column] = "wrong"
        else:
            rows[0]["row_offset"] = enrichment_input.dataset.count_rows()
        with pytest.raises(ValueError, match=f"Missing, duplicate, or misaligned {id_column} results"):
            publish_results(pa.Table.from_pylist(rows, schema=results.schema), enrichment_input)
        assert lance.dataset(enrichment_input.dataset.uri).version == enrichment_input.dataset.version

    def test_missing_fragment_results_never_publish(
        self, enrichment_input: inputs.EnrichmentInput, results: pa.Table
    ) -> None:
        """Valid results for one fragment cannot hide another fragment's absent work."""
        first_id = enrichment_input.dataset.get_fragments()[0].fragment_id
        first = results.filter(pc.equal(results["fragment_id"], first_id))
        with pytest.raises(ValueError, match="exactly cover"):
            publish_results(first, enrichment_input)
        assert lance.dataset(enrichment_input.dataset.uri).version == enrichment_input.dataset.version

    def test_mixed_fragments_are_rejected(self, enrichment_input: inputs.EnrichmentInput, results: pa.Table) -> None:
        """Only whole-fragment publication is a supported recovery state."""
        spec = enrichment_input.bindings.spec
        fragment = enrichment_input.dataset.get_fragments()[0]
        first = results.filter(pc.equal(results["fragment_id"], fragment.fragment_id))
        values = first.slice(0, 1).select([spec.id_column, *spec.output_fields.names])
        values = values.cast(pa.schema([spec.input_fields.field(spec.id_column), *spec.output_fields]))
        metadata, modified = fragment.update_columns(values, left_on=spec.id_column, right_on=spec.id_column)
        current = lance.LanceDataset.commit(
            enrichment_input.dataset.uri,
            lance.LanceOperation.Update(updated_fragments=[metadata], fields_modified=modified),
            read_version=enrichment_input.dataset.version,
        )
        with pytest.raises(ValueError, match="mixes pending and terminal"):
            inputs.pending_fragment_ids(current, enrichment_input.bindings)

    def test_rewritten_input_files_reject_publication(
        self, enrichment_input: inputs.EnrichmentInput, results: pa.Table
    ) -> None:
        """Stable IDs alone cannot rebind staged results to rewritten input files."""
        spec = enrichment_input.bindings.spec
        column = next(name for name in spec.input_fields.names if name != spec.id_column)
        values = enrichment_input.dataset.get_fragments()[0].to_table(columns=[column])[column]
        self._update_fragment(enrichment_input.dataset, spec, column, values.combine_chunks())
        with pytest.raises(ValueError, match="Input rows or file layout changed"):
            publish_results(results, enrichment_input)

    @pytest.mark.parametrize("during_commit", [False, True])
    def test_disjoint_writer_reprepares_only_stale_fragments(
        self,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        *,
        during_commit: bool,
    ) -> None:
        """Another writer's column file survives; only its fragment is rebuilt from staged results."""
        spec = enrichment_input.bindings.spec
        lance.dataset(enrichment_input.dataset.uri).add_columns(pa.schema([pa.field("other_writer", pa.string())]))
        fragments = enrichment_input.dataset.get_fragments()
        selected = tuple(int(fragment.fragment_id) for fragment in fragments)
        first_id = selected[0]
        rows = fragments[0].count_rows()
        lance_commit = lance.LanceDataset.commit
        raced = False

        def race(*args: object, **kwargs: object) -> lance.LanceDataset:
            nonlocal raced
            if not raced:
                raced = True
                self._update_fragment(enrichment_input.dataset, spec, "other_writer", pa.array(["keep"] * rows))
            return lance_commit(*args, **kwargs)

        if during_commit:
            monkeypatch.setattr(lance.LanceDataset, "commit", race)
        else:
            raced = True
            self._update_fragment(enrichment_input.dataset, spec, "other_writer", pa.array(["keep"] * rows))
        preparations = []

        def prepare(results_uri: str, *, selected: tuple[int, ...], **kwargs: Any) -> list[commit.PreparedFragment]:  # noqa: ANN401
            preparations.append((kwargs["prepare_fragment"].keywords["version"], selected))
            return prepare_results(results, results_uri, selected=selected, **kwargs)

        monkeypatch.setattr(execution, "prepare_enrichment_fragments", prepare)
        sink = commit.logger.add(sys.stderr, level="WARNING")
        try:
            published = execution.publish_enrichment(attempt_all(enrichment_input)).dataset
        finally:
            commit.logger.remove(sink)
        assert preparations == [
            (int(enrichment_input.dataset.version), selected),
            (int(published.version) - 1, (first_id,)),
        ]
        assert (
            published.get_fragment(first_id).to_table(columns=["other_writer"])["other_writer"].to_pylist()
            == ["keep"] * rows
        )
        assert inputs.pending_fragment_ids(published, enrichment_input.bindings) == ()
        if during_commit:
            assert "Retrying" in capfd.readouterr().err

    @pytest.mark.parametrize("append_after_commit", [False, True])
    def test_lost_commit_response_is_reconciled(
        self,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
        loguru_records: list[dict[str, Any]],
        *,
        append_after_commit: bool,
    ) -> None:
        """One warning reports the lost response while the receipt prevents a duplicate commit."""
        lance_commit = lance.LanceDataset.commit
        calls = []

        def lose_response(*args: object, **kwargs: object) -> None:
            calls.append(1)
            lance_commit(*args, **kwargs)
            if append_after_commit:
                _append_copy(enrichment_input.dataset, enrichment_input.bindings.spec, new_id=True)
            message = "lost response"
            raise OSError(message)

        monkeypatch.setattr(lance.LanceDataset, "commit", lose_response)
        sink = commit.logger.add(sys.stderr, level="WARNING")
        try:
            published = publish_results(results, enrichment_input)
        finally:
            commit.logger.remove(sink)
        assert published.version == enrichment_input.dataset.version + 1 + int(append_after_commit)
        assert len(calls) == 1
        (warning,) = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
        assert "OSError('lost response')" in warning
        assert enrichment_input.dataset.uri in warning
        assert f"publication verified from its durable receipt and canonical state at version {published.version}" in (
            warning
        )
        assert warning in capfd.readouterr().err

    @pytest.mark.parametrize("during_commit", [False, True])
    def test_appends_stay_pending_for_a_later_selection(
        self,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        *,
        during_commit: bool,
    ) -> None:
        """Rows appended before or during the commit are neither published nor certified as validated."""
        spec = enrichment_input.bindings.spec
        original_files = {
            f.fragment_id: f.metadata.to_json()["files"] for f in enrichment_input.dataset.get_fragments()
        }
        lance_commit = lance.LanceDataset.commit
        appended = False

        def race(*args: object, **kwargs: object) -> lance.LanceDataset:
            nonlocal appended
            if not appended:
                appended = True
                _append_copy(enrichment_input.dataset, spec, new_id=True)
            return lance_commit(*args, **kwargs)

        if during_commit:
            monkeypatch.setattr(lance.LanceDataset, "commit", race)
        else:
            _append_copy(enrichment_input.dataset, spec, new_id=True)
        published = publish_results(results, enrichment_input)
        transaction = published.read_transaction(published.version)
        assert transaction.transaction_properties[inputs.VALIDATED_INPUT_KEY] == enrichment_input.fingerprint
        assert not inputs.input_is_validated(
            published, fingerprint=inputs.input_fingerprint(published, enrichment_input.bindings)
        )
        for fragment_id, files in original_files.items():
            assert published.get_fragment(fragment_id).metadata.to_json()["files"][: len(files)] == files
        appended_ids = tuple(
            int(fragment.fragment_id)
            for fragment in published.get_fragments()
            if fragment.fragment_id not in original_files
        )
        assert len(appended_ids) == 1
        attempt = reconcile_and_select_attempt(inputs.open_enrichment_input(enrichment_input.bindings))
        assert attempt.selected == appended_ids
        tasks = execution.selection_datasource(attempt, columns=()).get_read_tasks(2)
        assert [row[spec.id_column] for task in tasks for batch in task() for row in batch.to_pylist()] == [
            f"{enrichment_input.dataset.to_table(columns=[spec.id_column])[spec.id_column][0].as_py()}-appended"
        ]

    def test_duplicate_append_requires_validation_again(
        self, enrichment_input: inputs.EnrichmentInput, results: pa.Table
    ) -> None:
        """Earlier validation evidence cannot hide an ID duplicated by a later append."""
        published = publish_results(results, enrichment_input)
        assert inputs.input_is_validated(published, fingerprint=enrichment_input.fingerprint)
        _append_copy(published, enrichment_input.bindings.spec, new_id=False)
        with pytest.raises(ValueError, match="non-null and globally unique"):
            inputs.open_enrichment_input(enrichment_input.bindings)

    def test_execution_tuning_keeps_the_workspace(
        self, recipe: RecipeUnderTest, config: object, dataset: lance.LanceDataset
    ) -> None:
        """Changing tuning reuses durable results; a different result contract does not."""
        selected = inputs.pending_fragment_ids(dataset, recipe.resolve_bindings(config))
        workspace = inputs.resolve_workspace(dataset, recipe.resolve_bindings(config), selected=selected)
        tuned = inputs.resolve_workspace(dataset, recipe.resolve_bindings(recipe.retune(config)), selected=selected)
        changed = inputs.resolve_workspace(
            dataset, recipe.resolve_bindings(recipe.recontract(config)), selected=selected
        )
        assert tuned == workspace
        assert changed != workspace

    @pytest.mark.parametrize("disjoint_write", [False, True])
    def test_rerun_verifies_publication_and_cleans_without_scanning(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        *,
        disjoint_write: bool,
    ) -> None:
        """A workspace left by a crash after commit is cleaned from receipts, without Ray or value scans."""
        runs = _skip_processing(recipe, results, monkeypatch)
        recipe.pipeline.run_config(config)
        ((_, outcome),) = runs
        attempt, published = outcome.attempt, outcome.dataset
        assert attempt.workspace is not None
        attempt.workspace.ensure(storage_profile=enrichment_input.bindings.execution.storage_profile)
        if disjoint_write:
            published.add_columns({"unrelated": "'keep'"})
        _forbid_ray(recipe, monkeypatch)
        monkeypatch.setattr(lance.LanceDataset, "sql", Mock(side_effect=AssertionError("Input must not be scanned")))
        monkeypatch.setattr(
            lance.fragment.LanceFragment, "scanner", Mock(side_effect=AssertionError("Results must not be scanned"))
        )
        verify = Mock(wraps=commit.verify_completed)
        monkeypatch.setattr(commit, "verify_completed", verify)
        summary = recipe.pipeline.run_config(config)
        assert summary["selected_fragments"] == summary["remaining_fragments"] == summary["selected_rows"] == 0
        assert summary["cleanup_complete"] is True
        assert not Path(attempt.workspace.root).exists()
        verify.assert_not_called()

    @pytest.mark.parametrize("incomplete", [False, True])
    def test_processing_marker_skips_processing_but_not_validation(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        *,
        incomplete: bool,
    ) -> None:
        """Publication-only recovery still refuses missing results before exposing any of them."""
        attempt = reconcile_and_select_attempt(enrichment_input)
        assert attempt.workspace is not None
        attempt.workspace.record_completion(storage_profile=enrichment_input.bindings.execution.storage_profile)
        _skip_processing(recipe, results, monkeypatch, process=Mock(side_effect=AssertionError("Processed again")))
        if incomplete:
            staged = results.filter(pc.equal(results["fragment_id"], attempt.selected[0]))
            monkeypatch.setattr(execution, "prepare_enrichment_fragments", partial(prepare_results, staged))
            with pytest.raises(ValueError, match="exactly cover"):
                recipe.pipeline.run_config(config)
            assert lance.dataset(enrichment_input.dataset.uri).version == enrichment_input.dataset.version
            assert Path(attempt.workspace.completion).exists()
        else:
            summary = recipe.pipeline.run_config(config)
            assert lance.dataset(enrichment_input.dataset.uri).version == enrichment_input.dataset.version + 1
            assert summary["selected_fragments"] == len(attempt.selected)
            assert summary["remaining_fragments"] == 0
            assert not Path(attempt.workspace.root).exists()

    def test_resumed_selection_reports_appends_as_remaining(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Finishing a saved selection leaves appended rows pending and counts them as remaining."""
        attempt = reconcile_and_select_attempt(enrichment_input)
        assert attempt.workspace is not None
        attempt.workspace.record_completion(storage_profile=enrichment_input.bindings.execution.storage_profile)
        input_rows = enrichment_input.dataset.count_rows()
        appended = _append_copy(enrichment_input.dataset, enrichment_input.bindings.spec, new_id=True)
        appended_ids = tuple(
            int(fragment.fragment_id)
            for fragment in appended.get_fragments()
            if fragment.fragment_id not in attempt.selected
        )
        _skip_processing(recipe, results, monkeypatch, process=Mock(side_effect=AssertionError("Processed again")))
        summary = recipe.pipeline.run_config(config)
        assert {name: summary[name] for name in attempt.counts()} == {
            "input_rows": input_rows + 1,
            "selected_rows": input_rows,
            "remaining_rows": 1,
            "selected_fragments": len(attempt.selected),
            "remaining_fragments": 1,
        }
        assert (
            inputs.pending_fragment_ids(lance.dataset(enrichment_input.dataset.uri), enrichment_input.bindings)
            == appended_ids
        )
        assert not Path(attempt.workspace.root).exists()

    def test_incompatible_workspace_is_preserved(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
    ) -> None:
        """Canonical completion never authorizes removing recovery state that fails validation."""
        attempt = attempt_all(enrichment_input)
        assert attempt.workspace is not None
        publish_results(results, enrichment_input)
        attempt.workspace.ensure(storage_profile=enrichment_input.bindings.execution.storage_profile)
        manifest = Path(attempt.workspace.root, "workspace.json")
        manifest.write_text(json.dumps({"incompatible": True}), encoding="utf-8")
        with pytest.raises(RecoveryStateError, match="Invalid fragment selection in recovery workspace"):
            recipe.pipeline.run_config(config)
        assert manifest.exists()

    def test_cleanup_failure_keeps_published_results(
        self,
        recipe: RecipeUnderTest,
        config: object,
        enrichment_input: inputs.EnrichmentInput,
        results: pa.Table,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Cleanup is best effort after publication and never a reason to process again."""
        _skip_processing(recipe, results, monkeypatch)

        def fail_cleanup(*_args: object, **_kwargs: object) -> None:
            message = "cleanup unavailable"
            raise OSError(message)

        monkeypatch.setattr(recovery_workspace, "remove_prefix", fail_cleanup)
        sink = recovery_workspace.logger.add(sys.stderr, level="WARNING")
        try:
            first = recipe.pipeline.run_config(config)
            second = recipe.pipeline.run_config(config)
        finally:
            recovery_workspace.logger.remove(sink)
        assert (first["selected_fragments"], first["cleanup_complete"]) == (
            len(enrichment_input.dataset.get_fragments()),
            False,
        )
        assert (second["selected_fragments"], second["cleanup_complete"]) == (0, False)
        assert "workspace cleanup failed" in capfd.readouterr().err
