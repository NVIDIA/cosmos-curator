# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run an enrichment attempt's Ray Data processing and publish it in one atomic Lance update.

``run_enrichment`` is the shared run sequence. It opens and validates the
recipe's input, resumes or selects an attempt, starts the recipe's Ray runtime
when fragments are selected, runs the recipe's plan inside
``checkpointed_processing`` unless processing already completed, and publishes.
The processing scope owns both ends of that read-to-Parquet plan: it reads the
frozen selection and writes validated staged results inside one native Ray
checkpoint. ``publish_enrichment`` prepares those results with Ray Data, commits
them with the shared recovery receipt, and removes the verified workspace.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import partial
from typing import Any, cast

import attrs
import lance
import pyarrow as pa
import pyarrow.fs as pafs
import ray
from ray.data.expressions import col

from cosmos_curator.next.core.lance_datasource import LanceFragmentDatasource
from cosmos_curator.next.core.ray_checkpoints import installed_checkpoint
from cosmos_curator.next.enrichment import commit, contracts
from cosmos_curator.next.enrichment.attempts import EnrichmentAttempt, reconcile_and_select_attempt
from cosmos_curator.next.enrichment.commit import PreparedFragment
from cosmos_curator.next.enrichment.workspace import RecoveryWorkspace
from cosmos_curator.next.utils.arrow_utils import arrow_filesystem


def selection_datasource(attempt: EnrichmentAttempt, *, columns: Sequence[str]) -> LanceFragmentDatasource:
    """Read the attempt's frozen selection at its pinned input version, as ``ProcessingScope.read`` does."""
    enrichment_input = attempt.input
    bindings = enrichment_input.bindings
    return LanceFragmentDatasource(
        enrichment_input.dataset,
        fragment_ids=attempt.selected,
        schema=bindings.spec.read_schema(columns),
        storage_profile=bindings.execution.storage_profile,
        batch_size=bindings.execution.lance_read_batch_size,
    )


def _require_publishable(block: pa.Table, *, spec: contracts.EnrichmentSpec) -> pa.Table:
    """Pass staged rows through to the sink only if publication will accept them."""
    if block.num_rows:
        contracts.canonical_results(block, spec)
    return block


@attrs.frozen
class ProcessingScope:
    """Both ends of one checkpointed read-to-Parquet plan over an attempt's selection."""

    attempt: EnrichmentAttempt
    workspace: RecoveryWorkspace
    filesystem: pafs.FileSystem

    @property
    def staged_schema(self) -> pa.Schema:
        """Return the schema of rows the plan passes to ``write``; build results in it on workers."""
        return self.attempt.input.bindings.spec.staged_schema

    def read(self, columns: Sequence[str], *, override_num_blocks: int | None = None) -> ray.data.Dataset:
        """Read the IDs, named input columns, and row addresses, with checkpoint filtering before expensive work.

        The columns must be declared input fields; see ``EnrichmentSpec.read_schema``.
        """
        return cast(
            "ray.data.Dataset",
            ray.data.read_datasource(
                selection_datasource(self.attempt, columns=columns),
                override_num_blocks=override_num_blocks,
            ),
        )

    def write(self, results: ray.data.Dataset) -> None:
        """Validate staged results, write them in strict checkpoint groups, then record that processing completed.

        Rows must match the spec's staged schema. Ray checkpoints a result's ID once
        its Parquet file is written, so every row is validated first: a checkpointed
        result can always be published.
        """
        bindings = self.attempt.input.bindings
        rows_per_file = bindings.execution.parquet_rows_per_file
        publishable = results.map_batches(
            partial(_require_publishable, spec=bindings.spec),
            batch_format="pyarrow",
            batch_size=None,
            zero_copy_batch=True,
            num_cpus=0.25,
        )
        groups = publishable.repartition(target_num_rows_per_block=rows_per_file, strict=True)
        groups.write_parquet(
            self.workspace.results,
            filesystem=self.filesystem,
            min_rows_per_file=rows_per_file,
            max_rows_per_file=rows_per_file,
        )
        self.workspace.record_completion(storage_profile=bindings.execution.storage_profile)


@contextmanager
def checkpointed_processing(attempt: EnrichmentAttempt) -> Iterator[ProcessingScope]:
    """Construct and execute one plan from ``scope.read`` to ``scope.write`` inside a Ray checkpoint.

    The checkpoint keys on the spec's ID column and shares the result sink's
    filesystem. It is retained until the attempt's publication is verified.
    """
    workspace = attempt.workspace
    if workspace is None or not attempt.selected:
        msg = "Processing requires an attempt with selected fragments and a recovery workspace"
        raise ValueError(msg)
    bindings = attempt.input.bindings
    filesystem = arrow_filesystem(workspace.root, storage_profile=bindings.execution.storage_profile)
    with installed_checkpoint(
        id_column=bindings.spec.id_column, checkpoint_path=workspace.checkpoints, filesystem=filesystem
    ):
        yield ProcessingScope(attempt, workspace, filesystem)


@attrs.frozen
class EnrichmentOutcome:
    """The table after an attempt and whether every eligible recovery workspace was removed."""

    attempt: EnrichmentAttempt
    dataset: lance.LanceDataset = attrs.field(eq=False)
    cleanup_complete: bool

    def summary(self) -> dict[str, object]:
        """Report the shared run-summary vocabulary; recipes add their own fields."""
        bindings = self.attempt.input.bindings
        return {
            **self.attempt.counts(),
            "cleanup_complete": self.cleanup_complete,
            "field_set": bindings.spec.field_set,
            "contract_digest": bindings.contract_digest,
        }


def summary_message(kind: str, summary: Mapping[str, object]) -> str:
    """Render the one-line CLI result shared by enrichment recipes."""
    return (
        f"{kind}: published results for {summary['selected_rows']} of {summary['input_rows']} input row(s); "
        f"{summary['remaining_fragments']} fragment(s) pending"
    )


def _bound_workspace(attempt: EnrichmentAttempt) -> RecoveryWorkspace:
    """Reject attempts assembled by hand from a different input, selection, or workspace."""
    enrichment_input = attempt.input
    if enrichment_input.fingerprint != contracts.input_fingerprint(enrichment_input.dataset, enrichment_input.bindings):
        msg = "Publication input no longer matches the validated snapshot"
        raise ValueError(msg)
    workspace = attempt.workspace
    if workspace is None or workspace != contracts.resolve_workspace(
        enrichment_input.dataset, enrichment_input.bindings, selected=attempt.selected
    ):
        msg = "Publication requires the workspace belonging to its validated input and selected fragments"
        raise ValueError(msg)
    return workspace


def publish_enrichment(
    attempt: EnrichmentAttempt, *, transaction_properties: Mapping[str, str] | None = None
) -> EnrichmentOutcome:
    """Publish the attempt's selection with proof of its validated input, then remove its workspace.

    The shared publisher records the spec's owner as ``kind``, its field set as
    ``field``, and the result contract's ``contract_digest``. It also owns the
    recovery receipt and validation evidence, including during descriptor refresh.
    Optional recipe properties add records that other recipes read. An attempt
    without selected fragments has nothing to publish. Cleanup is best effort and
    runs only after canonical verification.
    """
    enrichment_input = attempt.input
    bindings = enrichment_input.bindings
    shared = {
        contracts.KIND_KEY: bindings.spec.owner,
        contracts.FIELD_KEY: bindings.spec.field_set,
        contracts.CONTRACT_DIGEST_KEY: bindings.contract_digest,
        contracts.VALIDATED_INPUT_KEY: enrichment_input.fingerprint,
    }
    recipe_properties = dict(transaction_properties or {})
    if {*shared, contracts.PUBLICATION_RECEIPT_KEY} & recipe_properties.keys():
        msg = "Recipe properties cannot replace the shared provenance, recovery receipt, or validation evidence"
        raise ValueError(msg)
    if not attempt.selected:
        return EnrichmentOutcome(attempt, enrichment_input.dataset, attempt.cleanup_complete)
    workspace = _bound_workspace(attempt)
    execution = bindings.execution

    def prepare(version: int, selected: tuple[int, ...]) -> list[PreparedFragment]:
        return prepare_enrichment_fragments(
            workspace.results,
            selected=selected,
            filesystem=arrow_filesystem(workspace.root, storage_profile=execution.storage_profile),
            prepare_fragment=partial(commit.prepare_fragment, bindings=bindings, version=version),
            memory_bytes=int(execution.publication_memory_gib * 1024**3),
        )

    published = commit.commit_fragments(
        prepare,
        dataset=enrichment_input.dataset,
        bindings=bindings,
        selected=attempt.selected,
        receipt=workspace.manifest_digest,
        transaction_properties={**recipe_properties, **shared},
    )
    cleaned = workspace.cleanup(storage_profile=execution.storage_profile)
    return EnrichmentOutcome(attempt, published, attempt.cleanup_complete and cleaned)


def run_enrichment(
    bindings: contracts.EnrichmentBindings,
    *,
    start_ray: Callable[[contracts.EnrichmentBindings], None],
    process: Callable[[ProcessingScope], None],
    transaction_properties: Callable[[EnrichmentAttempt], Mapping[str, str]] | None = None,
    max_fragments: int | None = None,
) -> EnrichmentOutcome:
    """Open validated input, resume saved work or select pending fragments, process what is missing, and publish once.

    The input is opened with ``open_enrichment_input``, which registers the
    output fields and validates the pinned snapshot or reuses earlier proof.
    ``start_ray`` initializes the recipe's Ray runtime whenever fragments are
    selected, even when a valid processing marker skips ``process``: publication
    prepares column files with Ray Data. It receives the bindings, whose
    ``storage_env_vars`` cover the table and staging root. ``process`` builds and
    executes the recipe's plan from ``scope.read`` to ``scope.write``. An attempt
    without selected fragments starts no Ray. ``transaction_properties``
    optionally adds recipe records that other recipes read; the shared provenance
    needs none. ``max_fragments`` bounds only a new selection.
    """
    attempt = reconcile_and_select_attempt(contracts.open_enrichment_input(bindings), max_fragments=max_fragments)
    if attempt.selected:
        start_ray(bindings)
        if not attempt.processing_complete():
            with checkpointed_processing(attempt) as processing:
                process(processing)
    properties = None if transaction_properties is None else transaction_properties(attempt)
    return publish_enrichment(attempt, transaction_properties=properties)


def prepare_enrichment_fragments(
    results_uri: str,
    *,
    selected: tuple[int, ...],
    filesystem: pafs.FileSystem,
    prepare_fragment: Callable[[pa.Table], pa.Table],
    memory_bytes: int,
) -> list[PreparedFragment]:
    """Prepare selected whole fragments and collect only their compact descriptors.

    The publisher binds validation and a pinned Lance version to ``prepare_fragment``.
    A publication retry can select only stale fragments from the same staged results.
    """
    if not ray.is_initialized():
        # Ray Data would otherwise start a default local runtime without the
        # recipe's resources or storage environment.
        msg = "Start Ray with the recipe's runtime before publishing selected fragments"
        raise RuntimeError(msg)
    staged = ray.data.read_parquet(results_uri, filesystem=filesystem)
    staged = staged.filter(expr=col("fragment_id").is_in(list(selected)))
    prepared = staged.groupby("fragment_id", num_partitions=len(selected)).map_groups(
        cast("Any", prepare_fragment),
        batch_format="pyarrow",
        num_cpus=1,
        memory=memory_bytes,
    )
    candidates: list[PreparedFragment] = []
    for batch in prepared.iter_batches(batch_format="pyarrow", batch_size=None, prefetch_batches=0):
        candidates.extend(PreparedFragment.from_row(row) for row in cast("pa.Table", batch).to_pylist())
    return candidates
