# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage one embedding column group's computed vectors as checkpointed Parquet.

Computation and publication are separate phases. This module owns the first: it
turns owed rows into staged rows and records which rows are done. Nothing here
writes to the clips table, and no worker holds the capability to - publication is
one driver-side commit in ``publication.py``.

::

    read_datasource(EmbeddingScanDatasource)   owed rows, each carrying its
        |                                      fragment id and checkpoint id
        |   override_num_blocks = owed fragments
        v
    repartition(target_num_rows_per_block)     split a large fragment without
        |   strict=False                       mixing adjacent ones
        v
    map_batches(EmbeddingBatchWorker)          embedder built once per actor;
        |   ActorPoolStrategy(1, placeable)    compute only, no Lance handle
        v
    write_parquet(results_uri, APPEND)         Ray commits the checkpoint ids of
            partition_cols=[fragment_id]       each completed write bundle

The whole chain runs inside one ``installed_checkpoint`` block, and the shape
above is a CORRECTNESS requirement of it rather than a preference. Ray plans job
checkpointing only for a DAG that starts at a read and ends at a write: the read
is where the skip filter is inserted, and the write is the only moment at which a
row is provably finished. A ``materialize()`` terminal would disable the whole
mechanism with nothing but a Python warning.

The checkpoint id must therefore survive untouched from the read to the sink. It
is projected by the datasource, carried through the worker by
:class:`EmbeddingBatchWorker`, and written into the staged file, because Ray reads
it back off the sink to mark a row complete. A plan that dropped it anywhere in
between would record no progress while appearing to work.

What a crash costs is one write bundle per actor, not the phase: rows in bundles
that already committed are filtered out at the read on the next attempt, so their
GPU time is never paid twice.

A worker fault is FATAL to the attempt: nothing here tolerates a lost partition
of rows. The checkpoint is what makes that affordable - a re-run resumes rather
than restarting - and it is what a tolerated loss cannot offer, since publishing
around a failed subset produces a group that reads as complete while an unknown
set of rows holds nothing.
"""

import math
from typing import Any, Final

import lance
import pyarrow as pa
import ray
from loguru import logger
from ray.data import ActorPoolStrategy, SaveMode

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.core.ray_runtime import installed_checkpoint
from cosmos_curator.next.embeddings.schemas import KEY_COLUMN, EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.columns import pending_filter
from cosmos_curator.next.recipes.embeddings.datasource import (
    EmbeddingScanDatasource,
    EmbeddingScanPlan,
    PinnedScan,
)
from cosmos_curator.next.recipes.embeddings.modalities import ModalityFill, WorkerResources
from cosmos_curator.next.recipes.embeddings.staged import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    staged_schema,
)
from cosmos_curator.next.recipes.embeddings.workspace import EmbeddingWorkspace

# Written as a PAIR because Ray couples them: ``_apply_default_remote_args`` forces
# ``max_task_retries=-1`` whenever ``max_restarts`` is non-zero, so bounding the
# restarts alone would leave tasks retrying forever. Both are bounded rather than
# zero: an actor that dies on a transient fault should be replaced, but an actor
# dying on a permanent one - unreadable media, a model that cannot load - must
# stop the attempt instead of looping on it for the run's lifetime.
ACTOR_RESTART_BOUNDS: Final = {"max_restarts": 5, "max_task_retries": 5}

# The identity columns a staged row carries out of the worker, in staged-schema
# order. Taken from the INPUT batch rather than from the embedder, which never
# sees them.
_IDENTITY_COLUMNS: Final = (KEY_COLUMN, FRAGMENT_ID_COLUMN, CHECKPOINT_ID_COLUMN)

# Ray's own key for the heap reservation, both in the ``memory=`` request and in
# ``available_resources()``. Named once because the same string has to index the
# cluster snapshot AND select the byte rendering below: as two literals they could
# drift, and the divisor would then silently stop dividing by memory at all.
_MEMORY_RESOURCE: Final = "memory"

_BYTES_PER_GIB: Final = 1024**3


class EmbeddingBatchWorker:
    """Ray Data callable class computing one modality's vectors for a batch of owed rows.

    Constructed once per actor from picklable arguments only, so the model weights
    load once per actor rather than once per batch. It opens no Lance dataset and
    holds no storage client: this actor pool is the only place in the leg that
    holds a GPU, and giving it write capability is what the staged-then-publish
    split exists to avoid.
    """

    def __init__(
        self,
        *,
        group: EmbeddingColumnGroup,
        staged: pa.Schema,
        embedder_cls: type,
        embedder_kwargs: dict[str, Any],
    ) -> None:
        """Store the staged contract and build this actor's embedder once.

        Args:
            group: The modality's column group, named in failure messages.
            staged: The schema every returned batch is cast to.
            embedder_cls: Callable class taking an Arrow batch of the modality's
                source columns and returning exactly the group's columns.
            embedder_kwargs: Keyword arguments ``embedder_cls`` is built from.

        """
        self._group = group
        self._staged = staged
        self._embedder = embedder_cls(**embedder_kwargs)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one staged row per input row, in input order.

        Raises:
            ValueError: If the embedder returned a different row count, or columns
                that do not cast to the staged schema. Both messages are
                self-contained because Ray replaces whatever a UDF raises with a
                wrapper carrying neither the type nor the cause.

        """
        computed = self._embedder(batch)
        if computed.num_rows != batch.num_rows:
            msg = (
                f"embedder for group {self._group.name!r} returned {computed.num_rows} row(s) for a "
                f"{batch.num_rows}-row batch; the contract is one output row per input row, in input order"
            )
            raise ValueError(msg)
        # Attached POSITIONALLY, which is sound only because the embedder is
        # cardinality- and order-preserving - exactly what the check above
        # enforces. Without it a violation would silently pair vectors with the
        # wrong clips, and the checkpoint id with the wrong vector.
        staged = computed
        for index, name in enumerate(_IDENTITY_COLUMNS):
            staged = staged.add_column(index, name, batch.column(name))
        try:
            # Cast without a preceding ``select``: the cast is what rejects a
            # missing, renamed, extra or reordered column, and selecting first
            # would silently drop an extra one instead.
            return staged.cast(self._staged)
        except (ValueError, pa.ArrowTypeError, pa.ArrowNotImplementedError) as e:
            # Measured at pyarrow 25.0.1: a missing, renamed, extra or REORDERED
            # column raises a plain ValueError; an uncastable dtype raises
            # ArrowNotImplementedError; a wrong vector width raises ArrowInvalid,
            # which is caught by the ValueError arm it subclasses rather than by
            # the ArrowTypeError one. So do NOT trim ValueError as redundant with
            # the Arrow arms - it is what carries the width fault. The common base
            # pa.ArrowException would be shorter but also catches ArrowMemoryError,
            # and an allocation failure is transient, so promoting it here would
            # fail an attempt a retry would have completed.
            msg = (
                f"embedder for group {self._group.name!r} returned columns that do not match the staged "
                f"schema: {e}. Expected {self._staged.to_string(show_field_metadata=False)}, "
                f"got {staged.schema.to_string(show_field_metadata=False)}"
            )
            raise ValueError(msg) from e


def stage_embedding_group(  # noqa: PLR0913 - each argument is an independent input to one phase
    dataset: lance.LanceDataset,
    fill: ModalityFill,
    *,
    workspace: EmbeddingWorkspace,
    plan: EmbeddingScanPlan,
    rows_per_file: int,
    storage_options: dict[str, str] | None,
) -> None:
    """Compute one modality's owed rows and stage them, recording each completed row.

    Returns nothing because the staged rows ARE the output: they live in
    ``workspace.results_uri`` until publication reads them back. A count returned
    here would be this attempt's alone, while publication needs the total across
    every attempt that staged into this workspace.

    Args:
        dataset: The clips table, open at the version ``plan`` was built against.
        fill: The modality's embedder, source columns, group and actor shape.
        workspace: Namespace the staged rows and the checkpoint are written into.
        plan: The owed fragments. Must be non-empty; a modality owing nothing is
            skipped by the caller rather than executed with an empty plan.
        rows_per_file: Rows per staged Parquet file. With the modality's batch
            size it bounds what one checkpoint makes durable, and a crash costs
            the LARGER of the two - lowering this alone below the batch size
            buys no finer recovery.
        storage_options: Lance credentials each read task re-opens the table with.

    Raises:
        ValueError: If ``plan`` is empty, if the shared scan inputs do not resolve
            against the table, or if a worker violated the embedder contract.

    """
    if not plan.fragments:
        msg = (
            f"embedding group {fill.group.name!r}: refusing to execute an empty scan plan. A modality that "
            f"owes no work must be skipped before staging, or the run reports a phase it never ran"
        )
        raise ValueError(msg)
    # Re-derived rather than passed in: the predicate is a pure function of the
    # fill, so recomputing it cannot disagree with the one the caller planned
    # with, and threading it through the signature would invite a caller to pass
    # a different one than the plan was built from.
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    _check_fill_preconditions(dataset, fill, row_filter=row_filter)
    resources = fill.resources
    staged = staged_schema(fill.group)
    source = EmbeddingScanDatasource(
        PinnedScan.for_dataset(
            dataset,
            storage_options=storage_options,
            source_columns=fill.source_columns,
            row_filter=row_filter,
            batch_size=resources.scan_batch_size,
        ),
        plan.fragments,
    )
    max_actors = _placeable_actors(resources, owed_fragments=len(plan.fragments), group_name=fill.group.name)
    # The filesystem comes off the workspace so the checkpoint and the sink below
    # are reached through ONE credential set. Two independently resolved
    # filesystems let the sink write where the checkpoint cannot read, which
    # surfaces as a resume that silently redoes everything rather than as an
    # access error.
    with installed_checkpoint(
        id_column=CHECKPOINT_ID_COLUMN,
        checkpoint_path=workspace.checkpoints_uri,
        filesystem=workspace.filesystem,
    ):
        owed = ray.data.read_datasource(
            source,
            override_num_blocks=len(plan.fragments),
            ray_remote_args={"runtime_env": ray_data_gpu_runtime_env(resources.env_name)},
        ).repartition(
            target_num_rows_per_block=resources.scan_batch_size,
            # Non-strict processes each source block independently: it splits a
            # large fragment without coalescing or mixing adjacent fragments, so
            # one block still belongs to one fragment and the Hive partition the
            # sink writes stays a property of the read rather than of a shuffle.
            strict=False,
        )
        computed = owed.map_batches(
            EmbeddingBatchWorker,
            fn_constructor_kwargs={
                "group": fill.group,
                "staged": staged,
                "embedder_cls": fill.embedder_cls,
                "embedder_kwargs": fill.embedder_kwargs,
            },
            batch_format="pyarrow",
            batch_size=resources.scan_batch_size,
            compute=ActorPoolStrategy(min_size=1, max_size=max_actors),
            # Pinned for every modality, GPU and CPU alike: the worker needs the
            # interpreter that holds the model stack, and an unpinned worker would
            # silently inherit whichever env Ray happened to start.
            runtime_env=ray_data_gpu_runtime_env(resources.env_name),
            # Packs fractional-GPU actors rather than spreading them.
            scheduling_strategy="DEFAULT",
            # Ray declares all three as optional and treats an explicit ``None``
            # exactly as an omitted kwarg, so an unset resource passes through.
            num_gpus=resources.num_gpus,
            num_cpus=resources.num_cpus,
            memory=resources.memory_bytes,
            # Spread whole because the two keys are one decision; see
            # ``ACTOR_RESTART_BOUNDS``.
            **ACTOR_RESTART_BOUNDS,
        )
        computed.write_parquet(
            workspace.results_uri,
            filesystem=workspace.filesystem,
            partition_cols=[FRAGMENT_ID_COLUMN],
            # Append, so a resume adds to what an earlier attempt staged instead of
            # replacing it. The checkpoint guarantees those two sets are disjoint.
            # Passed explicitly even though it is Ray's current default: were that
            # default ever to become OVERWRITE, a resume would delete the very
            # vectors it exists to preserve.
            mode=SaveMode.APPEND,
            # Both bounds are HINTS in Ray's sink rather than guarantees, and are
            # set to one value only to keep a staged file small enough that a crash
            # mid-phase leaves usable partial work. No repartition is added to
            # enforce them: paying a shuffle to hit an exact file size would cost
            # more than the uneven files it avoids.
            min_rows_per_file=rows_per_file,
            max_rows_per_file=rows_per_file,
            ray_remote_args={"runtime_env": ray_data_gpu_runtime_env(resources.env_name)},
        )
    logger.info(
        f"embedding group {fill.group.name!r}: staged {plan.num_rows} owed row(s) across "
        f"{len(plan.fragments)} fragment(s) under {workspace.results_uri}"
    )


def _format_resource(name: str, amount: float) -> str:
    """Render one resource amount in the unit its name implies.

    ``memory`` is a byte count while GPU and CPU are device counts, so one format
    for all three renders the memory bound as an unlabelled ``1.9e+11`` - a figure
    an operator cannot weigh against a node spec or against the ``memory_gb`` they
    set, in the one line whose whole purpose is to name the bound they can act on.
    """
    if name == _MEMORY_RESOURCE:
        return f"{amount / _BYTES_PER_GIB:g} GiB"
    return f"{amount:g}"


def _placeable_actors(resources: WorkerResources, *, owed_fragments: int, group_name: str) -> int:
    """Return the actor ceiling this run's cluster snapshot and work can justify.

    Two independent bounds, whichever is smaller. The RESOURCE bound is what the
    live nodes can actually place: an unbounded pool autoscales against queue
    pressure, and an actor Ray cannot place still counts in the pool's own
    occupancy, so a saturated stage would report slack it does not have. The WORK
    bound is the owed fragment count, because a fragment-poor table starves a pool
    of any size.

    The resource bound divides by EVERY resource the modality reserves and keeps
    the tightest quotient. A modality holding a GPU, also reserving CPUs for
    host-side decode, and also reserving a heap for the clips in flight is
    placeable only where all three fit, so dividing by the GPU alone would
    reproduce on a CPU-poor or memory-poor node exactly the over-count this
    bound exists to prevent.

    Args:
        resources: The modality's per-actor request.
        owed_fragments: Fragments this attempt owes work in.
        group_name: Column group, for the log line.

    Returns:
        At least 1, so a cluster momentarily reporting nothing free still runs.

    """
    requested = {
        name: value
        for name, value in (
            ("GPU", resources.num_gpus),
            ("CPU", resources.num_cpus),
            # Divided by on the same footing as the other two: this is the figure
            # ``stage_embedding_group`` passes as ``memory=``, and Ray decrements
            # the node's advertised ``memory`` by it at admission - measured, an
            # actor requesting 64 MiB moved ``available_resources()["memory"]`` by
            # exactly 64 MiB. That Ray does not ENFORCE the figure against real
            # RSS is a separate fact and no reason to leave it out of the
            # division: an actor whose reservation does not fit stays pending
            # while still counting in the pool's occupancy, which is precisely
            # the over-count this bound exists to prevent.
            (_MEMORY_RESOURCE, resources.memory_bytes),
        )
        if value
    }
    if requested:
        # One snapshot for every resource: two calls could report capacities from
        # different moments, and the minimum of those is a cluster that never was.
        snapshot = ray.available_resources()  # type: ignore[no-untyped-call]
        # Compare the RAW quotients and clamp only the winner. Clamping first
        # floors every exhausted resource at 1, so the basis would be decided by
        # dict order rather than by scarcity - and the basis is the whole point
        # of the log line, which exists to name the bound the operator can act on.
        quotients = {name: math.floor(float(snapshot.get(name, 0.0)) / value) for name, value in requested.items()}
        basis = min(quotients, key=lambda name: quotients[name])
        placeable = max(1, quotients[basis])
        # The capacity says "registered so far" because it counts whichever nodes
        # had joined by the time this ran, not the nodes the job was allocated.
        # Stated on the number itself rather than left to a cluster snapshot
        # logged elsewhere, which is far enough away that a reader quoting this
        # one line loses the caveat.
        resource_clause = (
            f"the nodes registered so far advertise {_format_resource(basis, float(snapshot.get(basis, 0.0)))} free "
            f"{basis} against {_format_resource(basis, float(requested[basis]))} per actor, the tightest of the "
            f"{'/'.join(sorted(requested))} it reserves ({placeable} placeable)"
        )
    else:
        # A modality reserving no GPU, CPU or memory accepts Ray's own default
        # request, which this function cannot divide by. The work bound still
        # applies, so the pool is bounded either way.
        placeable = owed_fragments
        resource_clause = "it reserves no GPU, CPU or memory to divide, so no resource bound was measured"
    ceiling = min(placeable, owed_fragments)
    logger.info(
        f"embedding group {group_name!r}: capping the actor pool at {ceiling}. For this modality {resource_clause}, "
        f"and {owed_fragments} fragment(s) are owed"
    )
    return ceiling


def _check_fill_preconditions(dataset: lance.LanceDataset, fill: ModalityFill, *, row_filter: str) -> None:
    """Verify the scan inputs every read task shares, before any worker is started.

    The projection and the row filter are properties of the RUN, not of a
    fragment, so a broken one fails on every read task alike. Checked here so the
    fault arrives once, on the driver, with the inputs named - rather than as an
    opaque Ray wrapper after an actor pool has already been built and the model
    weights loaded onto a GPU.

    ``explain_plan`` is what makes checking this affordable: it resolves and
    optimizes the scan - which is where an unknown column or an unparsable
    predicate is rejected - without reading a row. Probing for a first matching
    row would instead scan until one appeared, which on a table whose owed rows
    sit at the end is most of the table.

    Raises:
        ValueError: If ``clip_id`` is absent from the projection, or the
            projection and filter do not resolve against the table.

    """
    if KEY_COLUMN not in fill.source_columns:
        msg = (
            f"embedding group {fill.group.name!r}: source columns {list(fill.source_columns)} omit "
            f"{KEY_COLUMN!r}; the embedder never reads it, but every staged row carries it and publication "
            f"joins on it, so it must be projected"
        )
        raise ValueError(msg)
    try:
        dataset.scanner(columns=list(fill.source_columns), filter=row_filter).explain_plan()
    except ValueError as e:
        # Lance raises ValueError for an unparsable predicate, a predicate naming
        # an absent column, and a projection naming an absent column alike. An IO
        # fault would be an OSError and is deliberately NOT caught here: it is not
        # a statement about the run's inputs, so it must not be relabelled as a
        # violated contract.
        msg = (
            f"embedding group {fill.group.name!r}: the per-fragment scan does not resolve against "
            f"{dataset.uri} at v{int(dataset.version)} (columns={list(fill.source_columns)}, "
            f"filter={row_filter!r}): {e}"
        )
        raise ValueError(msg) from e
