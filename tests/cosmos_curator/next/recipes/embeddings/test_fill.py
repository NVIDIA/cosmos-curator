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

"""Compute-only worker contract, and the checkpointed staging phase built on it.

Two layers, tested at two costs. ``EmbeddingBatchWorker`` is pure Arrow - it
embeds a batch and splices the row identity back on - so it is driven directly
with stub embedders and needs no Ray. That keeps the contract failures it pins
(a mis-sized output, a schema that will not cast, a mispaired key) observable in
isolation.

The staging tests DO start a real Ray execution, because what they pin is Ray's
behaviour rather than ours: whether the job checkpoint was planned at all, and
whether a second attempt's embedder is handed the rows a first attempt already
committed. Neither is observable without the actual plan, and a stub of the
checkpoint would only assert that the stub works.

Those tests observe the embedder from ANOTHER PROCESS, so they cannot count calls
in memory. Each invocation appends a file naming the clips it saw, which is what
makes "the embedder was handed nothing" a positive observation rather than the
absence of one.

The image group stands in for all three modalities: its per-row validity mask and
its single vector make it the smallest group that can express both a filled and a
failed row.
"""

import pathlib
import warnings

import attrs
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.schemas import (
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    IMAGE_GROUP_SCHEMA,
    image_columns_batch,
)
from cosmos_curator.next.recipes.embeddings.columns import pending_filter
from cosmos_curator.next.recipes.embeddings.config import Modality
from cosmos_curator.next.recipes.embeddings.datasource import plan_embedding_scan
from cosmos_curator.next.recipes.embeddings.fill import (
    EmbeddingBatchWorker,
    _check_fill_preconditions,
    _placeable_actors,
    stage_embedding_group,
)
from cosmos_curator.next.recipes.embeddings.modalities import (
    _IMAGE_APPLICABILITY_FILTER,
    ModalityFill,
    WorkerResources,
)
from cosmos_curator.next.recipes.embeddings.staged import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    staged_schema,
)
from cosmos_curator.next.recipes.embeddings.workspace import EmbeddingWorkspace

from .conftest import ClipsTableFactory, add_group_columns

_IMAGE_SOURCE_COLUMNS = ("clip_id", "clip_uri")
_MODEL_ID = "stub-image-model"
_STAGED_SCHEMA = staged_schema(IMAGE_COLUMN_GROUP)

# Where an out-of-process embedder records what it was handed. Set through an
# environment variable because the stub is constructed inside a Ray actor, which
# shares no memory with the test.
_OBSERVED_DIR_ENV = "COSMOS_TEST_EMBEDDER_OBSERVED_DIR"


class _ConstantImageEmbedder:
    """Emit one always-valid image group row per input row (the happy-path contract)."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return a constant vector for every row of ``batch``, in input order."""
        return _valid_image_group_rows(batch.num_rows)


class _UriEncodingImageEmbedder:
    """Encode each row's own ``clip_uri`` into its vector, making the key pairing observable.

    A constant vector cannot distinguish a correct positional splice from a
    shuffled one. This makes the row a vector came from recoverable from the
    vector itself, and requires the fixture's ``clipN.bin`` URI shape.
    """

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one vector per row whose first component is that row's ``clipN.bin`` index."""
        uris = batch.column("clip_uri").to_pylist()
        matrix = np.zeros((len(uris), IMAGE_DIM), dtype=np.float32)
        matrix[:, 0] = [_clip_index(uri) for uri in uris]
        return image_columns_batch(len(uris), matrix, _MODEL_ID, np.ones(len(uris), dtype=np.bool_))


class _OneRowFailingImageEmbedder:
    """Emit an all-NULL group for one clip, modelling a per-row compute failure.

    The only stub that stages a NULL vector beside surviving siblings, which is
    the row shape the ``fixed_size_list`` Parquet defect makes unreadable if the
    staged schema does not widen the vector.
    """

    def __init__(self, *, null_uri: str) -> None:
        """Bind the ``clip_uri`` whose group row is emitted all-NULL."""
        self._null_uri = null_uri

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one row per input row, the bound clip's row all-NULL."""
        uris = batch.column("clip_uri").to_pylist()
        matrix = np.zeros((len(uris), IMAGE_DIM), dtype=np.float32)
        matrix[:, 0] = [_clip_index(uri) for uri in uris]
        valid = np.array([uri != self._null_uri for uri in uris], dtype=np.bool_)
        return image_columns_batch(len(uris), matrix, _MODEL_ID, valid)


class _RowDroppingImageEmbedder:
    """Violate the cardinality contract by returning one row fewer than it was given."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return ``batch.num_rows - 1`` group rows, which the worker must reject."""
        return _valid_image_group_rows(max(batch.num_rows - 1, 0))


class _ColumnDroppingImageEmbedder:
    """Return the right number of rows, but omit one of the group's columns."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Drop the provenance column, so the result cannot cast to the staged schema."""
        return _valid_image_group_rows(batch.num_rows).drop_columns([IMAGE_COLUMN_GROUP.provenance_columns[0]])


class _ObservingImageEmbedder:
    """Record the clips of every batch it is handed, then embed them normally.

    Constructed inside a Ray actor, so it reports through the filesystem: one file
    per invocation under ``_OBSERVED_DIR_ENV``. A run whose embedder was handed
    nothing therefore leaves an EMPTY directory, which is a fact the test can
    assert rather than an absence it has to infer.
    """

    def __init__(self, *, observed_dir: str) -> None:
        """Bind the directory this actor appends its observations to."""
        self._observed = pathlib.Path(observed_dir)

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Record this batch's clip ids, then return one valid row per input row."""
        clips = sorted(batch.column("clip_id").to_pylist())
        self._observed.mkdir(parents=True, exist_ok=True)
        # Named by content so two actors observing the same clips cannot collide
        # into a single file and under-report; the test reads the union.
        (self._observed / f"{'-'.join(clips)}.txt").write_text("\n".join(clips), encoding="utf-8")
        return _valid_image_group_rows(batch.num_rows)


def _clip_index(uri: str) -> float:
    """Return the ``N`` of a fixture ``clipN.bin`` URI."""
    return float(uri.removeprefix("clip").removesuffix(".bin"))


def _valid_image_group_rows(rows: int) -> pa.Table:
    """Return ``rows`` well-formed image group rows, for stubs that then break one thing."""
    return image_columns_batch(
        rows,
        np.ones((rows, IMAGE_DIM), dtype=np.float32),
        _MODEL_ID,
        np.ones(rows, dtype=np.bool_),
    )


def _scanned_batch(clip_ids: list[str], *, fragment_id: int = 0) -> pa.Table:
    """Build the batch the datasource hands a worker: source columns plus row identity."""
    rows = len(clip_ids)
    return pa.table(
        {
            "clip_id": pa.array(clip_ids, pa.string()),
            "clip_uri": pa.array([f"clip{index}.bin" for index in range(rows)], pa.large_string()),
            FRAGMENT_ID_COLUMN: pa.array([fragment_id] * rows, pa.int64()),
            # Distinct per row, and deliberately not equal to the row offset, so a
            # worker that rebuilt the ids instead of carrying them through cannot
            # coincidentally produce the expected values.
            CHECKPOINT_ID_COLUMN: pa.array([100 + index for index in range(rows)], pa.uint64()),
        }
    )


def _worker(embedder_cls: type, **embedder_kwargs: str) -> EmbeddingBatchWorker:
    """Build an image-group worker around a stub embedder."""
    return EmbeddingBatchWorker(
        group=IMAGE_COLUMN_GROUP,
        staged=_STAGED_SCHEMA,
        embedder_cls=embedder_cls,
        embedder_kwargs=embedder_kwargs,
    )


def _image_fill(embedder_cls: type, **embedder_kwargs: str) -> ModalityFill:
    """Build an image-group fill spec around a stub embedder, sized for the local cluster."""
    return ModalityFill(
        modality=Modality.IMAGE,
        group=IMAGE_COLUMN_GROUP,
        source_columns=_IMAGE_SOURCE_COLUMNS,
        applicability_filter=_IMAGE_APPLICABILITY_FILTER,
        result_identity={"embedding_image_model_id": _MODEL_ID},
        embedder_cls=embedder_cls,
        embedder_kwargs=embedder_kwargs,
        resources=WorkerResources(scan_batch_size=2, num_cpus=1),
        weights_name=None,
    )


def _run_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the pixi ``py_executable`` so staging actors run in the interpreter running the test."""
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.fill.ray_data_gpu_runtime_env",
        lambda _env_name: ray_data_gpu_runtime_env(""),
    )


def _workspace(tmp_path: pathlib.Path) -> EmbeddingWorkspace:
    """Return a local workspace, built directly rather than resolved.

    Resolution is pinned in ``test_workspace.py``; what staging needs is only a
    root whose ``results`` and ``checkpoints`` are siblings.
    """
    return EmbeddingWorkspace(
        root_uri=str(tmp_path / "staging" / "image" / "namespace"),
        filesystem=pa.fs.LocalFileSystem(),
    )


def _stage(
    uri: str,
    fill: ModalityFill,
    workspace: EmbeddingWorkspace,
    *,
    max_fragments: int | None = None,
) -> None:
    """Plan and stage one modality against the table's current version."""
    dataset = lance.dataset(uri)
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    plan = plan_embedding_scan(dataset, row_filter=row_filter, max_fragments=max_fragments)
    stage_embedding_group(
        dataset,
        fill,
        workspace=workspace,
        plan=plan,
        rows_per_file=2,
        storage_options=None,
    )


def _staged_clips(workspace: EmbeddingWorkspace) -> list[str]:
    """Return every clip id staged under the workspace, across all partitions."""
    results = pathlib.Path(workspace.results_uri)
    files = sorted(results.rglob("*.parquet"))
    return sorted(clip for path in files for clip in pq.read_table(path).column("clip_id").to_pylist())


def _observed_clips(observed_dir: pathlib.Path) -> set[str]:
    """Return the union of the clip ids every embedder invocation recorded."""
    if not observed_dir.exists():
        return set()
    return {clip for path in observed_dir.iterdir() for clip in path.read_text(encoding="utf-8").split()}


def test_the_worker_pairs_each_vector_with_the_row_it_was_computed_from() -> None:
    """A vector spliced onto the wrong row is silent, so the pairing is pinned per row.

    The embedder encodes each row's own URI into its vector, so a shuffled or
    offset splice produces a detectably wrong vector rather than a plausible one.
    """
    batch = _scanned_batch(["c0", "c1", "c2"])

    staged = _worker(_UriEncodingImageEmbedder)(batch)

    first_components = [vector[0] for vector in staged.column("embedding_image").to_pylist()]
    assert list(zip(staged.column("clip_id").to_pylist(), first_components, strict=True)) == [
        ("c0", 0.0),
        ("c1", 1.0),
        ("c2", 2.0),
    ]


def test_the_worker_carries_the_checkpoint_id_through_untouched() -> None:
    """Ray reads this column back off the sink, so a worker that rebuilt it records nothing.

    The fixture's ids are deliberately not the row offsets, so a worker deriving
    them from position would produce different values rather than the same ones.
    """
    batch = _scanned_batch(["c0", "c1", "c2"])

    staged = _worker(_ConstantImageEmbedder)(batch)

    assert staged.column(CHECKPOINT_ID_COLUMN).to_pylist() == [100, 101, 102]


def test_the_worker_emits_exactly_the_staged_schema() -> None:
    """Publication reads these files back by name and type, so the staged shape is the contract."""
    staged = _worker(_ConstantImageEmbedder)(_scanned_batch(["c0", "c1"]))

    assert staged.schema == _STAGED_SCHEMA


def test_a_wrong_row_count_is_refused_naming_both_counts() -> None:
    """Ray erases the exception type, so the message must be self-contained.

    ``match=`` is what makes the expected failure attributable: without it the
    test would pass on any ``ValueError``, including one from a broken fixture.
    """
    worker = _worker(_RowDroppingImageEmbedder)

    with pytest.raises(ValueError, match="returned 2 row\\(s\\) for a 3-row batch"):
        worker(_scanned_batch(["c0", "c1", "c2"]))


def test_a_result_that_cannot_cast_is_refused_naming_both_schemas() -> None:
    """The reader of this message has neither schema to hand, so both are named in it."""
    worker = _worker(_ColumnDroppingImageEmbedder)

    with pytest.raises(ValueError, match="do not match the staged schema") as raised:
        worker(_scanned_batch(["c0", "c1"]))

    assert "Expected" in str(raised.value)
    assert "got" in str(raised.value)


def test_a_null_vector_survives_the_staged_cast_and_a_parquet_round_trip(tmp_path: pathlib.Path) -> None:
    """A per-row failure stages a NULL vector, which a fixed-size list cannot round-trip.

    Measured on pyarrow 25.0.1: a ``fixed_size_list`` column holding any NULL is
    unreadable once written to Parquet. The staged schema widens the vector for
    exactly this row, so the round trip is executed here rather than assumed - a
    schema assertion alone would still pass against the unreadable shape.
    """
    staged = _worker(_OneRowFailingImageEmbedder, null_uri="clip1.bin")(_scanned_batch(["c0", "c1", "c2"]))
    path = tmp_path / "staged.parquet"
    pq.write_table(staged, path)

    vectors = pq.read_table(path).column("embedding_image").to_pylist()

    assert vectors[1] is None
    assert [vector[0] for index, vector in enumerate(vectors) if index != 1] == [0.0, 2.0]


def test_the_actor_ceiling_is_bounded_by_the_owed_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fragment-poor table starves a pool of any size, so the work bounds it.

    The cluster is told it has far more CPU than the work needs, so only the work
    bound can produce the expected answer.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"CPU": 64.0})

    assert _placeable_actors(WorkerResources(scan_batch_size=2, num_cpus=1), owed_fragments=3, group_name="g") == 3


def test_the_actor_ceiling_is_bounded_by_what_the_cluster_can_place(monkeypatch: pytest.MonkeyPatch) -> None:
    """An actor Ray cannot place still counts in the pool's occupancy, so the pool is capped.

    The mirror of the work bound: here the work is ample and the cluster is
    scarce, so only the resource bound can produce the expected answer.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"GPU": 2.0})

    assert _placeable_actors(WorkerResources(scan_batch_size=2, num_gpus=0.5), owed_fragments=99, group_name="g") == 4


def test_the_actor_ceiling_divides_by_every_resource_the_modality_reserves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A modality reserving GPU and CPU is placeable only where BOTH fit.

    The cluster is deliberately GPU-rich and CPU-poor, and the two quotients are
    skewed 8 against 1, so a bound that divided by the GPU alone would answer 8
    and over-count by the factor that produces the unplaceable actors the bound
    exists to prevent.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"GPU": 8.0, "CPU": 4.0})

    resources = WorkerResources(scan_batch_size=2, num_gpus=1, num_cpus=4)
    assert _placeable_actors(resources, owed_fragments=99, group_name="image") == 1


def test_the_actor_ceiling_divides_by_the_reserved_heap(monkeypatch: pytest.MonkeyPatch) -> None:
    """A modality reserving a heap is placeable only where that heap fits too.

    Ray admits an actor against its ``memory=`` request as it does against GPU
    and CPU, so the reservation belongs in the division. The cluster here is
    deliberately device-rich and memory-poor, skewed so a bound that divided by
    the devices alone would answer 8 and size the pool for actors the nodes
    cannot admit.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"GPU": 8.0, "CPU": 64.0, "memory": float(16 * 1024**3)})

    resources = WorkerResources(scan_batch_size=2, num_gpus=1, num_cpus=2, memory_bytes=8 * 1024**3)
    assert _placeable_actors(resources, owed_fragments=99, group_name="image") == 2


def test_the_actor_ceiling_never_falls_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cluster momentarily reporting nothing free must not cap the pool at zero actors."""
    monkeypatch.setattr("ray.available_resources", dict)

    assert _placeable_actors(WorkerResources(scan_batch_size=2, num_gpus=1), owed_fragments=4, group_name="g") == 1


def test_the_derived_actor_ceiling_is_logged_with_its_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capped pool looks like a too-small cluster in every other log line.

    The inputs are logged beside the result because the number alone cannot say
    which of the two bounds produced it, and the two call for opposite responses.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"GPU": 2.0})
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO")
    try:
        _placeable_actors(WorkerResources(scan_batch_size=2, num_gpus=0.5), owed_fragments=99, group_name="image")
    finally:
        logger.remove(sink)

    logged = "".join(lines)
    assert "capping the actor pool at 4" in logged
    assert "2 free GPU" in logged
    assert "99 fragment(s) are owed" in logged


def test_the_reserved_heap_is_logged_in_the_unit_it_was_configured_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """The memory bound reports GiB, the unit ``memory_gb`` is written in.

    A byte count shares this line with GPU and CPU device counts, so rendered
    the way they are it arrives as an unlabelled ``1.7e+10`` - which the one
    line that exists to name the actionable bound cannot afford. Both terms are
    pinned together because the clause is what an operator reads, and either
    number alone could survive the divisor and the renderer disagreeing.
    """
    monkeypatch.setattr("ray.available_resources", lambda: {"memory": float(16 * 1024**3)})
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO")
    try:
        _placeable_actors(
            WorkerResources(scan_batch_size=2, memory_bytes=8 * 1024**3), owed_fragments=99, group_name="image"
        )
    finally:
        logger.remove(sink)

    assert "advertise 16 GiB free memory against 8 GiB per actor" in "".join(lines)


@pytest.mark.usefixtures("ray_local")
def test_staging_writes_every_owed_row_partitioned_by_fragment(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staged rows ARE the phase's output, so publication must find all of them.

    Partitioning is asserted through the directory layout rather than a column:
    ``fragment_id`` becomes a Hive key at write time, so a reader takes it from
    the path and a run that wrote it into the file instead would not be readable
    the way publication reads it.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)

    _stage(uri, _image_fill(_ConstantImageEmbedder), workspace)

    partitions = sorted(path.name for path in pathlib.Path(workspace.results_uri).iterdir() if path.is_dir())
    assert partitions == [f"{FRAGMENT_ID_COLUMN}=0", f"{FRAGMENT_ID_COLUMN}=1"]
    assert _staged_clips(workspace) == ["c0", "c1", "c2", "c3"]


@pytest.mark.usefixtures("ray_local")
def test_ray_plans_the_job_checkpoint_for_this_dag(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ray disables checkpointing with only a warning, so the absence of that warning is the check.

    An unsupported DAG shape - anything but read-to-write - leaves the run looking
    configured for recovery while retaining none of it. Nothing else in the run
    fails, so this warning is the only signal, and the resume tests below would
    still pass by simply recomputing everything.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _stage(uri, _image_fill(_ConstantImageEmbedder), _workspace(tmp_path))

    assert not [warning for warning in caught if "doesn't support checkpointing" in str(warning.message)]


@pytest.mark.usefixtures("ray_local")
def test_a_resume_hands_the_embedder_only_the_rows_no_attempt_committed(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the whole phase: a second attempt must not re-pay for staged rows.

    Everything lives in ONE fragment, and the first attempt is narrowed by ROW
    rather than by fragment. That is what makes the assertion discriminating: the
    checkpoint has to be keyed per row for the second attempt to see exactly the
    two rows the first never took on. Were it keyed per fragment - or on any
    column shared across a fragment's rows - the first attempt would have
    completed the only fragment there is and the second would embed NOTHING,
    which is a different set rather than a smaller one.

    Narrowing by row also avoids the nondeterminism of the realistic trigger:
    failing an attempt partway leaves which write bundles committed up to
    scheduling.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=4)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    first_observed = tmp_path / "observed-first"
    second_observed = tmp_path / "observed-second"
    narrowed = attrs.evolve(
        _image_fill(_ObservingImageEmbedder, observed_dir=str(first_observed)),
        applicability_filter="clip_id IN ('c0', 'c1')",
    )

    _stage(uri, narrowed, workspace)
    _stage(uri, _image_fill(_ObservingImageEmbedder, observed_dir=str(second_observed)), workspace)

    assert _observed_clips(first_observed) == {"c0", "c1"}
    assert _observed_clips(second_observed) == {"c2", "c3"}
    # The staged set is the union rather than either half, which is what
    # publication later reads.
    assert _staged_clips(workspace) == ["c0", "c1", "c2", "c3"]


@pytest.mark.usefixtures("ray_local")
def test_a_fully_checkpointed_phase_hands_the_embedder_nothing(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-run of a completed phase must cost no compute, not merely no duplicate rows.

    Distinct from the partial case: there the filter narrows the work, here it
    removes all of it, and only an empty observation directory separates "skipped
    every row" from "recomputed them and overwrote identical output".
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    second_observed = tmp_path / "observed-second"

    _stage(uri, _image_fill(_ConstantImageEmbedder), workspace)
    _stage(uri, _image_fill(_ObservingImageEmbedder, observed_dir=str(second_observed)), workspace)

    assert _observed_clips(second_observed) == set()


def test_an_empty_scan_plan_is_refused_before_any_actor_starts(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """A modality owing nothing must be skipped by the caller, not executed as an empty phase.

    Executed would report a phase that ran over no rows, which is
    indistinguishable in the log from one whose filter was wrong.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = _image_fill(_ConstantImageEmbedder)

    with pytest.raises(ValueError, match="refusing to execute an empty scan plan"):
        _stage(uri, fill, _workspace(tmp_path), max_fragments=0)


def test_a_projection_without_the_key_is_refused(make_clips_table: ClipsTableFactory) -> None:
    """Every staged row carries ``clip_id`` and publication joins on it, so it must be projected.

    Refused on the driver: reaching an actor pool first would load model weights
    onto a GPU before discovering an input fault that fails every task alike.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = attrs.evolve(_image_fill(_ConstantImageEmbedder), source_columns=("clip_uri",))

    with pytest.raises(ValueError, match="must be projected"):
        _stage(uri, fill, _workspace(pathlib.Path(uri).parent))


def test_a_filter_that_does_not_resolve_is_refused(make_clips_table: ClipsTableFactory) -> None:
    """The precondition check is called directly, because nothing reaches it with a bad filter.

    A caller plans before it stages, and ``plan_embedding_scan`` pushes the same
    predicate into a per-fragment ``count_rows``, so an unresolvable filter
    already surfaces there - as an ``OSError`` from Lance, on any table holding a
    fragment. Driving this arm through ``_stage`` would therefore pin the
    planner's failure while appearing to pin this one.

    The check is kept whole regardless: its other arm, the projection, is NOT
    validated by planning, and one ``explain_plan`` covers both.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = _image_fill(_ConstantImageEmbedder)

    with pytest.raises(ValueError, match="does not resolve against"):
        _check_fill_preconditions(lance.dataset(uri), fill, row_filter="no_such_column IS NOT NULL")
