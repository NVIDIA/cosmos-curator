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

"""Per-fragment fill worker contract, and the run-level skip policy built on it.

``_FragmentWorker`` is the whole write path: it scans one fragment's pending rows,
calls the embedder, attaches ``clip_id`` positionally, and rewrites only the
group's columns. Most of that needs no Ray, so the worker is driven directly with
stub embedders - which keeps those tests in the CPU suite and makes the failure
modes they pin (a rewritten complete fragment, a mis-sized embedder output, a
vector leaking to the driver) observable in isolation.

The tests that pin the RUN-level outcomes - which fragments commit when one of
them fails, which failure stops the run and what type the caller then sees, how a
run that owed no work is told apart from one whose every fragment failed, and
the ``max_fragments`` cap - go through ``fill_embedding_group`` and therefore
through a real actor pool, because turning the workers' payloads into a commit or
an exception is precisely the behaviour under test. They drop the pixi
``py_executable`` so the actors run in the interpreter running the test. The
exception is the driver's precondition check, which is tested without a pool
because never reaching one is the property being pinned.

The image group stands in for all three modalities: its per-row validity mask and
its single vector make it the smallest group that can express both a filled and a
failed row.
"""

import json
import pathlib
from typing import Any

import attrs
import lance
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.schemas import (
    IMAGE_COLUMN_GROUP,
    IMAGE_DIM,
    IMAGE_GROUP_SCHEMA,
    image_columns_batch,
)
from cosmos_curator.next.recipes.embeddings.columns import pending_filter
from cosmos_curator.next.recipes.embeddings.config import Modality
from cosmos_curator.next.recipes.embeddings.fill import (
    _ERROR_KEY,
    _FATAL_KEY,
    _RESULT_COLUMN,
    _RESULT_SCHEMA,
    FillContractError,
    _FragmentWorker,
    fill_embedding_group,
)
from cosmos_curator.next.recipes.embeddings.modalities import (
    _IMAGE_APPLICABILITY_FILTER,
    ModalityFill,
    WorkerResources,
)

from .conftest import CLIPS_BASE_SCHEMA, ClipsTableFactory, add_group_columns

_IMAGE_SOURCE_COLUMNS = ("clip_id", "clip_uri")
_MODEL_ID = "stub-image-model"

# Keys a written fragment's payload must carry for the driver to commit it: the
# fragment's identity, its serialized updated metadata, the field ids the write
# rebound, and the two counters the run's totals are summed from.
_WRITTEN_PAYLOAD_KEYS = {"fragment_id", "metadata_json", "modified_field_ids", "rows", "filled"}


class _ConstantImageEmbedder:
    """Emit one always-valid image group row per input row (the happy-path contract)."""

    def __init__(self, *, model_id: str = _MODEL_ID) -> None:
        """Bind the provenance value every row will carry."""
        self._model_id = model_id

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return a constant vector for every row of ``batch``, in input order."""
        rows = batch.num_rows
        return image_columns_batch(
            rows,
            np.ones((rows, IMAGE_DIM), dtype=np.float32),
            self._model_id,
            np.ones(rows, dtype=np.bool_),
        )


class _UriEncodingImageEmbedder:
    """Encode each row's own ``clip_uri`` into its vector, making the key pairing observable.

    The happy-path stub emits one constant vector, so it cannot distinguish a correct
    positional splice from a shuffled one. This one makes the row a vector came from
    recoverable from the vector itself. It requires the fixture's ``clipN.bin`` URI
    shape.
    """

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one vector per row whose first component is that row's ``clipN.bin`` index."""
        uris = batch.column("clip_uri").to_pylist()
        matrix = np.zeros((len(uris), IMAGE_DIM), dtype=np.float32)
        matrix[:, 0] = [float(uri.removeprefix("clip").removesuffix(".bin")) for uri in uris]
        return image_columns_batch(len(uris), matrix, _MODEL_ID, np.ones(len(uris), dtype=np.bool_))


def _valid_image_group_rows(rows: int) -> pa.Table:
    """Return ``rows`` well-formed image group rows, for stubs that then break one thing."""
    return image_columns_batch(
        rows,
        np.ones((rows, IMAGE_DIM), dtype=np.float32),
        _MODEL_ID,
        np.ones(rows, dtype=np.bool_),
    )


class _PartiallyFailingUriEncodingImageEmbedder:
    """Encode each row's ``clip_uri`` into its vector, but emit an all-NULL group for one clip.

    The only stub that can leave a fragment WRITTEN yet incomplete. Raising (as
    ``_PoisonedImageEmbedder`` does) skips the whole fragment, so nothing is written
    and there is no partially filled column to top up; an all-NULL group row is the
    per-row failure the leg contract actually defines.
    """

    def __init__(self, *, null_uri: str) -> None:
        """Bind the ``clip_uri`` whose group row is emitted all-NULL."""
        self._null_uri = null_uri

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one row per input row, the bound clip's row all-NULL."""
        uris = batch.column("clip_uri").to_pylist()
        matrix = np.zeros((len(uris), IMAGE_DIM), dtype=np.float32)
        matrix[:, 0] = [float(uri.removeprefix("clip").removesuffix(".bin")) for uri in uris]
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
        """Drop the provenance column, so the result cannot cast to the stored schema."""
        return _valid_image_group_rows(batch.num_rows).drop_columns([IMAGE_COLUMN_GROUP.provenance_columns[0]])


class _WrongDtypeImageEmbedder:
    """Return the group's columns with the vector typed as text."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Replace the vector with strings, which have no cast to a fixed-size list."""
        rows = _valid_image_group_rows(batch.num_rows)
        vector = IMAGE_COLUMN_GROUP.primary_vector
        return rows.set_column(rows.schema.get_field_index(vector), vector, pa.array(["not a vector"] * batch.num_rows))


class _LateRowDroppingImageEmbedder:
    """Honour the cardinality contract on the first batch, then violate it on the next.

    The worker embeds only the first scanned batch on its own stack; Lance pulls the
    rest. This stub puts the violation on the far side of that boundary.
    """

    def __init__(self) -> None:
        """Start this actor's batch counter."""
        self._calls = 0

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return one row per input row for the first batch, one row fewer after that."""
        self._calls += 1
        rows = batch.num_rows if self._calls == 1 else max(batch.num_rows - 1, 0)
        return image_columns_batch(
            rows,
            np.ones((rows, IMAGE_DIM), dtype=np.float32),
            _MODEL_ID,
            np.ones(rows, dtype=np.bool_),
        )


class _PoisonedImageEmbedder:
    """Fail outright on the batch holding ``poison_uri``, modelling a transient per-fragment fault.

    Stands in for the environment faults a fragment is skipped for (an unreadable
    object, a decode library crash) rather than for a broken embedder: it computes
    every other batch correctly, so exactly one fragment is lost.
    """

    def __init__(self, *, poison_uri: str) -> None:
        """Bind the ``clip_uri`` whose presence in a batch makes this embedder raise."""
        self._poison_uri = poison_uri

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Raise if the batch holds the poisoned clip, otherwise embed it normally."""
        if self._poison_uri in batch.column("clip_uri").to_pylist():
            msg = f"synthetic read failure for {self._poison_uri}"
            raise RuntimeError(msg)
        rows = batch.num_rows
        return image_columns_batch(
            rows,
            np.ones((rows, IMAGE_DIM), dtype=np.float32),
            _MODEL_ID,
            np.ones(rows, dtype=np.bool_),
        )


class _UnreadableImageEmbedder:
    """Fail on every batch, modelling a systemic fault that hits every fragment alike.

    The counterpart of ``_PoisonedImageEmbedder``: the fault is not a property of one
    fragment's data, so every fragment is skipped and the run writes nothing.
    """

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Raise for any batch, whatever it holds."""
        msg = f"synthetic storage outage for {batch.num_rows} row(s)"
        raise RuntimeError(msg)


def _worker(
    uri: str,
    embedder_cls: type,
    *,
    row_filter: str | None,
    scan_batch_size: int = 8,
    **embedder_kwargs: str,
) -> _FragmentWorker:
    """Build a worker pinned to the table's current version for the image group."""
    return _FragmentWorker(
        uri=uri,
        group=IMAGE_COLUMN_GROUP,
        read_version=int(lance.dataset(uri).version),
        storage_options=None,
        source_columns=_IMAGE_SOURCE_COLUMNS,
        row_filter=row_filter,
        scan_batch_size=scan_batch_size,
        embedder_cls=embedder_cls,
        embedder_kwargs=embedder_kwargs,
    )


def _work_items(*fragment_ids: int) -> pa.Table:
    """Build the one-fragment-per-row work batch Ray Data would hand the worker."""
    return pa.table({"fragment_id": pa.array(list(fragment_ids), pa.int64())})


def _payloads(result: pa.Table) -> list[dict[str, Any]]:
    """Decode a worker's result rows into the JSON payloads the driver collects."""
    return [json.loads(value) for value in result.column(_RESULT_COLUMN).to_pylist()]


def _image_fill(embedder_cls: type, **embedder_kwargs: str) -> ModalityFill:
    """Build an image-group fill spec around a stub embedder, sized for the local cluster."""
    return ModalityFill(
        modality=Modality.IMAGE,
        group=IMAGE_COLUMN_GROUP,
        source_columns=_IMAGE_SOURCE_COLUMNS,
        applicability_filter=_IMAGE_APPLICABILITY_FILTER,
        expected_provenance=None,
        embedder_cls=embedder_cls,
        embedder_kwargs=embedder_kwargs,
        resources=WorkerResources(scan_batch_size=8, num_cpus=1),
        weights_name=None,
    )


def _run_in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the pixi ``py_executable`` so fill actors run in the interpreter running the test."""
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.fill.ray_data_gpu_runtime_env",
        lambda _env_name: ray_data_gpu_runtime_env(""),
    )


def _data_files(uri: str) -> set[str]:
    """Return the names of every file in the dataset's data directory."""
    return {path.name for path in (pathlib.Path(uri) / "data").iterdir()}


def _image_vectors_by_clip(uri: str) -> dict[str, list[float] | None]:
    """Return each clip's stored image vector (``None`` where the group is still pending)."""
    result = lance.dataset(uri).to_table(columns=["clip_id", "embedding_image"]).to_pydict()
    return dict(zip(result["clip_id"], result["embedding_image"], strict=True))


def _filled_image_table(tmp_path: pathlib.Path, rows: int) -> str:
    """Write a single-fragment clips table whose image group is already complete."""
    table = pa.table(
        {
            "clip_id": pa.array([f"c{i}" for i in range(rows)], pa.string()),
            "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
            "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
            "clip_uri": pa.array([f"clip{i}.bin" for i in range(rows)], pa.large_string()),
            "action_data_uri": pa.array([f"act{i}.bin" for i in range(rows)], pa.large_string()),
            "source_dataset": pa.array(["ds_under_test"] * rows, pa.string()),
            "embedding_image": pa.array([[1.0] * IMAGE_DIM] * rows, pa.list_(pa.float32(), IMAGE_DIM)),
            "embedding_image_model_id": pa.array([_MODEL_ID] * rows, pa.string()),
        },
        schema=pa.schema([*list(CLIPS_BASE_SCHEMA), *list(IMAGE_GROUP_SCHEMA)]),
    )
    uri = str(tmp_path / "clips.lance")
    lance.write_dataset(table, uri, data_storage_version="2.2")
    return uri


def test_result_schema_carries_no_column_data(make_clips_table: ClipsTableFactory) -> None:
    """A worker may return one JSON accounting string per fragment, never a vector.

    This is the mechanism that keeps driver memory O(touched fragments) instead of
    O(rows): a lone string column structurally cannot be widened into a full-corpus
    vector gather without changing its type, so the schema is the enforcement point.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, _ConstantImageEmbedder, row_filter=None)(_work_items(0))

    assert result.schema == _RESULT_SCHEMA
    assert result.schema.types == [pa.large_string()]
    assert set(_payloads(result)[0]) == _WRITTEN_PAYLOAD_KEYS


def test_worker_reports_the_rows_it_scanned_and_filled(make_clips_table: ClipsTableFactory) -> None:
    """One fragment's result row carries that fragment's own scanned and filled counts.

    The run's ``selected`` is summed from these, never counted upfront, so a
    fragment that scanned 2 rows must say 2 rather than the table's row count.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, _ConstantImageEmbedder, row_filter=None)(_work_items(0))

    assert result.num_rows == 1
    payload = _payloads(result)[0]
    assert payload["fragment_id"] == 0
    assert payload["rows"] == 2
    assert payload["filled"] == 2
    assert payload["modified_field_ids"]  # the image field ids were rebound


def test_fragment_with_nothing_pending_is_not_rewritten(tmp_path: pathlib.Path) -> None:
    """A fully embedded fragment yields no result row and no new data file.

    The worker probes the scanner before calling ``update_columns``, because
    ``update_columns`` writes a column file the moment it is invoked. Without the
    probe an unchanged rerun would rewrite every fragment and commit a version that
    changes nothing.
    """
    uri = _filled_image_table(tmp_path, rows=3)
    before = _data_files(uri)

    worker = _worker(
        uri, _ConstantImageEmbedder, row_filter=pending_filter(IMAGE_COLUMN_GROUP, _IMAGE_APPLICABILITY_FILTER)
    )
    result = worker(_work_items(0))

    assert result.num_rows == 0
    assert _data_files(uri) == before


def test_each_vector_is_keyed_to_the_clip_it_was_computed_from(make_clips_table: ClipsTableFactory) -> None:
    """The positionally attached ``clip_id`` is the key of the row that produced the vector.

    The row-count check catches an embedder that drops rows, but not one that
    REORDERS them - and a reordering would pair every vector with the wrong clip
    while all counts still balance. Here each vector carries its own row's
    ``clip_uri`` index, so the splice is checkable. The input order is scrambled so
    that an implementation which sorts the scanned rows before embedding, yet
    attaches the unsorted key column, also fails.

    The table supplies only the stored column types the update batch is cast to; the
    rows embedded here are hand-built.
    """
    uri = make_clips_table(rows=1)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    worker = _worker(uri, _UriEncodingImageEmbedder, row_filter=None)
    source = pa.table(
        {
            "clip_id": pa.array(["c5", "c3", "c4"], pa.string()),
            "clip_uri": pa.array(["clip5.bin", "clip3.bin", "clip4.bin"], pa.large_string()),
        }
    )

    update = worker._embed(source, worker._stored_update_schema())

    vectors = [vector[0] for vector in update.column("embedding_image").to_pylist()]
    assert dict(zip(update.column("clip_id").to_pylist(), vectors, strict=True)) == {"c5": 5.0, "c3": 3.0, "c4": 4.0}


def test_an_embedder_that_changes_the_row_count_is_reported_as_fatal_not_skipped(
    make_clips_table: ClipsTableFactory,
) -> None:
    """An embedder returning fewer rows than it was given is fatal, not one more skipped fragment.

    ``clip_id`` is attached positionally, so a dropped row would shift every later
    vector onto the wrong clip. Unlike an environment fault the violation is a
    property of the run's own inputs, so every other fragment would fail the same
    way - reporting it through the skip payload would leave an empty group behind a
    success exit code. The distinction has to live in the payload because Ray Data
    does not carry the original exception object to the driver.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, _RowDroppingImageEmbedder, row_filter=None)(_work_items(0))

    payload = _payloads(result)[0]
    assert _ERROR_KEY not in payload  # not offered to the driver as a skippable fragment
    assert "one output row per input row" in payload[_FATAL_KEY]


@pytest.mark.parametrize("embedder", [_ColumnDroppingImageEmbedder, _WrongDtypeImageEmbedder])
def test_an_embedder_that_returns_the_wrong_schema_is_reported_as_fatal_not_skipped(
    make_clips_table: ClipsTableFactory,
    embedder: type,
) -> None:
    """A missing column and an uncastable dtype are the same fault as a wrong row count: a broken embedder.

    Both surface only when the output is cast to the stored schema, and both would
    otherwise reach the broad handler and be reported as one more skipped fragment
    - which fails every fragment identically and reduces a precise "the embedder
    returned the wrong schema" to a generic outage message.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, embedder, row_filter=None)(_work_items(0))

    payload = _payloads(result)[0]
    assert _ERROR_KEY not in payload  # not offered to the driver as a skippable fragment
    assert "do not match the stored schema" in payload[_FATAL_KEY]


def test_a_contract_violation_after_the_first_batch_is_still_reported_as_fatal(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Where in the update stream a violation happens must not decide fatal versus skipped.

    Only the first scanned batch is embedded on the worker's own stack; Lance pulls
    the rest through the Arrow C data interface, which carries the message but not the
    exception object, so a violation raised there re-emerges as a bare ``RuntimeError``.
    Reported as an ordinary fault it would become one more skipped fragment - and since
    a broken embedder fails every fragment identically, the run would commit nothing and
    still exit successfully. Only a fragment larger than one scan batch can reach this.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, _LateRowDroppingImageEmbedder, row_filter=None, scan_batch_size=1)(_work_items(0))

    payload = _payloads(result)[0]
    assert _ERROR_KEY not in payload  # not offered to the driver as a skippable fragment
    assert "one output row per input row" in payload[_FATAL_KEY]


def test_a_fragment_absent_at_the_pinned_version_is_reported_as_fatal(make_clips_table: ClipsTableFactory) -> None:
    """A planned fragment that does not exist at the pinned version is fatal, not a skip.

    Silently skipping it would let the run report success while leaving part of the
    planned work undone.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = _worker(uri, _ConstantImageEmbedder, row_filter=None)(_work_items(99))

    assert "not present" in _payloads(result)[0][_FATAL_KEY]


def test_a_failing_fragment_is_reported_as_skipped_instead_of_raising(make_clips_table: ClipsTableFactory) -> None:
    """An environment fault costs one fragment: the worker reports it and writes nothing.

    The counterpart to the cardinality rejection above. A fault the next run could
    get past must not discard the other fragments' completed work, so it comes back
    as an error payload the driver partitions out rather than as an exception.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    before = _data_files(uri)

    result = _worker(uri, _PoisonedImageEmbedder, row_filter=None, poison_uri="clip0.bin")(_work_items(0))

    payload = _payloads(result)[0]
    assert payload["fragment_id"] == 0
    assert _ERROR_KEY in payload
    assert "metadata_json" not in payload  # nothing to commit for a fragment that was never written
    assert _data_files(uri) == before


def test_a_skipped_fragment_is_announced_at_warning_level(
    make_clips_table: ClipsTableFactory,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The skip is logged as a WARNING, so silently unfinished work is visible to the operator.

    A skip leaves rows pending with a success exit code, so the log line is the
    only place the loss is reported at the time it happens.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    _worker(uri, _PoisonedImageEmbedder, row_filter=None, poison_uri="clip0.bin")(_work_items(0))

    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert any("SKIPPED" in message for message in warnings)


def test_a_projection_that_does_not_resolve_stops_the_run_instead_of_skipping_every_fragment(
    make_clips_table: ClipsTableFactory,
) -> None:
    """A source column absent from the table stops the run before any worker starts.

    The projection is a property of the run, not of a fragment, so left to the
    workers it would fail on every one of them - and because a failing fragment is
    skipped, the modality would report a successful run that embedded nothing. No
    actor pool is created here, which is the other half of the point: a run that
    cannot work must not first acquire GPUs.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = attrs.evolve(_image_fill(_ConstantImageEmbedder), source_columns=("clip_id", "no_such_column"))

    with pytest.raises(FillContractError, match="does not resolve"):
        fill_embedding_group(lance.dataset(uri), fill, storage_options=None)


def test_a_projection_without_the_join_key_is_refused(make_clips_table: ClipsTableFactory) -> None:
    """A fill whose projection omits ``clip_id`` is refused rather than skipped per fragment.

    The embedder never reads the key, but the fill attaches it positionally and
    ``update_columns`` joins on it, so a projection without it cannot write anything.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = attrs.evolve(_image_fill(_ConstantImageEmbedder), source_columns=("clip_uri",))

    with pytest.raises(FillContractError, match="must be projected"):
        fill_embedding_group(lance.dataset(uri), fill, storage_options=None)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_surviving_fragments_commit_when_one_fragment_fails(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One failed fragment costs only its own rows; every other fragment still commits.

    The point of the skip policy: a run that has already spent hours of GPU time on
    the other fragments must keep that work rather than discarding it for one flaky
    fragment.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill = _image_fill(_PoisonedImageEmbedder, poison_uri="clip2.bin")

    result = fill_embedding_group(lance.dataset(uri), fill, storage_options=None)

    assert result.committed_version is not None
    assert result.selected == 2  # only the surviving fragment's rows were counted
    vectors = _image_vectors_by_clip(uri)
    assert vectors["c0"] is not None
    assert vectors["c1"] is not None


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_skipped_fragments_rows_stay_pending_and_an_ordinary_rerun_fills_them(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A skipped fragment needs no retry machinery: its rows stay NULL and the next run re-selects them.

    This is what makes skipping safe rather than lossy - the pending predicate is
    already the retry mechanism, so an intra-run retry becomes an inter-run one.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill_embedding_group(
        lance.dataset(uri), _image_fill(_PoisonedImageEmbedder, poison_uri="clip2.bin"), storage_options=None
    )
    assert _image_vectors_by_clip(uri)["c2"] is None

    fill_embedding_group(lance.dataset(uri), _image_fill(_ConstantImageEmbedder), storage_options=None)

    assert all(vector is not None for vector in _image_vectors_by_clip(uri).values())


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_topping_up_one_pending_row_leaves_its_filled_siblings_byte_identical(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run fills only the NULL row of an already-written fragment.

    The sibling test to the skipped-fragment rerun above, one level finer: there the
    fragment was never written, so the rerun rewrote it wholesale and no existing
    vector was at risk. Here the first run wrote the fragment with one row failed, so
    the second run's update lands inside a column that already holds vectors it must
    not disturb.

    Starting from a FILLED column is what makes the assertion mean anything: over an
    all-NULL column "preserved" and "clobbered" are the same observation. The two runs
    also emit vectors neither could mistake for the other's - the first encodes each
    row's own URI in one component, the second is all ones - so a row the second run
    wrongly rewrote is visible in the stored bytes.
    """
    _run_in_process(monkeypatch)
    # One fragment, so the top-up is genuinely sub-fragment rather than a whole
    # untouched fragment being rewritten.
    uri = make_clips_table(rows=4, rows_per_file=4)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    fill_embedding_group(
        lance.dataset(uri),
        _image_fill(_PartiallyFailingUriEncodingImageEmbedder, null_uri="clip2.bin"),
        storage_options=None,
    )
    after_first = _image_vectors_by_clip(uri)
    assert after_first["c2"] is None
    assert all(after_first[clip] is not None for clip in ("c0", "c1", "c3"))

    fill_embedding_group(lance.dataset(uri), _image_fill(_ConstantImageEmbedder), storage_options=None)

    after_second = _image_vectors_by_clip(uri)
    assert after_second["c2"] is not None
    preserved = ("c0", "c1", "c3")
    assert {clip: after_second[clip] for clip in preserved} == {clip: after_first[clip] for clip in preserved}


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_contract_violation_reaches_the_caller_as_a_value_error_and_commits_nothing(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A violated fill contract surfaces as ``FillContractError``, and the table is left untouched.

    The type is the point: the CLI turns a ``ValueError`` into one logged line and a
    non-zero exit, and a worker's own raise cannot deliver one, because Ray Data
    replaces it with a wrapper that is no longer an instance of its own cause. This
    is the run-level counterpart of the worker's fatal payload above.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    version = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match="one output row per input row") as raised:
        fill_embedding_group(lance.dataset(uri), _image_fill(_RowDroppingImageEmbedder), storage_options=None)

    assert isinstance(raised.value, FillContractError)
    assert int(lance.dataset(uri).version) == version


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_partly_skipped_run_reports_the_skip_count_to_its_caller(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The number of fragments lost to a failure reaches the caller, not only the log.

    Every row count a skipped fragment would have contributed is zero by
    construction, so without this field a caller cannot tell a run that lost work
    from one that had none to do - which is what lets the total-outage refusal below
    be expressed at all.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = fill_embedding_group(
        lance.dataset(uri), _image_fill(_PoisonedImageEmbedder, poison_uri="clip2.bin"), storage_options=None
    )

    assert result.skipped_fragments == 1
    # The driver's own aggregate warning, distinct from each worker's per-fragment
    # one: it is the only line reporting the run-wide loss the commit hid.
    warnings = [record["message"] for record in loguru_records if record["level"].name == "WARNING"]
    assert any("see the per-fragment warnings" in message for message in warnings)


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_run_whose_every_fragment_failed_raises_instead_of_reporting_success(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fault that skips every fragment is an outage, not the sum of tolerable skips.

    The skip trades one fragment's work for the rest of the run's; with nothing
    written there is no rest, the group is untouched, and the counts are exactly
    those of a run that owed no work. Reported as a success it would make a total
    outage indistinguishable from an already-complete table.
    """
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    version = int(lance.dataset(uri).version)

    # ValueError is the outer type on purpose: it is what the CLI catches to turn a
    # failure into one logged line and a non-zero exit, so both halves are contract.
    with pytest.raises(ValueError, match="no fragment was written while 2 of the 2") as raised:
        fill_embedding_group(lance.dataset(uri), _image_fill(_UnreadableImageEmbedder), storage_options=None)

    assert isinstance(raised.value, FillContractError)
    # The message must not claim the failed fragments held pending rows: a fragment
    # can fault before its scan could report whether it had any.
    assert "not knowable" in str(raised.value)
    assert int(lance.dataset(uri).version) == version  # nothing was committed


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_a_rerun_over_a_fully_embedded_table_succeeds_and_commits_nothing(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Writing nothing because there was nothing to write stays a success.

    The other half of the refusal above, and the reason the discriminator is the
    skip count rather than "did this run commit": an idempotent re-run legitimately
    commits nothing, so a check keyed on the commit alone would fail every table
    that is already complete.
    """
    _run_in_process(monkeypatch)
    uri = _filled_image_table(tmp_path, rows=3)
    version = int(lance.dataset(uri).version)

    result = fill_embedding_group(lance.dataset(uri), _image_fill(_ConstantImageEmbedder), storage_options=None)

    assert result.skipped_fragments == 0
    assert result.selected == 0
    assert result.committed_version is None
    assert int(lance.dataset(uri).version) == version


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_max_fragments_bounds_the_run_to_the_first_fragments(
    make_clips_table: ClipsTableFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cap is fragment-granular, so a capped run fills whole fragments and leaves the rest pending."""
    _run_in_process(monkeypatch)
    uri = make_clips_table(rows=6, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)

    result = fill_embedding_group(
        lance.dataset(uri), _image_fill(_ConstantImageEmbedder), storage_options=None, max_fragments=1
    )

    assert result.selected == 2  # one fragment of two rows
    filled = {clip for clip, vector in _image_vectors_by_clip(uri).items() if vector is not None}
    assert filled == {"c0", "c1"}
