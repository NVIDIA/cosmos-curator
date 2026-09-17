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

"""Publishing staged rows into the clips table, and every refusal that precedes it.

The staged partitions are written HERE rather than by running the staging phase,
so a publication test costs no embedder and no actor pool. That is a restatement
of data, not of logic: the rows are built through ``staged_schema`` and
``image_columns_batch``, the producer's own declarations, so a change to either
lands in these fixtures rather than being papered over by a private copy. The one
test that must not do this - the seam where staging's output meets publication's
reader - runs the real staging phase end to end.

The refusals divide into two layers, and the split is deliberate. A partition
fault is raised inside a Ray task and must survive an Arrow reader boundary and
then Ray's own wrapper, so those tests drive ``prepare_fragment`` directly, where
the exception still has its type. The set-level refusals are driver-side and are
driven through their own helpers.

Every test asserting a refusal also asserts the table's version did not move.
"Refused" and "refused before any commit" are different claims, and only the
second one is worth having.
"""

import json
import pathlib
import re

import lance
import numpy as np
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import ray
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
from cosmos_curator.next.recipes.embeddings.fill import stage_embedding_group
from cosmos_curator.next.recipes.embeddings.modalities import (
    _IMAGE_APPLICABILITY_FILTER,
    ModalityFill,
    ModalityResult,
    WorkerResources,
)
from cosmos_curator.next.recipes.embeddings.publication import (
    TRANSACTION_KIND,
    EmbeddingPublicationError,
    _fragment_metadata,
    _require_expected_fragments,
    prepare_fragment,
    publish_embedding_group,
    staged_distinct_values,
)
from cosmos_curator.next.recipes.embeddings.staged import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    staged_schema,
)
from cosmos_curator.next.recipes.embeddings.workspace import EmbeddingWorkspace
from cosmos_curator.next.utils.identity import canonical_digest

from .conftest import ClipsTableFactory, add_group_columns

_IMAGE_SOURCE_COLUMNS = ("clip_id", "clip_uri")
_MODEL_ID = "stub-image-model"
_OLD_MODEL_ID = "stub-image-model-v0"
_STAGED_SCHEMA = staged_schema(IMAGE_COLUMN_GROUP)
_PROVENANCE_COLUMN = IMAGE_COLUMN_GROUP.provenance_columns[0]


class _ConstantImageEmbedder:
    """Emit one always-valid image group row per input row, for the end-to-end seam test."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        """Return a constant vector for every row of ``batch``, in input order."""
        rows = batch.num_rows
        return image_columns_batch(
            rows, np.ones((rows, IMAGE_DIM), dtype=np.float32), _MODEL_ID, np.ones(rows, dtype=np.bool_)
        )


def _image_fill(*, model_id: str = _MODEL_ID, applicability_filter: str = _IMAGE_APPLICABILITY_FILTER) -> ModalityFill:
    """Build an image-group fill spec whose result identity names ``model_id``."""
    return ModalityFill(
        modality=Modality.IMAGE,
        group=IMAGE_COLUMN_GROUP,
        source_columns=_IMAGE_SOURCE_COLUMNS,
        applicability_filter=applicability_filter,
        result_identity={"embedding_image_model_id": model_id},
        embedder_cls=_ConstantImageEmbedder,
        embedder_kwargs={},
        resources=WorkerResources(scan_batch_size=2, num_cpus=1),
        weights_name=None,
    )


def _workspace(tmp_path: pathlib.Path, *, name: str = "namespace") -> EmbeddingWorkspace:
    """Return a local workspace, built directly rather than resolved.

    Resolution is pinned in ``test_workspace.py``; publication needs only a root
    whose ``results`` holds the fragment partitions.
    """
    return EmbeddingWorkspace(
        root_uri=str(tmp_path / "staging" / "image" / name),
        filesystem=pafs.LocalFileSystem(),
    )


def _staged_rows(clip_ids: list[str], *, fragment_id: int, model_id: str, null_clips: tuple[str, ...] = ()) -> pa.Table:
    """Build one partition's staged rows, each vector encoding its own clip's index.

    Encoding the clip into the vector is what makes a mispaired publication
    detectable: a constant vector would be indistinguishable from one joined onto
    the wrong row.
    """
    rows = len(clip_ids)
    matrix = np.zeros((rows, IMAGE_DIM), dtype=np.float32)
    matrix[:, 0] = [_clip_index(clip_id) for clip_id in clip_ids]
    valid = np.array([clip_id not in null_clips for clip_id in clip_ids], dtype=np.bool_)
    group = image_columns_batch(rows, matrix, model_id, valid)
    staged = pa.table(
        {
            "clip_id": pa.array(clip_ids, pa.string()),
            FRAGMENT_ID_COLUMN: pa.array([fragment_id] * rows, pa.int64()),
            CHECKPOINT_ID_COLUMN: pa.array(range(rows), pa.uint64()),
            **{name: group.column(name) for name in group.schema.names},
        }
    )
    return staged.select(_STAGED_SCHEMA.names).cast(_STAGED_SCHEMA)


def _clip_index(clip_id: str) -> float:
    """Return the ``N`` of a fixture ``cN`` clip id."""
    return float(clip_id.removeprefix("c"))


def _write_partition(
    workspace: EmbeddingWorkspace,
    fragment_id: int,
    clip_ids: list[str],
    *,
    model_id: str = _MODEL_ID,
    null_clips: tuple[str, ...] = (),
) -> pathlib.Path:
    """Stage one fragment's rows as the Hive-partitioned Parquet publication reads."""
    partition = pathlib.Path(workspace.results_uri) / f"{FRAGMENT_ID_COLUMN}={fragment_id}"
    partition.mkdir(parents=True, exist_ok=True)
    rows = _staged_rows(clip_ids, fragment_id=fragment_id, model_id=model_id, null_clips=null_clips)
    pq.write_table(rows.drop_columns([FRAGMENT_ID_COLUMN]), partition / "0.parquet")
    return partition


def _publish(uri: str, fill: ModalityFill, workspace: EmbeddingWorkspace) -> object:
    """Plan against the table's current version and publish the staged rows."""
    dataset = lance.dataset(uri)
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    plan = plan_embedding_scan(dataset, row_filter=row_filter, max_fragments=None)
    return publish_embedding_group(dataset, fill, workspace=workspace, plan=plan, storage_options=None)


def _first_components(uri: str) -> dict[str, float | None]:
    """Return each clip's vector first component, or ``None`` where the vector is NULL."""
    table = lance.dataset(uri).to_table(columns=["clip_id", IMAGE_COLUMN_GROUP.primary_vector])
    vectors = table.column(IMAGE_COLUMN_GROUP.primary_vector).to_pylist()
    return {
        clip_id: None if vector is None else vector[0]
        for clip_id, vector in zip(table.column("clip_id").to_pylist(), vectors, strict=True)
    }


def _prepared(uri: str, workspace: EmbeddingWorkspace, fragment_id: int) -> dict[str, object]:
    """Prepare one fragment against the table's current version, without committing."""
    return prepare_fragment(lance.dataset(uri), workspace, fragment_id, group=IMAGE_COLUMN_GROUP)


@pytest.fixture
def seeded_table(make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path) -> str:
    """Return a 4-row, 2-fragment table whose image group is fully published at the OLD model.

    Seeded through the production publication path rather than by hand, so the
    "stale" rows under test are exactly the rows a previous run would have left.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path, name="seed")
    _write_partition(workspace, 0, ["c0", "c1"], model_id=_OLD_MODEL_ID)
    _write_partition(workspace, 1, ["c2", "c3"], model_id=_OLD_MODEL_ID)
    _publish(uri, _image_fill(model_id=_OLD_MODEL_ID), workspace)
    return uri


@pytest.mark.usefixtures("ray_local")
def test_a_stale_row_is_replaced_in_place_and_an_omitted_row_keeps_its_value(
    seeded_table: str, tmp_path: pathlib.Path
) -> None:
    """Publishing part of a fragment must not blank the rest of it.

    ``c0`` and ``c1`` share fragment 0, and only ``c0`` is staged. The left-outer
    join in ``update_columns`` is what keeps ``c1``; a right-outer or replacing
    write would leave it NULL, and nothing else in the run would say so.

    The staged vector for ``c0`` is deliberately different from the seeded one -
    it encodes the clip while the seed encodes it too, so the model id separates
    them - so "replaced" is observable rather than assumed.
    """
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0"])

    _publish(seeded_table, _image_fill(applicability_filter="clip_id = 'c0'"), workspace)

    provenance = lance.dataset(seeded_table).to_table(columns=["clip_id", _PROVENANCE_COLUMN])
    assert dict(
        zip(provenance.column("clip_id").to_pylist(), provenance.column(_PROVENANCE_COLUMN).to_pylist(), strict=True)
    ) == {"c0": _MODEL_ID, "c1": _OLD_MODEL_ID, "c2": _OLD_MODEL_ID, "c3": _OLD_MODEL_ID}


@pytest.mark.usefixtures("ray_local")
def test_a_row_whose_recompute_failed_loses_the_vector_it_held(seeded_table: str, tmp_path: pathlib.Path) -> None:
    """A stale vector must not outlive the attempt that failed to replace it.

    Distinct from publishing NULLs over an empty group: here the row already holds
    a vector from the older computation, and the staged row carries none. Keeping
    the old vector would leave the corpus holding two geometries under one
    identity once the rest of the group converges; keeping the old PROVENANCE
    would additionally hide the row from the next run's selection. Both columns
    going NULL is what keeps the row owed.

    ``c1`` succeeded in the same partition, which is what rules out publishing the
    failure by dropping its row: that leaves ``c0`` holding the older vector while
    ``c1`` takes its new one, and the run's counts balance either way.
    """
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"], null_clips=("c0",))

    _publish(seeded_table, _image_fill(applicability_filter="clip_id IN ('c0', 'c1')"), workspace)

    table = lance.dataset(seeded_table).to_table(
        columns=["clip_id", IMAGE_COLUMN_GROUP.primary_vector, _PROVENANCE_COLUMN], filter="clip_id = 'c0'"
    )
    assert table.column(IMAGE_COLUMN_GROUP.primary_vector).to_pylist() == [None]
    assert table.column(_PROVENANCE_COLUMN).to_pylist() == [None]
    assert _first_components(seeded_table)["c1"] == _clip_index("c1")


@pytest.mark.usefixtures("ray_local")
def test_a_column_outside_the_group_is_unchanged_by_the_publication(seeded_table: str, tmp_path: pathlib.Path) -> None:
    """``update_columns`` rewrites a fragment's metadata, so untouched columns are pinned.

    A prepared ``FragmentMetadata`` enumerates every one of its fragment's column
    files. Committing one that enumerated only the file this write produced would
    silently drop the source columns, and the group assertions above would all
    still pass.
    """
    before = lance.dataset(seeded_table).to_table(columns=["clip_id", "clip_uri", "task_name"])
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])

    _publish(seeded_table, _image_fill(applicability_filter="clip_id IN ('c0', 'c1')"), workspace)

    assert lance.dataset(seeded_table).to_table(columns=["clip_id", "clip_uri", "task_name"]) == before


@pytest.mark.usefixtures("ray_local")
def test_an_all_null_staged_group_publishes_as_null(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """A modality whose every row failed to compute still commits, with the failures recorded.

    Refusing here would strand the run: the staged rows are terminal, so the next
    attempt would recompute rows that already failed. The NULL is the durable
    verdict, and ``filled`` is what tells the operator apart from a run that
    published nothing.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"], null_clips=("c0", "c1"))

    result = _publish(uri, _image_fill(), workspace)

    assert _first_components(uri) == {"c0": None, "c1": None}
    assert (result.selected, result.filled, result.failed) == (2, 0, 2)


@pytest.mark.usefixtures("ray_local")
def test_the_result_counts_the_rows_the_commit_made_canonical(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """These tallies are the run's only source of counts, so a mixed batch is pinned.

    Counted as the batches stream into ``update_columns`` rather than from the
    staged files, which is the only point at which the two can disagree: a row
    staged but not consumed would inflate a file-based count.
    """
    uri = make_clips_table(rows=4, rows_per_file=4)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1", "c2", "c3"], null_clips=("c2",))

    result = _publish(uri, _image_fill(), workspace)

    assert (result.selected, result.filled, result.failed) == (4, 3, 1)
    assert result.committed_version == int(lance.dataset(uri).version)


@pytest.mark.usefixtures("ray_local")
def test_the_commit_carries_the_group_and_result_identity_it_published(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """The commit's properties are read back by the ambiguous-commit check, so they are a contract.

    The identity is digested rather than spelled out because a transaction
    property is one string; equality is all a later reader needs, and it is what
    attributes a version to the model that produced it.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])
    fill = _image_fill()

    result = _publish(uri, fill, workspace)

    properties = lance.dataset(uri).read_transaction(result.committed_version).transaction_properties
    assert properties["kind"] == TRANSACTION_KIND
    assert properties["group"] == IMAGE_COLUMN_GROUP.name
    assert properties["result_identity"] == canonical_digest(fill.result_identity)


@pytest.mark.usefixtures("ray_local")
def test_a_planned_fragment_with_no_staged_partition_refuses_before_any_commit(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """Half a group published is worse than none, so one absent partition stops the commit.

    Driven through ``publish_embedding_group`` rather than ``prepare_fragment``
    because what is under test is the driver abandoning the OTHER fragment: the
    version assertion is what distinguishes refusing from refusing in time.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    version_before = int(lance.dataset(uri).version)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])

    with pytest.raises(EmbeddingPublicationError, match="did not write this fragment"):
        _publish(uri, _image_fill(), workspace)

    assert int(lance.dataset(uri).version) == version_before
    assert _first_components(uri) == {"c0": None, "c1": None, "c2": None, "c3": None}


def test_a_partition_holding_no_file_refuses(make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path) -> None:
    """A directory with no file is a distinct fault from an absent one, and reads as success.

    An interrupted attempt can leave the partition directory behind, so the
    directory's existence cannot stand in for its contents.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    (pathlib.Path(workspace.results_uri) / f"{FRAGMENT_ID_COLUMN}=0").mkdir(parents=True)

    with pytest.raises(EmbeddingPublicationError, match="holds no file"):
        _prepared(uri, workspace, 0)


def test_an_unreadable_staged_file_refuses_rather_than_retrying(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """A file with no Parquet footer must not be left as the OSError family Ray retries.

    The mirror of ``staged_distinct_values``, which tolerates this exact fault. By
    publication the staging phase has reported success, so an unreadable file is
    corruption rather than a shard a later run reclaims - and retrying it would
    re-read a file that cannot become readable.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    partition = _write_partition(workspace, 0, ["c0", "c1"])
    (partition / "0.parquet").write_bytes(b"not parquet")

    with pytest.raises(EmbeddingPublicationError, match="cannot be read"):
        _prepared(uri, workspace, 0)


def test_a_staged_vector_of_the_wrong_width_refuses(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """The staged schema drops the vector width, so the publication cast is the only width gate.

    Widening was required for Parquet to round-trip a NULL vector. That trade is
    only sound if the width is re-asserted here, which is what this pins: a
    wrong-width vector must not reach the table's ``fixed_size_list`` column.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    partition = _write_partition(workspace, 0, ["c0", "c1"])
    narrow = _staged_rows(["c0", "c1"], fragment_id=0, model_id=_MODEL_ID).drop_columns(
        [FRAGMENT_ID_COLUMN, IMAGE_COLUMN_GROUP.primary_vector]
    )
    truncated = pa.array([[0.0] * (IMAGE_DIM - 1)] * 2, pa.list_(pa.float32()))
    pq.write_table(
        narrow.append_column(pa.field(IMAGE_COLUMN_GROUP.primary_vector, truncated.type), truncated),
        partition / "0.parquet",
    )

    with pytest.raises(EmbeddingPublicationError, match="do not cast to the stored schema"):
        _prepared(uri, workspace, 0)


def test_a_prepared_payload_carries_counts_rather_than_rows(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """Driver memory is O(fragments) only while the payload stays scalar, so its keys are pinned.

    A vector reaching the driver is not a slow publication, it is one that cannot
    run at corpus scale at all, and it would show up first as an extra payload
    key. Asserting the exact key set is therefore the cheap form of the memory
    bound, and it is also the contract ``_fragment_metadata`` decodes.
    """
    uri = make_clips_table(rows=3, rows_per_file=3)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1", "c2"], null_clips=("c1",))

    payload = _prepared(uri, workspace, 0)

    assert set(payload) == {FRAGMENT_ID_COLUMN, "metadata_json", "modified_field_ids", "rows", "filled"}
    assert (payload["rows"], payload["filled"]) == (3, 2)


def test_two_payloads_claiming_one_fragment_refuse(make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path) -> None:
    """Which of two metadata versions wins would otherwise be decided by iteration order.

    Reachable through a retried prepare task whose first attempt's payload was
    already collected, so the commit must refuse rather than pick.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])
    payload = _prepared(uri, workspace, 0)

    with pytest.raises(EmbeddingPublicationError, match="two metadata versions of one fragment"):
        _fragment_metadata([payload, payload], group_name=IMAGE_COLUMN_GROUP.name)


def test_a_payload_whose_metadata_names_another_fragment_refuses(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """The envelope id places the metadata in the commit, so the two disagreeing is fatal.

    Committing on the envelope's word would attach one fragment's column files to
    another fragment's row set - rows silently paired with the wrong vectors,
    which no later check in this recipe looks for.
    """
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 1, ["c2", "c3"])
    mislabelled = {**_prepared(uri, workspace, 1), FRAGMENT_ID_COLUMN: 0}

    with pytest.raises(EmbeddingPublicationError, match="carries metadata for fragment 1"):
        _fragment_metadata([mislabelled], group_name=IMAGE_COLUMN_GROUP.name)


@pytest.mark.parametrize(
    ("prepared", "expected_message"),
    [
        pytest.param((0,), "missing [1]", id="fewer-than-planned-leaves-rows-owed"),
        pytest.param((0, 1, 2), "unexpected [2]", id="more-than-planned-commits-foreign-rows"),
    ],
)
def test_a_prepared_set_differing_from_the_plan_refuses(prepared: tuple[int, ...], expected_message: str) -> None:
    """Both directions are faults, and ``update_columns`` would complain about neither.

    Fewer fragments than planned reports success over rows still owed; more means
    a foreign attempt's rows are in the commit. Each direction is named in the
    message because the remedies differ.
    """
    with pytest.raises(EmbeddingPublicationError, match=re.escape(expected_message)):
        _require_expected_fragments(prepared, (0, 1), uri="s3://bucket/clips.lance")


@pytest.mark.usefixtures("ray_local")
@pytest.mark.parametrize(
    ("staged_clips", "applicability", "expected_message"),
    [
        pytest.param(["c0"], "clip_id IN ('c0', 'c1')", "staged 1 of 2 owed row(s)", id="short-of-the-plan"),
        pytest.param(["c0", "c1"], "clip_id = 'c0'", "staged 2 of 1 owed row(s)", id="beyond-the-plan"),
    ],
)
def test_a_partition_that_does_not_tally_with_its_plan_refuses_and_leaves_the_group_untouched(
    seeded_table: str, tmp_path: pathlib.Path, staged_clips: list[str], applicability: str, expected_message: str
) -> None:
    """A readable partition holding the wrong number of rows is a fault, not a subset to publish.

    The prepared fragment SET matches the plan in both cases, so the set check
    passes and only the row tally separates them. Staging short is the silent
    one: ``update_columns`` leaves the unstaged rows holding what they had, so
    without this the commit reports success over rows still carrying the old
    model's vectors.

    The table is asserted unchanged rather than only the raise, because a
    refusal that fired after a partial commit would satisfy the raise alone.
    """
    before = lance.dataset(seeded_table).to_table(columns=["clip_id", _PROVENANCE_COLUMN])
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, staged_clips)

    with pytest.raises(EmbeddingPublicationError, match=re.escape(expected_message)):
        _publish(seeded_table, _image_fill(applicability_filter=applicability), workspace)

    assert lance.dataset(seeded_table).to_table(columns=["clip_id", _PROVENANCE_COLUMN]) == before


@pytest.mark.usefixtures("ray_local")
def test_a_lost_commit_response_is_recognized_through_the_attempt_marker(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit that landed and then lost its response must not be reported as a failure.

    The stub commits for real and then raises, which is the one shape that
    separates "the commit failed" from "the answer was lost": both arrive as the
    same exception. Told it failed, an operator would republish a group that is
    already canonical.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])
    real_commit = lance.LanceDataset.commit

    def commit_then_lose_the_response(*args: object, **kwargs: object) -> None:
        real_commit(*args, **kwargs)
        msg = "connection reset after the manifest was written"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", commit_then_lose_the_response)
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        result = _publish(uri, _image_fill(), workspace)
    finally:
        logger.remove(sink)

    assert result.committed_version == int(lance.dataset(uri).version)
    assert "the error was a lost response" in "".join(lines)


@pytest.mark.usefixtures("ray_local")
def test_a_commit_that_did_not_land_stays_an_error(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control for the marker: no marker in history means the group is untouched.

    Sits on the other side of the same ambiguity as the lost-response test. Were
    the check to accept an unmarked history, that test would still pass while this
    one turns a genuine failure into a reported success.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    version_before = int(lance.dataset(uri).version)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])
    monkeypatch.setattr(
        lance.LanceDataset,
        "commit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest rejected")),
    )

    with pytest.raises(ValueError, match="did not land and the group is untouched"):
        _publish(uri, _image_fill(), workspace)

    assert int(lance.dataset(uri).version) == version_before


@pytest.mark.usefixtures("ray_local")
def test_rows_staged_by_the_real_phase_publish_unchanged(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one test that must not hand-write its partitions: it pins the seam between the halves.

    Staging owns the layout, the Hive key, and the widened vector; publication
    reads all three back. Every other test here restates that layout, so only
    this one would fail if the two halves drifted apart.
    """
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.fill.ray_data_gpu_runtime_env",
        lambda _env_name: ray_data_gpu_runtime_env(""),
    )
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    fill = _image_fill()
    dataset = lance.dataset(uri)
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    plan = plan_embedding_scan(dataset, row_filter=row_filter, max_fragments=None)
    stage_embedding_group(dataset, fill, workspace=workspace, plan=plan, rows_per_file=2, storage_options=None)

    result = _publish(uri, fill, workspace)

    assert (result.selected, result.filled) == (4, 4)
    assert _first_components(uri) == {"c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 1.0}


@pytest.mark.usefixtures("ray_local")
def test_capped_waves_each_publish_their_own_fragment_and_together_finish_the_table(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_fragments`` bounds one attempt's footprint without stranding the rest of the table.

    Two waves over one workspace, which is the arrangement that could go wrong:
    the second wave's plan is computed from the rows the FIRST wave published, so
    a planner that re-offered the finished prefix would stage fragment 0 again and
    the table would never converge. Distinct commits are asserted because a wave
    that published nothing would leave the vectors of its fragment NULL while the
    row counts still balanced.
    """
    monkeypatch.setattr(
        "cosmos_curator.next.recipes.embeddings.fill.ray_data_gpu_runtime_env",
        lambda _env_name: ray_data_gpu_runtime_env(""),
    )
    uri = make_clips_table(rows=4, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    fill = _image_fill()

    waves = [_stage_and_publish(uri, fill, workspace, max_fragments=1) for _ in range(2)]

    assert [wave.selected for wave in waves] == [2, 2]
    assert waves[0].committed_version != waves[1].committed_version
    assert _first_components(uri) == {"c0": 1.0, "c1": 1.0, "c2": 1.0, "c3": 1.0}


def _stage_and_publish(
    uri: str, fill: ModalityFill, workspace: EmbeddingWorkspace, *, max_fragments: int | None
) -> ModalityResult:
    """Run one capped attempt end to end: plan, stage, then publish that plan."""
    dataset = lance.dataset(uri)
    row_filter = pending_filter(fill.group, fill.applicability_filter, fill.result_identity)
    plan = plan_embedding_scan(dataset, row_filter=row_filter, max_fragments=max_fragments)
    stage_embedding_group(dataset, fill, workspace=workspace, plan=plan, rows_per_file=2, storage_options=None)
    return publish_embedding_group(dataset, fill, workspace=workspace, plan=plan, storage_options=None)


def test_staged_distinct_values_reads_one_column_across_every_partition(tmp_path: pathlib.Path) -> None:
    """The action basis is resolved from this, so it must see the whole namespace.

    A value staged by any fragment is a value the run committed to, so reading one
    partition would let a second basis through.
    """
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"], model_id=_OLD_MODEL_ID)
    _write_partition(workspace, 1, ["c2"], model_id=_MODEL_ID)

    values = staged_distinct_values(workspace, _PROVENANCE_COLUMN, max_values=8)

    assert sorted(values) == sorted({_OLD_MODEL_ID, _MODEL_ID})


def test_staged_distinct_values_stops_once_the_caller_has_enough(tmp_path: pathlib.Path) -> None:
    """Two values already refuse, so the read must not walk the namespace to count them all.

    The bound is the caller's ceiling on ambiguity rather than a performance knob,
    and it is what keeps this pre-phase read from becoming a corpus scan.
    """
    workspace = _workspace(tmp_path)
    for fragment_id in range(4):
        _write_partition(workspace, fragment_id, [f"c{fragment_id}"], model_id=f"model-{fragment_id}")

    assert len(staged_distinct_values(workspace, _PROVENANCE_COLUMN, max_values=2)) == 2


def test_staged_distinct_values_of_an_unstaged_namespace_is_empty(tmp_path: pathlib.Path) -> None:
    """A first attempt has no namespace, and that is the ordinary case rather than a fault.

    Raising here would make the action leg unable to start: its basis resolution
    asks this question before anything has been staged.
    """
    assert staged_distinct_values(_workspace(tmp_path), _PROVENANCE_COLUMN, max_values=8) == ()


def test_staged_distinct_values_skips_a_torn_shard_and_warns(tmp_path: pathlib.Path) -> None:
    """A crashed attempt's last file has no footer, and the readable partitions still answer.

    Tolerated here and refused at publication, deliberately: this read runs BEFORE
    the phase whose checkpoint reclaims the shard, so failing would block the
    recovery the read exists to enable. The warning is asserted because a silent
    skip and a skip an operator can see are different things.
    """
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0"], model_id=_MODEL_ID)
    torn = _write_partition(workspace, 1, ["c1"], model_id=_OLD_MODEL_ID)
    (torn / "0.parquet").write_bytes(b"not parquet")
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING")
    try:
        values = staged_distinct_values(workspace, _PROVENANCE_COLUMN, max_values=8)
    finally:
        logger.remove(sink)

    assert values == (_MODEL_ID,)
    assert "skipping unreadable staged file" in "".join(lines)


@pytest.mark.usefixtures("ray_local")
def test_every_planned_fragment_gets_its_own_block_so_the_phase_is_as_wide_as_its_work(
    seeded_table: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prepare fan-out states its block count instead of letting Ray autodetect one.

    Task count follows block count, and ``from_items`` autodetects
    ``min(len(items), max(200, 2 * avail_cpus))`` (measured at Ray 2.58.0), a
    cluster-sized figure with no reason to equal the fragment count. A fixture
    cannot afford enough fragments to make the two differ, so what is pinned is
    that the count is passed at all and equals the planned fragments.
    """
    requested: list[int | None] = []
    real_from_items = ray.data.from_items

    def _record(items: list[dict[str, object]], **kwargs: object) -> ray.data.Dataset:
        override = kwargs.get("override_num_blocks")
        requested.append(None if override is None else int(str(override)))
        return real_from_items(items, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ray.data, "from_items", _record)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])
    _write_partition(workspace, 1, ["c2", "c3"])

    _publish(seeded_table, _image_fill(), workspace)

    assert requested == [2]


def test_a_prepared_payloads_metadata_survives_its_json_round_trip(
    make_clips_table: ClipsTableFactory, tmp_path: pathlib.Path
) -> None:
    """The metadata crosses a Ray boundary as a string, and a lossy trip commits the wrong files.

    Embedded as the exact string Lance produces rather than as a nested object, so
    the decoded metadata must still enumerate the fragment's files. Asserting the
    decode alone would pass on a metadata carrying no files at all.
    """
    uri = make_clips_table(rows=2, rows_per_file=2)
    add_group_columns(uri, IMAGE_GROUP_SCHEMA)
    workspace = _workspace(tmp_path)
    _write_partition(workspace, 0, ["c0", "c1"])

    payload = _prepared(uri, workspace, 0)

    decoded = _fragment_metadata([payload], group_name=IMAGE_COLUMN_GROUP.name)[0]
    assert int(decoded.id) == 0
    # The group's file joins the source columns' files rather than replacing them,
    # which is what leaves every untouched column bound to the file it had.
    assert len(decoded.files) > 1
    assert json.loads(str(payload["metadata_json"]))
