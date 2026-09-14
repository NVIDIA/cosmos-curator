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

"""Driver and action-basis tests for the direct-``clips.lance`` embedding pipeline.

Two layers, both runnable in the default CPU job:

- Pure driver-logic tests (no Ray): the fixed modality cascade, the action-only
  weight-staging no-op, the missing-table failure, the schema-widening commit, the
  no-work short circuit, the basis-staleness refusal, the total-outage guard, and
  the PCA sampling helpers. None of them starts a worker.
- One actor-pool dispatch test (``ray_local``): it starts a real actor but embeds
  nothing. It exists because the sampling-helper tests feed hand-built batches and
  therefore cannot catch a mis-scheduled extractor. It drops the pixi
  ``py_executable`` so the actor runs in the interpreter running the test rather
  than the container's environment.

Filling real vectors end to end needs the container's pixi environment, so that is
covered by the GPU job rather than here. The action leg is Mecka-only:
applicability is a non-empty ``action_data_uri``, and each artifact is a
self-describing ACT2 ``.bin`` decoded with no dataset registry, so a monkeypatched
dataset name resolves identically in the driver (which writes the bins) and in Ray
workers (which only decode them). The ``source_dataset`` column is retained in the
base schema but no longer read by the leg.
"""

import pathlib
from collections.abc import Callable, Iterator, Sequence

import lance
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.next.embeddings.action.embedder import DualWristMotionReadConfig
from cosmos_curator.next.embeddings.action.pca import action_pca_root_uri
from cosmos_curator.next.embeddings.action.wrist_motion import DESCRIPTOR_DIM, DESCRIPTOR_VERSION
from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    ACTION_DIM,
    ACTION_GROUP_SCHEMA,
    descriptor_batch,
)
from cosmos_curator.next.recipes.embeddings import action_pca, embed
from cosmos_curator.next.recipes.embeddings.action_pca import (
    _collect_pca_sample,
    _extract_candidates,
    _ranked_distinct_uris,
    _sample_rank,
    check_action_outcome,
    resolve_action_pca,
)
from cosmos_curator.next.recipes.embeddings.config import Modality
from cosmos_curator.next.recipes.embeddings.embed import (
    _enabled_groups,
    _generic_fills,
    _stage_weights,
    run_embedding_pipeline,
)
from cosmos_curator.next.recipes.embeddings.modalities import ModalityResult
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

from .conftest import (
    CLIPS_BASE_SCHEMA,
    ClipsTableFactory,
    EmbeddingsConfigFactory,
    add_group_columns,
)


def _write_clips_table(
    directory: pathlib.Path,
    clip_ids: Sequence[str],
    action_uris: Sequence[str],
    datasets: Sequence[str],
    *,
    name: str = "clips.lance",
) -> str:
    """Write a base-schema ``clips.lance`` with no clip media and return its URI.

    Reuses the recipe suite's ``CLIPS_BASE_SCHEMA`` (a faithful subset of the
    producer's ``OUTCOME_SCHEMA``) so the widen-then-append behaviour a test drives
    matches the real narrow-append path. ``clip_uri`` is always empty (these tests
    embed only the action modality, which reads ``action_data_uri``).
    """
    rows = len(clip_ids)
    table = pa.table(
        {
            "clip_id": pa.array(list(clip_ids), pa.string()),
            "task_name": pa.array(["pick up the block"] * rows, pa.string()),
            "subtask_name": pa.array(["grasp"] * rows, pa.string()),
            "clip_uri": pa.array([""] * rows, pa.large_string()),
            "action_data_uri": pa.array(list(action_uris), pa.large_string()),
            "source_dataset": pa.array(list(datasets), pa.string()),
        },
        schema=CLIPS_BASE_SCHEMA,
    )
    uri = str(directory / name)
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri


def _write_embedded_action_table(directory: pathlib.Path, *, descriptor_version: str, fingerprint: str) -> str:
    """Write a one-row ``clips.lance`` whose action group is already embedded.

    The provenance the row carries is what the driver reads back to decide which
    basis a later run must reuse, so it is written as real column data rather than
    added empty and patched afterwards.
    """
    schema = pa.schema([*list(CLIPS_BASE_SCHEMA), *list(ACTION_GROUP_SCHEMA)])
    table = pa.table(
        {
            "clip_id": pa.array(["c0"], pa.string()),
            "task_name": pa.array(["pick up the block"], pa.string()),
            "subtask_name": pa.array(["grasp"], pa.string()),
            "clip_uri": pa.array([""], pa.large_string()),
            "action_data_uri": pa.array(["act0.bin"], pa.large_string()),
            "source_dataset": pa.array(["ds_under_test"], pa.string()),
            "embedding_action": pa.array([[0.0] * ACTION_DIM], ACTION_GROUP_SCHEMA.field("embedding_action").type),
            "embedding_action_descriptor_version": pa.array([descriptor_version], pa.string()),
            "embedding_action_pca_fingerprint": pa.array([fingerprint], pa.string()),
        },
        schema=schema,
    )
    uri = str(directory / "clips.lance")
    lance.write_dataset(table, uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)
    return uri


def test_modalities_run_in_a_fixed_order_whatever_order_they_were_configured_in(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """The cascade is text -> image -> action; the configured tuple only records what is enabled.

    Text and image share one loop over their specs, so the loop's order IS the
    execution order. A run that embedded image before text would still be correct
    but would no longer match the order the summary and the logs report.
    """
    config = make_embeddings_config(modalities=["image", "text"])

    assert [fill.modality for fill in _generic_fills(config)] == [Modality.TEXT, Modality.IMAGE]


def test_the_action_group_is_widened_even_though_action_has_no_fill_spec(
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Action's columns are added by name, because its spec cannot exist before a basis is bound.

    Resolving the basis reads the action group's own provenance columns, so those
    columns must already exist when the resolve runs. Deriving the widening set
    from the built specs alone would leave action's columns absent and the resolve
    reading a column the planner cannot find.
    """
    config = make_embeddings_config(modalities=["action"])

    assert _enabled_groups(_generic_fills(config), config) == [ACTION_COLUMN_GROUP]


def test_stage_weights_is_a_noop_for_action_only(
    monkeypatch: pytest.MonkeyPatch, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Staging weights for an action-only run invokes no downloader (nothing to stage)."""
    called = False

    def _fail(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(embed, "download_models", _fail)
    config = make_embeddings_config(modalities=["action"])

    _stage_weights(_generic_fills(config), config)
    assert not called


def test_missing_clips_table_fails(tmp_path: pathlib.Path, make_embeddings_config: EmbeddingsConfigFactory) -> None:
    """A clips URI resolving to no table fails before any modality runs."""
    config = make_embeddings_config(clips_lance_uri=str(tmp_path / "absent.lance"), modalities=["action"])

    with pytest.raises(ValueError, match="not found"):
        run_embedding_pipeline(config)


def test_action_run_widens_schema_with_only_the_action_group(
    tmp_path: pathlib.Path, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Enabling only action adds ``embedding_action*`` and no text/image columns.

    Schema widening happens before any fill, so this holds even with zero
    applicable rows: a consumer must test column PRESENCE, not merely NULLs, and a
    single-modality run must not create another modality's columns.
    """
    clips_uri = _write_clips_table(tmp_path, ["c0", "c1"], ["", ""], ["not_dexterous", "not_dexterous"])

    run_embedding_pipeline(make_embeddings_config(clips_lance_uri=clips_uri, modalities=["action"]))

    names = set(lance.dataset(clips_uri).schema.names)
    assert "embedding_action" in names
    assert not any(name.startswith("embedding_text") for name in names)
    assert not any(name.startswith("embedding_image") for name in names)


def test_driver_starts_no_worker_when_the_action_leg_has_nothing_to_embed(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """No clip carries an action artifact, so the driver never starts an actor pool.

    The action leg can prove there is nothing to embed on the driver - the group is
    empty and no row has an ``action_data_uri``, so there is neither a basis to load
    nor a population to fit one from. It must report that as a skip rather than pay
    for an actor pool (and rather than failing a table that simply holds no action
    data).
    """

    def _fail(*_args: object, **_kwargs: object) -> object:
        msg = "no fill worker may start when the modality has nothing to embed"
        raise AssertionError(msg)

    monkeypatch.setattr(embed, "fill_embedding_group", _fail)
    clips_uri = _write_clips_table(tmp_path, ["c0", "c1"], ["", ""], ["not_dexterous", "not_dexterous"])

    result = run_embedding_pipeline(make_embeddings_config(clips_lance_uri=clips_uri, modalities=["action"]))

    assert result.action is None
    # No fit ran, so no basis was persisted: the content-addressed PCA root
    # directory was never created.
    assert not pathlib.Path(action_pca_root_uri(clips_uri)).exists()


def test_a_run_whose_only_commit_widened_the_schema_reports_that_version(
    tmp_path: pathlib.Path, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """The version reported is the table's, so a widening-only run names its own commit.

    The widening is a real version that a consumer opening the table next will
    read, and a first run of a modality with nothing pending makes it the run's
    only commit. Reporting nothing there would describe a table that had moved as
    unchanged, pointing whoever reads the summary at a version that no longer
    holds the columns the run added.
    """
    clips_uri = _write_clips_table(tmp_path, ["c0", "c1"], ["", ""], ["not_dexterous", "not_dexterous"])

    result = run_embedding_pipeline(make_embeddings_config(clips_lance_uri=clips_uri, modalities=["action"]))

    # Pinned together: no modality committed, yet the run's version is the live
    # table's -- which is the whole distinction the report has to carry.
    assert [modality.committed_version for modality in result.modalities] == [None]
    assert result.ending_version == lance.dataset(clips_uri).version


def test_a_rerun_that_adds_no_column_reports_no_version_of_its_own(
    tmp_path: pathlib.Path, make_embeddings_config: EmbeddingsConfigFactory
) -> None:
    """Only a commit this run made may be reported as this run's ending version.

    The second run re-opens a table whose columns already exist, so
    ``ensure_embedding_columns`` adds nothing and commits nothing -- but the
    version the open dataset carries is the FIRST run's widening. Crediting this
    run with it would report a commit that never happened.
    """
    clips_uri = _write_clips_table(tmp_path, ["c0", "c1"], ["", ""], ["not_dexterous", "not_dexterous"])
    config = make_embeddings_config(clips_lance_uri=clips_uri, modalities=["action"])
    first = run_embedding_pipeline(config)

    second = run_embedding_pipeline(config)

    assert first.ending_version is not None  # else the re-run's None proves nothing
    assert second.ending_version is None


def test_a_group_embedded_under_another_descriptor_version_refuses_to_load_its_basis(
    tmp_path: pathlib.Path,
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Descriptors whose meaning has changed cannot be re-projected onto the old basis.

    The refusal happens before the artifact is read, so the operator is told to
    reset the group rather than seeing the mismatch surface as a raw
    archive-validation error from inside the load.
    """
    clips_uri = _write_embedded_action_table(
        tmp_path, descriptor_version=f"not-{DESCRIPTOR_VERSION}", fingerprint="deadbeef"
    )
    config = make_embeddings_config(clips_lance_uri=clips_uri, modalities=["action"])

    with pytest.raises(ValueError, match="descriptor version"):
        resolve_action_pca(lance.dataset(clips_uri), config, root_uri=str(tmp_path / "pca"))


def test_an_action_fill_that_produced_nothing_at_all_fails_the_run(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Every visited row failing with an empty group is an outage, not a partial loss.

    Nothing about the table distinguishes "every artifact was unreadable" from "the
    export or the profile is misconfigured", so the run fails loudly instead of
    exiting zero over a group it left entirely NULL.
    """
    uri = make_clips_table(rows=2)
    add_group_columns(uri, ACTION_GROUP_SCHEMA)
    outage = ModalityResult(modality=Modality.ACTION, selected=2, filled=0, skipped_fragments=0, committed_version=None)

    with pytest.raises(ValueError, match="failed on all"):
        check_action_outcome(outage, lance.dataset(uri))


def test_a_run_whose_rows_all_failed_is_tolerated_when_the_group_already_holds_embeddings(
    tmp_path: pathlib.Path,
) -> None:
    """A top-up that embedded none of its new rows is retryable, not an outage.

    The group holding earlier vectors proves the artifacts and the basis can be
    read at all, so the loss is per-row and the ordinary pending filter re-selects
    those rows next run.
    """
    clips_uri = _write_embedded_action_table(tmp_path, descriptor_version=DESCRIPTOR_VERSION, fingerprint="deadbeef")
    all_failed = ModalityResult(
        modality=Modality.ACTION, selected=2, filled=0, skipped_fragments=0, committed_version=None
    )

    check_action_outcome(all_failed, lance.dataset(clips_uri))


@pytest.mark.integration
@pytest.mark.usefixtures("ray_local")
def test_pca_candidate_extraction_dispatches_through_a_real_actor_pool(
    monkeypatch: pytest.MonkeyPatch,
    make_mecka_bin: Callable[..., str],
    tmp_path: pathlib.Path,
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """Candidate extraction returns one descriptor row per URI when dispatched to a real actor pool.

    The pool is the whole point of the round trip: Ray Data refuses a bound method
    under ``ActorPoolStrategy``, so only a real dispatch proves the extractor is
    scheduled as a class whose constructor receives the read config. Every other
    action test feeds the sampling helpers hand-built batches and would still pass
    with the extractor mis-scheduled.

    Passing an empty pixi env name drops the ``py_executable`` indirection so the
    actor runs in the interpreter running the test; the production factory is still
    the one that builds the runtime env.
    """
    monkeypatch.setattr(action_pca, "ray_data_gpu_runtime_env", lambda _env_name: ray_data_gpu_runtime_env(""))
    uris = [make_mecka_bin(tmp_path / f"act{i}.bin", seed=i) for i in range(2)]
    config = make_embeddings_config(
        clips_lance_uri=str(tmp_path / "clips.lance"),
        modalities=["action"],
        # One URI per batch is what forces more than one dispatch; the read width
        # has to come down with it, because a width wider than the scan batch is
        # rejected rather than clamped.
        action={"batch_size": 1, "read_concurrency": 1},
    )

    batches = list(_extract_candidates(config, uris))

    assert sum(batch.num_rows for batch in batches) == len(uris)


def test_pca_candidate_extraction_carries_the_configured_read_width_to_the_extractor(
    monkeypatch: pytest.MonkeyPatch,
    make_embeddings_config: EmbeddingsConfigFactory,
) -> None:
    """A non-default ``read_concurrency`` reaches the extractor the PCA pass constructs.

    Pinned separately from the fill leg's equivalent because the two call sites each
    build their own ``DualWristMotionReadConfig``: a width threaded into one says
    nothing about the other. The value differs from that config's default on purpose,
    so dropping the threading here turns this red rather than silently reverting the
    PCA pass to serial reads.
    """
    captured: list[DualWristMotionReadConfig] = []

    class _RecordingDataset:
        """Records the extractor's constructor kwargs instead of scheduling a pool."""

        def map_batches(
            self,
            _fn: object,
            *,
            fn_constructor_kwargs: dict[str, DualWristMotionReadConfig],
            **_kwargs: object,
        ) -> "_RecordingDataset":
            captured.append(fn_constructor_kwargs["config"])
            return self

        def iter_batches(self, **_kwargs: object) -> Iterator[pa.Table]:
            return iter(())

    monkeypatch.setattr(action_pca.ray.data, "from_items", lambda _items: _RecordingDataset())
    config = make_embeddings_config(
        modalities=["action"],
        action={"batch_size": 16, "read_concurrency": 7},
    )

    list(_extract_candidates(config, ["s3://bucket/a.bin"]))

    assert [read_config.read_concurrency for read_config in captured] == [7]


def test_sample_rank_is_a_deterministic_function_of_the_uri() -> None:
    """``_sample_rank`` is a stable per-URI hash: same URI same rank, distinct URIs distinct ranks.

    The whole reproducibility argument rests on the rank being a pure function of
    the URI (not of scan order or clock), so a re-fit selects the identical spans.
    """
    assert _sample_rank("s3://b/a.bin") == _sample_rank("s3://b/a.bin")
    assert _sample_rank("s3://b/a.bin") != _sample_rank("s3://b/b.bin")


def test_ranked_distinct_uris_keeps_the_smallest_rank_set_regardless_of_batching() -> None:
    """The candidate set is the ``limit`` smallest-rank distinct URIs, batching-invariant.

    This is the first of the two bounded passes: it decides which artifacts the fit
    is even allowed to read. Because the rank is a pure function of the URI, the
    chosen set - and therefore the fitted basis - cannot change with scan batch
    boundaries. A repeated URI (a multi-view span) must occupy one slot, not two.
    """
    n_uris, limit = 10, 4
    uris = [f"s3://b/act{i}.bin" for i in range(n_uris)]
    expected = set(sorted(uris, key=_sample_rank)[:limit])

    def selected(batch_rows: int) -> set[str]:
        # A repeated URI and a blank one are interleaved: both must be ignored.
        values = [*uris, uris[0], ""]
        batches = [
            pa.table({"action_data_uri": pa.array(values[start : start + batch_rows], pa.string())})
            for start in range(0, len(values), batch_rows)
        ]
        return set(_ranked_distinct_uris(batches, limit=limit))

    assert selected(len(uris) + 2) == expected
    assert selected(3) == expected


def test_collect_pca_sample_selects_smallest_rank_spans_regardless_of_batching() -> None:
    """The fit sample is the ``sample_size`` smallest-``_sample_rank`` distinct spans, order-invariant.

    Each span's descriptor stores its integer id in column 0, so the selected set
    is recoverable from the returned matrix. Re-chunking the descriptor stream must
    not change which spans train the basis - the reproducibility a scan-order sample
    would break. A duplicated URI (a multi-view span) must contribute its descriptor
    exactly once, or it would be over-weighted in the basis.
    """
    n_spans, sample_size = 10, 6
    uris = [f"s3://b/act{i}.bin" for i in range(n_spans)]
    matrix = np.zeros((n_spans, DESCRIPTOR_DIM), dtype=np.float32)
    matrix[:, 0] = np.arange(n_spans)  # column 0 carries the span id back out of the sample
    # Append a second view of span 0 (same URI) to exercise the per-URI de-dup.
    table = descriptor_batch(
        [*uris, uris[0]],
        np.vstack([matrix, matrix[0]]),
        DESCRIPTOR_DIM,
        np.ones(n_spans + 1, dtype=np.bool_),
    )
    expected = set(sorted(range(n_spans), key=lambda i: _sample_rank(uris[i]))[:sample_size])

    def selected_span_ids(batch_rows: int) -> set[int]:
        batches = [table.slice(start, batch_rows) for start in range(0, table.num_rows, batch_rows)]
        sample, used = _collect_pca_sample(batches, sample_size)
        assert used == sample_size
        return {int(row[0]) for row in sample}

    assert selected_span_ids(table.num_rows) == expected
    assert selected_span_ids(3) == expected


def test_collect_pca_sample_selects_the_same_span_set_when_batches_arrive_out_of_order() -> None:
    """The fitted population is the same SET however the extract batches interleave.

    Widening the extractor's reads lets batches arrive in an order the scan did not
    produce, which re-chunking alone does not exercise. Only the SET is asserted:
    the sample's ROW order does follow arrival, and because float64 summation is not
    associative that permutes the fit's last bits and so changes the basis's content
    fingerprint outright. That is a new name for an equivalent basis, not a
    regression, which is why no test here pins the fingerprint.
    """
    n_spans, sample_size = 12, 5
    uris = [f"s3://b/act{i}.bin" for i in range(n_spans)]
    matrix = np.zeros((n_spans, DESCRIPTOR_DIM), dtype=np.float32)
    matrix[:, 0] = np.arange(n_spans)  # column 0 carries the span id back out of the sample
    batches = [
        descriptor_batch([uri], matrix[index : index + 1], DESCRIPTOR_DIM, np.ones(1, dtype=np.bool_))
        for index, uri in enumerate(uris)
    ]

    in_order, used = _collect_pca_sample(batches, sample_size)
    reversed_arrival, reversed_used = _collect_pca_sample(list(reversed(batches)), sample_size)

    assert used == reversed_used == sample_size
    assert {int(row[0]) for row in in_order} == {int(row[0]) for row in reversed_arrival}
