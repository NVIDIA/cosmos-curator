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

"""The read side of Curate: preflight, fit sample, label gather, scan, quota, publish.

Every stage of this leg is a pure function of a table plus a spec, and Ray Data
decides only WHERE each one runs. The tests below call the stage functions
directly against a real Lance table and compose them by hand in
``_run_cpu_curate``, which walks the whole leg to a real commit without a Ray
cluster. What that leaves unexercised is named in ``TestCpuEndToEnd``.

::

    _preflight ---> _fit_sample ---> centroid basis
        |                                  |
        v                                  v
    _label_rollup -> _reduce_gather -> _merge_maps
        |                                  |
        v                                  v
    _scan_fragments -> mark_duplicates -> apply_label_merge
        |                                          |
        v                                          v
    _build_quota -> select_within_quota -> _reason_counts
        |
        v
    update_one_fragment -> _collect -> _commit -> read the table back

Assertions are on VALUES, never on the absence of an exception, because both
defects this leg has produced were silent: a mis-sized vector that stays finite
and unit-norm, and a norm overflow leaving a row at the origin. A test that only
proved nothing raised would have passed on both.
"""

import hashlib
import itertools
import json
import math
import pathlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import attrs
import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pytest
import ray

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    EMBEDDING_COLUMN_GROUPS,
    IMAGE_COLUMN_GROUP,
    KEY_COLUMN,
    TEXT_COLUMN_GROUP,
    TEXT_DIM,
)
from cosmos_curator.next.recipes.curation import dedup, fairness, vectors
from cosmos_curator.next.recipes.curation.columns import (
    CANONICAL_TASK_COLUMN,
    CENTROID_ARCHIVE_KEYS,
    CENTROIDS_ROOT_SUFFIX,
    CURATE_CLUSTER_ID,
    CURATE_SELECTION_REASON,
    DEDUP_KEY_COLUMN,
    DEDUP_SCORE_COLUMN,
    DISTANCE_COLUMN,
    FRAGMENT_COLUMN,
    FUSED_BLOCKS,
    NO_DEDUP_GROUP,
    NO_DEDUP_GROUP_ZERO_NORM,
    NO_SUBTASK_CLUSTER,
    RAY_COUNT_COLUMN,
    SUBTASK_CLUSTER_COLUMN,
    SUBTASK_VECTOR_COLUMN,
    SUBTASK_WEIGHT_FIELD,
    TASK_COLUMN,
    TASK_VECTOR_COLUMN,
    WORKING_VECTOR_COLUMN,
    CurateReason,
    WithinGroupOrder,
)
from cosmos_curator.next.recipes.curation.config import CurateConfig, ModalityWeights, SelectionTarget
from cosmos_curator.next.recipes.curation.pipeline import (
    _DEDUP_EPS_LADDER,
    _FRAGMENT_ID_COLUMN,
    _GATHER_COUNT_COLUMN,
    _GATHER_FRAGMENT_COLUMN,
    _GATHER_LABEL_COLUMN,
    _GATHER_ROW,
    _GATHER_TIEBREAK_COLUMN,
    _GATHER_VECTOR_COLUMN,
    _HISTOGRAM_BINS,
    _ONE_GPU_BYTES,
    _RADIUS_METRIC,
    _RADIUS_QUANTILES,
    _RESULT_COLUMN,
    _SCAN_ROW,
    _SCORE_METRIC,
    _SCORE_QUANTILES,
    _SUBTASK_FIT_ROWS_PER_CENTROID,
    _build_quota,
    _CentroidsArtifact,
    _clips_moved,
    _collect,
    _commit,
    _commit_properties,
    _concat_aggregate_blocks,
    _eps_ladder,
    _fit_sample,
    _fit_sample_fragments,
    _FitResult,
    _FitSpec,
    _fold_metric_histograms,
    _histogram_max,
    _histogram_quantiles,
    _label_rollup,
    _LabelSet,
    _log_dedup_score,
    _merge_maps,
    _preflight,
    _reason_counts,
    _reduce_gather,
    _report_metrics,
    _report_subtask_basis,
    _report_unfunded,
    _requested_k,
    _require_basis_width,
    _require_merge_matrix,
    _require_single_producer,
    _require_source_columns,
    _scan_fragments,
    _ScanBases,
    _Source,
    _subtask_sample_rows,
    _text_matrix,
    _total_rows,
    _weight_map,
    _write_centroids,
    bypass_scan_rows,
    metric_histogram,
    scored_scan_rows,
    unscored_rows,
    update_one_fragment,
)

from .conftest import (
    CANONICAL_TASKS,
    LABEL_CYCLE,
    SUBTASK_PROSE_COLUMN,
    SUBTASK_REGIONS,
    ClipsTable,
    ClipsTableSpec,
    GroupState,
    subtask_region,
)

BuildTable = Callable[[ClipsTableSpec], ClipsTable]

_TASK_TEXT_VECTOR = TASK_VECTOR_COLUMN

# Every fixed-size-list column any embedding group defines, so a hand-built
# minimal source table can carry the columns preflight requires without
# restating a width the schema module owns.
_VECTOR_TYPES: dict[str, pa.DataType] = {
    field.name: field.type
    for group in EMBEDDING_COLUMN_GROUPS
    for field in group.schema
    if pa.types.is_fixed_size_list(field.type)
}

# Every provenance column any embedding group defines, for the same reason: the
# producer-identity contract is stated over these columns, and restating their
# names here would let the fixture drift away from the schema module that owns
# them.
_PROVENANCE_COLUMNS: tuple[str, ...] = tuple(
    column for group in EMBEDDING_COLUMN_GROUPS for column in group.provenance_columns
)

# The one identity every provenance column of a filled group carries in the shared
# clips fixture (``_group_columns`` writes one value per column, for all rows).
_FIXTURE_PRODUCER_SUFFIX = "-fixture"


def _config_for_uri(uri: str, **overrides: object) -> CurateConfig:
    """Build a config against one table URI, supplying the routing envelope.

    ``schema_version`` and ``kind`` are required on every config, and they say
    nothing about the behavior any test here pins, so they live in one place
    rather than in every construction.
    """
    return CurateConfig(schema_version=1, kind="curate", clips_lance_uri=uri, **overrides)


def _config(
    table: ClipsTable,
    *,
    target: SelectionTarget | None = None,
    weights: ModalityWeights | None = None,
    within_group_order: WithinGroupOrder = "farthest",
    merge_theta_task: float = 0.95,
    subtask_clusters: int = 16,
    dedup_eps: float | None = 0.01,
    fairness_residual_seed: int = 0,
) -> CurateConfig:
    """Build a config against one fixture table, defaulting to keeping every survivor."""
    return _config_for_uri(
        table.uri,
        target=target or SelectionTarget(),
        weights=weights or ModalityWeights(),
        within_group_order=within_group_order,
        merge_theta_task=merge_theta_task,
        subtask_clusters=subtask_clusters,
        dedup_eps=dedup_eps,
        fairness_residual_seed=fairness_residual_seed,
    )


def _warnings(records: Sequence[dict[str, Any]]) -> list[str]:
    """Return the WARNING messages captured so far."""
    return [record["message"] for record in records if record["level"].name == "WARNING"]


def _infos(records: Sequence[dict[str, Any]]) -> list[str]:
    """Return the messages captured at exactly INFO level.

    Exactly INFO rather than "INFO and above", so this and ``_warnings``
    partition the captured output instead of overlapping - which keeps an
    emptiness assertion about one of them from being satisfied by the other's
    absence.
    """
    return [record["message"] for record in records if record["level"].name == "INFO"]


def _work_batch(*fragment_ids: int) -> pa.Table:
    """Build the one-column work batch both read UDFs take."""
    return pa.table({_FRAGMENT_ID_COLUMN: pa.array(list(fragment_ids), type=pa.int64())})


def _split_by(table: pa.Table, columns: Sequence[str]) -> list[pa.Table]:
    """Group a table by ``columns``, preserving arrival order inside each group.

    Stands in for one ``groupby(...).map_groups(...)``: the kernels under test are
    per-group functions and what Ray contributes is only which worker runs them.
    """
    keys = list(zip(*(table.column(name).to_pylist() for name in columns), strict=True))
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    return [table.take(rows) for rows in groups.values()]


def _count_table(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    """Reduce a table to per-key counts under Ray Data's own ``count()`` column name."""
    counted = table.group_by(list(columns)).aggregate([([], "count_all")])
    payload: dict[str, pa.ChunkedArray] = {name: counted.column(name) for name in columns}
    payload[RAY_COUNT_COLUMN] = counted.column("count_all")
    return pa.table(payload)


def _fixture_text_vector(clip_id: str, table: ClipsTable, column: str) -> npt.NDArray[np.float32]:
    """Return the vector the fixture wrote for one clip, read back from the table itself."""
    rows = lance.dataset(table.uri).to_table(columns=[KEY_COLUMN, column])
    index = rows.column(KEY_COLUMN).to_pylist().index(clip_id)
    return np.asarray(rows.column(column)[index].as_py(), dtype=np.float32)


def _minimal_source_table(
    path: pathlib.Path,
    key: pa.Array,
    *,
    without: str | None = None,
    producers: Mapping[str, Sequence[str | None]] | None = None,
    with_vectors: frozenset[str] = frozenset(),
) -> str:
    """Write the narrowest table preflight accepts, with a caller-chosen key column.

    Vector columns read NULL by default, which is enough for the column and type
    checks and lets a test isolate one of them. ``with_vectors`` names the columns
    that instead carry a unit basis vector on every row, which is how the state only
    a consumer can see is built: a group holding vectors while its provenance stays
    NULL. The width comes from the schema module's own Arrow type, never restated here.

    ``without`` omits one column entirely - vector or provenance - which is how a
    per-column requirement is told apart from a per-GROUP one, since the fixture's
    embedding groups transition together.

    Provenance columns default to NULL, matching a group that exists but has not
    been filled. ``producers`` replaces named provenance columns with explicit
    per-row identities, which is the only way to build the state the embeddings
    leg cannot produce and only a consumer can see: one group filled twice, by two
    different producers.

    Raises:
        ValueError: If ``without`` would drop a column ``producers`` or
            ``with_vectors`` fills, or if ``with_vectors`` names a column no
            embedding group defines. Each leaves a table holding neither state,
            leaving a test that asked for both passing on nothing.

    """
    filled = frozenset(with_vectors)
    unknown = sorted(filled - _VECTOR_TYPES.keys())
    if unknown:
        msg = f"with_vectors={unknown} name no vector column; the table would hold no vector at all"
        raise ValueError(msg)
    if without is not None and (without in (producers or {}) or without in filled):
        msg = f"without={without!r} drops a column the caller asked to fill; that state would not exist"
        raise ValueError(msg)
    columns: dict[str, pa.Array] = {
        KEY_COLUMN: key,
        TASK_COLUMN: pa.array(["task-0"] * len(key), type=pa.string()),
    }
    for name, vector_type in _VECTOR_TYPES.items():
        if name in filled:
            dim = vector_type.list_size
            unit = [0.0] * dim
            unit[0] = 1.0
            columns[name] = pa.array([unit] * len(key), type=vector_type)
        else:
            columns[name] = pa.nulls(len(key), type=vector_type)
    for name in _PROVENANCE_COLUMNS:
        columns[name] = pa.nulls(len(key), type=pa.string())
    for name, identities in (producers or {}).items():
        columns[name] = pa.array(list(identities), type=pa.string())
    uri = str(path)
    lance.write_dataset(pa.table({name: array for name, array in columns.items() if name != without}), uri)
    return uri


class TestContractSurface:
    """The two derived maps every stage reads its column names and weights from."""

    def test_the_two_text_vector_columns_are_derived_not_spelled(self) -> None:
        """The two levels read two different stored vectors, and neither name is hardcoded.

        The subtask one is the fused text block's primary vector, which is what
        makes the level-2 basis free to fit when that block carries weight; the
        task one is the other text vector in the same group.
        """
        assert SUBTASK_VECTOR_COLUMN == "embedding_text_subtask"
        assert TASK_VECTOR_COLUMN == "embedding_text_task"

    def test_weight_map_is_keyed_by_the_fused_block_weight_field(self) -> None:
        """The kernels are handed weights under the fused-block field names, in block order.

        Built from NON-default weights: against the defaults a map that returned a
        module constant instead of this config's values would pass, and a later
        retune of those defaults would turn this red for an unrelated reason.
        """
        config = _config_for_uri(
            "/tmp/x.lance",  # noqa: S108
            weights=ModalityWeights(subtask=0.5, image=0.3, action=0.2),
        )
        weights = _weight_map(config)
        assert list(weights) == [field for _group, field in FUSED_BLOCKS]
        assert weights == {"subtask": 0.5, "image": 0.3, "action": 0.2}

    def test_a_de_weighted_block_stays_in_the_map_at_zero(self) -> None:
        """The kernels read a 0.0 weight as "skip"; a dropped key would be a KeyError."""
        config = _config_for_uri(
            "/tmp/x.lance",  # noqa: S108
            weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5),
        )
        assert _weight_map(config)["subtask"] == 0.0


class TestPreflight:
    """Data-contract checks, the schema widening, and the version every read pins."""

    def test_a_missing_table_is_not_a_contract_violation(self, tmp_path: pathlib.Path) -> None:
        """An absent table raises FileNotFoundError, distinguishable from a malformed one."""
        config = _config_for_uri(str(tmp_path / "absent.lance"))
        with pytest.raises(FileNotFoundError, match="no Lance table at"):
            _preflight(config)

    def test_a_weighted_block_with_no_column_is_named(self, build_clips_table: BuildTable) -> None:
        """A corpus the action leg never ran for fails naming the column, before any read."""
        table = build_clips_table(ClipsTableSpec(action=GroupState.ABSENT))
        with pytest.raises(ValueError, match="is missing column"):
            _preflight(_config(table))

    def test_the_task_text_vector_is_required_even_with_no_weighted_text_block(
        self, build_clips_table: BuildTable
    ) -> None:
        """The level-1 merge runs on every run, so the task vector is read whatever the weights."""
        table = build_clips_table(ClipsTableSpec(text=GroupState.ABSENT))
        config = _config(table, weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5))
        with pytest.raises(ValueError, match=_TASK_TEXT_VECTOR):
            _preflight(config)

    def test_a_vector_column_stored_as_float64_is_refused(self, tmp_path: pathlib.Path) -> None:
        """float64 list elements pass a width-only gate but break GPU memory assumptions."""
        key = pa.array(["clip-0", "clip-1"])
        columns: dict[str, pa.Array] = {
            KEY_COLUMN: key,
            TASK_COLUMN: pa.array(["task-0", "task-1"], type=pa.string()),
            TASK_VECTOR_COLUMN: pa.nulls(len(key), type=pa.list_(pa.float64(), TEXT_DIM)),
        }
        for name, vector_type in _VECTOR_TYPES.items():
            if name == TASK_VECTOR_COLUMN:
                continue
            columns[name] = pa.nulls(len(key), type=vector_type)
        uri = str(tmp_path / "float64-task-vector.lance")
        lance.write_dataset(pa.table(columns), uri)
        with pytest.raises(ValueError, match="fixed_size_list<float32>"):
            _require_source_columns(lance.dataset(uri), _weight_map(_config_for_uri(uri)))

    def test_the_subtask_prose_column_is_not_required(self, tmp_path: pathlib.Path) -> None:
        """Nothing reads ``subtask_name`` any more, so its absence is not a contract violation.

        The level-2 key is a partition of the subtask EMBEDDING, and dropping the
        prose from the read contract is what took the corpus-scale string
        vocabulary out of this leg. Written without the column at all rather than
        with it NULL, so a check that merely tolerated nulls would still fail here.
        """
        uri = _minimal_source_table(tmp_path / "no-subtask-prose.lance", pa.array(["clip-0", "clip-1"]))
        dataset = lance.dataset(uri)
        assert SUBTASK_PROSE_COLUMN not in dataset.schema.names

        _require_source_columns(dataset, _weight_map(_config_for_uri(uri)))

    def test_the_subtask_text_vector_is_required_only_when_its_block_is_funded(self, tmp_path: pathlib.Path) -> None:
        """At a zero subtask weight nothing reads the column and no level-2 basis is fitted.

        The requirement follows the eligibility predicate rather than the level-2
        key: the key partitions this column, but a corpus without it is still
        curatable on image and action alone, landing every row in the reserved
        level-2 cell. Asserted both ways round, because a table that satisfies the
        contract cannot show which weight the requirement was read from.
        """
        uri = _minimal_source_table(
            tmp_path / "no-subtask-vector.lance",
            pa.array(["clip-0", "clip-1"]),
            without=SUBTASK_VECTOR_COLUMN,
        )
        dataset = lance.dataset(uri)
        weights = _weight_map(_config_for_uri(uri))

        _require_source_columns(dataset, {**weights, "subtask": 0.0})
        with pytest.raises(ValueError, match=SUBTASK_VECTOR_COLUMN):
            _require_source_columns(dataset, {**weights, "subtask": 0.5})

    def test_a_non_string_key_column_is_refused(self, tmp_path: pathlib.Path) -> None:
        """``update_columns`` would reject the join key inside a worker, after the whole fit."""
        uri = _minimal_source_table(tmp_path / "int-key.lance", pa.array([1, 2], type=pa.int64()))
        with pytest.raises(ValueError, match="must be string"):
            _preflight(_config_for_uri(uri))

    def test_a_consumed_group_filled_by_two_producers_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Two producers of the same WIDTH pass every shape check and fuse into a wrong metric.

        This is the failure no later stage can report: the run concatenates both
        producers' vectors into one cosine space, where every distance stays finite
        and ordinary-looking, so the clusters, the duplicate verdicts and the
        quotas all come out plausible and meaningless. Refusing here is the only
        place it is visible.
        """
        column = IMAGE_COLUMN_GROUP.provenance_columns[0]
        uri = _minimal_source_table(
            tmp_path / "two-producers.lance",
            pa.array(["clip-0", "clip-1"]),
            producers={column: ["model-a", "model-b"]},
        )
        with pytest.raises(ValueError, match=f"more than one producer: '{column}'"):
            _require_single_producer(lance.dataset(uri), _weight_map(_config_for_uri(uri)))

    def test_every_provenance_column_of_a_consumed_group_is_read(self, tmp_path: pathlib.Path) -> None:
        """The action group identifies its producer with TWO columns, and both are checked.

        Its descriptor version and its PCA fingerprint are independent: one basis
        can be refitted from the same descriptors, and one descriptor version can
        be projected through two bases. A check that read only the first column of
        each group would accept the second case, which is precisely the one whose
        geometry is invalid.
        """
        column = ACTION_COLUMN_GROUP.provenance_columns[-1]
        uri = _minimal_source_table(
            tmp_path / "two-bases.lance",
            pa.array(["clip-0", "clip-1"]),
            producers={column: ["fp-a", "fp-b"]},
        )
        with pytest.raises(ValueError, match=f"more than one producer: '{column}'"):
            _require_single_producer(lance.dataset(uri), _weight_map(_config_for_uri(uri)))

    def test_a_de_weighted_group_s_producers_are_not_read(self, tmp_path: pathlib.Path) -> None:
        """A group this run does not fuse cannot corrupt its metric, so its provenance is ignored.

        Asserted both ways round on ONE table, because a table that passes cannot
        show which weight the read was gated on.
        """
        column = IMAGE_COLUMN_GROUP.provenance_columns[0]
        uri = _minimal_source_table(
            tmp_path / "unweighted-image.lance",
            pa.array(["clip-0", "clip-1"]),
            producers={column: ["model-a", "model-b"]},
        )
        dataset = lance.dataset(uri)
        unweighted = _weight_map(_config_for_uri(uri, weights=ModalityWeights(subtask=0.5, image=0.0, action=0.5)))
        weighted = _weight_map(_config_for_uri(uri, weights=ModalityWeights(subtask=0.5, image=0.5, action=0.0)))

        assert _require_single_producer(dataset, unweighted) == {}
        with pytest.raises(ValueError, match=f"more than one producer: '{column}'"):
            _require_single_producer(dataset, weighted)

    def test_the_text_group_s_producer_is_read_even_with_no_weighted_text_block(self, tmp_path: pathlib.Path) -> None:
        """The task merge reads a text vector on every run, so the text group is always consumed.

        The same asymmetry the column contract encodes for ``embedding_text_task``:
        no weight names it, and at ``subtask=0.0`` the text group is still the
        source of the vectors the level-1 merge compares.
        """
        column = TEXT_COLUMN_GROUP.provenance_columns[0]
        uri = _minimal_source_table(
            tmp_path / "two-text-models.lance",
            pa.array(["clip-0", "clip-1"]),
            producers={column: ["model-a", "model-b"]},
        )
        weights = _weight_map(_config_for_uri(uri, weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5)))
        with pytest.raises(ValueError, match=f"more than one producer: '{column}'"):
            _require_single_producer(lance.dataset(uri), weights)

    def test_a_consumed_group_missing_its_provenance_column_is_refused(self, tmp_path: pathlib.Path) -> None:
        """An unrecorded producer is as unusable as two: neither can be shown to be one.

        Accepting absence would also leave the gate with a trivial bypass - drop
        the column and the check has nothing to read. A group the embeddings leg
        wrote always carries its provenance, so this state means the columns came
        from somewhere else.
        """
        column = IMAGE_COLUMN_GROUP.provenance_columns[0]
        uri = _minimal_source_table(tmp_path / "no-provenance.lance", pa.array(["clip-0", "clip-1"]), without=column)
        with pytest.raises(ValueError, match=f"has no column '{column}'"):
            _require_single_producer(lance.dataset(uri), _weight_map(_config_for_uri(uri)))

    def test_an_unfilled_group_names_no_producer_rather_than_failing(self, tmp_path: pathlib.Path) -> None:
        """A group added but never filled has no identity to report, and that is not an error.

        The eligible-row count is what refuses that corpus, and it says something
        far more useful than a provenance message could.
        """
        uri = _minimal_source_table(tmp_path / "unfilled.lance", pa.array(["clip-0", "clip-1"]))
        assert _require_single_producer(lance.dataset(uri), _weight_map(_config_for_uri(uri))) == {}

    def test_a_consumed_group_with_vectors_but_no_producer_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Vectors without provenance are not an unfilled group and must not reach clustering."""
        vector = IMAGE_COLUMN_GROUP.primary_vector
        uri = _minimal_source_table(
            tmp_path / "vectors-without-producer.lance",
            pa.array(["clip-0", "clip-1"]),
            with_vectors=frozenset({vector}),
        )
        weights = _weight_map(_config_for_uri(uri))
        with pytest.raises(ValueError, match="no producer identity"):
            _require_single_producer(lance.dataset(uri), weights)

    def test_a_provenance_rejection_propagates_out_of_preflight(self, tmp_path: pathlib.Path) -> None:
        """The provenance gate is reached through ``_preflight``, not only when called directly.

        Every other test of this gate calls ``_require_single_producer`` itself,
        which cannot show that preflight runs it at all: a gate no caller invokes
        refuses nothing.
        """
        uri = _minimal_source_table(
            tmp_path / "preflight-reaches-provenance.lance",
            pa.array(["clip-0", "clip-1"]),
            with_vectors=frozenset({IMAGE_COLUMN_GROUP.primary_vector}),
        )

        with pytest.raises(ValueError, match="no producer identity"):
            _preflight(_config_for_uri(uri))

    def test_a_group_recording_only_one_of_its_two_producers_is_refused(self, tmp_path: pathlib.Path) -> None:
        """The action group's descriptor version and PCA basis are accounted for separately.

        A refit basis written over descriptors that were never re-versioned leaves
        the vectors attributed to the descriptors alone, and the centroids artifact
        would then archive a producer pair naming the descriptors but not the
        geometry those vectors actually live in. A per-GROUP check passes this
        table, because one filled provenance column is enough to satisfy it.
        """
        recorded, unrecorded = ACTION_COLUMN_GROUP.provenance_columns
        uri = _minimal_source_table(
            tmp_path / "action-half-attributed.lance",
            pa.array(["clip-0", "clip-1"]),
            producers={recorded: ["desc-v1", "desc-v1"]},
            with_vectors=frozenset({ACTION_COLUMN_GROUP.primary_vector}),
        )
        weights = _weight_map(_config_for_uri(uri))

        with pytest.raises(ValueError, match=f"no producer identity in .*{unrecorded}"):
            _require_single_producer(lance.dataset(uri), weights)

    def test_the_task_text_vector_alone_is_enough_to_require_the_text_producer(self, tmp_path: pathlib.Path) -> None:
        """The vector the level-1 merge reads is the one the text group must be attributed by.

        The fused text block is the SUBTASK vector, so a gate probing the group's
        primary vector reads a column this table leaves NULL and concludes the
        group is unfilled - while the merge goes on to fuse task vectors of
        unrecorded provenance, which is the state the gate exists to refuse.
        """
        uri = _minimal_source_table(
            tmp_path / "task-vectors-without-producer.lance",
            pa.array(["clip-0", "clip-1"]),
            with_vectors=frozenset({TASK_VECTOR_COLUMN}),
        )
        column = TEXT_COLUMN_GROUP.provenance_columns[0]
        weights = _weight_map(_config_for_uri(uri))

        with pytest.raises(ValueError, match=f"no producer identity in .*{column}"):
            _require_single_producer(lance.dataset(uri), weights)

    def test_a_vector_this_run_never_reads_does_not_require_a_producer(self, tmp_path: pathlib.Path) -> None:
        """At ``subtask=0.0`` the subtask vector leaves the metric, and so does its attribution.

        Refusing here would refuse a corpus whose subtask embeddings this run
        never touches.
        """
        uri = _minimal_source_table(
            tmp_path / "unread-subtask-vectors.lance",
            pa.array(["clip-0", "clip-1"]),
            with_vectors=frozenset({SUBTASK_VECTOR_COLUMN}),
        )
        weights = _weight_map(
            _config_for_uri(uri, weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5)),
        )

        assert _require_single_producer(lance.dataset(uri), weights) == {}

    def test_the_producer_of_each_consumed_group_is_recorded_on_the_source(self, build_clips_table: BuildTable) -> None:
        """Preflight resolves the identities, so the run can record what it curated.

        Every consumed group contributes every provenance column it defines; a
        de-weighted group contributes none, which is what makes this a record of
        the metric actually used rather than of the table's contents.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
        source = _preflight(_config(table, weights=ModalityWeights(subtask=0.6, image=0.0, action=0.4)))
        expected = {
            column: f"{column}{_FIXTURE_PRODUCER_SUFFIX}"
            for group in (TEXT_COLUMN_GROUP, ACTION_COLUMN_GROUP)
            for column in group.provenance_columns
        }
        assert source.producers == expected

    def test_the_widened_columns_are_visible_at_the_pinned_version(self, build_clips_table: BuildTable) -> None:
        """The version is pinned AFTER the widening, so the write's columns exist within it."""
        table = build_clips_table(ClipsTableSpec())
        source = _preflight(_config(table))
        pinned = lance.dataset(table.uri, version=source.read.read_version)
        assert {CURATE_SELECTION_REASON, CURATE_CLUSTER_ID} <= set(pinned.schema.names)

    def test_only_rows_satisfying_the_predicate_are_claimed(self, build_clips_table: BuildTable) -> None:
        """A fragment whose vectors read NULL contributes no eligible row, but keeps its rows."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4, empty_fragments=frozenset({1})))
        source = _preflight(_config(table))
        assert source.eligible_rows == 8
        assert source.fragment_rows == (4, 4, 4)

    def test_the_predicate_names_bare_identifiers(self, build_clips_table: BuildTable) -> None:
        """A quoted identifier is a string LITERAL to Lance, and would pass the whole corpus."""
        table = build_clips_table(ClipsTableSpec(empty_fragments=frozenset({0, 1})))
        source = _preflight(_config(table))
        assert '"' not in source.read.predicate
        assert source.eligible_rows == 4

    def test_a_corpus_with_no_claimable_row_is_refused(self, build_clips_table: BuildTable) -> None:
        """Zero eligible rows is a data-contract failure, not a successful empty run."""
        table = build_clips_table(ClipsTableSpec(fragments=2, empty_fragments=frozenset({0, 1})))
        with pytest.raises(ValueError, match="satisfies"):
            _preflight(_config(table))

    def test_every_fragment_is_reported_exactly_once(self, build_clips_table: BuildTable) -> None:
        """The work items and the fit prefix both index this tuple, so a gap loses a fragment.

        Named for what a freshly written table can actually show: here manifest
        order and ascending id coincide, so the ORDER itself is not observable and
        claiming it would be vacuous.
        """
        table = build_clips_table(ClipsTableSpec(fragments=4, rows_per_fragment=2))
        source = _preflight(_config(table))
        assert sorted(source.fragment_ids) == [0, 1, 2, 3]
        assert len(set(source.fragment_ids)) == 4


class TestRequestedK:
    """``k`` is derived from the target mean cluster size and from nothing else."""

    def test_k_is_the_ceiling_of_eligible_rows_over_the_target_mean(self) -> None:
        """A partial cluster still gets a centroid, so the configured mean is an upper bound.

        The exact multiple is the discriminating input: at 1,000,001 rows ``ceil``
        and ``floor + 1`` agree, so only a boundary case tells them apart.
        """
        assert _requested_k(1_000_001, 200_000) == 6
        assert _requested_k(400_000, 200_000) == 2

    def test_a_corpus_smaller_than_one_target_cluster_degenerates_to_one(self) -> None:
        """The current corpus against the 250M-row design target: one exhaustive cluster."""
        assert _requested_k(131_602, 200_000) == 1

    def test_the_degenerate_single_cluster_is_reported(self, loguru_records: list[dict[str, Any]]) -> None:
        """``curate_cluster_id`` becomes constant, so the run says so rather than accepting it."""
        _requested_k(131_602, 200_000)
        assert any("k=1" in message for message in _warnings(loguru_records))

    def test_a_single_cluster_that_no_card_can_score_names_the_knob_that_splits_it(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """At ``k == 1`` the one cluster is the whole corpus, and one cluster is one card.

        The ceiling comes from the retention stage's own model, so a corpus placed
        above it will be refused there; the driver says so first and names the only
        config field that raises ``k``. Asserted against the same
        ``dedup.max_group_rows`` the stage uses, because a hardcoded row count
        here would pass while disagreeing with the stage that enforces it.
        """
        ceiling = dedup.max_group_rows(width=vectors.FUSED_DIM, device_total_bytes=_ONE_GPU_BYTES)
        rows = ceiling + 1
        assert _requested_k(rows, rows) == 1
        assert any("lower target_mean_cluster_rows" in message for message in _warnings(loguru_records))

    def test_a_single_cluster_within_one_card_is_reported_without_the_remedy(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """Below the ceiling the exhaustive cluster runs, so naming a remedy would misdirect."""
        ceiling = dedup.max_group_rows(width=vectors.FUSED_DIM, device_total_bytes=_ONE_GPU_BYTES)
        assert _requested_k(ceiling, ceiling) == 1
        assert _warnings(loguru_records)
        assert not any("lower target_mean_cluster_rows" in message for message in _warnings(loguru_records))


class TestSubtaskSampleRows:
    """How many rows the level-2 fit reads: per-centroid coverage under the operator's budget."""

    def test_the_cap_scales_with_the_group_count_it_has_to_fit(self) -> None:
        """A basis of k cells needs rows per cell, and nothing about the corpus size."""
        assert _subtask_sample_rows(16, 4_000_000) == 16 * _SUBTASK_FIT_ROWS_PER_CENTROID

    def test_the_operator_budget_is_still_the_ceiling(self) -> None:
        """Ask for no more host memory than the locality fit already spends."""
        assert _subtask_sample_rows(1_000_000, 4_000) == 4_000


class TestSubtaskBasisReport:
    """An absent level-2 basis has two causes, and only one of them is a loss.

    The artifact records both as ``(0, TEXT_DIM)`` and every scan row takes
    ``NO_SUBTASK_CLUSTER`` either way, so the run's log is the only place they can
    still be told apart -- and the weights are what tells them apart, which is why
    the report lives beside the fit rather than at the fairness merge.
    """

    def _basis(self, centroids: int) -> npt.NDArray[np.float32]:
        return np.eye(centroids, TEXT_DIM, dtype=np.float32)

    def test_an_unweighted_block_reports_the_absent_basis_without_warning(
        self,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """A zero subtask weight is the documented escape, so fitting nothing was the request.

        Kept at INFO deliberately: warning on a configured state is what teaches
        an operator to stop reading warnings, and then the loss below goes unread
        too.
        """
        assert _report_subtask_basis(None, subtask_weight=0.0, requested=16) == 0

        assert any("carries no weight" in message for message in _infos(loguru_records))
        assert _warnings(loguru_records) == []

    def test_a_weighted_block_with_no_basis_warns_that_fairness_lost_granularity(
        self,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """A funded block that fitted nothing silently drops the level-2 cell.

        The whole point of the cell is that fairness groups are finer than the
        canonical task label; without a basis every row shares its task's single
        group, which is a reduction in what the run selected for and not a
        configured state. The line therefore names what was lost and the column
        to look at.
        """
        assert _report_subtask_basis(None, subtask_weight=0.6, requested=16) == 0

        warned = _warnings(loguru_records)
        assert len(warned) == 1
        assert "falls back to the canonical task label" in warned[0]
        assert SUBTASK_VECTOR_COLUMN in warned[0]

    def test_the_two_absent_basis_causes_do_not_share_a_line(
        self,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """Neither report is reachable from the other cause, which is the whole distinction.

        Asserted as an absence on both sides: a fix that widened one message to
        cover both cases would still satisfy either test above on its own.
        """
        _report_subtask_basis(None, subtask_weight=0.6, requested=16)

        assert not any("carries no weight" in message for message in _infos(loguru_records))
        assert any("falls back to the canonical task label" in message for message in _warnings(loguru_records))

    def test_a_fitted_basis_is_reported_at_its_observed_size(
        self,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """A basis that exists is neither cause, and its size is read off the fit."""
        assert _report_subtask_basis(self._basis(16), subtask_weight=0.6, requested=16) == 16

        assert _warnings(loguru_records) == []

    def test_a_clamped_basis_is_warned_about_as_a_clamp_and_not_as_a_fallback(
        self,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """Fewer centroids than requested is a coarser cell, not an absent one.

        A distinct regime from the two above and from each other: level-2 fairness
        still partitions each task, so the line reports the shortfall rather than
        a fallback that did not happen.
        """
        assert _report_subtask_basis(self._basis(3), subtask_weight=0.6, requested=16) == 3

        warned = _warnings(loguru_records)
        assert len(warned) == 1
        assert "holds 3 centroid(s)" in warned[0]
        assert "falls back" not in warned[0]


class TestFitSampler:
    """The sampler is a manifest-ordered fragment prefix bounded by eligible rows."""

    def _source(self, fragment_rows: tuple[int, ...], fragment_eligible_rows: tuple[int, ...] | None = None) -> _Source:
        eligible = fragment_rows if fragment_eligible_rows is None else fragment_eligible_rows
        return _Source(
            dataset=None,  # type: ignore[arg-type]
            read=None,  # type: ignore[arg-type]
            fragment_ids=tuple(range(len(fragment_rows))),
            fragment_rows=fragment_rows,
            fragment_eligible_rows=eligible,
            eligible_rows=sum(eligible),
            producers={},
        )

    def _taken(self, fragment_rows: tuple[int, ...], budget: int) -> tuple[int, ...]:
        return _fit_sample_fragments(self._source(fragment_rows), budget, _subtask_sample_rows(16, budget)).fragment_ids

    def test_a_fully_ineligible_leading_fragment_does_not_end_the_prefix(self) -> None:
        """A large leading fragment with zero eligible rows must not empty the sample.

        The prefix is budgeted on eligible rows, so a 40M-row leading fragment that
        contributes nothing extends into the later fragment that actually carries
        the corpus, rather than stopping on itself and starving ``_fit_sample``.
        """
        source = self._source(fragment_rows=(40_000_000, 100), fragment_eligible_rows=(0, 100))
        assert _fit_sample_fragments(source, 4_000_000, _subtask_sample_rows(16, 4_000_000)).fragment_ids == (0, 1)

    def test_an_ineligible_prefix_does_not_block_the_first_eligible_fragment_past_budget(self) -> None:
        """The fragment carrying the corpus is reached even when it alone exceeds the budget.

        The counterpart to the fixture above, whose eligible fragment fits inside
        the budget. Only eligible rows count against the budget and the leading
        fragment contributes none, so nothing about the next one crossing the
        budget stops the walk from getting there.
        """
        source = self._source(fragment_rows=(1_000, 5_000_000), fragment_eligible_rows=(0, 5_000_000))
        assert _fit_sample_fragments(source, 4_000_000, _subtask_sample_rows(16, 4_000_000)).fragment_ids == (0, 1)

    def test_a_budget_below_one_fragment_still_takes_one(self) -> None:
        """Undershooting the budget beats fitting a basis on nothing at all."""
        assert self._taken((100, 100), 10) == (0,)

    def test_the_prefix_crosses_the_budget_by_at_most_one_fragment(self) -> None:
        """The fragment that reaches the budget is taken, and the walk stops there.

        ``_fit_sample`` truncates inside that last fragment, so crossing is what
        lets the sample reach the budget at all; stopping short of it would leave
        the basis fitted on whatever the earlier fragments happened to hold.
        """
        assert self._taken((40, 40, 40, 40), 90) == (0, 1, 2)

    def test_a_prefix_that_lands_exactly_on_the_budget_stops_there(self) -> None:
        """Reaching the budget ends the walk; passing it is not required.

        The boundary the crossing test above cannot see - its fixture overshoots
        under either bound. A prefix that kept walking from here would take a
        further fragment whose rows the cap discards: read volume the basis never
        sees, and on a real manifest that fragment is not small.
        """
        assert self._taken((40, 50, 40), 90) == (0, 1)

    def test_a_thinly_eligible_leading_fragment_does_not_starve_the_sample(self) -> None:
        """One eligible row up front must not end a four-million-row prefix.

        The fit does not fail on a one-row sample - it clamps ``k`` to it and
        reports success, so the run partitions the corpus by a basis nobody asked
        for. A prefix that stopped before the fragment carrying the corpus is the
        one way to reach that state.
        """
        source = self._source(fragment_rows=(100, 5_000_000), fragment_eligible_rows=(1, 5_000_000))

        assert _fit_sample_fragments(source, 4_000_000, _subtask_sample_rows(16, 4_000_000)).fragment_ids == (0, 1)

    def test_the_whole_manifest_is_taken_when_the_budget_covers_it(self) -> None:
        """A corpus under the budget is fitted in full, in manifest order."""
        assert self._taken((40, 40, 40), 4_000_000) == (0, 1, 2)

    def test_the_matrices_are_sized_by_the_rows_the_prefix_can_fill(self) -> None:
        """Both caps come off ELIGIBLE rows, never off the prefix's physical size.

        A prefix that undershoots the budget is the whole manifest, so its physical
        rows are the corpus's. On a corpus the predicate thins out, sizing the
        buffers by those rows reserves host memory for rows the filtered scan can
        never return - here 12.9 GiB of it for a sample that holds 0.3 GiB.
        """
        source = self._source(fragment_rows=(40_000_000,), fragment_eligible_rows=(100_000,))

        prefix = _fit_sample_fragments(source, 4_000_000, _subtask_sample_rows(16, 4_000_000))

        assert (prefix.sample_row_cap, prefix.subtask_sample_row_cap) == (100_000, 100_000)

    def test_the_level_two_cap_keeps_its_own_budget_rather_than_the_locality_one(self) -> None:
        """Each matrix is bounded by its OWN budget, and only then by eligible rows.

        The fixture above cannot see this: its eligible count is below both budgets,
        so a subtask cap that had adopted ``fit_sample_rows`` would land on the same
        number. Here the eligible rows exceed the level-2 budget, which is the only
        shape where sharing one budget shows up - as a 384-wide matrix allocated for
        rows the level-2 fit never asks for.
        """
        subtask_budget = _subtask_sample_rows(16, 4_000_000)
        source = self._source(fragment_rows=(40_000_000,), fragment_eligible_rows=(3_000_000,))

        prefix = _fit_sample_fragments(source, 4_000_000, subtask_budget)

        assert (prefix.sample_row_cap, prefix.subtask_sample_row_cap) == (3_000_000, subtask_budget)

    def test_an_overshooting_prefix_does_not_warn_about_memory_it_will_never_allocate(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """The device estimate follows the row cap, which is what bounds the allocation.

        The first fragment is taken before the budget is consulted, so a 40M-row
        fragment against a 4M-row cap overshoots tenfold - but the fit matrix is
        still capped at 4M rows, which needs ~55 GiB of the reference device. An
        estimate scaled by the physical rows instead reports ~515 GiB and warns
        about a peak the run cannot reach.
        """
        self._taken((40_000_000, 100), 4_000_000)

        assert _warnings(loguru_records) == []

    def test_a_row_cap_that_really_exceeds_one_device_is_warned_about_with_a_target(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """When the cap is the problem, the warning names the value that would fit.

        "Lower ``fit_sample_rows``" alone leaves the operator to invert the peak
        model, so the warning does the arithmetic: 6,206,600 rows of 865 float32
        is the largest sample whose ~4x k-means peak stays inside 80 GiB.
        """
        self._taken((8_000_000,), 8_000_000)

        assert any("lower fit_sample_rows to at most 6206600" in message for message in _warnings(loguru_records))

    def test_the_host_matrix_size_is_reported_even_though_the_warning_is_device_only(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """The host cost is stated as fact, because no threshold for it is knowable here.

        ``_fit_sample`` allocates one buffer of ``min(cap, eligible) x 865``
        float32, a real ~12.9 GiB at the default cap that the device factor models
        nothing of. The fit task cannot know its share of node RAM, so the honest
        move is to report the number and leave the budget to whoever chose the
        node.
        """
        self._taken((40_000_000, 100), 4_000_000)

        assert any("12.9 GiB host matrix" in message for message in _infos(loguru_records))

    def test_the_level_two_matrix_is_reported_as_its_own_cost(self, loguru_records: list[dict[str, Any]]) -> None:
        """It is a SECOND host buffer in the same task, so a single number would understate it."""
        self._taken((40_000_000, 100), 4_000_000)

        assert any("subtask-text matrix" in message for message in _infos(loguru_records))

    def test_the_share_of_the_corpus_the_basis_saw_is_reported(self, loguru_records: list[dict[str, Any]]) -> None:
        """Neither row count says how biased the prefix is; their ratio does.

        Taken from the CAPPED rows, not the prefix's physical rows: on this
        fixture the prefix overshoots by 10x, so a share computed from the
        physical rows would report ~1000% coverage of a corpus the fit reads a
        tenth of.
        """
        self._taken((40_000_000, 100), 4_000_000)

        assert any("at most ~10.0% of the 40000100 eligible row(s)" in message for message in _infos(loguru_records))

    def test_a_prefix_covering_the_whole_corpus_reports_full_coverage(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """The counterfactual: an unbiased prefix has to read as one, or the share means nothing."""
        self._taken((40, 40), 4_000_000)

        assert any("at most ~100.0% of the 80 eligible row(s)" in message for message in _infos(loguru_records))


class TestFitSample:
    """What the k-means is handed: the working-vector matrix of the sampled prefix."""

    def _spec(self, source: _Source, config: CurateConfig, *, cap: int | None = None) -> _FitSpec:
        return _fit_spec(source, config, requested_k=1, cap=cap)

    def test_the_sample_matches_the_working_vectors_of_the_same_rows(self, build_clips_table: BuildTable) -> None:
        """Routed through ``working_vectors``, whose width gate then runs before the fit."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=4))
        config = _config(table)
        source = _preflight(config)
        sample = _fit_sample(source.read, self._spec(source, config))
        rows = lance.dataset(table.uri, version=source.read.read_version).to_table(filter=source.read.predicate)
        expected = vectors.working_vectors(rows, _weight_map(config)).working
        np.testing.assert_array_equal(sample.fused, expected)

    def test_every_sampled_row_is_unit_norm(self, build_clips_table: BuildTable) -> None:
        """The weights sum to 1, so the fused vector sits on the unit sphere by identity."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        config = _config(table)
        source = _preflight(config)
        sample = _fit_sample(source.read, self._spec(source, config))
        np.testing.assert_allclose(np.linalg.norm(sample.fused, axis=1), 1.0, rtol=0, atol=1e-6)

    def test_a_zero_norm_row_never_reaches_the_basis(self, build_clips_table: BuildTable) -> None:
        """A row with no direction would drag a centroid toward the origin.

        Asserted on WHICH rows survived, not how many: a count of three is equally
        satisfied by a sample that kept the origin row and dropped a healthy one,
        and that is the exact defect this guards.
        """
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=4, zero_norm_vector_rows=frozenset({2}))
        table = build_clips_table(spec)
        config = _config(table)
        source = _preflight(config)
        sample = _fit_sample(source.read, self._spec(source, config))
        rows = lance.dataset(table.uri, version=source.read.read_version).to_table(filter=source.read.predicate)
        fused = vectors.working_vectors(rows, _weight_map(config))
        assert fused.keep.tolist() == [True, True, False, True]
        np.testing.assert_array_equal(sample.fused, fused.working)
        np.testing.assert_allclose(np.linalg.norm(sample.fused, axis=1), 1.0, rtol=0, atol=1e-6)

    def test_the_row_cap_bounds_the_sample_exactly(self, build_clips_table: BuildTable) -> None:
        """The config bounds the sample whenever it is the smaller of the two bounds."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=4))
        config = _config(table)
        source = _preflight(config)
        assert _fit_sample(source.read, self._spec(source, config, cap=5)).fused.shape[0] == 5

    def test_a_thinned_prefix_still_fills_every_eligible_row(self, build_clips_table: BuildTable) -> None:
        """The eligible-bounded cap must not lose a row the filtered scan returns.

        Two of three fragments hold no eligible row, so the prefix's physical rows
        are three times its eligible ones - the shape where the cap is the eligible
        count rather than the config, and the direction where getting the bound
        wrong is a silently short sample the fit clamps ``k`` to rather than an
        error.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4, empty_fragments=frozenset({0, 2})))
        config = _config(table)
        source = _preflight(config)
        sample = _fit_sample(source.read, self._spec(source, config))
        rows = lance.dataset(table.uri, version=source.read.read_version).to_table(filter=source.read.predicate)
        np.testing.assert_array_equal(sample.fused, vectors.working_vectors(rows, _weight_map(config)).working)

    def test_both_matrices_come_out_of_one_pass(self, build_clips_table: BuildTable) -> None:
        """The read is the expensive part, so the level-2 basis must cost no extra one.

        Asserted as content, not merely presence: the subtask matrix has to be the
        UNIT subtask-text rows of the same sample, which is what makes it a basis
        over the stored text vector rather than over the fused one.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=4))
        config = _config(table)
        source = _preflight(config)

        sample = _fit_sample(source.read, self._spec(source, config))

        rows = lance.dataset(table.uri, version=source.read.read_version).to_table(filter=source.read.predicate)
        matrix, _present = _text_matrix(rows, SUBTASK_VECTOR_COLUMN)
        expected, usable = vectors.unit_rows(matrix)
        assert sample.subtask is not None
        np.testing.assert_array_equal(sample.subtask, expected[usable])

    def test_no_level_two_matrix_is_built_for_an_unfunded_subtask_block(self, build_clips_table: BuildTable) -> None:
        """A de-weighted modality leaves the metric AND stops being read at all."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=4))
        config = _config(table, weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5))
        source = _preflight(config)

        assert _fit_sample(source.read, self._spec(source, config)).subtask is None

    def test_a_small_level_two_cap_does_not_truncate_the_locality_sample(self, build_clips_table: BuildTable) -> None:
        """The two row caps are independent, and one filling first must not stop the read.

        The level-2 cap is the smaller of the two by construction (it scales with
        ``k``, not with the corpus), so a read that stopped when EITHER cap filled
        would silently fit the locality basis on a fraction of its own budget.
        """
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=4))
        config = _config(table)
        source = _preflight(config)
        spec = attrs.evolve(self._spec(source, config), subtask_sample_row_cap=2)

        sample = _fit_sample(source.read, spec)

        assert sample.subtask is not None
        assert (sample.fused.shape[0], sample.subtask.shape[0]) == (8, 2)

    def test_a_sample_holding_no_usable_vector_is_refused(self, build_clips_table: BuildTable) -> None:
        """Fitting on nothing would produce a basis that captures everything."""
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=2, zero_norm_vector_rows=frozenset({0, 1}))
        table = build_clips_table(spec)
        config = _config(table)
        source = _preflight(config)
        with pytest.raises(ValueError, match="no usable vector"):
            _fit_sample(source.read, self._spec(source, config))


class TestTextMatrix:
    """The gather's own width gate, and the all-NULL column it has to survive."""

    def test_a_mis_sized_column_is_refused(self) -> None:
        """A label embedded at another width stays finite and unit-norm, so width is checked."""
        values = pa.array(np.ones(16, dtype=np.float32), type=pa.float32())
        batch = pa.table({_TASK_TEXT_VECTOR: pa.FixedSizeListArray.from_arrays(values, 8)})
        with pytest.raises(ValueError, match=f"runs at {TEXT_DIM}"):
            _text_matrix(batch, _TASK_TEXT_VECTOR)

    def test_a_float64_column_is_refused(self) -> None:
        """float64 elements are rejected before any decode or merge."""
        values = pa.array(np.ones(TEXT_DIM, dtype=np.float64), type=pa.float64())
        batch = pa.table({_TASK_TEXT_VECTOR: pa.FixedSizeListArray.from_arrays(values, TEXT_DIM)})
        with pytest.raises(ValueError, match="float32"):
            _text_matrix(batch, _TASK_TEXT_VECTOR)

    def test_a_column_that_is_not_a_fixed_size_list_is_refused(self) -> None:
        """A variable-width column has no matrix to decode; its type is named."""
        batch = pa.table({_TASK_TEXT_VECTOR: pa.array([[1.0, 2.0]], type=pa.list_(pa.float32()))})
        with pytest.raises(ValueError, match="must be a fixed_size_list"):
            _text_matrix(batch, _TASK_TEXT_VECTOR)

    def test_an_all_null_column_decodes_to_zeros_and_no_presence(self) -> None:
        """Its child buffer can be zero-length, which no reshape could rebase."""
        batch = pa.table({_TASK_TEXT_VECTOR: pa.nulls(3, type=pa.list_(pa.float32(), TEXT_DIM))})
        matrix, present = _text_matrix(batch, _TASK_TEXT_VECTOR)
        assert not present.any()
        np.testing.assert_array_equal(matrix, np.zeros((3, TEXT_DIM), dtype=np.float32))

    def test_a_partly_null_column_reports_which_rows_carry_a_vector(self) -> None:
        """The presence mask is what stops a NULL slot becoming a label's representative."""
        rows = [[1.0] * TEXT_DIM, None, [2.0] * TEXT_DIM]
        batch = pa.table({_TASK_TEXT_VECTOR: pa.array(rows, type=pa.list_(pa.float32(), TEXT_DIM))})
        matrix, present = _text_matrix(batch, _TASK_TEXT_VECTOR)
        np.testing.assert_array_equal(present, np.array([True, False, True]))
        np.testing.assert_array_equal(matrix[0], np.ones(TEXT_DIM, dtype=np.float32))
        np.testing.assert_array_equal(matrix[2], np.full(TEXT_DIM, 2.0, dtype=np.float32))


class TestRequireMergeMatrix:
    """``merge_labels`` aliases only a float32 C-contiguous block; anything else it copies."""

    def _labels(self, count: int) -> tuple[str, ...]:
        return tuple(f"label-{index}" for index in range(count))

    def test_the_premise_the_guard_protects_is_that_a_conforming_buffer_is_aliased(self) -> None:
        """Why the guard is worth having at all: a conforming buffer is NOT copied.

        This pins numpy's behavior, not the guard's - the guard's power lives in
        the four rejection tests below. If ``ascontiguousarray`` ever started
        copying a conforming buffer, those rejections would still pass while
        guarding nothing, so the premise is asserted separately and named as one.
        """
        matrix = np.zeros((3, TEXT_DIM), dtype=np.float32)
        _require_merge_matrix(matrix, self._labels(3))
        assert np.ascontiguousarray(matrix, dtype=np.float32) is matrix

    def test_float64_is_refused(self) -> None:
        """``np.array`` of Python lists yields float64, and the resulting copy is silent."""
        matrix = np.zeros((2, TEXT_DIM), dtype=np.float64)
        with pytest.raises(ValueError, match="must be float32"):
            _require_merge_matrix(matrix, self._labels(2))  # type: ignore[arg-type]

    def test_a_non_contiguous_view_is_refused(self) -> None:
        """A sliced or transposed view is the other silent doubling of the driver peak."""
        matrix = np.zeros((4, TEXT_DIM), dtype=np.float32)[::2]
        with pytest.raises(ValueError, match="C-contiguous"):
            _require_merge_matrix(matrix, self._labels(2))

    def test_a_row_count_that_does_not_match_the_labels_is_refused(self) -> None:
        """Misalignment would merge labels against other labels' vectors, silently."""
        matrix = np.zeros((2, TEXT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match=r"must be \(3,"):
            _require_merge_matrix(matrix, self._labels(3))

    def test_a_mis_sized_width_is_refused(self) -> None:
        """The merge basis width follows the text model, never what happened to be stored."""
        matrix = np.zeros((2, TEXT_DIM // 2), dtype=np.float32)
        with pytest.raises(ValueError, match=f"must be \\(2, {TEXT_DIM}\\)"):
            _require_merge_matrix(matrix, self._labels(2))


class TestRequireBasisWidth:
    """A fitted basis is checked on the driver, before any worker scores against it."""

    def test_a_basis_of_the_scored_width_is_accepted(self) -> None:
        """The premise: the gate passes what the scan can actually use."""
        _require_basis_width(np.eye(3, TEXT_DIM, dtype=np.float32), TEXT_DIM, "subtask")

    def test_a_basis_of_the_wrong_width_is_refused_naming_which_one(self) -> None:
        """Two bases are fitted per run, so the message has to say which one is wrong."""
        centroids = np.eye(3, TEXT_DIM // 2, dtype=np.float32)
        with pytest.raises(ValueError, match="subtask centroids"):
            _require_basis_width(centroids, TEXT_DIM, "subtask")

    def test_a_zero_norm_centroid_is_refused_on_the_driver(self) -> None:
        """Otherwise the first scan task raises it, after the whole fit has been paid for."""
        centroids = np.zeros((2, TEXT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="~zero norm"):
            _require_basis_width(centroids, TEXT_DIM, "subtask")


class TestLabelGather:
    """The second read pass: ``O(labels)`` rows out, with a deterministic representative."""

    def _rollup(self, table: ClipsTable, *fragment_ids: int) -> pa.Table:
        source = _preflight(_config(table))
        return _label_rollup(_work_batch(*(fragment_ids or source.fragment_ids)), read=source.read)

    def test_one_row_per_distinct_label_carrying_its_clip_count(self, build_clips_table: BuildTable) -> None:
        """The fixture's uneven label cycle, reduced to two rows over twelve clips.

        Distinct means distinct CANONICAL label: the fixture spells each
        instruction several ways, so a rollup that keyed on the raw string would
        emit twelve rows here rather than two.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        rollup = self._rollup(table)
        counts = dict(
            zip(
                rollup.column(_GATHER_LABEL_COLUMN).to_pylist(),
                rollup.column(_GATHER_COUNT_COLUMN).to_pylist(),
                strict=True,
            )
        )
        assert counts == {CANONICAL_TASKS[0]: 6, CANONICAL_TASKS[1]: 6}

    def test_the_gather_reduces_only_the_task_vocabulary(self, build_clips_table: BuildTable) -> None:
        """No subtask label reaches the driver, which is the point of bounding level 2.

        A driver-side reduction over subtask prose is the ``O(L)`` state this key
        was changed to remove, and it grows with the corpus rather than with the
        annotation schema. Asserted against the fixture's own subtask vocabulary,
        so a rollup that started emitting one row per subtask again would show up
        here rather than only as memory pressure at scale.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        labels = set(self._rollup(table).column(_GATHER_LABEL_COLUMN).to_pylist())

        assert labels == set(CANONICAL_TASKS)
        assert not labels & {subtask for _task, subtask in LABEL_CYCLE}

    def test_the_gathered_clip_total_is_every_eligible_row(self, build_clips_table: BuildTable) -> None:
        """The merge reports its moved-clip count as a share of this total, documented as ELIGIBLE clips.

        Two properties of the fixture are what make the claim falsifiable, because
        they are the only two ways the counts could diverge. A fragment holding no
        eligible row is still read and must contribute nothing. A row whose vector
        is unusable IS claimed - it reaches the write as ``invalid_embedding`` -
        so it must still be counted here. A gather filtering either differently
        from the eligibility predicate would silently move the denominator the
        moved-clip warning is measured against.
        """
        table = build_clips_table(
            ClipsTableSpec(
                fragments=3,
                rows_per_fragment=4,
                empty_fragments=frozenset({1}),
                non_finite_vector_rows=frozenset({0}),
            )
        )
        source = _preflight(_config(table))
        rollup = _label_rollup(_work_batch(*source.fragment_ids), read=source.read)

        assert sum(_reduce_gather([rollup]).counts) == source.eligible_rows

    def test_the_rollup_is_the_declared_schema(self, build_clips_table: BuildTable) -> None:
        """Per-fragment rollups have to concatenate, so the schema is declared not inferred."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2))
        assert self._rollup(table).schema == _GATHER_ROW

    def test_the_representative_vector_belongs_to_the_lowest_clip_id(self, build_clips_table: BuildTable) -> None:
        """The tie-break that makes the merge basis independent of completion order.

        The shape matters: the first canonical task has to appear in more than one
        fragment, or the tie-break is never consulted and the assertion holds
        vacuously. At three rows per fragment the fixture's label cycle puts it in
        0 and 2.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=3))
        rollup = self._rollup(table)
        index = rollup.column(_GATHER_LABEL_COLUMN).to_pylist().index(CANONICAL_TASKS[0])
        winner = rollup.column(_GATHER_TIEBREAK_COLUMN)[index].as_py()
        assert winner == "clip-00-00"
        np.testing.assert_array_equal(
            np.asarray(rollup.column(_GATHER_VECTOR_COLUMN)[index].as_py(), dtype=np.float32),
            _fixture_text_vector(winner, table, _TASK_TEXT_VECTOR),
        )

    def test_the_reduction_is_independent_of_the_order_rollups_arrive_in(self, build_clips_table: BuildTable) -> None:
        """Ray guarantees no fragment order, so the merge basis must not depend on one."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=3))
        per_fragment = [self._rollup(table, fragment_id) for fragment_id in (0, 1, 2)]
        forward = _reduce_gather(per_fragment)
        reverse = _reduce_gather(list(reversed(per_fragment)))
        assert forward.labels == reverse.labels
        assert forward.counts == reverse.counts
        np.testing.assert_array_equal(forward.vectors, reverse.vectors)

    def test_the_reduced_matrix_is_the_shape_merge_labels_aliases(self, build_clips_table: BuildTable) -> None:
        """The reduction is the last place dtype and contiguity can still be got right."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        reduced = _reduce_gather([self._rollup(table)])
        assert reduced.vectors.dtype == np.float32
        assert reduced.vectors.flags.c_contiguous
        assert reduced.vectors.shape == (len(reduced.labels), TEXT_DIM)

    def test_a_gather_with_no_label_is_refused(self) -> None:
        """Every eligible row carries a task label, so an empty gather means rows were lost."""
        with pytest.raises(ValueError, match="no task label"):
            _reduce_gather([_GATHER_ROW.empty_table()])

    def test_a_label_reaching_the_merge_with_no_vector_is_reported(self, loguru_records: list[dict[str, Any]]) -> None:
        """It can merge with nothing, so it silently draws a fairness group of its own."""
        rollup = pa.table(
            {
                _GATHER_LABEL_COLUMN: pa.array(["task-0"], type=pa.string()),
                _GATHER_COUNT_COLUMN: pa.array([3], type=pa.int64()),
                _GATHER_VECTOR_COLUMN: pa.FixedSizeListArray.from_arrays(
                    pa.array(np.zeros(TEXT_DIM, dtype=np.float32), type=pa.float32()), TEXT_DIM
                ),
                _GATHER_TIEBREAK_COLUMN: pa.array([None], type=pa.string()),
                _GATHER_FRAGMENT_COLUMN: pa.array([None], type=pa.int64()),
            },
            schema=_GATHER_ROW,
        )
        _reduce_gather([rollup])
        assert any("no vector" in message for message in _warnings(loguru_records))

    def test_a_clip_id_shared_across_fragments_reduces_deterministically(self) -> None:
        """Two fragments may share a ``clip_id`` yet hold different vectors.

        ``clip_id`` alone is unique only within a fragment, so a reduction that
        tie-broke on it would keep whichever rollup Ray happened to fold first.
        The ``(clip_id, fragment_id)`` pair breaks that tie globally: the lower
        fragment id wins regardless of arrival order.
        """
        shared = "clip-shared"

        def _rollup(fragment_id: int, first_coordinate: float) -> pa.Table:
            vector = np.zeros(TEXT_DIM, dtype=np.float32)
            vector[0] = first_coordinate
            return pa.table(
                {
                    _GATHER_LABEL_COLUMN: pa.array(["task-0"], type=pa.string()),
                    _GATHER_COUNT_COLUMN: pa.array([1], type=pa.int64()),
                    _GATHER_VECTOR_COLUMN: pa.FixedSizeListArray.from_arrays(
                        pa.array(vector, type=pa.float32()), TEXT_DIM
                    ),
                    _GATHER_TIEBREAK_COLUMN: pa.array([shared], type=pa.string()),
                    _GATHER_FRAGMENT_COLUMN: pa.array([fragment_id], type=pa.int64()),
                },
                schema=_GATHER_ROW,
            )

        low = _rollup(1, first_coordinate=1.0)
        high = _rollup(4, first_coordinate=2.0)
        forward = _reduce_gather([low, high])
        reverse = _reduce_gather([high, low])
        np.testing.assert_array_equal(forward.vectors, reverse.vectors)
        assert forward.vectors[0, 0] == 1.0


class TestMergeMaps:
    """The driver-side merge: ONE call, over distinct task labels, at the config's theta."""

    def _label_set(self, matrix: npt.NDArray[np.float32]) -> _LabelSet:
        rows = matrix.shape[0]
        return _LabelSet(
            labels=tuple(f"task-{index}" for index in range(rows)),
            counts=tuple(range(rows, 0, -1)),
            vectors=np.ascontiguousarray(matrix, dtype=np.float32),
        )

    def _orthogonal(self, count: int) -> npt.NDArray[np.float32]:
        matrix = np.zeros((count, TEXT_DIM), dtype=np.float32)
        matrix[np.arange(count), np.arange(count)] = 1.0
        return matrix

    def _identical(self, count: int) -> npt.NDArray[np.float32]:
        matrix = np.zeros((count, TEXT_DIM), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix

    def test_orthogonal_labels_merge_into_nothing(self, build_clips_table: BuildTable) -> None:
        """R equals L, the worst case of the O(L * R) walk, and it has to be measurable."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        task_map, stats = _merge_maps(_config(table), self._label_set(self._orthogonal(3)))
        assert set(task_map.values()) == set(task_map)
        assert (stats.labels_in, stats.labels_out) == (3, 3)

    def test_identical_label_vectors_collapse_onto_the_most_frequent(self, build_clips_table: BuildTable) -> None:
        """The walk runs count-descending, so the wording the corpus uses most leads."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        task_map, stats = _merge_maps(_config(table), self._label_set(self._identical(3)))
        assert set(task_map.values()) == {"task-0"}
        assert stats.labels_out == 1

    def test_theta_comes_from_the_config(self, build_clips_table: BuildTable) -> None:
        """Nothing here derives, defaults or clamps theta.

        Asserted as a pair of runs over ONE label matrix, because a single run
        cannot distinguish "read the config" from "used a constant that happens to
        agree with it".
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        near = np.zeros((2, TEXT_DIM), dtype=np.float32)
        near[:, 0] = 1.0
        near[1, 1] = 0.4  # cosine ~0.93: merges at theta 0.9, not at 0.99
        label_set = self._label_set(near)

        _low, permissive = _merge_maps(_config(table, merge_theta_task=0.9), label_set)
        _high, strict = _merge_maps(_config(table, merge_theta_task=0.99), label_set)

        assert (permissive.labels_out, strict.labels_out) == (1, 2)

    def test_folding_one_large_label_re_pools_the_corpus_while_the_vocabulary_barely_moves(
        self,
        build_clips_table: BuildTable,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """The two merge warnings measure different things, and only the corpus one fires here.

        One of twelve labels is folded away, so the vocabulary ratio is nowhere
        near halving and ``fairness.merge_labels`` stays silent - yet the folded
        label holds 13% of the clips, which is the share every surviving group's
        quota was computed against. A single warning derived from ``L`` and ``R``
        cannot see this case at all.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        vectors_matrix = self._orthogonal(12)
        vectors_matrix[1] = vectors_matrix[0]  # the second label folds onto the first
        label_set = _LabelSet(
            labels=tuple(f"task-{index}" for index in range(12)),
            counts=(100, 30, *([10] * 10)),
            vectors=vectors_matrix,
        )

        _map, stats = _merge_maps(_config(table), label_set)

        assert (stats.labels_in, stats.labels_out, stats.clips_moved) == (12, 11, 30)
        assert [message for message in _warnings(loguru_records) if "re-pooled" in message] != []
        assert [message for message in _warnings(loguru_records) if "collapsed" in message] == []

    def test_a_merge_that_folds_nothing_moves_no_clip(
        self, build_clips_table: BuildTable, loguru_records: list[dict[str, Any]]
    ) -> None:
        """The counterfactual for the line above: no fold, so the share is zero and quiet.

        Without this a ``clips_moved`` hardwired to the corpus size would satisfy
        the assertions above.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))

        _map, stats = _merge_maps(_config(table), self._label_set(self._orthogonal(4)))

        assert stats.clips_moved == 0
        assert [message for message in _warnings(loguru_records) if "re-pooled" in message] == []

    def test_the_moved_clip_count_is_the_folded_labels_own_clips(self) -> None:
        """``clips_moved`` sums the counts of the labels the map re-pointed, not the leaders'.

        Counting the leader's clips as well would report the whole merged group,
        which at ``theta`` low enough to fold everything would read as 100% moved
        on a corpus where every clip kept its own leader's group.
        """
        label_set = _LabelSet(
            labels=("leader", "folded", "alone"),
            counts=(100, 7, 3),
            vectors=self._orthogonal(3),
        )

        assert _clips_moved(label_set, {"leader": "leader", "folded": "leader", "alone": "alone"}) == 7

    def test_the_merge_reports_its_own_wall_clock(self, build_clips_table: BuildTable) -> None:
        """The only instrument for the merge's realized cost; R alone cannot bound it.

        Bounded above by the call it is supposed to be timing, which catches a stat
        measured over the wrong span - process uptime, say. At this fixture's scale
        a real merge rounds to roughly zero, so this cannot distinguish a measured
        zero from a hardcoded one; it pins the span, not the value.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        before = time.monotonic()
        _map, stats = _merge_maps(_config(table), self._label_set(self._orthogonal(2)))
        elapsed = time.monotonic() - before
        assert math.isfinite(stats.seconds)
        assert 0.0 <= stats.seconds <= elapsed


def _fit_spec(source: _Source, config: CurateConfig, *, requested_k: int, cap: int | None = None) -> _FitSpec:
    """Build the fit spec one test needs, defaulting every bound to the config's.

    The caps come off the sampled prefix rather than straight off the config, so a
    test reads the same eligible-bounded numbers the driver would pass.
    """
    row_cap = cap if cap is not None else config.fit_sample_rows
    prefix = _fit_sample_fragments(source, row_cap, _subtask_sample_rows(config.subtask_clusters, row_cap))
    return _FitSpec(
        fragment_ids=prefix.fragment_ids,
        weights=_weight_map(config),
        requested_k=requested_k,
        subtask_k=config.subtask_clusters,
        sample_row_cap=prefix.sample_row_cap,
        subtask_sample_row_cap=prefix.subtask_sample_row_cap,
        random_state=config.kmeans_random_state,
    )


def _cpu_centroids(source: _Source, config: CurateConfig, *, cap: int | None = None) -> npt.NDArray[np.float32]:
    """Return the closed-form ``k == 1`` k-means basis for one table's fit sample.

    The sample mean IS the single-cluster solution, so the scan can be exercised
    against a real fitted basis without the GPU fit.
    """
    sample = _fit_sample(source.read, _fit_spec(source, config, requested_k=1, cap=cap))
    return np.ascontiguousarray(sample.fused.mean(axis=0, keepdims=True), dtype=np.float32)


def _cpu_subtask_centroids(source: _Source, config: CurateConfig) -> npt.NDArray[np.float32] | None:
    """Return a level-2 basis holding one centroid per subtask region, or ``None`` for none.

    The fixture draws its subtask vectors from ``SUBTASK_REGIONS`` bundles, so the
    per-bundle mean IS the k-means solution at that k, and the scan can be
    exercised against a real level-2 basis without the GPU fit. Read off the whole
    table by global row index rather than off the fit sample, because the sample
    holds only USABLE rows and a fixture with a zero-norm row would then shift
    every later row into the wrong bundle's mean.

    ``None`` for the two reasons ``_fit_sample`` returns no subtask matrix: an
    unweighted block is not read at all, and a weighted one whose population holds
    no direction leaves nothing to average. The second is unreachable through this
    fixture - a table whose subtask vectors are all unusable fails the locality fit
    first - so it is a guard against a bare ``np.stack([])`` rather than a covered
    branch.
    """
    if _weight_map(config)[SUBTASK_WEIGHT_FIELD] <= 0.0:
        return None
    rows = lance.dataset(source.read.uri, version=source.read.read_version).to_table(columns=[SUBTASK_VECTOR_COLUMN])
    matrix, present = _text_matrix(rows, SUBTASK_VECTOR_COLUMN)
    unit, usable = vectors.unit_rows(matrix)
    usable &= present
    regions = np.array([subtask_region(row) for row in range(rows.num_rows)])
    # Emit one centroid per POPULATED region only. An empty region's mean is NaN,
    # and a real GPU k-means never returns an empty-cluster centroid, so a
    # synthetic basis that stacked the empty regions too would feed a non-finite
    # row into CentroidAssigner.from_raw that no real fit could produce. A fully
    # populated fixture skips nothing, so the region-to-cell bijection the
    # partition tests assert is unchanged.
    means = [
        unit[members].mean(axis=0)
        for region in range(SUBTASK_REGIONS)
        if (members := usable & (regions == region)).any()
    ]
    if not means:
        return None
    return np.ascontiguousarray(np.stack(means), dtype=np.float32)


def _two_centroid_basis(source: _Source, config: CurateConfig) -> npt.NDArray[np.float32]:
    """Return a ``k == 2`` basis from the sample's own extremes, so both clusters land rows.

    Each centroid IS a sample row, so that row scores 1.0 against it and cannot be
    assigned elsewhere - which is what makes the partition non-empty on both sides
    without depending on how the fixture's random vectors happen to fall.
    """
    sample = _fit_sample(source.read, _fit_spec(source, config, requested_k=2)).fused
    farthest = int(np.argmin(sample @ sample[0]))
    return np.ascontiguousarray(np.stack([sample[0], sample[farthest]]), dtype=np.float32)


def _scan_table(config: CurateConfig) -> tuple[pa.Table, _Source, npt.NDArray[np.float32]]:
    """Preflight, fit a single-cluster basis, and scan every fragment of one table."""
    source = _preflight(config)
    centroids = _cpu_centroids(source, config)
    scanned = _scan_fragments(
        _work_batch(*source.fragment_ids),
        source.read,
        _weight_map(config),
        _ScanBases.from_raw(centroids, _cpu_subtask_centroids(source, config)),
    )
    return scanned, source, centroids


class TestScan:
    """The one row-scale read: labels, working vector, cluster and distance per eligible row."""

    def _scan(self, table: ClipsTable) -> tuple[pa.Table, _Source, npt.NDArray[np.float32]]:
        return _scan_table(_config(table))

    def test_one_row_per_eligible_row_in_fragment_order(self, build_clips_table: BuildTable) -> None:
        """The scan neither drops nor duplicates a claimed row."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=2))
        scanned, source, _centroids = self._scan(table)
        assert scanned.num_rows == source.eligible_rows
        assert scanned.column(KEY_COLUMN).to_pylist() == list(table.all_clip_ids)

    def test_the_scan_row_schema_is_declared(self, build_clips_table: BuildTable) -> None:
        """Blocks from different fragments must concatenate, an empty fragment's included."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2))
        scanned, _source, _centroids = self._scan(table)
        assert scanned.schema == _SCAN_ROW

    def test_each_row_is_stamped_with_the_fragment_it_was_read_from(self, build_clips_table: BuildTable) -> None:
        """``__frag`` is what routes a verdict back to the rows it belongs to."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=2))
        scanned, _source, _centroids = self._scan(table)
        assert scanned.column(FRAGMENT_COLUMN).to_pylist() == [0, 0, 1, 1, 2, 2]

    def test_an_empty_fragment_contributes_no_row(self, build_clips_table: BuildTable) -> None:
        """A fragment holding no eligible row is skipped, not represented by a NULL row."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=2, empty_fragments=frozenset({1})))
        scanned, _source, _centroids = self._scan(table)
        assert scanned.column(FRAGMENT_COLUMN).to_pylist() == [0, 0, 2, 2]

    def test_the_canonical_task_label_is_produced_here(self, build_clips_table: BuildTable) -> None:
        """The scan is the sole producer of the canonical task label column."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=2))
        scanned, _source, _centroids = self._scan(table)
        assert scanned.column(CANONICAL_TASK_COLUMN).to_pylist() == [CANONICAL_TASKS[0]] * 2

    def test_the_scan_folds_every_spelling_of_one_task_onto_one_key(self, build_clips_table: BuildTable) -> None:
        """Level-1 fairness groups on meanings, so the scan has to erase the wording differences.

        The fixture writes each instruction several ways - mixed case, a doubled
        space, a decomposed circumflex, a trailing mark - so all six raw task
        strings are distinct while only two canonical keys exist. Asserting the
        canonical column alone cannot see that, because it holds equally over a
        column that arrived pre-folded; the raw-side assertion is what gives the
        canonical-side one its power.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=len(LABEL_CYCLE)))
        scanned, _source, _centroids = self._scan(table)

        assert len({task for task, _subtask in LABEL_CYCLE}) == len(LABEL_CYCLE)
        assert set(scanned.column(CANONICAL_TASK_COLUMN).to_pylist()) == set(CANONICAL_TASKS)

    def test_the_level_two_key_partitions_the_subtask_text_not_the_subtask_prose(
        self, build_clips_table: BuildTable
    ) -> None:
        """Two rows share a level-2 cell exactly when their subtask VECTORS bundle together.

        This is the whole substitution: the level-2 key used to be the subtask
        string, whose vocabulary grows with the corpus. The fixture's bundles cross
        its subtask prose, so a key that had kept reading the prose would group
        these rows differently - and that crossing is asserted, because "two rows
        share a cell" holds trivially for a key that returned a constant.
        """
        rows = len(LABEL_CYCLE)
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=rows))
        scanned, _source, _centroids = self._scan(table)

        cells = scanned.column(SUBTASK_CLUSTER_COLUMN).to_pylist()
        by_region: dict[int, set[int]] = {}
        for row, cell in enumerate(cells):
            by_region.setdefault(subtask_region(row), set()).add(cell)
        assert all(len(observed) == 1 for observed in by_region.values())
        assert len({next(iter(observed)) for observed in by_region.values()}) == SUBTASK_REGIONS

        prose = [LABEL_CYCLE[row % len(LABEL_CYCLE)][1] for row in range(rows)]
        shared = {(one, two) for one, two in itertools.combinations(range(rows), 2) if cells[one] == cells[two]}
        assert shared
        assert all(prose[one] != prose[two] for one, two in shared)

    def test_every_level_two_cell_is_within_the_fitted_basis(self, build_clips_table: BuildTable) -> None:
        """The bound is the point: G is at most tasks x (k + 1) whatever the corpus holds."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        scanned, _source, _centroids = self._scan(table)
        assert set(scanned.column(SUBTASK_CLUSTER_COLUMN).to_pylist()) <= set(range(SUBTASK_REGIONS))

    def test_an_unusable_subtask_vector_takes_the_reserved_cell_rather_than_a_null(
        self, build_clips_table: BuildTable
    ) -> None:
        """Ray Data cannot shuffle a NULL group key, so an unusable row still gets one.

        Consistent with the eligibility contract rather than a new rule: the row is
        already reasoned ``invalid_embedding``, and it takes the reserved cell for
        exactly the reason it takes ``NO_DEDUP_GROUP`` - it has no direction to
        assign.
        """
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=3, zero_norm_vector_rows=frozenset({1}))
        scanned, _source, _centroids = self._scan(build_clips_table(spec))

        cells = scanned.column(SUBTASK_CLUSTER_COLUMN).to_pylist()
        assert cells[1] == NO_SUBTASK_CLUSTER
        assert scanned.column(SUBTASK_CLUSTER_COLUMN).null_count == 0
        assert NO_SUBTASK_CLUSTER not in {cells[0], cells[2]}

    def test_a_run_with_no_level_two_basis_puts_every_row_in_the_reserved_cell(
        self, build_clips_table: BuildTable
    ) -> None:
        """At a zero subtask weight no basis is fitted, and level 2 collapses to one cell.

        The de-weighting escape is meant to remove the modality from the metric,
        not to fail the run: fairness then reduces to the task level alone, which
        the quota's degeneracy WARNING reports if it matters.

        Routed through the shared CPU walk rather than by handing the scan a
        ``None`` basis directly, so it pins that the walk DERIVES the absent basis
        from the weights the way the fit does. Wired by hand, this passes over a
        walk that would have built a synthetic basis production never fits.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=4))
        config = _config(table, weights=ModalityWeights(subtask=0.0, image=0.5, action=0.5))
        scanned, _source, _centroids = _scan_table(config)
        assert set(scanned.column(SUBTASK_CLUSTER_COLUMN).to_pylist()) == {NO_SUBTASK_CLUSTER}

    def test_valid_rows_are_unit_norm_and_share_the_single_cluster(self, build_clips_table: BuildTable) -> None:
        """Cosine similarity IS the dot product only while the working vector is unit-norm."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=2))
        scanned, _source, _centroids = self._scan(table)
        working = vectors.vector_column_to_matrix(scanned.column(WORKING_VECTOR_COLUMN))
        np.testing.assert_allclose(np.linalg.norm(working, axis=1), 1.0, rtol=0, atol=1e-6)
        assert set(scanned.column(DEDUP_KEY_COLUMN).to_pylist()) == {0}

    def test_the_distance_is_the_cosine_distance_to_the_assigned_centroid(self, build_clips_table: BuildTable) -> None:
        """Recomputed independently: a mis-scaled distance reorders every selection."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=3))
        scanned, _source, centroids = self._scan(table)
        working = vectors.vector_column_to_matrix(scanned.column(WORKING_VECTOR_COLUMN))
        unit = centroids[0] / np.linalg.norm(centroids[0])
        np.testing.assert_allclose(
            np.asarray(scanned.column(DISTANCE_COLUMN).to_numpy(zero_copy_only=False), dtype=np.float32),
            1.0 - working @ unit,
            rtol=0,
            atol=1e-6,
        )

    def test_a_zero_norm_row_is_reasoned_and_routed_past_the_similarity_pass(
        self, build_clips_table: BuildTable
    ) -> None:
        """It has no direction, so it must reach neither a GEMM nor a cluster."""
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=3, zero_norm_vector_rows=frozenset({1}))
        scanned, _source, _centroids = self._scan(build_clips_table(spec))
        assert scanned.column(CURATE_SELECTION_REASON).to_pylist() == [
            None,
            str(CurateReason.INVALID_EMBEDDING),
            None,
        ]
        assert scanned.column(DEDUP_KEY_COLUMN).to_pylist()[1] == NO_DEDUP_GROUP_ZERO_NORM
        assert scanned.column(DISTANCE_COLUMN)[1].as_py() is None
        working = vectors.vector_column_to_matrix(scanned.column(WORKING_VECTOR_COLUMN))
        np.testing.assert_array_equal(working[1], np.zeros(vectors.FUSED_DIM, dtype=np.float32))

    def test_the_two_bypass_causes_take_different_sentinels(self, build_clips_table: BuildTable) -> None:
        """One key per cause is what makes the bypass INFO line a per-cause corpus count.

        Both sentinels are negative, so every consumer routes them identically and
        the split is observable only in the key itself. Asserted on one scan
        holding both causes, because two scans each holding one cannot show that
        the keys differ.
        """
        spec = ClipsTableSpec(
            fragments=1,
            rows_per_fragment=4,
            zero_norm_vector_rows=frozenset({1}),
            non_finite_vector_rows=frozenset({2}),
        )
        scanned, _source, _centroids = self._scan(build_clips_table(spec))

        keys = scanned.column(DEDUP_KEY_COLUMN).to_pylist()
        assert keys[1] == NO_DEDUP_GROUP_ZERO_NORM
        assert keys[2] == NO_DEDUP_GROUP
        assert keys[1] != keys[2]
        assert keys[1] < 0
        assert keys[2] < 0

    def test_a_non_finite_row_is_reasoned_rather_than_normalized(self, build_clips_table: BuildTable) -> None:
        """An inf row has infinite norm, passes a norm-only gate, and normalizes to NaN.

        Routing is asserted alongside finiteness, because a row normalized to
        finite garbage satisfies a finiteness check on its own.
        """
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=3, non_finite_vector_rows=frozenset({2}))
        scanned, _source, _centroids = self._scan(build_clips_table(spec))
        assert scanned.column(CURATE_SELECTION_REASON).to_pylist()[2] == str(CurateReason.INVALID_EMBEDDING)
        assert scanned.column(DEDUP_KEY_COLUMN).to_pylist()[2] == NO_DEDUP_GROUP
        assert scanned.column(DISTANCE_COLUMN)[2].as_py() is None
        working = vectors.vector_column_to_matrix(scanned.column(WORKING_VECTOR_COLUMN))
        assert np.isfinite(working).all()
        np.testing.assert_array_equal(working[2], np.zeros(vectors.FUSED_DIM, dtype=np.float32))

    def test_each_row_is_assigned_to_its_nearest_centroid(self, build_clips_table: BuildTable) -> None:
        """The dedup key IS the cluster id, and it is what bounds the retention GEMM.

        Scanned against a hand-built TWO-centroid basis. Every other test here
        fits ``k == 1``, where "routed to its nearest centroid" and "wrote 0" are
        indistinguishable, so nothing would catch a hardcoded cluster.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=6))
        config = _config(table)
        source = _preflight(config)
        basis = _two_centroid_basis(source, config)
        scanned = _scan_fragments(
            _work_batch(*source.fragment_ids),
            source.read,
            _weight_map(config),
            _ScanBases.from_raw(basis, None),
        )
        working = vectors.vector_column_to_matrix(scanned.column(WORKING_VECTOR_COLUMN))
        unit = basis / np.linalg.norm(basis, axis=1, keepdims=True)
        expected = np.argmax(working @ unit.T, axis=1)
        assert scanned.column(DEDUP_KEY_COLUMN).to_pylist() == expected.tolist()
        assert set(expected.tolist()) == {0, 1}

    def test_the_clusters_partition_the_rows_the_retention_pass_compares(self, build_clips_table: BuildTable) -> None:
        """Retention runs per cluster, so the basis decides which pairs are ever compared.

        At ``k == 2`` the scan must hand the GEMM two disjoint groups covering
        every valid row, and a duplicate pair inside one of them must still be
        marked. Byte-identical vectors cannot be split by a nearest-centroid
        assignment, so what is observable here is the partition, not a missed pair.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=6, duplicate_vectors=(0, 3)))
        config = _config(table)
        source = _preflight(config)
        scanned = _scan_fragments(
            _work_batch(*source.fragment_ids),
            source.read,
            _weight_map(config),
            _ScanBases.from_raw(_two_centroid_basis(source, config), None),
        )
        groups = _split_by(scanned, [DEDUP_KEY_COLUMN])
        assert len(groups) == 2
        assert sum(group.num_rows for group in groups) == scanned.num_rows
        deduped = pa.concat_tables([dedup.mark_duplicates(group, eps=config.dedup_eps) for group in groups])
        reasons = deduped.column(CURATE_SELECTION_REASON).to_pylist()
        assert reasons.count(str(CurateReason.DUPLICATE)) == 1

    def test_a_pathological_row_is_reported_to_the_driver(
        self, build_clips_table: BuildTable, loguru_records: list[dict[str, Any]]
    ) -> None:
        """How many rows were committed as invalid is observable only from this log line."""
        spec = ClipsTableSpec(fragments=1, rows_per_fragment=3, zero_norm_vector_rows=frozenset({0}))
        table = build_clips_table(spec)
        self._scan(table)
        assert any(str(CurateReason.INVALID_EMBEDDING) in message for message in _warnings(loguru_records))


class TestBuildQuota:
    """The quota is allocated over survivors only, and sums to the resolved target."""

    def _counts(self, rows: Sequence[tuple[str, int, str | None, int]]) -> pa.Table:
        return pa.table(
            {
                CANONICAL_TASK_COLUMN: pa.array([row[0] for row in rows], type=pa.string()),
                SUBTASK_CLUSTER_COLUMN: pa.array([row[1] for row in rows], type=pa.int32()),
                CURATE_SELECTION_REASON: pa.array([row[2] for row in rows], type=pa.string()),
                RAY_COUNT_COLUMN: pa.array([row[3] for row in rows], type=pa.int64()),
            }
        )

    def test_a_run_with_no_survivor_is_refused(self, build_clips_table: BuildTable) -> None:
        """Committing it would publish "nothing was selected" as a successful run."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        with pytest.raises(ValueError, match="no row survived"):
            _build_quota(_config(table), self._counts([("t", 0, str(CurateReason.DUPLICATE), 4)]))

    def test_the_refusal_does_not_blame_a_pass_that_may_not_have_run(self, build_clips_table: BuildTable) -> None:
        """With ``dedup_eps`` unset every reasoned row is invalid, so naming dedup misdirects.

        The message is asserted not to name the pass at all, rather than to name
        the right one per path: the raise sees only survivor counts and cannot
        tell which path produced them.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        counts = self._counts([("t", 0, str(CurateReason.INVALID_EMBEDDING), 4)])
        with pytest.raises(ValueError, match="no row survived") as raised:
            _build_quota(_config(table, dedup_eps=None), counts)
        assert "dedup" not in str(raised.value)
        assert "duplicat" not in str(raised.value)

    def test_reasoned_rows_neither_consume_nor_fund_a_quota(self, build_clips_table: BuildTable) -> None:
        """The target's denominator is the survivor population, not the eligible rows."""
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        counts = self._counts([("t", 0, None, 10), ("t", 0, str(CurateReason.DUPLICATE), 90)])
        _quota, target = _build_quota(_config(table, target=SelectionTarget(target_fraction=0.5)), counts)
        assert target == 5

    def test_the_allocation_is_nested_and_sums_to_the_target(self, build_clips_table: BuildTable) -> None:
        """Places are split across TASKS first, then within each task across its cells.

        Asserted per task, because the grand total cannot tell the nesting apart:
        a flat water-fill over the three pairs also sums to 6 and also respects
        the cap of 2, but spends 2/2/2 and gives the one-cell task only a third.
        The per-task split is what shows that ``_build_quota`` handed the
        capacities to the right level. Which of ``t``'s two cells takes the odd
        row is the residual order's call and is pinned in ``test_fairness``.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        counts = self._counts([("t", 0, None, 2), ("t", 1, None, 8), ("u", 0, None, 5)])
        quota, target = _build_quota(_config(table, target=SelectionTarget(target_count=6)), counts)
        quotas = quota.quotas()
        assert target == 6
        assert sorted(value for (task, _cell), value in quotas.items() if task == "t") == [1, 2]
        assert quotas[("u", 0)] == 3

    def test_the_configured_residual_seed_reaches_the_allocation(self, build_clips_table: BuildTable) -> None:
        """The config field must decide which tasks are funded, not just exist.

        The allocator's own default seed is also ``0``, so dropping this argument
        at the call site changes no result and leaves every fairness test passing.
        Driving it from a config is the only thing that pins the wiring.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        counts = self._counts([(f"task {index}", 0, None, 4) for index in range(8)])

        def funded(seed: int) -> frozenset[str]:
            config = _config(table, target=SelectionTarget(target_count=4), fairness_residual_seed=seed)
            quota, _target = _build_quota(config, counts)
            return frozenset(task for (task, _cell), allotted in quota.quotas().items() if allotted > 0)

        assert len({funded(seed) for seed in range(8)}) > 1


class TestUnfundedGroups:
    """How many fairness groups the target could not reach, reported and returned."""

    def _allotment(self, allotments: Sequence[int]) -> tuple[fairness.FairnessQuota, dict[fairness.Level2Key, int]]:
        """Return a quota object and a hand-written allotment map over its keys.

        The map is written rather than allocated because these tests are about
        the driver's line, not the water-fill that produced the numbers in it.
        The quota is built at target zero for the same reason: that silences
        ``FairnessQuota``'s own degeneracy WARNING, whose threshold is pinned
        against the allocator that owns it, leaving only the count and share
        this helper's callers assert on.
        """
        quotas = {("t", cell): allotted for cell, allotted in enumerate(allotments)}
        quota = fairness.FairnessQuota.build(
            level2_keys=list(quotas),
            level2_counts=[max(allotted, 1) for allotted in allotments],
            target=0,
        )
        return quota, quotas

    def test_the_count_is_the_number_of_groups_holding_a_zero_quota(self) -> None:
        """It is the read-out for a level-2 resolution the target cannot fund."""
        assert _report_unfunded(*self._allotment([3, 0, 1, 0, 0])) == 3

    def test_the_count_comes_from_the_quota_and_is_not_re_derived(
        self,
        monkeypatch: pytest.MonkeyPatch,
        loguru_records: list[dict[str, Any]],
    ) -> None:
        """The allocator owns the zero-quota predicate; the driver only reports it.

        Pinned against a stubbed count that the map itself contradicts, because
        a driver re-deriving the number from the map would agree with fairness
        today and could stop agreeing the moment either definition moved.
        """
        monkeypatch.setattr(fairness.FairnessQuota, "unfunded_groups", lambda _self, _quotas: 99)

        assert _report_unfunded(*self._allotment([1, 1])) == 99
        assert any("99 of 2 group(s)" in message for message in _infos(loguru_records))

    def test_the_count_and_the_share_are_both_reported_at_any_non_zero_value(
        self, loguru_records: list[dict[str, Any]]
    ) -> None:
        """One unfunded group of a thousand and one of two are the same count and different news.

        Reported unconditionally rather than above a threshold, so the share is
        what makes the count readable. Asserted on a single unfunded group,
        because that is the smallest non-zero value a threshold rule would drop.
        """
        _report_unfunded(*self._allotment([1, 1, 1, 0]))

        assert any("1 of 4 group(s) (25.0%) received no quota" in message for message in _infos(loguru_records))

    def test_a_fully_funded_allocation_still_reports_its_zero(self, loguru_records: list[dict[str, Any]]) -> None:
        """Silence would be indistinguishable from the line never having been reached."""
        assert _report_unfunded(*self._allotment([2, 2])) == 0

        assert any("0 of 2 group(s) (0.0%) received no quota" in message for message in _infos(loguru_records))

    def test_an_empty_group_set_is_reported_without_dividing_by_it(self, loguru_records: list[dict[str, Any]]) -> None:
        """Unreachable through ``_build_quota``, which refuses first, but the share divides by G."""
        assert _report_unfunded(*self._allotment([])) == 0

        assert any("0 of 0 group(s) (0.0%) received no quota" in message for message in _infos(loguru_records))


class TestStageOrder:
    """Where the label merge sits relative to the count that funds the quotas."""

    def test_the_group_count_is_taken_after_the_labels_are_merged(self, build_clips_table: BuildTable) -> None:
        """Counting first would fund the pre-merge vocabulary, so the cut would find no quota.

        Asserted as a value: collapsing the task labels has to change the quota's
        group STRUCTURE. If the count ran on unmerged labels, the merged key the
        cut addresses would be absent from the map it was allocated in.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        config = _config(table)
        scanned, _source, _centroids = _scan_table(config)
        tasks = sorted(set(scanned.column(CANONICAL_TASK_COLUMN).to_pylist()))
        keys = [CANONICAL_TASK_COLUMN, SUBTASK_CLUSTER_COLUMN, CURATE_SELECTION_REASON]

        unmerged, _target = _build_quota(config, _count_table(scanned, keys))
        merged_rows = fairness.apply_label_merge(scanned, dict.fromkeys(tasks, tasks[0]))
        merged, _merged_target = _build_quota(config, _count_table(merged_rows, keys))
        assert len(unmerged.level1_keys) == len(tasks) == 2
        assert merged.level1_keys == (tasks[0],)


class TestAggregateBlockConcat:
    """Reading one aggregate back from partitions that typed a NULL group key differently."""

    def _block(self, reasons: Sequence[str | None], reason_type: pa.DataType) -> pa.Table:
        """One aggregate output partition, its reason column typed by the caller."""
        return pa.table(
            {
                CURATE_SELECTION_REASON: pa.array(reasons, type=reason_type),
                RAY_COUNT_COLUMN: pa.array(range(1, len(reasons) + 1), type=pa.int64()),
            }
        )

    def test_a_partition_that_saw_only_null_keys_joins_one_that_saw_a_value(self) -> None:
        """The unreasoned partition's lost type is restored from the reasoned partition's.

        The shape every run produces: nearly every row's reason is NULL, so a
        partition holding no reasoned row types the column ``null`` and Arrow
        refuses to concatenate it with a sibling that typed it ``string``.
        """
        unreasoned = self._block([None, None], pa.null())
        reasoned = self._block(["duplicate"], pa.string())

        table = _concat_aggregate_blocks([unreasoned, reasoned])

        assert table.schema.field(CURATE_SELECTION_REASON).type == pa.string()
        assert table.column(CURATE_SELECTION_REASON).to_pylist() == [None, None, "duplicate"]

    def test_a_type_no_partition_observed_is_not_invented(self) -> None:
        """Every partition holding only NULL keys leaves the type genuinely unknown.

        Promotion needs a sibling's evidence, so with none the column stays
        ``null`` and the reason gate downstream sees the NULLs it must refuse -
        rather than a ``string`` guess that would make the refusal look like data.
        """
        table = _concat_aggregate_blocks([self._block([None], pa.null()), self._block([None, None], pa.null())])

        assert table.schema.field(CURATE_SELECTION_REASON).type == pa.null()
        assert table.num_rows == 3

    def test_two_partitions_naming_different_concrete_types_are_refused(self) -> None:
        """A concrete disagreement is schema drift, and widening it would hide a real defect.

        ``string`` against ``large_string`` is the pair a permissive promotion
        would fold silently; only ``null`` is allowed to be promoted here.
        """
        with pytest.raises(ValueError, match="schema drift"):
            _concat_aggregate_blocks(
                [self._block(["duplicate"], pa.string()), self._block(["selected"], pa.large_string())]
            )

    def test_an_empty_partition_contributes_no_schema(self) -> None:
        """A partition past the last group is schemaless, so it cannot be read for field names.

        Ray gives an empty partition zero COLUMNS, not a zero-row copy of its
        siblings' schema, and the partition count far exceeds the group count -
        so a schemaless block arriving FIRST is the ordinary case. Reading field
        names from it drops every column and the loss surfaces only downstream.
        """
        table = _concat_aggregate_blocks([pa.table({}), self._block(["selected"], pa.string()), pa.table({})])

        assert table.schema.names == [CURATE_SELECTION_REASON, RAY_COUNT_COLUMN]
        assert table.column(CURATE_SELECTION_REASON).to_pylist() == ["selected"]

    def test_a_populated_partition_missing_a_field_is_refused(self) -> None:
        """A block with rows but a narrower field set is drift, not an empty partition.

        The distinction matters because empty blocks are dropped silently: a
        populated block must not get the same tolerance, or a genuinely lost
        column would be dropped with it.
        """
        with pytest.raises(ValueError, match="different field sets"):
            _concat_aggregate_blocks(
                [
                    self._block(["duplicate"], pa.string()),
                    self._block(["selected"], pa.string()).drop_columns([CURATE_SELECTION_REASON]),
                ]
            )

    def test_an_aggregate_that_returned_no_rows_is_refused(self) -> None:
        """Both callers reduce a non-empty set, so all-empty partitions mean rows were lost."""
        with pytest.raises(ValueError, match="returned no rows across 2 output partition"):
            _concat_aggregate_blocks([pa.table({}), pa.table({})])

    def test_an_aggregate_that_produced_no_block_is_refused(self) -> None:
        """Distinct from every partition being empty, and neither may reach the caller as a table."""
        with pytest.raises(ValueError, match="returned no rows across 0 output partition"):
            _concat_aggregate_blocks([])


class TestReasonCounts:
    """The pre-commit gate: every claimed row leaves carrying exactly one reason."""

    def _reasons(self, rows: Sequence[tuple[str | None, int]]) -> pa.Table:
        return pa.table(
            {
                CURATE_SELECTION_REASON: pa.array([row[0] for row in rows], type=pa.string()),
                RAY_COUNT_COLUMN: pa.array([row[1] for row in rows], type=pa.int64()),
            }
        )

    def test_an_unreasoned_row_blocks_the_commit(self) -> None:
        """A NULL reason on a claimed row means some stage failed to judge it."""
        with pytest.raises(ValueError, match="no reason"):
            _reason_counts(self._reasons([(str(CurateReason.SELECTED), 3), (None, 1)]), 4)

    def test_a_verdict_set_that_changed_cardinality_blocks_the_commit(self) -> None:
        """The commit is all-or-nothing, so a lost row is caught before it is published."""
        with pytest.raises(ValueError, match="a stage changed cardinality"):
            _reason_counts(self._reasons([(str(CurateReason.SELECTED), 3)]), 4)

    def test_the_counts_are_returned_per_reason(self) -> None:
        """The run's only report of what it published, since nothing is stamped on the table."""
        counts = _reason_counts(self._reasons([(str(CurateReason.SELECTED), 3), (str(CurateReason.DUPLICATE), 1)]), 4)
        assert counts == {str(CurateReason.SELECTED): 3, str(CurateReason.DUPLICATE): 1}


# The producer identities the artifact tests archive. Deliberately spelled so the
# columns and the identities sort in OPPOSITE orders: the archive stores them as
# two arrays parallel by position, so an implementation that sorted each array
# independently would still round-trip a pair whose two sort orders agreed.
_PRODUCERS_UNDER_TEST: Mapping[str, str] = {
    ACTION_COLUMN_GROUP.provenance_columns[-1]: "zeta-basis",
    TEXT_COLUMN_GROUP.provenance_columns[0]: "alpha-model",
}


class TestCentroidsArtifact:
    """The basis is published before the commit, keyed by the hash of its own bytes."""

    def _fit(self, *, subtask: bool = True) -> _FitResult:
        """Build a fit result whose two bases are distinguishable from each other."""
        centroids = np.arange(2 * vectors.FUSED_DIM, dtype=np.float32).reshape(2, vectors.FUSED_DIM)
        subtask_centroids = np.arange(3 * TEXT_DIM, dtype=np.float32).reshape(3, TEXT_DIM) if subtask else None
        return _FitResult(
            centroids=centroids,
            subtask_centroids=subtask_centroids,
            effective_k=2,
            subtask_k=3 if subtask else 0,
            fit_rows=17,
            fragment_ids=(0, 2),
        )

    def _written(
        self,
        table: ClipsTable,
        *,
        version: int,
        fit: _FitResult | None = None,
        producers: Mapping[str, str] | None = None,
    ) -> tuple[_CentroidsArtifact, dict[str, npt.NDArray[np.float32]], _FitResult]:
        """Write a basis under NON-default weights, so the archive must carry this run's.

        ``producers`` is compared against None rather than taken for its
        truthiness, because an empty mapping is a state under test - the run that
        resolved no identity at all - and not an omitted argument.
        """
        fitted = fit if fit is not None else self._fit()
        config = _config(table, weights=ModalityWeights(subtask=0.5, image=0.3, action=0.2))
        resolved = _PRODUCERS_UNDER_TEST if producers is None else producers
        artifact = _write_centroids(config, fitted, read_version=version, producers=resolved)
        with np.load(artifact.uri) as archive:
            return artifact, {name: archive[name] for name in archive.files}, fitted

    def test_the_stored_bytes_hash_to_the_name_they_are_stored_under(self, build_clips_table: BuildTable) -> None:
        """The name IS the hash, so a reader can prove it loaded the intended basis.

        Asserted against the bytes read back rather than against the value the
        writer returned: those agree by construction inside the writer, and it is
        the stored object a reader hashes.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        artifact, _archive, _fit = self._written(table, version=7)
        stored = pathlib.Path(artifact.uri)
        assert artifact.uri == f"{table.uri}{CENTROIDS_ROOT_SUFFIX}/{artifact.fingerprint}.npz"
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == artifact.fingerprint

    def test_republishing_one_fit_lands_on_the_same_object(self, build_clips_table: BuildTable) -> None:
        """A repeat of the same fit writes the same name, so the name is the fit's alone.

        The serialization has to be byte-stable for a content hash to be a usable
        name at all: a wall-clock stamp or a run id anywhere in the archive would
        give one fit as many objects as it has attempts, and none of them would be
        reachable from a commit that referenced another. This pins the serializer,
        not the fit -- a GPU refit is not bitwise reproducible, so a retry does
        normally publish a new object.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        fit = self._fit()
        first, _archive, _fit = self._written(table, version=3, fit=fit)
        again, _archive_again, _fit_again = self._written(table, version=3, fit=fit)
        assert again.fingerprint == first.fingerprint

    def test_the_archive_holds_exactly_the_declared_key_set(self, build_clips_table: BuildTable) -> None:
        """The writer and every reader agree on the archive's shape through one set.

        A reader holding its own copy of the full set cannot tell "a key I do not
        consume" from "a key no Curate run wrote", so the writer gaining a key
        silently makes every healthy artifact look foreign to it. Failing here is
        what forces the shared set to be updated in the same change.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=7)
        assert set(archive) == CENTROID_ARCHIVE_KEYS

    def test_the_artifact_carries_the_metric_the_basis_was_fitted_in(self, build_clips_table: BuildTable) -> None:
        """A fused coordinate means nothing without the block order and the weights.

        The weights are THIS run's, not the module defaults: a basis read back
        against another run's metric is a wrong distance, not a missing one.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=7)
        np.testing.assert_array_equal(archive["block_dims"], np.asarray(vectors.BLOCK_DIMS, dtype=np.int64))
        np.testing.assert_allclose(archive["block_weights"], np.array([0.5, 0.3, 0.2]))
        assert [str(name) for name in archive["block_columns"]] == [
            group.primary_vector for group, _field in FUSED_BLOCKS
        ]

    def test_the_artifact_records_what_was_fitted_and_on_how_much(self, build_clips_table: BuildTable) -> None:
        """``effective_k`` is observed from the basis, never the value that was requested.

        The version recorded is the one the sample was READ at, which is the only
        one that exists when the basis is published; the version this run goes on
        to commit has not been assigned yet.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=3)
        assert (int(archive["effective_k"]), int(archive["fit_rows"]), int(archive["read_version"])) == (2, 17, 3)

    def test_the_artifact_records_what_would_be_needed_to_refit_it(self, build_clips_table: BuildTable) -> None:
        """The seed and the sampled prefix, because the sample is a prefix and not the corpus.

        Without both, the basis cannot be reproduced from its own file, and a
        re-fit that lands elsewhere is indistinguishable from a corpus that moved.
        One seed covers both fits, because one task produced both from one sample.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=3)
        assert int(archive["kmeans_random_state"]) == _config(table).kmeans_random_state
        np.testing.assert_array_equal(archive["fit_fragment_ids"], np.array([0, 2], dtype=np.int64))

    def test_the_basis_is_persisted_raw(self, build_clips_table: BuildTable) -> None:
        """The consumer owns the unit-normalization, so a normalized archive is a wrong basis.

        Asserted as an exact array: a shape check passes for a basis that was
        normalized, cast or transposed on the way out, and nothing downstream
        could tell, because the artifact is the only record of the fit.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, fit = self._written(table, version=3)
        np.testing.assert_array_equal(archive["centroids"], fit.centroids)
        assert archive["centroids"].dtype == np.float32

    def test_the_level_two_basis_is_archived_beside_the_locality_one(self, build_clips_table: BuildTable) -> None:
        """A cell id is as unreadable without its centroids as a cluster id is without its.

        Nothing persisted on the table records which region of instruction meaning
        a level-2 group was, so the artifact is the only place that can say. Both
        bases are asserted in one test because what matters is that they are
        distinguishable: swapping them would type-check and pass a shape check.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, fit = self._written(table, version=3)
        assert fit.subtask_centroids is not None
        np.testing.assert_array_equal(archive["subtask_centroids"], fit.subtask_centroids)
        assert int(archive["subtask_k"]) == 3
        assert str(archive["subtask_column"]) == SUBTASK_VECTOR_COLUMN

    def test_a_run_that_fitted_no_level_two_basis_says_so_unambiguously(self, build_clips_table: BuildTable) -> None:
        """Zero ROWS, not a missing key: a fitted basis always holds at least one centroid.

        A reader can therefore tell "the subtask block carried no weight" from
        "this artifact predates the level-2 basis" without consulting the config
        the run was launched with.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=3, fit=self._fit(subtask=False))
        assert archive["subtask_centroids"].shape == (0, TEXT_DIM)
        assert int(archive["subtask_k"]) == 0

    def test_the_artifact_names_whose_vectors_the_basis_was_fitted_over(self, build_clips_table: BuildTable) -> None:
        """The block order and weights say how the coordinates were assembled; this says from whom.

        Two runs whose blocks, weights and dims all match are still incomparable
        when one producer differs, and no other archived field can tell them apart.
        Read back as a column-to-identity mapping, because that is how a consumer
        uses it - the two arrays are parallel by position, and a shape check alone
        would pass for a pair that had been sorted independently.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=3)
        recorded = dict(
            zip(
                [str(name) for name in archive["producer_columns"]],
                [str(identity) for identity in archive["producer_identities"]],
                strict=True,
            )
        )
        assert recorded == dict(_PRODUCERS_UNDER_TEST)

    def test_a_run_over_an_unfilled_group_archives_an_empty_producer_pair(self, build_clips_table: BuildTable) -> None:
        """No identity resolved is an empty pair of arrays, not a missing or float-typed key.

        A reader loading the archive must be able to iterate the two arrays
        unconditionally; an untyped empty list would arrive as a float array and
        break that on the one run that resolved nothing.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        _artifact, archive, _fit = self._written(table, version=3, producers={})
        assert archive["producer_columns"].shape == (0,)
        assert archive["producer_columns"].dtype.kind == "U"
        assert archive["producer_identities"].shape == (0,)

    def test_the_archived_rules_are_the_ones_the_commit_stamped_a_digest_of(
        self, build_clips_table: BuildTable
    ) -> None:
        """Hashing the archived text reproduces the digest the version carries.

        This is what makes the two artifacts one record rather than two claims:
        the commit says which rules produced a version but not what they were,
        the archive says what they were but sits in a sibling directory anyone
        can write to, and only the recomputation ties a file to a version.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        config = _config(table, weights=ModalityWeights(subtask=0.5, image=0.3, action=0.2))

        _artifact, archive, _fit = self._written(table, version=3)

        properties = _commit_properties(config, centroids_fingerprint=_artifact.fingerprint)
        assert str(archive["resolved_config"]) == config.result_defining_json()
        assert str(archive["config_digest"]) == properties["config_digest"]

    def test_the_commit_names_the_object_the_basis_was_published_to(self, build_clips_table: BuildTable) -> None:
        """The commit's fingerprint resolves to a file, which is what makes it a reference.

        Nothing else links the two: the object's name carries no version and the
        commit carries no path, so a fingerprint that did not name a stored object
        would leave a curated version unable to say which basis produced its
        cluster ids.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))
        config = _config(table)

        artifact, _archive, _fit = self._written(table, version=3)

        named = _commit_properties(config, centroids_fingerprint=artifact.fingerprint)["centroids_fingerprint"]
        assert pathlib.Path(f"{table.uri}{CENTROIDS_ROOT_SUFFIX}/{named}.npz").exists()

    def test_the_archived_rules_omit_the_table_they_were_published_on(self, build_clips_table: BuildTable) -> None:
        """A table copied elsewhere keeps reporting the identity of the rules it holds.

        Named separately from the digest's own exclusions because the archive is
        where a reader looks to see WHERE two runs differ, so an address landing
        in that text would read as a rule difference.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=1))

        _artifact, archive, _fit = self._written(table, version=3)

        assert "clips_lance_uri" not in json.loads(str(archive["resolved_config"]))


def _cpu_retain(config: CurateConfig, scanned: pa.Table) -> pa.Table:
    """Apply the retention stage, or the projection ``_retain`` substitutes for it.

    The same branch ``_retain`` takes, on the same four functions, so the CPU walk
    exercises whichever path the config asks for rather than only the scored one.
    It routes through the production predicates rather than local masks so the
    bypass boundary rides under every test that walks the leg, and so does the
    schema agreement the two paths owe each other: ``concat_tables`` refuses a
    mismatch here exactly as ``Dataset.union`` would in the pipeline.
    """
    if config.dedup_eps is None:
        return unscored_rows(scanned)
    parts = [unscored_rows(bypass_scan_rows(scanned))]
    parts.extend(
        dedup.mark_duplicates(group, eps=config.dedup_eps)
        for group in _split_by(scored_scan_rows(scanned), [DEDUP_KEY_COLUMN])
    )
    return pa.concat_tables(parts)


def _run_cpu_curate(config: CurateConfig) -> tuple[int, pa.Table]:
    """Walk the whole leg on the CPU, in the order ``run_curate`` composes it.

    Ray Data decides only WHERE each stage runs, so a local groupby and the
    closed-form single-cluster basis exercise every kernel, the stage ORDER, the
    quota arithmetic and the real commit.

    One position differs from production and cannot be reproduced here: the basis
    is published after the corpus read rather than before it, because ``_scan_table``
    bundles the fit and the scan. What the walk does preserve is the property the
    ordering exists for, that the commit names an already-durable object.

    Returns:
        ``(committed_version, verdict rows)``.

    """
    scanned, source, centroids = _scan_table(config)
    _requested_k(source.eligible_rows, config.target_mean_cluster_rows)

    # Published HERE, before any of the work below, because that ordering is the
    # completion contract: the commit at the bottom may only name an object that is
    # already durable. The basis is the synthetic closed-form one, which is what
    # makes this a wiring assertion rather than a fit assertion.
    subtask = _cpu_subtask_centroids(source, config)
    published = _write_centroids(
        config,
        _FitResult(
            centroids=centroids,
            subtask_centroids=subtask,
            effective_k=int(centroids.shape[0]),
            subtask_k=0 if subtask is None else int(subtask.shape[0]),
            fit_rows=source.eligible_rows,
            fragment_ids=tuple(source.fragment_ids),
        ),
        read_version=source.read.read_version,
        producers=source.producers,
    )

    rollups = [_label_rollup(_work_batch(fragment_id), read=source.read) for fragment_id in source.fragment_ids]
    task_merge, _stats = _merge_maps(config, _reduce_gather(rollups))

    merged = fairness.apply_label_merge(_cpu_retain(config, scanned), task_merge)

    group_keys = [CANONICAL_TASK_COLUMN, SUBTASK_CLUSTER_COLUMN]
    quota, _target = _build_quota(config, _count_table(merged, [*group_keys, CURATE_SELECTION_REASON]))
    quotas = quota.quotas()
    verdicts = pa.concat_tables(
        [
            fairness.select_within_quota(group, quotas, config.within_group_order)
            for group in _split_by(merged, group_keys)
        ]
    )
    _reason_counts(_count_table(verdicts, [CURATE_SELECTION_REASON]), source.eligible_rows)

    payloads = [
        str(
            update_one_fragment(
                group,
                uri=source.read.uri,
                read_version=source.read.read_version,
                storage_options=source.read.storage_options,
            )
            .column(_RESULT_COLUMN)[0]
            .as_py()
        )
        for group in _split_by(verdicts, [FRAGMENT_COLUMN])
    ]
    collected = _total_rows(_collect(payloads), source.eligible_rows)
    version = _commit(
        source.dataset,
        collected,
        properties=_commit_properties(config, centroids_fingerprint=published.fingerprint),
        storage_options=source.read.storage_options,
    )
    return version, verdicts


class TestCpuEndToEnd:
    """The composed leg, from an untouched table to a committed one.

    What runs here is every stage KERNEL and the order they run in, against a real
    Lance table, ending in a real basis publication and a real commit. What does
    not, and is therefore UNVERIFIED anywhere in this file:

    - the cuML k-means in ``_fit_kmeans`` and the cuPy retention GEMM in
      ``dedup_group``, neither of which has a CPU path;
    - every Ray-scheduled wrapper, and the logic each one carries alone: the
      ``ObjectRef``-stays-a-reference assumption and the per-batch streaming in
      ``_scan_verdicts``, the basis-width and degenerate-basis checks in
      ``_fit_centroids``, and the k clamp in ``_fit_kmeans``;
    - ``RAY_COUNT_COLUMN``, which ``_count_table`` substitutes rather than
      verifies, and the block sizing in ``_work_items``;
    - that ``run_curate`` itself places the publication before the commit. This
      walk composes that order, but only the source does so for a real run; what
      is asserted here is the consequence an operator sees, that a committed
      version resolves to verifiable basis bytes.
    """

    def _committed(self, table: ClipsTable, config: CurateConfig) -> pa.Table:
        version, _verdicts = _run_cpu_curate(config)
        return lance.dataset(table.uri, version=version).to_table(
            columns=[KEY_COLUMN, CURATE_SELECTION_REASON, CURATE_CLUSTER_ID]
        )

    def _by_clip(self, rows: pa.Table) -> dict[str, str | None]:
        return dict(
            zip(
                rows.column(KEY_COLUMN).to_pylist(),
                rows.column(CURATE_SELECTION_REASON).to_pylist(),
                strict=True,
            )
        )

    def test_every_eligible_row_is_published_with_exactly_one_reason(self, build_clips_table: BuildTable) -> None:
        """The leg's central promise: a claimed row leaves the run carrying a verdict."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        rows = self._committed(table, _config(table))
        assert rows.column(CURATE_SELECTION_REASON).null_count == 0
        assert set(rows.column(CURATE_SELECTION_REASON).to_pylist()) <= {str(reason) for reason in CurateReason}

    def test_the_committed_version_resolves_to_the_basis_bytes_it_names(self, build_clips_table: BuildTable) -> None:
        """A committed version alone is enough to fetch and verify the basis behind it.

        This is the whole point of publishing first, and the assertion is made
        against DISK rather than against the writer's return value: the run threads
        a real fingerprint onto the commit only if the object that hash names is
        already there. A stand-in constant, or a hash of anything other than the
        published bytes, fails here.
        """
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        version, _verdicts = _run_cpu_curate(_config(table))
        stamped = lance.dataset(table.uri, version=version).read_transaction(version)
        assert stamped is not None
        named = stamped.transaction_properties["centroids_fingerprint"]
        stored = pathlib.Path(f"{table.uri}{CENTROIDS_ROOT_SUFFIX}/{named}.npz")
        assert stored.exists(), f"the commit names {named} but no such object was published"
        assert hashlib.sha256(stored.read_bytes()).hexdigest() == named

    def test_exactly_the_target_number_of_rows_is_selected(self, build_clips_table: BuildTable) -> None:
        """The quota is spent in full and never overspent."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4))
        rows = self._committed(table, _config(table, target=SelectionTarget(target_count=5)))
        assert rows.column(CURATE_SELECTION_REASON).to_pylist().count(str(CurateReason.SELECTED)) == 5

    def test_one_half_of_an_identical_pair_is_published_as_a_duplicate(self, build_clips_table: BuildTable) -> None:
        """Retention keeps one of a byte-identical pair, and marks exactly the other."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4, duplicate_vectors=(0, 5)))
        by_clip = self._by_clip(self._committed(table, _config(table)))
        pair = [by_clip[table.clip_id_at(row)] for row in (0, 5)]
        assert pair.count(str(CurateReason.DUPLICATE)) == 1

    def test_an_identical_pair_both_survive_when_de_duplication_is_off(self, build_clips_table: BuildTable) -> None:
        """With ``dedup_eps`` unset the pair is two selection candidates, not one and a verdict.

        Asserted on the same fixture the scored path marks exactly one of, so the
        difference is the config field and nothing else, and on the whole corpus
        carrying no ``duplicate`` at all - the pair could survive while some other
        row was marked, which would still mean the stage ran.
        """
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=4, duplicate_vectors=(0, 5)))
        rows = self._committed(table, _config(table, dedup_eps=None))
        by_clip = self._by_clip(rows)

        assert str(CurateReason.DUPLICATE) not in set(rows.column(CURATE_SELECTION_REASON).to_pylist())
        assert by_clip[table.clip_id_at(0)] != str(CurateReason.DUPLICATE)
        assert by_clip[table.clip_id_at(5)] != str(CurateReason.DUPLICATE)
        assert rows.column(CURATE_SELECTION_REASON).null_count == 0

    def test_the_run_with_no_de_duplication_still_publishes_the_geometry(self, build_clips_table: BuildTable) -> None:
        """The fit runs on both paths, because the cluster and the distance come from the scan.

        Skipping the retention stage must not skip the basis: ``curate_cluster_id``
        is a persisted column and the within-group order reads the distance, so a
        skip that also skipped the fit would publish NULL clusters for selected rows.
        """
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        rows = self._committed(table, _config(table, dedup_eps=None))
        assert set(rows.column(CURATE_CLUSTER_ID).to_pylist()) == {0}

    def test_the_post_retention_schema_is_the_same_whether_or_not_the_stage_ran(
        self, build_clips_table: BuildTable
    ) -> None:
        """Two schemas out of one stage cannot concatenate, so the skip must project, not pass through.

        The vector is the load-bearing half: at corpus scale it is the largest
        column in the run and nothing after retention reads it, so a skip path that
        merely returned its input would carry it to the write.
        """
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        scored = _config(table)
        scanned, _source, _centroids = _scan_table(scored)

        assert _cpu_retain(scored, scanned).schema == _cpu_retain(_config(table, dedup_eps=None), scanned).schema
        assert WORKING_VECTOR_COLUMN not in _cpu_retain(_config(table, dedup_eps=None), scanned).schema.names

    def test_an_invalid_embedding_is_published_with_no_cluster(self, build_clips_table: BuildTable) -> None:
        """The routing sentinel is mapped to NULL, so no reader sees a fabricated cluster."""
        spec = ClipsTableSpec(fragments=2, rows_per_fragment=3, zero_norm_vector_rows=frozenset({4}))
        table = build_clips_table(spec)
        rows = self._committed(table, _config(table))
        index = rows.column(KEY_COLUMN).to_pylist().index(table.clip_id_at(4))
        assert rows.column(CURATE_SELECTION_REASON)[index].as_py() == str(CurateReason.INVALID_EMBEDDING)
        assert rows.column(CURATE_CLUSTER_ID)[index].as_py() is None

    def test_a_non_finite_row_is_published_with_no_cluster_too(self, build_clips_table: BuildTable) -> None:
        """The second bypass cause takes a different sentinel and must map to NULL as well.

        The sentinel split is invisible after the write - both causes commit the
        same reason and the same NULL cluster - so only a test on the non-finite
        cause catches a NULL mapping written for one sentinel value.
        """
        spec = ClipsTableSpec(fragments=2, rows_per_fragment=3, non_finite_vector_rows=frozenset({4}))
        table = build_clips_table(spec)
        rows = self._committed(table, _config(table))
        index = rows.column(KEY_COLUMN).to_pylist().index(table.clip_id_at(4))
        assert rows.column(CURATE_SELECTION_REASON)[index].as_py() == str(CurateReason.INVALID_EMBEDDING)
        assert rows.column(CURATE_CLUSTER_ID)[index].as_py() is None

    def test_a_row_the_run_never_claimed_stays_null(self, build_clips_table: BuildTable) -> None:
        """Curate widens the table in place, so an unclaimed row must read as untouched."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=2, empty_fragments=frozenset({1})))
        by_clip = self._by_clip(self._committed(table, _config(table)))
        assert by_clip[table.clip_id_at(2)] is None
        assert by_clip[table.clip_id_at(3)] is None

    def test_a_selected_row_carries_the_cluster_it_was_scored_in(self, build_clips_table: BuildTable) -> None:
        """``curate_cluster_id`` is the persisted half of the geometry the run fitted."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        rows = self._committed(table, _config(table))
        clusters = rows.column(CURATE_CLUSTER_ID).to_pylist()
        assert set(clusters) == {0}

    def test_the_commit_advances_the_table_by_one_version(self, build_clips_table: BuildTable) -> None:
        """One Update transaction for the whole run: no partial publication can exist."""
        table = build_clips_table(ClipsTableSpec(fragments=3, rows_per_fragment=2))
        config = _config(table)
        before = _preflight(config).read.read_version
        version, _verdicts = _run_cpu_curate(config)
        assert version == before + 1

    def test_every_scanned_row_reaches_the_radius_histogram(self, build_clips_table: BuildTable) -> None:
        """The reduction consumes the scan's own distance column, not a hand-built one."""
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        config = _config(table)
        scanned, _source, _centroids = _scan_table(config)
        retained = _cpu_retain(config, scanned)

        assert int(_fold_metric_histograms([metric_histogram(retained)])[DISTANCE_COLUMN].sum()) == scanned.num_rows

    def test_farthest_selects_the_more_distant_rows_and_nearest_the_closer_ones(
        self, build_clips_table: BuildTable
    ) -> None:
        """The ordering is result-defining, which is why it is a config field and not a default.

        Asserted on the DIRECTION, not merely that the two sets differ: two
        comparators swapped against each other also select different rows, and a
        set inequality cannot tell which way round they are. The distance is a
        transient scan column and is never committed, so it is recovered by
        re-scanning the same table against the same basis.
        """
        spec = ClipsTableSpec(fragments=3, rows_per_fragment=4)
        target = SelectionTarget(target_count=3)
        farthest = build_clips_table(spec)
        nearest = build_clips_table(spec)
        totals = []
        for table, order in ((farthest, "farthest"), (nearest, "nearest")):
            config = _config(table, target=target, within_group_order=order)
            scanned, _source, _centroids = _scan_table(config)
            distance = dict(
                zip(
                    scanned.column(KEY_COLUMN).to_pylist(),
                    scanned.column(DISTANCE_COLUMN).to_pylist(),
                    strict=True,
                )
            )
            selected = [
                clip
                for clip, reason in self._by_clip(self._committed(table, config)).items()
                if reason == str(CurateReason.SELECTED)
            ]
            totals.append(sum(distance[clip] for clip in selected))
        assert totals[0] > totals[1]


class TestUnscoredRows:
    """The projection that stands in for the retention stage when it does not run."""

    def test_the_working_vector_is_shed_exactly_as_the_scoring_stage_sheds_it(
        self, build_clips_table: BuildTable
    ) -> None:
        """It is the run's largest column and the last reader is the stage being skipped.

        A skip that returned its input unchanged would carry the vector through
        the fairness shuffle and into the write.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=3))
        scanned, _source, _centroids = _scan_table(_config(table))

        projected = unscored_rows(scanned)
        assert WORKING_VECTOR_COLUMN in scanned.schema.names
        assert WORKING_VECTOR_COLUMN not in projected.schema.names
        assert projected.num_rows == scanned.num_rows

    def test_no_row_is_marked_and_every_score_reads_null(self, build_clips_table: BuildTable) -> None:
        """Nothing was compared, so the score is absent rather than zero.

        A zero would enter bin 0 of the report and read as a genuine "no earlier
        row was similar" verdict on a corpus where no comparison happened.
        """
        table = build_clips_table(ClipsTableSpec(fragments=1, rows_per_fragment=3, duplicate_vectors=(0, 2)))
        scanned, _source, _centroids = _scan_table(_config(table, dedup_eps=None))

        projected = unscored_rows(scanned)
        assert projected.column(DEDUP_SCORE_COLUMN).null_count == projected.num_rows
        assert str(CurateReason.DUPLICATE) not in set(projected.column(CURATE_SELECTION_REASON).to_pylist())


class TestRetentionRouting:
    """The two predicates that split the scan between the GEMM and the bypass."""

    @staticmethod
    def _keys(values: Sequence[int]) -> pa.Table:
        """Return a batch carrying only the routing column the predicates read."""
        return pa.table({DEDUP_KEY_COLUMN: pa.array(values, type=pa.int64())})

    def test_cluster_zero_is_scored_rather_than_bypassed(self) -> None:
        """The boundary is the sign of the key, and a valid cluster id starts at zero.

        A strict comparison would route the whole of cluster 0 down the bypass, so
        every one of its rows would reach the fairness cut unscored: a recall loss
        on one cluster in k that no stage reports, because both paths still
        account for the row.
        """
        keys = self._keys([0, -1])
        assert scored_scan_rows(keys).column(DEDUP_KEY_COLUMN).to_pylist() == [0]
        assert bypass_scan_rows(keys).column(DEDUP_KEY_COLUMN).to_pylist() == [-1]

    def test_every_row_is_routed_exactly_once_and_keeps_its_arrival_order(self) -> None:
        """The stage rebuilds the corpus from the two halves, so they must partition it.

        A row matched by both predicates would be committed twice; one matched by
        neither would vanish before the completeness check could miss it. Order
        within each half matters as well, because the retention rule reads its
        input as already sorted farthest-first.
        """
        keys = self._keys([3, -1, 0, 7, -2, 3])
        scored = scored_scan_rows(keys).column(DEDUP_KEY_COLUMN).to_pylist()
        bypassed = bypass_scan_rows(keys).column(DEDUP_KEY_COLUMN).to_pylist()
        assert sorted(scored + bypassed) == sorted(keys.column(DEDUP_KEY_COLUMN).to_pylist())
        assert scored == [3, 0, 7, 3]
        assert bypassed == [-1, -2]


MetricBatch = Callable[[Sequence[float | None]], pa.Table]


@pytest.fixture
def distances() -> MetricBatch:
    """Return a builder for a batch carrying only radii; every score reads NULL."""

    def build(values: Sequence[float | None]) -> pa.Table:
        return pa.table(
            {
                DISTANCE_COLUMN: pa.array(values, type=pa.float32()),
                DEDUP_SCORE_COLUMN: pa.nulls(len(values), type=pa.float32()),
            }
        )

    return build


@pytest.fixture
def scores() -> MetricBatch:
    """Return a builder for a batch carrying only retention scores; every radius reads NULL."""

    def build(values: Sequence[float | None]) -> pa.Table:
        return pa.table(
            {
                DISTANCE_COLUMN: pa.nulls(len(values), type=pa.float32()),
                DEDUP_SCORE_COLUMN: pa.array(values, type=pa.float32()),
            }
        )

    return build


def _radius_counts(tables: Sequence[pa.Table]) -> npt.NDArray[np.int64]:
    """Fold the reduction's output and return the radius metric's count vector."""
    return _fold_metric_histograms(tables)[_RADIUS_METRIC.column]


def _score_counts(tables: Sequence[pa.Table]) -> npt.NDArray[np.int64]:
    """Fold the reduction's output and return the retention score's count vector."""
    return _fold_metric_histograms(tables)[_SCORE_METRIC.column]


class TestClusterRadius:
    """The cluster-radius reduction: a fixed-bin histogram read for its percentiles.

    A histogram exists because the driver cannot hold the column: at 250M rows a
    percentile by collect-and-sort is 1 GB of float32 on one node. These tests
    therefore pin the two properties a histogram can get wrong that a sort cannot
    -- which rows enter a bin, and how bins from different batches combine.
    """

    _WIDTH = _RADIUS_METRIC.bin_width

    def test_a_row_carrying_no_distance_is_not_binned(self, distances: MetricBatch) -> None:
        """A row scored against no centroid has no radius, so it must not weigh one down.

        Arrow renders a NULL float as NaN on the way to NumPy, and an integer cast
        of NaN is platform-defined rather than an error, so an unfiltered NULL is
        counted in some bin instead of raising.
        """
        counts = _radius_counts([metric_histogram(distances([0.5, None, 1.5]))])

        assert int(counts.sum()) == 2

    def test_the_far_end_of_the_range_lands_in_the_last_bin(self, distances: MetricBatch) -> None:
        """Distance 2.0 is attainable - antipodal unit vectors - and has no bin of its own."""
        counts = _radius_counts([metric_histogram(distances([_RADIUS_METRIC.upper]))])

        assert int(counts[_HISTOGRAM_BINS - 1]) == 1

    def test_a_distance_below_the_metric_range_is_binned_rather_than_raised(self, distances: MetricBatch) -> None:
        """A diagnostic reduction must not be able to abort a run that already ran its GPU pass.

        The magnitude is far outside float32 rounding on purpose. A stray -1e-8
        truncates to bin 0 on the integer cast alone, so it cannot tell whether the
        lower bound is present; only a value whose scaled magnitude exceeds one
        reaches ``np.bincount``, which rejects a negative index outright.
        """
        counts = _radius_counts([metric_histogram(distances([-0.5]))])

        assert int(counts[0]) == 1

    def test_one_bin_repeated_inside_one_folded_table_sums_rather_than_overwrites(self, distances: MetricBatch) -> None:
        """Ray Data concatenates blocks, so one table can name a bin twice.

        The two histograms are joined into ONE table, which is what block
        concatenation does. Folding them as two separate tables would not exercise
        this at all: the loop's second iteration accumulates correctly whatever
        indexing it uses, and only a duplicated index INSIDE one table can lose a
        write.
        """
        one_bin = 5.5 * self._WIDTH
        folded = pa.concat_tables(
            [metric_histogram(distances([one_bin] * 3)), metric_histogram(distances([one_bin] * 4))]
        )

        assert int(_radius_counts([folded])[5]) == 7

    def test_each_percentile_lands_within_half_a_bin_of_the_true_order_statistic(self, distances: MetricBatch) -> None:
        """The histogram's resolution is its whole error budget, and it is half a bin width.

        Evenly spaced distances make every order statistic known in closed form, so
        each reported percentile is checked against the value it approximates
        rather than against another run of the same code.

        The row count is deliberately NOT a multiple of 20. At a round hundred
        every ``quantile * total`` is already an integer, both rounding
        conventions pick the same rank, and the test cannot see which one the
        reduction uses; 101 rows separate them by one rank, which at this spacing
        is twenty bins.
        """
        rows = 101
        spread = [index * 0.019 for index in range(rows)]
        counts = _radius_counts([metric_histogram(distances(spread))])

        reported = _histogram_quantiles(counts, _RADIUS_QUANTILES, self._WIDTH)
        expected = [sorted(spread)[math.ceil(quantile * rows) - 1] for quantile in _RADIUS_QUANTILES]
        assert all(
            abs(value - truth) <= self._WIDTH / 2 + 1e-9 for value, truth in zip(reported, expected, strict=True)
        )

    def test_a_corpus_with_no_scored_row_reports_no_percentile(self) -> None:
        """An empty histogram must not report a percentile of nothing as bin zero."""
        assert _histogram_quantiles(np.zeros(_HISTOGRAM_BINS, dtype=np.int64), _RADIUS_QUANTILES, self._WIDTH) == ()


class TestDedupScoreReport:
    """The retention score's distribution: the same fold, read for its upper tail.

    The score is never persisted, so these log lines are the only thing that makes
    ``dedup_eps`` interpretable after a run - and the only thing that says what a
    different threshold would have flagged.
    """

    _WIDTH = _SCORE_METRIC.bin_width

    def test_both_metrics_come_out_of_one_reduction_pass(self, build_clips_table: BuildTable) -> None:
        """Two distributions, one narrow projection over the materialized corpus.

        The reduction is fed the scan's own columns rather than a hand-built batch,
        so a metric silently dropped from the reported set shows up as a missing
        count vector here.
        """
        table = build_clips_table(ClipsTableSpec(fragments=2, rows_per_fragment=3))
        scanned, _source, _centroids = _scan_table(_config(table))
        retained = pa.concat_tables(
            [dedup.mark_duplicates(group, eps=0.01) for group in _split_by(scanned, [DEDUP_KEY_COLUMN])]
        )

        folded = _fold_metric_histograms([metric_histogram(retained)])

        assert int(folded[_RADIUS_METRIC.column].sum()) == scanned.num_rows
        assert int(folded[_SCORE_METRIC.column].sum()) == scanned.num_rows

    def test_a_row_that_bypassed_the_similarity_pass_contributes_no_bin(self, scores: MetricBatch) -> None:
        """A NULL score is "never compared", which is not the same as "compared and scored 0".

        Binning it into bin 0 would put every ``invalid_embedding`` row under the
        percentile the operator reads their threshold against, dragging the whole
        reported tail down by however many rows had no vector.
        """
        counts = _score_counts([metric_histogram(scores([0.99, None, None]))])

        assert int(counts.sum()) == 1

    def test_the_leading_row_of_every_group_clips_into_the_underflow_bin(self, scores: MetricBatch) -> None:
        """Row 0 of a group is scored against nothing and an antipodal pair scores negative.

        Both are legitimate outputs of the retention kernel rather than defects, so
        the low end of the range has to absorb them instead of raising - the same
        reason the radius clips both bounds.
        """
        counts = _score_counts([metric_histogram(scores([0.0, -1.0]))])

        assert int(counts[0]) == 2

    def test_the_reported_tail_matches_the_constructed_cosines(self, scores: MetricBatch) -> None:
        """The percentiles and the maximum are read off the histogram to within half a bin.

        A thousand rows at one cosine and one exact duplicate: both percentiles
        then land in the bulk while the maximum is the duplicate, so a maximum
        taken as another percentile - or a percentile taken as the maximum - fails
        on the gap between them. The bulk is one row larger than the p99.9 rank
        needs, which is what keeps the single top row out of that percentile.
        """
        bulk = [0.5] * 1000
        counts = _score_counts([metric_histogram(scores([*bulk, 1.0]))])

        reported = _histogram_quantiles(counts, _SCORE_QUANTILES, self._WIDTH)
        assert all(abs(value - 0.5) <= self._WIDTH / 2 + 1e-9 for value in reported)
        highest = _histogram_max(counts, self._WIDTH)
        assert highest is not None
        assert abs(highest - 1.0) <= self._WIDTH / 2 + 1e-9

    def test_the_ladder_counts_the_rows_each_candidate_threshold_would_flag(self, scores: MetricBatch) -> None:
        """Each rung is the folded histogram's own tail above ``1 - eps``, and nothing more.

        One cosine is placed inside each interval between consecutive candidate
        thresholds, and every one of them sits several bin widths clear of the
        nearest threshold, so the count each rung must report is known by counting
        the construction rather than by re-deriving the binning. The expected
        counts are written out for the same reason: computing them from
        ``_DEDUP_EPS_LADDER`` would restate the arithmetic under test.
        """
        constructed = [0.9997, 0.997, 0.993, 0.985, 0.96, 0.92, 0.5]
        counts = _score_counts([metric_histogram(scores(constructed))])

        assert dict(_eps_ladder(counts, self._WIDTH, None)) == {
            0.001: 1,
            0.005: 2,
            0.01: 3,
            0.02: 4,
            0.05: 5,
            0.10: 6,
        }
        assert set(_DEDUP_EPS_LADDER) == {0.001, 0.005, 0.01, 0.02, 0.05, 0.10}

    def test_a_threshold_outside_the_static_span_still_gets_a_rung(self, scores: MetricBatch) -> None:
        """A run looser than the widest static rung must still find its own eps on the ladder.

        The static rungs stop at 0.10, so a run at 0.30 would otherwise read a
        ladder of pure counterfactuals. The fixture puts one cosine between 0.10
        and 0.30 that NO static rung can reach, so the spliced rung is the only
        thing that can report it and deleting the splice drops the count to the
        0.10 value.
        """
        counts = _score_counts([metric_histogram(scores([0.995, 0.8]))])

        ladder = dict(_eps_ladder(counts, self._WIDTH, 0.30))

        assert ladder[0.10] == 1
        assert ladder[0.30] == 2

    def test_a_threshold_equal_to_a_static_rung_is_not_reported_twice(self, scores: MetricBatch) -> None:
        """Splicing the run's eps onto the static rungs must not duplicate an existing one.

        ``0.01`` is both the shipped default and a static rung, so the common case
        is exactly the collision case; a list-append splice would render it twice.
        """
        counts = _score_counts([metric_histogram(scores([0.995]))])

        rungs = [rung for rung, _ in _eps_ladder(counts, self._WIDTH, 0.01)]

        assert rungs.count(0.01) == 1

    def test_a_spliced_rung_is_ordered_among_the_static_ones(self, scores: MetricBatch) -> None:
        """The ladder reads ascending, so a spliced rung sorts into place rather than onto the end.

        ``0.03`` is deliberately not a static rung and falls between ``0.02`` and
        ``0.05``, so an append-without-sort splice puts it last and this fails.
        A rung equal to a static one could not detect that.
        """
        counts = _score_counts([metric_histogram(scores([0.995]))])

        rungs = [rung for rung, _ in _eps_ladder(counts, self._WIDTH, 0.03)]

        assert rungs == sorted(rungs)
        assert rungs.index(0.03) == rungs.index(0.02) + 1

    def test_a_corpus_with_no_score_at_all_reports_no_maximum(self) -> None:
        """An empty histogram has no highest occupied bin, which is not bin zero.

        The state a skipped retention pass leaves behind, so it must be
        distinguishable from a corpus whose every score really was zero.
        """
        assert _histogram_max(np.zeros(_HISTOGRAM_BINS, dtype=np.int64), self._WIDTH) is None

    def test_the_reported_line_is_the_upper_tail_and_not_the_middle(
        self, scores: MetricBatch, loguru_records: list[dict[str, Any]]
    ) -> None:
        """Near-duplicates live in the last thousandth, so a median score says nothing about them.

        Read off the emitted line rather than the constant, because the line is
        what an operator actually reads a threshold against: p50 and p95 of a
        corpus at any ``dedup_eps`` are both far below the threshold and would
        make the report unactionable.
        """
        _log_dedup_score(_score_counts([metric_histogram(scores([0.5, 0.99]))]), 0.01)

        line = next(message for message in _infos(loguru_records) if message.startswith("curate dedup score:"))
        assert "p99=" in line
        assert "p99.9=" in line
        assert "max=" in line
        assert "p50=" not in line

    def test_the_reported_line_marks_which_rung_the_run_actually_used(
        self, scores: MetricBatch, loguru_records: list[dict[str, Any]]
    ) -> None:
        """The operating point must be identifiable on the ladder, not inferred from it.

        A threshold outside the static span is the case that makes this load
        bearing: every other rung on the line is a counterfactual, so without the
        marker a reader has no way to tell which count describes the verdicts the
        run actually wrote.
        """
        _log_dedup_score(_score_counts([metric_histogram(scores([0.5, 0.99]))]), 0.20)

        line = next(message for message in _infos(loguru_records) if message.startswith("curate dedup score:"))
        assert "eps=0.2*->" in line
        assert "eps=0.1*->" not in line
        assert "* the configured dedup_eps" in line

    @pytest.mark.integration
    @pytest.mark.usefixtures("ray_local")
    def test_the_configured_threshold_reaches_the_reported_ladder(self, loguru_records: list[dict[str, Any]]) -> None:
        """The run's own ``dedup_eps`` must survive the hop from the config to the line.

        The defect this marker exists to fix was never in the ladder arithmetic -
        it was the configured value not reaching the renderer, which every test
        that calls ``_log_dedup_score`` with a literal threshold passes straight
        over. Driven through a real dataset so the assertion covers the whole
        reduction rather than a stand-in for it.
        """
        rows = pa.table(
            {
                DISTANCE_COLUMN: pa.array([0.25, 0.30], type=pa.float32()),
                DEDUP_SCORE_COLUMN: pa.array([0.5, 0.99], type=pa.float32()),
            }
        )

        _report_metrics(ray.data.from_arrow(rows), 0.20)

        line = next(message for message in _infos(loguru_records) if message.startswith("curate dedup score:"))
        assert "eps=0.2*->" in line

    def test_the_absent_score_line_is_not_a_warning(self, loguru_records: list[dict[str, Any]]) -> None:
        """``dedup_eps=None`` is a configured state, so reporting it as degenerate misleads.

        The radius takes the opposite branch on the same emptiness - it can only be
        empty when every row was invalid - so the two must not share a level.
        """
        _log_dedup_score(np.zeros(_HISTOGRAM_BINS, dtype=np.int64), None)

        assert _warnings(loguru_records) == []
        assert any("no row carried a retention score" in message for message in _infos(loguru_records))
