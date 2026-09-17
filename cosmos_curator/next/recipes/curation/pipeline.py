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

"""Curate's Lance and Ray layer: the whole leg, from preflight to the one commit.

The geometry, the retention rule and the quota arithmetic live in the pure
kernels beside this module (``vectors``, ``dedup``, ``fairness``); what is owned
here is the distributed plumbing and every place Curate touches the table's
manifest. ``clips.lance`` is read once, widened with the two nullable
``curate_*`` columns in a metadata-only commit, and finally rebound for the
fragments that carried an eligible row - all in one transaction.

::

    _preflight            open, check the source contract, widen, PIN the version
        |                 refuse a corpus with no eligible row
        v
    _fit_centroids        manifest-ordered fragment prefix, ONE pass, ONE
        |                 whole-GPU task, ONE seed -> two cuML KMeans:
        |                   locality  raw (k, FUSED_DIM)  over working vectors
        |                   subtask   raw (k2, TEXT_DIM)  over subtask text
        v
    _write_centroids      publish both bases as ONE immutable <sha256>.npz,
        |                 BEFORE the corpus-wide passes below, so the commit can
        |                 reference an object that is already durable
        v
    _gather_labels        second bounded pass: per distinct canonical TASK label,
        |                 its clip count and ONE text vector.  O(tasks) driver.
        v
    _merge_maps           merge_labels once, driver-side over TASK labels
        |
        v
    _scan                 from_items(fragment_ids).map_batches, one task per
        |                 fragment: eligibility filter -> canonical task ->
        |                 working vectors -> nearest locality centroid ->
        |                 nearest subtask centroid -> stamp __frag
        v
    groupby(__dedup_key).map_groups(dedup_group)         GPU, retention rule
        |                 CONDITIONAL: at dedup_eps=None the rows are projected
        |                 by unscored_rows instead, so both paths shed the
        v                 working vector and emit the same score column
    map_batches(apply_label_merge) -> materialize()      barrier: merged tasks
        |
        +- metric_histogram over distance + score  ->  radius p50/p95,
        |                                               score tail + eps ladder
        |
        +- groupby(task, cluster, reason).count()  ->  quotas   (O(G) driver)
        v
    groupby(task, cluster).map_groups(select_within_quota) -> materialize()
        |                 every remaining row gets a reason; barrier: the verdicts
        |                 are counted for the pre-commit gate and then written back
        v
    ... grouped on __frag ...
        v
    update_one_fragment  (per group, in a worker)     blank_one_fragment
        |     reopen at read_version, get_fragment      |   the fragments no
        |     assert clip_id is unique within it        |   verdict group named
        |     __dedup_key < 0 -> NULL cluster id        |   -> NULL, NULL
        |     claimed rows take their verdict,          |
        |     every OTHER stored row takes NULL         |
        |     update_columns(left_on=clip_id)           |
        +-----------------+-----------------------------+
                          v
    _collect   decode, refuse two metadata versions of one fragment
        v
    _commit    ONE Transaction(Update(updated_fragments, fields_modified))

The write is TOTAL: every row of every fragment is written on every run, so both
columns always describe the run that committed last and NULL has exactly one
meaning - "the latest run did not claim this row". Three properties make that
cheap and safe:

- ``update_columns`` writes a new file holding only the two columns, for the
  fragment's existing rows in their existing offset order. No row is replaced,
  so no deletion vector is written and every row address survives.
- The join is left-outer on the ordinary persisted ``clip_id``, which is why an
  unclaimed row must be present carrying NULL rather than merely omitted: an
  omitted row keeps its previous value, an explicitly NULL one is overwritten.
- ``fields_modified`` scopes the rebinding to Curate's two field ids, so every
  other column - source fields and ``embedding_*`` alike - stays bound to the
  file it already had.

Totality is what makes a re-run over a SHRUNK eligible set safe. A verdict is not
a per-row computation, so a row curated by an earlier run under looser weights
would otherwise survive indistinguishable from a current one, and the table's
selected count could exceed the target this run was given.

The commit is ALL-OR-NOTHING, and this is where Curate deliberately parts from
the embeddings leg, which skips a failed fragment and commits the rest. A Curate
verdict is not a per-row computation: ``below_quota`` and ``unfunded`` are
decided against a corpus-global quota, so committing the fragments that happened
to succeed would publish verdicts derived from a population that was never
written. A failure therefore propagates out of the worker and the run ends with
no verdict published - narrower than "the table untouched", because a run that
is the first to curate a table has already committed the preflight's
metadata-only schema widening. That version stamps no ``kind``, so no reader
counts it as a curate run.

The commit carries the run's IDENTITY, and nothing more: the leg's ``kind``, the
config generation, a digest of every config field that decides an outcome, and the
content fingerprint of the basis those outcomes were assigned against. That
artifact archives the text the digest is taken over, so a reader who finds two
versions with different digests can see which field moved.

That commit is therefore the ONLY boundary at which a verdict becomes visible -
the preflight widening above publishes names, never values. Every object the
verdict commit names - the two rebound column files and the basis - is durable
before it lands,
so there is no window in which a verdict is readable while what produced it is
still being written, and no failure mode that needs a repair path the run's own
in-memory state is the only source for. The basis is named by its own hash instead
of by the version it produced, which is what allows it to be written first: a
version-derived name cannot be computed until the commit that would depend on it
has already happened.

Two passes read the corpus, not one, and the second is what keeps the first
narrow. The main lineage must never carry a 384-wide text vector past the scan,
because everything after it shuffles; the task merge nonetheless needs one vector
per distinct task label. So the gather is its own reduction that projects the task
text vector and emits O(distinct tasks) rows, and the main scan reads the subtask
vector but reduces it to one int32 cell before yielding. Fusing the two into one
pass would put the widest column in the run through the dedup shuffle to save a
read of one column.

Every driver-side intermediate is bounded by config or by the annotation schema
rather than by the corpus. The label gather reduces the TASK vocabulary, which the
schema fixes; the quota's group table is at most distinct merged tasks times
``subtask_clusters + 1``; the metric report is a fixed bin count. The level-2 key
is a partition of the subtask embedding computed per row inside the scan, so the
free-form subtask vocabulary - which would otherwise be the merge's input size,
the quota's group count and a shuffle key at once - never reaches the driver in
any form.

See docs/curator/design/curator-next-curation.md.
"""

import hashlib
import io
import json
import math
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence

import attrs
import lance
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import ray
from lance.fragment import FragmentMetadata
from loguru import logger
from ray.data import Dataset

from cosmos_curator.core.utils.pixi_runtime_envs import ray_data_gpu_runtime_env
from cosmos_curator.core.utils.storage.storage_utils import StorageWriter, get_lance_storage_options
from cosmos_curator.next.core.ray_runtime import configure_ray_data_progress
from cosmos_curator.next.embeddings.schemas import (
    KEY_COLUMN,
    TEXT_DIM,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.curation import dedup, fairness, vectors
from cosmos_curator.next.recipes.curation.columns import (
    CANONICAL_TASK_COLUMN,
    CENTROIDS_ROOT_SUFFIX,
    CURATE_CLUSTER_ID,
    CURATE_COLUMNS,
    CURATE_SELECTION_REASON,
    CURATE_UPDATE_SCHEMA,
    DEDUP_KEY_COLUMN,
    DEDUP_SCORE_COLUMN,
    DISTANCE_COLUMN,
    FINGERPRINT_PROPERTY,
    FRAGMENT_COLUMN,
    FUSED_BLOCKS,
    KIND_PROPERTY,
    NO_DEDUP_GROUP,
    NO_DEDUP_GROUP_ZERO_NORM,
    NO_SUBTASK_CLUSTER,
    RAY_COUNT_COLUMN,
    SUBTASK_CLUSTER_COLUMN,
    SUBTASK_VECTOR_COLUMN,
    SUBTASK_WEIGHT_FIELD,
    TASK_COLUMN,
    TASK_VECTOR_COLUMN,
    TRANSACTION_KIND,
    VERDICT_ROW,
    WORKING_VECTOR_COLUMN,
    CurateReason,
    consumed_columns,
    consumed_groups,
    consumed_vectors,
    eligibility_filter,
    weighted_vectors,
)
from cosmos_curator.next.recipes.curation.config import CurateConfig
from cosmos_curator.next.utils.lance_utils import distinct_non_null_values, open_dataset

# The pixi environment holding cuML and cuPy. An internal constant, not config:
# no operator needs to change which environment the GPU stages run in, and the
# embeddings leg pins its own the same way. A Ray task whose imports live outside
# the driver's environment must name it via runtime_env, or it only works when
# the driver happens to have been launched there.
_GPU_ENV_NAME = "cuml"

# The worker-to-driver channel. A single string column STRUCTURALLY cannot carry
# column data back, which is what keeps driver state O(fragments) rather than
# O(rows) however wide the verdict rows were.
_RESULT_COLUMN = "write_result"
_RESULT_SCHEMA = pa.schema([pa.field(_RESULT_COLUMN, pa.large_string(), nullable=False)])

# The column ``from_items`` carries into the scan and the gather: one fragment id
# per work item. Fragments are fed directly rather than derived from a row
# address, so neither pass needs a shuffle to reach fragment granularity.
_FRAGMENT_ID_COLUMN = "fragment_id"
_FRAGMENTS_PER_TASK = 1

# Rows per scanner batch inside one fragment. It bounds the per-call score matrix
# (``n x k``, which ``CentroidAssigner.score`` does not chunk) and the per-call
# working-vector buffer, so it is a memory knob and not a throughput one.
_SCAN_ROW_BATCH = 8192

_FIT_TASK_GPUS = 1

# Measured ratio of cuML KMeans peak device memory to the raw sample bytes, and
# the device budget the estimate is reported against. Used only to LOG whether a
# configured ``fit_sample_rows`` fits one GPU: the ceiling is a device property,
# so a wrong guess here must not silently shrink an operator's sample.
#
# ASSUMED rather than probed, and deliberately unlike the dedup stage - which
# reads ``cuda.Device().mem_info`` and reserves a fraction of it. That stage runs
# INSIDE the GPU task, so the card is present and its own refusal is binding.
# This runs on the DRIVER, before the fit task is scheduled and in an environment
# that has no cuPy in it, so there is nothing here to ask. The two consequences
# are accepted rather than papered over: the number is advisory only, and it
# carries no reserve fraction, because the factor below was measured end to end
# against a real peak and so already contains the context and workspace cost that
# a reserve exists to cover.
_FIT_PEAK_FACTOR = 4
_ONE_GPU_BYTES = 80 * 1024**3

# Sample rows the subtask-text k-means is fitted on per centroid it has to place,
# capped by the operator's own fit_sample_rows. Expressed per centroid because
# that is the quantity a k-means sample has to cover: subtask_clusters is small
# and bounded by config, so tying the sample to it bounds the SECOND host matrix
# this task allocates instead of letting it track the fused one - which is
# 865-wide and sized for a k derived from the whole corpus. 100k rows per cell is
# far past where a 384-dimensional fit at k in the tens stops moving, and at the
# k=16 default it caps the subtask matrix at ~2.3 GiB - the figure the run's own
# fit-sample line prints, so the two agree.
_SUBTASK_FIT_ROWS_PER_CENTROID = 100_000

# Transient gather-row column names. Local to the two functions that produce and
# reduce them, but named rather than inlined because the rollup and the reduction
# are the same producer/consumer pair the routing names in columns.py exist for.
_GATHER_LABEL_COLUMN = "label"
_GATHER_COUNT_COLUMN = "rows"
_GATHER_VECTOR_COLUMN = "vector"
_GATHER_TIEBREAK_COLUMN = "vector_clip_id"
_GATHER_FRAGMENT_COLUMN = "vector_fragment_id"

# A gathered label matrix must be a 2-D (L, d) block; see _require_merge_matrix.
_MERGE_MATRIX_NDIM = 2

# Warn when more than one clip in this many changed fairness group in the task
# merge. Measures the CORPUS where fairness.merge_labels' own collapse line
# measures the vocabulary, and the two are independent: a merge can keep most of
# its representatives - so that line stays silent - while folding two of the
# largest labels and re-pooling a large share of the clips. Nothing about a merge
# is persisted, so both lines are the only signal either exists.
#
# Deliberately NOT a CurateConfig field, for the same reason its sibling in
# fairness is not: a threshold on a diagnostic is the coordinator's call, and
# nobody has evidence to set it per run.
_CLIPS_MOVED_WARN_FACTOR = 10

# Reported distributions: fixed bins over the closed range each metric can
# occupy. The bin COUNT is shared, because it is a property of the reporting
# precision rather than of either metric - 2048 bins put a value within ~0.001 of
# its true one on the radius range and ~0.0005 on the score range, both orders
# finer than any dedup_eps an operator would set.
#
# A histogram rather than a collect-and-sort. The percentile of an O(N) column
# cannot be taken on the driver at 250M rows - that is 1 GB of float32 before the
# sort - and Ray Data has no exact quantile aggregator, so each task reduces its
# own rows to a fixed-length count vector and the driver sums O(bins) of them.
_HISTOGRAM_BINS = 2048
_METRIC_COLUMN = "metric"
_HISTOGRAM_BIN_COLUMN = "histogram_bin"
_HISTOGRAM_ROWS_COLUMN = "histogram_rows"

_METRIC_HISTOGRAM: pa.Schema = pa.schema(
    [
        pa.field(_METRIC_COLUMN, pa.string(), nullable=False),
        pa.field(_HISTOGRAM_BIN_COLUMN, pa.int32(), nullable=False),
        pa.field(_HISTOGRAM_ROWS_COLUMN, pa.int64(), nullable=False),
    ]
)
"""One row per NON-EMPTY bin of one reduced batch: ``(metric, bin index, rows)``.

Sparse triples rather than a dense fixed-length vector, because Ray Data may split
or concatenate a UDF's output block: a dense vector's meaning depends on its
offset within the block, so a split would silently reduce a partial histogram as
if it were a whole one. A triple carries its own index, so the driver's fold is
correct under any block boundary and needs no ordering guarantee.

The metric label is what lets ONE reduction carry both distributions. Two passes
over a materialized 250M-row corpus to report two float32 columns would double the
scan for nothing; stamping the metric costs one dictionary-shaped string per
occupied bin.
"""

_RADIUS_QUANTILES: tuple[float, ...] = (0.50, 0.95)
_SCORE_QUANTILES: tuple[float, ...] = (0.99, 0.999)

# Candidate thresholds the retention score's upper tail is reported against, so
# an operator can read what a different dedup_eps WOULD have flagged. The ladder
# is free: each entry is the folded histogram's own tail above 1 - eps, which the
# driver already holds. Spanning an order of magnitude either side of the 0.01
# default, because the question it answers ("was my eps far too loose or far too
# tight") is not a local one.
#
# These are the counterfactual rungs only. The run's own dedup_eps is added at
# report time by _ladder_rungs, so the operating point is always on the ladder
# even when it lies outside this span.
_DEDUP_EPS_LADDER: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


@attrs.frozen
class _Metric:
    """One reported distribution: the in-flight column it reads and its bin range.

    Attributes:
        column: The column holding the per-row value. It doubles as the metric's
            label in the emitted triples, so one reduction folds two distributions
            without a second vocabulary to keep in step with the column names.
        upper: Top of the closed range the bins span. Values outside it clip into
            the end bins rather than being dropped or raising.

    """

    column: str
    upper: float

    @property
    def bin_width(self) -> float:
        """Return one bin's width: the resolution of every value read off it."""
        return self.upper / _HISTOGRAM_BINS


# The radius spans the full range of a cosine distance between unit vectors. The
# score is a cosine similarity between unit vectors, whose useful range is [0, 1]:
# row 0 of every group is forced to 0.0 and an antipodal pair scores negative, so
# bin 0 doubles as the underflow bucket for both.
_RADIUS_METRIC = _Metric(column=DISTANCE_COLUMN, upper=2.0)
_SCORE_METRIC = _Metric(column=DEDUP_SCORE_COLUMN, upper=1.0)
_REPORTED_METRICS: tuple[_Metric, ...] = (_RADIUS_METRIC, _SCORE_METRIC)


# The vector a label carries before one is observed for it. Read-only and shared:
# _LabelAccumulator.observe never mutates a vector in place, it rebinds.
_ZERO_TEXT_VECTOR: npt.NDArray[np.float32] = np.zeros(TEXT_DIM, dtype=np.float32)
# Every vector-less label accumulator shares this one array, so an in-place fold
# would corrupt all of them at once. Read-only makes that a raise, not a bug.
_ZERO_TEXT_VECTOR.flags.writeable = False

_GATHER_ROW: pa.Schema = pa.schema(
    [
        pa.field(_GATHER_LABEL_COLUMN, pa.string(), nullable=False),
        pa.field(_GATHER_COUNT_COLUMN, pa.int64(), nullable=False),
        pa.field(_GATHER_VECTOR_COLUMN, pa.list_(pa.float32(), TEXT_DIM), nullable=False),
        pa.field(_GATHER_TIEBREAK_COLUMN, pa.string(), nullable=True),
        pa.field(_GATHER_FRAGMENT_COLUMN, pa.int64(), nullable=True),
    ]
)
"""One row per distinct canonical TASK label seen by one gather task.

The tie-break columns are what make the merge reproducible. Several raw labels
fold onto one canonical label and each embeds differently, so "the" vector of a
canonical label is a CHOICE; it is resolved as the vector of the lowest
``(clip_id, fragment_id)`` carrying it, which no fragment completion order can
perturb. ``clip_id`` is unique only WITHIN a fragment, so the fragment id is
carried alongside it: two fragments may legitimately share a ``clip_id`` while
holding different vectors, and the pair breaks that tie globally. Both are NULL
together when this task saw the label only on rows whose task vector was NULL,
and the vector is then a zero row, which ``merge_labels`` reads as having no
direction.

There is no level column, because there is only one merged level. The level-2
group is a partition of the subtask embedding computed per ROW in the scan, so it
never reaches a driver-side reduction at all - which is the whole reason the
level-2 key was bounded.
"""

_SCAN_ROW: pa.Schema = pa.schema(
    [
        *VERDICT_ROW,
        pa.field(CANONICAL_TASK_COLUMN, pa.string(), nullable=False),
        pa.field(SUBTASK_CLUSTER_COLUMN, pa.int32(), nullable=False),
        pa.field(DISTANCE_COLUMN, pa.float32(), nullable=True),
        pa.field(WORKING_VECTOR_COLUMN, pa.list_(pa.float32(), vectors.FUSED_DIM), nullable=False),
    ]
)
"""What one scanned row carries into the retention pass: ``VERDICT_ROW`` plus four.

Built on ``VERDICT_ROW`` rather than restating it, so the four columns the write
reads cannot drift. A row the scan already judged ``invalid_embedding`` carries a
NULL distance - it was scored against no centroid, so it is a distance to
nothing - and a ZERO vector rather than a NULL one, because ``dedup`` rejects a
NULL vector inside a scored group and the zero row is only ever reachable on a
negative-key bypass, which returns before any vector is read.

The level-2 group is non-nullable and int32 like the other two routing keys: an
unassignable row carries ``NO_SUBTASK_CLUSTER`` rather than a NULL, so ONE key
type - ``fairness.Level2Key``, a ``(str, int)`` pair - serves the quota's key, the
selection shuffle's key, and the map ``select_within_quota`` looks itself up in.
Not because a null key cannot be shuffled: Ray Data groups NULL as its own group
(verified on Ray 2.55.1) and the group count deliberately relies on that for
``curate_selection_reason``, which is NULL for exactly the survivors
``fairness.survivor_group_counts`` keeps. A nullable cell would instead widen that
one key type to ``int | None`` in three places that have to agree on it.

Declared as a schema, not inferred, so a fragment holding no eligible row still
emits a block Ray Data can concatenate with the rest.
"""


class CurateWriteError(RuntimeError):
    """A write-back contract violation that no retry can repair.

    Raised for the states where the write cannot be trusted to mean what it says:
    a verdict group spanning more than one fragment, a ``clip_id`` repeated inside
    one fragment (Lance would resolve it by taking one matching row's value), duplicate
    or NULL ``clip_id`` values already stored in the target fragment, verdict
    ``clip_id`` values absent from that fragment, two metadata versions of one fragment
    reaching the commit, and a group naming a fragment the pinned version does not
    contain.

    Catchable by type only when raised on the driver. A raise inside a Ray Data
    UDF reaches the driver wrapped in a different type, so the abort is
    guaranteed by ANY exception propagating, not by this class.
    """


def ensure_curate_columns(dataset: lance.LanceDataset) -> int:
    """Add the ``curate_*`` columns if absent, in one metadata commit; return fields added.

    ``add_columns`` with an all-nullable schema rewrites no data file: Lance records the fields
    in the manifest and reads them as NULL for existing rows, so this cannot create a tombstone.
    The two columns are atomic siblings, so exactly one present is a corrupt schema rather than
    a state to repair - it means something outside Curate wrote one of its names.

    Args:
        dataset: The clips table to widen; ``add_columns`` advances this handle in place to the
            version the write-back must read and pin, so pass it onward.

    Raises:
        ValueError: If exactly one column is present, or a present one mismatches the declared
            type or nullability.

    """
    names = set(dataset.schema.names)
    present = [field.name for field in CURATE_COLUMNS if field.name in names]
    if not present:
        dataset.add_columns(CURATE_COLUMNS)
        return len(CURATE_COLUMNS)
    if len(present) != len(CURATE_COLUMNS):
        missing = [field.name for field in CURATE_COLUMNS if field.name not in names]
        msg = (
            f"curate columns are partially present on {dataset.uri}: has {present}, missing {missing}; "
            f"they are written together, so drop {present} and rerun"
        )
        raise ValueError(msg)
    for expected in CURATE_COLUMNS:
        stored = dataset.schema.field(expected.name)
        if not stored.type.equals(expected.type) or not stored.nullable:
            msg = (
                f"curate column {expected.name!r} on {dataset.uri} is {stored.type} "
                f"(nullable={stored.nullable}) but must be {expected.type} and nullable"
            )
            raise ValueError(msg)
    return 0


def update_one_fragment(
    group: pa.Table,
    *,
    uri: str,
    read_version: int,
    storage_options: dict[str, str] | None = None,
) -> pa.Table:
    """Write one fragment's verdicts onto EVERY row it stores; return its metadata as JSON.

    One group is one fragment's verdict rows, keyed on ``__frag``, written where they were read
    from. Nothing joins across fragments, so a ``clip_id`` shared with another fragment is
    resolved by that fragment's own call. Rows the group does not claim are written NULL rather
    than left alone, which is what keeps an earlier run's verdict from surviving into this one.

    Args:
        group: One fragment's verdicts: ``clip_id``, ``__frag``, ``__dedup_key``, the reason.
        uri: The clips table.
        read_version: The version every read of this run used.
        storage_options: Lance storage options for ``uri``.

    Raises:
        CurateWriteError: If the group names more than one fragment, a ``clip_id`` repeats
            within it, the persisted fragment carries duplicate or NULL ``clip_id`` values,
            a verdict ``clip_id`` is absent from that fragment, or the fragment is absent
            at ``read_version``.

    """
    # Guaranteed by groupby("__frag"), checked anyway because the failure is
    # SILENT: a row from another fragment matches no clip_id here, so its verdict
    # would be dropped with no error rather than written to the wrong place.
    fragments = pc.unique(group.column(FRAGMENT_COLUMN))  # type: ignore[attr-defined]
    if len(fragments) != 1:
        msg = f"one verdict group must name exactly one fragment of {uri}, got {fragments.to_pylist()}"
        raise CurateWriteError(msg)
    fragment_id = int(fragments[0].as_py())
    ids = group.column(KEY_COLUMN)
    # A duplicate key inside ONE fragment is the whole hazard. Lance resolves it
    # by taking one matching row's value, and Curate legitimately assigns
    # DIFFERENT verdicts to identical-vector rows - the second is a duplicate of
    # the first - so the write would be plausible and wrong. It is also what lets
    # the projection below key on this column. The check is free on data already
    # in hand; harmless across fragments.
    distinct = len(pc.unique(ids))  # type: ignore[attr-defined]
    if distinct != len(ids):
        msg = (
            f"fragment {fragment_id} of {uri} carries {len(ids)} verdict rows under "
            f"{distinct} distinct clip_id(s); refusing to let Lance pick one"
        )
        raise CurateWriteError(msg)
    payload = _write_fragment_columns(
        fragment_id,
        group,
        uri=uri,
        read_version=read_version,
        storage_options=storage_options,
    )
    return pa.table({_RESULT_COLUMN: pa.array([payload], type=pa.large_string())}, schema=_RESULT_SCHEMA)


def blank_one_fragment(
    batch: pa.Table,
    *,
    uri: str,
    read_version: int,
    storage_options: dict[str, str] | None = None,
) -> pa.Table:
    """Write NULL into both columns for every row of each fragment named in the batch.

    The complement of ``update_one_fragment``. A fragment holding no eligible row produces no
    verdict group, so a shuffle keyed on ``__frag`` can never reach it, and without this pass a
    fragment whose rows were all curated by an earlier run and are all ineligible now would keep
    that run's verdicts. Not a rare path: the table is appended as per-dataset slabs, so a
    modality absent for one whole dataset takes whole fragments out of the eligible set together.

    Args:
        batch: Work items carrying the fragment ids to blank.
        uri: The clips table.
        read_version: The version every read of this run used.
        storage_options: Lance storage options for ``uri``.

    Raises:
        CurateWriteError: If a fragment is absent at ``read_version``, or carries duplicate or
            NULL ``clip_id`` values.

    """
    empty = VERDICT_ROW.empty_table()
    payloads = [
        _write_fragment_columns(
            int(fragment_id),
            empty,
            uri=uri,
            read_version=read_version,
            storage_options=storage_options,
        )
        for fragment_id in batch.column(_FRAGMENT_ID_COLUMN).to_pylist()
    ]
    return pa.table({_RESULT_COLUMN: pa.array(payloads, type=pa.large_string())}, schema=_RESULT_SCHEMA)


def _write_fragment_columns(
    fragment_id: int,
    group: pa.Table,
    *,
    uri: str,
    read_version: int,
    storage_options: dict[str, str] | None,
) -> str:
    """Write both ``curate_*`` columns for every row of one fragment; return the commit payload.

    Shared by the claimed and the blanking pass, which differ only in whether ``group`` holds
    any row. Both write the whole fragment, so the guards below apply to every fragment of the
    table on every run rather than only to the ones that carry verdicts.

    Args:
        fragment_id: The fragment to write, already resolved by the caller.
        group: Verdict rows for that fragment; empty when the fragment is being blanked.
        uri: The clips table.
        read_version: The version every read of this run used.
        storage_options: Lance storage options for ``uri``.

    Returns:
        The JSON payload ``_collect`` reduces: fragment id, metadata, field ids, claimed rows.

    Raises:
        CurateWriteError: If the fragment is absent at ``read_version``, stores duplicate or
            NULL ``clip_id`` values, or is missing a ``clip_id`` the group claims.

    """
    fragment = lance.dataset(uri, version=read_version, storage_options=storage_options).get_fragment(fragment_id)
    if fragment is None:
        msg = f"fragment {fragment_id} is not present in {uri} at version {read_version}"
        raise CurateWriteError(msg)
    stored_ids = fragment.to_table(columns=[KEY_COLUMN]).column(KEY_COLUMN)
    if stored_ids.null_count:
        msg = (
            f"fragment {fragment_id} of {uri} carries {stored_ids.null_count} NULL "
            f"{KEY_COLUMN} value(s); refusing to join verdicts onto ambiguous rows"
        )
        raise CurateWriteError(msg)
    stored_distinct = len(pc.unique(stored_ids))  # type: ignore[attr-defined]
    if stored_distinct != len(stored_ids):
        msg = (
            f"fragment {fragment_id} of {uri} stores {len(stored_ids)} rows under "
            f"{stored_distinct} distinct {KEY_COLUMN} value(s); refusing to let "
            f"update_columns match one verdict to multiple rows"
        )
        raise CurateWriteError(msg)
    if group.num_rows:
        group_keys = group.column(KEY_COLUMN).combine_chunks()
        stored_values = stored_ids.combine_chunks()
        in_fragment = pc.is_in(group_keys, value_set=stored_values)  # type: ignore[attr-defined]
        missing_count = int(pc.sum(pc.invert(in_fragment)).as_py())  # type: ignore[attr-defined]
        if missing_count:
            missing_keys = pc.filter(group_keys, pc.invert(in_fragment))  # type: ignore[attr-defined]
            example_missing = missing_keys.slice(0, min(5, missing_count)).to_pylist()
            msg = (
                f"fragment {fragment_id} of {uri} is missing {missing_count} verdict "
                f"{KEY_COLUMN} value(s), for example {example_missing}; refusing a partial fragment update"
            )
            raise CurateWriteError(msg)
    update = _total_verdicts(group, stored_ids)
    reader = pa.RecordBatchReader.from_batches(CURATE_UPDATE_SCHEMA, update.to_batches())
    metadata, modified_field_ids = fragment.update_columns(reader, left_on=KEY_COLUMN, right_on=KEY_COLUMN)
    # These keys ARE _collect's contract, so a change here lands there in the
    # same edit. The metadata is embedded as the exact string FragmentMetadata
    # expects rather than as a nested object, so the round trip cannot lose a
    # field to re-encoding. "rows" counts the CLAIMED rows, not every row this
    # rewrites, because it is reported as the run's verdict count against
    # eligible_rows - so a blanking call reports 0 however many rows it NULLed.
    return json.dumps(
        {
            "fragment_id": fragment_id,
            "metadata_json": json.dumps(metadata.to_json()),
            "modified_field_ids": [int(field_id) for field_id in modified_field_ids],
            "rows": group.num_rows,
        }
    )


def _total_verdicts(group: pa.Table, stored_ids: pa.ChunkedArray) -> pa.Table:
    """Project verdict rows onto ``CURATE_UPDATE_SCHEMA``, one row per STORED row.

    Two mappings land here, and both turn something into a NULL that must not be readable as a
    value. ``__dedup_key`` is a routing value: a row that never entered the similarity pass
    carries a negative key, which must become a NULL ``curate_cluster_id`` so no sentinel is ever
    readable as a cluster. And a stored row the group does not claim takes NULL in both columns -
    carried explicitly, because ``update_columns`` overwrites a row present with NULL and
    preserves one that is merely absent.

    Args:
        group: The fragment's verdict rows; empty when the fragment is being blanked.
        stored_ids: Every ``clip_id`` the fragment stores, in offset order.

    Returns:
        One row per stored id, in stored order, NULL in both columns where unclaimed.

    """
    keys = group.column(DEDUP_KEY_COLUMN)
    negative = pc.less(keys, 0)  # type: ignore[attr-defined]
    cluster_id = pc.if_else(negative, pa.scalar(None, type=keys.type), keys)  # type: ignore[attr-defined]
    # index_in needs a unique value set, which update_one_fragment's duplicate
    # refusal guarantees. An unclaimed stored row maps to a null index, and take()
    # turns a null index into a null value - so the blanks cost no branch and the
    # empty-group case (the blanking pass) needs no special handling.
    positions = pc.index_in(stored_ids, value_set=group.column(KEY_COLUMN).combine_chunks())  # type: ignore[attr-defined]
    projected = pa.table(
        {
            KEY_COLUMN: stored_ids,
            CURATE_SELECTION_REASON: group.column(CURATE_SELECTION_REASON).take(positions),
            CURATE_CLUSTER_ID: cluster_id.take(positions),
        }
    )
    return projected.cast(CURATE_UPDATE_SCHEMA)


@attrs.frozen
class _CollectedWrite:
    """The worker payloads reduced to what one ``Update`` transaction needs.

    Attributes:
        fragments: Updated metadata, one entry per written fragment.
        field_ids: Union of the field ids those writes rebound.
        rows: Verdict rows CLAIMED across all of them - rows that received a
            non-NULL verdict, NOT rows the transaction rewrote, which is every
            stored row of every fragment. Claimed equals matched, because the
            write refuses before the join every key shape that would let the two
            diverge. ``_total_rows`` holds the sum to the eligible count.

    """

    fragments: tuple[FragmentMetadata, ...]
    field_ids: tuple[int, ...]
    rows: int


def _collect(payloads: Sequence[str]) -> _CollectedWrite:
    """Decode the worker payloads into the commit's inputs.

    Args:
        payloads: One JSON string per written fragment.

    Returns:
        The fragments, field ids, and row count the commit reports.

    Raises:
        CurateWriteError: If two payloads name the same fragment. Each fragment
            is one work item, so a repeat means a retried task's write was also
            counted, and committing two metadata versions of one fragment would
            make the outcome depend on their order.

    """
    decoded = [json.loads(payload) for payload in payloads]
    by_fragment = {
        int(entry["fragment_id"]): FragmentMetadata.from_json(str(entry["metadata_json"])) for entry in decoded
    }
    if len(by_fragment) != len(decoded):
        msg = (
            f"{len(decoded)} write result(s) named only {len(by_fragment)} distinct fragment(s); "
            f"refusing to commit two metadata versions of one fragment"
        )
        raise CurateWriteError(msg)
    field_ids = {int(field_id) for entry in decoded for field_id in entry["modified_field_ids"]}
    return _CollectedWrite(
        fragments=tuple(by_fragment.values()),
        field_ids=tuple(sorted(field_ids)),
        rows=sum(int(entry["rows"]) for entry in decoded),
    )


def _commit_properties(config: CurateConfig, *, centroids_fingerprint: str) -> dict[str, str]:
    """Return the identity to stamp on the commit.

    ``kind`` attributes the version to this leg; ``config_digest`` says which rules
    produced it, so two versions holding the same two columns are distinguishable.
    ``schema_version`` is stamped separately even though the digest covers it,
    because a digest is opaque and the generation is what a reader needs first.
    ``centroids_fingerprint`` names the basis object this version's cluster ids were
    assigned against, which is the only value that can distinguish "this version did
    not refit" from "this version's own basis is missing" - a directory listing
    cannot, because the two look identical from outside.

    The read version is deliberately NOT stamped: Lance persists it on the
    transaction itself, so a copy here could disagree with the version actually
    read.

    Args:
        config: The resolved run configuration.
        centroids_fingerprint: Content hash of the already-published basis artifact.

    Returns:
        Lance transaction properties.

    """
    return {
        KIND_PROPERTY: TRANSACTION_KIND,
        "schema_version": str(config.schema_version),
        "config_digest": config.result_defining_digest(),
        FINGERPRINT_PROPERTY: centroids_fingerprint,
    }


def _commit(
    dataset: lance.LanceDataset,
    collected: _CollectedWrite,
    *,
    properties: dict[str, str],
    storage_options: dict[str, str] | None = None,
) -> int:
    """Publish every written fragment as one ``Update`` transaction; return the version.

    The write is total, so a successful run names every fragment of the table here.

    Args:
        dataset: The handle pinned at the version every read used; its version is the read version.
        collected: The decoded worker payloads.
        properties: Identity stamped on the version; see ``_commit_properties``.
        storage_options: Lance storage options for the table.

    Raises:
        ValueError: If the commit fails, so a caller reports a cause not a traceback.

    """
    # This is the only point at which a VERDICT becomes visible; the preflight
    # widening publishes the two column names, never a value in them. A run that
    # aborts before here leaves the column files its finished workers wrote:
    # unreferenced by any manifest and therefore inert, so extra files after a
    # failure are garbage to collect, never a partial commit.
    read_version = int(dataset.version)
    transaction = lance.Transaction(
        read_version=read_version,
        operation=lance.LanceOperation.Update(
            updated_fragments=list(collected.fragments),
            fields_modified=list(collected.field_ids),
        ),
        transaction_properties=properties,
    )
    try:
        committed = lance.LanceDataset.commit(dataset.uri, transaction, storage_options=storage_options)
    except (OSError, RuntimeError) as e:
        # Lance routes only IO faults to OSError; its commit-conflict and
        # write-contention variants fall through to RuntimeError, so both must be
        # caught to name a cause.
        msg = f"curate commit against v{read_version} of {dataset.uri} failed ({e})"
        raise ValueError(msg) from e
    return int(committed.version)


@attrs.frozen
class MergeStats:
    """The task merge's realized cost, measured on the driver around the call.

    Attributes:
        labels_in: Distinct canonical task labels handed to ``merge_labels``
            (``L``).
        labels_out: Representatives returned (``R``). The walk streams a
            growing-prefix GEMV per label, so this is the realized inner extent
            of the ``O(L * R)`` similarity work, not merely an output size.
        clips_moved: Clips whose label was folded into another label's group, so
            whose fairness group the merge CHANGED. Counts the folded labels' own
            clips, never the leaders'.
        seconds: Wall-clock of the ``merge_labels`` call alone.

    """

    labels_in: int
    labels_out: int
    clips_moved: int
    seconds: float


@attrs.frozen
class CurateResult:
    """What one Curate run produced, for a caller that persists nothing itself.

    Attributes:
        clips_lance_uri: The table read and widened.
        read_version: The pinned version every read used, after widening.
        committed_version: The version the one ``Update`` produced.
        eligible_rows: Rows the predicate claimed; the denominator of everything.
        written_rows: Verdict rows the write claimed; always ``eligible_rows``,
            which the write-back enforces rather than reports.
        requested_k: ``k`` derived from ``target_mean_cluster_rows``.
        effective_k: Locality centroids the fit actually returned.
        subtask_k: Subtask-text centroids the fit actually returned, observed
            rather than requested; ``0`` when no level-2 basis was fitted.
        fit_rows: Rows the locality k-means sample held.
        fairness_groups: Distinct merged ``(task, cluster)`` groups holding a
            survivor; the ``G`` of the quota arithmetic.
        unfunded_groups: Of those groups, how many drew a quota of zero, so the
            residual order alone chose among siblings. Never persisted.
        target: The resolved keep-count the quota was allocated at.
        reason_counts: Rows per ``CurateReason``; sums to ``eligible_rows``.
        merge_stats: The task merge's realized ``L``, ``R``, moved clips and
            wall-clock.
        centroids_uri: The basis artifact, keyed by the hash its commit stamps.

    """

    clips_lance_uri: str
    read_version: int
    committed_version: int
    eligible_rows: int
    written_rows: int
    requested_k: int
    effective_k: int
    subtask_k: int
    fit_rows: int
    fairness_groups: int
    unfunded_groups: int
    target: int
    reason_counts: dict[str, int]
    merge_stats: MergeStats
    centroids_uri: str


@attrs.frozen
class _ReadSpec:
    """What a worker needs to reopen the pinned table and read its eligible rows.

    One object rather than four arguments because every read of the run - the fit,
    the label gather, the scan, the write - takes exactly these four, and a stage
    that reopened at a different version or predicate would read a different
    corpus while reporting success.

    Attributes:
        uri: The clips table.
        read_version: The version pinned after the schema widening.
        predicate: The eligibility filter, built once from the weights.
        storage_options: Lance storage options for ``uri``.

    """

    uri: str
    read_version: int
    predicate: str
    storage_options: dict[str, str] | None


@attrs.frozen
class _Source:
    """The pinned, contract-checked table every stage of one run reads.

    Attributes:
        dataset: Driver handle pinned at ``read.read_version``, already widened.
            It is also what the commit is issued against.
        read: How every worker reopens it.
        fragment_ids: Fragment ids in manifest order - the work items, and the
            order the fit sampler takes its prefix from.
        fragment_rows: Physical rows per fragment, parallel to ``fragment_ids``.
            REPORTING ONLY - they feed the fit log's coverage and overshoot
            figures; nothing budgets or allocates from them.
        fragment_eligible_rows: Predicate-passing rows per fragment, parallel to
            ``fragment_ids``. The fit sampler budgets its prefix AND sizes its
            matrices on these, so a large but fully ineligible leading fragment
            can neither end the prefix on an empty sample nor inflate the
            allocation.
        eligible_rows: Rows matching the predicate at that version; the sum of
            ``fragment_eligible_rows``.
        producers: ``{provenance_column: identity}`` for every consumed column
            that recorded one; a group can be absent entirely. It is what makes
            the fused space one metric instead of several, so the run records it
            into the centroids artifact beside the block order and weights. See
            ``_require_recorded_producer`` for when an absence is legal.

    """

    dataset: lance.LanceDataset
    read: _ReadSpec
    fragment_ids: tuple[int, ...]
    fragment_rows: tuple[int, ...]
    fragment_eligible_rows: tuple[int, ...]
    eligible_rows: int
    producers: dict[str, str]


# One value beyond the single one the gate tolerates, which is all it takes to
# tell one identity from more than one.
_PRODUCERS_READ = 2


def _weight_map(config: CurateConfig) -> dict[str, float]:
    """Return the per-block weight dict the kernels take, keyed by fused-block field.

    Read off ``FUSED_BLOCKS`` rather than spelled out, so a new modality reaching
    the fused order without a ``ModalityWeights`` field fails here with the field
    name instead of silently contributing nothing.
    """
    return {field: float(getattr(config.weights, field)) for _, field in FUSED_BLOCKS}


_VECTOR_WIDTHS: dict[str, int] = {
    TASK_VECTOR_COLUMN: TEXT_DIM,
    **{group.primary_vector: width for (group, _), width in zip(FUSED_BLOCKS, vectors.BLOCK_DIMS, strict=True)},
}
"""Declared width of every vector column this leg can read, weights aside.

Lives here rather than beside the read set in ``columns`` because the fused
widths come from ``vectors.BLOCK_DIMS`` and ``vectors`` imports ``columns``.
Which of these a given run checks is still ``columns``' answer: preflight
indexes this by ``consumed_columns``, so the set checked for geometry is the
set required to exist.
"""


def _require_source_columns(dataset: lance.LanceDataset, weights: Mapping[str, float]) -> None:
    """Reject a table this run cannot read: a missing or mistyped source column.

    Raises:
        ValueError: Naming what is missing, mistyped, or mis-sized. The ``clip_id``
            type check is the load-bearing one: ``update_columns`` rejects a
            key-type mismatch inside the write worker, which is after the fit and
            the whole de-duplication pass, so this moves that failure onto the
            driver. The width checks do the same for a column stored at a
            geometry this run cannot fuse or merge.

    """
    schema = dataset.schema
    names = set(schema.names)
    # The task text vector is required regardless of weight, because the task
    # merge runs on every run and it is not a fused block, so no weight would ever
    # pull it in. The SUBTASK text vector is required exactly when its block
    # carries weight, which is the same rule the eligibility predicate applies to
    # it: at subtask=0.0 nothing reads the column, the level-2 partition is not
    # fitted, and a table that never had it is curatable.
    #
    # subtask_name itself is required by nothing. The level-2 key is a partition of
    # the subtask EMBEDDING, so the prose is never read - which is precisely what
    # took the corpus-scale string vocabulary out of this leg.
    required = [KEY_COLUMN, TASK_COLUMN, *consumed_columns(weights)]
    missing = sorted({name for name in required if name not in names})
    if missing:
        msg = f"{dataset.uri} is missing column(s) {missing} that this run reads; has {sorted(names)}"
        raise ValueError(msg)
    for name in (KEY_COLUMN, TASK_COLUMN):
        stored = schema.field(name).type
        if not pa.types.is_string(stored):
            msg = (
                f"{name} on {dataset.uri} is {stored} but must be string; the write joins on "
                f"{KEY_COLUMN} and Lance rejects a key-type mismatch inside the write worker"
            )
            raise ValueError(msg)
    # Vector WIDTH, gated on the driver from the declared fixed_size_list type.
    # This is a metadata read, not a vector read, so it belongs in preflight:
    # without it a mis-sized column reaches _label_rollup inside a Ray worker or
    # vectors.working_vectors inside the GPU fit task, i.e. after the run has
    # already claimed cluster and GPU resources.
    #
    # Checked over consumed_columns, the SAME set required to exist above, so a
    # column this run reads cannot be present-but-unchecked. _VECTOR_WIDTHS is a
    # pure lookup and raises KeyError on a consumed vector whose geometry nothing
    # declares - the correct failure, since defaulting a width would gate the
    # column against a dimension no one chose for it.
    for name in consumed_columns(weights):
        width = _VECTOR_WIDTHS[name]
        stored = schema.field(name).type
        if (
            not pa.types.is_fixed_size_list(stored)
            or int(stored.list_size) != width
            or stored.value_type != pa.float32()
        ):
            msg = f"{name} on {dataset.uri} stores {stored} but this run needs fixed_size_list<float32>[{width}]"
            raise ValueError(msg)


def _require_recorded_producer(
    dataset: lance.LanceDataset,
    group: EmbeddingColumnGroup,
    weights: Mapping[str, float],
    recorded: Mapping[str, str],
) -> None:
    """Refuse a consumed group holding vectors no provenance column accounts for.

    The floor under ``_require_single_producer``: an identity that agrees with
    itself says nothing when the column recording it is empty while the vectors
    beside it are not.

    Args:
        dataset: The clips table, read before the schema widening.
        group: One group ``consumed_groups`` returned.
        weights: Per-block weight keyed by the ``FUSED_BLOCKS`` weight field.
        recorded: The identities already resolved for this group; a provenance
            column missing from it recorded nothing at all.

    Raises:
        ValueError: If any vector this run reads from ``group`` is filled while
            one of the group's provenance columns records nothing.

    """
    unrecorded = [column for column in group.provenance_columns if column not in recorded]
    if not unrecorded:
        return
    consumed = consumed_vectors(group, weights)
    # A real scan, linear in rows: a fixed-size-list column carries no scalar
    # index and no Lance statistic answers IS NOT NULL, so this reads the
    # column's validity. It runs only when a consumed group recorded no identity
    # for some column - usually a table about to be refused, but NOT always: a
    # text group nobody filled, curated at zero subtask weight, is read here and
    # still passes, because eligibility_filter tests only weighted blocks and so
    # never names the task vector. A completing run can therefore pay this once.
    # The exact count is kept rather than an early-terminating probe because the
    # row figure is what tells an operator whether a stray partial backfill or
    # the whole group is at fault; count_rows discards **kwargs in pylance 9, so
    # bounding it would mean an ``ds.sql(... LIMIT 1)`` scanner, not a ``limit=``.
    filled_rows = int(dataset.count_rows(filter=" OR ".join(f"{column} IS NOT NULL" for column in consumed)))
    if filled_rows == 0:
        return
    msg = (
        f"embedding group {group.name!r} on {dataset.uri} has {filled_rows} row(s) with a vector this "
        f"run reads ({list(consumed)}) but no producer identity in {unrecorded}; vectors without a "
        f"recorded producer cannot be curated in one metric. Re-run the embeddings leg to refill "
        f"the group - an ordinary run recomputes any row whose recorded identity is missing or "
        f"names another computation - then curate again"
    )
    raise ValueError(msg)


def _require_single_producer(dataset: lance.LanceDataset, weights: Mapping[str, float]) -> dict[str, str]:
    """Resolve the one producer each consumed provenance column names.

    Shape is not a sufficient source contract: two producers of the same width -
    two text models, or two action PCA bases - fuse into one cosine space whose
    distances stay finite and plausible while meaning nothing. See
    ``docs/curator/design/curator-next-curation.md``, "Producer identity is part
    of the source contract".

    Args:
        dataset: The clips table, read before the schema widening.
        weights: Per-block weight keyed by the ``FUSED_BLOCKS`` weight field.

    Returns:
        ``{provenance_column: identity}`` for every consumed column that named
        one; a column that is entirely NULL is absent from the mapping.

    Raises:
        ValueError: If a consumed group's provenance column is absent or names
            more than one producer, or if ``_require_recorded_producer`` refuses
            the group.

    """
    identities: dict[str, str] = {}
    names = set(dataset.schema.names)
    for group in consumed_groups(weights):
        group_identities: dict[str, str] = {}
        for column in group.provenance_columns:
            if column not in names:
                msg = (
                    f"{dataset.uri} has no column {column!r}, so the producer of embedding group "
                    f"{group.name!r} cannot be established; the group is partially present, so detach "
                    f"its remaining columns and re-run the embeddings leg to refill it. Table has "
                    f"{sorted(names)}"
                )
                raise ValueError(msg)
            # One distinct-value scan per consumed provenance column, bounded at
            # _PRODUCERS_READ values: a pushed-down pass over one encoded string
            # column, whatever the corpus length.
            producers = distinct_non_null_values(dataset, column, max_values=_PRODUCERS_READ)
            if len(producers) > 1:
                msg = (
                    f"embedding group {group.name!r} on {dataset.uri} was filled by more than one "
                    f"producer: {column!r} holds {list(producers)}. This run would compare those rows "
                    f"in one cosine space, where vectors from two producers share no geometry. Two "
                    f"producers is what a group looks like PART-WAY through a rebuild, so finish the "
                    f"embeddings leg - an ordinary re-run recomputes every row still naming the older "
                    f"computation - then curate again"
                )
                raise ValueError(msg)
            if producers:
                group_identities[column] = producers[0]
        # Per COLUMN, not per group: the action group records its descriptor
        # version and its PCA basis separately, and either one alone leaves the
        # fused geometry unattributed.
        _require_recorded_producer(dataset, group, weights, group_identities)
        identities.update(group_identities)
    return identities


def _preflight(config: CurateConfig) -> _Source:
    """Open, contract-check and widen the table, then pin the version every read uses.

    Data-contract checks only: a table this run cannot read, cannot compare in one
    metric, or that holds no claimable row. No VECTOR data is read here - the
    declared widths and ``float32`` element types are gated from the schema
    (metadata) in ``_require_source_columns``, and per-row finiteness stays the
    fit's job via ``vectors._decode_blocks``. Data reads are confined to
    ``_require_single_producer``: one bounded distinct scan per consumed
    provenance column, plus - only for a group that recorded no identity for some
    column - one filtered count over the vectors this run consumes from it,
    because no metadata Lance exposes can prove either that a column holds a
    single value or that a vector column is empty.

    Raises:
        FileNotFoundError: If no table exists at ``clips_lance_uri``.
        ValueError: If a source column is missing or mistyped, if a consumed
            embedding group names more than one producer or holds vectors whose
            producer no provenance column records, if no block carries weight, or
            if no row satisfies the eligibility predicate.

    """
    storage_options = get_lance_storage_options(config.clips_lance_uri, profile_name=config.storage_profile)
    dataset = open_dataset(config.clips_lance_uri, storage_options=storage_options)
    if dataset is None:
        msg = f"no Lance table at {config.clips_lance_uri}"
        raise FileNotFoundError(msg)

    weights = _weight_map(config)
    predicate = eligibility_filter(weights)
    _require_source_columns(dataset, weights)
    # BEFORE the widening, so a table this run refuses gains no version. Safe for
    # the version pinned below because the widening is metadata-only: it cannot
    # move a provenance value. It does assume no embeddings fill is running
    # concurrently, which is not a supported mode.
    producers = _require_single_producer(dataset, weights)
    added = ensure_curate_columns(dataset)
    # AFTER the widening: it is a commit, so a version read before it would name a
    # schema without the columns the write is about to rebind.
    read_version = int(dataset.version)
    logger.info(f"curate preflight: {config.clips_lance_uri} v{read_version}, {added} column(s) added")
    logger.info(f"curate preflight: producer per consumed provenance column: {dict(sorted(producers.items()))}")

    # Per-fragment eligible counts, not a single corpus count: the fit sampler
    # needs the breakdown to budget its prefix on eligible rows, and summing them
    # is the same predicate evaluation the corpus-wide count would have paid, so
    # this is cost-neutral rather than an extra pass. Counted from THIS handle, so
    # they describe read_version exactly - the fit also SIZES its matrices from
    # them (see _FitPrefix), and a count taken against another version would
    # under-allocate into a silently truncated sample rather than an error.
    fragments = dataset.get_fragments()
    fragment_ids = tuple(int(fragment.fragment_id) for fragment in fragments)
    fragment_rows = tuple(int(fragment.count_rows()) for fragment in fragments)
    fragment_eligible_rows = tuple(int(fragment.count_rows(filter=predicate)) for fragment in fragments)
    eligible_rows = sum(fragment_eligible_rows)
    if eligible_rows == 0:
        msg = (
            f"no row of {config.clips_lance_uri} v{read_version} satisfies {predicate!r}; "
            f"the weighted modalities have not been embedded on this corpus"
        )
        raise ValueError(msg)

    source = _Source(
        dataset=dataset,
        read=_ReadSpec(
            uri=config.clips_lance_uri,
            read_version=read_version,
            predicate=predicate,
            storage_options=storage_options,
        ),
        fragment_ids=fragment_ids,
        fragment_rows=fragment_rows,
        fragment_eligible_rows=fragment_eligible_rows,
        eligible_rows=eligible_rows,
        producers=producers,
    )
    logger.info(
        f"curate preflight: {eligible_rows} eligible of {int(dataset.count_rows())} row(s) "
        f"across {len(source.fragment_ids)} fragment(s) under {predicate!r}"
    )
    return source


def _requested_k(eligible_rows: int, target_mean_cluster_rows: int) -> int:
    """Derive the cluster count from the target mean cluster size; warn at the degenerate 1.

    ``k == 1`` is benign for de-duplication up to a point - one exhaustive cluster
    has perfect duplicate recall and no boundary false negatives - but it leaves
    ``curate_cluster_id`` carrying no information, so it is reported rather than
    silently accepted.

    Past a point it is not benign at all: the single cluster holds every eligible
    row, and the retention stage loads one cluster onto one card, so above the
    per-group device ceiling that stage refuses. The clause naming that is
    ADVISORY, matching ``_FIT_PEAK_FACTOR``'s precedent - the driver has no device
    to ask, so it evaluates ``dedup.max_group_rows`` against an assumed total and
    the in-UDF refusal stays the binding check.
    """
    k = max(1, math.ceil(eligible_rows / target_mean_cluster_rows))
    if k != 1:
        return k
    ceiling = dedup.max_group_rows(width=vectors.FUSED_DIM, device_total_bytes=_ONE_GPU_BYTES)
    beyond_one_card = (
        ""
        if eligible_rows <= ceiling
        else (
            f", and that one cluster holds all {eligible_rows} of them, past the {ceiling} rows a "
            f"{_ONE_GPU_BYTES / 1024**3:.0f} GiB card can score at any GEMM tile size, so the retention "
            f"stage will refuse it - lower target_mean_cluster_rows to raise k and split the corpus"
        )
    )
    logger.warning(
        f"curate fit: k=1 for {eligible_rows} eligible row(s) at target_mean_cluster_rows="
        f"{target_mean_cluster_rows}; de-duplication is exhaustive and curate_cluster_id is constant"
        f"{beyond_one_card}"
    )
    return k


def _subtask_sample_rows(subtask_k: int, fit_sample_rows: int) -> int:
    """Return the row cap for the subtask-text fit: per-centroid coverage, never above the budget.

    The operator's ``fit_sample_rows`` remains the hard ceiling, so this can only
    ever ask for LESS host memory than the locality fit already spends.
    """
    return min(fit_sample_rows, subtask_k * _SUBTASK_FIT_ROWS_PER_CENTROID)


@attrs.frozen
class _FitPrefix:
    """The sampled fragment prefix, and the rows the fit can actually fill from it.

    Attributes:
        fragment_ids: The sampled prefix, in manifest order.
        sample_row_cap: Rows the locality matrix is allocated for, which is
            ``min(fit_sample_rows, the prefix's eligible rows)``. Eligible rather
            than physical, because the sample scan reads only predicate-passing
            rows: a prefix the predicate thins out cannot fill a bound taken from
            what it stores, so taking one would reserve host memory for rows that
            never arrive.
        subtask_sample_row_cap: The same bound for the subtask-text matrix. Its
            own budget never exceeds the locality one, so this stays at or below
            ``sample_row_cap``.

    """

    fragment_ids: tuple[int, ...]
    sample_row_cap: int
    subtask_sample_row_cap: int


def _fit_sample_fragments(source: _Source, fit_sample_rows: int, subtask_sample_rows: int) -> _FitPrefix:
    """Return the fragment prefix the locality basis is fitted on, and the rows it can fill.

    The prefix ends on the fragment that REACHES the eligible sample budget, or on
    the manifest's last when the corpus holds fewer eligible rows than that - it
    undershoots rather than failing, and at least one fragment is always taken.
    Bounded by ELIGIBLE (predicate-passing) rows: preflight counted those per
    fragment, so no data scan is needed and the result is deterministic under any
    scheduling. Eligible rather than physical, because a fully ineligible leading
    fragment would fill a physical budget by itself and leave nothing to fit.

    It takes each fragment BEFORE testing the budget, so the prefix crosses the
    budget rather than stopping short, and ``_fit_sample`` truncates inside the
    crossing fragment. The overshoot is bounded against stopping short, never
    against the budget: this prefix is the short one plus at most one fragment.
    ``rows`` has no budget-relative ceiling - zero-eligible leading fragments add
    physical rows and none eligible - so it is the coverage figure the log prints
    and bounds no allocation; the returned caps bound the matrices.

    What the prefix does not give is a uniform sample - it is the earliest-written
    fragments - so a corpus whose write order correlates with content fits a basis
    on that correlation. Since the cap cuts inside a fragment, the effective sample
    is a prefix of ROWS in write order, a strictly stronger bias than a prefix of
    fragments.
    """
    # Keeping this shape is a decision, not an oversight: a strided or random
    # sampler is rejected on determinism and needing no scan, and the measurement
    # that gates revisiting it is named under "Before proposing a different fit
    # sampler" in docs/curator/guides/curate-runbook.md.
    taken: list[int] = []
    rows = 0
    eligible = 0
    for fragment_id, fragment_rows, fragment_eligible in zip(
        source.fragment_ids, source.fragment_rows, source.fragment_eligible_rows, strict=True
    ):
        taken.append(fragment_id)
        rows += fragment_rows
        eligible += fragment_eligible
        # Tested AFTER the take, so a leading fragment holding one eligible row
        # cannot end the prefix at one row. The fit would not fail on that: it
        # clamps k to the sample and succeeds, having partitioned the corpus by
        # something the config never asked for.
        if eligible >= fit_sample_rows:
            break
    # What the fit will actually allocate, which is NOT `rows`: the matrices hold
    # ELIGIBLE rows under the budget, so `rows` over-reports the allocation twice
    # over - by the prefix's overshoot past the budget (measured at 11.8x on a
    # two-fragment table), and by whatever share of the prefix the predicate
    # rejects. Both numbers are logged because they answer different questions -
    # `rows` says how much of the corpus the basis saw, the caps say what it cost.
    sample_row_cap = min(fit_sample_rows, eligible)
    subtask_row_cap = min(subtask_sample_rows, eligible)
    sample_bytes = sample_row_cap * vectors.FUSED_DIM * 4
    subtask_bytes = subtask_row_cap * TEXT_DIM * 4
    # The ratio is how much of the eligible corpus the prefix can cover. Use the
    # eligible row cap, not the prefix's physical rows, so a mostly-ineligible
    # prefix cannot report more than 100% coverage.
    sampled_share = sample_row_cap / source.eligible_rows if source.eligible_rows else 0.0
    logger.info(
        f"curate fit sample: {len(taken)} of {len(source.fragment_ids)} fragment(s) holding {rows} "
        f"physical row(s); the basis is fitted on the first {sample_row_cap} eligible of them "
        f"(budget {fit_sample_rows}), at most ~{sampled_share:.1%} of the {source.eligible_rows} eligible "
        f"row(s), a ~{sample_bytes / 1024**3:.1f} GiB host matrix, plus a "
        f"~{subtask_bytes / 1024**3:.1f} GiB subtask-text matrix from the same pass"
    )
    # DEVICE peak only, and it says so: the host matrix above is a separate cost
    # this factor models nothing of, and widening the check to cover it would mean
    # inventing a host-memory budget - which is the same mistake as guessing a
    # device one, and the fit task cannot know its share of node RAM anyway. The
    # host number is therefore reported as fact and left to the operator, who does.
    device_peak_bytes = sample_bytes * _FIT_PEAK_FACTOR
    if device_peak_bytes > _ONE_GPU_BYTES:
        fitting_rows = _ONE_GPU_BYTES // (vectors.FUSED_DIM * 4 * _FIT_PEAK_FACTOR)
        logger.warning(
            f"curate fit sample: ~{device_peak_bytes / 1024**3:.1f} GiB estimated device peak for the "
            f"k-means on {sample_row_cap} row(s) exceeds {_ONE_GPU_BYTES / 1024**3:.0f} GiB; lower "
            f"fit_sample_rows to at most {fitting_rows} so the fit task does not run out of memory"
        )
    return _FitPrefix(
        fragment_ids=tuple(taken),
        sample_row_cap=sample_row_cap,
        subtask_sample_row_cap=subtask_row_cap,
    )


@attrs.frozen
class _FitSpec:
    """What the k-means task fits, beyond how to read the table.

    Attributes:
        fragment_ids: The sampled fragment prefix, in manifest order.
        weights: Per-block fusion weight; the metric the locality basis is fitted
            in, and what decides whether a subtask basis is fitted at all.
        requested_k: The driver's locality ``k``; the task may only clamp it
            downward.
        subtask_k: The configured level-2 group count; likewise clamp-only.
        sample_row_cap: Hard row bound on the locality sample: the smaller of the
            configured budget and the rows the sample scan can actually return, so
            host and device memory are bounded by both.
        subtask_sample_row_cap: The same bound for the subtask-text sample, always
            at or below ``sample_row_cap``.
        random_state: Seed for BOTH fits. One seed, because the two bases are
            produced by one task from one sample and reproducing a run means
            reproducing both.

    """

    fragment_ids: tuple[int, ...]
    weights: dict[str, float]
    requested_k: int
    subtask_k: int
    sample_row_cap: int
    subtask_sample_row_cap: int
    random_state: int


@attrs.frozen
class _FitSample:
    """The two host matrices one fit pass produces.

    Attributes:
        fused: ``(n, FUSED_DIM)`` working vectors; the locality metric.
        subtask: ``(m, TEXT_DIM)`` unit subtask-text vectors, or ``None`` when
            there is nothing to fit a level-2 basis on - either because the
            subtask block carries no weight, or because no sampled row carried a
            usable subtask direction. ``m <= n`` because the two matrices have
            independent row caps.

    """

    fused: npt.NDArray[np.float32]
    subtask: npt.NDArray[np.float32] | None


def _fit_sample(read: _ReadSpec, spec: _FitSpec) -> _FitSample:
    """Build both fit matrices in ONE pass over the sampled fragment prefix.

    The fused matrix is routed through ``working_vectors`` rather than reading
    vectors raw. That decoder's width check is the only block-width gate in the
    landed path, so running it here means a corpus embedded against a different
    geometry fails before the k-means burns a fit rather than partway through the
    row scan.

    The subtask matrix is filled from the SAME scanner batches, which is what
    makes the level-2 basis cost no extra read: when the subtask block carries
    weight it is already in the projection, and when it does not there is nothing
    to fit. Its rows are the sample's PRESENT and usable subtask directions - the
    same pair of gates ``_subtask_cells`` applies at the scan, so the basis is
    fitted over exactly the population it will later be scored against. They are
    NOT restricted to the fused ``keep`` mask, because a row unusable in some
    OTHER block still has a perfectly good subtask direction and excluding it
    would bias the level-2 basis by whatever makes a row invalid.

    Raises:
        ValueError: If a sampled fragment is absent at the pinned version, if a
            weighted block is stored at a width this run cannot fuse, or if the
            whole sample holds no usable vector.

    """
    dataset = lance.dataset(read.uri, version=read.read_version, storage_options=read.storage_options)
    fits_subtask = spec.weights[SUBTASK_WEIGHT_FIELD] > 0.0
    columns = sorted(set(weighted_vectors(spec.weights)))
    fragments = []
    for fragment_id in spec.fragment_ids:
        fragment = dataset.get_fragment(int(fragment_id))
        if fragment is None:
            msg = f"fragment {fragment_id} is not present in {read.uri} at version {read.read_version}"
            raise ValueError(msg)
        fragments.append(fragment)
    # Filled in place rather than concatenated from a list of batches, because a
    # join holds the batches AND their copy at once - at the default cap that
    # doubles a 12.9 GiB sample to 25.8 GiB of host memory inside the fit task,
    # which the device-side budget WARNING models nothing of. Both caps arrive
    # already bounded by the prefix's ELIGIBLE rows (see `_FitPrefix`), which is
    # an exact ceiling on what the filtered scan below can return, so a corpus
    # smaller than the cap - or one the predicate thins out - allocates for the
    # rows it can fill rather than for the rows it stores.
    ceiling = spec.sample_row_cap
    subtask_ceiling = spec.subtask_sample_row_cap if fits_subtask else 0
    fused_sample = np.empty((ceiling, vectors.FUSED_DIM), dtype=np.float32)
    subtask_sample = np.empty((subtask_ceiling, TEXT_DIM), dtype=np.float32)
    rows = subtask_rows = 0
    for fragment in fragments:
        scanner = fragment.scanner(columns=columns, filter=read.predicate, batch_size=_SCAN_ROW_BATCH)
        for record_batch in scanner.to_batches():
            table = pa.Table.from_batches([record_batch])
            working = vectors.working_vectors(table, spec.weights).working
            take = min(working.shape[0], ceiling - rows)
            fused_sample[rows : rows + take] = working[:take]
            rows += take
            if subtask_rows < subtask_ceiling:
                matrix, present = _text_matrix(table, SUBTASK_VECTOR_COLUMN)
                unit, usable = vectors.unit_rows(matrix)
                # A NULL slot of a fixed-size-list column still occupies its
                # width in the child buffer with values Arrow does not define, so
                # presence has to gate the decode rather than be inferred from
                # the norm. Today the eligibility predicate already requires this
                # column non-NULL whenever the block is weighted, so the mask is
                # all-true here - but that predicate is owned elsewhere, and this
                # is the same gate assign_clusters applies at the scan.
                directed = unit[usable & present]
                subtask_take = min(directed.shape[0], subtask_ceiling - subtask_rows)
                subtask_sample[subtask_rows : subtask_rows + subtask_take] = directed[:subtask_take]
                subtask_rows += subtask_take
            # Both caps must be reached before the read stops, or a small subtask
            # cap would truncate the locality sample it is unrelated to.
            if rows >= ceiling and subtask_rows >= subtask_ceiling:
                break
        if rows >= ceiling and subtask_rows >= subtask_ceiling:
            break
    if rows == 0:
        msg = (
            f"the fit sample of {read.uri} v{read.read_version} holds no usable vector; "
            f"every sampled row was non-finite or zero-norm"
        )
        raise ValueError(msg)
    # A weighted-but-directionless subtask population is not an error here: the
    # predicate guarantees the column is non-NULL, not that it has a direction.
    # It is a LOSS rather than a refusal, and it is not distinguishable from an
    # unweighted block once the matrix is gone, so the driver reports the two
    # apart from the weights it still holds (see _report_subtask_basis).
    return _FitSample(fused=fused_sample[:rows], subtask=subtask_sample[:subtask_rows] if subtask_rows else None)


@attrs.frozen(eq=False)
class _FitBases:
    """What one GPU fit task hands back, before the driver validates it.

    ``eq=False`` because it holds ndarrays.

    Attributes:
        locality: ``(effective_k, FUSED_DIM)`` RAW locality centroids.
        subtask: ``(subtask_k, TEXT_DIM)`` RAW level-2 centroids, or ``None`` when
            the sample held no subtask matrix to fit one on.
        fit_rows: Usable rows the locality sample held.

    """

    locality: npt.NDArray[np.float32]
    subtask: npt.NDArray[np.float32] | None
    fit_rows: int


def _fit_kmeans(read: _ReadSpec, spec: _FitSpec) -> _FitBases:
    """Fit both bases on one GPU; return the RAW centroids and the sample's rows.

    ONE task rather than two, because the read is the expensive part and both
    bases come from the same scanner batches. A second GPU task would re-read a
    fragment prefix to fit a basis that is 384-wide at a k in the tens.

    Raises:
        ValueError: From ``_fit_sample``, for an unreadable or unusable sample.

    """
    # Deferred and untyped: cuML lives in the GPU pixi environment named by
    # _GPU_ENV_NAME, not in the driver's, so this module must import without it.
    from cuml.cluster import KMeans  # type: ignore[import-not-found]  # noqa: PLC0415

    sample = _fit_sample(read, spec)

    def fit(matrix: npt.NDArray[np.float32], requested: int, label: str) -> npt.NDArray[np.float32]:
        # KMeans requires k <= n_samples and the driver cannot know a sample's
        # realized size, because eligibility is a filter rather than a count.
        # Clamping here and reporting the observed k back keeps the artifact honest.
        effective = min(requested, matrix.shape[0])
        if effective != requested:
            logger.warning(f"curate fit: {label} k clamped from {requested} to {effective}, the sample's row count")
        logger.info(f"curate fit: {label} KMeans k={effective} on {matrix.shape[0]} x {matrix.shape[1]} sample")
        kmeans = KMeans(n_clusters=effective, random_state=spec.random_state, n_init=1, output_type="numpy")
        kmeans.fit(matrix)
        return np.ascontiguousarray(kmeans.cluster_centers_, dtype=np.float32)

    return _FitBases(
        locality=fit(sample.fused, spec.requested_k, "locality"),
        subtask=fit(sample.subtask, spec.subtask_k, "subtask") if sample.subtask is not None else None,
        fit_rows=int(sample.fused.shape[0]),
    )


@attrs.frozen
class _FitResult:
    """The fitted bases and what they were fitted on.

    Attributes:
        centroids: ``(effective_k, FUSED_DIM)`` RAW locality centroids. Raw
            because ``CentroidAssigner`` owns the unit-normalization and the
            artifact records what was fitted, not what was scored with.
        subtask_centroids: ``(subtask_k, TEXT_DIM)`` RAW level-2 centroids, or
            ``None`` when no basis was fitted; every row then takes
            ``NO_SUBTASK_CLUSTER`` and level 2 collapses to one group per task.
            The two causes of ``None`` are one configured and one degenerate, and
            ``_report_subtask_basis`` is what tells them apart.
        effective_k: Locality centroids the fit returned, observed rather than
            assumed: k-means clamps ``k`` to the sample it actually got.
        subtask_k: The same, observed, for the level-2 basis; ``0`` when none was
            fitted.
        fit_rows: Usable rows the locality sample held.
        fragment_ids: The sampled prefix, recorded so the artifact says which
            region of the corpus the bases were fitted on.

    """

    centroids: npt.NDArray[np.float32]
    subtask_centroids: npt.NDArray[np.float32] | None
    effective_k: int
    subtask_k: int
    fit_rows: int
    fragment_ids: tuple[int, ...]


def _report_subtask_basis(
    subtask_centroids: npt.NDArray[np.float32] | None,
    *,
    subtask_weight: float,
    requested: int,
) -> int:
    """Report the level-2 basis and return the centroids it holds; ``0`` for none.

    An absent basis has two causes, they are indistinguishable everywhere
    downstream - the artifact records both as ``(0, TEXT_DIM)`` - and only one of
    them is a loss, so they are separated here, at the last point that still holds
    the weights. A zero-weight subtask block is the documented escape and fitted
    nothing on purpose, which is a configured state and stays at INFO. A WEIGHTED
    block that produced no basis means no sampled row carried a usable subtask
    direction, so every row takes ``NO_SUBTASK_CLUSTER`` and level-2 fairness
    silently falls back to the canonical task label alone.

    Args:
        subtask_centroids: The RAW level-2 basis the fit returned, or ``None``.
        subtask_weight: The subtask block's fusion weight, which is what decides
            whether an absent basis was asked for.
        requested: ``subtask_clusters``, for the clamp line.

    Returns:
        Centroids the basis holds.

    """
    if subtask_centroids is not None:
        held = int(subtask_centroids.shape[0])
        if held != requested:
            logger.warning(
                f"curate fit: requested subtask_clusters={requested} but the level-2 basis holds {held} centroid(s)"
            )
        return held
    if subtask_weight <= 0.0:
        logger.info(
            "curate fit: no subtask basis was fitted because the subtask block carries no weight; "
            "every row takes one level-2 group per task"
        )
        return 0
    logger.warning(
        f"curate fit: the subtask block carries weight {subtask_weight} but no level-2 basis was fitted, "
        f"so fairness granularity falls back to the canonical task label alone and every row takes one "
        f"level-2 group per task; no sampled row carried a usable {SUBTASK_VECTOR_COLUMN} direction, so "
        f"check that column on the fit sample's fragment prefix for zero-norm or non-finite vectors"
    )
    return 0


def _fit_centroids(config: CurateConfig, source: _Source, requested_k: int) -> _FitResult:
    """Run both fits on one whole-GPU task and check each basis is the width it is scored at.

    Raises:
        ValueError: If either basis has the wrong width or holds a zero-norm row.
            Both are checked HERE, on the driver, because either one otherwise
            surfaces from the first scan task - after the label gather and the
            whole driver-side merge have already been paid for - and arrives with
            its type erased by Ray's UDF wrapping.

    """
    task = ray.remote(num_gpus=_FIT_TASK_GPUS, runtime_env=ray_data_gpu_runtime_env(_GPU_ENV_NAME))(_fit_kmeans)
    subtask_sample_rows = _subtask_sample_rows(config.subtask_clusters, config.fit_sample_rows)
    prefix = _fit_sample_fragments(source, config.fit_sample_rows, subtask_sample_rows)
    weights = _weight_map(config)
    bases: _FitBases = ray.get(
        task.remote(
            source.read,
            _FitSpec(
                fragment_ids=prefix.fragment_ids,
                weights=weights,
                requested_k=requested_k,
                subtask_k=config.subtask_clusters,
                sample_row_cap=prefix.sample_row_cap,
                subtask_sample_row_cap=prefix.subtask_sample_row_cap,
                random_state=config.kmeans_random_state,
            ),
        )
    )
    effective_k = int(bases.locality.shape[0])
    if effective_k != requested_k:
        logger.warning(f"curate fit: requested k={requested_k} but the basis holds {effective_k} centroid(s)")
    _require_basis_width(bases.locality, vectors.FUSED_DIM, "locality")
    subtask_k = _report_subtask_basis(
        bases.subtask,
        subtask_weight=weights[SUBTASK_WEIGHT_FIELD],
        requested=config.subtask_clusters,
    )
    if bases.subtask is not None:
        _require_basis_width(bases.subtask, TEXT_DIM, "subtask")
    return _FitResult(
        centroids=bases.locality,
        subtask_centroids=bases.subtask,
        effective_k=effective_k,
        subtask_k=subtask_k,
        fit_rows=bases.fit_rows,
        fragment_ids=tuple(int(one) for one in prefix.fragment_ids),
    )


def _require_basis_width(centroids: npt.NDArray[np.float32], width: int, label: str) -> None:
    """Reject a fitted basis the scan would score the wrong vectors against.

    Raises:
        ValueError: If the basis is not ``width`` wide, or holds a zero-norm row.
            The second check is delegated to ``CentroidAssigner.from_raw``, whose
            result is discarded: the point is to raise on the driver rather than
            inside the first scan task.

    """
    if centroids.shape[1] != width:
        msg = f"the fit returned {centroids.shape[1]}-wide {label} centroids but they are scored at {width}"
        raise ValueError(msg)
    vectors.CentroidAssigner.from_raw(centroids)


def _work_items(fragment_ids: Sequence[int]) -> Dataset:
    """Return one Ray Data row per fragment id: the unit of work of the fragment-keyed passes.

    The block count is left to Ray, deliberately. Ray's 200 is a read-parallelism
    FLOOR that its heuristics raise to cover the cluster's CPUs; passing it as
    ``override_num_blocks`` would turn that floor into a hard ceiling and pin both
    row-scale passes to 200 concurrent tasks however large the cluster is.
    """
    logger.info(f"curate: {len(fragment_ids)} fragment work item(s)")
    return ray.data.from_items([{_FRAGMENT_ID_COLUMN: int(fragment_id)} for fragment_id in fragment_ids])


def _eligible_batches(batch: pa.Table, read: _ReadSpec, columns: Sequence[str]) -> Iterable[tuple[int, pa.Table]]:
    """Yield ``(fragment_id, eligible batch)`` for every fragment id in one work batch.

    The shared read shape of both row-scale passes: reopen at the pinned version,
    push the eligibility predicate and the projection into the scanner, and stream
    fixed-size batches so no stage ever holds a whole fragment.

    Raises:
        ValueError: If a work item names a fragment absent at the pinned version.

    """
    dataset = lance.dataset(read.uri, version=read.read_version, storage_options=read.storage_options)
    for fragment_id in batch.column(_FRAGMENT_ID_COLUMN).to_pylist():
        fragment = dataset.get_fragment(int(fragment_id))
        if fragment is None:
            msg = f"fragment {fragment_id} is not present in {read.uri} at version {read.read_version}"
            raise ValueError(msg)
        scanner = fragment.scanner(columns=list(columns), filter=read.predicate, batch_size=_SCAN_ROW_BATCH)
        for record_batch in scanner.to_batches():
            yield int(fragment_id), pa.Table.from_batches([record_batch])


def _text_matrix(batch: pa.Table, name: str) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Decode one text vector column; return its matrix and the per-row presence mask.

    Raises:
        ValueError: If the column is not ``fixed_size_list<float32>[TEXT_DIM]``.
            Checked off the declared type before any decode, because a mismatched
            label embedding would merge on a basis nobody chose while remaining
            finite and unit-norm.

    """
    column = batch.column(name)
    n_rows = len(column)
    if not pa.types.is_fixed_size_list(column.type):
        msg = f"{name} must be a fixed_size_list column, got {column.type}"
        raise ValueError(msg)
    if int(column.type.list_size) != TEXT_DIM:
        msg = f"{name} stores {column.type.list_size}-wide vectors but the label merge runs at {TEXT_DIM}"
        raise ValueError(msg)
    if column.type.value_type != pa.float32():
        msg = f"{name} stores {column.type} but the label merge runs on float32 vectors"
        raise ValueError(msg)
    if column.null_count == n_rows:
        # An all-NULL fixed-size-list column can carry a zero-length child buffer,
        # which no reshape can rebase; nothing is present, so nothing is decoded.
        return np.zeros((n_rows, TEXT_DIM), dtype=np.float32), np.zeros(n_rows, dtype=np.bool_)
    present = np.asarray(column.combine_chunks().is_valid().to_numpy(zero_copy_only=False), dtype=np.bool_)
    return vectors.vector_column_to_matrix(column), present


@attrs.define
class _LabelAccumulator:
    """One canonical label's running count and its lowest-``(clip_id, fragment_id)`` vector.

    Attributes:
        rows: Clips seen carrying this label.
        vector: The representative embedding, zeros until one is seen.
        tiebreak: The ``(clip_id, fragment_id)`` of the clip ``vector`` came
            from; None while none is held. ``clip_id`` alone is unique only
            WITHIN a fragment, so the fragment id is carried with it: it is the
            tie-break that makes the choice of representative independent of
            fragment and batch completion order even when two fragments share a
            ``clip_id`` while holding different vectors.

    """

    rows: int
    vector: npt.NDArray[np.float32]
    tiebreak: tuple[str, int] | None

    def observe(
        self,
        rows: int,
        vector: npt.NDArray[np.float32] | None,
        clip_id: str | None,
        fragment_id: int | None,
    ) -> None:
        """Fold one observation in, taking the vector only if it wins the tie-break."""
        self.rows += rows
        if vector is None or clip_id is None or fragment_id is None:
            return
        candidate = (clip_id, fragment_id)
        if self.tiebreak is None or candidate < self.tiebreak:
            self.vector = vector
            self.tiebreak = candidate


def _accumulate_labels(accumulators: dict[str, _LabelAccumulator], canonical: pa.Table, fragment_id: int) -> None:
    """Fold one fragment batch's per-task-label counts and representatives into ``accumulators``.

    Both reductions run in Arrow and touch Python once per DISTINCT label, never
    once per row: the count is a group-by, and the representative row is located
    by ``index_in`` of each group's minimum ``clip_id``. ``clip_id`` is unique
    within a fragment, so the winner within this single-fragment batch is
    unambiguous; ``fragment_id`` is carried so the driver-side reduction can
    break a tie between two fragments that share a ``clip_id``.

    Args:
        accumulators: Per-canonical-label running state, mutated in place.
        canonical: One fragment's canonicalized rows.
        fragment_id: The fragment ``canonical`` was read from; part of the
            representative tie-break.

    Raises:
        ValueError: If the task text vector column is mis-sized.

    """
    matrix, present = _text_matrix(canonical, TASK_VECTOR_COLUMN)
    flat = pa.table(
        {
            _GATHER_LABEL_COLUMN: canonical.column(CANONICAL_TASK_COLUMN),
            KEY_COLUMN: canonical.column(KEY_COLUMN),
        }
    )
    counted = flat.group_by(_GATHER_LABEL_COLUMN).aggregate([(KEY_COLUMN, "count")])
    for label, rows in zip(
        counted.column(_GATHER_LABEL_COLUMN).to_pylist(),
        counted.column(f"{KEY_COLUMN}_count").to_pylist(),
        strict=True,
    ):
        accumulators.setdefault(label, _LabelAccumulator(0, _ZERO_TEXT_VECTOR, None)).observe(
            int(rows), None, None, None
        )

    if not bool(present.any()):
        return
    # Resolved INSIDE the vector-bearing rows, then mapped back through their
    # original positions. Against the whole batch, index_in returns the FIRST row
    # carrying that clip_id, which for a repeated id can be one whose vector is
    # NULL - and an unvalidated child-buffer slot would then define the label's
    # merge geometry, finite and non-zero, without tripping any check.
    with_vector = flat.filter(pa.array(present))
    original_rows = np.flatnonzero(present)
    winners = with_vector.group_by(_GATHER_LABEL_COLUMN).aggregate([(KEY_COLUMN, "min")])
    positions = pc.index_in(  # type: ignore[attr-defined]
        winners.column(f"{KEY_COLUMN}_min").combine_chunks(),
        value_set=with_vector.column(KEY_COLUMN).combine_chunks(),
    )
    for label, clip_id, position in zip(
        winners.column(_GATHER_LABEL_COLUMN).to_pylist(),
        winners.column(f"{KEY_COLUMN}_min").to_pylist(),
        positions.to_pylist(),
        strict=True,
    ):
        # The copy is load-bearing: a row VIEW would keep the whole decoded batch
        # matrix alive for as long as this label is accumulated.
        accumulators[label].observe(0, matrix[int(original_rows[int(position)])].copy(), clip_id, fragment_id)


def _label_rollup(batch: pa.Table, *, read: _ReadSpec) -> pa.Table:
    """Reduce one fragment's eligible rows to one ``_GATHER_ROW`` per distinct task label.

    The second read pass, and the reason the first stays narrow: the task merge
    needs one text vector per distinct task label, and emitting them here keeps a
    384-wide column out of the retention shuffle. Output is ``O(distinct tasks)``,
    never ``O(rows)``, and that is now a BOUND rather than a hope - the annotation
    schema fixes the task vocabulary, where the subtask vocabulary this pass used
    to also reduce grew with the corpus.

    Raises:
        ValueError: If a task label is NULL, or the task vector column is
            mis-sized.

    """
    accumulators: dict[str, _LabelAccumulator] = {}
    columns = [KEY_COLUMN, TASK_COLUMN, TASK_VECTOR_COLUMN]
    for fragment_id, rows in _eligible_batches(batch, read, columns):
        _accumulate_labels(accumulators, fairness.canonicalize_labels(rows), fragment_id)
    return _gather_table(accumulators)


def _gather_table(accumulators: Mapping[str, _LabelAccumulator]) -> pa.Table:
    """Serialize one task's accumulators as a ``_GATHER_ROW`` table.

    The vector column is built from one contiguous matrix rather than from Python
    lists, so its cost is a memcpy of ``labels x TEXT_DIM`` floats.
    """
    items = sorted(accumulators.items())
    stacked = (
        np.stack([accumulator.vector for _key, accumulator in items])
        if items
        else np.zeros((0, TEXT_DIM), dtype=np.float32)
    )
    return pa.table(
        {
            _GATHER_LABEL_COLUMN: pa.array([label for label, _acc in items], type=pa.string()),
            _GATHER_COUNT_COLUMN: pa.array([acc.rows for _key, acc in items], type=pa.int64()),
            _GATHER_VECTOR_COLUMN: pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(stacked, dtype=np.float32).reshape(-1), type=pa.float32()),
                TEXT_DIM,
            ),
            _GATHER_TIEBREAK_COLUMN: pa.array(
                [acc.tiebreak[0] if acc.tiebreak is not None else None for _key, acc in items], type=pa.string()
            ),
            _GATHER_FRAGMENT_COLUMN: pa.array(
                [acc.tiebreak[1] if acc.tiebreak is not None else None for _key, acc in items], type=pa.int64()
            ),
        },
        schema=_GATHER_ROW,
    )


@attrs.frozen
class _LabelSet:
    """The distinct task labels with their counts and representative vectors.

    Attributes:
        labels: The distinct canonical labels, in ascending order.
        counts: Clip count per label, parallel to ``labels``.
        vectors: ``(len(labels), TEXT_DIM)`` C-contiguous float32 embeddings,
            parallel to ``labels``.

    """

    labels: tuple[str, ...]
    counts: tuple[int, ...]
    vectors: npt.NDArray[np.float32]


def _require_merge_matrix(matrix: npt.NDArray[np.float32], labels: Sequence[str]) -> None:
    """Reject a label matrix ``merge_labels`` would have to convert rather than alias.

    ``merge_labels`` calls ``np.ascontiguousarray(vectors, dtype=float32)``, which
    aliases only an already-float32 C-contiguous array and otherwise COPIES -
    silently, at full matrix size, on top of the two buffers the walk allocates.
    The task vocabulary is small enough that this is no longer a memory cliff, but
    the shape is still asserted rather than hoped for: a silent copy is also a
    silent dtype coercion, and the walk's similarities would then be computed on
    something other than what the gather measured.

    Raises:
        ValueError: On a wrong rank, a row count that does not match ``labels``, a
            width other than ``TEXT_DIM``, a non-float32 dtype, or a
            non-C-contiguous buffer.

    """
    if matrix.ndim != _MERGE_MATRIX_NDIM or matrix.shape != (len(labels), TEXT_DIM):
        msg = f"task label matrix must be ({len(labels)}, {TEXT_DIM}), got shape {matrix.shape}"
        raise ValueError(msg)
    if matrix.dtype != np.float32:
        msg = f"task label matrix must be float32, got {matrix.dtype}; merge_labels would copy it"
        raise ValueError(msg)
    if not matrix.flags.c_contiguous:
        msg = "task label matrix must be C-contiguous; merge_labels would copy it"
        raise ValueError(msg)


def _reduce_gather(tables: Iterable[pa.Table]) -> _LabelSet:
    """Reduce the per-fragment rollups to one ``_LabelSet`` of task labels.

    Raises:
        ValueError: If no label was gathered, or the reduced matrix is not the
            shape ``merge_labels`` can alias.

    """
    accumulators: dict[str, _LabelAccumulator] = {}
    for table in tables:
        labels = table.column(_GATHER_LABEL_COLUMN).to_pylist()
        counts = table.column(_GATHER_COUNT_COLUMN).to_pylist()
        clip_ids = table.column(_GATHER_TIEBREAK_COLUMN).to_pylist()
        fragment_ids = table.column(_GATHER_FRAGMENT_COLUMN).to_pylist()
        matrix = vectors.vector_column_to_matrix(table.column(_GATHER_VECTOR_COLUMN))
        for index, label in enumerate(labels):
            accumulators.setdefault(label, _LabelAccumulator(0, _ZERO_TEXT_VECTOR, None)).observe(
                int(counts[index]), matrix[index].copy(), clip_ids[index], fragment_ids[index]
            )

    items = sorted(accumulators.items())
    if not items:
        msg = "the label gather returned no task label; every eligible row carries one"
        raise ValueError(msg)
    labels = tuple(label for label, _acc in items)
    stacked = np.ascontiguousarray(np.stack([acc.vector for _label, acc in items]), dtype=np.float32)
    _require_merge_matrix(stacked, labels)
    # The merge's OWN threshold, not an exact-zero test: it drops any label whose
    # norm falls at or below it, so a test for zero would under-report exactly
    # the population this warning exists to surface.
    undirected = int((np.linalg.norm(stacked, axis=1) <= fairness.MIN_LABEL_NORM).sum())
    if undirected:
        logger.warning(
            f"curate merge: {undirected} of {len(labels)} task label(s) reached the merge with no vector; "
            f"each stays its own fairness group"
        )
    return _LabelSet(
        labels=labels,
        counts=tuple(int(acc.rows) for _label, acc in items),
        vectors=stacked,
    )


def _gather_labels(source: _Source) -> _LabelSet:
    """Run the task-label pass and reduce it on the driver; ``O(distinct tasks)`` driver state."""
    rollups = _work_items(source.fragment_ids).map_batches(
        # Ray Data types its UDF as taking one positional batch, so a UDF with
        # keyword-only arguments filled from fn_kwargs never matches that shape.
        _label_rollup,  # type: ignore[arg-type]
        batch_size=_FRAGMENTS_PER_TASK,
        batch_format="pyarrow",
        fn_kwargs={"read": source.read},
    )
    # One generator expression, not a list: the reduction folds each rollup and
    # drops it, so the driver holds one rollup at a time rather than all of them.
    # Materializing the list first would peak at blocks x labels x TEXT_DIM to
    # produce a reduction whose own size is O(distinct tasks).
    label_set = _reduce_gather(ray.get(ref) for ref in rollups.to_arrow_refs())
    logger.info(f"curate merge: {len(label_set.labels)} distinct task label(s) gathered")
    return label_set


def _clips_moved(label_set: _LabelSet, merged: Mapping[str, str]) -> int:
    """Return how many clips the merge re-pooled into another label's fairness group.

    Free to compute: the driver already holds the per-label clip counts it handed
    the merge and the map the merge returned, so this is one pass over the task
    vocabulary and needs neither a corpus pass nor a fold from a stage.
    """
    return sum(count for label, count in zip(label_set.labels, label_set.counts, strict=True) if merged[label] != label)


def _merge_maps(config: CurateConfig, label_set: _LabelSet) -> tuple[dict[str, str], MergeStats]:
    """Merge the task labels on the driver; return the map and its measured cost.

    ``theta`` is read from the config unchanged - that model is the single
    validation point for it, so nothing here defaults, clamps or derives one.
    """
    started = time.perf_counter()
    merged = fairness.merge_labels(
        label_set.labels,
        label_set.counts,
        label_set.vectors,
        theta=config.merge_theta_task,
    )
    elapsed = time.perf_counter() - started
    representatives = len(set(merged.values()))
    # The only instrument for the merge's realized cost: R has no config bound and
    # the walk has no early exit, so L * R is knowable only from here.
    logger.info(
        f"curate merge: task L={len(label_set.labels)} -> R={representatives} "
        f"at theta={config.merge_theta_task} in {elapsed:.2f}s"
    )
    moved = _clips_moved(label_set, merged)
    clips = sum(label_set.counts)
    share = moved / clips if clips else 0.0
    logger.info(f"curate merge: {moved} of {clips} clip(s) ({share:.1%}) changed fairness group")
    # Strictly above the share warns. The corpus-side counterpart to
    # fairness.merge_labels' own line, which measures the label vocabulary: a
    # merge can leave R comfortably high and still re-pool most of the clips, and
    # that is the case worth stopping to read.
    if moved * _CLIPS_MOVED_WARN_FACTOR > clips:
        logger.warning(
            f"curate merge re-pooled {moved} of {clips} clip(s) ({share:.1%}) into another task's fairness "
            f"group at theta={config.merge_theta_task}: more than one clip in "
            f"{_CLIPS_MOVED_WARN_FACTOR} changed group"
        )
    return merged, MergeStats(
        labels_in=len(label_set.labels),
        labels_out=representatives,
        clips_moved=moved,
        seconds=elapsed,
    )


@attrs.frozen(eq=False)
class _ScanBases:
    """The two bases one scan task scores against.

    ``eq=False`` because it holds assigners over ndarrays. Bundled rather than
    passed as two positional arguments of the same type, which would type-check
    when swapped and then partition each geometry by the other's centroids.

    Attributes:
        locality: Scores the fused working vector; produces ``__dedup_key`` and
            ``distance_to_centroid``.
        subtask: Scores the stored subtask text vector; produces the level-2
            fairness group. ``None`` when the subtask block carries no weight, in
            which case every row takes ``NO_SUBTASK_CLUSTER``.

    """

    locality: vectors.CentroidAssigner
    subtask: vectors.CentroidAssigner | None

    @classmethod
    def from_raw(
        cls,
        locality: npt.NDArray[np.float32],
        subtask: npt.NDArray[np.float32] | None,
    ) -> "_ScanBases":
        """Unit-normalize both raw bases once, at the top of a scan task."""
        return cls(
            locality=vectors.CentroidAssigner.from_raw(locality),
            subtask=vectors.CentroidAssigner.from_raw(subtask) if subtask is not None else None,
        )


def _scan_verdicts(
    batch: pa.Table,
    *,
    read: _ReadSpec,
    weights: Mapping[str, float],
    centroids: "ray.ObjectRef[npt.NDArray[np.float32]]",
    subtask_centroids: "ray.ObjectRef[npt.NDArray[np.float32]] | None",
) -> Iterator[pa.Table]:
    """Scan one fragment into ``_SCAN_ROW``: task label, subtask cell, working vector, cluster, distance.

    The single row-scale read of the run. It is the producer of both fairness
    group keys and of every column the retention pass needs, so no later stage has
    to reopen the table.

    Raises:
        ValueError: If a task label is NULL, a weighted block is mis-sized, or a
            fitted basis is degenerate.

    """
    # fn_kwargs are captured in the map closure, NOT passed as top-level task
    # arguments, so Ray does not dereference these - the ObjectRefs arrive as
    # refs. Resolved once per work batch, which at _FRAGMENTS_PER_TASK=1 is once
    # per fragment; the unit-normalization they feed is k x d, so cheap.
    bases = _ScanBases.from_raw(
        ray.get(centroids), ray.get(subtask_centroids) if subtask_centroids is not None else None
    )
    # Yielded per scanned batch, never accumulated. A generator UDF is Ray Data's
    # own remedy for a large output: back-pressure then applies per block, where
    # returning one table per fragment would hold a whole fragment's working
    # vectors and their concatenated copy at once.
    yield from _scan_blocks(batch, read, weights, bases)


def _scan_fragments(
    batch: pa.Table,
    read: _ReadSpec,
    weights: Mapping[str, float],
    bases: _ScanBases,
) -> pa.Table:
    """Scan one work batch into a single table; the streamed blocks, joined.

    Raises:
        ValueError: If a task label is NULL or a weighted block is mis-sized.

    """
    blocks = list(_scan_blocks(batch, read, weights, bases))
    return pa.concat_tables(blocks) if blocks else _SCAN_ROW.empty_table()


def _scan_blocks(
    batch: pa.Table,
    read: _ReadSpec,
    weights: Mapping[str, float],
    bases: _ScanBases,
) -> Iterator[pa.Table]:
    """Yield one ``_SCAN_ROW`` block per scanned batch of every fragment in a work batch.

    ``subtask_name`` is deliberately NOT projected. The level-2 key is a partition
    of the subtask EMBEDDING, so the prose that produced it is never read at row
    scale again - which is what removes it as a corpus-scale string vocabulary
    from this pass.

    Raises:
        ValueError: If a task label is NULL or a weighted block is mis-sized.

    """
    columns = sorted(
        {KEY_COLUMN, TASK_COLUMN, *weighted_vectors(weights)}
        | ({SUBTASK_VECTOR_COLUMN} if bases.subtask is not None else set())
    )
    non_finite = zero_norm = 0
    for fragment_id, rows in _eligible_batches(batch, read, columns):
        canonical = fairness.canonicalize_labels(rows)
        fused = vectors.working_vectors(canonical, weights)
        # The two causes partition the dropped rows, and zero-norm is a subset of
        # the finite ones, so the non-finite rows are the dropped rows that are
        # not zero-norm. Derived here rather than returned as a count because the
        # mask is what routes each row to its own bypass sentinel.
        non_finite += int((~fused.keep & ~fused.zero_norm).sum())
        zero_norm += int(fused.zero_norm.sum())
        yield _verdict_block(canonical, fragment_id, fused, bases)
    if non_finite or zero_norm:
        logger.warning(
            f"curate scan: {non_finite} non-finite and {zero_norm} zero-norm row(s) reached the write as "
            f"{CurateReason.INVALID_EMBEDDING}"
        )


def _subtask_cells(canonical: pa.Table, bases: _ScanBases) -> npt.NDArray[np.int32]:
    """Assign one scanned batch's rows to their level-2 subtask cell.

    Every row is assigned, including the ones the scan is about to reason
    ``invalid_embedding``: the fairness stage groups on this key as one half of a
    non-nullable ``fairness.Level2Key``, so a row without a usable subtask
    direction takes ``NO_SUBTASK_CLUSTER`` instead of leaving the key set. See
    ``_SCAN_ROW`` for why the sentinel rather than a NULL, which Ray Data would in
    fact group.

    Raises:
        ValueError: If the subtask vector column is stored at a width other than
            ``TEXT_DIM``, which would partition on a basis nobody chose.

    """
    if bases.subtask is None:
        return np.full(canonical.num_rows, NO_SUBTASK_CLUSTER, dtype=np.int32)
    matrix, present = _text_matrix(canonical, SUBTASK_VECTOR_COLUMN)
    return vectors.assign_clusters(matrix, present, bases.subtask, NO_SUBTASK_CLUSTER)


def _verdict_block(
    canonical: pa.Table,
    fragment_id: int,
    fused: vectors.WorkingVectors,
    bases: _ScanBases,
) -> pa.Table:
    """Assemble one scanned batch's ``_SCAN_ROW`` rows, in input order.

    The kept rows are scored and routed to their cluster; the rest are reasoned
    ``invalid_embedding`` here and routed to a negative bypass key, which is the
    only way a vector that is not on the unit sphere reaches the write without
    entering a similarity GEMM.

    Which of the two bypass keys a dropped row takes is decided by its cause. The
    fate is identical either way - every consumer tests negativity - so the split
    buys exactly one thing, and only on the scored path: the retention stage groups
    on this key corpus-wide, so its per-group line reports each cause's exact total
    in one aggregate. With that stage skipped no such line exists and
    ``_scan_blocks``' own per-task warning is what keeps the causes apart.
    """
    n_rows = canonical.num_rows
    keep = fused.keep
    cluster_id, distance = bases.locality.score(fused.working)
    dedup_key = np.where(fused.zero_norm, NO_DEDUP_GROUP_ZERO_NORM, NO_DEDUP_GROUP).astype(np.int32)
    dedup_key[keep] = cluster_id
    distances = np.zeros(n_rows, dtype=np.float32)
    distances[keep] = distance
    # A zero row, not a NULL: dedup rejects a NULL vector inside a scored group,
    # and these rows only ever reach the bypass, which reads no vector at all.
    stored = np.zeros((n_rows, vectors.FUSED_DIM), dtype=np.float32)
    stored[keep] = fused.working
    usable = pa.array(keep)
    reason = pc.if_else(  # type: ignore[attr-defined]
        usable,
        pa.scalar(None, type=pa.string()),
        pa.scalar(str(CurateReason.INVALID_EMBEDDING), type=pa.string()),
    )
    return pa.table(
        {
            KEY_COLUMN: canonical.column(KEY_COLUMN),
            FRAGMENT_COLUMN: pa.array(np.full(n_rows, fragment_id, dtype=np.int32)),
            DEDUP_KEY_COLUMN: pa.array(dedup_key),
            CURATE_SELECTION_REASON: reason,
            CANONICAL_TASK_COLUMN: canonical.column(CANONICAL_TASK_COLUMN),
            SUBTASK_CLUSTER_COLUMN: pa.array(_subtask_cells(canonical, bases)),
            DISTANCE_COLUMN: pa.array(distances, mask=~keep),
            WORKING_VECTOR_COLUMN: pa.FixedSizeListArray.from_arrays(
                pa.array(stored.reshape(-1), type=pa.float32()), vectors.FUSED_DIM
            ),
        },
        schema=_SCAN_ROW,
    )


def _scan(config: CurateConfig, source: _Source, fit: _FitResult) -> Dataset:
    """Return the run's scanned rows: one ``_SCAN_ROW`` per eligible row."""
    return _work_items(source.fragment_ids).map_batches(
        _scan_verdicts,  # type: ignore[arg-type]
        batch_size=_FRAGMENTS_PER_TASK,
        batch_format="pyarrow",
        fn_kwargs={
            "read": source.read,
            "weights": _weight_map(config),
            "centroids": ray.put(fit.centroids),
            "subtask_centroids": ray.put(fit.subtask_centroids) if fit.subtask_centroids is not None else None,
        },
    )


def unscored_rows(batch: pa.Table) -> pa.Table:
    """Shape one scanned batch the way the retention stage would, without scoring it.

    A plain module-level function because Ray reads ``fn.__name__`` off the
    callable. It reproduces the two things that stage does BESIDES marking
    duplicates - shedding the working vector, which at 250M rows is ~865 GB of
    column nothing downstream reads, and emitting the score column, NULL here
    because nothing was compared - so the fairness shuffle and the report pass see
    one schema whether or not de-duplication ran.
    """
    return dedup.with_scores(batch.drop_columns([WORKING_VECTOR_COLUMN]), None)


def scored_scan_rows(batch: pa.Table) -> pa.Table:
    """Keep rows whose dedup key routes them through the retention GEMM."""
    return batch.filter(pc.greater_equal(batch.column(DEDUP_KEY_COLUMN), 0))  # type: ignore[attr-defined]


def bypass_scan_rows(batch: pa.Table) -> pa.Table:
    """Keep rows whose dedup key bypasses the retention GEMM."""
    return batch.filter(pc.less(batch.column(DEDUP_KEY_COLUMN), 0))  # type: ignore[attr-defined]


def _retain(config: CurateConfig, scanned: Dataset) -> Dataset:
    """Mark near-duplicates cluster by cluster on the GPU, and shed the working vector.

    At ``dedup_eps=None`` the stage does not run and the rows are projected
    instead. Both paths are owned here so the post-retain schema has one author;
    the fit still runs either way, because ``curate_cluster_id`` and the
    within-group ordering come from the scan and are needed by both.

    Rows with a negative ``__dedup_key`` bypass the GPU group stage entirely:
    they are filtered out, projected through ``unscored_rows``, and unioned back
    with the scored groups so invalid rows never materialize two corpus-wide bypass
    partitions.
    """
    if config.dedup_eps is None:
        logger.info(
            "curate dedup: skipped (dedup_eps is unset), so no row is marked duplicate and the rows a "
            "retention pass would have flagged reach the fairness cut as selection candidates"
        )
        return scanned.map_batches(
            unscored_rows,  # type: ignore[arg-type]
            batch_size=None,
            batch_format="pyarrow",
        )
    launch = dedup.dedup_launch_args(gpu_env_name=_GPU_ENV_NAME, concurrency=config.dedup_concurrency)
    scored = (
        scanned.map_batches(
            scored_scan_rows,  # type: ignore[arg-type]
            batch_format="pyarrow",
        )
        .groupby(DEDUP_KEY_COLUMN)
        .map_groups(
            dedup.dedup_group,  # type: ignore[arg-type]
            fn_kwargs={"eps": config.dedup_eps},
            **launch,
        )
    )
    bypassed = scanned.map_batches(
        bypass_scan_rows,  # type: ignore[arg-type]
        batch_format="pyarrow",
    ).map_batches(
        unscored_rows,  # type: ignore[arg-type]
        batch_format="pyarrow",
    )
    return scored.union(bypassed)


def _merge_groups(deduped: Dataset, task_merge: Mapping[str, str]) -> Dataset:
    """Rewrite the task label column to its representatives, as its own row-wise stage.

    A stage of its own and placed here deliberately: the count that funds the
    quotas and the shuffle that cuts against it must both address the MERGED
    group, so the rewrite has to complete before either. The level-2 key needs no
    equivalent stage - the scan already wrote its final value.
    """
    return deduped.map_batches(
        fairness.apply_label_merge,  # type: ignore[arg-type]
        batch_size=None,
        batch_format="pyarrow",
        fn_args=(dict(task_merge),),
    )


def _unified_field(blocks: Sequence[pa.Table], name: str) -> pa.Field:
    """Return the single field type ``name`` carries across every populated aggregate block.

    Raises:
        ValueError: If two blocks name different concrete types for the field.

    """
    observed: list[pa.DataType] = []
    for block in blocks:
        field_type = block.schema.field(name).type
        if not pa.types.is_null(field_type) and field_type not in observed:
            observed.append(field_type)
    if len(observed) > 1:
        msg = f"aggregate blocks disagree on the type of {name!r}: {observed}; this is schema drift, not a lost type"
        raise ValueError(msg)
    # Nullable throughout: a group key that is NULL for a whole partition is what
    # this function exists to carry, and widening the count's nullability costs
    # the caller nothing - it reads values, never the flag.
    return pa.field(name, observed[0] if observed else pa.null(), nullable=True)


def _concat_aggregate_blocks(blocks: Sequence[pa.Table]) -> pa.Table:
    """Concatenate one aggregate's output partitions into the table the driver reads.

    Ray describes each output PARTITION independently, and a partition that saw
    only part of the data describes only that part, so two blocks of one
    reduction can disagree about the table they belong to in two ways:

    - An EMPTY partition carries no schema at all - zero columns, no field names
      - not a zero-row copy of its siblings'. Every partition past the number of
      distinct groups is empty, and the default partition count far exceeds the
      handful of groups either caller reduces to, so most blocks are these.
    - A partition whose rows all hold NULL in a nullable group key types that key
      ``null``, while a sibling that observed a value types it ``string``. Curate
      reaches this every run: ``curate_selection_reason`` is NULL for exactly the
      survivors, which is nearly every row, so a partition holding no reasoned
      row is the common case rather than the corner.

    Empty partitions are therefore dropped rather than unified - they carry no
    row to keep and no type to learn from - and the surviving blocks agree on a
    schema by promoting ``null`` to a type a sibling actually observed. A
    disagreement between two CONCRETE types, or a populated block whose field set
    differs, means the blocks really do describe different tables and is left to
    fail. That last part is what rules out
    ``pa.concat_tables(..., promote_options="permissive")``: it would absorb both
    of the shapes above and a genuine drift alike, turning a lost column into an
    all-NULL one that reads downstream as data rather than as a defect.

    Args:
        blocks: One Arrow table per aggregate output partition, in any order.

    Returns:
        The populated blocks concatenated under one schema, field order taken
        from the first populated block.

    Raises:
        ValueError: If no block holds a row, if two populated blocks carry
            different field sets, or if two disagree on a field's concrete type.

    """
    populated = [block for block in blocks if block.num_rows > 0]
    if not populated:
        msg = (
            f"the aggregate returned no rows across {len(blocks)} output partition(s); "
            "a reduction over the eligible rows yields at least one group"
        )
        raise ValueError(msg)
    names = [field.name for field in populated[0].schema]
    for block in populated:
        if set(block.schema.names) != set(names):
            msg = (
                f"populated aggregate blocks carry different field sets: {sorted(names)} "
                f"vs {sorted(block.schema.names)}; this is schema drift, not an empty partition"
            )
            raise ValueError(msg)
    unified = pa.schema([_unified_field(populated, name) for name in names])
    return pa.concat_tables([block.select(names).cast(unified) for block in populated])


def _group_counts(merged: Dataset) -> pa.Table:
    """Reduce the merged rows to the ``O(G)`` count table the quota is built from.

    The only driver intermediate that scales with anything, and now the only one
    that is BOUNDED: three group keys and a count, one row per observed
    ``(task, cluster, reason)``, so ``G`` is at most distinct merged tasks times
    ``subtask_clusters + 1`` times the reason cardinality however large the
    corpus is.
    """
    return _concat_aggregate_blocks(
        ray.get(
            merged.groupby([CANONICAL_TASK_COLUMN, SUBTASK_CLUSTER_COLUMN, CURATE_SELECTION_REASON])
            .count()
            .to_arrow_refs()
        )
    )


def _metric_bins(batch: pa.Table, metric: _Metric) -> pa.Table:
    """Bin one metric's finite values into non-empty ``(metric, bin, rows)`` triples.

    Rows carrying no value are skipped rather than binned, and each metric has its
    own reason for holding none: a row the scan judged ``invalid_embedding`` was
    scored against no centroid, and a row that bypassed the retention GEMM - or
    every row, when the stage was skipped - was compared against nothing. Arrow
    renders such a NULL as NaN on the way to NumPy, which is why finiteness, not
    nullness, is the filter.
    """
    column = batch.column(metric.column).combine_chunks()
    values = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
    scored = values[np.isfinite(values)]
    if scored.size == 0:
        return _METRIC_HISTOGRAM.empty_table()
    # Clipped rather than trusted, and BOTH bounds are load-bearing for the same
    # reason: this reduction only reports, so it must not be able to abort a run.
    # np.bincount raises on a negative index, and an index at _HISTOGRAM_BINS would
    # widen the count vector past the dense buffer the driver folds into. A value
    # outside a metric's range means the scan scored against a basis the metric
    # does not describe - worth failing on, but at the fit's own gates, not from a
    # log line downstream of a committed GPU pass.
    scaled = scored * (_HISTOGRAM_BINS / metric.upper)
    indices = np.clip(scaled.astype(np.int64), 0, _HISTOGRAM_BINS - 1)
    counts = np.bincount(indices, minlength=_HISTOGRAM_BINS)
    occupied = np.flatnonzero(counts)
    return pa.table(
        {
            _METRIC_COLUMN: pa.array([metric.column] * occupied.size, type=pa.string()),
            _HISTOGRAM_BIN_COLUMN: pa.array(occupied.astype(np.int32)),
            _HISTOGRAM_ROWS_COLUMN: pa.array(counts[occupied].astype(np.int64)),
        },
        schema=_METRIC_HISTOGRAM,
    )


def metric_histogram(batch: pa.Table) -> pa.Table:
    """Reduce one batch's reported metrics to non-empty ``_METRIC_HISTOGRAM`` triples.

    A plain module-level function because Ray reads ``fn.__name__`` for the
    operator label, which is also why the per-metric binning is a private helper
    rather than a ``functools.partial`` over this one.

    Args:
        batch: Any rows carrying every ``_REPORTED_METRICS`` column.

    Returns:
        One row per non-empty bin per metric. An all-NULL batch returns no rows.

    """
    return pa.concat_tables([_metric_bins(batch, metric) for metric in _REPORTED_METRICS])


def _fold_metric_histograms(tables: Iterable[pa.Table]) -> dict[str, npt.NDArray[np.int64]]:
    """Sum the per-batch triples into one dense count vector per reported metric."""
    folded: dict[str, npt.NDArray[np.int64]] = {
        metric.column: np.zeros(_HISTOGRAM_BINS, dtype=np.int64) for metric in _REPORTED_METRICS
    }
    for table in tables:
        if table.num_rows == 0:
            continue
        names = np.asarray(table.column(_METRIC_COLUMN).to_pylist(), dtype=np.str_)
        bins = np.asarray(table.column(_HISTOGRAM_BIN_COLUMN).to_numpy(zero_copy_only=False), dtype=np.int64)
        rows = np.asarray(table.column(_HISTOGRAM_ROWS_COLUMN).to_numpy(zero_copy_only=False), dtype=np.int64)
        for metric, counts in folded.items():
            selected = names == metric
            # Unbuffered add. One UDF call emits each (metric, bin) pair at most
            # once, but Ray Data concatenates blocks, so ONE table reaching this
            # fold can carry a pair twice - and `counts[bins] += rows` keeps only
            # the last write per repeated index, silently under-counting the mode
            # of the distribution.
            np.add.at(counts, bins[selected], rows[selected])
    return folded


def _histogram_quantiles(
    counts: npt.NDArray[np.int64],
    quantiles: Sequence[float],
    bin_width: float,
) -> tuple[float, ...]:
    """Read ``quantiles`` off a folded histogram, as bin midpoints.

    Each quantile is the midpoint of the first bin whose cumulative count reaches
    rank ``ceil(q * total)``, so the reported value is within half a bin width of
    the true order statistic. An empty histogram yields no quantiles.
    """
    total = int(counts.sum())
    if total == 0:
        return ()
    cumulative = np.cumsum(counts)
    return tuple(
        (int(np.searchsorted(cumulative, math.ceil(quantile * total), side="left")) + 0.5) * bin_width
        for quantile in quantiles
    )


def _histogram_max(counts: npt.NDArray[np.int64], bin_width: float) -> float | None:
    """Return the highest occupied bin's midpoint, or None for an empty histogram.

    Reported alongside the percentiles because the tail is where the threshold
    sits: a p99.9 comfortably below ``1 - eps`` with a maximum above it is a
    corpus whose duplicates are a rounding error, and the two numbers together say
    so where either alone does not.
    """
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        return None
    return (int(occupied[-1]) + 0.5) * bin_width


def _ladder_rungs(eps: float | None) -> tuple[float, ...]:
    """Return the thresholds the score's tail is reported against, ascending.

    Args:
        eps: The run's configured threshold, or ``None`` when de-duplication was
            skipped and every rung is therefore a counterfactual.

    """
    if eps is None:
        return _DEDUP_EPS_LADDER
    return tuple(sorted({*_DEDUP_EPS_LADDER, eps}))


def _eps_ladder(counts: npt.NDArray[np.int64], bin_width: float, eps: float | None) -> tuple[tuple[float, int], ...]:
    """Return, per candidate eps, how many rows scored above its duplicate threshold.

    The counterfactual an operator cannot otherwise get: the score is not
    persisted, so without this a different ``dedup_eps`` can only be evaluated by
    re-running the whole leg. It costs nothing here - each entry is the folded
    histogram's own upper tail, read at one more offset.

    Rows are counted from the first bin whose LOWER edge reaches ``1 - eps``, so
    the bin straddling the threshold is excluded and each figure is a lower bound
    accurate to one bin width. That is also why the rung at the run's own eps does
    not have to equal the exact ``duplicate`` verdict count.

    Args:
        counts: The folded score histogram.
        bin_width: Width of one histogram bin.
        eps: The run's configured threshold, spliced in as a rung.

    """
    # rung <= 1.0 is enforced by CurateConfig.dedup_eps (le=1.0) and by
    # _DEDUP_EPS_LADDER, so the slice start is never negative. A negative start
    # would silently sum the histogram's last N bins instead of its whole upper
    # range, reporting a tail smaller than the one it names.
    return tuple((rung, int(counts[math.ceil((1.0 - rung) / bin_width) :].sum())) for rung in _ladder_rungs(eps))


def _percentile_labels(quantiles: Sequence[float], values: Sequence[float]) -> str:
    """Render quantile / value pairs as ``p50=0.1234``, keeping a fractional p99.9."""
    return " ".join(f"p{quantile * 100:g}={value:.4f}" for quantile, value in zip(quantiles, values, strict=True))


def _resolution(metric: _Metric) -> str:
    """Render a metric's reporting resolution, so a percentile is read with its error bar."""
    return f"+/- {metric.bin_width / 2:.5f}, {_HISTOGRAM_BINS} bins over [0, {metric.upper:g}]"


def _log_cluster_radius(counts: npt.NDArray[np.int64]) -> None:
    """Log the corpus-wide cluster radius: the ``distance_to_centroid`` percentiles.

    The radius is what makes ``dedup_eps`` interpretable. ``k`` rises with the
    corpus (it is ``ceil(eligible_rows / target_mean_cluster_rows)``) while
    ``dedup_eps`` is a fixed constant, so the same eps is a different fraction of
    a cell's extent at every corpus size.
    """
    quantiles = _histogram_quantiles(counts, _RADIUS_QUANTILES, _RADIUS_METRIC.bin_width)
    if not quantiles:
        logger.warning("curate cluster radius: no scored row carried a distance; every row was invalid_embedding")
        return
    logger.info(
        f"curate cluster radius: {DISTANCE_COLUMN} {_percentile_labels(_RADIUS_QUANTILES, quantiles)} over "
        f"{int(counts.sum())} scored row(s) ({_resolution(_RADIUS_METRIC)})"
    )


def _log_dedup_score(counts: npt.NDArray[np.int64], eps: float | None) -> None:
    """Log the retention score's upper tail and what other thresholds would have flagged.

    The other half of the radius line: the radius says how wide a cell is, this
    says where in that cell the corpus actually sits relative to the threshold
    applied to it. The run's own eps is marked on the ladder so the operating
    point is readable beside its counterfactuals rather than inferred from them.

    The absence branch is INFO rather than a warning, unlike the radius, because
    it is a configured state and not a degenerate one - ``dedup_eps=None`` skips
    the stage on purpose, so there is genuinely nothing to report.

    Args:
        counts: The folded score histogram.
        eps: The run's configured ``dedup_eps``.

    """
    quantiles = _histogram_quantiles(counts, _SCORE_QUANTILES, _SCORE_METRIC.bin_width)
    highest = _histogram_max(counts, _SCORE_METRIC.bin_width)
    if not quantiles or highest is None:
        logger.info(
            "curate dedup score: no row carried a retention score; de-duplication was skipped or every "
            "row bypassed the similarity pass"
        )
        return
    ladder = " ".join(
        f"eps={rung:g}{'*' if rung == eps else ''}->{rows}"
        for rung, rows in _eps_ladder(counts, _SCORE_METRIC.bin_width, eps)
    )
    legend = " (* the configured dedup_eps)" if eps is not None else ""
    logger.info(
        f"curate dedup score: {DEDUP_SCORE_COLUMN} {_percentile_labels(_SCORE_QUANTILES, quantiles)} "
        f"max={highest:.4f} over {int(counts.sum())} scored row(s) ({_resolution(_SCORE_METRIC)}); "
        f"rows above 1 - eps at {ladder}{legend}"
    )


def _report_metrics(merged: Dataset, dedup_eps: float | None) -> None:
    """Log the two corpus-wide distributions the run's thresholds are read against.

    Both metrics are computed in flight and deliberately never persisted, so these
    lines are the only way the meaning of "duplicate" is observable at all.

    ONE narrow pass over both columns rather than one pass each: the projection
    reads two float32 columns off the already-materialized blocks (~2 GB at 250M
    rows, distributed), and the driver holds ``_HISTOGRAM_BINS`` integers per
    metric.

    ``batch_size=None`` pins whole-block batching, as every other ``map_batches``
    here does: the reduction emits one triple per occupied bin per call, so a
    row-count batch size would multiply the emitted rows by the number of batches
    a block was cut into.

    Args:
        merged: Materialized post-retention rows carrying both metric columns.
        dedup_eps: The run's configured threshold, marked on the reported ladder.

    """
    histograms = merged.select_columns([metric.column for metric in _REPORTED_METRICS]).map_batches(
        metric_histogram,  # type: ignore[arg-type]
        batch_size=None,
        batch_format="pyarrow",
    )
    folded = _fold_metric_histograms(ray.get(ref) for ref in histograms.to_arrow_refs())
    _log_cluster_radius(folded[_RADIUS_METRIC.column])
    _log_dedup_score(folded[_SCORE_METRIC.column], dedup_eps)


def _build_quota(config: CurateConfig, group_counts: pa.Table) -> tuple[fairness.FairnessQuota, int]:
    """Resolve the target against the survivor population and build the quota over it.

    Returns:
        ``(quota, target)``. The survivor total is the target's denominator:
        duplicates and invalid rows already carry a verdict, so they neither
        consume a quota nor enlarge the group that funds one.

    Raises:
        ValueError: If no eligible row reached the cut unreasoned, so there is
            nothing to select and a committed run would mean "nothing was
            selected". The message names no single cause, because it must be true
            whether or not de-duplication ran: with the stage skipped every
            reasoned row is ``invalid_embedding``, and naming a pass that never
            executed would send an operator to the wrong knob.

    """
    survivors_by_group = fairness.survivor_group_counts(group_counts)
    survivors = sum(survivors_by_group.values())
    if survivors == 0:
        msg = "no row survived to be selected; every eligible row already carries a reason"
        raise ValueError(msg)
    target = config.target.resolve(survivors)
    logger.info(
        f"curate fairness: {survivors} survivor(s) across {len(survivors_by_group)} merged group(s), target={target}"
    )
    return (
        fairness.FairnessQuota.build(
            level2_keys=list(survivors_by_group.keys()),
            level2_counts=list(survivors_by_group.values()),
            target=target,
            residual_seed=config.fairness_residual_seed,
        ),
        target,
    )


def _report_unfunded(quota: fairness.FairnessQuota, quotas: Mapping[fairness.Level2Key, int]) -> int:
    """Report how many fairness groups received a quota of zero, and return the count.

    The count comes from ``FairnessQuota.unfunded_groups``, which owns the
    predicate and the degeneracy threshold; this adds the share, which is what
    makes the count readable, and reports it unconditionally so the
    sub-threshold regime is observable too. Nothing about the group set is
    persisted, so this line and ``CurateResult.unfunded_groups`` are the only
    places it can be seen.
    """
    unfunded = quota.unfunded_groups(quotas)
    share = unfunded / len(quotas) if quotas else 0.0
    logger.info(f"curate fairness: {unfunded} of {len(quotas)} group(s) ({share:.1%}) received no quota")
    return unfunded


def _select(config: CurateConfig, merged: Dataset, quotas: Mapping[fairness.Level2Key, int]) -> Dataset:
    """Cut each merged group to its quota, leaving every row with exactly one reason."""
    return merged.groupby([CANONICAL_TASK_COLUMN, SUBTASK_CLUSTER_COLUMN]).map_groups(
        fairness.select_within_quota,  # type: ignore[arg-type]
        fn_args=(dict(quotas), config.within_group_order),
        batch_format="pyarrow",
    )


def _reason_table(verdicts: Dataset) -> pa.Table:
    """Reduce the verdict rows to one count per reason; at most six rows."""
    return _concat_aggregate_blocks(ray.get(verdicts.groupby(CURATE_SELECTION_REASON).count().to_arrow_refs()))


def _reason_counts(reason_table: pa.Table, eligible_rows: int) -> dict[str, int]:
    """Return rows per reason, refusing a verdict set the write must not publish.

    Raises:
        ValueError: If any row reaches the write unreasoned, or the reasoned rows
            do not account for every eligible row. Either means a stage dropped or
            duplicated rows, and the commit is all-or-nothing, so it is refused
            here rather than discovered from the committed table.

    """
    observed = list(
        zip(
            reason_table.column(CURATE_SELECTION_REASON).to_pylist(),
            reason_table.column(RAY_COUNT_COLUMN).to_pylist(),
            strict=True,
        )
    )
    counts = {str(reason): int(count) for reason, count in observed if reason is not None}
    unreasoned = sum(int(count) for reason, count in observed if reason is None)
    if unreasoned:
        msg = f"{unreasoned} row(s) reached the write with no reason; every claimed row must carry one"
        raise ValueError(msg)
    total = sum(counts.values())
    if total != eligible_rows:
        msg = f"the verdict set holds {total} row(s) but {eligible_rows} were eligible; a stage changed cardinality"
        raise ValueError(msg)
    logger.info(f"curate verdicts: {counts}")
    return counts


def _write_back(source: _Source, verdicts: Dataset) -> _CollectedWrite:
    """Write both columns for every row of every fragment; reduce the payloads for the commit.

    Two passes, because the write is total and a shuffle keyed on ``__frag`` reaches only the
    fragments that carry verdicts: the first writes those, the second blanks the rest. Splitting
    them keeps the verdict pass a pure shuffle over rows already in flight, and makes the second
    pass cost nothing when every fragment holds an eligible row. ``_total_payloads`` then holds
    the two passes to their sum, so totality survives a change to this wiring, and
    ``_total_rows`` holds the rows they claimed to the eligible count, which is the axis the
    fragment count cannot see.

    Raises:
        CurateWriteError: If the passes do not cover every fragment, or the rows they
            claimed do not account for every eligible row.

    """
    results = (
        verdicts.groupby(FRAGMENT_COLUMN)
        .map_groups(
            update_one_fragment,  # type: ignore[arg-type]
            fn_kwargs={
                "uri": source.read.uri,
                "read_version": source.read.read_version,
                "storage_options": source.read.storage_options,
            },
            batch_format="pyarrow",
        )
        .take_all()
    )
    claimed = [str(row[_RESULT_COLUMN]) for row in results]
    uncovered = _uncovered_fragments(source.fragment_ids, claimed)
    collected = _collect(_total_payloads(uncovered, claimed, _blank_uncovered(source, uncovered)))
    return _total_rows(collected, source.eligible_rows)


def _uncovered_fragments(fragment_ids: Sequence[int], claimed: Sequence[str]) -> list[int]:
    """Return the fragment ids no verdict payload named, in manifest order.

    The covered set is read back out of the payloads rather than aggregated from the verdicts,
    because the payloads are already on the driver and number one per fragment, whereas asking
    the verdict dataset for its distinct fragment ids is another distributed pass over every row.

    Args:
        fragment_ids: Every fragment of the pinned table, in manifest order.
        claimed: One payload per fragment the verdict pass wrote.

    Returns:
        The fragments the blanking pass must visit for the write to be total.

    """
    covered = {int(json.loads(payload)["fragment_id"]) for payload in claimed}
    return [fragment_id for fragment_id in fragment_ids if fragment_id not in covered]


def _total_payloads(uncovered: Sequence[int], claimed: Sequence[str], blanked: Sequence[str]) -> list[str]:
    """Return one payload per fragment of the table, refusing a set that misses any.

    This refusal is what makes the write total rather than merely intended: without
    it, a blanking pass that was skipped, or wired away, would commit only the
    fragments carrying verdicts and leave the rest holding a previous run's.

    It counts payloads instead of re-decoding them because the blanking pass is one
    work item per uncovered fragment and ``_collect`` refuses duplicates, so one
    payload per uncovered fragment says the same thing as coverage does.

    Raises:
        CurateWriteError: If the blanking pass did not answer for every fragment
            the verdict pass left uncovered.

    """
    if len(blanked) != len(uncovered):
        msg = (
            f"the blanking pass returned {len(blanked)} payload(s) for {len(uncovered)} fragment(s) holding "
            f"no eligible row; refusing to commit a write that would leave a previous run's verdicts in place"
        )
        raise CurateWriteError(msg)
    return [*claimed, *blanked]


def _total_rows(collected: _CollectedWrite, eligible_rows: int) -> _CollectedWrite:
    """Return the collected write, refusing one that does not account for every eligible row.

    The payload count cannot answer this. ``_uncovered_fragments`` reads the covered
    set out of the payloads that came back, so a verdict payload that never arrives
    moves its fragment into the uncovered set, where the blanking pass answers for
    it: the fragment arithmetic balances while that fragment's rows go to NULL. A
    reader cannot tell those rows from ones the run legitimately did not claim, and
    the run would report the eligible count it never wrote.

    Raises:
        CurateWriteError: If the claimed rows do not sum to ``eligible_rows``.

    """
    if collected.rows != eligible_rows:
        msg = (
            f"the write claimed {collected.rows} verdict row(s) for {eligible_rows} eligible row(s); "
            f"refusing to commit a selection that does not account for each of them exactly once"
        )
        raise CurateWriteError(msg)
    return collected


def _blank_uncovered(source: _Source, uncovered: Sequence[int]) -> list[str]:
    """Blank both columns on every fragment the verdict pass did not reach; return its payloads.

    Args:
        source: The pinned table, for the read the workers reopen it at.
        uncovered: The fragments no verdict payload named, in manifest order.

    Returns:
        One payload per blanked fragment; empty when every fragment carried a verdict.

    """
    if not uncovered:
        return []
    logger.info(f"curate write-back: blanking both columns on {len(uncovered)} fragment(s) holding no eligible row")
    results = (
        _work_items(uncovered)
        .map_batches(
            blank_one_fragment,  # type: ignore[arg-type]
            fn_kwargs={
                "uri": source.read.uri,
                "read_version": source.read.read_version,
                "storage_options": source.read.storage_options,
            },
            batch_format="pyarrow",
        )
        .take_all()
    )
    return [str(row[_RESULT_COLUMN]) for row in results]


@attrs.frozen
class _CentroidsArtifact:
    """One published basis object and the hash that names it.

    Attributes:
        uri: Where the bases were written.
        fingerprint: ``sha256`` hex of the archive bytes, which is both the object's
            name and what the commit records to point at it. A reader that re-hashes
            the bytes it fetched can therefore prove it loaded the intended basis.

    """

    uri: str
    fingerprint: str


def _write_centroids(
    config: CurateConfig, fit: _FitResult, *, read_version: int, producers: Mapping[str, str]
) -> _CentroidsArtifact:
    """Publish BOTH fitted bases beside the table, named by their content hash.

    Written BEFORE the commit, which keeps the commit the only boundary at which a
    verdict becomes visible and lets it reference an already-durable object; the
    fingerprint it records is what locates that object. Nothing time- or
    run-derived enters the archive, so one fit always serializes to one name.

    The archive carries what a fused coordinate cannot be read without (block
    order, dims, weights, and the producer identities saying whose vectors it was
    assembled from), what a refit needs (the seed and the sampled fragment ids),
    and the run's rules as text beside the digest the commit stamps. A run that
    fitted no level-2 basis writes ``subtask_centroids`` at ``(0, TEXT_DIM)``,
    unambiguous because a fitted basis always holds at least one centroid; which
    cause left it empty comes from the log (see ``_report_subtask_basis``).

    Args:
        config: The resolved run configuration.
        fit: The fitted bases and what they were fitted on.
        read_version: The pinned version the sample was drawn from, not the one this run commits.
        producers: Producer identity per consumed provenance column.

    Returns:
        The published object and the fingerprint the commit must record.

    """
    weights = _weight_map(config)
    producer_columns = sorted(producers)
    subtask_centroids = (
        fit.subtask_centroids if fit.subtask_centroids is not None else np.zeros((0, TEXT_DIM), dtype=np.float32)
    )
    buffer = io.BytesIO()
    np.savez(
        buffer,
        centroids=fit.centroids,
        effective_k=np.int64(fit.effective_k),
        fused_dim=np.int64(vectors.FUSED_DIM),
        block_dims=np.asarray(vectors.BLOCK_DIMS, dtype=np.int64),
        block_columns=np.asarray([group.primary_vector for group, _ in FUSED_BLOCKS]),
        block_weights=np.asarray([weights[field] for _, field in FUSED_BLOCKS], dtype=np.float64),
        read_version=np.int64(read_version),
        fit_rows=np.int64(fit.fit_rows),
        kmeans_random_state=np.int64(config.kmeans_random_state),
        fit_fragment_ids=np.asarray(list(fit.fragment_ids), dtype=np.int64),
        subtask_centroids=subtask_centroids,
        subtask_k=np.int64(fit.subtask_k),
        subtask_column=np.asarray(SUBTASK_VECTOR_COLUMN),
        # Two parallel arrays rather than one "column=identity" string per entry,
        # so a reader can look a column up without parsing. Typed explicitly
        # because an unfilled group contributes no entry and an untyped empty
        # list would land in the archive as a float array.
        producer_columns=np.asarray(producer_columns, dtype=np.str_),
        producer_identities=np.asarray([producers[column] for column in producer_columns], dtype=np.str_),
        # The rules, as text, beside the geometry they produced. The commit carries
        # only the DIGEST of this, which tells a reader that two versions differ but
        # never where; the text is what answers that, and hashing it back must
        # reproduce config_digest, so the two artifacts check each other.
        resolved_config=np.asarray(config.result_defining_json()),
        config_digest=np.asarray(config.result_defining_digest()),
    )
    payload = buffer.getvalue()
    fingerprint = hashlib.sha256(payload).hexdigest()
    uri = f"{config.clips_lance_uri.rstrip('/')}{CENTROIDS_ROOT_SUFFIX}/{fingerprint}.npz"
    StorageWriter(uri, profile_name=config.storage_profile).write(payload)
    logger.info(
        f"curate centroids: wrote {fit.centroids.shape[0]} x {fit.centroids.shape[1]} locality basis "
        f"and {fit.subtask_k} x {TEXT_DIM} subtask basis to {uri}"
    )
    return _CentroidsArtifact(uri=uri, fingerprint=fingerprint)


def run_curate(config: CurateConfig) -> CurateResult:
    """Run one whole Curate leg against ``config.clips_lance_uri``.

    Fits a basis on a bounded sample, reads the table once at row scale, marks
    near-duplicates, cuts each fairness group to its quota, and publishes every
    verdict in ONE transaction. The write-back is all-or-nothing: a failure
    before that commit leaves no verdict update. ``_preflight`` may already have
    widened the schema via ``ensure_curate_columns()`` before the fit or scan
    work begins.

    Raises:
        FileNotFoundError: If the table does not exist.
        ValueError: On a source-contract violation, an unusable fit sample, a
            corpus with no eligible row or no survivor, or a verdict set that
            does not account for every eligible row. All of these are raised on
            the driver and so are catchable by type.
        CurateWriteError: On a write-back contract violation raised on the driver.
        ray.exceptions.RayError: On any failure raised INSIDE a Ray Data UDF - the
            NULL-label and block-width gates included - because Ray replaces a
            UDF's exception with its own wrapper and the original type does not
            survive the task boundary. Such a violation is identifiable only by
            message, never by ``except ValueError``.

    """
    # Set before the first dataset exists: DataContext is read when an operator is
    # built, so a later call would leave the earliest stages on the old setting.
    configure_ray_data_progress(progress=True)

    source = _preflight(config)
    requested_k = _requested_k(source.eligible_rows, config.target_mean_cluster_rows)
    fit = _fit_centroids(config, source, requested_k)
    # Published before the corpus-wide passes below, so the basis is already durable by
    # the time the commit can reference it. A failure below this line deletes nothing --
    # it leaves this archive and any finished column files behind as objects no manifest
    # references, so they are unreadable rather than half visible.
    centroids = _write_centroids(config, fit, read_version=source.read.read_version, producers=source.producers)
    task_merge, merge_stats = _merge_maps(config, _gather_labels(source))

    scanned = _scan(config, source, fit)
    # Materialized because the count that funds the quotas and the cut that spends
    # it are two passes over the same rows, and the lineage behind them holds the
    # run's GPU stage; the same reason applies to the verdicts below.
    merged = _merge_groups(_retain(config, scanned), task_merge).materialize()
    _report_metrics(merged, config.dedup_eps)
    quota, target = _build_quota(config, _group_counts(merged))
    quotas = quota.quotas()
    unfunded = _report_unfunded(quota, quotas)
    verdicts = _select(config, merged, quotas).materialize()
    counts = _reason_counts(_reason_table(verdicts), source.eligible_rows)

    collected = _write_back(source, verdicts)
    committed_version = _commit(
        source.dataset,
        collected,
        properties=_commit_properties(config, centroids_fingerprint=centroids.fingerprint),
        storage_options=source.read.storage_options,
    )
    logger.info(f"curate: committed v{committed_version} with {collected.rows} verdict row(s)")
    return CurateResult(
        clips_lance_uri=config.clips_lance_uri,
        read_version=source.read.read_version,
        committed_version=committed_version,
        eligible_rows=source.eligible_rows,
        written_rows=collected.rows,
        requested_k=requested_k,
        effective_k=fit.effective_k,
        subtask_k=fit.subtask_k,
        fit_rows=fit.fit_rows,
        fairness_groups=len(quota.level2_keys),
        unfunded_groups=unfunded,
        target=target,
        reason_counts=counts,
        merge_stats=merge_stats,
        centroids_uri=centroids.uri,
    )
