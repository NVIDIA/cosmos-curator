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

"""Fairness: which task gets its share, decided on LABEL SEMANTICS, never on the fused metric.

Curate carries two distinct geometries, computed over different metrics, and this
module owns exactly one of them. LOCALITY (``curate_cluster_id``) is a k-means
partition of the FUSED vector and answers "is this row a near-duplicate of that
one". FAIRNESS is the ``(canonical task, subtask cluster)`` pair and answers "did
this task get its share".

Distinct is not independent, and the difference matters when reading a report.
The two keys share information by construction, because the fused vector weights
subtask text at 0.6 and the level-2 cell is k-means over that same embedding:
their measured normalized mutual information is 0.50, with 55% of the fairness key
already fixed once the locality cluster is known; the apparatus is in
``docs/curator/design/curator-next-curation.md``, "What must stay separate".
What follows from that is only
that neither may be read as the other - they are NOT interchangeable. It has no
operational consequence, because nothing here reads a cluster: ``FairnessQuota``
is built from ``survivor_group_counts`` alone.

The direction that would be a defect is the reverse one, and it is a measurement
rather than a preference: the action block funds 0.2 of the fused metric and is
measurably blind to which task is being performed - task-separation ratio 0.990,
same-task-closer AUC 0.543, which is chance - so no function of the fused vector
can stand in for a task identity. Measurement and apparatus:
``docs/curator/design/curator-next-curation.md``, "Why each modality is present".

The separation is between the two METRICS, not between strings and vectors. Both
levels here are functions of the label's own text embedding - the one block that
does carry label semantics, measured at AUC 0.853 - and neither ever reads the
fused vector. Level 1 folds the task vocabulary pairwise, because the annotation
schema bounds it: the distinct-task count is a property of that schema and does
not move with the row count. Level 2 PARTITIONS the subtask text embedding into
``subtask_clusters`` cells, because its vocabulary is free-form annotator prose
resolved per shard, so its size grows with the corpus: a pairwise fold over it is
both a quadratic driver kernel and a group count no quota can divide.

Four steps, in this order, each a pure function here and wired by ``pipeline``::

    scan     canonicalize_label per row     -> __canonical_task   (level 1 key)
      |      nearest subtask centroid       -> __subtask_cluster  (level 2 key)
      v
    merge    merge_labels once over the DISTINCT TASK labels at merge_theta_task
      |      (driver, O(distinct tasks) state). apply_label_merge then REWRITES
      |      __canonical_task to its representative, so every later step - the
      |      count, the shuffle key, and the cut - sees the merged task.
      v
    count    survivor_group_counts -> per-group capacity, survivors only
      |
      v
    quota    FairnessQuota.build(...).quotas() -> one integer per group
      |
      v
    cut      select_within_quota per group -> selected / below_quota / unfunded

Level 2 has no merge stage and needs none. A merge exists to put two spellings of
one instruction in one group, and a partition of the same embedding does that
already and more aggressively - two spellings of one instruction sit at cosine
~0.99 and land in one cell of 16. The stronger reason is structural: a merge is
driven by one vector per DISTINCT label, which only a driver-side gather over the
whole vocabulary can produce, and that gather is the O(L) state this key was
bounded to remove. Assigning per ROW at the scan is what makes the level-2 key
cost nothing on the driver at all.

``quotas()`` guarantees six properties, and the suite asserts each as an
identity rather than by example:

1. Exact total: quotas sum to ``min(target, total)``.
2. Capacity: ``0 <= quota[g] <= capacity[g]``.
3. Coverage, WITHIN a tier: every group gets one before any gets a second, and at
   a target below the tier's group count exactly ``target`` of its groups are
   funded, whatever their size, chosen by that tier's residual order. Nesting
   weakens it across the two: a task holding one cell spends its whole parent
   share there, so a target equal to the level-2 group count does not promise
   every cell a clip.
4. Uniformity: unsaturated quotas differ by at most one.
5. Determinism: the keys and the seed decide every boundary, so input row order
   cannot reach the result.
6. No float arithmetic anywhere in the allocation.

Size never enters an order: ``capacity`` is consulted only for saturation, so the
tie-break that DENIES representation cannot depend on size any more than the
quota itself does. ``FairnessQuota.build`` normalizes the keys into ascending
order once, which leaves no comparator downstream to get wrong and makes each
parent's children contiguous, so the nested pass reads one slice per parent.

What the ascending key order must NOT decide is who wins the remainder, and at
``target`` below a tier's group count the remainder is the whole allocation. Both
tiers therefore fund their remainder by ``residual_ranks``, a seeded digest, and
each has its own correlation that the key order would otherwise fold into the
selected set:

- Level 1 hashes the LABEL. Canonical task labels are verb-initial annotation
  prose, so their alphabet is ordered by action type: funding a prefix of it drops
  whole families of instructions together, and drops the same families on every
  run.
- Level 2 hashes the ``(task, cell)`` PAIR. A fitted cell id is an arbitrary
  index, but it is a GLOBAL one, so ascending cell id is the same order under
  every parent - a cell that loses its task's last place loses it under every
  task at once, and whole regions of subtask meaning reach the selected set with
  no clips at all. Hashing the pair rather than the cell is what decorrelates the
  tiers from each other: the loss still falls somewhere, but not on the same cell
  everywhere.

Within ONE pass the unfunded count is identical under any order, because the fill
line is a function of the capacity multiset and the target alone. Nesting leaves a
residue: the level-1 residual decides which parents win an extra row, and parents
differ in occupied-cell count, so a swapped parent moves the level-2 total by one.
The variation is bounded by the level-1 REMAINDER, which is not small - a corpus
mixing one-cell and five-cell tasks moves the count by tens of groups across seeds.
That makes the count an even weaker detector of either correlation, not a stronger
one: it is a scarcity reading, never a comparator reading.

The allocation is nested rather than flat because neither single pass is fair at
both granularities: a flat pass over the ``(task, subtask)`` pairs lets a task win
by owning more subtasks, while a pass over tasks alone never guarantees that each
subtask is covered. Both levels and the target are run-invariant state, and the
two levels must derive from identical inputs or they disagree about how much a
parent received.

The two verdicts a losing row can carry differ in what lost it: ``unfunded``
means the group's quota is zero - the water-fill never funded the group at all
- while ``below_quota`` means the group was funded and this row ranked outside
its share.

Group identity is deliberately not persisted, so ``below_quota`` and ``unfunded``
are not auditable from stored state: recovering "which group" needs the task label
set, the task threshold, the clip counts that ordered the representatives, and
the centroid artifact that fixed the subtask cells. Reproducibility rests entirely
on this module being deterministic, which is why every ordering here is total.
The concessions are transient WARNINGs. A task merge that folds away most of its
vocabulary says so, because a mis-set threshold would otherwise redistribute a
large part of the corpus with nothing anywhere recording that it did. Starvation
says so too, and separately per level, because the two levels starve for
different reasons and answer to different knobs: a target below the task count is
not a ``subtask_clusters`` problem and cannot be fixed by changing it.

See docs/curator/design/curator-next-curation.md.
"""

import collections
import hashlib
import itertools
import operator
import unicodedata
from collections.abc import Mapping, Sequence

import attrs
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from loguru import logger

from cosmos_curator.next.embeddings.schemas import KEY_COLUMN
from cosmos_curator.next.recipes.curation.columns import (
    CANONICAL_TASK_COLUMN,
    CURATE_SELECTION_REASON,
    DEDUP_KEY_COLUMN,
    DISTANCE_COLUMN,
    FRAGMENT_COLUMN,
    RAY_COUNT_COLUMN,
    SUBTASK_CLUSTER_COLUMN,
    TASK_COLUMN,
    VERDICT_ROW,
    CurateReason,
    WithinGroupOrder,
)

# One level-2 fairness group: a canonical task paired with one subtask-text
# cluster cell. Named because it is the type of the quota's key, of the shuffle
# key, and of the map select_within_quota looks itself up in - three places that
# have to agree on the pair's shape.
Level2Key = tuple[str, int]

# Trailing marks (and the space) canonicalization strips. The space is IN the set
# so a mark separated by a space ("open folder . ") strips cleanly to "open
# folder"; stripping is trailing-only, so a leading mark survives into the key.
_TRAILING_PUNCTUATION = ".!?,;: "

# Labels whose folded form IDENTIFIES the canonicalize_label rule: probes covering
# every step it performs, plus the blank-group edge its docstring promises. The
# rule is result-defining - two runs that fold labels differently build different
# fairness groups - so the run identity carries what these fold TO rather than a
# version suffix someone has to remember to bump.
#
# Most probes are MINIMAL witnesses of one step: DELETE that step and the probe's
# result moves. Deletion is the cheap case, and one probe per step covers it.
# SUBSTITUTING a step - another normalization form, another spelling of the
# collapse, the same steps in a different order - is covered only where a probe
# was built for it, because a substitute normally agrees with the original on a
# probe chosen to witness the step's mere presence. Two probes below exist for
# that case alone and say so.
#
# They are deliberately synthetic rather than plausible task labels, because
# nothing reads them as data - a reader who takes them for examples goes looking
# for a meaning they do not carry.
#
# Two are built FROM _TRAILING_PUNCTUATION instead of spelling a mark by hand, so
# every member of that set is exercised. A hand-written "." witnesses only itself,
# which leaves a mark added to or removed from the set invisible to the identity
# even though the set decides which labels reach one key.
#
# The space before the marks in that strip probe is load-bearing, and it is the
# one member the collapse step can hide: whitespace at either end is already gone
# by the time rstrip runs, so the space earns its place in the set only when a
# mark sits BEHIND one ("a ." -> "a"). Close that gap and the set's space becomes
# unreachable while every probe still agrees.
_CANONICALIZATION_PROBES: tuple[str, ...] = (
    # NFC composes the two codepoints "e" + combining acute into the one "e-acute".
    # The base letter is constrained, not decorative: it has to be one with a
    # precomposed form. "q" + combining acute has none, stays two codepoints, and
    # would witness nothing.
    "e\u0301",
    # SUBSTITUTION witness for step 1, NFC against NFKC. U+00B2 (superscript two)
    # is chosen because it has a COMPATIBILITY decomposition and no canonical one,
    # which is exactly where the two forms disagree: NFC leaves it alone, NFKC
    # folds it to "2", so under NFKC a label spelled with the superscript joins the
    # group of one spelled with the digit. The ligature U+FB01 looks like the
    # obvious character for this and is NOT one - casefold already maps it to "fi",
    # so both forms end up agreeing and the probe would witness nothing.
    "\u00b2",
    # A run of MIXED whitespace collapses to exactly one space. "a" and "b" are
    # placeholders, not literals - "test" and "qwerty" fold identically - but they
    # are not free-form either: an anchor must be INERT under the other three
    # steps, or this probe stops isolating THIS one. "Test" also moves under
    # casefold, and a mark like "." is eaten by the strip; lowercase ASCII is
    # simply the cheapest character that nothing else touches. Something must also
    # sit on each side, or the run is leading/trailing, split() drops it, and the
    # probe stops showing whether one space survived or none. Mixing tab and
    # newline in catches a narrower rewrite (str.replace of a double space) that a
    # spaces-only probe folds identically under.
    "a \t\n b",
    # SUBSTITUTION witness for step 2 and for the order of steps 2 and 4:
    # whitespace at the ENDS with a mark behind it. " ".join(split()) trims the
    # ends, so a collapse that only rewrites internal runs (re.sub of r"\s+")
    # leaves the leading space in the key; and an rstrip moved AHEAD of the
    # collapse halts on the tab and keeps the mark. Either edit folds this probe
    # differently while the anchored collapse probe above still agrees.
    " a.\t",
    # casefold, not lower: the sharp s is the character where the two disagree
    # ("ss" against an unchanged "sharp s"). Any ordinary letter folds the same
    # under both and cannot say which one the code calls.
    "\u00df",
    # Every trailing mark strips and a leading one survives. "a" is an anchor
    # again; the space in front of the marks is not, see the note above.
    f".a {_TRAILING_PUNCTUATION}",
    # A label of nothing but marks folds to the blank group - deliberately
    # unanchored, so there is nothing left for the strip to stop at.
    _TRAILING_PUNCTUATION,
)

# Below this norm a label embedding has no direction, so cosine similarity is
# undefined for it. Such a label neither joins a representative nor absorbs one.
#
# PUBLIC because it is a shared floor, not an implementation detail: a caller that
# reports how many labels reached the merge without a direction has to test the
# same bound the merge itself applies, and a second copy of the number would drift
# from this one unnoticed.
MIN_LABEL_NORM = 1e-12

_VECTOR_NDIM = 2

# Warn when the merge keeps fewer than one representative per this many input
# labels - i.e. when it more than halves the group count. Under uniform max-min a
# level-1 share is approximately target / R, so the REPRESENTATIVE COUNT is what
# exactly determines every surviving group's share, not a proxy for it: halving R
# doubles every share, whatever the labels underneath were. That is why this line
# is computed from the counts alone and needs nothing from the corpus.
#
# It is therefore only half the picture, and the other half is a different
# quantity rather than a finer version of this one: how much of the CORPUS changed
# group. A fold of rare wording variants barely moves R and moves almost no clips;
# two of the largest labels merging moves a large share of the clips while barely
# moving R. The caller reports that share separately for exactly that reason.
#
# Deliberately NOT a CurateConfig field. A knob here would be a tuning surface
# nobody has evidence to set, and it is the coordinator's call rather than a
# per-run choice. Cheap to revisit once a theta-0.95 merge has actually been
# observed over real label prose.
_MERGE_COLLAPSE_WARN_FACTOR = 2

# Warn when more than one level-2 group in this many receives nothing at all --
# i.e. above a fifth of the partition. Under uniform max-min water-fill a child's
# quota is zero exactly when its parent's budget was smaller than the parent's
# child count - the parent had nothing to distribute, so the residual order alone
# decided which of its children were funded. Counting the zero-quota groups
# therefore MEASURES that regime, where a maximum over the quotas only stands in
# for it: one task with too few rows to populate subtask_clusters cells
# contributes a single cell holding that whole task's level-1 quota, so a handful
# of small tasks lift the maximum clear of any threshold while the corpus is still
# allocated by the residual order.
#
# A fifth rather than a half, because a fifth of the level-2 partition being
# decided by a seed instead of by the corpus is already too much to leave unsaid.
# Seeding the order fixed WHERE the loss falls, and very nearly how large it is:
# within one pass the count below is the same under any order, and nesting leaves
# only a residue bounded by the level-1 remainder (see the module docstring). So
# this line measures scarcity and is not weakened by the comparator, even though
# reseeding can move it by tens of groups on a corpus of mixed cell occupancy.
# A measured sweep is what moved the factor off a halving: at scarce targets a
# quarter to over a third of the partition received nothing while a halving factor
# stayed silent, and neither share is a regime an operator would want to discover
# from the selected set. The swept numbers are in
# docs/curator/design/curator-next-curation.md.
#
# It is a SMALL-CORPUS signal either way. At the 250M-500M design envelope G is
# bounded by the task vocabulary times the cell count, which the doc above works
# out to orders of magnitude below a multi-million target, so every group is
# funded, unfunded is zero, and this line is silent whatever the factor. What it
# catches is a pilot run, or a k set far too high for the corpus it was pointed
# at.
#
# This does NOT overlap _MERGE_COLLAPSE_WARN_FACTOR, which fires on too FEW
# level-1 groups (a task merge that folded away most of its vocabulary). This
# regime is caused by too MANY level-2 groups: more cells than the target can
# reach. The two can hold at once and mean opposite things, so neither warning
# can be widened to cover the other. Since the level-2 key became a bounded
# partition this line is also the run-time read-out for subtask_clusters: it
# fires when k * distinct tasks is large against the target.
#
# It is a BACKSTOP rather than the primary check, because the same condition is
# computable before a run: a parent starves cells exactly when its budget,
# approximately target / MERGED tasks, falls below the number of cells it
# OCCUPIES - at most k, and less for a task too small to populate them all. A k
# modestly above that quotient is already allocated by the residual order while
# staying under a fifth, so silence here does not certify the value. The operator-facing
# arithmetic is in docs/curator/guides/curate-runbook.md.
_DEGENERATE_UNFUNDED_WARN_FACTOR = 5


def canonicalize_label(text: str) -> str:
    """Fold one raw annotation label to its canonical fairness key.

    The four steps run in a fixed order that is part of the contract:

    1. ``unicodedata.normalize("NFC", ...)`` so two byte sequences denoting the
       same characters can compare equal at all;
    2. ``" ".join(text.split())`` so tabs, newlines, and runs of spaces collapse
       before case is folded;
    3. ``casefold()`` for case-insensitive, non-ASCII-aware equality;
    4. ``rstrip`` of trailing punctuation and spaces, so an instruction written
       with and without its full stop reaches one key.

    A wholly-punctuation label folds to ``""`` and joins the (legitimate) blank
    fairness group.

    Args:
        text: One raw ``task_name`` or ``subtask_name``.

    Returns:
        The canonical key; possibly the empty string.

    """
    normalized = unicodedata.normalize("NFC", text)
    collapsed = " ".join(normalized.split())
    folded = collapsed.casefold()
    return folded.rstrip(_TRAILING_PUNCTUATION)


def canonicalization_contract() -> list[list[str]]:
    """Return the level-1 grouping rule's identity, as JSON-ready probe/result pairs.

    Which labels share a fairness group decides which rows compete for one quota,
    so a release that folds labels differently moves verdicts while every config
    file stays byte-identical. The run identity therefore has to carry this rule,
    and it carries it as what the rule DOES to fixed inputs - computed here rather
    than recorded, so no second value has to be kept in step with the code.

    A finite probe set cannot promise that every edit moves the identity. What
    this one delivers is each step's PRESENCE, the ORDER of the collapse and the
    strip - the pair whose order is known to change the result - and every MEMBER
    of ``_TRAILING_PUNCTUATION``. Substituting a step for a near-equivalent moves
    the identity only where a probe was built for that substitution.

    Pairs rather than a hash: the digest already hashes this, and keeping the text
    readable lets an operator diffing two archived configs see WHICH fold moved.

    Returns:
        One ``[probe, folded]`` pair per entry in ``_CANONICALIZATION_PROBES``.

    """
    return [[probe, canonicalize_label(probe)] for probe in _CANONICALIZATION_PROBES]


def canonicalize_labels(batch: pa.Table) -> pa.Table:
    """Append the canonical task column to one scanned batch.

    Level 1 only. The level-2 key is a cluster cell over the subtask TEXT
    EMBEDDING rather than over the subtask string, so no canonicalization of
    ``subtask_name`` exists to do: NFC, case, and trailing punctuation do not
    move an embedding far enough to change which of ``subtask_clusters`` cells
    it lands in, and the embedding already identifies two spellings of one
    instruction that this fold could not (word order, synonyms, tense).

    Args:
        batch: Rows carrying ``task_name``.

    Returns:
        The batch with ``__canonical_task`` appended.

    Raises:
        ValueError: If ``task_name`` holds a NULL. A row with no task has no
            fairness group, so it would silently leave the population that the
            quota is computed over.

    """
    column = batch.column(TASK_COLUMN)
    if column.null_count:
        msg = f"{column.null_count} row(s) carry a NULL {TASK_COLUMN}, which has no fairness group"
        raise ValueError(msg)
    folded = [canonicalize_label(value) for value in column.to_pylist()]
    return batch.append_column(
        pa.field(CANONICAL_TASK_COLUMN, pa.string(), nullable=False),
        pa.array(folded, type=pa.string()),
    )


def _unit_rows(matrix: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Return row-normalized ``matrix`` and the mask of rows that had a direction.

    A row below ``MIN_LABEL_NORM`` is left as zeros rather than divided, so it
    scores 0 against every representative instead of producing NaN.

    Deliberately NOT ``vectors.unit_rows``, despite the shape: this runs over one
    vector per distinct LABEL, where a non-finite value is a programming error
    ``merge_labels`` raises on, so there is no finiteness mask to carry. The
    corpus-row path must instead route a bad row silently to a reserved cell.
    """
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
    directed = norms > MIN_LABEL_NORM
    unit = np.zeros_like(matrix)
    # The denominator stays float64: a finite float32 row can carry a norm past
    # the float32 ceiling, and narrowing it first would make it inf and zero a
    # row that IS directed - which merges nothing and absorbs nothing, so the
    # label silently keeps its own quota. The float32 ``out`` is what keeps that
    # promotion free: the ufunc buffers it rather than building a float64 copy.
    np.divide(matrix, norms[:, None], out=unit, where=directed[:, None])
    return unit, directed


def merge_labels(
    labels: Sequence[str],
    counts: Sequence[int],
    vectors: npt.NDArray[np.floating],
    *,
    theta: float,
) -> dict[str, str]:
    """Return every distinct TASK label mapped to its group's representative, itself for a leader.

    Each label joins the first representative it strictly exceeds ``theta`` against, or becomes one;
    membership is direct, never transitive. A label with norm at or below ``MIN_LABEL_NORM`` never
    merges, alone in its own group.

    Level 1 only. This is an O(L*R) driver kernel over one vector per DISTINCT label, which is
    affordable exactly because the annotation schema bounds the task vocabulary; the level-2
    vocabulary has no such bound and is partitioned in the scan instead.

    Args:
        labels: The distinct canonical task labels.
        counts: Clip count per label, parallel to ``labels``; the walk runs count DESC, label ASC.
        vectors: ``(len(labels), dim)`` label embeddings, parallel to ``labels``; re-normalized here.
        theta: Cosine a label must strictly exceed to join a representative.

    Raises:
        ValueError: On a length mismatch, a duplicate label, a misaligned ``vectors``,
            or non-finite values in ``vectors``.

    """
    if len(labels) != len(counts):
        msg = f"labels and counts must have equal length, got {len(labels)} and {len(counts)}"
        raise ValueError(msg)
    if len(set(labels)) != len(labels):
        msg = "labels must be distinct; merge runs over the distinct label set, not over rows"
        raise ValueError(msg)
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != _VECTOR_NDIM or matrix.shape[0] != len(labels):
        msg = f"vectors must be a ({len(labels)}, dim) matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    if not np.isfinite(matrix).all():
        msg = "vectors must be finite"
        raise ValueError(msg)
    if not labels:
        return {}

    unit, directed = _unit_rows(matrix)
    # Representatives are appended in walk order, so row 0 of this buffer is the
    # earliest representative and argmin-of-hits IS "the first it exceeds".
    representative_vectors = np.empty_like(unit)
    representative_labels: list[str] = []
    merged: dict[str, str] = {}

    # Why this exists: post-repair labels are prose, so exact-string grouping
    # fragments one task's quota across its wording variants - a task written N
    # ways draws roughly N times its share. Frequent labels lead (count DESC), so
    # variants are absorbed into the wording the corpus actually uses, and the
    # label tie-break keeps two equal-count labels resolving the same way.
    #
    # The walk runs over DISTINCT labels, never rows, at O(L * R) similarity work:
    # one input string always embeds to one vector, so any row's vector represents
    # its whole label.
    order = sorted(range(len(labels)), key=lambda index: (-counts[index], labels[index]))
    for index in order:
        label = labels[index]
        found = len(representative_labels)
        if found and directed[index]:
            similarity = representative_vectors[:found] @ unit[index]
            hits = np.flatnonzero(similarity > theta)
            if hits.size:
                merged[label] = representative_labels[int(hits[0])]
                continue
        representative_vectors[found] = unit[index]
        representative_labels.append(label)
        merged[label] = label

    n_in, n_out = len(labels), len(representative_labels)
    if n_out * _MERGE_COLLAPSE_WARN_FACTOR < n_in:
        logger.warning(f"fairness merge collapsed task groups {n_in} -> {n_out} (theta={theta})")
    return merged


def _remap_column(column: pa.ChunkedArray, merged: Mapping[str, str], name: str) -> pa.Array:
    """Rewrite one label column through a merge map.

    Raises:
        ValueError: If the column holds a label the map does not name. The map is
            built from the distinct labels of the same table version, so a
            missing one means the driver and the rows disagree about the
            vocabulary - which would create a group with no quota.

    """
    if column.null_count:
        msg = f"{column.null_count} row(s) carry a NULL {name}; canonicalization emits a value for every row"
        raise ValueError(msg)
    try:
        rewritten = [merged[value] for value in column.to_pylist()]
    except KeyError as error:
        msg = f"{name} value {error.args[0]!r} is absent from the label merge map"
        raise ValueError(msg) from error
    return pa.array(rewritten, type=pa.string())


def apply_label_merge(batch: pa.Table, task_merge: Mapping[str, str]) -> pa.Table:
    """Rewrite a batch's canonical task label to its representative.

    Runs before the count and the shuffle, so the group counts, the group key,
    and the cut all address the same merged group. Overwrites the column in place
    rather than adding a second: nothing downstream needs the pre-merge label,
    and nothing persists either form.

    Merging tasks merges their level-2 namespaces. Two wording variants of one
    task carrying the same subtask cluster cell become ONE level-2 group --
    intended, because it is one semantic task and one region of subtask meaning,
    but a consequence of level-1 merging rather than of the level-2 partition.

    Args:
        batch: Rows carrying ``__canonical_task``.
        task_merge: Task label -> representative, from ``merge_labels``.

    Returns:
        The batch with ``__canonical_task`` rewritten; one row out per row in,
        every other column untouched.

    Raises:
        ValueError: If the label is NULL or absent from the merge map.

    """
    index = batch.schema.get_field_index(CANONICAL_TASK_COLUMN)
    if index < 0:
        msg = f"batch carries no {CANONICAL_TASK_COLUMN} column; canonicalization must run before the merge"
        raise ValueError(msg)
    return batch.set_column(
        index,
        pa.field(CANONICAL_TASK_COLUMN, pa.string(), nullable=False),
        _remap_column(batch.column(index), task_merge, CANONICAL_TASK_COLUMN),
    )


def survivor_group_counts(group_counts: pa.Table) -> dict[Level2Key, int]:
    """Return the per-group survivor capacity the quota is allocated over.

    A survivor is a row de-duplication left unreasoned: duplicates and
    invalid-embedding rows already have a verdict, so they neither consume a
    quota nor enlarge the group that funds one.

    Args:
        group_counts: The ``O(G)`` reduction of the post-dedup rows, grouped by
            ``(__canonical_task, __subtask_cluster, curate_selection_reason)`` and
            counted. Rows whose reason is non-NULL are dropped here.

    Returns:
        ``(__canonical_task, cluster)`` -> survivor count, over the groups that
        hold at least one survivor. Its length is the observed group count ``G``,
        bounded by distinct tasks times ``subtask_clusters`` rather than growing
        with the corpus. The ``+ 1`` for the reserved ``NO_SUBTASK_CLUSTER`` cell
        in the published ceiling is slack: that cell never appears beside a fitted
        one, so no run reaches it.

    """
    rows = zip(
        group_counts.column(CANONICAL_TASK_COLUMN).to_pylist(),
        group_counts.column(SUBTASK_CLUSTER_COLUMN).to_pylist(),
        group_counts.column(CURATE_SELECTION_REASON).to_pylist(),
        group_counts.column(RAY_COUNT_COLUMN).to_pylist(),
        strict=True,
    )
    return {(task, int(cluster)): int(count) for task, cluster, reason, count in rows if reason is None and count > 0}


def residual_ranks(keys: Sequence[str], seed: int) -> npt.NDArray[np.int64]:
    """Rank ``keys`` by a seeded digest, so residual funding does not follow key order.

    The rank of a key depends on its full text and on ``seed``, never on its
    position, so a prefix of the ranking is not a prefix of the input. Uses
    blake2b rather than ``hash()``, which is salted per interpreter and would make
    the selected set vary between processes.

    Args:
        keys: Distinct digest inputs, positional and parallel to the tier's groups.
            Each tier supplies its own encoding, so these need not be sorted.
        seed: Result-defining seed; a different value draws a different subset.

    Returns:
        ``(len(keys),)`` ranks in ``[0, len(keys))``, positional and parallel to
        ``keys``. Ties on an equal digest fall back to the key itself, so the
        order is total even under a collision.

    """
    digests = [hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=16).digest() for key in keys]
    order = sorted(range(len(keys)), key=lambda index: (digests[index], keys[index]))
    ranks = np.empty(len(keys), dtype=np.int64)
    ranks[order] = np.arange(len(keys), dtype=np.int64)
    return ranks


def _pair_digest_keys(pairs: Sequence[Level2Key]) -> list[str]:
    """Encode ``(task, cell)`` pairs as digest inputs, injectively.

    The task is in the key even though one call sees only one parent's children:
    it is what makes a cell's rank differ per task, so a cell starved under one
    task can win under another.

    The cell id leads because it is an integer and so holds no delimiter, which
    leaves the task label - free-form prose that may contain the delimiter - as
    the unambiguous tail. Distinct pairs therefore encode to distinct strings, so
    no two level-2 groups can be handed one rank. Changing this encoding changes
    every scarce-budget selection.
    """
    return [f"{cell}|{task}" for task, cell in pairs]


def _water_fill(
    capacity: npt.NDArray[np.int64],
    target: int,
    residual_order: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Allocate ``target`` over one tier by integer max-min.

    Clamp the target to ``[0, sum]``; find the largest integer fill line ``L`` with
    ``sum(min(capacity, L)) <= target``; set ``quota = min(capacity, L)``; hand the
    remainder one row each to groups whose CAPACITY exceeds ``L``, never to the ones
    the line left whole, in ``residual_order`` - the whole allocation when ``L`` is 0.

    ``#`` is a kept row, ``.`` capacity above the line::

        M = 8, L = 2 (L = 3 would keep 9)
        k0  size 5  [ # # | . . . ]   active, may take the leftover row
        k1  size 2  [ # # ]           saturated
        k2  size 4  [ # # | . . ]     active, may take the leftover row
        k3  size 1  [ # ]             saturated

    Args:
        capacity: Per-group survivor counts, positional in key-ascending order.
        target: Rows to allocate across this tier.
        residual_order: Per-group rank deciding who wins the remainder, positional
            and parallel to ``capacity``.

    Returns:
        Per-group quotas, positional and parallel to ``capacity``.

    """
    n_groups = int(capacity.size)
    if n_groups == 0:
        return np.zeros(0, dtype=np.int64)
    total = int(capacity.sum())
    demand = max(0, min(target, total))
    if demand == 0:
        return np.zeros(n_groups, dtype=np.int64)
    if demand == total:
        return capacity.astype(np.int64, copy=True)

    # L is a function of the multiset of capacities and the target alone; the
    # sorted copy exists only to evaluate that monotone function, and never
    # decides who is funded (residual_order does that, below).
    capacity_sorted = np.sort(capacity)
    prefix = np.cumsum(capacity_sorted)

    def saturated_sum(fill_line: int) -> int:
        below_count = int(np.searchsorted(capacity_sorted, fill_line, side="left"))
        below_total = int(prefix[below_count - 1]) if below_count > 0 else 0
        return below_total + fill_line * (n_groups - below_count)

    low, high = 0, int(capacity_sorted[-1])
    while low < high:
        mid = (low + high + 1) // 2
        if saturated_sum(mid) <= demand:
            low = mid
        else:
            high = mid - 1
    fill_line = low

    quota = np.minimum(capacity, fill_line)
    remainder = demand - int(quota.sum())
    if remainder > 0:
        active = np.flatnonzero(capacity > fill_line)  # positional -> key order
        # Stable, so an equal rank still resolves by key and the order stays
        # total. Sorting only the active subset keeps this O(A log A).
        active = active[np.argsort(residual_order[active], kind="stable")]
        quota[active[:remainder]] += 1
    return quota


def _validate_level1_slices(
    level1_keys: tuple[str, ...],
    level1_capacity: npt.NDArray[np.int64],
    level2_keys: tuple[Level2Key, ...],
    level2_capacity: npt.NDArray[np.int64],
) -> None:
    """Reject level-1 metadata that does not match contiguous level-2 slices."""
    derived_level1_keys: list[str] = []
    derived_level1_capacity: list[int] = []
    for (parent, _child), count in zip(level2_keys, level2_capacity, strict=True):
        if not derived_level1_keys or derived_level1_keys[-1] != parent:
            derived_level1_keys.append(parent)
            derived_level1_capacity.append(int(count))
        else:
            derived_level1_capacity[-1] += int(count)
    if tuple(derived_level1_keys) != level1_keys:
        msg = "level1_keys must match the ordered parents in level2_keys"
        raise ValueError(msg)
    if not np.array_equal(level1_capacity, np.array(derived_level1_capacity, dtype=np.int64)):
        msg = "level1_capacity must equal the summed child capacities"
        raise ValueError(msg)


@attrs.frozen(eq=False)
class FairnessQuota:
    """Nested integer max-min quotas over canonical label groups at one target.

    ``eq=False`` because it holds ndarrays.

    Tasks are funded equally, then each task's own quota is split equally among its subtask cells::

        level-1 capacities --water-fill(target)--> level-1 quotas, each its children's target
        level-2 capacities --water-fill(parent quota)--> level-2 quotas

    Both tiers fund their remainder by ``residual_ranks`` under one seed, level 1
    over the task label and level 2 over the ``(task, cell)`` pair, for the reason
    given in the module docstring.

    Attributes:
        level1_keys: Canonical task keys in ascending order.
        level1_capacity: Per-task survivor counts, parallel to ``level1_keys``.
        level2_keys: ``(task, cluster)`` pairs in ascending order.
        level2_capacity: Per-pair survivor counts, parallel to ``level2_keys``.
        target: The resolved integer target for this run.
        residual_seed: Seeds both tiers' remainder order; result-defining.

    """

    level1_keys: tuple[str, ...]
    level1_capacity: npt.NDArray[np.int64]
    level2_keys: tuple[Level2Key, ...]
    level2_capacity: npt.NDArray[np.int64]
    target: int
    residual_seed: int = 0

    def __attrs_post_init__(self) -> None:
        """Reject state ``quotas()`` cannot trust: length mismatch, negative target, unsorted keys.

        Also rejects non-positive level-2 capacities and level-1 metadata that does
        not match the contiguous level-2 slices ``quotas()`` reads.

        Raises:
            ValueError: On any invariant violation.

        """
        if len(self.level1_keys) != self.level1_capacity.size:
            msg = "level1_keys and level1_capacity must have equal length"
            raise ValueError(msg)
        if len(self.level2_keys) != self.level2_capacity.size:
            msg = "level2_keys and level2_capacity must have equal length"
            raise ValueError(msg)
        if self.target < 0:
            msg = f"target must be >= 0, got {self.target}"
            raise ValueError(msg)
        if any(not (lower < upper) for lower, upper in itertools.pairwise(self.level1_keys)):
            msg = "level1_keys must be strictly ascending; construct via build"
            raise ValueError(msg)
        if any(not (lower < upper) for lower, upper in itertools.pairwise(self.level2_keys)):
            msg = "level2_keys must be strictly ascending; construct via build"
            raise ValueError(msg)
        if np.any(self.level2_capacity <= 0):
            msg = "level2_capacity must be strictly positive"
            raise ValueError(msg)
        _validate_level1_slices(
            self.level1_keys,
            self.level1_capacity,
            self.level2_keys,
            self.level2_capacity,
        )

    @classmethod
    def build(
        cls,
        *,
        level2_keys: Sequence[Level2Key],
        level2_counts: Sequence[int],
        target: int,
        residual_seed: int = 0,
    ) -> "FairnessQuota":
        """Build a normalized quota from the observed groups and their survivor counts.

        Level-1 capacities are derived by summing the pair counts per task. Keys
        are sorted into ascending order once here; nothing downstream re-sorts,
        so input iteration order cannot reach the result.

        Args:
            level2_keys: Distinct observed ``(task, cluster)`` pairs.
            level2_counts: Survivor count per pair; strictly positive.
            target: Resolved integer target; ``>= 0``.
            residual_seed: Seeds both tiers' remainder order; result-defining.

        Returns:
            The normalized ``FairnessQuota``.

        Raises:
            ValueError: On a length mismatch, a non-positive count, a duplicate
                pair, or a negative target.

        """
        if len(level2_keys) != len(level2_counts):
            msg = "level2_keys and level2_counts must have equal length"
            raise ValueError(msg)
        if len(set(level2_keys)) != len(level2_keys):
            msg = "level2_keys must be distinct observed pairs"
            raise ValueError(msg)
        if any(count <= 0 for count in level2_counts):
            msg = "level2_counts must be strictly positive (already-reasoned rows are excluded upstream)"
            raise ValueError(msg)

        ordered = sorted(zip(level2_keys, level2_counts, strict=True), key=operator.itemgetter(0))
        sorted_keys: tuple[Level2Key, ...] = tuple(key for key, _ in ordered)
        sorted_caps = np.array([count for _, count in ordered], dtype=np.int64)

        # Distinct level-1 keys appear in ascending order already, so derive them
        # order-preserving (equality only) rather than sorting a second time.
        level1_keys: list[str] = []
        level1_caps: list[int] = []
        for (parent, _child), count in zip(sorted_keys, sorted_caps, strict=True):
            if not level1_keys or level1_keys[-1] != parent:
                level1_keys.append(parent)
                level1_caps.append(int(count))
            else:
                level1_caps[-1] += int(count)

        return cls(
            level1_keys=tuple(level1_keys),
            level1_capacity=np.array(level1_caps, dtype=np.int64),
            level2_keys=sorted_keys,
            level2_capacity=sorted_caps,
            target=int(target),
            residual_seed=int(residual_seed),
        )

    def quotas(self) -> dict[Level2Key, int]:
        """Return the level-2 quota for every observed pair.

        Runs the level-1 water-fill at ``target``, then one water-fill per parent
        over that parent's contiguous children using the parent's level-1 quota
        as the target. Level-2 quotas therefore sum to their parent's quota by
        construction, and the whole allocation sums to ``min(target, total)``.

        Returns:
            ``(task, cluster)`` -> integer quota. Level-1 quotas are an
            intermediate of this call and deliberately never escape it.

        """
        level1_quota = _water_fill(
            self.level1_capacity,
            self.target,
            residual_ranks(self.level1_keys, self.residual_seed),
        )
        result: dict[Level2Key, int] = {}
        start = 0
        n_pairs = len(self.level2_keys)
        for parent_index, parent_key in enumerate(self.level1_keys):
            stop = start
            while stop < n_pairs and self.level2_keys[stop][0] == parent_key:
                stop += 1
            children = self.level2_keys[start:stop]
            child_quota = _water_fill(
                self.level2_capacity[start:stop],
                int(level1_quota[parent_index]),
                residual_ranks(_pair_digest_keys(children), self.residual_seed),
            )
            for offset, pair in enumerate(children):
                result[pair] = int(child_quota[offset])
            start = stop
        return result

    def unfunded_groups(self, quotas: Mapping[Level2Key, int]) -> int:
        """Return how many level-2 groups received nothing, warning per CAUSE.

        A quota is zero when the parent's budget was under its child count, and the
        two situations that produce it have DIFFERENT remedies, so they warn
        separately. A starved TASK means ``target`` is below the task count, which
        only a larger target or a harder merge fixes. Cells starved inside a funded
        task are the read-out for ``subtask_clusters``.

        A starved task suppresses the level-2 line, and that is what keeps this
        method the sole owner of ONE zero-quota number: with every task funded the
        zero-quota groups ARE the cells starved inside funded tasks, so the count
        the caller logs is the count this threshold tested. Nothing about the group
        set is persisted, so these lines are the only signal either way.

        Args:
            quotas: The map ``quotas()`` returned for this same allocation.

        Returns:
            Groups holding a quota of zero, both causes together.

        """
        unfunded = sum(1 for quota in quotas.values() if quota == 0)
        if self.target <= 0:
            return unfunded

        allocated_per_task = collections.Counter[str]()
        for (task, _cell), quota in quotas.items():
            allocated_per_task[task] += quota
        starved_tasks = sum(1 for task in self.level1_keys if allocated_per_task[task] == 0)

        # Binary rather than thresholded, and that is a property of the water-fill
        # rather than a choice: a task is starved only when the level-1 fill line
        # is zero, which happens exactly when the target is below the task count.
        # So any starved task at all names one cause and one remedy - and the
        # level-2 share is uninformative here, because every funded task then holds
        # a quota of exactly one, making that share a function of subtask_clusters
        # rather than of the corpus. Reporting it would contradict this line.
        if starved_tasks > 0:
            logger.warning(
                f"fairness target {self.target} is below the {len(self.level1_keys)} task group(s), so "
                f"{starved_tasks} task(s) draw no selected clips and every survivor in them lands as unfunded; "
                f"raise the target above the task count, or LOWER merge_theta_task to fold the task "
                f"vocabulary harder. Lowering subtask_clusters cannot help, because the shortfall is at the "
                f"task level, so the level-2 line is suppressed until every task is funded"
            )
            return unfunded

        # Strictly past a fifth warns, exactly a fifth does not; kept as an integer
        # product because every other line in this allocation is integer-only.
        if _DEGENERATE_UNFUNDED_WARN_FACTOR * unfunded > len(quotas):
            logger.warning(
                f"fairness quota is degenerate at level 2: {unfunded} of {len(quotas)} group(s) received "
                f"nothing, so which cells are funded is decided by fairness_residual_seed and not by the "
                f"corpus; lower subtask_clusters or raise the target"
            )
        return unfunded


# Within-group sort keys per ordering mode. clip_id and fragment_id ascending are
# the final tie-breaks in every mode, which is what makes the selected set
# reproducible under any read or shuffle order; neutral consults clip_id alone.
# farthest keeps the atypical rows and so sorts distance DESCENDING; nearest
# inverts it. The distance is to a FUSED centroid, so farthest buys appearance
# and motion atypicality - never task diversity, which the labels already carry.
# No RNG and no row or file order ever enters here.
_SORT_KEYS: dict[WithinGroupOrder, list[tuple[str, str]]] = {
    "farthest": [
        (DISTANCE_COLUMN, "descending"),
        (KEY_COLUMN, "ascending"),
        (FRAGMENT_COLUMN, "ascending"),
    ],
    "nearest": [
        (DISTANCE_COLUMN, "ascending"),
        (KEY_COLUMN, "ascending"),
        (FRAGMENT_COLUMN, "ascending"),
    ],
    "neutral": [(KEY_COLUMN, "ascending"), (FRAGMENT_COLUMN, "ascending")],
}


def _one_level2_key(group: pa.Table) -> Level2Key:
    """Return the group's single ``(canonical task, subtask cell)`` key.

    Guaranteed by grouping on both columns, checked anyway because a mixed group
    fails SILENTLY: row 0 would name one member's key, and every other member's
    rows would then be ranked against - and written under - a quota that was
    counted for a group they are not in.

    Raises:
        ValueError: If either column names anything other than exactly one value.

    """
    task = pc.unique(group.column(CANONICAL_TASK_COLUMN))  # type: ignore[attr-defined]
    cluster = pc.unique(group.column(SUBTASK_CLUSTER_COLUMN))  # type: ignore[attr-defined]
    if len(task) != 1 or len(cluster) != 1:
        msg = (
            f"one fairness group must name exactly one ({CANONICAL_TASK_COLUMN}, {SUBTASK_CLUSTER_COLUMN}) "
            f"pair, got {task.to_pylist()} x {cluster.to_pylist()}"
        )
        raise ValueError(msg)
    return (task[0].as_py(), int(cluster[0].as_py()))


def _verdict_row(group: pa.Table, reasons: Sequence[str | None]) -> pa.Table:
    """Project one reasoned group onto ``VERDICT_ROW``, dropping labels and distance.

    Fairness is the last stage before the write, so it emits exactly the row the
    write reads: the canonical labels and the distance have no consumer past this
    point and would only widen the final shuffle.
    """
    return pa.table(
        {
            KEY_COLUMN: group.column(KEY_COLUMN),
            FRAGMENT_COLUMN: group.column(FRAGMENT_COLUMN),
            DEDUP_KEY_COLUMN: group.column(DEDUP_KEY_COLUMN),
            CURATE_SELECTION_REASON: pa.array(reasons, type=pa.string()),
        },
        schema=VERDICT_ROW,
    )


def select_within_quota(
    group: pa.Table,
    quotas: Mapping[Level2Key, int],
    order: WithinGroupOrder,
) -> pa.Table:
    """Reason one fairness group's rows against its quota, one ``VERDICT_ROW`` per input row.

    Invoked once per merged ``(task, cluster)`` key. A row that already carries a reason - a duplicate,
    or one whose embedding was invalid - passes through untouched and is excluded from the ranking, so
    it can neither win a place nor displace a survivor. The rest are ordered by ``order``, the first
    ``quota`` ``selected`` and the remainder ``below_quota``; unfunded groups make every survivor
    ``unfunded``.

    Args:
        group: One group's post-dedup rows, all sharing one merged key.
        quotas: The broadcast group -> quota map for the whole run.
        order: The within-group ordering mode.

    Raises:
        ValueError: If a group holding survivors names more than one key, or is absent from
            ``quotas`` so count and cut disagree.

    """
    # A plain module-level function because Ray reads fn.__name__ for the operator
    # label, taking the whole quota map rather than one group's quota because that
    # map is passed unchanged to every invocation.
    existing = group.column(CURATE_SELECTION_REASON).to_pylist()
    survivor = np.fromiter((reason is None for reason in existing), dtype=bool, count=group.num_rows)
    survivor_count = int(survivor.sum())
    if survivor_count == 0:
        return _verdict_row(group, existing)

    key = _one_level2_key(group)
    quota = quotas.get(key)
    if quota is None:
        msg = f"group {key!r} holds {survivor_count} survivor(s) but was never counted; count and cut disagree"
        raise ValueError(msg)

    assigned = np.full(
        group.num_rows,
        str(CurateReason.UNFUNDED if quota == 0 else CurateReason.BELOW_QUOTA),
        dtype=object,
    )
    if quota > 0:
        # Rank the whole group once, then keep only the survivor positions: the
        # already-reasoned rows are dropped AFTER the sort, so their presence
        # cannot shift which survivor wins.
        ranked = pc.sort_indices(group, sort_keys=_SORT_KEYS[order]).to_numpy()  # type: ignore[attr-defined]
        assigned[ranked[survivor[ranked]][:quota]] = str(CurateReason.SELECTED)

    return _verdict_row(
        group,
        [
            reason if reason is not None else verdict
            for reason, verdict in zip(existing, assigned.tolist(), strict=True)
        ],
    )
