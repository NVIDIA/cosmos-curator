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

"""Shared Lance utilities for Curator Next recipes.

Deliberately small: the things every Curator Next recipe needs from Lance before
it can decide anything - "does this table exist yet, how many rows does it hold,
and how is it laid out" - plus one bounded column read and the constants every
recipe must agree on because they describe the FORMAT rather than any one table:
the storage version each sink writes at, and Lance's row-address layout. Absence
is answerable either way:
``open_dataset`` returns ``None`` for a caller that treats a missing table as a
valid state, and ``open_dataset_or_raise`` raises for one that cannot proceed
without it. Everything table-shaped (schemas, key columns, what a vector means)
belongs to the owning recipe, not here; anything answerable from PyArrow alone
belongs in ``arrow_utils``, so that a caller inspecting a Parquet schema need not
import a Lance module to ask it.

``distinct_non_null_values`` is here rather than in one recipe because more than
one recipe needs to prove a column carries a single value, and a second
implementation of that read could answer differently. Which recipe asks, and what
it concludes, stays with the caller.

``report_table_shape`` is shared for the same reason and answers only the
physical question - how many fragments, how the rows sit across them. A recipe
whose unit of work is a fragment reads its own parallelism ceiling out of that;
one that lets Ray choose its block count reads nothing of the sort, so the
interpretation belongs at the call site and not here. Sharing it also writes one
rendering once, so two recipes cannot describe one table's geometry in two
shapes. It states the one fact this module *can* establish without knowing its
consumer: whether the fragments are small by Lance's own storage predicate.
Whether that is a problem, or whether they are too FEW, is a statement about the
reader's width, so no helper here makes it and none recommends an operation.

Write paths are NOT here on purpose. Each recipe owns its own commit semantics:
``video_cut`` and ``robot_action_split`` append fragments, the embeddings recipe
updates column groups in place. A shared "write a Lance table" helper would have
to encode one of those choices for all of them.
"""

import statistics
from typing import Literal

import attrs
import lance
import pyarrow as pa
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options

# The Lance data-storage version every Curator Next table is written at, so they
# all share one fragment granularity. Public because other sinks that write their
# own fragments must use the same value; a second definition there would let the
# two tables silently diverge.
#
# Independent of the video pipeline's CLIP_METADATA_LANCE_DATA_STORAGE_VERSION,
# which looks identical but is an enforced per-table read gate: that pipeline
# raises when an existing dataset's version differs, so sharing a symbol would
# turn a bump made here into a breaking read change there. Kept separate
# deliberately.
LANCE_DATA_STORAGE_VERSION: Literal["2.2"] = "2.2"

# Lance's own row-address layout, shared for the same reason as the storage
# version above: it is a property of the FORMAT, not of any one recipe's table, so
# a second definition elsewhere could disagree with the format rather than merely
# with its neighbour. A scan opts into the pseudo-column with
# ``with_row_address=True``; Lance then packs each address as
# ``(fragment_id << 32) | offset_within_fragment``.
#
# Exported as the two raw facts rather than as a decode helper: callers split an
# address in materially different ways - columnar via ``pc.shift_right``, per row
# via a Python shift, and one that also needs the offset - and a helper shaped for
# any one of them would be bypassed by the others.
ROW_ADDRESS_COLUMN: Literal["_rowaddr"] = "_rowaddr"
ROW_ADDRESS_FRAGMENT_SHIFT: Literal[32] = 32


def open_dataset(uri: str, *, storage_options: dict[str, str] | None = None) -> lance.LanceDataset | None:
    """Open a Lance dataset, returning ``None`` when it does not exist.

    Lance (9.x) exposes no dataset-exists predicate, so absence is classified by
    the error message: a "not found" / "does not exist" ``(OSError, ValueError)``
    reads as ``None`` and anything else re-raises. The message match cannot tell a
    permissions or bucket error whose text happens to contain "not found" apart
    from a genuinely absent table.
    """
    try:
        return lance.dataset(uri, storage_options=storage_options)
    except (OSError, ValueError) as exc:
        message = str(exc).lower()
        if "not found" in message or "does not exist" in message:
            return None
        raise


def open_dataset_or_raise(uri: str, *, storage_options: dict[str, str] | None = None) -> lance.LanceDataset:
    """Open a Lance dataset, raising ``ValueError`` when it is absent or unreadable.

    The raising counterpart to ``open_dataset``, for callers that cannot proceed
    without the table. A storage-level ``OSError`` is translated too, so one
    ``except ValueError`` covers both failure modes.

    Raises:
        ValueError: If no table exists at ``uri``, or it cannot be opened.

    """
    try:
        dataset = open_dataset(uri, storage_options=storage_options)
    except OSError as exc:
        msg = f"Lance table at {uri} could not be opened: {exc}"
        raise ValueError(msg) from exc
    if dataset is None:
        msg = f"Lance table not found at {uri}"
        raise ValueError(msg)
    return dataset


def distinct_non_null_values(
    dataset: lance.LanceDataset, column: str, *, max_values: int, where: str | None = None
) -> tuple[str, ...]:
    """Return an arbitrary bounded subset of a string column's distinct non-null values, sorted.

    Bounded so a caller can tell "one distinct value" from "more than one" by
    asking for two, with no value per row crossing into the driver. The result is
    whichever values Lance reached before the bound, ordered only for stable
    output: a caller wanting the smallest ``max_values`` must sort in the query.

    Lance resolves the distinctness and applies the bound, but must still visit
    every non-null value to prove only one is present - no fragment or dataset
    statistic it exposes answers that, and these columns carry no index. That
    pass is a projected scan of one encoded column.

    Args:
        dataset: The table to read.
        column: The string column whose distinct values are wanted.
        max_values: Upper bound on values returned; at least 1.
        where: Optional extra restriction, ANDed with the non-null condition.

            This is **DataFusion** SQL, not Lance's scanner-filter dialect: here
            double quotes are IDENTIFIERS and single quotes are LITERALS. That is
            the exact INVERSE of the scanner-filter surface
            (``count_rows(filter=...)``, ``scanner(filter=...)``), where a
            double-quoted name parses as a string literal and silently matches
            every row. A predicate written for one surface is wrong on the other,
            so do not pass a filter built for the scanner here.

    Returns:
        Up to ``max_values`` distinct non-null values, sorted.

    Raises:
        ValueError: If ``column`` is absent, not a string column, or
            ``max_values`` is below one.

    """
    if max_values < 1:
        msg = f"max_values must be at least 1 to read any value of {column!r}, got {max_values}"
        raise ValueError(msg)
    if column not in dataset.schema.names:
        msg = f"cannot read distinct values of {column!r}: {dataset.uri} has no such column"
        raise ValueError(msg)
    # Gated because the returned values are typed str: Lance yields a non-string
    # column's values as their own Python type, which would satisfy no caller and
    # no type checker while raising nothing.
    stored = dataset.schema.field(column).type
    if not (pa.types.is_string(stored) or pa.types.is_large_string(stored)):
        msg = f"cannot read distinct values of {column!r}: {dataset.uri} stores it as {stored}, not a string"
        raise ValueError(msg)
    # The column name is interpolated rather than quoted, which is safe because it
    # was just checked against the schema. Do NOT carry a quoting habit here from
    # the scanner filter surface (``count_rows(filter=...)``), where a
    # double-quoted name is read as a string literal and silently matches every
    # row: this statement goes through DataFusion, where double quotes are
    # identifiers and it is SINGLE quotes that would produce a literal.
    restriction = f"{column} IS NOT NULL" if where is None else f"{column} IS NOT NULL AND ({where})"
    statement = f"SELECT DISTINCT {column} FROM dataset WHERE {restriction} LIMIT {max_values}"  # noqa: S608
    batches = dataset.sql(statement).build().to_batch_records()
    return tuple(sorted(value for batch in batches for value in batch.column(0).to_pylist()))


def count_rows(uri: str, *, storage_profile: str = "default") -> int:
    """Return the row count of a Lance table, or 0 when it does not exist yet.

    Reads fragment metadata only (no column scan). A missing table is 0 while a
    real storage / I/O error propagates.
    """
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        return 0
    return int(dataset.count_rows())


# Lance's own default for ``write_dataset(max_rows_per_group=...)``, measured 1024 at
# pylance 13.0.0; the threshold tracks Lance's figure, not this literal, so
# re-measure if that default ever moves.
#
# Do NOT read it as a physical unit and do NOT retune it to whatever the writer's
# current unit is. The Lance file format dropped row groups at data storage version
# 2.0 and stores pages instead, and varying this knob over 64 / 1024 / 8192 leaves
# page count and per-page item counts unchanged at storage versions 2.0 / 2.1 / 2.2
# across int, wide-mixed, string, fixed-size-list and list schemas.
#
# What anchors the threshold is that Lance reads this same default as its own
# SMALL-FILE predicate: ``dataset.stats.dataset_stats()`` counts data files through
# ``num_small_files(max_rows_per_group)``, which is strictly rows < 1024 - the same
# boundary the comparison below draws. A fragment under it is one Lance's own tooling
# calls small, so its per-fragment costs - manifest entry, file open, footer read,
# scanner setup - are unamortized against a file the format itself considers too
# thin. That is why the threshold is anchored to a per-fragment ROW COUNT and not to
# a fragment COUNT: how many fragments is "too few" depends on the reader's
# parallelism, which a storage utility cannot know.
#
# Deliberately NOT Lance's fragment target (max_rows_per_file / compact_files
# target_rows_per_fragment, 1048576): this project's writers intentionally target
# thousands of rows per fragment, so that bound would warn on nearly every table we
# write, including the 8,000-row fragments whose problem is the opposite direction.
# That rejection is also why the emitted note names no remedy operation: the
# obvious one is compact_files, whose default target IS 1048576, so a bare call
# would collapse a table this note fires on into a single fragment - the opposite
# direction, and the one whose cost this layer cannot see.
_LANCE_SMALL_FILE_ROW_THRESHOLD = 1024


@attrs.define(frozen=True)
class LanceTableShape:
    """How a Lance table is physically laid out at one version.

    Fragment geometry, not content. A caller that hands one fragment to one unit
    of work reads ``num_fragments`` as its concurrency ceiling and
    ``max_fragment_rows`` as the bound on its last unit's runtime; whether that
    mapping holds is the caller's to know, so this type states the shape and
    never what it implies.

    ``num_rows`` is the sum of the per-fragment counts by construction, and every
    count is net of deletions, so the total agrees with ``count_rows()``.

    Attributes:
        version: The dataset version these counts describe.
        num_fragments: Fragment count.
        num_rows: Total rows, net of deletions.
        min_fragment_rows: Smallest per-fragment row count; 0 when there are no
            fragments, as are the other two distribution fields.
        median_fragment_rows: A per-fragment row count with as many fragments
            below it as above; always a count some fragment actually has.
        max_fragment_rows: Largest per-fragment row count.
        num_fields: Schema field count.

    """

    version: int
    num_fragments: int
    num_rows: int
    min_fragment_rows: int
    median_fragment_rows: int
    max_fragment_rows: int
    num_fields: int

    def as_message(self) -> str:
        """Return the one operator-facing line describing this shape.

        Owned here rather than composed per call site so two recipes cannot
        describe one table's geometry in two different shapes.
        """
        # Terse because the small-fragment note is appended to this line, and the
        # combined form has to stay readable in a terminal.
        head = f"v{self.version}: {self.num_rows:,} rows, {self.num_fragments:,} fragments"
        if not self.num_fragments:
            # The three distribution fields are 0 for an empty table, which would
            # otherwise render as a real reading of zero-row fragments.
            return f"{head}, {self.num_fields} fields"
        return (
            f"{head} (min={self.min_fragment_rows:,} median={self.median_fragment_rows:,} "
            f"max={self.max_fragment_rows:,} per fragment), {self.num_fields} fields"
        )

    @property
    def small_fragment_note(self) -> str | None:
        """Return where this table sits against Lance's small-file boundary, or ``None``.

        Keyed on the median so one small remainder fragment - the ordinary tail of
        any write - cannot trigger it; it takes at least half the fragments being
        under Lance's own small-file threshold.

        States the boundary and the cost in BOTH directions, and stops there. It
        issues no imperative and names no operation, because which direction is
        better for this table depends on the reader's width - the one thing this
        layer cannot see. A remedy phrased as a call would be acted on with its
        default target, and for a table small enough to trip this note that
        default collapses it to a single fragment, which is the fault this note
        cannot even detect. The reasoning behind the boundary lives at
        ``_LANCE_SMALL_FILE_ROW_THRESHOLD`` and in ``docs/``, not in a line
        emitted on every table open; the count and the median live in
        ``as_message()``, which this is appended to.
        """
        if self.num_fragments <= 1 or self.median_fragment_rows >= _LANCE_SMALL_FILE_ROW_THRESHOLD:
            return None
        return (
            f"median fragment under Lance's {_LANCE_SMALL_FILE_ROW_THRESHOLD:,}-row small-file boundary: "
            "smaller fragments leave per-fragment open and footer costs unamortized, "
            "larger ones cost a fragment-per-worker reader parallel units and lengthen its last-block tail"
        )


def report_table_shape(dataset: lance.LanceDataset, label: str) -> LanceTableShape:
    """Summarise a Lance table's fragment geometry from manifest metadata, log it, and return it.

    Reads no column data, so it is safe on the driver against remote object
    storage at any table size: it costs the manifest the handle already holds,
    and no request per fragment. Takes an open handle rather than a URI because
    every caller has one, and re-opening would pin a version the caller is not
    reading.

    One record rather than a summary line plus a conditional second one, so the
    distribution the note is drawn from always travels with it. Always INFO: the
    only judgement available here is that the median fragment sits under Lance's
    small-file boundary, which is a true statement about storage and not a fault,
    since whether the layout is wrong depends on the reader's width. *label* and
    every word of ``as_message()`` are identical with and without the note, so
    one filter matches both forms.

    Args:
        dataset: Open handle to the table to summarise.
        label: What this table is to the caller, e.g. ``"embedding source <uri>"``.
            Doubles as the grep anchor, so it should be stable per call site.

    Returns:
        The reported geometry, so a caller can act on it rather than re-read it.

    """
    # physical_rows less num_deletions, rather than fragment.count_rows(), because
    # a property pair cannot later acquire a filter argument - and count_rows WITH
    # a filter is a scan. Both properties resolve from the manifest: measured by
    # moving a table's data/ away while leaving _deletions/ in place, so a scan is
    # the only thing the absence can break, and re-reading them.
    rows_per_fragment = sorted(
        int(fragment.metadata.physical_rows) - int(fragment.metadata.num_deletions)
        for fragment in dataset.get_fragments()
    )
    shape = LanceTableShape(
        version=int(dataset.version),
        num_fragments=len(rows_per_fragment),
        # Summed rather than read from count_rows() so the total cannot disagree
        # with the distribution printed beside it.
        num_rows=sum(rows_per_fragment),
        # The three distribution fields stay 0 for a table with no fragments,
        # which as_message() renders as no distribution rather than as a reading
        # of zero-row fragments.
        min_fragment_rows=rows_per_fragment[0] if rows_per_fragment else 0,
        # median_low, not median: it returns an observed fragment size instead of
        # the midpoint of two, so an even fragment count yields a count some
        # fragment really has rather than a synthetic half-row figure.
        median_fragment_rows=statistics.median_low(rows_per_fragment) if rows_per_fragment else 0,
        max_fragment_rows=rows_per_fragment[-1] if rows_per_fragment else 0,
        num_fields=len(dataset.schema),
    )
    message = f"{label} {shape.as_message()}"
    if (note := shape.small_fragment_note) is not None:
        message = f"{message} - {note}"
    logger.info(message)
    return shape
