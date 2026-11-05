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

"""A synthetic wide ``clips.lance`` builder, shared by every Curate test.

Curate reads and writes ONE table, so almost every question about it is a
question about the shape of that table: how many fragments it has, which
embedding groups exist, which rows carry a vector, whether ``clip_id`` is really
unique, and which vectors are pathological. ``ClipsTableSpec`` makes each of
those an argument, so a test states the shape it needs in one line and asserts
against a table it fully owns.

::

    ClipsTableSpec(fragments=3, action=GroupState.NULL, ...)
        |
        v
    per fragment: base columns generated from CLIP_SCHEMA
        |         + the enabled embedding groups' columns
        v
    write_clips_table   one lance write per fragment, so the fragment
        |               count and the row-to-fragment mapping are exact
        v
    ClipsTable(uri, clip_ids per fragment)

The base columns are GENERATED from the producer's own ``CLIP_SCHEMA`` rather
than hand-listed, so the fixture stays faithful when that schema gains a field
and fails loudly - on an unhandled Arrow type - rather than silently drifting
narrow. Only the identity and label columns (``clip_id``, ``task_name``,
``subtask_name``) get meaningful values; the rest are filled from their type,
because no Curate behaviour reads them and asserting on generated values would
couple a test to this file instead of to a mechanism. ``subtask_name`` is
meaningful even though Curate no longer reads it, because the producer writes it
and a fixture that dropped it would stop being a faithful source table.

Four generated properties are deliberate and load-bearing, because getting any of
them wrong makes a whole class of test unable to fail:

- Every vector is drawn from a generator seeded on its own ``(row, column name)``
  pair, so two runs of one spec give byte-identical tables while no two vectors
  in the table are equal. Seeding on the row alone would give every modality of a
  row the same direction, and swapping or reweighting two equal blocks is a
  no-op; seeding on a column's POSITION would do the same, since all three groups
  number their vector field 0. The only intended source of an identical pair is
  ``duplicate_vectors``.
- The ``(task_name, subtask_name)`` pairs come from an explicit cycle that makes
  the three fairness groups uneven, in a 1:2:3 ratio. Two independent moduli look
  varied and are exactly uniform, and a uniform population never asks a group for
  more than it holds, so the water-fill's redistribution branch is unreachable.
- Those pairs are RAW annotation spellings, not canonical keys: several wordings
  of one instruction differ in case, interior whitespace, Unicode composition or a
  trailing mark. A fixture already in canonical form makes the pipeline's fold an
  identity, so every downstream assertion would hold with the fold deleted.
  ``CANONICAL_TASKS`` names the task keys they collapse to.
- The subtask text vectors lie in ``SUBTASK_REGIONS`` tight bundles that CROSS the
  subtask label partition (see ``subtask_region``). The level-2 fairness key is a
  k-means cell over that block, so the shape of the block is the only thing that
  can make the key observable. Both natural shapes hide it: per-row directions in
  a high-dimensional space put every row in its own cell, and one shared direction
  puts every row in one cell. Either way substituting the cell for the label
  changes nothing downstream, no single assertion is at fault, and a level-2 key
  computed from anything at all still passes.

``ClipsTableSpec`` refuses any knob it could not honour - an index past the end of
the table, a pair naming one row twice, a duplicate pair or an unusable vector
placed where the vectors would read NULL - because a fixture that quietly drops a
configured property leaves the test that asked for it passing on nothing.

``run_child`` is here for the same reason the table builder is: several modules
assert their own import purity, and the only honest way to check it is in an
interpreter this session has not already imported them into. ``POISONED_IMPORTS``,
``poisoning`` and ``assert_poisoning_fired`` are the harness those tests run
under; the latter two are plain functions rather than fixtures because neither
holds per-test state, one being a string builder and the other an assertion.
"""

import enum
import os
import pathlib
import subprocess
import sys
import textwrap
import zlib
from collections.abc import Callable, Collection, Mapping
from typing import Protocol

import attrs
import lance
import numpy as np
import pyarrow as pa
import pytest

from cosmos_curator.next.embeddings.schemas import (
    ACTION_COLUMN_GROUP,
    IMAGE_COLUMN_GROUP,
    KEY_COLUMN,
    TEXT_COLUMN_GROUP,
    EmbeddingColumnGroup,
)
from cosmos_curator.next.recipes.robot_action_split.contracts import CLIP_SCHEMA
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# The producer's free-form subtask label. Named here rather than imported from
# Curate because Curate has no name for it: the level-2 key is a partition of the
# subtask EMBEDDING, so the prose is only ever an input column of the source table
# this fixture writes. Public so a test can assert that Curate does not read it.
SUBTASK_PROSE_COLUMN: str = "subtask_name"

_LABEL_COLUMNS = ("task_name", SUBTASK_PROSE_COLUMN)
_ERROR_COLUMNS = ("error_stage", "error_message")

_ROWS_FOR_INTRA_DUPLICATE = 2


class RunChild(Protocol):
    """What the ``run_child`` fixture hands a test.

    A protocol rather than a ``Callable`` alias so the optional environment
    overlay stays typed; several modules take this, and one spelling of the
    signature is the point.
    """

    def __call__(self, code: str, *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """Run ``code`` in a fresh interpreter, with ``env`` overlaid on this process's."""
        ...


POISONED_IMPORTS: tuple[str, ...] = ("ray", "cuml", "cudf", "cupy", "lance", "pylance")
"""Driver-only dependencies every import-purity test blocks.

One spelling for every purity test and its control, which is what a shared
constant delivers: each control validates the very harness its own purity test
runs under, rather than a second set that happens to look like it.

``pylance`` is the DISTRIBUTION name, whose import name is ``lance`` - already
in the set - so poisoning it can never fire and five of the six names carry the
guarantee. Kept rather than dropped because removing it narrows a pinned literal,
which is a decision the pin exists to force someone to make deliberately.

It does NOT protect the set's CONTENTS. Dropping a name leaves every test green
while the purity guarantee quietly covers one dependency fewer, so the contents
are pinned against a literal in ``test_writeback.py``.
"""


def poisoning(module: str, *modules: str) -> str:
    """Return a ``run_child`` program that blocks every driver dependency, then imports ``modules``.

    ``sys.modules[name] = None`` does not REMOVE a module - it makes a later
    ``import name`` RAISE ``ModuleNotFoundError: import of <name> halted; None in
    sys.modules``. A control therefore asserts that message and never a
    dependency name: the message names nothing, so it neither pins which of the
    six raises first - import order decides, and a re-sort moves it - nor can be
    produced by an unrelated import-time crash. A name is a short token matched
    against a whole traceback, where ``lance`` appears in the frame paths of
    ``lance_utils.py`` and ``ray`` is a substring of ``array``.

    Shared rather than copied because three files carried drifted copies, which is
    how a vacuous-pass control survived being fixed in one of them.

    Args:
        module: First module to import, separate from ``modules`` so an
            argument-less call raises instead of yielding a purity test that
            imports nothing and passes.
        modules: Further modules to import in the same child.

    """
    prelude = textwrap.dedent(
        f"""
        import sys
        for name in {POISONED_IMPORTS!r}:
            sys.modules[name] = None
        """
    )
    # Only source TEXT is built here; the child runs it through "python -c" and
    # never imports this module, which is why a conftest whose whole job is to
    # block these dependencies may itself import them at the top of the file.
    #
    # dedent covers the literal above and nothing else: the joined import lines
    # start at column zero, so passing them through dedent would read as a step
    # that strips indentation while being incapable of changing anything.
    return prelude + "\n".join(f"import {name}  # noqa: F401" for name in (module, *modules))


def assert_poisoning_fired(result: subprocess.CompletedProcess[str]) -> None:
    """Assert a child died from the poisoning and not from something else.

    Owns the message because ``poisoning`` owns the mechanism that emits it: the
    two are one contract, and a per-file copy of this assertion is exactly what
    let a weaker form of it - a substring match over the poisoned NAMES - survive
    in two of three files after being repaired in the third.
    """
    assert result.returncode != 0, result.stdout
    assert "halted; None in sys.modules" in result.stderr, result.stderr


# The canonical fairness keys LABEL_CYCLE below folds onto. Public because every
# label a test can observe downstream of the scan is a CANONICAL label, so an
# assertion should read the fixture's contract instead of restating a literal that
# the raw cycle no longer contains. Only the TASK level has such keys: the level-2
# key is a partition of the subtask embedding, so no canonical subtask string
# exists downstream to name.
CANONICAL_TASKS: tuple[str, str] = ("plate the cr\u00eape", "stack the plates")

# How many bundles the subtask text vectors are drawn from, and how far a row is
# allowed to sit from its bundle's centre.
#
# Three is the smallest count that makes the level-2 partition tell a story: it is
# COARSER than the six raw subtask spellings, FINER than nothing, and - because
# 3 does not divide the label cycle's period of 6 - it is INCOMPARABLE with the
# subtask label partition, so a cell holds rows of two different labels and one
# label spans two cells. A level-2 key that read the label instead of the geometry
# therefore produces a visibly different partition rather than the same one.
#
# The jitter scale sets the intra-bundle cosine to about 1 / (1 + s^2) = 0.9,
# which is close enough that k-means recovers the bundles at any k >= 3 and far
# enough that no two rows of a bundle are near-duplicates of each other: the
# default dedup_eps of 0.01 needs 0.99, and the subtask block carries only 0.6 of
# the fused distance, so the pair is scored around 0.54.
SUBTASK_REGIONS: int = 3
_SUBTASK_JITTER: float = 1.0 / 3.0

# Namespace for the region seeds, so a region index cannot collide with the row
# index that seeds the jitter and make a row's vector parallel to its own bundle.
_SUBTASK_REGION_SALT: int = zlib.crc32(b"subtask-region")

# The raw (task_name, subtask_name) pairs rows cycle through. Two properties are
# load-bearing:
#
# - The pairs are deliberately UNEVEN, so the fairness groups stand in a 1:2:3
#   ratio (see _label_value).
# - Every entry is spelled the way an annotator would write it, NOT pre-folded, so
#   canonicalization has something to do. Handing the pipeline labels already in
#   canonical form makes the fold an identity, and a test over its output then
#   holds whatever the fold does - including nothing at all. Between them the six
#   entries cover all four steps: NFC (row 2's decomposed circumflex), whitespace
#   collapse (row 1's doubled space, row 2's tab, row 3's newline), casefold (rows
#   0, 1 and 4) and the trailing-mark strip (rows 0, 1, 3 and 4). Row 5 is already
#   canonical, so the fold is exercised as idempotent as well as effective.
#
# All six spellings are distinct, so a bypassed fold yields six task groups where
# the canonical fold yields two.
LABEL_CYCLE: tuple[tuple[str, str], ...] = (
    ("Plate The Cr\u00eape", "Place it on the Scale."),
    ("plate the  cr\u00eape ", "Wipe The Rim  "),
    ("plate the cre\u0302pe", "wipe\tthe rim"),
    ("stack the plates!", "place it on\nthe scale"),
    ("STACK THE PLATES", "place it on the scale ;"),
    ("stack the plates", "place it on the scale"),
)


class GroupState(enum.StrEnum):
    """Whether one modality's column group exists on the table, and is filled.

    ``ABSENT`` is the state of a table the embeddings leg never ran for: the
    columns do not exist at all. ``NULL`` is a group that was added but not
    filled. ``FILLED`` carries real vectors and provenance.
    """

    ABSENT = "absent"
    NULL = "null"
    FILLED = "filled"


@attrs.frozen
class ClipsTableSpec:
    """The shape of one synthetic clips table.

    Attributes:
        fragments: How many Lance fragments to write; each is one write.
        rows_per_fragment: Rows in every fragment.
        text: State of the text group (two vectors plus a model id).
        image: State of the image group.
        action: State of the action group.
        empty_fragments: Fragments whose rows read NULL for every vector even in
            a ``FILLED`` group, which is how a fragment holding no eligible row
            is built.
        duplicate_within_fragment: Fragment whose second row repeats its first
            row's ``clip_id``. Needs ``rows_per_fragment >= 2``.
        duplicate_across_fragments: Two fragments whose first rows share one
            ``clip_id``.
        duplicate_vectors: Two ROW indices given byte-identical vectors in every
            filled group, so the pair is a de-duplication duplicate at any
            ``eps``. Distinct ``clip_id``s, unlike the two knobs above.
        non_finite_vector_rows: Row indices whose first coordinate is ``inf``, so
            they must be routed past the similarity pass rather than allowed to
            propagate NaN across their cluster.
        zero_norm_vector_rows: Row indices whose vectors are all zeros, which no
            normalization can rescue and which therefore have no direction.

    """

    fragments: int = 3
    rows_per_fragment: int = 4
    text: GroupState = GroupState.FILLED
    image: GroupState = GroupState.FILLED
    action: GroupState = GroupState.FILLED
    empty_fragments: frozenset[int] = frozenset()
    duplicate_within_fragment: int | None = None
    duplicate_across_fragments: tuple[int, int] | None = None
    # Row indices below are GLOBAL - fragment-major, so row r of fragment f is
    # f * rows_per_fragment + r - because a caller placing a duplicate pair or a
    # broken vector cares whether the two rows share a fragment, and a global
    # index states that directly.
    duplicate_vectors: tuple[int, int] | None = None
    non_finite_vector_rows: frozenset[int] = frozenset()
    zero_norm_vector_rows: frozenset[int] = frozenset()

    def __attrs_post_init__(self) -> None:
        """Reject any knob the table could not actually honour.

        One rule, applied to every knob: a spec must never quietly fail to deliver
        a property it was configured with. An out-of-range index, two knobs that
        cancel each other, and a knob placed where no vector is written all end in
        a table missing what the caller asked for with nothing to say so, which is
        worse in a fixture than in production - the test still passes, having
        verified nothing.

        Cheapest fault first: an out-of-range index is reported before any
        combination check, so a spec wrong in two ways names the simpler one.

        Raises:
            ValueError: On an out-of-range index, a combination whose outcome would
                be ambiguous, or a knob whose property the table would not hold.

        """
        self._reject_out_of_range()
        if (
            self.duplicate_within_fragment is not None
            and self.duplicate_across_fragments is not None
            and self.duplicate_within_fragment in self.duplicate_across_fragments
        ):
            msg = (
                f"duplicate_within_fragment={self.duplicate_within_fragment} overlaps "
                f"duplicate_across_fragments={self.duplicate_across_fragments}; the second "
                f"knob would overwrite the first knob's row and neither pair would exist"
            )
            raise ValueError(msg)
        pathological = self.non_finite_vector_rows | self.zero_norm_vector_rows
        # Only the knobs actually set are named, so the message a caller reads - and
        # the string a test matches on - identifies which one it was.
        named = " / ".join(
            name
            for name, rows in (
                ("non_finite_vector_rows", self.non_finite_vector_rows),
                ("zero_norm_vector_rows", self.zero_norm_vector_rows),
            )
            if rows
        )
        self._reject_rows_whose_vectors_would_read_null(f"{named}={sorted(pathological)}", pathological)
        if self.duplicate_vectors is not None and pathological.intersection(self.duplicate_vectors):
            msg = (
                f"duplicate_vectors={self.duplicate_vectors} names a row that is also "
                f"pathological ({sorted(pathological)}); a broken vector cannot be half of a "
                f"duplicate pair, so the pair would silently not exist"
            )
            raise ValueError(msg)
        if self.duplicate_vectors is not None:
            self._reject_rows_whose_vectors_would_read_null(
                f"duplicate_vectors={self.duplicate_vectors}", self.duplicate_vectors
            )

    def _reject_rows_whose_vectors_would_read_null(self, knob: str, rows: Collection[int]) -> None:
        """Fail when a knob names rows the table would write NULL vectors for anyway.

        Two states erase a per-row vector before any knob can reach it: a row inside
        an empty fragment, and a table where no group is ``FILLED``. Either way the
        row exists and every one of its vectors reads NULL, so the property the
        caller configured - an identical pair, an unusable vector - is simply not in
        the table, and the test asking for it passes having covered nothing.

        Args:
            knob: The knob and its value, already rendered, to name in the message.
            rows: Global row indices that knob placed; empty means nothing to check,
                including the group states, which no unset knob has an opinion on.

        """
        if not rows:
            return
        empty = {row // self.rows_per_fragment for row in rows} & self.empty_fragments
        if empty:
            msg = (
                f"{knob} names a row in empty fragment(s) {sorted(empty)}; its vectors "
                f"read NULL, so the requested property would silently not exist"
            )
            raise ValueError(msg)
        if not any(state is GroupState.FILLED for state in (self.text, self.image, self.action)):
            msg = (
                f"{knob} needs a filled group to be written into, but text={self.text}, "
                f"image={self.image}, and action={self.action}"
            )
            raise ValueError(msg)

    def _reject_out_of_range(self) -> None:
        """Fail on any index naming a fragment or row the table will not contain."""
        if self.fragments < 1 or self.rows_per_fragment < 1:
            msg = f"a table needs at least one row in one fragment, got {self.fragments}x{self.rows_per_fragment}"
            raise ValueError(msg)
        rows = self.fragments * self.rows_per_fragment
        fragment_knobs: dict[str, tuple[int, ...]] = {
            "empty_fragments": tuple(sorted(self.empty_fragments)),
            "duplicate_within_fragment": ()
            if self.duplicate_within_fragment is None
            else (self.duplicate_within_fragment,),
            "duplicate_across_fragments": self.duplicate_across_fragments or (),
        }
        for name, values in fragment_knobs.items():
            outside = [value for value in values if not 0 <= value < self.fragments]
            if outside:
                msg = f"{name}={values} names fragment(s) {outside} outside range(0, {self.fragments})"
                raise ValueError(msg)
        row_knobs: dict[str, tuple[int, ...]] = {
            "duplicate_vectors": self.duplicate_vectors or (),
            "non_finite_vector_rows": tuple(sorted(self.non_finite_vector_rows)),
            "zero_norm_vector_rows": tuple(sorted(self.zero_norm_vector_rows)),
        }
        for name, values in row_knobs.items():
            outside = [value for value in values if not 0 <= value < rows]
            if outside:
                msg = f"{name}={values} names global row(s) {outside} outside range(0, {rows})"
                raise ValueError(msg)
        # A pair of one names no pair. Caught here rather than left to produce a
        # table where one row is trivially "identical to itself".
        for name, pair in (
            ("duplicate_across_fragments", self.duplicate_across_fragments),
            ("duplicate_vectors", self.duplicate_vectors),
        ):
            if pair is not None and pair[0] == pair[1]:
                msg = f"{name}={pair} names one index twice; a pair needs two distinct members"
                raise ValueError(msg)
        if self.duplicate_within_fragment is not None and self.rows_per_fragment < _ROWS_FOR_INTRA_DUPLICATE:
            msg = (
                f"duplicate_within_fragment={self.duplicate_within_fragment} repeats a fragment's "
                f"first clip_id on its second row, so rows_per_fragment must be at least "
                f"{_ROWS_FOR_INTRA_DUPLICATE}, got {self.rows_per_fragment}"
            )
            raise ValueError(msg)


@attrs.frozen
class ClipsTable:
    """A written synthetic clips table and the row identities it holds.

    Attributes:
        uri: Path the table was written to.
        spec: The spec it was built from.
        clip_ids: Per fragment, its ``clip_id`` values in row order. Under a
            duplicate knob two entries repeat, which is the point.

    """

    uri: str
    spec: ClipsTableSpec
    clip_ids: tuple[tuple[str, ...], ...]

    @property
    def all_clip_ids(self) -> tuple[str, ...]:
        """Return every ``clip_id`` in fragment then row order."""
        return tuple(clip_id for fragment in self.clip_ids for clip_id in fragment)

    def clip_id_at(self, row: int) -> str:
        """Return the ``clip_id`` at one global row index.

        Lets a caller that placed a duplicate pair or a broken vector by row
        index name the resulting clip without recomputing the layout.
        """
        return self.all_clip_ids[row]


def _base_value(field: pa.Field, row: int) -> str | int | float:
    """Return a deterministic value for one generated base column.

    Raises:
        TypeError: On an Arrow type this builder has no rule for, so a schema
            change surfaces here instead of as an opaque Arrow cast failure.

    """
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        return f"{field.name}-{row:04d}"
    if pa.types.is_integer(field.type):
        return row
    if pa.types.is_floating(field.type):
        return float(row)
    msg = f"clips fixture has no value rule for column {field.name!r} of type {field.type}"
    raise TypeError(msg)


def _vector(dim: int, *, row: int, field: str) -> list[float]:
    """Return the vector belonging to one ``(row, column)`` pair.

    Seeded on the pair rather than drawn from a shared stream, which is what
    makes the table byte-identical across runs while keeping every vector in it
    distinct. Both halves of the seed are load-bearing. Seeding on the row alone
    would make every modality of a row carry the SAME direction, which silently
    un-falsifies any test asserting that the fused block order or the per-block
    weights change the result - swapping or reweighting two equal blocks is a
    no-op. Seeding on the column alone would make every row identical.

    The column half is the name, not its position in the group: three groups each
    number their own vector field 0, so positions collide across groups exactly
    where names cannot.
    """
    return np.random.default_rng([row, zlib.crc32(field.encode())]).standard_normal(dim).tolist()


def subtask_region(row: int) -> int:
    """Return which subtask-text bundle one global row's vector is drawn from.

    Public because it is the fixture's level-2 contract: two rows share a subtask
    cluster cell exactly when they share a region, so a test asserts against this
    rather than against a cell id that k-means is free to number as it likes.
    """
    return row % SUBTASK_REGIONS


def _subtask_vector(dim: int, *, row: int, field: str) -> list[float]:
    """Return one row's subtask text vector: its region's direction, plus jitter.

    The jitter is what keeps every vector in the table distinct while leaving the
    region recoverable by k-means; see ``SUBTASK_REGIONS``.
    """
    salted = [_SUBTASK_REGION_SALT, subtask_region(row), zlib.crc32(field.encode())]
    base = np.random.default_rng(salted).standard_normal(dim)
    jitter = np.random.default_rng([row, zlib.crc32(field.encode())]).standard_normal(dim)
    return (base + _SUBTASK_JITTER * jitter).tolist()


def _field_vector(dim: int, *, row: int, field: str) -> list[float]:
    """Return the vector one ``(row, column)`` pair carries, before the pathology knobs.

    Only the subtask text vector is bundled; every other column stays per-row, so
    the fused distance keeps distinguishing rows that share a subtask region.
    """
    if field == TEXT_COLUMN_GROUP.primary_vector:
        return _subtask_vector(dim, row=row, field=field)
    return _vector(dim, row=row, field=field)


def _row_vector(spec: ClipsTableSpec, *, dim: int, field: str, row: int) -> list[float]:
    """Return one row's vector for one field, applying the pathology knobs.

    Order matters: a zero-norm or non-finite row wins over a duplicate pairing,
    so a caller cannot accidentally build a row that is both a duplicate and
    unusable and then be surprised by which property the pipeline reports.
    ``ClipsTableSpec`` rejects that combination up front, so the precedence here
    is a safety net rather than the contract.
    """
    if row in spec.zero_norm_vector_rows:
        return [0.0] * dim
    if row in spec.non_finite_vector_rows:
        return [float("inf"), *_field_vector(dim, row=row, field=field)[1:]]
    if spec.duplicate_vectors is not None and row == spec.duplicate_vectors[1]:
        # Its partner's ROW under this column's own name, so the pair is
        # byte-identical in every filled group and their cosine similarity is
        # exactly 1, while the two rows stay distinct from every other row. Going
        # through the row rather than through a raw seed is what keeps that true
        # for the bundled subtask vector, whose region is a function of the row.
        return _field_vector(dim, row=spec.duplicate_vectors[0], field=field)
    return _field_vector(dim, row=row, field=field)


def _group_columns(
    group: EmbeddingColumnGroup, spec: ClipsTableSpec, *, rows: int, first_row: int, filled: bool
) -> dict[str, pa.Array]:
    """Build one group's columns for one fragment, all-filled or all-NULL.

    Every field of a group transitions together, so a row is either complete
    across the group or empty across it; a half-filled row is not a state the
    producer can create and is therefore not a state this builder offers.
    """
    columns: dict[str, pa.Array] = {}
    for field in group.schema:
        if not filled:
            columns[field.name] = pa.nulls(rows, type=field.type)
        elif pa.types.is_fixed_size_list(field.type):
            width = field.type.list_size
            values = [_row_vector(spec, dim=width, field=field.name, row=first_row + row) for row in range(rows)]
            columns[field.name] = pa.array(values, type=field.type)
        else:
            columns[field.name] = pa.array([f"{field.name}-fixture"] * rows, type=field.type)
    return columns


def _label_value(column: str, row: int) -> str:
    """Return one row's ``task_name`` or ``subtask_name``, forming UNEVEN fairness groups.

    A fairness fixture is only interesting when the groups differ in size: a
    uniform population never asks a group for more than it can supply, so the
    water-fill never reaches the branch that redistributes an underfunded group's
    quota. Independent moduli do NOT give that - ``row % 2`` against ``row % 3``
    cycles with period 6 and makes all six pairs exactly equal - so the pairs are
    assigned from an explicit cycle instead, in a 1:2:3 ratio.

    The returned value is RAW, in the non-canonical spelling ``LABEL_CYCLE`` gives
    it; the canonical task key it folds onto is in ``CANONICAL_TASKS``. Nothing
    folds ``subtask_name``: it is written because the producer writes it, and
    Curate no longer reads it.
    """
    task, subtask = LABEL_CYCLE[row % len(LABEL_CYCLE)]
    return task if column == "task_name" else subtask


def _fragment_clip_ids(spec: ClipsTableSpec) -> tuple[tuple[str, ...], ...]:
    """Return the per-fragment ``clip_id`` values, applying both duplicate knobs."""
    ids = [
        [f"clip-{fragment:02d}-{row:02d}" for row in range(spec.rows_per_fragment)]
        for fragment in range(spec.fragments)
    ]
    if spec.duplicate_within_fragment is not None:
        fragment = ids[spec.duplicate_within_fragment]
        fragment[1] = fragment[0]
    if spec.duplicate_across_fragments is not None:
        source, target = spec.duplicate_across_fragments
        ids[target][0] = ids[source][0]
    return tuple(tuple(fragment) for fragment in ids)


def _fragment_table(
    spec: ClipsTableSpec, clip_ids: tuple[str, ...], *, first_row: int, filled: bool
) -> tuple[pa.Table, pa.Schema]:
    """Build one fragment's table and the schema every fragment shares."""
    columns: dict[str, pa.Array] = {}
    fields: list[pa.Field] = []
    for field in CLIP_SCHEMA:
        fields.append(field)
        if field.name == KEY_COLUMN:
            columns[field.name] = pa.array(list(clip_ids), type=field.type)
        elif field.name in _ERROR_COLUMNS:
            # Every fixture clip was produced; a non-null error_stage would mark it failed.
            columns[field.name] = pa.nulls(len(clip_ids), type=field.type)
        elif field.name == "clip_uri":
            columns[field.name] = pa.array(
                [f"s3://fixture/clips/{clip_id}.mp4" for clip_id in clip_ids],
                type=field.type,
            )
        elif field.name == "action_data_uri":
            columns[field.name] = pa.array(
                [f"s3://fixture/actions/{clip_id}.bin" for clip_id in clip_ids],
                type=field.type,
            )
        elif field.name in _LABEL_COLUMNS:
            values = [_label_value(field.name, first_row + row) for row in range(len(clip_ids))]
            columns[field.name] = pa.array(values, type=field.type)
        else:
            values = [_base_value(field, first_row + row) for row in range(len(clip_ids))]
            columns[field.name] = pa.array(values, type=field.type)
    for group, state in (
        (TEXT_COLUMN_GROUP, spec.text),
        (IMAGE_COLUMN_GROUP, spec.image),
        (ACTION_COLUMN_GROUP, spec.action),
    ):
        if state is GroupState.ABSENT:
            continue
        fields.extend(group.schema)
        columns.update(
            _group_columns(
                group,
                spec,
                rows=len(clip_ids),
                first_row=first_row,
                filled=filled and state is GroupState.FILLED,
            )
        )
    schema = pa.schema(fields)
    return pa.table(columns, schema=schema), schema


def write_clips_table(path: pathlib.Path, spec: ClipsTableSpec) -> ClipsTable:
    """Write one synthetic clips table and return its identities.

    Each fragment is written by its own call, which is what makes the fragment
    count and the row-to-fragment mapping exact rather than a consequence of
    Lance's file sizing.

    Args:
        path: Destination directory for the Lance table.
        spec: The shape to build.

    Returns:
        The written table's URI and per-fragment ``clip_id`` values.

    """
    clip_ids = _fragment_clip_ids(spec)
    uri = str(path)
    for index, fragment_ids in enumerate(clip_ids):
        table, schema = _fragment_table(
            spec,
            fragment_ids,
            first_row=index * spec.rows_per_fragment,
            filled=index not in spec.empty_fragments,
        )
        lance.write_dataset(
            table,
            uri,
            schema=schema,
            mode="create" if index == 0 else "append",
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
    return ClipsTable(uri=uri, spec=spec, clip_ids=clip_ids)


@pytest.fixture
def run_child(repo_root: pathlib.Path) -> RunChild:
    """Return a runner executing one snippet in a fresh interpreter.

    The import-purity tests poison ``sys.modules`` to make a driver-only import
    fail, which a same-process run would leak into every test after it - and
    which would in any case prove nothing, since this session has already
    imported the modules under test. The ``env`` overlay serves the other reason
    to leave the process: an interpreter-level setting such as ``PYTHONHASHSEED``
    is fixed at startup and cannot be varied in-process at all.
    """

    def _run(code: str, *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
            cwd=repo_root,
            env=None if env is None else os.environ | dict(env),
        )

    return _run


@pytest.fixture
def build_clips_table(tmp_path: pathlib.Path) -> Callable[[ClipsTableSpec], ClipsTable]:
    """Return a builder writing a synthetic clips table under the test's tmp path."""
    counter = 0

    def _build(spec: ClipsTableSpec) -> ClipsTable:
        nonlocal counter
        counter += 1
        return write_clips_table(tmp_path / f"clips-{counter}.lance", spec)

    return _build
