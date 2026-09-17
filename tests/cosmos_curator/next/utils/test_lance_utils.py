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

"""Read + version utilities: open_dataset classification, count_rows, distinct values.

``lance_utils`` is deliberately read-only, so this file is its whole suite: it
exercises the recipe-agnostic helpers against a local generic table, with no
embeddings schema involved. Each recipe's own write path is covered by that
recipe's tests, and the Arrow-only predicates by ``test_arrow_utils``.

The distinct-value read is tested here rather than in either recipe because both
consume it - the embeddings leg to refuse filling a group that already carries two
producer identities, the curation leg to refuse fusing one - so a behaviour change
must redden one shared file instead of depending on which recipe happens to cover
it.

The shape summary is fixtured with DELIBERATELY UNEVEN fragments (5 / 20 / 100
rows), so its minimum, median, maximum and mean are four different numbers and no
assertion below can be satisfied by the wrong statistic.
"""

import pathlib
import shutil
from collections.abc import Sequence
from typing import Any

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.robot_action_split import contracts, lance_sink
from cosmos_curator.next.utils import lance_utils
from cosmos_curator.next.utils.lance_utils import count_rows, distinct_non_null_values, report_table_shape

# Distinct values a producer-identity read asks for: one is the legal state, and
# the second exists only so "more than one" is distinguishable from it.
_PRODUCERS_INSPECTED = 2

_IDENTITY_COLUMN = "producer"

_KEEPER_COLUMN = "keeper"

# Rows per data file, so every multi-row table below spans several fragments and a
# read that stopped at the first one would be visibly wrong.
_ROWS_PER_FILE = 3


def _write_identities(path: pathlib.Path, values: Sequence[str | None]) -> lance.LanceDataset:
    """Write a one-column table of producer identities across several fragments."""
    uri = str(path)
    lance.write_dataset(
        pa.table({_IDENTITY_COLUMN: pa.array(list(values), type=pa.string())}),
        uri,
        max_rows_per_file=_ROWS_PER_FILE,
        data_storage_version=lance_utils.LANCE_DATA_STORAGE_VERSION,
    )
    return lance.dataset(uri)


def _write_restricted_identities(
    path: pathlib.Path, values: Sequence[str | None], keepers: Sequence[str | None]
) -> lance.LanceDataset:
    """Write producer identities alongside a second column a ``where`` clause can restrict on."""
    uri = str(path)
    lance.write_dataset(
        pa.table(
            {
                _IDENTITY_COLUMN: pa.array(list(values), type=pa.string()),
                _KEEPER_COLUMN: pa.array(list(keepers), type=pa.string()),
            }
        ),
        uri,
        max_rows_per_file=_ROWS_PER_FILE,
        data_storage_version=lance_utils.LANCE_DATA_STORAGE_VERSION,
    )
    return lance.dataset(uri)


def test_count_rows_is_zero_for_absent_table(tmp_path: pathlib.Path) -> None:
    """A table that does not exist yet counts as zero rows, not an error."""
    assert count_rows(str(tmp_path / "missing.lance")) == 0


def test_count_rows_returns_written_row_count(tmp_path: pathlib.Path) -> None:
    """An existing table reports its row count from fragment metadata."""
    uri = str(tmp_path / "t.lance")
    lance.write_dataset(pa.table({"clip_id": ["a", "b", "c"]}), uri, data_storage_version="2.2")
    assert count_rows(uri) == 3


def test_open_dataset_propagates_non_absence_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    """A permissions-style error is re-raised, not misread as an absent table.

    Absence is classified by message substring; this pins that an error whose text
    is not "not found" / "does not exist" propagates rather than reading as None.
    """

    def raise_permission_error(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        message = "Access Denied: caller lacks permission"
        raise OSError(message)

    monkeypatch.setattr(lance, "dataset", raise_permission_error)
    with pytest.raises(OSError, match="Access Denied"):
        lance_utils.open_dataset(str(tmp_path / "t.lance"))


def test_open_dataset_or_raise_raises_for_absent_table(tmp_path: pathlib.Path) -> None:
    """A caller that cannot proceed gets a ValueError (not None) when the table is absent.

    ``open_dataset`` returns None for absence; ``open_dataset_or_raise`` is the
    raising counterpart, so the absent path must surface as the documented
    ValueError rather than a None a caller could dereference.
    """
    with pytest.raises(ValueError, match="not found"):
        lance_utils.open_dataset_or_raise(str(tmp_path / "missing.lance"))


def test_open_dataset_or_raise_translates_unreadable_oserror_to_valueerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A non-absence storage OSError is translated to ValueError, so one ``except`` covers both failures.

    Unlike absence (which ``open_dataset`` maps to None), a permissions-style
    OSError propagates out of ``open_dataset``; the raising helper wraps it as
    ValueError so callers need not also catch OSError.
    """

    def raise_permission_error(*_args: object, **_kwargs: object) -> lance.LanceDataset:
        message = "Access Denied: caller lacks permission"
        raise OSError(message)

    monkeypatch.setattr(lance, "dataset", raise_permission_error)
    with pytest.raises(ValueError, match="could not be opened"):
        lance_utils.open_dataset_or_raise(str(tmp_path / "t.lance"))


def test_distinct_non_null_values_returns_the_single_identity(tmp_path: pathlib.Path) -> None:
    """A column filled by one producer reads back as exactly that one value."""
    dataset = _write_identities(tmp_path / "single.lance", ["fp-abc"] * 3)
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == ("fp-abc",)


def test_distinct_non_null_values_detects_an_identity_confined_to_the_final_row(tmp_path: pathlib.Path) -> None:
    """A second identity present in only the last row is still surfaced.

    This is the case that separates a bounded read from a truncated one: every
    fragment but the last carries a single identity, so the answer is only settled
    once the whole column has been considered. Stopping early here would hide
    exactly the mixed-producer corruption both callers exist to catch.
    """
    rows = 32
    dataset = _write_identities(tmp_path / "late.lance", ["fp-first"] * (rows - 1) + ["fp-last"])
    values = distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)
    assert values == ("fp-first", "fp-last")


def test_distinct_non_null_values_caps_the_result_at_max_values(tmp_path: pathlib.Path) -> None:
    """A column holding more identities than asked for yields exactly ``max_values``.

    What keeps the read bounded for a caller that only needs to tell one identity
    from several: a corpus column of arbitrary variety must not arrive in the
    driver row by row.

    The bound must be filled by DISTINCT values, so the set is asserted alongside
    the count. Both callers read "more than one value came back" as "more than one
    producer wrote this column", so a cap reached by the same identity twice would
    abort a run over a perfectly well formed table.

    Each identity is written twice in CONSECUTIVE rows for that assertion to be
    reachable, and the adjacency is the load-bearing half: it is what makes the
    first two rows a non-deduplicating read would return the same value. Over one
    row per identity - or the same nine pairs interleaved - such a read still hands
    back distinct values and the assertion can no longer fail.
    """
    dataset = _write_identities(tmp_path / "many.lance", [f"fp-{index}" for index in range(9) for _ in range(2)])
    values = distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)
    assert len(values) == _PRODUCERS_INSPECTED
    assert len(set(values)) == _PRODUCERS_INSPECTED


def test_distinct_non_null_values_ignores_null_rows(tmp_path: pathlib.Path) -> None:
    """Unfilled rows contribute nothing, so a partly filled column names one identity."""
    dataset = _write_identities(tmp_path / "partial.lance", ["fp-abc", None, "fp-abc", None])
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == ("fp-abc",)


def test_distinct_non_null_values_reads_an_identity_containing_a_quote(tmp_path: pathlib.Path) -> None:
    """An identity containing a single quote is read back verbatim.

    The read is expressed as a SQL statement, so a value able to terminate a
    string literal must never reach one. Were it to, a lone legal identity could
    be misread as two and abort a run that was in fact well formed.
    """
    quoted = "fp-o'brien"
    dataset = _write_identities(tmp_path / "quoted.lance", [quoted] * 6)
    assert distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED) == (quoted,)


def test_distinct_non_null_values_names_a_column_the_table_lacks(tmp_path: pathlib.Path) -> None:
    """An absent column raises a named error rather than a planner one.

    The name is interpolated into the statement, so the schema check is what
    stands in for quoting it; without the check the caller would get whatever
    DataFusion says about an unresolvable identifier.
    """
    dataset = _write_identities(tmp_path / "absent.lance", ["fp-abc"])
    with pytest.raises(ValueError, match="no such column"):
        distinct_non_null_values(dataset, "not_a_column", max_values=_PRODUCERS_INSPECTED)


def test_distinct_non_null_values_refuses_a_non_string_column(tmp_path: pathlib.Path) -> None:
    """A non-string column is refused, because its values would not be the promised str.

    Lance yields each value as its own Python type, so an int column would return
    ints under a ``tuple[str, ...]`` annotation - satisfying the type checker and
    no caller. Refusing is what keeps the annotation true at runtime.
    """
    uri = str(tmp_path / "int-identity.lance")
    lance.write_dataset(pa.table({_IDENTITY_COLUMN: pa.array([1, 2], type=pa.int64())}), uri)
    with pytest.raises(ValueError, match="not a string"):
        distinct_non_null_values(lance.dataset(uri), _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED)


def test_distinct_non_null_values_refuses_a_bound_that_reads_nothing(tmp_path: pathlib.Path) -> None:
    """A bound below one is a caller bug: it would report "unfilled" for any column.

    ``LIMIT 0`` is valid SQL returning no rows, so the read would succeed and the
    caller would conclude the column names no producer whatever it holds.
    """
    dataset = _write_identities(tmp_path / "zero-bound.lance", ["fp-abc"])
    with pytest.raises(ValueError, match="must be at least 1"):
        distinct_non_null_values(dataset, _IDENTITY_COLUMN, max_values=0)


def test_a_where_clause_narrows_which_rows_contribute_an_identity(tmp_path: pathlib.Path) -> None:
    """Only rows satisfying ``where`` are considered, so an excluded producer is not reported.

    What makes the read usable for a caller asking "which identities does THIS
    subset carry" - reading the whole column instead would report a producer that
    wrote rows the caller is not asking about.
    """
    dataset = _write_restricted_identities(
        tmp_path / "restricted.lance",
        ["fp-kept", "fp-dropped", "fp-kept", "fp-dropped"],
        ["yes", "no", "yes", "no"],
    )
    values = distinct_non_null_values(
        dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED, where=f"{_KEEPER_COLUMN} = 'yes'"
    )
    assert values == ("fp-kept",)


def test_a_double_quoted_name_in_where_is_an_identifier_not_a_literal(tmp_path: pathlib.Path) -> None:
    """A double-quoted name resolves the column here, the inverse of a Lance scanner filter.

    The two SQL surfaces Lance exposes disagree, and both accept the other's
    spelling silently. On a scanner filter (``count_rows(filter=...)``) a
    double-quoted name is a STRING LITERAL, so ``"keeper" IS NOT NULL`` is a
    constant that matches every row. Here the same text resolves the column, and a
    caller who ports a predicate between the two gets a wrong answer rather than
    an error. The fixture is what tells the two readings apart: were the name
    treated as a literal, the clause would be trivially true and the excluded
    producer would come back too.
    """
    dataset = _write_restricted_identities(
        tmp_path / "quoted-identifier.lance",
        ["fp-kept", "fp-dropped", "fp-kept", "fp-dropped"],
        ["present", None, "present", None],
    )
    values = distinct_non_null_values(
        dataset, _IDENTITY_COLUMN, max_values=_PRODUCERS_INSPECTED, where=f'"{_KEEPER_COLUMN}" IS NOT NULL'
    )
    assert values == ("fp-kept",)


def test_lance_sink_and_utils_share_one_storage_version() -> None:
    """The Extract sink writes at the version lance_utils owns, not a private copy.

    Executed coverage for the single-owner arrangement: robot_action_split's own
    suite skips entirely without a libopenh264 encoder, so it cannot gate this.
    """
    assert lance_sink.LANCE_DATA_STORAGE_VERSION == lance_utils.LANCE_DATA_STORAGE_VERSION


def test_contracts_no_longer_defines_a_storage_version() -> None:
    """The duplicate definition is gone, so the two next/ tables cannot disagree."""
    assert not hasattr(contracts, "LANCE_DATA_STORAGE_VERSION")


def _write_uneven_fragments(path: pathlib.Path, sizes: Sequence[int]) -> lance.LanceDataset:
    """Write a two-column table holding one fragment per entry of ``sizes``.

    One append per size, so the fragment row counts are exactly ``sizes`` and the
    dataset version is ``len(sizes)``.
    """
    uri = str(path)
    for index, rows in enumerate(sizes):
        lance.write_dataset(
            pa.table(
                {
                    "clip_id": pa.array([f"c{index}-{row}" for row in range(rows)], type=pa.string()),
                    "value": pa.array(range(rows), type=pa.int64()),
                }
            ),
            uri,
            mode="create" if index == 0 else "append",
            max_rows_per_file=max(sizes),
            data_storage_version=lance_utils.LANCE_DATA_STORAGE_VERSION,
        )
    return lance.dataset(uri)


def test_report_table_shape_reports_the_largest_fragment_not_the_mean(tmp_path: pathlib.Path) -> None:
    """The maximum is the largest fragment, which no average of these sizes equals.

    The largest fragment is what bounds the tail of a fragment-parallel read, so
    reporting a mean (41.66 here) in its place would describe a table that
    finishes in a third of the time the real one takes.
    """
    shape = report_table_shape(_write_uneven_fragments(tmp_path / "uneven.lance", (5, 100, 20)), "source")
    assert shape.max_fragment_rows == 100
    assert shape.min_fragment_rows == 5
    assert shape.max_fragment_rows != shape.num_rows // shape.num_fragments


def test_report_table_shape_median_is_a_size_some_fragment_actually_has(tmp_path: pathlib.Path) -> None:
    """An even fragment count yields an observed size, not the midpoint of two.

    Sizes 5 / 20 / 40 / 100 have a conventional median of 30, which no fragment
    holds; the reported 20 is a real fragment's row count.
    """
    shape = report_table_shape(_write_uneven_fragments(tmp_path / "even.lance", (5, 100, 20, 40)), "source")
    assert shape.median_fragment_rows == 20


def test_report_table_shape_counts_rows_net_of_deletions(tmp_path: pathlib.Path) -> None:
    """Deleted rows leave both the total and the distribution, matching count_rows.

    Fragment metadata carries physical rows, which still counts deleted ones, so
    an unadjusted read would over-report a compacted-but-not-rewritten table and
    disagree with the row count every other helper returns.
    """
    dataset = _write_uneven_fragments(tmp_path / "deleted.lance", (5, 100, 20))
    dataset.delete("value < 2")
    reopened = lance.dataset(str(tmp_path / "deleted.lance"))

    shape = report_table_shape(reopened, "source")

    assert shape.num_rows == reopened.count_rows() == 119
    assert shape.max_fragment_rows == 98
    assert shape.version == 4


# Deletions the payload-stripped reads below are fixtured with: distinct per
# fragment, and one fragment left untouched. A num_deletions that resolved to any
# constant - zero included - therefore yields a different distribution than the one
# asserted, so the assertion cannot be satisfied by a default standing in for a
# manifest read.
_DELETIONS_PER_FRAGMENT = (1, 2, 0, 3)
_ROWS_PER_STRIPPED_FRAGMENT = 100


def _reopen_without(tmp_path: pathlib.Path, payload: str) -> lance.LanceDataset:
    """Write the deletion-skewed table, move ONE payload directory aside, and reopen.

    Exactly one of ``data`` / ``_deletions`` goes per call, so a scan that then
    fails names the directory the caller took away and nothing else. That is what
    lets each control below attribute its own failure: removing both at once
    leaves a scan failing on the deletion file first, which can only establish
    that something is absent.

    The returned handle is constructed after the move and carries its own
    metadata cache, so it cannot answer from the pre-move handle's.
    """
    uri = tmp_path / "stripped.lance"
    dataset = _write_uneven_fragments(uri, (_ROWS_PER_STRIPPED_FRAGMENT,) * len(_DELETIONS_PER_FRAGMENT))
    doomed = ", ".join(
        f"'c{fragment}-{row}'" for fragment, deletions in enumerate(_DELETIONS_PER_FRAGMENT) for row in range(deletions)
    )
    dataset.delete(f"clip_id IN ({doomed})")
    shutil.move(str(uri / payload), str(tmp_path / f"stashed-{payload}"))
    return lance.dataset(str(uri))


def _assert_stripped_shape(shape: lance_utils.LanceTableShape) -> None:
    """Assert the five manifest facts of the deletion-skewed fixture.

    Both strips below assert the same five, because both make the same claim
    about where the numbers came from and differ only in which payload directory
    is absent.
    """
    # Four appends then one delete; clip_id and value.
    assert (shape.version, shape.num_fields, shape.num_fragments) == (5, 2, 4)
    # 100 rows per fragment less (1, 2, 0, 3) deletions, sorted: 97 / 98 / 99 / 100.
    assert shape.num_rows == 394
    assert (shape.min_fragment_rows, shape.median_fragment_rows, shape.max_fragment_rows) == (97, 98, 100)


def test_report_table_shape_reads_no_column_data(tmp_path: pathlib.Path) -> None:
    """Every shape field resolves with the data files - and only those - moved away.

    The helper runs on the driver at every table open, so a column read here
    would be a per-run cost against remote object storage. With the data alone
    gone, the control fails naming a missing path under ``data/``, and that is
    the reading which establishes no per-fragment data was touched.

    The control reads ``clip_id``, the column the delete predicate scanned while
    the fixture was built, so an in-process page cache able to serve a shape read
    would serve the control too and the control would not raise.
    """
    stripped = _reopen_without(tmp_path, "data")

    shape = report_table_shape(stripped, "source")

    _assert_stripped_shape(shape)
    # Non-vacuous: without this the test would also pass against a table whose
    # data files were never needed in the first place. Keyed on the data path
    # because a missing deletion file names ``_deletions/`` and never ``data/``.
    with pytest.raises(pa.ArrowInvalid, match=r"stripped\.lance/data/"):
        stripped.to_table(columns=["clip_id"])


def test_report_table_shape_reads_no_per_fragment_deletion_file(tmp_path: pathlib.Path) -> None:
    """num_deletions resolves from the manifest, not from the per-fragment deletion file.

    Lance writes one deletion file per fragment that has any, so reading those
    would be per-fragment I/O exactly as costly as reading the data - and the
    strip above leaves them in place, so it cannot rule that out. This is the
    other half of the same claim, split off rather than folded in so each
    control stays attributable to the directory its own test removed.
    """
    stripped = _reopen_without(tmp_path, "_deletions")

    shape = report_table_shape(stripped, "source")

    _assert_stripped_shape(shape)
    with pytest.raises(pa.ArrowInvalid, match=r"stripped\.lance/_deletions/"):
        stripped.to_table(columns=["clip_id"])


def test_report_table_shape_summarises_an_empty_table_without_a_distribution(tmp_path: pathlib.Path) -> None:
    """A table with no fragments reports no rows-per-fragment reading at all.

    The three distribution fields are 0 for an empty table, which rendered as a
    reading would claim fragments exist and hold nothing. Pinned whole rather
    than by the absence of the phrase "per fragment", which a line still
    carrying ``min=0 median=0 max=0`` would satisfy - the exact rendering this
    branch exists to suppress.
    """
    uri = str(tmp_path / "empty.lance")
    lance.write_dataset(
        pa.table({"clip_id": pa.array([], type=pa.string())}),
        uri,
        data_storage_version=lance_utils.LANCE_DATA_STORAGE_VERSION,
    )

    shape = report_table_shape(lance.dataset(uri), "source")

    # Guards the pin below against a fixture that quietly stopped being empty.
    assert shape.num_fragments == 0
    assert shape.as_message() == "v1: 0 rows, 0 fragments, 1 fields"


def test_as_message_renders_every_fact_an_operator_reads(tmp_path: pathlib.Path) -> None:
    """The rendered line is pinned whole, because operators read it and nothing else does.

    It is the only surface on which this shape reaches a human, so a dropped
    field or a re-ordered clause changes what a job log means without changing
    any other assertion in this file.
    """
    shape = report_table_shape(_write_uneven_fragments(tmp_path / "message.lance", (5, 100, 20)), "source")
    assert shape.as_message() == "v3: 125 rows, 3 fragments (min=5 median=20 max=100 per fragment), 2 fields"


def _shape(
    num_fragments: int,
    median_fragment_rows: int,
    *,
    min_fragment_rows: int | None = None,
    max_fragment_rows: int | None = None,
    num_rows: int | None = None,
) -> lance_utils.LanceTableShape:
    """Build a shape for the small-fragment rule, skewable away from its median.

    min, max, and the total default to a flat table only so a caller that does
    not care about skew stays short; every test about WHICH statistic the rule
    reads must set the relevant field apart, or it cannot tell the fields from
    one another. ``num_rows`` is the one that separates the median from the
    MEAN: left at its default the two coincide, so a rule reading
    ``num_rows / num_fragments`` would satisfy every assertion here.
    """
    return lance_utils.LanceTableShape(
        version=1,
        num_fragments=num_fragments,
        num_rows=num_fragments * median_fragment_rows if num_rows is None else num_rows,
        min_fragment_rows=median_fragment_rows if min_fragment_rows is None else min_fragment_rows,
        median_fragment_rows=median_fragment_rows,
        max_fragment_rows=median_fragment_rows if max_fragment_rows is None else max_fragment_rows,
        num_fields=4,
    )


def test_small_fragment_note_is_pinned_whole_for_a_table_below_lances_threshold() -> None:
    """The note is pinned literally, because an operator reads it and nothing else does.

    It is the only surface on which this rule reaches a human. Pinning it whole
    is what makes the three defects it was rewritten to shed - an imperative, a
    named remedy operation, and a verdict this layer cannot reach - unable to
    return without failing here.
    """
    assert _shape(num_fragments=400, median_fragment_rows=1023).small_fragment_note == (
        "median fragment under Lance's 1,024-row small-file boundary: "
        "smaller fragments leave per-fragment open and footer costs unamortized, "
        "larger ones cost a fragment-per-worker reader parallel units and lengthen its last-block tail"
    )


def test_small_fragment_note_is_silent_at_exactly_lances_small_file_threshold() -> None:
    """A fragment at the threshold is not small, matching Lance's own strict comparison.

    Pins the boundary rather than a value well clear of it: an inclusive
    comparison here would warn about tables Lance's ``num_small_files`` counts as
    healthy, since that predicate is strictly rows below the threshold.
    """
    assert _shape(num_fragments=400, median_fragment_rows=1024).small_fragment_note is None


def test_small_fragment_note_stays_silent_on_the_too_few_fragments_fault() -> None:
    """The incident table's shape draws no note from this helper.

    17 fragments for 131,602 rows throttled a 32-GPU run, but that is a claim
    about the reader's parallelism. A storage utility that guessed at it would
    have to invent a cluster width, so this rule must not fire here - the
    caption recipe owns that direction.
    """
    assert _shape(num_fragments=17, median_fragment_rows=8_000).small_fragment_note is None


def test_small_fragment_note_ignores_a_single_fragment_table() -> None:
    """One fragment holding a handful of rows has no distribution to read."""
    assert _shape(num_fragments=1, median_fragment_rows=3).small_fragment_note is None


def test_small_fragment_note_reads_the_median_not_the_largest_fragment() -> None:
    """One huge fragment does not excuse a table of tiny ones.

    Skewed so the median and the max fall on opposite sides of the threshold:
    keyed on the max this table looks healthy, which is the confusion that would
    let the pathology through. ``num_rows`` is the total those 400 fragments
    really hold, which puts the MEAN on the max's side of the threshold too, so
    a rule averaging instead of ordering also fails here.
    """
    skewed = _shape(
        num_fragments=400,
        median_fragment_rows=10,
        max_fragment_rows=5_000_000,
        num_rows=399 * 10 + 5_000_000,
    )

    assert skewed.small_fragment_note is not None


def test_small_fragment_note_tolerates_one_small_remainder_fragment() -> None:
    """A healthy table whose last fragment holds the write's remainder is left alone.

    Every write ends with a partial fragment, so keying on the min would warn
    about essentially every table in the project.
    """
    remainder = _shape(
        num_fragments=17,
        median_fragment_rows=8_000,
        min_fragment_rows=12,
        num_rows=16 * 8_000 + 12,
    )

    assert remainder.small_fragment_note is None


def test_small_fragment_note_reads_the_median_not_the_mean_fragment_size() -> None:
    """A table whose average fragment is thin stays silent while its median is not.

    The other skew tests order fragments; this one averages them. Two fragments
    at the threshold and one near-empty give a mean well under it, so a rule
    dividing ``num_rows`` by ``num_fragments`` reports a small-fragment table
    where Lance's own predicate counts only one small file.
    """
    averaged = _shape(
        num_fragments=3,
        median_fragment_rows=1_024,
        min_fragment_rows=1,
        max_fragment_rows=1_024,
        num_rows=1 + 1_024 + 1_024,
    )

    assert averaged.small_fragment_note is None


def test_report_table_shape_notes_small_fragments_at_info_and_keeps_the_numbers(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A small-fragment table is reported at INFO in one record carrying both facts.

    INFO rather than WARNING because the layer can establish only that the
    fragments are small by Lance's own predicate, never that the layout is
    wrong - the run that made this rule fire is the run refragmenting rescued.
    Asserting the level alone would pass for a record that dropped the
    distribution, and asserting the note alone would pass for a record that
    carried the observation without the numbers it is drawn from.
    """
    fragmented = _write_uneven_fragments(tmp_path / "fragmented.lance", (5, 100, 20))

    report_table_shape(fragmented, "embedding source s3://bucket/t")

    assert len(loguru_records) == 1
    assert loguru_records[0]["level"].name == "INFO"
    message = loguru_records[0]["message"]
    assert "1,024-row small-file boundary" in message
    assert "125 rows, 3 fragments" in message
    assert "median=20" in message


def test_report_table_shape_recommends_no_operation_to_the_operator(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The emitted line states facts and issues no instruction.

    The rule fired on a layout that was correct for its reader, so any imperative
    it carried argued against the fix. Naming the obvious remedy is the dangerous
    half: ``compact_files`` defaults to a fragment target three orders of
    magnitude above the boundary this note reports, so complying literally with
    a bare mention collapses such a table to one fragment - one input block, one
    reader, and a tail this layer cannot see because it does not know the reader.

    Scanning for the vocabulary rather than pinning the text, so a rewrite that
    reintroduces a recommendation in new words still fails.
    """
    fragmented = _write_uneven_fragments(tmp_path / "fragmented.lance", (5, 100, 20))

    shape = report_table_shape(fragmented, "embedding source s3://bucket/t")

    # Guarded on the rule having fired at all, so an empty note cannot satisfy
    # the scan below. Read off the shape rather than the text, so a rewording of
    # the note leaves this test failing on the vocabulary it exists to reject.
    assert shape.small_fragment_note is not None
    message = loguru_records[0]["message"].lower()
    for instruction in ("compact", "should", "must", "recommend", "consider", "rewrite", "over-fragment"):
        assert instruction not in message


def test_report_table_shape_makes_no_retracted_row_group_claim(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The boundary is attributed to Lance's small-file predicate, not to a row group.

    Lance dropped row groups at data storage version 2.0, so 1,024 cannot be
    described to an operator as a row-group size. It survives as a threshold only
    because Lance's own ``num_small_files`` predicate reads the same figure.
    """
    fragmented = _write_uneven_fragments(tmp_path / "fragmented.lance", (5, 100, 20))

    shape = report_table_shape(fragmented, "embedding source s3://bucket/t")

    assert shape.small_fragment_note is not None
    message = loguru_records[0]["message"].lower()
    assert "row group" not in message
    assert "row-group" not in message


def test_report_table_shape_appends_no_note_for_a_healthy_table(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A table clear of the boundary is reported at INFO with nothing appended.

    The negative control for the rule: without it, a note emitted
    unconditionally would satisfy every assertion above. Keyed on the boundary
    phrase, which only this rule emits.
    """
    healthy = _write_uneven_fragments(tmp_path / "healthy.lance", (5,))

    report_table_shape(healthy, "Caption source s3://bucket/t")

    assert len(loguru_records) == 1
    assert loguru_records[0]["level"].name == "INFO"
    assert "small-file boundary" not in loguru_records[0]["message"]


def test_report_table_shape_reports_one_record_per_open(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The note and the numbers arrive together, not as two records.

    A reader who greps out the note must not have to go find a separate line to
    learn the distribution it was drawn from.
    """
    fragmented = _write_uneven_fragments(tmp_path / "fragmented.lance", (5, 100, 20))

    report_table_shape(fragmented, "embedding source s3://bucket/t")

    assert [record["level"].name for record in loguru_records] == ["INFO"]


def test_report_table_shape_label_prefixes_both_forms(
    tmp_path: pathlib.Path,
    loguru_records: list[dict[str, Any]],
) -> None:
    """The label prefixes the record with and without the note, so one grep matches both.

    The note is a suffix rather than a rewrite of the line precisely so a filter
    anchored on the label keeps matching when a table's layout changes.
    """
    healthy = _write_uneven_fragments(tmp_path / "healthy.lance", (5,))
    fragmented = _write_uneven_fragments(tmp_path / "fragmented.lance", (5, 100, 20))

    report_table_shape(healthy, "embedding source s3://bucket/t")
    report_table_shape(fragmented, "embedding source s3://bucket/t")

    assert len(loguru_records) == 2
    assert all(record["message"].startswith("embedding source s3://bucket/t ") for record in loguru_records)
