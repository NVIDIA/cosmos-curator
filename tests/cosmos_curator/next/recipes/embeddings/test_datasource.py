# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owed-row scan: which fragments a run takes on, and the identity each row carries.

The read tasks are exercised by calling them directly rather than through
``ray.data.read_datasource``: the behaviours under test are the scan's own, and
starting a Ray cluster would make them slower to check without making them
better checked.

The row filters here are deliberately plain predicates over base columns rather
than the modality predicate ``columns.pending_filter`` builds. The datasource
takes the filter as an opaque string, so borrowing the real one would couple
these tests to a contract they do not exercise; what "owed" means is pinned in
``test_columns.py``. A run advancing past fragments it has finished is modelled
by NARROWING the filter between two plans, which is what finishing them does.
"""

import lance
import pytest

from cosmos_curator.next.recipes.embeddings.datasource import (
    EmbeddingScanDatasource,
    PinnedScan,
    plan_embedding_scan,
    read_owed_rows,
)
from cosmos_curator.next.recipes.embeddings.staged import CHECKPOINT_ID_COLUMN, FRAGMENT_ID_COLUMN
from cosmos_curator.next.utils.lance_utils import ROW_ADDRESS_FRAGMENT_SHIFT

from .conftest import ClipsTableFactory

# Six rows in fragments of two, so a cap of one leaves two fragments behind and
# an "advance past what is finished" assertion has somewhere to advance to.
_ROWS = 6
_ROWS_PER_FILE = 2
_FRAGMENT_COUNT = _ROWS // _ROWS_PER_FILE

_EVERYTHING = "clip_id IS NOT NULL"
_SOURCE_COLUMNS = ("clip_id", "clip_uri")


def _scan(uri: str, row_filter: str, *, batch_size: int = 2) -> PinnedScan:
    """Bind a scan to the table's current version with the standard projection."""
    return PinnedScan.for_dataset(
        lance.dataset(uri),
        storage_options=None,
        source_columns=_SOURCE_COLUMNS,
        row_filter=row_filter,
        batch_size=batch_size,
    )


def test_a_fragment_owing_nothing_is_left_out_of_the_plan(make_clips_table: ClipsTableFactory) -> None:
    """Only fragments holding an owed row are planned, so a resume reads nothing it finished."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)

    plan = plan_embedding_scan(lance.dataset(uri), row_filter="clip_id IN ('c4', 'c5')", max_fragments=None)

    assert plan.fragment_ids == (2,)
    assert plan.num_rows == 2


def test_the_cap_is_applied_after_the_finished_fragments_are_dropped(make_clips_table: ClipsTableFactory) -> None:
    """A capped run advances, rather than re-offering a prefix it has already finished.

    The load-bearing ordering in the planner. Capping the raw manifest prefix
    first would hand back fragment 0 here, find it owes nothing, drop it, and
    return an empty plan - so every capped attempt after the first would make no
    progress while reporting success.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    first = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=1)

    finished = "clip_id NOT IN ('c0', 'c1')"
    second = plan_embedding_scan(lance.dataset(uri), row_filter=finished, max_fragments=1)

    assert first.fragment_ids == (0,)
    assert second.fragment_ids == (1,)


def test_a_capped_plan_stops_counting_once_the_cap_is_filled(
    make_clips_table: ClipsTableFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap bounds the planning READ, not only the plan it returns.

    Every turn of the walk is a filtered count, which reads the predicate's
    columns, so a cap applied by slicing a completed walk would make a
    one-fragment smoke run pay for the whole corpus. Asserted by counting the
    counts, because the returned plan is identical either way - which is what
    lets the saving go missing unnoticed.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    counted: list[int] = []
    unspied = lance.LanceFragment.count_rows

    def spy(fragment: lance.LanceFragment, *args: object, **kwargs: object) -> int:
        counted.append(int(fragment.fragment_id))
        return int(unspied(fragment, *args, **kwargs))

    monkeypatch.setattr(lance.LanceFragment, "count_rows", spy)

    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=1)

    assert plan.fragment_ids == (0,)
    assert counted == [0]


def test_a_cap_of_zero_plans_nothing(make_clips_table: ClipsTableFactory) -> None:
    """A cap of zero takes on no fragment at all.

    The boundary that decides where the stop condition may sit: asked after a
    fragment has been appended it can never match, because the tally goes from
    zero owed straight to one, and the walk would then run to the end uncapped.
    Pinned here as well as end to end, where the same fault instead surfaces as
    an actor pool starting for a plan that should have been refused.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)

    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=0)

    assert plan.fragments == ()


def test_an_uncapped_plan_takes_on_every_owed_fragment(make_clips_table: ClipsTableFactory) -> None:
    """With no cap the plan spans the whole table, and its row count is the table's."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)

    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=None)

    assert plan.fragment_ids == tuple(range(_FRAGMENT_COUNT))
    assert plan.num_rows == _ROWS


def test_a_table_owing_nothing_plans_no_fragments(make_clips_table: ClipsTableFactory) -> None:
    """An up-to-date table yields an empty plan, which is how the caller learns there is no work."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)

    plan = plan_embedding_scan(lance.dataset(uri), row_filter="clip_id = 'absent'", max_fragments=None)

    assert plan.fragments == ()
    assert plan.num_rows == 0


def test_the_read_tasks_cover_every_planned_row_exactly_once(make_clips_table: ClipsTableFactory) -> None:
    """Splitting the plan across tasks partitions the rows; it neither drops nor duplicates one.

    Read at a parallelism below the fragment count so at least one task owns
    more than one fragment, which is the arrangement in which a slicing error
    can overlap two tasks.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=None)
    source = EmbeddingScanDatasource(_scan(uri, _EVERYTHING), plan.fragments)

    tasks = source.get_read_tasks(parallelism=2)

    scanned = [row for task in tasks for block in task() for row in block.to_pylist()]
    assert sorted(row["clip_id"] for row in scanned) == [f"c{index}" for index in range(_ROWS)]
    assert sum(task.metadata.num_rows or 0 for task in tasks) == _ROWS


def test_every_scanned_row_carries_a_unique_id_naming_its_own_fragment(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Checkpoint ids are unique table-wide and decompose to the fragment the plan named.

    Uniqueness is what Ray's row checkpoint rests on: a repeated id would mark
    an uncomputed row complete. Asserted across the whole read rather than per
    task, because the ids are only at risk of colliding BETWEEN fragments.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=None)
    source = EmbeddingScanDatasource(_scan(uri, _EVERYTHING), plan.fragments)

    rows = [row for task in source.get_read_tasks(parallelism=3) for block in task() for row in block.to_pylist()]

    checkpoint_ids = [row[CHECKPOINT_ID_COLUMN] for row in rows]
    assert len(set(checkpoint_ids)) == _ROWS
    for row in rows:
        assert row[CHECKPOINT_ID_COLUMN] >> ROW_ADDRESS_FRAGMENT_SHIFT == row[FRAGMENT_ID_COLUMN]
    assert {row[FRAGMENT_ID_COLUMN] for row in rows} == set(plan.fragment_ids)


def test_reading_more_tasks_than_fragments_yields_one_task_per_fragment(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Parallelism above the fragment count does not produce empty tasks Ray would schedule for nothing."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=None)
    source = EmbeddingScanDatasource(_scan(uri, _EVERYTHING), plan.fragments)

    tasks = source.get_read_tasks(parallelism=_FRAGMENT_COUNT * 4)

    assert len(tasks) == _FRAGMENT_COUNT


def test_a_plan_with_no_fragments_produces_no_read_tasks(make_clips_table: ClipsTableFactory) -> None:
    """Nothing owed means no task, so the caller need not special-case an empty read."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    source = EmbeddingScanDatasource(_scan(uri, _EVERYTHING), ())

    assert source.get_read_tasks(parallelism=4) == []


def test_a_non_positive_parallelism_is_refused(make_clips_table: ClipsTableFactory) -> None:
    """Zero tasks would read nothing while the run reported success."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    plan = plan_embedding_scan(lance.dataset(uri), row_filter=_EVERYTHING, max_fragments=None)
    source = EmbeddingScanDatasource(_scan(uri, _EVERYTHING), plan.fragments)

    with pytest.raises(ValueError, match="parallelism must be positive"):
        source.get_read_tasks(parallelism=0)


def test_a_fragment_absent_at_the_pinned_version_names_the_table_and_version(
    make_clips_table: ClipsTableFactory,
) -> None:
    """The failure names where it looked, because a worker's message is all the operator gets.

    The pinned version is what makes this reachable: the driver planned against a
    snapshot, and a worker reading later resolves fragment ids against that same
    snapshot rather than whatever the table has become.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    scan = _scan(uri, _EVERYTHING)
    absent = _FRAGMENT_COUNT + 99

    with pytest.raises(ValueError, match=rf"fragment {absent} is not present in .* at v{scan.read_version}"):
        list(read_owed_rows(scan, (absent,)))


def test_a_batch_size_below_one_is_refused_at_construction(make_clips_table: ClipsTableFactory) -> None:
    """A zero batch size is caught on the driver, not inside a worker's scanner."""
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)

    with pytest.raises(ValueError, match="batch_size"):
        _scan(uri, _EVERYTHING, batch_size=0)


def test_the_scan_reopens_the_version_it_was_pinned_to(make_clips_table: ClipsTableFactory) -> None:
    """A row deleted after planning is still read, so a run's scope cannot shrink under it.

    A delete rather than an append, because an append lands in a NEW fragment
    that no planned task asks for and so could not tell a pinned read from an
    unpinned one. A delete instead changes a fragment a task DOES own, which is
    the dangerous direction: the plan counted the row, the task would silently
    not produce it, and the run would report having embedded rows it never saw.
    """
    uri = make_clips_table(rows=_ROWS, rows_per_file=_ROWS_PER_FILE)
    scan = _scan(uri, _EVERYTHING)
    lance.dataset(uri).delete("clip_id = 'c0'")

    rows = [row for block in read_owed_rows(scan, tuple(range(_FRAGMENT_COUNT))) for row in block.to_pylist()]

    assert sorted(row["clip_id"] for row in rows) == [f"c{index}" for index in range(_ROWS)]
    assert lance.dataset(uri).count_rows() == _ROWS - 1
