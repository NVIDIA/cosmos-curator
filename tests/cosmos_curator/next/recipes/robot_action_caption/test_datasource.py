# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planning and scanning: shard ownership, owed counts, and checkpoint identity.

The read tasks are exercised by calling ``read_pending_clips`` directly rather
than through ``ray.data.read_datasource``: the behaviours under test are the
scan's own, and starting a Ray cluster would make them slower to check without
making them better checked.

Fixtures are local to this module rather than in a ``conftest.py`` so it owns
the table shape it needs, and the caption columns are written AT CREATION rather
than with a later ``dataset.update``, which would tombstone and re-append the
row - moving it into a different fragment, and so changing the very ids and
ownership these tests assert on.

Ownership is asserted over the FRAGMENT ids a fixture actually produces rather
than over an assumed dense range, and the partition properties - disjointness,
coverage, order-independence - are asserted here against a real manifest even
though ``test_sharding.py`` already asserts them against a synthetic universe.
The two are different claims: that the assignment partitions, and that planning
applies it to the ids Lance hands back.

The one context column the fixture writes is TEST-OWNED (:data:`CONTEXT_FIELD`)
and names a column no source table has. What the scan owes is a relation - a
resolved field is projected and rendered - so pinning that against the shipped
``SUPPORTED_CONTEXT_FIELDS`` would make landing a new column upstream fail these
tests without the scan having changed at all.
"""

import pathlib
from collections.abc import Callable, Sequence

import lance
import pyarrow as pa
import pytest

from cosmos_curator.next.core.sharding import ShardConfig
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_FIELDS,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import (
    CHECKPOINT_ID_COLUMN,
    FRAGMENT_ID_COLUMN,
    PROMPT_COLUMN,
    SCAN_SCHEMA,
    CaptionScanPlan,
    PinnedScan,
    plan_caption_scan,
    read_pending_clips,
    scanned_rows,
)
from cosmos_curator.next.recipes.robot_action_caption.prompt import PromptContextField
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    caption_work_filter,
)
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

ROWS = 7
ROWS_PER_FILE = 3

# Fragment count for the ownership tests, built one row per fragment so a
# fragment id and a clip index coincide and an assertion reads directly. Chosen
# so the shard counts exercised below both divide it (2, 3) and do not (5), since
# an uneven split is where an off-by-one in the assignment would surface.
FRAGMENTS = 12

# Lance's own name for the address column a scan adds under ``with_row_address``,
# and the packing it uses: (fragment_id << 32) | offset_within_fragment. Spelled
# out rather than imported from the module under test, so an assertion here
# cannot be satisfied by the same constant it is checking.
ROW_ADDRESS_COLUMN = "_rowaddr"
ROW_ADDRESS_FRAGMENT_SHIFT = 32

# Two well-formed digests belonging to no real contract. ``caption_work_filter``
# validates the shape and then compares for equality only, so nothing below
# depends on what the shipped contract actually hashes to. They differ in the
# last character alone, which is all an equality comparison needs.
DIGEST = f"{'0' * 63}a"
STALE_DIGEST = f"{'0' * 63}b"
MODEL = "m"

# A test-owned context field, standing in for whatever subset a run resolves. Its
# column name is one no source table holds, so landing a new supported column
# upstream cannot change what these tests project or render.
CONTEXT_FIELD = PromptContextField(column="scene_note", label="Scene note")

# Builds a clips.lance carrying the caption columns; see ``make_clips_table``.
ClipsTableFactory = Callable[..., str]


@pytest.fixture
def make_clips_table(tmp_path: pathlib.Path) -> ClipsTableFactory:
    """Return a builder for a captionable ``clips.lance`` under ``tmp_path``.

    The caption columns are always present, since the scan filters on
    ``caption_status``, and so is :data:`CONTEXT_FIELD`'s column, so a test can
    resolve it without rebuilding the table. Keyword arguments: ``rows``
    (default 7), ``rows_per_file`` (default 3, giving three fragments),
    ``clip_uris`` (per-row ``clip_uri`` override, defaulting to a distinct URI
    per row), ``notes`` (per-row :data:`CONTEXT_FIELD` override, defaulting to a
    distinct value per row), ``statuses`` (per-row ``caption_status``, defaulting
    to all null), ``digests`` (per-row ``caption_contract_digest``), ``selection``
    (per-row curation verdict, defaulting to all selected), and
    ``omit_selection``. A row given a status defaults to a valid terminal caption
    group under :data:`DIGEST` and :data:`MODEL`, unless ``digests`` says otherwise.
    """

    def build(
        *,
        rows: int = ROWS,
        rows_per_file: int = ROWS_PER_FILE,
        clip_uris: Sequence[str | None] | None = None,
        notes: Sequence[str | None] | None = None,
        statuses: Sequence[str | None] | None = None,
        digests: Sequence[str | None] | None = None,
        selection: Sequence[str | None] | None = None,
        omit_selection: bool = False,
    ) -> str:
        verdicts = list(statuses) if statuses is not None else [None] * rows
        published = list(digests) if digests is not None else [DIGEST if v else None for v in verdicts]
        columns = {
            CLIP_ID_COLUMN: pa.array([f"c{i}" for i in range(rows)], pa.string()),
            CLIP_URI_COLUMN: pa.array(
                list(clip_uris) if clip_uris is not None else [f"s3://b/c{i}.mp4" for i in range(rows)],
                pa.large_string(),
            ),
            CONTEXT_FIELD.column: pa.array(
                list(notes) if notes is not None else [f"note {i}" for i in range(rows)],
                pa.large_string(),
            ),
            # Split by verdict the way validate_terminal_values() requires: a
            # success or truncated row carries a nonblank caption and a null
            # error, an error row carries the reverse. A caption paired with a
            # null error under "error" is a row no producer can write, and the
            # pending predicate reads it as still owed.
            CAPTION_COLUMN: pa.array(
                [f"caption {i}" if verdict in {"success", "truncated"} else None for i, verdict in enumerate(verdicts)],
                pa.large_string(),
            ),
            CAPTION_STATUS_COLUMN: pa.array(verdicts, pa.string()),
            CAPTION_ERROR_COLUMN: pa.array(
                [f"error {i}" if verdict == "error" else None for i, verdict in enumerate(verdicts)],
                pa.large_string(),
            ),
            CAPTION_DIGEST_COLUMN: pa.array(published, pa.string()),
            CAPTION_MODEL_COLUMN: pa.array([MODEL if v else None for v in verdicts], pa.string()),
            CURATE_SELECTION_REASON: pa.array(
                list(selection) if selection is not None else [CURATE_SELECTED_VALUE] * rows,
                pa.string(),
            ),
        }
        schema = pa.schema(
            [
                pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
                # Nullable, matching the source contract: a row naming no media
                # location is a real shape the scan must pass through.
                pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=True),
                pa.field(CONTEXT_FIELD.column, pa.large_string()),
                *CAPTION_FIELDS,
                pa.field(CURATE_SELECTION_REASON, pa.string()),
            ]
        )
        if omit_selection:
            columns.pop(CURATE_SELECTION_REASON)
            schema = pa.schema([field for field in schema if field.name != CURATE_SELECTION_REASON])
        table = pa.table(columns, schema=schema)
        uri = str(tmp_path / "clips.lance")
        lance.write_dataset(
            table, uri, max_rows_per_file=rows_per_file, data_storage_version=LANCE_DATA_STORAGE_VERSION
        )
        return uri

    return build


def read_all(
    uri: str,
    *,
    batch_size: int = 2,
    context_fields: tuple[PromptContextField, ...] = (),
) -> pa.Table:
    """Read every pending clip of every pending fragment as one table.

    Plans with ``plan_caption_scan`` and reads with ``read_pending_clips``, which
    is the production pairing: the plan decides which fragments hold pending rows
    and the read task re-derives the same predicate on its own worker.
    """
    dataset = lance.dataset(uri)
    row_filter = caption_work_filter(DIGEST, MODEL, selected_only=False)
    scan = PinnedScan.for_dataset(
        dataset,
        storage_options=None,
        batch_size=batch_size,
        context_fields=context_fields,
        row_filter=row_filter,
    )
    tables = list(read_pending_clips(scan, plan_caption_scan(dataset, row_filter, shard=None).fragment_ids))
    return pa.concat_tables(tables) if tables else SCAN_SCHEMA.empty_table()


def ids_by_clip(scanned: pa.Table) -> dict[str, int]:
    """Return the scanned checkpoint id of each clip."""
    return {row[CLIP_ID_COLUMN]: row[CHECKPOINT_ID_COLUMN] for row in scanned.to_pylist()}


def test_checkpoint_id_is_an_unsigned_integer(make_clips_table: ClipsTableFactory) -> None:
    """The id is a row address, so it is the width Lance emits and not text.

    The type is asserted at the scan boundary because Ray's checkpoint reads this
    column straight off the Read output and the staged Parquet has to declare the
    same thing; a wider or signed id here would make the whole recovery path
    disagree about one column's type.
    """
    scanned = read_all(make_clips_table())

    assert scanned.schema.field(CHECKPOINT_ID_COLUMN).type == pa.uint64()


def test_checkpoint_ids_are_unique_across_fragments(make_clips_table: ClipsTableFactory) -> None:
    """Within one snapshot every id is distinct, so the id has no collision case."""
    scanned = read_all(make_clips_table())
    ids = scanned.column(CHECKPOINT_ID_COLUMN).to_pylist()

    assert len(ids) == ROWS
    assert len(set(ids)) == ROWS


def test_checkpoint_ids_are_stable_across_two_reads(make_clips_table: ClipsTableFactory) -> None:
    """Two reads of one pinned version return the same id for the same clip.

    This is what makes the id reusable as a resume key at all: a second run
    pinning the same snapshot must recognise the ids the first run committed.
    """
    uri = make_clips_table()

    first = ids_by_clip(read_all(uri, batch_size=2))
    second = ids_by_clip(read_all(uri, batch_size=5))

    assert first == second
    assert len(first) == ROWS


def test_the_fragment_id_is_the_high_half_of_the_checkpoint_id(make_clips_table: ClipsTableFactory) -> None:
    """Both identity columns come from one scanned value, so they cannot disagree.

    Publication addresses a staged partition by ``fragment_id`` while Ray matches
    a resume against ``__checkpoint_id``. Deriving the first from the second is
    what stops a row being checkpointed under one fragment and staged under
    another; a join back to the table could resolve differently.
    """
    scanned = read_all(make_clips_table())

    for row in scanned.to_pylist():
        assert row[FRAGMENT_ID_COLUMN] == row[CHECKPOINT_ID_COLUMN] >> ROW_ADDRESS_FRAGMENT_SHIFT


def test_two_fragments_holding_one_clip_id_get_distinct_checkpoint_ids() -> None:
    """The id separates rows a clip id alone would fold together.

    Built synthetically because the table boundary refuses a duplicate clip id
    upstream; what is under test is that the id is taken from the row's address
    rather than from its business key, which is what stops a resume marking one
    fragment's row complete because a different fragment's row finished.
    """
    scanned = pa.table(
        {
            ROW_ADDRESS_COLUMN: pa.array([0, 1 << ROW_ADDRESS_FRAGMENT_SHIFT], pa.uint64()),
            CLIP_ID_COLUMN: pa.array(["c0", "c0"], pa.string()),
            CLIP_URI_COLUMN: pa.array(["s3://b/c0.mp4", "s3://b/c0.mp4"], pa.large_string()),
        }
    )

    ids = scanned_rows(scanned, ()).column(CHECKPOINT_ID_COLUMN).to_pylist()

    # The addresses themselves rather than their distinctness: distinctness alone
    # is satisfied by any injective function of position, including a per-batch
    # counter that would not name a row at all.
    assert ids == [0, 1 << ROW_ADDRESS_FRAGMENT_SHIFT]


def test_fragment_ids_span_every_fragment(make_clips_table: ClipsTableFactory) -> None:
    """A three-fragment fixture yields rows attributed to all three."""
    scanned = read_all(make_clips_table())

    assert sorted(set(scanned.column(FRAGMENT_ID_COLUMN).to_pylist())) == [0, 1, 2]


def test_scan_yields_every_pending_row_exactly_once(make_clips_table: ClipsTableFactory) -> None:
    """Each pending clip appears once, with all five fields populated."""
    scanned = read_all(make_clips_table())

    assert scanned.column(CLIP_ID_COLUMN).to_pylist() == [f"c{i}" for i in range(ROWS)]
    for field in SCAN_SCHEMA:
        assert scanned.column(field.name).null_count == 0


def test_scan_excludes_an_already_captioned_row(make_clips_table: ClipsTableFactory) -> None:
    """A published clip is filtered out at the scan, so it never reaches the engine."""
    uri = make_clips_table(statuses=[None, None, None, None, "success", None, None])

    scanned = read_all(uri)

    assert scanned.column(CLIP_ID_COLUMN).to_pylist() == ["c0", "c1", "c2", "c3", "c5", "c6"]


def test_planning_counts_each_pending_row_once(make_clips_table: ClipsTableFactory) -> None:
    """The plan's row count is the number of rows the scan then yields.

    Planning and counting are one pass over the manifest, and the caller both
    decides there is work to do and sizes the engine pool from this one number.
    Two passes could disagree; a fragment counted whole rather than filtered
    would over-provision against rows that do not exist.

    Pinned against the rows actually read as well as against a literal, so
    neither an unfiltered count nor a plan that silently skipped a fragment can
    satisfy it.
    """
    uri = make_clips_table(statuses=[None, None, None, None, "success", None, None])

    plan = plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=False),
        shard=None,
    )

    assert plan.num_rows == ROWS - 1
    assert plan.num_rows == read_all(uri).num_rows


def test_planning_excludes_a_fragment_with_only_unselected_pending_rows(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Do not spend a work item on a fragment outside the attempt's scope."""
    uri = make_clips_table(
        rows=3,
        rows_per_file=3,
        selection=[
            "duplicate",
            "below_quota",
            None,
        ],
    )

    plan = plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=True),
        shard=None,
    )

    assert plan.fragments == ()


def test_planning_counts_only_selected_pending_rows_in_a_mixed_fragment(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Size the attempt from the same selected-and-pending rows it scans."""
    uri = make_clips_table(
        rows=3,
        rows_per_file=3,
        statuses=[None, None, "success"],
        selection=[
            "duplicate",
            CURATE_SELECTED_VALUE,
            CURATE_SELECTED_VALUE,
        ],
    )

    plan = plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=True),
        shard=None,
    )

    assert plan.fragment_ids == (0,)
    assert plan.num_rows == 1


def test_datasource_reads_exactly_the_selected_pending_rows(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Read the same selected-and-pending row counted by planning."""
    uri = make_clips_table(
        rows=3,
        rows_per_file=3,
        statuses=[None, None, "success"],
        selection=[
            "duplicate",
            CURATE_SELECTED_VALUE,
            CURATE_SELECTED_VALUE,
        ],
    )
    dataset = lance.dataset(uri)
    row_filter = caption_work_filter(DIGEST, MODEL, selected_only=True)
    plan = plan_caption_scan(dataset, row_filter, shard=None)
    scan = PinnedScan.for_dataset(
        dataset,
        storage_options=None,
        batch_size=3,
        context_fields=(),
        row_filter=row_filter,
    )

    scanned = pa.concat_tables(read_pending_clips(scan, plan.fragment_ids))

    assert scanned.column(CLIP_ID_COLUMN).to_pylist() == ["c1"]


def test_planning_a_fully_captioned_table_owes_nothing(make_clips_table: ClipsTableFactory) -> None:
    """A table captioned under this digest plans no fragment, so a resume reads nothing."""
    uri = make_clips_table(statuses=["success"] * ROWS)

    plan = plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=False),
        shard=None,
    )

    assert plan.fragments == ()
    assert plan.num_rows == 0


def test_planning_re_plans_a_table_captioned_under_another_contract(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Latest contract wins at the planning boundary: a superseded digest is owed again.

    The counterfactual to the test above, on the same fully-published table, and
    the reason planning takes a digest at all. Without it a contract change would
    be unresumable in the worst way: the plan would skip every already-captioned
    fragment, so the run would report success having re-captioned nothing.
    """
    uri = make_clips_table(statuses=["success"] * ROWS, digests=[STALE_DIGEST] * ROWS)

    plan = plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=False),
        shard=None,
    )

    assert plan.fragment_ids == (0, 1, 2)
    assert plan.num_rows == ROWS


def plan_for_shard(uri: str, shard: ShardConfig | None) -> CaptionScanPlan:
    """Plan one shard's scope over a table that owes every row."""
    return plan_caption_scan(
        lance.dataset(uri),
        caption_work_filter(DIGEST, MODEL, selected_only=False),
        shard=shard,
    )


def test_an_unsharded_plan_owns_every_fragment(make_clips_table: ClipsTableFactory) -> None:
    """``None`` is the unsharded scope and must claim the whole manifest.

    The baseline the ownership tests below are relative to: without it a
    partition assertion could be satisfied by a planner that owned nothing.
    """
    uri = make_clips_table(rows=FRAGMENTS, rows_per_file=1)

    plan = plan_for_shard(uri, None)

    assert plan.owned_fragment_ids == tuple(range(FRAGMENTS))
    assert plan.fragment_ids == tuple(range(FRAGMENTS))
    assert plan.num_rows == FRAGMENTS


@pytest.mark.parametrize("count", [2, 3, 5])
def test_the_shards_of_one_campaign_partition_the_fragments(make_clips_table: ClipsTableFactory, count: int) -> None:
    """Every fragment is planned by exactly one shard: no gap, no overlap.

    This is the whole correctness claim of a sharded campaign. A gap is a clip no
    job ever captions and nothing reports; an overlap is two jobs captioning one
    clip, which publication then refuses.

    Counts that do and do not divide the fragment total are both covered, since
    an uneven split is where an off-by-one in the assignment would show.
    """
    uri = make_clips_table(rows=FRAGMENTS, rows_per_file=1)

    planned = [plan_for_shard(uri, ShardConfig(count=count, index=index)).fragment_ids for index in range(count)]

    claimed = [fragment_id for ids in planned for fragment_id in ids]
    assert sorted(claimed) == list(range(FRAGMENTS))
    assert len(claimed) == len(set(claimed))


def test_ownership_is_decided_on_the_fragment_id_not_the_owed_count(
    make_clips_table: ClipsTableFactory,
) -> None:
    """A captioned fragment stays owned, so the partition cannot move as work completes.

    Were ownership derived from anything the pending predicate sees, a job
    resuming after a sibling published would compute a different scope than it
    started with, and the fragments that changed hands would be owned by nobody.

    Pinned against literals rather than against a second plan: the owned set is
    what a modulo over ``count=3, index=1`` selects from twelve fragments, and the
    owed set is the half of those the fixture left pending. The two differing is
    what makes the first assertion a claim about ids rather than about counts.
    """
    captioned_fragments = 6
    uri = make_clips_table(
        rows=FRAGMENTS,
        rows_per_file=1,
        statuses=["success"] * captioned_fragments + [None] * (FRAGMENTS - captioned_fragments),
    )

    plan = plan_for_shard(uri, ShardConfig(count=3, index=1))

    assert plan.owned_fragment_ids == (1, 4, 7, 10)
    assert plan.fragment_ids == (7, 10)


def test_a_shard_that_owns_no_owed_fragment_plans_nothing(make_clips_table: ClipsTableFactory) -> None:
    """A no-op shard is a clean empty plan, and still reports the scope it owned.

    The distinction the run summary rests on: ``fragments`` empty means THIS JOB
    owes nothing, while ``owned_fragment_ids`` non-empty says that verdict was
    reached over a real scope rather than by owning none of the table.
    """
    uri = make_clips_table(rows=FRAGMENTS, rows_per_file=1, statuses=["success"] * FRAGMENTS)

    plan = plan_for_shard(uri, ShardConfig(count=3, index=1))

    assert plan.fragments == ()
    assert plan.owned_fragment_ids


def test_a_count_above_the_fragment_total_leaves_shards_empty(make_clips_table: ClipsTableFactory) -> None:
    """Over-sharding is a no-op for the surplus jobs, not an error.

    An operator sizing a campaign against a table that later compacts should get
    idle jobs rather than failures, and the fragments must still all be owned.
    """
    uri = make_clips_table(rows=2, rows_per_file=1)
    count = 8

    planned = [plan_for_shard(uri, ShardConfig(count=count, index=index)).fragment_ids for index in range(count)]

    assert sorted(fragment_id for ids in planned for fragment_id in ids) == [0, 1]
    assert sum(1 for ids in planned if not ids) == count - 2


def test_a_null_clip_uri_is_carried_through_the_scan(make_clips_table: ClipsTableFactory) -> None:
    """A row naming no media location is emitted with a null ``clip_uri``, not refused.

    Refusing it here would fail the read task and so the whole attempt, whereas
    the media stage can refuse the single row that names no location and let the
    rest of the corpus caption. :data:`SCAN_SCHEMA` declares the column nullable
    for exactly this row.
    """
    uri = make_clips_table(rows=2, clip_uris=[None, "s3://b/c1.mp4"])

    scanned = read_all(uri)

    assert scanned.column(CLIP_ID_COLUMN).to_pylist() == ["c0", "c1"]
    assert scanned.column(CLIP_URI_COLUMN).to_pylist() == [None, "s3://b/c1.mp4"]


def test_a_present_context_column_reaches_the_prompt(make_clips_table: ClipsTableFactory) -> None:
    """A resolved context column is projected by the scan and rendered into ``prompt_text``.

    Two steps have to agree for this to hold, and the failure modes differ: were
    the column left out of the projection, ``scanned_rows`` would refuse the
    batch by name; were it projected but not rendered, the prompt would silently
    lose a line the digest says it carries.

    The field is test-owned, so this pins the RELATION between a resolved field
    and the rendered prompt rather than today's supported field list.
    """
    uri = make_clips_table(rows=1, notes=["a cluttered worktop"])

    prompt = read_all(uri, context_fields=(CONTEXT_FIELD,)).column(PROMPT_COLUMN).to_pylist()[0]

    assert f"{CONTEXT_FIELD.label}: a cluttered worktop" in prompt


def test_an_unresolved_context_column_is_not_rendered(make_clips_table: ClipsTableFactory) -> None:
    """A column the run did not resolve stays out of the prompt even though the table holds it.

    This is what makes the resolved subset the contract rather than a hint: the
    digest is computed over the fields passed in, so a prompt that read the table
    directly would carry context no digest accounts for.
    """
    uri = make_clips_table(rows=1, notes=["a cluttered worktop"])

    prompt = read_all(uri, context_fields=()).column(PROMPT_COLUMN).to_pylist()[0]

    assert "cluttered worktop" not in prompt
    assert CONTEXT_FIELD.label not in prompt


def test_scan_carries_the_clip_uri_for_the_media_stage(make_clips_table: ClipsTableFactory) -> None:
    """The media stage's input travels on the same row as the id it will be keyed by."""
    scanned = read_all(make_clips_table(rows=1))

    assert scanned.column(CLIP_URI_COLUMN).to_pylist() == ["s3://b/c0.mp4"]


def test_scanned_rows_names_a_missing_required_column() -> None:
    """A projection that lost a source column fails naming it, not inside the prompt renderer."""
    incomplete = pa.table(
        {
            ROW_ADDRESS_COLUMN: pa.array([0], pa.uint64()),
            CLIP_ID_COLUMN: pa.array(["c0"], pa.string()),
        }
    )

    with pytest.raises(ValueError, match=CLIP_URI_COLUMN):
        scanned_rows(incomplete, ())


def test_scanned_rows_names_a_missing_resolved_context_column() -> None:
    """A resolved context column absent from the batch is refused rather than read as absent.

    Resolution already decided this snapshot supplies the column, so an absent
    one means the projection and the render disagree. Left to the renderer it is
    silent: every caption in the run would be stamped under a contract whose
    context line the prompt never carried.
    """
    without_context = pa.table(
        {
            ROW_ADDRESS_COLUMN: pa.array([0], pa.uint64()),
            CLIP_ID_COLUMN: pa.array(["c0"], pa.string()),
            CLIP_URI_COLUMN: pa.array(["s3://b/c0.mp4"], pa.large_string()),
        }
    )

    with pytest.raises(ValueError, match=CONTEXT_FIELD.column):
        scanned_rows(without_context, (CONTEXT_FIELD,))


@pytest.mark.parametrize(
    ("invalid_clip_id", "clip_id_type"),
    [
        pytest.param(None, pa.string(), id="null"),
        pytest.param("", pa.string(), id="empty"),
        pytest.param(" \t", pa.string(), id="whitespace"),
        pytest.param(7, pa.int64(), id="non-string"),
    ],
)
def test_scanned_rows_rejects_an_invalid_logical_key(
    invalid_clip_id: object,
    clip_id_type: pa.DataType,
) -> None:
    """Refuse malformed keys at the read boundary before Arrow can coerce them."""
    scanned = pa.table(
        {
            ROW_ADDRESS_COLUMN: pa.array([0], pa.uint64()),
            CLIP_ID_COLUMN: pa.array([invalid_clip_id], clip_id_type),
            CLIP_URI_COLUMN: pa.array(["s3://b/c0.mp4"], pa.large_string()),
        }
    )

    with pytest.raises(ValueError, match="clip_id must be a nonblank string"):
        scanned_rows(scanned, ())


def test_scanned_rows_preserves_a_nonblank_key_byte_for_byte() -> None:
    """Preserve whitespace around a nonblank logical key byte-for-byte."""
    scanned = pa.table(
        {
            ROW_ADDRESS_COLUMN: pa.array([0], pa.uint64()),
            CLIP_ID_COLUMN: pa.array(["  c0  "], pa.string()),
            CLIP_URI_COLUMN: pa.array(["s3://b/c0.mp4"], pa.large_string()),
        }
    )

    assert scanned_rows(scanned, ()).column(CLIP_ID_COLUMN).to_pylist() == ["  c0  "]


def test_a_read_task_rejects_a_fragment_absent_at_the_pinned_version(
    make_clips_table: ClipsTableFactory,
) -> None:
    """A fragment the snapshot does not carry is an error, not a silently empty read."""
    dataset = lance.dataset(make_clips_table())
    scan = PinnedScan.for_dataset(
        dataset,
        storage_options=None,
        batch_size=2,
        context_fields=(),
        row_filter=caption_work_filter(DIGEST, MODEL, selected_only=False),
    )

    with pytest.raises(ValueError, match="is not present in"):
        # Consumed, because ``read_pending_clips`` is a generator and validates
        # nothing until its first item is pulled.
        list(read_pending_clips(scan, (99,)))


def test_direct_pinned_scan_construction_rejects_a_non_positive_batch_size() -> None:
    """Prevent direct construction from bypassing the positive-size contract."""
    with pytest.raises(ValueError, match=r"'batch_size' must be >= 1"):
        PinnedScan(
            uri="memory://clips.lance",
            read_version=1,
            storage_options=None,
            batch_size=0,
            context_fields=(),
            row_filter=caption_work_filter(DIGEST, MODEL, selected_only=False),
        )


def test_direct_pinned_scan_construction_copies_storage_options() -> None:
    """Prevent caller mutation from retargeting a pinned scan."""
    storage_options = {"aws_region": "us-east-1"}
    scan = PinnedScan(
        uri="s3://bucket/clips.lance",
        read_version=1,
        storage_options=storage_options,
        batch_size=1,
        context_fields=(),
        row_filter=caption_work_filter(DIGEST, MODEL, selected_only=False),
    )

    storage_options["aws_region"] = "us-west-2"

    assert scan.storage_options == {"aws_region": "us-east-1"}
    assert scan.storage_options is not storage_options
