# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The source contract over ``clips.lance``: structural columns, context resolution, pending rows.

Four behaviours: refuse a table no row of which is captionable, resolve which
optional context this snapshot supplies, register the caption columns, and -
once they are registered - ask which rows this contract still owes a caption.
``pipeline.py`` owns the order a run calls them in; the order below is this
file's own.

Fixtures live in this module rather than in a ``conftest.py`` so it owns the
exact table shape each behaviour needs. Two builders, because the two concerns
want opposite tables: ``make_clips_table`` writes several fragments of real rows,
because the per-fragment pending query has nothing to scope on a single-fragment
table and the row-level predicate needs rows to select; ``make_schema_table``
writes a zero-row table from a schema handed in whole, because the
column-contract functions read ``dataset.schema`` and nothing else, so rows
would only obscure which part of the table is under test.

``make_clips_table`` writes already-published verdicts AT CREATION rather than
with a later ``dataset.update``, because an update tombstones and re-appends the
row, which would move the very fragment membership these tests assert on.
"""

import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import lance
import pyarrow as pa
import pytest

import cosmos_curator.next.recipes.robot_action_caption.table as caption_table
from cosmos_curator.next.recipes.curation.columns import (
    CURATE_COLUMNS,
    CurateReason,
)
from cosmos_curator.next.recipes.curation.columns import (
    CURATE_SELECTION_REASON as PRODUCER_SELECTION_REASON,
)
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_COLUMNS,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_FIELDS,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
)
from cosmos_curator.next.recipes.robot_action_caption.prompt import SUPPORTED_CONTEXT_FIELDS
from cosmos_curator.next.recipes.robot_action_caption.table import (
    _REGISTRATION_ATTEMPTS,
    CLIP_ID_COLUMN,
    CLIP_URI_COLUMN,
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    REQUIRED_SOURCE_COLUMNS,
    caption_work_filter,
    count_rows_in_fragments,
    ensure_caption_fields,
    owed_clip_ids,
    require_source_columns,
    resolve_context_fields,
)
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# Two digests that are well-formed but belong to no real contract, so no assertion
# below depends on what the shipped contract currently hashes to. They differ in
# the last character only, which is all the predicate needs: it compares for
# equality and never for distance.
CURRENT_DIGEST = "0" * 63 + "a"
STALE_DIGEST = "0" * 63 + "b"
MODEL_ID = "model"

# The subset of the robot-action-split clip schema this recipe reads. Restated
# here rather than imported so a producer-side column addition cannot turn a
# caption behaviour test red.
CLIPS_SOURCE_SCHEMA = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        pa.field("task_name", pa.string(), nullable=False),
        pa.field("subtask_name", pa.string(), nullable=False),
        pa.field("camera_motion_annotation", pa.large_string()),
        pa.field(CURATE_SELECTION_REASON, pa.string()),
    ]
)

# The two structural columns at the encodings the producer writes them under, so
# a schema built from these alone satisfies the structural contract.
STRUCTURAL_FIELDS = (
    pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
    pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False),
)


def register(uri: str) -> lance.LanceDataset:
    """Register the caption group on a local table, returning the widened handle.

    ``storage_options`` is always ``None`` here: every fixture is a local
    directory, so credentials would be a value the subject never reads.
    """
    return ensure_caption_fields(uri, storage_options=None)


# Builds a clips.lance and returns its URI; see ``make_clips_table``.
ClipsTableFactory = Callable[..., str]

# Writes a zero-row table from an explicit schema and returns its URI.
SchemaTableFactory = Callable[[pa.Schema], str]


def source_columns(
    *,
    rows: int,
    motion: Sequence[str | None] | None,
    selection: Sequence[str | None] | None,
) -> dict[str, pa.Array]:
    """Return ``rows`` source clip columns, with ``c0..c{rows-1}`` as clip ids."""
    return {
        "clip_id": pa.array([f"c{i}" for i in range(rows)], pa.string()),
        "clip_uri": pa.array([f"s3://b/c{i}.mp4" for i in range(rows)], pa.large_string()),
        "task_name": pa.array([f"task{i}" for i in range(rows)], pa.string()),
        "subtask_name": pa.array([f"subtask{i}" for i in range(rows)], pa.string()),
        "camera_motion_annotation": pa.array(
            list(motion) if motion is not None else [f"pan {i}" for i in range(rows)],
            pa.large_string(),
        ),
        CURATE_SELECTION_REASON: pa.array(
            list(selection) if selection is not None else [CURATE_SELECTED_VALUE] * rows,
            pa.string(),
        ),
    }


def context_schema(column: str, arrow_type: pa.DataType) -> pa.Schema:
    """Return a structurally valid schema carrying one context column at ``arrow_type``."""
    return pa.schema([*STRUCTURAL_FIELDS, pa.field(column, arrow_type)])


def structural_schema(column: str, arrow_type: pa.DataType) -> pa.Schema:
    """Return the structural schema with ``column`` stored at ``arrow_type`` instead.

    One retype at a time is the point: the two structural columns are held to
    DIFFERENT types, so a fixture that retyped both could not attribute a
    refusal to either gate.
    """
    return pa.schema(
        [
            pa.field(field.name, arrow_type, nullable=field.nullable) if field.name == column else field
            for field in STRUCTURAL_FIELDS
        ]
    )


def warnings_of(records: Sequence[dict[str, Any]]) -> list[str]:
    """Return the WARNING messages captured so far."""
    return [record["message"] for record in records if record["level"].name == "WARNING"]


@pytest.fixture
def make_clips_table(tmp_path: pathlib.Path) -> ClipsTableFactory:
    """Return a builder for a ``clips.lance`` under ``tmp_path``.

    Keyword arguments: ``rows`` (default 7), ``rows_per_file`` (default 3, giving
    three fragments at the default row count), ``motion`` (per-row
    ``camera_motion_annotation`` override), ``statuses`` (per-row
    ``caption_status``; when given the caption columns are written at creation,
    otherwise they are absent entirely), ``captions``, ``errors``, ``digests``,
    and ``models`` (per-row caption-group overrides), ``selection`` (per-row
    ``curate_selection_reason``, defaulting to all selected), ``omit_selection``
    (remove the curation column), and ``name``. Unspecified caption members form
    a valid terminal row under :data:`CURRENT_DIGEST` and :data:`MODEL_ID`.
    """

    def build(
        *,
        rows: int = 7,
        rows_per_file: int = 3,
        motion: Sequence[str | None] | None = None,
        statuses: Sequence[str | None] | None = None,
        captions: Sequence[str | None] | None = None,
        errors: Sequence[str | None] | None = None,
        digests: Sequence[str | None] | None = None,
        models: Sequence[str | None] | None = None,
        selection: Sequence[str | None] | None = None,
        omit_selection: bool = False,
        name: str = "clips.lance",
    ) -> str:
        columns = source_columns(rows=rows, motion=motion, selection=selection)
        schema = CLIPS_SOURCE_SCHEMA
        if omit_selection:
            columns.pop(CURATE_SELECTION_REASON)
            schema = pa.schema([field for field in schema if field.name != CURATE_SELECTION_REASON])
        if statuses is not None:
            schema = pa.schema([*schema, *CAPTION_FIELDS])
            caption_values = (
                list(captions)
                if captions is not None
                else [
                    f"caption {i}" if status in {"success", "truncated"} else None for i, status in enumerate(statuses)
                ]
            )
            error_values = (
                list(errors)
                if errors is not None
                else [f"error {i}" if status == "error" else None for i, status in enumerate(statuses)]
            )
            published = (
                list(digests)
                if digests is not None
                else [CURRENT_DIGEST if status is not None else None for status in statuses]
            )
            model_values = (
                list(models)
                if models is not None
                else [MODEL_ID if status is not None else None for status in statuses]
            )
            columns |= {
                CAPTION_COLUMN: pa.array(caption_values, pa.large_string()),
                CAPTION_STATUS_COLUMN: pa.array(list(statuses), pa.string()),
                CAPTION_ERROR_COLUMN: pa.array(error_values, pa.large_string()),
                CAPTION_DIGEST_COLUMN: pa.array(published, pa.string()),
                CAPTION_MODEL_COLUMN: pa.array(model_values, pa.string()),
            }
        uri = str(tmp_path / name)
        lance.write_dataset(
            pa.table(columns, schema=schema),
            uri,
            max_rows_per_file=rows_per_file,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        return uri

    return build


@pytest.fixture
def make_schema_table(tmp_path: pathlib.Path) -> SchemaTableFactory:
    """Return a builder writing a zero-row table under ``tmp_path`` from one schema.

    Zero rows is the point, not a shortcut: ``require_source_columns`` and
    ``resolve_context_fields`` read the schema and never a value, so a table with
    rows would invite an assertion about contents that neither function looks at.
    Lance preserves the field types handed in here, a dictionary encoding
    included, so the schema the test wrote is the schema the function sees.
    """
    written = 0

    def build(schema: pa.Schema) -> str:
        nonlocal written
        written += 1
        uri = str(tmp_path / f"schema{written}.lance")
        lance.write_dataset(
            pa.Table.from_pylist([], schema=schema),
            uri,
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        return uri

    return build


@pytest.mark.parametrize("omitted", REQUIRED_SOURCE_COLUMNS)
def test_a_table_missing_a_structural_column_is_refused(make_schema_table: SchemaTableFactory, omitted: str) -> None:
    """Neither structural column can be absent, and the error names the one that is.

    Parametrized over the shipped tuple rather than over two literals, so a
    column promoted to structural is covered the moment it is promoted.
    """
    uri = make_schema_table(pa.schema([field for field in STRUCTURAL_FIELDS if field.name != omitted]))

    with pytest.raises(ValueError, match=f"missing required source column.*{omitted}"):
        require_source_columns(lance.dataset(uri))


@pytest.mark.parametrize("omitted", REQUIRED_SOURCE_COLUMNS)
def test_caption_registration_refuses_a_missing_structural_column_before_widening(
    make_schema_table: SchemaTableFactory,
    omitted: str,
) -> None:
    """Refuse a missing clip_id or clip_uri without bumping the Lance version."""
    uri = make_schema_table(pa.schema([field for field in STRUCTURAL_FIELDS if field.name != omitted]))
    before = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match=f"missing required source column.*{omitted}"):
        register(uri)

    # Re-read rather than inspected through a handle held across the call:
    # registration opens its own, so a handle from before it could not observe a
    # commit even if one had happened, and the assertion would pass vacuously.
    after = lance.dataset(uri)
    assert int(after.version) == before
    assert set(after.schema.names).isdisjoint(CAPTION_COLUMNS)


def test_a_non_text_clip_uri_is_refused(make_schema_table: SchemaTableFactory) -> None:
    """A retyped media location is caught on the driver, not inside a read task.

    Presence alone is not enough: the column is read as text, and a producer that
    retyped it would otherwise fail per fragment, where Ray erases the exception
    type and the message reaches an operator wrapped in a ``RayError``.

    Named rather than parametrized over the structural tuple, because the two
    columns are held to different types - see the ``clip_id`` case below. A
    column promoted to structural therefore needs its own gate as well as its own
    case here.
    """
    uri = make_schema_table(structural_schema(CLIP_URI_COLUMN, pa.int64()))

    with pytest.raises(ValueError, match=f"non-text {CLIP_URI_COLUMN}"):
        require_source_columns(lance.dataset(uri))


@pytest.mark.parametrize(
    "unusable",
    [
        pytest.param(pa.int64(), id="int64"),
        pytest.param(pa.large_string(), id="large-string"),
        pytest.param(pa.dictionary(pa.int32(), pa.string()), id="dictionary-of-string"),
    ],
)
def test_a_clip_id_the_update_join_cannot_publish_on_is_refused(
    make_schema_table: SchemaTableFactory, unusable: pa.DataType
) -> None:
    """``clip_id`` is held to ``string``, not merely to a type that reads back as text.

    The narrower gate is about the WRITE path rather than the read: publication
    types the caption update from the STORED schema so ``update_columns`` never
    coerces, which means a large_string or dictionary id is joined at that type on
    both sides. Whether that join matches is not something this gate can promise,
    and a join matching nothing would surface inside a Ray task, where the
    exception type is erased.

    The two text encodings are the cases that matter: both reach Python as
    ``str``, so the read-oriented gate ``clip_uri`` gets would accept them.
    """
    uri = make_schema_table(structural_schema(CLIP_ID_COLUMN, unusable))

    with pytest.raises(ValueError, match=f"stores {CLIP_ID_COLUMN} as .*must be exactly non-nullable string"):
        require_source_columns(lance.dataset(uri))


def test_a_nullable_clip_id_is_refused(make_schema_table: SchemaTableFactory) -> None:
    """Require ``clip_id`` nullability to match the structural identity contract."""
    nullable_id = pa.schema(
        [
            pa.field(CLIP_ID_COLUMN, pa.string(), nullable=True),
            pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False),
        ]
    )

    with pytest.raises(ValueError, match=r"nullable=True.*must be exactly non-nullable string"):
        require_source_columns(lance.dataset(make_schema_table(nullable_id)))


def test_the_structural_columns_as_the_producer_writes_them_are_accepted(
    make_schema_table: SchemaTableFactory,
) -> None:
    """The counterfactual for the two refusals: ``string`` and ``large_string`` both pass.

    Worth pinning because the two columns are stored under DIFFERENT text
    encodings, so a check that accepted only ``string`` would refuse every real
    table while still passing a test that built both columns the same way.
    """
    require_source_columns(lance.dataset(make_schema_table(pa.schema(STRUCTURAL_FIELDS))))


def test_a_dictionary_encoded_clip_uri_is_accepted(make_schema_table: SchemaTableFactory) -> None:
    """A dictionary of text decodes to ``str`` on the way out, so it is text enough.

    The non-obvious accept: a low-cardinality column may reach the recipe
    dictionary-encoded, and refusing it would block a table whose values the
    prompt reads perfectly well. ``clip_id`` gets no such latitude, because the
    recipe joins on it rather than only reading it.
    """
    dictionary_encoded = structural_schema(CLIP_URI_COLUMN, pa.dictionary(pa.int32(), pa.string()))

    require_source_columns(lance.dataset(make_schema_table(dictionary_encoded)))


def test_an_absent_context_column_is_ignored(make_schema_table: SchemaTableFactory) -> None:
    """A supported context column the table omits narrows the run rather than failing it.

    Context is optional by contract: its absence is recorded in the digest, so
    those captions are honestly stamped as produced without it.

    The present column is taken from the shipped tuple rather than named, which
    keeps this asserting the relation - present resolves, absent does not -
    instead of pinning today's field list.
    """
    present, *absent = SUPPORTED_CONTEXT_FIELDS
    uri = make_schema_table(context_schema(present.column, pa.string()))

    resolved = resolve_context_fields(lance.dataset(uri))

    assert absent, "the relation under test is vacuous unless some supported column is absent"
    assert resolved == (present,)


def test_a_non_text_context_column_is_refused_without_mutating_the_table(
    make_schema_table: SchemaTableFactory,
) -> None:
    """Refuse a present context column with the wrong type.

    Treating it as absent would silently remove a prompt line from every row.
    The caller resolves context before schema widening, so the resolver must
    fail on the driver without advancing the dataset handle.
    """
    field, *_ = SUPPORTED_CONTEXT_FIELDS
    uri = make_schema_table(context_schema(field.column, pa.int64()))
    dataset = lance.dataset(uri)
    before = int(dataset.version)

    with pytest.raises(ValueError, match=rf"context column {field.column!r}.*non-text type int64"):
        resolve_context_fields(dataset)

    assert int(dataset.version) == before


def test_caption_registration_validates_context_before_mutating_the_table(
    make_schema_table: SchemaTableFactory,
) -> None:
    """Validate supported context fields before caption registration commits."""
    field, *_ = SUPPORTED_CONTEXT_FIELDS
    uri = make_schema_table(context_schema(field.column, pa.int64()))
    before = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match=rf"context column {field.column!r}.*non-text type int64"):
        register(uri)

    # Re-read: registration opens its own handle, so only a fresh one can show
    # whether the refusal came before or after a commit.
    after = lance.dataset(uri)
    assert int(after.version) == before
    assert set(after.schema.names).isdisjoint(CAPTION_COLUMNS)


def test_an_absent_context_column_is_skipped_without_a_warning(
    make_schema_table: SchemaTableFactory, loguru_records: list[dict[str, Any]]
) -> None:
    """Absence is a normal source shape, so warning about it would train operators to ignore it.

    A malformed present field is rejected instead; an absent optional field
    needs no operator action.
    """
    resolve_context_fields(lance.dataset(make_schema_table(pa.schema(STRUCTURAL_FIELDS))))

    assert warnings_of(loguru_records) == []


def test_a_table_carrying_no_context_column_resolves_an_empty_field_set(
    make_schema_table: SchemaTableFactory,
) -> None:
    """A table with none of the supported columns still runs, captioning from video alone."""
    assert resolve_context_fields(lance.dataset(make_schema_table(pa.schema(STRUCTURAL_FIELDS)))) == ()


def test_the_resolved_fields_keep_the_supported_order(make_schema_table: SchemaTableFactory) -> None:
    """Resolution follows the supported order, not the order the columns sit in the table.

    The resolved tuple is both the render order and a digest input, so were it to
    track the source schema's field order, two tables holding the same columns in
    a different order would caption differently under one digest.
    """
    reversed_in_table = pa.schema(
        [*STRUCTURAL_FIELDS, *[pa.field(field.column, pa.string()) for field in reversed(SUPPORTED_CONTEXT_FIELDS)]]
    )

    resolved = resolve_context_fields(lance.dataset(make_schema_table(reversed_in_table)))

    assert resolved == SUPPORTED_CONTEXT_FIELDS


def test_registration_adds_every_caption_field(make_clips_table: ClipsTableFactory) -> None:
    """A table without the caption columns gains all five in one commit.

    The version is read off the handle the call RETURNED rather than off a
    re-open, because ``pipeline.py`` pins its read version from that same object:
    the returned handle already sitting at the widened version is what makes the
    pinning structural, and a re-open would pass whether it did or not.
    """
    uri = make_clips_table()
    before = int(lance.dataset(uri).version)

    widened = register(uri)

    assert int(widened.version) == before + 1
    assert set(CAPTION_COLUMNS) <= set(widened.schema.names)


def test_registration_does_not_rewrite_data_files(make_clips_table: ClipsTableFactory) -> None:
    """Widening is metadata-only: no fragment's row count or data-file count changes."""
    uri = make_clips_table()
    before = [
        (fragment.fragment_id, fragment.count_rows(), len(fragment.data_files()))
        for fragment in lance.dataset(uri).get_fragments()
    ]

    register(uri)

    after = [
        (fragment.fragment_id, fragment.count_rows(), len(fragment.data_files()))
        for fragment in lance.dataset(uri).get_fragments()
    ]
    assert after == before


def test_registration_is_idempotent_without_a_version_bump(make_clips_table: ClipsTableFactory) -> None:
    """A second call adds nothing and commits nothing.

    A spurious bump would break the pinning contract outright: the first run
    would pin the version its widening created and its resume would pin a later
    one, landing the two in different recovery namespaces with nothing in common.
    """
    uri = make_clips_table()
    widened_version = int(register(uri).version)

    resumed = register(uri)

    assert int(resumed.version) == widened_version
    assert int(lance.dataset(uri).version) == widened_version


def test_registration_refuses_a_partially_present_field_set_without_a_version_bump(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Refuse a strict caption-field subset as an inconsistent published schema.

    Registration is all-or-none. Naming both sides makes the refusal actionable,
    and preserving the version proves the function did not repair or commit the
    partial group.
    """
    uri = make_clips_table()
    lance.dataset(uri).add_columns(pa.schema([CAPTION_FIELDS.field(CAPTION_STATUS_COLUMN)]))
    before = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match="partial caption column group") as exc_info:
        register(uri)

    message = str(exc_info.value)
    assert CAPTION_STATUS_COLUMN in message
    assert all(name in message for name in CAPTION_COLUMNS if name != CAPTION_STATUS_COLUMN)
    assert int(lance.dataset(uri).version) == before


def test_registration_refuses_a_mistyped_field_it_would_otherwise_widen_around(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Refuse a malformed strict subset as partial before any repair.

    The all-or-none state check takes precedence over member validation: a
    partial group is not a schema generation the recipe may top up, even when a
    present member is also mistyped. The version assertion pins refusal before
    any widening commit.
    """
    uri = make_clips_table()
    lance.dataset(uri).add_columns(pa.schema([pa.field(CAPTION_STATUS_COLUMN, pa.int64(), nullable=True)]))
    before = int(lance.dataset(uri).version)

    with pytest.raises(ValueError, match=rf"partial caption column group.*present.*{CAPTION_STATUS_COLUMN}.*missing"):
        register(uri)

    assert int(lance.dataset(uri).version) == before


def test_registration_rejects_an_incompatible_existing_field(make_clips_table: ClipsTableFactory) -> None:
    """A caption column stored at the wrong type is refused before any work runs.

    The match pins the stored type BESIDE the contract's, which is what makes the
    refusal actionable: told only which columns mismatched, an operator cannot
    see whether the difference is the type or the nullability.
    """
    uri = make_clips_table()
    lance.dataset(uri).add_columns(pa.schema([pa.field(name, pa.int64(), nullable=True) for name in CAPTION_COLUMNS]))

    with pytest.raises(ValueError, match=r"incompatible with the frozen contract.*int64.*contract: "):
        register(uri)


def test_registration_preserves_the_source_columns(make_clips_table: ClipsTableFactory) -> None:
    """Every source column keeps its name, type, and nullability across the widening."""
    uri = make_clips_table()
    register(uri)

    schema = lance.dataset(uri).schema
    assert [schema.field(field.name) for field in CLIPS_SOURCE_SCHEMA] == list(CLIPS_SOURCE_SCHEMA)


def test_a_lost_schema_race_is_resolved_by_reopening(
    make_clips_table: ClipsTableFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit that loses to a concurrent widener resolves on the next pass.

    This is the whole reason the function takes a URI instead of a handle: the
    losing job cannot retry from the handle that lost, because the group it needs
    exists only at a version that handle cannot see. The stub widens the table
    for real before raising, so the reopened pass finds a genuinely present group
    rather than a mock's say-so, and the returned handle must sit at THAT
    version - the one the winner created.
    """
    uri = make_clips_table()
    real_add_columns = lance.LanceDataset.add_columns

    def lose_once(self: lance.LanceDataset, *args: object, **kwargs: object) -> None:
        monkeypatch.setattr(lance.LanceDataset, "add_columns", real_add_columns)
        real_add_columns(self, *args, **kwargs)  # type: ignore[arg-type]  # Untyped in Lance stubs.
        msg = "commit conflict"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", lose_once)

    widened = register(uri)

    assert set(CAPTION_COLUMNS) <= set(widened.schema.names)
    assert int(widened.version) == int(lance.dataset(uri).version)


def test_a_deterministic_widening_failure_surfaces_once_the_attempts_are_spent(
    make_clips_table: ClipsTableFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A storage fault that never resolves is reported, not retried forever.

    The counterfactual to the test above, and what the bound buys: the same
    exception type, raised by a stub that never lets the commit through, must end
    the attempt with the cause attached rather than loop.
    """
    uri = make_clips_table()

    def always_fail(_self: lance.LanceDataset, *_args: object, **_kwargs: object) -> None:
        msg = "storage is unreachable"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", always_fail)

    with pytest.raises(RuntimeError, match="could not register caption fields") as exc_info:
        register(uri)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert set(lance.dataset(uri).schema.names).isdisjoint(CAPTION_COLUMNS)


def test_a_partial_group_found_after_the_attempts_are_spent_is_refused_as_partial(
    make_clips_table: ClipsTableFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A group that turns partial while the commits are failing is named partial.

    The loop's last look at the table is an ordinary classification pass, not a
    present-or-fail check, so the state it finds decides the refusal. Here every
    commit loses and a concurrent writer leaves a strict subset behind: reporting
    that as "still absent, so no concurrent writer registered it either" would
    send an operator looking for contention instead of for the half-written group
    that is actually blocking them.
    """
    uri = make_clips_table()
    real_add_columns = lance.LanceDataset.add_columns
    commits = 0

    def lose_every_commit(_self: lance.LanceDataset, *_args: object, **_kwargs: object) -> None:
        # The subset lands during the LAST commit pass, so the loop cannot see it
        # until the read-only pass. Planting it earlier would be caught by an
        # in-loop check that already existed and would prove nothing.
        nonlocal commits
        commits += 1
        if commits == _REGISTRATION_ATTEMPTS:
            real_add_columns(  # type: ignore[call-arg]  # Untyped in Lance stubs.
                lance.dataset(uri), pa.schema([CAPTION_FIELDS.field(CAPTION_STATUS_COLUMN)])
            )
        msg = "commit conflict"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "add_columns", lose_every_commit)

    with pytest.raises(ValueError, match=rf"partial caption column group.*{CAPTION_STATUS_COLUMN}"):
        register(uri)


def test_an_uncaptioned_row_is_pending(make_clips_table: ClipsTableFactory) -> None:
    """A row carrying no verdict at all is owed a caption under any contract."""
    uri = make_clips_table(rows=7, statuses=[None, None, None, None, "success", None, None])

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 6


def test_selected_only_work_filter_owes_only_selected_pending_rows(make_clips_table: ClipsTableFactory) -> None:
    """Intersect caption completion with Curate's final selected verdict."""
    selection = [
        CURATE_SELECTED_VALUE,
        "duplicate",
        "below_quota",
        "unfunded",
        "invalid_embedding",
        None,
        CURATE_SELECTED_VALUE,
    ]
    uri = make_clips_table(rows=7, statuses=[None, None, None, None, None, None, "success"], selection=selection)

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=True)

    assert lance.dataset(uri).count_rows(filter=row_filter) == 1


def test_unscoped_work_filter_owes_every_pending_selection_value(make_clips_table: ClipsTableFactory) -> None:
    """Disabling selection makes every Curate verdict and NULL eligible."""
    selection = [
        CURATE_SELECTED_VALUE,
        "duplicate",
        "below_quota",
        "unfunded",
        "invalid_embedding",
        None,
    ]
    uri = make_clips_table(rows=6, statuses=[None] * 6, selection=selection)

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)

    assert lance.dataset(uri).count_rows(filter=row_filter) == 6


def test_selected_only_preflight_refuses_an_absent_selection_column(
    make_clips_table: ClipsTableFactory,
) -> None:
    """Refuse before caption widening when the default scope cannot be evaluated."""
    dataset = lance.dataset(make_clips_table(omit_selection=True))
    initial_version = int(dataset.version)

    with pytest.raises(ValueError, match=r"selected-only captioning requires column.*selected_only=false"):
        caption_table.require_selection_column(dataset, selected_only=True)

    assert int(dataset.version) == initial_version


@pytest.mark.parametrize("wrong_type", [pa.bool_(), pa.int32()])
def test_selected_only_preflight_refuses_a_retyped_selection_column(
    make_schema_table: SchemaTableFactory,
    wrong_type: pa.DataType,
) -> None:
    """Require the producer's exact string contract at the input boundary."""
    schema = pa.schema([*STRUCTURAL_FIELDS, pa.field(CURATE_SELECTION_REASON, wrong_type)])
    dataset = lance.dataset(make_schema_table(schema))

    with pytest.raises(ValueError, match=rf"{CURATE_SELECTION_REASON}.*exactly string.*{wrong_type}"):
        caption_table.require_selection_column(dataset, selected_only=True)


def test_the_selection_contract_matches_the_producer_that_writes_it() -> None:
    """Pin the locally declared read contract against Curate's own declarations.

    ``table.py`` restates the column name, the selected value, and the stored
    type instead of importing them, so this is the only place drift between the
    two recipes can be caught. The type is asserted because the preflight
    demands exactly ``string``, which is a claim about the producer's schema.
    """
    assert CURATE_SELECTION_REASON == PRODUCER_SELECTION_REASON
    assert CurateReason.SELECTED.value == CURATE_SELECTED_VALUE
    assert CURATE_COLUMNS.field(PRODUCER_SELECTION_REASON).type.equals(pa.string())


def test_unscoped_preflight_needs_no_selection_column(make_clips_table: ClipsTableFactory) -> None:
    """Allow an explicit all-pending run against a pre-curation table."""
    caption_table.require_selection_column(
        lance.dataset(make_clips_table(omit_selection=True)),
        selected_only=False,
    )


def test_a_current_digest_row_is_not_pending(make_clips_table: ClipsTableFactory) -> None:
    """A row already captioned under this run's contract is not owed another caption.

    The predicate's whole purpose: a resume must skip these, or every re-run
    re-captions the corpus.
    """
    uri = make_clips_table(rows=3, statuses=["success"] * 3, digests=[CURRENT_DIGEST] * 3)

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 0


@pytest.mark.parametrize(
    "stored_model",
    [
        pytest.param(None, id="null"),
        pytest.param("another/model", id="wrong"),
    ],
)
def test_a_current_row_without_the_expected_model_is_pending(
    make_clips_table: ClipsTableFactory,
    stored_model: str | None,
) -> None:
    """A current digest cannot make a missing or foreign model identity complete."""
    uri = make_clips_table(rows=1, statuses=["success"], models=[stored_model])

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 1


def test_the_expected_model_is_escaped_as_a_sql_literal(make_clips_table: ClipsTableFactory) -> None:
    """A quote in a model id stays data and cannot change the pending predicate."""
    expected_model = "publisher/model' OR 1=1 --"
    uri = make_clips_table(
        rows=2,
        statuses=["success", "success"],
        models=[expected_model, "another/model"],
    )

    assert (
        lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, expected_model, selected_only=False))
        == 1
    )


@pytest.mark.parametrize(
    ("status", "caption", "error"),
    [
        pytest.param("unknown", "caption", None, id="unsupported-status"),
        pytest.param("success", None, None, id="success-null-caption"),
        pytest.param("success", " \t", None, id="success-blank-caption"),
        pytest.param("success", "caption", "failure", id="success-with-error"),
        pytest.param("truncated", None, None, id="truncated-null-caption"),
        pytest.param("truncated", "\n ", None, id="truncated-blank-caption"),
        pytest.param("truncated", "caption", "failure", id="truncated-with-error"),
        pytest.param("error", "caption", "failure", id="error-with-caption"),
        pytest.param("error", None, None, id="error-null-error"),
        pytest.param("error", None, " \t", id="error-blank-error"),
    ],
)
def test_a_malformed_current_terminal_group_is_pending(
    make_clips_table: ClipsTableFactory,
    status: str,
    caption: str | None,
    error: str | None,
) -> None:
    """A current digest cannot hide terminal values publication would refuse."""
    uri = make_clips_table(rows=1, statuses=[status], captions=[caption], errors=[error])

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 1


def test_a_stale_digest_row_is_pending_again(make_clips_table: ClipsTableFactory) -> None:
    """Latest contract wins: a caption produced under a different contract is re-owed.

    The same table shape as the row above and the opposite verdict, so this pins
    the digest comparison itself rather than the presence of a digest.
    """
    uri = make_clips_table(rows=3, statuses=["success"] * 3, digests=[STALE_DIGEST] * 3)

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 3


def test_a_status_row_without_a_digest_is_pending(make_clips_table: ClipsTableFactory) -> None:
    """A published verdict carrying a NULL digest is pending, not silently retained.

    The ``digest IS NULL`` arm of the predicate is load-bearing rather than
    defensive. Under SQL three-valued logic ``NULL != 'x'`` evaluates to NULL and
    not to true, so the inequality alone would never select this row and it would
    stay uncaptionable forever. The status is ``success`` here precisely so the
    ``status IS NULL`` arm cannot be what selects it.
    """
    uri = make_clips_table(rows=3, statuses=["success"] * 3, digests=[None] * 3)

    assert lance.dataset(uri).count_rows(filter=caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)) == 3


def test_a_malformed_digest_is_refused_before_it_reaches_sql(make_clips_table: ClipsTableFactory) -> None:
    """The digest is validated, not escaped, which is what makes interpolating it safe.

    The accepted alphabet - 64 lowercase hex characters - cannot express a quote
    or an operator, so a value that passes validation cannot alter the
    predicate's structure. A value that does not pass must therefore be refused
    rather than interpolated.
    """
    uri = make_clips_table(rows=3, statuses=["success"] * 3)

    with pytest.raises(ValueError, match="64-character lowercase hex sha256"):
        lance.dataset(uri).count_rows(filter=caption_work_filter("' OR 1=1 --", MODEL_ID, selected_only=False))


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param(CURRENT_DIGEST.upper(), id="uppercase-hex"),
        pytest.param(CURRENT_DIGEST[:-1], id="too-short"),
        pytest.param(CURRENT_DIGEST + "a", id="too-long"),
        pytest.param(f"{CURRENT_DIGEST[:-1]}\n", id="trailing-newline"),
    ],
)
def test_a_digest_of_the_wrong_shape_is_refused(malformed: str) -> None:
    """Only an exact 64-character lowercase hex string is accepted.

    Uppercase is refused rather than folded, so one contract cannot reach the
    column under two spellings; the length and newline cases pin that the match
    is anchored at both ends rather than merely searching.
    """
    with pytest.raises(ValueError, match="64-character lowercase hex sha256"):
        caption_work_filter(malformed, MODEL_ID, selected_only=False)


def test_owed_clip_ids_excludes_a_captioned_row(make_clips_table: ClipsTableFactory) -> None:
    """A published clip drops out of its own fragment's pending list."""
    uri = make_clips_table(rows=7, rows_per_file=3, statuses=[None, None, None, None, "success", None, None])

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
    assert owed_clip_ids(lance.dataset(uri), 1, row_filter) == ("c3", "c5")


def test_owed_clip_ids_re_lists_a_row_captioned_under_another_contract(
    make_clips_table: ClipsTableFactory,
) -> None:
    """The per-fragment read plan follows the digest too, not just the presence of a status.

    Asserted at the clip-id level as well as at the row-count level because this
    is the value the engine is actually fed: a fragment counted as pending but
    read back without its stale clip would caption nothing.
    """
    uri = make_clips_table(
        rows=7,
        rows_per_file=3,
        statuses=[None, None, None, None, "success", None, None],
        digests=[None, None, None, None, STALE_DIGEST, None, None],
    )

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
    assert owed_clip_ids(lance.dataset(uri), 1, row_filter) == ("c3", "c4", "c5")


def test_owed_clip_ids_is_empty_for_a_fully_captioned_fragment(make_clips_table: ClipsTableFactory) -> None:
    """Every row published means no pending clip, not a partial list."""
    uri = make_clips_table(rows=7, rows_per_file=3, statuses=["success", "truncated", "error", *[None] * 4])

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
    assert owed_clip_ids(lance.dataset(uri), 0, row_filter) == ()


def test_owed_clip_ids_refuses_an_unregistered_table(make_clips_table: ClipsTableFactory) -> None:
    """Querying before registration names the missing fields rather than failing inside SQL."""
    dataset = lance.dataset(make_clips_table())

    with pytest.raises(ValueError, match="are not present on"):
        owed_clip_ids(dataset, 0, caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False))


def test_owed_clip_ids_rejects_an_absent_fragment(make_clips_table: ClipsTableFactory) -> None:
    """A fragment id the pinned version does not carry is an error, not an empty list."""
    uri = make_clips_table(rows=7, rows_per_file=3)
    register(uri)

    with pytest.raises(ValueError, match="is not present in"):
        owed_clip_ids(
            lance.dataset(uri),
            99,
            caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False),
        )


def test_a_scoped_count_ignores_the_pending_rows_of_fragments_it_was_not_given(
    make_clips_table: ClipsTableFactory,
) -> None:
    """The count covers exactly the fragments named, never the table.

    Built adversarially: every pending row sits in the fragment NOT counted, and
    the counted fragment's rows are all captioned. A table-wide count would
    report 3 here and, in the caller that uses this, would blame Curate for a
    backlog belonging to another job - sending an operator to a remedy that
    changes nothing.
    """
    uri = make_clips_table(
        rows=6,
        rows_per_file=3,
        statuses=["success", "success", "success", None, None, None],
    )
    register(uri)
    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
    dataset = lance.dataset(uri)

    assert count_rows_in_fragments(dataset, [0], row_filter) == 0
    # The pending rows do exist, which is what stops the zero above being a
    # property of the fixture rather than of the scoping.
    assert count_rows_in_fragments(dataset, [1], row_filter) == 3
    assert count_rows_in_fragments(dataset, [0, 1], row_filter) == 3


def test_a_scoped_count_over_no_fragments_is_zero(make_clips_table: ClipsTableFactory) -> None:
    """A shard owning nothing counts nothing, rather than falling back to the table."""
    uri = make_clips_table(rows=3, rows_per_file=3)
    register(uri)

    assert (
        count_rows_in_fragments(
            lance.dataset(uri), [], caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
        )
        == 0
    )


def test_a_scoped_count_rejects_an_absent_fragment(make_clips_table: ClipsTableFactory) -> None:
    """A fragment id this version does not carry is an error, not a silent zero.

    A silent zero would read exactly like "this shard owes nothing", which is the
    verdict the caller acts on.
    """
    uri = make_clips_table(rows=3, rows_per_file=3)
    register(uri)

    with pytest.raises(ValueError, match="is not present in"):
        count_rows_in_fragments(
            lance.dataset(uri),
            [0, 99],
            caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False),
        )


@pytest.mark.parametrize(
    ("invalid_clip_id", "clip_id_type"),
    [
        pytest.param(None, pa.string(), id="null"),
        pytest.param("", pa.string(), id="empty"),
        pytest.param(" \t", pa.string(), id="whitespace"),
        pytest.param(7, pa.int64(), id="non-string"),
    ],
)
def test_owed_clip_ids_rejects_an_invalid_logical_key(
    tmp_path: pathlib.Path,
    invalid_clip_id: object,
    clip_id_type: pa.DataType,
) -> None:
    """Refuse malformed keys at the fragment boundary without normalization."""
    uri = str(tmp_path / f"invalid-{type(invalid_clip_id).__name__}.lance")
    schema = pa.schema(
        [
            pa.field(CLIP_ID_COLUMN, clip_id_type, nullable=invalid_clip_id is None),
            pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False),
            *CAPTION_FIELDS,
        ]
    )
    columns = {
        CLIP_ID_COLUMN: pa.array([invalid_clip_id], clip_id_type),
        CLIP_URI_COLUMN: pa.array(["s3://b/clip.mp4"], pa.large_string()),
        **{field.name: pa.nulls(1, field.type) for field in CAPTION_FIELDS},
    }
    lance.write_dataset(pa.table(columns, schema=schema), uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)

    with pytest.raises(ValueError, match="clip_id must be a nonblank string"):
        owed_clip_ids(
            lance.dataset(uri),
            0,
            caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False),
        )


def test_owed_clip_ids_preserves_a_nonblank_key_byte_for_byte(tmp_path: pathlib.Path) -> None:
    """Preserve whitespace around a nonblank logical key byte-for-byte."""
    uri = str(tmp_path / "nonblank-key.lance")
    schema = pa.schema(
        [
            pa.field(CLIP_ID_COLUMN, pa.string(), nullable=False),
            pa.field(CLIP_URI_COLUMN, pa.large_string(), nullable=False),
            *CAPTION_FIELDS,
        ]
    )
    columns = {
        CLIP_ID_COLUMN: pa.array(["  c0  "], pa.string()),
        CLIP_URI_COLUMN: pa.array(["s3://b/clip.mp4"], pa.large_string()),
        **{field.name: pa.nulls(1, field.type) for field in CAPTION_FIELDS},
    }
    lance.write_dataset(pa.table(columns, schema=schema), uri, data_storage_version=LANCE_DATA_STORAGE_VERSION)

    row_filter = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)
    assert owed_clip_ids(lance.dataset(uri), 0, row_filter) == ("  c0  ",)


def test_the_pending_predicate_names_the_digest_column_bare(make_clips_table: ClipsTableFactory) -> None:
    """The predicate keeps identifiers unquoted, which the Lance filter surface requires.

    On this surface a double-quoted name parses as a STRING LITERAL, so a quoted
    identifier would compare two constants and match every row instead of
    raising. Pinned as text because the failure mode is silent over-selection: a
    quoted predicate still executes, so every row-count assertion above would
    keep passing while the run re-captioned the whole corpus.
    """
    predicate = caption_work_filter(CURRENT_DIGEST, MODEL_ID, selected_only=False)

    assert f'"{CAPTION_DIGEST_COLUMN}"' not in predicate
    assert f"{CAPTION_DIGEST_COLUMN} IS NULL" in predicate
    assert lance.dataset(make_clips_table(rows=3, statuses=["success"] * 3)).count_rows(filter=predicate) == 0
