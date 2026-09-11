# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase B guards and the single atomic commit, against real Lance and real Parquet.

Every guard here exists because ``update_columns`` cannot report the fault
itself: it preserves unmatched left rows and silently drops unmatched right ones,
so a wave that is missing a clip, carries an extra one, or was generated under
other semantics commits without complaint. The tests therefore assert the
REFUSAL, and the commit tests assert that the table is untouched when one fires.

The one exception is the ambiguous commit, where the error carries no verdict at
all: Lance raises the same way whether the manifest was written or not. Those
tests drive both outcomes through a faked commit and assert that canonical table
state plus this attempt's durable marker, not the error, decides what happened.

``prepare_fragment`` is exercised directly rather than through the Ray fan-out:
the fan-out only decodes payloads, while the whole cardinality contract lives in
the function it calls.
"""

import inspect
import pathlib
import re
from collections.abc import Callable
from typing import Any
from uuid import UUID

import lance
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from cosmos_curator.next.recipes.robot_action_caption import contracts as caption_contracts
from cosmos_curator.next.recipes.robot_action_caption import publication as publication_module
from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMNS,
    CAPTION_FIELDS,
    CaptionStatus,
)
from cosmos_curator.next.recipes.robot_action_caption.publication import (
    CaptionPublicationConflictError,
    CaptionPublicationError,
    commit_prepared,
    fold_staged_rows,
    prepare_fragment,
    publish_captions,
    require_exact_coverage,
    staged_captions,
)
from cosmos_curator.next.recipes.robot_action_caption.request import STAGED_SCHEMA
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CURATE_SELECTED_VALUE,
    CURATE_SELECTION_REASON,
    caption_work_filter,
    owed_clip_ids,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import CaptionWorkspace
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

# A 64-character lowercase hex sha256 rather than a readable label: everything
# reached from ``prepare_fragment`` runs the digest through ``caption_work_filter``,
# which refuses any other shape before it interpolates it into SQL.
DIGEST = "a" * 64
FOREIGN_DIGEST = "b" * 64
MODEL = "org/model"
ROW_FILTER = caption_work_filter(DIGEST, MODEL, selected_only=False)
SELECTED_ROW_FILTER = caption_work_filter(DIGEST, MODEL, selected_only=True)

# The staging namespace the guards name in their refusals. A literal rather than the
# ``workspace`` fixture's path because these guards are pure over rows already read:
# they take the namespace as a string precisely so testing them needs no filesystem.
NAMESPACE_URI = "s3://staging/inference/" + "c" * 64

CLIPS_SCHEMA = pa.schema(
    [
        pa.field("clip_id", pa.string(), nullable=False),
        pa.field("clip_uri", pa.large_string(), nullable=False),
        *CAPTION_FIELDS,
        pa.field(CURATE_SELECTION_REASON, pa.string(), nullable=True),
    ]
)

# One staged partition's columns: the schema Phase A writes, minus the Hive
# partition key and the checkpoint id Parquet keeps but publication never reads.
_PARTITION_SCHEMA = pa.schema([STAGED_SCHEMA.field("clip_id"), *CAPTION_FIELDS])


StagedRowFactory = Callable[..., dict[str, object]]
StagedShardWriter = Callable[[int, list[dict[str, object]]], None]
# Stages and prepares the named fragments, returning their payloads; see ``prepare``.
WaveFactory = Callable[..., list[dict[str, object]]]


@pytest.fixture
def staged_row() -> StagedRowFactory:
    """Return a builder for one successful terminal staged row.

    Keyword arguments vary one durable value at a time for publication refusal
    tests. A ``None`` digest is the row a foreign staging tree can hold, so it is
    supported rather than a type violation at this fixture boundary.
    """

    def build(
        clip_id: str,
        *,
        caption: object = "a caption",
        status: object = CaptionStatus.SUCCESS,
        error: object = None,
        digest: str | None = DIGEST,
        model_id: object = MODEL,
    ) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "caption": caption,
            "caption_status": status,
            "caption_error": error,
            "caption_contract_digest": digest,
            "caption_model": model_id,
        }

    return build


@pytest.fixture
def staged_error_row() -> StagedRowFactory:
    """Return a builder for one terminal ERROR staged row.

    Null caption text beside a non-empty error is the shape ``terminal_row``
    enforces for this verdict, so a counting test reads the row publication would
    really be handed rather than a success row relabelled.
    """

    def build(clip_id: str) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "caption": None,
            "caption_status": CaptionStatus.ERROR,
            "caption_error": "MediaObjectError: media object is missing",
            "caption_contract_digest": DIGEST,
            "caption_model": MODEL,
        }

    return build


@pytest.fixture
def staged_truncated_row() -> StagedRowFactory:
    """Return a builder for one terminal TRUNCATED staged row.

    Non-empty caption text beside a null error is the shape ``terminal_row``
    enforces for this verdict, and it is the same shape a success row carries -
    which is precisely why a count is the only thing that tells the two apart.
    """

    def build(clip_id: str) -> dict[str, object]:
        return {
            "clip_id": clip_id,
            "caption": "a caption that stops mid-",
            "caption_status": CaptionStatus.TRUNCATED,
            "caption_error": None,
            "caption_contract_digest": DIGEST,
            "caption_model": MODEL,
        }

    return build


@pytest.fixture
def clips(tmp_path: pathlib.Path) -> lance.LanceDataset:
    """Write a two-fragment widened clip table whose every row is pending."""
    rows = 4
    uri = str(tmp_path / "clips.lance")
    lance.write_dataset(
        pa.table(
            {
                "clip_id": pa.array([f"c{i}" for i in range(rows)], pa.string()),
                "clip_uri": pa.array([f"s3://b/c{i}.mp4" for i in range(rows)], pa.large_string()),
                **{field.name: pa.array([None] * rows, field.type) for field in CAPTION_FIELDS},
                CURATE_SELECTION_REASON: pa.array([CURATE_SELECTED_VALUE] * rows, pa.string()),
            },
            schema=CLIPS_SCHEMA,
        ),
        uri,
        max_rows_per_file=2,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    return lance.dataset(uri)


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> CaptionWorkspace:
    """Return a local workspace whose results tree the tests populate by hand.

    Constructed directly rather than through ``resolve_workspace``: publication
    reads a path and never re-derives it, so the namespace digest is irrelevant
    here and a readable directory keeps a failure message legible.
    """
    return CaptionWorkspace(root_uri=str(tmp_path / "ws"), filesystem=pafs.LocalFileSystem())


@pytest.mark.parametrize(
    ("status", "caption", "error", "expected"),
    [
        (CaptionStatus.SUCCESS, "complete", None, CaptionStatus.SUCCESS),
        ("truncated", "cut off", None, CaptionStatus.TRUNCATED),
        (CaptionStatus.ERROR, None, "media missing", CaptionStatus.ERROR),
        ("error", None, "media missing", CaptionStatus.ERROR),
    ],
)
def test_terminal_values_accept_enum_and_stored_string_statuses(
    status: object,
    caption: object,
    error: object,
    expected: CaptionStatus,
) -> None:
    """Stored strings and in-process enum members normalize to one terminal enum."""
    assert caption_contracts.validate_terminal_values(status=status, caption=caption, error=error) is expected


def test_terminal_values_reject_unknown_string_status() -> None:
    """A stored string outside the three persisted statuses is not terminal."""
    with pytest.raises(ValueError, match="terminal caption status"):
        caption_contracts.validate_terminal_values(status="unknown", caption="caption", error=None)


@pytest.mark.parametrize("status", [None, 1])
def test_terminal_values_reject_non_string_statuses(status: object) -> None:
    """A stored terminal status must be text rather than an arbitrary scalar."""
    with pytest.raises(ValueError, match="terminal caption status"):
        caption_contracts.validate_terminal_values(status=status, caption="caption", error=None)


def test_error_terminal_values_require_a_null_caption() -> None:
    """An error verdict cannot publish caption text beside its failure."""
    with pytest.raises(ValueError, match="null caption"):
        caption_contracts.validate_terminal_values(
            status=CaptionStatus.ERROR,
            caption="contradictory text",
            error="media missing",
        )


@pytest.mark.parametrize("error", [None, 7, "", " \t"])
def test_error_terminal_values_require_a_nonblank_string_error(error: object) -> None:
    """An error verdict must carry a human-readable failure rather than blank data."""
    with pytest.raises(ValueError, match="nonblank string error"):
        caption_contracts.validate_terminal_values(status=CaptionStatus.ERROR, caption=None, error=error)


@pytest.mark.parametrize("status", [CaptionStatus.SUCCESS, CaptionStatus.TRUNCATED])
@pytest.mark.parametrize("caption", [None, 7, "", " \t"])
def test_caption_terminal_values_require_nonblank_string_text(status: CaptionStatus, caption: object) -> None:
    """Successful and truncated verdicts must both carry meaningful caption text."""
    with pytest.raises(ValueError, match="nonblank string caption"):
        caption_contracts.validate_terminal_values(status=status, caption=caption, error=None)


@pytest.mark.parametrize("status", [CaptionStatus.SUCCESS, CaptionStatus.TRUNCATED])
def test_caption_terminal_values_require_a_null_error(status: CaptionStatus) -> None:
    """Successful and truncated verdicts cannot also carry an error."""
    with pytest.raises(ValueError, match="null error"):
        caption_contracts.validate_terminal_values(status=status, caption="caption", error="contradictory failure")


def test_terminal_row_uses_the_shared_terminal_validator() -> None:
    """The in-process row builder must reject the same whitespace-only text as publication."""
    with pytest.raises(ValueError, match="nonblank string caption"):
        caption_contracts.terminal_row(
            status=CaptionStatus.SUCCESS,
            digest=DIGEST,
            model_id=MODEL,
            caption=" \t",
        )


def test_publication_signatures_match_the_frozen_interface() -> None:
    """Publication threads model identity through preparation and commit verification."""
    assert tuple(inspect.signature(publish_captions).parameters) == (
        "dataset",
        "workspace",
        "fragment_ids",
        "digest",
        "model_id",
        "row_filter",
        "storage_options",
    )
    assert tuple(inspect.signature(staged_captions).parameters) == ("workspace", "fragment_id")
    assert tuple(inspect.signature(prepare_fragment).parameters) == (
        "dataset",
        "workspace",
        "fragment_id",
        "digest",
        "model_id",
        "row_filter",
    )
    assert tuple(inspect.signature(commit_prepared).parameters) == (
        "dataset",
        "payloads",
        "expected_fragment_ids",
        "row_filter",
        "storage_options",
    )


@pytest.fixture
def stage(workspace: CaptionWorkspace) -> StagedShardWriter:
    """Return a writer placing one fragment's staged rows as a Hive-partitioned shard."""

    def write(fragment_id: int, rows: list[dict[str, object]]) -> None:
        partition = pathlib.Path(workspace.results_uri) / f"fragment_id={fragment_id}"
        partition.mkdir(parents=True, exist_ok=True)
        shard = partition / f"part-{len(list(partition.iterdir()))}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=_PARTITION_SCHEMA), shard)

    return write


@pytest.fixture
def prepare(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> WaveFactory:
    """Return a builder that stages one terminal row per owed clip and prepares the fragments.

    Which clips a fragment owes is read from the fragment itself rather than
    restated here, so a commit test asserts against the same owed set
    publication will validate against.

    ``row`` chooses the verdict staged, and defaults to a successful caption,
    which is the shape every guard test needs.
    """

    def build(*fragment_ids: int, row: StagedRowFactory | None = None) -> list[dict[str, object]]:
        build_row = row if row is not None else staged_row
        for fragment_id in fragment_ids:
            stage(
                fragment_id,
                [build_row(clip_id) for clip_id in owed_clip_ids(clips, fragment_id, ROW_FILTER)],
            )
        return [
            prepare_fragment(
                clips,
                workspace,
                fragment_id,
                digest=DIGEST,
                model_id=MODEL,
                row_filter=ROW_FILTER,
            )
            for fragment_id in fragment_ids
        ]

    return build


def test_an_absent_partition_reads_as_no_staged_row(workspace: CaptionWorkspace) -> None:
    """A fragment Phase A never wrote reads as empty, not as a missing-directory error."""
    assert staged_captions(workspace, 0).num_rows == 0


def test_an_empty_partition_reads_as_no_staged_row(workspace: CaptionWorkspace) -> None:
    """A partition created before its shard is written reads as empty."""
    partition = pathlib.Path(workspace.results_uri) / "fragment_id=0"
    partition.mkdir(parents=True)

    assert staged_captions(workspace, 0).num_rows == 0


def test_staged_reads_use_the_workspace_filesystem(
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Publication exposes no second filesystem resolver while workspace reads succeed."""
    stage(0, [staged_row("c0")])

    assert "arrow_filesystem" not in vars(publication_module)
    assert staged_captions(workspace, 0).num_rows == 1


@pytest.mark.parametrize(
    ("dependency", "error", "message"),
    [
        ("staged_captions", OSError("staged object disappeared"), "could not read staged captions"),
        ("staged_captions", pa.ArrowInvalid("truncated footer"), "could not read staged captions"),
        ("owed_clip_ids", ValueError("invalid stored clip id"), "could not determine owed clips"),
    ],
)
def test_prepare_converts_known_data_failures_to_typed_refusals(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
    monkeypatch: pytest.MonkeyPatch,
    dependency: str,
    error: Exception,
    message: str,
) -> None:
    """Known staged-storage and owed-row faults become fragment-named refusals."""
    owed = owed_clip_ids(clips, 0, ROW_FILTER)
    stage(0, [staged_row(clip_id) for clip_id in owed])

    def fail_dependency(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(publication_module, dependency, fail_dependency)

    with pytest.raises(CaptionPublicationError, match=rf"fragment 0.*{message}"):
        prepare_fragment(clips, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)


def test_identical_duplicate_staged_rows_fold_to_one(staged_row: StagedRowFactory) -> None:
    """A replayed shard writes the same answer twice, which is one answer."""
    staged = pa.Table.from_pylist([staged_row("c0"), staged_row("c0")], schema=_PARTITION_SCHEMA)

    assert fold_staged_rows(staged, fragment_id=0, namespace_uri=NAMESPACE_URI) == {"c0": staged_row("c0")}


@pytest.mark.parametrize("clip_id", [None, 7, "", " \t"])
def test_invalid_staged_clip_ids_are_refused_without_coercion(clip_id: object) -> None:
    """A staged logical key must already be a nonblank string at the publication boundary."""
    staged = pa.table({"clip_id": [clip_id]})

    with pytest.raises(
        CaptionPublicationError,
        match=rf"fragment 7.*invalid clip_id {re.escape(repr(clip_id))}",
    ):
        fold_staged_rows(staged, fragment_id=7, namespace_uri=NAMESPACE_URI)


def test_nonblank_staged_clip_id_is_not_stripped() -> None:
    """Whitespace around a nonblank identity remains part of the exact stored key."""
    staged = pa.table({"clip_id": [" c0 "]})

    assert fold_staged_rows(staged, fragment_id=0, namespace_uri=NAMESPACE_URI) == {" c0 ": {"clip_id": " c0 "}}


def test_conflicting_duplicate_staged_rows_are_refused(staged_row: StagedRowFactory) -> None:
    """Two differing captions for one clip are two answers, and the refusal names the way out.

    The situation does not clear itself: every input to the namespace digest is
    stable across re-runs, so a later attempt reads the same disagreeing rows and
    refuses again until that directory is deleted. The pattern spans both halves of
    the message so it can only pass on the conflict refusal AND only while that
    refusal still names the namespace.
    """
    staged = pa.Table.from_pylist(
        [staged_row("c0", caption="first"), staged_row("c0", caption="second")],
        schema=_PARTITION_SCHEMA,
    )

    with pytest.raises(
        CaptionPublicationError,
        match=rf"conflicting staged caption rows.*delete {re.escape(NAMESPACE_URI)} before re-running",
    ):
        fold_staged_rows(staged, fragment_id=0, namespace_uri=NAMESPACE_URI)


def test_a_clip_id_owed_twice_is_refused(staged_row: StagedRowFactory) -> None:
    """A repeated owed id would give the update join two right-side rows for one key.

    Validation reads the owed clips as a SET while the update table is built
    from the tuple, so a repeat survives every other guard here and reaches
    ``update_columns``, which would publish whichever of the two it picked.
    """
    folded = {"c0": staged_row("c0")}

    with pytest.raises(CaptionPublicationError, match=r"2 owed clip\(s\) but only 1 distinct"):
        require_exact_coverage(folded, ("c0", "c0"), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_a_staged_row_carrying_no_digest_is_named_as_null(staged_row: StagedRowFactory) -> None:
    """A null digest is reported as ``<null>``, not as a "None" that reads like a real one.

    This is the message an operator uses to identify a foreign staging tree, so
    the difference between "carries no digest" and "carries a digest spelled
    None" is the whole diagnosis.
    """
    folded = {"c0": staged_row("c0", digest=None)}

    with pytest.raises(CaptionPublicationError, match=r"digest\(s\) \['<null>'\] instead of"):
        require_exact_coverage(folded, ("c0",), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_an_owed_clip_without_a_staged_row_is_refused(staged_row: StagedRowFactory) -> None:
    """An unfinished clip must stop the wave rather than publish a partial fragment."""
    folded = {"c0": staged_row("c0")}

    with pytest.raises(CaptionPublicationError, match="no staged caption row"):
        require_exact_coverage(folded, ("c0", "c1"), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_a_staged_clip_that_is_not_owed_is_refused(staged_row: StagedRowFactory) -> None:
    """update_columns drops an unmatched staged row silently, so it is caught here."""
    folded = {"c0": staged_row("c0"), "c9": staged_row("c9")}

    with pytest.raises(CaptionPublicationError, match="does not owe"):
        require_exact_coverage(folded, ("c0",), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_a_foreign_contract_digest_is_refused(staged_row: StagedRowFactory) -> None:
    """A row generated under other semantics is not this run's to publish.

    The refusal rests on the namespace digest covering the contract, so it names
    the namespace it read: that claim is unverifiable against an unnamed path, and
    which path is foreign is what an operator needs to tell a stray writer from two
    contracts sharing one staging root.
    """
    folded = {"c0": staged_row("c0", digest=FOREIGN_DIGEST)}

    with pytest.raises(CaptionPublicationError, match=rf"instead of.*{re.escape(NAMESPACE_URI)} can hold rows"):
        require_exact_coverage(folded, ("c0",), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_exact_coverage_under_this_contract_is_accepted(staged_row: StagedRowFactory) -> None:
    """One current-contract row per owed clip is the only shape that passes."""
    folded = {"c0": staged_row("c0"), "c1": staged_row("c1")}

    require_exact_coverage(folded, ("c0", "c1"), fragment_id=0, digest=DIGEST, namespace_uri=NAMESPACE_URI)


def test_preparing_a_fragment_writes_no_visible_caption(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Preparation writes data files and rebinds field ids; only the commit publishes."""
    stage(0, [staged_row("c0"), staged_row("c1")])

    payload = prepare_fragment(clips, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)

    assert payload["rows"] == 2
    assert lance.dataset(clips.uri).count_rows(filter="caption_status IS NULL") == 4


def test_unselected_pending_rows_do_not_count_as_missing_staged_rows(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Validate coverage against this attempt's selected scope, not global pending rows."""
    clips.update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where="clip_id = 'c1'",
    )
    stage(0, [staged_row("c0")])

    payload = prepare_fragment(
        clips,
        workspace,
        0,
        digest=DIGEST,
        model_id=MODEL,
        row_filter=SELECTED_ROW_FILTER,
    )

    assert payload["rows"] == 1


def test_a_staged_unselected_row_is_refused_as_extra(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Keep the guard that prevents a foreign scope from publishing silently."""
    clips.update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where="clip_id = 'c1'",
    )
    stage(0, [staged_row("c0"), staged_row("c1")])

    with pytest.raises(CaptionPublicationError, match=r"does not owe.*c1"):
        prepare_fragment(
            clips,
            workspace,
            0,
            digest=DIGEST,
            model_id=MODEL,
            row_filter=SELECTED_ROW_FILTER,
        )


def test_preparing_an_absent_fragment_is_refused(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
) -> None:
    """A fragment id the pinned version does not carry is a refusal, not an empty prepare."""
    with pytest.raises(CaptionPublicationError, match="is not present in"):
        prepare_fragment(clips, workspace, 99, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)


@pytest.mark.parametrize(
    ("status", "caption", "error", "message", "malformed_index"),
    [
        ("unknown", "caption", None, "terminal caption status", -1),
        (CaptionStatus.ERROR, "contradictory text", "media missing", "null caption", 0),
        (CaptionStatus.ERROR, None, " \t", "nonblank string error", 0),
        (CaptionStatus.SUCCESS, " \t", None, "nonblank string caption", 0),
        (CaptionStatus.TRUNCATED, "cut off", "contradictory failure", "null error", 0),
    ],
)
def test_malformed_terminal_rows_are_refused_before_commit(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
    status: object,
    caption: object,
    error: object,
    message: str,
    malformed_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every terminal row is validated before update_columns or a visible commit."""
    owed = owed_clip_ids(clips, 0, ROW_FILTER)
    malformed_clip = owed[malformed_index]
    rows = [staged_row(clip_id) for clip_id in owed]
    rows[malformed_index] = staged_row(malformed_clip, status=status, caption=caption, error=error)
    stage(0, rows)
    starting_version = int(clips.version)
    fragment = clips.get_fragment(0)
    assert fragment is not None
    update_columns_calls = 0

    def reject_update_columns(*_args: object, **_kwargs: object) -> None:
        nonlocal update_columns_calls
        update_columns_calls += 1
        msg = "update_columns ran before every terminal row was validated"
        raise AssertionError(msg)

    monkeypatch.setattr(fragment, "update_columns", reject_update_columns)
    original_get_fragment = lance.LanceDataset.get_fragment

    def get_fragment(dataset: lance.LanceDataset, fragment_id: int) -> object:
        if dataset is clips and fragment_id == 0:
            return fragment
        return original_get_fragment(dataset, fragment_id)

    monkeypatch.setattr(lance.LanceDataset, "get_fragment", get_fragment)

    with pytest.raises(
        CaptionPublicationError,
        match=rf"fragment 0.*clip {re.escape(repr(malformed_clip))}.*{message}",
    ):
        prepare_fragment(clips, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)

    assert update_columns_calls == 0
    assert int(lance.dataset(clips.uri).version) == starting_version


def test_a_foreign_caption_model_is_refused_before_commit(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """A staged row from another model cannot become canonical under this model."""
    owed = owed_clip_ids(clips, 0, ROW_FILTER)
    foreign_clip = owed[0]
    rows = [staged_row(clip_id) for clip_id in owed]
    rows[0] = staged_row(foreign_clip, model_id="other/model")
    stage(0, rows)
    starting_version = int(clips.version)

    with pytest.raises(
        CaptionPublicationError,
        match=rf"fragment 0.*clip {re.escape(repr(foreign_clip))}.*other/model.*{MODEL}",
    ):
        prepare_fragment(clips, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)

    assert int(lance.dataset(clips.uri).version) == starting_version


def test_contract_digest_is_refused_before_terminal_values(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Identity, coverage, and digest checks precede terminal interpretation."""
    owed = owed_clip_ids(clips, 0, ROW_FILTER)
    rows = [staged_row(clip_id) for clip_id in owed]
    rows[0] = staged_row(owed[0], status="unknown", digest=FOREIGN_DIGEST)
    stage(0, rows)
    starting_version = int(clips.version)

    with pytest.raises(CaptionPublicationError, match="contract digest"):
        prepare_fragment(clips, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)

    assert int(lance.dataset(clips.uri).version) == starting_version


def test_one_commit_publishes_every_prepared_fragment(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
) -> None:
    """Both fragments become canonical at one version, with no intermediate state.

    Exercised through the module's own commit path rather than the Ray fan-out so
    the assertion is about the transaction, not about Ray.
    """
    starting_version = int(clips.version)
    payloads = prepare(0, 1)

    result = commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert result.published_rows == 4
    assert result.ending_version == starting_version + 1
    published = lance.dataset(clips.uri, version=result.ending_version)
    assert published.count_rows(filter="caption_status IS NULL") == 0
    assert published.to_table(columns=["caption"]).column("caption").to_pylist() == ["a caption"] * 4


def test_equal_clip_ids_in_different_fragments_publish_independently(
    tmp_path: pathlib.Path,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Fragment scope prevents equal logical ids from cross-applying captions."""
    uri = str(tmp_path / "duplicate-ids.lance")
    lance.write_dataset(
        pa.table(
            {
                "clip_id": pa.array(["duplicate", "duplicate"], pa.string()),
                "clip_uri": pa.array(["s3://b/first.mp4", "s3://b/second.mp4"], pa.large_string()),
                **{field.name: pa.nulls(2, field.type) for field in CAPTION_FIELDS},
                CURATE_SELECTION_REASON: pa.array([CURATE_SELECTED_VALUE] * 2, pa.string()),
            },
            schema=CLIPS_SCHEMA,
        ),
        uri,
        max_rows_per_file=1,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    dataset = lance.dataset(uri)
    stage(0, [staged_row("duplicate", caption="first fragment caption")])
    stage(1, [staged_row("duplicate", caption="second fragment caption")])
    payloads = [
        prepare_fragment(
            dataset,
            workspace,
            fragment_id,
            digest=DIGEST,
            model_id=MODEL,
            row_filter=ROW_FILTER,
        )
        for fragment_id in (0, 1)
    ]

    result = commit_prepared(
        dataset,
        payloads,
        (0, 1),
        row_filter=ROW_FILTER,
        storage_options=None,
    )

    published = lance.dataset(uri, version=result.ending_version).to_table(columns=["clip_uri", "caption"]).to_pylist()
    assert published == [
        {"clip_uri": "s3://b/first.mp4", "caption": "first fragment caption"},
        {"clip_uri": "s3://b/second.mp4", "caption": "second fragment caption"},
    ]


def test_commit_stamps_the_literal_transaction_properties(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit carries its durable kind and one UUID attempt marker."""
    payloads = prepare(0)
    real_commit = lance.LanceDataset.commit
    observed: list[dict[str, str]] = []

    def capture_transaction(base_uri: str, transaction: lance.Transaction, **kwargs: object) -> lance.LanceDataset:
        observed.append(dict(transaction.transaction_properties))
        return real_commit(base_uri, transaction, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(capture_transaction))

    commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)

    assert len(observed) == 1
    assert set(observed[0]) == {"kind", "publication_attempt"}
    assert observed[0]["kind"] == "curator-next-robot-action-caption"
    assert str(UUID(observed[0]["publication_attempt"])) == observed[0]["publication_attempt"]


def test_each_commit_attempt_gets_a_distinct_marker(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two commit attempts sharing one semantic digest remain distinguishable."""
    payloads = prepare(0)
    observed: list[str] = []

    def reject_the_commit(_base_uri: str, transaction: lance.Transaction, **_kwargs: object) -> None:
        observed.append(transaction.transaction_properties["publication_attempt"])
        msg = "Commit conflict: another transaction has already been committed"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(reject_the_commit))

    for _attempt in range(2):
        with pytest.raises(ValueError, match=r"row\(s\).*still owed"):
            commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)

    assert len(observed) == 2
    assert observed[0] != observed[1]


def test_a_wave_of_terminal_errors_counts_every_one(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    staged_error_row: StagedRowFactory,
) -> None:
    """A fault that errored every clip commits as a success, so only the count shows it.

    A missing media object is row-local in cause yet can be total in extent - a
    table restored without its media errors every clip - and a published error row
    satisfies the full completion contract, so the next run finds nothing owed and
    says so. Publication does not refuse this; it reports it, and the report has to
    distinguish it from four published captions.
    """
    payloads = prepare(0, 1, row=staged_error_row)

    result = commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert (result.published_rows, result.error_rows) == (4, 4)


def test_a_wave_of_truncated_captions_counts_every_one(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    staged_truncated_row: StagedRowFactory,
) -> None:
    """A cut-off caption publishes as a success, so only the count separates the two.

    Truncation is not a failure: the text is real, merely incomplete, so it
    carries the same non-null caption a complete one does and satisfies the
    pending predicate identically. Nothing here refuses it and nothing should -
    without this count, a corpus every caption of which stops mid-sentence reads
    exactly like a corpus of whole ones.
    """
    payloads = prepare(0, 1, row=staged_truncated_row)

    result = commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert (result.published_rows, result.truncated_rows) == (4, 4)


def test_a_wave_of_captions_counts_no_error(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
) -> None:
    """The healthy wave reports zero, which is what makes a non-zero count mean something."""
    payloads = prepare(0, 1)

    result = commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert (result.published_rows, result.error_rows) == (4, 0)


def test_two_payloads_for_one_fragment_are_refused(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
) -> None:
    """Committing two metadata versions of one fragment would depend on collection order."""
    payload = prepare(0)[0]

    with pytest.raises(ValueError, match="two metadata versions of one fragment"):
        commit_prepared(clips, [payload, payload], (0,), row_filter=ROW_FILTER, storage_options=None)


def test_a_missing_prepared_fragment_blocks_the_commit(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
) -> None:
    """A wave short of a planned fragment would publish a partial corpus as a success.

    The commit itself cannot notice: an ``Update`` names only the fragments it
    carries, so a one-fragment transaction is a perfectly valid one. The clips of
    the unnamed fragment would simply stay pending while the run reported success.
    """
    payloads = prepare(0)

    with pytest.raises(CaptionPublicationError, match=r"missing \[1\]"):
        commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert lance.dataset(clips.uri).count_rows(filter="caption_status IS NULL") == 4


def test_an_unplanned_prepared_fragment_blocks_the_commit(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
) -> None:
    """A fragment nobody planned carries rows from a foreign attempt's row set."""
    payloads = prepare(0, 1)

    with pytest.raises(CaptionPublicationError, match=r"unexpected \[1\]"):
        commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)

    assert lance.dataset(clips.uri).count_rows(filter="caption_status IS NULL") == 4


def test_embedded_fragment_metadata_must_match_its_envelope(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse metadata for another fragment before constructing the transaction."""
    payloads = prepare(0, 1)
    payloads[0]["metadata_json"] = payloads[1]["metadata_json"]

    def reject_commit(*_args: object, **_kwargs: object) -> None:
        msg = "commit reached with metadata from the wrong fragment"
        raise AssertionError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(reject_commit))

    with pytest.raises(CaptionPublicationError, match=r"envelope fragment 0.*metadata fragment 1"):
        commit_prepared(
            clips,
            payloads,
            (0, 1),
            row_filter=ROW_FILTER,
            storage_options=None,
        )


def test_lost_response_recovery_scans_past_an_unmarked_sibling(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
    loguru_records: list[dict[str, Any]],
) -> None:
    """A later unmarked sibling cannot hide the caption attempt's committed marker.

    Lance raises the same way whether the manifest was written or not, so trusting
    the error would send an operator to re-publish a wave that already succeeded -
    and generation is unseeded, so the second publication would write different
    captions over the first. The fake commits for real, adds an unrelated sibling
    commit, and only then loses the response. Recovery must scan the full intervening
    history, warn that it is continuing, and return the later observed version.
    """
    payloads = prepare(0, 1)
    starting_version = int(clips.version)
    commit = lance.LanceDataset.commit
    sibling_versions: list[int] = []

    def commit_then_add_sibling_and_lose_response(
        base_uri: str,
        transaction: lance.Transaction,
        **kwargs: object,
    ) -> None:
        committed = commit(base_uri, transaction, **kwargs)  # type: ignore[arg-type]
        caption_version = int(committed.version)
        committed.add_columns(pa.field("unrelated_sibling", pa.string(), nullable=True))
        sibling_version = int(committed.version)
        sibling_transaction = committed.read_transaction(sibling_version)

        assert sibling_version == caption_version + 1
        assert sibling_transaction is not None
        assert sibling_transaction.transaction_properties == {}
        sibling_versions.append(sibling_version)
        msg = "Generic error: connection reset by peer"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(commit_then_add_sibling_and_lose_response))

    result = commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)

    assert sibling_versions == [starting_version + 2]
    assert result.ending_version == sibling_versions[0]
    assert lance.dataset(clips.uri).version == sibling_versions[0]
    assert lance.dataset(clips.uri, version=result.ending_version).count_rows(filter=ROW_FILTER) == 0
    warnings = [str(record["message"]) for record in loguru_records if record["level"].name == "WARNING"]
    assert warnings == [
        f"Caption publication commit against v{starting_version} of {clips.uri} reported "
        "Generic error: connection reset by peer, "
        f"but no row of the wave is owed at v{sibling_versions[0]}: the commit landed and the error "
        "was a lost response. Treating the publication as successful."
    ]


def test_ambiguous_commit_verification_ignores_unselected_pending_rows(
    clips: lance.LanceDataset,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The attempt's exact scope, not global pending rows, proves its lost response landed."""
    clips.update(
        {CURATE_SELECTION_REASON: "'duplicate'"},
        where="clip_id IN ('c1', 'c3')",
    )
    for fragment_id, selected_clip in ((0, "c0"), (1, "c2")):
        stage(fragment_id, [staged_row(selected_clip)])
    payloads = [
        prepare_fragment(
            clips,
            workspace,
            fragment_id,
            digest=DIGEST,
            model_id=MODEL,
            row_filter=SELECTED_ROW_FILTER,
        )
        for fragment_id in (0, 1)
    ]
    starting_version = int(clips.version)
    commit = lance.LanceDataset.commit

    def commit_then_lose_response(
        base_uri: str,
        transaction: lance.Transaction,
        **kwargs: object,
    ) -> None:
        commit(base_uri, transaction, **kwargs)  # type: ignore[arg-type]
        msg = "Generic error: connection reset by peer"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(commit_then_lose_response))

    result = commit_prepared(
        clips,
        payloads,
        (0, 1),
        row_filter=SELECTED_ROW_FILTER,
        storage_options=None,
    )
    published = lance.dataset(clips.uri, version=result.ending_version)

    assert result.ending_version == starting_version + 1
    assert published.count_rows(filter=SELECTED_ROW_FILTER) == 0
    assert published.count_rows(filter=ROW_FILTER) == 2


def test_an_ambiguous_commit_with_a_foreign_marker_is_refused(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical rows written by another attempt do not prove this attempt landed."""
    payloads = prepare(0)
    commit = lance.LanceDataset.commit

    def commit_foreign_then_lose_response(
        base_uri: str,
        transaction: lance.Transaction,
        **kwargs: object,
    ) -> None:
        foreign = lance.Transaction(
            read_version=transaction.read_version,
            operation=transaction.operation,
            transaction_properties={
                "kind": "curator-next-robot-action-caption",
                "publication_attempt": "00000000-0000-4000-8000-000000000001",
            },
        )
        commit(base_uri, foreign, **kwargs)  # type: ignore[arg-type]
        msg = "Generic error: connection reset by peer"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(commit_foreign_then_lose_response))

    with pytest.raises(CaptionPublicationConflictError, match=r"cannot attribute.*publication attempt"):
        commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)


def test_an_ambiguous_commit_without_a_marker_is_refused(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical rows from an unmarked commit do not prove this attempt landed."""
    payloads = prepare(0)
    commit = lance.LanceDataset.commit

    def commit_unmarked_then_lose_response(
        base_uri: str,
        transaction: lance.Transaction,
        **kwargs: object,
    ) -> None:
        unmarked = lance.Transaction(
            read_version=transaction.read_version,
            operation=transaction.operation,
            transaction_properties={"kind": "curator-next-robot-action-caption"},
        )
        commit(base_uri, unmarked, **kwargs)  # type: ignore[arg-type]
        msg = "Generic error: connection reset by peer"
        raise OSError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(commit_unmarked_then_lose_response))

    with pytest.raises(CaptionPublicationConflictError, match=r"cannot attribute.*publication attempt"):
        commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)


def test_an_ambiguous_commit_that_really_failed_names_the_owed_rows(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected commit leaves the whole wave owed, and the refusal counts it.

    Raised as ``RuntimeError`` rather than ``OSError``: Lance routes commit
    conflicts and write contention there rather than to the IO error, so this is
    the other arm of the pair the verification catches.
    """
    payloads = prepare(0, 1)

    def reject_the_commit(*_args: object, **_kwargs: object) -> None:
        msg = "Commit conflict: another transaction has already been committed"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(reject_the_commit))

    with pytest.raises(ValueError, match=r"4 row\(s\) across the wave's fragments are still owed"):
        commit_prepared(clips, payloads, (0, 1), row_filter=ROW_FILTER, storage_options=None)


def test_an_ambiguous_commit_whose_fragment_vanished_is_refused_as_unverifiable(
    clips: lance.LanceDataset,
    prepare: WaveFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expected fragment absent at latest makes the outcome unverifiable, not successful.

    The dangerous shape of the ambiguous commit: retiring the wave's only
    fragment leaves NO owed row to count, so treating an absent fragment as
    nothing to check would read this rejected commit as landed and report a
    publication that wrote nothing as a success.
    """
    payloads = prepare(0)
    # A concurrent compactor retiring the fragment the wave prepared. Deleting
    # every row it holds drops it from the manifest, which is the state the
    # verification has to interpret.
    retired = ", ".join(f"'{clip_id}'" for clip_id in owed_clip_ids(clips, 0, ROW_FILTER))
    lance.dataset(clips.uri).delete(f"clip_id IN ({retired})")

    def reject_the_commit(*_args: object, **_kwargs: object) -> None:
        msg = "Commit conflict: another transaction has already been committed"
        raise RuntimeError(msg)

    monkeypatch.setattr(lance.LanceDataset, "commit", staticmethod(reject_the_commit))

    with pytest.raises(ValueError, match=r"fragment\(s\) \[0\] the wave published into are absent"):
        commit_prepared(clips, payloads, (0,), row_filter=ROW_FILTER, storage_options=None)


def test_publishing_preserves_a_caption_that_is_not_in_this_wave(
    tmp_path: pathlib.Path,
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """update_columns keeps unmatched LEFT rows, so an earlier caption survives the wave."""
    uri = str(tmp_path / "clips.lance")
    lance.write_dataset(
        pa.table(
            {
                "clip_id": pa.array(["c0", "c1"], pa.string()),
                "clip_uri": pa.array(["s3://b/c0.mp4", "s3://b/c1.mp4"], pa.large_string()),
                "caption": pa.array(["earlier", None], pa.large_string()),
                "caption_status": pa.array([CaptionStatus.SUCCESS, None], pa.string()),
                "caption_error": pa.array([None, None], pa.large_string()),
                "caption_contract_digest": pa.array([DIGEST, None], pa.string()),
                "caption_model": pa.array([MODEL, None], pa.string()),
                CURATE_SELECTION_REASON: pa.array([CURATE_SELECTED_VALUE] * 2, pa.string()),
            },
            schema=CLIPS_SCHEMA,
        ),
        uri,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    dataset = lance.dataset(uri)
    stage(0, [staged_row("c1", caption="new")])

    payload = prepare_fragment(dataset, workspace, 0, digest=DIGEST, model_id=MODEL, row_filter=ROW_FILTER)
    result = commit_prepared(dataset, [payload], (0,), row_filter=ROW_FILTER, storage_options=None)

    captions = lance.dataset(uri, version=result.ending_version).to_table(columns=["clip_id", "caption"]).to_pylist()
    assert captions == [{"clip_id": "c0", "caption": "earlier"}, {"clip_id": "c1", "caption": "new"}]


def test_the_staged_partition_carries_exactly_the_published_columns(
    workspace: CaptionWorkspace,
    stage: StagedShardWriter,
    staged_row: StagedRowFactory,
) -> None:
    """Publication reads clip_id plus the five caption columns and nothing else."""
    stage(0, [staged_row("c0")])

    assert staged_captions(workspace, 0).schema.names == ["clip_id", *CAPTION_COLUMNS]
