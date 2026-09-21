# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Define what ``clips.lance`` must contain and which rows an attempt owes.

Every function here reads a dataset the CALLER opened, following the same shape
as the embeddings recipe's ``columns`` module, so which version a call reads is
visible at the call site. ``ensure_caption_fields`` is the deliberate exception
and takes the URI: it COMMITS, and a lost schema race cannot be retried while
holding the handle that lost it. It RETURNS the advanced handle, which makes "the
read version is pinned from the handle that widened the table" structural rather
than a call-site convention.

The table boundary keeps validation before mutation even when registration is
called before explicit context resolution::

    require_source_columns(ds)  refuse a table no row of which is captionable
          |
          v
    require_selection_column    refuse an unavailable selected-only scope
          |
          v
    validate context fields     shared by resolution and registration
          |
          v
    ensure_caption_fields(uri)  register the complete caption group; opens its
          |                     own handle per attempt (metadata-only commit)
          v
    pin read_version            from the handle that call RETURNED
          |
          v
    owed_clip_ids(ds, id, row_filter)   owed clips in one fragment

Two column classes, and the difference is the whole validation policy. ``clip_id``
and ``clip_uri`` are STRUCTURAL: without them no row in the table can be
captioned, so their absence is refused. Every context column is OPTIONAL: its
absence narrows the prompt for the whole run and is recorded in the digest, so it
is resolved rather than required.

``curate_selection_reason`` is conditional input scope: required at exact
``string`` type only when selected-only mode is enabled. Curate writes it
nullable; equality safely excludes NULL without requiring that schema flag. The
local name and selected value describe persisted data without importing Curate.

The write side is deliberately absent apart from the schema commit: preparing
and committing caption values belongs to the publication phase, which owns its
own ``LanceOperation.Update`` semantics.
"""

import enum
import re
from collections.abc import Sequence
from typing import Final, assert_never

import lance
import pyarrow as pa
from lance.fragment import LanceFragment
from loguru import logger

from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_COLUMNS,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_FIELDS,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
    CaptionStatus,
)
from cosmos_curator.next.recipes.robot_action_caption.prompt import (
    SUPPORTED_CONTEXT_FIELDS,
    PromptContextField,
)
from cosmos_curator.next.utils.arrow_utils import is_text_type
from cosmos_curator.next.utils.lance_fragment_recovery import sql_string_literal

CLIP_ID_COLUMN: Final = "clip_id"
CLIP_URI_COLUMN: Final = "clip_uri"

# Local read contract rather than a cross-recipe import: captioning consumes
# persisted table data, not the Curate implementation. A focused boundary test
# pins these values against the producer so schema drift still fails in CI.
CURATE_SELECTION_REASON: Final = "curate_selection_reason"
CURATE_SELECTED_VALUE: Final = "selected"

# Structurally required: a table missing either of these has no captionable row,
# so the run is refused rather than producing an error row per clip. Context
# columns are deliberately NOT here - see `resolve_context_fields`.
REQUIRED_SOURCE_COLUMNS: Final[tuple[str, ...]] = (CLIP_ID_COLUMN, CLIP_URI_COLUMN)

# How many COMMIT passes a contended widening gets, on top of which the loop
# always spends one final read-only pass. Deliberately not a parameter: the bound
# exists so a deterministic storage failure still surfaces, and the number of
# jobs racing to widen is not something an operator tunes against. Three is
# enough for any number of them, because every losing job reopens to find the
# group already present rather than retrying its own commit.
_REGISTRATION_ATTEMPTS: Final = 3

_HEX64: Final = re.compile(r"\A[0-9a-f]{64}\Z")


class _CaptionGroupPresence(enum.StrEnum):
    """How much of the caption field group one snapshot already carries."""

    ABSENT = "absent"
    PARTIAL = "partial"
    PRESENT = "present"


def require_source_columns(dataset: lance.LanceDataset) -> None:
    """Refuse a table whose structural columns are absent or unusable.

    Presence alone is not enough. Both columns are read as text - one as the
    caption's identity, one as the media location - and a producer that retyped
    either fails inside a read task, where the exception type is erased by Ray
    and the message reaches an operator wrapped in a ``RayError``.

    The two get DIFFERENT gates, because the recipe uses them differently:
    ``clip_uri`` is only ever read, so any encoding a scan hands back as ``str``
    serves; ``clip_id`` is also a join key and a logical identity, so its exact
    contract is plain Arrow ``string`` with no nulls.

    Runs on the driver, and before ``ensure_caption_fields``, so an invalid source
    is refused with a usable error and without having mutated the table.

    Args:
        dataset: The clip table, before widening.

    Raises:
        ValueError: If a required column is absent, if ``clip_id`` is not
            non-nullable ``string``, or if ``clip_uri`` does not hold text.

    """
    schema = dataset.schema
    present = set(schema.names)
    missing = [name for name in REQUIRED_SOURCE_COLUMNS if name not in present]
    if missing:
        msg = (
            f"clip table {dataset.uri} is missing required source column(s) {missing}; "
            f"the recipe reads {list(REQUIRED_SOURCE_COLUMNS)} and a missing one makes every row uncaptionable"
        )
        raise ValueError(msg)
    clip_id_field = schema.field(CLIP_ID_COLUMN)
    # Narrower than the text gate its sibling gets, because this column is not
    # merely readable text: `publication.prepare_fragment` joins the caption
    # update on it, and types that update from the STORED schema so
    # `update_columns` never coerces. A dictionary-encoded or large_string id
    # would therefore be joined at that type on both sides, which is not a
    # documented capability of that call and which no table this recipe has run
    # against exercises - and the failure would land inside a Ray task, where
    # Ray erases the exception type. Widening this gate means proving the join
    # handles the type, not merely that the value reads back as `str`.
    # Nullability is equally load-bearing: a null logical identity cannot name
    # the row to publish, checkpoint, or reconcile, so it is refused before the
    # caption group can be added to the table.
    # Type and nullability only: stored VALUES are the row-level identity check's
    # to police, so this boundary stays schema-cheap rather than buying per-value
    # certainty with a table-wide scan every run pays.
    if not clip_id_field.type.equals(pa.string()) or clip_id_field.nullable:
        msg = (
            f"clip table {dataset.uri} stores {CLIP_ID_COLUMN} as {clip_id_field.type} with "
            f"nullable={clip_id_field.nullable}; it is the logical key the caption update joins on, "
            f"so it must be exactly non-nullable string"
        )
        raise ValueError(msg)
    clip_uri_type = schema.field(CLIP_URI_COLUMN).type
    if not is_text_type(clip_uri_type):
        msg = (
            f"clip table {dataset.uri} holds non-text {CLIP_URI_COLUMN} of type {clip_uri_type}; "
            f"it must be string, large_string, string_view, or a dictionary of those"
        )
        raise ValueError(msg)


def require_selection_column(dataset: lance.LanceDataset, *, selected_only: bool) -> None:
    """Refuse a table that cannot answer the configured input scope.

    Args:
        dataset: The clip table, before widening.
        selected_only: Whether the run requires Curate's final selection.

    Raises:
        ValueError: If selected-only scope lacks the canonical string column.

    """
    if not selected_only:
        return
    if CURATE_SELECTION_REASON not in dataset.schema.names:
        msg = (
            f"selected-only captioning requires column {CURATE_SELECTION_REASON!r}, written by the curate leg; "
            "set input.selected_only=false to caption every pending row instead"
        )
        raise ValueError(msg)
    stored_type = dataset.schema.field(CURATE_SELECTION_REASON).type
    if not stored_type.equals(pa.string()):
        msg = (
            f"selected-only captioning requires {CURATE_SELECTION_REASON} to be exactly string, "
            f"but {dataset.uri} stores {stored_type}; set input.selected_only=false to caption every pending row"
        )
        raise ValueError(msg)


def _validated_context_fields(dataset: lance.LanceDataset) -> tuple[PromptContextField, ...]:
    """Return supported context fields after validating every present field."""
    schema = dataset.schema
    present = set(schema.names)
    resolved = []
    for field in SUPPORTED_CONTEXT_FIELDS:
        if field.column not in present:
            continue
        arrow_type = schema.field(field.column).type
        if not is_text_type(arrow_type):
            # Refused rather than skipped: absence is a supported source shape,
            # but a producer that retypes a present field has violated the prompt
            # contract. Continuing would make every caption silently lose that
            # line and then look current to the pending predicate.
            msg = (
                f"caption context column {field.column!r} on {dataset.uri} has non-text type {arrow_type}; "
                f"supported context columns must hold text when present"
            )
            raise ValueError(msg)
        resolved.append(field)
    return tuple(resolved)


def resolve_context_fields(dataset: lance.LanceDataset) -> tuple[PromptContextField, ...]:
    """Return the supported context fields this snapshot can actually supply.

    Resolution is per SNAPSHOT, not per row, which is what keeps one attempt's
    prompt contract single-valued: every row in the run renders the same field
    set, the digest records exactly that set, and ``datasource.SCAN_SCHEMA`` never
    varies with which optional columns happen to exist.

    An absent column narrows the prompt normally. A present non-text column is a
    source-contract fault: treating it as absent would silently lose one prompt
    line from every caption while stamping the run as successful.

    Args:
        dataset: The clip table.

    Returns:
        The resolvable fields, in :data:`SUPPORTED_CONTEXT_FIELDS` order.

    Raises:
        ValueError: If a supported context column is present but does not hold
            text.

    """
    resolved = _validated_context_fields(dataset)
    logger.info(
        f"resolved caption context columns {[field.column for field in resolved]} on {dataset.uri} "
        f"out of supported {[field.column for field in SUPPORTED_CONTEXT_FIELDS]}"
    )
    return resolved


def _pending_filter(digest: str, model_id: str) -> str:
    """Return the SQL predicate selecting rows this contract still owes.

    Latest contract wins, and a current digest is complete only when its
    model and terminal values satisfy the publication contract.

    Args:
        digest: This run's contract digest.
        model_id: The resolved model identity this run expects.

    Returns:
        A parenthesized SQL predicate.

    Raises:
        ValueError: If ``digest`` is not a 64-character lowercase hex sha256,
            or ``model_id`` is blank.

    """
    # Validated before interpolation, which is what makes the interpolation
    # injection-free: the accepted alphabet cannot express a quote or an operator.
    if not _HEX64.fullmatch(digest):
        msg = f"contract digest must be a 64-character lowercase hex sha256, got {digest!r}"
        raise ValueError(msg)
    if not model_id.strip():
        msg = f"expected model identity must be nonblank, got {model_id!r}"
        raise ValueError(msg)
    captioned_statuses = (CaptionStatus.SUCCESS, CaptionStatus.TRUNCATED)
    captioned_status_literals = ", ".join(sql_string_literal(status.value) for status in captioned_statuses)
    terminal_status_literals = ", ".join(sql_string_literal(status.value) for status in CaptionStatus)
    nonblank_pattern = sql_string_literal(r"\S")
    model_literal = sql_string_literal(model_id)

    # Every nullable member gets an explicit IS NULL arm where NULL means
    # malformed. SQL three-valued logic otherwise turns comparisons against NULL
    # into NULL rather than true, and the row can disappear from the scan.
    #
    # Identifiers stay bare, never quoted: on the Lance scanner filter surface a
    # double-quoted name parses as a STRING LITERAL, which silently compares two
    # constants and matches every row instead of raising.
    #
    # `!=` rather than `<>` because `!=` is the spelling already proven against
    # `count_rows(filter=...)` on this surface in `robot_action_split/recovery.py`.
    #
    # `regexp_like(..., '\S')` is Lance's supported nonblank check. `trim` is not
    # accepted by its filter planner, while length alone treats whitespace as
    # content and would diverge from validate_terminal_values().
    return (
        f"({CAPTION_DIGEST_COLUMN} IS NULL OR {CAPTION_DIGEST_COLUMN} != '{digest}' "
        f"OR {CAPTION_MODEL_COLUMN} IS NULL OR {CAPTION_MODEL_COLUMN} != {model_literal} "
        f"OR {CAPTION_STATUS_COLUMN} IS NULL "
        f"OR {CAPTION_STATUS_COLUMN} NOT IN ({terminal_status_literals}) "
        f"OR ({CAPTION_STATUS_COLUMN} IN ({captioned_status_literals}) "
        f"AND ({CAPTION_COLUMN} IS NULL OR NOT regexp_like({CAPTION_COLUMN}, {nonblank_pattern}) "
        f"OR {CAPTION_ERROR_COLUMN} IS NOT NULL)) "
        f"OR ({CAPTION_STATUS_COLUMN} = {sql_string_literal(CaptionStatus.ERROR.value)} "
        f"AND ({CAPTION_COLUMN} IS NOT NULL OR {CAPTION_ERROR_COLUMN} IS NULL "
        f"OR NOT regexp_like({CAPTION_ERROR_COLUMN}, {nonblank_pattern})))"
        ")"
    )


def caption_work_filter(digest: str, model_id: str, *, selected_only: bool) -> str:
    """Return the SQL predicate selecting the rows this attempt owes."""
    pending = _pending_filter(digest, model_id)
    if not selected_only:
        return pending
    # Keep the identifier bare: Lance's scanner filter parser treats a
    # double-quoted identifier as a string literal and silently matches the
    # wrong rows. Values still go through the shared SQL literal escaper.
    selected = sql_string_literal(CURATE_SELECTED_VALUE)
    return f"({pending} AND {CURATE_SELECTION_REASON} = {selected})"


def _caption_group_presence(dataset: lance.LanceDataset) -> _CaptionGroupPresence:
    """Classify the caption field group's presence on one snapshot.

    Raises:
        ValueError: If a complete group's fields do not match
            :data:`CAPTION_FIELDS` exactly.

    """
    stored = set(dataset.schema.names)
    present = [name for name in CAPTION_COLUMNS if name in stored]
    if not present:
        return _CaptionGroupPresence.ABSENT
    if len(present) != len(CAPTION_COLUMNS):
        return _CaptionGroupPresence.PARTIAL
    _require_fields_match(dataset.schema, dataset.uri, CAPTION_COLUMNS)
    return _CaptionGroupPresence.PRESENT


def _partial_group_message(uri: str, schema: pa.Schema) -> str:
    """Return the refusal naming which caption fields exist and which do not."""
    stored = set(schema.names)
    present = [name for name in CAPTION_COLUMNS if name in stored]
    missing = [name for name in CAPTION_COLUMNS if name not in stored]
    return (
        f"partial caption column group on {uri}: present={present}, missing={missing}; "
        f"the group is registered atomically and a strict subset cannot be repaired safely"
    )


def ensure_caption_fields(
    uri: str,
    *,
    storage_options: dict[str, str] | None,
) -> lance.LanceDataset:
    """Register or validate the complete caption field group, returning the handle to pin.

    Args:
        uri: The clip table to widen.
        storage_options: Backend credentials and endpoint settings, or ``None``
            for the backend's own resolution.

    Returns:
        The table open at a version where the complete caption group exists. The
        caller pins the run's read version from this handle.

    Raises:
        ValueError: If a required source column is absent or unusable, if only
            part of the group exists, if a complete group's fields do not match
            :data:`CAPTION_FIELDS` exactly, or if a supported context column has
            a non-text type.
        RuntimeError: If every commit lost the schema race and the group is
            still absent.

    """
    # Takes a URI rather than a handle - the one place this module does not read
    # a caller-supplied dataset - because it COMMITS. A concurrent first launch
    # (several jobs dividing one campaign, all finding the table unwidened) loses
    # the race on every job but one, and a lost commit cannot be retried from the
    # handle that lost it: the group the loser needs exists only at a version its
    # handle cannot see. So every pass reopens.
    #
    # One pass more than the commit budget, and the extra one is read-only. A
    # conflict on the final commit is itself evidence that SOME writer committed,
    # so the loop has to look once more before reporting failure - and by making
    # that final look an ordinary iteration, a group that turns out partial is
    # refused as partial rather than misreported as absent.
    last_error: Exception | None = None
    for attempt in range(_REGISTRATION_ATTEMPTS + 1):
        dataset = lance.dataset(uri, storage_options=storage_options)
        # Structural identity first: a missing clip_id / clip_uri makes every row
        # uncaptionable, and add_columns below would still bump the Lance version.
        # Context fields stay optional and are validated next, before that commit.
        require_source_columns(dataset)
        # Keep registration fail-closed when a caller has not resolved context yet:
        # a metadata commit here must not precede reporting a malformed source field.
        _validated_context_fields(dataset)
        presence = _caption_group_presence(dataset)
        match presence:
            case _CaptionGroupPresence.PRESENT:
                # No commit, so no version bump - a contract rather than an
                # optimization. An already-complete field set is the case a resume
                # must hit; widening runs at most once per table per contract
                # generation, so this is the path every later attempt takes.
                return dataset
            case _CaptionGroupPresence.PARTIAL:
                # Refused, never topped up: a strict subset has ambiguous
                # ownership, and completing it would turn an unexplained schema
                # into a published one that looks deliberate.
                raise ValueError(_partial_group_message(uri, dataset.schema))
            case _CaptionGroupPresence.ABSENT:
                pass
            case _:
                assert_never(presence)
        if attempt == _REGISTRATION_ATTEMPTS:
            break
        try:
            # Metadata-only: an all-nullable schema is recorded in the manifest
            # and read as NULL for every existing row, so no data file is
            # rewritten and no row is replaced. That is what makes retrying this
            # cheap on a large table, and why it cannot create a tombstone.
            dataset.add_columns(CAPTION_FIELDS)
        except OSError as e:
            # Measured at the locked pylance 10.0.0: a lost widening race is a
            # bare OSError, and RuntimeError is raised on this path at neither
            # 10.0.0 nor 11.0.0 (where the same race is CommitConflictError,
            # itself an OSError). Re-measure on a pylance bump.
            #
            # A deterministic "Invalid user input" fault is the SAME type there,
            # so the two cannot be told apart by type, and the only thing that
            # separates them is a human-readable message no code may parse. What
            # keeps the broad catch acceptable is that both such faults are
            # unreachable here: this call is gated on the group being absent, so
            # no column can already exist, and CAPTION_FIELDS is frozen
            # all-nullable. One arriving anyway still surfaces, as the cause of
            # the RuntimeError below.
            last_error = e
            continue
        # add_columns advances THIS handle, so the version logged is the
        # widening's own and no reopen is needed.
        logger.info(f"registered caption fields {list(CAPTION_COLUMNS)} on {uri} as v{int(dataset.version)}")
        return dataset
    msg = (
        f"could not register caption fields {list(CAPTION_COLUMNS)} on {uri} "
        f"in {_REGISTRATION_ATTEMPTS} attempt(s); "
        f"the group is still absent, so no concurrent writer registered it either"
    )
    raise RuntimeError(msg) from last_error


def require_clip_id(value: object) -> str:
    """Return a nonblank string clip id without normalization.

    Raises:
        ValueError: If ``value`` is not a nonblank string.

    """
    # Whitespace is inspected only to reject an all-whitespace identity. Leading
    # and trailing whitespace on a nonblank key is retained byte-for-byte.
    if not isinstance(value, str) or not value.strip():
        msg = f"{CLIP_ID_COLUMN} must be a nonblank string, got {value!r}"
        raise ValueError(msg)
    return value


def owed_clip_ids(
    dataset: lance.LanceDataset,
    fragment_id: int,
    row_filter: str,
) -> tuple[str, ...]:
    """Return the ``clip_id`` of every owed row in one fragment.

    Args:
        dataset: The clip table, open at the run's pinned version.
        fragment_id: Fragment to read.
        row_filter: Predicate defining what this attempt owes.

    Returns:
        The owed clip ids, in the fragment's physical row order.

    Raises:
        ValueError: If the caption fields are unregistered, the fragment is
            absent at this version, or an owed row has an invalid ``clip_id``.

    """
    _require_registered(dataset)
    fragment = _fragment_or_raise(dataset, fragment_id)
    scanner = fragment.scanner(columns=[CLIP_ID_COLUMN], filter=row_filter)
    return tuple(
        require_clip_id(value) for batch in scanner.to_batches() for value in batch.column(CLIP_ID_COLUMN).to_pylist()
    )


def count_rows_in_fragments(
    dataset: lance.LanceDataset,
    fragment_ids: Sequence[int],
    row_filter: str,
) -> int:
    """Return how many rows match ``row_filter`` across exactly these fragments.

    Scoped rather than table-wide on purpose. A job that owns part of the table
    must not report a count covering the rest of it: a diagnostic drawn from the
    whole table sends an operator after a backlog some other job already owns.

    Args:
        dataset: The clip table, open at the run's pinned version.
        fragment_ids: The fragments to count over, all present at this version.
        row_filter: Predicate to count matches of.

    Returns:
        How many rows in those fragments match. Zero for an empty fragment list.

    Raises:
        ValueError: If a fragment is absent at this version.

    """
    # Ids are resolved first, and that is what keeps a planned-but-absent
    # fragment a refusal rather than a smaller number. It is close to free:
    # measured at pylance 10.0.0, resolving 1,000 ids cost 0.8 ms against the
    # 56 ms the count itself took.
    fragments = [_fragment_or_raise(dataset, fragment_id) for fragment_id in fragment_ids]
    # One scanner across the named fragments, NOT a filtered count per fragment:
    # measured over those same 1,000 fragments, 56 ms against 828 ms, because the
    # per-fragment form pays a separate filtered scan every time. At the ~31k
    # fragments a 250M-row table reaches, that is the difference between about
    # two seconds of driver time and about twenty-six.
    #
    # And NOT `dataset.count_rows(filter=..., fragments=...)`, whose signature
    # invites exactly that: it accepts `fragments` into `**kwargs` and forwards
    # it NOWHERE, so it silently counts the WHOLE TABLE - measured returning
    # 10,000 where the one named fragment held 10. The scanner form is
    # scope-correct, and counts an empty list as zero rather than as everything.
    scanner = dataset.scanner(columns=[], with_row_id=True, filter=row_filter, fragments=fragments)
    return int(scanner.count_rows())  # type: ignore[no-untyped-call]  # Untyped in Lance stubs.


def _fragment_or_raise(dataset: lance.LanceDataset, fragment_id: int) -> LanceFragment:
    """Return one fragment, refusing a planned id the pinned version does not hold.

    Absence is a planning fault, not an empty result: the plan named this
    fragment at the version it pinned, so a caller that skipped it would report a
    count or a work list covering less of the table than it claims to own.
    """
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"fragment {fragment_id} is not present in {dataset.uri} at v{int(dataset.version)}"
        raise ValueError(msg)
    return fragment


def _require_registered(dataset: lance.LanceDataset) -> None:
    """Raise unless every caption field is present on the dataset."""
    missing = [name for name in CAPTION_COLUMNS if name not in set(dataset.schema.names)]
    if missing:
        msg = (
            f"caption fields {missing} are not present on {dataset.uri}; "
            f"ensure_caption_fields must run before the read version is pinned"
        )
        raise ValueError(msg)


def _require_fields_match(schema: pa.Schema, uri: str, names: Sequence[str]) -> None:
    """Raise unless each named caption field matches the frozen contract exactly.

    Args:
        schema: The stored schema.
        uri: Clip table URI, for the refusal message.
        names: The complete caption field group present on ``schema``.

    Raises:
        ValueError: If a named field's type or nullability differs.

    """
    # Rendered as fields rather than as bare names so the message carries the
    # stored type AND nullability beside the contract's - a nullability-only
    # mismatch is otherwise reported as a column that looks identical.
    mismatched = [
        f"{schema.field(name)} (contract: {CAPTION_FIELDS.field(name)})"
        for name in names
        if not schema.field(name).equals(CAPTION_FIELDS.field(name))
    ]
    if mismatched:
        msg = (
            f"existing caption fields on {uri} are incompatible with the frozen contract: {mismatched}; "
            f"their type and nullability are a published contract and cannot be narrowed in place"
        )
        raise ValueError(msg)
