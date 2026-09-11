# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase B: validate every staged row, then publish all of them in ONE commit.

::

    per fragment, in a Ray task (no commit):
        read  results/fragment_id=<id>/*.parquet
        fold  duplicate clip_ids -> one row, conflicting ones -> REFUSE
        check exactly one terminal row per owed clip, no extras,
              every row carrying THIS run's contract digest and model,
              every terminal status matching its caption/error nulls
        write fragment.update_columns(clip_id + the five caption columns)
              -> new column file + rebound field ids; nothing visible yet
              |
    driver:   v
        ONE Transaction(Update(every prepared fragment, fields_modified))
              |
              v
        clips.lance @ v+1     <- the sole completion signal

A refusal can leave invisible, unreferenced Lance column files; operator-owned
table maintenance reclaims them (see ``docs/curator/design/curator-next-robot-action-caption.md``).

Everything is refused BEFORE the commit because after it there is nothing left to
refuse: a caption is stochastic and unseeded, so a wrongly published row cannot be
recomputed into the right one. That is also why a conflicting duplicate is a hard
error rather than last-write-wins - two differing rows for one clip are two
different answers, not a retry artifact.

One commit rather than one per fragment. With N commits there are N+1 observable
table states and a crash between them leaves the table genuinely half-published,
which is what forces commit counters and a reconciliation state machine
downstream. With one there are two states, the full caption-group pending
predicate is honest at every instant, and the only failure mode is "nothing
published, staged rows intact" - the case recovery already handles.

``update_columns`` preserves unmatched LEFT rows, so a clip already captioned and
absent from this wave keeps its values. It silently DROPS unmatched right rows,
which is why a staged clip missing from the owed set is checked for here
rather than left for the commit to complain about: it never would.
"""

import json
from collections.abc import Generator, Iterable, Sequence
from contextlib import closing
from typing import Any, Final
from uuid import uuid4

import attrs
import lance
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import ray
from lance.fragment import FragmentMetadata
from loguru import logger

from cosmos_curator.next.recipes.robot_action_caption.contracts import (
    CAPTION_COLUMN,
    CAPTION_COLUMNS,
    CAPTION_DIGEST_COLUMN,
    CAPTION_ERROR_COLUMN,
    CAPTION_MODEL_COLUMN,
    CAPTION_STATUS_COLUMN,
    CaptionStatus,
    validate_terminal_values,
)
from cosmos_curator.next.recipes.robot_action_caption.datasource import FRAGMENT_ID_COLUMN
from cosmos_curator.next.recipes.robot_action_caption.request import STAGED_SCHEMA
from cosmos_curator.next.recipes.robot_action_caption.table import (
    CLIP_ID_COLUMN,
    owed_clip_ids,
)
from cosmos_curator.next.recipes.robot_action_caption.workspace import CaptionWorkspace
from cosmos_curator.next.utils.arrow_utils import filesystem_path

TRANSACTION_KIND: Final = "curator-next-robot-action-caption"
TRANSACTION_KIND_PROPERTY: Final = "kind"
PUBLICATION_ATTEMPT_PROPERTY: Final = "publication_attempt"

# The one column a worker may send the driver: a JSON payload per fragment
# carrying its serialized metadata, the rebound field ids and a row count. A lone
# string column structurally cannot carry caption text back, which is what keeps
# driver memory O(touched fragments) rather than O(published rows).
_RESULT_COLUMN: Final = "result_json"
_RESULT_SCHEMA: Final = pa.schema([pa.field(_RESULT_COLUMN, pa.large_string())])

# Marks a refused fragment. Carried as a PAYLOAD rather than raised, because Ray
# Data replaces whatever a UDF raises with a wrapper that is no longer an instance
# of its own cause: a raise here would reach the driver as an untyped RayError and
# the operator would lose the one thing that tells them what to fix.
_REFUSAL_KEY: Final = "refusal"

_FRAGMENTS_PER_TASK: Final = 1

# What a staged partition is read as, and what one prepared update table holds.
# ``fragment_id`` is absent on purpose: it is the Hive partition key, so reading
# it back would only restate the directory name it was written from.
_STAGED_COLUMNS: Final[tuple[str, ...]] = (CLIP_ID_COLUMN, *CAPTION_COLUMNS)

# How a staged row carrying NO contract digest is named in a refusal message.
# Rendering the null itself yields "None", which reads as a real digest in
# exactly the message an operator uses to identify a foreign staging tree. Any
# value that is not a 64-character hex digest works, since a real digest can
# never collide with it.
_NULL_DIGEST_LABEL: Final = "<null>"


class CaptionPublicationError(ValueError):
    """A staged-row fault that stops the run with nothing published.

    Always reaches the caller from the DRIVER: a worker detects it and reports it
    as a payload, because the raise itself does not survive the Ray task boundary
    with its type intact. A ``ValueError`` subclass so it joins the recipe's other
    pre-commit refusals as one thing a caller handles - the run stopped and the
    staged rows are still there to re-publish once the cause is fixed.
    """


class CaptionPublicationConflictError(ValueError):
    """An ambiguous commit outcome that cannot be attributed to this attempt."""


@attrs.frozen
class PublicationResult:
    """What one publication changed, in counts the driver can hold at any scale.

    Attributes:
        ending_version: A Lance version at which this wave is verified canonical.
            Normally the exact caption commit; after an ambiguous response it can
            be a later version that also includes an allowed sibling commit.
            Never null: this publication either verifies canonical state or raises.
        published_rows: Rows the commit published, of every terminal verdict.
        error_rows: How many of ``published_rows`` carry the ``error`` verdict, so
            a SUBSET of that count rather than an addition to it. Reported rather
            than gated on: a share approaching the whole wave means a fault true
            of every clip was charged one row at a time, and a published error row
            is terminal under this contract, so this count is the only thing that
            tells that apart from a wave of captions.
        truncated_rows: How many carry the ``truncated`` verdict, a subset on the
            same terms and disjoint from ``error_rows``. A caption stopped at a
            generation bound is usable text, so the wave commits either way and
            only this count separates a corpus of complete captions from one
            largely cut off. With both here the third verdict needs no field:
            ``success`` is ``published_rows`` less these two.

    """

    ending_version: int
    published_rows: int
    error_rows: int
    truncated_rows: int


def publish_captions(  # noqa: PLR0913 - a phase boundary; each input is independently meaningful
    dataset: lance.LanceDataset,
    workspace: CaptionWorkspace,
    fragment_ids: tuple[int, ...],
    *,
    digest: str,
    model_id: str,
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> PublicationResult:
    """Validate the staged rows and publish every touched fragment atomically.

    Args:
        dataset: The clip table, open at the run's pinned version. Both the read
            base of the transaction and the table it commits to.
        workspace: The namespace Phase A staged into.
        fragment_ids: Fragments the attempt owes. Never empty: the
            caller returns before reaching publication when nothing is owed, and
            an empty wave has no staged row this could validate.
        digest: The contract digest every staged row must carry.
        model_id: The resolved model id every staged row must carry.
        row_filter: Predicate defining what this attempt owes.
        storage_options: Lance credentials for the workers and the commit.

    Returns:
        The publication's accounting.

    Raises:
        CaptionPublicationError: If any fragment's staged rows do not exactly
            cover its owed clips under this run's scope and contract, if the prepared
            fragment set differs from the planned one, or if no fragment could be
            prepared while owed fragments existed.
        ValueError: If two workers claim one fragment, or the commit fails.

    """
    payloads = _prepare_fragments(
        dataset,
        workspace,
        fragment_ids,
        digest=digest,
        model_id=model_id,
        row_filter=row_filter,
        storage_options=storage_options,
    )
    if not payloads:
        # Every selected fragment held an owed row when the attempt planned it,
        # so preparing none of them means the staged tree does not describe this
        # attempt at all - and an empty commit would report that as a success.
        msg = (
            f"caption publication prepared no fragment although {len(fragment_ids)} fragment(s) of {dataset.uri} "
            f"were owed; the staged results under {workspace.results_uri} do not cover this attempt"
        )
        raise CaptionPublicationError(msg)
    return commit_prepared(
        dataset,
        payloads,
        fragment_ids,
        row_filter=row_filter,
        storage_options=storage_options,
    )


def commit_prepared(  # Coordinator inputs independently guard one atomic commit.
    dataset: lance.LanceDataset,
    payloads: list[dict[str, Any]],
    expected_fragment_ids: tuple[int, ...],
    *,
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> PublicationResult:
    """Publish every prepared fragment in ONE transaction.

    Args:
        dataset: The clip table at the version the transaction reads against.
        payloads: One decoded payload per prepared fragment.
        expected_fragment_ids: Exactly the fragments the attempt planned. A
            prepared set that differs is refused before anything is committed.
        row_filter: Predicate used to verify an ambiguous commit.
        storage_options: Lance credentials for the commit.

    Returns:
        The publication's accounting.

    Raises:
        CaptionPublicationError: If the prepared set is not exactly the expected
            set, or if decoded fragment metadata names a different fragment than
            its payload envelope.
        CaptionPublicationConflictError: If canonical rows cannot be attributed
            to this publication attempt after an ambiguous commit response.
        ValueError: If two payloads claim the same fragment, or the commit fails
            and the table shows it did not land.

    """
    prepared = {int(payload[FRAGMENT_ID_COLUMN]): payload for payload in payloads}
    if len(prepared) != len(payloads):
        # Each fragment is a single work item, so a duplicate means a retried
        # task's preparation was also collected; committing two metadata versions
        # of one fragment would make the outcome depend on collection order.
        msg = (
            f"caption publication received {len(payloads)} result(s) naming only {len(prepared)} distinct "
            f"fragment(s) of {dataset.uri}; refusing to commit two metadata versions of one fragment"
        )
        raise ValueError(msg)
    _require_expected_fragments(prepared.keys(), expected_fragment_ids, uri=dataset.uri)
    metadata = []
    for fragment_id, payload in prepared.items():
        fragment_metadata = FragmentMetadata.from_json(str(payload["metadata_json"]))
        if int(fragment_metadata.id) != fragment_id:
            msg = (
                f"caption publication payload for envelope fragment {fragment_id} carries metadata fragment "
                f"{int(fragment_metadata.id)}; refusing to commit a prepared descriptor under the wrong fragment id"
            )
            raise CaptionPublicationError(msg)
        metadata.append(fragment_metadata)
    field_ids = sorted({int(field_id) for payload in prepared.values() for field_id in payload["modified_field_ids"]})
    published_rows = sum(int(payload["rows"]) for payload in prepared.values())
    error_rows = sum(int(payload["error_rows"]) for payload in prepared.values())
    truncated_rows = sum(int(payload["truncated_rows"]) for payload in prepared.values())
    ending_version = _commit_verified(
        dataset,
        metadata,
        field_ids,
        expected_fragment_ids=expected_fragment_ids,
        row_filter=row_filter,
        storage_options=storage_options,
    )
    logger.info(
        f"Published {published_rows} caption row(s), {error_rows} of them terminal error(s) and "
        f"{truncated_rows} truncated, across {len(prepared)} fragment(s) of {dataset.uri}: "
        f"v{int(dataset.version)} -> v{ending_version}"
    )
    return PublicationResult(
        ending_version=ending_version,
        published_rows=published_rows,
        error_rows=error_rows,
        truncated_rows=truncated_rows,
    )


def staged_captions(workspace: CaptionWorkspace, fragment_id: int) -> pa.Table:
    """Read one fragment's staged terminal rows from the Hive-partitioned Parquet.

    Args:
        workspace: The namespace Phase A staged into.
        fragment_id: The fragment whose partition to read.

    Returns:
        A table of ``clip_id`` plus the five caption columns, empty when the
        partition holds no file. ``fragment_id`` is deliberately absent: it is the
        partition key, so reading it back would only restate the directory name.

    """
    filesystem = workspace.filesystem
    partition = filesystem_path(f"{workspace.results_uri.rstrip('/')}/{FRAGMENT_ID_COLUMN}={fragment_id}")
    info = filesystem.get_file_info(partition)
    if info.type != pafs.FileType.Directory:
        return pa.table({name: pa.array([], STAGED_SCHEMA.field(name).type) for name in _STAGED_COLUMNS})
    dataset = pads.dataset(partition, filesystem=filesystem, format="parquet")
    if not dataset.files:
        return pa.table({name: pa.array([], STAGED_SCHEMA.field(name).type) for name in _STAGED_COLUMNS})
    return dataset.to_table(columns=list(_STAGED_COLUMNS))


def fold_staged_rows(staged: pa.Table, *, fragment_id: int, namespace_uri: str) -> dict[str, dict[str, Any]]:
    """Collapse duplicate staged rows for one clip, refusing rows that disagree.

    A duplicate is expected because a second attempt appends its shards beside the
    first's. An IDENTICAL pair comes from a DETERMINISTIC row: a media fault or an
    oversized clip yields the same error text on every attempt, so re-staging it
    under one contract reproduces the same bytes, and the two rows are one answer
    written twice. A DIFFERING pair comes from generation, which is unseeded, so
    the two rows are two different answers and there is no basis on which to pick
    one. Why a re-generated row can reach here at all belongs to
    ``inference.py``, which owns the checkpoint's commit guarantee.

    Args:
        staged: Rows read from one fragment's partition.
        fragment_id: The fragment they belong to, for the refusal message.
        namespace_uri: The staging namespace these rows came from, named in the refusal
            as the directory to delete: every input to the namespace digest is stable
            across re-runs, so nothing else clears the conflict.

    Returns:
        One row per ``clip_id``, keyed by it.

    Raises:
        CaptionPublicationError: If two rows for one ``clip_id`` differ.

    """
    folded: dict[str, dict[str, Any]] = {}
    for row in staged.to_pylist():
        value = row[CLIP_ID_COLUMN]
        if not isinstance(value, str) or not value.strip():
            msg = f"fragment {fragment_id} has a staged caption row with invalid clip_id {value!r}"
            raise CaptionPublicationError(msg)
        clip_id = value
        existing = folded.get(clip_id)
        if existing is not None and existing != row:
            # The whole namespace, never the fragment's results partition alone. The
            # checkpoint under it still records these clips as complete, so deleting
            # only the partition makes the next run stage nothing for them and lands
            # the refusal in ``require_exact_coverage``'s missing-row branch, whose
            # remedy is to re-run Phase A - which cannot help.
            msg = (
                f"fragment {fragment_id} has conflicting staged caption rows for clip {clip_id!r}; "
                f"generation is unseeded, so these are two different answers rather than a retry artifact, "
                f"and there is no correct one to publish; every re-run resolves to the same namespace and "
                f"refuses again, so delete {namespace_uri} before re-running - the completed captions under "
                f"it are discarded and those clips are recaptioned"
            )
            raise CaptionPublicationError(msg)
        folded[clip_id] = row
    return folded


def require_exact_coverage(
    folded: dict[str, dict[str, Any]],
    owed: tuple[str, ...],
    *,
    fragment_id: int,
    digest: str,
    namespace_uri: str,
) -> None:
    """Refuse anything but exactly one current-contract row per owed clip.

    Four distinct faults, kept separate because they have different causes: a
    repeated owed clip means the producer's uniqueness guarantee broke, a
    missing row means Phase A did not finish that clip, an extra row means the
    staged tree belongs to a different attempt's row set, and a foreign digest
    means it belongs to different semantics. Only the second is fixed by
    re-running Phase A.

    Args:
        folded: One staged row per ``clip_id``.
        owed: The clips this fragment still owes a caption.
        fragment_id: The fragment, for the refusal messages.
        digest: The contract digest every row must carry.
        namespace_uri: The staging namespace these rows were read from. Named in the
            foreign-digest refusal, which rests on the namespace digest covering the
            contract - a claim a reader cannot check against an unnamed path.

    Raises:
        CaptionPublicationError: If a ``clip_id`` is owed twice, if an owed clip
            has no staged row, if a staged clip is not owed, or if any row
            carries another contract's digest.

    """
    owed_set = set(owed)
    # Checked first: a repeated clip makes every count below misleading.
    #
    # Everything else here validates against the SET, while ``prepare_fragment``
    # builds the update table from the TUPLE - one row per owed entry - so a
    # repeat would reach ``update_columns`` as two right-side rows for one join
    # key and it would publish whichever it picked. The producer does enforce
    # uniqueness (``robot_action_split/recovery.py`` refuses a duplicate on every
    # run), and this asserts that where it is load-bearing rather than re-deriving
    # it, because other consumers of this same table decline to rely on it and
    # this recipe's write path is what breaks if they are right.
    if len(owed_set) != len(owed):
        msg = (
            f"fragment {fragment_id} lists {len(owed)} owed clip(s) but only {len(owed_set)} distinct "
            f"clip_id(s); the update table carries one row per owed clip, so a repeated id becomes two rows "
            f"for one join key and update_columns would publish an arbitrary one of them"
        )
        raise CaptionPublicationError(msg)
    missing = sorted(owed_set - folded.keys())
    if missing:
        msg = (
            f"fragment {fragment_id} has {len(owed_set)} owed clip(s) but no staged caption row for "
            f"{len(missing)} of them (first: {missing[:3]}); Phase A did not finish this fragment, so re-run "
            f"it before publishing"
        )
        raise CaptionPublicationError(msg)
    extra = sorted(folded.keys() - owed_set)
    if extra:
        msg = (
            f"fragment {fragment_id} has staged caption row(s) for {len(extra)} clip(s) that this attempt does not owe "
            f"(first: {extra[:3]}); update_columns drops an unmatched staged row silently, so this is refused "
            f"here rather than published as a partial wave"
        )
        raise CaptionPublicationError(msg)
    # ONE expression feeds both the test and the message, so the two can never
    # disagree about what a row carries. A missing digest is foreign by
    # construction here, since a real digest cannot equal the null label.
    staged_digests = {str(row[CAPTION_DIGEST_COLUMN] or _NULL_DIGEST_LABEL) for row in folded.values()}
    foreign = sorted(staged_digests - {digest})
    if foreign:
        # Names the path and stops there. Unlike the conflicting duplicate, this
        # fault does not identify its own cause - rows the namespace digest could
        # not have produced are there either because two contracts reached one
        # staging tree or because something outside this recipe wrote into it - so
        # which of those it is decides whether deleting is repair or data loss.
        msg = (
            f"fragment {fragment_id} has staged caption row(s) carrying contract digest(s) {foreign} instead of "
            f"{digest}; the namespace digest covers the contract, so {namespace_uri} can hold rows staged only "
            f"under {digest}, and this run cannot account for what is there instead; the staged tree is foreign "
            f"to it and must not be published"
        )
        raise CaptionPublicationError(msg)


def prepare_fragment(  # noqa: PLR0913 - fragment identity, staged contract, and owed-row scope are independent
    dataset: lance.LanceDataset,
    workspace: CaptionWorkspace,
    fragment_id: int,
    *,
    digest: str,
    model_id: str,
    row_filter: str,
) -> dict[str, Any]:
    """Validate one fragment's staged rows and write its caption columns, without committing.

    Args:
        dataset: The clip table at the run's pinned version.
        workspace: The namespace Phase A staged into.
        fragment_id: The fragment to prepare.
        digest: The contract digest every staged row must carry.
        model_id: The resolved model id every staged row must carry.
        row_filter: Predicate defining what this attempt owes.

    Returns:
        The payload :func:`commit_prepared` commits from. Its keys are that
        function's contract, so a change here lands in both in one edit.

    Raises:
        CaptionPublicationError: If the fragment is absent at the pinned version,
            its staged rows do not exactly cover its owed clips, or a row has
            invalid terminal values or a foreign model id. Unreadable staged
            storage and invalid owed-row data are converted to this type here.

    """
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"fragment {fragment_id} is not present in {dataset.uri} at v{int(dataset.version)}"
        raise CaptionPublicationError(msg)
    try:
        staged = staged_captions(workspace, fragment_id)
    except (OSError, pa.ArrowInvalid) as e:
        msg = f"fragment {fragment_id} could not read staged captions from {workspace.results_uri}: {e}"
        raise CaptionPublicationError(msg) from e
    # Both guards receive the namespace as a STRING rather than the workspace: they
    # are pure over rows already read, so handing them the workspace would attach a
    # credentialed filesystem handle to functions that perform no I/O.
    folded = fold_staged_rows(staged, fragment_id=fragment_id, namespace_uri=workspace.root_uri)
    try:
        owed = owed_clip_ids(dataset, fragment_id, row_filter)
    except ValueError as e:
        msg = f"fragment {fragment_id} could not determine owed clips in {dataset.uri}: {e}"
        raise CaptionPublicationError(msg) from e
    require_exact_coverage(
        folded,
        owed,
        fragment_id=fragment_id,
        digest=digest,
        namespace_uri=workspace.root_uri,
    )

    # Ordered by the fragment's own owed order rather than by the staged read.
    # update_columns joins on clip_id so order does not decide correctness, but a
    # deterministic order makes the written column file reproducible for a given
    # staged set, which is what lets a re-prepared fragment be compared at all.
    # The same walk validates each durable row before it can reach update_columns.
    update_rows: list[dict[str, Any]] = []
    for clip_id in owed:
        row = folded[clip_id]
        # Only refusal matters here: _prepare_batch selects CaptionPublicationError
        # for its payload; a bare ValueError crosses Ray in a wrapper that is not
        # an instance of its cause.
        try:
            validate_terminal_values(
                status=row[CAPTION_STATUS_COLUMN],
                caption=row[CAPTION_COLUMN],
                error=row[CAPTION_ERROR_COLUMN],
            )
        except ValueError as e:
            msg = f"fragment {fragment_id} has invalid terminal values for clip {clip_id!r}: {e}"
            raise CaptionPublicationError(msg) from e
        staged_model = row[CAPTION_MODEL_COLUMN]
        if staged_model != model_id:
            msg = (
                f"fragment {fragment_id} has staged caption row for clip {clip_id!r} carrying model "
                f"{staged_model!r} instead of {model_id!r}"
            )
            raise CaptionPublicationError(msg)
        update_rows.append(row)
    update = pa.Table.from_pylist(update_rows, schema=_stored_schema(dataset))
    metadata, modified_field_ids = fragment.update_columns(
        update,
        left_on=CLIP_ID_COLUMN,
        right_on=CLIP_ID_COLUMN,
    )
    return {
        FRAGMENT_ID_COLUMN: fragment_id,
        # Embedded as the exact string ``FragmentMetadata.from_json`` expects,
        # rather than as a nested object, so the round trip through the worker's
        # result column cannot lose a field to re-encoding.
        "metadata_json": json.dumps(metadata.to_json()),
        "modified_field_ids": [int(field_id) for field_id in modified_field_ids],
        "rows": update.num_rows,
        "error_rows": _status_row_count(update_rows, CaptionStatus.ERROR),
        "truncated_rows": _status_row_count(update_rows, CaptionStatus.TRUNCATED),
    }


def _status_row_count(rows: Sequence[dict[str, Any]], status: CaptionStatus) -> int:
    """Return how many of these terminal rows carry ``status``.

    Counted off the rows the caller has already materialized to build its update
    table, so no count costs a second read of the staged partition and every one
    covers exactly the rows the commit will publish.

    Args:
        rows: Terminal caption rows. A sequence rather than an iterable because
            the caller scans the same rows once per verdict it counts.
        status: The verdict to count.

    Returns:
        How many rows carry it.

    """
    # Compared by value, not by identity: a row read back from Parquet carries a
    # plain string while contracts.terminal_row builds an enum member, and
    # ``CaptionStatus`` is a ``StrEnum`` so the two compare equal.
    return sum(1 for row in rows if row[CAPTION_STATUS_COLUMN] == status)


def _prepare_fragments(  # noqa: PLR0913 - mirrors the phase boundary it fans out
    dataset: lance.LanceDataset,
    workspace: CaptionWorkspace,
    fragment_ids: tuple[int, ...],
    *,
    digest: str,
    model_id: str,
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Prepare every fragment in parallel Ray tasks and decode their payloads.

    Raises:
        CaptionPublicationError: On the first refusal reported by any worker,
            while results are still streaming in, so the fragments not yet
            started are abandoned rather than preparing files nothing commits.

    """
    items = [{FRAGMENT_ID_COLUMN: fragment_id} for fragment_id in fragment_ids]
    # Ray accepts fn_kwargs for a keyword-only batch UDF; its public stub does not
    # model that callable shape.
    results = ray.data.from_items(items).map_batches(
        _prepare_batch,  # type: ignore[arg-type]
        batch_format="pyarrow",
        batch_size=_FRAGMENTS_PER_TASK,
        fn_kwargs={
            # Picklable values only: the worker re-opens the table at the pinned
            # version rather than receiving this driver-side handle.
            "uri": dataset.uri,
            "read_version": int(dataset.version),
            "workspace": workspace,
            "digest": digest,
            "model_id": model_id,
            "row_filter": row_filter,
            "storage_options": storage_options,
        },
    )
    payloads = []
    # Closed explicitly rather than left to refcounting. Ray Data shuts the
    # streaming executor down from the ``finally`` of its own batch generator, so
    # the shutdown only happens once this iterator is closed - and on the refusal
    # path the raised exception's traceback keeps this frame, and therefore the
    # iterator, alive for as long as the exception is unhandled. Without the
    # close, the prepare tasks still queued keep writing fragment column files
    # that no commit will ever reference.
    with closing(_iter_result_rows(results)) as rows:
        for row in rows:
            payload = json.loads(str(row))
            refusal = payload.get(_REFUSAL_KEY)
            if refusal is not None:
                raise CaptionPublicationError(str(refusal))
            payloads.append(payload)
    return payloads


def _iter_result_rows(results: ray.data.Dataset) -> Generator[str]:
    """Yield each worker's payload string as it arrives."""
    for row in results.iter_rows():
        yield str(row[_RESULT_COLUMN])


def _prepare_batch(  # noqa: PLR0913 - the Ray worker boundary carries the whole prepare contract
    batch: pa.Table,
    *,
    uri: str,
    read_version: int,
    workspace: CaptionWorkspace,
    digest: str,
    model_id: str,
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> pa.Table:
    """Prepare the fragments named in one work item, reporting a refusal as a payload."""
    # Outside the refusal try on purpose. The driver already opened this exact uri at
    # this exact version, so a failure here is this worker's environment rather than a
    # staged-row fault an operator could fix, and the refusal payload is terminal on
    # first sight. Left native, the error is matched by the OSError family in
    # ray_runtime's retried_map_errors and retried; wrapped, it would end the phase.
    dataset = lance.dataset(uri, version=read_version, storage_options=storage_options)
    payloads = []
    for fragment_id in batch.column(FRAGMENT_ID_COLUMN).to_pylist():
        try:
            payload = prepare_fragment(
                dataset,
                workspace,
                int(fragment_id),
                digest=digest,
                model_id=model_id,
                row_filter=row_filter,
            )
        except CaptionPublicationError as e:
            logger.error(f"Caption publication refused fragment {fragment_id} of {uri}: {e}")
            payload = {FRAGMENT_ID_COLUMN: int(fragment_id), _REFUSAL_KEY: str(e)}
        payloads.append(json.dumps(payload))
    return pa.table({_RESULT_COLUMN: pa.array(payloads, type=pa.large_string())}, schema=_RESULT_SCHEMA)


def _require_expected_fragments(prepared: Iterable[int], expected: tuple[int, ...], *, uri: str) -> None:
    """Refuse a prepared set that is not exactly the set the attempt planned.

    A silent partial publication is the specific failure this prevents. Preparing
    fewer fragments than planned leaves clips owed while the run reports
    success, and preparing MORE means rows from a foreign attempt are about to be
    committed - and ``update_columns`` would drop unmatched right rows without
    complaint either way.

    Args:
        prepared: Fragment ids that produced a payload.
        expected: Fragment ids the attempt planned.
        uri: Clip table URI, for the refusal message.

    Raises:
        CaptionPublicationError: If the two sets differ.

    """
    prepared_ids = set(prepared)
    expected_ids = set(expected)
    if prepared_ids == expected_ids:
        return
    msg = (
        f"caption publication prepared a different fragment set than {uri} planned: "
        f"missing {sorted(expected_ids - prepared_ids)}, unexpected {sorted(prepared_ids - expected_ids)}; "
        f"refusing to commit a partial or foreign wave"
    )
    raise CaptionPublicationError(msg)


def _commit_verified(  # noqa: PLR0913 - one commit, and every input decides whether it landed
    dataset: lance.LanceDataset,
    fragments: list[FragmentMetadata],
    field_ids: list[int],
    *,
    expected_fragment_ids: tuple[int, ...],
    row_filter: str,
    storage_options: dict[str, str] | None,
) -> int:
    """Commit every prepared fragment as ONE ``Update`` transaction.

    Args:
        dataset: The clip table at the version the transaction reads against.
        fragments: The prepared fragment metadata.
        field_ids: The field ids the update modified.
        expected_fragment_ids: Fragments that must hold no owed row once the
            commit has landed.
        row_filter: Predicate defining what this attempt owed.
        storage_options: Lance credentials for the commit.

    Returns:
        A version at which the captions are verified canonical. This is the
        exact commit version normally and may be a later version after an
        ambiguous response.

    Raises:
        ValueError: If the commit failed and the table confirms it did not land.

    """
    read_version = int(dataset.version)
    publication_attempt = str(uuid4())
    # LanceOperation.Update accepts a PARTIAL fragment list, which makes one
    # multi-fragment commit possible: only the fragments that changed are named.
    #
    # Columns outside the caption group survive because each prepared
    # FragmentMetadata enumerates EVERY one of its fragment's column files, not
    # just the file update_columns wrote, and this module stages only
    # _STAGED_COLUMNS. fields_modified does not decide that: it lists modified
    # field ids so Lance can drop those fragments from indices covering them
    # (see lance.LanceDataset.commit).
    transaction = lance.Transaction(
        read_version=read_version,
        operation=lance.LanceOperation.Update(
            updated_fragments=fragments,
            fields_modified=field_ids,
        ),
        transaction_properties={
            TRANSACTION_KIND_PROPERTY: TRANSACTION_KIND,
            PUBLICATION_ATTEMPT_PROPERTY: publication_attempt,
        },
    )
    try:
        committed = lance.LanceDataset.commit(dataset.uri, transaction, storage_options=storage_options)
    except (OSError, RuntimeError) as e:
        # Lance routes only IO faults to OSError; its commit-conflict and
        # write-contention variants fall through to RuntimeError, so both must be
        # caught. Neither, though, tells us whether the commit LANDED: a response
        # lost after the manifest was written raises here just as an outright
        # rejection does. Re-raising unconditionally would send an operator to
        # re-publish a wave that already succeeded, and re-publishing under
        # unseeded generation writes different captions.
        #
        # The table itself is the only authority, so ask it. This is the ONE
        # place a post-commit check is legitimate: it does not assert the
        # publication was correct - that was refused in advance, before any
        # commit - it only reads which of the two states the table is in.
        return _committed_version_after(
            dataset,
            expected_fragment_ids,
            row_filter,
            publication_attempt=publication_attempt,
            storage_options=storage_options,
            cause=e,
        )
    return int(committed.version)


def _committed_version_after(  # noqa: PLR0913 - every input independently verifies ambiguous commit ownership
    dataset: lance.LanceDataset,
    expected_fragment_ids: tuple[int, ...],
    row_filter: str,
    *,
    publication_attempt: str,
    storage_options: dict[str, str] | None,
    cause: Exception,
) -> int:
    """Return a version at which an ambiguous publication is verified canonical.

    Args:
        dataset: The clip table the commit targeted, at the version it read against.
        expected_fragment_ids: Fragments the wave published into.
        row_filter: The predicate defining what this attempt owed.
        publication_attempt: Unique marker stamped on this commit attempt.
        storage_options: Lance credentials for the re-open.
        cause: The error the commit raised.

    Returns:
        The latest observed version at which no expected fragment holds a
        row this attempt owes. It may include an allowed sibling commit after this wave.

    Raises:
        CaptionPublicationConflictError: If no owed row remains but no
            committed transaction carries this attempt's marker.
        ValueError: If owed rows remain, so the commit really did fail, or if
            an expected fragment is absent, leaving the outcome unverifiable.

    """
    uri = dataset.uri
    read_version = int(dataset.version)
    # Re-opened at LATEST rather than reusing the handle: the passed one is
    # pinned to the read version, where every row of the wave is owed by
    # definition, so asking it would always conclude the commit failed.
    latest = lance.dataset(uri, storage_options=storage_options)
    absent: list[int] = []
    owed_counts: list[int] = []
    for fragment_id in expected_fragment_ids:
        fragment = latest.get_fragment(fragment_id)
        if fragment is None:
            absent.append(fragment_id)
            continue
        # O(fragments) filtered counts on one column, never O(rows) on the
        # driver: the same per-fragment count planning already uses.
        owed_counts.append(int(fragment.count_rows(filter=row_filter)))
    if absent:
        # An absent fragment is UNVERIFIABLE, not benign, so it must not be
        # skipped. A successful `update_columns` commit rewrites a fragment's
        # column files and keeps its id, so a missing id cannot mean "committed";
        # it means the table moved underneath this check - a concurrent
        # compaction, say - in a way this helper cannot interpret. Skipping it
        # would contribute zero owed rows and so report a genuinely FAILED
        # commit as landed: nothing published, and the operator told it worked.
        msg = (
            f"caption publication commit against v{read_version} of {uri} failed ({cause}), and fragment(s) "
            f"{absent} the wave published into are absent at v{int(latest.version)}; whether the commit landed "
            f"cannot be determined from a table that no longer carries them"
        )
        raise ValueError(msg) from cause
    remaining = sum(owed_counts)
    if remaining:
        msg = (
            f"caption publication commit against v{read_version} of {uri} failed ({cause}); "
            f"{remaining} row(s) across the wave's fragments are still owed, so nothing was published"
        )
        raise ValueError(msg) from cause
    ending_version = int(latest.version)
    if not _history_contains_publication_attempt(
        latest,
        publication_attempt,
        after_version=read_version,
    ):
        msg = (
            f"caption publication commit against v{read_version} of {uri} failed ({cause}), and no row of the "
            f"wave is owed at v{ending_version}, but no committed version after v{read_version} carries this "
            "publication attempt's marker; cannot attribute the canonical rows to this publication attempt"
        )
        raise CaptionPublicationConflictError(msg) from cause
    logger.warning(
        f"Caption publication commit against v{read_version} of {uri} reported {cause}, "
        f"but no row of the wave is owed at v{ending_version}: the commit landed and the error "
        f"was a lost response. Treating the publication as successful."
    )
    return ending_version


def _history_contains_publication_attempt(
    dataset: lance.LanceDataset,
    publication_attempt: str,
    *,
    after_version: int,
) -> bool:
    """Return whether a committed version after the read carries this attempt marker."""
    for version in range(after_version + 1, int(dataset.version) + 1):
        transaction = dataset.read_transaction(version)
        if transaction is None:
            continue
        properties = transaction.transaction_properties or {}
        if properties.get(PUBLICATION_ATTEMPT_PROPERTY) == publication_attempt:
            return True
    return False


def _stored_schema(dataset: lance.LanceDataset) -> pa.Schema:
    """Return ``clip_id`` plus the caption columns typed exactly as the table stores them.

    Taken from the dataset schema rather than rebuilt, so the update table is cast
    to the stored types and ``update_columns`` never has to coerce - a coercion it
    would perform per row, and which would hide a real type drift.
    """
    return pa.schema([dataset.schema.field(name) for name in _STAGED_COLUMNS])
