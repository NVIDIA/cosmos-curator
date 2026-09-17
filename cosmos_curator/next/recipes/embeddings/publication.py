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

"""Publish one modality's staged rows into the clips table in one verified commit.

The second half of the leg. Staging owns computation and knows nothing about
Lance writes; this module owns the write and needs no GPU, no model, and no
embedder - only the staged Parquet a staging phase left behind.

::

    plan.fragment_ids
        |
        v
    Ray tasks, one fragment each        prepare_fragment
        |   staged Parquet partition --> cast to stored types --> update_columns
        |   returns ONE json payload: metadata + field ids + two counters
        v
    driver: refuse a duplicate, a mismatched id, a set differing from the plan,
            or a fragment whose staged rows do not tally with its owed count
        |
        v
    ONE Transaction(Update(prepared_fragments, fields_modified))

Every refusal is raised BEFORE the commit, which is what makes the commit the
leg's only completion signal: there is no state in which some fragments are
published and others are not.

A vector never reaches the driver. Each task streams its partition's batches
straight into ``update_columns``, and returns one JSON string - a lone string
column structurally cannot carry column data back, so driver memory is
O(fragments) rather than O(rows).

``update_columns`` joins on the persisted ``clip_id`` with left-outer semantics: a
fragment row absent from the staged set keeps its previous group value, which is
what makes publishing only the owed subset of a fragment safe. It writes a NEW
file holding only the group's columns for the fragment's existing rows, in their
existing offset order, so no row is replaced, no deletion vector is written, and
every column outside the group stays bound to the file it already had.

That same left-outer join is why the row tallies are checked and not merely
reported. It cannot distinguish a fragment row the plan never owed from one it
owed and lost: both keep their previous value, and the commit succeeds either
way. Only the plan's own per-fragment owed count separates the two, so the
staged rows are measured against it before anything is committed.
"""

import json
from collections.abc import Generator, Iterable, Iterator
from contextlib import closing
from typing import Any, Final
from uuid import uuid4

import attrs
import lance
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import ray
from lance.fragment import FragmentMetadata
from loguru import logger

from cosmos_curator.next.embeddings.schemas import KEY_COLUMN, EmbeddingColumnGroup
from cosmos_curator.next.recipes.embeddings.datasource import EmbeddingScanPlan
from cosmos_curator.next.recipes.embeddings.modalities import ModalityFill, ModalityResult
from cosmos_curator.next.recipes.embeddings.staged import FRAGMENT_ID_COLUMN, staged_columns
from cosmos_curator.next.recipes.embeddings.workspace import EmbeddingWorkspace
from cosmos_curator.next.utils.arrow_utils import filesystem_path, normalize_location_root
from cosmos_curator.next.utils.identity import canonical_digest

TRANSACTION_KIND: Final = "curator-next-embedding"
"""Value of the commit's ``kind`` property, making an embedding commit attributable.

Read back by the ambiguous-commit check, so it is a persisted contract rather
than a log label: changing it strands the marker of any attempt in flight.
"""

_TRANSACTION_KIND_PROPERTY: Final = "kind"
_GROUP_PROPERTY: Final = "group"
_RESULT_IDENTITY_PROPERTY: Final = "result_identity"
_PUBLICATION_ATTEMPT_PROPERTY: Final = "publication_attempt"

# The single column a prepare task may return: one JSON payload per fragment.
_RESULT_COLUMN: Final = "result_json"
_RESULT_SCHEMA: Final = pa.schema([pa.field(_RESULT_COLUMN, pa.large_string())])

# Discriminates the refusal payload shape from the prepared one.
_REFUSAL_KEY: Final = "refusal"

# One fragment per task, so a task's memory is bounded by one fragment's
# in-flight batch rather than by however many fragments Ray bundled together.
_FRAGMENTS_PER_TASK: Final = 1


class EmbeddingPublicationError(ValueError):
    """A staged-row fault refused before any commit.

    A ``ValueError`` subclass so it joins the recipe's other pre-commit refusals
    as one thing a caller handles: the run stopped with the group untouched.

    It always reaches the caller from the DRIVER. A prepare task raises it
    internally, but its own handler turns that into a refusal payload, because a
    raise would otherwise arrive at the driver as a Ray wrapper of another type.
    """


@attrs.define
class _StagedTally:
    """Row tallies and any refusal seen while one fragment's update stream is consumed.

    ``update_columns`` pulls the batches itself, so neither the counts nor a cast
    failure is known until it returns; a small mutable holder lets the generator
    report what it saw without buffering the batches to count them first.

    Attributes:
        rows: Staged rows this fragment's partition yielded. Reported in the
            run's accounting AND compared against the fragment's planned owed
            count before the commit, so it is a contract rather than a statistic.
        filled: Of those, the rows whose primary vector is non-NULL.
        refusal: Message of a refusal the generator raised, recorded because the
            raise itself does not survive the Arrow reader boundary.

    """

    rows: int = 0
    filled: int = 0
    refusal: str | None = None


def staged_batches(
    workspace: EmbeddingWorkspace, fragment_id: int, *, group: EmbeddingColumnGroup
) -> Iterator[pa.RecordBatch]:
    """Stream one fragment's staged rows out of its Hive partition.

    Never calls ``to_table``: one fragment's text group runs to hundreds of
    megabytes, and materializing it would defeat the streaming hand-off into
    ``update_columns`` that keeps a task's memory at one batch.

    Args:
        workspace: The namespace a staging phase wrote into.
        fragment_id: The fragment whose partition to read.
        group: The modality's column group, deciding the projection.

    Yields:
        Batches of ``clip_id`` plus the group's columns, in staged types.
        ``fragment_id`` is deliberately absent: it is the partition key, so
        reading it back would only restate the directory name.

    Raises:
        EmbeddingPublicationError: If the partition is missing, is not a
            directory, or holds no file. Unlike a caption wave, this leg plans
            only fragments that owe rows, so an empty partition means the staging
            phase did not finish - not that there was nothing to stage.

    """
    filesystem = workspace.filesystem
    partition = filesystem_path(f"{normalize_location_root(workspace.results_uri)}/{FRAGMENT_ID_COLUMN}={fragment_id}")
    info = filesystem.get_file_info(partition)
    if info.type != pafs.FileType.Directory:
        msg = (
            f"staged partition {partition} for fragment {fragment_id} is {info.type!s} rather than a directory; "
            f"the staging phase did not write this fragment, so its rows are not available to publish"
        )
        raise EmbeddingPublicationError(msg)
    files = [file.path for file in filesystem.get_file_info(pafs.FileSelector(partition, recursive=True))]
    if not files:
        msg = (
            f"staged partition {partition} for fragment {fragment_id} holds no file, but the fragment was "
            f"planned as owing rows; refusing to publish a fragment whose staged rows are absent"
        )
        raise EmbeddingPublicationError(msg)
    try:
        dataset = pads.dataset(partition, filesystem=filesystem, format="parquet")
        yield from dataset.to_batches(columns=list(staged_columns(group)))
    except (OSError, pa.ArrowInvalid) as e:
        # Read faults are converted rather than propagated, unlike in
        # ``staged_distinct_values`` where the same fault is tolerated. There a
        # partial shard is expected, because that read runs before the phase that
        # writes it; here the staging phase has already reported success, so an
        # unreadable file is corruption. Left native it would match
        # ray_runtime's retried OSError family and be retried against a file that
        # cannot become readable.
        msg = (
            f"staged partition {partition} for fragment {fragment_id} cannot be read ({e!r}); "
            f"refusing to publish a fragment whose staged rows are unreadable"
        )
        raise EmbeddingPublicationError(msg) from e


def staged_distinct_values(workspace: EmbeddingWorkspace, column: str, *, max_values: int) -> tuple[str, ...]:
    """Return up to ``max_values`` distinct non-NULL values of one staged column.

    The only staged read that runs BEFORE a phase rather than after it, and what
    makes the action basis resolvable on a retry that crashed between staging and
    publication.

    Args:
        workspace: The namespace to read. A namespace that does not exist yet
            yields no values, which is the ordinary first-attempt case.
        column: Staged column to read.
        max_values: Stop once this many distinct values have been seen. The
            caller's ceiling on ambiguity, not a performance knob: two values are
            already enough to refuse.

    Returns:
        The distinct values found, in no particular order.

    """
    filesystem = workspace.filesystem
    results = filesystem_path(normalize_location_root(workspace.results_uri))
    info = filesystem.get_file_info(results)
    if info.type != pafs.FileType.Directory:
        return ()
    values: set[str] = set()
    # Read file by file rather than through one ``pyarrow.dataset``: a single torn
    # shard would fail a whole-directory read, and this runs BEFORE the phase that
    # would reclaim it. Per-file, one bad shard costs its own rows only.
    #
    # No name-based filter excludes a partial file, and none would help: Ray's
    # ``*.pending.parquet`` markers live under the checkpoint path, never beside
    # the data, so a suffix filter here would match nothing while reading as
    # though it guarded something. A torn shard is recognized by FAILING TO READ.
    for file_info in filesystem.get_file_info(pafs.FileSelector(results, recursive=True)):
        if file_info.type != pafs.FileType.File:
            continue
        try:
            staged = pq.read_table(file_info.path, columns=[column], filesystem=filesystem)
        except (OSError, pa.ArrowInvalid) as e:
            # A file the writer did not finish has no Parquet footer, so this is
            # the expected shape of a crashed attempt's last shard rather than an
            # anomaly. Warned rather than raised: the next staging phase's
            # checkpoint load reclaims it.
            logger.warning(
                f"skipping unreadable staged file {file_info.path} while reading {column!r} ({e!r}); "
                f"a partial shard from an interrupted attempt is reclaimed by the next run"
            )
            continue
        values.update(value for value in staged.column(column).to_pylist() if value is not None)
        if len(values) >= max_values:
            break
    return tuple(values)


def prepare_fragment(
    dataset: lance.LanceDataset, workspace: EmbeddingWorkspace, fragment_id: int, *, group: EmbeddingColumnGroup
) -> dict[str, Any]:
    """Write one fragment's staged group columns and return the payload the commit needs.

    Runs inside a Ray task. The staged batches are streamed into
    ``update_columns``, so peak memory is one batch rather than one fragment.

    Args:
        dataset: The clips table at the run's pinned version.
        workspace: The namespace holding this fragment's staged rows.
        fragment_id: The fragment to write.
        group: The modality's column group.

    Returns:
        ``{"fragment_id", "metadata_json", "modified_field_ids", "rows", "filled"}``.

    Raises:
        EmbeddingPublicationError: If the fragment is absent at the pinned
            version, its staged partition is unusable, or a staged batch does not
            cast to the stored types - which is where a wrong-width vector is
            caught, the staged schema having deliberately dropped the width so
            Parquet could round-trip a NULL row.

    """
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"fragment {fragment_id} is not present in {dataset.uri} at v{int(dataset.version)}"
        raise EmbeddingPublicationError(msg)
    stored = _stored_schema(dataset, group)
    tally = _StagedTally()
    reader = pa.RecordBatchReader.from_batches(stored, _stored_batches(workspace, fragment_id, group, stored, tally))
    try:
        metadata, modified_field_ids = fragment.update_columns(reader, left_on=KEY_COLUMN, right_on=KEY_COLUMN)
    except Exception as e:
        # Lance pulls the batches through the Arrow C data interface, which
        # carries an error MESSAGE but not the Python exception object, so a
        # refusal raised inside the generator re-emerges here as a bare
        # RuntimeError. Re-raising it from the recorded message is what keeps the
        # refusal attributable to the staged row that caused it.
        if tally.refusal is not None:
            raise EmbeddingPublicationError(tally.refusal) from e
        raise
    # These keys ARE the driver's contract; a change here lands in
    # ``_prepared_fragments`` in the same edit. The metadata is embedded as the
    # exact string ``FragmentMetadata.from_json`` expects rather than as a nested
    # object, so the round trip cannot lose a field to re-encoding.
    return {
        FRAGMENT_ID_COLUMN: fragment_id,
        "metadata_json": json.dumps(metadata.to_json()),
        "modified_field_ids": [int(field_id) for field_id in modified_field_ids],
        "rows": tally.rows,
        "filled": tally.filled,
    }


def publish_embedding_group(
    dataset: lance.LanceDataset,
    fill: ModalityFill,
    *,
    workspace: EmbeddingWorkspace,
    plan: EmbeddingScanPlan,
    storage_options: dict[str, str] | None,
) -> ModalityResult:
    """Publish every planned fragment's staged rows as one commit, or publish nothing.

    Args:
        dataset: The clips table at the version the transaction reads against.
        fill: The modality whose group is published; its result identity is
            stamped on the commit.
        workspace: The namespace holding the staged rows.
        plan: The fragments the attempt planned. Exactly this set must be
            prepared.
        storage_options: Lance credentials for the prepare tasks and the commit.

    Returns:
        The run's accounting, summed from the prepare tasks. These tallies are
        the run's only source of counts.

    Raises:
        EmbeddingPublicationError: On any staged-row fault, a duplicate or
            mismatched fragment payload, a prepared set differing from the plan,
            or a fragment whose staged rows do not cover exactly what it owed.
            Always before the commit.
        ValueError: If the commit failed and the table confirms it did not land.

    """
    payloads = _prepared_fragments(dataset, fill.group, workspace=workspace, plan=plan, storage_options=storage_options)
    fragments = _fragment_metadata(payloads, group_name=fill.group.name)
    _require_expected_fragments(fragments, plan.fragment_ids, uri=dataset.uri)
    # Strictly after the set check, which is what guarantees every payload has a
    # planned counterpart to be measured against.
    _require_exact_coverage(payloads, plan, group_name=fill.group.name)
    field_ids = sorted({int(field_id) for payload in payloads for field_id in payload["modified_field_ids"]})
    read_version = int(dataset.version)
    committed_version = _commit_verified(
        dataset,
        list(fragments.values()),
        field_ids,
        fill=fill,
        storage_options=storage_options,
    )
    selected = sum(int(payload["rows"]) for payload in payloads)
    filled = sum(int(payload["filled"]) for payload in payloads)
    logger.info(
        f"embedding group {fill.group.name!r}: published {filled}/{selected} row(s) across "
        f"{len(fragments)} fragment(s); v{read_version} -> v{committed_version}"
    )
    return ModalityResult(
        modality=fill.modality,
        result_identity=dict(fill.result_identity),
        selected=selected,
        filled=filled,
        committed_version=committed_version,
    )


def _stored_batches(
    workspace: EmbeddingWorkspace,
    fragment_id: int,
    group: EmbeddingColumnGroup,
    stored: pa.Schema,
    tally: _StagedTally,
) -> Generator[pa.RecordBatch]:
    """Yield one fragment's staged batches cast to the stored types, tallying as it goes.

    The cast is the width guard as well as a type conversion: the staged vector is
    a variable-width list so Parquet can round-trip a NULL row, and this is where
    it becomes the fixed-size list the table declares. A row of the wrong width
    therefore fails here rather than being written.

    EVERY refusal leaves through the one handler at the bottom, which records it on
    ``tally``, because the caller pulls these batches from beneath an Arrow reader
    boundary that keeps a message and discards its type. A refusal raised on a
    path that skipped the recording would reach the caller as a bare
    ``RuntimeError`` naming nothing an operator can act on.
    """
    try:
        for batch in staged_batches(workspace, fragment_id, group=group):
            if not batch.num_rows:
                continue
            try:
                converted = pa.Table.from_batches([batch]).cast(stored)
            except (ValueError, pa.ArrowTypeError, pa.ArrowNotImplementedError) as e:
                # The three arms a contract violation reaches, not the whole Arrow
                # family: pa.ArrowException would be shorter but also catches
                # ArrowMemoryError, and an allocation failure is transient rather
                # than a statement about the staged rows. ValueError is load-bearing
                # and not redundant with the Arrow arms: measured at pyarrow 25.0.1,
                # a wrong vector width raises ArrowInvalid, which subclasses
                # ValueError rather than ArrowTypeError.
                msg = (
                    f"staged rows for fragment {fragment_id} of group {group.name!r} do not cast to the stored "
                    f"schema: {e}. Expected {stored.to_string(show_field_metadata=False)}, "
                    f"got {batch.schema.to_string(show_field_metadata=False)}"
                )
                raise EmbeddingPublicationError(msg) from e
            tally.rows += converted.num_rows
            tally.filled += converted.num_rows - converted.column(group.primary_vector).null_count
            yield from converted.to_batches()
    except EmbeddingPublicationError as e:
        tally.refusal = str(e)
        raise


def _prepared_fragments(
    dataset: lance.LanceDataset,
    group: EmbeddingColumnGroup,
    *,
    workspace: EmbeddingWorkspace,
    plan: EmbeddingScanPlan,
    storage_options: dict[str, str] | None,
) -> list[dict[str, Any]]:
    """Prepare every planned fragment in parallel Ray tasks and decode their payloads.

    Raises:
        EmbeddingPublicationError: On the first refusal any task reports, while
            results are still streaming in, so the fragments not yet started are
            abandoned rather than writing column files nothing will reference.

    """
    items = [{FRAGMENT_ID_COLUMN: fragment_id} for fragment_id in plan.fragment_ids]
    # One block per fragment, stated rather than autodetected. Task count follows
    # BLOCK count, and ``from_items`` autodetects
    # ``min(len(items), max(200, 2 * avail_cpus))`` - a cluster-sized figure that
    # has no reason to equal the fragment count. Left to autodetect, a batch_size
    # of one fragment does not buy one task per fragment; it buys that many tasks
    # each walking its block's fragments in series, and every fragment after the
    # first in a block commits only once its predecessors have. The override
    # cannot over-split, since the autodetected value is itself clamped to
    # ``len(items)``. Equal by construction means the owed-fragment count already
    # logged IS the width of this phase.
    results = ray.data.from_items(items, override_num_blocks=len(items)).map_batches(
        _prepare_batch,  # type: ignore[arg-type]
        batch_format="pyarrow",
        batch_size=_FRAGMENTS_PER_TASK,
        fn_kwargs={
            # Picklable values only: the task re-opens the table at the pinned
            # version rather than receiving this driver-side handle.
            "uri": dataset.uri,
            "read_version": int(dataset.version),
            "workspace": workspace,
            "group": group,
            "storage_options": storage_options,
        },
    )
    payloads = []
    # Closed explicitly rather than left to refcounting. Ray Data shuts the
    # streaming executor down from the ``finally`` of its own batch generator, so
    # the shutdown only happens once this iterator is closed - and on the refusal
    # path the raised exception's traceback keeps this frame, and therefore the
    # iterator, alive for as long as the exception is unhandled.
    with closing(_iter_result_rows(results)) as rows:
        for row in rows:
            payload = json.loads(row)
            refusal = payload.get(_REFUSAL_KEY)
            if refusal is not None:
                raise EmbeddingPublicationError(str(refusal))
            payloads.append(payload)
    return payloads


def _iter_result_rows(results: ray.data.Dataset) -> Generator[str]:
    """Yield each task's payload string as it arrives."""
    for row in results.iter_rows():
        yield str(row[_RESULT_COLUMN])


def _prepare_batch(  # noqa: PLR0913 - the Ray task boundary carries the whole prepare contract
    batch: pa.Table,
    *,
    uri: str,
    read_version: int,
    workspace: EmbeddingWorkspace,
    group: EmbeddingColumnGroup,
    storage_options: dict[str, str] | None,
) -> pa.Table:
    """Prepare the fragments named in one work item, reporting a refusal as a payload."""
    # Outside the refusal try on purpose. The driver already opened this exact uri
    # at this exact version, so a failure here is this task's environment rather
    # than a staged-row fault an operator could fix, and a refusal payload is
    # terminal on first sight. Left native, the error is matched by the OSError
    # family in ray_runtime's retried_map_errors and retried; wrapped, it would
    # end the phase.
    dataset = lance.dataset(uri, version=read_version, storage_options=storage_options)
    payloads = []
    for fragment_id in batch.column(FRAGMENT_ID_COLUMN).to_pylist():
        try:
            payload = prepare_fragment(dataset, workspace, int(fragment_id), group=group)
        except EmbeddingPublicationError as e:
            logger.error(f"embedding publication refused fragment {fragment_id} of {uri}: {e}")
            payload = {FRAGMENT_ID_COLUMN: int(fragment_id), _REFUSAL_KEY: str(e)}
        payloads.append(json.dumps(payload))
    return pa.table({_RESULT_COLUMN: pa.array(payloads, type=pa.large_string())}, schema=_RESULT_SCHEMA)


def _fragment_metadata(payloads: list[dict[str, Any]], *, group_name: str) -> dict[int, FragmentMetadata]:
    """Decode each payload's metadata, refusing a duplicate or a self-inconsistent one.

    Raises:
        EmbeddingPublicationError: If two payloads claim one fragment, or a
            payload's decoded metadata names a different fragment than its
            envelope. The first would make the commit's outcome depend on
            ordering; the second means the payload and the file it describes
            disagree, so the envelope's id cannot be trusted to place it.

    """
    metadata = {}
    for payload in payloads:
        fragment_id = int(payload[FRAGMENT_ID_COLUMN])
        if fragment_id in metadata:
            msg = (
                f"embedding group {group_name!r}: two publication payloads claim fragment {fragment_id}; "
                f"refusing to commit two metadata versions of one fragment"
            )
            raise EmbeddingPublicationError(msg)
        decoded = FragmentMetadata.from_json(str(payload["metadata_json"]))
        if int(decoded.id) != fragment_id:
            msg = (
                f"embedding group {group_name!r}: publication payload for fragment {fragment_id} carries "
                f"metadata for fragment {int(decoded.id)}; refusing to commit metadata whose identity "
                f"disagrees with the fragment it was prepared for"
            )
            raise EmbeddingPublicationError(msg)
        metadata[fragment_id] = decoded
    return metadata


def _require_expected_fragments(prepared: Iterable[int], expected: tuple[int, ...], *, uri: str) -> None:
    """Refuse a prepared set that is not exactly the set the attempt planned.

    A silent partial publication is the specific failure this prevents. Preparing
    FEWER fragments than planned leaves rows owed while the run reports success,
    and preparing MORE means rows from a foreign attempt are about to be
    committed - and ``update_columns`` would drop unmatched right rows without
    complaint either way.

    Raises:
        EmbeddingPublicationError: If the two sets differ.

    """
    prepared_ids = set(prepared)
    expected_ids = set(expected)
    if prepared_ids == expected_ids:
        return
    msg = (
        f"embedding publication prepared a different fragment set than {uri} planned: "
        f"missing {sorted(expected_ids - prepared_ids)}, unexpected {sorted(prepared_ids - expected_ids)}; "
        f"refusing to commit a partial or foreign set"
    )
    raise EmbeddingPublicationError(msg)


def _require_exact_coverage(payloads: list[dict[str, Any]], plan: EmbeddingScanPlan, *, group_name: str) -> None:
    """Refuse a fragment whose staged rows are not exactly the rows it owed.

    The set check above establishes that every planned FRAGMENT was prepared; this
    establishes that each one carried every planned ROW. Without it a partition
    holding some of its rows publishes as a success, because ``update_columns``
    is left-outer: an owed row absent from the staged set keeps whatever it held,
    so the group commits with those rows still NULL or still stale.

    Equality is exact in both directions rather than a floor. The namespace pins
    the read version and the recovery contract, the checkpoint keeps the
    attempts' staged rows disjoint, and the worker emits one row per scanned
    row, so a resume re-derives the same per-fragment count. It does NOT pin
    ``applicability_filter``, which also feeds ``pending_filter``: narrowing one
    between attempts shrinks the owed count while resolving the same namespace,
    and this refuses rather than publish the surplus. Action's contract is its
    descriptor version, its basis being re-bound from the staged rows instead.

    Args:
        payloads: One prepared payload per fragment, each carrying its ``rows``
            tally. Must already have passed :func:`_require_expected_fragments`,
            which is what guarantees the lookup below resolves.
        plan: The attempt's plan, holding each fragment's owed-row count.
        group_name: Named in the refusal.

    Raises:
        EmbeddingPublicationError: If any fragment's tally differs from its plan.

    """
    owed = {fragment.fragment_id: fragment.num_rows for fragment in plan.fragments}
    mismatched = []
    for payload in payloads:
        fragment_id = int(payload[FRAGMENT_ID_COLUMN])
        staged = int(payload["rows"])
        # Counts rather than checkpoint ids: both numbers are already in hand, so
        # this is O(fragments) on the driver, while comparing id sets would cost
        # O(rows) to catch only a same-cardinality substitution of one owed row
        # for another - which no scan of this fragment can produce.
        if staged != owed[fragment_id]:
            mismatched.append((fragment_id, staged, owed[fragment_id]))
    if not mismatched:
        return
    detail = ", ".join(
        f"fragment {fragment_id} staged {staged} of {expected} owed row(s)"
        for fragment_id, staged, expected in sorted(mismatched)
    )
    msg = (
        f"embedding group {group_name!r}: staged rows do not cover the plan ({detail}); refusing to commit a "
        f"group whose staged rows and planned rows disagree. Fewer staged than owed would publish as a "
        f"success while leaving those rows stale; more means the namespace holds rows this plan did not "
        f"produce. Delete the staging namespace to recompute the group from the table"
    )
    raise EmbeddingPublicationError(msg)


def _commit_verified(
    dataset: lance.LanceDataset,
    fragments: list[FragmentMetadata],
    field_ids: list[int],
    *,
    fill: ModalityFill,
    storage_options: dict[str, str] | None,
) -> int:
    """Commit every prepared fragment as ONE ``Update`` transaction.

    Returns:
        A version at which the group is verified canonical: the commit's own
        version normally, and a later observed version after an ambiguous
        response.

    Raises:
        ValueError: If the commit failed and the table cannot confirm it landed.

    """
    read_version = int(dataset.version)
    publication_attempt = str(uuid4())
    # LanceOperation.Update accepts a PARTIAL fragment list, which is what makes
    # one multi-fragment commit possible: only the fragments that changed are
    # named, and every other fragment keeps its existing metadata.
    #
    # Columns outside the group survive because each prepared FragmentMetadata
    # enumerates EVERY one of its fragment's column files, not just the file
    # update_columns wrote. fields_modified does not decide that: it lists the
    # modified field ids so Lance can drop those fragments from indices covering
    # them (see lance.LanceDataset.commit).
    transaction = lance.Transaction(
        read_version=read_version,
        operation=lance.LanceOperation.Update(
            updated_fragments=fragments,
            fields_modified=field_ids,
        ),
        transaction_properties={
            _TRANSACTION_KIND_PROPERTY: TRANSACTION_KIND,
            _GROUP_PROPERTY: fill.group.name,
            # Digested rather than spelled out: the identity is a mapping whose
            # values are model fingerprints, and a transaction property is one
            # string. Equality is all a later reader needs from it.
            _RESULT_IDENTITY_PROPERTY: canonical_digest(fill.result_identity),
            _PUBLICATION_ATTEMPT_PROPERTY: publication_attempt,
        },
    )
    try:
        committed = lance.LanceDataset.commit(dataset.uri, transaction, storage_options=storage_options)
    except (OSError, RuntimeError) as e:
        # Lance routes only IO faults to OSError; its commit-conflict and
        # write-contention variants fall through to RuntimeError, so both must be
        # caught. Neither, though, says whether the commit LANDED: a response lost
        # after the manifest was written raises here exactly as an outright
        # rejection does. Re-raising unconditionally would send an operator to
        # re-publish a group that already succeeded.
        return _committed_version_after(
            dataset,
            publication_attempt=publication_attempt,
            storage_options=storage_options,
            cause=e,
        )
    return int(committed.version)


def _committed_version_after(
    dataset: lance.LanceDataset,
    *,
    publication_attempt: str,
    storage_options: dict[str, str] | None,
    cause: Exception,
) -> int:
    """Return the version an ambiguous commit landed at, by looking for its own marker.

    The marker is unique per attempt, so finding it in the table's history is
    proof this attempt's commit landed. No owed-row recount is performed
    alongside: a row count can be explained by a sibling attempt, whereas the
    marker cannot, so the second scan would cost a corpus walk and add nothing.

    Raises:
        ValueError: If no committed version after the read carries this attempt's
            marker, so the commit really did fail.

    """
    uri = dataset.uri
    read_version = int(dataset.version)
    # Re-opened at LATEST rather than reusing the handle, which is pinned to the
    # read version and so cannot see the commit under investigation.
    latest = lance.dataset(uri, storage_options=storage_options)
    ending_version = int(latest.version)
    marker_version = _publication_attempt_version(latest, publication_attempt, after_version=read_version)
    if marker_version is None:
        msg = (
            f"embedding group publication commit against v{read_version} of {uri} failed ({cause}), and no "
            f"committed version through v{ending_version} carries this attempt's marker, so the commit did "
            f"not land and the group is untouched. Re-run to republish from the staged rows"
        )
        raise ValueError(msg) from cause
    logger.warning(
        f"embedding group publication commit against v{read_version} of {uri} reported {cause}, but "
        f"v{marker_version} carries this attempt's marker: the commit landed and the error was a lost "
        f"response. Treating the publication as successful"
    )
    return marker_version


def _publication_attempt_version(
    dataset: lance.LanceDataset, publication_attempt: str, *, after_version: int
) -> int | None:
    """Return the committed version that carries this attempt's marker, if any."""
    for version in range(after_version + 1, int(dataset.version) + 1):
        transaction = dataset.read_transaction(version)
        if transaction is None:
            continue
        properties = transaction.transaction_properties or {}
        if properties.get(_PUBLICATION_ATTEMPT_PROPERTY) == publication_attempt:
            return version
    return None


def _stored_schema(dataset: lance.LanceDataset, group: EmbeddingColumnGroup) -> pa.Schema:
    """Return ``clip_id`` plus the group's fields typed exactly as the table stores them.

    Taken from the dataset schema rather than rebuilt, so the update table is cast
    to the stored types (``clip_id`` may be ``large_string``) and
    ``update_columns`` never has to coerce - a coercion it would perform per row,
    and which would hide a real type drift.
    """
    return pa.schema([dataset.schema.field(name) for name in staged_columns(group)])
