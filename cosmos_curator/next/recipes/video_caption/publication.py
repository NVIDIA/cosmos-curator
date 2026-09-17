# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase B: validate durable results and atomically publish one Lance fragment at a time."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import ray
from lance.fragment import FragmentMetadata, LanceFragment
from loguru import logger
from ray.data.expressions import col

from cosmos_curator.next.recipes.video_caption.bindings import CaptionRunBindings
from cosmos_curator.next.recipes.video_caption.inference import staged_result_files
from cosmos_curator.next.recipes.video_caption.lance_state import (
    CaptionAttempt,
    FragmentCaptionState,
    classify_fragment,
    fragment_fingerprint,
    fragment_metadata_json,
    leaf_field_ids,
    protected_binding_fingerprint,
)
from cosmos_curator.next.recipes.video_caption.workspace import CaptionWorkspace

_BYTES_PER_GIB = 1024**3
_PUBLICATION_PREPARE_MEMORY_BYTES = 8 * _BYTES_PER_GIB
_FINGERPRINT_BATCH_ROWS = 4_096
_RESULT_PATH_SCHEMA = pa.schema([pa.field("path", pa.large_string(), nullable=False)])
PREPARED_PUBLICATION_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("payload", pa.large_string(), nullable=False),
    ]
)


class PublicationError(ValueError):
    """A staged-result or physical-fragment contract violation."""


@dataclass(frozen=True)
class AttemptFragmentGuard:
    """The V_attempt identity that staged results are allowed to enrich."""

    fragment_id: int
    row_count: int
    ordered_row_ids_fingerprint: str
    base_binding_fingerprint: str
    pending_binding_fingerprint: str


@dataclass(frozen=True)
class PreparedFragmentUpdate:
    """A validated uncommitted caption-column descriptor."""

    fragment_id: int
    source_version: int
    source_metadata_json: str
    source_fingerprint: str
    source_pending_binding_fingerprint: str
    updated_metadata_json: str
    modified_field_ids: tuple[int, ...]
    row_count: int


@dataclass(frozen=True)
class PreparedFragmentPublication:
    """Small worker-to-driver control record for one complete staged fragment."""

    guard: AttemptFragmentGuard
    staged_values_fingerprint: str
    descriptor: PreparedFragmentUpdate | None


@dataclass(frozen=True)
class PublicationSummary:
    """Counts from fragment-level reconciliation."""

    published_fragments: int
    already_committed_fragments: int


def publish_staged_results(  # noqa: PLR0913 -- the publication boundary keeps recovery inputs explicit
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Publish every fragment that was pending at ``V_attempt`` from durable Parquet."""
    if commit_attempts < 1:
        msg = f"Caption commit attempts must be positive, got {commit_attempts}"
        raise ValueError(msg)
    logger.info(
        "Starting caption Lance publication for field {} from attempt v{}: {} selected fragment(s) "
        "({} pending, {} already complete)",
        bindings.result.caption_field_name,
        attempt.version,
        len(attempt.selected_fragment_ids),
        len(attempt.pending_fragment_ids),
        len(attempt.complete_fragment_ids),
    )
    if not attempt.pending_fragment_ids:
        verify_selected_fragments(uri, attempt, bindings, storage_options=storage_options)
        return PublicationSummary(published_fragments=0, already_committed_fragments=0)

    # Ray Data otherwise replaces user exceptions with UserCodeException, which
    # would hide the recipe's actionable staging-contract errors at this boundary.
    with _raise_original_map_exceptions():
        return _publish_pending_staged_results(
            uri,
            attempt,
            workspace,
            bindings,
            storage_options=storage_options,
            commit_attempts=commit_attempts,
        )


def _publish_pending_staged_results(  # noqa: PLR0913 -- the publication boundary keeps recovery inputs explicit
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> PublicationSummary:
    """Prepare pending fragments across Ray workers and serialize their commits."""
    files = staged_result_files(workspace)
    if not files:
        msg = f"No staged caption results exist for {len(attempt.pending_fragment_ids)} pending fragment(s)"
        raise PublicationError(msg)
    logger.info(
        "Preparing {} pending caption fragment(s) across Ray workers from {} staged Parquet file(s)",
        len(attempt.pending_fragment_ids),
        len(files),
    )
    _validate_parquet_schemas_distributed(files, workspace, bindings)

    attempt_dataset = lance.dataset(uri, version=attempt.version, storage_options=storage_options)
    guards = _capture_fragment_guards(attempt_dataset, attempt.complete_fragment_ids, bindings)
    prepared_publications = _prepared_publication_dataset(
        files,
        uri=uri,
        attempt=attempt,
        workspace=workspace,
        bindings=bindings,
        storage_options=storage_options,
    )

    published = 0
    already_committed = 0
    pending_fragment_ids = set(attempt.pending_fragment_ids)
    seen_fragment_ids: set[int] = set()
    staged_value_fingerprints: dict[int, str] = {}
    for prepared in _iter_prepared_publications(prepared_publications):
        fragment_id = prepared.guard.fragment_id
        if fragment_id not in pending_fragment_ids:
            msg = f"Ray prepared an unexpected caption fragment {fragment_id}"
            raise PublicationError(msg)
        if fragment_id in seen_fragment_ids:
            msg = f"Ray prepared duplicate caption descriptors for fragment {fragment_id}"
            raise PublicationError(msg)
        seen_fragment_ids.add(fragment_id)
        guards[fragment_id] = prepared.guard
        staged_value_fingerprints[fragment_id] = prepared.staged_values_fingerprint
        logger.info(
            "Received caption fragment {} preparation from Ray: rows={}, source_version={}",
            fragment_id,
            prepared.guard.row_count,
            None if prepared.descriptor is None else prepared.descriptor.source_version,
        )
        was_published = _publish_one_fragment(
            uri,
            attempt_dataset,
            attempt,
            workspace,
            prepared,
            bindings,
            storage_options=storage_options,
            commit_attempts=commit_attempts,
        )
        if was_published:
            published += 1
        else:
            already_committed += 1

    missing_fragment_ids = pending_fragment_ids - seen_fragment_ids
    if missing_fragment_ids:
        msg = "Ray did not prepare staged caption results for fragment(s): " + ", ".join(
            str(fragment_id) for fragment_id in sorted(missing_fragment_ids)
        )
        raise PublicationError(msg)

    verify_selected_fragments(
        uri,
        attempt,
        bindings,
        storage_options=storage_options,
        guards=guards,
        staged_value_fingerprints=staged_value_fingerprints,
    )
    return PublicationSummary(published_fragments=published, already_committed_fragments=already_committed)


def verify_selected_fragments(  # noqa: PLR0913 -- explicit canonical-verification contract
    uri: str,
    attempt: CaptionAttempt,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    guards: dict[int, AttemptFragmentGuard] | None = None,
    staged_value_fingerprints: dict[int, str] | None = None,
) -> None:
    """Require all finite-attempt fragments to be canonical before cleanup."""
    attempt_dataset = lance.dataset(uri, version=attempt.version, storage_options=storage_options)
    current = lance.dataset(uri, storage_options=storage_options)
    logger.info(
        "Verifying caption publication at Lance v{} before cleanup: field={}, {} selected fragment(s)",
        current.version,
        bindings.result.caption_field_name,
        len(attempt.selected_fragment_ids),
    )
    _assert_nullable_schema_extension(attempt_dataset.schema, current.schema)
    resolved_guards = guards or _capture_fragment_guards(attempt_dataset, attempt.selected_fragment_ids, bindings)
    staged_fingerprints = staged_value_fingerprints or {}
    for fragment_id in attempt.selected_fragment_ids:
        fragment = _require_fragment(current, fragment_id)
        guard = resolved_guards[fragment_id]
        _assert_base_identity(current, fragment, guard, bindings)
        state = classify_fragment(fragment, bindings=bindings)
        if state is not FragmentCaptionState.COMPLETE:
            msg = f"Fragment {fragment_id} selected from attempt v{attempt.version} is still pending"
            raise PublicationError(msg)
        staged_fingerprint = staged_fingerprints.get(fragment_id)
        if staged_fingerprint is not None and _caption_values_fingerprint(fragment, bindings) != staged_fingerprint:
            msg = f"Fragment {fragment_id} is terminal but differs from its authoritative staged result"
            raise PublicationError(msg)
    logger.info(
        "Verified caption publication at Lance v{}: field={}, {} selected fragment(s), {} row(s) canonical",
        current.version,
        bindings.result.caption_field_name,
        len(attempt.selected_fragment_ids),
        sum(resolved_guards[fragment_id].row_count for fragment_id in attempt.selected_fragment_ids),
    )


def prepare_fragment_update(
    dataset: lance.LanceDataset,
    fragment: LanceFragment,
    staged: pa.Table,
    bindings: CaptionRunBindings,
) -> PreparedFragmentUpdate:
    """Write uncommitted caption column files and return their constrained descriptor."""
    source_metadata = fragment.metadata
    result = bindings.result
    expected_modified = leaf_field_ids(dataset, (result.caption_field_name, result.metadata_field_name))
    pending_fields = (
        *bindings.publication.protected_field_names,
        result.caption_field_name,
        result.metadata_field_name,
    )
    pending_ids = leaf_field_ids(dataset, pending_fields)
    update_schema = pa.schema(
        [
            bindings.input.mapped_schema.field(bindings.input.row_id_column),
            result.fields.field(result.caption_field_name),
            result.fields.field(result.metadata_field_name),
        ]
    )
    update_table = staged.select(
        [bindings.input.row_id_column, result.caption_field_name, result.metadata_field_name]
    ).cast(update_schema)
    reader = pa.RecordBatchReader.from_batches(update_schema, update_table.to_batches())
    updated_metadata, modified_field_ids = fragment.update_columns(
        reader,
        left_on=bindings.input.row_id_column,
        right_on=bindings.input.row_id_column,
    )
    descriptor = PreparedFragmentUpdate(
        fragment_id=int(fragment.fragment_id),
        source_version=int(dataset.version),
        source_metadata_json=fragment_metadata_json(source_metadata),
        source_fingerprint=fragment_fingerprint(source_metadata),
        source_pending_binding_fingerprint=protected_binding_fingerprint(source_metadata, pending_ids),
        updated_metadata_json=fragment_metadata_json(updated_metadata),
        modified_field_ids=tuple(sorted(int(field_id) for field_id in modified_field_ids)),
        row_count=int(fragment.count_rows()),
    )
    _validate_descriptor(descriptor, expected_modified)
    return descriptor


def commit_prepared_update(
    uri: str,
    descriptor: PreparedFragmentUpdate,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
) -> int:
    """Commit exactly one prepared descriptor without Lance attaching stale metadata to latest."""
    transaction = lance.Transaction(
        read_version=descriptor.source_version,
        operation=lance.LanceOperation.Update(
            updated_fragments=[FragmentMetadata.from_json(descriptor.updated_metadata_json)],
            fields_modified=list(descriptor.modified_field_ids),
        ),
        transaction_properties={
            **bindings.publication.transaction_properties,
            "source_version": str(descriptor.source_version),
            "fragment_id": str(descriptor.fragment_id),
            "field": bindings.result.caption_field_name,
            "contract_digest": bindings.result.digest,
        },
    )
    committed = lance.LanceDataset.commit(
        uri,
        transaction,
        storage_options=storage_options,
        max_retries=0,
    )
    return int(committed.version)


def _publish_one_fragment(  # noqa: C901, PLR0913 -- reconciliation state is deliberately explicit
    uri: str,
    attempt_dataset: lance.LanceDataset,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    prepared: PreparedFragmentPublication,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
    commit_attempts: int,
) -> bool:
    guard = prepared.guard
    staged_values_fingerprint = prepared.staged_values_fingerprint
    descriptor = prepared.descriptor
    last_commit_error: Exception | None = None
    for commit_attempt in range(1, commit_attempts + 1):
        current = lance.dataset(uri, storage_options=storage_options)
        _assert_nullable_schema_extension(attempt_dataset.schema, current.schema)
        fragment = _require_fragment(current, guard.fragment_id)
        state = classify_fragment(fragment, bindings=bindings)
        if state is FragmentCaptionState.COMPLETE:
            if _caption_values_fingerprint(fragment, bindings) == staged_values_fingerprint:
                logger.info(
                    "Caption fragment {} is already canonical at Lance v{}: {} row(s), no commit needed",
                    guard.fragment_id,
                    current.version,
                    guard.row_count,
                )
                return False
            msg = f"Fragment {guard.fragment_id} has a terminal caption that differs from staged recovery data"
            raise PublicationError(msg)

        _assert_pending_identity(current, fragment, guard, bindings)
        current_fingerprint = fragment_fingerprint(fragment.metadata)
        if descriptor is None or descriptor.source_fingerprint != current_fingerprint:
            logger.info(
                "Refreshing caption fragment {} publication descriptor against Lance v{} before commit attempt {}/{}",
                guard.fragment_id,
                current.version,
                commit_attempt,
                commit_attempts,
            )
            reprepared = _reprepare_fragment_with_ray(
                guard.fragment_id,
                uri=uri,
                attempt=attempt,
                workspace=workspace,
                bindings=bindings,
                storage_options=storage_options,
            )
            if reprepared.guard != guard or reprepared.staged_values_fingerprint != staged_values_fingerprint:
                msg = f"Restaged caption control data changed for fragment {guard.fragment_id}"
                raise PublicationError(msg)
            descriptor = reprepared.descriptor
            if descriptor is None:
                continue
        _validate_descriptor(
            descriptor,
            leaf_field_ids(
                current,
                (bindings.result.caption_field_name, bindings.result.metadata_field_name),
            ),
        )
        if not attempt.version <= descriptor.source_version <= int(current.version):
            msg = f"Prepared update source version is invalid for fragment {guard.fragment_id}"
            raise PublicationError(msg)
        if descriptor.source_pending_binding_fingerprint != guard.pending_binding_fingerprint:
            msg = f"Prepared update source for fragment {guard.fragment_id} is not the pending attempt fragment"
            raise PublicationError(msg)
        logger.info(
            "Committing caption fragment {}: {} row(s), prepared from Lance v{}, latest v{}, attempt {}/{}",
            guard.fragment_id,
            guard.row_count,
            descriptor.source_version,
            current.version,
            commit_attempt,
            commit_attempts,
        )
        try:
            committed_version = commit_prepared_update(
                uri,
                descriptor,
                bindings,
                storage_options=storage_options,
            )
        except (OSError, RuntimeError) as exc:
            # The response may be a deterministic conflict or an ambiguous IO error.
            # Reopening latest and applying the same state machine resolves both safely.
            last_commit_error = exc
            canonical_version = _latest_canonical_version(
                uri,
                guard,
                staged_values_fingerprint,
                bindings,
                storage_options=storage_options,
            )
            if canonical_version is not None:
                logger.info(
                    "Caption fragment {} is canonical at Lance v{} after {} during commit; "
                    "treating the response as already committed",
                    guard.fragment_id,
                    canonical_version,
                    type(exc).__name__,
                )
                return False
            descriptor = None
            continue

        canonical_version = _latest_canonical_version(
            uri,
            guard,
            staged_values_fingerprint,
            bindings,
            storage_options=storage_options,
        )
        if canonical_version is not None:
            logger.info(
                "Caption fragment {} published: {} row(s), commit Lance v{}, canonical at latest v{}",
                guard.fragment_id,
                guard.row_count,
                committed_version,
                canonical_version,
            )
            return True
        logger.info(
            "Caption fragment {} commit returned Lance v{} but latest is not canonical; repreparing",
            guard.fragment_id,
            committed_version,
        )
        descriptor = None

    msg = f"Could not reconcile caption commit for fragment {guard.fragment_id} after {commit_attempts} attempt(s)"
    raise PublicationError(msg) from last_commit_error


def _latest_canonical_version(
    uri: str,
    guard: AttemptFragmentGuard,
    staged_values_fingerprint: str,
    bindings: CaptionRunBindings,
    *,
    storage_options: dict[str, str] | None,
) -> int | None:
    """Return the latest version holding staged terminal values, rejecting a mismatch."""
    latest = lance.dataset(uri, storage_options=storage_options)
    latest_fragment = _require_fragment(latest, guard.fragment_id)
    if classify_fragment(latest_fragment, bindings=bindings) is not FragmentCaptionState.COMPLETE:
        return None
    if _caption_values_fingerprint(latest_fragment, bindings) == staged_values_fingerprint:
        return int(latest.version)
    msg = f"Fragment {guard.fragment_id} committed a terminal value different from staging"
    raise PublicationError(msg)


def _capture_fragment_guards(
    dataset: lance.LanceDataset,
    fragment_ids: tuple[int, ...],
    bindings: CaptionRunBindings,
) -> dict[int, AttemptFragmentGuard]:
    base_ids = leaf_field_ids(dataset, bindings.publication.protected_field_names)
    pending_ids = leaf_field_ids(
        dataset,
        (
            *bindings.publication.protected_field_names,
            bindings.result.caption_field_name,
            bindings.result.metadata_field_name,
        ),
    )
    guards: dict[int, AttemptFragmentGuard] = {}
    for fragment_id in fragment_ids:
        fragment = _require_fragment(dataset, fragment_id)
        guards[fragment_id] = _fragment_guard(fragment, bindings=bindings, base_ids=base_ids, pending_ids=pending_ids)
    return guards


def _fragment_guard(
    fragment: LanceFragment,
    *,
    bindings: CaptionRunBindings,
    base_ids: set[int],
    pending_ids: set[int],
) -> AttemptFragmentGuard:
    row_id_schema = pa.schema([bindings.input.mapped_schema.field(bindings.input.row_id_column)])
    row_ids = fragment.to_table(columns=[bindings.input.row_id_column]).cast(row_id_schema)
    return AttemptFragmentGuard(
        fragment_id=int(fragment.fragment_id),
        row_count=row_ids.num_rows,
        ordered_row_ids_fingerprint=_arrow_table_fingerprint(row_ids),
        base_binding_fingerprint=protected_binding_fingerprint(fragment.metadata, base_ids),
        pending_binding_fingerprint=protected_binding_fingerprint(fragment.metadata, pending_ids),
    )


def prepare_fragment_publication(
    staged: pa.Table,
    *,
    uri: str,
    attempt_version: int,
    storage_options: dict[str, str] | None,
    bindings: CaptionRunBindings,
) -> pa.Table:
    """Validate one Ray-grouped fragment and stage its uncommitted Lance update."""
    raw_fragment_ids = pc.unique(staged["fragment_id"]).to_pylist()
    if len(raw_fragment_ids) != 1:
        msg = "One distributed caption publication group must contain exactly one fragment ID"
        raise PublicationError(msg)
    fragment_id = int(raw_fragment_ids[0])
    attempt_dataset = lance.dataset(uri, version=attempt_version, storage_options=storage_options)
    guard = _capture_fragment_guards(attempt_dataset, (fragment_id,), bindings)[fragment_id]

    current = lance.dataset(uri, storage_options=storage_options)
    _assert_nullable_schema_extension(attempt_dataset.schema, current.schema)
    fragment = _require_fragment(current, fragment_id)
    table = _validate_staged_group(staged, fragment, guard, bindings)
    staged_values_fingerprint = _caption_table_fingerprint(table, bindings)

    descriptor: PreparedFragmentUpdate | None = None
    state = classify_fragment(fragment, bindings=bindings)
    if state is FragmentCaptionState.COMPLETE:
        _assert_base_identity(current, fragment, guard, bindings)
        if _caption_values_fingerprint(fragment, bindings) != staged_values_fingerprint:
            msg = f"Fragment {fragment_id} has a terminal caption that differs from staged recovery data"
            raise PublicationError(msg)
    else:
        _assert_pending_identity(current, fragment, guard, bindings)
        descriptor = prepare_fragment_update(current, fragment, table, bindings)

    prepared = PreparedFragmentPublication(
        guard=guard,
        staged_values_fingerprint=staged_values_fingerprint,
        descriptor=descriptor,
    )
    return pa.Table.from_pylist(
        [{"fragment_id": fragment_id, "payload": _serialize_prepared_publication(prepared)}],
        schema=PREPARED_PUBLICATION_SCHEMA,
    )


def _prepared_publication_dataset(  # noqa: PLR0913 -- Ray plan inputs are independently meaningful
    files: tuple[str, ...],
    *,
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    storage_options: dict[str, str] | None,
) -> ray.data.Dataset:
    staged = ray.data.read_parquet(
        list(files),
        filesystem=workspace.filesystem,
        schema=bindings.result.staged_schema,
        override_num_blocks=max(len(files), len(attempt.pending_fragment_ids)),
    )
    pending = staged.filter(expr=col("fragment_id").is_in(list(attempt.pending_fragment_ids)))
    return pending.groupby(
        "fragment_id",
        num_partitions=len(attempt.pending_fragment_ids),
    ).map_groups(
        cast("Any", prepare_fragment_publication),
        batch_format="pyarrow",
        zero_copy_batch=True,
        fn_kwargs={
            "uri": uri,
            "attempt_version": attempt.version,
            "storage_options": storage_options,
            "bindings": bindings,
        },
        num_cpus=1,
        memory=_PUBLICATION_PREPARE_MEMORY_BYTES,
    )


def _iter_prepared_publications(dataset: ray.data.Dataset) -> Iterator[PreparedFragmentPublication]:
    """Stream small worker control records without fetching staged caption rows."""
    try:
        batches = dataset.iter_batches(prefetch_batches=0, batch_size=None, batch_format="pyarrow")
        for raw_batch in batches:
            batch = raw_batch if isinstance(raw_batch, pa.Table) else pa.Table.from_batches([raw_batch])
            for row in batch.to_pylist():
                yield _parse_prepared_publication(str(row["payload"]), expected_fragment_id=int(row["fragment_id"]))
    except Exception as exc:
        publication_error = _find_publication_error(exc)
        if publication_error is not None:
            raise publication_error from exc
        raise


def _validate_staged_group(
    table: pa.Table,
    fragment: LanceFragment,
    guard: AttemptFragmentGuard,
    bindings: CaptionRunBindings,
) -> pa.Table:
    expected_schema = bindings.result.staged_schema
    try:
        table = table.cast(expected_schema)
    except (pa.ArrowException, ValueError) as exc:
        msg = f"Staged rows for fragment {guard.fragment_id} do not cast to the canonical result schema"
        raise PublicationError(msg) from exc
    if table.num_rows != guard.row_count:
        msg = (
            f"Staged fragment {guard.fragment_id} has {table.num_rows} row(s), expected exactly {guard.row_count}; "
            "missing or duplicate recovery rows are not publishable"
        )
        raise PublicationError(msg)
    indices = pc.sort_indices(table, sort_keys=[("row_offset", "ascending")])
    table = table.take(indices)
    offsets = [int(value) for value in table["row_offset"].to_pylist()]
    if offsets != list(range(guard.row_count)):
        msg = f"Staged fragment {guard.fragment_id} offsets are not exactly 0..{guard.row_count - 1}"
        raise PublicationError(msg)
    row_id = bindings.input.row_id_column
    row_id_schema = pa.schema([bindings.input.mapped_schema.field(row_id)])
    staged_row_ids = table.select([row_id]).cast(row_id_schema)
    current_row_ids = fragment.to_table(columns=[row_id]).cast(row_id_schema)
    if (
        _arrow_table_fingerprint(staged_row_ids) != guard.ordered_row_ids_fingerprint
        or _arrow_table_fingerprint(current_row_ids) != guard.ordered_row_ids_fingerprint
    ):
        msg = f"Staged fragment {guard.fragment_id} row-ID order does not match the attempt and current fragment"
        raise PublicationError(msg)
    current_bindings = _current_staged_bindings(fragment, bindings)
    staged_bindings = table.select(list(bindings.input.staged_binding_columns)).cast(current_bindings.schema)
    if not staged_bindings.equals(current_bindings):
        msg = f"Staged fragment {guard.fragment_id} input bindings do not match the current fragment"
        raise PublicationError(msg)
    fragment_ids = [int(value) for value in table["fragment_id"].to_pylist()]
    if any(fragment_id != guard.fragment_id for fragment_id in fragment_ids):
        msg = f"Staged group for fragment {guard.fragment_id} contains a different fragment ID"
        raise PublicationError(msg)
    for row in table.to_pylist():
        try:
            bindings.result.validate_staged_row(row)
        except (TypeError, ValueError) as exc:
            msg = f"Staged fragment {guard.fragment_id} contains a non-canonical terminal row: {exc}"
            raise PublicationError(msg) from exc
    return table


def _current_staged_bindings(fragment: LanceFragment, bindings: CaptionRunBindings) -> pa.Table:
    source = fragment.scanner(columns=list(bindings.input.source_columns), with_row_address=True).to_table()
    projected = bindings.input.project_batch(source)
    return projected.select(list(bindings.input.staged_binding_columns))


def _validate_parquet_schemas_distributed(
    files: tuple[str, ...],
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
) -> None:
    """Validate result footers across Ray workers instead of serially on the driver."""
    expected = bindings.result.staged_schema
    paths = ray.data.from_items(
        [{"path": path} for path in files],
        override_num_blocks=len(files),
    )
    validation = paths.map_batches(
        cast("Any", _validate_parquet_schema_batch),
        batch_format="pyarrow",
        batch_size=1,
        zero_copy_batch=True,
        fn_kwargs={"filesystem": workspace.filesystem, "expected": expected},
        num_cpus=0.1,
    )
    try:
        validation.materialize()
    except Exception as exc:
        publication_error = _find_publication_error(exc)
        if publication_error is not None:
            raise publication_error from exc
        raise


def _validate_parquet_schema_batch(
    batch: pa.Table,
    *,
    filesystem: pafs.FileSystem,
    expected: pa.Schema,
) -> pa.Table:
    for path in batch["path"].to_pylist():
        try:
            actual = pq.read_schema(str(path), filesystem=filesystem)
        except (OSError, pa.ArrowException) as exc:
            msg = f"Staged caption result is not a readable Parquet file: {path}"
            raise PublicationError(msg) from exc
        if not actual.equals(expected, check_metadata=True):
            msg = f"Staged caption result has an incompatible schema: {path}"
            raise PublicationError(msg)
    return batch.cast(_RESULT_PATH_SCHEMA)


def _reprepare_fragment_with_ray(  # noqa: PLR0913 -- retry task reproduces the complete worker contract
    fragment_id: int,
    *,
    uri: str,
    attempt: CaptionAttempt,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    storage_options: dict[str, str] | None,
) -> PreparedFragmentPublication:
    """Restage a descriptor on a worker after a same-fragment commit race."""
    task = ray.remote(num_cpus=1, memory=_PUBLICATION_PREPARE_MEMORY_BYTES)(
        cast("Any", _prepare_fragment_from_workspace)
    )
    try:
        payload = ray.get(
            task.remote(
                fragment_id,
                uri=uri,
                attempt_version=attempt.version,
                workspace=workspace,
                bindings=bindings,
                storage_options=storage_options,
            )
        )
    except Exception as exc:
        publication_error = _find_publication_error(exc)
        if publication_error is not None:
            raise publication_error from exc
        raise
    return _parse_prepared_publication(str(payload), expected_fragment_id=fragment_id)


def _prepare_fragment_from_workspace(  # noqa: PLR0913 -- remote retry inputs are independently meaningful
    fragment_id: int,
    *,
    uri: str,
    attempt_version: int,
    workspace: CaptionWorkspace,
    bindings: CaptionRunBindings,
    storage_options: dict[str, str] | None,
) -> str:
    files = staged_result_files(workspace)
    if not files:
        msg = f"No staged caption results exist while restaging fragment {fragment_id}"
        raise PublicationError(msg)
    staged_dataset = pads.dataset(list(files), format="parquet", filesystem=workspace.filesystem)
    staged = staged_dataset.to_table(filter=pc.field("fragment_id") == fragment_id)
    prepared = prepare_fragment_publication(
        staged,
        uri=uri,
        attempt_version=attempt_version,
        storage_options=storage_options,
        bindings=bindings,
    )
    return str(prepared["payload"][0].as_py())


def _serialize_prepared_publication(prepared: PreparedFragmentPublication) -> str:
    descriptor = prepared.descriptor
    payload = {
        "guard": {
            "fragment_id": prepared.guard.fragment_id,
            "row_count": prepared.guard.row_count,
            "ordered_row_ids_fingerprint": prepared.guard.ordered_row_ids_fingerprint,
            "base_binding_fingerprint": prepared.guard.base_binding_fingerprint,
            "pending_binding_fingerprint": prepared.guard.pending_binding_fingerprint,
        },
        "staged_values_fingerprint": prepared.staged_values_fingerprint,
        "descriptor": (
            None
            if descriptor is None
            else {
                "fragment_id": descriptor.fragment_id,
                "source_version": descriptor.source_version,
                "source_metadata_json": descriptor.source_metadata_json,
                "source_fingerprint": descriptor.source_fingerprint,
                "source_pending_binding_fingerprint": descriptor.source_pending_binding_fingerprint,
                "updated_metadata_json": descriptor.updated_metadata_json,
                "modified_field_ids": list(descriptor.modified_field_ids),
                "row_count": descriptor.row_count,
            }
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _parse_prepared_publication(payload: str, *, expected_fragment_id: int) -> PreparedFragmentPublication:
    try:
        raw: Any = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = "Ray returned an unreadable caption publication descriptor"
        raise PublicationError(msg) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("guard"), dict):
        msg = "Ray returned an invalid caption publication descriptor"
        raise PublicationError(msg)
    raw_guard = raw["guard"]
    guard = AttemptFragmentGuard(
        fragment_id=_payload_int(raw_guard, "fragment_id"),
        row_count=_payload_int(raw_guard, "row_count"),
        ordered_row_ids_fingerprint=_payload_string(raw_guard, "ordered_row_ids_fingerprint"),
        base_binding_fingerprint=_payload_string(raw_guard, "base_binding_fingerprint"),
        pending_binding_fingerprint=_payload_string(raw_guard, "pending_binding_fingerprint"),
    )
    if guard.fragment_id != expected_fragment_id:
        msg = f"Ray publication descriptor names fragment {guard.fragment_id}, expected {expected_fragment_id}"
        raise PublicationError(msg)

    raw_descriptor = raw.get("descriptor")
    descriptor: PreparedFragmentUpdate | None
    if raw_descriptor is None:
        descriptor = None
    elif isinstance(raw_descriptor, dict):
        raw_modified_field_ids = raw_descriptor.get("modified_field_ids")
        if not isinstance(raw_modified_field_ids, list):
            msg = f"Ray publication descriptor for fragment {guard.fragment_id} has invalid modified field IDs"
            raise PublicationError(msg)
        descriptor = PreparedFragmentUpdate(
            fragment_id=_payload_int(raw_descriptor, "fragment_id"),
            source_version=_payload_int(raw_descriptor, "source_version"),
            source_metadata_json=_payload_string(raw_descriptor, "source_metadata_json"),
            source_fingerprint=_payload_string(raw_descriptor, "source_fingerprint"),
            source_pending_binding_fingerprint=_payload_string(
                raw_descriptor,
                "source_pending_binding_fingerprint",
            ),
            updated_metadata_json=_payload_string(raw_descriptor, "updated_metadata_json"),
            modified_field_ids=tuple(
                _payload_list_int(value, label="modified_field_ids") for value in raw_modified_field_ids
            ),
            row_count=_payload_int(raw_descriptor, "row_count"),
        )
        if descriptor.fragment_id != guard.fragment_id or descriptor.row_count != guard.row_count:
            msg = f"Ray publication descriptor does not match its fragment guard for {guard.fragment_id}"
            raise PublicationError(msg)
    else:
        msg = f"Ray publication descriptor for fragment {guard.fragment_id} has an invalid update payload"
        raise PublicationError(msg)

    return PreparedFragmentPublication(
        guard=guard,
        staged_values_fingerprint=_payload_string(raw, "staged_values_fingerprint"),
        descriptor=descriptor,
    )


def _payload_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"Ray publication descriptor field {key!r} must be a nonnegative integer"
        raise PublicationError(msg)
    return value


def _payload_list_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"Ray publication descriptor field {label!r} must contain nonnegative integers"
        raise PublicationError(msg)
    return value


def _payload_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Ray publication descriptor field {key!r} must be a nonempty string"
        raise PublicationError(msg)
    return value


@contextmanager
def _raise_original_map_exceptions() -> Iterator[None]:
    """Preserve typed publication failures across Ray Data's UDF boundary."""
    context = ray.data.DataContext.get_current()
    previous = context.raise_original_map_exception
    context.raise_original_map_exception = True
    try:
        yield
    finally:
        context.raise_original_map_exception = previous


def _find_publication_error(error: BaseException) -> PublicationError | None:
    """Recover a typed user-code failure through Ray's exception wrappers."""
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, PublicationError):
            return current
        if isinstance(current, ray.exceptions.RayTaskError):
            cause = current.as_instanceof_cause()  # type: ignore[no-untyped-call]
            if cause is not current:
                pending.append(cause)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return None


def _caption_values_fingerprint(fragment: LanceFragment, bindings: CaptionRunBindings) -> str:
    result = bindings.result
    table = fragment.to_table(
        columns=[bindings.input.row_id_column, result.caption_field_name, result.metadata_field_name]
    )
    return _caption_table_fingerprint(table, bindings)


def _caption_table_fingerprint(table: pa.Table, bindings: CaptionRunBindings) -> str:
    result = bindings.result
    schema = pa.schema(
        [
            bindings.input.mapped_schema.field(bindings.input.row_id_column),
            result.fields.field(result.caption_field_name),
            result.fields.field(result.metadata_field_name),
        ]
    )
    normalized = table.select(schema.names).cast(schema)
    return _arrow_table_fingerprint(normalized)


def _arrow_table_fingerprint(table: pa.Table) -> str:
    """Hash a canonical Arrow stream in fixed row batches, independent of input chunking."""
    normalized = table.combine_chunks()
    fingerprint = sha256()
    fingerprint.update(normalized.schema.serialize().to_pybytes())
    for batch in normalized.to_batches(max_chunksize=_FINGERPRINT_BATCH_ROWS):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, normalized.schema) as writer:
            writer.write_batch(batch)
        fingerprint.update(sink.getvalue().to_pybytes())
    return fingerprint.hexdigest()


def _validate_descriptor(descriptor: PreparedFragmentUpdate, expected_modified: set[int]) -> None:
    source = FragmentMetadata.from_json(descriptor.source_metadata_json)
    updated = FragmentMetadata.from_json(descriptor.updated_metadata_json)
    source_json: Any = source.to_json()
    updated_json: Any = updated.to_json()
    if not isinstance(source_json, dict) or not isinstance(updated_json, dict):
        msg = "Lance returned an invalid fragment descriptor"
        raise PublicationError(msg)
    source_fragment_id = int(source_json.get("id", -1))
    updated_fragment_id = int(updated_json.get("id", -1))
    if source_fragment_id != descriptor.fragment_id or updated_fragment_id != descriptor.fragment_id:
        msg = f"Prepared update does not name target fragment {descriptor.fragment_id} exactly"
        raise PublicationError(msg)
    if fragment_fingerprint(source) != descriptor.source_fingerprint:
        msg = f"Prepared source fingerprint for fragment {descriptor.fragment_id} does not match its metadata"
        raise PublicationError(msg)
    if int(source_json.get("physical_rows", -1)) != descriptor.row_count:
        msg = f"Prepared source descriptor for fragment {descriptor.fragment_id} has an unexpected row count"
        raise PublicationError(msg)
    if int(updated_json.get("physical_rows", -1)) != descriptor.row_count:
        msg = f"Prepared update descriptor for fragment {descriptor.fragment_id} has an unexpected row count"
        raise PublicationError(msg)
    layout_keys = ("id", "physical_rows", "deletion_file", "overlays", "row_id_meta")
    if any(source_json.get(key) != updated_json.get(key) for key in layout_keys):
        msg = f"Prepared update descriptor for fragment {descriptor.fragment_id} changed physical row identity"
        raise PublicationError(msg)
    if len(descriptor.modified_field_ids) != len(set(descriptor.modified_field_ids)):
        msg = f"Prepared update descriptor for fragment {descriptor.fragment_id} repeats a modified field ID"
        raise PublicationError(msg)
    if set(descriptor.modified_field_ids) != expected_modified:
        msg = f"Prepared update descriptor for fragment {descriptor.fragment_id} modifies an unexpected field set"
        raise PublicationError(msg)


def _assert_pending_identity(
    dataset: lance.LanceDataset,
    fragment: LanceFragment,
    guard: AttemptFragmentGuard,
    bindings: CaptionRunBindings,
) -> None:
    _assert_base_identity(dataset, fragment, guard, bindings)
    pending_ids = leaf_field_ids(
        dataset,
        (
            *bindings.publication.protected_field_names,
            bindings.result.caption_field_name,
            bindings.result.metadata_field_name,
        ),
    )
    current = protected_binding_fingerprint(fragment.metadata, pending_ids)
    if current != guard.pending_binding_fingerprint:
        msg = f"Pending fragment {guard.fragment_id} changed a split- or caption-owned physical binding"
        raise PublicationError(msg)


def _assert_base_identity(
    dataset: lance.LanceDataset,
    fragment: LanceFragment,
    guard: AttemptFragmentGuard,
    bindings: CaptionRunBindings,
) -> None:
    row_id = bindings.input.row_id_column
    row_id_schema = pa.schema([bindings.input.mapped_schema.field(row_id)])
    row_ids = fragment.to_table(columns=[row_id]).cast(row_id_schema)
    if row_ids.num_rows != guard.row_count or _arrow_table_fingerprint(row_ids) != guard.ordered_row_ids_fingerprint:
        msg = f"Fragment {guard.fragment_id} changed row layout or canonical row identity after attempt capture"
        raise PublicationError(msg)
    base_ids = leaf_field_ids(dataset, bindings.publication.protected_field_names)
    current = protected_binding_fingerprint(fragment.metadata, base_ids)
    if current != guard.base_binding_fingerprint:
        msg = (
            f"Fragment {guard.fragment_id} changed a {bindings.publication.base_field_owner}-owned physical binding "
            "after attempt capture"
        )
        raise PublicationError(msg)


def _assert_nullable_schema_extension(source: pa.Schema, current: pa.Schema) -> None:
    if source.metadata != current.metadata or len(current) < len(source):
        msg = "The Lance schema changed incompatibly after caption attempt capture"
        raise PublicationError(msg)
    for index, source_field in enumerate(source):
        if not source_field.equals(current.field(index), check_metadata=True):
            msg = f"Lance field {source_field.name!r} changed incompatibly after caption attempt capture"
            raise PublicationError(msg)
    non_nullable = [field.name for field in list(current)[len(source) :] if not field.nullable]
    if non_nullable:
        msg = f"Concurrent schema extensions must be nullable: {', '.join(non_nullable)}"
        raise PublicationError(msg)


def _require_fragment(dataset: lance.LanceDataset, fragment_id: int) -> LanceFragment:
    fragment = dataset.get_fragment(fragment_id)
    if fragment is None:
        msg = f"Fragment {fragment_id} selected for captioning is absent at Lance version {dataset.version}"
        raise PublicationError(msg)
    return fragment
