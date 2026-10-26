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

"""Generic canonical-table bootstrap, fragment staging, and idempotent appends.

Deprecated: new append producers use shared append publication in
``cosmos_curator.next.append``, which video-cut adopts. The producers still using
this module migrate explicitly; see ``docs/curator/design/curator-next-checkpoint-recovery.md``.

Shared by any Curator Next recipe that publishes one row per deterministically
identified unit of work into an append-only Lance table and needs that table to
double as a cross-run recovery checkpoint. Originated in ``video_split``'s
incremental publication path; extracted once ``robot_action_split`` needed the
identical contract (schema bootstrap with create-only semantics, producer-owned
vs. nullable-extension schema validation, worker-stage/driver-commit fragment
split, and all/none/partial candidate-ID presence resolution for idempotent
retries and ambiguous commit responses).

None of this module inspects a recipe-specific column beyond the caller-supplied
``id_column`` used to key presence checks. Recipe ``lance_sink.py`` modules own
their schema, identity semantics, and transaction-property labels; this module
owns only the bootstrap/stage/commit mechanics.
"""

import json
import time
from typing import Any, Literal

import lance
import pyarrow as pa
from lance.fragment import write_fragments
from loguru import logger

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.utils.lance_utils import open_dataset

# Matches ``do_with_retries``'s defaults so a commit retry backs off the same way
# every other retry path in the codebase does.
_COMMIT_RETRY_BACKOFF_FACTOR = 2
_COMMIT_RETRY_MAX_WAIT_S = 16.0

# Mirrors ``lance.write_dataset``'s ``data_storage_version`` literal so callers'
# recipe-owned version constants (e.g. ``Literal["2.2"]``) type-check here too.
LanceDataStorageVersion = Literal["stable", "2.0", "2.1", "2.2", "2.3", "next", "legacy", "0.1"]


def open_or_create_table(
    *,
    uri: str,
    storage_profile: str,
    schema: pa.Schema,
    data_storage_version: LanceDataStorageVersion,
    kind: str,
) -> lance.LanceDataset:
    """Open the canonical table or atomically bootstrap its zero-row schema."""
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        empty = pa.Table.from_batches([], schema=schema)
        try:
            dataset = lance.write_dataset(
                empty,
                uri,
                schema=schema,
                mode="create",
                data_storage_version=data_storage_version,
                storage_options=storage_options,
                enable_v2_manifest_paths=True,
                transaction_properties={"kind": kind, "operation": "bootstrap"},
            )
        except OSError:  # A storage failure may be an ambiguous successful create.
            dataset = open_dataset(uri, storage_options=storage_options)
            if dataset is None:
                raise

    validate_table(dataset, uri=uri, schema=schema, data_storage_version=data_storage_version)
    return dataset


def validate_table(
    dataset: lance.LanceDataset,
    *,
    uri: str,
    schema: pa.Schema,
    data_storage_version: LanceDataStorageVersion,
) -> None:
    """Require the producer-owned schema while allowing nullable extension fields."""
    actual_data_storage_version = getattr(dataset, "data_storage_version", None)
    if actual_data_storage_version != data_storage_version:
        msg = (
            f"Existing Lance table {uri} has data_storage_version={actual_data_storage_version!r}; "
            f"expected {data_storage_version!r}"
        )
        raise ValueError(msg)

    expected_by_name = {field.name: field for field in schema}
    actual_by_name = {field.name: field for field in dataset.schema}
    missing = [name for name in expected_by_name if name not in actual_by_name]
    if missing:
        msg = f"Existing Lance table {uri} is missing producer-owned field(s): {', '.join(missing)}"
        raise ValueError(msg)

    # write_row_fragment stages a new fragment using the caller-supplied schema's
    # own field order, not the dataset's on-disk order — Lance's write_fragments
    # binds columns positionally against that argument, not by name against the
    # already-open table. A producer schema that reorders fields relative to how
    # the table was originally bootstrapped would pass every other check here
    # (same names, types, nullability) while silently writing each value under
    # the wrong column name. Comparing order is what catches that before it ever
    # reaches a commit.
    expected_order = list(expected_by_name)
    actual_order = [name for name in actual_by_name if name in expected_by_name]
    if expected_order != actual_order:
        msg = (
            f"Existing Lance table {uri} has producer-owned fields in a different order than the caller's schema: "
            f"table order is {actual_order}, expected {expected_order}"
        )
        raise ValueError(msg)

    mismatched = [
        name
        for name, expected in expected_by_name.items()
        if not actual_by_name[name].equals(expected, check_metadata=True)
    ]
    if mismatched:
        msg = f"Existing Lance table {uri} has incompatible producer-owned field(s): {', '.join(mismatched)}"
        raise ValueError(msg)

    non_nullable_extensions = [
        field.name for field in dataset.schema if field.name not in expected_by_name and not field.nullable
    ]
    if non_nullable_extensions:
        msg = f"Existing Lance table {uri} has non-nullable extension field(s): {', '.join(non_nullable_extensions)}"
        raise ValueError(msg)


def write_row_fragment(  # noqa: PLR0913 -- generic over schema/id/storage; each parameter is independent
    rows: pa.Table,
    *,
    uri: str,
    storage_profile: str,
    schema: pa.Schema,
    id_column: str,
    data_storage_version: LanceDataStorageVersion,
) -> str | None:
    """Stage one canonical row batch and serialize its recovery candidate."""
    if not rows.schema.equals(schema):
        msg = "Row fragment input does not match the canonical schema"
        raise ValueError(msg)
    if rows.num_rows == 0:
        return None

    row_ids = [str(value) for value in rows[id_column].to_pylist()]
    if len(set(row_ids)) != len(row_ids):
        msg = f"Row fragment contains duplicate {id_column} values"
        raise ValueError(msg)

    # ``overwrite`` applies only to uncommitted worker-local fragment ID allocation.
    # The coordinator publishes this descriptor exclusively with ``Append``.
    fragments = write_fragments(
        rows,
        uri,
        schema=schema,
        mode="overwrite",
        max_rows_per_file=rows.num_rows,
        data_storage_version=data_storage_version,
        storage_options=get_lance_storage_options(uri, profile_name=storage_profile),
    )
    if len(fragments) != 1:
        msg = f"One publication batch must stage exactly one Lance fragment, got {len(fragments)}"
        raise RuntimeError(msg)
    fragment = fragments[0]
    if fragment.num_rows != len(row_ids):
        msg = f"Staged fragment contains {fragment.num_rows} rows for {len(row_ids)} candidate {id_column} values"
        raise RuntimeError(msg)
    data_files = tuple(str(data_file.path) for data_file in fragment.files)
    logger.info(
        "Staged Lance fragment with {} row(s) for {}: data_files={}",
        len(row_ids),
        uri,
        data_files,
    )

    return json.dumps(
        {"fragment": fragment.to_json(), "row_ids": row_ids},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def append_row_fragment(  # noqa: PLR0913 -- generic over schema/id/storage/transaction labels; all independent
    candidate_json: str,
    *,
    uri: str,
    storage_profile: str,
    schema: pa.Schema,
    id_column: str,
    kind: str,
    operation: str,
    data_storage_version: LanceDataStorageVersion,
    attempts: int,
) -> int:
    """Idempotently append one staged fragment and return a version containing it."""
    if attempts < 1:
        msg = f"Append attempts must be positive, got {attempts}"
        raise ValueError(msg)
    fragment, row_ids = _parse_candidate(candidate_json)
    data_files = tuple(str(data_file.path) for data_file in fragment.files)
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    transaction_properties = {"kind": kind, "operation": operation}

    for attempt in range(1, attempts + 1):
        dataset, present = _candidate_presence(
            row_ids,
            uri=uri,
            storage_options=storage_options,
            schema=schema,
            id_column=id_column,
            data_storage_version=data_storage_version,
        )
        if len(present) == len(row_ids):
            logger.info(
                "Staged Lance fragment with {} row(s) is already canonical in {} at version {}; skipping append: "
                "data_files={}",
                len(row_ids),
                uri,
                dataset.version,
                data_files,
            )
            return int(dataset.version)
        if present:
            msg = (
                f"Atomic append invariant violated for {uri}: {len(present)} of "
                f"{len(row_ids)} candidate {id_column} values are canonical"
            )
            raise RuntimeError(msg)

        transaction = lance.Transaction(
            read_version=dataset.version,
            operation=lance.LanceOperation.Append([fragment]),
            transaction_properties=transaction_properties,
        )
        logger.info(
            "Committing staged Lance fragment with {} row(s) to {} from version {} (attempt {}/{}): data_files={}",
            len(row_ids),
            uri,
            dataset.version,
            attempt,
            attempts,
            data_files,
        )
        try:
            committed = lance.LanceDataset.commit(
                uri,
                transaction,
                storage_options=storage_options,
                enable_v2_manifest_paths=True,
            )
        except Exception as exc:  # All failed commit responses are potentially ambiguous.
            latest, present = _candidate_presence(
                row_ids,
                uri=uri,
                storage_options=storage_options,
                schema=schema,
                id_column=id_column,
                data_storage_version=data_storage_version,
            )
            if len(present) == len(row_ids):
                logger.info(
                    "Confirmed staged Lance fragment with {} row(s) in {} at version {} after an ambiguous "
                    "commit response ({}: {!s}): data_files={}",
                    len(row_ids),
                    uri,
                    latest.version,
                    type(exc).__name__,
                    exc,
                    data_files,
                )
                return int(latest.version)
            if present:
                msg = (
                    f"Atomic append invariant violated for {uri}: {len(present)} of "
                    f"{len(row_ids)} candidate {id_column} values are canonical"
                )
                raise RuntimeError(msg) from exc
            if attempt == attempts:
                raise
            sleep_time = min(_COMMIT_RETRY_BACKOFF_FACTOR**attempt, _COMMIT_RETRY_MAX_WAIT_S)
            logger.warning(
                "Commit attempt {}/{} for {} failed with {}: {!s}. Retrying in {}s...",
                attempt,
                attempts,
                uri,
                type(exc).__name__,
                exc,
                sleep_time,
            )
            time.sleep(sleep_time)
            continue
        logger.info(
            "Committed staged Lance fragment with {} row(s) to {} at version {}: data_files={}",
            len(row_ids),
            uri,
            committed.version,
            data_files,
        )
        return int(committed.version)

    msg = "Append attempt loop exited unexpectedly"
    raise AssertionError(msg)


def _parse_candidate(candidate_json: str) -> tuple[lance.FragmentMetadata, tuple[str, ...]]:
    candidate: Any = json.loads(candidate_json)
    if not isinstance(candidate, dict) or not isinstance(candidate.get("fragment"), dict):
        msg = "Invalid staged fragment candidate payload"
        raise TypeError(msg)
    raw_row_ids = candidate.get("row_ids")
    if not isinstance(raw_row_ids, list) or not raw_row_ids or not all(isinstance(value, str) for value in raw_row_ids):
        msg = "Staged fragment candidate must contain nonempty string row IDs"
        raise ValueError(msg)
    row_ids = tuple(raw_row_ids)
    if len(set(row_ids)) != len(row_ids):
        msg = "Staged fragment candidate contains duplicate row IDs"
        raise ValueError(msg)
    fragment = lance.FragmentMetadata.from_json(json.dumps(candidate["fragment"]))
    if fragment.num_rows != len(row_ids):
        msg = f"Staged fragment has {fragment.num_rows} rows for {len(row_ids)} candidate row IDs"
        raise ValueError(msg)
    return fragment, row_ids


def _candidate_presence(  # noqa: PLR0913 -- generic over schema/id/storage; each parameter is independent
    row_ids: tuple[str, ...],
    *,
    uri: str,
    storage_options: dict[str, str] | None,
    schema: pa.Schema,
    id_column: str,
    data_storage_version: LanceDataStorageVersion,
) -> tuple[lance.LanceDataset, set[str]]:
    dataset = open_dataset(uri, storage_options=storage_options)
    if dataset is None:
        msg = f"Canonical Lance table disappeared while resolving an append: {uri}"
        raise RuntimeError(msg)
    validate_table(dataset, uri=uri, schema=schema, data_storage_version=data_storage_version)
    # Use Lance's SQL filter parser rather than a PyArrow Expression. Lance's
    # Substrait bridge cannot currently lower a string Expression when the
    # table also contains nullable struct enrichment fields (for example the
    # video-caption metadata field), even though only id_column is projected.
    literals = ",".join(sql_string_literal(row_id) for row_id in row_ids)
    candidate_filter = f"{id_column} IN ({literals})"
    committed = dataset.to_table(columns=[id_column], filter=candidate_filter)
    committed_ids = [str(value) for value in committed[id_column].to_pylist()]
    if len(set(committed_ids)) != len(committed_ids):
        msg = f"Canonical table contains duplicate candidate {id_column} values"
        raise RuntimeError(msg)
    return dataset, set(committed_ids)


def sql_string_literal(value: str) -> str:
    """Quote one candidate ID for Lance's SQL predicate parser.

    Public so any recipe-owned scan filtering by a string ID column (e.g. a
    reconciliation scan run outside this module's own commit path) can use the
    same SQL-string filter form rather than a PyArrow Expression, which Lance's
    Substrait bridge cannot currently lower once the table also contains
    nullable struct enrichment fields.
    """
    return "'" + value.replace("'", "''") + "'"
