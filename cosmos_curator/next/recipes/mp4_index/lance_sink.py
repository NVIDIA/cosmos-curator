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

"""Schema bootstrap, distributed fragment staging, and incremental appends.

Thin ``mp4-index``-schema binding over the generic canonical-table primitives in
``cosmos_curator.next.utils.lance_fragment_recovery`` -- the same binding shape as
``robot_action_split.lance_sink``.
"""

from collections.abc import Iterable
from typing import Any

import lance  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from loguru import logger  # type: ignore[import-untyped]

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.recipes.mp4_index.contracts import ERROR_SCHEMA, RECORD_SCHEMA_VERSION, VIDEO_MANIFEST_SCHEMA
from cosmos_curator.next.utils import lance_fragment_recovery
from cosmos_curator.next.utils.lance_utils import (
    LANCE_DATA_STORAGE_VERSION,
    open_dataset,
    open_dataset_or_raise,
)

_KIND = "mp4-index"
_ERRORS_KIND = "mp4-index-errors"
_APPEND_OPERATION = "append-video"
_COMMIT_ATTEMPTS = 5

#: BTREE fits all three -- exact-match lookups only, session_camera_id not unique. Rebuilt
#: (replace=True, Lance's default) after every run, via _ensure_scalar_indices below.
_IDENTITY_INDEXED_COLUMNS = ("video_id", "media_location", "session_camera_id")

#: video_id (join back to the manifest) and error_type.
_ERROR_INDEXED_COLUMNS = ("video_id", "error_type")

#: The tiny control record stage_manifest_batch returns per block -- either "fragment" (one
#: staged candidate covering the whole block's successes) or "failure" (one per failed video,
#: carried through unchanged).
_PUBLISH_RESULT_SCHEMA = pa.schema(
    [
        pa.field("result_type", pa.string(), nullable=False),
        pa.field("candidate_json", pa.large_string(), nullable=True),
        pa.field("extracted_count", pa.int64(), nullable=True),
        pa.field("video_id", pa.large_string(), nullable=True),
        pa.field("media_location", pa.large_string(), nullable=True),
        pa.field("error_type", pa.string(), nullable=True),
        pa.field("error_message", pa.large_string(), nullable=True),
    ]
)


def open_or_create_manifest_table(*, uri: str, storage_profile: str) -> lance.LanceDataset:
    """Open the canonical manifest table or atomically bootstrap its zero-row schema."""
    return lance_fragment_recovery.open_or_create_table(
        uri=uri,
        storage_profile=storage_profile,
        schema=VIDEO_MANIFEST_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind=_KIND,
    )


def validate_committed_versions(dataset: lance.LanceDataset, *, extraction_contract_id: str) -> None:
    """Refuse a table containing any row from an incompatible schema/extraction-contract version.

    Scans the whole table, not just this run's own candidate ``video_id``s: unlike
    ``robot_action_split``'s ``clip_id``, ``video_id`` doesn't bake the contract into the identity,
    so a stale row outside this run's own discovered set would otherwise go unchecked forever.
    """
    mismatched_schema = dataset.count_rows(filter=f"record_schema_version != {RECORD_SCHEMA_VERSION}")
    if mismatched_schema:
        msg = (
            f"Manifest table contains {mismatched_schema} row(s) with record_schema_version other than "
            f"{RECORD_SCHEMA_VERSION}. Recreate the table and rerun, or backfill the stale rows, before "
            "continuing."
        )
        raise ValueError(msg)
    mismatched_contract = dataset.count_rows(filter=f"extraction_contract_id != '{extraction_contract_id}'")
    if mismatched_contract:
        msg = (
            f"Manifest table contains {mismatched_contract} row(s) with extraction_contract_id other than "
            f"{extraction_contract_id!r} (the current run's own container_kind/session_id_path_depth/"
            "extraction_version). Recreate the table and rerun, or backfill the stale rows, before "
            "continuing."
        )
        raise ValueError(msg)


def _in_filter(column: str, values: Iterable[str]) -> str | None:
    """Return a ``column IN (...)`` predicate for *values*, or None if *values* is empty."""
    literals = [lance_fragment_recovery.sql_string_literal(value) for value in sorted(set(values))]
    return f"{column} IN ({','.join(literals)})" if literals else None


def committed_video_ids(dataset: lance.LanceDataset, candidate_video_ids: Iterable[str]) -> set[str]:
    """Return which of *candidate_video_ids* already have a committed row, one bounded BTREE lookup."""
    row_filter = _in_filter("video_id", candidate_video_ids)
    if row_filter is None:
        return set()
    table = dataset.to_table(columns=["video_id"], filter=row_filter)
    return {str(value) for value in table.column("video_id").to_pylist()}


def committed_session_camera_owners(
    dataset: lance.LanceDataset, candidate_session_camera_ids: Iterable[str]
) -> dict[str, tuple[str, str]]:
    """Return ``{session_camera_id: (video_id, media_location)}`` for committed rows matching the candidates.

    A collision can only ever involve one of the candidates, so no other committed row matters here.
    """
    row_filter = _in_filter("session_camera_id", candidate_session_camera_ids)
    if row_filter is None:
        return {}
    table = dataset.to_table(columns=["session_camera_id", "video_id", "media_location"], filter=row_filter)
    return {
        session_camera_id: (video_id, media_location)
        for session_camera_id, video_id, media_location in zip(
            table.column("session_camera_id").to_pylist(),
            table.column("video_id").to_pylist(),
            table.column("media_location").to_pylist(),
            strict=True,
        )
    }


#: error_type values that are not a deterministic property of the video itself -- e.g. a
#: misconfigured or not-yet-created bucket, which S3Client.stat's own docstring notes a HEAD's
#: bodyless 404 cannot tell apart from a genuinely missing key. Excluded from
#: known_failed_video_ids: ERROR_SCHEMA's own docstring already says a row records one failed
#: attempt, not the video's current status, and skip_known_failures must not promote one of
#: these into that claim.
_RETRYABLE_ERROR_TYPES = ("FileNotFoundError",)


def known_failed_video_ids(errors_lance_uri: str, storage_profile: str, candidate_video_ids: Iterable[str]) -> set[str]:
    """Return which of *candidate_video_ids* have at least one non-retryable failure in errors_lance_uri.

    A video whose only recorded failures are all in _RETRYABLE_ERROR_TYPES is never
    considered "known failed" -- it's always retried, not reconciled away.
    """
    row_filter = _in_filter("video_id", candidate_video_ids)
    if row_filter is None:
        return set()
    storage_options = get_lance_storage_options(errors_lance_uri, profile_name=storage_profile)
    dataset = open_dataset(errors_lance_uri, storage_options=storage_options)
    if dataset is None:
        return set()
    retryable = ",".join(lance_fragment_recovery.sql_string_literal(t) for t in _RETRYABLE_ERROR_TYPES)
    table = dataset.to_table(columns=["video_id"], filter=f"({row_filter}) AND error_type NOT IN ({retryable})")
    return {str(value) for value in table.column("video_id").to_pylist()}


def stage_manifest_batch(batch: pa.Table, *, uri: str, storage_profile: str) -> pa.Table:
    """Stage one bounded block's worth of successful rows as a single Lance fragment."""
    is_success = pc.equal(batch["status"], "success")
    successes = batch.filter(is_success)
    failures = batch.filter(pc.invert(is_success))

    rows: list[dict[str, Any]] = []
    if successes.num_rows:
        manifest_table = successes.select(VIDEO_MANIFEST_SCHEMA.names).cast(VIDEO_MANIFEST_SCHEMA)
        candidate = lance_fragment_recovery.write_row_fragment(
            manifest_table,
            uri=uri,
            storage_profile=storage_profile,
            schema=VIDEO_MANIFEST_SCHEMA,
            id_column="video_id",
            data_storage_version=LANCE_DATA_STORAGE_VERSION,
        )
        if candidate is not None:
            rows.append(
                {
                    "result_type": "fragment",
                    "candidate_json": candidate,
                    "extracted_count": manifest_table.num_rows,
                    "video_id": None,
                    "media_location": None,
                    "error_type": None,
                    "error_message": None,
                }
            )
    rows.extend(
        {
            "result_type": "failure",
            "candidate_json": None,
            "extracted_count": None,
            "video_id": failure["video_id"],
            "media_location": failure["media_location"],
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
        }
        for failure in failures.to_pylist()
    )
    return pa.Table.from_pylist(rows, schema=_PUBLISH_RESULT_SCHEMA)


def append_manifest_fragment(candidate_json: str, *, uri: str, storage_profile: str) -> int:
    """Idempotently append one staged manifest fragment and return the version containing it."""
    return lance_fragment_recovery.append_row_fragment(
        candidate_json,
        uri=uri,
        storage_profile=storage_profile,
        schema=VIDEO_MANIFEST_SCHEMA,
        id_column="video_id",
        kind=_KIND,
        operation=_APPEND_OPERATION,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        attempts=_COMMIT_ATTEMPTS,
    )


def _ensure_scalar_indices(uri: str, storage_profile: str, columns: tuple[str, ...]) -> list[str]:
    """(Re)build a BTREE scalar index on every column in *columns*."""
    storage_options = get_lance_storage_options(uri, profile_name=storage_profile)
    dataset = open_dataset_or_raise(uri, storage_options=storage_options)
    for column in columns:
        dataset.create_scalar_index(column, index_type="BTREE")
    return list(columns)


def rebuild_manifest_indices(uri: str, storage_profile: str) -> list[str]:
    """(Re)build the manifest's own video_id/media_location scalar indices."""
    return _ensure_scalar_indices(uri, storage_profile, _IDENTITY_INDEXED_COLUMNS)


def append_error_log(failures: list[dict[str, Any]], *, errors_lance_uri: str, storage_profile: str) -> list[str]:
    """Bootstrap the error log, append this run's failures if any, and (re)build its indices.

    Plain sequential append, not ``lance_fragment_recovery``'s staging machinery -- the caller
    already collected every failure itself, and an error schema has no uniqueness constraint to
    protect.
    """
    storage_options = get_lance_storage_options(errors_lance_uri, profile_name=storage_profile)
    lance_fragment_recovery.open_or_create_table(
        uri=errors_lance_uri,
        storage_profile=storage_profile,
        schema=ERROR_SCHEMA,
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        kind=_ERRORS_KIND,
    )
    if not failures:
        return []
    table = pa.Table.from_pylist(failures, schema=ERROR_SCHEMA)
    lance.write_dataset(
        table,
        errors_lance_uri,
        mode="append",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
        storage_options=storage_options,
        transaction_properties={"kind": _ERRORS_KIND, "operation": "append-error"},
    )
    logger.info("{}: appended {} failure(s) to {}", _KIND, len(failures), errors_lance_uri)

    error_indices_built = _ensure_scalar_indices(errors_lance_uri, storage_profile, _ERROR_INDEXED_COLUMNS)
    logger.info("{}: rebuilt scalar index on {} (errors)", _KIND, error_indices_built)
    return error_indices_built
