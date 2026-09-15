# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical episode validation and flatten-only selected-camera projection."""

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

from cosmos_curator.next.recipes.multimodal_caption.contracts import validate_terminal_binding
from cosmos_curator.next.recipes.multimodal_split.identities import MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.multimodal_split.lance_sink import validate_episode_table
from cosmos_curator.next.recipes.multimodal_split.records import (
    EPISODE_SCHEMA,
    RECORD_SCHEMA_VERSION,
    resolve_camera_media,
)

_ROW_ADDRESS_FRAGMENT_SHIFT = 32
_ROW_ADDRESS_OFFSET_MASK = (1 << _ROW_ADDRESS_FRAGMENT_SHIFT) - 1

EPISODE_SOURCE_COLUMNS = ("episode_id", "materialization_digest", "camera_media")
EPISODE_SOURCE_SCHEMA = pa.schema([EPISODE_SCHEMA.field(name) for name in EPISODE_SOURCE_COLUMNS])
EPISODE_MAPPED_INPUT_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
    ]
)
EPISODE_STAGED_BINDING_COLUMNS = (
    "episode_id",
    "materialization_digest",
    "camera_key",
    "media_location",
    "media_size_bytes",
    "media_sha256",
)


def validate_episode_input(
    dataset: lance.LanceDataset,
    *,
    uri: str,
    camera_key: str,
) -> str | None:
    """Validate global episode identity and exact camera resolution on one pinned snapshot.

    Returns one representative selected S3 media URI when present for Ray's
    storage-compatibility environment.
    """
    validate_episode_table(dataset, uri=uri)
    query = (
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT episode_id) AS episode_ids, "
        "SUM(CASE WHEN episode_id IS NULL THEN 1 ELSE 0 END) AS null_episode_ids, "
        "MIN(record_schema_version) AS min_record_schema_version, "
        "MAX(record_schema_version) AS max_record_schema_version, "
        "MIN(media_contract_version) AS min_media_contract_version, "
        "MAX(media_contract_version) AS max_media_contract_version FROM dataset"
    )
    rows = [row for batch in dataset.sql(query).build().to_batch_records() for row in batch.to_pylist()]
    if len(rows) != 1:
        msg = f"Could not validate canonical episode identities in {uri}"
        raise RuntimeError(msg)
    row_count = int(rows[0]["rows"])
    distinct = int(rows[0]["episode_ids"])
    nulls = int(rows[0]["null_episode_ids"] or 0)
    if nulls:
        msg = f"Canonical episode_id values in {uri} must all be non-null"
        raise ValueError(msg)
    if distinct != row_count:
        msg = f"Canonical episode_id values in {uri} are not globally unique ({distinct} distinct for {row_count} rows)"
        raise ValueError(msg)
    version_extrema = (
        rows[0]["min_record_schema_version"],
        rows[0]["max_record_schema_version"],
        rows[0]["min_media_contract_version"],
        rows[0]["max_media_contract_version"],
    )
    expected_versions = (
        RECORD_SCHEMA_VERSION,
        RECORD_SCHEMA_VERSION,
        MEDIA_CONTRACT_VERSION,
        MEDIA_CONTRACT_VERSION,
    )
    if row_count and version_extrema != expected_versions:
        msg = (
            f"Canonical episode contract versions in {uri} must be "
            f"record_schema_version={RECORD_SCHEMA_VERSION} and media_contract_version={MEDIA_CONTRACT_VERSION}"
        )
        raise ValueError(msg)

    representative_s3_uri: str | None = None
    scanner = dataset.scanner(columns=["episode_id", "camera_media"])
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            selected = resolve_camera_media(row, camera_key)
            location = str(selected["media_location"])
            if representative_s3_uri is None and location.lower().startswith("s3://"):
                representative_s3_uri = location
    return representative_s3_uri


def project_episode_camera(batch: pa.Table, *, camera_key: str) -> pa.Table:
    """Flatten one selected camera descriptor while preserving physical row identity."""
    required = {"_rowaddr", *EPISODE_SOURCE_COLUMNS}
    missing = sorted(required - set(batch.schema.names))
    if missing:
        msg = f"Lance episode caption read is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for row in batch.select(["_rowaddr", *EPISODE_SOURCE_COLUMNS]).to_pylist():
        selected = resolve_camera_media(row, camera_key)
        row_address = int(row["_rowaddr"])
        rows.append(
            {
                "fragment_id": row_address >> _ROW_ADDRESS_FRAGMENT_SHIFT,
                "row_offset": row_address & _ROW_ADDRESS_OFFSET_MASK,
                "episode_id": row["episode_id"],
                "media_location": selected["media_location"],
                "media_size_bytes": selected["media_size_bytes"],
                "media_sha256": selected["media_sha256"],
                "materialization_digest": row["materialization_digest"],
                "camera_key": selected["camera_key"],
            }
        )
    return pa.Table.from_pylist(rows, schema=EPISODE_MAPPED_INPUT_SCHEMA)


def validate_complete_fragment(
    fragment: LanceFragment,
    *,
    camera_key: str,
    metadata_field_name: str,
) -> None:
    """Validate terminal metadata against current selected-camera bindings."""
    columns = [*EPISODE_SOURCE_COLUMNS, metadata_field_name]
    for raw_batch in fragment.scanner(columns=columns).to_batches():
        for row in raw_batch.to_pylist():
            metadata = row[metadata_field_name]
            if not isinstance(metadata, dict):
                msg = "terminal caption metadata must be a mapping"
                raise TypeError(msg)
            selected = resolve_camera_media(row, camera_key)
            validate_terminal_binding(
                metadata,
                episode_id=row["episode_id"],
                materialization_digest=row["materialization_digest"],
                camera_key=selected["camera_key"],
                media_sha256=selected["media_sha256"],
            )
