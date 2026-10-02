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

"""Canonical episode validation and selected-camera projection for video vectors."""

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

from cosmos_curator.next.recipes.multimodal_split.identities import MEDIA_CONTRACT_VERSION
from cosmos_curator.next.recipes.multimodal_split.lance_sink import validate_episode_table
from cosmos_curator.next.recipes.multimodal_split.records import (
    EPISODE_SCHEMA,
    RECORD_SCHEMA_VERSION,
    resolve_camera_media,
)

_ROW_ADDRESS_FRAGMENT_SHIFT = 32
_ROW_ADDRESS_OFFSET_MASK = (1 << _ROW_ADDRESS_FRAGMENT_SHIFT) - 1

EPISODE_SOURCE_COLUMNS = (
    "episode_id",
    "materialization_digest",
    "start_ns",
    "exclusive_end_ns",
    "camera_media",
)
EPISODE_SOURCE_SCHEMA = pa.schema([EPISODE_SCHEMA.field(name) for name in EPISODE_SOURCE_COLUMNS])
EPISODE_MAPPED_INPUT_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
    ]
)
EPISODE_STAGED_BINDING_COLUMNS = (
    "episode_id",
    "materialization_digest",
    "camera_key",
    "media_location",
    "media_size_bytes",
    "media_sha256",
    "start_ns",
    "exclusive_end_ns",
)


def _validate_episode_identity(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Check global episode identity and Split contract versions once per snapshot."""
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
    expected_versions = (
        RECORD_SCHEMA_VERSION,
        RECORD_SCHEMA_VERSION,
        MEDIA_CONTRACT_VERSION,
        MEDIA_CONTRACT_VERSION,
    )
    observed_versions = (
        rows[0]["min_record_schema_version"],
        rows[0]["max_record_schema_version"],
        rows[0]["min_media_contract_version"],
        rows[0]["max_media_contract_version"],
    )
    if row_count and observed_versions != expected_versions:
        msg = (
            f"Canonical episode contract versions in {uri} must be "
            f"record_schema_version={RECORD_SCHEMA_VERSION} and media_contract_version={MEDIA_CONTRACT_VERSION}"
        )
        raise ValueError(msg)


def validate_episode_input(dataset: lance.LanceDataset, *, uri: str, camera_key: str) -> str | None:
    """Validate the canonical table and every selected camera before registration."""
    validate_episode_table(dataset, uri=uri)
    _validate_episode_identity(dataset, uri=uri)
    representative_s3_uri: str | None = None
    scanner = dataset.scanner(columns=list(EPISODE_SOURCE_COLUMNS))
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            start_ns = row["start_ns"]
            end_ns = row["exclusive_end_ns"]
            if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
                msg = f"Episode {row['episode_id']!r} must have positive start_ns/exclusive_end_ns duration"
                raise ValueError(msg)
            selected = resolve_camera_media(row, camera_key)
            location = str(selected["media_location"])
            if representative_s3_uri is None and location.lower().startswith("s3://"):
                representative_s3_uri = location
    return representative_s3_uri


def project_episode_camera(batch: pa.Table, *, camera_key: str) -> pa.Table:
    """Flatten one selected camera descriptor and preserve the physical row address."""
    required = {"_rowaddr", *EPISODE_SOURCE_COLUMNS}
    missing = sorted(required - set(batch.schema.names))
    if missing:
        msg = f"Lance episode embedding read is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for row in batch.select(["_rowaddr", *EPISODE_SOURCE_COLUMNS]).to_pylist():
        start_ns = row["start_ns"]
        end_ns = row["exclusive_end_ns"]
        if not isinstance(start_ns, int) or not isinstance(end_ns, int) or end_ns <= start_ns:
            msg = f"Episode {row['episode_id']!r} must have positive start_ns/exclusive_end_ns duration"
            raise ValueError(msg)
        selected = resolve_camera_media(row, camera_key)
        row_address = int(row["_rowaddr"])
        rows.append(
            {
                "fragment_id": row_address >> _ROW_ADDRESS_FRAGMENT_SHIFT,
                "row_offset": row_address & _ROW_ADDRESS_OFFSET_MASK,
                "episode_id": row["episode_id"],
                "materialization_digest": row["materialization_digest"],
                "start_ns": start_ns,
                "exclusive_end_ns": end_ns,
                "camera_key": selected["camera_key"],
                "media_location": selected["media_location"],
                "media_size_bytes": selected["media_size_bytes"],
                "media_sha256": selected["media_sha256"],
            }
        )
    return pa.Table.from_pylist(rows, schema=EPISODE_MAPPED_INPUT_SCHEMA)


def validate_complete_fragment(fragment: LanceFragment, *, camera_key: str, metadata_field_name: str) -> None:
    """Check completed metadata against the current episode and selected media."""
    columns = [*EPISODE_SOURCE_COLUMNS, metadata_field_name]
    for batch in fragment.scanner(columns=columns).to_batches():
        for row in batch.to_pylist():
            metadata = row[metadata_field_name]
            if not isinstance(metadata, dict):
                msg = "Terminal embedding metadata must be a mapping"
                raise TypeError(msg)
            selected = resolve_camera_media(row, camera_key)
            expected = {
                "episode_id": row["episode_id"],
                "materialization_digest": row["materialization_digest"],
                "camera_key": selected["camera_key"],
                "media_sha256": selected["media_sha256"],
                "episode_duration_ns": row["exclusive_end_ns"] - row["start_ns"],
            }
            mismatched = [name for name, value in expected.items() if metadata.get(name) != value]
            if mismatched:
                msg = "Terminal embedding metadata differs from its episode: " + ", ".join(mismatched)
                raise ValueError(msg)
