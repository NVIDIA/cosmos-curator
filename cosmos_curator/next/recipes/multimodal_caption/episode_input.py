# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical episode validation and flatten-only selected-camera projection."""

from typing import cast

import lance
import pyarrow as pa
from lance.fragment import LanceFragment

from cosmos_curator.next.recipes.multimodal_caption.contracts import (
    EGO_REQUIRED_TIMESTAMP_SOURCE,
    validate_terminal_binding,
)
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
EGO_EPISODE_SOURCE_COLUMNS = (
    "episode_id",
    "materialization_digest",
    "camera_media",
    "timestamp_source",
    "start_ns",
    "exclusive_end_ns",
    "sensor_media",
)
EGO_EPISODE_SOURCE_SCHEMA = pa.schema([EPISODE_SCHEMA.field(name) for name in EGO_EPISODE_SOURCE_COLUMNS])
EGO_EPISODE_MAPPED_INPUT_SCHEMA = pa.schema(
    [
        pa.field("fragment_id", pa.int64(), nullable=False),
        pa.field("row_offset", pa.int64(), nullable=False),
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("materialization_digest", pa.string(), nullable=False),
        pa.field("timestamp_source", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("exclusive_end_ns", pa.int64(), nullable=False),
        pa.field("camera_key", pa.string(), nullable=False),
        pa.field("media_location", pa.large_string(), nullable=False),
        pa.field("media_size_bytes", pa.int64(), nullable=False),
        pa.field("media_sha256", pa.string(), nullable=False),
        pa.field("ego_sensor_key", pa.string(), nullable=False),
        pa.field("ego_sensor_type", pa.string(), nullable=False),
        pa.field("ego_media_location", pa.large_string(), nullable=False),
        pa.field("ego_media_size_bytes", pa.int64(), nullable=False),
        pa.field("ego_media_sha256", pa.string(), nullable=False),
    ]
)
# Bounds are transient context inputs, not staged recovery columns. The
# producer's versioned make_episode_id(source_id, span) contract commits to
# both bounds; revisit this binding if that invariant changes.
EGO_EPISODE_STAGED_BINDING_COLUMNS = (
    *EPISODE_STAGED_BINDING_COLUMNS,
    "timestamp_source",
    "ego_sensor_key",
    "ego_sensor_type",
    "ego_media_location",
    "ego_media_size_bytes",
    "ego_media_sha256",
)


def _validate_episode_identity(dataset: lance.LanceDataset, *, uri: str) -> None:
    """Validate global episode identity and base contract versions once."""
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
    _validate_episode_identity(dataset, uri=uri)

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


def validate_ego_episode_input(
    dataset: lance.LanceDataset,
    *,
    uri: str,
    camera_key: str,
    sensor_key: str,
    max_episode_duration_ns: int,
) -> tuple[str | None, str | None]:
    """Validate camera/Ego input together and return representative camera then Ego S3 URIs."""
    validate_episode_table(dataset, uri=uri)
    _validate_episode_identity(dataset, uri=uri)
    representative_camera_s3_uri: str | None = None
    representative_ego_s3_uri: str | None = None
    scanner = dataset.scanner(
        columns=["episode_id", "timestamp_source", "start_ns", "exclusive_end_ns", "camera_media", "sensor_media"]
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            # Preflight validates every row before result fields are registered.
            _validate_timestamp_source(row)
            _validate_episode_duration(row, max_episode_duration_ns=max_episode_duration_ns)
            camera = resolve_camera_media(row, camera_key)
            ego = _resolve_ego_sensor(row, sensor_key)
            camera_location = str(camera["media_location"])
            ego_location = str(ego["media_location"])
            if representative_camera_s3_uri is None and camera_location.lower().startswith("s3://"):
                representative_camera_s3_uri = camera_location
            if representative_ego_s3_uri is None and ego_location.lower().startswith("s3://"):
                representative_ego_s3_uri = ego_location
    return representative_camera_s3_uri, representative_ego_s3_uri


def project_episode_camera_and_ego(
    batch: pa.Table,
    *,
    camera_key: str,
    ego_sensor_key: str,
    max_episode_duration_ns: int,
) -> pa.Table:
    """Flatten one camera and one Ego descriptor with canonical episode bounds."""
    required = {"_rowaddr", *EGO_EPISODE_SOURCE_COLUMNS}
    missing = sorted(required - set(batch.schema.names))
    if missing:
        msg = f"Lance Ego caption read is missing required column(s): {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for row in batch.select(["_rowaddr", *EGO_EPISODE_SOURCE_COLUMNS]).to_pylist():
        # Projection repeats the guards against the pinned Lance snapshot used by
        # execution, which can differ from the earlier pre-registration scan.
        _validate_timestamp_source(row)
        _validate_episode_duration(row, max_episode_duration_ns=max_episode_duration_ns)
        camera = resolve_camera_media(row, camera_key)
        ego = _resolve_ego_sensor(row, ego_sensor_key)
        row_address = int(row["_rowaddr"])
        rows.append(
            {
                "fragment_id": row_address >> _ROW_ADDRESS_FRAGMENT_SHIFT,
                "row_offset": row_address & _ROW_ADDRESS_OFFSET_MASK,
                "episode_id": row["episode_id"],
                "materialization_digest": row["materialization_digest"],
                "timestamp_source": row["timestamp_source"],
                "start_ns": row["start_ns"],
                "exclusive_end_ns": row["exclusive_end_ns"],
                "camera_key": camera["camera_key"],
                "media_location": camera["media_location"],
                "media_size_bytes": camera["media_size_bytes"],
                "media_sha256": camera["media_sha256"],
                "ego_sensor_key": ego["sensor_key"],
                "ego_sensor_type": ego["sensor_type"],
                "ego_media_location": ego["media_location"],
                "ego_media_size_bytes": ego["media_size_bytes"],
                "ego_media_sha256": ego["media_sha256"],
            }
        )
    return pa.Table.from_pylist(rows, schema=EGO_EPISODE_MAPPED_INPUT_SCHEMA)


def validate_complete_fragment(
    fragment: LanceFragment,
    *,
    camera_key: str,
    metadata_field_name: str,
    ego_sensor_key: str | None = None,
) -> None:
    """Validate terminal metadata against current selected-camera bindings."""
    columns = [*EPISODE_SOURCE_COLUMNS, metadata_field_name]
    if ego_sensor_key is not None:
        columns.extend(["timestamp_source", "sensor_media"])
    for raw_batch in fragment.scanner(columns=columns).to_batches():
        for row in raw_batch.to_pylist():
            metadata = row[metadata_field_name]
            if not isinstance(metadata, dict):
                msg = "terminal caption metadata must be a mapping"
                raise TypeError(msg)
            selected = resolve_camera_media(row, camera_key)
            ego_binding: dict[str, object] = {}
            if ego_sensor_key is not None:
                # Completed-fragment validation compares publication against the
                # current live row, independently of preflight and projection.
                _validate_timestamp_source(row)
                ego = _resolve_ego_sensor(row, ego_sensor_key)
                ego_binding = {
                    "timestamp_source": row["timestamp_source"],
                    "ego_sensor_key": ego["sensor_key"],
                    "ego_sensor_type": ego["sensor_type"],
                    "ego_media_sha256": ego["media_sha256"],
                }
            validate_terminal_binding(
                metadata,
                episode_id=row["episode_id"],
                materialization_digest=row["materialization_digest"],
                camera_key=selected["camera_key"],
                media_sha256=selected["media_sha256"],
                **ego_binding,
            )


def _resolve_ego_sensor(record: dict[str, object], sensor_key: str) -> dict[str, object]:
    sensors = record.get("sensor_media")
    if not isinstance(sensors, list):
        msg = f"Episode {record.get('episode_id')!r} sensor_media must be a list"
        raise TypeError(msg)
    matches = [
        sensor
        for sensor in sensors
        if str(sensor.get("sensor_key")) == sensor_key and str(sensor.get("sensor_type")) == "egotrajectory"
    ]
    if len(matches) != 1:
        msg = f"Episode {record.get('episode_id')!r} has {len(matches)} egotrajectory descriptors for {sensor_key!r}"
        raise ValueError(msg)
    return dict(matches[0])


def _validate_episode_duration(record: dict[str, object], *, max_episode_duration_ns: int) -> None:
    duration_ns = cast("int", record["exclusive_end_ns"]) - cast("int", record["start_ns"])
    if duration_ns <= 0:
        msg = f"Episode {record.get('episode_id')!r} must have a positive duration; got {duration_ns} ns"
        raise ValueError(msg)
    if duration_ns > max_episode_duration_ns:
        msg = (
            f"Episode {record.get('episode_id')!r} duration {duration_ns} ns exceeds this Ego caption context "
            f"version's {max_episode_duration_ns} ns maximum; use multimodal-split clip.duration_s <= 10.0"
        )
        raise ValueError(msg)


def _validate_timestamp_source(record: dict[str, object]) -> None:
    timestamp_source = record.get("timestamp_source")
    if timestamp_source != EGO_REQUIRED_TIMESTAMP_SOURCE:
        msg = (
            f"Episode {record.get('episode_id')!r} has timestamp_source {timestamp_source!r}; "
            f"Ego caption contexts require {EGO_REQUIRED_TIMESTAMP_SOURCE!r}"
        )
        raise ValueError(msg)
