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
"""Direct tests for the ``mp4-index``-schema binding over lance_fragment_recovery."""

import pathlib
from datetime import UTC, datetime

import lance  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from cosmos_curator.next.recipes.mp4_index import lance_sink
from cosmos_curator.next.recipes.mp4_index.contracts import (
    ERROR_SCHEMA,
    RECORD_SCHEMA_VERSION,
    VIDEO_IDENTITY_SCHEMA,
    make_session_camera_id,
)
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION

_CURRENT_CONTRACT_ID = "contract-v1"


def _publish_manifest_row(uri: str, *, video_id: str, record_schema_version: int, extraction_contract_id: str) -> None:
    """Stage one VIDEO_IDENTITY_SCHEMA row directly, skipping process_one_video entirely."""
    table = pa.Table.from_pylist(
        [
            {
                "video_id": video_id,
                "source_session_id": "sess-a",
                "camera_key": "camera",
                "session_camera_id": make_session_camera_id("sess-a", "camera"),
                "media_location": f"s3://unused/{video_id}.mp4",
                "media_size_bytes": 1,
                "frame_index_location": "s3://unused/frames.arrow",
                "metadata_location": "s3://unused/metadata.arrow",
                "moov_location": f"s3://unused/{video_id}.moov",
                "moof_table_location": None,
                "container_kind": "nonfragmented",
                "record_schema_version": record_schema_version,
                "extraction_contract_id": extraction_contract_id,
            }
        ],
        schema=VIDEO_IDENTITY_SCHEMA,
    )
    lance.write_dataset(table, uri, mode="create", data_storage_version=LANCE_DATA_STORAGE_VERSION)


def _publish_error_rows(errors_lance_uri: str, *, rows: list[tuple[str, str]]) -> None:
    """Stage multiple (video_id, error_type) ERROR_SCHEMA rows directly -- pre-seeds a known-failure table."""
    table = pa.Table.from_pylist(
        [
            {
                "video_id": video_id,
                "media_location": "s3://unused/unused.mp4",
                "error_type": error_type,
                "error_message": "pre-seeded for a test",
                "attempted_at": datetime.now(UTC),
            }
            for video_id, error_type in rows
        ],
        schema=ERROR_SCHEMA,
    )
    lance.write_dataset(table, errors_lance_uri, mode="create", data_storage_version=LANCE_DATA_STORAGE_VERSION)


def _publish_error_row(errors_lance_uri: str, *, video_id: str, error_type: str = "BoxNotFoundError") -> None:
    """Stage one ERROR_SCHEMA row directly -- pre-seeds a known-failure table for a test."""
    _publish_error_rows(errors_lance_uri, rows=[(video_id, error_type)])


def test_known_failed_video_ids_empty_when_table_does_not_exist(tmp_path: pathlib.Path) -> None:
    """A not-yet-bootstrapped error log reads back as an empty set, not an error."""
    uri = str(tmp_path / "errors.lance")

    assert lance_sink.known_failed_video_ids(uri, "default", {"a"}) == set()


def test_known_failed_video_ids_reads_real_error_rows(tmp_path: pathlib.Path) -> None:
    """A real error log's own video_id column comes back as the known-failed set."""
    uri = str(tmp_path / "errors.lance")
    _publish_error_row(uri, video_id="a")

    assert lance_sink.known_failed_video_ids(uri, "default", {"a"}) == {"a"}


def test_known_failed_video_ids_ignores_a_failure_outside_the_candidate_set(tmp_path: pathlib.Path) -> None:
    """A video_id's own failure row is irrelevant to a run that doesn't discover that video."""
    uri = str(tmp_path / "errors.lance")
    _publish_error_row(uri, video_id="a")

    assert lance_sink.known_failed_video_ids(uri, "default", {"b"}) == set()


def test_known_failed_video_ids_excludes_a_file_not_found_error(tmp_path: pathlib.Path) -> None:
    """A video whose only recorded failure is FileNotFoundError (e.g. a not-yet-created bucket) is never 'known failed'.

    S3Client.stat's own docstring: a missing bucket and a missing key answer the same bodyless
    404, both translated to FileNotFoundError -- a wrong-but-fixable bucket must not durably
    mark every video in a run as a permanent failure.
    """
    uri = str(tmp_path / "errors.lance")
    _publish_error_row(uri, video_id="a", error_type="FileNotFoundError")

    assert lance_sink.known_failed_video_ids(uri, "default", {"a"}) == set()


def test_known_failed_video_ids_still_includes_a_deterministic_failure_alongside_a_retryable_one(
    tmp_path: pathlib.Path,
) -> None:
    """A FileNotFoundError row for one video never hides a genuinely deterministic failure for another."""
    uri = str(tmp_path / "errors.lance")
    _publish_error_rows(uri, rows=[("a", "FileNotFoundError"), ("b", "AmbiguousVideoTrackError")])

    assert lance_sink.known_failed_video_ids(uri, "default", {"a", "b"}) == {"b"}


def test_committed_session_camera_owners_matches_against_the_candidate_set(tmp_path: pathlib.Path) -> None:
    """The returned mapping keys on the real stored session_camera_id column, with (video_id, location)."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="vid-a", record_schema_version=RECORD_SCHEMA_VERSION, extraction_contract_id="c"
    )
    dataset = lance.dataset(uri)
    session_camera_id = make_session_camera_id("sess-a", "camera")

    owners = lance_sink.committed_session_camera_owners(dataset, [session_camera_id])

    assert owners == {session_camera_id: ("vid-a", "s3://unused/vid-a.mp4")}


def test_committed_session_camera_owners_ignores_a_row_outside_the_candidate_set(tmp_path: pathlib.Path) -> None:
    """A committed row's own session_camera_id is irrelevant to a run whose candidates don't share it."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="vid-a", record_schema_version=RECORD_SCHEMA_VERSION, extraction_contract_id="c"
    )
    dataset = lance.dataset(uri)

    owners = lance_sink.committed_session_camera_owners(dataset, [make_session_camera_id("sess-b", "other-camera")])

    assert owners == {}


def test_committed_video_ids_matches_against_the_candidate_set(tmp_path: pathlib.Path) -> None:
    """Only video_ids present in both the table and the candidate set come back."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="vid-a", record_schema_version=RECORD_SCHEMA_VERSION, extraction_contract_id="c"
    )
    dataset = lance.dataset(uri)

    assert lance_sink.committed_video_ids(dataset, ["vid-a", "vid-b"]) == {"vid-a"}
    assert lance_sink.committed_video_ids(dataset, ["vid-b"]) == set()
    assert lance_sink.committed_video_ids(dataset, []) == set()


def test_validate_committed_versions_passes_on_an_empty_table(tmp_path: pathlib.Path) -> None:
    """A freshly bootstrapped, zero-row table never has anything to refuse."""
    uri = str(tmp_path / "manifest.lance")
    dataset = lance_sink.open_or_create_manifest_table(uri=uri, storage_profile="default")

    lance_sink.validate_committed_versions(dataset, extraction_contract_id=_CURRENT_CONTRACT_ID)


def test_validate_committed_versions_passes_when_every_row_matches_the_current_contract(
    tmp_path: pathlib.Path,
) -> None:
    """A table whose rows all carry the current version/contract is accepted."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="a", record_schema_version=RECORD_SCHEMA_VERSION, extraction_contract_id=_CURRENT_CONTRACT_ID
    )
    dataset = lance.dataset(uri)

    lance_sink.validate_committed_versions(dataset, extraction_contract_id=_CURRENT_CONTRACT_ID)


def test_validate_committed_versions_raises_on_a_mismatched_record_schema_version(tmp_path: pathlib.Path) -> None:
    """A row stamped with a stale row-shape version refuses the whole table, naming the count."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="a", record_schema_version=RECORD_SCHEMA_VERSION + 1, extraction_contract_id=_CURRENT_CONTRACT_ID
    )
    dataset = lance.dataset(uri)

    with pytest.raises(ValueError, match=r"1 row.*record_schema_version"):
        lance_sink.validate_committed_versions(dataset, extraction_contract_id=_CURRENT_CONTRACT_ID)


def test_validate_committed_versions_raises_on_a_mismatched_extraction_contract_id(tmp_path: pathlib.Path) -> None:
    """A row extracted under a different container_kind/session_id_path_depth/extraction_version refuses the table."""
    uri = str(tmp_path / "manifest.lance")
    _publish_manifest_row(
        uri, video_id="a", record_schema_version=RECORD_SCHEMA_VERSION, extraction_contract_id="contract-v0-stale"
    )
    dataset = lance.dataset(uri)

    with pytest.raises(ValueError, match=r"1 row.*extraction_contract_id"):
        lance_sink.validate_committed_versions(dataset, extraction_contract_id=_CURRENT_CONTRACT_ID)
