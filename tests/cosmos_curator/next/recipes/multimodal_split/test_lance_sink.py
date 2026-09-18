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

"""Recipe-level Lance publication and extension-boundary tests."""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from cosmos_curator.core.sensors.types.types import TimestampSource
from cosmos_curator.next.recipes.multimodal_split.lance_sink import (
    append_episode_fragment,
    open_episode_table,
    open_or_create_episode_table,
    write_episode_fragment,
)
from cosmos_curator.next.recipes.multimodal_split.records import (
    EPISODE_SCHEMA,
    EpisodeFacts,
    build_episode_record,
    episode_table,
)
from cosmos_curator.next.utils.lance_utils import LANCE_DATA_STORAGE_VERSION


def _facts(episode_id: str) -> EpisodeFacts:
    return EpisodeFacts(
        episode_id=episode_id,
        split_clip_id=f"split-{episode_id}",
        source_id="source",
        source_session_id="session",
        source_session_location="/input/session",
        start_ns=0 if episode_id == "a" else 1_000_000_000,
        exclusive_end_ns=1_000_000_000 if episode_id == "a" else 2_000_000_000,
        sampling_fps=30,
        row_count=30,
        timestamp_source=TimestampSource.SENSOR,
        split_contract_digest="contract",
    )


def _record(episode_id: str) -> dict[str, object]:
    camera = {
        "camera_key": "front",
        "source_media_location": "/input/front.mp4",
        "media_location": f"/output/{episode_id}/front.mp4",
        "media_size_bytes": 100,
        "media_sha256": episode_id * 64,
        "timeline_location": f"/output/{episode_id}/front.mp4.timestamps.npy",
        "timeline_sha256": "1" * 64,
        "sensor_timestamps_location": f"/output/{episode_id}/front.mp4.sensor_timestamps.npy",
        "sensor_timestamps_sha256": "2" * 64,
    }
    return build_episode_record(_facts(episode_id), [camera])


def _append(uri: str, episode_id: str) -> int:
    candidate = write_episode_fragment(episode_table([_record(episode_id)]), uri=uri, storage_profile="default")
    assert candidate is not None
    return append_episode_fragment(candidate, uri=uri, storage_profile="default", attempts=1)


def test_absent_episode_table_bootstraps_appends_and_reopens(tmp_path: Path) -> None:
    """Nested camera rows travel through the actual stage/append boundary."""
    uri = str(tmp_path / "episodes.lance")
    bootstrap = open_or_create_episode_table(uri=uri, storage_profile="default")

    version = _append(uri, "a")
    reopened = open_episode_table(uri=uri, storage_profile="default")

    assert bootstrap.version == 1
    assert version == 2
    assert reopened.schema == EPISODE_SCHEMA
    assert reopened.to_table().to_pylist() == [_record("a")]


def test_existing_nullable_extension_values_survive_later_base_appends(tmp_path: Path) -> None:
    """The base publisher neither owns nor erases a downstream nullable group."""
    uri = str(tmp_path / "episodes.lance")
    extension = pa.field(
        "result__test_v1",
        pa.struct([pa.field("value", pa.string(), nullable=False)]),
        nullable=True,
    )
    extended_schema = EPISODE_SCHEMA.append(extension)
    first = {**_record("a"), "result__test_v1": {"value": "kept"}}
    lance.write_dataset(
        pa.Table.from_pylist([first], schema=extended_schema),
        uri,
        mode="create",
        data_storage_version=LANCE_DATA_STORAGE_VERSION,
    )
    open_or_create_episode_table(uri=uri, storage_profile="default")

    _append(uri, "b")
    rows = sorted(lance.dataset(uri).to_table().to_pylist(), key=lambda row: row["episode_id"])

    assert rows[0]["result__test_v1"] == {"value": "kept"}
    assert rows[1]["result__test_v1"] is None


def test_incompatible_storage_version_base_field_and_extension_are_rejected(tmp_path: Path) -> None:
    """Existing tables fail closed at each schema compatibility boundary."""
    wrong_version = str(tmp_path / "wrong-version.lance")
    lance.write_dataset(
        pa.Table.from_batches([], schema=EPISODE_SCHEMA),
        wrong_version,
        mode="create",
        data_storage_version="2.1",
    )
    with pytest.raises(ValueError, match="data_storage_version"):
        open_or_create_episode_table(uri=wrong_version, storage_profile="default")

    wrong_base = str(tmp_path / "wrong-base.lance")
    schema = EPISODE_SCHEMA.set(
        EPISODE_SCHEMA.get_field_index("episode_id"),
        pa.field("episode_id", pa.large_string(), nullable=False),
    )
    lance.write_dataset(pa.Table.from_batches([], schema=schema), wrong_base, mode="create", data_storage_version="2.2")
    with pytest.raises(ValueError, match=r"incompatible producer-owned field.*episode_id"):
        open_or_create_episode_table(uri=wrong_base, storage_profile="default")

    wrong_extension = str(tmp_path / "wrong-extension.lance")
    schema = EPISODE_SCHEMA.append(pa.field("required_result", pa.string(), nullable=False))
    lance.write_dataset(
        pa.Table.from_batches([], schema=schema),
        wrong_extension,
        mode="create",
        data_storage_version="2.2",
    )
    with pytest.raises(ValueError, match=r"non-nullable extension field.*required_result"):
        open_or_create_episode_table(uri=wrong_extension, storage_profile="default")
