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

"""Tests for complete-camera episode assembly."""

import pyarrow as pa
import pytest

from cosmos_curator.next.recipes.multimodal_split.publication import assemble_episode_group, stage_episode_batch
from cosmos_curator.next.recipes.multimodal_split.split import CAMERA_OUTCOME_SCHEMA


def _outcome(camera_key: str, *, status: str = "success") -> dict[str, object]:
    marker = "1" if camera_key == "front" else "2"
    success = status == "success"
    return {
        "status": status,
        "record_schema_version": 1,
        "media_contract_version": 2,
        "episode_id": "episode",
        "split_clip_id": "split",
        "source_id": "source",
        "source_session_id": "session",
        "source_session_location": "/input/session",
        "start_ns": 0,
        "exclusive_end_ns": 1_000_000_000,
        "sampling_fps": 30,
        "expected_row_count": 30,
        "timestamp_source": "capture",
        "split_contract_digest": "contract",
        "planned_camera_keys": ["front", "left"],
        "camera_key": camera_key,
        "sensor_type": "camera",
        "source_media_location": f"/input/session/{camera_key}.mp4",
        "media_location": f"/output/{camera_key}.mp4" if success else None,
        "media_size_bytes": 100 if success else None,
        "media_sha256": marker * 64 if success else None,
        "timeline_location": f"/output/{camera_key}.mp4.timestamps.npy" if success else None,
        "timeline_sha256": "a" * 64 if success else None,
        "observations_location": f"/output/{camera_key}.mp4.observations.npy" if success else None,
        "observations_sha256": "b" * 64 if success else None,
    }


def _assemble(rows: list[dict[str, object]]) -> dict[str, object]:
    table = pa.Table.from_pylist(rows, schema=CAMERA_OUTCOME_SCHEMA)
    return assemble_episode_group(table).to_pylist()[0]


def test_receipt_order_does_not_change_the_canonical_episode() -> None:
    """One multiview clip is one row with canonical camera serialization."""
    first = _assemble([_outcome("left"), _outcome("front")])
    second = _assemble([_outcome("front"), _outcome("left")])

    assert first == second
    assert first["assembly_status"] == "complete"
    assert [camera["camera_key"] for camera in first["camera_media"]] == ["front", "left"]


@pytest.mark.parametrize(
    "rows",
    [
        [_outcome("front")],
        [_outcome("front", status="failure"), _outcome("left", status="failure")],
        [_outcome("front"), _outcome("left", status="failure")],
    ],
)
def test_incomplete_and_all_failed_camera_sets_emit_no_episode_row(rows: list[dict[str, object]]) -> None:
    """Failures remain countable without producing a partial canonical row."""
    result = _assemble(rows)

    assert result["assembly_status"] == "dropped"
    assert result["camera_media"] is None
    assert result["camera_clips_written"] == sum(row["status"] == "success" for row in rows)


@pytest.mark.parametrize("defect", ["duplicate", "unexpected", "conflicting"])
def test_malformed_camera_groups_fail_instead_of_being_reconciled(defect: str) -> None:
    """Only an exact, internally consistent planned camera set is accepted."""
    rows = [_outcome("front"), _outcome("left")]
    if defect == "duplicate":
        rows[1] = _outcome("front")
    elif defect == "unexpected":
        rows[1]["camera_key"] = "rear"
    else:
        rows[1]["source_session_location"] = "/different/session"

    expected = {"duplicate": "duplicate", "unexpected": "unexpected", "conflicting": "conflicting"}[defect]
    with pytest.raises(ValueError, match=expected):
        _assemble(rows)


@pytest.mark.parametrize("planned_camera_keys", [[], ["left", "front"]])
def test_planned_camera_set_must_be_nonempty_and_canonical(planned_camera_keys: list[str]) -> None:
    """Planner facts retain canonical camera ordering across Arrow serialization."""
    rows = [_outcome("front"), _outcome("left")]
    for row in rows:
        row["planned_camera_keys"] = planned_camera_keys

    with pytest.raises(ValueError, match="non-canonical planned camera set"):
        _assemble(rows)


def test_a_successful_camera_requires_every_artifact_descriptor() -> None:
    """The nullable outcome union must not admit a partial success into publication."""
    rows = [_outcome("front"), _outcome("left")]
    rows[0]["media_sha256"] = None

    with pytest.raises(ValueError, match="missing artifact descriptors"):
        _assemble(rows)


def test_staging_rejects_mixed_assembly_states_before_writing() -> None:
    """Only complete assembly rows may cross the Lance staging boundary."""
    batch = pa.table({"assembly_status": ["complete", "dropped"]})

    with pytest.raises(ValueError, match=r"assembly states.*complete.*dropped"):
        stage_episode_batch(batch, uri="/unused", storage_profile="default")
