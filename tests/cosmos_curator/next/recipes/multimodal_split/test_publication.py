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
from cosmos_curator.next.recipes.multimodal_split.split import SENSOR_OUTCOME_SCHEMA


def _outcome(
    camera_key: str,
    *,
    status: str = "success",
    sensor_type: str = "camera",
    planned_sensor_keys: list[str] | None = None,
) -> dict[str, object]:
    marker = "1" if camera_key == "front" else "2"
    success = status == "success"
    is_camera = sensor_type == "camera"
    # A sensor clip is an MCAP. Fabricating .mp4 paths for one would let these
    # tests pass against an outcome shape the splitter never emits.
    suffix = ".mp4" if is_camera else ".mcap"

    def sidecar(value: str) -> str | None:
        return value if success and is_camera else None

    return {
        "status": status,
        "record_schema_version": 2,
        "media_contract_version": 3,
        "episode_id": "episode",
        "split_clip_id": "split",
        "source_id": "source",
        "source_session_id": "session",
        "source_session_location": "/input/session",
        "start_ns": 0,
        "exclusive_end_ns": 1_000_000_000,
        "sampling_fps": 30,
        "expected_row_count": 30,
        "timestamp_source": "sensor",
        "split_contract_digest": "contract",
        "planned_camera_keys": ["front", "left"],
        "planned_sensor_keys": planned_sensor_keys or [],
        "sensor_key": camera_key,
        "sensor_type": sensor_type,
        "source_media_location": f"/input/session/{camera_key}{suffix}",
        "media_location": f"/output/{camera_key}{suffix}" if success else None,
        "media_size_bytes": 100 if success else None,
        "media_sha256": marker * 64 if success else None,
        # A cut sensor has no grid to record and no per-row selection to account
        # for, so its sidecar descriptors are null where a camera's are set.
        "timeline_location": sidecar(f"/output/{camera_key}.mp4.timestamps.npy"),
        "timeline_sha256": sidecar("a" * 64),
        "sensor_timestamps_location": sidecar(f"/output/{camera_key}.mp4.sensor_timestamps.npy"),
        "sensor_timestamps_sha256": sidecar("b" * 64),
    }


def _assemble(rows: list[dict[str, object]]) -> dict[str, object]:
    table = pa.Table.from_pylist(rows, schema=SENSOR_OUTCOME_SCHEMA)
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
        rows[1]["sensor_key"] = "rear"
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


def _sensor_rows(*, planned: list[str], received: list[str]) -> list[dict[str, object]]:
    """Return one camera outcome plus a sensor outcome per received key."""
    rows: list[dict[str, object]] = [_outcome(key, planned_sensor_keys=planned) for key in ("front", "left")]
    rows += [_outcome(key, sensor_type="egotrajectory", planned_sensor_keys=planned) for key in received]
    return rows


def test_a_sensor_outcome_joins_the_episode_it_belongs_to() -> None:
    """A trajectory and the video of the same moment assemble into one row."""
    result = _assemble(_sensor_rows(planned=["ego"], received=["ego"]))

    assert result["assembly_status"] == "complete"
    assert [camera["camera_key"] for camera in result["camera_media"]] == ["front", "left"]
    assert [sensor["sensor_key"] for sensor in result["sensor_media"]] == ["ego"]
    assert result["sensor_media"][0]["sensor_type"] == "egotrajectory"
    assert result["camera_clips_written"] == 2
    assert result["sensor_clips_written"] == 1


def test_a_planned_sensor_that_never_arrived_drops_the_episode() -> None:
    """A trajectory the clip was meant to hold and does not costs it its video too."""
    result = _assemble(_sensor_rows(planned=["ego"], received=[]))

    assert result["assembly_status"] == "dropped"
    assert result["sensor_media"] is None
    # The cameras it did write are still accounted for, so a dropped episode
    # does not hide the work it cost.
    assert result["camera_clips_written"] == 2
    assert result["sensor_clips_written"] == 0


def test_a_failed_sensor_outcome_drops_the_episode() -> None:
    """A trajectory that was cut and failed is not a clip published without it."""
    rows = _sensor_rows(planned=["ego"], received=["ego"])
    rows[-1] = _outcome("ego", status="failure", sensor_type="egotrajectory", planned_sensor_keys=["ego"])

    result = _assemble(rows)

    assert result["assembly_status"] == "dropped"


@pytest.mark.parametrize("defect", ["duplicate", "unexpected"])
def test_malformed_sensor_groups_fail_instead_of_being_reconciled(defect: str) -> None:
    """A sensor set that is not the planned one is a broken stream, not an outcome.

    Short is ordinary and drops the episode. A key nobody planned, or one
    delivered twice, means the outcome stream itself is wrong, and publishing
    from it would put an artifact on an episode that never asked for it.
    """
    received = ["ego", "ego"] if defect == "duplicate" else ["rear"]

    with pytest.raises(ValueError, match=defect):
        _assemble(_sensor_rows(planned=["ego"], received=received))


def test_a_non_canonical_planned_sensor_set_is_refused() -> None:
    """An unsorted planned set could make a short episode look complete."""
    with pytest.raises(ValueError, match="non-canonical planned sensor set"):
        _assemble(_sensor_rows(planned=["imu", "ego"], received=["ego", "imu"]))


def test_a_dropped_episode_reports_the_sensor_bytes_it_did_write() -> None:
    """A trajectory written for an episode whose camera failed is still counted."""
    rows = [
        _outcome("front", status="failure", planned_sensor_keys=["ego"]),
        _outcome("left", planned_sensor_keys=["ego"]),
        _outcome("ego", sensor_type="egotrajectory", planned_sensor_keys=["ego"]),
    ]

    result = _assemble(rows)

    assert result["assembly_status"] == "dropped"
    assert result["camera_clips_written"] == 1
    assert result["sensor_clips_written"] == 1
    assert result["sensor_clip_bytes"] == 100
