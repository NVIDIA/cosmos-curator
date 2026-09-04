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
"""Tests for MCAP protobuf ``EgoTrajectorySensor``."""

from pathlib import Path

import numpy as np
import pytest
from google.protobuf import descriptor_pb2

from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import (
    DEFAULT_FRAME,
    DEFAULT_TOPIC,
    EgoTrajectorySensor,
)
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    one_window_spec,
    protobuf_descriptor_set_from_proto,
    protobuf_message_class,
    write_protobuf_mcap,
)

_REPO_ROOT = Path(__file__).parents[5]
_REFERENCE_SCHEMA_NAME = "cosmos_curator.sensors.egotrajectory.v1.EgotrajectorySample"
_REFERENCE_MAPPING_PATH = (
    _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "examples" / "egotrajectory_protobuf_mapping.yaml"
)
_REFERENCE_PROTO_PATH = _REPO_ROOT / "cosmos_curator" / "core" / "sensors" / "schemas" / "egotrajectory.proto"
_IDENTITY_ORIENTATION_XYZW = (0.0, 0.0, 0.0, 1.0)


def _reference_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build the descriptor set for the checked-in reference ego-trajectory schema."""
    return protobuf_descriptor_set_from_proto(_REFERENCE_PROTO_PATH)


def _reference_payload(
    sensor_timestamp_ns: int,
    *,
    host_timestamp_ns: int | None = None,
    sequence_counter: int = 1,
    position_m: tuple[float, ...] = (0.0, 0.0, 0.0),
    orientation_xyzw: tuple[float, ...] = _IDENTITY_ORIENTATION_XYZW,
    pose_valid: bool = True,
) -> bytes:
    """Serialize one reference ego-trajectory protobuf sample."""
    message_cls = protobuf_message_class(_reference_descriptor_set(), _REFERENCE_SCHEMA_NAME)
    sample = message_cls()
    sample.sensor_timestamp_ns = sensor_timestamp_ns
    sample.host_timestamp_ns = sensor_timestamp_ns + 50 if host_timestamp_ns is None else host_timestamp_ns
    sample.sequence_counter = sequence_counter
    sample.position_m.extend(position_m)
    sample.orientation_xyzw.extend(orientation_xyzw)
    sample.pose_valid = pose_valid
    return sample.SerializeToString()


def _write_reference_mcap(path: Path, samples: list[McapSample]) -> None:
    """Write an MCAP using the checked-in reference ego-trajectory schema."""
    write_protobuf_mcap(
        path,
        samples,
        topic=DEFAULT_TOPIC,
        schema_name=_REFERENCE_SCHEMA_NAME,
        schema_data=_reference_descriptor_set().SerializeToString(),
        library="cosmos_curator reference egotrajectory sensor test",
    )


def _reference_sensor(
    path: Path,
    *,
    protobuf_mapping: str | Path | dict[str, object] = _REFERENCE_MAPPING_PATH,
    frame: str = DEFAULT_FRAME,
) -> EgoTrajectorySensor:
    """Create an ego-trajectory sensor using the checked-in reference mapping by default."""
    return EgoTrajectorySensor(
        path,
        schema_name=_REFERENCE_SCHEMA_NAME,
        protobuf_mapping=protobuf_mapping,
        frame=frame,
    )


def _minimal_mapping(**overrides: object) -> dict[str, object]:
    """Required-only ego-trajectory mapping (no host/sequence optionals)."""
    fields: dict[str, object] = {
        "sensor_timestamp_ns": {"from": "sensor_timestamp_ns", "type": "timestamp", "unit": "ns"},
        "position_m": {"from": "position_m", "type": "float"},
        "orientation_xyzw": {"from": "orientation_xyzw", "type": "float"},
        "pose_valid": {"from": "pose_valid", "type": "bool"},
    }
    fields.update(overrides)
    return {"fields": fields}


def test_ego_trajectory_sensor_reads_reference_schema_with_checked_in_mapping(tmp_path: Path) -> None:
    """The shipped reference schema and YAML should produce complete EgoTrajectory data."""
    path = tmp_path / "reference_egotrajectory.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_payload(100, sequence_counter=10, position_m=(1.0, 2.0, 3.0)),
            ),
            McapSample(
                log_time_ns=200,
                data=_reference_payload(
                    200,
                    sequence_counter=11,
                    position_m=(4.0, 5.0, 6.0),
                    orientation_xyzw=(0.0, 0.0, float(np.sqrt(0.5)), float(np.sqrt(0.5))),
                ),
            ),
        ],
    )

    batch = next(_reference_sensor(path).sample(one_window_spec(100, 300)))

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100, 200], dtype=np.int64))
    np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([150, 250], dtype=np.int64))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([10, 11], dtype=np.uint64))
    np.testing.assert_array_equal(batch.pose_valid, np.array([True, True], dtype=np.bool_))
    np.testing.assert_allclose(batch.positions_m[0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(batch.positions_m[1], (4.0, 5.0, 6.0))
    half_sqrt = float(np.sqrt(0.5))
    np.testing.assert_allclose(
        batch.orientations_xyzw,
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt]]),
    )
    assert batch.frame == DEFAULT_FRAME


def test_ego_trajectory_sensor_preserves_invalid_identity_poses(tmp_path: Path) -> None:
    """Keep-and-mask rows should retain zero-position identity poses when invalid."""
    path = tmp_path / "invalid_pose.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_payload(100, position_m=(1.0, 0.0, 0.0), pose_valid=True),
            ),
            McapSample(
                log_time_ns=200,
                data=_reference_payload(200, pose_valid=False, sequence_counter=2),
            ),
        ],
    )

    batch = next(_reference_sensor(path).sample(one_window_spec(100, 300)))

    assert batch.pose_valid.tolist() == [True, False]
    np.testing.assert_array_equal(batch.positions_m[1], np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(batch.orientations_xyzw[1], np.array([0.0, 0.0, 0.0, 1.0]))


def test_ego_trajectory_reference_mapping_empty_window_preserves_optional_arrays(tmp_path: Path) -> None:
    """Fully mapped reference optionals should remain present in empty windows."""
    path = tmp_path / "reference_empty_window.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])

    batch = next(_reference_sensor(path).sample(one_window_spec(200, 300)))

    assert batch.align_timestamps_ns.shape == (0,)
    assert batch.sensor_timestamps_ns.shape == (0,)
    assert batch.positions_m.shape == (0, 3)
    assert batch.orientations_xyzw.shape == (0, 4)
    assert batch.pose_valid.shape == (0,)
    assert batch.host_timestamps_ns is not None
    assert batch.host_timestamps_ns.shape == (0,)
    assert batch.sequence_counter is not None
    assert batch.sequence_counter.shape == (0,)
    assert batch.frame == DEFAULT_FRAME


def test_ego_trajectory_sensor_windows_by_log_time(tmp_path: Path) -> None:
    """Sampling windows should include only messages whose log_time falls inside."""
    path = tmp_path / "windowed.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(200, sequence_counter=2)),
            McapSample(log_time_ns=300, data=_reference_payload(300, sequence_counter=3)),
        ],
    )

    batch = next(_reference_sensor(path).sample(one_window_spec(150, 250)))

    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([200], dtype=np.int64))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([2], dtype=np.uint64))


def test_ego_trajectory_sensor_rejects_missing_required_mapping_fields(tmp_path: Path) -> None:
    """Init should reject mappings that omit required EgoTrajectory destinations."""
    path = tmp_path / "missing_required.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])

    with pytest.raises(ValueError, match="must map required field"):
        EgoTrajectorySensor(
            path,
            schema_name=_REFERENCE_SCHEMA_NAME,
            protobuf_mapping={
                "fields": {
                    "sensor_timestamp_ns": {"from": "sensor_timestamp_ns", "type": "timestamp", "unit": "ns"},
                }
            },
        )


def test_ego_trajectory_sensor_rejects_missing_mapped_topic(tmp_path: Path) -> None:
    """Accessing timeline properties should fail when the configured topic is absent."""
    path = tmp_path / "other_topic.mcap"
    write_protobuf_mcap(
        path,
        [McapSample(log_time_ns=100, data=_reference_payload(100))],
        topic="/other",
        schema_name=_REFERENCE_SCHEMA_NAME,
        schema_data=_reference_descriptor_set().SerializeToString(),
    )
    sensor = EgoTrajectorySensor(
        path,
        schema_name=_REFERENCE_SCHEMA_NAME,
        protobuf_mapping=_REFERENCE_MAPPING_PATH,
    )

    with pytest.raises(ValueError, match="no MCAP messages on topic"):
        _ = sensor.start_ns


def test_ego_trajectory_sensor_read_all(tmp_path: Path) -> None:
    """read_all should decode the full topic timeline into one batch."""
    path = tmp_path / "read_all.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(200, sequence_counter=2)),
        ],
    )

    batch = _reference_sensor(path).read_all()

    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100, 200], dtype=np.int64))
    assert len(batch.positions_m) == 2


def test_ego_trajectory_sensor_rejects_duplicate_mapped_align_timestamps(tmp_path: Path) -> None:
    """Mapped ego-trajectory rows require strictly increasing alignment timestamps."""
    path = tmp_path / "duplicate_timestamps.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(100, sequence_counter=2)),
        ],
    )

    with pytest.raises(ValueError, match="strictly sorted in ascending order with no duplicates"):
        next(_reference_sensor(path).sample(one_window_spec(100, 300)))


def test_ego_trajectory_sensor_read_all_rejects_recording_wide_duplicate_alignment(tmp_path: Path) -> None:
    """Eager decoding requires one globally strictly increasing source timeline."""
    path = tmp_path / "recording_duplicate_timestamps.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(100, sequence_counter=2)),
        ],
    )

    with pytest.raises(ValueError, match="strictly sorted in ascending order with no duplicates"):
        _reference_sensor(path).read_all()


def test_ego_trajectory_sensor_custom_mapping_preserves_optional_absence_for_all_windows(tmp_path: Path) -> None:
    """Unmapped host/sequence outputs stay absent for populated and empty windows."""
    path = tmp_path / "custom_optional_egotrajectory.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])
    sensor = _reference_sensor(path, protobuf_mapping=_minimal_mapping())

    populated = next(sensor.sample(one_window_spec(100, 200)))
    empty = next(sensor.sample(one_window_spec(200, 300)))
    for batch in (populated, empty):
        assert batch.host_timestamps_ns is None
        assert batch.sequence_counter is None
    np.testing.assert_array_equal(populated.sensor_timestamps_ns, np.array([100], dtype=np.int64))
    assert empty.align_timestamps_ns.shape == (0,)


def test_ego_trajectory_sensor_rejects_empty_frame(tmp_path: Path) -> None:
    """Init should reject an empty library frame name."""
    path = tmp_path / "empty_frame.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])

    with pytest.raises(ValueError, match="frame must be a non-empty string"):
        _reference_sensor(path, frame="")


def test_ego_trajectory_sensor_stream_timestamps_not_implemented(tmp_path: Path) -> None:
    """stream_timestamps is camera-only; EgoTrajectorySensor raises NotImplementedError."""
    path = tmp_path / "stream_timestamps.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])

    with pytest.raises(NotImplementedError, match="only implemented for CameraSensor"):
        list(_reference_sensor(path).stream_timestamps())


def test_ego_trajectory_sensor_rejects_wrong_position_length(tmp_path: Path) -> None:
    """Decoded positions must contain exactly three doubles."""
    path = tmp_path / "short_position.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_payload(100, position_m=(1.0, 0.0)),
            )
        ],
    )

    with pytest.raises(ValueError, match=r"fields\.position_m expected 3 values"):
        next(_reference_sensor(path).sample(one_window_spec(100, 200)))


def test_ego_trajectory_sensor_rejects_wrong_orientation_length(tmp_path: Path) -> None:
    """Decoded orientations must contain exactly four doubles."""
    path = tmp_path / "short_orientation.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_reference_payload(100, orientation_xyzw=(0.0, 0.0, 1.0)),
            )
        ],
    )

    with pytest.raises(ValueError, match=r"fields\.orientation_xyzw expected 4 values"):
        next(_reference_sensor(path).sample(one_window_spec(100, 200)))
