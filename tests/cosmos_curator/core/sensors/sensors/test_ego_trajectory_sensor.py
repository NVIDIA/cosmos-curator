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
from foxglove_schemas_protobuf.PoseInFrame_pb2 import PoseInFrame
from google.protobuf import descriptor_pb2
from google.protobuf.descriptor import FileDescriptor
from mcap.reader import make_reader

from cosmos_curator.core.sensors.data.egotrajectory_data import EgoTrajectory
from cosmos_curator.core.sensors.exceptions import AlignmentError, AlignmentFailureReason
from cosmos_curator.core.sensors.sampling.policy import NearestTimestampPolicy, NoSamplingPolicy
from cosmos_curator.core.sensors.sampling.spec import SamplingSpec
from cosmos_curator.core.sensors.sensors.ego_trajectory_sensor import (
    DEFAULT_FRAME,
    DEFAULT_TOPIC,
    EgoTrajectorySensor,
)
from cosmos_curator.core.sensors.sensors.group import SensorGroup
from tests.cosmos_curator.core.sensors.test_utils import (
    McapSample,
    make_sampling_grid,
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
_FOXGLOVE_SCHEMA_NAME = "foxglove.PoseInFrame"
_FOXGLOVE_TOPIC = "/foxglove/egotrajectory"
_FOXGLOVE_MAPPING_PATH = (
    _REPO_ROOT
    / "cosmos_curator"
    / "core"
    / "sensors"
    / "examples"
    / "foxglove_pose_in_frame_egotrajectory_mapping.yaml"
)


def _reference_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build the descriptor set for the checked-in reference ego-trajectory schema."""
    return protobuf_descriptor_set_from_proto(_REFERENCE_PROTO_PATH)


def _foxglove_pose_in_frame_descriptor_set() -> descriptor_pb2.FileDescriptorSet:
    """Build a complete descriptor set from Foxglove's official definition."""
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    added_files: set[str] = set()

    def add_file_descriptor(file_descriptor: FileDescriptor) -> None:
        file_name = file_descriptor.name
        if file_name in added_files:
            return
        for dependency in file_descriptor.dependencies:
            add_file_descriptor(dependency)
        file_descriptor.CopyToProto(descriptor_set.file.add())
        added_files.add(file_name)

    add_file_descriptor(PoseInFrame.DESCRIPTOR.file)
    return descriptor_set


def _foxglove_pose_in_frame_payload(
    *,
    timestamp_ns: int,
    frame_id: str,
    position_m: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
) -> bytes:
    """Serialize one official Foxglove PoseInFrame message."""
    message = PoseInFrame(frame_id=frame_id)
    message.timestamp.seconds = timestamp_ns // 1_000_000_000
    message.timestamp.nanos = timestamp_ns % 1_000_000_000
    message.pose.position.x, message.pose.position.y, message.pose.position.z = position_m
    message.pose.orientation.x, message.pose.orientation.y = orientation_xyzw[:2]
    message.pose.orientation.z, message.pose.orientation.w = orientation_xyzw[2:]
    return bytes(message.SerializeToString())


def _foxglove_pose_in_frame_without_pose_payload(*, timestamp_ns: int, frame_id: str) -> bytes:
    """Serialize a Foxglove message with its proto3-default pose left unset."""
    message = PoseInFrame(frame_id=frame_id)
    message.timestamp.seconds = timestamp_ns // 1_000_000_000
    message.timestamp.nanos = timestamp_ns % 1_000_000_000
    return bytes(message.SerializeToString())


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
    return bytes(sample.SerializeToString())


def _write_reference_mcap(path: Path, samples: list[McapSample], *, chunk_size: int = 1_048_576) -> None:
    """Write an MCAP using the checked-in reference ego-trajectory schema."""
    write_protobuf_mcap(
        path,
        samples,
        topic=DEFAULT_TOPIC,
        schema_name=_REFERENCE_SCHEMA_NAME,
        schema_data=_reference_descriptor_set().SerializeToString(),
        library="cosmos_curator reference egotrajectory sensor test",
        chunk_size=chunk_size,
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


def _nearest_sample(sensor: EgoTrajectorySensor, spec: SamplingSpec) -> EgoTrajectory:
    """Advance one sensor-alignment window under the required nearest policy."""
    return next(sensor.sample(spec, policy=NearestTimestampPolicy()))


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

    batch = _nearest_sample(_reference_sensor(path), one_window_spec(100, 300))

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(batch.host_timestamps_ns, np.array([150], dtype=np.int64))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([10], dtype=np.uint64))
    np.testing.assert_array_equal(batch.pose_valid, np.array([True], dtype=np.bool_))
    np.testing.assert_allclose(batch.positions_m[0], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(batch.orientations_xyzw, np.array([[0.0, 0.0, 0.0, 1.0]]))
    assert batch.frame == DEFAULT_FRAME


def test_ego_trajectory_sensor_reads_foxglove_pose_in_frame_with_checked_in_mapping(tmp_path: Path) -> None:
    """The Foxglove YAML maps one PoseInFrame message into one trajectory row."""
    path = tmp_path / "foxglove_pose_in_frame.mcap"
    write_protobuf_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_foxglove_pose_in_frame_payload(
                    timestamp_ns=10_000,
                    frame_id="arbitrary_upstream_frame",
                    position_m=(1.0, 2.0, 3.0),
                    orientation_xyzw=(0.0, 0.0, float(np.sqrt(0.5)), float(np.sqrt(0.5))),
                ),
            ),
            McapSample(
                log_time_ns=200,
                data=_foxglove_pose_in_frame_payload(
                    timestamp_ns=20_000,
                    frame_id="another_upstream_frame",
                    position_m=(4.0, 5.0, 6.0),
                    orientation_xyzw=_IDENTITY_ORIENTATION_XYZW,
                ),
            ),
        ],
        topic=_FOXGLOVE_TOPIC,
        schema_name=_FOXGLOVE_SCHEMA_NAME,
        schema_data=_foxglove_pose_in_frame_descriptor_set().SerializeToString(),
        library="cosmos_curator Foxglove PoseInFrame sensor test",
    )

    sensor = EgoTrajectorySensor(
        path,
        topic=_FOXGLOVE_TOPIC,
        schema_name=_FOXGLOVE_SCHEMA_NAME,
        protobuf_mapping=_FOXGLOVE_MAPPING_PATH,
    )
    batch = _nearest_sample(sensor, one_window_spec(10_000, 30_000))

    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([10_000], dtype=np.int64))
    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([10_000], dtype=np.int64))
    np.testing.assert_allclose(batch.positions_m, np.array([[1.0, 2.0, 3.0]]))
    half_sqrt = float(np.sqrt(0.5))
    np.testing.assert_allclose(batch.orientations_xyzw, np.array([[0.0, 0.0, half_sqrt, half_sqrt]]))
    np.testing.assert_array_equal(batch.pose_valid, np.array([True], dtype=np.bool_))
    assert batch.frame == DEFAULT_FRAME


def test_ego_trajectory_sensor_rejects_foxglove_pose_in_frame_without_pose(tmp_path: Path) -> None:
    """An unset Foxglove pose must not be silently converted to a trajectory row."""
    path = tmp_path / "foxglove_pose_in_frame_without_pose.mcap"
    write_protobuf_mcap(
        path,
        [
            McapSample(
                log_time_ns=100,
                data=_foxglove_pose_in_frame_without_pose_payload(
                    timestamp_ns=10_000,
                    frame_id="arbitrary_upstream_frame",
                ),
            ),
        ],
        topic=_FOXGLOVE_TOPIC,
        schema_name=_FOXGLOVE_SCHEMA_NAME,
        schema_data=_foxglove_pose_in_frame_descriptor_set().SerializeToString(),
        library="cosmos_curator Foxglove PoseInFrame sensor test",
    )
    sensor = EgoTrajectorySensor(
        path,
        topic=_FOXGLOVE_TOPIC,
        schema_name=_FOXGLOVE_SCHEMA_NAME,
        protobuf_mapping=_FOXGLOVE_MAPPING_PATH,
    )

    with pytest.raises(ValueError, match=r"fields\.position_m path 'pose\.position\.x' was not found"):
        _nearest_sample(sensor, one_window_spec(100, 200))


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

    batch = _nearest_sample(_reference_sensor(path), one_window_spec(200, 300))

    assert batch.pose_valid.tolist() == [False]
    np.testing.assert_array_equal(batch.positions_m[0], np.zeros(3, dtype=np.float64))
    np.testing.assert_array_equal(batch.orientations_xyzw[0], np.array([0.0, 0.0, 0.0, 1.0]))


def test_ego_trajectory_sensor_uses_the_complete_clip_as_its_candidate_set(tmp_path: Path) -> None:
    """A request outside MCAP log-time coverage can use a pose from the clip."""
    path = tmp_path / "reference_empty_window.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])

    batch = _nearest_sample(_reference_sensor(path), one_window_spec(200, 300))

    np.testing.assert_array_equal(batch.align_timestamps_ns, [200])
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, [100])
    assert batch.positions_m.shape == (1, 3)
    assert batch.orientations_xyzw.shape == (1, 4)
    assert batch.pose_valid.shape == (1,)
    assert batch.host_timestamps_ns is not None
    assert batch.host_timestamps_ns.shape == (1,)
    assert batch.sequence_counter is not None
    assert batch.sequence_counter.shape == (1,)
    assert batch.frame == DEFAULT_FRAME


def test_ego_trajectory_sensor_selects_halo_poses_from_a_clip(tmp_path: Path) -> None:
    """Predecessor and successor clip halos remain nearest-pose candidates."""
    path = tmp_path / "clip_egotrajectory.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(
                log_time_ns=90,
                data=_reference_payload(90, sequence_counter=9, position_m=(9.0, 0.0, 0.0)),
            ),
            McapSample(
                log_time_ns=140,
                data=_reference_payload(140, sequence_counter=14, position_m=(14.0, 0.0, 0.0)),
            ),
            McapSample(
                log_time_ns=200,
                data=_reference_payload(200, sequence_counter=20, position_m=(20.0, 0.0, 0.0)),
            ),
        ],
    )
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 125, 150, 175, 200], dtype=np.int64),
            stride_ns=100,
            duration_ns=100,
        )
    )

    [batch] = list(_reference_sensor(path).sample(spec, policy=NearestTimestampPolicy()))

    np.testing.assert_array_equal(batch.align_timestamps_ns, [100, 125, 150, 175])
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, [90, 140, 140, 200])
    np.testing.assert_array_equal(batch.positions_m[:, 0], [9.0, 14.0, 14.0, 20.0])
    assert batch.sequence_counter is not None
    np.testing.assert_array_equal(batch.sequence_counter, [9, 14, 14, 20])


def test_ego_trajectory_sensor_selects_from_the_clip_using_alignment_time(tmp_path: Path) -> None:
    """Windows request rows; MCAP log time does not filter nearest candidates."""
    path = tmp_path / "windowed.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(200, sequence_counter=2)),
            McapSample(log_time_ns=300, data=_reference_payload(300, sequence_counter=3)),
        ],
    )

    batch = _nearest_sample(_reference_sensor(path), one_window_spec(150, 250))

    np.testing.assert_array_equal(batch.align_timestamps_ns, np.array([150], dtype=np.int64))
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(batch.sequence_counter, np.array([1], dtype=np.uint64))


def test_ego_trajectory_sensor_routes_nearest_policy_through_sensor_group(tmp_path: Path) -> None:
    """SensorGroup receives one aligned ego pose for every requested row."""
    path = tmp_path / "group_aligned.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(200, sequence_counter=2)),
        ],
    )
    sensor = _reference_sensor(path)

    assert sensor.supports_sampling_policy(NearestTimestampPolicy())
    assert not sensor.supports_sampling_policy(NoSamplingPolicy())
    frame = next(
        SensorGroup({"ego": sensor}).sample(
            one_window_spec(150, 250),
            policies={"ego": NearestTimestampPolicy()},
        )
    )
    trajectory = frame.sensor_data["ego"]

    np.testing.assert_array_equal(trajectory.align_timestamps_ns, [150])
    np.testing.assert_array_equal(trajectory.sensor_timestamps_ns, [100])

    with pytest.raises(TypeError, match="EgoTrajectorySensor requires NearestTimestampPolicy"):
        next(sensor.sample(one_window_spec(150, 250), policy=None))  # type: ignore[arg-type]


def test_ego_trajectory_sensor_propagates_nearest_tolerance_errors_through_the_group(tmp_path: Path) -> None:
    """SensorGroup attributes an ego nearest-tolerance failure to its configured sensor."""
    path = tmp_path / "group_tolerance.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])
    sensor = _reference_sensor(path)

    with pytest.raises(AlignmentError) as error_info:
        next(
            SensorGroup({"ego": sensor}).sample(
                one_window_spec(200, 300),
                policies={"ego": NearestTimestampPolicy(max_delta_ns=50)},
            )
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.EGO_TRAJECTORY_MAX_DELTA_EXCEEDED
    assert error.sensor_id == "ego"
    assert error.delta_ns == 100
    assert error.max_delta_ns == 50
    np.testing.assert_array_equal(error.align_timestamps_ns, [200])
    np.testing.assert_array_equal(error.sensor_timestamps_ns, [100])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [100])


def test_ego_trajectory_sensor_reports_an_empty_topic_as_an_alignment_error(tmp_path: Path) -> None:
    """A configured but empty ego topic is unsatisfiable data, not a sensor defect."""
    path = tmp_path / "empty_egotrajectory.mcap"
    _write_reference_mcap(path, [])
    sensor = _reference_sensor(path)

    with pytest.raises(AlignmentError) as error_info:
        next(
            SensorGroup({"ego": sensor}).sample(
                one_window_spec(100, 200),
                policies={"ego": NearestTimestampPolicy()},
            )
        )

    error = error_info.value
    assert error.reason is AlignmentFailureReason.EGO_TRAJECTORY_NO_ELIGIBLE_POSE
    assert error.sensor_id == "ego"
    assert error.max_delta_ns is None
    np.testing.assert_array_equal(error.align_timestamps_ns, [100])
    np.testing.assert_array_equal(error.known_source_timestamps_ns, [])


def test_ego_trajectory_sensor_preserves_shapes_for_an_empty_raw_read_range(tmp_path: Path) -> None:
    """An empty half-open MCAP range is a valid shape-preserving source batch."""
    path = tmp_path / "empty_read_range.mcap"
    _write_reference_mcap(path, [McapSample(log_time_ns=100, data=_reference_payload(100))])
    sensor = _reference_sensor(path)

    with sensor._mcap.open_reader() as reader:
        sensor._resolve_message_class(reader)
        batch = sensor._read_messages(reader, 200, 300)

    assert batch.positions_m.shape == (0, 3)
    assert batch.orientations_xyzw.shape == (0, 4)
    assert batch.host_timestamps_ns is not None
    assert batch.host_timestamps_ns.shape == (0,)
    assert batch.sequence_counter is not None
    assert batch.sequence_counter.shape == (0,)


def test_ego_trajectory_sensor_reuses_one_decoded_source_across_multiple_windows(
    tmp_path: Path,
) -> None:
    """Each output window aligns against the complete clip source."""
    half_sqrt = float(np.sqrt(0.5))
    path = tmp_path / "multiple_windows.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=90, data=_reference_payload(90, sequence_counter=9, position_m=(9.0, 0.0, 0.0))),
            McapSample(log_time_ns=140, data=_reference_payload(140, sequence_counter=14, position_m=(14.0, 0.0, 0.0))),
            McapSample(
                log_time_ns=220,
                data=_reference_payload(
                    220,
                    sequence_counter=22,
                    position_m=(22.0, 0.0, 0.0),
                    orientation_xyzw=(0.0, 0.0, half_sqrt, half_sqrt),
                ),
            ),
        ],
    )
    sensor = _reference_sensor(path)
    spec = SamplingSpec(
        grid=make_sampling_grid(
            timestamps_ns=np.array([100, 150, 200, 250, 300], dtype=np.int64),
            stride_ns=100,
            duration_ns=100,
        )
    )
    first, second = list(sensor.sample(spec, policy=NearestTimestampPolicy()))

    np.testing.assert_array_equal(first.align_timestamps_ns, [100, 150])
    np.testing.assert_array_equal(first.sensor_timestamps_ns, [90, 140])
    np.testing.assert_array_equal(first.positions_m[:, 0], [9.0, 14.0])
    np.testing.assert_array_equal(first.sequence_counter, [9, 14])
    np.testing.assert_array_equal(second.align_timestamps_ns, [200, 250])
    np.testing.assert_array_equal(second.sensor_timestamps_ns, [220, 220])
    np.testing.assert_array_equal(second.positions_m[:, 0], [22.0, 22.0])
    np.testing.assert_allclose(second.orientations_xyzw, [[0.0, 0.0, half_sqrt, half_sqrt]] * 2)
    np.testing.assert_array_equal(second.sequence_counter, [22, 22])


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


def test_ego_trajectory_alignment_is_invariant_to_internal_mcap_chunk_boundaries(tmp_path: Path) -> None:
    """Equivalent clip records align identically regardless of how MCAP chunks them."""
    samples = [
        McapSample(log_time_ns=100, data=_reference_payload(1_100, sequence_counter=10, position_m=(1.0, 2.0, 3.0))),
        McapSample(log_time_ns=200, data=_reference_payload(1_200, sequence_counter=20, position_m=(4.0, 5.0, 6.0))),
        McapSample(log_time_ns=400, data=_reference_payload(1_400, sequence_counter=40, position_m=(7.0, 8.0, 9.0))),
    ]
    small_chunks_path = tmp_path / "small_chunks.mcap"
    large_chunks_path = tmp_path / "large_chunks.mcap"
    _write_reference_mcap(small_chunks_path, samples, chunk_size=1)
    _write_reference_mcap(large_chunks_path, samples, chunk_size=1_048_576)

    with small_chunks_path.open("rb") as stream:
        small_chunks_summary = make_reader(stream).get_summary()
    with large_chunks_path.open("rb") as stream:
        large_chunks_summary = make_reader(stream).get_summary()
    assert small_chunks_summary is not None
    assert large_chunks_summary is not None
    assert len(small_chunks_summary.chunk_indexes) > 1
    assert len(large_chunks_summary.chunk_indexes) == 1

    spec = one_window_spec(1_150, 1_250)
    policy = NearestTimestampPolicy(max_delta_ns=100)
    small_chunks = next(_reference_sensor(small_chunks_path).sample(spec, policy=policy))
    large_chunks = next(_reference_sensor(large_chunks_path).sample(spec, policy=policy))

    np.testing.assert_array_equal(small_chunks.align_timestamps_ns, large_chunks.align_timestamps_ns)
    np.testing.assert_array_equal(small_chunks.sensor_timestamps_ns, large_chunks.sensor_timestamps_ns)
    np.testing.assert_array_equal(small_chunks.positions_m, large_chunks.positions_m)
    np.testing.assert_array_equal(small_chunks.orientations_xyzw, large_chunks.orientations_xyzw)
    np.testing.assert_array_equal(small_chunks.pose_valid, large_chunks.pose_valid)
    assert small_chunks.host_timestamps_ns is not None
    assert large_chunks.host_timestamps_ns is not None
    np.testing.assert_array_equal(small_chunks.host_timestamps_ns, large_chunks.host_timestamps_ns)
    assert small_chunks.sequence_counter is not None
    assert large_chunks.sequence_counter is not None
    np.testing.assert_array_equal(small_chunks.sequence_counter, large_chunks.sequence_counter)


def test_ego_trajectory_alignment_uses_log_time_order_across_physically_reordered_chunks(tmp_path: Path) -> None:
    """Physical MCAP write order does not change mapped-time nearest selection."""
    path = tmp_path / "physically_reordered.mcap"
    _write_reference_mcap(
        path,
        [
            McapSample(log_time_ns=400, data=_reference_payload(400, sequence_counter=4)),
            McapSample(log_time_ns=100, data=_reference_payload(100, sequence_counter=1)),
            McapSample(log_time_ns=200, data=_reference_payload(200, sequence_counter=2)),
        ],
        chunk_size=1,
    )

    batch = _nearest_sample(_reference_sensor(path), one_window_spec(150, 250))

    np.testing.assert_array_equal(batch.align_timestamps_ns, [150])
    np.testing.assert_array_equal(batch.sensor_timestamps_ns, [100])
    assert batch.sequence_counter is not None
    np.testing.assert_array_equal(batch.sequence_counter, [1])


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
        _nearest_sample(_reference_sensor(path), one_window_spec(100, 300))


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

    first = _nearest_sample(sensor, one_window_spec(100, 200))
    second = _nearest_sample(sensor, one_window_spec(200, 300))
    for batch in (first, second):
        assert batch.host_timestamps_ns is None
        assert batch.sequence_counter is None
    np.testing.assert_array_equal(first.sensor_timestamps_ns, np.array([100], dtype=np.int64))
    np.testing.assert_array_equal(second.align_timestamps_ns, np.array([200], dtype=np.int64))
    np.testing.assert_array_equal(second.sensor_timestamps_ns, np.array([100], dtype=np.int64))


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
        _nearest_sample(_reference_sensor(path), one_window_spec(100, 200))


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
        _nearest_sample(_reference_sensor(path), one_window_spec(100, 200))
