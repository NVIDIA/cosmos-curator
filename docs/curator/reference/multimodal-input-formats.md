<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Multimodal input formats accepted by readers and pipelines

This reference describes the input formats and measurement contracts accepted by Curator's
sensor readers and multimodal recipes. Use it to choose a compatible format and check
timestamps, units, coordinate frames, and measurement validity before processing a recording.

Reader support and recipe support differ: a format that a sensor reader accepts may not
be accepted by a particular recipe. The compatibility table below identifies these differences.
For execution instructions, see the [multimodal AV end-to-end guide](../guides/multimodal-av-e2e.md).

## Supported inputs at a glance

The multimodal split recipe creates clips; multimodal captioning reads the published episodes.

| Input | Sensor-reader support | Multimodal split | Multimodal captioning |
| --- | --- | --- | --- |
| Camera/video | MP4 and related video containers; limited MKV support | Direct video files | Reads episode MP4s |
| Images | JPEG, PNG, and other Pillow-supported images | Not accepted | Not accepted |
| IMU | ImuSample protobuf in MCAP | Copies messages into MCAP clips | Not used for ego context |
| GPS | LocationFix protobuf in MCAP | Requires LocationFix; aligns positions to clip times and writes new messages | Not used for ego context |
| Egotrajectory | PoseInFrame protobuf in MCAP | Copies messages into MCAP clips | Requires PoseInFrame for ego context |
| LiDAR | Undecoded packets in MCAP with a compatible decoder | Not accepted | Not accepted |

## Shared input requirements

### Containers, encodings, and schemas

A **container** stores messages or media; an **encoding** specifies how payload bytes are
serialized; a **schema** describes message fields. For example: MCAP container → protobuf
encoding → LocationFix schema. A topic name is configurable and does not identify its format.

IMU, GPS, and egotrajectory inputs use protobuf messages in MCAP. A mapper adapts message
fields and units to the reader; including a schema alone does not establish compatible
measurement semantics. Reader and recipe requirements must both be satisfied.

### Clocks and alignment

An **alignment timestamp** is the clock value used to compare or sample measurements.
MCAP `log_time` and `publish_time` are nanosecond-valued envelope timestamps. Mapped payload
timestamps can use nanoseconds, converted scalar units, or `google.protobuf.Timestamp`.
Declare each field's clock domain and origin: nanosecond units or a timestamp field name
do not establish Unix UTC or synchronization. Keep host-arrival timestamps separate from
sensor timestamps when the producer provides both.

The protobuf readers map `align_timestamp_ns` into the Data batch's `align_timestamps_ns`
array. If omitted, it defaults to mapped `sensor_timestamp_ns`. The mapper also accepts
`$mcap.logtime` for explicit envelope-time alignment. Decoded batches require strictly
increasing alignment timestamps; a mapper does not reconcile clocks or repair duplicates.

The protobuf readers' `start_ns`, `end_ns`, and `timestamps_ns` describe MCAP `log_time`,
even when their mapped alignment timestamps use another clock. Sampling behavior differs
by modality, as described below.

### Units, coordinate frames, and validity

Supply measurements in the units and coordinate frames required by the selected reader.
Readers do not apply rig transforms. Protobuf defaults are not proof of valid measurements:
choose a mapping that preserves available validity masks, and use mappings that default
validity to true only for valid measurements. Fields retained in an MCAP are not necessarily
exposed by the mapping or returned Data class.

## Requirements by modality

### Cameras and images

- **Accepted format:** direct MP4/related video media; MKV has limited testing and MPEG-TS
  is unsupported by the camera reader. Image sequences use Pillow-supported formats.
- **Required measurements:** video frames or image pixels. Camera and image readers return
  decoded RGB pixels in their Data objects.
- **Timestamp behavior:** `CameraSensor` uses presentation timestamps (PTS) by default;
  caller-supplied sensor timestamps replace that timeline, while `origin_ns` shifts its
  origin. `StreamingCameraSensor` uses caller-supplied per-frame timestamps; its `from_index()`
  constructor reports container decode timestamps (DTS) when sensor timestamps are omitted.
  `ImageSensor` accepts per-image sensor timestamps; without them it uses sequential indices.
  Container timestamps and image indices do not establish sensor measurement time.
- **Limitations:** prefer video without B-frames or frame reordering, which adds buffering
  and frame dependencies. `CameraSensor` supports reordered video through presentation-order
  indexing. A streaming reader needs video data through `source` to decode frames; building
  it from an index without a source permits metadata-only use.

### IMU

- **Accepted format:** ImuSample protobuf messages in MCAP, with a compatible mapping.
- **Required measurements:** three-axis angular velocity in rad/s and linear acceleration
  in m/s², expressed in the IMU sensor frame. The full mapping reads per-axis validity,
  temperature, sequence, host time, and biases. The minimal mapping omits masks and biases,
  defaults measurement/temperature validity to true, and leaves bias arrays absent.
- **Timestamp behavior:** sampling selects messages by half-open MCAP `log_time` windows.
  Changing the mapped alignment clock does not change window membership.
- **Limitations:** the split recipe copies message bytes without decoding or validating
  measurements. Copies include the message immediately preceding each span when available.
  IMU data is not used for captioning ego context.

### GPS

- **Accepted format:** LocationFix protobuf messages in MCAP, as required by the multimodal
  split recipe.
- **Required measurements:** WGS-84 latitude/longitude in degrees and WGS-84 ellipsoidal
  height in metres.
- **Timestamp behavior:** sampling decodes the source and selects or interpolates using mapped
  alignment timestamps. Sampling grids must use that clock. The split recipe requires the
  LocationFix payload timestamp to equal MCAP `log_time`.
- **Limitations:** the LocationFix mapping reads position and timestamp, assumes every position
  component is valid, and omits covariance, heading, velocity, frame ID, and key/value metadata.
  It does not check receiver quality. GPS is not used for captioning ego context.

### Egotrajectory

- **Accepted format:** PoseInFrame protobuf messages in MCAP for the split-to-captioning workflow.
- **Required measurements:** three position values in metres and a unit quaternion in X/Y/Z/W
  order, representing the sensor-origin pose in a clip-local East-North-Up (ENU) world frame.
  PoseInFrame mappings default validity to true and omit host time/sequence; frame naming
  comes from reader configuration, not wire `frame_id`.
- **Timestamp behavior:** sampling selects poses using mapped alignment timestamps. The generic
  PoseInFrame mapping aligns on the payload timestamp; the captioning mapping retains that
  sensor timestamp but aligns on MCAP `log_time`. These agree only when the two timestamps agree.
- **Limitations:** the split recipe copies message bytes by `log_time`, including the preceding
  message when available, without enforcing the trajectory schema or validating poses.
  Captioning requires PoseInFrame, reads all decoded poses, and rejects invalid rows.
  Ego context requires sensor-timestamped episodes no longer than 10 seconds.
  A downstream reader must choose a compatible alignment clock.

### LiDAR

- **Accepted format:** undecoded packet bytes in MCAP with `schema_id=0` and decoder-specific
  encoding. Supply a compatible packet decoder and any required calibration.
- **Required measurements:** the decoder supplies point XYZ, intensity, ring, and sensor
  timestamps; the reader associates points with alignment rows. XYZ units and intensity scale
  are decoder-defined and passed through without normalization. XYZ must be finite.
  `reference_frame` defaults to `sensor` and must match the decoder's coordinates.
- **Timestamp behavior:** bounds and `timestamps_ns` use recorder `log_time`; decoded point
  timestamps, output alignment timestamps, and sampling windows use the decoder's sensor clock.
  Build windows in that clock or explicitly reconcile the clocks before mixing domains.
- **Limitations:** the reader does not select a vendor decoder, apply motion compensation, or
  populate per-point validity masks. Decoded sweeps require strictly increasing sweep timestamps
  and disjoint point-timestamp ranges; overlapping or touching ranges are rejected. The multimodal
  split and captioning recipes do not accept LiDAR input.

## Preparing and validating inputs

### Preparation checklist

Before producing MCAP inputs for a recipe:

- Choose a schema the recipe accepts: LocationFix for multimodal split GPS and PoseInFrame
  for captioning ego context.
- Express measurements in the units and coordinate frames expected by the reader. GPS
  interpolation requires WGS-84 ellipsoidal height; readers apply no geoid correction or rig transforms.
- Establish a shared alignment clock. Timestamp units alone do not establish synchronization.
- Preserve measurement validity. Use mappings that default validity to true only when every
  supplied measurement is valid.
- Bind protobuf channels to their schema descriptors and write MCAP summary/chunk indexes
  for recipe access.

### Validate a protobuf mapping

The [mapping validator](../../../cosmos_curator/core/sensors/scripts/validate_protobuf_mapping.py)
checks a YAML mapping against the protobuf descriptor embedded in an MCAP. For example,
run this from the repository root to check a LocationFix mapping:

```bash
pixi run -e default python -m \
  cosmos_curator.core.sensors.scripts.validate_protobuf_mapping \
  --target gps \
  --topic /gps \
  --mapping cosmos_curator/core/sensors/examples/foxglove_location_fix_protobuf_mapping.yaml \
  --mcap /data/gps.mcap
```

Replace the MCAP path and topic with your input. Supported targets are `gps`, `imu`, and
`egotrajectory`; select the corresponding mapping.

A passing result establishes structural mapping compatibility. The validator does not
decode measurement payloads or verify physical units, coordinate-frame meaning, timestamp
ordering, clock synchronization, or recipe compatibility.

## Reader and mapping reference

| Reader | Mapping files | Sampling / Data result |
| --- | --- | --- |
| [ImuSensor](../../../cosmos_curator/core/sensors/sensors/imu_sensor.py) | [Full ImuSample](../../../cosmos_curator/core/sensors/examples/imu_protobuf_mapping.yaml), [minimal ImuSample](../../../cosmos_curator/core/sensors/examples/imu_minimal_protobuf_mapping.yaml) | `NoSamplingPolicy` → `ImuData`; `read_all()` returns all decoded source measurements |
| [GpsSensor](../../../cosmos_curator/core/sensors/sensors/gps_sensor.py) | [LocationFix](../../../cosmos_curator/core/sensors/examples/foxglove_location_fix_protobuf_mapping.yaml) | `NearestTimestampPolicy` or `GpsInterpolationPolicy` → `GpsData`; `read_all()` does not resample |
| [EgoTrajectorySensor](../../../cosmos_curator/core/sensors/sensors/ego_trajectory_sensor.py) | [generic PoseInFrame](../../../cosmos_curator/core/sensors/examples/foxglove_pose_in_frame_egotrajectory_mapping.yaml), [captioning PoseInFrame](../../../cosmos_curator/next/recipes/multimodal_caption/ego_context_foxglove_pose_in_frame_mapping.yaml) | `NearestTimestampPolicy` → `EgoTrajectory`; the captioning consumer calls `read_all()` and rejects invalid pose rows |

Other protobuf schemas may work with a compatible descriptor and mapping; that reader
extension mechanism does not automatically add support to a recipe's selected format.

`PreintegratedImuSensor` wraps `ImuSensor` and returns `PreintegratedImuData` in memory.
It introduces neither a new serialized input schema nor an IMU conversion entry point.

### Protobuf and MCAP requirements

For protobuf IMU, GPS, and egotrajectory inputs:

- Both channel `message_encoding` and schema `encoding` must be `protobuf`.
- Schema data must contain a serialized `FileDescriptorSet`, including needed dependencies,
  and define the configured fully qualified `schema_name`.
- The selected channel must reference that schema, with a stable schema ID per topic.
  An unrelated descriptor elsewhere in the archive does not describe its messages.
- Configure a YAML/dict mapping into the reader's required fields. A mapper adapts protobuf
  fields and units to the reader; embedding a schema alone does not establish compatible semantics.

The [protobuf resolver](../../../cosmos_curator/core/sensors/utils/mcap.py) checks these
contracts. These protobuf readers can scan MCAP without summary indexes, but the splitter's
schema-backed topic writers require a summary and a referenced Schema record. Write
summary/chunk indexes for indexed recipe access.

### Schema names

| Message | Fully qualified schema name |
| --- | --- |
| [ImuSample](../../../cosmos_curator/core/sensors/schemas/imu.proto) | `cosmos_curator.sensors.imu.v1.ImuSample` |
| [LocationFix](https://github.com/foxglove/foxglove-sdk/blob/main/schemas/proto/foxglove/LocationFix.proto) | `foxglove.LocationFix` |
| [PoseInFrame](https://github.com/foxglove/foxglove-sdk/blob/main/schemas/proto/foxglove/PoseInFrame.proto) | `foxglove.PoseInFrame` |

LocationFix and PoseInFrame come from the public
[`foxglove-schemas-protobuf` schemas](https://github.com/foxglove/foxglove-sdk/tree/main/schemas/proto/foxglove)
(Python import: `foxglove_schemas_protobuf`). The schema provider's application is not required.
Default MCAP reader topics are `/imu`, `/gps`, `/egotrajectory`, and `/lidar`; configure them
to match the source channels.
