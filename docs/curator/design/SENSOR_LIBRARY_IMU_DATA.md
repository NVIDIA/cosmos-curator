# Sensor Library IMU Data Design

This note documents the design rationale for the first generic IMU data
structure in the Cosmos Curator Sensor Library. `ImuData` is implemented in
`cosmos_curator/core/sensors/data/imu_data.py`. This note does not define
`ImuSensor`, IMU parsing, preintegration, bias correction, calibration, or
coordinate transforms.

## Implemented Data Model

`ImuData` is a structure-of-arrays batch of IMU point samples. Each row
represents one aligned output sample and satisfies the existing `SensorData`
protocol:

- `align_timestamps_ns` is the reference timeline requested from the
  `SamplingGrid`.
- `sensor_timestamps_ns` is the source IMU sample time actually used for each
  row.
- vector payloads use shape `(N, 3)`.
- optional scalar payloads and row-level validity masks use shape `(N,)`.
- optional per-axis validity masks use shape `(N, 3)`.
- optional covariance payloads use shape `(N, 3, 3)`.

This lets `ImuData` represent decoded source samples when the reference grid is
the IMU's own timestamp stream
(`align_timestamps_ns == sensor_timestamps_ns`) and aligned point samples when
an IMU sensor samples onto a camera, lidar, or arbitrary reference grid.

In this note, "decoded source samples" means parsed IMU messages converted to
Sensor Library field names, NumPy arrays, and SI units. It does not mean raw
vendor bytes, CAN packets, MCAP payloads, or ADC counts.

Do not make the first `ImuData` a preintegrated-window structure;
preintegration has interval semantics, depends on bias and frame conventions,
and should be a separate derived structure.

## Existing Sensor Library Conventions

The current Sensor Library establishes these conventions:

- `SensorData` requires `align_timestamps_ns` and `sensor_timestamps_ns`
  ([SensorData](#ref-sensor-data)).
- `AlignedFrame` requires every payload's `align_timestamps_ns` to exactly
  match the frame's reference timeline
  ([AlignedFrame](#ref-aligned-frame)).
- `CameraData` and `VideoIndex` use attrs classes, structure-of-arrays layout,
  read-only NumPy views, and explicit shape/dtype validators
  ([CameraData](#ref-camera-data), [VideoIndex](#ref-video-index)).
- public timestamp arrays use `np.int64` nanoseconds.
- boolean masks use `np.bool_`.

`ImuData` should follow these conventions rather than introducing a
sensor-specific batch model.

## Reference Models Reviewed

### DriveWorks IMU

Public DriveWorks 6.0.9 IMU documentation describes decode/read APIs for IMU
frames and a public `dwIMUFrame` structure
([DriveWorks IMU](#ref-driveworks-imu), [DriveWorks IMU types](#ref-driveworks-imu-types)).
The `IMUTypes.h` reference also lists `dwIMUFrameNew`, but marks it deprecated
in favor of `dwIMUFrame`; `dwIMUFrame.timestamp_us` is also deprecated in
favor of the split `hostTimestamp` and `sensorTimestamp` fields
([DriveWorks IMU types](#ref-driveworks-imu-types)).

Across the DriveWorks fields, the reusable generic signals are:

- angular rate / gyroscope vector in rad/s
- acceleration vector in m/s^2
- optional orientation in roll/pitch/yaw and quaternion form
- optional heading and magnetometer fields
- per-signal validity metadata indicating which fields are valid
- distinct source sensor and host timestamps in the current `dwIMUFrame`

This supports keeping angular velocity and linear acceleration as the required
generic fields, treating orientation and magnetometer as optional, and
representing validity explicitly rather than relying only on sentinel values.
It also supports keeping `sensor_timestamps_ns` as the canonical measurement
time and `host_timestamps_ns` as optional receive-time metadata, instead of
building the generic API around DriveWorks-only timestamp/status fields.

### ROS IMU Conventions

ROS `sensor_msgs/Imu` groups orientation, angular velocity, and linear
acceleration with one 3x3 covariance matrix for each measurement family
([ROS Imu](#ref-ros-imu)). ROS also standardizes SI units and coordinate-frame
conventions, including rad/s for angular velocity and m/s^2 for linear
acceleration ([REP 103](#ref-ros-rep-103)). REP 145 recommends reporting
accelerometer, gyroscope, magnetometer, and orientation data in one consistent
sensor frame, and leaving downstream consumers to do additional transformations
([REP 145](#ref-ros-rep-145)).

This supports a generic `ImuData` that standardizes units and shape while
deferring extrinsics and frame transforms to calibration/extrinsics layers.

### PX4 SensorCombined

PX4's `SensorCombined` message keeps gyro and accelerometer data as SI-unit
vectors, records timing for gyro and accelerometer samples, and carries
clipping/calibration counters ([PX4 SensorCombined](#ref-px4-sensor-combined)).
This supports keeping timing and status fields available without promoting
every source-specific counter into the required generic core.

## Implemented `ImuData`

```python
@attrs.define(hash=False, frozen=True)
class ImuData:
    __hash__ = None

    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    angular_velocity_rad_s: npt.NDArray[np.float64]
    linear_acceleration_m_s2: npt.NDArray[np.float64]

    orientation_quat_xyzw: npt.NDArray[np.float64] | None = None
    angular_velocity_covariance: npt.NDArray[np.float64] | None = None
    linear_acceleration_covariance: npt.NDArray[np.float64] | None = None
    orientation_covariance: npt.NDArray[np.float64] | None = None

    angular_velocity_valid: npt.NDArray[np.bool_] | None = None
    linear_acceleration_valid: npt.NDArray[np.bool_] | None = None
    orientation_valid: npt.NDArray[np.bool_] | None = None

    host_timestamps_ns: npt.NDArray[np.int64] | None = None
    sequence_counter: npt.NDArray[np.uint64] | None = None
    temperature_c: npt.NDArray[np.float64] | None = None
```

### Required Fields

| Field | dtype | shape | unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `align_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Reference timestamps requested from the `SamplingGrid`; must be strictly increasing. |
| `sensor_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Source IMU sample timestamps selected for each row; may differ from alignment times and may repeat under supersampling. |
| `angular_velocity_rad_s` | `np.float64` | `(N, 3)` | rad/s | yes | Gyroscope measurement in x/y/z axis order. |
| `linear_acceleration_m_s2` | `np.float64` | `(N, 3)` | m/s^2 | yes | Accelerometer proper acceleration in x/y/z axis order, including gravity unless a parser explicitly documents otherwise. |

### Optional Generic Fields

| Field | dtype | shape | unit | Notes |
| --- | --- | --- | --- | --- |
| `orientation_quat_xyzw` | `np.float64` | `(N, 4)` | unitless | Optional fused orientation estimate as quaternion x/y/z/w. |
| `angular_velocity_covariance` | `np.float64` | `(N, 3, 3)` | `(rad/s)^2` | Optional covariance for gyroscope rows. |
| `linear_acceleration_covariance` | `np.float64` | `(N, 3, 3)` | `(m/s^2)^2` | Optional covariance for accelerometer rows. |
| `orientation_covariance` | `np.float64` | `(N, 3, 3)` | rad^2 | Optional covariance for orientation, expressed about x/y/z axes. |
| `angular_velocity_valid` | `np.bool_` | `(N, 3)` | unitless | Optional per-axis validity mask. |
| `linear_acceleration_valid` | `np.bool_` | `(N, 3)` | unitless | Optional per-axis validity mask. |
| `orientation_valid` | `np.bool_` | `(N,)` | unitless | Optional row-level orientation validity. |
| `host_timestamps_ns` | `np.int64` | `(N,)` | ns | Optional host/container receive timestamps, distinct from source sensor measurement timestamps. |
| `sequence_counter` | `np.uint64` | `(N,)` | unitless | Optional source sequence counter when available. |
| `temperature_c` | `np.float64` | `(N,)` | deg C | Optional IMU temperature. |

### Timestamp Semantics

`align_timestamps_ns` is the reference timeline requested by the
`SamplingGrid`. It is the timestamp downstream aligned-frame consumers use to
compare rows across sensors.

`sensor_timestamps_ns` is the selected source measurement timestamp for each
output row. For a DriveWorks parser, prefer a valid `dwIMUFrame.sensorTimestamp`
when available and convert it to public `np.int64` nanoseconds. If a source API
only exposes a single frame timestamp, the parser may map that timestamp to
`sensor_timestamps_ns` but should document the source meaning.

`host_timestamps_ns` is optional receive-time metadata. For DriveWorks, it maps
to `dwIMUFrame.hostTimestamp` when that signal is valid. It should not replace
`sensor_timestamps_ns` for alignment unless a parser explicitly documents that
the source has no separate sensor measurement time.

Keep timestamp quality, time-sync status, timestamp-format enums, and the
original units/source field names in parser-specific metadata unless multiple
parsers expose compatible semantics and downstream code needs them generically.

### Parser-Specific Metadata

Keep these out of the required generic `ImuData` core unless multiple parser
implementations expose the field with compatible semantics and downstream code
needs to consume it generically:

- vendor quality/status enums, alignment status, time-sync status, and
  diagnostic error IDs
- clipping bits
- calibration counters
- timestamp quality/format metadata
- magnetometer, heading, and GNSS/INS fused navigation outputs
- offsets, bias estimates, and bias covariance
- angular acceleration / gyroscope acceleration

Parsers can preserve these in parser-specific result types or sidecar metadata
until that compatibility and downstream need are clear.

## Validity Representation

Use the proposed boolean validity-mask fields instead of NaNs for generic
validity. These fields are `angular_velocity_valid`,
`linear_acceleration_valid`, and `orientation_valid`. Required vector arrays
should have concrete finite values for rows marked valid. Optional measurements
should be `None` when unavailable for the whole batch. If a field is available
but only some rows or axes are valid, include the corresponding validity mask.
Do not add generic status arrays in the first `ImuData`; keep source-specific
quality/status values in parser-specific metadata until a generic consumer
needs them.

This avoids overloading floating-point values with data-quality semantics and
matches DriveWorks' explicit validity model more closely than NaN-only
handling. It also keeps compatibility with ROS-style covariance conventions:
unknown covariance can be represented by `None`, and known covariance by an
explicit `(N, 3, 3)` array.

## Coordinate Frames

`ImuData` should not silently transform coordinate frames. The vector fields
are in x/y/z axis order for the IMU sensor frame represented by the sensor
object and its calibration/extrinsics metadata. The first implementation
should document the parser's source frame and whether any normalization was
performed.

Recommended defaults:

- use right-handed frames at the Sensor Library API boundary
- express angular velocity and linear acceleration in the IMU sensor frame
- keep orientation optional because not every IMU reports a fused orientation
- defer frame transforms, vehicle-body transforms, and camera/IMU extrinsics
  to the calibration/extrinsics layer

## Validation Constraints

The reviewed DriveWorks, ROS, and PX4 references support validation
constraints, not universal physical min/max ranges. DriveWorks leaves numeric
min/max unspecified for core IMU values, while ROS `sensor_msgs/Imu` and PX4
`SensorCombined` specify SI units and measurement semantics rather than hard
bounds ([DriveWorks IMU types](#ref-driveworks-imu-types),
[ROS Imu](#ref-ros-imu), [PX4 SensorCombined](#ref-px4-sensor-combined)).

Timestamp dtype and ordering constraints are covered in the required-field
table. Non-timestamp fields should use these constraints:

| Field | Constraint |
| --- | --- |
| all payload arrays | Same leading length `N` as `align_timestamps_ns`. |
| `angular_velocity_rad_s` | `np.float64`, shape `(N, 3)`, finite values; no generic min/max. |
| `linear_acceleration_m_s2` | `np.float64`, shape `(N, 3)`, finite values; no generic min/max. |
| `orientation_quat_xyzw` | Optional `np.float64`, shape `(N, 4)`, finite values, nonzero norms; prefer unit norm within tolerance without silently normalizing caller data. |
| `angular_velocity_covariance` | Optional `np.float64`, shape `(N, 3, 3)`, finite values; should be symmetric positive semidefinite within tolerance when present. |
| `linear_acceleration_covariance` | Optional `np.float64`, shape `(N, 3, 3)`, finite values; should be symmetric positive semidefinite within tolerance when present. |
| `orientation_covariance` | Optional `np.float64`, shape `(N, 3, 3)`, finite values; should be symmetric positive semidefinite within tolerance when present. |
| `angular_velocity_valid` | Optional `np.bool_`, shape `(N, 3)`. |
| `linear_acceleration_valid` | Optional `np.bool_`, shape `(N, 3)`. |
| `orientation_valid` | Optional `np.bool_`, shape `(N,)`. |
| `sequence_counter` | Optional `np.uint64`, shape `(N,)`; no generic monotonicity rule because wrap, gaps, and reset behavior are source-specific. |
| `temperature_c` | Optional `np.float64`, shape `(N,)`, finite values; no generic min/max. |

ROS covariance sentinels should be normalized at parser boundaries: all-zero
covariance means unknown covariance, and a first covariance element of `-1`
means the corresponding estimate is unavailable
([ROS Imu](#ref-ros-imu)). In `ImuData`, represent unknown covariance as
`None` and unavailable estimates with `None` fields or validity masks rather
than preserving ROS sentinel values inside covariance arrays.

Follow the existing pattern from `CameraData`: attach shared batch-length
validation after all required fields have been set, and expose read-only views
without mutating caller-owned arrays.

## Related Structures

Do not add a separate undecoded IMU payload type immediately. If one is needed
later, avoid names such as `RawImuData` because "raw" is ambiguous, and be
careful with names such as `VendorImuData` because the payload may come from a
container format rather than directly from a vendor API. A more precise future
name would be something like `UndecodedImuPayload`: vendor bytes, CAN packets,
MCAP message payloads, ADC counts, or similar source payloads that have not yet
been parsed into SI-unit IMU measurements. Decoded source-sample streams can
use `ImuData` with identity alignment timestamps, while aligned streams can use
the same type with reference-grid timestamps.

Consider a separate `PreintegratedImuData` only when a consumer needs interval
integration. That type should not be a small extension of `ImuData`; it needs
fields such as:

- `start_timestamps_ns` and `exclusive_end_timestamps_ns`
- `delta_rotation_quat_xyzw`
- `delta_velocity_m_s`
- `delta_position_m`
- integration covariance
- bias values used for integration
- sample count and integration duration

Consider an `ImuWindowData` or ragged source-sample window structure only if
downstream pipelines need all high-rate decoded IMU source samples attached to
each lower-rate alignment row.

## Implementation Status

The implementation adds:

1. `cosmos_curator/core/sensors/data/imu_data.py`, with an attrs-based
   `ImuData` class matching this design note.
2. Shared validation helpers in `cosmos_curator/core/sensors/utils/validation.py`
   for finite `float64` arrays and 1-D `uint64` arrays.
3. Tests under `tests/cosmos_curator/core/sensors/data/test_imu_data.py` and
   `tests/cosmos_curator/core/sensors/utils/test_validation.py`.

DriveWorks 6.0.9 documentation was reviewed as part of this design. The
implementation keeps `sensor_timestamps_ns` as the canonical selected source
measurement timestamp, maps DriveWorks `hostTimestamp` to optional
`host_timestamps_ns` when valid, and keeps DriveWorks quality/status,
alignment, and time-sync fields parser-specific unless a generic downstream
consumer needs them. DriveWorks, ROS, and PX4 references support validation
constraints, not universal physical min/max ranges; `ImuData` validates dtype,
shape, finite-value, quaternion-norm, covariance, and validity-mask constraints
for non-timestamp fields, but does not add generic physical min/max bounds for
angular velocity, linear acceleration, or temperature.

## Open Follow-Up Questions

- Should parser-specific status fields such as DriveWorks quality, alignment,
  and time-sync values live in sidecar metadata, a typed optional status
  structure, or a future per-parser subclass?
- Should the first IMU sensor implementation use nearest-neighbor, previous,
  interpolation, or interval aggregation when sampling onto a lower-rate
  reference grid?
- Do downstream consumers need all decoded IMU source samples per aligned
  camera/lidar row soon enough to justify an `ImuWindowData` structure in the
  future implementation phase?

## References

- <a id="ref-sensor-data"></a>`SensorData` protocol:
  `cosmos_curator/core/sensors/data/sensor_data.py`
- <a id="ref-aligned-frame"></a>`AlignedFrame`:
  `cosmos_curator/core/sensors/data/aligned_frame.py`
- <a id="ref-camera-data"></a>`CameraData`:
  `cosmos_curator/core/sensors/data/camera_data.py`
- <a id="ref-video-index"></a>`VideoIndex`:
  `cosmos_curator/core/sensors/data/video.py`
- <a id="ref-validation-helpers"></a>Sensor validation helpers:
  `cosmos_curator/core/sensors/utils/validation.py`
- <a id="ref-driveworks-imu"></a>DriveWorks SDK Reference: IMU Sensor
  (Drive OS 6.0.9, DriveWorks 5.16.65):
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/group__imu__group.html>
- <a id="ref-driveworks-imu-types"></a>DriveWorks SDK Reference:
  `IMUTypes.h` (Drive OS 6.0.9, DriveWorks 5.16.65):
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/IMUTypes_8h.html>
- <a id="ref-ros-imu"></a>ROS 2 Jazzy `sensor_msgs/msg/Imu`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/Imu.html>
- <a id="ref-ros-rep-103"></a>ROS REP 103: Standard Units of Measure and
  Coordinate Conventions: <https://ros.org/reps/rep-0103.html>
- <a id="ref-ros-rep-145"></a>ROS REP 145: Conventions for IMU Sensor
  Drivers: <https://www.ros.org/reps/rep-0145.html>
- <a id="ref-px4-sensor-combined"></a>PX4 `SensorCombined`:
  <https://docs.ros.org/en/rolling/p/px4_msgs/msg/SensorCombined.html>
