# Sensor Library IMU Data Design

This note documents the generic raw and preintegrated IMU data structures in
the Cosmos Curator Sensor Library. `ImuData` stores decoded point samples;
`PreintegratedImuData` stores derived interval deltas. The current
implementation also includes an MCAP-backed `ImuSensor`, an eager
preintegration engine, and a grid-aligned `PreintegratedImuSensor`. Calibration,
rig transforms, and egomotion state estimation remain separate concerns.

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

    angular_velocity_bias_rad_s: npt.NDArray[np.float64] | None = None
    linear_acceleration_bias_m_s2: npt.NDArray[np.float64] | None = None
    angular_velocity_bias_valid: npt.NDArray[np.bool_] | None = None
    linear_acceleration_bias_valid: npt.NDArray[np.bool_] | None = None

    host_timestamps_ns: npt.NDArray[np.int64] | None = None
    sequence_counter: npt.NDArray[np.uint64] | None = None
    temperature_c: npt.NDArray[np.float64] | None = None
    temperature_valid: npt.NDArray[np.bool_] | None = None
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
| `angular_velocity_bias_rad_s` | `np.float64` | `(N, 3)` | rad/s | Optional gyroscope bias estimate in x/y/z order. |
| `linear_acceleration_bias_m_s2` | `np.float64` | `(N, 3)` | m/s^2 | Optional accelerometer bias estimate in x/y/z order. |
| `angular_velocity_bias_valid` | `np.bool_` | `(N, 3)` | unitless | Per-axis gyroscope-bias validity; paired with the bias values. |
| `linear_acceleration_bias_valid` | `np.bool_` | `(N, 3)` | unitless | Per-axis accelerometer-bias validity; paired with the bias values. |
| `host_timestamps_ns` | `np.int64` | `(N,)` | ns | Optional host/container receive timestamps, distinct from source sensor measurement timestamps. |
| `sequence_counter` | `np.uint64` | `(N,)` | unitless | Optional source sequence counter when available. |
| `temperature_c` | `np.float64` | `(N,)` | deg C | Optional IMU temperature. |
| `temperature_valid` | `np.bool_` | `(N,)` | unitless | Optional row-level temperature validity; paired with `temperature_c`. |

### Protobuf YAML Mapping

`ImuSensor` requires a sensor-owned YAML mapping for MCAP protobuf schemas. The [`imu.proto` reference schema](../../../cosmos_curator/core/sensors/schemas/imu.proto) and [`imu_protobuf_mapping.yaml`](../../../cosmos_curator/core/sensors/examples/imu_protobuf_mapping.yaml) are an exact pair. Use the mapping unchanged for that message, or adapt its source paths and units, and provide it as a `Path` together with the customer topic and schema name. Sensor timestamp, angular velocity, and linear acceleration are required; omitted vector validity defaults to true. In YAML, the per-message destination is named `align_timestamp_ns` (singular); the sensor batches those values into `ImuData.align_timestamps_ns` (plural). It defaults to the singular `sensor_timestamp_ns` mapping, which similarly becomes `ImuData.sensor_timestamps_ns`, but an integration may explicitly map a stable host or MCAP clock when sensor timestamps repeat or are otherwise unsuitable for alignment.

A mapping includes `host_timestamps_ns` and `sequence_counter` only when
`host_timestamp_ns` and `sequence_counter` are mapped. Mapping `temperature_c`
includes both temperature arrays; if `temperature_valid` is omitted, its values
default to true. Omitting `temperature_c` leaves both temperature arrays as
`None`. Bias value mappings similarly control whether their value and validity
arrays are present, and omitted bias validity defaults to true. A bias validity
mapping without its paired value is rejected. Populated and empty windows use
the same optional-array presence.

```python
from pathlib import Path

sensor = ImuSensor(
    source,
    topic="/vehicle/imu",
    schema_name="vendor.ImuEnvelope",
    protobuf_mapping=Path("imu_protobuf_mapping.yaml"),
)
```

Before pipeline execution, smoke-test the mapping with the
[standalone protobuf mapping validator](sensor-library.md#validate-a-mapping-before-pipeline-execution).

### Timestamp Semantics

`align_timestamps_ns` is the reference timeline requested by the
`SamplingGrid`. It is the timestamp downstream aligned-frame consumers use to
compare rows across sensors.

`sensor_timestamps_ns` is the selected source measurement timestamp for each
output row. When a source distinguishes measurement time from receive time, a
parser should prefer the measurement timestamp and convert it to public
`np.int64` nanoseconds. If a source API only exposes a single frame timestamp,
the parser may map that timestamp to `sensor_timestamps_ns` but should document
the source meaning.

`host_timestamps_ns` is optional receive-time metadata. A parser should
populate it from the source's receive-time timestamp when the source exposes
one and marks it valid. It should not replace `sensor_timestamps_ns` for
alignment by default. An integration may explicitly choose a stable host clock
when the sensor clock repeats or is unsuitable, but must document that policy.

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
- source-specific bias quality codes, bias timestamps without defined estimate-time semantics, and bias covariance
- angular acceleration / gyroscope acceleration

Parsers can preserve these in parser-specific result types or sidecar metadata
until that compatibility and downstream need are clear.

## Validity Representation

Use boolean validity-mask fields instead of source-specific status codes for
generic validity. These fields include `angular_velocity_valid`,
`linear_acceleration_valid`, `orientation_valid`,
`angular_velocity_bias_valid`, and `linear_acceleration_bias_valid`. Required vector arrays
should have concrete finite values for rows marked valid. Optional measurements
should be `None` when unavailable for the whole batch. If a field is available
but only some rows or axes are valid, include the corresponding validity mask.
Do not add raw measurement-quality or timestamp-quality arrays to `ImuData`;
their integer encodings are source-specific. Keep those values in the source
protobuf or parser-specific metadata until a source-neutral quality contract
and explicit per-source translations are defined. Bias-estimate timestamps are
also deferred: a sample timestamp must not be relabeled as the estimate time of
a bias that may have been computed earlier.

Explicit boolean masks keep data-quality semantics out of the floating-point
values themselves, so a consumer never has to distinguish a real measurement
from a sentinel. It also keeps compatibility with ROS-style covariance
conventions: unknown covariance can be represented by `None`, and known
covariance by an explicit `(N, 3, 3)` array.

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

The reviewed ROS and PX4 references support validation constraints, not
universal physical min/max ranges. ROS `sensor_msgs/Imu` and PX4
`SensorCombined` specify SI units and measurement semantics rather than hard
bounds ([ROS Imu](#ref-ros-imu), [PX4 SensorCombined](#ref-px4-sensor-combined)).

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

## Undecoded IMU Data

Do not add a separate undecoded IMU payload type immediately. If one is needed
later, avoid names such as `RawImuData` because "raw" is ambiguous, and be
careful with names such as `VendorImuData` because the payload may come from a
container format rather than directly from a vendor API. A more precise future
name would be something like `UndecodedImuPayload`: vendor bytes, CAN packets,
MCAP message payloads, ADC counts, or similar source payloads that have not yet
been parsed into SI-unit IMU measurements. Decoded source-sample streams can
use `ImuData` with identity alignment timestamps, while aligned streams can use
the same type with reference-grid timestamps.

## Implemented Stateful Preintegration

`PreintegratedImuData` is a separate immutable structure-of-arrays contract
because interval deltas do not have point-sample semantics. For a complete
alignment grid, row zero is an invalid, zero-duration identity. Every later row
aligned to `t_i` represents the causal interval `[t_{i-1}, t_i)`. A sliced
window may begin with a valid row whose interval start precedes the first
alignment timestamp retained in that slice.

The public output includes:

- exact external-clock `align_interval_start_timestamps_ns` and
  `align_interval_end_timestamps_ns` bounds
- converted sensor-clock endpoints and physical `integration_duration_ns`
- `delta_rotation_quat_xyzw`, `delta_velocity_m_s`, and `delta_position_m`
- the gyroscope and accelerometer bias values used, with per-axis availability
- total, used, and rejected source-sample counts
- maximum observed source-sample gap
- interval validity and an `ImuIntegrationInvalidReason` bit mask

`preintegrate_imu(imu_data, align_timestamps_ns)` performs midpoint SO(3)
integration for the complete requested grid. The requested grid and raw
`ImuData.align_timestamps_ns` share the external reference-clock domain. Their
paired raw sensor timestamps convert each requested boundary into the IMU
clock. Measurement interpolation, segment `dt`, duration, bias averaging,
and maximum-gap reporting then use only that physical
sensor timeline. Adjacent intervals reuse a boundary measurement for
interpolation but do not duplicate elapsed time.

For each sensor-time segment, the engine averages endpoint angular velocity to
form the SO(3) rotation increment. It rotates the left acceleration with the
segment's starting orientation and the right acceleration with the ending
orientation, then averages those two vectors in the common interval-start
frame. Velocity and position use that conventional endpoint-frame midpoint
force.

Sensor endpoints are represented as integer nanoseconds. After align-to-sensor
conversion rounds an endpoint, measurement values, biases, and validity
recompute their interpolation fraction from that same rounded sensor
coordinate. This keeps every boundary payload consistent with the timestamp
used for physical integration.

Recording-wide preparation normalizes measurement validity, replaces invalid
bias axes with zero, and computes bias-corrected gyroscope and accelerometer
arrays once. Each interval then locates contiguous source slices and
interpolates only its two boundaries. Interior rows are assembled from NumPy
array slices rather than per-point Python interpolation.

Measurements and valid biases come entirely from `ImuData`. The correction
rule is `corrected = measurement - bias`. An unavailable bias axis uses zero
correction and remains marked unavailable in the output; unavailable bias does
not by itself invalidate an interval. Invalid measurements, missing boundary
support, or fewer than two usable supporting samples do invalidate an interval.
Raw sensor timestamps may repeat because `ImuData` permits a nondecreasing
sensor clock. A repeated timestamp invalidates only external-grid intervals
whose support contains the zero-duration segment, using
`NON_INCREASING_SENSOR_TIME`; later intervals resume when sensor time advances.
No external alignment row is dropped.

Deltas are expressed in the interval's starting IMU frame. Accelerometer values
remain specific force: preintegration does not apply a world gravity vector,
initial pose, initial velocity, or rig extrinsics. Those belong to a future
egomotion estimator.

`PreintegratedImuSensor` wraps an `ImuSensor`. On demand it decodes the MCAP
topic and prepares recording-wide validity, bias-availability, and
bias-corrected arrays once. It then preintegrates each nonempty output window
only when that window is emitted. `SamplingWindow` remains an output-batching
mechanism, not an IMU episode boundary.

The external pipeline may call `reset_pose()` to request a new episode. The
request is forward-only and survives empty output windows. The first row of the
next nonempty batch is the canonical invalid zero-duration identity row with
`FIRST_ALIGNMENT`; no valid interval crosses that caller-defined boundary. In
the absence of a reset, each later nonempty batch prepends the final emitted
alignment timestamp from the preceding batch for integration, then emits only
the current window's rows. Emitted nonempty alignment timestamps must therefore
be strictly increasing; overlapping or decreasing preintegrated output is
rejected even after `reset_pose()`, though other sensor types may use
overlapping windows.

Boundary interpolation may use decoded observations outside the output window
or caller-defined episode, while the integrated interval itself never crosses a
reset boundary.

The lifecycle is:

1. decodes the complete MCAP IMU topic once with `ImuSensor.read_all()`
2. prepares and caches recording-wide arrays once
3. preintegrates each emitted nonempty window from current episode state
4. yields rows whose `align_timestamps_ns` equal `window.timestamps_ns`

This stateful lifecycle makes the derived sensor compatible with `SensorGroup`,
avoids repeating high-rate decoding and preparation, and lets a pipeline choose
episode boundaries while it consumes output.

`read_all()` requires mapped `ImuData.align_timestamps_ns` to be globally
strictly increasing across the complete recording so external boundaries can
be mapped unambiguously. Its sensor timestamps remain nondecreasing; duplicate
sensor times are handled per interval as described above.

`PreintegratedImuSensor` produces derived rows exactly on the supplied grid.
`NearestTimestampPolicy` is not supported by preintegrated IMU sampling in the
current policy-routing contract, so it does not constrain the distance to the
raw rows used for interpolation. Consumers should use
`max_inter_sample_gap_ns` to assess source support until an
integration-specific gap policy is introduced.

```python
from cosmos_curator.core.sensors.sampling.policy import NoSamplingPolicy

raw_imu = ImuSensor(
    source,
    topic="/imu",
    schema_name="vendor.ImuMessage",
    protobuf_mapping=Path("imu_protobuf_mapping.yaml"),
)
imu = PreintegratedImuSensor(raw_imu)

for batch in imu.sample(spec, policy=NoSamplingPolicy()):
    consume_preintegrated_intervals(batch)
    if begins_new_pose_episode(batch):
        imu.reset_pose()
```

## Future Windowed IMU Data

Consider an `ImuWindowData` or ragged source-sample window structure only if
downstream pipelines need all high-rate decoded IMU source samples attached to
each lower-rate alignment row.

## Implementation Status

The implementation includes:

1. `cosmos_curator/core/sensors/data/imu_data.py`, with an attrs-based
   `ImuData` class matching this design note.
2. `cosmos_curator/core/sensors/data/preintegrated_imu_data.py`, with the
   interval-delta, quality, and bias contract.
3. `cosmos_curator/core/sensors/preintegration/imu_preintegrator.py`, with
   recording-wide preparation, boundary interpolation, contiguous slicing, and
   midpoint SO(3) integration.
4. `cosmos_curator/core/sensors/sensors/preintegrated_imu_sensor.py`, with
   full-recording decode/integration caching and exact window slicing.
5. Shared batch-shape, dtype, quaternion, and covariance validation helpers in
   `cosmos_curator/core/sensors/utils/validation.py`.
6. Contract, analytic integration, boundary, bias, covariance, eager-cache,
   generic MCAP, and `SensorGroup` tests under
   `tests/cosmos_curator/core/sensors/`.

The implementation keeps `sensor_timestamps_ns` as the canonical selected
source measurement timestamp, maps a source's receive-time timestamp to
optional `host_timestamps_ns` when valid, and keeps source quality/status,
alignment, and time-sync fields parser-specific unless a generic downstream
consumer needs them. Validation is constraint-based rather than range-based:
`ImuData` validates dtype, shape, finite-value, quaternion-norm, covariance,
and validity-mask constraints for non-timestamp fields, but does not add
generic physical min/max bounds for angular velocity, linear acceleration, or
temperature.

## Open Follow-Up Questions

- Should parser-specific status fields such as sample quality, alignment,
  and time-sync values live in sidecar metadata, a typed optional status
  structure, or a future per-parser subclass?
- Do downstream consumers need all decoded IMU source samples per aligned
  camera/lidar row soon enough to justify an `ImuWindowData` structure in the
  future implementation phase?
- Which calibrated noise-density and bias-random-walk models should a future
  egomotion estimator require in addition to per-sample covariance?
- How should initial pose, velocity, gravity, rig extrinsics, and GPS-derived
  state updates be represented when preintegrated deltas are consumed by
  egomotion?

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
- <a id="ref-ros-imu"></a>ROS 2 Jazzy `sensor_msgs/msg/Imu`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/Imu.html>
- <a id="ref-ros-rep-103"></a>ROS REP 103: Standard Units of Measure and
  Coordinate Conventions: <https://ros.org/reps/rep-0103.html>
- <a id="ref-ros-rep-145"></a>ROS REP 145: Conventions for IMU Sensor
  Drivers: <https://www.ros.org/reps/rep-0145.html>
- <a id="ref-px4-sensor-combined"></a>PX4 `SensorCombined`:
  <https://docs.ros.org/en/rolling/p/px4_msgs/msg/SensorCombined.html>
