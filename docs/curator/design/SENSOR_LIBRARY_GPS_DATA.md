# Sensor Library GPS Data Design

This note documents the research and implemented data model for the first
generic GPS/GNSS data structure in the Cosmos Curator Sensor Library. The
implementation is `GpsData` under `cosmos_curator/core/sensors/data/`.
This note does not define `GpsSensor`, GPS/GNSS parsing, map projection,
localization fusion, filtering, or coordinate transforms.

## Recommendation

GNSS means Global Navigation Satellite System, the broader category of
satellite-positioning systems that includes GPS, Galileo, GLONASS, BeiDou, and
others. GPS is one GNSS constellation, but this design uses `GpsData` for
consistency with the existing Sensor Library terminology.

WGS-84 is the global geodetic coordinate reference system commonly used by GPS
receivers. In this note, WGS-84 coordinates mean latitude, longitude, and
ellipsoid altitude rather than a local projected map coordinate or vehicle
frame.

In GPS/GNSS terminology, a fix is the receiver's computed position solution at
one point in time. A fix can include latitude, longitude, altitude, timestamp,
status/type, and optional accuracy metadata.

`GpsData` is a structure-of-arrays batch of GPS/GNSS fix rows. Each row
represents one decoded or aligned fix sample and satisfies the existing
`SensorData` protocol:

- `align_timestamps_ns` is the reference timeline requested from the
  `SamplingGrid`.
- `sensor_timestamps_ns` is the source GPS/GNSS fix timestamp selected for
  each row.
- scalar payloads use shape `(N,)`.
- optional vector payloads use shape `(N, 3)`.
- optional covariance payloads use shape `(N, 3, 3)`.

The first generic structure uses the existing Sensor Library name
`GpsData`, even though the payload may represent any GNSS source. This keeps
the public API consistent with current Sensor Library examples while the
documentation can state that the coordinate fields represent GNSS fixes.

`GpsData` represents point fix samples, not windows or fused trajectories.
It can represent decoded source fixes when the reference grid is the GPS/GNSS
receiver's own timestamp stream
(`align_timestamps_ns == sensor_timestamps_ns`) and aligned fix rows when a GPS
sensor samples onto a camera, lidar, or arbitrary reference grid.

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

`GpsData` follows these conventions rather than introducing a separate
GPS-specific batch model.

## Reference Models Reviewed

### DriveWorks GPS

Public DriveWorks 6.0.9 GPS documentation exposes GPS frame APIs in two forms:
the GPS sensor group page lists `dwGPSFrameNew`, while the `GPSFrame.h`
reference for the same release defines `dwGPSFrame` with `validityInfo` fields
and deprecates older `dwGPSFlags` values in favor of those per-signal validity
fields ([DriveWorks GPS](#ref-driveworks-gps),
[DriveWorks GPS frame](#ref-driveworks-gps-frame)). Across those GPS frame
docs, the reusable fields are latitude, longitude, WGS-84 ellipsoid altitude,
speed, course, climb, horizontal/vertical/position dilution of precision,
horizontal/vertical accuracy, satellite count, fix status, GPS mode, UTC time,
timestamp quality, and per-signal validity metadata.
DriveWorks also notes that a GPS frame can be only partially filled depending
on sensor capabilities, with validity flags indicating which fields are valid
([DriveWorks GPS read](#ref-driveworks-gps-read)).

DriveWorks also documents raw decode APIs such as
`dwSensorGPS_processRawDataNew`, which accept undecoded GPS bytes and enqueue
decoded GPS frames ([DriveWorks GPS](#ref-driveworks-gps)). This supports
keeping `GpsData` as a decoded SI-unit payload while leaving raw receiver bytes,
protocol packets, and virtual-sensor log layouts parser-specific.

DriveWorks timestamp documentation shows that GPS frame timestamps can be
configured as host, synced, smoothed, or raw depending on sensor protocol
([DriveWorks GPS timestamps](#ref-driveworks-gps-timestamps)). This supports
keeping `sensor_timestamps_ns` as the generic selected source timestamp, adding
optional host/UTC timestamp arrays, and documenting parser-specific timestamp
meaning instead of baking one DriveWorks timestamp mode into `GpsData`.

### ROS NavSatFix and NavSatStatus

ROS `sensor_msgs/NavSatFix` models a GNSS fix in WGS-84 coordinates with
latitude, longitude, altitude, and an ENU position covariance matrix
([ROS NavSatFix](#ref-ros-navsat-fix)). ROS `NavSatStatus` separates fix status
from GNSS service bitmasks, with a fix considered valid when status is at least
`STATUS_FIX` ([ROS NavSatStatus](#ref-ros-navsat-status)).

This supports standardizing the required position fields as WGS-84 geodetic
coordinates and representing position uncertainty as optional ENU covariance.
ROS allows NaN altitude when unavailable, but Sensor Library data structures
should prefer explicit validity/status fields over NaN sentinels.

### PX4 SensorGps

PX4 `SensorGps` uses WGS-84 latitude and longitude, MSL and ellipsoid
altitudes, fix type, horizontal/vertical accuracy, HDOP/VDOP, ground speed,
NED velocity, course over ground, UTC time, satellite count, and receiver
health metadata such as jamming/spoofing/authentication state
([PX4 SensorGps](#ref-px4-sensor-gps)).

This supports including a small set of generic optional fields for velocity,
accuracy, dilution of precision, UTC time, and satellite count, while keeping
receiver health, RTCM, jamming, spoofing, authentication, and antenna-offset
details parser-specific until a downstream consumer needs compatible semantics.

## Implemented `GpsData`

```python
class GpsFixType(enum.IntEnum):
    NO_FIX_OR_UNKNOWN = 0
    FIX_2D = 2
    FIX_3D = 3
    DIFFERENTIAL = 4
    RTK_FLOAT = 5
    RTK_FIXED = 6
    EXTRAPOLATED = 8


@attrs.define(hash=False, frozen=True)
class GpsData:
    __hash__ = None

    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    latitude_deg: npt.NDArray[np.float64]
    longitude_deg: npt.NDArray[np.float64]
    altitude_m: npt.NDArray[np.float64]
    position_valid: npt.NDArray[np.bool_]

    position_covariance_enu_m2: npt.NDArray[np.float64] | None = None
    velocity_enu_m_s: npt.NDArray[np.float64] | None = None
    velocity_valid: npt.NDArray[np.bool_] | None = None

    fix_type: npt.NDArray[np.uint8] | None = None
    satellites_used: npt.NDArray[np.uint16] | None = None
    horizontal_accuracy_m: npt.NDArray[np.float64] | None = None
    vertical_accuracy_m: npt.NDArray[np.float64] | None = None
    hdop: npt.NDArray[np.float64] | None = None
    vdop: npt.NDArray[np.float64] | None = None
    pdop: npt.NDArray[np.float64] | None = None

    host_timestamps_ns: npt.NDArray[np.int64] | None = None
    utc_timestamps_ns: npt.NDArray[np.int64] | None = None
    sequence_counter: npt.NDArray[np.uint64] | None = None
```

### Required Fields

| Field | dtype | shape | unit | Notes |
| --- | --- | --- | --- | --- |
| `align_timestamps_ns` | `np.int64` | `(N,)` | ns | Reference timestamps requested from the `SamplingGrid`; must be strictly increasing. |
| `sensor_timestamps_ns` | `np.int64` | `(N,)` | ns | Source GPS/GNSS fix timestamps selected for each row; may differ from alignment times and may repeat under supersampling. |
| `latitude_deg` | `np.float64` | `(N,)` | deg | WGS-84 latitude, positive north. |
| `longitude_deg` | `np.float64` | `(N,)` | deg | WGS-84 longitude, positive east. |
| `altitude_m` | `np.float64` | `(N,)` | m | Altitude above the WGS-84 ellipsoid. Parsers that only have MSL altitude should document the source and either convert or defer the field until conversion is available. |
| `position_valid` | `np.bool_` | `(N, 3)` | unitless | Per-axis validity for latitude, longitude, and altitude. No-fix rows can be represented without NaN payloads by setting the relevant axes false. |

### Optional Generic Fields

| Field | dtype | shape | unit | Notes |
| --- | --- | --- | --- | --- |
| `position_covariance_enu_m2` | `np.float64` | `(N, 3, 3)` | m^2 | Optional position covariance in local ENU order, matching ROS `NavSatFix` covariance semantics. |
| `velocity_enu_m_s` | `np.float64` | `(N, 3)` | m/s | Optional velocity in local ENU order. Parsers with NED velocity should convert N/E/D to E/N/U. |
| `velocity_valid` | `np.bool_` | `(N, 3)` | unitless | Optional per-axis velocity validity mask. |
| `fix_type` | `np.uint8` | `(N,)` | unitless | Optional normalized `GpsFixType` value. `GpsFixType` is PX4-inspired but not source-native: `0` no fix or unknown, `2` 2D fix, `3` 3D fix, `4` differential, `5` RTK float, `6` RTK fixed, `8` extrapolated. Parser adapters should map source-specific status values into this enum or leave the field `None`. |
| `satellites_used` | `np.uint16` | `(N,)` | count | Optional number of satellites used by the fix. |
| `horizontal_accuracy_m` | `np.float64` | `(N,)` | m | Optional horizontal position accuracy estimate. |
| `vertical_accuracy_m` | `np.float64` | `(N,)` | m | Optional vertical position accuracy estimate. |
| `hdop` | `np.float64` | `(N,)` | unitless | Optional horizontal dilution of precision. |
| `vdop` | `np.float64` | `(N,)` | unitless | Optional vertical dilution of precision. |
| `pdop` | `np.float64` | `(N,)` | unitless | Optional position dilution of precision. |
| `host_timestamps_ns` | `np.int64` | `(N,)` | ns | Optional host/container receive timestamps, distinct from source measurement timestamps. |
| `utc_timestamps_ns` | `np.int64` | `(N,)` | ns | Optional UTC timestamp reported by the receiver, converted to nanoseconds when available. |
| `sequence_counter` | `np.uint64` | `(N,)` | unitless | Optional source sequence counter when available. |

## Timestamp Semantics

`align_timestamps_ns` is the reference timeline requested by the
`SamplingGrid`. It is the timestamp downstream aligned-frame consumers use to
compare rows across sensors.

`sensor_timestamps_ns` is the selected source fix timestamp for each output
row. If a source exposes multiple timestamp concepts, parsers should map the
timestamp used for alignment to `sensor_timestamps_ns` and document its source
meaning.

`host_timestamps_ns` is optional receive-time metadata. For DriveWorks, it can
map to host timestamp output modes when configured. It should not replace
`sensor_timestamps_ns` for alignment unless a parser explicitly documents that
the source has no better measurement timestamp.

`utc_timestamps_ns` is optional GNSS/receiver UTC time. It should be present
only when the source provides a meaningful UTC timestamp. PX4 documents `0` as
unavailable for `time_utc_usec`; a parser should represent unavailable UTC time
with `None` for the whole batch or a parser-specific validity mask rather than
using `0` as a generic timestamp sentinel.

## Coordinate Frames

The required position fields are WGS-84 geodetic coordinates:

- `latitude_deg`: degrees north of the equator
- `longitude_deg`: degrees east of the prime meridian
- `altitude_m`: meters above the WGS-84 ellipsoid

Optional `position_covariance_enu_m2` and `velocity_enu_m_s` use local ENU
axis order at the reported position: east, north, up. `GpsData` should not
silently project to ENU/ECEF positions, transform into a vehicle frame, or fuse
GPS/GNSS with IMU/odometry. Those outputs should be separate localization data
structures.

## Validity and Status

Use explicit validity arrays and normalized status fields instead of NaN
sentinels for generic validity. Required numeric arrays should contain finite
values. Downstream code should use `position_valid`, `velocity_valid`, and
`fix_type` when deciding whether a row or axis is usable.

Keep these source-specific values out of the required `GpsData` core unless a
consumer needs compatible cross-source semantics:

- DriveWorks GPS mode, timestamp quality, and per-signal validity details
- ROS GNSS service bitmasks
- PX4 jamming, spoofing, authentication, RTCM, receiver system error, and
  antenna-offset fields
- raw receiver bytes, NMEA sentences, CAN packets, MCAP payloads, or vendor log
  records

## Validation Constraints

The reviewed DriveWorks, ROS, and PX4 references define units and message
semantics more clearly than universal physical min/max ranges. `GpsData`
enforces structural and basic geographic constraints, while avoiding arbitrary
vehicle- or receiver-specific bounds.

Timestamp dtype and ordering constraints are covered in the required-field
table. Non-timestamp fields use these constraints:

| Field | Constraint |
| --- | --- |
| all payload arrays | Same leading length `N` as `align_timestamps_ns`. |
| `latitude_deg` | `np.float64`, shape `(N,)`, finite values in `[-90.0, 90.0]`. |
| `longitude_deg` | `np.float64`, shape `(N,)`, finite values in `[-180.0, 180.0]`. |
| `altitude_m` | `np.float64`, shape `(N,)`, finite values; no generic min/max. |
| `position_valid` | `np.bool_`, shape `(N, 3)`. |
| `position_covariance_enu_m2` | Optional `np.float64`, shape `(N, 3, 3)`, finite values; must be symmetric positive semidefinite within tolerance when present. |
| `velocity_enu_m_s` | Optional `np.float64`, shape `(N, 3)`, finite values; no generic min/max. |
| `velocity_valid` | Optional `np.bool_`, shape `(N, 3)`. |
| `fix_type` | Optional `np.uint8`, shape `(N,)`; values must be in `GpsFixType`. |
| `satellites_used` | Optional `np.uint16`, shape `(N,)`; no generic upper bound beyond dtype. |
| `horizontal_accuracy_m` | Optional `np.float64`, shape `(N,)`, finite nonnegative values. |
| `vertical_accuracy_m` | Optional `np.float64`, shape `(N,)`, finite nonnegative values. |
| `hdop` | Optional `np.float64`, shape `(N,)`, finite nonnegative values. |
| `vdop` | Optional `np.float64`, shape `(N,)`, finite nonnegative values. |
| `pdop` | Optional `np.float64`, shape `(N,)`, finite nonnegative values. |
| `host_timestamps_ns` | Optional `np.int64`, shape `(N,)`. |
| `utc_timestamps_ns` | Optional `np.int64`, shape `(N,)`; unavailable source sentinels should be normalized at parser boundaries. |
| `sequence_counter` | Optional `np.uint64`, shape `(N,)`; no generic monotonicity rule because wrap, gaps, and reset behavior are source-specific. |

## Related Structures

Do not add a separate undecoded GPS/GNSS payload type immediately. If one is
needed later, avoid names such as `RawGpsData` because "raw" is ambiguous. A
more precise future name would be something like `UndecodedGpsPayload`: NMEA
sentences, vendor binary messages, CAN packets, MCAP message payloads, or
similar source payloads that have not yet been parsed into SI-unit GPS/GNSS
measurements.

Consider separate localization or trajectory structures only when a consumer
needs map-projected positions, ECEF/ENU position series, filtered/fused
vehicle poses, or GPS+IMU odometry. Those have different frame, covariance,
and filtering semantics than GPS/GNSS fix rows.

## Phase-1 Answers

1. The first structure should represent decoded source fixes and aligned fix
   rows using the same `GpsData` type. It should not represent raw receiver
   bytes or fused localization.
2. Public alignment timestamps should remain `np.int64` nanoseconds:
   `align_timestamps_ns` and `sensor_timestamps_ns` are required;
   `host_timestamps_ns` and `utc_timestamps_ns` are optional.
3. Required physical fields should be WGS-84 latitude, longitude, altitude, and
   explicit position validity.
4. Optional generic fields should cover covariance, velocity, fix type,
   satellites used, accuracy, DOP, host/UTC timestamps, and sequence counters.
5. Units should be degrees for latitude/longitude, meters for altitude and
   accuracy, meters per second for velocity, nanoseconds for public timestamps,
   and unitless DOP/status/count values.
6. `GpsData` should document WGS-84 geodetic coordinates and optional local ENU
   covariance/velocity. It should not silently project or transform frames.
7. Batch shapes should use `(N,)` scalar arrays, `(N, 3)` axis vectors/masks,
   and `(N, 3, 3)` covariance arrays.
8. Validity should use explicit boolean masks and optional normalized status
   arrays instead of NaNs.
9. Construction-time validation should cover dtype, shape, finite values,
   latitude/longitude ranges, nonnegative accuracy/DOP, covariance symmetry and
   positive semidefiniteness, normalized fix-type values, read-only views, and
   shared leading batch length.

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
- <a id="ref-driveworks-gps"></a>DriveWorks SDK Reference: GPS Sensor
  (Drive OS 6.0.9, DriveWorks 5.16.65):
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/group__gps__group.html>
- <a id="ref-driveworks-gps-frame"></a>DriveWorks SDK Reference:
  `GPSFrame.h` (Drive OS 6.0.9, DriveWorks 5.16.65):
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/GPSFrame_8h.html>
- <a id="ref-driveworks-gps-read"></a>DriveWorks SDK Reference: Reading GPS
  data from sensor:
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/gps_usecase1.html>
- <a id="ref-driveworks-gps-timestamps"></a>DriveWorks SDK Reference:
  Timestamp options:
  <https://developer.nvidia.com/docs/drive/drive-os/6.0.9/public/driveworks-nvsdk/gps_usecase3.html>
- <a id="ref-ros-navsat-fix"></a>ROS 2 Jazzy `sensor_msgs/msg/NavSatFix`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/NavSatFix.html>
- <a id="ref-ros-navsat-status"></a>ROS 2 Jazzy
  `sensor_msgs/msg/NavSatStatus`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/NavSatStatus.html>
- <a id="ref-px4-sensor-gps"></a>PX4 `SensorGps`:
  <https://docs.px4.io/main/en/msg_docs/SensorGps>
