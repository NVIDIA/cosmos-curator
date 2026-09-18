# Sensor Library LiDAR Data Design

This note documents the proposed design for the first generic LiDAR data
structure in the Cosmos Curator Sensor Library. `LidarData` would be
implemented at `cosmos_curator/core/sensors/data/lidar_data.py`. This note
does not define `LidarSensor`, packet parsing, calibration, motion-compensation
algorithms, or downstream perception models.

The note is structured as a decision document: each open question is laid out
with options, trade-offs, and a tentative recommendation that reviewers can
ratify, alter, or defer.

## Proposed Data Model

`LidarData` is a structure-of-arrays batch of LiDAR points with **two
independent batch dimensions**: `N` (alignment rows) and `P_total`
(points). The two `SensorData`-protocol timestamp arrays sit at row
granularity `(N,)` — matching `CameraData` / `ImuData` / `EgoTrajectory`
— while the point payload is flat with leading dimension `P_total`. `N`
and `P_total` are independent; there is no per-row partition of the
points.

`LidarData` satisfies the existing `SensorData` protocol
([SensorData](#ref-sensor-data)):

- `align_timestamps_ns: (N,) int64`, strictly increasing. When
  `motion_compensated=True`, each entry doubles as the per-row
  motion-compensation reference instant, so a multi-sweep batch can
  record independent reference instants per row.
- `sensor_timestamps_ns: (N,) int64`, non-decreasing. Per-row
  source-reported reference time.
- Required per-point payload: `points_xyz: (P_total, 3) float32` and
  `points_timestamps_ns: (P_total,) int64` (non-decreasing globally).
- Optional per-point payload: `points_intensity`, `points_ring`,
  `points_return_index`, `points_reflectivity`, `points_ambient`,
  `points_validity`, `points_radial_velocity`, `points_sweep_index`,
  and `points_align_index`.
- `metadata` (`LidarMetadata`, required) carries the
  motion-compensation flag, the reference frame of `points_xyz`, the
  static sensor-to-rig `extrinsics`, and the sensor model.

Point → alignment-row mapping is either recorded explicitly in the
optional `points_align_index` field, or reconstructed by consumers on
demand via the existing
[`find_closest_indices`](#ref-find-closest-indices) helper (which
wraps `np.searchsorted` with the necessary `np.clip` to `[0, N-1]`)
under a nearest-alignment convention (see
[D1](#d1-payload-shape-and-alignment-granularity)). Today
`find_closest_indices` requires both inputs to be strictly increasing;
`points_timestamps_ns` is only non-decreasing (LiDAR can return
simultaneous points), so this reuse is tracked as an
[open follow-up](#open-follow-up-questions).

Rig-to-global ego pose is **not** a field on `LidarData`. It is carried by a
sibling `SensorData` type, `EgoTrajectory` (see
[Related Structures](#related-structures) and the
[Tentative `LidarData`](#tentative-lidardata) sketch). See
[D2](#d2-ego-pose-placement) for the rationale.

Because both timestamp arrays sit at `(N,)`, `LidarData` slots into
`AlignedFrame` on the existing row-matching contract with no
validator changes.

"Decoded source points" here means parsed LiDAR packets converted to Sensor
Library field names, NumPy arrays, and SI units. It does not mean raw vendor
UDP packets, MCAP message payloads, or pre-motion-compensated point streams.

## Existing Sensor Library Conventions

The current Sensor Library establishes these conventions, and `LidarData`
should follow them rather than introduce a sensor-specific batch model:

- `SensorData` requires `align_timestamps_ns` and `sensor_timestamps_ns`
  ([SensorData](#ref-sensor-data)).
- `AlignedFrame` requires every payload's `align_timestamps_ns` to exactly
  match the frame's reference timeline ([AlignedFrame](#ref-aligned-frame)).
- `CameraData` and `VideoIndex` use attrs classes, structure-of-arrays layout,
  read-only NumPy views via `as_readonly_view`, and explicit shape/dtype
  validators ([CameraData](#ref-camera-data),
  [validation helpers](#ref-validation-helpers)).
- Public timestamp arrays use `np.int64` nanoseconds.
- Boolean masks use `np.bool_`.
- Static sensor-to-rig calibration is represented by a single `SensorExtrinsics`
  4x4 row-major `float64` matrix ([SensorExtrinsics](#ref-sensor-extrinsics)).
- Ragged side data aligned 1:1 with the primary payload has one existing
  precedent: tuple-of-frozen-containers ([MotionVectorData](#ref-motion-vector-data)
  for video frames).

LiDAR does **not** adopt the MotionVectorData pattern. After four rounds
of design iteration (see
[D1](#d1-payload-shape-and-alignment-granularity) for the trail), the
current shape is: `(N,) align_timestamps_ns` and `(N,) sensor_timestamps_ns`
at row granularity — same as `CameraData` / `ImuData` — plus flat
per-point arrays including `(P_total,) points_timestamps_ns`. `N` and
`P_total` are independent; there is no per-row partition of the points.
When `motion_compensated=True`, `align_timestamps_ns[i]` doubles as the
per-row motion-comp reference instant, so a multi-sweep batch can record
independent reference instants per row. Sweep semantics, when needed,
live in an optional per-point `points_sweep_index` column.

## Reference Models Reviewed

LiDAR vocabulary and schema choices are well-established across open AV
datasets, vendor SDKs, and robotics middleware. The following sources are
reviewed for this design, prioritized by their direct relevance to the data
model.

### ROS `sensor_msgs/PointCloud2`

`sensor_msgs/PointCloud2` is the dominant generic point-cloud message in
robotics ([ROS PointCloud2](#ref-ros-pointcloud2)). Its `PointField` schema is
self-describing: each field has a name, byte offset, datatype enum (`FLOAT32`,
`UINT16`, etc.), and a count, allowing arbitrary per-point attributes. A
`PointCloud2` has one `header.stamp` per cloud and no per-point timestamps;
per-point timing, when present, is conventionally encoded as an additional
field (e.g., `t` or `time` in nanoseconds relative to the cloud's reference
time).

This supports providing a fixed set of named per-point arrays on
`LidarData` rather than a self-describing offset/datatype table: the
sensor library is consumed by Python ML pipelines that benefit more from
predictable typed access than from arbitrary-schema flexibility. It also
supports carrying a per-point sensor-timestamp array
(`points_timestamps_ns: (P_total,) int64`) alongside the row-level
`align_timestamps_ns` / `sensor_timestamps_ns`, rather than relying on
a single sweep-level timestamp.

### nuScenes

nuScenes serializes each sweep as a flat `.pcd.bin` buffer with 5 `float32`
channels per point: `x`, `y`, `z`, `intensity`, `ring`
([nuScenes data format](#ref-nuscenes)). Each sweep has a single reference
timestamp at the file level, not per-point timestamps. Multi-sweep aggregation
(`LidarPointCloud.from_file_multisweep` in the devkit) composes
`T_global_egopose * T_egopose_lidar` per sweep to bring all points into a
common reference frame.

This supports the choice of `(P_i, 3) float32` for `xyz` and a separate
`(P_i,) intensity` channel. It also supports keeping ego pose external to the
per-sweep payload and bringing it together at the batch level rather than
duplicating per-point.

### Waymo Open Dataset

Waymo preserves the LiDAR's range-image topology (a `(rings, azimuth_bins, C)`
dense tensor with `range`, `intensity`, `elongation`, `is_in_no_label_zone`)
alongside a derived point-cloud projection ([Waymo Open Dataset](#ref-waymo)).
Each range-image pixel has a per-pixel `range_image_pose` providing
motion-compensation transforms.

This is the strongest existing reference for explicit per-point timing and
motion compensation. It supports carrying `points_timestamps_ns` per point
and the sibling `EgoTrajectory` for rig-to-global poses adopted in
[D2](#d2-ego-pose-placement). It also supports treating range-image
representation as an optional later extension rather than a required first
form: most downstream ML stacks consume unordered point lists.

### KITTI

KITTI's Velodyne `.bin` files are flat `float32` buffers with 4 channels per
point: `x`, `y`, `z`, `reflectance` ([KITTI](#ref-kitti)). No per-point
timestamps, no ring index, no return index, no per-point motion compensation
at the file level. KITTI is the historical minimum schema.

This supports keeping `points_ring` and `points_return_index` optional
rather than required: a `LidarData` loaded from a KITTI-style source
should be valid with only the required per-point payload
(`points_xyz` and `points_timestamps_ns`) plus the required row-level
`align_timestamps_ns` / `sensor_timestamps_ns`. Per-point timestamps
are synthesized by uniform interpolation across the sweep duration
when the source has only a sweep-level time; the row-level arrays
carry that single sweep time (`N = 1`). Optional `points_intensity`
maps from `reflectance`.

### Ouster SDK

Ouster's `LidarScan` representation in the Ouster SDK preserves both the
range-image topology (`(rings, columns, channels)`) and the projected point
cloud, with explicit per-pixel `t_offset_ns` fields for sub-sweep timing and
per-pixel `signal`, `reflectivity`, `near_ir` channels
([Ouster SDK](#ref-ouster-sdk)). Dual returns are exposed as a separate axis.

This supports carrying `points_timestamps_ns` per point (Ouster's
`t_offset_ns` becomes part of `points_timestamps_ns` directly when the
parser materialises the point cloud from the range image) and treating
multiple returns as a per-point `points_return_index` rather than a
separate batch axis. It also supports keeping additional channels
(reflectivity, near-infrared) as optional, vendor-shaped extensions
rather than required core fields.

### PCL Point Types and PCD Format

The Point Cloud Library defines a set of canonical point types
(`PointXYZ`, `PointXYZI`, `PointXYZRGB`, `PointXYZINormal`, etc.) widely
echoed in downstream libraries ([PCL](#ref-pcl)). PCD files use a header that
explicitly enumerates `FIELDS`, `SIZE`, `TYPE`, `COUNT`, `WIDTH`, `HEIGHT`,
`VIEWPOINT`, and `POINTS` to describe arbitrary per-point schemas.

This supports field naming choices for `LidarData`'s per-point arrays:
matching PCL vocabulary where possible reduces surprise for users coming
from PCL/ROS ecosystems.

### OpenPCDet Dataset Normalization

OpenPCDet's per-dataset loaders (KITTI, nuScenes, Waymo, Lyft) normalize their
respective source formats into a common in-memory dict layout
([OpenPCDet](#ref-openpcdet)). The common keys are `points: (N, C) float32`
where `C` covers `xyz`, `intensity`, and dataset-specific extras like
`elongation` or `ring`; per-sweep poses are carried separately as part of the
sample metadata.

This supports the structure-of-arrays approach: a community-evolved
"canonical" lidar schema looks more like a small set of typed arrays than a
self-describing schema, and is consumed flat-concatenated when feeding 3-D
detectors. Combined with review feedback that not every consumer organizes
points by sweep, this directly motivated the flat `(P_total,)` per-point
payload choice in
[D1](#d1-payload-shape-and-alignment-granularity) — no per-sweep
partitioning is baked into the type; sweep semantics remain available
via the optional `points_sweep_index` column.

### Proprietary AV SDK

A widely-used proprietary embedded-AV LiDAR SDK was also reviewed for this
design. The specific SDK is intentionally left unnamed; the relevant
structural takeaways are summarized below.

The SDK's decoded-packet representation pairs per-packet metadata (a host
timestamp, the sensor timestamp of the first point, a packet duration, FOV
extents, a scan-complete flag, and a returns table with up to ten entries
per packet) with two per-point variants: a Cartesian form (`x`, `y`, `z`,
intensity) and a spherical form (azimuth, elevation, radius, intensity).

Sensor capabilities are surfaced in a separate properties struct (device
string, spin frequency, points/packets per spin, horizontal/vertical FOV,
per-row calibration tables, an available-returns bitmask, and a validity
bitmask over a set of auxiliary per-point data types). The aux types
include per-point time, the spherical mirror coordinates, radial velocity,
SNR, pulse width, ring/line index, detector and zone identifiers, an
existence probability, a blockage / weather-obstruction flag, a per-point
validity flag, and an invalidity-reason enumeration.

The decoded point cloud also carries a coordinate-frame enum
(sensor / rig / custom) and a motion-compensation sub-struct (a compensated
flag plus the reference timestamp all points were warped to) on every batch.

This supports several choices already in this note and motivates a few
additions:

- **Motion-compensation contract (D3) maps directly.** The proposed
  `motion_compensated` flag mirrors the SDK's compensated flag, and
  `reference_frame` mirrors the SDK's coordinate-frame enum (sensor / rig,
  with custom corresponding to `"world"` / `"map"`). The SDK also carries
  the **reference instant** all points were warped to; this design note
  encodes that instant on the per-row `align_timestamps_ns` array (see
  [D1](#d1-payload-shape-and-alignment-granularity)) rather than as a
  separate metadata scalar, which additionally supports multi-sweep
  batches where different rows have different reference instants.

- **Multi-return (D5) confirmed.** The SDK supports up to ten returns per
  pulse via its returns table. A `uint8 return_index` covers this range.
  The SDK additionally distinguishes return **type** (first / last /
  strongest, plus variants thereof) from return **index**; how to document
  `points_return_index` ordering semantics is added to open follow-ups.

- **Per-point timing (D6) confirmed required.** The SDK treats per-point
  time as a standard aux channel and provides packet sensor timestamp +
  duration for sources without explicit per-point timing.

- **New optional channels motivated.** `points_validity` (boolean mask,
  mirroring the SDK's per-point validity flag) and `points_radial_velocity`
  (`float32`, mirroring the SDK's per-point radial-velocity channel) are
  added to the optional per-point field list in D6 — they appear in this
  SDK's schema and have no nearby analogue in the other references
  reviewed. The remaining aux channel types (SNR, pulse width, ring/line
  index, detector ID, zone ID, etc.) remain vendor-shaped extensions; they
  should not be promoted to core fields until a concrete consumer needs
  them.

- **Sensor capabilities stay outside `LidarData`.** Properties-struct
  fields (per-row calibration tables, FOV, spin frequency, available
  returns) belong on a future `LidarSensor` rather than the per-batch
  data class — they are static sensor metadata, not per-row payload.

- **Nominal vs. corrected extrinsics.** The SDK separates an as-designed
  (nominal) sensor-to-rig transform from a calibrated (corrected)
  sensor-to-rig transform. The current `extrinsics` slot carries a single
  matrix; how to surface the nominal/corrected distinction (if at all) is
  added to open follow-ups.

**Status**: reviewed against the SDK's decoded-packet, per-point
Cartesian / spherical, sensor-properties, decoded-point-cloud,
motion-compensation, and auxiliary-per-point-data-type public types.

## Open Design Decisions

The choices below shape the structure of `LidarData` and `LidarMetadata`.
Each lists options, trade-offs, and a tentative recommendation.

### D1. Payload shape and alignment granularity

How should the per-point data be represented, and at what granularity does
the alignment grid live? Four rounds of iteration.

**Option A — tuple of `LidarSweepData`** (mirrors `MotionVectorData`)

```python
sweeps: tuple[LidarSweepData, ...]   # length N; each row self-contained
```

Pros: matches the existing ragged-side-data pattern in
`MotionVectorData`; each row is independently validatable; trivial to
slice one row out; explicit `LidarSweepData.empty()` placeholder
mirrors `MotionVectorFrameData.empty()` for empty rows.

Cons: `N` Python objects per batch; cross-row vectorized numpy
operations require iteration in Python. Forces a "sweep" mental model
onto consumers that don't use it (e.g., solid-state LiDAR pipelines and
pipelines that query points by time window irrespective of sweep
boundaries).

**Option B — flat per-point arrays + CSR-style `align_offsets`**

```python
points_xyz:        npt.NDArray[np.float32]   # shape (P_total, 3)
points_intensity:  npt.NDArray[np.float32]   # shape (P_total,)
align_offsets:     npt.NDArray[np.int64]     # shape (N+1,)
# points for alignment row i = points_xyz[align_offsets[i]:align_offsets[i+1]]
```

Pros: vectorizes cross-row operations in single numpy calls; closer to
how KITTI/nuScenes/Waymo serialize and how OpenPCDet/MMDetection3D batch
for training. The `align_offsets` array is a partition that consumers can
use or ignore.

Cons: more bookkeeping during filtering and re-batching; mutation
patterns require offset rebuilds; `align_offsets` carries semantics no
current consumer actually uses (we observed this only after adopting it).

**Option C — flat per-point arrays, no per-row partition, `(N,)` alignment grid**

```python
align_timestamps_ns:  npt.NDArray[np.int64]   # shape (N,) — alignment grid
sensor_timestamps_ns: npt.NDArray[np.int64]   # shape (N,) — per-row source-reported time
points_xyz:           npt.NDArray[np.float32] # shape (P_total, 3)
points_timestamps_ns: npt.NDArray[np.int64]   # shape (P_total,) globally non-decreasing
# N and P_total are independent. Consumers that want "points near
# alignment row i" filter points_timestamps_ns directly.
```

Pros: keeps `LidarData` compatible with `AlignedFrame`'s row-matching
protocol — the `(N,)` alignment grid lines up with other sensors' grids
in the same bundle.

Cons: initially the `(N,)` grid felt like it carried no per-point
information — "what's the alignment timestamp of point `j`?" had no
direct answer. Round 3 addressed this by collapsing to per-point
alignment (Option D). Round 4 reversed that decision — see below.

**Option D — per-point alignment; no `(N,)` grid at all**  (rejected on reflection)

```python
align_timestamps_ns:  npt.NDArray[np.int64]    # shape (P_total,) non-decreasing
sensor_timestamps_ns: npt.NDArray[np.int64]    # shape (P_total,) non-decreasing
points_xyz:           npt.NDArray[np.float32]  # shape (P_total, 3)
```

Pros: maximally simple; "what's the alignment timestamp of point `j`?"
has an immediate answer (`align_timestamps_ns[j]`).

Cons: **breaks `AlignedFrame`'s row-matching invariant.** A camera's
`(30,)` and a lidar's `(P_total,)` `align_timestamps_ns` can't match
`np.array_equal`, so `LidarData` and other row-shaped sensors can't
coexist in the same bundle. Concretely blocks the primary use case:
LiDAR-camera fusion inside an `AlignedFrame`.

**Option C revisited — with alignment grid doubling as motion-comp reference instants**  (recommended)

```python
align_timestamps_ns:  npt.NDArray[np.int64]     # (N,) strictly increasing
                                                 #   grid AND (when motion_compensated=True) per-row motion-comp
                                                 #   reference instants
sensor_timestamps_ns: npt.NDArray[np.int64]     # (N,) non-decreasing — per-row source-reported time
points_xyz:           npt.NDArray[np.float32]   # (P_total, 3)
points_timestamps_ns: npt.NDArray[np.int64]     # (P_total,) non-decreasing — per-point sensor timestamp
```

Same `(N,)` shape as Option C, but the semantic of `align_timestamps_ns`
is enriched: when `motion_compensated=True`, `align_timestamps_ns[i]` is
the reference instant that all points with per-point time closest to
`align_timestamps_ns[i]` have been warped to. Different points in a batch
can be compensated to different instants — a batch spanning 5 sweeps can
compensate each sweep to its own midpoint, with
`align_timestamps_ns = [midpoint_0, ..., midpoint_4]`. This makes the
old `LidarMetadata.motion_compensation_timestamp_ns` scalar redundant —
it is removed.

**Recommendation: Option C revisited.** Rationale:

- Four rounds of iteration. Round 1 (review feedback) moved from
  Option A to Option B — sweep boundaries shouldn't be load-bearing.
  Round 2 observed that the `align_offsets` partition was unused —
  drop it (Option C). Round 3 collapsed to per-point alignment
  (Option D). Round 4 reversed Round 3: the `AlignedFrame`
  incompatibility was too expensive, and enriching Option C's
  `align_timestamps_ns` with motion-comp reference instants restores the
  answer to "what's each point compensated to?" without breaking the
  bundle protocol.
- **`AlignedFrame` integration works with zero validator changes.**
  Both timestamp arrays are `(N,)`, matching `CameraData` / `ImuData` /
  `EgoTrajectory`. No `isinstance` carve-outs; no new bundle abstraction.
- **Motion-comp anchors are per-row, not per-batch.** A LiDAR batch
  with multiple sweeps can compensate each sweep to its own reference
  instant. More expressive than the old "one scalar for the whole batch"
  model.
- **Per-point ↔ align-row mapping.** The mapping can either be recorded
  by the producer in the optional `points_align_index` field, or
  reconstructed by consumers via the existing
  [`find_closest_indices`](#ref-find-closest-indices) helper (which
  wraps `np.searchsorted` with `np.clip` to `[0, N-1]`). Producers use
  the explicit field when they applied a non-obvious policy (e.g.,
  binning by `align_timestamps_ns ± Δ` rather than nearest); otherwise
  the field is absent and the standard nearest-alignment convention
  applies. Note the helper today requires strictly-increasing inputs
  on both sides; `points_timestamps_ns` is only non-decreasing, so
  either the helper needs a variant or the recovery must be inlined —
  tracked as an [open follow-up](#open-follow-up-questions).
  Consumers that don't need the mapping filter by per-point timestamps
  directly.
- **Sweep semantics** stay optional via `points_sweep_index`
  ([D6](#d6-required-vs-optional-per-point-fields)) — orthogonal to the
  alignment grid.

### D2. Ego-pose placement

LiDAR consumers commonly need a rig pose at each alignment row's reference
time for motion compensation and multi-row aggregation. Where should ego
pose live? No part of the sensors package currently carries ego pose;
`LidarData` is the first place that needs it.

**Option A — optional `ego_poses` field on `LidarData`**

```python
ego_poses: npt.NDArray[np.float64] | None   # shape (N, 4, 4); rig-to-global
```

Pros: localizes the change to LiDAR; doesn't disturb `CameraData` or
`AlignedFrame`; consumers fetch pose directly from the LiDAR object.

Cons: ego pose is logically a rig property, not a LiDAR property. If
other sensors (radar, etc.) later need ego pose at sweep/scan time, the
field will need to be lifted up, requiring a refactor. Also conflates
two independent frame names on `LidarMetadata` (the frame of `points_xyz`
and the global frame `ego_poses` targets).

**Option B — add `ego_poses` to `AlignedFrame`**

```python
class AlignedFrame:
    align_timestamps_ns: ...
    sensor_data: Mapping[str, SensorData]
    ego_poses: npt.NDArray[np.float64] | None   # new field
```

Pros: ego pose is shared across all sensors at the bundle level, matching
its physical meaning; only one trajectory stored per bundle.

Cons: many places currently construct `AlignedFrame`; modifies a class
shared across all sensors; couples `AlignedFrame` semantics to
"trajectory of a rig," which may not always be defined (e.g., static
captures).

**Option C — add an `EgoTrajectory` type satisfying `SensorData`**  (recommended)

```python
@attrs.define(hash=False, frozen=True)
class EgoTrajectory:
    align_timestamps_ns: npt.NDArray[np.int64]
    sensor_timestamps_ns: npt.NDArray[np.int64]
    positions_m: npt.NDArray[np.float64]          # shape (N, 3) rig-to-global translation
    orientations_xyzw: npt.NDArray[np.float64]    # shape (N, 4) rig-to-global unit quaternion
    frame: str   # e.g., "world", "map", "odom"
```

Then carry it inside `AlignedFrame.sensor_data` keyed by id (e.g.,
`"ego"`):

```python
frame.sensor_data["ego"]         # an EgoTrajectory
frame.sensor_data["lidar_top"]   # a LidarData
```

Pros: ego pose participates in the same alignment guarantees as every
other stream; no `AlignedFrame` schema change; cleanly handles multiple
trajectory sources (e.g., ground-truth GPS+IMU vs. SLAM estimate) by
using different keys. `LidarData` stays focused on the lidar payload;
`LidarMetadata.reference_frame` is the only frame name on the lidar side.

Cons: changes the "what counts as a sensor?" mental model;
`SensorData`-as-trajectory is a slight overload.

**Recommendation: Option C.** Rationale:

- Option C is the right end-state. Earlier drafts staged through Option A
  first as a transitional shape on `LidarData`, but on reflection there is
  no real benefit to that staging: Option A is a strict subset of Option C,
  and migrating later would touch every consumer that reads ego pose. Better
  to write consumers against the final shape from the start.
- Option C cleanly handles ego pose for any future sensor that needs it
  (radar, additional LiDAR with independent timing, etc.) without any
  per-sensor refactoring — they all read `frame.sensor_data["ego"]`.
- Option C supports multiple trajectory sources by using different keys
  (e.g., `"ego_gnss_imu"` and `"ego_slam"`), and each trajectory satisfies
  the `SensorData` alignment guarantees just like a sensor stream.
- Option B is rejected because it modifies the shared `AlignedFrame` class
  for what is conceptually an "additional sensor stream" — Option C
  achieves the same sharing without the schema change.

`EgoTrajectory` stores translation and XYZW unit quaternions as its canonical
pose representation. `as_homogeneous_matrices()` provides row-major
`(N, 4, 4)` transforms to consumers that need matrix composition.

### D3. Motion-compensation contract

When `LidarData` is constructed, are the `xyz` coordinates expected to be in
the sensor frame at the per-point sensor timestamp (raw), or already transformed
into a single reference frame for the sweep (motion-compensated)?

**Option A — `LidarData` is agnostic; producer declares**

A `LidarMetadata` field captures the contract:

```python
@attrs.define(hash=False, frozen=True)
class LidarMetadata:
    motion_compensated: bool
    reference_frame: str                              # frame of points_xyz: "sensor", "rig", "world", "map", ...
    extrinsics: SensorExtrinsics | None = None        # static sensor-to-rig calibration (batch-level metadata)
    sensor_model: str | None = None
    # When motion_compensated=True, the per-row motion-comp reference instants
    # live on LidarData.align_timestamps_ns[i]; there is no separate scalar
    # field for the batch's compensation timestamp.
```

Pros: supports both pipelines; doesn't force a costly transform at
parse time when the consumer doesn't need it; makes the contract
explicit and validatable. The per-row motion-comp reference instants
live on `LidarData.align_timestamps_ns`, so a batch spanning multiple
sweeps can record independent reference instants per row.

Cons: consumers must check the flag and apply or skip the transform
themselves; a misconfigured producer can mislead downstream stages.

**Option B — `LidarData` is always motion-compensated**

Pros: removes ambiguity; downstream perception stages get the clean
shape they want.

Cons: requires the producer to know an ego trajectory at parse time;
loses information for stages that want raw points; not all source
formats provide compensation natively.

**Option C — `LidarData` is always raw**

Pros: closest to source data; producers do less work.

Cons: pushes the transform into every consumer; for multi-sweep
aggregation each consumer reimplements compensation.

**Recommendation: Option A.** Rationale:

- The sensors package convention is to carry source data with explicit
  semantics, not to force a specific transformation contract
  ([ImuData](#ref-imu-data) "Coordinate Frames" section).
- A motion-compensated stage can be added as a derived helper without
  changing `LidarData`.
- The flag plus `reference_frame` together let validators reject
  inconsistent payloads (e.g., `motion_compensated=True` with
  `reference_frame="sensor"` is a contradiction).
- When `motion_compensated=True`, the reference instant for each row is
  stored in `LidarData.align_timestamps_ns[i]` (see
  [D1](#d1-payload-shape-and-alignment-granularity)). Individual points
  can therefore be compensated to different instants: a batch spanning
  multiple sweeps records the per-sweep reference instants as successive
  entries in `align_timestamps_ns`. Consumers resolve "which reference
  instant is point `j` compensated to?" via
  [`find_closest_indices`](#ref-find-closest-indices) on
  `align_timestamps_ns` — no dedicated per-point index column.

### D4. Range-image representation

Some sensors (Ouster, Waymo) preserve a dense `(rings, azimuth_bins, C)`
range-image representation alongside the point cloud. Should the first
`LidarData` support it?

**Option A — point list only**

Add range-image support later as an optional sibling payload when a consumer
needs it.

**Option B — both, with point list as required and range image as optional**

```python
range_image: RangeImageData | None  # optional sibling like MotionVectorData
```

**Recommendation: Option A for the first cut.** Rationale:

- Most downstream ML stacks (OpenPCDet, MMDetection3D, PyTorch3D batched
  detectors) consume point lists, not range images.
- Range-image support requires per-sensor convention bookkeeping (ring
  ordering, azimuth zero direction, missing-pixel encoding) that varies by
  vendor.
- Range image can be added as an optional sibling payload in v2 without
  disturbing existing v1 consumers, mirroring how `motion_vectors` was added
  to `CameraData`.

### D5. Multi-return handling

Spinning LiDARs (Velodyne, Hesai) and time-of-flight LiDARs (Ouster) can
report multiple returns per laser pulse (first, last, strongest). How is this
represented?

**Option A — `points_return_index` as an optional per-point field**

```python
points_return_index: npt.NDArray[np.uint8] | None   # 0 = first, 1 = second, ...
```

All returns from all pulses are concatenated in the flat point arrays; the
per-point `points_return_index` disambiguates.

**Option B — separate flat point arrays per return**

```python
points_xyz_first_return: npt.NDArray[np.float32]            # (P_first, 3)
points_xyz_last_return:  npt.NDArray[np.float32] | None     # (P_last, 3)
# plus parallel arrays for every other per-point field, per return
```

**Recommendation: Option A.** Rationale:

- Matches Ouster SDK, KITTI, and nuScenes conventions: multi-return points
  share one cloud with a return-index column.
- Keeps the data class shape stable across single-return and dual-return
  sensors.
- Avoids `None` proliferation for the (common) case where only first returns
  are used.

### D6. Required vs optional per-point fields

What is the minimum set of per-point fields a `LidarData` must carry? All
per-point arrays are flat with leading dimension `P_total` (see D1).

**Required (proposed):**

- `points_xyz: (P_total, 3) float32`
- `points_timestamps_ns: (P_total,) int64`

(`align_timestamps_ns` and `sensor_timestamps_ns` are also required
per-batch but live at row granularity `(N,)` — see
[D1](#d1-payload-shape-and-alignment-granularity) for the shape.)

**Optional generic:**

- `points_intensity: (P_total,) float32` (or `uint16`; see below)
- `points_ring: (P_total,) uint16`
- `points_return_index: (P_total,) uint8`
- `points_reflectivity: (P_total,) uint16`
- `points_ambient: (P_total,) uint16` (Ouster near-IR; vendor-specific)
- `points_validity: (P_total,) bool` (False marks points the source flagged
  as invalid / blocked / filtered; mirrors the per-point validity flag in
  the reviewed proprietary AV SDK)
- `points_radial_velocity: (P_total,) float32` m/s (Doppler-capable / FMCW
  lidars only; mirrors the per-point radial-velocity channel in the
  reviewed proprietary AV SDK)
- `points_sweep_index: (P_total,) uint16` (producer-defined sweep id per
  point; absent when the producer doesn't track sweep boundaries; sized
  for fleets with many short sweeps in a single batch)
- `points_align_index: (P_total,) uint16` (producer-authoritative
  point → align-row mapping; each entry is an index into
  `align_timestamps_ns`; absent when the producer expects consumers to
  compute the mapping via `np.searchsorted`)

**Recommendation as listed.** Rationale:

- `points_xyz` is universally required.
- `points_timestamps_ns` is required. Motion-compensation math, per-point
  time-window queries, and per-point ↔ alignment-row recovery all depend
  on it. If a source provides only sweep-level timing, the parser
  synthesises per-point times (uniform interpolation is acceptable) and
  documents the synthesis.
- `points_intensity` is common but not universal: post-processed
  CAD-style clouds, depth-camera-derived clouds, and some specialty
  sensors omit it. Keeping it optional admits those sources without
  forcing parsers to synthesize placeholder values.
- `points_sweep_index` is the first-class home for "which producer-defined
  sweep did this point come from?" — the only sweep-shaped semantic that
  survived the D1 iterations and is cheap to populate for sweep-shaped
  producers.
- `points_ring` and `points_return_index` are sensor-specific; KITTI lacks
  both. Keeping them optional allows loading minimal-schema sources.
- `points_reflectivity` (calibrated reflectance) is distinct from raw
  `points_intensity` (return strength); some sensors report both. Optional.
- `points_validity` carries an explicit per-point invalid/blocked mask when
  the source provides one. Default policy is still "drop invalid points at
  parser time"; the field exists for consumers (e.g., blockage analysis,
  weather classification) that need to retain them.
- `points_radial_velocity` is only meaningful for FMCW/Doppler-capable
  sensors (e.g., Aeva) and absent on time-of-flight lidars. Kept optional.

Choice of `points_intensity` dtype: `float32` is recommended over `uint16`
for homogeneity with `points_xyz` and downstream tensor conversion, at the
cost of 2 bytes per point versus a packed `uint16`. Reconsider if storage
footprint becomes a constraint.

## Tentative `LidarData`

This section reflects the recommendations above (D1 = Option C revisited
with alignment grid doubling as motion-comp reference instants, D2 = C
`EgoTrajectory` sibling, D3=A, D4=A, D5=A, D6 as listed). It is tentative
and will be updated as decisions are ratified.

```python
@attrs.define(hash=False, frozen=True)
class LidarMetadata:
    motion_compensated: bool
    reference_frame: str                                # frame of points_xyz
    extrinsics: SensorExtrinsics | None = None          # static sensor-to-rig calibration
    sensor_model: str | None = None
    # When motion_compensated=True, per-row motion-comp reference instants
    # live on LidarData.align_timestamps_ns[i]. No dedicated scalar field.


@attrs.define(hash=False, frozen=True)
class LidarData:
    __hash__ = None

    # SensorData protocol — row-level dual timestamps (length N).
    align_timestamps_ns:  npt.NDArray[np.int64]                # (N,) strictly increasing
                                                                #   grid AND (when motion_compensated=True) per-row
                                                                #   motion-comp reference instants
    sensor_timestamps_ns: npt.NDArray[np.int64]                # (N,) non-decreasing — per-row source-reported time

    # Required per-point payload (length P_total; independent of N).
    points_xyz:           npt.NDArray[np.float32]              # (P_total, 3)
    points_timestamps_ns: npt.NDArray[np.int64]                # (P_total,) non-decreasing — per-point sensor timestamp

    metadata: LidarMetadata                                    # carries extrinsics + frame + comp state

    # Optional per-point payload.
    points_intensity:       npt.NDArray[np.float32] | None = None
    points_ring:            npt.NDArray[np.uint16]  | None = None
    points_return_index:    npt.NDArray[np.uint8]   | None = None
    points_reflectivity:    npt.NDArray[np.uint16]  | None = None
    points_ambient:         npt.NDArray[np.uint16]  | None = None
    points_validity:        npt.NDArray[np.bool_]   | None = None
    points_radial_velocity: npt.NDArray[np.float32] | None = None
    points_sweep_index:     npt.NDArray[np.uint16]  | None = None   # producer-defined sweep id per point
    points_align_index:     npt.NDArray[np.uint16]  | None = None   # index into align_timestamps_ns (producer-authoritative)


@attrs.define(hash=False, frozen=True)
class EgoTrajectory:                                           # satisfies SensorData
    __hash__ = None
    align_timestamps_ns:  npt.NDArray[np.int64]                # (N,) strictly increasing
    sensor_timestamps_ns: npt.NDArray[np.int64]                # (N,) non-decreasing
    positions_m:          npt.NDArray[np.float64]              # (N, 3) rig-to-global translation
    orientations_xyzw:    npt.NDArray[np.float64]              # (N, 4) rig-to-global unit quaternion
    pose_valid:           npt.NDArray[np.bool_]                # (N,) false rows use zero position and identity quaternion
    frame:                str                                  # "world" | "map" | "odom" | ...
```

Notes:

- `LidarData` has two batch dimensions: `N` (alignment rows) and
  `P_total` (points). They are independent — the alignment grid is not
  a partition of the points, and `P_total` need not be a multiple of
  `N` in any way.
- **Per-point ↔ alignment-row mapping.** When the producer applied a
  non-obvious mapping (e.g., binning by `align_timestamps_ns ± Δ` rather
  than nearest), it can be recorded explicitly in the optional
  `points_align_index` field. When absent, consumers reconstruct the
  mapping on demand via
  `np.searchsorted(align_timestamps_ns, points_timestamps_ns)` under a
  nearest-alignment convention. Producers document which path they took.
- **`AlignedFrame` integration is direct.** Both `LidarData` and
  `EgoTrajectory` expose `(N,) align_timestamps_ns` and `sensor_timestamps_ns`,
  matching the existing protocol; no validator changes are needed.
- Multiple trajectories (e.g., ground-truth GPS+IMU and SLAM estimate)
  coexist inside a bundle under different `EgoTrajectory` keys
  (`"ego_gnss_imu"`, `"ego_slam"`).

### Required Fields (`LidarData`)

| Field | dtype | shape | unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `align_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Strictly increasing. When `motion_compensated=True`, each entry is also the reference instant that all points nearest to it (by `points_timestamps_ns`) have been warped to. |
| `sensor_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Per-row source-reported reference time. Non-decreasing; may repeat under supersampling. |
| `points_xyz` | `np.float32` | `(P_total, 3)` | m | yes | Point coordinates in the frame declared by `LidarMetadata.reference_frame`. |
| `points_timestamps_ns` | `np.int64` | `(P_total,)` | ns | yes | Per-point sensor timestamp. Non-decreasing globally across the batch. |
| `metadata` | `LidarMetadata` | — | — | yes | Carries `motion_compensated`, `reference_frame` (frame of `points_xyz`), `extrinsics` (static sensor-to-rig calibration), and `sensor_model`. |

### Optional Fields (`LidarData`)

| Field | dtype | shape | unit | Notes |
| --- | --- | --- | --- | --- |
| `points_intensity` | `np.float32` | `(P_total,)` | sensor-defined | Return strength; sensor calibration determines absolute units. Absent for sources that don't report intensity. |
| `points_ring` | `np.uint16` | `(P_total,)` | — | Laser/ring index for spinning sensors; absent for solid-state. Sized for sensors with more than 256 channels (e.g., next-generation Ouster / Hesai). |
| `points_return_index` | `np.uint8` | `(P_total,)` | — | 0 = first return, 1 = second, etc. Absent if only first returns retained. |
| `points_reflectivity` | `np.uint16` | `(P_total,)` | sensor-defined | Calibrated reflectance distinct from raw `points_intensity`. |
| `points_ambient` | `np.uint16` | `(P_total,)` | sensor-defined | Near-infrared ambient channel (Ouster-style sensors). |
| `points_validity` | `np.bool_` | `(P_total,)` | — | `False` marks points the source flagged invalid / blocked / filtered. |
| `points_radial_velocity` | `np.float32` | `(P_total,)` | m/s | Per-point radial velocity from FMCW / Doppler lidars. |
| `points_sweep_index` | `np.uint16` | `(P_total,)` | — | Producer-defined sweep / scan id per point. Absent when the producer doesn't track sweep boundaries. Sized for fleets with many short sweeps in a single batch. |
| `points_align_index` | `np.uint16` | `(P_total,)` | — | Producer-authoritative point → alignment-row mapping; each entry is an index into `align_timestamps_ns` (must be in `[0, N)`). Absent when consumers should compute the mapping via `np.searchsorted` and `np.clip`. |

### Required Fields (`EgoTrajectory`)

| Field | dtype | shape | unit | Required | Notes |
| --- | --- | --- | --- | --- | --- |
| `align_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Same `SensorData` protocol requirement as `LidarData`. |
| `sensor_timestamps_ns` | `np.int64` | `(N,)` | ns | yes | Same `SensorData` protocol requirement as `LidarData`. |
| `positions_m` | `np.float64` | `(N, 3)` | m | yes | Rig-to-global translation at each alignment row's reference time. |
| `orientations_xyzw` | `np.float64` | `(N, 4)` | — | yes | Rig-to-global unit quaternion in XYZW component order. |
| `pose_valid` | `np.bool_` | `(N,)` | — | yes | `False` preserves timeline continuity with a zero-position, identity XYZW quaternion placeholder; consumers must not trust that pose. |
| `frame` | `str` | — | — | yes | Name of the target global frame (e.g., `"world"`, `"map"`, `"odom"`). |

## Timestamp Semantics

`LidarData` carries three timestamp arrays:

- **`align_timestamps_ns: (N,)`** — the alignment grid, strictly
  increasing. Satisfies the `SensorData` protocol at row granularity.
  When `motion_compensated=True`, each `align_timestamps_ns[i]` is also
  the reference instant that points nearest to it (by
  `points_timestamps_ns`) have been warped to. Different rows can carry
  independent reference instants — a batch spanning multiple sweeps
  records the per-sweep motion-comp anchors as successive entries.
- **`sensor_timestamps_ns: (N,)`** — per-row source-reported reference
  time. Non-decreasing. Typically the sweep start / midpoint / query
  instant, producer's choice; parsers document the convention.
- **`points_timestamps_ns: (P_total,)`** — per-point sensor timestamp.
  Non-decreasing globally across the batch. This is what makes
  per-point-time-window queries and motion-comp math possible.

**Recovering "which alignment row is point `j` associated with?"** is a
consumer-side operation, not a stored field. The intended primitive is
the existing sensor-library helper
[`find_closest_indices`](#ref-find-closest-indices), which wraps
`np.searchsorted` with the necessary `np.clip` to `[0, N-1]`:

```python
# Nearest-align policy (typical for motion-comp):
align_idx = find_closest_indices(lidar.align_timestamps_ns,
                                 lidar.points_timestamps_ns)  # (P_total,)
```

Caveat: `find_closest_indices` today requires both inputs to be
strictly increasing, while `points_timestamps_ns` is only
non-decreasing (LiDAR can return simultaneous points). Reusing it
verbatim requires relaxing the grid-side check, exposing a variant
that skips it, or inlining the equivalent `searchsorted` + `np.clip`
+ nearest pick. This is tracked as an
[open follow-up](#open-follow-up-questions). The inlined form is:

```python
# Same math as find_closest_indices, tolerant of non-strictly-increasing grid:
prev = np.searchsorted(lidar.align_timestamps_ns,
                       lidar.points_timestamps_ns, side="right") - 1
prev = np.clip(prev, 0, len(lidar.align_timestamps_ns) - 1)
nxt  = np.clip(prev + 1, 0, len(lidar.align_timestamps_ns) - 1)
dp = np.abs(lidar.points_timestamps_ns - lidar.align_timestamps_ns[prev])
dn = np.abs(lidar.align_timestamps_ns[nxt] - lidar.points_timestamps_ns)
align_idx = np.where(dn < dp, nxt, prev)   # (P_total,) int — nearest align row per point
```

The producer's motion-compensation policy (which alignment row each point
was actually warped to) should be documented by the parser so consumers
know which recovery rule to use — nearest-align is the recommended
default.

**Parsers ship identity by default** — `align_timestamps_ns` at parser
output typically records the natural alignment moments of the source
(sweep midpoints, camera frame times, etc.). A separate downstream
grid-binning helper (see [Related Structures](#related-structures)) can
map an existing batch onto a coarser or different `align_timestamps_ns`
grid when a consumer needs it; the helper rewrites the `(N,)` timestamp
arrays but never touches `points_timestamps_ns`.

**Grid-binning interacts with motion compensation.** When
`motion_compensated=True`, each `align_timestamps_ns[i]` is not just a
label — it is the physical reference instant that points nearest to
row `i` have already been warped to. Silently re-labelling those
entries would invalidate the compensation anchors while leaving
`points_xyz` untouched. The grid-binning helper must therefore either
(a) require `motion_compensated=False` on input (parser-native batches,
the common case), or (b) accept a pose source (typically an
`EgoTrajectory`) and re-warp `points_xyz` onto the new anchors before
rewriting `align_timestamps_ns`. Pure label-only re-binning of a
compensated batch is not a supported operation.

`LidarData` deliberately does not carry a `pts_stream` (producer-native
presentation timestamp) array analogous to `CameraData.pts_stream`. In all
common LiDAR source formats (MCAP, ROS bags, KITTI / nuScenes / Waymo file
readers, and the static-dataset paths), the source's native time domain is
already nanoseconds, so a `pts_stream` field would be a redundant copy of
`sensor_timestamps_ns`. The video-style `time_base` precision problem that
motivates `pts_stream` on `CameraData` does not arise. Re-add the field if
a future LiDAR source format uses a non-ns native time domain where
lossless re-seek would otherwise be impossible.

If a source provides only a sweep-level timestamp and no per-point timing,
the parser should synthesize `points_timestamps_ns` (uniform interpolation
across an estimated sensor acquisition duration is acceptable) and document the
synthesis.

## Coordinate Frames

`LidarData` does not silently transform coordinate frames. Points in
`points_xyz` are expressed in the frame declared by
`LidarMetadata.reference_frame`. The metadata flag `motion_compensated`
declares whether all points in the batch share a single instantaneous
reference frame (`True`) or whether each point is in the sensor frame at
its own sensor timestamp (`False`). The flag applies to every point in the
batch; mixing motion-compensated and raw points in a single `LidarData` is
not supported.

Recommended defaults at the Sensor Library API boundary:

- Right-handed frames.
- SI units: meters for `xyz`, seconds (via ns timestamps) for time.
- Static sensor-to-rig calibration in `LidarMetadata.extrinsics` (batch-level metadata).
- Time-varying rig-to-global pose (if available) in a sibling `EgoTrajectory` carried inside the same bundle (both `LidarData` and `EgoTrajectory` expose `(N,)` alignment, so they can sit in the same `AlignedFrame`); the target global frame is named by `EgoTrajectory.frame` and is independent of `LidarMetadata.reference_frame`.
- Frame conventions (axis direction, rotation order) follow ROS REP 103 where
  applicable ([REP 103](#ref-ros-rep-103)).

The valid combinations of `motion_compensated` and `reference_frame` are:

| `motion_compensated` | `reference_frame` | Meaning |
| --- | --- | --- |
| `False` | `"sensor"` | Raw — each point in the sensor frame at its own sensor timestamp. No ego trajectory required. Cloud is time-sheared by ego motion. |
| `False` | `"rig"` | Each point in the rig frame at its own sensor timestamp, obtained from sensor-frame coordinates via the static `extrinsics`. Cloud is time-sheared because the rig frame itself moves with the car. No ego trajectory required. |
| `False` | `"world"` / `"map"` | **Invalid** — rejected by validators. Reaching a fixed global frame requires per-point `T_world_rig(t_sensor)` lookup, which is mathematically equivalent to motion compensation; producers should declare `True` and record the chosen anchor instant. |
| `True` | `"sensor"` | **Invalid** — rejected by validators. A single-instant snapshot of a moving frame is ill-defined; producers should target `"rig"` or a global frame instead. |
| `True` | `"rig"` | Points motion-compensated into the rig frame. When `N == 1`, `align_timestamps_ns[0]` is the single reference instant. When `N > 1`, different points are compensated to different instants — point `j` was warped to whichever `align_timestamps_ns[i]` its per-point time is nearest to. |
| `True` | `"world"` / `"map"` | Same as above, but the target frame is a fixed global frame. Per-row reference instants live in `align_timestamps_ns`. |

## Validation Constraints

Timestamp dtype, length, and ordering constraints are covered in the field
tables. Non-timestamp fields use these constraints:

| Field | Constraint |
| --- | --- |
| `LidarData` row-level arrays | Both `align_timestamps_ns` and `sensor_timestamps_ns` share the leading length `N`. |
| `LidarData` per-point arrays | All share the leading length `P_total` (applies to `points_xyz`, `points_timestamps_ns`, and every present optional `points_*` field). `P_total` is independent of `N`. |
| `LidarData.align_timestamps_ns` | `np.int64`, shape `(N,)`, strictly increasing. |
| `LidarData.sensor_timestamps_ns` | `np.int64`, shape `(N,)`, non-decreasing. |
| `LidarData.points_xyz` | `np.float32`, shape `(P_total, 3)`, finite values (no NaN/Inf). |
| `LidarData.points_timestamps_ns` | `np.int64`, shape `(P_total,)`, non-decreasing globally across the batch. |
| `LidarData.points_intensity` | Optional `np.float32`, shape `(P_total,)`, finite values. |
| `LidarData.points_ring` | Optional `np.uint16`, shape `(P_total,)`. |
| `LidarData.points_return_index` | Optional `np.uint8`, shape `(P_total,)`. |
| `LidarData.points_reflectivity` | Optional `np.uint16`, shape `(P_total,)`. |
| `LidarData.points_ambient` | Optional `np.uint16`, shape `(P_total,)`. |
| `LidarData.points_validity` | Optional `np.bool_`, shape `(P_total,)`. |
| `LidarData.points_radial_velocity` | Optional `np.float32`, shape `(P_total,)`, finite values. |
| `LidarData.points_sweep_index` | Optional `np.uint16`, shape `(P_total,)`. No ordering or contiguity constraint; sweep ids can repeat and can appear out-of-order. |
| `LidarData.points_align_index` | Optional `np.uint16`, shape `(P_total,)`. Every entry must satisfy `0 <= v < N`. No ordering or contiguity constraint across points. |
| `LidarMetadata.extrinsics` | Optional `SensorExtrinsics`; existing 4x4 `float64` validators apply ([SensorExtrinsics](#ref-sensor-extrinsics)). |
| `LidarMetadata` cross-field | `motion_compensated=True` requires `reference_frame != "sensor"`. `motion_compensated=False` requires `reference_frame in {"sensor", "rig"}` — `"world"` / `"map"` are rejected because reaching a fixed global frame is mathematically equivalent to motion compensation and should be declared as such. |
| `EgoTrajectory.positions_m` | `np.float64`, shape `(N, 3)`, finite values. `N` matches the length of `align_timestamps_ns`. |
| `EgoTrajectory.orientations_xyzw` | `np.float64`, shape `(N, 4)`, finite unit XYZW quaternions. `N` matches the length of `align_timestamps_ns`. |
| `EgoTrajectory.pose_valid` | `np.bool_`, shape `(N,)`. `False` rows must use a zero position and identity XYZW quaternion placeholder. |
| `EgoTrajectory.frame` | Non-empty `str`; typically `"world"`, `"map"`, or `"odom"`. |

Follow the existing pattern from `CameraData`: attach shared length-match
validation to the last required field so that every required `points_*`
array and every present optional `points_*` array have been bound by the
time the validator runs; expose read-only views via `as_readonly_view`
without mutating caller-owned arrays.

Per-point `xyz` finite-value validation rejects NaN and Inf to keep the
contract simple. Sources that emit "no return" placeholders (some Velodyne
streams) should drop those points at parser time rather than encoding them as
NaN in `LidarData`.

## Related Structures

`LidarSensor` is out of scope for this note. It is expected to be a future
addition under `cosmos_curator/core/sensors/sensors/` that parses MCAP, ROS
bag, or vendor-native LiDAR streams and produces `LidarData` payloads. A
`McapLidarSensor` would be the most likely first concrete implementation,
mirroring `McapCameraSensor`.

A future `RangeImageData` sibling payload may be added as an optional field
on `LidarData` for sensors and consumers that need to preserve the
range-image topology. See [D4](#d4-range-image-representation).

`EgoTrajectory` is part of this design (not a future addition). It is a
`SensorData`-compliant container for the rig-to-global trajectory with
the usual `(N,)` shape. See [D2](#d2-ego-pose-placement). `LidarData` and
`EgoTrajectory` both expose `(N,)` alignment arrays, so they sit in the
same `AlignedFrame` cleanly — no bundle changes required. Other future
sensors (radar, additional LiDARs at the same `(N,)` granularity) read
`frame.sensor_data["ego"]` without any per-sensor refactor; multiple
trajectory sources coexist under different keys (e.g., `"ego_gnss_imu"`
and `"ego_slam"`).

A future motion-compensation helper (transform a
`motion_compensated=False, reference_frame="sensor"` `LidarData` into a
`motion_compensated=True, reference_frame="rig"` one, using an
`EgoTrajectory` from the same `AlignedFrame`) belongs alongside the data
classes but is not part of them.

A future **grid-binning / re-alignment helper** (map an existing
`LidarData`'s `align_timestamps_ns` onto a different target grid — for
example, a camera's frame times) also belongs alongside the data
classes. Parsers ship the source's natural alignment moments; consumers
that need a different alignment invoke this helper explicitly. For a
`motion_compensated=False` batch, the helper produces a new `LidarData`
with re-labelled `align_timestamps_ns` and `sensor_timestamps_ns` at
the new `N`; `points_timestamps_ns` and the flat per-point payload
are never touched. For a `motion_compensated=True` batch, the same
re-labelling would silently invalidate the compensation anchors stored
in `align_timestamps_ns[i]` — so the helper must either reject such
input or additionally re-warp `points_xyz` onto the new anchors using
an `EgoTrajectory`, which turns it into a re-compensation operation
rather than a pure label rewrite.

Do not introduce a separate `RawLidarData` or `UndecodedLidarPayload` type
until concretely needed; matches the precedent from
[ImuData](#ref-imu-data).

## Implementation Status

The core data-model portion of this proposal is implemented:

1. `cosmos_curator/core/sensors/data/lidar_data.py` provides attrs-based
   `LidarData` and `LidarMetadata` classes.
2. `cosmos_curator/core/sensors/data/egotrajectory_data.py` provides the
   `EgoTrajectory` `SensorData` class.
3. `cosmos_curator/core/sensors/utils/validation.py` provides reusable
   validators, including `nondecreasing_int64_array` and
   `unit_quaternion_batch`.
4. Tests cover the data contracts under
   `tests/cosmos_curator/core/sensors/data/test_lidar_data.py` and
   `tests/cosmos_curator/core/sensors/data/test_egotrajectory_data.py`.

Concrete sensor readers and motion-compensation or re-alignment helpers remain
future work as described below.

## Open Follow-Up Questions

- Should the first `LidarSensor` implementation source data from MCAP, a
  vendor SDK (Ouster, Velodyne), a proprietary AV SDK, or a static dataset
  reader (KITTI, nuScenes, Waymo)? The choice affects which timestamp
  conventions and which optional fields are exercised first.
- When per-point timestamps must be synthesized from a sweep-level
  timestamp, is uniform interpolation across an estimated sensor acquisition duration
  acceptable to downstream consumers, or do they require
  sensor-model-specific timing tables?
- Should `LidarData` carry a per-batch `frame_id` string (`"lidar_top"`,
  `"lidar_left"`, ...) for multi-LiDAR rigs, or is that pipeline-task
  metadata?
- What sentinel handling, if any, is needed for missing returns? Current
  recommendation is "drop at parser time"; revisit if downstream stages need
  to know which laser/azimuth combos failed.
- Will the first non-AV consumer (e.g., robotics, indoor scanning) require
  a different `reference_frame` vocabulary than the AV-typical
  `"sensor"`/`"rig"`/`"world"`/`"map"` set?
- What ordering does `points_return_index` denote — first-temporal,
  strongest, or producer-defined? At least one reviewed proprietary AV SDK
  distinguishes return **type** (first / last / strongest / variants
  thereof) from return **index** explicitly. Either pick one canonical
  ordering and document it, or add a per-batch field declaring which
  convention the producer used.
- Should `LidarMetadata.extrinsics` carry the nominal (as-designed)
  transform, the calibrated (corrected) transform, or both? Some
  calibration systems (including the reviewed proprietary AV SDK) separate
  a nominal sensor-to-rig transform from a calibrated one; the current
  single-matrix slot has to pick one (likely the calibrated one when
  available) and loses the nominal fallback.
- Validity policy: keep "drop invalid points at parser time" as the default,
  or require parsers to populate the `points_validity` mask when the source
  carries per-point invalid / blockage flags? The latter preserves more
  information for blockage / weather analysis but enlarges the per-batch
  payload.
- Reusing [`find_closest_indices`](#ref-find-closest-indices) for the
  point → align-row recovery: the helper requires strictly increasing
  inputs on both sides, but `points_timestamps_ns` is only
  non-decreasing (LiDAR can return simultaneous points). Options:
  relax the grid-side check in place, add a variant that accepts a
  non-strictly-increasing grid, or leave the helper untouched and
  inline the equivalent `searchsorted` + `np.clip` + nearest pick at
  the `LidarData` recovery call sites. Pick before the implementation
  MR so the parser and any downstream helpers agree on one primitive.
- Should `EgoTrajectory.frame` use the same vocabulary as
  `LidarMetadata.reference_frame` (`"sensor"` / `"rig"` / `"world"` /
  `"map"`) or a different set (e.g., adding `"odom"`)? They serve
  different purposes — points' frame vs trajectory's target frame — so
  the natural sets aren't necessarily the same.

## References

- <a id="ref-sensor-data"></a>`SensorData` protocol:
  `cosmos_curator/core/sensors/data/sensor_data.py`
- <a id="ref-aligned-frame"></a>`AlignedFrame`:
  `cosmos_curator/core/sensors/data/aligned_frame.py`
- <a id="ref-camera-data"></a>`CameraData`:
  `cosmos_curator/core/sensors/data/camera_data.py`
- <a id="ref-motion-vector-data"></a>`MotionVectorData` /
  `MotionVectorFrameData` (camera ragged side-data precedent):
  `cosmos_curator/core/sensors/data/camera_data.py`
- <a id="ref-sensor-extrinsics"></a>`SensorExtrinsics`:
  `cosmos_curator/core/sensors/data/extrinsics.py`
- <a id="ref-imu-data"></a>`ImuData` design note:
  `docs/curator/design/sensor-library-imu-data.md`
- <a id="ref-validation-helpers"></a>Sensor validation helpers:
  `cosmos_curator/core/sensors/utils/validation.py`
- <a id="ref-find-closest-indices"></a>`find_closest_indices` helper
  (searchsorted + clip nearest-index primitive used elsewhere in the
  sensor library):
  `cosmos_curator/core/sensors/sampling/sampler.py`
- <a id="ref-ros-pointcloud2"></a>ROS 2 `sensor_msgs/msg/PointCloud2`:
  <https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/PointCloud2.html>
- <a id="ref-ros-rep-103"></a>ROS REP 103: Standard Units of Measure and
  Coordinate Conventions: <https://ros.org/reps/rep-0103.html>
- <a id="ref-nuscenes"></a>nuScenes data format:
  <https://www.nuscenes.org/nuscenes#data-format>
- <a id="ref-waymo"></a>Waymo Open Dataset:
  <https://github.com/waymo-research/waymo-open-dataset>
- <a id="ref-kitti"></a>KITTI Vision Benchmark Suite:
  <https://www.cvlibs.net/datasets/kitti/>
- <a id="ref-ouster-sdk"></a>Ouster SDK (LidarScan, range-image and
  point-cloud APIs): <https://github.com/ouster-lidar/ouster_sdk>
- <a id="ref-pcl"></a>Point Cloud Library (point types and PCD format):
  <https://pointclouds.org/documentation/>
- <a id="ref-openpcdet"></a>OpenPCDet (per-dataset loaders normalizing
  KITTI / nuScenes / Waymo / Lyft into a common in-memory layout):
  <https://github.com/open-mmlab/OpenPCDet>
- <a id="ref-proprietary-av-sdk"></a>Proprietary AV SDK: reviewed against
  the SDK's public types covering the decoded-packet container, per-point
  Cartesian and spherical representations, sensor properties, return-type
  enumeration, auxiliary per-point data types, invalidity-reason
  enumeration, decoded point-cloud wrapper, motion-compensation
  sub-struct, point-cloud layout mapping, and coordinate-frame
  enumeration.
