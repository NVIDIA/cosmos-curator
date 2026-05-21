# Cosmos Curator Sensor Library Interface Design

A Pythonic sensor library for reading video, IMU, GPS, lidars, radar, and other sensors that supports streaming through data to create aligned frames. Designed for cars and robots with arbitrary sensor combinations.

This library separates container file formats and packet formats into modular components, and aligns data cross sensors.

All components are meant to be modular and replacable, so that the library can be adapted as needed.

For example: this library does not prescribe a specific schema for storing sensor extrinsics on disk. Instead, it defines a protocol for returning a 4x4 matrix of a sensor's extrinsics, allowing for custom parsers to load the data from disk.

End users can adapt this library to their existing data, as opposed to adapting their data to this library.

## How Sensor Data Tends to be Stored

AV and robotics data files tend to be stored in files with these characteristics:

- Time-indexed container files
- Packets of binary data inside of the container files

Examples of containers:

- MP4 or fMP4 (Standard ISO-based multimedia wrapper)
- MCAP (file format defined at [https://mcap.dev](https://mcap.dev))

Examples of packets:

- h264, h265, av1, vp9 encoded video packets (also called NAL units)
- Lidar binary data
- IMU binary data
- GPS binary data

## Design Principles

- **Timestamp-first**: Timestamps are the canonical identifier. Only expose sane frame indices after alignment.
- **Streaming**: Iterate over data without loading entire files into memory.
- **Modular architecture**: Extensible, modular architecture that allows for users of this library to use what they need, and allows for adding proprietary components as separate modules.
- **Arbitrary sensor combinations**: Support any mix of cameras, lidars, IMU, GPS, radar, ultrasonics, etc. Frames may omit sensors (e.g., IMU + lidar only).
- **Format-agnostic**: Binary data loaders are modular and pluggable.
- **SoA data**: Data classes for sensor data contain structures of arrays for better performance.
- **Self-contained:** cosmos-curator relies on this library, not the other way around. Goal is maximum reusability.

### Timestamp-First Design

**Avoid sensor-specific frame indices.** When a camera drops a frame, its frame indices no longer align with other cameras—e.g., primary frame 30 and left frame 29 may both correspond to t=1.0s. Downstream consumers that assume "index N = same moment across cameras" get off-by-one or off-by-few errors.

**Use timestamps as the canonical identifier.** Each `AlignedFrame` carries `align_timestamps_ns` `(N,)` for the batch and per-sensor `sensor_timestamps_ns` inside each payload (e.g. `frame[sensor_id]`). Downstream code aligns on those arrays, not on opaque frame indices. For stream position (e.g., progress bars), use `enumerate(sensor_group.sample(spec))` with a **`SamplingSpec`** (see below).

### Streaming Design

**Reduce memory usage.** Iterate over data without loading entire files into memory. Vendor-specific binary formats (e.g., lidar) can be 3–4× smaller before parsing. Loading a 70 GB lidar file and expanding it by 4× yields ~280 GB for a single file—often infeasible. Streaming keeps only a bounded working set in memory.

**Unlock parallelism.** Lower memory pressure allows more workers or processes to run concurrently on the same hardware, improving throughput for large datasets.

### Modular Architecture

This library:

- Ships with support for mp4 and mcap
  - But does not prescribe specific container or binary packet formats
  - End users can bring their own container and binary packet formats
- Supports sampling by timestamps on a grid
  - But also supports sampling using arbitrary timestamps
- Uses sampling to determine:
  - nearest neighbor data point
  - points in time bands
  - But can support arbitrary methods of using timestamp sampling to align data

## Architecture

```text
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Sensors:                                                                   │
  │  - one per data source; composed of                                         │
  │    - TimeIndexedContainer                                                   │
  │    - BinaryPacketParser                                                     │
  │  - Implemented as Sensor(                                                   │
  │        TimeIndexedContainer,                                                │
  │        BinaryPacketParser,                                                  │
  │        Intrinsics,                                                          │
  │        Extrinsics)                                                          │
  │                                                                             │
  │  CameraSensor │ ImuSensor │ GpsSensor │ LidarSensor │ ...                   │
  │  - start_ns, end_ns                                                         │
  │  - sample(spec) → Generator[SensorData] (SoA, batch dim N per yield)        │
  └─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  SensorGroup                                                                │
  │  - sensors: dict[str, Sensor]                                               │
  │  - start_ns, end_ns (from sensor bounds)                                    │
  └─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  sensor_group.sample(spec) → Generator[AlignedFrame]                        │
  │  - ``spec: SamplingSpec`` has ``.grid`` and optional ``.policy``            │
  │  - Each step: ``window`` from ``for window in spec.grid``, (N,)             │
  │  - Each sensor: next batch from ``sensor.sample(spec)`` (same spec)         │
  │  - ``AlignedFrame(align_timestamps_ns=window.timestamps_ns,``               │
  │    ``sensor_data={id → …})``                                                │
                                          │
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  AlignedFrame (batch, SoA)                                                  │
  │  - align_timestamps_ns: (N,) alignment timestamps (= active window)         │
  │  - sensor_data: dict[str, SensorData] e.g. CameraData (SoA, len=N)          │
  └─────────────────────────────────────────────────────────────────────────────┘
```

## Alignment and Sampling Contract

### Alignment Contract

Cross-sensor alignment is defined by shared reference timestamps, not by shared
source-sample counts.

In this section, a source sample means a decoded source measurement before
alignment or aggregation. It does not mean raw vendor bytes, container packets,
or encoded payloads.

For a given window:

- every sensor is sampled against the same reference timestamps
- each output row corresponds to one reference timestamp
- each sensor payload records the source sensor timestamp or timestamps actually
  used to build that row

Downstream consumers should read an aligned batch as:

- “these sensor payload rows correspond to these reference timestamps”

not:

- “these are all source samples that happened to occur during this
  wall-clock interval”

### Reference-Grid Sampling Contract

`SamplingGrid` defines the canonical reference timeline for alignment. It is
the source of truth for which timestamps downstream code cares about, and it
also defines batching.

Sampling and alignment are related but not identical:

- sampling is the mechanism that defines half-open windows and selects data for
  reference timestamps
- alignment is the result of applying that sampling mechanism so sensor payload
  rows line up on a shared reference timeline

A yielded `SamplingWindow` should be interpreted as one sampling interval plus
the active alignment timestamps that belong to it, not as a promise about how
many source samples exist in that wall-clock span.

For one window:

- `window.timestamps_ns` are the active reference timestamps for the
  batch
- `window.start_ns` and `window.exclusive_end_ns` define the half-open sampling interval
  `[start_ns, exclusive_end_ns)`
- batch shape is determined by `len(window)`, not by the number of sensor
  samples inside that time span

Sensors are then sampled onto those reference timestamps.

For each reference timestamp in `window.timestamps_ns`, a sensor applies its own
sampling rule to produce the payload aligned to that reference timestamp. The
sampling rule is sensor-specific:

- Cameras typically use nearest-neighbour frame selection.
- Lidar may collect, interpolate, or aggregate rays near the reference
  timestamp.
- IMU may select decoded source samples, interpolate, or preintegrate over a
  band around the reference timestamp. The first generic IMU data structure,
  `ImuData`, represents point samples rather than preintegrated windows; see
  `cosmos_curator/core/sensors/data/imu_data.py` and the design rationale in
  `sensor-library-imu-data.md`.
- GPS/GNSS may select the nearest decoded fix, meaning the receiver's computed
  position solution at one point in time, or interpolate between decoded fixes.
  GNSS is the broader satellite-positioning category that includes GPS; WGS-84
  is the global latitude/longitude/ellipsoid-altitude coordinate reference
  system commonly used by GPS/GNSS receivers. The first generic GPS data
  structure, `GpsData`, represents WGS-84 fix rows rather than map-projected
  localization output. See
  `sensor-library-gps-data.md`.

This has several important consequences:

- Source sensor timestamps that fall inside the wall-clock span of the window
  may be unused.
- Multiple reference timestamps may map to the same source sample.
- A sensor may use one source sample, many source samples, or an interpolation
  of source samples to produce one aligned payload row.
- Equal-sized aligned batches come from the reference grid, not from source
  sample counts.

This is intentional. The goal is to project heterogeneous, jittery, drifting
sensor timelines onto a clean shared reference grid in a resampling model,
rather than simply returning all source samples that fall inside each window.

### Half-Open Window Semantics

Sampling windows use the half-open interval `[window.start_ns, window.exclusive_end_ns)`.

This convention exists to define batch boundaries and avoid double-counting at
window edges. A timestamp exactly equal to `window.exclusive_end_ns` belongs to
the next window, not the current one.

The half-open interval should not be interpreted to mean that every source
sample inside that interval must appear in the sampled output. A source sample
may be unused if it is not selected by the sensor’s sampling rule for any
reference timestamp in `window.timestamps_ns`.

## Sampling Data by Timestamp

Two ways of sampling data by timestamp are supported:

- nearest_neighbor: for each reference timestamp, pick the closest source
  sample
- time_spans: for each reference timestamp, derive a time span and collect or
  aggregate source samples inside that span
  - Auto-generated time spans from a list of reference timestamps
    - center: map each t_i such that t_i is centered between [t_i - 1, t_i + 1]
    - next: map each t_i -> [t_i, t_i+1)
    - prev: map each t_i -> [t_i-1, t_i)

### nearest_neighbor example

Let's say we want to sample on a timestamp grid:

```text
|---------|---------|---------|
t0        t1        t2        t3
```

In real life, there's timestamp jitter, drift and drop, so timestamps may arrive at

```text
-|-------|-------------------|
 t0'     t1'                t2'
```

Reference timestamps are the primary object. Source timestamps are projected onto
that reference grid. In this example, `t2'` may be reused if it is the closest
sample for more than one reference timestamp, and some source timestamps may be
unused if they are not the closest sample for any reference timestamp.

For example, if the nearest-neighbour matches land that way, the sampled data
stream could be:

```text
[t0', t1', t2', t2']
```

### time_span example

Lidar data is usually collected at 10hz. However, that 1/10th of a second could have hundreds of thousands of points, each 1e-6 seconds apart.

Let's say we want to sample on a timestamp grid:

```text
|---------|---------|---------|
t0        t1        t2        t3
```

We may not want to sample single data points at t0, t1, t2, and t3. There
could be 1e5 data points between t0 and t1. Instead, for each reference
timestamp we may want to collect or aggregate the data points that are
**around** t0, t1, t2, and t3.

Example of the time_spans:

centered

```text
|---------|---------|---------|
t0        t1        t2        t3
|<-->|<------->|<------->|<-->|
```

Returns data points centered around the sample timestamp

next:

```text
|---------|---------|---------|
t0        t1        t2        t3
|<------->|<------->|<------->|
```

Returns the data points between `[t_i, t_i+1)`

previous:

```text
|---------|---------|---------|
t0        t1        t2        t3
|<------->|<------->|<------->|
```

Returns the data points between `[t_i-1, t_i)`

Under the hood, `next` and `previous` share the same code path, it's just a matter of timestamp ordering.

These sampling methods are sensor specific.

For example, CameraSensor will often use nearest_neighbor, while lidar and imu
may use time_span or other aggregation / interpolation rules.

Lidar could use any of the time_span methods, while imu may prefer `previous`
or another causal integration rule.

## SamplingGrid, SamplingPolicy, and SamplingSpec

The library splits **what** to sample (reference timeline + **windowing**) from **how strictly** to enforce alignment (tolerances, overlap, and future QC rules).

### SamplingGrid

A **`SamplingGrid`** holds the sorted reference **`timestamps_ns`** array plus **`stride_ns`**, and **`duration_ns`**.

Iteration advances window interval starts ``first + k * stride_ns`` for ``k = 0, 1, …`` while that start is ``<= last``, so the final sample is always included in a **window** when it falls on that schedule.

``for window in grid`` yields **``numpy`` ``int64``** views into the parent ``timestamps_ns`` for each **window**. Windows may overlap in time when ``stride_ns < duration_ns``, so index ranges into the buffer may overlap.

A **`SamplingGrid`** holds a half-open timeline (`start_ns`, `exclusive_end_ns`), the active reference `timestamps_ns`, and the window geometry
(`stride_ns`, `duration_ns`).

Iteration advances nominal window starts `start_ns + k * stride_ns` for  `k = 0, 1, …` while the start is `< exclusive_end_ns`.

``for window in grid`` yields **``SamplingWindow``** objects whose ``window.start_ns`` and ``window.exclusive_end_ns`` are the nominal half-open
bounds of that window, and whose `window.timestamps_ns` are the active reference timestamps that fall within those bounds.

### SamplingPolicy

A **`SamplingPolicy`** holds **alignment and quality parameters** that are not intrinsic to the reference timeline — for example:

- **`tolerance_ns`** — maximum allowed time delta between a reference timestamp and the chosen canonical sample (per-sensor or global; exact semantics are implementation-defined).
- **`sensor_overlap`** — minimum temporal overlap across sensors within a **window** (or similar multi-sensor gate).

It does **not** own ``timestamps_ns`` or window geometry from **`SamplingGrid`**. New fields (max gap, fail vs skip window, per-sensor maps) can extend this type without changing **`SamplingGrid`**.

A **`SamplingSpec`** may omit policy entirely. Use **`policy=None`** when no sampling policy should be applied.

### SamplingSpec

A **`SamplingSpec`** bundles **`grid: SamplingGrid`** and optional **`policy: SamplingPolicy | None`**. It is the **only** argument type for **`sensor.sample(spec)`** — there is no separate overload for bare **`grid`** / **`policy`**.

Internals may read **`spec.grid`** and **`spec.policy`**; CLIs and configs pass a single **`spec`**.

### Batch contract: each window defines `N`

Concrete sensors implement **`sample(spec)`** as a **generator**. For each
**window**, ``for window in spec.grid`` produces a 1-D **`SamplingWindow`**.
Under the half-open contract:

- ``window.timestamps_ns`` are the active reference timestamps for the batch
- ``window.start_ns`` is the inclusive left boundary
- ``window.exclusive_end_ns`` is the exclusive right boundary
- the batch size is **`N = len(window)`**

- **Reference times** — Row ``i`` is requested at reference time
  ``window.timestamps_ns[i]``. For ``CameraData``, ``align_timestamps_ns``
  should match those active reference times.
- **Sensor times** — ``sensor_timestamps_ns`` has shape ``(N,)``;
  ``sensor_timestamps_ns[i]`` is the actual capture / message time chosen
  for row ``i`` (for example, the nearest neighbour to the active reference
  timestamp in the source stream).
- **Payloads** — For cameras, ``frames`` has shape ``(N, H, W, 3)`` for RGB
  (see ``CameraData`` in ``camera_data.py``). **``H``** and **``W``** come from
  the decoded stream or container metadata; for MCAP ``rgb8`` topics aligned
  with ``make_mcap_from_mp4``, **``H``** and **``W``** are read from the
  channel metadata on that topic.

So **`len(window)`**, **`len(align_timestamps_ns)`**,
**`len(sensor_timestamps_ns)`**, and **`frames.shape[0]`** are the same
**`N`** for each yield. Supersampling (denser reference grid than the source
rate) **duplicates rows** when multiple reference times map to the same source
sample; subsampling uses a smaller **`N`** per window because the active
reference portion of **`window.timestamps_ns`** contains fewer points.

Aligned batches in an **`AlignedFrame`** use this same **`N`** for every sensor present in that frame so rows stay time-aligned across sensors.

## Proposed High-Level API

This proposed example shows how the library can be used to align data from
multiple sensors.

```python

# Containers
cam0_ct = Mp4Container(cam0_file)
cam1_ct = Mp4Container(cam1_file)
imu0_ct = McapContainer(imu0_file)
gps0_ct = McapContainer(gps0_file)
lidar0_ct = McapContainer(lidar0_ct)

# Parsers - decodes packets inside the container
cam0_parser = VideoParser(cam0_ct.metadata)
cam1_parser = VideoParser(cam1_ct.metadata)
imu0_parser = ImuParser()
gps0_parser = GpsParser()
lidar0_parser = HesaiLidarParser()

# Intrinsics
cam0_int = CameraIntrinsics(parser=CameraIntrinsicsParser(rig_metadata))
cam1_int = CameraIntrinsics(parser=CameraIntrinsicsParser(rig_metadata))

# Extrinsics
cam0_ext = CameraExtrinsics(parser=CameraExtrinsicsParser(metadata))
cam1_ext = CameraExtrinsics(parser=CameraExtrinsicsParser(metadata))
lidar0_ext = LidarExtrinsics(parser=LidarExtrinsicsParser(metadata))

# Sensors
cam0 = CameraSensor(cam0_ct, cam0_parser, cam0_int, cam0_ext)  # camera data collection rate is 30 Hz
cam1 = CameraSensor(cam1_ct, cam1_parser, cam0_int, cam0_ext)  # camera data collection rate is 30 Hz
imu0 = ImuSensor(imu0_ct, imu0_parser)  # imu collection rate is 100hz
gps0 = GpsSensor(gps0_ct, gps0_parser)  # gps collection rate is 1hz
lidar0 = LidarSensor(lidar0_ct, lidar0_parser, lidar0_ext)  # lidar collection rate is 10hz

# Group sensors into a group
sensor_group = SensorGroup(sensors={
    "cam0": cam0,
    "cam1": cam1,
    "imu0": imu0,
    "gps0": gps0,
    "lidar0": lidar0
})

# Reference timeline + windowing (reusable across groups)
start_ns, exclusive_end_ns, timestamps = make_ts_grid(sensor_group.start_ns, sensor_group.end_ns, 30.0)
grid = SamplingGrid(
    start_ns=start_ns,
    exclusive_end_ns=exclusive_end_ns,
    timestamps_ns=timestamps,  # reference timestamps
    stride_ns=10 * 1_000_000_000,   # time between window interval starts
    duration_ns=10 * 1_000_000_000,  # query interval width on the timeline (paired with stride)
)

# Alignment / quality rules (always paired with a grid in a SamplingSpec)
policy = SamplingPolicy(
    tolerance_ns=5_000_000,       # e.g. max |ref − canonical| for a matched sample (5 ms)
    sensor_overlap=0.99,          # e.g. min fraction of a window where all sensors overlap
)

spec = SamplingSpec(grid=grid, policy=policy)  # optional policy; sole handle for sample / align

# Iterate over the group, yielding aligned frames
# Each frame will contain 1 or more sensor sources
# Data from each sensor source will be aligned to the same reference timestamps
for frame in sensor_group.sample(spec):
    frame.align_timestamps_ns  # (N,) active alignment timestamps for this batch
    frame["cam0"]   # CameraData: frames (N, H, W, 3) uint8 RGB
    frame["cam1"]   # CameraData: same layout; H, W from stream metadata
    frame["imu0"]   # ImuData, SoA, point samples aligned to the reference grid
    frame["gps0"]   # GpsData, SoA, WGS-84 fixes selected or interpolated
    frame["lidar0"]  # LidarData, rays aggregated or bucketed around the same reference timestamps
```

### Unit conventions

On **`SamplingGrid`**, `start_ns`, `exclusive_end_ns`, `stride_ns`, and
`duration_ns` are `int` **nanoseconds**; `timestamps_ns` uses the same unit
— aligned with MCAP's `log_time` / `publish_time`
(uint64 on the wire) convention. On **`SamplingPolicy`**, `tolerance_ns` and
similar fields use the same nanosecond unit so comparisons to alignment and
sensor times stay exact.

### Dtype conventions

At the public API boundary, timestamp arrays are expected to use
`np.int64`. Boolean flag arrays (for example packet masks such as
`is_keyframe` / `is_discard`) use `np.bool_`. RGB frame tensors use
`np.uint8`.

These dtypes are enforced at key public boundaries such as `SamplingGrid`,
`VideoIndex`, and `CameraData` so malformed arrays fail early with a clear
error rather than being silently coerced downstream.

### DataSource contract

`DataSource` is `Path | bytes | io.BufferedIOBase` (see
`cosmos_curator/core/sensors/types/types.py`). The library is intentionally
backend-agnostic: it does **not** accept URIs and does **not** import any cloud
client. Callers wrap any non-local backend as a `BinaryIO` themselves and pass
it in as the `io.BufferedIOBase` arm.

- **`Path`**: a local filesystem path. The library opens it in `"rb"` mode and
  owns the resulting handle for the duration of the call (closes it on exit).
  Sensors may open or rewind their input source multiple times across metadata
  loading, indexing, and sampling without any caller cooperation.

- **`bytes`**: an owned byte buffer. Wrapped in a fresh `io.BytesIO` per call,
  so multi-phase sensors (e.g. `CameraSensor` index build + `.sample()`) reuse
  the same bytes safely.

- **`io.BufferedIOBase`**: a caller-owned, **readable** and **seekable** binary
  stream. The library treats it as an absolute-offset random-access buffer:
  each public sensor entry point positions the stream at offset 0 before
  handing it to PyAV / PIL and the underlying decoder performs absolute seeks
  within the stream as it pleases. The library makes no promises about the
  stream's position before, during, or after use, and does **not** restore the
  caller's position on exit. Concurrent / overlapping use of one stream by
  multiple readers (e.g. driving two sensors over the same `BinaryIO`
  simultaneously) is unsupported. If the caller's "real data" lives at a
  non-zero offset within a larger buffer, wrap it in a slicing adapter that
  presents absolute offset 0 as the data origin.

Non-readable or non-seekable buffered streams are rejected with a clear
`ValueError` at `open_data_source`. Single-pass network streams (e.g. raw
`urllib.urlopen` responses, `boto3 StreamingBody`) must be buffered into a
seekable stream or downloaded to a `Path` / `bytes` first.

## Components

- **Containers**: Time indexed container files are the only supported container format, like MCAP, or MP4.
- **SamplingGrid**: Half-open timeline (`start_ns`, `exclusive_end_ns`) and active `timestamps_ns` plus `stride_ns` / `duration_ns`; iterable in **window** order (`for window in grid`). Each **`window`** is a `SamplingWindow` with `start_ns`, `exclusive_end_ns`, and `timestamps_ns` `(N,)`. Drives batch shape **`N`** per yield. Held by a **`SamplingSpec`**; not passed alone to **`sample`** / **`align`**.
- **SamplingPolicy**: Alignment and QC parameters (`tolerance_ns`, etc.); **no** timeline geometry. May be attached to a **`SamplingSpec`**.
- **SamplingSpec**: **Required** argument to **`sensor.sample(spec)`**. Always contains **`grid: SamplingGrid`** and may also carry **`policy: SamplingPolicy | None`**. Single object threaded through sampling and alignment.
- **Binary Parsers**: Parse binary data, like vendor-specific lidar data, or h265 encoded video packets, into easily handled Pythonic data formats like CameraData, LidarData, ImuData, etc.
- **Sensors**: One per data source (CameraSensor, ImuSensor, GpsSensor, LidarSensor, etc.). Expose `start_ns`, `end_ns`, and **`sample(spec)`** — a generator over **`spec.grid`** **windows**. Each yield returns SoA with batch dimension **`N`**. Uses **`spec.policy`** when validating matches (e.g. tolerance). Each sensor type implements its own sampling strategy (nearest, interpolate, preintegrate, bucket).
- **SensorGroup**: Group of sensors. Holds `sensors`; provides `start_ns`, `end_ns` from sensor bounds. Call `.sample(spec)` to iterate.
- **SensorGroup.sample()**: Takes **`spec: SamplingSpec`**. Walks **`spec.grid`** in window order (`for window in spec.grid`). For each **`window`**, `window.timestamps_ns` is the alignment batch; each sensor's **`sample(spec)`** is advanced in lockstep so every payload has batch dimension `N`. Uses **`spec.policy`** for cross-sensor checks (e.g. minimum temporal overlap, tolerance). Assembles `AlignedFrame(align_timestamps_ns=window.timestamps_ns, sensor_data=…)`.
- **AlignedFrame**: Batch of timestamp-aligned data (SoA). Fields: `align_timestamps_ns` `(N,)` (that window’s active alignment times from the `SamplingGrid`), and `sensor_data: dict[str, SensorData]` (e.g. `CameraData` with `align_timestamps_ns`, `sensor_timestamps_ns`, `frames` each of length `N`). Index with `frame[sensor_id]` or `frame.sensor_data[sensor_id]`. See `cosmos_curator/core/sensors/data/aligned_frame.py`.
- **ImuData**: `cosmos_curator/core/sensors/data/imu_data.py` implements the
  first generic IMU payload as SoA point samples with required angular velocity
  and linear acceleration vectors plus optional orientation, covariance,
  validity, host timestamp, sequence, and temperature fields. The design
  rationale in `docs/curator/design/sensor-library-imu-data.md` also documents
  why preintegrated windows and ragged source-sample windows should remain
  separate future structures.
- **GpsData**: The first generic GPS/GNSS payload is a SoA batch of
  WGS-84 fix rows with required latitude, longitude, altitude, and position
  validity plus optional covariance, velocity, fix type, satellite count,
  accuracy, dilution of precision, host/UTC timestamps, and sequence fields.
  See `docs/curator/design/sensor-library-gps-data.md`.
- **Cloud sources**: The sensor library does **not** accept URIs. Callers feed
  data from S3, Azure Blob, GCS, or any other remote store by opening their
  own seekable `BinaryIO` (e.g. via `smart_open.open(uri, "rb",
  transport_params=...)`, `boto3.client("s3").get_object(...)["Body"]` buffered
  through `io.BytesIO`, or the equivalent for Azure / GCS) and passing it as
  the `io.BufferedIOBase` arm of `DataSource`. Header-only indexing
  (`make_index_and_metadata(..., FROM_HEADER)`) continues to work over any
  seekable cloud-backed stream because PyAV uses absolute seeks in the AVIO
  layer. A minimal caller-side pattern using `smart_open`:

  ```python
  import smart_open
  from cosmos_curator.core.sensors.sensors.camera_sensor import CameraSensor

  with smart_open.open("s3://bucket/clip.mp4", "rb", transport_params={"client": boto3_s3}) as stream:
      sensor = CameraSensor(stream)
      for batch in sensor.sample(spec):
          ...
  ```

  See
  [`docs/curator/guides/sensor-library-cloud-storage.md`](../guides/sensor-library-cloud-storage.md)
  for usage (`BinaryIO` wrapping patterns for S3 / Azure, code example) and
  [`docs/curator/design/sensor-library-cloud-storage.md`](sensor-library-cloud-storage.md)
  for measured numbers, parity validation, and current limitations.

## Module Layout

```text
cosmos_curator
└── core
    └── sensors
        ├── containers
        │    ├── __init__.py
        │    ├── mp4.py            # Mp4Container
        │    └── mcap.py           # McapContainer
        ├── parsers
        │    ├── __init__.py
        │    ├── video.py          # VideoParser
        │    ├── hesai.py          # HesaiLidarParser
        │    ├── intrinsics.py     # Generic intrinsics parsing
        │    └── extrinsics.py     # Generic extrinsics parsing
        ├── sensors
        │    ├── __init__.py
        │    ├── camera.py         # CameraSensor, CameraIntrinsics
        │    ├── imu.py            # ImuSensor
        │    ├── gps.py            # GpsSensor
        │    ├── lidar.py          # LidarSensor
        │    └── group.py          # SensorGroup
        ├── sampling
        │    ├── __init__.py
        │    ├── grid.py           # SamplingGrid (reference timeline + window iteration)
        │    ├── policy.py         # SamplingPolicy (tolerance, overlap, alignment QC)
        │    ├── spec.py           # SamplingSpec(grid, optional policy) — required for sample / align
        │    └── sampler.py        # nearest_neighbor, time_spans
        └── data
             ├── __init__.py       # Structure-of-Arrays (SoA) data structures
             ├── camera.py         # CameraData, MotionVectorData, CameraIntrinsics
             ├── gps_data.py       # GpsData
             ├── lidar.py          # LidarData, LidarRays
             ├── aligned_frame.py  # AlignedFrame
             ├── extrinsics.py     # SensorExtrinsics
             └── intrinsics.py     # CameraIntrinsics, etc.
```

## Timestamp Dtype and Units

All timestamps throughout this library — reference grids, sensor timestamps,
and `AlignedFrame` fields — are `int64` **nanoseconds**.

Represents this many seconds and hours, this is expected to be sufficiently
large, sensor sessions are expected to be less than 10 hours.

| unit        | calculation                  | total               |
| ------------| ---------------------------- | ------------------- |
| nanoseconds | 2**63 - 1                    | 9223372036854775807 |
| seconds     | (2**63 - 1) // 1_000_000_000 | 9223372036          |
| hours       | 9223372036  // 60            | 153722867           |

If int64 in nanoseconds is used to represent seconds since the unix epoch, this
timestamp will overflow on Apr 11 2262 23:47:17.

The authors have decided that this impending apocalypse is somebody else's
problem.

### Why int64 nanoseconds

MCAP and many capture pipelines use nanosecond resolution. Using int64 ns
end-to-end matches `VideoIndex` presentation times derived from packet PTS and
MCAP camera streams without truncating to microseconds.

Additional benefits:

- **Exact arithmetic** — integer subtraction and comparison have no rounding error, which matters when evaluating `tolerance_ns` constraints.
- **Useful sentinel** — `INT64_MIN` is a natural invalid-timestamp value.
- **No mixed-dtype bugs** — a single canonical unit eliminates the class of errors that arise from accidentally mixing seconds and nanoseconds.

### int64 vs uint64

MCAP uses uint64, not int64. However, this presents some significant footguns:

The underflow problem is real everywhere you compute deltas:

```python
# Looks innocent, blows up if t_ref > t for any reason
relative_ns = timestamp_ns - t_ref_ns   # uint64: wraps to ~1.84e19 instead of negative

# Sorting/alignment checks
delta = pts_ns[i] - pts_ns[i-1]         # fine if sorted, catastrophic if not
gap = window_start - sensor_start       # dangerous if sensor starts after window
```

Any place the code checks if delta < 0 or expects a negative result silently
gets a huge positive number instead.

The practical headroom argument against switching:

- Current Unix epoch in nanoseconds: ~1.75e18
- int64_max: ~9.22e18 → rolls over around year 2262
- uint64_max: ~1.84e19 → rolls over around year 2554

The gain is ~292 years of headroom. In exchange you give up negative arithmetic
everywhere. That's a bad trade.

The mcap -> sensor lib boundary strategy:

Accept uint64 at the MCAP ingestion point, cast to int64 immediately, work in
int64 throughout:

```python
# At the MCAP boundary only
ts_ns: int = int(np.int64(mcap_timestamp_uint64))
```

### The non-integer sample rate wrinkle

Common sample rates like 30 Hz do not divide evenly into nanoseconds:
1/30 s ≈ 33,333,333.33... ns. Stepping a grid in fixed integer steps
accumulates small drift over long sessions, which can violate a tight
tolerance unless the grid is built in float and rounded once.

### Resolution

`make_ts_grid` generates the grid in `float64` seconds and rounds once to int64
nanoseconds:

```python
sample_interval = 1.0 / sample_rate_hz
start = start_ns / 1_000_000_000
end = end_ns / 1_000_000_000
timestamps_s = np.arange(start, end, sample_interval, dtype=np.float64)
timestamps = np.round(timestamps_s * 1_000_000_000).astype(np.int64)
```

Maximum rounding error is 0.5 ns per sample after that single conversion.
**`SamplingGrid`** accepts `timestamps_ns` as `npt.NDArray[np.int64]`
directly and applies no further conversion.

## Proposed Method for LanceDB Integration

No LanceDB integration is provided by this library, but integration is expected
to be straightforward.

To store aligned sensor data in LanceDB, like for vector search, metadata, or
clip indexing:

1. Conversion step: Turn NumPy SoAs into a pa.Table (e.g. via pa.Table.from_pydict or column-by-column construction).
2. Write: Pass the pa.Table to LanceDB’s ingest API.
3. Read: LanceDB returns Arrow data, which can be converted back to NumPy if needed.
