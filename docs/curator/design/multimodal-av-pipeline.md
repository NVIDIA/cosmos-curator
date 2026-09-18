# Multimodal AV Pipeline Design

**Status:** Initial design. This document defines the stable AV episode
contract and an initial Ray Data runtime plan for producing aligned
multimodal Lance records on top of the sensor library.

## Purpose

Design a Cosmos Curator pipeline that converts longer autonomous-vehicle
recordings into aligned, fixed-stride Lance records suitable for durable
refinement, downstream curation stages, and optional recipe-specific exports
such as LeRobotDataset v3.

The design has one central decision:

> Each aligned multiview AV clip is one logical episode record in Lance. One
> configured output FPS defines the episode timeline, and every required
> modality is sampled or derived against that timeline using an explicit
> per-modality policy.

The sensor library remains responsible for truthful timestamped sensor reads and
alignment results. The pipeline owns clip construction, episode-relative
timestamps, Lance publication, run policy, curation outputs, and optional export
sinks. The initial runtime uses Ray Data pipeline stages directly rather than
Xenna.

This document separates the stable pipeline contract from the initial runtime
plan. The contract defines the durable episode, timing, calibration, ownership,
and failure semantics. The runtime plan describes one implementation path that
satisfies that contract.

## Table of Contents

- [Pipeline Contract](#pipeline-contract)
  - [Goals](#goals)
  - [Durable Format Target](#durable-format-target)
  - [Aligned Episode Contract](#aligned-episode-contract)
  - [Sampling Policies](#sampling-policies)
  - [Ownership Boundaries](#ownership-boundaries)
  - [Failure and Quarantine Semantics](#failure-and-quarantine-semantics)
- [Initial Runtime Plan](#initial-runtime-plan)
  - [Stage Plan](#stage-plan)
  - [Expected Runtime Outcomes](#expected-runtime-outcomes)
  - [Illustrative Runtime Configuration](#illustrative-runtime-configuration)
  - [Engineering Stage Notes](#engineering-stage-notes)
  - [Acceptance Criteria](#acceptance-criteria)

## Pipeline Contract

### Goals

- Target Lance as the default durable refinement boundary, with media and bulk
  sensor payloads stored externally and referenced from Lance.
- Treat a synchronized multiview clip as one episode, not as independent
  per-camera examples.
- Use one configured output FPS for the regular episode timeline.
- Support source sensors with different nominal rates, jitter, and missing
  observations.
- Cover multiview cameras, preintegrated IMU, and GPS/GNSS position in the
  initial design.
- Record modality-agnostic rig calibration for cameras, IMU, GPS/GNSS, and
  future sensors.
- Leave room for LiDAR without requiring initial LiDAR serialization.
- Keep sensor-library and pipeline ownership separate.
- Preserve source timing and recording provenance while exposing regular
  episode-relative timestamps to consumers.

### Non-Goals

- No code implementation in this document.
- No detailed task breakdown.
- No new sensor-library data type.
- No production LeRobot writer API wrapper selection.
- No model-specific feature recipe, token layout, or training schema.
- No LiDAR output in the initial runtime.
- No advanced recovery policy for partially valid clips.

### Open Decisions

Initial runtime decisions:

- Which multimodal function should be built first: IMU-only, IMU + GPS/GNSS, or
  an egomotion-based feature.
- Which calibration tensors should be training-visible features versus
  metadata-only fields.

Follow-on decisions:

- Which embedding model family, embedding granularity, and vector schema should
  be included in the training dataset.
- Whether downstream consumers need LiDAR in the first follow-on increment.
- Which additional curation outputs are useful after the initial runtime.

### Durable Format Target

The durable refinement boundary for this pipeline is Lance. Lance stores
tabular episode, row, provenance, curation, and media-reference records. Media
and bulk sensor payloads, such as per-camera MP4 clips, remain in the configured
object store or filesystem and are referenced from Lance rows.

The Lance contract should be inspectable before any export sink runs. It should
hold:

- episode records with source session ID, source interval, output FPS, row
  count, required sensor keys, task strings, calibration ID, config digest, and
  provenance ID;
- row-level records or nested row payloads with episode-relative timestamps,
  target source timestamps, sensor selected timestamps, tabular sensor features,
  validity flags, and alignment provenance;
- media-reference records for per-camera MP4 clip artifacts;
- curation records for free-form captions, train-of-cognition reasoning,
  embeddings, optional FST labels, and selected multimodal outputs when those
  stages are enabled;
- calibration and provenance records needed to inspect or regenerate an
  episode.

LeRobotDataset v3 is a recipe-specific export sink, not the durable refinement
boundary. A LeRobot exporter can consume Lance records and media references and
materialize `data/`, `videos/`, and `meta/` when a downstream training workflow
requires that format. Export-specific schema choices, task index mapping,
statistics, and chunk sizing belong to the exporter.

References:

- [Cosmos Curator Next](curator-next.md)
- [LeRobotDataset v3.0](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
- [Porting Large Datasets to LeRobot Dataset v3.0](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)

### Aligned Episode Contract

An episode is one fixed-stride interval over one recording session:

```text
episode = {
  episode_id,
  source_session_id,
  source_interval_ns = [start_ns, end_ns),
  output_fps,
  row_count,
  required_sensor_keys,
  calibration_id,
  tasks,
  config_digest,
  provenance_id
}
```

`required_sensor_keys` includes cameras and non-camera modalities. The sensor
configuration or manifest resolves each key to its modality type.
`tasks` is the episode's natural-language task label set. Export sinks may map
task strings to format-specific indices, such as LeRobot `task_index`, when
needed.
`config_digest` identifies the pipeline configuration used to produce the
episode. `provenance_id` links the episode to heavier audit details in Lance
provenance records or companion metadata.

Before computing episode IDs, spans, row counts, or timestamps, the runtime
normalizes clip geometry into canonical integer values:

```text
duration_ns = normalized clip.duration_s in nanoseconds
stride_ns = normalized clip.stride_s in nanoseconds
min_duration_ns = normalized clip.min_duration_s in nanoseconds
output_fps = normalized positive integer clip.output_fps
source_interval_start_ns = source_interval_ns[0]
source_interval_end_ns = source_interval_ns[1]
```

The design contract is that these normalized values are deterministic for a
given configuration. The delivered camera-only split converts positive seconds
values with exact decimal arithmetic and truncates any fractional-nanosecond
tail; the episode timeline requires a positive integer FPS. All formulas in this
document use these normalized integer values.

Rows are episode-relative and regular:

```text
row i:
  frame_index = i
  episode_time_ns[i] = delivered grid offset defined by the canonical publication contract
  timestamp = episode_time_ns[i] / 1e9
  row_count = length of the delivered sensor-clock grid
  target_source_time_ns[i] = source_interval_start_ns + episode_time_ns[i]
```

`output_fps` is the normalized integer FPS for the episode timeline. Rows are
emitted for `0 <= i < row_count`. The delivered camera-only grid construction
and rounding rule are defined by
[Canonical Multimodal AV Episode Publication](multimodal-av-episode-publication.md#sensor-clock-grid).
The grid construction makes row count a deterministic function of integer clip
duration and output FPS without independently re-deriving it from a second
formula. The episode timestamp grid is explicitly half-open:
`0 <= episode_time_ns[i] < duration_ns`. There is no row at the exclusive end.

`target_source_time_ns` is the ideal grid point on the source timeline, not the
timestamp of any selected sensor observation. Actual selected sensor timestamps
are retained separately in per-sensor provenance.
`source_interval_ns` defines target episode rows. Per-sensor support intervals
may extend outside it by the configured interpolation or preintegration
tolerance and must be recorded in provenance.

Each required camera contributes one visual observation per episode row under a
stable recipe feature key:

```text
observation.images.<camera_key>
```

Tabular sensor features use the same row timeline. The initial feature families
are:

- `observation.imu.*` for preintegrated IMU interval deltas and validity.
- `observation.gps.*` for WGS-84 GPS/GNSS position fields and validity.
- `index`, `timestamp`, `frame_index`, `episode_index`, `is_first`, `is_last`,
  and `is_terminal` for recipe-level row indexing. Export sinks may add
  format-specific fields such as LeRobot `task_index`.

Pipeline-owned provenance must retain:

- source session identifier and source URI set;
- absolute source interval `[start_ns, end_ns)`;
- per-row target source time in nanoseconds;
- per-sensor selected source timestamps;
- per-sensor alignment deltas or interval bounds;
- calibration ID and calibration version used for the episode;
- sensor IDs and camera keys used for the episode;
- sampling policy names and relevant thresholds;
- producer version and `config_digest`.

#### Provenance Placement

Provenance should be placed according to the granularity needed by consumers:

- Episode-level provenance goes in Lance episode records, including source
  session ID, source interval, calibration ID, configured sensor set, tasks,
  `config_digest`, and `provenance_id`.
- Row-level alignment provenance goes in Lance row-level records or nested row
  payloads, including target source time, selected per-sensor timestamps,
  alignment deltas, interval bounds, and validity flags.
- Larger audit details go in Lance provenance records or companion metadata
  keyed by `provenance_id`, including source object identifiers, supporting
  sample counts, detailed alignment diagnostics, and retry or quarantine
  records.

Training code can ignore audit-only metadata, while debugging and regeneration
tools can use it to trace each episode back to the selected source observations.

### Calibration Contract

Calibration is modality-agnostic rig metadata. It is not limited to cameras.
The canonical calibration store should be a versioned Lance table or companion
manifest such as:

```text
meta/calibration.json
```

The manifest should be keyed by `calibration_id` and include:

- coordinate frame names and conventions;
- per-sensor modality and stable sensor key;
- per-sensor extrinsics, such as `vehicle_T_sensor`;
- camera intrinsics, distortion parameters, image size, and image convention;
- IMU, GPS/GNSS, LiDAR, radar, and future modality calibration fields when
  available and relevant to downstream consumers;
- calibration version, producer, and validity interval.

Each episode references the calibration used to generate its aligned rows. If
calibration is constant for a recording session, every episode from that session
can share a `calibration_id`. If calibration changes within a session, episode
metadata must reference the correct calibration interval. A single-calibration
episode must be wholly contained within that calibration validity interval;
otherwise the stage must split, truncate, fail the episode, or use row-level
calibration IDs under an explicitly reviewed policy.

By default, calibration remains metadata. If a training model needs calibration
as input, Lance publication should materialize selected calibration values into
declared row-level features; export sinks can then project those fields into
their format-specific schemas.

### Sampling Policies

#### Cameras

Initial camera policy: nearest source frame to each grid timestamp.

Required configuration:

- `sampling: nearest`
- `max_delta_ms`
- optional camera role or layout metadata

The default policy fails the episode when a required camera cannot provide a
frame within `max_delta_ms` for any row. A source frame must not be selected for
more than one output row unless `allow_repeated_frames: true` is configured.
Ties choose the earlier source frame.

#### Preintegrated IMU

Initial IMU policy: causal preintegration over adjacent grid timestamps.

For a complete grid, row zero is the existing sensor-library bootstrap row:
identity delta, zero duration, and invalid integration status. Each later row
aligned to `t_i` represents the interval `[t_{i-1}, t_i)`.

The writer exposes:

- delta rotation quaternion;
- delta velocity;
- delta position;
- interval start/end timestamps;
- integration duration;
- source sample counts and maximum supporting gap;
- integration validity and invalid-reason mask.

The IMU policy must declare the output frame convention, quaternion component
order, gravity-compensation convention, duration clock, and reference-clock
alignment interval bounds.

The default required-IMU policy fails the episode if any non-bootstrap interval
is invalid or exceeds the configured support-gap threshold. A future policy may
allow masked intervals for consumers that can train through sparse quality
markers.

#### GPS/GNSS

GPS arrives around 1 Hz against a 30 Hz grid, so the grid is denser than the
source and every row is placed rather than selected down. The config chooses how,
because the two ways produce materially different data and neither is right for
every consumer:

- `sampling: nearest_neighbor` repeats the closest recorded fix, bounded by
  `max_delta_ms`. Every row is a position that was actually measured, carrying
  whatever staleness the gap to it implies.
- `sampling: interpolate` derives a position between the two fixes bracketing the
  row, bounded by `max_gap_ms`. Rows read smoothly and no instrument reported any
  of them.

Interpolation runs in ECEF and publishes WGS-84, uses only valid bracketing
fixes, and never extrapolates. A row it cannot serve — no bracket, or a bracket
wider than `max_gap_ms` — fails the clip.

Set the bounds between the jitter a healthy receiver shows and the gap a dropped
fix leaves. Measured on real sessions those are about ±15 µs and about two
seconds, so the defaults are 1500 ms for a bracket and half that for nearest
selection. A bound set at the nominal interval refuses healthy data.

Each row records how it was produced, and from which fix or pair of fixes,
because once a position sits on the grid nothing about the row itself says
whether it was measured or derived.

#### LiDAR

LiDAR is not serialized by the initial pipeline. The design leaves room for it
by keeping the episode contract centered on a shared time interval and regular
row grid, while allowing future LiDAR features to use either:

- row-aligned sweep features keyed by episode timestamp; or
- external point-cloud payload references with per-point timing and episode
  provenance.

The exact Lance and export representation for LiDAR remains deferred.

### Ownership Boundaries

#### Sensor Library Owns

- Reading caller-provided `DataSource` inputs.
- Parsing source media or MCAP payloads into typed sensor data.
- Producing `CameraData`, `GpsData`, `PreintegratedImuData`, and future
  modality data structures.
- Preserving selected source timestamps and alignment timestamps.
- Enforcing sensor-level dtype, shape, monotonicity, and alignment invariants.
- Reporting sensor-local alignment or validity failures.

#### Pipeline Owns

- Session discovery and URI traversal.
- Source artifact existence checks and opening source objects for sensor
  components.
- Required/optional modality policy.
- Fixed-stride clip span construction.
- Episode IDs and clip IDs.
- Episode-relative timestamps and row count.
- Lance feature naming, schema, media references, and metadata.
- Optional export sinks such as LeRobotDataset v3.
- Atomic episode success/failure policy.
- Quarantine or rerun state.
- Scenario labels, captions, embeddings, and other curation outputs.
- End-to-end provenance and run summaries.

This boundary keeps Lance publication, export sinks, and curation policy out of
`cosmos_curator.core.sensors` while still using the sensor library as the
source of aligned modality data.

### Failure and Quarantine Semantics

Failures are classified by the narrowest unit that can be retried:

| Failure type | Unit | Initial behavior |
| --- | --- | --- |
| Missing required session object | session | Reject session before span generation. |
| Unsupported configuration | run | Fail fast. |
| Required camera alignment miss | episode | Fail episode. |
| Required IMU integration invalid | episode | Fail episode. |
| Required GPS/GNSS row that cannot be placed on the grid | episode | Fail episode. |
| Optional modality miss | episode | Write null or masked values for configured optional fields. |
| Lance publication failure | attempt | Do not publish episode records or a completion marker for that attempt. |
| LeRobot export failure | export attempt | Do not mark the export attempt complete; keep the Lance dataset as the durable boundary. |
| Derived curation failure | artifact attempt | Quarantine the stage output key; keep the aligned episode and video artifacts available for retry; do not publish the episode until required outputs exist. |

The initial runtime should be conservative: failed required modalities fail the
whole episode. More permissive policies can be added later only when downstream
consumers can distinguish and handle masked rows.

Each failed episode should produce a structured diagnostic record containing:

- episode ID and source interval;
- failing stage;
- sensor key or writer component;
- error category;
- short message;
- policy that made the failure terminal.

### Downstream Curation Extensions

Scenario labeling, captioning, embedding, filtering, and multimodal model
processing consume the episode contract rather than the source recordings
directly. Training-visible outputs are written into Lance refinement records
during publication.

The delivered single-camera free-form caption contract is specified in
[`Multimodal AV Captioning`](multimodal-av-captioning.md). It consumes the canonical episode table independently of
episode splitting and records the exact selected-camera artifact binding with each result.

They may add:

- episode-level labels;
- per-row or interval annotations;
- per-camera captions;
- episode or window embeddings;
- filtering decisions and quality scores;
- model-specific materialized features.

These stages should not change the base episode timeline. If a model needs a
different sampling rate, camera canvas, history window, or token layout, that is
a derived view with its own recipe and provenance.
Episode-level or window-level outputs are visible through LeRobot training
loads only after the optional LeRobot exporter materializes them as declared
row-level features. Other consumers may read the Lance records directly.

### Deferred Capabilities

- LiDAR serialization and point-cloud payload references.
- Pose trajectories and advanced GPS/GNSS smoothing policies.
- Clock-drift estimation beyond configured timestamp mappings.
- Rig-frame derivations and calibration-version selection.
- Audio and action streams.
- Non-fixed-stride span providers.
- Partial-episode recovery after writer failure.
- Version-aware regeneration policy.
- Exact LeRobot exporter API and chunk-size tuning.

## Initial Runtime Plan

### Stage Plan

The initial runtime starts with lightweight session discovery, then has
two main processing phases:

- generate aligned episodes and early video artifacts from source sessions;
- produce curation features and publish an inspectable Lance dataset with
  media payload references.

```text
source sessions
  -> session discovery
  -> fixed-stride splitting + alignment
  -> VLM free-form captions
  -> VLM train-of-cognition reasoning
  -> video + text embeddings
  -> optional FST top-level scenario labels
  -> selected multimodal processing
  -> Lance publication
  -> optional LeRobotDataset v3 export
```

Runtime priority is defined by whether a capability is required for the
initial end-to-end implementation or can follow after the core path works.
Required initial capabilities include multimodal sensor sampling, required
curation outputs, embeddings, and the complete Lance schema needed to inspect
the result. Follow-on capabilities harden, enrich, or export the dataset after
the core path works.

| Stage | Initial scope | Outcome | Design note |
| --- | --- | --- | --- |
| Session discovery | Required | Enumerate candidate sessions. | Driver-side input setup only. It does not recurse through every session or verify artifacts. |
| Fixed-stride splitting + alignment | Required | Generate aligned multimodal episodes from longer sessions. | First Ray Data processing stage. Runs in parallel and performs per-session artifact existence checks before common time range calculation, fixed-stride spans, camera sampling, episode assembly, immutable MP4 content writes, and VLM-ready frame preparation. Deeper data-integrity checks run in a separate pipeline. |
| Multimodal sensor sampling | Required | Add camera, IMU, and GPS/GNSS rows on the episode timeline. | Extends camera sampling with required non-camera sensor outputs. |
| VLM free-form captions | Required | Add vanilla natural-language captions for VLM-ready episode frames. | Produces training-visible caption records keyed by aligned episode, without requiring a fixed FST taxonomy in the first run. |
| VLM train-of-cognition reasoning | Required | Explain why the observed situation occurred. | Consumes VLM-ready frames and optional caption context; it does not require FST labels in the initial runtime. |
| Video + text embeddings | Required | Add retrieval and training features. | Required for the initial feature set. |
| Lance publication | Required | Publish an inspectable durable refinement dataset. | Writes the complete Lance episode, row, provenance, curation, embedding, calibration-reference, and media-reference schema after required artifacts are complete. |
| Optional FST top-level scenario labels | Follow-on | Add taxonomy-bound scenario labels. | Can be added after the free-form captioning path is validated and the active label set is approved. |
| Selected multimodal processing | Follow-on | Add a consumer-specific camera/sensor feature. | Optional until a later decision names the first function. |
| Optional LeRobotDataset v3 export | Follow-on | Export recipe-specific training format. | Consumes Lance records and media references; it is not the durable refinement boundary. |

Publication is staged. The splitting and alignment stage may write final
per-camera MP4 clip content early to reduce Ray object-store pressure, but those
files are not treated as published episodes until Lance publication records the
episode, media references, and required metadata. Downstream stages should write
append-only refinement records or stage outputs; they should not mutate existing
published Lance rows in place.
Any Ray Data stage that writes external artifacts must use attempt-scoped or
content-addressed paths so retries do not corrupt already-written outputs.

For the initial runtime described by this design, Lance publication requires
the configured multimodal sensor outputs, free-form captions, train-of-cognition
reasoning, embeddings, and curation joining/check outputs. FST top-level
scenario labels and selected multimodal processing remain follow-on work unless
a later decision names one as a required function.

### Expected Runtime Outcomes

The initial runtime should make it possible to:

- exercise the sensor-library integration path for fixed-stride multiview AV
  episode generation;
- create Lance episode records from longer multiview AV sessions;
- write per-camera MP4 clip artifacts while aligned episodes are generated;
- prepare downsampled and reshaped frame payloads for downstream VLM stages;
- preserve IMU and GPS/GNSS alignment provenance;
- generate free-form captions and reasoning records from video;
- produce video/text embeddings for retrieval, training, or dataset
  balancing;
- publish a complete Lance schema for the required initial outputs.

Follow-on features should make it possible to:

- preserve full modality calibration, including intrinsics, extrinsics, and
  calibration versioning;
- reserve a selected multimodal processing stage that becomes required once its
  first function is selected;
- harden publication idempotency, retry, and quarantine semantics;
- export recipe-specific downstream formats such as LeRobotDataset v3.

### Illustrative Runtime Configuration

The initial runtime should expose configuration in these groups:

```yaml
input:
  sessions:
    - uri: s3://bucket/recordings/session-000/
  session_id_pattern: "{name}"
  limit: null

clip:
  duration_s: 10.0
  stride_s: 10.0
  min_duration_s: 10.0
  output_fps: 30.0

sensors:
  cameras:
    front:
      required: true
      uri_pattern: "camera/front.mp4"
      sampling: nearest
      max_delta_ms: 50
    left:
      required: true
      uri_pattern: "camera/left.mp4"
      sampling: nearest
      max_delta_ms: 50
  imu:
    required: true
    source: "imu.mcap"
    topic: "/imu"
    sampling: preintegrated
    max_gap_ms: 20
  gps:
    required: true
    source: "gps.mcap"
    topic: "/gps"
    sampling: interpolate   # or nearest_neighbor, with max_delta_ms
    max_gap_ms: 1500

calibration:
  manifest_uri: "calibration/rig.json"
  episode_reference: calibration_id

output:
  lance_uri: s3://bucket/av-pipeline/refinement.lance
  media_root: s3://bucket/av-pipeline/media/
  task_name: "autonomous driving"
  export:
    lerobot_v3:
      enabled: false
      dataset_root: s3://bucket/av-pipeline/lerobot-v3/

failure:
  required_sensor_policy: fail_episode
  publish_completion_marker_last: true
```

Exact field names are implementation details. The stable semantic groups are:
input discovery, clip geometry, output FPS, required sensors, sampling policies,
calibration, output layout, and failure policy.

### Engineering Stage Notes

#### Session Discovery

Discovery runs in the driver and should remain a lightweight candidate
enumeration function, not a Ray Data stage. It supports two modes:

- `input_path_prefix` only: list the immediate child directories below the
  prefix. Each child directory name is a candidate `source_session_id`.
- `input_path_prefix` plus `session_id_list_path`: read the session IDs from
  the file and join each ID with `input_path_prefix`.

In both modes, discovery emits candidate session work items for Ray Data
processing. It should not recurse through every session, resolve configured
camera and sensor URIs, verify required artifacts, decode media, or sample
sensors.

#### Fixed-Stride Splitting and Alignment

This is the first Ray Data processing stage and the first episode-generating
stage. It fuses the work that would otherwise be split across artifact
resolution, required artifact existence checks, common time range calculation,
fixed-stride span generation, sensor reads, alignment, episode assembly,
immutable MP4 content writes, and VLM frame preparation.

For each discovered session, the stage resolves configured camera, sensor, and
calibration artifacts. It rejects the session when required artifacts are
absent. The pipeline opens source artifacts and passes caller-provided
`DataSource` inputs to the sensor library; sensors own decoding, timestamp
interpretation, and alignment from those inputs. Optional modalities may be
omitted from the session-level inputs. If a configured optional modality is
absent for an episode, its feature keys remain in the dataset schema and the
writer fills those fields with null or masked values plus explicit validity
fields.

For each session, the stage computes the usable common time range for required
modalities. For multiview cameras, aligned clips share the same span and stable
ID across views.

Fixed-stride spans use half-open intervals bounded by the usable common time
range. `duration_ns` and `stride_ns` are the normalized integer values from the
aligned episode contract:

```text
span_start_ns = common_start_ns + k * stride_ns
span_end_ns = span_start_ns + duration_ns
span_k = [span_start_ns, span_end_ns)
```

The initial runtime generates full spans only when `span_end_ns <= common_end_ns`.
If trailing partial spans are supported later, they must be clamped to
`common_end_ns` and kept only when `span_end_ns - span_start_ns` is at least
`min_duration_ns`.
Span IDs should be stable hashes of source session identity, interval, output
FPS, required sensor set, and the episode contract version.

The stage constructs one regular timestamp grid per episode from
`source_interval_start_ns`, `source_interval_end_ns`, and `output_fps`.

It asks sensor-library components for modality outputs aligned to that grid:

- Cameras produce `CameraData` rows selected for each grid timestamp.
- IMU uses `PreintegratedImuSensor` / `PreintegratedImuData` semantics.
- GPS/GNSS uses `GpsData` rows interpolated for each grid timestamp.

The sensor library owns source decoding, timestamp comparison, per-sensor
alignment results, and alignment failures. The pipeline owns the policy decision
about whether an alignment failure fails the episode, quarantines it, or is
represented as a masked row. The initial policy is `fail_episode` for required
modalities.

The stage writes immutable per-episode MP4 content keyed by stable episode ID.
Lance publication records accepted episodes and their media references after
ordering and filtering are frozen. The stage also prepares bounded, downsampled,
and reshaped frame payloads for the downstream VLM stages so those stages do
not need to repeat clip extraction work. These VLM payloads are stage inputs,
not a replacement for the media artifacts referenced from Lance.

Episode assembly performs the minimal structural checks needed to produce a
well-formed dataset artifact. It confirms that stage outputs can be joined into
the configured episode timeline and serialized consistently. It checks:

- row count equals the deterministic count derived from duration and
  `output_fps`;
- required camera and sensor outputs have the expected episode ID and row count;
- configured optional feature columns remain schema-stable across episodes;
- episode calibration references resolve to calibration records;
- feature arrays match the declared Lance schema shapes and dtypes;
- required provenance references are present for every episode.

Assembly errors are episode-local unless they indicate a broken session
manifest or unsupported configuration.

#### Lance Publication

The publication stage writes the durable Lance refinement dataset:

- one episode record per accepted fixed-stride interval;
- one row-level record or nested row payload per episode timestamp;
- media-reference records for each per-camera MP4 clip artifact;
- calibration records or references;
- provenance records for source objects, selected timestamps, alignment support,
  and retry or quarantine diagnostics;
- curation records from free-form captions, train-of-cognition reasoning,
  embeddings, optional FST labels, and selected multimodal stages when those
  stages are enabled.

Publication consumes aligned episode descriptors, MP4 artifact references from
the splitting and alignment stage, tabular sensor rows, calibration data, and
training-visible outputs from downstream curation stages. It writes durable
Lance records once; later stages should append refinement records or publish a
new attempt rather than mutate existing rows in place.

Publication outputs should be written under an attempt-scoped temporary prefix
or content-addressed paths. The publish step must happen after MP4 references
are complete, Lance writes are committed, and required metadata is present.
Episode records or an equivalent completion marker must be published last so
incomplete outputs are not treated as published episodes. Retries ignore
uncommitted attempts.

#### Optional LeRobotDataset v3 Export

LeRobotDataset v3 export is downstream of Lance publication. The exporter maps
Lance episode, row, media-reference, task, calibration, and curation records
into the LeRobot `data/`, `videos/`, and `meta/` layout when a consumer needs
that training format. It also owns LeRobot-specific `task_index` mapping,
`meta/info.json`, `meta/tasks.parquet`, `meta/stats.json`, chunk sizing, and
video-reference metadata.

The initial runtime may skip LeRobot export. A production LeRobot export must
include all required LeRobot metadata and statistics.

#### Derived Curation Stages

Free-form captions, train-of-cognition reasoning, embeddings, optional FST
labels, and selected multimodal processing are curation stages.
They consume aligned episodes and produce feature records for Lance publication.
Derived feature stages consume aligned episode IDs and timestamps from the
splitting and alignment stage.

Each derived stage must declare:

- input artifact contract;
- output schema;
- producer version, including model and prompt version when applicable;
- `required_for_publish`, which is specific to the active runtime;
- missing-output policy;
- success marker;
- retry key;
- quarantine scope.

The default quarantine scope is the stage output attempt, not the aligned
episode or already-written MP4 artifacts. Finalization must not publish an
episode with a missing stage output that is required by the active runtime.

#### Free-Form Caption Prompt

The initial captioning prompt is:

> Describe the driving scene and ego-vehicle context visible in this clip.

The initial captioning run produces vanilla natural-language captions rather
than taxonomy-bound labels. Captions should describe visible road context,
traffic actors, ego-vehicle behavior, and material uncertainty without forcing
the output into a fixed scenario class. The output should preserve the caption
text, model and prompt version, confidence or evidence-quality metadata, and the
aligned episode or frame interval it describes.

Optional FST top-level labels are follow-on work. When enabled, FST labels should
map an aligned episode to exactly one approved top-level scenario label and
preserve the selected label, model and prompt version, and a short rationale or
evidence span when the selected model supports it.

#### Train-of-Cognition Reasoning Prompt

The train-of-cognition stage reasons over the VLM-ready episode frames.
It explains why the observed driving situation happened and why the ego vehicle
or other actors behaved as they did. It is not a top-level scenario classifier;
it consumes VLM-ready frames and optional free-form caption context to produce a
more detailed video-only explanation.

The initial reasoning model is [Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super).

Reasoning output should capture causal and contextual factors such as:

- traffic controls, right-of-way, lane geometry, road topology, and occlusions;
- actor intent or interaction cues visible in the episode;
- why the ego vehicle slowed, stopped, yielded, changed path, or maintained
  course;
- which visible conditions made the situation nominal, constrained, contested,
  or terminal;
- uncertainty when the video does not provide enough evidence to explain a
  behavior confidently.

The stable training-visible contract should distinguish structured reasoning
fields from debug prose. At minimum, the stage should preserve the model and
prompt version, any caption or frame-interval context it consumed, a concise
natural-language explanation, confidence or evidence-quality metadata, and any
structured cause or interaction tags selected by the configured reasoning schema.
Longer rationale text remains provenance for inspection unless a downstream
training contract explicitly requires it.

The first open multimodal design decision is the function family:

- `f(IMU)`: consume preintegrated IMU features only.
- `f(IMU + GPS)`: combine motion deltas with WGS-84 position context.
- `f(egomotion)`: consume a derived trajectory or pose stream once the
  egomotion contract is available.

### Acceptance Criteria

The implementation work that follows this design should be complete when it
can:

- generate deterministic fixed-stride episode intervals from longer sessions;
- produce inspectable Lance episode, row, media-reference, provenance, and
  curation records;
- write one logical episode per aligned multiview clip;
- compute row count as the length of the delivered sensor-clock grid;
- emit episode rows only for the half-open interval
  `0 <= episode_time_ns[i] < duration_ns`;
- preserve source timestamps and alignment provenance;
- write modality calibration metadata and episode calibration references;
- apply explicit camera, IMU, and GPS/GNSS policies;
- fail or quarantine episodes according to configured policy;
- rerun without treating incomplete episode writes as published outputs;
- allow downstream curation stages to consume episodes without direct session
  traversal.
