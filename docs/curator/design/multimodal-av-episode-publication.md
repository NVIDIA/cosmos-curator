# Canonical Multimodal AV Episode Publication

## Purpose

`multimodal-split` publishes one canonical Lance row for each complete synchronized multimodal episode. Camera MP4s,
their timing sidecars, and planned non-camera sensor MCAPs remain external payloads, while the row records their
locations, sizes, and strong hashes. This gives downstream captioning and sensor-alignment recipes a stable episode
identity without requiring them to traverse source sessions or infer whether the planned artifact set is complete.

One row represents a synchronized interval, not one individual sensor. That keeps shared episode facts authoritative
and prevents different views of the same interval from receiving inconsistent episode-level enrichments.

## Background and use cases

Configured sensors may be stored under different recorder paths. Planning resolves the sensors present in each
session; an optional missing sensor is not part of its episodes. For each non-canonical span, `multimodal-split` emits
one terminal outcome per planned camera or non-camera sensor.

Sensor outcomes do not append base rows independently. Publication groups them by `split_clip_id` and writes one
episode row only when every planned camera and planned non-camera sensor has a successful outcome:

```text
independent sensor outcomes
  front_left camera ──┐
  front_wide camera ──┼──> assemble complete artifact set ──> one episodes.lance row
  ego MCAP         ──┘
```

The published row has three complementary shapes:

- `episode_id` names the logical source-session interval. It excludes camera selection, encoding, and downstream
  models, providing the join key for later episode-scoped data.
- `camera_media` is the complete nested list of materialized camera descriptors planned for the episode, serialized
  lexicographically by `camera_key`.
- `sensor_media` is the complete nested list of materialized non-camera sensor descriptors planned for the episode,
  serialized lexicographically by `sensor_key`; it is empty when none is planned.

```text
episode_id = one synchronized source interval
├── camera_media
│   ├── front_left
│   │   └── MP4 location + size/hash
│   │       timeline location + hash
│   │       sensor timestamps location + hash
│   ├── front_right
│   │   └── MP4 and timing-artifact bindings
│   └── front_wide
│       └── MP4 and timing-artifact bindings
└── sensor_media
    └── ego
        └── sensor type + MCAP location + size/hash
```

The displayed camera and sensor names are example keys stored as list items, not fields in either item struct.

### Why a list of structs

`camera_media` is one top-level `large_list` whose item type is a fixed struct. Lance stores that struct's children as
nested fields. The list and the nested fields are stacked parts of one representation, not competing designs:

```text
camera_media                         large_list
└── item                             fixed struct
    ├── camera_key                   nested primitive field
    ├── media_location               nested primitive field
    ├── media_sha256                 nested primitive field
    └── ...                          nested primitive fields
```

`sensor_media` uses the same list-of-fixed-structs pattern with its own descriptor fields.

| Representation | Shape | Decision |
|---|---|---|
| List of fixed structs | `camera_media: large_list<struct<...>>` | Selected: variable camera count with one typed descriptor contract |
| Struct keyed by camera | `struct<front_left: struct<...>, front_right: struct<...>>` | Camera names become schema fields; a rig change is schema evolution |
| Top-level camera columns | `front_left_uri`, `front_right_uri`, … | Duplicates field families and binds the table to one rig |
| One row per camera | `(episode_id, camera_key)` per row | Every consumer must regroup rows and prove the camera set is complete |

Existing clip-oriented recipes intentionally use different row units. `video-cut` publishes one row per `span_id`,
which `video-transcode` enriches with one flat MP4 and `clip_id`. `robot-action-split` publishes one independently
appendable view row per `clip_id`; a consumer needing
synchronized views groups those rows by `span_group_id`. Here the complete multimodal episode artifact set, rather
than an individual camera clip, is the canonical row.

Different episodes may have different list lengths under one schema. An already-published `episode_id` whose resolved
camera or non-camera sensor set does not match is incompatible and fails before rewrite. The publisher writes
`camera_media` and `sensor_media` as complete values; it does not grow an existing row.

Camera count and camera names are list data. The item struct is schema. A new per-camera descriptor property is
producer-owned schema evolution. Consumer enrichments read `camera_media`; they do not add children inside it.

The same base row supports:

1. **Single-view captioning.** Resolve one required `camera_key`, verify that view's MP4, and bind the visual result
   using:

   ```text
   (episode_id, materialization_digest, camera_key,
    media_location, media_size_bytes, media_sha256)
   ```

2. **Multiview captioning or fusion.** Resolve an explicit ordered camera selection from one already-assembled list
   instead of discovering and joining separate camera rows. The consumer records that selection in its own result
   contract and binds the result to `materialization_digest`, which identifies the complete published artifact set.

3. **Stacked episode enrichment.** Free-form captions, other caption styles, quality passes, and later typed modality
   tables remain independent result contracts on the same episode. Each owns its fields or table and joins through
   `episode_id`; none rediscovers source camera paths or reassembles the camera set. A visual result additionally records
   `materialization_digest`, its explicit camera selection, and the applicable artifact hashes so the episode-level join
   does not erase visual provenance. Adding another result style does not change the base episode schema.

4. **Artifact verification.** Use `media_size_bytes` and the per-artifact hashes to verify only the MP4, timing, or
   sensor MCAP files a consumer reads. `materialization_digest` identifies the complete published artifact set. Row
   presence alone does not prove that referenced objects remain intact in storage.

5. **Publication completeness.** At publication time, a canonical row means every planned camera and every planned
   non-camera sensor required by that split contract produced a successful terminal outcome. Any missing or failed
   planned sensor drops the episode. Successful sibling artifacts may remain on storage for diagnosis or retry, but no
   partial Lance row is published.

This is the multi-phase consequence of assembling the complete artifact set at publication: later phases join an
already-complete interval and select from its published descriptors instead of reconstructing a rig. Those phases share
the episode and, when consuming media, materialization identity; they do not share result identity. Each retains its
own model, prompt, schema, version, and writer contract.

For an unchanged source interval, adding, removing, or renaming a camera or non-camera sensor leaves `episode_id`
unchanged while changing `split_contract_digest` and `materialization_digest`. A changed sensor set can also change the
synchronized intervals that planning produces.

## State and output layout

The output root has one fixed layout:

```text
<output.path>/
  episodes.lance
  sessions/<source_session_id>/<split_clip_id>/
    clip.json
    <camera_key>.mp4
    <camera_key>.mp4.timestamps.npy
    <camera_key>.mp4.sensor_timestamps.npy
    <sensor_key>.mcap
  errors/
```

`episodes.lance` is the canonical episode authority. Once referenced by a row, the clip's artifacts are immutable by
contract. The per-clip `clip.json` and sparse files under `errors/` remain path-local diagnostics; they are not
publication inputs or current canonical state. Sensor outcomes carried within one invocation are transient assembly
inputs and are not another durable table.

Each MP4 is written after its two timing sidecars and is the per-camera artifact commit point. The episode row becomes
visible only after every planned camera and every planned non-camera sensor has produced a complete receipt.

## Episode record

The ordered producer-owned Arrow schema is:

| Field | Arrow type | Meaning |
|---|---|---|
| `record_schema_version` | `int32 not null` | Base row field and semantic version |
| `media_contract_version` | `int32 not null` | `multimodal-split` media contract version |
| `episode_id` | `string not null` | Stable logical source-interval identity |
| `split_clip_id` | `string not null` | Split lineage and artifact-path identity |
| `source_id` | `string not null` | Digest of the normalized source-session location |
| `source_session_id` | `large_string not null` | Human-visible source-session child name |
| `source_session_location` | `large_string not null` | Resolved source-session location |
| `start_ns` | `int64 not null` | Inclusive selected episode-clock start |
| `exclusive_end_ns` | `int64 not null` | Exclusive selected episode-clock end |
| `sampling_fps` | `int32 not null` | Shared regular-grid rate |
| `row_count` | `int64 not null` | Grid points and encoded frames per camera |
| `timestamp_source` | `string not null` | Sensor Library timestamp provenance; `sensor`, or `container_pts` when a camera's `allow_pts_fallback` fires |
| `split_contract_digest` | `string not null` | Result-defining split configuration identity |
| `materialization_digest` | `string not null` | Exact complete published artifact-set identity |
| `camera_media` | `large_list<struct<...>> not null` | Complete camera descriptors, sorted by `camera_key` |
| `sensor_media` | `large_list<struct<...>> not null` | Complete non-camera descriptors, sorted by `sensor_key`; empty when none is configured |

Every list item and every child of the `camera_media` struct is non-nullable:

| Child | Arrow type | Meaning |
|---|---|---|
| `camera_key` | `string not null` | Stable configured camera key |
| `source_media_location` | `large_string not null` | Source camera recording |
| `media_location` | `large_string not null` | Materialized camera MP4 |
| `media_size_bytes` | `int64 not null` | Exact MP4 byte count |
| `media_sha256` | `string not null` | SHA-256 of MP4 bytes |
| `timeline_location` | `large_string not null` | Regular-grid timestamp sidecar |
| `timeline_sha256` | `string not null` | SHA-256 of grid-sidecar bytes |
| `sensor_timestamps_location` | `large_string not null` | Selected source sensor-timestamp sidecar |
| `sensor_timestamps_sha256` | `string not null` | SHA-256 of sensor-timestamp sidecar bytes |

Every list item and every child of the `sensor_media` struct is likewise non-nullable. It carries no grid or
sensor-timestamp sidecars, whichever way its clip was produced. A copied sensor was never put on the episode's grid, so
there is no grid to record. An aligned one sits on the episode's own grid rather than a timeline of its own and records
per-row provenance inside the rows, so there is nothing to write beside them.

| Child | Arrow type | Meaning |
|---|---|---|
| `sensor_key` | `string not null` | Stable configured sensor key |
| `sensor_type` | `string not null` | Configured sensor kind, such as `egotrajectory` or `gps` |
| `source_media_location` | `large_string not null` | Resolved session artifact the clip was cut from |
| `media_location` | `large_string not null` | Materialized clip MCAP |
| `media_size_bytes` | `int64 not null` | Exact MCAP byte count |
| `media_sha256` | `string not null` | SHA-256 of MCAP bytes |

Camera storage order is lexicographic serialization, not a model-input order. A schema-level resolver requires an
explicit `camera_key` and returns exactly one descriptor, rejecting missing or duplicate keys. A visual consumer uses:

```text
(episode_id, materialization_digest, camera_key,
 media_location, media_size_bytes, media_sha256)
```

The base record intentionally has no default camera. Each consumer owns its camera choice or explicit multiview order.

## Identity and timeline contracts

All identities are SHA-256 hashes of compact, sorted canonical JSON. The layers answer different questions:

- `episode_id` identifies the logical source interval and is the durable join key.
- `split_clip_id` preserves the delivered split identity used by paths and diagnostics.
- `split_contract_digest` identifies the result-defining multimodal artifact transformation.
- `materialization_digest` binds downstream results to the complete set of exact published camera, timing, and sensor
  MCAP bytes.

`split_clip_id` includes the media-contract version and excludes encoder settings, camera selection, and sampling, so
a rerun addresses the same object paths for unchanged camera keys. Changes to result-defining settings are instead
visible in `split_contract_digest`, while the per-artifact hashes bind the exact payload bytes.

### Logical episode identity

```text
{
  "kind": "multimodal-av-episode",
  "episode_identity_version": 1,
  "source_id": <source session location digest>,
  "start_ns": <inclusive start>,
  "exclusive_end_ns": <exclusive end>
}
```

The identity excludes cameras, encoders, sampling, and derived models so episode-keyed sensor data remains joinable
across rendering choices.

### Split contract identity

```text
{
  "kind": "multimodal-av-split-contract",
  "split_contract_digest_version": 4,
  "media_contract_version": <media contract version>,
  "clip": {
    "duration_ns": <resolved integer duration>,
    "sampling_fps": <resolved sampling rate>
  },
  "cameras": [
    {"camera_key": <key>, "sensor_type": <type>, "allow_pts_fallback": <bool>},
    ... sorted by camera_key
  ],
  "mcap_sensors": [
    {
      "sensor_key": <key>,
      "sensor_type": <type>,
      "topic": <topic or null>,
      "sampling": <policy or null>,
      "sampling_bound_ns": <bound the policy consults, or null>
    },
    ... sorted by sensor_key
  ],
  "video": {
    "max_delta_ns": <resolved tolerance>,
    "encoder": {
      "video_encoder": <encoder>,
      "video_bitrate": <canonical bitrate>,
      "gop_size": <GOP size>,
      "faststart": <fast-start setting>
    }
  }
}
```

Discovery limits, URI patterns, input/output roots, credentials, resources, concurrency, progress, buffering, retries,
and publication block sizing are excluded. URI patterns locate sources rather than defining the transform; each
resolved `source_media_location` remains explicit in the row and in pre-write compatibility checks.

### Materialization identity

```text
{
  "kind": "multimodal-av-episode-materialization",
  "materialization_digest_version": 3,
  "episode_id": <episode_id>,
  "split_clip_id": <split_clip_id>,
  "split_contract_digest": <split contract digest>,
  "sampling_fps": <sampling rate>,
  "row_count": <row count>,
  "camera_artifacts": [
    {
      "camera_key": <key>,
      "media_sha256": <MP4 hash>,
      "timeline_sha256": <grid-sidecar hash>,
      "sensor_timestamps_sha256": <sensor-timestamp-sidecar hash>
    },
    ... sorted by camera_key
  ],
  "sensor_artifacts": [
    {
      "sensor_key": <key>,
      "sensor_type": <type>,
      "media_sha256": <MCAP hash>
    },
    ... sorted by sensor_key; empty when no non-camera sensor is planned
  ]
}
```

Artifact locations and sizes are excluded from this digest; `media_size_bytes` remains a separate inexpensive
truncation check. This does not make arbitrary source relocation identity-preserving: `source_id` is derived from the
normalized source-session location, so moving a source session changes `source_id`, `episode_id`, and the downstream
materialization identity.

The per-artifact hashes and the composite digest serve different scopes. Each `*_sha256` verifies one external
artifact, allowing a consumer to validate only the files it reads. `materialization_digest` is a derived manifest
fingerprint over those stored hashes and the episode, split, and grid facts above; it is not an additional checksum of
the bytes. It gives a consumer one compact binding to the complete published artifact set and changes whenever any
published MP4, timeline, sensor-timestamp sidecar, or non-camera sensor MCAP changes.

Canonical row presence alone does not prove that external objects remain intact. Byte verification requires reading
the selected artifacts and comparing their stored hashes. Publication and reconciliation do not reread or rehash
already-canonical media, so this contract enables detection, not automatic repair: a normal retry will not repair a
damaged already-canonical artifact. Repair requires an explicit verification-and-repair workflow.

### Episode clock and grid

Record schema version 2 publishes half-open intervals `[start_ns, exclusive_end_ns)` with
`timestamp_source="sensor"` unless a camera's `allow_pts_fallback` config fired because its
sensor-timestamp sidecar was missing. In that case the episode uses `"container_pts"`: one
camera reading its timeline from the container sets the provenance for the whole episode,
since a clip is one moment across every view the rig promised. Media contract version 3
names the selected-source sidecar `<camera>.mp4.sensor_timestamps.npy`.

There is no in-place migration from the previous record or media contract. An existing
`episodes.lance` table written under record schema version 1 and media contract version 2 fails
schema validation, and surviving rows also conflict during reconciliation because their contract
versions and timestamp-source value differ. Run this version against a fresh output root.

```text
step_ns = 1_000_000_000.0 / sampling_fps
offset[k] = round(step_ns * k) using NumPy float64 ties-to-even rounding
timestamp[k] = start_ns + offset[k]
grid = consecutive timestamp[k] values in [start_ns, exclusive_end_ns)
row_count = len(grid)
```

The delivered `make_ts_grid` half-open grid is authoritative for row count. Offsets are independently rounded from the
origin; they are neither floored nor built by accumulating a rounded period. The regular grid is stored in each
camera's timeline sidecar, while the corresponding selected sensor timestamps are stored in its sensor-timestamp sidecar.

Together, `source_id` and `timestamp_source` record the provenance of the selected session-local episode clock. A
non-camera enrichment must either rely on a producer contract that places its timestamps in that domain or record the
mapping used to transform them.

The delivered `multimodal-caption` Ego contexts rely on the producer contract that required Ego MCAP `log_time` and
the episode clock share a session-local domain; captioning does not independently verify that relationship. They
require `timestamp_source: sensor`, which means the camera timelines came from recorded sensor timestamps rather than
the configured `container_pts` fallback, but does not prove camera-to-Ego clock equivalence. Within the Ego MCAP,
alignment uses `log_time`; the embedded
`PoseInFrame.timestamp` is decoded as non-serialized source provenance. The `video_only` context does not consume the
Ego clock and continues to support canonical `container_pts` episodes.

## Publication and recovery

After non-empty discovery and storage preflight, the driver creates an absent table with zero rows or validates the
existing table. It pins that finite Lance version and reads only reconciliation fields for the discovered source IDs.
Workers receive an inert source-keyed mapping; planning workers do not open Lance.

For each planned logical episode:

1. An absent `episode_id` proceeds to manifest and planned sensor writes.
2. An existing episode must match the record/media versions, split contract, sampling rate, timestamp source, and exact
   sorted camera and non-camera sensor source mappings. A match emits one `already_canonical` accounting record and
   touches no artifact path.
3. An incompatible match fails before a manifest, sidecar, MP4, or sensor MCAP is rewritten.

Reconciliation assumes resolved source artifacts—and each camera's input sensor-timestamp sidecar derived from its
source URI—remain immutable in place. Canonical reuse does not rehash those inputs. Corrected inputs must use a new
session location, and therefore a new source identity.

Each non-reconciled sensor emits exactly one current-run outcome per planned span. Both successes and failures carry the
complete shared episode facts and expected row count. Successes additionally carry all artifact descriptors that apply
to that sensor kind; a non-camera sensor has no grid or selection sidecars, so those descriptors are null for it. Durable
error JSON supplies detailed failure category and message but does not veto a later successful current-run outcome.

Assembly groups small outcomes by `split_clip_id`, compares the complete shared facts once, and requires exactly one
outcome for every planned camera and every planned non-camera sensor. Duplicate, unexpected, or conflicting input is an
internal contract error. A missing or failed sensor of either kind drops the episode without publishing a partial row.
Successful sibling artifacts remain available for diagnosis or retry. The splitter refuses an encode whose written row
count differs from the planned grid before it emits a successful outcome.

Complete rows are staged in bounded Lance fragments by workers and appended serially by the driver. Append recovery is
idempotent by `episode_id`: an ambiguous response is resolved by reopening the table and accepting all candidate IDs as
committed, retrying when none are present, and failing on partial presence. Earlier committed fragments survive a later
fragment failure.

The supported consistency model has one `multimodal-split` base-row publisher. A fully reconciled rerun stages no rows,
writes no artifacts, and does not advance the Lance version. The materialized plan independently counts one skip row
per canonical episode and the unique remaining `split_clip_id` values projected from camera plans. After publication,
the pipeline verifies:

```text
episodes_planned =
  episodes_already_canonical + episodes_newly_published + episodes_dropped
```

## Downstream extension seam

Base-schema validation requires the producer-owned fields in exact order, type, nullability, and metadata while allowing
additional nullable top-level fields. Later base appends omit those fields, preserving existing enrichment values and
presenting null for newly appended rows.

This section governs *downstream* consumers -- work that runs after base publication and enriches an already-published
episode. It does not govern the split producer, which owns every column it writes.

Compact singular results may use consumer-owned nullable top-level episode fields. Dense, repeated, or independently
versioned data belongs in typed tables or immutable sidecars keyed by `episode_id`. A visual result records the
applicable `materialization_digest` in its own contract. A downstream non-camera modality does not become an entry in
`camera_media` or `sensor_media`; the episode identity and episode-clock interval/grid are its join seam.

A modality the splitter itself cuts is not downstream work. It is base data the producer owns, catalogued in
`sensor_media` alongside `camera_media` and committed in the same fragment, so one clip is one row rather than a join
a reader has to reconstruct.

Concrete field names, result metadata, completion semantics, and concurrent enrichment publication belong to the
consumer that introduces them. Base publication does not reserve placeholder columns or define a universal result
lifecycle.

`multimodal-caption` validates canonical rows before registering its own caption/metadata pair, using the effective
result key in `caption_{context_variant}_{effective_result_key}_{model_key}_v1` and its `_metadata` sibling.
Opt-in `output.tag_camera_key: true` appends the selected camera key to the base key; custom-prompt base keys cannot
start with `default`.
These consumer-owned results enrich existing episode rows without changing the base schema; the
[captioning design](multimodal-av-captioning.md) owns their configuration, contract, provenance, publication, and
recovery rules.

`multimodal-embedding` independently reads selected camera MP4s from these same rows and adds a vector/metadata pair
per camera. One run can select a camera list or `all`; the selected pending results become visible together in one
Lance result update. Captions are not required for video embedding. The
[embedding design](multimodal-av-video-embedding.md) owns its result and recovery contracts, and the
[end-to-end guide](../guides/multimodal-av-e2e.md) connects the split, caption, and embedding examples.

## Configuration and boundaries

Canonical publication does not bump the configuration schema. It is an unconditional completion step of version 1:
existing configs remain valid, always use `<output.path>/episodes.lance`, and have no publication setting or opt-out.

```yaml
schema_version: 1
kind: multimodal-split

output:
  path: s3://bucket/av-mvp

execution:
  storage_profile: default
```

Local input and output locations may be written as bare filesystem paths,
`file:///...` URIs, or `file://localhost/...` URIs. Storage preflight resolves
those forms through the shared local-URI boundary before checking the input
directory or creating the temporary output probe; diagnostics retain the
original configured location. Completion reporting uses the same boundary when
it lists and opens persisted failure reports, so a `file://` output cannot be
summarized as clean merely because its URI text is not a filesystem path. The
open boundary resolves `file://localhost/...` itself rather than relying on a
dependency that only handles the empty-authority form. A `file://` URI with any
other authority is refused because it does not name a path on the local machine.

The Lance URI is always `<output.path>/episodes.lance`; there is no location override or publication tuning surface.
The invocation succeeds only after canonical publication succeeds. Artifacts committed before a publication failure
remain available for retry and diagnosis.

The contract does not migrate incompatible base schemas or artifacts from earlier record or media contracts, coordinate
concurrent base publishers, proactively detect out-of-band artifact replacement without a reader verifying the stored
hashes, or define general output-root cleanup. Cleanup must preserve `episodes.lance` and every artifact referenced by a
canonical row.

## Implementation ownership

`multimodal_split.plan` owns synchronized span facts and pre-write reconciliation decisions. `split` retains the
sensor-per-task processing and camera sidecar-first/MP4-last writes while emitting complete terminal receipts.
`records` owns the episode schema, construction invariants, and exact camera resolver. `publication` owns
engine-neutral complete-artifact-set assembly and fragment staging; the `pipeline` driver owns bounded Ray batching,
serial commits, and accounting. `lance_sink` is a thin episode-schema binding over the shared Curator Next fragment
bootstrap and recovery primitives.

The existing split URI, grid, identity, storage, and diagnostic helpers remain authoritative for their delivered
contracts. Digest-bound nullable caption-field registration is an extension precedent, not a dependency of base
publication.
