# Multimodal AV Captioning

## Summary

`multimodal-caption` enriches the canonical `episodes.lance` table produced by
[`multimodal-split`](multimodal-av-episode-publication.md). Splitting and captioning are independently runnable:
captioning selects one explicitly configured camera from each complete episode row and never rediscovers sessions or
reassembles camera sets.

The separate [`multimodal-embedding` recipe](multimodal-av-video-embedding.md) adds Cosmos-Embed1 video vectors to the
same episode table for selected cameras or all cameras in one run. It reads camera MP4s directly, so caption results
and Ego prompt context are not inputs to those vectors. See the
[end-to-end guide](../guides/multimodal-av-e2e.md) for the combined workflow and example configs.

Each invocation selects a model, context, prompt, and result identity. Model compatibility is the fixed product of
three model bindings and three context modes, for nine supported execution combinations:

| Axis | Supported values | Owner |
|---|---|---|
| Model | `qwen3_8_27b_fp8`, `qwen3_8_27b`, `cosmos3_nano` | Recipe-owned model runtime |
| Context | `video_only`, `ego_trajectory_summary`, `ego_trajectory_absolute_poses` | Context representation and default prompt |
| Prompt/result | Required `default` or `custom` prompt selection; `output.result_key` defaults to `default` | Caller |

The count of nine covers model × context only. Prompt selection and result identity are per-invocation settings, so
each model/context combination can publish multiple independently keyed field pairs.

There is no model/context pair registry. Every model works with every context. Qwen samples video at 2 FPS;
`cosmos3_nano` samples at 4 FPS, uses the native Cosmos3 vLLM executor, and supplies its closed-reasoning assistant
prefill. Each model binding owns its fixed 32,768-token model-length ceiling. Contexts do not own or configure model
runtime.

The recipe uses the current `video-caption` complete-MP4 inference, checkpoint, staging, and fragment-atomic Lance
publication engine. `video-caption` keeps its clip schema, fields, literal contract digests, prompt, and runtime; the
recipes share only private execution machinery.

## Context modes

All three context modes caption the selected camera video. The two Ego modes additionally use deterministic,
recipe-owned preprocessing to derive canonical trajectory context from the episode-level `egotrajectory` MCAP
produced by `multimodal-split`. This gives the model information about the Ego vehicle's motion without requiring it
to infer that context from video alone.

`video_only` sends the selected complete camera MP4 with this context-owned default prompt:

```text
Elaborate on the visual and narrative elements of the video in detail.
```

Both Ego modes use the same driving-oriented default prompt, followed by exactly one literal
`{ego_trajectory_context}` sentinel:

```text
The video depicts the view from a camera mounted on a car as it is driving. Pay special attention to the motion of the cars, including the primary car whose point-of-view we observe in the video. Also note important factors that would relate to driving safety like the relative positions of pedestrians, lane markers, road signs, traffic signals, and any aggressive driving behavior of other vehicles. Also pay attention to interesting landmarks and describe them in detail.

Ego trajectory context for this episode (duration is the full episode span; motion covers only the recorded in-span poses):
{ego_trajectory_context}
```

The adapter replaces that literal sentinel once with canonical JSON. It does not use Python format-string rendering,
so unrelated braces remain unchanged.

The two Ego representations share the same decoder, qualification, and numeric summary:

- `ego_trajectory_summary` serializes `frame`, `representation_version`, and `summary` only.
- `ego_trajectory_absolute_poses` adds one `samples` item per qualification request. Each item contains
  `relative_time_ns`, `requested_align_timestamp_ns`, `position_m`, and `orientation_xyzw`.

Summary metrics use every recorded pose whose MCAP `log_time` lies in the half-open episode span, in time order.
`path_length_m` is the sum of the 3D Euclidean distances between consecutive in-span positions;
`net_displacement_m` is the 3D Euclidean distance from the first in-span position to the last. Both are zero when the
span contains one recorded pose. Path length is sensitive to recording rate and pose noise, so it is descriptive rather
than a trajectory-precision or quality measure. The summary describes the native decoded artifact; it does not infer
missing poses, classify motion, or assume a vehicle-forward axis.

Both Ego modes perform the same fixed-grid qualification. Summary mode omits the `samples` array; absolute mode
serializes the nearest qualified pose for every request. In each absolute sample, `relative_time_ns` and
`requested_align_timestamp_ns` describe the requested grid instant, not an observed source-pose timestamp. The
associated position and orientation come from the nearest recorded pose within the inclusive 20 ms tolerance.

## Input and source binding

The input must be a canonical episode table with globally unique `episode_id` values and the current record and media
contract versions. Every invocation requires one non-empty `context.camera_key`. Before any result fields are
registered, preflight scans the complete table and resolves exactly one matching descriptor from every row's complete,
lexicographically serialized `camera_media` list. For `video_only`, the camera-resolution scan reads `episode_id` and
`camera_media` but does not scan `sensor_media`; separate base-schema and global-identity checks validate the contract
versions and episode IDs. Missing or duplicate camera descriptors fail without claiming the result namespace. The scan
cost is paid up front; after successful preflight and digest-bound field registration, the recipe reopens and pins the
attempt version. Its execution projection then carries `materialization_digest` with the selected camera binding.

`materialization_digest` binds the complete published episode artifact set: every camera MP4 and timing sidecar plus
every planned non-camera sensor MCAP. The selected camera's `media_sha256` and, for Ego modes, `ego_media_sha256`
identify the exact artifacts consumed by this invocation. The composite digest and per-artifact hashes therefore serve
different, complementary scopes.

The selected MP4 is checked against its exact byte size and SHA-256. Its durable source binding is:

```text
(episode_id, materialization_digest, camera_key,
 media_location, media_size_bytes, media_sha256)
```

Both Ego modes extend that same pre-registration scan by reading the episode bounds, timestamp source, and
`sensor_media` descriptors for every row. The scan resolves the configured `egotrajectory` descriptor and validates a
positive episode duration within the maximum and `timestamp_source == "sensor"`; it does not fetch the referenced
camera or Ego media bytes. The projector repeats both guards against the pinned attempt and carries the bounds into
context construction. Bounds are not staged recovery columns because the producer's versioned `episode_id` contract
commits to them. The projector stages the actual timestamp source together with the Ego URI, size, and SHA-256.
Publication and recovery require both the staged and current live value to remain `sensor`. There is no clock
conversion or fallback.

`video_only` has no Ego restriction and continues to accept canonical episodes whose `timestamp_source` is
`container_pts` after a configured camera used its explicit PTS fallback.

## Ego decoder and qualification contract

The two Ego modes consume the complete split MCAP named by `context.ego.sensor_key`. The source is fixed to the
`foxglove.PoseInFrame` schema and a recipe-owned protobuf mapping. The mapping decodes the embedded
`PoseInFrame.timestamp` as `sensor_timestamp_ns` source provenance and derives `align_timestamp_ns` from MCAP
`log_time`. MCAP log time controls span membership, predecessor selection, nearest-pose qualification, and summary
calculation. The embedded timestamp is not an alignment coordinate and is not serialized or durably persisted in Ego
context version 1. Both timestamp roles and the parsed mapping digest are contract identity. Callers configure only
the sensor key, topic, and frame.

Captioning relies on the producer contract that Ego MCAP `log_time` and the episode clock share a session-local domain.
`timestamp_source: sensor` identifies the recorded sensor timestamps as the source of the camera timelines rather than
the configured `container_pts` fallback, while the 20 ms gate checks only local request-to-pose proximity. Captioning
does not independently verify the cross-stream relationship; neither condition proves camera-to-Ego clock equivalence.

Ego context version 1 requires:

- `timestamp_source: sensor` for the episode row;
- an episode duration of at most 10 seconds;
- at least one recorded pose in the half-open episode span;
- `pose_valid == true` for every decoded row, including predecessor halo rows;
- requests at `episode_start + k * 500 ms` while the request remains before `exclusive_end_ns`; and
- nearest recorded poses with an inclusive maximum absolute delta of 20 ms, without interpolation.

The request count is `ceil(duration_ns / 500_000_000)`. The recipe does not interpolate, pair camera frames with Ego
poses, or claim frame-accurate synchronization between sampled video frames and serialized poses.

`context.ego.frame` is a caller-supplied source-coordinate label serialized into the context payload. Version 1 does
not inspect `PoseInFrame.frame_id`, transform coordinates, infer frame meaning, or verify a physical vehicle convention.

Invalid table identity or versions, missing or duplicate selected camera or Ego descriptors, non-sensor timestamps,
and invalid or over-limit episode spans fail during the full-table preflight without registering result fields. After
registration, item failures classified by the camera/Ego media readers or Ego context builder—byte-integrity,
unavailable-object, decode, pose-validity, and alignment failures—produce a terminal error only for the affected
episode. Ego media is fetched only after the selected camera MP4 fetch and validation succeed, so a camera failure
skips the Ego download for that episode. Unhandled shared-storage, infrastructure, or stage failures abort the phase
instead of being converted into item results.

Tables written before record schema version 2 and media contract version 3 are incompatible with this input contract
and must be regenerated by `multimodal-split` into a fresh output root.

## Public configuration

Configuration is strict and versioned; unknown fields are rejected. Context and prompt mode are required. The model is
optional and defaults to `qwen3_8_27b_fp8`; the whole output block is optional for a default-prompt run and resolves to
`result_key: default`, `tag_camera_key: false`, and the derived staging root. Omitting output, writing `output: {}`,
and writing the explicit default resolve identically on both single-camera and multicamera tables. The block, key,
and tagging flag are not nullable.

A complete default-prompt invocation is:

```yaml
schema_version: 1
kind: multimodal-caption

input:
  episodes_lance_uri: s3://example-bucket/curated/multimodal-split/episodes.lance

context:
  variant: ego_trajectory_summary
  camera_key: front_wide
  ego:
    sensor_key: ego
    topic: /vehicle/pose
    frame: world

model:
  variant: qwen3_8_27b_fp8

inference:
  prompt:
    mode: default

execution:
  storage_profile: default
```

The generated full template shows `output.result_key: default` and `output.tag_camera_key: false` for discoverability
even though that default-prompt configuration may omit them.

A custom context-free prompt can use an opaque result identity such as `safety_focus`; that is an example result key,
not a context variant or preset:

```yaml
schema_version: 1
kind: multimodal-caption

input:
  episodes_lance_uri: s3://example-bucket/curated/multimodal-split/episodes.lance

context:
  variant: video_only
  camera_key: rear

model:
  variant: cosmos3_nano

inference:
  prompt:
    mode: custom
    template: >-
      Describe the driving scene visible from this camera, including safety-relevant motion.

output:
  result_key: safety_focus

execution:
  storage_profile: default
```

`output.result_key` is the base label for an independently retained caption result. It defaults to `default` and must
match `^[a-z][a-z0-9_]{0,47}$`. Default prompt mode may use any valid base key. Default-prefixed keys are reserved for
default-prompt results; they do not reserve a key for a particular camera. Custom mode requires an explicit base key
that does not start with `default`, rejecting `default`, `default_front_wide`, and `defaulted` before tagging,
registration, or recovery. Published results remain readable; custom-prompt configurations using default-prefixed
base keys cannot resume or publish additional results under those keys.

With `output.tag_camera_key: false` (the default), the effective key is the literal base key. With tagging enabled,
it is `<result_key>_<camera_key>` using exact configured strings: base `default` and camera `front_wide` produce
`default_front_wide`. Tagging always appends, even if the base already contains the camera name. The effective key
must match the same pattern and 48-character limit. Cameras containing uppercase letters, dots, dashes, or names
too long for the composed key require tagging off and an explicit valid result key; names are never slugified or
truncated. Resolved configuration preserves the base and flag, so reloading cannot append twice.

Custom `video_only` prompts contain zero literal `{ego_trajectory_context}` sentinels. Custom Ego prompts contain
exactly one. A custom template is persisted byte-for-byte after YAML parsing, so scalar folding, indentation, and
chomping affect result identity.

Prompt mode is provenance in resolved configuration and the run summary only. The exact resolved, unrendered prompt
template—not the mode—is persisted in terminal metadata and bound into the normalized contract and its digest.

`output.staging_root_uri` defaults to `<episodes.lance parent>/staging/multimodal-caption`. Model weights must already
exist under the standard model mount on every inference worker; the recipe never downloads them. `cosmos3_nano` is not
in the default weight set and must be staged explicitly when used:

```bash
pixi run --as-is model-download --models cosmos3_nano
```

## Result contract

Each invocation owns one nullable caption field and one nullable metadata field:

```text
caption_{context_variant}_{effective_result_key}_{model_key}_v1
caption_{context_variant}_{effective_result_key}_{model_key}_v1_metadata
```

For example, the default Qwen FP8 summary fields are
`caption_ego_trajectory_summary_default_qwen3_8_27b_fp8_v1` and its `_metadata` sibling. The same result key can coexist
across contexts because the context is part of the field name. Different effective keys create distinct field
namespaces and contract digests even for identical prompts.

Opt-in tagging is the camera-naming exception: the camera appears through the effective result-key segment, with
no separate camera axis in the v1 formula. The selected camera remains in terminal metadata and the normalized
contract. Model and context already distinguish results; retaining another same-model/context result requires a new
key rather than another episode row. See the [end-to-end guide](../guides/multimodal-av-e2e.md).

Reusing `(table, context_variant, effective_result_key, model_key)` with another camera, prompt, model revision, or
context setting fails before inference or episode-result publication, even if registered fields still contain null
values. The multimodal consumer preserves the original diagnostic and suggests a new result key or enabling camera
tagging; if tagging is enabled, it suggests a new base key. Collision classification uses both fields' schema metadata,
without inspecting rows or guessing which input changed. Partial field pairs and unrelated schema errors keep their
original diagnostics. No retry under another key or replacement occurs. Ambiguous concatenations such as base `a_b`
with camera `c` and base `a` with camera `b_c` are protected by the same full-contract check.

The normalized contract binds result key, context, camera, exact model identity and precision, unrendered prompt,
resolved message, sampling and model runtime, logical result schema, terminal policy, and all applicable Ego source and
representation values. Its digest is attached to both registered top-level fields.

The normalized contract, contract digest, row metadata, transaction properties, and recovery workspace use only the
effective key as `result_key`. The run summary records `base_result_key`, `tag_camera_key`, and effective `result_key`.
Base and flag are configuration provenance, excluded from row metadata, transaction properties, and contract identity.
A manually named key and a tagged key resolving to the same effective key recover the same matching contract.

Terminal metadata repeats the direct result, context, camera, model, prompt, and contract identifiers. Ego metadata
also carries representation version, frame, cadence, maximum delta, sensor identity and hash, decoder-mapping digest,
and the canonical context digest. Terminal states are `success`, `truncated`, and `error`.

An item-scoped error leaves the caption null and records `status: error` with non-empty `error_type` and
`error_message` values in terminal metadata. For current item-scoped Ego failures, `context_digest` is null because
camera fetch, Ego fetch, decoding, pose validation, and alignment occur before context construction completes.
Multimodal preparation, inference, shared-storage, and runtime failures abort the phase rather than producing terminal
item rows.

The workspace is rooted by caption field and contract digest. A run with a different caption field or contract digest
does not resume, or automatically remove, an interrupted run's previous workspace. Its fail-closed manifest contains
the complete normalized contract and direct field, model, context, and result identifiers. Transaction properties carry
recipe kind, result key, context, model key, and contract digest. Completion classification, durable Phase A recovery,
and fragment-atomic publication operate only on that selected field pair. An unchanged rerun skips valid terminal
`success`, `truncated`, and `error` results; it does not retain a fresh sample or automatically retry terminal errors.
A successful fragment commit remains canonical if later work fails.

## Extension, privacy, and unsupported behavior

A future multiview design must decide whether results are separate per camera, fused across cameras, or both before a
formal camera naming axis is added. Audio, clock conversion, new Ego representations, learned motion classifications,
split-and-caption orchestration, and automatic prompt registries are outside this version.

For Ego modes, neither the published Lance result nor the durable recovery Parquet stores the per-row rendered prompt
or canonical Ego JSON. Published terminal metadata retains the exact unrendered prompt template and source identifiers;
Ego metadata also retains the canonical context digest. The durable recovery Parquet additionally retains the staged
source binding. Camera MP4s, Ego MCAPs, runtime requests containing rendered context, temporary execution data, and
generated captions may contain sensitive location, trajectory, or scene information. Runtime checkpoints, spills, or
logs may also contain sensitive intermediates. Tests and repository fixtures use synthetic data only. Operators must
apply their storage access, retention, and logging policies to source artifacts, staging workspaces, terminal metadata,
and captions. The recipe records provenance and integrity; it does not anonymize or redact content.
