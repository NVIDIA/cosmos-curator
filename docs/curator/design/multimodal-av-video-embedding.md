<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multimodal AV Video Embedding

## Summary

`multimodal-embedding` reads canonical `episodes.lance` rows from
[`multimodal-split`](multimodal-av-episode-publication.md). It embeds selected camera MP4s per episode and writes
the results to the same Lance table. Every run checkpoints an unambiguous work identity derived from
`(episode_id, camera_key)`, including runs selecting one camera. Cosmos-Embed1 is supported at 224p,
336p (default), and 448p. Stage the selected model assets under `/config/models` on the driver and all nodes that
may run recipe actors. GPU inference needs checkpoint weights; CPU preparation needs checkpoint processor files.
Preflight checks that weights are present and that model and processor config values (frames, resolution, dimension)
match the pinned contract; it does not verify artifact revision. The model and processor revisions in result metadata
identify the intended contract, not an independently verified revision of the staged files. A shared model-cache mount
can make the assets available to both CPU and GPU workers.

## Camera selection

Embed all cameras by default. Omitting `video`, leaving it empty, or leaving both selectors unset resolves to
`video.camera_keys: all`. Limit a run to selected views with:

```yaml
video:
  camera_keys: [front_wide, left_fisheye]
```

The generated template makes the default explicit:

```yaml
video:
  camera_keys: all
```

Select one camera with `video.camera_keys: [front_wide]`. Lists must be nonempty with distinct names and resolve
in deterministic order. `all` discovers the camera names from the input table and resolves to a list before execution;
every selected camera must be present in every episode. Missing cameras and heterogeneous camera sets fail preflight.

Each selected camera gets its own vector and metadata fields. `output.tag_camera_key` defaults to `true` and uses the
exact camera-tagging rule from multimodal captioning: `<base_result_key>_<camera_key>`. For base `default`, camera
`front_wide` publishes
`embedding_video_default_front_wide_ce1_336p_v1`, and `left_fisheye` publishes
`embedding_video_default_left_fisheye_ce1_336p_v1`. The config retains `output.result_key: default`;
metadata and field contracts use the effective key. Rendering and reloading a config does not append a second suffix.

Effective keys must match `^[a-z][a-z0-9_]{0,47}$`. Camera names are never renamed, hashed, or truncated.
Invalid or overlong tagged keys fail before output registration; use a shorter base key, or select exactly one camera
and set `output.tag_camera_key: false` with an explicit valid `output.result_key`. Disabling tagging uses the base key
unchanged, matching captioning's naming opt-out. Multi-camera lists and `all` require tagging to prevent collisions.
Namespaces depend only on the configuration and camera identity, never on which fields are already registered.
The scalar `video.camera_key` option is not supported. There is no fallback to earlier untagged embedding fields and
no migration of episode-only staging/checkpoints; all selections use episode/camera work identities.

The run summary reports the base key, effective keys, and field names for each camera. Adding another camera computes
only its missing results; list order and execution tuning do not change individual camera contracts.
One-camera lists and larger selections share the same execution, staging, and publication path.

Internally, each `CameraEmbeddingPlan` owns its camera identity, effective key, fields, and contract digest.
An `EmbeddingRunPlan` collects those plans and the shared episode publication state. Publication builds a narrow
adapter for the existing Lance guard helpers explicitly; it does not inherit a first camera's state or callbacks.
Aggregate guards project episode identity only, while each camera's staged media bindings are validated separately.
The deterministic workspace field anchor and contract digests are preserved across this internal representation change.

An empty table does no media, staging, Ray, or model work. Explicit-list selectors still register missing output fields,
which can create new Lance versions. With `all`, no output fields are registered; the table stays unchanged and the
summary contains no cameras.

See [the multi-camera embedding example](../../../examples/multimodal-multicamera-embedding.yaml) for the
`front_wide`/`left_fisheye` selection used by the caption examples and
[end-to-end guide](../guides/multimodal-av-e2e.md#embed-selected-cameras-together).

One pipeline batches videos from different cameras through shared model actors. Inference batch size counts videos,
not episodes. Durable camera rows are validated and assembled into episode rows before one atomic result commit across
all selected cameras and fragments. Already completed camera columns are carried through unchanged. Restart under the
same selection reuses camera checkpoints; unpublished staging for a different selection is isolated by its workspace
contract. Published camera results remain reusable across selections. Use one active writer per selected result-field set.
Runs for disjoint camera fields and unrelated caption updates can share the table: publication rebuilds stale descriptors
from durable staging and preserves those unrelated changes. A competing write to any selected embedding field is
rejected, even if it contains identical vectors; this recipe does not adopt a live competing writer's results or take
an execution lock. A rerun after that writer finishes captures a fresh snapshot and reuses its completed results.
Only this run's publication receipt, together with matching canonical values, reconciles its own ambiguous commit.
Any pending results still become visible in one atomic update without repeating completed inference.
If another selection publishes a camera while a run is interrupted, recovery validates the old staged row's source
binding and the canonical terminal result, then carries the published result through unchanged. Staged cameras outside
the run's selection or rows with changed episode/media bindings remain errors.

## Result contract and eligibility

The recipe writes a nullable fixed-size float32 vector field and a metadata sibling. The selected camera key, model
and processor revisions, frame sampling rule, vector shape, and metadata schema determine the result contract. Both
fields carry its digest, so an existing result key cannot be reused with an incompatible contract.

Vector fields are named `embedding_video_{effective_result_key}_{model_field_key}_v1`, with a `_metadata` sibling.
The model field key is `ce1_224p`, `ce1_336p`, or `ce1_448p`. Camera-tagged effective keys distinguish selected views;
camera identity is also recorded in metadata and protected by the contract digest. All camera selections use these
fields by default. An explicit one-camera list with tagging disabled uses the base key, while retaining the same
camera provenance, contract checks, and recovery path.

All variants take eight endpoint-inclusive frames selected with `np.linspace(0, frame_count - 1, 8, dtype=np.int64)`
in decoded presentation order, matching the Cosmos model-card example.
Videos shorter than eight frames repeat indices. The pinned processor receives RGB uint8 BTCHW without resize,
scaling, or normalization overrides. Cosmos-Embed1 produces 256 values at 224p and 768 values at 336p/448p.
Inference runs in BF16; the model-normalized `visual_proj` is cast to float32 without another normalization step.
Both row and batch validators use `np.isclose(norm, 1.0, atol=0.02)` to tolerate BF16 rounding.
Vectors are not guaranteed to be numerically equivalent to the classic splitting pipeline's for the same clip.

These sampling and output semantics change the contract digest. The unreleased schema and field suffix stay at v1,
but previous demo fields require a fresh result key. Existing fields or staging with incompatible contracts are
refused; there is no implicit migration or overwrite.

The configured maximum episode duration determines eligibility and is recorded per row. An over-limit episode receives
`unsupported_duration` with no vector. This differs from the preflight rejection used by the ego-trajectory caption
context; caption's video-only context has no duration preflight limit. Raising the maximum currently does not revisit
already terminal rows. `selection.retry_errors: true` is rejected until completed-fragment rewrite support is added.
Lowering the maximum does not remove previously published vectors. Readers can inspect the recorded episode duration
and maximum used for each row.

## Publication and recovery

A finite attempt reads pending Lance fragments and stages one terminal row per episode in Parquet. Workers then
validate exact row coverage and prepare fragment updates. The driver collects only the small publication descriptors
and commits all pending fragment results in one Lance Update. On a backend meeting the storage requirements below,
readers see the selected new results together in one atomic table version.
Atomic publication is the recipe's only publication path; there is no mode setting. The embedding recipe's
`publication.py` owns its transaction, receipt, and recovery logic, reusing shared fragment preparation and guards. Registering nullable result fields
may be a separate setup commit; it does not publish any embeddings. Caption recipes retain their existing per-fragment
publication entry point.

### Storage requirements and qualification

Atomic publication depends on Lance's commit handler and the storage backend enforcing atomic create-if-absent
operations, such as conditional PUT or atomic rename-if-not-exists. They ensure only one writer can publish a given
manifest version. See the [Lance commit protocol](https://lance.org/format/table/transaction/).

The recipe currently accepts local paths and `s3://` locations. Backend status is:

- **Native AWS S3:** supplies the required conditional-write primitive through `If-None-Match`, including rejection
  of competing writes to an existing object. This is a documented backend guarantee, not an end-to-end qualification
  of a production storage profile. See [AWS conditional writes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/conditional-writes.html).
- **Local filesystems:** require atomic create-if-absent or equivalent no-replace rename semantics. The recipe's
  CPU publication and recovery tests exercise local filesystem storage. This coverage does not qualify every
  filesystem or shared mount used in a deployment.
- **Custom S3-compatible endpoints:** require confirmation that the deployed service, bucket configuration, and
  any intervening gateway enforce the conditional-write operation used by Lance. S3 API compatibility alone does
  not establish this guarantee; these endpoints have not been qualified for this recipe's atomic publication.
- **Shared filesystem mounts, including NFS:** require deployment-specific verification of atomic manifest creation
  and read visibility across writers. A local-looking path is not evidence of these guarantees; these mounts have
  not been qualified for this recipe's atomic publication.

No named production `execution.storage_profile` is qualified for this recipe's atomic publication. Profiles select
credentials and connection settings; names such as `default` do not identify verified storage capabilities. These are deployment requirements;
the recipe does not probe or enforce them during preflight. Azure and GCS URI schemes are not accepted by this
recipe's current location configuration.

The recipe does not supply an external `commit_lock`. A backend lacking the required atomic operations needs an
appropriate commit-coordination mechanism shared by every writer to the table before this recipe can be used safely
there. Brief receipt-read retries after an ambiguous commit response do not provide that coordination or qualify a
backend with arbitrary read-visibility delays.

### Staging and recovery boundaries

The temporary Parquet vector is a nullable variable-length list so item errors can be staged; publication validates
its length and casts it to the canonical fixed-width Lance field. Publication guards protect episode bindings, row
identities, field contracts, and owned results. Producer appends outside the finite selection remain pending. Disjoint
caption updates are preserved by rebuilding stale fragment descriptors from staging. Conflicting input or embedding
writes are rejected; guards do not lock separate runs.

Publication deliberately re-reads and hashes the vectors and metadata of every previously complete fragment in the
finite selection. This detects changes to completed results since the attempt snapshot, including results that the
current update does not modify. The checks repeat on commit retries and ambiguous-response reconciliation reads,
so their cost scales with the total completed result data selected, rather than just the new results being published.
For scale, one million 768-dimensional float32 vectors contain about 3 GB of vector data per full pass, before
metadata. This is a data-size estimate; no production-scale benchmark or wall-clock threshold has been established.
The initial selection and no-op rerun validation also scan completed rows. A future optimization could avoid
rehashing unchanged immutable Lance files after verifying their identities; cached hashes alone would not detect
changes to the results they describe.

A scoped transaction receipt binds the contract, workspace, selected input identities, and authoritative staged
result fingerprints. If a commit response is lost, the receipt and canonical values must both match before the attempt
accepts success. A row is complete when its vector field and metadata sibling represent a valid terminal state. A
failure during preparation or before commit leaves all selected pending fragments unpublished. A driver failure after
commit leaves all selected results published; restart validates them and performs no additional result commit.

Policy-scoped staging and Ray row checkpoints preserve completed inference across attempts. The completion marker is
written only after the full staging pass. Publication failures can rebuild descriptors without running the model again.
An unchanged rerun performs no inference or Lance commit, although it still scans terminal rows for validation. A later
run discovers appended episode fragments.

If a crash leaves a staged Parquet row without its Ray row checkpoint, a restart may encode that episode again. Before
publication, replay rows are checked against the current episode binding and result contract. One valid row per
episode is chosen deterministically, preferring a verified success over a later item error. Only the active policy
workspace is cleaned after verified publication. Abandoned workspaces under older duration or retry policies are
ignored and require explicit operator cleanup. Recovery after managed-cluster head loss has not yet been qualified.

## Failure behavior

Preflight validates the exact input snapshot selected for processing, including global episode-ID uniqueness and
record/media contract versions across pending and completed fragments. Every episode must contain exactly one
descriptor for the selected camera. If a camera is missing, repair the input or select a camera present in every
episode. Completed rows are also checked against their episode ID, materialization digest, camera key, media SHA-256,
and episode duration. If those bindings changed, restore the original binding or recompute under a new
`output.result_key`; ordinary reruns refuse to reuse stale results.

Published item errors, including `media_object_error` and `invalid_vector`, are terminal. Ordinary reruns keep them
even if the object later becomes available or a later model call would return a valid vector. Retrying published
errors is not available yet: use a new `output.result_key` to recompute results. This recomputes the selected episodes,
including previous successes, under separate fields; it does not overwrite the original results.

Download actors reserve one `curator_io` slot and 0.25 CPU, with one task in flight per actor. They fetch and
verify each eligible MP4 using the shared `MediaObjectReader`, then pass its encoded bytes through Ray's object
store to a separate CPU preparation stage. The reader checks the object-size ceiling before downloading, caches
storage clients, and retries transport failures; the actor also verifies the exact declared size and SHA-256. Over-limit episodes skip fetching, and download errors pass through without decoding or model execution.
CPU actors open the MP4 bytes in in-memory buffers and decode sequentially, retaining only selected RGB frames,
then preprocess those frames in `legacy-transformers`. The declared frame count is verified against the full decode;
if absent, a count pass precedes sampling. Corrupt frames, count mismatches, and inconsistent sampled dimensions
produce `decode_error`. Media duration is checked against the episode span with a two-frame-period tolerance.
Decode work covers the entire clip; there is no bounded keyframe-seek-work claim. Neither download nor preparation writes temporary
media files. They reserve `execution.decode_cpus`
(default 1) and `execution.decode_memory_gib` (default 2 GiB), without IO slots or GPUs. The memory setting is a
scheduling reservation, not a hard limit. PyAV decoder and Torch preprocessing threads are each limited to
`decode_cpus` on every actor incarnation, including the fallback frame-count pass. CPU preparation emits only a compact tensor and releases the MP4 payload;
GPU actors batch eligible tensors without holding IO slots. Neither MP4 bytes nor prepared tensors are durable
checkpoint outputs; the final result rows are checkpointed by episode/camera identity before atomic publication.

Before downloading, projected episode metadata is repartitioned into blocks of at most
`execution.lance_read_batch_size` rows. This exposes large Lance fragments as multiple work blocks without
partitioning downloaded MP4 payloads. Episode IDs and physical row addresses are preserved for checkpointing and publication.

All three actor pools can use workers that join after the run starts. Download and CPU preparation pools autoscale
using Ray's normal controls. The shared `configure_download_actor_autoscaling` callback releases surplus pending
download actors when ready actors have spare capacity, allowing idle actors to scale down and release their IO slots.
The callback targets only pools reserving `curator_io`. Both download and preparation scale
without a fixed maximum derived from startup resources. `execution.media_concurrency` and
`execution.decode_concurrency` default to `auto`; a positive integer caps the corresponding actor pool to limit
storage traffic or preparation memory demand. The GPU pool ceiling is the pending row count for
`inference_concurrency: auto`; an explicit limit is capped by that count. This uses datasource metadata rather than a
Ray counting job or a startup GPU snapshot. Every actor accepts one in-flight task and executes one task at a time.
GPU inference batches default to four episodes; operators can increase `execution.inference_batch_size` after checking
memory use on their target GPU and resolution. These settings do not change the vector contract digest.
This recipe's execution context uses a reachable actor scaling threshold of 1.0 and retains Ray's conservative
maximum of one additional actor per scaling decision. Ray Data controls admission and backpressure within those limits.

This separation follows `ray-data.md`'s download admission guidance. Encoded MP4 bytes now cross the object store,
which adds payload memory and transfer overhead compared with keeping an MP4 inside a combined actor. The declared
media size bounds each verified payload, and Ray backpressure regulates queued data. Throughput has not been
measured; measure fetch time, decode/preprocessing time, episode throughput, and GPU utilization before claiming
scaling performance.

All three stages allow five restarts per actor and five retries per actor task, following the robot embedding and
robot-action-caption recipes. Repeated worker crashes or node losses can exhaust these limits and fail the attempt;
operators can relaunch using the existing staging checkpoints. This differs from multimodal-caption's unbounded
worker recovery. These limits do not enable retries for ordinary Python exceptions from processor or model code.

Shared-reader object faults (including missing, empty, or oversized objects) produce terminal `media_object_error`
rows. Exact-size and SHA-256 mismatches after a successful read retain their specific item codes; decode failures
and invalid vectors also remain item errors. Transport failures after retry exhaustion and systemic storage faults
abort the attempt without publishing item errors. Archived S3 objects (`InvalidObjectState`) follow the shared
reader's transport classification and abort after retries, rather than producing a permanent `media_unavailable`
row. This replaces the previous recipe-specific fetch error codes and changes the contract digest; an existing
result key created under the earlier contract needs a new result key. The GPU actor encodes only rows with prepared tensors and carries media errors and over-limit rows through to
publication. A bad projection affects its own row; a GPU failure such as out-of-memory aborts the batch and leaves the
attempt retryable. The run summary counts terminal statuses and error codes.
