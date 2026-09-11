# Curator Next: Video Captioning

## Summary

`video-caption` enriches the canonical clip table produced by Curator Next
[`video-split`](curator-next-video-split.md). It reads complete clip MP4s, generates one caption per clip with Qwen3.8
through Ray Data LLM and vLLM, and fills a versioned caption field set on the existing Lance rows.

The design separates expensive inference from canonical publication:

1. **Phase A: checkpointed inference** runs one streaming Ray Data plan and writes terminal caption rows to durable
   Parquet. Ray checkpoints completed `clip_id` values at the file-sink boundary so a restart skips media fetch and
   inference for completed work.
2. **Phase B: fragment publication** validates the staged rows, prepares Lance column files on Ray workers, and commits
   one complete physical fragment at a time from the driver.

These phases address different recovery units. Ray's checkpoints prevent repeated row-level GPU work, while Lance
transactions make fragment-level progress canonical. Parquet connects the two: it is durable recovery data until the
corresponding fragments are verified in Lance.

## Context and Responsibilities

`video-split` owns row creation, fragment creation, and clip media. Captioning owns only its selected caption and
metadata fields. It does not discover sources, split videos, rewrite media, append rows, or change row order.

The design relies on these properties of the upstream table:

- `clip_id` is globally unique and immutable.
- `clip_uri` and `clip_size_bytes` are non-null and immutable for an existing row.
- Each clip is a standalone H.264 MP4 that was probed after transcoding.
- Splitting is append-only; new fragments may appear while captioning is running.
- Nullable enrichment fields may be added without changing the splitting-owned fields.

One logical writer owns a given caption field set. `video-split` and enrichment writers for disjoint field sets may run
concurrently when they follow the fragment-reconciliation protocol described below. Multiple writers for the same
field set are outside the consistency model.

## Design Invariants

The implementation preserves the following invariants across normal execution, task replay, process failure, and Ray
head loss:

1. **Lance is canonical.** Downstream readers never need the staging workspace.
2. **Publication is fragment-atomic.** A fragment contains either all pending values or a complete, valid caption field
   set; mixed state is rejected.
3. **Durable inference is reusable.** Once a result shard and its checkpoint are committed, a restart does not fetch or
   infer those clips again.
4. **Staged rows cannot silently rebind.** Phase B proves that staged `clip_id` values still describe the selected
   physical fragment before publishing them.
5. **Concurrent disjoint changes are preserved.** A stale same-fragment descriptor is rebuilt from Parquet against the
   latest fragment rather than attached to a newer read version.
6. **An attempt is finite.** Work is selected from one pinned Lance snapshot. Rows appended later wait for a subsequent
   attempt.
7. **Result identity is independent of execution tuning.** Cluster size, batching, worker placement, and retry timing do
   not change the caption contract.

## State and Authority

Several durable objects participate in recovery, but they are not equally authoritative:

| State | Role | Authority and lifetime |
| --- | --- | --- |
| Lance clip table | Published clip metadata and caption fields | Canonical and long-lived. A fragment commit survives later failures. |
| Result Parquet | Terminal caption rows produced by Phase A | Authoritative recovery input until all selected fragments are canonical. |
| Ray checkpoint files | Committed `clip_id` values for result shards | An index over safely written Parquet. Used to filter work, not as caption data. |
| `workspace.json` | Contract and schema compatibility manifest | Protects a workspace from reuse by an incompatible caption contract. |
| `phase-a-complete.json` | Fragment-level completion fast path | An optimization only. Its absence never invalidates Parquet or checkpoints. |

The workspace is scoped by caption field and contract digest rather than by run or Lance version. That choice allows a
new attempt to reuse completed clips from an earlier attempt while still selecting newly appended fragments from the
latest table.

## Attempts and End-to-End Flow

Each invocation captures the latest table as `V_attempt` after registering or validating the selected caption fields.
Every fragment in that snapshot is classified as one of two valid states:

- **pending:** both caption-owned fields are null on every row;
- **complete:** every row is terminal and matches the selected contract digest.

A fragment with mixed pending and terminal rows, a partial field set, stale contract metadata, or invalid terminal
values fails before new work begins.

The execution flow is:

```text
Driver
  validate input and caption fields
        |
        v
  capture finite Lance snapshot V_attempt
        |
        +-- complete fragments ------------------------------+
        |                                                     |
        v                                                     |
  pending fragments                                           |
        |                                                     |
        v                                                     |
  Phase A: install Ray CheckpointConfig                       |
        |                                                     |
        v                                                     |
Ray Data                                                      |
  pinned Lance read -> CheckpointFilter -> fetch -> vLLM -> result Parquet
        |                                                     |
        v                                                     |
Driver                                                        |
  clear CheckpointConfig -> Phase B publication <-------------+
        |
        v
  verify every selected fragment at latest Lance -> clean workspace
```

If no fragment is pending, the invocation skips model setup and Ray inference. It still verifies the selected canonical
state and validates any pre-existing workspace before cleanup.

## Caption Result Contract

### Field ownership and registration

The recipe adds the caption and metadata fields together in one Lance schema transaction. Both top-level fields are
nullable so existing rows and future split appends begin pending. Their Arrow metadata records the owner and field set:

```text
cosmos_curator.owner = video-caption
cosmos_curator.field_set = <caption-field-name>
```

An existing field set must match its expected names, Arrow types, nullability, and ownership metadata exactly. A
result-defining change receives a new versioned field name instead of changing the meaning of an existing `_v1` field.
Schema registration retries disjoint schema races, but a partial or conflicting caption field set fails.

The FP8 field set is representative:

```text
caption_qwen3_8_27b_fp8_v1: large_string

caption_qwen3_8_27b_fp8_v1_metadata: struct<
  caption_schema_version: int32 not null,
  contract_digest: string not null,
  status: string not null,
  model_id: string not null,
  model_revision: string not null,
  prompt: large_string not null,
  prompt_token_count: int64,
  generated_token_count: int64,
  error_type: string,
  error_message: large_string
>
```

The BF16 variant uses the same types under its variant-specific field names.

### Terminal values

A non-pending row has exactly one terminal state:

| Status | Caption | Error fields | Meaning |
| --- | --- | --- | --- |
| `success` | Non-null | Null | Generation completed below the output-token limit. |
| `truncated` | Non-null | Null | Generation reached the output-token limit. |
| `error` | Null | Non-null type and message | A deterministic per-clip media failure exhausted its retries. |

Every terminal metadata value records the schema version, contract digest, model ID and revision, prompt, and available
token counts. The metadata struct is null only while the row is pending.

### Contract identity

The logical result identity is `(clip_id, caption_contract_digest)`. The digest is the SHA-256 hash of canonical JSON
containing every result-defining value:

- caption schema version and exact Arrow field schemas;
- model ID, immutable revision, and precision;
- prompt, message order, and chat-template arguments;
- sampling and output-token settings;
- video preprocessing settings; and
- terminal-state and failure semantics.

Data-URL encoding, block sizes, concurrency, retry counts, and worker placement are execution details and are excluded
from the digest. Sampling is intentionally stochastic, so the first compatible result durably committed to Parquet is
authoritative; retries are not expected to reproduce identical text bit-for-bit.

### Current v1 contract

The current contract follows the revision-pinned
[Qwen3.8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/blob/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/README.md):

| Setting | Value |
| --- | --- |
| Prompt | `Elaborate on the visual and narrative elements of the video in detail.` |
| Message | One user turn containing the exact MP4 as a `video_url` data URL, followed by the prompt. |
| Thinking | Disabled through the model-shipped chat template. |
| Sampling | Temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0.0`, presence penalty `1.5`, repetition penalty `1.0`. |
| Maximum output | 2,048 tokens. |
| Video input | Complete MP4, sampled at 2 FPS and clamped to 4-768 frames; prepared frames are not sampled a second time. |
| Audio | Ignored. |

Supported variants resolve their model IDs and immutable revisions from
`cosmos_curator/configs/all_models.json`:

| Variant | Model ID | Precision | Caption field |
| --- | --- | --- | --- |
| `qwen3_8_27b_fp8` | `Qwen/Qwen3.8-27B-FP8` | FP8 | `caption_qwen3_8_27b_fp8_v1` |
| `qwen3_8_27b` | `Qwen/Qwen3.8-27B` | BF16 | `caption_qwen3_8_27b_v1` |

## Recovery Workspace

The default workspace layout is:

```text
<staging-root>/<caption-field>/<caption-contract-digest>/
  workspace.json
  phase-a-complete.json
  results/
  checkpoints/
```

`workspace.json` is created atomically before the first inference attempt. It records normalized media and Lance URIs,
the output schemas, pinned model identity and runtime path, the normalized caption contract, and workspace/checkpoint
schema versions. It deliberately records neither `V_attempt` nor fixed work membership.

An existing manifest must match exactly before any staged data is reused or removed. Because a compatible workspace may
contain rows produced by several attempts, physical locations stored in Parquet are treated as hints. Phase B validates
them against the current fragment instead of assuming that an old `(fragment_id, row_offset)` is still valid.

### Ray Data checkpoint protocol

Phase A installs a job-level `CheckpointConfig` with `clip_id` as its identity column, `checkpoints/` as its durable
path, and checkpoint deletion on success disabled. Ray places `CheckpointFilter` directly after the datasource read.
When the workspace uses GCS through its S3/XML endpoint, driver and worker
filesystem construction follow the endpoint-scoped compatibility policy in
[Ray Data: S3-compatible object storage (GCS)](ray-data.md#s3-compatible-object-storage-gcs).

For each compact output shard, Ray's file-sink protocol performs this durability handshake:

1. write a pending checkpoint containing the shard's `clip_id` values;
2. write the corresponding result Parquet file;
3. mark the checkpoint committed only after the result write succeeds.

On recovery, Ray removes outputs associated with pending checkpoints, reads compact committed ID files, and filters
those IDs before media access. Any shard whose checkpoint remains pending may be replayed, but each replay unit is
bounded to one compact shard. A committed shard skips both download and inference. Curator does not duplicate this
protocol by scanning result Parquet or collecting checkpoint IDs on the driver.

After the complete Ray write succeeds, `phase-a-complete.json` records the attempt version, contract digest, and pending
fragment IDs. If a later attempt's pending set is covered by that marker, the recipe enters Phase B without loading the
model. If the marker is absent or does not cover the new attempt, the normal Ray plan runs and `CheckpointFilter`
removes already completed clips.

The job-level checkpoint configuration is always cleared before Phase B. This keeps Ray's checkpoint injection scoped
to the inference sink and prevents publication's own Ray Data reads from participating in the Phase A protocol.

The workspace is removed only after every fragment selected from `V_attempt` is canonical at the latest Lance version.
A cleanup failure does not invalidate committed captions; Lance remains authoritative and cleanup can be retried.

## Phase A: Checkpointed Inference

Phase A is one streaming execution over all fragments that were pending at `V_attempt`:

```text
pinned Lance read
  -> Ray CheckpointFilter
  -> physical row-address mapping
  -> strict lightweight blocks
  -> MP4 fetch and byte validation
  -> multimodal preparation
  -> chat template and tokenization
  -> vLLM
  -> terminal rows
  -> compact checkpoint/result shards
```

### Pinned Lance input

A recipe-local Ray datasource plans from inert fragment metadata. Read tasks contain only the Lance URI, pinned version,
fragment IDs, resolved storage options, and batch size; workers reopen Lance with those values. Live `LanceFragment`
objects are never serialized into tasks because their pickle path reopens the dataset without the configured storage
options, which can lose S3 credentials before execution begins.

The read projects only `clip_id`, `clip_uri`, `clip_size_bytes`, and Lance's physical row address. Checkpoint filtering
happens before the row address is decomposed into `fragment_id` and fragment-local `row_offset`, and before any media is
downloaded.

Lance fragments determine the number of source read tasks, but they do not cap later parallelism. A strict streaming
repartition of the lightweight rows fans a few large fragments out into fetch-sized work as an elastic cluster grows.

### Media and inference

Fetch tasks download the exact `clip_uri`, validate its byte length against `clip_size_bytes`, and encode the MP4 as a
data URL that can cross the Ray object store. The top-level data URL is removed once it has been moved into the OpenAI
message, and multimodal preparation replaces that message with decoded inputs. Large media values therefore do not
travel through tokenization, inference output, or Parquet.

The implementation uses Ray Data LLM's public processor surface and built-in multimodal preparation. Multimodal or
inference failures stop Phase A instead of being published as bad-media rows: the upstream splitter already transcoded
and probed each clip, so a later decode or engine failure indicates a runtime or contract problem. Deterministic
per-clip download failures become terminal `error` rows; credentials, storage availability, model loading, and other
shared failures stop the phase.

The vLLM actor pool has a work-sized upper bound rather than a GPU count captured at startup. Each replica requests
`tensor_parallel_size * pipeline_parallel_size` GPUs. The default TP=1, PP=1 topology places one replica per live GPU.
Currently available GPUs seed the initial actor count so model loading begins in parallel, while Ray's eager actor
autoscaling allows GPUs on later-joining managed Slurm nodes to be claimed without changing the plan.

Immediately before vLLM, a strict repartition creates exact `inference_batch_size` blocks. This compensates for the
ordinary `map_batches` rebundler, which may pass a 33-row multimodal block as 32 rows followed by a recurring one-row
engine call. Only the final global block may be smaller.

### Backpressure and reference tuning

The reference settings are execution choices, not result-contract inputs:

| Boundary | Default | Rationale |
| --- | --- | --- |
| Lance/read fan-out | 256 rows per strict block | Exposes media work beyond the number of input fragments. |
| Media fetch | 4 clips, 0.25 CPU, one `curator_io` slot | Bounds simultaneous downloads and active encoded MP4s per node. |
| Multimodal preparation | 24 GiB scheduling reservation per actor | Covers measured batch-32 heap peaks without host overcommit. |
| Chat-template stage | 16 GiB scheduling reservation per actor | Bounds CPU actor growth on large nodes. |
| vLLM | Batch 32, 8 concurrent batch calls | Matches the benchmarked legacy Ray Data captioning baseline. |
| Result/checkpoint shard | 4,096 rows | Amortizes object listing and validation while bounding replay work. |

Memory values are Ray scheduling reservations rather than eager allocations. Media fetch uses the per-node
`curator_io` resource advertised by managed Ray; the default 16 slots bound concurrent fetch tasks independently on
each node. Explicit concurrency limits remain available for diagnostics or deliberate resource sharing, while `auto`
lets CPU and GPU pools follow live resources.

### Durable output

After inference, the pipeline keeps only physical identity, caption values, metadata, and token counts. Terminal rows
are strictly repartitioned into compact groups before `write_parquet`, aligning result files with Ray's checkpoint
units. Each row contains:

```text
fragment_id: int64
row_offset: int64
clip_id: string
<caption field>
<caption metadata field>
```

The complete Phase A plan finishes before publication begins, so the model remains loaded for the inference workload
and no Lance transaction can be affected by a still-installed checkpoint configuration.

## Phase B: Fragment-Atomic Publication

Phase B turns durable Parquet into canonical Lance columns without sending caption rows through the driver:

```text
Ray workers
  validate Parquet footers
    -> read staged rows
    -> hash-group by fragment_id
    -> validate one complete fragment group
    -> prepare uncommitted Lance column files
    -> return compact descriptor + fingerprints

Driver
  stream descriptors with prefetch disabled
    -> reopen latest Lance
    -> reconcile fragment state
    -> commit one fragment transaction
    -> verify all selected fragments
```

Each Ray group is sorted by `row_offset` and checked against the current fragment. Validation proves that:

- the fragment was pending in `V_attempt`;
- offsets are exactly `0..row_count - 1`, without gaps or duplicates;
- ordered `clip_id` values match the current fragment;
- every row has the selected field set and contract digest;
- every row is a valid terminal value; and
- both fields cast exactly to their registered Arrow types.

The worker calls `LanceFragment.update_columns(..., left_on="clip_id", right_on="clip_id")` with an explicitly typed
Arrow reader. It returns only a compact descriptor containing the source version and fragment fingerprint, updated
fragment metadata, modified leaf field IDs, and row count. Uncommitted column files created by task replay are not
visible to readers and are harmless.

Before each commit, the driver reopens the latest table and follows this reconciliation state machine:

| Latest target state | Action |
| --- | --- |
| Terminal and value-for-value equivalent to staging | Treat the fragment as already committed. |
| Pending with the same physical fingerprint used by the worker | Commit the prepared descriptor. |
| Pending with only disjoint schema or enrichment changes | Discard the descriptor and ask a Ray task to rebuild it from the same Parquet rows against latest. |
| Mixed, conflicting, rewritten, deleted, or differently terminal | Fail without changing the fragment. |

Restaging is conservative by design. Lance may reject two same-fragment updates even when their fields are disjoint,
and attaching old fragment metadata to a newer read version can discard the winning writer's file bindings. Rebuilding
only the caption column files preserves the other enrichment without repeating inference.

Each transaction updates exactly one fragment and every leaf under the two caption-owned fields. Transaction properties
record the source version, fragment ID, field name, and contract digest. Updates to other fragments and split appends may
rebase independently.

An ambiguous commit response is resolved by reopening latest and applying the same state machine. Exact staged data is
accepted as committed; a still-pending target is retried or restaged; conflicting state fails. This makes an uncertain
client response a reconciliation event rather than a reason to repeat inference.

## Concurrency and Consistency Model

The pinned `V_attempt` separates selection from concurrent appends. A fragment appended by `video-split` after that
snapshot is not part of the active attempt and remains pending for the next invocation. Earlier selected fragments can
still be published while splitting appends other fragments.

Disjoint enrichment writers can proceed concurrently when they preserve the fragment ID, row count, ordered
`clip_id` values, and all split- and caption-owned physical bindings. A same-fragment race causes restaging from Parquet
against latest. No GPU work is repeated.

The following changes are intentionally outside the current protocol:

- multiple active writers for one caption field set;
- row mutation, deletion, or reordering;
- fragment compaction or remapping while an attempt is staged; and
- another enrichment implementation that attaches stale descriptors without equivalent reconciliation.

These cases fail closed when detectable rather than risking a caption-to-media mismatch.

## Failure and Recovery

| Last durable event | Behavior on the next invocation |
| --- | --- |
| Before or during Phase A | Capture a new `V_attempt`; reuse compatible checkpoints and process unfinished or newly selected clips. |
| Result file written but checkpoint still pending | Ray removes the pending output and replays that compact shard. |
| Phase A complete | Reuse Parquet and enter Phase B without model setup when the completion marker covers the attempt. |
| Prepared update not known committed | Reopen latest and reconcile; restage from Parquet when the change is disjoint. |
| Some fragments committed | Skip canonical fragments and publish only fragments that remain pending. |
| `video-split` appended after `V_attempt` | Leave the new fragment pending until the next attempt. |
| Canonical verification succeeded but cleanup failed | Keep the Lance result; retry workspace cleanup later. |

Deterministic per-clip media failures are terminal data and are not retried by an ordinary rerun. Shared failures stop
the phase so broken credentials, storage, model setup, or inference infrastructure cannot be mistaken for a large set
of bad clips. Checkpoints, result Parquet, and already committed fragments remain available after such a failure.

## Configuration and Runtime Assumptions

The recipe uses strict, versioned YAML or JSON. A minimal configuration is:

```yaml
schema_version: 1
kind: video-caption

input:
  media_root: s3://example-bucket/curated/video-split

model:
  variant: qwen3_8_27b_fp8

output: {}

execution:
  storage_profile: default
```

`clips_lance_uri` defaults to `<media_root>/lance`, and `staging_root_uri` defaults to
`<media_root>/staging/video-caption`. The staging location is visible to every Ray worker and survives Ray head loss.

Model weights are pinned by variant and pre-staged under the standard `/config/models` mount on every inference node.
The runtime validates `config.json` and model weight files but does not download weights. This keeps model acquisition
outside the recovery protocol and ensures every actor resolves the same local path.

## Known Boundaries

The v1 design does not:

- split videos or create alternate caption windows;
- use audio or produce structured Cosmos-style captions;
- download models at pipeline runtime;
- replace an existing caption contract in place;
- coordinate multiple writers for the same field set;
- support row mutation, fragment compaction, or fragment remapping during recovery; or
- promise bitwise-identical stochastic output across retries or hardware.

## Implementation Map

| Module | Responsibility |
| --- | --- |
| `pipeline.py` | Attempt orchestration, phase boundaries, Ray setup, verification, and cleanup. |
| `config.py` | Strict config validation and derived storage locations. |
| `contracts.py` | Model resolution, field schemas, normalized contract digest, and terminal-value rules. |
| `lance_state.py` | Field registration, input validation, attempt capture, and physical-fragment identity. |
| `workspace.py` | Contract-scoped manifest, completion marker, filesystem resolution, and cleanup. |
| `inference.py` | Lance datasource adapter, Ray checkpoint configuration, media fetch, Ray Data LLM, and Parquet sink. |
| `publication.py` | Distributed staged-row validation, descriptor preparation, reconciliation, and Lance commits. |

Unit tests exercise contract and publication logic without launching Ray jobs. Environment-marked integration tests run
the real Ray checkpoint/recovery path and distributed publication boundary on provisioned infrastructure. The
publication suite covers corrupt or incomplete staging, partial progress, ambiguous responses, disjoint same-fragment
races, stale-descriptor restaging, split-owned rewrites, and appends after `V_attempt`.
