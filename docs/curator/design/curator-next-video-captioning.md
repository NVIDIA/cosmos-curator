# Curator Next: Video Captioning

`video-caption` enriches the canonical clip table produced by Curator Next
[`video-transcode`](curator-next-video-transcode.md). It reads complete clip MP4s, generates one caption per clip with
Qwen3.8 through Ray Data LLM and vLLM, and fills a versioned caption field set on the existing Lance rows.

```text
video-cut -> video-transcode --+--> video-caption
                              +--> video-embed

pending Lance clips -> checkpoint filter -> fixed-size blocks
  -> CPU media-fetch tasks -> multimodal preparation, chat template, tokenization
  -> exact-batch vLLM actors -> terminal caption rows
  -> strict repartition -> checkpointed Arrow IPC
  -> group by original fragment -> prepare column files on workers
  -> ONE Lance Update containing every selected fragment
```

The implementation lives in `cosmos_curator.next.recipes.video_caption`. Checkpoint reuse, atomic enrichment, commit
reconciliation, and cleanup follow the [shared checkpoint and recovery contract](curator-next-checkpoint-recovery.md),
also used by video-transcode and video-embed. Captioning supplies clip identity, its model and result contract, clip
readiness, and terminal-result validation, which its `bindings.py` declares to the shared contract; media fetch and
vLLM inference remain recipe-owned.

## Context and Responsibilities

`video-cut` owns span rows and fragment creation. `video-transcode` owns clip media and its metadata. Captioning owns
only its selected caption and metadata fields. It does not discover sources, split videos, rewrite media, append rows,
or change row order.

Empty/error source outcomes from video-cut have null span and clip IDs. Captioning skips these rows during reading,
completion checks, and publication coverage, preserving their physical offsets and leaving caption fields null.
Fragments containing only source outcomes need no caption work. The clip invariants below apply to real spans.

The design relies on these properties of the upstream table:

- `clip_id` is globally unique and immutable.
- `clip_uri` and `clip_size_bytes` are non-null and immutable for an existing row; byte sizes are positive.
- Each clip is a standalone H.264 MP4 that was probed after transcoding.
- Transcoding publishes all clip metadata in one commit while preserving the cut fragment layout.
- Nullable enrichment fields may be added without changing the upstream-owned fields.

Input readiness comes from data. Shared validation requires the canonical clip schema and globally unique `clip_id`
values; captioning adds the readiness validator it shares with embedding, which requires a non-null `clip_id`, a
non-null `clip_uri`, and a positive `clip_size_bytes` on every real span. No transcode publication receipt is required,
so history pruning and later empty/error source outcomes do not block ready clips. These checks cover the entire
pinned snapshot: newly appended spans without clip metadata must finish transcoding before captioning starts.

One logical writer owns a given caption field set. Writers for disjoint field sets, including embedding and other
caption variants, may publish concurrently under the shared contract. Multiple writers for the same field set are
outside the consistency model.

## Caption Result Contract

### Field ownership and registration

The recipe declares the caption and metadata fields with the shared `owned_fields`, and shared registration adds them
together in one Lance schema transaction. Both top-level fields are nullable so existing rows and any future producer
appends begin pending. Their Arrow metadata records the owner and field set:

```text
cosmos_curator.owner = video-caption
cosmos_curator.field_set = <caption-field-name>
```

An existing field set must match its expected names, Arrow types, nullability, and ownership metadata exactly. A
result-defining change receives a new versioned field name instead of changing the meaning of an existing `_v1` field.

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
token counts. The metadata struct is null only while the row is pending. Every physical fragment in the pinned table
must be wholly pending or wholly terminal under the requested contract; mixed fragments and terminal values with a
different digest are rejected before selection.

### Contract identity

The logical result identity is `(clip_id, caption_contract_digest)`. The digest is the SHA-256 hash of canonical JSON
containing every result-defining value:

- caption schema version and exact Arrow field schemas;
- model ID, immutable revision, and precision;
- prompt, message order, and chat-template arguments;
- sampling and output-token settings;
- video preprocessing settings;
- a set `execution.max_model_len`, which can end generation early; and
- terminal-state and failure semantics.

The normalized contract is the spec's `result_contract`, so the shared publication's `contract_digest` provenance is the
same digest that terminal metadata records. Data-URL encoding, block sizes, concurrency, retry counts, and worker
placement are execution details and are excluded from the digest. Sampling is intentionally stochastic, so the first
compatible result durably checkpointed is authoritative; retries are not expected to reproduce identical text
bit-for-bit.

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

## Checkpoints and Publication

New work selects all pending caption fragments unless `execution.max_fragments` bounds it. Every selected fragment
becomes visible in one Lance `Update`, so nothing is visible until the selection publishes; set `max_fragments` to
publish smaller selections sooner, and rerun while `remaining_fragments` is nonzero. Saved selections follow the
[shared recovery rules](curator-next-checkpoint-recovery.md#recovery-workspaces-and-receipts), and fragment counts
follow the [shared run summary definitions](curator-next-checkpoint-recovery.md#run-summaries).

Ray filters checkpointed `clip_id` values at the read, before media fetch or inference. Staged results hold
`fragment_id`, `row_offset`, `clip_id`, the caption, and its metadata; media never enters durable result files. The
shared writer validates every terminal row before Ray checkpoints its ID, and publication repeats the check.
`execution.rows_per_checkpoint` sets the strict repartition and checkpoint shard boundary, defaulting to 1,000 rows.

The workspace is `<output.staging_root_uri>/<contract-namespace>/`. Captioning contributes its pinned model, field
schemas, and result contract to the
[shared compatibility manifest](curator-next-checkpoint-recovery.md#enrichment-recovery-identity), so each variant and
each set `max_model_len` uses its own contract namespace. A valid processing-completion marker lets a publication retry
skip model validation and inference.

Publication follows the
[shared publication and concurrency rules](curator-next-checkpoint-recovery.md#canonical-publication-and-concurrent-writers),
and the [shared failure boundaries](curator-next-checkpoint-recovery.md#failure-and-recovery-boundaries) cover
interrupted processing, ambiguous commit responses, and cleanup; publication retries reuse staged captions without
inference.

Earlier versions published one fragment per commit from a separate Parquet workspace under
`<staging-root>/<caption-field>/<contract-digest>/`. The caption fields and contract digests are unchanged, so tables
captioned that way are already complete: their values validate, and only fragments still pending are selected. An
unfinished workspace from that implementation is not resumed; delete it, and rerun to caption its fragments again.

## Media and Inference

The plan reads `clip_id`, `clip_uri`, `clip_size_bytes`, and the row addresses through the shared Lance reader, which
balances fragment row ranges across Ray's read tasks. Ray merges each read task's output into one block and sizes read
tasks from the CPUs present when the plan starts. A strict repartition after the read therefore gives every media-fetch
task `execution.lance_read_batch_size` clips, independent of the fragment count and read parallelism, including the
clips a resume leaves after checkpoint filtering. Fetch tasks download the exact
`clip_uri`, validate its byte length against `clip_size_bytes`, and encode the MP4 as a data URL that can cross the
Ray object store. The top-level data URL is removed once it has been moved into the OpenAI message, and multimodal
preparation replaces that message with decoded inputs. Large media values therefore do not travel through
tokenization, inference output, or staged results.

The plan uses Ray Data LLM's public processor surface and built-in multimodal preparation. Immediately before vLLM, a
strict repartition creates exact `inference_batch_size` blocks. This compensates for the ordinary `map_batches`
rebundler, which may pass a 33-row multimodal block as 32 rows followed by a recurring one-row engine call. Only the
final global block may be smaller.

The vLLM actor pool has a work-sized upper bound, the selection's row count from `ProcessingScope.selected_rows`,
rather than a GPU count captured at startup. Each replica requests `tensor_parallel_size * pipeline_parallel_size`
GPUs; the default TP=1, PP=1 topology places one replica per live GPU. Currently available GPUs seed the initial actor
count so model loading begins in parallel, while Ray's eager actor autoscaling lets GPUs on later-joining managed Slurm
nodes join without changing the plan.

Multimodal-caption's two-phase runner reuses this media-fetch and inference plan through `inference.caption_rows`,
supplying its own media columns, prompt context, and terminal rows.

### Backpressure and reference tuning

The reference settings are execution choices, not result-contract inputs:

| Boundary | Default | Rationale |
| --- | --- | --- |
| Media-fetch block | 256 rows | `lance_read_batch_size`, the scanner batch and fetch-task size; fixed blocks spread fetches across the cluster whatever the read parallelism. |
| Media fetch | 4 clips, 0.25 CPU, one `curator_io` slot | Bounds simultaneous downloads and active encoded MP4s per node. |
| Multimodal preparation | 24 GiB scheduling reservation per actor | Covers measured batch-32 heap peaks without host overcommit. |
| Chat-template stage | 16 GiB scheduling reservation per actor | Bounds CPU actor growth on large nodes. |
| vLLM | Batch 32, 8 concurrent batch calls | Matches the benchmarked legacy Ray Data captioning baseline. |
| vLLM context length | Unset, the model's own maximum | `execution.max_model_len` caps prompt plus generated tokens per request. vLLM refuses to start unless one request of that length fits in the KV cache, so lowering it lets the model load on GPUs with less memory beyond the weights; a clip over the limit fails the run. A set limit is part of the caption contract and digest; null leaves both unchanged. |
| Checkpoint group | 1,000 rows | The shared default; bounds the in-flight results that replay after an interruption. |

Memory values are Ray scheduling reservations rather than eager allocations. Media fetch uses the per-node
`curator_io` resource advertised by managed Ray; the default 16 slots bound concurrent fetch tasks independently on
each node. Fetches are plain tasks, as in [video-embed](curator-next-video-embed.md#failure-and-concurrency-boundaries),
not a download actor pool. Ray Data may submit fetch tasks that then wait for slots, counting each against its CPU
and object-store budgets. That can slow later stages' scale-up but does not affect GPU or heap-memory reservations.
Explicit concurrency limits remain available for diagnostics or deliberate resource sharing, while `auto` lets CPU and
GPU pools follow live resources.

## Failure and Concurrency Boundaries

Deterministic per-clip download failures, such as a missing or archived object or a byte-length mismatch, become
terminal `error` rows and are not retried by an ordinary rerun. Credentials, access denials, storage availability,
model loading, multimodal preparation, and inference failures stop processing: the upstream transcoder already probed
each clip, so a later decode or engine failure indicates a runtime or contract problem rather than bad media. Failed
processing leaves checkpointed results and the workspace for the next invocation.

Each attempt covers the clip snapshot selected at its start; appended fragments wait for a later selection. Row
mutation, deletion, compaction, and multiple writers for one caption field set are unsupported and fail closed when
detectable.

## Configuration and Validation

```yaml
schema_version: 1
kind: video-caption

input:
  media_root: s3://example-bucket/curated/video-transcode

model:
  variant: qwen3_8_27b_fp8
```

`input.clips_lance_uri` defaults to `<media_root>/lance`, and `output.staging_root_uri` defaults to
`<media_root>/staging/video-caption`. The staging location is visible to every Ray worker and survives Ray head loss.
`cosmos-curator pipeline template video-caption` shows execution defaults. Run on managed Ray through the usual entry:

```bash
cosmos-curator slurm ray submit CLUSTER_CONFIG -- pixi run --as-is run-pipeline RECIPE_CONFIG
```

Model weights are pinned by variant and pre-staged under the standard `/config/models` mount on every inference node.
The runtime validates `config.json` and model weight files but does not download weights.

CPU unit tests cover the contract digest, field set, terminal states, media failure classification, processor
construction, plan wiring, and clip readiness. The shared enrichment conformance suite runs against caption's clip
fixtures, covering coverage rejection, concurrent writers, ambiguous publication, and recovery. A CPU integration test
runs the shared fresh-cluster scenario with a vLLM stand-in: the native Ray plan resumes across fresh local clusters
with one `curator_io` slot, previously checkpointed clips are neither fetched nor inferred again, a clip whose media
disappears still records a terminal error, and the recipe's `run_config` publishes the result. These checks do not
constitute managed Slurm worker-churn or head-loss qualification, and GPU throughput of the shared-enrichment plan has
not been re-measured.

## Known Boundaries

The v1 design does not:

- split videos or create alternate caption windows;
- use audio or produce structured Cosmos-style captions;
- download models at pipeline runtime;
- replace an existing caption contract in place;
- coordinate multiple writers for the same field set;
- support row mutation, fragment compaction, or fragment remapping during recovery;
- select individually ready fragments from a growing clip table; or
- promise bitwise-identical stochastic output across retries or hardware.

## Implementation Map

| Module | Responsibility |
| --- | --- |
| `bindings.py` | The shared enrichment spec: clip input, owned caption fields, result contract, and validators. |
| `config.py` | Strict config validation, derived storage locations, and execution tuning. |
| `contracts.py` | Model resolution, field schemas, normalized contract digest, and terminal-value rules. |
| `inference.py` | The shared caption plan: media fetch, Ray Data LLM processor, and terminal rows. |
| `pipeline.py` | Ray setup, the checkpointed processing plan, and the `run_enrichment` entry point. |
