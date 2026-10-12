# Curator Next: Video Embedding

`video-embed` adds one visual Cosmos-Embed1 vector per completed `video-transcode` clip. The clip table may originate
from either fixed-stride or TransNetV2 `video-cut`. Caption fields are neither read nor required. Captioning and
embedding can run independently and publish disjoint columns concurrently.

```text
video-cut -> video-transcode --+--> video-caption
                              +--> video-embed

pending Lance clips -> checkpoint filter
  -> CPU download tasks -> GPU decode / sample / pinned processor / embedding actors
  -> strict repartition -> checkpointed Parquet
  -> group by original fragment -> prepare column files on workers
  -> ONE Lance Update containing every selected fragment
```

The implementation lives in `cosmos_curator.next.recipes.video_embed`. It uses native Ray Data and existing storage,
Pixi environment, and media-reading utilities. It imports no legacy pipeline or Xenna model wrapper. The robotics
embedding recipe's modality and publication contracts are not part of this recipe.

Checkpoint reuse, atomic enrichment, commit reconciliation, and cleanup follow the
[shared checkpoint and recovery contract](curator-next-checkpoint-recovery.md), also used by video-transcode. Embedding
supplies clip identity, its model and result contract, clip readiness, and terminal-result validation, which its
`bindings.py` declares to the shared contract; media handling and GPU inference remain recipe-owned.

## Model-card contract

The model IDs and immutable revisions come from `cosmos_curator/configs/all_models.json`. These are the reference
model cards:

| Variant | Default input | Vector width | Reference |
| --- | --- | --- | --- |
| `224p` | 8 RGB frames, 224 × 224 | 256 | [Pinned model card](https://huggingface.co/nvidia/Cosmos-Embed1-224p/blob/85f56274be7ce439175b4587fa6e3525556d4832/README.md) |
| `336p` (default) | 8 RGB frames, 336 × 336 | 768 | [Pinned model card](https://huggingface.co/nvidia/Cosmos-Embed1-336p/blob/5d8309dd5c9ec8f856b16d589693004c907c9a57/README.md) |
| `448p` | 8 RGB frames, 448 × 448 | 768 | [Pinned model card](https://huggingface.co/nvidia/Cosmos-Embed1-448p/blob/9f4ff4d8d3df0968276b2191806344000d6752a9/README.md) |

Temporal sampling uses the reference expression `np.linspace(0, frame_count - 1, 8, dtype=int)`. It includes the first
and last displayed frames, and naturally repeats frame indices for clips shorter than eight frames. PyNvVideoCodec
uses NVDEC to produce presentation-order RGB frames on the GPU; it replaces the reference example's Decord decoder.
PyAV reads container metadata without decoding. NVDEC drains delayed B frames and verifies the declared frame count,
or counts and reopens a clip whose container does not declare one. Only selected frames are retained. Encoded bytes
are written to an actor-local temporary file because PyNvVideoCodec 2.0's memory callback cannot seek through all MP4
layouts; the file is removed after decoding.

Each inference actor retains up to four NVDEC sessions, keyed by GPU/stream, codec, dimensions, bit depth, chroma
format, and color metadata. Fully drained sessions are reused across clips; failed or interrupted decodes are discarded.
Selected-frame copies finish before a session is reused. This avoids per-clip decoder teardown while preserving the
sampling and validation contract.

Each model's staged `AutoProcessor` receives uint8 BTCHW data with no preprocessing overrides. The pinned processor
owns square resizing, interpolation, antialiasing, and scaling. Each inference actor calls it separately for each clip
on CUDA tensors so different source sizes can feed the same inference batch. Decoded frames and prepared tensors stay
on that actor's GPU.

`AutoModel` runs in evaluation/inference mode with BF16 inputs and weights. Only `get_video_embeddings` is called.
Its already-normalized `visual_proj` is converted to float32 for storage, with no additional pooling or normalization.
There is no text input, audio input, caption dependency, extra temporal window, or per-frame output.

Model acquisition remains outside the recipe. Exact registry-pinned weights and processor code must be staged at
`/config/models/<model_id>` on every inference node. The runtime checks configuration and weight-file
presence, and uses `local_files_only=True`; it relies on model staging for revision integrity. GPU actors use the
`legacy-transformers` Pixi environment, including PyNvVideoCodec. The sampling, processor contract, model revision,
precision, output representation, and failure semantics all participate in result identity.

## Canonical columns and identity

The recipe owns two nullable fields per variant, registered together before processing. For `336p`:

```text
embedding_cosmos_embed1_336p_v1: fixed_size_list<float32, 768>
embedding_cosmos_embed1_336p_v1_metadata: struct<
  embedding_schema_version: int32 not null,
  contract_digest: string not null,
  model_id: string not null,
  model_revision: string not null,
  status: string not null,
  error_type: string,
  error_message: large_string
>
```

Arrow field metadata records owner `video-embed` and the variant's field set. Existing fields must match exactly.
A successful row has a finite vector of the model's dimension, terminal `success` metadata, and no error details.
A terminal `error` row has a null vector and nonempty error details. Both fields are null only while pending.
Every physical fragment in the pinned table must be wholly pending or wholly terminal under the requested contract;
partial, mixed, and incompatible fragments anywhere in the snapshot are rejected before selection.

The logical result identity is `(clip_id, embedding_contract_digest)`. Upstream clips follow the
[shared input immutability requirements](curator-next-checkpoint-recovery.md#assumptions-and-guarantees).
Existing vectors are not overwritten when model behavior changes:
such a change requires a new versioned field contract. Ordinary reruns skip all terminal rows, including media errors.
There is no error-reset or in-place replacement operation. The initial version 1 contract records NVDEC RGB conversion
and CUDA processor execution. Earlier development results and checkpoints with a different contract digest cannot
be reused, even if their fields were also named version 1.

## Checkpoints and publication

An attempt validates the canonical clip schema, registers its fields, and checks clip values in the pinned Lance
snapshot. New work selects all pending embedding fragments; the recipe exposes no public selection-size limit.
Saved selections follow the [shared recovery rules](curator-next-checkpoint-recovery.md#recovery-identity-and-workspace).
Ray filters checkpointed string `clip_id` values before media reads or GPU work.
Fragment counts follow the [shared run summary definitions](curator-next-checkpoint-recovery.md#run-summaries).

Input readiness comes from data: the table must have the canonical clip schema, globally unique non-null `clip_id`
values, non-null `clip_uri` values, and positive non-null `clip_size_bytes`. No transcode publication receipt is required;
valid clip tables remain usable after upstream history is pruned. Empty tables are valid. These checks cover the entire
pinned snapshot; selecting individually ready fragments from a growing table remains follow-up work. Payload reads
and decoding still validate the media during processing.

Parquet stores `fragment_id`, `row_offset`, `clip_id`, the vector, and metadata. Encoded media and prepared tensors
never enter durable result files. The shared staged schema stores vectors as variable-length lists because Arrow 25
cannot read back null fixed-size lists from Parquet. Publication validates dimensions and casts to the fixed-size
canonical fields.
`execution.parquet_rows_per_file` sets the strict repartition and checkpoint shard boundary, defaulting to 1,000 rows.
Shared non-null `int64` fields define the fragment and row addresses; the logical `clip_id` retains its canonical field
definition. These staged schema definitions isolate incompatible earlier development workspaces.

The workspace is `<output.staging_root_uri>/<contract-namespace>/<selection-digest>/`. Embedding contributes its
pinned model and result contract to the
[shared compatibility manifest](curator-next-checkpoint-recovery.md#recovery-identity-and-workspace), so each variant
uses its own contract namespace.
A valid processing-completion marker allows publication retries to skip model setup.

Workers validate terminal metadata and use vectorized checks for vector dimensions and finite values before preparing
the two owned columns. Publication follows the
[shared publication and concurrency rules](curator-next-checkpoint-recovery.md#canonical-publication-and-concurrent-writers).
The [shared failure boundaries](curator-next-checkpoint-recovery.md#failure-and-recovery-boundaries) cover interrupted
processing, ambiguous commit responses, and cleanup; publication retries reuse staged vectors without inference.

## Failure and concurrency boundaries

Missing/empty/size-mismatched objects and identifiable corrupt video become terminal error rows. Credential faults,
storage outages after transport retries, model/processor setup failures, unexpected decoder faults, invalid inference
output, and GPU failures stop the attempt. There is no catch-all conversion of inference failures to clip errors.

One logical writer owns each variant's field set. Disjoint caption or other variant writers can publish concurrently
under the shared contract; each attempt covers the clip snapshot selected at its start.

CPU tasks use `map` to download clips, reusing storage connections within each worker. GPU actors use `map_batches`
to decode, prepare, and embed those clips. Only metadata and compressed clips enter the object store between stages;
decoded frames never do. Download attempts remain controlled by `media_attempts`. The former `media_concurrency`,
`decode_concurrency`, `decode_cpus`, and `decode_memory_gib` settings are removed. Each download task reserves one
`curator_io` slot (default: 16 per node), including while Ray backpressures its output. A production run without this
bound admitted 91 download tasks that retained about 85 GiB of private heap alongside 24 GiB of object-store data and
triggered Ray OOM kills. Ray's output backpressure alone did not bound worker heap usage. Downloads remain plain tasks;
there is no download actor pool or actor-autoscaling workaround. Ray Core enforces each node's IO slots, including
on late workers; Ray Data still regulates queued output and may submit tasks that wait for those slots.

Lance scanner batches are sized by `lance_read_batch_size`; the shared reader balances fragment row ranges across Ray's
requested read tasks, without a separate input repartition. Download tasks reserve 0.25 CPUs; each inference actor
reserves one CPU and one GPU. The plan retains Ray's default target block size. Model pools use Ray's default scale-up
threshold and task prefetching, executing one call per actor at a time. This inference plan sets the utilization downscaling threshold
to `0.0` for the inference pool: one idle GPU alongside a busy GPU no longer triggers model retirement. Fully idle pools
can still shrink; resource-pressure downscaling and end-of-input cleanup remain enabled. This addresses repeated model loads
at Ray's default `0.5` threshold. Pools start small and may grow with live resources, including late workers, unless
explicitly capped. Reserve enough CPUs for the model pool, resume checkpoint filter, and transient
reads/repartition/writes; a small local smoke cluster should provide at least six.
Whole encoded clips must fit in worker memory, and decoder surfaces plus eight source-resolution frames and the
inference batch must fit on a GPU. The default inference batch is 16 clips, configurable through
`execution.inference_batch_size`.

On twelve sampled production clips, isolated NVDEC decoding took 0.89 s versus 2.23 s for the prior CPU decoder.
The pinned 336p processor's median time fell from 58.2 ms on CPU to 1.04 ms on GPU including transfer, or 0.18 ms
for GPU-resident inputs. These component measurements motivated moving preparation into the inference actor;
they are not an end-to-end throughput claim. CUDA processing and NVDEC RGB conversion changed some pixel values,
so the contract records the decode backend and processor device.

A bounded local run at inference batch size 4 on two RTX PRO 6000 Blackwell GPUs processed 512 sampled production clips
(2.10 GiB of compressed media) in 59.6 s, including model setup and Lance publication. The same sample with an 8 MiB
target block size took 59.4 s, so the default 128 MiB target remains unchanged. Each fresh Ray cluster initialized
exactly two models with no reloads; all 512 results succeeded. This small run does not establish sustained production
throughput or eliminate the possibility of downscaling during longer input stalls.

With inference batch size 4 and downloads bounded to 16 IO slots, a 2,048-clip sample (8.33 GiB) completed in 200.3 s on the
same two GPUs. All clips succeeded, with two model initializations and no Ray OOM kills. Instrumented download calls
never exceeded 16 concurrently. Five-second memory samples reached 7.6 GiB of private download-worker heap and
19.8 GiB for the whole container. These measurements validate the bounded sample; they are not hard memory limits.

## Configuration and validation

```yaml
schema_version: 1
kind: video-embed
input:
  media_root: s3://example-bucket/curated/videos
model:
  variant: 336p
```

`input.clips_lance_uri` defaults to `<media_root>/lance`; `output.staging_root_uri` defaults to
`<media_root>/staging/video-embed`. `cosmos-curator pipeline template video-embed` shows execution defaults, and the
generic validate/render/schema/run commands use the same strict resolver. Run on managed Ray through the usual entry:

```bash
cosmos-curator slurm ray submit CLUSTER_CONFIG -- pixi run --as-is run-pipeline RECIPE_CONFIG
```

CPU unit coverage includes sample indices for short clips, media failure classification,
Parquet null-vector round trips, output validation, and caption coexistence. The shared conformance suite runs against
embedding's clip fixtures, covering coverage rejection, concurrent writers, ambiguous publication, and recovery. A CPU integration test uses model stand-ins to run the native Ray plan across fresh local clusters
with one `curator_io` slot and verifies that previously checkpointed clips are neither downloaded nor inferred again.
Each observed download also asserts that it holds its IO resource. GPU tests verify short-clip repeats and reordered
frames against a full CPU decode with an RGB rounding tolerance, compare all three variants' preprocessing and visual
output with direct pinned-processor/model calls, and exercise the complete recipe.

These checks do not constitute managed Slurm worker-churn or head-loss qualification. Cluster-scale throughput and
memory measurements, actual worker removal/requeue, and concurrent multi-job qualification remain follow-up work.
For fragment memory and checkpoint-index limits, see the
[shared publication sizing guidance](curator-next-checkpoint-recovery.md#publication-sizing).
