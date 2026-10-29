# Curator Next: Video Cut

[`video_cut`](../../../cosmos_curator/next/recipes/video_cut/) plans fixed-stride or TransNetV2 scene spans without creating clip media.
[`video-transcode`](curator-next-video-transcode.md) encodes those spans. Together they replace the removed
`video-split` recipe.

## Decision and rationale

The design separates source processing from tabular publication:

```text
select new sources -> probe + plan each source -> checkpointed Arrow IPC with spans[]
                   -> Arrow expansion -> packed Lance fragments -> atomic append
                   -> workspace cleanup
```

- **Separate cutting from transcoding.** Boundaries and source references can be reused across encoding choices.
- **Recover by source.** Keeping a complete source plan together avoids reconciling separate source and span progress.
- **Pack spans independently.** Checkpoint size bounds replay; Lance fragment size controls storage layout.
- **Append one finite selection.** One atomic Lance commit publishes the new spans and empty/error outcomes together,
  allowing staging to be deleted afterward while preserving earlier fragments and every source's completion.

The tradeoff is that interrupted publication rebuilds fragments from staged plans. This repeats tabular work while reusing
checkpointed source plans, avoiding a separate publication ledger. Video-cut is the reference recipe for
[shared append publication](curator-next-checkpoint-recovery.md#append-publication), which owns selection, checkpoint
scopes, publication of all source outcomes, and recovery proof. Video-cut owns discovery, source probing and planning,
its terminal-error policy, and span expansion.

## Input, identity, and output contract

Inputs are explicit S3 MP4 URIs or a recursive S3 root. `discovery.discover_items(config)` normalizes, sorts, and
deduplicates selected URIs, preserves literal S3 keys, and returns identified work items in `INPUT_SCHEMA`. Production
and the conformance suite use this same entry point. The invocation opens the target table and reconciles saved work
before discovery. An unfinished selection resumes without listing. For new work, listing failures propagate before
selection or Ray starts; an empty target table may already have been created. A successful empty listing selects no
work and starts no Ray.

Each URI must identify immutable content. Each output is bound to one cutting contract, but its source set can grow.
Root inputs are listed when selecting new work. Sources already published are skipped before probing, including recorded
empty plans and terminal errors. Removing a source from a later input selection does not remove its existing spans.
Explicit URIs and root discovery are interchangeable when they select identical sources.

| Identity | Meaning |
| --- | --- |
| `source_id` | Hash of the normalized immutable source URI; the append work item. |
| `span_id` | Hash of source ID, first video stream (`stream_index: 0`), start, and end. |
| Cutting contract | Method, result settings, stream selection, behavior version, and (for TransNetV2) pinned model identity; the append result contract, which binds the span table. |

Span identity is independent of cutting settings. Intervals are half-open `[start_ns, end_ns)` on a zero-relative video
timeline. The source endpoint is FFprobe's stream duration, falling back to container duration.
[`next.media.spans`](../../../cosmos_curator/next/media/spans.py) defines rounding and fixed-stride tail rules.

Lance stores flat span rows and durable source outcomes. Every row carries source references and media metadata.
Span rows have a `span_id` and interval boundaries. A source without spans has one outcome row with null `span_id`,
`start_ns`, and `end_ns`; a terminal failure's outcome row also has `error_stage` and `error_message`.
[`contracts.py`](../../../cosmos_curator/next/recipes/video_cut/contracts.py) defines the schemas. Current source-ID
presence records completion even after history is pruned.

Transcode, embedding, and captioning skip outcome rows. Query `span_id IS NOT NULL` for spans or
`error_stage IS NOT NULL` for diagnostics. No append receipts or JSONL error reports are required.

One logical cutter owns source selection for each output. Transcode and other producers of nullable enrichment columns
may run concurrently. Cut only appends fragments; existing rows, fragment boundaries, and span columns are immutable.
Compaction, deletion, and overwrites are outside this contract. Execution tuning is excluded from identity and recovery
compatibility.

## Execution and recovery

Selection, resumption, `execution.max_items`, history pruning, and cleanup follow
[shared append publication](curator-next-checkpoint-recovery.md#append-publication). By default, each invocation
selects and publishes every pending source; set `execution.max_items` to publish fewer per invocation, and rerun while
`remaining_items` is nonzero or null. A resumed selection skips source discovery and reports null input and remaining
counts; the following invocation discovers any new work. Each table and cutting contract reuses one fixed workspace.
Cleanup must succeed before a new selection starts; a cleanup failure leaves already-published results valid.
This section describes what video-cut adds.

Both cutting methods read the selected sources through the shared processing scope, one source per read block, so
native checkpoint filtering skips completed sources before media access, each probe or download task handles one
source, and a task retry repeats only that source. Both emit one canonical `SOURCE_SCHEMA` row per source. Each method
owns its execution graph and shares only the source failure policy in
[`sources.py`](../../../cosmos_curator/next/recipes/video_cut/sources.py):

```text
fixed-stride: one task per source: remote probe -> local download/probe if needed -> plan
transnetv2:   download -> probe + decode (CPU, or NVDEC on the GPU actor) -> inference + plan
```

Fixed-stride needs only metadata and normally finishes through remote probing. An inconclusive remote probe streams
the source to a temporary file and probes it there within the same task. Source bytes never enter Ray, and the task's
CPU reservation bounds concurrency, including rare fallback downloads. It needs no `curator_io` resources or actor
pool. If many sources require fallback, increase `execution.fixed_stride.probe_cpus` to reduce concurrent disk and
network use.

TransNetV2 needs every source's frames, so its download actors pass encoded bytes through Ray to decoding. Each actor
reserves one `curator_io` slot and defaults to 0.25 CPU, following the
[large-download guidance](ray-data.md#large-source-downloads-and-curator_io). The pool autoscales and accepts late
workers without a startup-derived maximum.

Native checkpoints use `source_id` and advance in groups controlled by `rows_per_checkpoint`. Both methods default
to 1,000 source plans per file. Lower this setting to reduce replay at the cost of more checkpoint writes; setting it
to one gives single-source durability. An unfinished group and other in-flight sources can replay. Completed plans
contain no frames or encoded bytes.

Publication expands the source plans with Arrow and packs spans and source outcomes across source boundaries into Lance fragments
controlled by `rows_per_fragment`. Previous fragments are never repacked, including an earlier run's partially filled
final fragment. Every committed fragment is ready for transcode; there is no whole-table completion gate. An empty
table can coexist with an active cut run. Append preserves enrichment columns, and newly appended narrow span rows read
those nullable columns as null.

This implementation assumes freshly created datasets and recovery workspaces. Tables from earlier video-cut formats
lack the shared contract binding or the current span schema and are rejected at open with a request to use a new output
location; they have no migration path.

The fixed-stride method has completed a managed Ray run on Slurm. Failure recovery is tested locally with synthetic
probe metadata, through the shared append conformance suite and the fresh-cluster scenario, which runs for both methods.

### TransNetV2 execution

`split.method: transnetv2` defaults to PyAV preprocessing (`split.decoder: pyav`), with separate CPU decoding
and GPU inference stages:

```text
checkpoint-filtered source blocks -> download actors (curator_io)
  -> encoded bytes through Ray -> CPU tasks: probe, PyAV decode, resize
  -> tiny RGB frames, timestamps, and media fields through Ray -> GPU actors: TransNetV2 inference
  -> one complete source plan -> checkpointed Arrow IPC -> shared Lance append
```

PyAV decode tasks default to 3 CPUs (`execution.transnetv2.cpus`) and 2 GiB of host memory (`memory_gib`). Inference
actors use 1 CPU (`inference_cpus`), 0.25 GPU (`gpus`), and 2 GiB of host memory (`inference_memory_gib`). The
different resource requests keep Ray from fusing decoding back into inference, so CPU decode concurrency can grow
independently of the model pool.

Set `split.decoder: nvdec` to opt into GPU preprocessing. Encoded bytes flow directly from download actors to GPU
actors, where NVDEC decoding and inference share the same actor and GPU reservation. The actor's host memory
reservation is the sum of `memory_gib` and `inference_memory_gib` (4 GiB by default). PyNvVideoCodec decodes in
presentation order and CV-CUDA performs antialiased linear resizing to 48x27 RGB. Full-resolution frames stay on the
GPU; only the tiny resized frames return to host memory for common windowing. NVDEC requires supported NVIDIA
decoding hardware. Unclassified decoder/CUDA failures propagate.

Both backends use actor pools that can grow with available resources, including late workers; neither has a
startup-derived maximum. The actor autoscaling threshold is 1.0 for the media plan.

NVDEC uses native demuxer timestamps, including variable cadence, and drains delayed B-frames at end of input.
Its color conversion and resizing differ from PyAV, so decoder selection belongs to the cutting contract.
Switching decoders requires a separate `output.media_root`; PyAV checkpoints and published spans are not reused for NVDEC.
To resume existing NVDEC output, keep `split.decoder: nvdec` explicit. PyAV is the default based on the managed Slurm
comparison below; NVDEC remains opt-in while throughput and output quality are evaluated further.

The shared download-autoscaling utility cancels surplus pending IO actors that would otherwise prevent Ray 2.58 from
scaling down an idle download pool, preserving normal cooldown and scale-up.

Each inference actor loads the pinned FP32 PyTorch TransNetV2 checkpoint once. The driver checks that the checkpoint
file exists when constructing the processing plan, and every new actor checks before loading it. A valid processing
completion marker skips this preflight and all inference setup, so publication can finish without mounted weights.
Staging must supply the revision in
`configs/all_models.json` and keep the weights immutable across runs; video-cut does not verify their checksum.
The checkpoint comes from the registry's runtime model directory:
`/config/models/Sn4kehead/TransNetV2/transnetv2-pytorch-weights.pth`. Local Docker and managed Slurm launches mount the
workspace at `/config`. To use another host model directory, mount it at `/config/models` in the driver and all worker
containers, including replacement nodes. For local launch, use `--extra-volumes /data/models:/config/models:ro`.
The host directory must contain `Sn4kehead/TransNetV2/transnetv2-pytorch-weights.pth`. No model download runs as part of
cutting. The registered model ID, revision, and precision are part of the cutting contract; host mount locations and
resource/batch settings are not.

With PyAV, each CPU decode task writes a temporary source file, probes it, and decodes the first video stream at its native cadence.
Resizing reuses one single-threaded scaler per source. Decoded 48x27 RGB uint8 frames travel in Arrow binary buffers,
alongside integer timestamps and probed media fields; encoded source bytes stop at the decode stage. These transient
buffers are reconstructed as NumPy views by inference actors and are never part of the durable source-result schema.
Presentation timestamps are normalized to the first decoded frame,
so variable-frame-rate input does not rely on average-FPS arithmetic. Missing/non-increasing timestamps or timestamps
outside the probed duration produce terminal media errors. Sources must fit in worker memory and temporary storage;
the memory reservation is not a hard cap. Encoded bytes and decoded frames occupy Ray object-store memory at their
respective stage boundaries, with Ray's normal backpressure bounding queued work. Source plans and checkpoints retain
the same cutting contract; changing these execution settings does not invalidate existing progress.

Inference uses 100-frame windows with a 50-frame stride, repeating edge frames for padding and retaining only each
window's central 50 predictions. `window_batch_size` defaults to one and controls the number of windows per model call.
Windowing and inference live in `cosmos_curator.models.transnetv2` and are shared with the legacy splitters, which now
also repeat the last frame to pad their final windows. This corrects their missing end padding and can change legacy
predictions near the end of a source.
Predictions strictly above `split.threshold` identify transition frames. Consecutive non-transition frames form scenes;
transition runs are omitted, a source without transitions forms one scene, and an all-transition source has no spans.
Long scenes are tiled into `max_duration_s` spans, and scenes/tails shorter than `min_duration_s` are dropped. The final
scene includes the last frame and ends at the probed source duration. There is no extra boundary crop or frame-count
minimum. These semantics are versioned independently of the legacy splitter.

Inference failures never become successful empty plans. If NVDEC selects a different video stream from the required
first video stream, the source receives a terminal `source-decode` error. Rerun those sources with
`split.decoder: pyav` and a fresh output location.

A CPU Ray integration test exercises the separate decode and inference boundary with a deterministic CPU model and
terminal missing/invalid-source outcomes; the shared fresh-cluster scenario covers source checkpoints across an
interrupted run and Lance publication. The combined predecessor also completed a local GPU smoke run using the pinned
model checkpoint. Both backends completed the managed Slurm runs below; qualification for changes in worker membership
and worker loss remains outstanding.

The NVDEC prototype was compared locally with three 720p H.264 videos totaling 22,254 frames. Native timestamps and
thresholded transition decisions matched PyAV for these samples, while antialiased GPU preprocessing was 2.4–3.7 times
faster than three-thread CPU preprocessing. Pixels were not identical; these measurements do not establish output
equivalence or end-to-end distributed throughput. GPU tests cover variable cadence and B-frame draining. A local Ray/GPU
smoke run with the pinned checkpoint processed two sources and a terminal missing-source outcome, published 11 spans,
and reused all three source checkpoints.

#### Managed Slurm comparison (2026-09-28)

Revision `7861ebb3` processed the same 22,067 sources from `s3://nemo-curation-mvp/benchmark-input-data/` on
`oci-aga-slurm-2`, with 10 worker nodes and four GB200 GPUs per worker (40 GPUs total). Both configurations used
`execution.transnetv2.gpus: 1.0` and `window_batch_size: 2`; these measurements do not cover the default GPU sharing
(`gpus: 0.25`) or window batching (`window_batch_size: 1`).

| Decoder | Pipeline runtime | Slurm wall time | Published spans | Source errors |
| --- | ---: | ---: | ---: | ---: |
| PyAV | 14m27s | 16m24s | 546,179 | 16 |
| NVDEC | 26m58s | 28m50s | 546,023 | 16 |

Pipeline runtime runs from the logged pipeline command launch to the final result summary, including Lance publication.
Slurm wall time includes cluster startup and shutdown. Both runs exited successfully. The managed run IDs are
`cc-ray-be2d0573e5ad` for PyAV (head job `41119`) and `cc-ray-147e55c5cd6a` for NVDEC (head job `41130`).

Both runs recorded the same 16 sources with non-increasing decoded timestamps. Both produced spans for the same
22,029 sources, with no duplicate, nonpositive, or overlapping intervals. Of those sources, 16,476 have identical
intervals and 5,553 (25.2%) differ in boundaries or counts; 1,578 have different span counts. The net difference of
156 spans therefore does not establish equivalent output. These results support PyAV as the default while NVDEC
remains an explicit choice with a separate cutting contract. GPU sharing, larger inference batches, and keeping
resized frames on the GPU are follow-up performance experiments; visual checks of differing cuts remain outstanding.

## Media access and failure policy

Fixed-stride always starts by letting FFprobe read a presigned S3 URL, using HTTP ranges where possible. A source the
remote probe cannot settle is downloaded in full and probed locally in the same task, which decides whether it is
unusable. There is no probe-mode setting. The task reserves `execution.fixed_stride.probe_cpus` (0.25 by default)
through probing, any fallback download, and planning. The selection read reserves the same CPUs and sets no other
remote args, so Ray runs each source's read and probe as one task.
`execution.probe_timeout_s` is shared by both cutters. Signed URLs are never persisted, and avoiding a full-object
transfer is not guaranteed.

Both methods classify storage failures with the same helper in
[`sources.py`](../../../cosmos_curator/next/recipes/video_cut/sources.py). Input storage is stable, so a source that is
unusable as stored stays unusable.

- **Terminal errors:** missing or archived objects (`NoSuchKey`, `InvalidObjectState`, or a bare 404 whose bucket
  exists), unsupported media after a successful download, local FFprobe failures explicitly reporting
  `AVERROR_INVALIDDATA`, and for TransNetV2, recognized invalid decoded media. These are checkpointed and skipped on
  later runs; retrying them requires a fresh output. A nonzero probe exit alone does not establish a terminal media
  problem.
- **Retryable failures:** other storage failures, which describe the environment or a transient condition: missing
  buckets, credentials, permissions such as `AccessDenied`, endpoints, and throttling. Missing tools, local probe
  timeouts, process termination, unclassified probe or decoding failures, model setup, CUDA, and unexpected errors are
  retryable too. These propagate without terminal checkpoints, and the saved selection resumes once the cause is
  fixed. Inconclusive probes receive up to three Ray map retries; exhaustion fails the attempt without publishing
  completion.

Terminal errors are published in the same Lance append as successful and empty outcomes, with their `source_id`,
`source_uri`, `error_stage`, and `error_message`. The [shared summary](curator-next-checkpoint-recovery.md#run-summaries)
counts newly published rows (including outcomes) and terminal errors. Video-cut also reports the table's
`spans_lance_uri` and `spans_lance_version`. Summaries use the invocation's counters without scanning the accumulated
table for a cumulative span count.

## Running

Generate an editable config and run it in the runtime environment:

```bash
cosmos-curator pipeline template video-cut > video-cut.yaml
cosmos-curator pipeline validate video-cut.yaml
pixi run --as-is run-pipeline video-cut.yaml
```

A minimal config is:

```yaml
schema_version: 1
kind: video-cut
input:
  uris: [s3://example-bucket/raw/example.mp4]
output:
  media_root: s3://example-bucket/curated/video-cut
```

For recursive discovery, replace `input.uris` with `input.root_uri: s3://example-bucket/raw/`.
Output paths default to `<media_root>/lance` and `<media_root>/staging/video-cut`, with overrides through
`output.spans_lance_uri` and `output.staging_root_uri`.

Defaults are ten-second spans at ten-second strides, retaining tails of at least two seconds. The generated template
and schema describe the remaining settings. Config validation and rendering perform no source discovery.

Outputs may use S3 or local/shared paths accessible to every Ray process. Workers require FFprobe.

For TransNetV2, replace `split` and stage the checkpoint under `/config/models`:

```yaml
split:
  method: transnetv2
  decoder: pyav  # Default; opt into nvdec with a separate output.media_root.
  threshold: 0.4
  min_duration_s: 2.0
  max_duration_s: 60.0
execution:
  transnetv2:
    cpus: 3
    memory_gib: 2.0
    gpus: 0.25
    inference_cpus: 1.0
    inference_memory_gib: 2.0
    window_batch_size: 1
```

Use a separate output location when changing methods or cutting settings. A Lance table is bound to one cutting
contract, while more sources can be appended under that same contract. TransNetV2 requires the default runtime's CUDA,
PyTorch, PyAV, and FFprobe dependencies, plus PyNvVideoCodec and CV-CUDA when opting into NVDEC.
