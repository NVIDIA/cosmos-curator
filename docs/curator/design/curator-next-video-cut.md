# Curator Next: Video Cut

[`video_cut`](../../../cosmos_curator/next/recipes/video_cut/) plans fixed-stride or TransNetV2 scene spans without creating clip media.
[`video-transcode`](curator-next-video-transcode.md) encodes those spans. Together they replace the removed
`video-split` recipe.

## Decision and rationale

The design separates source processing from tabular publication:

```text
select new sources -> probe + plan each source -> checkpointed Parquet with spans[]
                   -> Arrow expansion -> packed Lance fragments -> atomic append
                   -> workspace cleanup
```

- **Separate cutting from transcoding.** Boundaries and source references can be reused across encoding choices.
- **Recover by source.** Keeping a complete source plan together avoids reconciling separate source and span progress.
- **Pack spans independently.** Checkpoint size bounds replay; Lance fragment size controls storage layout.
- **Append one finite selection.** One atomic Lance commit publishes the new spans and run receipt together, allowing
  staging to be deleted afterward while preserving earlier fragments.

The tradeoff is that interrupted publication rebuilds fragments from Parquet. This repeats tabular work while reusing
checkpointed source plans, avoiding a separate publication ledger.

## Input, identity, and output contract

Inputs are explicit S3 MP4 URIs or a recursive S3 root. Discovery normalizes, sorts, and deduplicates selected URIs while
preserving literal S3 keys. Empty selections and listing failures stop before output creation.

Each URI must identify immutable content. Each output is bound to one cutting contract, but its source set can grow.
Root inputs are listed on every invocation. Sources already published are skipped before probing, including recorded
empty plans and terminal errors. Removing a source from a later input selection does not remove its existing spans.
Explicit URIs and root discovery are interchangeable when they select identical sources.

| Identity | Meaning |
| --- | --- |
| `source_id` | Hash of the normalized immutable source URI. |
| `span_id` | Hash of source ID, first video stream (`stream_index: 0`), start, and end. |
| Cutting contract | Method, result settings, stream selection, behavior version, and (for TransNetV2) pinned model identity. |
| Input snapshot | Hash of one invocation's normalized, sorted source URI set; identifies its workspace and receipt. |

Span identity is independent of cutting settings. Intervals are half-open `[start_ns, end_ns)` on a zero-relative video
timeline. The source endpoint is FFprobe's stream duration, falling back to container duration.
[`next.media.spans`](../../../cosmos_curator/next/media/spans.py) defines rounding and fixed-stride tail rules.

Lance is the canonical span dataset. Rows contain source references, media metadata, cutting provenance, and span
boundaries; the v1 schemas are defined in [`contracts.py`](../../../cosmos_curator/next/recipes/video_cut/contracts.py).
Sources with empty plans or terminal errors contribute no spans, but remain included in the committed outcome counts.

One logical cutter owns source selection for each output. Transcode and other producers of nullable enrichment columns
may run concurrently. Cut only appends fragments; existing rows, fragment boundaries, and span columns are immutable.
Compaction, deletion, and overwrites are outside this contract. Execution tuning is excluded from identity and recovery
compatibility.

## Execution and recovery

Phase A exposes source identities through a Ray Data datasource, allowing native checkpoint filtering to skip completed
sources before media access. Fixed-stride planning tasks process sources sequentially within each read block, producing one Parquet
row per source with its metadata, spans, and outcome. A task retry repeats the block's probes. `execution.probe_cpus`
controls probe concurrency and defaults to `0.25` CPUs per task. Empty plans and terminal errors are checkpointed too.

Native checkpoints use `source_id` and advance in groups controlled by `sources_per_checkpoint`. An unfinished group
can replay; setting the group size to one gives single-source durability. Plans and checkpoints are retained until
publication succeeds. The workspace path hashes the input snapshot, cutting contract, full source-result schema, output
destination, and workspace version to isolate compatible recovery state.

On resume, checkpointed outcomes for sources already published by an intervening cut invocation are excluded.
Phase B expands the remaining source plans with Arrow and packs spans across source boundaries into Lance fragments controlled
by `spans_per_fragment`. Workers stage the fragments; the driver commits them with one atomic `Append` transaction.
The transaction's `video_cut.publication` property uses publication format version 1 and records the input snapshot,
processed source IDs, outcome counts, and error report. Even a zero-span run commits its receipt. Reruns recognize the
receipt despite intervening enrichment commits, and later input selections skip its source IDs. Retain Lance transaction
history for these run receipts, especially the identities of sources with no span rows.

Every committed fragment is ready for transcode; there is no whole-table completion gate. An empty table can coexist
with an active cut run. Append preserves enrichment columns, and newly appended narrow span rows read those nullable
columns as null. Previous fragments are never repacked, including an earlier run's partially filled final fragment.

This implementation assumes freshly created datasets and recovery workspaces. Source completion is determined from
transaction receipts, including zero-span outcomes; earlier schema receipts have no migration path. Tables carrying
schema receipts are rejected at open with a request to use a new output location.

| Failure boundary | Recovery |
| --- | --- |
| Before a source group is checkpointed | Replay incomplete source work. |
| After source checkpoints, before the Lance commit | Reuse source plans and rebuild publication fragments. |
| Commit succeeds but its response is lost | Recognize completion in Lance; the active invocation can proceed to cleanup. |
| Driver exits after commit, or cleanup fails or is interrupted | A later invocation recognizes completion and returns success without starting Ray or touching staging. |

After confirming publication, an invocation attempts to delete only the configuration-derived workspace it used or
resumed. Cleanup is best effort: success returns `cleanup_complete: true`; a failure logs a warning and returns
`cleanup_complete: false` without invalidating the output. An invocation that finds publication already complete skips
cleanup and returns `cleanup_complete: null`, even if its staging root or storage profile has changed. Publication
metadata never supplies a deletion path. Crashes and cleanup failures may leave orphaned staging for explicit cleanup
or a storage expiration policy. Uncommitted fragment files from failed attempts can be removed through Lance cleanup.

The fixed-stride method has completed a managed Ray run on Slurm. Failure recovery is tested locally with synthetic probe metadata.

### TransNetV2 execution

`split.method: transnetv2` defaults to PyAV preprocessing (`split.decoder: pyav`), with separate CPU decoding
and GPU inference stages:

```text
checkpoint-filtered source blocks -> download actors
  -> encoded bytes through Ray -> CPU tasks: PyAV decode, resize
  -> tiny RGB frames and timestamps through Ray -> GPU actors: TransNetV2 inference
  -> one complete source plan -> checkpointed Parquet -> existing Lance append
```

Downloads reserve 0.25 CPU and one `curator_io` slot per actor, with one in-flight call. PyAV decode tasks default to
3 CPUs (`execution.transnetv2.cpus`) and 2 GiB of host memory (`memory_gib`). Inference actors use
1 CPU (`inference_cpus`), 0.25 GPU (`gpus`), and 2 GiB of host memory (`inference_memory_gib`). The different resource
requests keep Ray from fusing decoding back into inference, so CPU decode concurrency can grow independently of the
model pool.

Set `split.decoder: nvdec` to opt into GPU preprocessing. Encoded bytes flow directly from download actors to GPU
actors, where NVDEC decoding and inference share the same actor and GPU reservation. The actor's host memory
reservation is the sum of `memory_gib` and `inference_memory_gib` (4 GiB by default). PyNvVideoCodec decodes in
presentation order and CV-CUDA performs antialiased linear resizing to 48x27 RGB. Full-resolution frames stay on the
GPU; only the tiny resized frames return to host memory for common windowing. NVDEC requires supported NVIDIA
decoding hardware. Unclassified decoder/CUDA failures propagate.

Both backends use actor pools that can grow with available resources, including late workers; neither has a
startup-derived maximum. The actor autoscaling threshold is 1.0 for the media plan.
See the [large-download guidance](ray-data.md#large-source-downloads-and-curator_io).

NVDEC uses native demuxer timestamps, including variable cadence, and drains delayed B-frames at end of input.
Its color conversion and resizing differ from PyAV, so decoder selection belongs to the cutting contract.
Switching decoders requires a separate `output.media_root`; PyAV checkpoints and published spans are not reused for NVDEC.
To resume existing NVDEC output, keep `split.decoder: nvdec` explicit. PyAV is the default based on the managed Slurm
comparison below; NVDEC remains opt-in while throughput and output quality are evaluated further.

The shared download-autoscaling utility is scoped to the TransNetV2 media plan. It cancels surplus pending IO actors
that would otherwise prevent Ray 2.58 from scaling down an idle download pool, preserving normal cooldown and scale-up.

Download tasks process sources sequentially within each read block. Ray streams encoded bytes downstream using its
normal block-size targets; inference still plans one source per call. Task retries may repeat multiple sources from a block,
while durable recovery remains source-level through the existing checkpoints.

Each inference actor loads the pinned FP32 PyTorch TransNetV2 checkpoint once. The driver checks that the checkpoint
file exists before starting Ray, and every new actor checks before loading it. Staging must supply the revision in
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
alongside integer timestamps and probe metadata; encoded source bytes stop at the decode stage. These transient buffers
are reconstructed as NumPy views by inference actors and are never part of the durable source-result schema.
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

TransNetV2 defaults to 32 sources per checkpoint when that setting is omitted (fixed-stride remains 1,024). Completed
plans contain no frames or encoded bytes. Incomplete groups and other in-flight sources may replay after failure.
Missing objects and recognized invalid or unsupported media become terminal errors; storage infrastructure, model setup,
CUDA, and unclassified decoding failures propagate. If NVDEC selects a different video stream from the required first
video stream, the source receives a terminal `source-decode` error. Rerun those sources with `split.decoder: pyav` and
a fresh output location. Inference failures never become successful empty plans.

A CPU Ray integration test exercises the separate decode and inference boundary with a deterministic CPU model,
terminal missing/invalid-source outcomes, Lance publication, and reuse of source checkpoints. The combined predecessor
also completed a local GPU smoke run using the pinned model checkpoint. Both backends completed the managed Slurm
runs below; qualification for changes in worker membership and worker loss remains outstanding.

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

For fixed-stride, `probe_mode: auto` lets FFprobe read a presigned S3 URL, using HTTP ranges where possible and falling back to a temporary
download if probing fails. `probe_mode: download` always downloads first. Signed URLs are never persisted, and avoiding
a full-object transfer is not guaranteed.

- **Terminal errors:** missing objects, unsupported media after a successful download, or local FFprobe failures
  explicitly reporting `AVERROR_INVALIDDATA`. These are checkpointed and skipped on recovery; retrying them requires a
  fresh output. A nonzero probe exit alone does not establish a terminal media problem.
- **Retryable failures:** storage access problems (including archived objects and credentials), missing tools, local
  probe timeouts, process termination, unclassified probe failures, and unexpected errors. These propagate without
  terminal checkpoints, leaving the source retryable. Inconclusive probes receive up to three Ray map retries; exhaustion
  fails the attempt without publishing completion.

Terminal error details survive cleanup at `<spans_lance_uri>/video-cut/<input_snapshot>/errors.jsonl`. Each report is
complete only when referenced by its committed run receipt. Summary `source_results` and `errors` describe that run's
new sources; `spans` counts all canonical spans accumulated in the table.

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
  sources_per_checkpoint: 32
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
