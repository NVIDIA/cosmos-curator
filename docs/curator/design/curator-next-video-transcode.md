# Curator Next: Video Transcode

`video-transcode` encodes pending fragments produced by [video-cut](curator-next-video-cut.md), uploads standalone
MP4 clips, and fills clip metadata on the existing Lance rows. Cut and producers of other nullable enrichment columns
can continue writing while transcode runs.

Checkpoint reuse, atomic enrichment, commit reconciliation, and cleanup follow the
[shared checkpoint and recovery contract](curator-next-checkpoint-recovery.md), also used by video-embed. Transcode
supplies span identity, encoding and media durability rules, and successful-clip validation; its `bindings.py`
declares them to the shared contract.

## Execution

```text
register nullable clip columns, then pin the latest Lance span table
  -> recover a saved fragment selection, or select all pending fragments
  -> native Ray checkpoint filter on span_id
  -> group remaining spans by source_id
  -> sort source groups by descending source size
  -> autoscaling download actors (0.25 CPU + one curator_io slot per actor, one in-flight task each)
  -> source bytes and span addresses through Ray's object store
  -> flat_map transcodes each source and yields individual clips from bounded FFmpeg batches
  -> independent clip uploads, dropping bytes
  -> checkpointed result Parquet
  -> regroup metadata by original Lance fragment
  -> validate coverage and stage column files on workers
  -> one atomic Lance column Update, including a run receipt
  -> best-effort workspace cleanup
```

Each invocation validates the span schema and adds the nullable clip fields if absent. This is a metadata-only schema
commit: it creates no placeholder column files and rewrites no span data. The invocation then pins the latest Lance
version and passes it to workers. Reads within an attempt describe one consistent snapshot. Committed cut fragments
are independently ready; a whole-table cut completion record is not required.

Every fragment in the table must have either all clip fields null on every row (pending), or valid completed clip
metadata on every row for the requested encoding contract and media destination. Mixed, partially filled, or incompatible
fragments anywhere in the pinned snapshot are rejected. New work selects all pending fragments; the recipe has no
public selection-size limit. Saved selections follow the
[shared recovery rules](curator-next-checkpoint-recovery.md#recovery-identity-and-workspace), and fragment counts follow
the [shared run summary definitions](curator-next-checkpoint-recovery.md#run-summaries).
Input validation also requires globally unique, non-null span IDs.

Source grouping happens after checkpoint filtering. A source whose spans are all durable is never downloaded.
A partially completed source is downloaded once for its remaining spans. Grouping shuffles only lightweight span
metadata using sort tasks, so shuffle actors do not hold CPU and memory reservations during transcoding.
Each source work row carries its identity, URI, and byte size once; nested spans carry only their identity, interval,
and original Lance row address.
`execution.source_partitions` sets the target read block count used to size this sort.
Source groups are sorted by descending source size before downloading to start large sources early.
Source ID breaks size ties so equally sized sources can occupy separate sort partitions.
Each group and its downloaded source must fit on one worker. Sorted metadata blocks pass directly to downloads.
Each download actor task processes several source rows one at a time, streaming completed videos through Ray's output
blocks. Ray may buffer small videos together. A task retry can repeat downloads for multiple sources.
Download actors reserve 0.25 CPU and one `curator_io` slot from the existing per-node IO budget for their lifetime,
including while idle or backpressured. Ray Data submits at most one task per ready actor. This bounds admission as well
as execution: Ray 2.58's task-pool resource accounting does not include custom resources, so an uncapped task pool can
charge CPU and memory budgets for many submitted downloads that are still waiting for IO slots.

The pool starts with one actor and has no startup-derived maximum. Late workers can supply additional slots, and
actors and their in-flight tasks can restart after worker loss. The download callable retains only configuration,
not required progress or a source-payload cache. The media plan sets the actor upscaling threshold to 1.0: Ray's default
1.75 cannot be reached with one in-flight task per actor. Pending actors count toward utilization, preventing repeated scale-up while
the next actor waits for capacity. These settings do not change the ambient context used by publication.
The shared download-autoscaling utility also cancels surplus pending IO actors when ready actors are idle, allowing
Ray 2.58's normal scale-down to proceed. Ray's cooldown, minimum pool size, and future scale-up remain in effect.

Uploads reserve 0.25 CPU and rely on Ray's normal backpressure. Only the transcode stage reserves
`execution.transcode_cpus`. It also reserves `execution.transcode_memory_gib` (default: 4 GiB) per task, independently of
the CPU reservation.
This replaces Ray's automatic reservation (10 GiB at five CPUs in the Curator container). The budget covers the Python
worker, source staging, and the FFmpeg subprocess; Ray accounts for object-store payloads separately. Increase it for workloads
with larger sources, higher-resolution video, or larger FFmpeg batches. It is a scheduling reservation, not a hard
process memory limit.
See the shared [IO concurrency guidance](ray-data.md#large-source-downloads-and-curator_io).

Source bytes travel with their remaining spans through Ray's object store, allowing download and transcode tasks
to run on different nodes. This uses object-store capacity for prefetched sources. Ray provides output buffering
and backpressure, while the IO-backed actor pool bounds admitted download tasks. This is not a hard memory cap:
whole-source payloads and serialization copies must still fit alongside other work on each node. Reservation-based
high-memory warnings are suppressed for these large payloads; logical memory reservations and backpressure remain enabled.
The worker releases its Python copy of the source bytes after writing the temporary source file.
The `flat_map` transcode generator retains its source file and at most one FFmpeg batch of clip files, yielding one
clip dictionary at a time. Separate upload tasks remove bytes and restore the canonical Arrow metadata types before
metadata reaches the checkpoint sink.

The recipe reuses the shared CPU FFmpeg implementation: independently seekable multi-output encoding, optional audio
copy/AAC, and an individual-span fallback after a multi-output failure. Decoder, filter, and encoder threads are
bounded, and subprocess stdin is disconnected to prevent background-worker hangs. Source sizes, when recorded by
cutting, are checked after download. Every encoded clip is probed before upload. Metadata is coalesced by a separate
streaming repartition into bounded groups before checkpoint writes.

## Identity and canonical output

- `source_id` identifies the immutable source URI under video-cut's contract.
- `span_id` is the native checkpoint ID and remains independent of encoding.
- `clip_id` hashes the span ID and versioned encoding contract.
- Media destinations are `<input.media_root>/clips/<clip_id>.mp4`. Retries replace the same destination.

The encoding contract includes encoder, bitrate, audio mode, and a behavior version. CPU and memory reservations,
encoder threads, FFmpeg batch size, checkpoint size, retries, and worker placement do not affect identity.

Video-transcode preserves the cut fields, row order, fragment IDs, and existing fragment boundaries. It adds
`transcode_contract_id`, `clip_id`, `clip_uri`, byte size, duration, dimensions, frame rate, optional frame count, and video
codec. The exact schemas live in
[contracts.py](../../../cosmos_curator/next/recipes/video_transcode/contracts.py). Arrow field metadata records owner
`video-transcode` and field set `clip`. Added fields are nullable in the Arrow schema so the upstream producer can append
narrow span rows; a completed fragment requires usable clip metadata on every span. Each completed fragment retains its
original span file and adds a file holding the transcode columns.
Fragment IDs and row offsets stay unchanged; the column update creates no row deletions.

All selected pending fragments are filled in one `Update` transaction under the
[shared publication and concurrency rules](curator-next-checkpoint-recovery.md#canonical-publication-and-concurrent-writers).
One logical transcode writer owns the clip columns. The transaction adds a `video_transcode.publication` receipt using
publication format version 1, identifying the input version, selected fragments, encoding contract, media root, and
completed span count, alongside the shared recovery receipt. Fragment values determine readiness and rerun selection;
receipts do not declare later appended fragments complete. Completed rows and transaction receipts must match the
requested encoding contract and destination. An empty table without a transcode receipt still records a zero-span
run, without starting Ray, because video-caption requires a whole-table receipt. Transcode commits that receipt
directly rather than through shared publication: recording it twice is harmless, so it needs no recovery workspace.
This extension goes away once caption checks clip readiness from data.

`video-caption` and `video-embed` have not yet been adapted to select only ready transcode fragments from a growing
table. Caption requires a whole-table transcode receipt; embedding checks the clip schema, IDs, URIs, and byte sizes
directly. Both reject tables with pending clip inputs; selecting ready fragments is separate work.

## Checkpoints and recovery

`execution.parquet_rows_per_file` defaults to **1,000 spans per Parquet/checkpoint group**, using the same setting name as
video-embed. It replaces the development-era `spans_per_checkpoint` setting. At 650,000 spans, that is approximately 650
result shards.
The 100,000-row fragment layout chosen by video-cut remains about seven Lance fragments. These are independent
boundaries: checkpoint size controls durability granularity and fragment size controls canonical storage layout.
Incomplete write groups and other in-flight work may replay; 1,000 is not a bound on total work in flight.

The checkpoint ID is `span_id`. Result Parquet contains span IDs, original fragment IDs and row offsets, plus clip
metadata. It never contains media bytes. Upload must succeed before a result reaches the sink; an upload whose result
was not checkpointed may repeat at the same deterministic clip destination.

The workspace is `<output.staging_root_uri>/<contract-namespace>/<selection-digest>/`. Transcode contributes its
encoding contract and media destination to the
[shared compatibility manifest](curator-next-checkpoint-recovery.md#recovery-identity-and-workspace). Span inputs and
source media follow the [shared immutability requirements](curator-next-checkpoint-recovery.md#assumptions-and-guarantees).
Earlier development receipt formats and workspace layouts have no migration path.

Processing, publication, and cleanup follow the
[shared failure and recovery boundaries](curator-next-checkpoint-recovery.md#failure-and-recovery-boundaries). A retry
can rebuild publication from staged clip metadata without downloading sources or encoding checkpointed spans again.
Retain transcode publication history for video-caption's receipt-based input validation.

## Failure policy and validation

Only successful clips are checkpointed: shared processing validates clip metadata with the publication validator before
Ray records a span ID. Source access, FFmpeg, probe, and upload failures stop the attempt after their applicable
retries. Failed spans are never marked complete or silently omitted. The recipe currently has no quarantine
or terminal bad-media rows, so a completed fragment supplies a usable clip for every span.

Local tests exercise native checkpoint recovery across two fresh Ray clusters and a disjoint schema commit, including
a source spanning multiple Lance fragments, plus actual CPU FFmpeg encoding and publication. The shared conformance
suite runs against transcode's span fixtures, covering incomplete result rejection despite a processing marker, an
ambiguous commit response, concurrent writers, and cleanup failure; recipe tests cover the downstream receipt, empty
input, and clip validation. See the [shared validation criteria](curator-next-checkpoint-recovery.md#validation-and-adoption-criteria).
A CPU-only multi-node test checks submitted-task counts against live IO capacity, adds a late worker, removes every
worker, and finishes publication on a replacement worker.
The 22,000-source / 650,000-span managed benchmark and full Slurm worker-loss qualification have not been run for this
recipe.
For fragment memory and checkpoint-index limits, see the
[shared publication sizing guidance](curator-next-checkpoint-recovery.md#publication-sizing).

## Running

```bash
cosmos-curator pipeline template video-transcode > video-transcode.yaml
cosmos-curator pipeline validate video-transcode.yaml
pixi run --as-is run-pipeline video-transcode.yaml
```

```yaml
schema_version: 1
kind: video-transcode
input:
  media_root: s3://example-bucket/curated/videos
execution:
  transcode_memory_gib: 4
  parquet_rows_per_file: 1000
```

Clips are written under `<input.media_root>/clips`. `input.spans_lance_uri` defaults to `<input.media_root>/lance`, and
`output.staging_root_uri` defaults to `<input.media_root>/staging/video-transcode`; both can be overridden. Local/shared
paths are supported for local execution and must be visible to the relevant Ray workers.

Generate a video-caption config using the same media root and canonical table after transcode completes.
