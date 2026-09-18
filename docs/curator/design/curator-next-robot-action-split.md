# Curator Next: Robot Action Split

## Summary

`robot-action-split` builds on the original Curator Next video-split design to support
**semantically-bounded span extraction** from LeRobot/Mecka robot datasets. Where
`video-split` produces fixed-stride clips from a flat MP4 inventory, this recipe
reads parquet-backed episode metadata, identifies subtask boundaries, and cuts
each span as a standalone MP4 together with its per-span robot action data (`.bin`
file) and a JSON metadata sidecar. Multiple camera views of the same span are
treated as a coherent group and share a stable `span_group_id`.

The comparisons with `video-split` below describe the removed recipe's historical contract. Flat-video workflows now
use [video-cut](curator-next-video-cut.md) and [video-transcode](curator-next-video-transcode.md). Robot-action splitting
retains its own execution and publication contract and uses shared utilities without importing the removed recipe.

The design reuses the Curator Next programming model (config system, Lance
publication) and the `cut_plan` smart cut for GOP-aware stream copying, but
replaces fixed-stride span generation with a parquet-driven span provider.

---

## Motivation

This recipe brings robot action dataset curation into the Curator Next programming
model, giving it:

- The shared config/manifest/receipt contract the planner already understands
- Curator's scalability invariants (bounded per-attempt work, no global Lance scans)
- GOP-aware smart cut — stream-copy for interior frames, re-encode only the head
- Clean separation between span discovery (CPU-bound, parquet-driven) and media
  work (I/O-bound, per-source decode + encode)

---

## Input Format

### LeRobot/Mecka Dataset Layout

```text
{shard}/
  meta/info.json                         # fps, feature shapes, view list
  meta/subtasks.parquet                  # subtask_index -> subtask label
  meta/tasks.parquet                     # task_index -> task label
  meta/episodes/chunk-NNN/file-MMM.parquet  # episode/view locations, optional task_description
  data/chunk-NNN/file-MMM.parquet        # per-frame: episode_index, frame_index,
                                         #   subtask_index, observation.state.*, action
  videos/<view>/chunk-NNN/file-MMM.mp4   # one dir per camera view
```

`NNN` and `MMM` are three-digit and zero-padded. Discovery lists `data/` by
parsing any digit width, but rebuilds every data and video path with three, so a
name padded to another width is listed and then unreadable. That raises rather
than skipping the file: a skip drops the file's spans while surviving spans from
its siblings keep the run reporting success, so the loss is invisible.

Episodes can span multiple shards. A single episode's video and its data parquet
may live in different chunk files (multi-file layout). `meta/episodes/*/` records,
per episode and per view, which `chunk_index`/`file_index` holds that episode's
frames and the `from_timestamp` offset within that MP4. Its optional
`task_description` is episode metadata: discovery resolves it by `(shard,
episode_index)` and copies it into every span for that episode. The exact Python
string is preserved, including whitespace and the empty string; a missing column
or source null becomes `None`. A present column must use an Arrow text type.
Discovery does not join descriptions through `tasks.parquet`, because numeric
task indices are shard-local and need not identify one description.

### Label Resolution

`task_name` and `subtask_name` come from the meta parquets. The label column is
resolved against the file's **Arrow schema**, by name and by type:

| Meta file | Index column | Accepted label column, in precedence order | Required type |
| --- | --- | --- | --- |
| `meta/tasks.parquet` | `task_index` | `task`, `task_name`, `__index_level_0__` | any Arrow string |
| `meta/subtasks.parquet` | `subtask_index` | `subtask`, `subtask_name`, `__index_level_0__` | any Arrow string |

`__index_level_0__` is last so a real named column always wins over a preserved
index. "Any Arrow string" means `string`, `large_string`, `string_view`, or a
dictionary of those — every encoding that survives a parquet round trip as text.
`binary` is excluded: its values arrive as `bytes`, which stringify to
`b'make coffee'` rather than to the text, and that repr would also stop matching
the span filters.

Exporters carry the label either as a real column or as a pandas index, which
`DataFrame.to_parquet` preserves as an ordinary Arrow column — named (`task`) when
the index had a name, `__index_level_0__` when it did not. Reading the file
through `Table.to_pandas()` would move a preserved index back into
`DataFrame.index`, where a column scan cannot see it and the label appears
absent, so resolution reads `Table.column_names` directly and never goes through
pandas. See `_read_label_map` in
`cosmos_curator/next/recipes/robot_action_split/discovery.py`.

The **type** half of the contract matters as much as the name: `__index_level_0__`
means "pandas preserved whatever the index was", and a non-`RangeIndex` integer
index lands there as `int64`. Accepting it on name alone would turn row numbers
into labels, which is the defect this contract exists to prevent, so a candidate
column that does not hold text is rejected as though it were absent.

The type gate rejects *numeric* identifiers, not identifiers in general: a
preserved index holding strings such as `task_0001` satisfies it and is accepted
as a label. Telling an id-shaped string from prose would need a heuristic, which
is exactly what this contract replaced, so the name precedence carries that
weight instead — a real named column always wins over a preserved index, so an
export that also carries prose resolves to the prose.

Labels are never synthesized from an index. Three failure modes follow from that.

A meta file **raises**, failing the whole discovery run, when it:

- carries no accepted label column of a text type — the error names the file, the
  expected name and every observed column with its type;
- maps one index to two different labels;
- has rows but none carrying a usable label;
- is `tasks.parquet` and yields no labels at all.

A span is **dropped** when its `task_index` or `subtask_index` resolves to no
label — either the index has no row in the meta parquet, or its row's label is
null or blank. Spans also drop, for a reason unrelated to labels, when their
`episode_index` has no row in `meta/episodes/`. Each data file reports both
causes, with the indices behind them, in one warning per cause.

A **run** raises when its data files produced no span at all. A drop is per-span
and tolerable in isolation, but a meta file that overlaps no index in `data/` —
stale, or numbered from 1 against 0-based data — drops *every* span, whether it
is `subtasks.parquet` missing the labels or `meta/episodes/` missing the rows.
Returning nothing is indistinguishable from a dataset with nothing to do, which
the caller reports as a successful empty run, so total data loss would exit 0.
The check runs before the span filters, because filtering every span out is a
legitimate outcome of the configured rules rather than a data fault.

Two cases are **not** drops, and both fall back to `task_name`: a dataset whose
`data/` declares no `subtask_index` column, and a shard whose `subtask_map` is
empty — no `subtasks.parquet`, or one that resolves to nothing. The fallback is
keyed on the *map*, not on the index: a `subtask_map` that exists but has no row
for a declared `subtask_index` is the drop case above, not this one.

The distinction is what the label would claim.
Where no subtask bounds exist the span *is* the task run, so the task label
describes it exactly; where a subtask bounds it, the span is a fragment of that
task and the task label would over-claim the whole. Falling back there would
relabel a fragment as the entire task — quietly, and in the field, at whatever
rate the meta file is incomplete.

A fallback label also carries no dedup weight. Per-episode dedup caps spans per
subtask label, so spans sharing one stand-in label would compete for a single
allowance and lose every run past it — deleting a label table would then cost
spans that the geometry still supports. Fallback spans are bucketed by
`subtask_index` instead; two indices resolving to the same real text stay one
subtask-label bucket and share one allowance.

The enumeration is deliberately no narrower than what it replaced. The previous
reader resolved `tasks.parquet` with a wildcard — *any* column that was not
`task_index` — and `subtasks.parquet` against `("subtask", "subtask_name")`.
Dropping the wildcard is the point of this contract: it could not tell prose from
an identifier, so a wrong pick was indistinguishable from a right one. Dropping
`task_name` or `subtask_name` would not have been, since both name the label
explicitly, so both are still accepted. A column under any *other* name now
raises rather than being guessed at, and the error names it: admitting a new
spelling is a one-line change.

### Span Definition

A **span** is a maximal contiguous run of the same `subtask_index` within one
episode. Datasets without `subtask_index` fall back to segmenting by `task_index`.
Span boundaries come entirely from parquet metadata; no video decode is required
during discovery. Boundaries are detected with a single numpy `diff` pass over
the sorted frame rows.

---

## Recipe Design

### New Kind: `robot-action-split`

This is a distinct kind from `video-split`. The two recipes share toolkit
components but differ in:

| Dimension | `video-split` | `robot-action-split` |
|-----------|--------------|----------------------|
| Span source | fixed-stride | parquet subtask boundaries |
| Source unit | single MP4 | dataset chunk MP4 + data parquet |
| Source input | `input.uris` (.mp4 list) | `input.uris` (dataset roots) |
| Extra output | none | action `.bin` sidecar per span |
| Multi-view | no | yes (one clip per view per span) |
| Action data | n/a | ACT2 fixed-stride binary |
| Lance row unit | one per clip | one per (span, view) |

### Config Contract

```yaml
schema_version: 1
kind: robot-action-split

input:
  uris:
    - s3://example-bucket/robot_data/lerobot_v30/dataset_name/
  source_dataset: dataset_name
  limit: null   # max source chunk MP4s to process; matches video-split semantics

split:
  min_duration_s: 4.0
  max_duration_s: 20.0
  skip_labels: ["no action", "no actions"]
  skip_label_prefixes: ["hold", "adjust"]
  skip_label_substrings: ["idle"]
  max_keep_per_description: 3
  dedup_prefer_min_s: 5.0
  dedup_prefer_max_s: 10.0

output:
  media_root: s3://example/clips/
  lance_uri: s3://example/robot_clips.lance
  action_format: bin   # "bin" (ACT2, default) or "pickle" (debug)
  video_bitrate: "2M"  # bitrate for libopenh264 head re-encode (only ~0.5 frames/clip on GOP=2 sources)
  views: []            # empty = all available views

execution:             # all fields optional; shown with defaults
  storage_profile: default
  discovery_workers: 4
  max_segments_per_batch: 50
  clips_per_publish_batch: 8000   # commit a Lance fragment every N successful clip rows
  storage_attempts: 3             # retries for the idempotent Lance fragment commit
```

---

## Span Provider

The span provider is the only truly new toolkit component this recipe adds. It runs
before the processing loop and emits an ordered list of `SpanWorkItem` objects — one
per (span, view):

```python
@attrs.define(frozen=True)
class SpanWorkItem:
    # Identity
    source_id: str             # hash of normalized dataset root URI
    span_group_id: str         # hash of (source_id, episode_id, subtask_index, frame_start)
    clip_id: str               # hash of (media_contract_version, span_group_id, video_bitrate[, view_name])
    view_name: str

    # Source location (resolved from episode parquet metadata)
    chunk_mp4_uri: str
    data_parquet_uri: str

    # Span geometry (integer frames)
    episode_index: int
    episode_frame_base: int    # first frame_index of this episode in the chunk
    frame_start: int
    frame_end: int             # exclusive
    native_fps: float
    episode_from_timestamp: float   # episode start offset within the chunk MP4 (seconds)

    # Labels
    subtask_index: int
    subtask_name: str
    task_index: int
    task_name: str
    task_description: str | None

    # Source episode identity
    episode_id: str

    # Camera
    camera_intrinsics: list[float] | None
```

Discovery phases:

1. **Shard enumeration** — list `shard_*/` dirs under each `input.uri`.
2. **Per-shard prep** (thread pool) — read `meta/info.json`, `subtasks.parquet`,
   `tasks.parquet`; discover `videos/` view dirs; list `data/chunk-*/file-*.parquet`.
3. **Per-data-file segment build** (thread pool) — read each
   `data/chunk-*/file-*.parquet` + `meta/episodes/*/` parquet; run numpy
   change-detection to emit segments; resolve the per-view video file and optional
   `task_description` from episode metadata.
4. **Filter + dedup** — apply `split.*` config (duration bounds, label skip,
   per-episode dedup per subtask label).
5. **Batch assembly** — group `SpanWorkItem` objects by `(chunk_mp4_uri, data_parquet_uri)`.

No source media is touched during discovery.

---

## Timestamp-Based Cutting

### Smart Cut

The cutting layer uses a **smart cut** strategy implemented in
`cosmos_curator/next/media/smart_cut.py`. For each clip, `cut_plan` builds a
per-frame PTS index from container packets (no decode), then:

- **Stream-copies** interior GOPs bit-exact from the source when the clip starts on
  a keyframe or the source is all-intra.
- **Re-encodes only the head** frames (those before the first keyframe inside the
  clip boundary) when a stream copy would be inexact.
- **Falls back** to a full re-encode for any clip that fails the fast path or any
  source that is not copy-safe.

With Mecka's keyframe-dense encoding (~GOP=2), re-encoding is limited to ~0.5
frames per clip on average. Copied frames are bit-exact — no generation loss.

### Cutting Flow

`process_batch` handles all spans from one `(chunk_mp4_uri, data_parquet_uri)` pair:

```text
process_batch(ChunkSpanBatch(chunk_mp4_uri, data_parquet_uri, [SpanWorkItem, ...]))
  │
  ├─ read_bytes(chunk_mp4_uri)    → chunk_bytes   ─┐ one try/except:
  ├─ read_bytes(data_parquet_uri) → parquet_bytes  ─┘ any failure → all items failed
  ├─ parse parquet → action_arrays for all episodes in this batch
  ├─ write chunk_bytes to temp file (cut_plan requires a local path)
  │
  ├─ cut_plan(chunk_local, cut_specs, bitrate=video_bitrate, smart_cut=True)
  │      builds per-frame PTS index from container packets (no decode)
  │      for each span: stream-copy interior GOPs + re-encode head → local clip MP4
  │      falls back to full re-encode if stream copy would be inexact
  │      any failure → all items failed
  │
  └─ for each SpanWorkItem (independently; one failure does not skip others):
       ├─ write_media(clip_uri, ...)       — failure → failed outcome
       ├─ encode_action_bin + write_media  — failure → failed outcome
       └─ write_media(sidecar_uri, ...)    — failure → failed outcome
```

### Frame Accuracy and Timestamp Translation

Frame-accurate cutting requires staying in **integer frame-index space** when
translating parquet span geometry to video coordinates. `episode_from_timestamp`
is sourced directly from the vendor's parquet files and is exact — the actual
container PTS of the episode's first frame. Converting to an integer frame index
via `round(ts * fps)` keeps all arithmetic integer and avoids floating-point
accumulation:

```python
chunk_frame_offset = round(episode_from_timestamp * native_fps)  # integer
abs_start = chunk_frame_offset + (frame_start - episode_frame_base)
abs_end   = chunk_frame_offset + (frame_end   - episode_frame_base)

start_ns = round(abs_start / native_fps * 1e9)
end_ns   = round(abs_end   / native_fps * 1e9)
```

`cut_plan` uses the inclusive `abs_start` / `abs_end - 1` frame indices to seek
into the chunk and determine the copy/re-encode boundary.

### Sensor Library for Downstream Stages

`CameraSensor` (`cosmos_curator/core/sensors/`) is the decode layer for stages
**downstream of cutting** — embedding and captioning. Those stages open the stored
clip MP4s and need decoded frames to feed a model. `SamplingGrid` provides temporal
control; `SensorGroup` provides multi-view alignment if a model sees multiple
cameras. The cutting stage does not use `CameraSensor` — it never needs decoded
pixels.

---

## Multi-View Support

Each span produces one clip per selected camera view. Views are discovered from
`shard/videos/observation.images.*/` at prep time. The `source_is_multiview` flag
(set before any view filter is applied) controls whether `clip_id` folds in the
view name:

- single-view source: `clip_id = hash(media_contract_version, span_group_id, video_bitrate)`
- multi-view source: `clip_id = hash(media_contract_version, span_group_id, video_bitrate, view_name)`

All views of the same span share the same `action/<action_id>.bin` file.
Per-view clips and JSON sidecars use the per-view `clip_id` as their filename stem.
The sidecar places the episode's nullable `task_description` directly after
`task_name`.

---

## Action Data Output

Each span (not each view) produces one file at `<media_root>/action/<action_id>.bin` where `action_id` encodes `(span_group_id, action_format, source_dataset, action_contract_version)`.

The `output.action_format` config field controls serialization:

- **`"bin"` (default)** — ACT2 fixed-stride binary: a 1 KB self-describing JSON
  header followed by packed `float32` frame records and a per-clip tail. Implemented
  in `cosmos_curator/next/media/action_binary.py`. Requires `source_dataset` to be
  registered in `ACTION_BINARY_SPEC_BY_DATASET`; an unregistered dataset fails at
  serialization time with a clear error. Registered platforms: `mecka`, `libero`,
  `droid_lerobot`, `robomind_franka`, `robomind_franka_dual`, `robomind_ur`.

- **`"pickle"` (debug/development)** — Python pickle bytes. No dataset registration
  required. Use this for smoke runs against unregistered datasets or local testing
  where downstream ACT2 consumers are not involved.

---

## Lance Schema

`records.py` publishes successful outcomes to the canonical, append-only `CLIP_SCHEMA` table and
projects failed outcomes separately to a replaced-each-run `errors.json` report — never to
Lance. Keeping failures out of the canonical table (rather than a mixed success/failure schema)
is what makes the append-once-per-`clip_id` recovery invariant below hold: a retried failure never
collides with an already-committed row, because a failed attempt never gets a canonical row to
retry against in the first place.

`CLIP_SCHEMA`:

| Column | Arrow type | Non-null | Meaning |
|--------|-----------|---------|---------|
| `record_schema_version` | `int32` | yes | `CLIP_RECORD_SCHEMA_VERSION` at write time |
| `media_contract_version` | `int32` | yes | `MEDIA_CONTRACT_VERSION` at write time |
| `clip_id` | `string` | yes | Stable per-view clip identifier |
| `span_group_id` | `string` | yes | Stable id shared across all views of this span |
| `view_name` | `string` | yes | Camera view (e.g. `observation.images.wrist_image`) |
| `source_id` | `string` | yes | Hash of the normalized dataset root URI |
| `source_dataset` | `string` | yes | Registered dataset name (`source_dataset` config field) |
| `episode_id` | `string` | yes | Source episode identifier (from episode parquet) |
| `episode_index` | `int32` | yes | Episode index within the shard |
| `subtask_index` | `int32` | yes | Subtask index within the episode |
| `subtask_name` | `string` | yes | Human-readable subtask label |
| `task_index` | `int32` | yes | Task index |
| `task_name` | `string` | yes | Human-readable task label |
| `task_description` | `large_string` | no | Exact episode description from `meta/episodes`, when present |
| `frame_start` | `int32` | yes | Episode-local start frame index (inclusive) |
| `frame_end` | `int32` | yes | Episode-local end frame index (exclusive) |
| `start_ns` | `int64` | yes | Span start as nanoseconds within the chunk MP4 |
| `end_ns` | `int64` | yes | Span end (exclusive) as nanoseconds within the chunk MP4 |
| `native_fps` | `float64` | yes | Source video frame rate |
| `episode_from_timestamp` | `float64` | yes | Episode start offset within the chunk MP4 (seconds) |
| `clip_uri` | `large_string` | yes | Written clip MP4 URI |
| `action_data_uri` | `large_string` | yes | Written action `.bin`/`.pickle` URI |
| `camera_motion_annotation` | `large_string` | no | Computed motion description, when available |

The `errors.json` report carries the same identity, labels, episode metadata, and
geometry columns as `CLIP_SCHEMA` plus non-null `error_stage` and `error_message`;
it has no `clip_uri`/`action_data_uri`/`camera_motion_annotation`, and it is never
diffed for dedup — a rerun simply replaces it.

`clip_id` and `span_group_id` are SHA-256 digests. `span_group_id` hashes
`(source_id, episode_id, subtask_index, frame_start)`. `clip_id` always hashes
`(media_contract_version, span_group_id, video_bitrate)`, plus `view_name` for
multi-view sources. `action_data_uri` is keyed by `action_id`, which hashes
`(action_contract_version, span_group_id, action_format, source_dataset)`.
`task_description` participates in none of these identities. Record schema
version 4 adds that nullable metadata field; reconciliation rejects rows stamped
with version 3. `MEDIA_CONTRACT_VERSION` remains unchanged because it versions
generated MP4 bytes and clip identity, not metadata-only Lance or sidecar fields.

**Downstream — embeddings.** The embed leg
([curator-next-embeddings.md](curator-next-embeddings.md)) consumes a projection
of this table (`clip_id`, `task_name`, `subtask_name`, `clip_uri`,
`action_data_uri`, `source_dataset`) and adds its vectors as additive, nullable
per-modality column groups (`embedding_<modality>_*`) written directly onto
`clips.lance`, plus a fitted action-PCA basis. `CLIP_SCHEMA` declares all six of those
columns non-null, which the embed leg's read contract (`EMBED_SOURCE_ROW`) already expects.

---

## Cross-Run Recovery

The recovery protocol follows [Curator Next Incremental Curation](curator-next-incremental-curation.md): a canonical Lance table that
is both published output and cross-run checkpoint, deterministic identities, idempotent fragment-scoped
commits, and driver-side reconciliation before any expensive work starts.

**v3-to-v4 rollout.** Record schema v4 adds nullable `task_description`.
The recipe intentionally provides no in-place migration. Strict schema
validation refuses an existing v3 table that lacks the field before
reconciliation starts. Manually adding the field is not enough: reconciliation
separately refuses old rows still stamped with `record_schema_version = 3`.
Recreating the table is the simplest rollout, but rerunning this recipe recuts
and republishes the media. For an expensive existing table, use a separate
metadata-only backfill keyed by `(source_id, episode_id)` to populate
`task_description` for every matching span/view row and stamp the completed
rows with schema version 4.

**Recovery unit.** `source_id` here hashes the *whole dataset root*, which can span many shards, chunk MP4s,
and thousands of spans — too coarse a unit to check "is this already done" against. The natural recovery unit
is `ChunkSpanBatch`'s key, `(chunk_mp4_uri, data_parquet_uri)`: the same unit `_iter_sequential` downloads
once and reuses, and the unit Ray Data parallelizes over. Reconciliation (`recovery.reconcile_batches`)
operates at `clip_id` granularity within each batch: a batch whose every `(span, view)` `clip_id` is already
committed is dropped before any chunk MP4 download; a partially-committed batch is still downloaded and cut,
but only for its still-missing `SpanWorkItem`s.

**Why no known-source replan shortcut.** Unlike `video-split`, span discovery here (`discover_spans`) is cheap,
pure parquet/JSON metadata reading — no video decode, no probing (see "Span Provider" above: *"No source media
is touched during discovery"*). `video-split`'s reconciliation goes to real effort reconstructing a known
source's expected clip geometry from stored Lance fields specifically to avoid re-probing, since probing costs
an object read. That shortcut isn't needed here: rerunning `discover_spans` in full is itself the cheap,
always-correct way to recompute "what should exist," so reconciliation is a direct diff against freshly
discovered `clip_id`s rather than a replan from stored metadata.

**Idempotent commits.** Successful clip rows are buffered and staged/committed as one Lance fragment every
`clips_per_publish_batch` rows (buffer-to-threshold, plus one final smaller flush at run end — the same shape
`video-split` uses), via the shared `cosmos_curator.next.utils.lance_fragment_recovery` primitives: before each
commit, the candidate fragment's `clip_id`s are checked against the latest table — all present means the
fragment already landed (skip), none present means it's safe to append, and partial presence is a protocol
violation (raised, never silently resolved). The same check resolves an ambiguous commit response before retry.

**Action sidecar and multi-view rows.** Unlike `video-split`'s one-row-per-unit-of-work, multiple `CLIP_SCHEMA`
rows (one per view) share one action `.bin` file. Reconciliation and commit still operate at `clip_id` (per-row)
granularity — a partially-committed span (one view's clip committed, another's still missing) is a normal,
independently-retryable state, not a violation. The action `.bin` write inherits the same
durable-payload-before-metadata-commit property as clip media: `action_id` and its destination path are
deterministic, so a redundant write (from a retry, or from each view's independent write attempt at the same
shared path) is safe to blindly repeat.

**Ordering under `input.limit`.** `limit` caps the number of distinct chunk MP4s discovery processes. Because
reconciliation only ever diffs against the current run's freshly discovered batch set — never a wider stored
expectation — a `limit`-capped smoke run can't be misread as "this shard is fully processed." This does require
chunk selection order to be deterministic across runs (e.g. sorted by URI), so two `limit`-capped runs over the
same input see the same missing-work set rather than a shifting window.

**Known gap.** Commit granularity is row-count-based (buffer-to-`clips_per_publish_batch`), not chunk-based —
a crash mid-buffer loses that buffer's uncommitted work, same as `video-split`. A `robot-action-split` run
smaller than one publish batch (well under the 8,000-row default) gets no more incremental checkpointing
benefit than a single-shot writer would; only large-scale runs see the crash-safety improvement. Lowering
`clips_per_publish_batch` trades checkpoint frequency for fragment count / downstream Ray Data parallelism.

---

## Pipeline Execution

`pipeline.run` reconciles freshly discovered spans against the canonical clip table
before processing, then processes only missing work — sequentially (`_iter_sequential`)
or in parallel via Ray Data `flat_map` (`_iter_ray_data`, the default). Successful clip
rows are buffered and committed as a Lance fragment every `clips_per_publish_batch`
rows, not only once at the end of the run.

```text
pipeline.run(config)
  │
  ├─ discover_spans(config)                          → list[ChunkSpanBatch]
  ├─ open_or_create_clip_table(lance_uri)
  ├─ reconcile_batches(batches, dataset)              → drop committed spans, keep the rest
  │
  ├─ for each remaining outcome (sequential or Ray Data flat_map):
  │     buffer successful rows; flush a Lance fragment every clips_per_publish_batch
  │
  └─ write run_summary.json (local media_root only)
```

Batching groups `SpanWorkItem` objects by `(chunk_mp4_uri, data_parquet_uri)` so
each call to `process_batch` reads one chunk file and one parquet shard. The
`max_segments_per_batch` config cap splits large chunks so each batch stays bounded.

---

## Relationship to `video-split`

| Concern | `video-split` | `robot-action-split` |
|---------|--------------|----------------------|
| Config kind | `video-split` | `robot-action-split` |
| Span provider | fixed-stride | parquet subtask |
| Cut | `transcode_span` (full re-encode) | `cut_plan` (smart cut: stream copy + head re-encode) |
| Downstream decode | `CameraSensor` | `CameraSensor` (shared) |
| Toolkit: Lance publish | shared | shared |
| Source manifest item | `{source_uri}` | `{dataset_root, chunk_mp4_uri, data_parquet_uri, ...}` |
| Extra outputs | none | `.bin` per span |
| Multi-view | no | yes |

---

## Open Questions

1. **Duplicate-episode filter** — cross-shard dedup of repeated episodes. Planner-level
   concern or v1 recipe scope?

2. **Face-blur episode filter** — cloud-backed episode blocklist. In scope for v1
   or deferred?

3. **`moov_offset`/`moov_size`** — cheap post-encode step for efficient clip streaming
   access. Defer to later iteration?

---

## Validation Criteria

In addition to the shared recipe configuration, identity, publication, and recovery checks, the first
`robot-action-split` implementation is complete when:

- span provider emits the correct segments for a fixture LeRobot/Mecka shard,
  including multi-file and multi-view layouts
- `episode_from_timestamp` offsets are applied correctly so cut clips match the
  frame range in the source parquet
- single-view and multi-view datasets both produce stable `clip_id`s across
  attempts and batch configurations
- `span_group_id` is identical across per-view rows for the same span
- action `.bin` files contain the expected per-frame fields in the registered spec
- cross-run recovery is implemented per "Cross-Run Recovery" above: freshly discovered
  spans are reconciled against the canonical clip table before any chunk MP4 is
  downloaded, and successful clip rows commit incrementally as bounded, idempotent
  Lance fragments
