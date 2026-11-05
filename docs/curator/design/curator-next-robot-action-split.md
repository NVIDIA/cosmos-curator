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

- The shared config contract and checkpointed append publication
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
  limit: null   # max data chunk directories (data/chunk-NNN/) discovery reads; for smoke runs
  # storage_profile: <profile>   # defaults to execution.storage_profile; set it when sources live in another store

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
  lance_uri: s3://example/clips/lance/clips.lance              # default <media_root>/lance/clips.lance
  staging_root_uri: s3://example/clips/staging/robot-action-split  # default; recovery workspace
  action_format: bin   # "bin" (ACT2, default) or "pickle" (debug)
  video_bitrate: "2M"  # bitrate for libopenh264 head re-encode (only ~0.5 frames/clip on GOP=2 sources)
  views: []            # empty = all available views

execution:             # all fields optional; shown with defaults
  storage_profile: default        # serves the clip table, staging root, and media payloads
  discovery_workers: 4
  cut_cpus: 4.0                   # CPUs per chunk task, and concurrent cuts within it
  max_items: 1000                 # chunks per published selection; null = every pending chunk
  rows_per_checkpoint: 8          # chunk outcomes per checkpoint group
  rows_per_fragment: 8000         # clip rows per appended Lance fragment
  media_write_attempts: 3         # attempts per clip, action, or sidecar write before the task fails
  chunk_download_attempts: 3      # attempts to stage a remote chunk MP4 before the task fails
  no_progress_timeout_s: 7200     # Ray Data's wedge detector; one task cuts a whole chunk
  commit_attempts: 5
  tmp_dir: null
```

The split settings, `source_dataset`, `action_format`, `video_bitrate`, and the media and action contract
versions form the result contract that binds the clip table (see "Cross-Run Recovery"); changing any of them
needs a new `lance_uri`. Locations, `views`, `input.limit`, and every `execution` setting select or schedule work
without changing a chunk's outcome, so they stay outside it.

After the first successful publication from a discovered shard, its source
media and span-defining Parquet/JSON metadata are immutable inputs. Corrected
source versions follow the new-output or complete-rebuild procedure in
"Cross-Run Recovery"; overwriting those artifacts in place is unsupported.

---

## Span Provider

The span provider is the only truly new toolkit component this recipe adds. It runs
before the processing loop and emits an ordered list of `SpanWorkItem` objects — one
per (span, view):

```python
@attrs.define(frozen=True)
class SpanWorkItem:
    # Identity
    source_id: str             # hash of normalized discovered shard URI
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
5. **Chunk assembly** — group `SpanWorkItem` objects by `(chunk_mp4_uri, data_parquet_uri)`. Each group is
   one work item: `chunk_id` digests both URIs, and the group's spans travel as canonical JSON in `spans_json`,
   which round-trips their float geometry exactly through the selection manifest.
   `spans_json` is a persisted recovery format: decoding ignores span fields a later version no longer has, and a
   field added to `SpanWorkItem` needs a default so saved selections still resume.

No source media is touched during discovery. A listing failure, or an input root with no shard, fails discovery
instead of reading as an empty dataset. Overlapping inputs, such as a root listed with one of its shards, discover
each shard once, and a repeated `clip_id` fails discovery.

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

`cut_chunk` handles all spans of one work item, the `(chunk_mp4_uri, data_parquet_uri)` pair, and returns one
terminal chunk outcome:

```text
cut_chunk(work item: chunk_id, chunk_mp4_uri, data_parquet_uri, spans_json)
  │
  ├─ read_bytes(data_parquet_uri)        missing/archived → chunk error "load"
  ├─ parse parquet → action_arrays       unparseable      → chunk error "parquet-parse"
  ├─ download chunk_mp4_uri to a temp file (cut_plan requires a local path), chunk_download_attempts times
  │                                      missing/archived → chunk error "chunk-stage"
  │
  ├─ cut_plan(chunk_local, cut_specs, bitrate=video_bitrate, smart_cut=True, workers=cut_cpus, on_cut=finish)
  │      builds per-frame PTS index from container packets (no decode)
  │      for each span: stream-copy interior GOPs + re-encode head → local clip MP4
  │      falls back to full re-encode if stream copy would be inexact
  │      span outside the chunk          → clip error "cut", before cut_plan runs
  │      invalid or unindexable source (MediaError) → chunk error "cut-index"
  │      inconclusive index probe (ProbeError)      → raise
  │
  └─ finish(cut record), as each cut completes on cut_plan's worker threads:
       ├─ cut the source cannot yield (CutError)    → clip error "cut"
       ├─ inconclusive run (FFmpegRunError, ProbeError) → raise (failed or killed ffmpeg, timeout,
       │                                       OS error such as a full disk, uncountable output)
       ├─ slice + encode action data        → clip error "action" on ValueError/KeyError/IndexError
       ├─ write clip, action, and sidecar   → retried media_write_attempts times, then raise
       └─ remove the local clip
```

A chunk error fails every clip the chunk owes with the same stage and message. Every other failure — storage
environment, credentials, permissions, throttling, tools, or anything unclassified — raises without an outcome,
so the chunk stays retryable (Ray Data's map retries cover transient errors first). Media failures follow the
shared contract in [`media/failures.py`](../../../cosmos_curator/next/media/failures.py): `smart_cut` reports a
`MediaError` only on positive evidence -- FFprobe's structured invalid-data error, an index that is not a frame
index, a decode that selects other frames, or a completed cut with the wrong frame count -- and everything else as
`ProbeError` or `FFmpegRunError`, which the shared runtime retries by name. A full scratch disk or a timeout under
load therefore never becomes a published clip error. Payloads are durable before the outcome is returned, and their
paths are deterministic, so a replayed chunk rewrites the same objects. Each clip is uploaded and removed as soon
as its cut finishes, so a task's local disk holds the chunk MP4 plus at most `cut_cpus` clips.

A chunk that keeps raising — for example, a deterministic failure the policy does not recognize — fails every
run that resumes its selection, and the selection's other chunks wait with it. The raised error carries a note
naming the chunk's `chunk_id` and `chunk_mp4_uri`. Fix the cause, or remove the
recovery workspace under `output.staging_root_uri` to discard the saved selection; its staged outcomes are then
recomputed by the next run.

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

Each chunk stages one terminal outcome (`CHUNK_SCHEMA`: the chunk's identity and URIs, a list of clip outcomes,
and a chunk-level `error_stage`/`error_message`). Publication expands outcomes into `CLIP_SCHEMA`, one row per
clip the chunk owes, finished or failed. A failed clip keeps its identity, labels, and geometry, with null
`clip_uri` and `action_data_uri` and a non-null `error_stage`, so every discovered clip — and therefore every
chunk — remains present in current rows.

`CLIP_SCHEMA`:

| Column | Arrow type | Non-null | Meaning |
|--------|-----------|---------|---------|
| `clip_id` | `string` | yes | Stable per-view clip identifier |
| `span_group_id` | `string` | yes | Stable id shared across all views of this span |
| `view_name` | `string` | yes | Camera view (e.g. `observation.images.wrist_image`) |
| `source_id` | `string` | yes | Hash of the normalized discovered shard URI |
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
| `clip_uri` | `large_string` | no | Written clip MP4 URI; null exactly when the clip failed |
| `action_data_uri` | `large_string` | no | Written action `.bin`/`.pickle` URI; null exactly when the clip failed |
| `camera_motion_annotation` | `large_string` | no | Computed motion description, when available |
| `chunk_id` | `string` | yes | Work item: digest of `chunk_mp4_uri` and `data_parquet_uri` |
| `chunk_mp4_uri` | `string` | yes | Source chunk MP4 the clip was cut from |
| `data_parquet_uri` | `string` | yes | Source data parquet holding the clip's episode |
| `error_stage` | `string` | no | Terminal failure stage; null for a finished clip |
| `error_message` | `string` | no | Terminal failure diagnostic, bounded to 8,192 characters |

`clip_id` and `span_group_id` are SHA-256 digests. `span_group_id` hashes
`(source_id, episode_id, subtask_index, frame_start)`. `clip_id` always hashes
`(media_contract_version, span_group_id, video_bitrate)`, plus `view_name` for
multi-view sources. `action_data_uri` is keyed by `action_id`, which hashes
`(action_contract_version, span_group_id, action_format, source_dataset)`.
`source_id` is location-based: it hashes the normalized discovered shard URI,
not the media or metadata bytes at that location. Its stability therefore
depends on the source media and span-defining Parquet/JSON metadata remaining
immutable at that versioned shard URI.
`task_description` participates in none of these identities.
`MEDIA_CONTRACT_VERSION` versions generated MP4 bytes and clip identity, not metadata-only Lance or sidecar
fields. The table no longer carries per-row schema or media version columns: the append contract binding in the
schema metadata names the result contract, which includes both contract versions.

Terminal diagnostics are ordinary rows:

```python
import lance

clips = lance.dataset("s3://example/clips/lance/clips.lance")
failed = clips.scanner(
    filter="error_stage IS NOT NULL",
    columns=["clip_id", "chunk_mp4_uri", "error_stage", "error_message"],
)
```

**Downstream — embeddings.** The embed leg
([curator-next-embeddings.md](curator-next-embeddings.md)) consumes a projection
of this table (`clip_id`, `task_name`, `subtask_name`, `clip_uri`,
`action_data_uri`, `source_dataset`) and adds its vectors as additive, nullable
per-modality column groups (`embedding_<modality>_*`) written directly onto
`clips.lance`, plus a fitted action-PCA basis. Its read contract (`EMBED_SOURCE_ROW`) already declares both
URIs nullable; the text and image modalities apply only to rows with a `clip_uri`, and action only to rows with
an `action_data_uri`, so failed clips receive no vectors and Curate never claims them.

**Downstream — captioning.** Robot-action-caption owes only rows with a non-null `clip_uri`, in both its
`selected_only` scopes; see [curator-next-robot-action-caption.md](curator-next-robot-action-caption.md).

---

## Cross-Run Recovery

`robot-action-split` uses shared append publication
([checkpoint recovery](curator-next-checkpoint-recovery.md#append-publication)), with
[video-cut](curator-next-video-cut.md) as the reference producer. The shared code owns selection, Ray Data
checkpoints, packed fragment writes, atomic publication, and workspace cleanup; this recipe owns discovery, the
processing graph, payload durability, and the terminal-error policy.

**Migration decisions.** Recorded per
[append migration decisions](curator-next-checkpoint-recovery.md#append-migration-decisions):

| Decision | `robot-action-split` |
| --- | --- |
| Work item and identity | One source chunk: one view's chunk MP4 and the data parquet holding its episodes, identified by `chunk_id`. Its spans travel in the item as `spans_json`. A chunk downloads once and `cut_plan` cuts its spans `cut_cpus` at a time; per-clip work items would download each chunk once per clip, and grouping clips back into chunks inside the plan needs a Ray Data shuffle whose `map_groups` tasks each process a whole hash partition of chunks. Rows keep `clip_id` as their own identity. |
| Durable result and payloads | One `CHUNK_SCHEMA` outcome per chunk, listing every clip. Clip MP4s, action payloads, and sidecars are written to deterministic paths before the outcome returns. Views of one span share one action payload and rewrite identical bytes. |
| Publication and visibility | Each run publishes one selection of at most `execution.max_items` chunks (default 1,000) atomically; rows are no longer visible every `clips_per_publish_batch` rows. Rerun while `remaining_items` is nonzero or null. `max_items` also bounds the selection manifest, which saves each chunk's spans. |
| Failure policy | Terminal: a missing or archived source object, unparseable data parquet, an unindexable chunk, a completed cut without the exact frames, and action data that cannot be sliced or encoded. A chunk error marks the chunk outcome, which publication counts as a terminal error, and every clip it owes. Cuts and index probes the environment failed, chunk downloads, and payload writes retry or raise; everything else raises and stays retryable. A published terminal error is not retried on a later run; retrying it needs a new output location. |
| Compatibility and ownership | The table is bound to `owner: robot-action-split`, `id_column: chunk_id`, and the split contract (split settings, `source_dataset`, `action_format`, `video_bitrate`, `output.media_root`, media and action contract versions, `behavior_version`). `media_root` is bound because validation checks that every finished clip's payload URIs sit beneath it, as video-transcode does for its clips; the table and staging locations are not. One producer appends to a table. Tables written before this migration carry no binding and are refused; use a new output location. |

**Run sequence.** `run_config` calls `run_append` with discovery as a callback:

1. Open or create the clip table and validate its contract binding.
2. Resume a saved selection without discovery, or discover chunks, look up published `chunk_id`s in current rows
   with one projected scan, and select up to `max_items` pending chunks in `chunk_id` order.
3. Start Ray, then run `cut_clips` inside the shared checkpoint scope: `scope.read()` yields one block per chunk,
   filtered by the checkpoint before any download, and `cut_chunks` runs in the same task with `cut_cpus` CPUs.
   Outcomes are validated and staged in checkpoint groups of `rows_per_checkpoint` chunks.
4. Expand staged outcomes into clip rows, pack them into `rows_per_fragment`-row fragments, and append them in one
   transaction; then remove the verified workspace.

An interrupted run repeats only chunks that were in flight or waiting in an unwritten checkpoint group; a resumed
selection keeps its saved chunks even if discovery or `max_items` changes. The run summary is the shared append
summary plus `failed_clips`, `failed_clips_by_stage`, `clips_lance_uri`, and `clips_lance_version`. Its `errors`
counts chunks that failed as a whole; `failed_clips` counts every failed clip this run published, read from the
fragments it appended, and the CLI message reports it too.
`errors.json` and `run_summary.json` are no longer written.

**Immutable source contract.** Exact rerun and recovery require the source media
and the Parquet/JSON metadata under each discovered shard URI to remain
unchanged after rows from that shard are published. Completion is per chunk: once a chunk is published, a
corrected or newly added span in that chunk is not discovered as pending work.

Publish corrected source artifacts at versioned shard URIs, normally beneath a
versioned dataset root, and write to a new `output.lance_uri` and
`output.media_root`, or completely rebuild the existing outputs. Changing only
the source root while retaining the cumulative output is insufficient: moving
the shards creates new source identities, but the old rows remain canonical
because this recipe only appends; it does not retire or replace rows
from earlier source versions.

**Ordering under `input.limit`.** `limit` caps the number of data chunk directories (`data/chunk-NNN/`) discovery
reads, counted across all input URIs and shards in a deterministic order, so a `limit`-capped smoke run selects the
same directories on every run. Every data file in a kept directory is read, and the chunk MP4s its spans reference
become work items. The cap applies before span parsing, so a smoke run skips parsing the rest of the dataset, and
therefore before dedup. LeRobot v3 stores each episode in one data file, so keeping or dropping whole directories
never splits the per-episode dedup group: a limited run publishes the same rows for its chunks as an unlimited run.

---

## Pipeline Execution

```text
run_config(config)
  │
  └─ run_append(resolve_bindings(config), discover_items, start_ray, cut_clips)
       ├─ open_target(lance_uri)                       → validate the contract binding
       ├─ resume saved selection, or discover_items    → chunk work items, minus published chunk_ids
       ├─ start_ray: storage env for outputs and sources, progress, map retries, no-progress timeout
       ├─ cut_clips: scope.read(num_cpus=cut_cpus) → map_batches(cut_chunks) → scope.write
       └─ publish: expand to clip rows → pack rows_per_fragment → one Lance append → clean workspace
```

---

## Relationship to `video-split`

| Concern | `video-split` | `robot-action-split` |
|---------|--------------|----------------------|
| Config kind | `video-split` | `robot-action-split` |
| Span provider | fixed-stride | parquet subtask |
| Cut | `transcode_span` (full re-encode) | `cut_plan` (smart cut: stream copy + head re-encode) |
| Downstream decode | `CameraSensor` | `CameraSensor` (shared) |
| Toolkit: Lance publish | shared | shared append publication |
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
- cross-run recovery is implemented per "Cross-Run Recovery" above: the recipe runs the shared append
  conformance suite and the fresh-cluster resume scenario on its real Ray Data plan, and published chunks are
  never downloaded again
