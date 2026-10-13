# Curator Next: MP4 Index

[`mp4-index`](../../../cosmos_curator/next/recipes/mp4_index/) discovers MP4 video files (S3, local, or
`file://`) and indexes them directly, without consuming another recipe's output -- there is no upstream
recipe that already knows where these videos live. Box parsing is
[a separate Sensor Library component](sensor-library-mp4-parsing.md); this recipe owns discovery,
extraction orchestration, and publication only.

## Decision and rationale

- **Container kind is declared per run, not detected per video.** A real `moov` can carry a genuine
  `mvex` child without the file actually being fragmented (measured against production data), so byte
  presence alone is not a safe per-video signal. A wrong declaration is not a clean per-video error: a
  fragmented walk over a non-fragmented file buffers unboundedly past `mdat`. Per-file detection is
  tracked separately; an earlier attempt against real corpus data surfaced complications needing
  dedicated investigation, not a quick fix folded into this recipe.
- **Video payloads are keyed by source identity.** `video_id` (location-derived) names the
  raw `moov`/`moof` extraction caches. `session_camera_id` remains a queryable logical key;
  this recipe continues to refuse ambiguous ownership of one session/camera across videos.
- **Metadata and frame indexes are canonical Lance fields.** One video row contains
  `metadata: struct<…>` and `frames: large_list<struct<…>>`, published together with identity
  and provenance. There is no derived metadata/index sidecar.
- **Extraction-affecting settings are hashed into one contract id**, stamped on every manifest row and
  checked whole-table before reconciliation starts -- refusing a run against a stale-contract table
  rather than silently skipping or reprocessing it.

## Input, identity, and output contract

`input` specifies `uris` (an explicit list, taken verbatim -- no listing, no `limit`) xor
`path_prefix`+`globs` (scanned recursively). `file://` is normalized to a bare local path at config
resolution; discovery and extraction only ever see `s3://` or a plain local path.

`path_prefix` discovery assumes many session "directories", each independently listable, with
video files at arbitrary depth within one. A single recursive listing across the whole prefix
would walk that as one long serial chain of round trips before a single video is even touched, so
over `s3://` the session level is listed first (one cheap call) and each session's own subtree is
walked concurrently -- capped at 8 workers, measured against a real ~10,000-session bucket: 32
concurrent listings produced a `429` (Too Many Requests), 8 completed cleanly. Every other backend
falls back to a single full listing.

| Identity | Meaning |
| --- | --- |
| `video_id` | Hash of the normalized `media_location`. Deliberately stable: changing it would orphan every already-extracted `moov`/`moof` sidecar on disk. |
| `session_camera_id` | Hash of `(source_session_id, camera_key)`, derived from `media_location`'s path `execution.session_id_path_depth` directories up (default 2). Two distinct `video_id`s deriving the same `session_camera_id` refuse the whole run upfront (`SessionCameraIdCollisionError`), before any Ray dispatch. Consumers can query this logical key in Lance. |
| `extraction_contract_id` | Hash of `{kind, extraction_version, container_kind, session_id_path_depth}` -- see "Reconciliation" below. |

One row per video in `manifest.lance`
([`VIDEO_MANIFEST_SCHEMA`](../../../cosmos_curator/next/recipes/mp4_index/contracts.py)):

The record schema version is 6. Version 5 lacks the nullable EOF/SOF/MOE frame fields;
existing version-5 datasets require a fresh output dataset. Moov/moof caches can be reused.

| Field | Nullable | Notes |
| --- | --- | --- |
| `video_id`, `media_location`, `media_size_bytes` | no | |
| `source_session_id`, `camera_key` | no | |
| `source_width`, `source_height`, `source_frame_rate`, `source_frame_count`, `source_duration_ns`, `source_video_codec` | no | Summary fields derived from the same metadata object, named to match `video_cut`. Queries can project these columns without loading frame arrays. |
| `metadata` | no | Struct containing every `VIDEO_METADATA_SCHEMA` field, including extradata and edit-list bytes. |
| `frames` | no | Large list of non-null `FRAME_INDEX_SCHEMA` structs, in decode order. Composition offsets may be all null when absent from the source. EOF/SOF/MOE timestamps are nullable int64 nanoseconds. |
| `moov_location` | no | Raw `moov` bytes, named by `video_id`. |
| `moof_table_location` | yes | Raw `moof` bytes, one row per fragment; null for a nonfragmented video. |
| `container_kind` | no | `"fragmented"`/`"nonfragmented"`, this video's own declared run shape. |
| `record_schema_version`, `extraction_contract_id` | no | See "Reconciliation" below. |

`video_id` presence is the single source of truth for "already indexed." Workers stage complete
video records as Lance fragments; the driver receives only fragment descriptors, counts, and
failures, and appends each fragment atomically. Metadata, frame index, summaries, and provenance
become visible under the same Lance commit. Uncommitted work can be recomputed from the raw-box
caches without fetching the source again.

Extraction actors construct frame columns directly from NumPy arrays and return typed Arrow
batches. No Python dictionary per frame or nested IPC schema-metadata blob is created. Publication
batches retain the existing default of 1,000 videos; tune
`execution.videos_per_fragment` and actor memory for the corpus. This is a row-count bound, not a
hard byte bound: exceptionally long recordings may require smaller batches, and one complete
video index must fit in memory.

A consumer can project summaries for selection, then read selected `metadata`/`frames` fields
in distributed input workers. `read_video_record(table, row_index=…)` unwraps the nested frame
list into an ordinary frame table while reusing Arrow value buffers. The reader requires each
frame column's declared type and rejects mismatches without casting. `Mp4Metadata.from_row` and
`Mp4FrameIndex.from_frame_index_table` remain independent of Lance. Arrow unwrapping does not
eliminate allocations for subsequent NumPy conversion or derived timestamp calculations.

The ordinary `mp4-index` recipe leaves EOF/SOF/MOE fields null. Extensions can supply timing
columns within the fixed shared frame schema; recorder-specific requirements and sentinel
normalization belong in the extension. The shared reader preserves partially missing timing as
masked `int64` arrays (`True` marks missing timing), fully populated columns as ordinary NumPy
arrays, and entirely missing columns as `None`. The sensor library checks dtype and frame count
without requiring EOF when SOF or MOE is present.

Raw `moof` Arrow caches are still fetched wholesale before IPC parsing; they are extraction caches,
not the canonical metadata/index output. Source media remains in its original storage.

Failures append to `errors.lance` (`ERROR_SCHEMA`): `video_id`, `media_location`, `error_type`,
`error_message`, `attempted_at`.

## Reconciliation, caching, and failure policy

Reconciliation keys on `video_id` alone, which is permanent once committed -- so a fix to how a video is
extracted needs a separate signal to find rows it predates.
[`ResolvedMp4IndexConfig.extraction_contract`](../../../cosmos_curator/next/recipes/mp4_index/config.py)
hashes exactly the settings that change what gets extracted for identical input bytes:

```python
{
    "kind": config.kind,
    "extraction_version": 1,  # bumped manually when extraction semantics change
    "container_kind": config.input.container_kind,
    "session_id_path_depth": config.execution.session_id_path_depth,
}
```

Interpretation-layer behavior (e.g. edit-list application) is deliberately excluded: that code runs at
read time against already-persisted undecoded fields, so a fix applies to every existing sidecar on its
next read, with no version bump or re-extraction needed. Execution-only tuning (`chunk_bytes`,
`skip_existing_sidecars`, actor pool sizing) is excluded too -- it never changes the output.

Before reconciliation, [`lance_sink.validate_committed_versions`](../../../cosmos_curator/next/recipes/mp4_index/lance_sink.py)
scans the whole manifest table (not just this run's own candidates -- a row for a video outside this
run's set would otherwise go unchecked forever) and raises, naming how many rows are stale, if any row's
`record_schema_version`/`extraction_contract_id` doesn't match the current run's. Fixing stale rows --
recreating the table and rerunning, or a scoped backfill -- is a deliberate, separate action; a normal
run never attempts it automatically.

Before fetching/walking a video's `moov`/`moof` (`execution.skip_existing_sidecars`, default on), the
deterministic sidecar path is checked first and read back instead of refetched, atomically on both
local and `s3://` backends. A cache entry that writes completely but fails to parse retries once with
caching off, treated as a cache miss rather than a video failure. For uncommitted videos, the metadata/index record is always
rebuilt -- cheap once `moov`/`moof` bytes are in hand. **Known gap**: the guard trusts a
sidecar's mere existence, not whether `media_location`'s content still matches what produced it -- a
source file replaced in place between two runs would read back stale bytes.

| Outcome | Where it lands |
| --- | --- |
| `UnindexableVideoError` (box-format refusal) | `errors.lance`, per-video |
| `UnresolvableMediaLocationError` (path too shallow for session/camera derivation) | `errors.lance`, per-video |
| `FileNotFoundError` (an explicit `uris` entry that's missing -- no upfront `HEAD`) | `errors.lance`, per-video |
| Manifest table has rows from a different contract/schema version | Whole run refuses before dispatch |
| Extension contract violations or unexpected extension errors | Whole run fails; no per-video error row |

Before assembling a successful video record, the recipe checks every frame's byte range against
the source file size. Negative offsets or sizes and ranges extending past EOF produce an
`InvalidFrameByteRangeError` per-video failure without decoding frames.

Extensions report malformed source data through explicit `UnindexableVideoError` subclasses.
Refusals from extension construction or handler registration become per-video failures without
cache recovery. Refusals after parsing cached boxes may trigger the same one-time refetch as
standard MP4 parse failures, with a fresh extension for the retry.
Unknown or duplicate columns, schema/type mismatches, and incorrectly shaped extension batches fail the run
without retrying cached boxes or adding the video to `errors.lance`. Previously recorded failures can
be retried with `--set execution.skip_known_failures=false` after fixing the extension.

Every video's own extraction is a real network read against `media_location`, and a fragmented run's
moof walk reads the whole source sequentially -- the large-source-download pattern
[`ray-data.md`](ray-data.md)'s own `curator_io` section covers. `run_config` reserves one `curator_io`
slot per actor task unconditionally.

Indexing actors retain MP4 boxes and frame arrays, stream source reads when needed, and do not
keep whole videos or decoded frames in memory. Observed actor memory use is about 300–400 MB,
so this recipe can support a higher IO budget than stages that buffer full videos. For a new local
Ray node, the slot budget defaults to the process's available CPU count;
`COSMOS_CURATOR_RAY_IO_SLOTS_PER_NODE` overrides it. Existing clusters retain their launch-time
resource budgets. Indexing actors reserve 0.5 GiB by default, adjustable through
`execution.actor_memory_gb`.

## Running

```bash
cosmos-curator pipeline template mp4-index > mp4-index.yaml
cosmos-curator pipeline validate mp4-index.yaml
pixi run --as-is run-pipeline mp4-index.yaml
```

```yaml
schema_version: 1
kind: mp4-index
input:
  uris: [s3://example-bucket/raw/example.mp4]
  container_kind: nonfragmented
output:
  path: s3://example-bucket/mp4-index/
```

`manifest.lance`, `errors.lance`, `moov/`, and `moof/` are all written under
`output.path`. Local/shared paths are supported in place of S3, visible to every Ray worker.

Older manifest layouts require a fresh output dataset. Existing moov/moof caches
remain reusable. No standalone Arrow export is included.
