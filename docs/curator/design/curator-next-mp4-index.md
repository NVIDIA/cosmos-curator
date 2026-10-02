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
- **Two independent identities, not one.** `video_id` (location-derived) names `moov`/`moof` sidecars
  and stays stable so they're never orphaned by a location change. `session_camera_id`
  (session/camera-derived) names `frame_index`/`metadata` sidecars instead, so a relocated file for the
  same (session, camera) still resolves to the same sidecar.
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
| `session_camera_id` | Hash of `(source_session_id, camera_key)`, derived from `media_location`'s path `execution.session_id_path_depth` directories up (default 2). Two distinct `video_id`s deriving the same `session_camera_id` refuse the whole run upfront (`SessionCameraIdCollisionError`), before any Ray dispatch. A consumer holding only `source_session_id`/`camera_key` can compute the exact sidecar path a producer wrote, with no Lance query required. |
| `extraction_contract_id` | Hash of `{extraction_version, container_kind, session_id_path_depth}` -- see "Reconciliation" below. |

One row per video in `manifest.lance`
([`VIDEO_IDENTITY_SCHEMA`](../../../cosmos_curator/next/recipes/mp4_index/contracts.py)):

| Field | Nullable | Notes |
| --- | --- | --- |
| `video_id`, `media_location`, `media_size_bytes` | no | |
| `source_session_id`, `camera_key` | no | |
| `frame_index_location`, `metadata_location` | no | Per-video Arrow IPC sidecars, named by `session_camera_id`. |
| `moov_location` | no | Raw `moov` bytes, named by `video_id`. |
| `moof_table_location` | yes | Raw `moof` bytes, one row per fragment; null for a nonfragmented video. |
| `container_kind` | no | `"fragmented"`/`"nonfragmented"`, this video's own declared run shape. |
| `record_schema_version`, `extraction_contract_id` | no | See "Reconciliation" below. |

`video_id` presence is the single source of truth for "already indexed." A producer must write both
`frame_index`/`metadata` sidecars before committing the row: the Lance commit is the completion marker,
since only it is atomic.

Two per-video Arrow IPC sidecars hold what the `moov` box states about the stream --
[`FRAME_INDEX_SCHEMA`](../../../cosmos_curator/next/recipes/mp4_index/contracts.py) (one row per
sample) and `VIDEO_METADATA_SCHEMA` (one row). Both are always read wholesale for one already-identified
video, never queried across videos, so a single-video read needs only `pyarrow`, never `lance`. Failures
append to `errors.lance` (`ERROR_SCHEMA`): `video_id`, `media_location`, `error_type`, `error_message`,
`attempted_at`.

## Reconciliation, caching, and failure policy

Reconciliation keys on `video_id` alone, which is permanent once committed -- so a fix to how a video is
extracted needs a separate signal to find rows it predates.
[`ResolvedMp4IndexConfig.extraction_contract`](../../../cosmos_curator/next/recipes/mp4_index/config.py)
hashes exactly the settings that change what gets extracted for identical input bytes:

```python
{
    "extraction_version": 1,  # bumped manually when moov/moof parsing semantics change
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
caching off, treated as a cache miss rather than a video failure. `frame_index`/`metadata` are always
rebuilt regardless -- cheap once `moov`/`moof` bytes are in hand. **Known gap**: the guard trusts a
sidecar's mere existence, not whether `media_location`'s content still matches what produced it -- a
source file replaced in place between two runs would read back stale bytes.

| Outcome | Where it lands |
| --- | --- |
| `UnindexableVideoError` (box-format refusal) | `errors.lance`, per-video |
| `UnresolvableMediaLocationError` (path too shallow for session/camera derivation) | `errors.lance`, per-video |
| `FileNotFoundError` (an explicit `uris` entry that's missing -- no upfront `HEAD`) | `errors.lance`, per-video |
| Manifest table has rows from a different contract/schema version | Whole run refuses before dispatch |

Every video's own extraction is a real network read against `media_location`, and a fragmented run's
moof walk reads the whole source sequentially -- the large-source-download pattern
[`ray-data.md`](ray-data.md)'s own `curator_io` section covers. `run_config` reserves one `curator_io`
slot per actor task unconditionally.

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

`manifest.lance`, `errors.lance`, `frames/`, `metadata/`, `moov/`, and `moof/` are all written under
`output.path`. Local/shared paths are supported in place of S3, visible to every Ray worker.
