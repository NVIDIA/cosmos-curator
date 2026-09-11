# Curator Next: Video Split

**Status:** Deprecated in favor of [`video-cut`](curator-next-video-cut.md) and the planned `video-transcode` recipe.
Retain the existing implementation until both replacements are available; migration and removal follow separately.

## Summary

`video-split` turns source videos into fixed-stride MP4 clips. Ray Data overlaps source downloads, CPU transcoding, and
clip uploads as independent streaming operators. S3 stores clip media, Lance stores canonical clip metadata in
incrementally committed fragments, and a JSON report records only source or clip failures from the run.

Each invocation reconciles deterministic clip identities with the canonical rows already committed. It skips complete
sources without downloading them, processes only missing clips from partial sources, and commits each bounded
fragment as soon as it is written. The Lance table is both published output and cross-run split progress.

The recipe assumes one logical splitter at a time. Source and fragment work remains parallel within that splitter, but
canonical commits have one coordinator.

Config kind dispatch is exact: `video-split` selects this recipe, while `video_split` selects the deprecated Ray Data
pipeline. The similar names are temporary while the older package is being retired.

## Scope

The recipe:

- accepts MP4 sources from S3
- generates fixed-duration spans at a fixed stride, with a configurable minimum tail duration
- transcodes the first video stream with a CPU H.264 encoder
- copies the first audio stream when present
- writes each clip as a standalone MP4 to S3
- reconciles expected clip IDs with committed rows across invocations
- publishes successful clip metadata through bounded Lance fragment commits
- replaces one run-level JSON error report after processing finishes

Scene detection, GPU transcoding, durable zero-clip or source-failure outcomes, filtering, captioning, and embedding are
outside this increment.

## Inputs

A run uses exactly one input form:

- `uris`: an explicit set of S3 MP4 object URIs
- `root_uri`: an S3 prefix whose MP4 descendants are discovered recursively

Before Ray starts, the recipe normalizes and deduplicates the source selection. Recursive discovery schedules larger
objects first, with URI order breaking size ties; explicit selections remain URI-ordered. Scheduling order does not
affect clip identity.

A normalized URI is assumed to identify immutable content. Replacing an object at the same URI is outside the current
contract. An empty selection fails before table bootstrap or processing so a mistyped root cannot change the canonical
table or replace a previous error report.

## Outputs

Media and metadata remain separate:

- clip MP4s live at `<media_root>/clips/<clip_id>.mp4`
- `<media_root>/lance` contains one row per successfully uploaded clip
- `errors.json` contains an array of source and clip errors from the latest completed run; a clean run writes `[]`

Each clip row contains the source URI and ID, requested span, deterministic clip identity and URI, source media
properties, and output media properties. Clip identity depends only on source URI, requested span, and the media
contract. It does not depend on run order, worker placement, or execution tuning.

The Lance table is append-only for splitting. Omitting a previously processed source from a later input selection does
not delete its clips. After validating a nonempty input selection, the splitter creates an absent table with the
splitting-owned schema and zero rows, or validates the splitting-owned fields of an existing table. Later curation
streams may add nullable fields, and subsequent split fragments begin with those fields null.

The error report is operational rather than a canonical downstream dataset. Source errors have `scope: "source"` and
null clip fields. Errors after planning have `scope: "clip"` and identify the affected clip and span. Successful sources
do not emit a terminal source record; valid sources that plan zero clips are silent. Because canonical fragments commit
before the report is replaced, a failed run may leave the previous report beside newer canonical clip rows.

## Dataflow

```text
S3 source selection
        |
        v
schema-only Lance create or schema validation
        |
        v
normalized sources + committed Lance clip rows
        |
        v
source reconciliation
  complete source -> skip without download
  known partial   -> reuse metadata, plan missing clips
  unknown source  -> plan after download + probe
        |
        v
source download                           0.25 CPU + one curator_io slot
        |
        v
source flat_map generator                 configured transcode CPUs
  FFmpeg batch -> clip file -> bytes -> yield one clip/error at a time
        |
        v
clip upload map                           0.25 CPU + one curator_io slot
  upload bytes -> drop bytes -> emit clip metadata/error
        |
        v
streaming metadata repartition
        |
        +---------------------------+
        |                           |
        v                           v
worker-written Lance fragments      streamed JSON error rows
        |                           |
        v                           |
atomic Append per fragment           |
        |                           |
        +-------------> replace errors.json
```

Before Ray starts, the driver opens the canonical Lance table. If it is absent, the driver atomically creates it with
the splitting-owned schema and zero rows using create-only semantics. If another creator wins that race, the driver
reopens and validates the resulting table rather than overwriting it. An existing table may contain additional nullable
curation fields; the splitter validates its own fields and preserves the rest. A schema-only table remains valid if the
run later fails or produces no successful clips.

Reconciliation happens before expensive media work. If a source has committed clips, their duplicated source metadata
is sufficient to reconstruct the current fixed-stride plan and expected clip IDs. A complete source is omitted from the
Ray input. A partial source reuses that metadata, downloads the source once, and sends only missing spans to
transcoding. A source with no committed clip row follows the full download, probe, and plan path.

The download stage hands one source byte payload to one source-level transcode task. That task writes one temporary
source file and reuses it for all missing planned spans. Spans are processed in bounded multi-output FFmpeg batches. A
failed batch is retried clip-by-clip so one damaged span does not discard successful siblings.

The transcode function is a generator. It reads and yields one completed clip at a time rather than building a list for
the source. Ray's byte-sized output blocks and streaming executor backpressure bound the clip payloads in the object
store. Independent upload tasks consume those blocks while other source tasks continue transcoding. Upload removes the
payload column before metadata reaches publication.

Publication uses Ray's streaming row-count repartition to coalesce terminal metadata into bounded batches without an
all-to-all shuffle. Workers write uncommitted Lance fragments and return fragment descriptors plus individual JSON error
records. The driver consumes those results incrementally and commits each fragment descriptor with `Append` as it
arrives. This includes the first fragment and fragments from future invocations. When the stream finishes cleanly, a
smaller final fragment is committed rather than held for a later run. Error records participate in the repartition
boundary, so a fragment can also be smaller than the configured batch size when its batch contains sparse failures.

The driver streams errors into a temporary file while fragments commit, then replaces `errors.json`, including an empty
array for a clean run. Error reporting does not delay or roll back canonical fragment visibility.

## Failure Semantics

Expected source and media failures become data records:

- `source-read`: a source could not be downloaded after retries
- `source-probe`: FFprobe identified invalid or unsupported media in the downloaded source
- `transcode`: FFmpeg could not produce one planned clip
- `clip-probe`: FFprobe identified invalid or unsupported media in one produced clip

Source and clip probes use `execution.probe_attempts`. Clip-probe retries inspect the same encoded file without
repeating the transcode.

Unexpected exceptions, missing worker tools, inconclusive probes after retries, invalid internal records, exhausted
clip-upload retries, Lance failures, and error-report upload failures fail the task or run. This prevents a broken
environment or shared destination from being reported as a large collection of bad media.

Successful clip uploads use deterministic destinations and tolerate Ray task replay by replacement. Media is durable
before its metadata row becomes canonical. If a run fails after upload but before its fragment commits, the MP4 or an
uncommitted Lance data file may remain invisible to canonical readers. Reconciliation still sees the clip row as
missing, and a later run safely recreates or reuses the deterministic media location.

Each Lance fragment transaction is an independent canonical reader-visibility and recovery boundary. Earlier fragment
commits survive a later run failure. Before each append, and again after an ambiguous commit response, the splitter
checks the candidate fragment's clip IDs against the latest table: all present means the descriptor was already
published, none present permits the append or retry, and partial presence violates the atomic single-writer protocol
and fails loudly instead of blindly appending duplicates.

`errors.json` is written after the Ray stream finishes. If that write fails, the run fails even though any committed
fragments remain visible. A rerun reconciles those rows and repairs the report.

## Configuration

```yaml
schema_version: 1
kind: video-split
input:
  root_uri: s3://example-bucket/raw
output:
  media_root: s3://example-bucket/curated/video-split
```

`clips_lance_uri` defaults to `<media_root>/lance`, and `errors_uri` defaults to
`<media_root>/errors.json`. The Lance destination may be a driver-local path for local runs; media and errors
remain on S3.

Execution tuning controls CPU reservations, FFmpeg batch size and threads, retry/timeout behavior, publication batch
size, and progress output. `clips_per_publish_batch` bounds terminal records; sparse errors count toward that boundary,
so it is an upper bound rather than an exact clip-fragment size. Execution settings do not participate in clip identity.

## Cross-Run Recovery

The recovery protocol follows [Curator Next Incremental Curation](curator-next-incremental-curation.md). Video splitting
is the sole producer of rows and fragments in the canonical Lance clip table, and there is one logical splitter at a
time.

At the start of an invocation, the splitter reads committed clip IDs and source metadata for the selected normalized
source URIs. For any known source, it reconstructs the current fixed-stride plan from the stored duration and derives
the expected clip IDs using the current media contract. If every expected ID is present, it skips the source without a
download. If some are missing, it downloads the source once and processes only those spans. An unknown source is
downloaded and probed before planning.

This is clip-level recovery even though download and transcode execution remain source-level. A source may cross a
fragment boundary: clips in the committed fragment are skipped, while uncommitted siblings are recreated. The
deterministic media destination makes an upload without a metadata commit safe to replay.

The initial implementation re-evaluates sources that produce no clips or fail before producing a clip because there is
no canonical clip row from which to recover their plan. A small durable source-outcome table can be added later if that
work becomes significant. Captioning, embedding, and other later curation preserve the committed rows and fragment
boundaries while adding nullable columns of their own.

## Validation

The implementation must demonstrate:

- deterministic identities and canonical clip schema
- an absent destination being bootstrapped as a zero-row table with the canonical clip schema
- every clip fragment appending, including the first one, and a clean smaller final fragment
- one source download reused by all of its missing clip transcodes
- a fully committed source being skipped without a download
- a source crossing a fragment commit boundary and a restart recreating only its missing clips
- repeated runs leaving exactly one canonical row per expected `clip_id`
- uploaded media without a metadata commit being safe to replay
- ambiguous fragment commits being resolved from candidate clip IDs before retry
- clips appended after nullable enrichment fields are registered beginning with null values
- generator fan-out before a long source finishes
- separate download, transcode, and upload Ray Data operators
- streaming bounded fragment publication and error-report collection
- successful sibling clips surviving an isolated media failure
- a replaced JSON report after each successfully completed run
- an empty source selection leaving prior outputs untouched
