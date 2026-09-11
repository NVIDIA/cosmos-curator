# Curator Next: Video Cut

[`video_cut`](../../../cosmos_curator/next/recipes/video_cut/) plans fixed-stride video spans without creating clip media.
A planned `video-transcode` recipe will encode those spans. Together they replace the deprecated
[`video-split`](curator-next-video-split.md), whose implementation remains until both replacements are ready.

## Decision and rationale

The design separates source processing from tabular publication:

```text
frozen source set -> probe + plan each source -> checkpointed Parquet with spans[]
                  -> Arrow expansion -> packed Lance fragments -> atomic commit
                  -> workspace cleanup
```

- **Separate cutting from transcoding.** Boundaries and source references can be reused across encoding choices.
- **Recover by source.** Keeping a complete source plan together avoids reconciling separate source and span progress.
- **Pack spans independently.** Checkpoint size bounds replay; Lance fragment size controls storage layout.
- **Publish a frozen snapshot.** One atomic Lance commit publishes spans and completion together, allowing staging to be
  deleted afterward.

The tradeoff is that interrupted publication rebuilds fragments from Parquet. This repeats tabular work while reusing
checkpointed source plans, avoiding a separate publication ledger.

## Input, identity, and output contract

Inputs are explicit S3 MP4 URIs or a recursive S3 root. Discovery normalizes, sorts, and deduplicates selected URIs while
preserving literal S3 keys. Empty selections and listing failures stop before output creation.

Each URI must identify immutable content. Each output is bound to one source set and cutting contract; changing either
requires a new output location. Root inputs are listed on every invocation and must resolve to the same set. Explicit
URIs and root discovery are interchangeable when they select identical sources.

| Identity | Meaning |
| --- | --- |
| `source_id` | Hash of the normalized immutable source URI. |
| `span_id` | Hash of source ID, first video stream (`stream_index: 0`), start, and end. |
| Cutting contract | Method, integer-nanosecond settings, stream selection, and behavior version. |
| Input snapshot | Hash of the normalized, sorted source URI set, stored separately from the cutting contract. |

Span identity is independent of cutting settings. Intervals are half-open `[start_ns, end_ns)` on a zero-relative video
timeline. The source endpoint is FFprobe's stream duration, falling back to container duration.
[`next.media.spans`](../../../cosmos_curator/next/media/spans.py) defines rounding and fixed-stride tail rules.

Lance is the canonical span dataset. Rows contain source references, media metadata, cutting provenance, and span
boundaries; the v1 schemas are defined in [`contracts.py`](../../../cosmos_curator/next/recipes/video_cut/contracts.py).
Sources with empty plans or terminal errors contribute no spans, but remain included in the committed outcome counts.

One logical driver owns each output. Execution tuning is excluded from identity and recovery compatibility.

## Execution and recovery

Phase A exposes source identities through a Ray Data datasource, allowing native checkpoint filtering to skip completed
sources before media access. Each source is probed and planned in its own task, producing one Parquet row with its
metadata, spans, and outcome. `execution.probe_cpus` controls probe concurrency and defaults to `0.25` CPUs per task.
Empty plans and terminal errors are checkpointed too.

Native checkpoints use `source_id` and advance in groups controlled by `sources_per_checkpoint`. An unfinished group
can replay; setting the group size to one gives single-source durability. Plans and checkpoints are retained until
publication succeeds. The workspace path hashes the input snapshot, cutting contract, full source-result schema, output
destination, and workspace version to isolate compatible recovery state.

Phase B expands plans with Arrow and packs spans across source boundaries into Lance fragments controlled by
`spans_per_fragment`. Workers stage the fragments; the driver commits them atomically with the `video_cut.publication`
completion record in schema metadata. It records outcome counts and the error report.
Publication rejects concurrent table changes and fields owned by another producer.

**Readers must require `video_cut.publication`.** The table is created before processing, so its existence alone does
not distinguish an incomplete run from a completed run with zero spans.

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

The recipe has completed a managed Ray run on Slurm. Failure recovery is tested locally with synthetic probe metadata.

## Media access and failure policy

`probe_mode: auto` lets FFprobe read a presigned S3 URL, using HTTP ranges where possible and falling back to a temporary
download if probing fails. `probe_mode: download` always downloads first. Signed URLs are never persisted, and avoiding
a full-object transfer is not guaranteed.

- **Terminal errors:** missing objects, unsupported media after a successful download, or local FFprobe failures
  explicitly reporting `AVERROR_INVALIDDATA`. These are checkpointed and skipped on recovery; retrying them requires a
  fresh output. A nonzero probe exit alone does not establish a terminal media problem.
- **Retryable failures:** storage access problems (including archived objects and credentials), missing tools, local
  probe timeouts, process termination, unclassified probe failures, and unexpected errors. These propagate without
  terminal checkpoints, leaving the source retryable. Inconclusive probes receive up to three Ray map retries; exhaustion
  fails the attempt without publishing completion.

Terminal error details survive cleanup at `<spans_lance_uri>/video-cut/errors.jsonl`. The report is complete only when
referenced by the Lance completion record.

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
