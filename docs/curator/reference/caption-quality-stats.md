# Caption Quality Stats

## Motivation

For split-video captioning runs, `summary.json` remains the main run accounting
artifact: input counts, processed-video counts, clip counts, duration, runtime,
and token totals.

Captioning runs also need a compact structural-health view. Today the source
signals exist on per-window metadata, but consumers must scan per-clip files
under `metas/v0/` to answer run-level questions about caption status, failure
reasons, heuristic flags, and empty or sentinel text.

`caption_quality_stats.json` is a supplemental sibling artifact that provides
those counters in one file. The writer builds the counters while it already
inspects caption windows for metadata output, so the artifact does not add a
separate metadata scan.

`caption_quality_stats.json` has its own schema contract: known counter keys,
exactness invariants, and `schema_version` for contract changes. Keeping these
counters separate from `summary.json` leaves room for richer caption-quality
metrics later without expanding the main run accounting artifact.

Version 1 stores counters only. It does not store rates, scores, thresholds,
percentiles, raw caption text, or pass/fail verdicts. Consumers derive rates from
the emitted counts and denominators.

The artifact answers structural-health questions such as:

- how many caption windows succeeded, truncated, blocked, errored, or skipped
- which failure reasons were observed
- how many heuristic flags were evaluated and how many were true
- whether any OK-status windows emitted empty or sentinel caption text

These counters indicate caption structural integrity. They do not define policy,
thresholds, baseline eligibility, quality scoring, or pass/fail verdicts.

## Artifact Contract

The split-video pipeline owns the artifact schema and emission semantics.
Benchmarking, dashboards, and downstream validation tools may consume the
artifact, but they do not define its contract.

When caption generation is enabled, `--no-caption-quality-stats` is not set,
and the run is in scope, the artifact is written at the output root:

```text
{output_clip_path}/
├── summary.json
├── caption_quality_stats.json
└── ...
```

`caption_quality_stats.json` uses a stable filename and declares its schema
version in the payload. The writer must use overwrite behavior consistent with
`summary.json` when the artifact is emitted.

### Scope

Version 1 counts subject-caption windows that belong to passed/output clips. It
does not count filtered clips, filter-window captioning, enhanced captions, or
per-model dimensions. Version 1 uses the run's single configured subject-caption
field for text checks.

Operators can disable emission with `--no-caption-quality-stats`; the default is
on for in-scope runs.

Multi-camera runs use a per-session storage layout that the version 1
aggregation path does not walk, so they are outside this artifact's version 1
emission scope.

## Schema

Example shape:

```json
{
  "schema_version": 1,
  "pipeline": "split_video_pipeline",
  "caption_windows_checked": 1000,
  "caption_status_counts": {
    "success": 700,
    "truncated": 100,
    "blocked": 80,
    "error": 100,
    "skipped": 20
  },
  "caption_failure_reason_counts": {
    "exception": 60,
    "timeout": 20
  },
  "caption_quality_flags_evaluated_count": 800,
  "caption_quality_flag_counts": {
    "flag_length_outlier": 50,
    "flag_repetition": 30,
    "flag_near_duplicate": 40
  },
  "empty_caption_count": 70,
  "sentinel_caption_count": 10
}
```

### Identity Fields

| Field | Rule |
|---|---|
| `schema_version` | Integer schema version. Version 1 is the initial contract. |
| `pipeline` | Stable identifier string for the emitting pipeline. Version 1 uses `split_video_pipeline`. |

A `schema_version` change signals a contract evolution: removed or renamed keys,
changed counter semantics, or invariant tightening. Additive fields that
preserve version 1 counter meanings do not bump the version.

`pipeline` is provenance metadata, not a discriminator. Consumers should not
branch on its value. Schema differences across pipelines are versioned through
`schema_version`, not through per-pipeline code paths.

### Counters

| Field | Rule |
|---|---|
| `caption_windows_checked` | Number of in-scope subject-caption windows inspected by the writer, across passed/output clips. |
| `caption_status_counts` | Counts by `caption_status`, captured after the captioning path maps raw backend results to the known key set. Missing or `null` status maps to `skipped`. |
| `caption_failure_reason_counts` | Counts observed non-empty `caption_failure_reason` values. Failure reasons are a subset of error windows; some error windows may not have a structured reason. |
| `caption_quality_flags_evaluated_count` | Count of windows where the v1 caption-quality heuristic flag set was evaluated. |
| `caption_quality_flag_counts` | Per-flag numerator: windows where the flag value is exactly `true`. |
| `empty_caption_count` | OK-status windows whose active subject-caption text is empty or whitespace-only after stripping. |
| `sentinel_caption_count` | OK-status windows whose active subject-caption text exactly matches a canonical sentinel string after stripping. |

The shared heuristic-flag evaluated count is equivalent to the count of windows
where the known flag values are non-null. The active subject-caption text is the
value of the run's configured subject-caption field on the inspected window.

Empty and sentinel checks apply only to OK-status windows (`success` and
`truncated`). A sentinel string is non-empty, so a window cannot count as both
empty and sentinel. Backend paths that classify sentinel output as an error
before metadata writing are reflected in `caption_status_counts["error"]`, not
in `sentinel_caption_count`.

The current canonical sentinel is
`cosmos_curator.models.vllm_sentinels.VLLM_UNKNOWN_CAPTION` (`"Unknown caption"`).
The contract counts the canonical sentinel value after stripping; the symbol can
move or be retargeted without a schema change.

For OK-status windows, missing or `null` caption text counts as empty. Normal
captioning paths should prevent that state, so this is a defensive structural
check.

`caption_windows_checked` can differ from
`summary.json`'s `total_num_caption_windows`. The summary field counts
caption-bearing windows for run accounting, while this artifact counts every
in-scope subject-caption window inspected for structural integrity, including
blocked, error, skipped, and missing-status windows.

On resume runs, existing `processed_videos/` records can cause already-processed
videos to be skipped from execution but still included in summary aggregation.
In that case, `caption_quality_stats.json` describes the cumulative processed
state of the output directory, not only videos processed by the current
invocation. Version 1 does not validate that reused chunks came from identical
captioning settings.

Inputs that have no `processed_videos/` record are treated the same way as
`summary.json`: they are unprocessed inputs, contribute zero caption-quality
counters, and do not by themselves cause the artifact to be omitted. This covers
legitimate partial or list-limited runs where the summary input set is larger
than the set of videos processed by the invocation.

### Known Keys

| Map | Known version 1 keys |
|---|---|
| `caption_status_counts` | `success`, `truncated`, `blocked`, `error`, `skipped` |
| `caption_failure_reason_counts` | `exception`, `timeout` |
| `caption_quality_flag_counts` | `flag_length_outlier`, `flag_repetition`, `flag_near_duplicate` |

Version 1 emits only the known status, failure-reason, and quality-flag keys
listed above, with zero defaults when the count is absent. New keys require a
schema review.

If serialized aggregate data contains a key outside any version 1 known counter
map, the artifact is omitted rather than emitting an additive key.

## Artifact Invariants

The summary writer validates these invariants before emitting
`caption_quality_stats.json`:

- status counts sum to `caption_windows_checked`
- failure-reason counts do not exceed the `error` count
- each flag's true count is never greater than `caption_quality_flags_evaluated_count`
- empty plus sentinel counts do not exceed the `success` plus `truncated` count
- all known status, failure-reason, and quality-flag keys are present, and no
  unknown keys are present

## Emission Behavior

The writer emits `caption_quality_stats.json` by default for captioning runs
when it can build exact counters from the aggregate metadata read for summary
writing. Emission is gated on captioning mode and the
`--no-caption-quality-stats` operator setting rather than inferred from missing
per-window caption fields.

When caption generation is disabled or `--no-caption-quality-stats` is set, the
artifact is omitted silently because the run intentionally does not request
caption-quality stats. When emission is disabled or validation fails, the writer
removes any existing root `caption_quality_stats.json` at the output prefix so
file presence reflects the current summary run.

When caption-quality flags are disabled, the artifact is still emitted for
captioning runs, but no flag values are computed, so the shared evaluated count
and true counts are zero. Status, failure-reason, empty-caption,
sentinel-caption, and `caption_windows_checked` counters still apply.

Multi-camera runs are omitted with a warning.

Unprocessed inputs with no `processed_videos/` record are skipped during
aggregation, matching `summary.json`'s `processed: false` semantics. The
exactness requirement applies to inputs that do have processed-video metadata.

If the writer cannot prove that counters are exact for the processed output
metadata, it omits the artifact instead of writing partial or zero-filled
compatibility data. Concrete examples include missing or malformed chunk
aggregate data, no single configured subject-caption field, unknown serialized
status, failure-reason, or quality-flag keys outside the version 1 known keys,
and invariant failures listed above. These
exactness failures should be logged as warnings so they are distinguishable from
captioning-disabled runs.

## Aggregation Flow

The artifact follows the same aggregation direction as existing run accounting:

1. Caption stages populate per-window status, failure-reason, and quality flag
   fields.
2. The clip metadata writer inspects each passed clip's subject-caption windows
   while it already builds per-clip metadata.
3. Per-clip counters are merged into per-video chunk metadata under
   `processed_clip_chunks/`.
4. The summary writer reads processed video/chunk metadata, skips unprocessed
   input entries, and writes the run-level `caption_quality_stats.json` artifact
   beside `summary.json`.

This keeps the run-level artifact exact without adding a second scan over
`metas/v0/`.

The counters are derived from fields already written by captioning and
metadata-writing code:

- `caption_status`
- `caption_failure_reason`
- `flag_length_outlier`
- `flag_repetition`
- `flag_near_duplicate`

## Consumer Behavior

Downstream tools can read `caption_quality_stats.json` directly when they need
exact caption structural-health counters. If the artifact is missing, consumers
should treat exact run-level caption-quality stats as unavailable, not as a
healthy zero-count result.

Consumers should validate `schema_version`, known map keys, and artifact
invariants before using the counters. Derived rates should be computed outside
the artifact from the emitted counts and denominators.

## Future Extensions

Future versions may add compatible fields such as derived rates after schema
review. Additive fields should preserve the version 1 counter meanings.

If future heuristic flags no longer share one applicability rule, add per-flag
evaluated denominators without replacing `caption_quality_flags_evaluated_count`.
