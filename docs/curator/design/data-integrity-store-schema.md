# Data-Integrity Store — Lance Schema

**Status:** built, and this doc is its reference. Persistence only — no new metric math and no curation filter; the
thresholds it names are the policy the metrics already carry.

The DI CLIs (`di-check` for one stream, `di-session` for a whole session) and the Ray pipeline run the checks and
print the results; given `--store-path` they also save them. This doc is where those results go, and why.

---

## 1. The idea

Two different kinds of thing come out of a run:

- A **measurement** is a *fact* about the data. "There were 3 backward timestamps." That fact never changes, because
  the recording never changes. It only goes wrong if we change the *code that measures it*, or if the bytes behind
  the path get replaced.
- An **evaluation** is a *judgment* about that fact. "3 backward timestamps is too many → FAIL." That changes
  whenever we change our thresholds.

So we store them **separately**. That way, when someone tightens a threshold next month, we re-judge the saved facts
and get new verdicts — without re-opening a single file. That's the main point of this work.

There's a third kind of table for a boring but important reason: if a file is corrupt and won't open, it produces *no
measurements at all*. If we only stored measurements, that broken stream would just vanish from the store — and a
broken stream is exactly the thing curation most needs to know about. So we keep a row per stream too.

### Four kinds of table

| Table | One row per | What it holds |
|---|---|---|
| `stream.lance` | stream | facts about the data source itself, plus "couldn't open it" |
| `measurements/<metric>.lance` | stream | the measured facts — **one table per metric** |
| `evaluation.lance` | policy × stream × metric | the verdicts |
| `run.lance` | finished run | the commit marker, plus who wrote it and under what policy |

The first three have a session-grain twin each, for the metrics whose subject is a whole session rather than one
stream: `session.lance`, `session_measurements/<metric>.lance` and `session_evaluation.lance`, carrying `session_id`
where the original carries `stream_id`. They are additive — everything in sections 1–4 keeps its shape, and a reader
that does not care about sessions never learns they exist. Section 5 is their spec.

```text
<store-root>/                 # a local dir or an s3:// path
├── stream.lance
├── measurements/
│   ├── timestamp_ordering.lance
│   ├── rate.lance
│   ├── timestamp_gap.lance
│   ├── jitter.lance
│   └── frame_reordering_present.lance
├── evaluation.lance
├── session.lance             # session-grain; absent from a store di-check wrote
├── session_measurements/
│   ├── multi_sensor_spread.lance
│   └── multi_sensor_overlap.lance
├── session_evaluation.lance
├── run.lance                 # written last; a run's rows are invisible until its row lands
└── manifest.json             # small provenance file: run id, timestamps, thresholds used, metric versions
```

Nothing here is video-specific by design. The metric kernel only ever sees timestamps (`NDArray[np.int64]`), so IMU,
GPS and lidar streams fit these same tables; `codec_name` and `has_bframes` are the only video-shaped columns, and
they're nullable for exactly that reason.

### Append-only

Tables are **append-only** — we never edit rows in place. A re-run or a re-judge adds rows; readers asking "what's
the current state?" take the newest row per key. Keeps history, keeps it auditable.

### One run, several tables — so a run needs a commit point

A write touches `stream.lance`, five metric tables and `evaluation.lance` in sequence — and, for a session run, the
three session-grain tables beside them. Kill it halfway and the store holds a *newer* `run_id` for a run that never
finished — which "newest row per key" would happily serve as current state.

So `run.lance` gets its row **after** every other write, and readers drop rows whose `run_id` has no row there. A
torn write leaves rows that are present but not believed; the previous complete run stays current. The filter runs
*before* the newest-per-key pass, otherwise the half-written row would win the sort and then be dropped, leaving the
key empty. Re-evaluation commits the same way, so a partial generation of verdicts can't appear either.

`manifest.json` is *not* the commit point. It is overwritten on every run, so it only ever describes the latest one;
`run.lance` is the history, and readers rely on it rather than on a file that a reader may find half-written.

Its fields split two ways, and the split is worth knowing before reading one. `run_id`, `created_at`, `tool`,
`session_path`, `num_streams` and the policy fields describe *that run*. `store_schema_version`, `datasets` and
`instruments` describe *the layout and registry of the build that wrote it* — every dataset the schema defines and
every metric it knows, not the subset whose rows the run happened to produce. So a `di-check` run lists the
session datasets it could not have written, and a run whose only stream failed to open lists `measurements` while
writing no measurement row. That is the useful way round for an overwritten file: the alternative is a store
`di-session` filled and `di-check` later wrote to ending up with a manifest that denies the `session.lance` sitting
next to it. What a *particular* run wrote is answered by `run_id` and the rows.

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `run_id` | `string` | no | the run this row completes |
| `created_at` | `timestamp[us, UTC]` | no | the stamp every row of the run carries |
| `committed_at` | `timestamp[us, UTC]` | no | when the run finished writing |
| `tool` | `string` | no | `di-check` \| `di-session` \| `di-reevaluate` \| `data-integrity` (the Ray recipe) |
| `session_path` | `string` | yes | `null` for a single-video run and for a re-judge |
| `policy_id` | `string` | no | the policy the run's verdicts were produced under |
| `num_streams` | `int64` | yes | `null` for a re-judge, which touches no stream |
| `store_schema_version` | `int32` | no | layout version at write time |

### Layout versions

`store_schema_version` counts changes to the store *as a whole* — a dataset added, moved or renamed. A change inside
one metric's schema needs no number here, because Lance records that table's own schema.

| Version | Change | Reads an older store? | Takes new rows in an older store? |
|---|---|---|---|
| 1 | the original per-stream tables | — | — |
| 2 | added the three session-grain tables | yes | yes |
| 3 | added `timestamp_source` to `stream.lance` and to `session.lance` | yes | **no** |

Version 2 was additive: a dataset that was never written reads as no rows, so nothing had to change. Version 3 is
not. Appends go through `lance.write_dataset(..., mode="append")`, which requires the incoming schema to match the
dataset's, so a root written at version 1 or 2 rejects new `stream.lance` or `session.lance` rows and needs a fresh
one. That is a deliberate choice rather than an oversight: migration machinery for a store whose only readers are the
two CLIs and the rollup would cost more than re-measuring, and re-measuring is cheap because the sources have not
changed.

### Dedup keys

"Newest per key" only works if everyone agrees on the key, so it's defined per table:

| Table | Dedup key | Order by |
|---|---|---|
| `stream.lance` | (`stream_id`) | `created_at` DESC, then `run_id` DESC |
| `measurements/<metric>.lance` | (`stream_id`) | `created_at` DESC, then `run_id` DESC |
| `evaluation.lance` | (`stream_id`, `metric_name`, `policy_id`) | `created_at` DESC, then `run_id` DESC |
| `session.lance` | (`session_id`) | `created_at` DESC, then `run_id` DESC |
| `session_measurements/<metric>.lance` | (`session_id`) | `created_at` DESC, then `run_id` DESC |
| `session_evaluation.lance` | (`session_id`, `metric_name`, `policy_id`) | `created_at` DESC, then `run_id` DESC |
| `run.lance` | — | `committed_at`, then `run_id` |

The session keys are the per-stream ones with `session_id` substituted, which is the other half of why the two
grains are separate tables: one table holding both would carry two rules for picking its newest row.

`run.lance` has no dedup key: a run is written once and never superseded, so every row stands.

Metric tables don't need `metric_name` in the key — the table *is* the metric.

The `run_id` tie-break matters because every row written by one invocation shares the same timestamp, so timestamps
alone don't give a total order. `run_id` is a UUID, so DESC is arbitrary — but it is *deterministic*, which is the
point: two readers de-duplicating the same data must land on the same row.

Note that `instrument_version` is deliberately **not** in the measurement key: re-measuring with a newer instrument
should supersede the old row, not sit beside it.

A single-video run and a whole-session run write to the *same* tables. The only difference is that `session_path` is
`null` for a single-video run.

Throughout this doc, `null` means a genuine Arrow null, never an empty string.

### Stream identity

A "stream" is not always a file. One MCAP holds many topics, each of which is its own timeline, so a stream is
addressed by a **file plus a selector**:

| Column | Example (video) | Example (MCAP) |
|---|---|---|
| `selector_type` | `video_stream` | `mcap_topic` |
| `selector_value` | `0` | `/camera/front` |

`stream_id` is a deterministic hash of the normalized `(source, selector_type, selector_value)` triple. `source` is
demoted to a display column — readable, but not the key.

Deriving the id from the *absolute* source rather than from `(session_id, relative_key, ...)` is deliberate, even
though the two are equivalent (a session root plus a relative key reconstructs the source). The absolute form is
invariant to **how the stream was discovered**: check one video directly with `di-check`, then check its whole
session with `di-session`, and both runs produce the same `stream_id` instead of two rows for the same bytes.

For a plain video file there is one stream to address, so `selector_value` is the `--stream-idx` that was opened —
`0` unless someone asks for another. It's part of the hash rather than a display column because two indices of one
file are two timelines: without it, `di-check --stream-idx 1` would overwrite the rows written for index `0`.

Normalization is narrow on purpose: lowercase the URI scheme, strip trailing slashes, and make local paths absolute
(which folds `.` / `..`). Paths and S3 keys are case-sensitive, so case is never touched below the scheme.

Nothing *inside* a cloud object key is rewritten. A key is an opaque string, not a path: `s3://b/a/../c.mp4`,
`s3://b//c.mp4` and `s3://b/c.mp4` are three different objects, and folding them together would merge distinct
streams into one row. Local paths are different — there `..` really does mean the parent directory — so they keep
being folded. The trailing slash is the one thing stripped from a key, because a prefix gets written both ways by
hand and `s3://b/clips/` and `s3://b/clips` are the same session.

`session_id`, `locator_namespace` and `relative_key` ride along on `stream.lance` as descriptive columns. They were
chosen so that a `session.lance` would be a pure projection of columns that already existed, and that is how it
turned out: section 5's table needed no backfill, and `session_id` on the stream rows is what lets a rollup count
how much of a session was measured without joining on a path.

---

## 2. `stream.lance`

One row per stream a run touched, **including ones that failed to open**.

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `stream_id` | `string` | no | the dedup key; see above |
| `run_id` | `string` | no | one per CLI invocation |
| `created_at` | `timestamp[us, UTC]` | no | |
| `session_id` | `string` | yes | hash of `locator_namespace` + normalized `session_path` |
| `session_path` | `string` | **yes** | the session dir; `null` for a single-video run |
| `locator_namespace` | `string` | no | `local` \| `s3` \| `az` |
| `source` | `string` | no | full path or URI, for humans |
| `relative_key` | `string` | yes | normalized path within the session; `null` for a single-video run |
| `selector_type` | `string` | no | `video_stream` \| `mcap_topic` |
| `selector_value` | `string` | no | `0` for a single-video-stream file |
| `content_etag` | `string` | yes | opaque change token; see below |
| `content_size_bytes` | `int64` | yes | |
| `content_last_modified` | `timestamp[us, UTC]` | yes | |
| `codec_name` | `string` | yes | e.g. `h264` |
| `has_bframes` | `bool` | yes | frame-reordering flag from the file header |
| `num_samples` | `int64` | yes | how many timestamps we read |
| `start_ns` / `end_ns` | `int64` | yes | first / last timestamp |
| `expected_hz` | `float64` | yes | the rate we judged against |
| `expected_hz_source` | `string` | yes | `user` \| `header` \| `unavailable` |
| `timestamp_source` | `string` | yes | `capture` \| `container_pts` — which clock the timing columns were measured on |
| `error` | `string` | yes | if set, the source couldn't be read and has no measurements |

`expected_hz` is worked out per stream and saved because it explains the results — if it's `unavailable`, that's
*why* the rate/gap/jitter checks got skipped.

`timestamp_source` is saved for a stronger reason: it can't be recovered later. The row keeps the numbers, not the
sidecar that explains them, and a stored jitter of nearly zero is ambiguous without it — a rig that is genuinely
steady and a constant-frame-rate mux that regenerated the timeline both look like that. Only one is a measurement.
A query for streams whose cadence is real filters on this column. See
[Timestamp provenance](data-integrity-metrics.md#timestamp-provenance).

### Content identity is free

The three `content_*` columns cost **zero extra requests**. The progress display already issues one `HEAD` (S3) /
`get_blob_properties` (Azure) per stream, and that response carries `ETag` and `LastModified` right next to the
`ContentLength` it wants — so `storage_io.object_stat` returns all three and the size-only caller keeps a thin wrapper
over it. Local files get size and mtime from a `stat()`; `content_etag` stays `null`.

Recording them means re-evaluation can ask *"did the bytes change?"* instead of trusting that a path still points at
the same data.

Treat `content_etag` as an **opaque change token, not a checksum**. S3 returns an MD5 for single-part uploads but a
`<hash>-<part-count>` composite for multipart ones, so two objects with identical bytes can have different ETags if
they were uploaded differently. A *changed* ETag reliably means "something happened"; an unchanged one plus an
unchanged size is good evidence nothing did.

---

## 3. Measurement tables — one per metric

Each metric gets its own Lance dataset with its measured fields as **native typed columns**. These are the tables we
never want to have to rebuild.

Every metric table starts with the same identity prefix:

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `stream_id` | `string` | no | the dedup key |
| `run_id` | `string` | no | links back to `stream.lance` |
| `created_at` | `timestamp[us, UTC]` | no | |
| `session_path` | `string` | yes | |
| `source` | `string` | no | for humans |
| `instrument_version` | `int32` | no | version of the code that measured this |
| `is_defined` | `bool` | **yes — 3 states** | see below |

...then the metric's own fields, lifted straight off the frozen `attrs` measurement class:

| Table | Typed columns |
|---|---|
| `timestamp_ordering.lance` | `decreasing_count` `int64`, `duplicate_count` `int64`, `first_decreasing_index` `int64?`, `first_duplicate_index` `int64?`, `num_samples` `int64` |
| `rate.lance` | `period_deviation_percent` `float64`, `expected_hz` `float64`, `expected_period_ns` `int64`, `actual_mean_hz` `float64`, `actual_mean_period_ns` `float64`, `num_samples` `int64`, `num_intervals` `int64`, `num_filtered` `int64` |
| `timestamp_gap.lance` | `max_gap_ns` `int64`, `expected_period_ns` `int64`, `expected_hz` `float64`, `num_samples` `int64`, `num_gaps` `int64`, `first_gap_index` `int64?` |
| `jitter.lance` | `jitter_percent` `float64`, `expected_hz` `float64`, `num_samples` `int64`, `num_intervals` `int64`, `num_filtered` `int64` |
| `frame_reordering_present.lance` | `has_reordering` `bool?` |

(`?` marks nullable. Metric columns are additionally all-null on a "never ran" row — see below.)

`expected_hz_source` and `timestamp_source` live only on `stream.lance`. Both are per-stream context, not measured
facts, so metric tables reach them by joining on `stream_id` rather than carrying a copy.

### Why one table per metric

The alternative is a single tall table with the metric-specific numbers packed into a JSON string. Typed columns win
on four counts:

1. **`NaN` needs no encoding.** JSON cannot represent `NaN`, so a JSON payload has to smuggle it through as a
   sentinel string like `"__nan__"` and decode on the way back — a hand-written contract, easy to get subtly wrong,
   and the exact place where "we measured and it's undefined" silently degrades into "we didn't measure". Arrow
   float64 stores `NaN` natively as an IEEE-754 bit pattern and tracks null separately in the validity bitmap, so
   `null` (never ran), `NaN` (measured, undefined) and a real value are three distinct states for free.
2. **Re-evaluation is per-metric anyway.** Judging `rate` reads `period_deviation_percent` off a frozen
   `RateMeasurement`, not off a generic bag of fields, so re-judging has to rebuild that typed instance either way —
   and a typed row rebuilds it directly, where JSON would need a decode registry first.
3. **The numbers are queryable.** `jitter_percent > 8` is a predicate Lance can push down, not a string scan.
4. **The schema is enforced.** Renaming or re-typing a field fails at write time instead of quietly producing rows
   with a different JSON key.

The cost is more datasets — five now, maybe twenty eventually. That's bounded, and it lands mostly on writes, because
the two questions that *would* want a single table both live elsewhere: cross-metric verdicts ("what failed on this
stream?") and run completeness ("did every metric run?") are answered by `evaluation.lance`, which stays single and
tall. Metric tables are read for re-evaluation and deep inspection, both of which already know which metric they
want.

Adding a metric still costs no schema redesign — it adds a *new* dataset and leaves every existing one untouched.

### `is_defined` has three states, not two

Today the CLI prints `SKIPPED` for two situations that are actually different, and we don't want to lose that:

- `true` / `false` — the metric ran. `false` means it ran but couldn't produce a meaningful answer (e.g. only one
  timestamp, so there's no interval to look at). We still have real numbers to save.
- `null` — the metric **never ran at all**. This happens when we don't know the expected rate, so rate, gap and
  jitter are never even created. The row exists, but every metric column is `null`.

That "never ran" row is written deliberately rather than left absent, so *"we considered this metric and it couldn't
run"* stays distinguishable from *"this metric wasn't part of the run"*.

### The five metrics

| `metric_name` | What gets judged | Type | Default threshold |
|---|---|---|---|
| `timestamp_ordering` | `strict_violation_count` | int | 0 |
| `rate` | `period_deviation_percent` | float | 5.0 |
| `timestamp_gap` | `num_gaps` | int | 0 |
| `jitter` | `jitter_percent` | float | 10.0 |
| `frame_reordering_present` | `has_reordering` as 0/1 | int | 0 (i.e. not allowed) |

---

## 4. `evaluation.lance`

One row per policy × stream × metric — the one table that stays tall, because verdicts have a uniform shape no
matter which metric produced them. Re-judging writes a **new set of rows under a new `policy_id`**; the old verdicts
stay.

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `stream_id` | `string` | no | |
| `run_id` | `string` | no | the invocation that produced this verdict |
| `measurement_run_id` | `string` | no | the measuring run whose facts were judged |
| `policy_id` | `string` | no | hash of the thresholds used |
| `thresholds_json` | `string` | no | the thresholds themselves, so a verdict explains itself |
| `created_at` | `timestamp[us, UTC]` | no | |
| `session_path` | `string` | yes | |
| `source` | `string` | no | |
| `metric_name` | `string` | no | which metric table the input came from |
| `check_status` | `string` | no | `PASS` \| `FAIL` \| `SKIPPED` — the single verdict column |
| `margin` | `float64` | yes | how far from the threshold (negative = over) |
| `threshold` | `float64` | yes | the limit applied |
| `reason` | `string` | yes | the one-line summary the CLI prints |
| `instrument_version` | `int32` | no | which measuring code produced the input |

If a measurement was undefined or never taken, the row is `SKIPPED` with `margin` and `threshold` both `null`. So
it's structurally impossible to store a verdict for something that was never judged.

**There is deliberately only one verdict column.** An earlier draft also had a `status` column holding the kernel's
`PASS`/`FAIL`, but the kernel derives `CheckStatus` straight from `EvaluationStatus`, so the two could never
disagree — `status` was just `check_status` minus the `SKIPPED` case. Two columns meaning the same thing only
invites the question of which to filter on. **Filter on `check_status`.** The kernel's `EvaluationResult` still
reconstructs losslessly: `check_status` gives the status and `margin` gives the margin.

`margin` is an int for ordering/gap/reordering and a float for rate/jitter. We store `float64` and convert back on
read using the metric registry, rather than adding a column just to say which.

`metric_name` and `check_status` are plain strings rather than Arrow dictionary (categorical) columns. At five and
three distinct values the saving is negligible — Lance already compresses low-cardinality string columns — while
append-only writes produce many fragments, which is exactly where dictionaries have to be unified across fragments.
Plain strings are also what every other enum-ish column in the repo uses; `ISSUE_SCHEMA.code` is the lone exception.

### Why two run ids

`run_id` always means **"the invocation that wrote this row"**, in every table. `stream` and the metric tables are
written by the same invocation, so they share one value. `evaluation` needs a second id because a verdict points at
someone else's work: `measurement_run_id` says **which facts were judged**.

On the first pass the CLI measures *and* judges, so `run_id == measurement_run_id`. A later re-judge is a new
invocation, so they differ — which gives you a free way to separate original verdicts from re-judged ones:

```sql
WHERE run_id <> measurement_run_id   -- re-judged under a later policy
```

Join keys (distinct from the dedup keys in section 1 — these link tables, those pick the newest row):

- evaluation → its metric table on (`measurement_run_id`, `stream_id`), picking the table named by `metric_name`.
  Resolves to exactly one row, since a run measures each metric once per stream.
- evaluation → stream on (`measurement_run_id`, `stream_id`).
- metric table → stream on (`run_id`, `stream_id`).

---

## 5. Session-grain — `session.lance` and session verdicts

Sections 1–4 all assume the subject of a measurement is a stream. That was true of the first five metrics and stops
being true here: **how far apart a rig's sensors started, and how much of the session they were all recording for,
are facts about the session, not about any one camera in it.** There is no stream to hang such a row on — picking one
would blame it for the others — and `stream_id` / `source` are `nullable=False` on both the measurement identity
prefix and `evaluation.lance`, so there is nowhere to put one among them.

So the three tables below are parallel to sections 2–4 rather than folded into them. Both writers fill them: the
`di-session` CLI writes the verdicts it just printed, and the Ray pipeline writes one set per session it measured.
`reevaluate` re-judges them alongside the per-stream ones, so tightening a policy moves both grains together.

### `session.lance`

One row per session per run. Dedup on `session_id` by `(created_at DESC, run_id DESC)`, the same rule
`stream.lance` uses.

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `session_id` | `string` | no | the dedup key: hash of `locator_namespace` + normalized `session_path` |
| `run_id` | `string` | no | one per invocation |
| `created_at` | `timestamp[us, UTC]` | no | |
| `session_path` | `string` | no | the session dir / prefix |
| `locator_namespace` | `string` | no | `local` \| `s3` \| `az` |
| `num_streams` | `int64` | no | how many streams the run found under it |
| `num_unreadable` | `int64` | no | how many of those could not be opened |
| `timestamp_source` | `string` | yes | `capture` \| `container_pts` \| `mixed` — the clock rolled up across the session's streams; `null` when none contributed |
| `error` | `string` | yes | if set, the session itself could not be listed and has no streams and no measurements |

`timestamp_source` is the session-grain counterpart of the `stream.lance` column, rolled up by one rule: `capture`
only when *every* contributing stream reported it, `container_pts` only when every one did, and `mixed` when they
disagree. A stream that contributed no bounds — unreachable, or read with an empty timeline — belongs to neither
answer and is left out, so a session no stream contributed to has `null` rather than a name: it was measured on no
clock, which is not the same as a bad one.

The column exists because it is the *only* thing that can explain a skipped session-grain metric. Both session metrics
declare `requires_capture_timestamps`, so on anything but `capture` they never run, and a never-ran row is all nulls
by construction — it can say the metric did not run but not why. Writing the cause here also makes the fleet
queryable: "which sessions have real alignment numbers" is a filter on this column rather than a join against every
camera's row. `di-reevaluate` reads it back to phrase the same reason the original run gave.

`error` is the session-level analogue of `stream.error`, and it settles a question the pipeline review raised: a
stale path in a thousand-path manifest gets a row saying so, rather than costing the run every measurement already
taken.

There is deliberately **no `num_unreachable` column**. A stream that could not be *reached* — an expired token, a
403 — gets no row anywhere, for the reason in [the pipeline doc](data-integrity-pipeline.md): a failure the
environment manufactured must not be persisted as a finding about the data. So `num_streams` counts what the run
*found*, and a session with unreachable streams has more of them than it has rows in `stream.lance`.

That difference is the record of the gap, and `session_rollup` reads it: a session holding fewer stream rows than
`num_streams` rolls up `ERROR`, on the same grounds an errored stream does — nothing was measured there, so there is
nothing to call a pass. Counting rows rather than storing a shortfall is what keeps the answer true over time: a
stream one run could not reach and a later one could is measured, and the session completes without either run
having to revise a column. A stored `num_unreachable` would pin that session at `ERROR` for good, which is the
concrete form of the reason not to have it — an unreachable stream is a fact about the run, not about the session,
and the same session re-run with working credentials has a different number. The run says so too: the pipeline
raises `IncompleteRunError` after committing, and its message carries the count.

### `session_measurements/<metric>.lance`

One dataset per session-grain metric, same reasoning as section 3 — typed columns, `NaN` distinct from `null`,
`is_defined` with three states.

The identity prefix is `MEASUREMENT_IDENTITY_FIELDS` with `session_id` where `stream_id` sits, and **no `source`**:

| Column | Type | Null? | Meaning |
|---|---|---|---|
| `session_id` | `string` | no | the dedup key |
| `run_id` | `string` | no | links back to `session.lance` |
| `created_at` | `timestamp[us, UTC]` | no | |
| `session_path` | `string` | no | for humans, in place of `source` |
| `instrument_version` | `int32` | no | version of the code that measured this |
| `is_defined` | `bool` | **yes — 3 states** | as in section 3 |

...then the metric's own fields:

| Table | Typed columns |
|---|---|
| `multi_sensor_spread.lance` | `start_spread_ns` `int64`, `stop_spread_ns` `int64`, `num_sensors` `int64` |
| `multi_sensor_overlap.lance` | `overlap_fraction` `float64`, `non_overlap_fraction` `float64`, `effective_duration_ns` `int64`, `total_duration_ns` `int64`, `num_sensors` `int64` |

`num_sensors` is on both because it is what makes an undefined row legible: a session of one camera has nothing to
compare, and the count is the whole explanation.

The third `is_defined` state is reachable here too, and by only one route. Both session metrics declare
`requires_capture_timestamps`, so unless the session's rolled-up `timestamp_source` is `capture` they never run and
the row is written all-null. That shape is load-bearing rather than incidental: on container PTS the numbers would be
a zero spread and total overlap that every threshold accepts, and if those were *stored*, `di-reevaluate` would
rebuild them and produce a `PASS` the rig never earned. There is nothing to rebuild from a null row. The cause is on
`session.lance.timestamp_source`, since the row itself cannot carry it.

Note which skip is which: one camera on capture times gives `is_defined = false` with `num_sensors = 1` — measured,
nothing to compare — while one camera on container PTS gives all nulls. The first is a fact a later policy can
re-judge; the second never was one. See
[The session grain skips without a shared capture clock](data-integrity-metrics.md#the-session-grain-skips-without-a-shared-capture-clock).

### `session_evaluation.lance`

One row per `policy_id` × `session_id` × `metric_name`, mirroring `evaluation.lance` column for column with
`session_id` in place of `stream_id` and `session_path` in place of `source`. Everything section 4 says about it
holds unchanged: one verdict column, `margin` / `threshold` `null` on a `SKIPPED` row, `measurement_run_id`
separate from `run_id` so a re-judge is distinguishable from an original verdict.

### The two session metrics

| `metric_name` | What gets judged | Type | Default threshold |
|---|---|---|---|
| `multi_sensor_spread` | `max(start_spread_ns, stop_spread_ns)` | int | 1_000_000_000 — **a placeholder**, see below |
| `multi_sensor_overlap` | `non_overlap_percent` | float | 5.0 |

Overlap is judged through its complement because a spec can only ask for a value to stay *below* a limit, and an
overlap is the one quantity here a session wants more of.

The spread default is the one number in the tool with no first-principles basis. Every other default is a real limit
(zero backward steps, zero gaps); a perfectly synchronised rig spreads by `0`, but no real rig does, so `0` would
fail everything. One second is a placeholder. For scale, one internal 12-camera session measured 133 ms of spread
and 1.3% non-overlap on its capture clock, so the default is roughly 7× the observed skew — the right order of
magnitude, and loose enough that it would catch only gross failures.

### Three tables rather than one

A single `session.lance` carrying the verdict inline is the obvious smaller option, and it is wrong for the same
reason section 3 and section 4 are separate: it fuses facts with judgements. Measurements are the expensive,
never-rebuild half; verdicts are cheap and policy-dependent, and re-judging under new thresholds has to be able to
write a second generation of them without touching what was measured. That is what `reevaluate` exists to do, and a
verdict welded onto the session row cannot participate.

The other alternative is **widening the existing tables** — make `stream_id` and `source` nullable, add
`session_id`, and let one measurement table hold rows of both grains. That is fewer datasets, and it means one
reader gets both. It loses more than it saves:

- Every existing reader has to learn to ask which grain a row is, and `nullable=False` on `stream_id` is currently
  what guarantees it does not have to.
- The two grains have different dedup keys, so one table would carry two rules for picking the newest row.
- A metric's dataset is named after the metric, so a widened table would still be one dataset per metric — the
  saving is in the *identity prefix*, not in the number of datasets.

Additive parallel tables leave every reader in sections 1–4 untouched: nothing that exists today changes shape, and
a reader that does not care about sessions never learns these tables exist.

---

## 6. Putting it together — every possible state

| Situation | `stream.error` | `is_defined` | metric columns | `check_status` | `margin` / `threshold` |
|---|---|---|---|---|---|
| Passed | `null` | `true` | set | `PASS` | set |
| Failed | `null` | `true` | set | `FAIL` | set |
| Ran, but undefined (e.g. 1 frame) | `null` | `false` | set — `NaN` for `rate` / `jitter` | `SKIPPED` | `null` |
| Never ran (no known rate) | `null` | `null` | all `null` | `SKIPPED` | `null` |
| Source wouldn't open | **set** | *no rows* | *no rows* | *no rows* | — |

Only `rate` and `jitter` ever produce `NaN`. The other three carry real values even when undefined: an undefined
`timestamp_ordering` still has integer counts, `timestamp_gap` has `max_gap_ns = 0`, and `frame_reordering_present`
has `has_reordering = null`. So `is_defined = false` does not imply `NaN`.

That last row is why `stream.lance` exists.

The session-grain tables have the same states with `session.error` in place of `stream.error`, and one difference of
emphasis: the "ran, but undefined" row is the *common* case rather than an edge one. A session of a single sensor has
nothing to compare, so both metrics come back `is_defined = false` with `num_sensors = 1` and a `SKIPPED` verdict.
"Never ran" is unreachable today, since neither session metric needs an expected rate — the shape is the table's,
not the metric's.

---

## 7. Telling if a saved measurement is out of date

Three independent signals, covering the three ways a saved fact can stop being true:

- **`instrument_version`** — a number we bump **by hand** when a metric's math changes. Only a human knows whether a
  change actually altered the meaning.
- **The dataset schema itself** — Lance records each metric table's schema, so a renamed or re-typed field is visible
  without our help, and a mismatched write fails outright. This is why there's no `fields_fingerprint` column: with
  typed columns the schema *is* the fingerprint, enforced rather than advisory.
- **Content identity** — `content_etag` / `content_size_bytes` / `content_last_modified`. Catches the case the other
  two can't see: the code is unchanged but the bytes behind the path were replaced.

Stale = any one of the three disagrees with what we'd get today. We avoided using a git SHA or package version as a
signal: any unrelated change in the repo would mark everything stale.

---

## 8. Example

One session, three streams: a good one, one with a single frame and no known rate, and a corrupt one.

`stream.lance` (abridged — `stream_id` shortened for readability)

| stream_id | source | selector | codec_name | num_samples | expected_hz | expected_hz_source | timestamp_source | content_etag | error |
|---|---|---|---|---|---|---|---|---|---|
| `s_a1b2` | `camera_front.mp4` | `video_stream`/`0` | `h264` | 600 | 30.0 | `header` | `capture` | `"9f8e..."` | `null` |
| `s_c3d4` | `camera_left.mp4` | `video_stream`/`0` | `h264` | 1 | `null` | `unavailable` | `container_pts` | `"1a2b..."` | `null` |
| `s_e5f6` | `camera_rear.mp4` | `video_stream`/`0` | `null` | `null` | `null` | `null` | `null` | `"7c8d..."` | `moov atom not found` |

`measurements/rate.lance`

| stream_id | is_defined | period_deviation_percent | expected_hz | actual_mean_hz | num_samples |
|---|---|---|---|---|---|
| `s_a1b2` | `true` | 0.42 | 30.0 | 29.87 | 600 |
| `s_c3d4` | **`null`** | `null` | `null` | `null` | `null` |

`measurements/timestamp_ordering.lance`

| stream_id | is_defined | decreasing_count | duplicate_count | first_decreasing_index | num_samples |
|---|---|---|---|---|---|
| `s_a1b2` | `true` | 0 | 0 | `null` | 600 |
| `s_c3d4` | `false` | 0 | 0 | `null` | 1 |

`camera_left.mp4` has a `rate` row with `is_defined = null` (the metric never ran — no known rate) but a real
`timestamp_ordering` row with `is_defined = false` (it ran, one sample isn't enough to mean anything).
`camera_rear.mp4` has no rows in any metric table at all — just the `stream.lance` row above.

`evaluation.lance`

| run_id | measurement_run_id | policy_id | stream_id | metric_name | check_status | margin | threshold |
|---|---|---|---|---|---|---|---|
| `r1` | `r1` | `p_default` | `s_a1b2` | `rate` | `PASS` | 4.58 | 5.0 |
| `r1` | `r1` | `p_default` | `s_c3d4` | `rate` | `SKIPPED` | `null` | `null` |

Both ids are `r1` because that first run measured and judged in one go.

Now someone tightens the rate threshold to 0.1. That's a new invocation, `r2`, and it reads `measurements/rate.lance`
only — **no source is opened**:

| run_id | measurement_run_id | policy_id | stream_id | metric_name | check_status | margin | threshold |
|---|---|---|---|---|---|---|---|
| `r2` | `r1` | `p_strict` | `s_a1b2` | `rate` | `FAIL` | −0.32 | 0.1 |

`run_id` is `r2` (who judged) but `measurement_run_id` is still `r1` (whose facts). The `p_default` rows are
untouched — both verdicts coexist.

---

## 9. One note on where the code lives

The store lives at `cosmos_curator/next/recipes/data_integrity/`, alongside the CLIs that write it. It began under
`core/sensors/data_integrity/` because both CLIs did, and a CI test forbids anything under `core/sensors/` from
importing the rest of `cosmos_curator` — so a store that had to be reachable from a CLI had to live inside that
boundary too. Moving the CLIs out of the sensor library moved the store with them, and the constraint no longer
applies to it: what stays behind is the reusable half (the metrics, their policy, the result vocabulary and the
per-stream engine), which a second CI test keeps free of `lance` / `pyarrow` so a plain in-memory check never pays
for a columnar format.

Cloud access has since been consolidated onto `core.utils.storage`. It used to go through a private helper module
under `core/sensors/scripts/`, because sensor scripts shared it and cannot import outside `core/sensors/`; that
module is gone and its callers now use `core/utils/storage_cli.py`, which sits above the storage clients and
supplies the CLI half — argument flags, actionable errors, and credentials resolved from the standard AWS profile
namespace rather than Curator's own. `storage_utils.get_lance_storage_options` is still not what the store calls,
and that is deliberate: it resolves credentials through `get_s3_client_config`, a different namespace, and pulls in
`ray`. `storage_cli.get_lance_storage_options` builds the same option names from the profile the CLI was given.
Local + S3 today; Azure is still a follow-up, and `get_lance_storage_options` refuses `az://` explicitly rather
than writing a store it cannot authenticate.

Sharing a directory with the CLIs doesn't mean depending on them. The types a result is expressed in (`CheckResult`,
`StreamResult`, the verdict enums) live in a leaf module, `results.py`, which imports only the metric kernel and
stayed with it. The CLIs, the reports and the store all build on that, so the store never imports a CLI module to
find out what a verdict is — a test asserts this, since it's the kind of thing an import added in a hurry would
undo.

---

## 10. Open questions

1. **Append-only vs overwrite.** Proposed append-only (keeps history, makes multiple verdict generations natural),
   but `split_comparison` overwrites, and append means readers must de-duplicate on the keys in section 1. Worth it?
2. **`policy_id`.** Proposed: a hash of the threshold values, so identical policies collide on purpose. Should a
   human-readable label be required instead of optional?
3. ~~**Does `session.lance` belong in v1?**~~ **Settled — see section 5.** It was deferred while a session row
   would have carried nothing of its own; multi-sensor metrics gave it something to carry.
4. **`measurements/` subdirectory, or metric datasets flat at the store root?** Nesting keeps the root readable at
   twenty metrics; flat is one less path to construct.
5. **One store per session, or one shared store for many?** The schema handles both. We should just agree a
   convention.
6. **Is `stream.lance` the right name**, given it also carries the "couldn't open it" record?
