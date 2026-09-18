# Data Integrity Pipeline

## Summary

`data-integrity` is a [Curator Next](curator-next.md) recipe that runs the existing
data-integrity metrics across many autonomous-vehicle sessions with Ray Data, and persists
the resulting measurements and evaluations into one store root. The first target is a large
multi-session autonomous-vehicle recording bucket.

Where `di-check` measures one video and `di-session` measures one session, this recipe
takes an input that identifies many sessions, fans out to one Ray task per session, and
commits everything a single invocation measured under one `run_id`.

It adds no metric, no modality and no evaluation policy: the metric kernel
(`cosmos_curator/core/sensors/data_integrity/`) is consumed as it is. It does add to the
Lance store (`cosmos_curator/next/recipes/data_integrity/store.py`), which grew the three
session-grain tables this recipe writes alongside the per-stream ones — additively, so every
reader of sections 1–4 of the [store schema][store-schema] is untouched.

---

## Motivation

[Data Integrity Design](data-integrity-design.md) draws its boundary explicitly: the kernel
"measures and judges", and "everything needed to *run* it at recording scale is
deliberately not designed here". Of the six open problems that section lists, `di-check` and
`di-session` answered one — an end-to-end entry point — at single-input and single-session
granularity. This recipe carries that same problem to many sessions. Of the other five it
resolves none outright: full-fidelity sensor input, composing metrics into checks, declaring
checks as data, and generalizing one-read-many-metrics past the camera path all stand. The
fifth, multi-sensor metrics, is under way rather than untouched — two of them exist in the
kernel, and this recipe computes and stores them per session — but the problem as the design
doc states it is broader than two camera-timing checks.

That same section also places three constraints on any solution, and all three are met
without touching the kernel:

- **An undefined measurement must not be evaluated.** The engine still decides this per
  metric, the store still records `is_defined` in three states, and a skipped metric's
  verdict row still carries a null `margin` and `threshold`.
- **The input is a stream of full-fidelity windows, larger than memory, and one read should
  drive many metrics.** A session is the unit of distributed work, but its streams are still
  measured one at a time by the existing engine, which reads each once and folds it into
  every instrument. Nothing buffers a recording.
- **Measurement stays separate from evaluation policy.** Both go through the same builders
  the CLIs use, into the same separate tables.

## Constraints

- Build directly on Ray Data, not the Xenna stage model in `core/interfaces` that
  `pipelines/video/` is built on. (`pipelines/ray_data/` is already Ray-Data-native.)
- Every session processed by one invocation lands in one store root.
- Preserve the store's append, commit, identity, measurement, evaluation and
  re-evaluation compatibility semantics.
- A run that covered its input exits 0 regardless of data-quality findings. A nonzero exit
  means the pipeline could not meet its operational contract — invalid config, Ray failure, a
  failed write — or could not cover everything it was asked to.
- Data-quality failures and unreadable inputs are persisted as findings; the run continues.
- A failure to *reach* data is not a finding about it, and is never persisted as one.
- Consume the recipe's library APIs, not the internals of its CLIs.
- The recipe package must not import Ray at import time, so
  `python -m cosmos_curator.next.recipes.data_integrity.cli --help` stays fast and
  Ray-free.

---

## Current Behavior

### The APIs this recipe consumes

| Symbol | Location | Purpose |
| --- | --- | --- |
| `run_checks` | `sources.py:90` | measure one stream URI, returning check results, video info and the resolved config |
| `discover_streams` | `discovery.py:86` | expand one session path into sorted stream URIs |
| `new_run_id` | `store.py:100` | `uuid4().hex`, shared by every row in one run |
| `content_identity` | `store.py:130` | provenance `HEAD` / `stat` for one source; best-effort, never raises |
| `stream_key` | `store.py:171` | the dedup key one stream's rows agree on |
| `INSTRUMENTS` / `instrument` | `core/sensors/data_integrity/instruments.py:330,341` | the five instrument specs, and the name → spec lookup both row builders take |
| `append_rows` | `store.py:312` | create-or-append rows to one Lance dataset |
| `commit_run` | `store.py:329` | the commit point; must be the last dataset write |
| `write_run` | `store.py:385` | the CLIs' one-session path: build rows, append, commit, write the manifest |
| `write_manifest` | `store.py:533` | overwrite `manifest.json` to describe the most recent run |
| `stream_result` | `core/sensors/data_integrity/results.py:286` | package metrics and video info into a `StreamResult`, default selector included |
| `StreamResult` | `core/sensors/data_integrity/results.py` | per-stream result; `error` carries an unreadable input |

All paths are under `cosmos_curator/next/recipes/data_integrity/` unless noted. This recipe
imports nothing from `cli.py` or `session_cli.py`.

`discover_streams` expands exactly one session path — local directories via a recursive
`rglob`, cloud prefixes via `storage_cli.list_storage_objects` — filtered to
`VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".mkv")` and returned sorted. Its `limit` caps
streams per session, where 0 means no cap. Its `stream_globs` narrows the recursion to the
subprefixes that hold the session's real streams (see [What a session
is](#what-a-session-is)); with globs set the cloud listing is issued uncapped, since a cap
spent before filtering would come back holding mostly excluded keys.

### What a session is

The session layout is already settled, and already exercised against the target bucket.
`session_cli.py` defines a session as "one recording -- typically a `clips/<uuid>/`
directory (or cloud prefix) holding one video per camera", and its usage block gives
`--session-path s3://<bucket>/clips/<uuid>/` as a worked example.

One recording can span several recorder computers: a car with multiple Orins writes a
`recorder-<NN>/` directory per Orin under one session uuid, and their clocks are expected
to be synchronized. So the session is the uuid, not the recorder — splitting it per
recorder would ask `multi_sensor_spread` and `multi_sensor_overlap` whether each Orin
agrees with itself, which is not the question those metrics exist to answer.

That makes `input.stream_globs` load-bearing rather than a convenience. Discovery recurses,
and an observed bucket kept copies of the same cameras beside the recorders (`__original/`,
`triage/`, `derived/` siblings), which unfiltered would measure every camera several times
and present one rig as having twenty sensors. `stream_globs: ["recorder-*/*"]` keeps the
session whole while measuring each camera once. Patterns are matched against the path below
the session with `PurePosixPath.full_match`, anchored there: `*` stops at a `/`, so
`recorder-*/*` does not reach `triage/recorder-00/front.mp4`, and `**` spans depth for a
recorder that nests its cameras further down.

### Nothing enumerates sessions

`storage_cli.list_storage_objects` (`cosmos_curator/core/utils/storage_cli.py`) lists
**objects**, with no `Delimiter` / `CommonPrefixes` support, so asking which sessions
exist under a dataset root means paging every object beneath it. Session enumeration is the
one capability this recipe needs that the DI code does not have — but it is not unwritten:
`multimodal-split` has the same listing, and the [Input Contract](#input-contract) hoists and
shares it rather than adding a second one.

### `write_run` is per-session by construction

`write_run` takes `session_path` as a single scalar, stamps it onto every stream,
measurement and evaluation row, then commits — "Last, and deliberately so: until this lands,
nothing written above is visible to a reader" (`store.py:506`) — and writes the manifest.
Because the commit lives inside it, `write_run` cannot be called once per session and still
produce one run. Everything it composes is public, though — see [Persistence](#persistence)
— except two row builders, `_stream_row` and `_measurement_row`, which this recipe promotes.

### How config-backed recipes run today

A recipe registers a `PipelineKind` (`cosmos_curator/next/core/pipeline_kind.py`) by adding
it to the `BUILTIN_PIPELINE_KINDS` registry in
[builtin_pipeline_kinds.py](../../../cosmos_curator/client/pipeline_cli/builtin_pipeline_kinds.py)
— one import and one tuple entry, in one file. `PipelineKindRegistry` indexes kinds by name
and rejects duplicates, and both CLI entry points resolve a kind through it.

`PipelineKind` is a frozen dataclass of a name plus seven callables, all of them reachable.
`pipeline template` / `validate` / `render` / `schema` call the first five, `pipeline presets
list` calls `list_presets` on every registered kind, and `prepare_run` is the run path:
`pipeline_runtime.main` reads the config's `kind`, fetches that kind, calls
`prepare_run(config, set_overrides=...)` for a deferred `PreparedPipelineRun`, and invokes it
to get a `PipelineRunOutput` carrying a JSON payload and a one-line message.

The `kind` string is matched exactly — `load_pipeline_kind_name` returns it verbatim and the
registry looks it up unchanged — so hyphen and underscore are different kinds rather than two
spellings of one. For example, `robot-action-split` is registered and `robot_action_split`
is rejected. The remaining `video_split` kind selects the legacy pipeline. This recipe is
`data-integrity` everywhere: in `kind:`, as the CLI's kind argument, and as the registered
name. `test_robot_action_split_underscore_spelling_is_not_supported` pins the rule.

That runtime already implements the exit-status behavior this recipe needs. An `OSError`,
`TypeError`, `ValueError` or pydantic `ValidationError` from the lookup or from `prepare_run`
becomes `typer.Exit(2)` — and because `prepare_run` resolves the config before it returns its
callable, a bad config fails there rather than after Ray has started. An exception from the
run itself propagates (or becomes `_fail("runtime", ...)` under `--json`), nonzero either
way. A completed run echoes its message or payload and exits 0. `caption_judge` is the
precedent for findings — it returns `passed: False` in its payload and still exits 0. There is
no third status between those two, which is why a run that committed its rows but covered less
than its input has to say so by raising: see [Failure Isolation and Exit
Status](#failure-isolation-and-exit-status).
Execution is `pixi run run-pipeline`; there is deliberately no `cosmos-curator pipeline run`
subcommand, and `test_pipeline_run_is_not_host_cli_command` pins that.

---

## Recipe Design

### New Kind: `data-integrity`

A distinct kind, structured like other Next recipes: a Pydantic config module, a
`pipeline_kind.py` exposing the config and runtime surface, and a `pipeline.py` driver that
owns the Ray Data work. Unlike the other Next recipes, there is no media output — the only
output is store rows.

The recipe lives beside the DI code it drives, under
`cosmos_curator/next/recipes/data_integrity/`. The original, now removed `video-split` provided
the precedent: a Ray Data pipeline under `next/` whose `pipeline_kind.py` deferred every
import so the CLI stayed Ray-free, with `run_config(resolved) -> dict` as the shape
`prepare_run` wrapped. It refused an input selection that realized nothing rather than
publishing an empty result — the same rule this recipe applies to an empty session
expansion.

The Ray conventions are already in `next/core/ray_runtime.py`: `ensure_ray_initialized`,
`configure_ray_data_progress` and `configure_ray_data_stability`, called in that order at the
top of the original splitter's driver. This recipe imports the same three. The helpers that would
size the stage from the live cluster — `capped_slots_for_items` and `live_ray_node_count`,
with `DEFAULT_IO_SLOTS_PER_NODE` — have no public home: they are private to
`pipelines/ray_data/_runtime.py` and `pipelines/ray_data/constants.py:19`, which
`next/AGENTS.md` forbids importing from `next/`. `session_concurrency` therefore takes a
plain default.

### Config Contract

```yaml
schema_version: 1
kind: data-integrity

input:
  sessions: []              # explicit session paths
  session_list_uri: null    # file of session paths: one per line, or a JSON array
  session_roots: []         # dataset roots to expand into session paths
  session_depth: 1          # prefix levels below each root that name a session
  limit: null               # max streams per session; null maps to discover_streams limit=0
  stream_globs: []          # keep only streams matching these globs, relative to the session

checks:
  expected_hz: null         # null = per-stream header rate, or unavailable (see below)
  batch_size: 0             # timestamps per metric update; 0 feeds the whole array at once
  thresholds:               # shown at their kernel defaults; omit any key to keep it
    max_strict_violations: 0
    max_rate_deviation_percent: 5.0
    max_gaps: 0
    max_jitter_percent: 10.0
    allow_frame_reordering: false
    max_sensor_spread_ns: 1000000000  # session-grain, as is the key below
    max_non_overlap_percent: 5.0

output:
  store_root: s3://example-bucket/di_store/

execution:                  # all fields optional; shown with defaults
  s3_profile_name: null
  azure_profile_name: default
  endpoint_url: null
  session_concurrency: 8      # concurrent session-measurement tasks
  stream_attempts: 3          # attempts per stream when the failure looks transient
  append_batch_size: 512      # sessions per driver-side append
  progress: false             # Ray Data progress bars
  no_progress_timeout_s: 7200 # how long one session may run before Ray calls it wedged; -1 disables
```

`execution.no_progress_timeout_s` widens Ray Data's own 30-minute window, which is far too
short here: one task measures a session's streams in sequence, emitting nothing until the
whole session is done, and a recorder holding several multi-gigabyte videos that need a full
packet scan takes over an hour of cloud reads. At Ray's default such a run dies on its
slowest session — after every other one was measured, and before the commit that would have
made them readable. The two-hour default is not a bound anyone can prove, since neither the
stream count nor the stream size is capped, so the field is configurable: raise it for larger
sessions, or set `-1` to drop the check entirely. It stays finite by default so a genuine
deadlock is still reported rather than waited on.

`checks:` carries only the knobs `run_checks` already takes. It is not the check-declaration
format that [Data Integrity Design](data-integrity-design.md) leaves open, and taking the
name should not be read as a position on it.

Leaving `expected_hz` null is the right default across a heterogeneous dataset, since
`resolve_expected_hz` then takes each stream's own nominal `avg_frame_rate`
(`ExpectedHzSource.HEADER`). Where a container reports no nominal rate it resolves to
`UNAVAILABLE`, and `run_metrics` builds no rate, gap or jitter metric at all — three of the
five come back undefined for that stream rather than wrong. That is the `is_defined` null row
in the exit-status table.

`execution` names the three credential fields explicitly rather than taking the single
`storage_profile` that `robot-action-split` and `multimodal-split` take, because
`run_checks`, `discover_streams` and `content_identity` already take them individually and
this recipe passes them through. The two are not the same field under different names —
`storage_profile` selects a Curator profile while `s3_profile_name` selects an AWS one — and
[Root expansion](#root-expansion-shares-one-lister-with-multimodal-split) is where that
difference has to be handled.

`output.store_root` must be a local directory or an `s3://` prefix.
`get_lance_storage_options` raises on an `az://` root rather than silently writing an
unauthenticated store, and `store_cli.validate_store_path` rejects one at parse time for the
two CLIs — because the store is written last, so a typo should not cost a whole session. This
config should validate it the same way. Reading *sessions* from `az://` is unaffected.

### Input Contract

`input` accepts any combination of three forms:

- **`sessions`** — explicit session paths. The primitive case, and what the tests use.
- **`session_list_uri`** — a newline-delimited or JSON file of session paths, for large
  runs where the list is produced elsewhere.
- **`session_roots`** with **`session_depth`** — expand a dataset root by listing the
  prefixes below it. This is what makes a whole recording bucket usable directly.
  `session_depth` is validated at 1, the shared lister's native scope and the layout
  `di-session` documents; a deeper root is rejected rather than silently flattened, and
  supporting one is additive (see [Open Questions](#open-questions)).

At least one must be non-empty, and an expansion that yields zero sessions is a
configuration error rather than a successful empty run — the same rule `video_split` applies
to an input selection that realizes nothing.

#### Root expansion shares one lister with `multimodal-split`

Root expansion needs a delimiter listing, `list_objects_v2(Delimiter="/")`, and nothing
shared offered one: `S3Client.list_recursive_directory` in
`core/utils/storage/s3_client.py` pages every object beneath a prefix, and so does the
CLI listing wrapped around it. [MR 1104][mr1104] added exactly the listing this recipe needs, for
`multimodal-split`, but private to that recipe: `_list_child_session_ids` in
`next/recipes/multimodal_split/discovery.py`, S3 through a paginated delimited list and local
through a `scandir` of immediate children. The primitive is storage-level rather than
recipe-level, so it is hoisted once and consumed twice rather than written a second time
here:

- `list_child_prefixes(s3_client, *, bucket, prefix)` in `core/utils/storage/s3_client.py`,
  holding the paginated delimited loop and its `FileNotFoundError` for a prefix that holds
  nothing. A thin `S3Client.list_child_prefixes(uri)` method delegates to it, so
  `multimodal-split` keeps the ergonomics it had.
- `list_child_directories(path)` in `core/utils/storage/storage_utils.py`, the `scandir`
  half, symlink semantics unchanged.

`multimodal_split._list_child_session_ids` keeps its own scheme dispatch and now calls both;
its tests pass untouched, which is what makes the hoist observably behavior-preserving.

**The hoisted S3 helper takes a client rather than building one**, and this recipe hands it
one built the same way as every other cloud read here:
`storage_cli.make_s3_client(source, s3_profile_name, endpoint_url)` — an AWS named profile
plus an explicit endpoint override, with boto3's default credential chain behind it. What it
must *not* do is build its own through `get_s3_client_config`, which reads Curator's own
creds file at `S3_PROFILE_PATH` and falls back to the NVCF secret store, raising when neither
exists. Those are two different credential namespaces, and a lister on the second one would
let a run measure a bucket it cannot enumerate. Since the client is now always an `S3Client`,
`sessions.py` calls the bound `S3Client.list_child_prefixes(prefix)` rather than the
module-level function.

`session_roots` on `az://` is rejected rather than expanded: neither shared lister offers a
delimited listing there. Sessions themselves may still live on `az://`; only expanding a root
into them cannot.

Consuming it settles four smaller things:

- **The unit of reuse is the lister, not `discover_candidate_sessions`.** That entry point is
  typed to `MultimodalSplitInputConfig` and returns a `pa.Table` of
  `CANDIDATE_SESSION_SCHEMA` (`source_session_id`, `session_uri`). This recipe keeps its own
  three input forms above the lister rather than adopting that config.
- **No store column is renamed.** Their `session_uri` is the string this doc calls
  `session_path`, and their `source_session_id` is the child directory name — which is *not*
  the store's `session_id`. That stays `identity.session_id(session_path)`, a
  backend-namespaced digest. The child name belongs in logs, not in a row.
- **`limit` means two different things.** Theirs caps sessions, applied after dedup and
  sort; this config's caps streams per session, and keeps that meaning. A session cap is
  worth having here too, but it has to arrive as `session_limit` rather than widening
  `limit` into a field that means whichever the reader assumes.
- **Local sessions stay absolute paths, not `file://` URIs.** MR 1104 resolves them with
  `os.path.abspath`, which is what `identity.normalize_source` does to a scheme-less source.
  A `file://` URI would keep its scheme verbatim there and hash to a different `stream_id`
  than the same file named by path.

---

## Work Granularity

**One Ray Data record per session, not per stream.** A task lists its session and then
measures every stream in it, in sequence.

The session is the unit because a session is what a check is about. Most metrics judge one
stream on its own, but the cross-sensor ones do not: how far apart the rig's sensors started,
how much of the session they were all recording for. Those have no per-stream task to run
in. Giving one worker the whole session is what makes them expressible without another
change to the execution model — and it is what the two session-grain metrics now use.

Throughput is a separate question from what the measurement is about, and it does not argue
the other way in practice. A rig carries the same sensor suite from session to session, so
sessions hold similar stream counts rather than skewed ones, and Ray Data hands the next
session to whichever worker frees up first. If a session ever does run long enough to matter,
the lever is inside the task — `run_session` already takes `max_workers` for exactly this —
not a different record.

Aligning the unit of work first is what kept the store's session-grain half a schema
addition rather than a rewrite: each task now judges its session as well as its streams and
emits the three tables in [data-integrity-store-schema.md][store-schema] section 5. Both
tools reach that verdict through one function, `session_runner.session_metrics`, so a
session the CLI fails is not one the pipeline records as passing — and a session the CLI
skips for want of a shared sensor clock is not one the pipeline records at all.

Within a session the kernel's requirement is unchanged: each stream is read once, through
`run_checks`, which takes exactly one URI, and folded into every instrument.

### Ray Dataset Record Structure

Two record shapes — what the driver feeds in and what the stage emits — each a flat dict so
every record shares one Arrow schema:

```python
# driver-built input: one record per session
{"session_path": str}

# after map(check_session): one record per session
{
    "session_path": str,
    "streams": int,         # streams the session held; 0 if it could not be listed
    "unreadable": int,      # of those, how many were read and found unreadable
    "unreachable": int,     # of those, how many could not be reached at all
    "failed_metrics": int,
    "failed_session_metrics": int,  # of the session's own verdicts, how many failed
    "listing_error": str,   # "" unless the session could not be listed
    "rows": bytes,          # serialized {dataset_name: [store row, ...]}, the whole session
}
```

Every column is present on every record, whatever happened to the session, and one
constructor builds them all for that reason. `iter_batches` concatenates blocks, and a
column that is a string in one block and null in another has no common Arrow type to
concatenate under — which is why `listing_error` is `""` rather than null when there was no
error, and why an unlistable session still reports zeroed counters rather than omitting them.

The counters are what the run summary is built from, and — for the last two — the exit
status: the driver cannot count streams before the run, since listing now happens inside the
tasks, so it sums them as batches arrive.

Workers build the store rows rather than handing the `StreamResult` back for the driver to
convert, for two reasons. The first is the per-stream `content_identity` HEAD: `write_run`
avoids issuing those serially only because the session CLI hands it `storage_stats` gathered
for its progress display, and a Ray driver has no such collection, so building rows
centrally would mean one blocking HEAD per stream inside that loop. Doing it in the worker
puts the request where the stream is already being opened. The second is that rows arrive
already stamped with their own session, which leaves the driver a plain appender with
nothing to regroup.

`rows` is opaque because the store keeps one Lance dataset per metric, each with its own
schema (`store_schema.MEASUREMENT_SCHEMAS`), so a session's rows do not share a single
Arrow type and cannot travel as typed columns of a single Ray record. The flat sibling
columns carry everything logging and the run summary need, so nothing reads `rows` except
the append step.

All Lance writes stay on the driver. Concurrent `append_rows` from many workers would hit
the create-or-append race that `robot_action_split/lance_sink.py` flags as an open TODO,
and there is no reason to inherit it.

`stream_idx` is not in the record: it selects which video stream to open *inside* a
container, and stays at the `run_checks` default of 0, as `di-session` does — so a container
holding more than one video stream has only its first measured. `check_session` packages each
successful run through the shared `results.stream_result` helper and takes the default
selector, exactly as the session runner does, because `stream_key` hashes the selector
alongside the source: hand-building a `StreamResult` with a different selector would fork
stream identity from what the CLIs write for the same file. `run_id` and `created_at` are
computed once on the driver and passed into `check_session`, so every row in one run shares
them.

## Pipeline Execution

```text
pipeline.run(config)
  │
  ├─ expand_sessions(config.input)          → list[str] of session paths
  │      explicit list + list file + root expansion; distinct, and none nested
  │      inside another; an empty result is a config error
  │
  ├─ run_id = store.new_run_id()            → one id for the whole invocation
  │
  ├─ ray.data.from_items([{"session_path": p} for p in sessions])
  │      .map(check_session)                → one record per session, carrying its rows
  │           lists the session, then measures each stream in sequence
  │           never raises: a failed listing comes back as `listing_error`
  │           an unreadable stream becomes a stream row carrying `error`
  │           an unreachable stream becomes a count, and no row
  │      .iter_batches()                    → finished sessions arrive on the driver
  │
  ├─ per batch, on the driver:
  │      sum the counters, then
  │      append_rows(...) → stream.lance, measurements/<metric>.lance (x5), evaluation.lance
  │                         session.lance, session_measurements/<metric>.lance (x2),
  │                         session_evaluation.lance
  │
  ├─ zero rows across every session         → error, before the commit; nothing visible
  ├─ commit_run(root, run_id=run_id, ...)   → run.lance; everything above becomes visible
  ├─ write_manifest(root, run_id=run_id, ...)  → manifest.json
  ├─ any session unlisted or stream unreachable → IncompleteRunError, after the commit
  └─ return summary dict                    → PipelineRunOutput; runtime echoes it; exit 0
```

Items are wrapped into single-key dicts because `from_items` on bare strings names the
column `item`. Seeding from
`from_items` also spreads the sessions across blocks on its own. Were this recipe to seed
from `multimodal-split`'s discovery table instead, it would have to repartition first —
`ray.data.from_arrow` maps one table to one block however the table is chunked, so the whole
run would collapse into a single task.

One stage means the driver never holds the run: nothing is materialized between listing and
measurement, and a batch of finished sessions is the most that crosses it. It also means
there is one concurrency knob rather than two. Listing is IO-bound and measurement is
decode-bound, which would argue for sizing them separately, but a task now does both, so
`session_concurrency` sizes the pair; each task takes Ray's default single CPU, since the
engine decodes one stream at a time.

The cost of one stage is that the run's stream count is not known until every session has
been listed. Two things follow. The summary counts are summed from the batch counters as they
arrive rather than known up front, and the "nothing measured anywhere" error can only be
raised after the append loop — which leaves rows written but uncommitted, and therefore
invisible, exactly as any other run that does not reach its commit. That error names *every*
way it got there, not the first one it found, because they ask different things of an
operator: sessions that could not be listed (the paths or the credentials), streams that
could not be reached (the credentials or the endpoint), and sessions that genuinely hold no
video (the paths or `input.limit`). A run can arrive by several of those roads at once — one
stale path beside two empty sessions is a mix, not a single cause — and naming one would send
an operator after the wrong thing, with no summary and no readable row behind it to correct
the impression.

De-duplicating sources is no longer possible mid-run, and does not need to be: since no
session is nested inside another (see [One source reached twice in one
run](#one-source-reached-twice-in-one-run)), no source is reachable from two of them.

Retries happen inside the worker, not around it, because nothing escapes the worker at all.
`configure_ray_data_stability` (`next/core/ray_runtime.py`) sets
`DataContext.max_map_retries = 3` with `retried_map_errors` limited to transient transport
failures — `OSError`, `ConnectionError`, `TimeoutError`, `EndpointConnectionError`,
`ReadTimeoutError`, `IncompleteRead` — which is the class of error a long S3 `GET` hits. That
machinery only sees exceptions that *escape* the map function, and `check_session` never lets
one out, so it never fires here; the helper's own docstring says as much of any recipe that
turns IO errors into per-item outcomes. The call stays because it is the shared default for a
`next/` Ray Data driver and would still cover a genuine bug escaping the UDF, but no failure
this design reasons about reaches it.

Two things about that machinery are worth stating, because reaching for it was the earlier
plan. Its patterns are matched as substrings of the formatted `"ClassName: message"`, so the
`OSError` entry only ever matches an exception literally *named* `OSError` — a
`FileNotFoundError` from a stale session path never qualified. And a map error that exhausts
its retries aborts the dataset, since `max_errored_blocks` is 0 by default and a block holds
several sessions, so the run that dies takes every session already measured with it.

The retry that does fire is `run_one_stream`'s `max_attempts`, which `check_session` passes
from `execution.stream_attempts`. Per-stream isolation is unchanged — a stream that fails
every attempt is still one row rather than a failed session — and a read timeout stops being
recorded as a data-quality finding manufactured by the network. `max_attempts` defaults to 1,
so `di-session` retries nothing and behaves exactly as before.

Listing has no retry, which is a real cost of catching it here rather than letting it escape:
a transient blip while listing costs that session its measurement for this run, where the
map-level retry would have given it two more attempts. Accepted rather than overlooked. The
retry only ever applied to a listing failure that had already put the whole block at risk,
the run now names the unlisted session and exits nonzero rather than quietly covering less,
and a bounded retry around the listing itself — the shape `run_one_stream` already has — is
additive if blips prove common in practice.

Which failures count as transient is decided by a short list of exception class names beside
`run_one_stream`, matched against the raised exception's MRO — a third copy of Ray's own list
in principle, but not in practice: the two existing copies (`_MAP_RETRY_PATTERNS` in
`next/core/ray_runtime.py` and its twin in the legacy `_runtime.py`) are private, and
`next/AGENTS.md` forbids `next/` importing the legacy tree at all. Matching the hierarchy,
which Ray's string match cannot do, is also what lets this list be shorter: bare `OSError` is
deliberately absent, because here it would drag in `FileNotFoundError` and `PermissionError`
and spend the whole budget re-confirming that a path is still missing. A second list beside it
answers the different question of which failures mean the data was *unreachable* rather than
retryable — see [Failure Isolation and Exit
Status](#failure-isolation-and-exit-status).

---

## Persistence

One `run_id` for the whole invocation, and one commit as the last dataset write, so a run
that dies partway through leaves rows that are present but never believed by a reader. See
[Data-Integrity Store — Lance Schema](data-integrity-store-schema.md), section "One run,
several tables — so a run needs a commit point". This recipe does not change that rule,
only how many sessions precede it.

The driver's whole sequence is already public API, and not a novel use of it: `reevaluate`
already writes this way, calling `append_rows` and then `commit_run` directly rather than going
through `write_run`. `storage_options` is built once with
`get_lance_storage_options(root, ...)` — the `storage_cli` helper that `store.py` and every
`read_*` already use — and passed to the appends and the commit below; `write_manifest` takes
the profile and endpoint themselves instead:

1. `append_rows(rows, join(root, dataset), schema, storage_options)` per dataset, as
   batches of finished sessions arrive — `stream.lance`, five `measurements/<metric>.lance`
   (`store_schema.metric_dataset_path`, one per entry in `INSTRUMENTS`), and
   `evaluation.lance`; then the session-grain three, `session.lance`, two
   `session_measurements/<metric>.lance` (`store_schema.session_metric_dataset_path`, one
   per entry in `SESSION_INSTRUMENTS`) and `session_evaluation.lance`.
2. `commit_run(...)` once, appending the single `run.lance` row that makes everything above
   visible. Its `num_streams` is every stream the invocation *recorded* across all sessions,
   unreadable ones included, since those are rows too — but not the unreachable ones, which
   produced none. `write_run` counts every stream it attempted, and for a single session
   those are the same number; here they are not, and the run row is a claim about what is in
   the store. `reevaluate` is the only caller passing null there, because a re-judge touches
   no stream.
3. `write_manifest(...)`, overwriting the store's single `manifest.json` to describe the
   most recent run.

This sequence has been run end to end on local Ray — four streams across two sessions, one
unreadable, plus a third session path dropped for being nested inside one of them — and
produces one committed run, each session's provenance on its own rows, no verdict rows for
the unreadable stream, and a store `reevaluate` then re-judges. What remains unproven is
scale and the cloud paths, not the store contract.

Nothing is grouped by session on the way in. Every row already carries its own
`session_path`, stamped by the worker that produced it, so the driver appends in arrival
order and per-session provenance survives without a groupby. Row payloads cross the driver a
batch at a time rather than a run at a time, and for the whole run it holds only the session
list and the counters.

### What this needs from `store.py`

Two renames: `_stream_row` and `_measurement_row` become `stream_row` and
`measurement_row`, joining the already-public `build_evaluation_row`, with `write_run`
updated to the new names. `write_run` gathers a whole session before writing anything, which
a driver measuring thousands of streams across many sessions cannot do, so the builders it
uses are the public seam a driver needs. Workers then build every row through public API —
`content_identity` for provenance, `stream_key(result)` for the dedup key, `instrument` for
each result's spec, and the three per-stream builders — plus `session_key` and the three
session-grain builders for the session's own rows. `content_identity` has to be handed the
run's `s3_profile_name` / `azure_profile_name` / `endpoint_url`, exactly as `write_run` hands
them: it never raises, so an unprofiled `HEAD` records empty provenance instead of failing.

The session-grain half then added what it needed: three builders, `session_key`, three dataset
names and a dedup key of their own, and `STORE_SCHEMA_VERSION` at 2 — all of it additive, so a
version-1 store still reads (see [data-integrity-store-schema.md][store-schema] section 5).
Sensor timestamps then took it to 3 by adding `timestamp_source` to `stream.lance`, which is
*not* additive: an append has to match the dataset's schema, so a store written at 1 or 2 needs
a fresh root rather than a migration.
What did not change is the commit rule, or `write_run`'s one production call site,
`store_cli.persist_run`, which is what both CLIs go through, so neither CLI changes.

### The run row's `session_path` is null

`commit_run` and `write_manifest` both take a scalar `session_path` describing the run as a
whole. A multi-session run passes `None`, which the schema permits
(`store_schema.py:157`) and which `di-check` already writes today for a single-stream run.

This is not the rejected alternative below. That one passes `None` *down into every row*,
erasing `session_id`, `session_path` and `relative_key` per stream. Here only the
run-level summary is null and the data rows keep their own session. Nothing reads a run row
by session either: `completed_runs` selects the `run_id` column alone, and `read_runs`
sorts by `(committed_at, run_id)` without filtering.

### `session_rollup` answers a wider question now

`session_rollup(root)` takes no session argument: it reads every committed stream and verdict
under the root and reduces them to one status. On a store `di-session` wrote, that is a session
verdict; on a store this recipe wrote, it is a whole-dataset verdict under the same name. No
stored row changes, and nothing in the pipeline calls it — `session_id` and `session_path` are
on every row, so a per-session rollup is a group-by away. Adding that argument is follow-up
work rather than part of this recipe.

### One source reached twice in one run

`stream.lance` and the metric tables de-duplicate on `stream_id` alone, ordered by
`created_at` DESC then `run_id` DESC. `stream_id` hashes the absolute
`(source, selector_type, selector_value)` triple, deliberately independent of the session it
was discovered under, so `di-check` and `di-session` produce one row for the same bytes
rather than two (store schema, "Dedup keys" and "Stream identity").

Every row in one invocation shares one `run_id` and one `created_at`. So if a source is
reached through two session paths — overlapping `session_roots`, a repeated entry, or a root
nested inside another — its two rows collide on the key with nothing left to break the tie.
`_latest_per_key` sorts on `(created_at, run_id)` and keeps the first row per key, so the tie
falls through to Lance's physical row order: stable for a given dataset, but decided by which
Ray task finished first rather than by anything meaningful. One row wins, the other's
`session_path` goes with it, and nothing reports that it happened. Writing one run this way
confirms it: two sessions sharing a source leave a single stream row, carrying whichever
session was appended first. This is the one way the design can lose the per-session
provenance it exists to preserve.

Expansion is therefore the only place this is prevented, since each session is measured on
its own and no later stage sees two of them together. `expand_sessions` returns distinct
paths, and then drops any session that lies under another — matching whole path segments, so
`clips/a` swallows `clips/a/b` but not `clips/ab`. The enclosing session is the one kept,
because listing recurses, so it already covers the nested one's streams; dropping it would
lose coverage where dropping the nested one loses nothing. Sorting first is what makes the
choice deterministic rather than dependent on which input form named which path, and both
drops are logged.

Two spellings of one path and two paths for one source are all this catches. Two genuinely
distinct sessions that both point at the same bytes — through a symlink, or a bucket alias —
would still collide, and are out of scope: nothing in a listing distinguishes them.

`tool` is `"data-integrity"`, joining the existing `di-check` / `di-session` /
`di-reevaluate` values; the comment in `store_schema` that enumerates them needs the new
entry.

One invocation applies one threshold policy. `commit_run` takes a single `thresholds` and
stamps `policy_id(thresholds)` on the run row and on every evaluation row, so the `checks:`
block configures the run rather than individual sessions.

## Failure Isolation and Exit Status

Nothing propagates out of `check_session`. What varies is what a failure is taken as evidence
*of*, and there are three answers rather than two.

**The data is bad.** A failed open or decode becomes `StreamResult(error=str(exc),
metrics=[])`, which the store already persists as a stream row with no measurement or
evaluation rows, and the session's other streams are measured regardless. That is not merely
the same *representation* `session_runner` produces today — it is the same logic, so
`check_session` calls it rather than restating it: `_run_one_stream` is promoted to
`run_one_stream`. Data-quality FAILs are ordinary evaluation rows. Neither reaches the exit
status.

**We could not reach the data.** An expired token, a 403, a 503 or a transport error that
exhausts `max_attempts` says nothing about the stream, so it is counted and no row is written.
Recording it as that stream's `error` would publish a data-quality claim the environment
manufactured — and because readers resolve to the newest row per `stream_id`, that claim
supersedes a healthy measurement from an earlier run, so a token that expires halfway through
a fleet run leaves every stream after it reading back as broken. Writing nothing leaves the
last real answer standing. `run_one_stream` raises `InfrastructureError` for these, under an
opt-in `raise_infrastructure_errors` flag: `di-session` keeps the old contract, where the
operator is watching and every failure is one line of a report, and only a run over thousands
of streams needs the distinction.

Classification is `is_infrastructure_error`, reading three signals in order — the exception's
class hierarchy (`NoCredentialsError`, `ProfileNotFound`, our own `StorageCliError`, which on
this path can only be a failure to build a credentialled client), the backend's error code
(`ExpiredToken`, `AccessDenied`, `SignatureDoesNotMatch`), and the HTTP status (401, 403, and
anything 5xx). The last two are read defensively off whatever attributes the exception
carries, because the alternative is importing botocore and azure.core to name two types. The
404 family is deliberately absent: an object that is gone is a fact about the dataset, and
belongs in a row.

Both are read through the whole exception chain, not off the exception itself. Reads on this
path go through `smart_open`, which replaces every failed S3 call with a plain `IOError` and
keeps the `botocore` error only as `backend_error` and `__cause__` — so absent, denied and
unavailable all arrive as one exception type carrying none of the two signals above. Read from
the top only, a 403 looked like nothing at all and the stream got the manufactured
data-quality error this section exists to prevent. `sources.error_chain` does the walk and
`sources.error_code` / `sources.http_status` read the two signals off it, all three shared with
the sidecar reader below, which has to tell the same three apart for the same reason. Sited
there rather than here because that is the module the `smart_open` wrapper comes from, and
because one reading of an error shape is what keeps a denial from being anything else.

**The sensor timeline was not there, or was not usable.** `run_checks` looks for a
`<video>.mp4.timestamps.npy` sidecar beside each stream and hands it to the sensor, which then
measures and reports sensor times instead of container PTS (see
[Timestamp provenance][metrics-provenance] for why that matters). The four outcomes are
deliberately different. **Absent** is the ordinary case — most sources have no sidecar — so it
is not a finding: the stream is measured on PTS and `timestamp_source` says so. **Present but
unusable** — corrupt, or a length that disagrees with the video's frame count — fails the
stream, because falling back silently would hide the one thing an operator can act on, and
because a timeline matched to frames by position leaves *every* frame's sensor time in doubt
when the count is wrong. **Unreachable** is neither, and is classified exactly as an
unreachable video is: it is the same bucket and the same credentials, so a denial there says
nothing about the recording. Telling those apart is what `error_chain` is for above — through
`smart_open`'s wrapper, a missing sidecar and a denied one are the same exception type. The
fourth is **present but disowned**, below.

**This recipe grades source sessions, not split clips.** The distinction is not
enforceable in general — a path is a path, and a clip directory holds an MP4 and a sidecar
like any other — but it decides what the numbers mean. `multimodal-split` cuts a session by
resampling every camera onto one evenly spaced grid at the configured output rate, and writes
*that grid* to `<clip>.mp4.timestamps.npy`, deliberately, so a later stage reads a clip the
way it read a session. Graded as a recording, such a session answers the session grain before
it is asked: every camera sits on the same lattice, so spread is zero and overlap total
however ragged the rig behind the clips was, because the alignment is the splitter's and the
original skew is already gone. The instants a camera observed are kept beside the grid as
`<clip>.mp4.sensor_timestamps.npy`, and that file is the discriminator: nothing a recorder writes
produces it, so `read_sensor_timestamps` treats its presence as the tree disowning the timeline
next to it, logs which artifact it found, and measures container PTS instead. For clips written
before the rename, the reader recognizes `<clip>.mp4.observations.npy` as the same discriminator;
new split runs write only `.sensor_timestamps.npy`. Existence is the
whole signal — the marker holds one observation per output frame and repeats wherever a frame
was dropped, so it is non-decreasing where a timeline must strictly ascend, and could not be
read as one. Keyed per stream rather than per directory, since twelve cameras share a clip
directory and each names its own artifacts. This is cheap insurance on an unsupported input,
not support for it: the honest answer for a derived clip is that rig sync is unanswerable, and
that is the answer it now gives.

Absence is not a finding *per stream*, but it does decide the session grain. `session_metrics`
rolls the contributing streams' clocks into one value — `sensor` only when every one of them
reported it, else `container_pts` or `mixed` — and both session-grain metrics declare
`requires_sensor_timestamps`, so on anything but `sensor` they never run. Their rows are
written all-null and their verdicts are `SKIPPED`, naming the clock that was measured instead.
Storing the numbers would be worse than storing nothing: a mux puts every camera on one
regenerated grid, so the zero spread it produces is the encoder's, and a stored zero is one
`di-reevaluate` would rebuild into a pass. `session.lance.timestamp_source` carries the cause,
because a null row cannot.

**We could not reach the session.** A listing failure — a stale path, a moved bucket, a
denied prefix, an expired token — comes back as `listing_error` on that session's record
*and* as a `session.lance` row with `error` set. Not symmetric with an unreachable stream,
and deliberately: an unreachable stream has a healthy row from an earlier run that a
manufactured error would supersede, while an unlisted session has a `session_id` nothing
else will ever say a word about. Writing the row is the only way that path is recorded at
all. The run still raises `IncompleteRunError` — the row explains the gap, it does not make
the run complete.

The last two are why a run has a third outcome: **completed, but did not cover its input.**
The driver commits everything it measured, writes the manifest, and only *then* raises
`IncompleteRunError`. Ordering is the whole point — the measurements are durable and readable,
and the nonzero exit says only that the coverage is short. A stale entry in a manifest of
thousands of session paths is close to inevitable, and it should cost the run its exit status,
not its rows. A raising run is reported by its exception's text and nothing else —
`{"ok": false, "error": "runtime", "message": ...}` under `--json`, the propagated traceback
otherwise — so that message has to stand alone: it names the run id, the store root, the
counts, and up to five unlisted paths before summarising the rest.

Closing the gap means re-running, which re-measures the sessions that already succeeded as
well: the store dedups on `stream_id` and keeps the newest row, so the result is correct
rather than cheap. Covering only what was missed is incremental restart, which stays a
[non-goal](#non-goals).

A session that lists cleanly and holds nothing is not a failure of the *run* — an empty
session is a legitimate finding about the input — but it is counted and reported, because a
stale S3 prefix lists clean and empty too, and the count is the only thing that separates
"the prefix moved" from "this session has no video in it". Its own verdict is a different
question: nothing was measured, so the session rolls up `ERROR`, exactly as a live report
grades one.

| Situation | Persisted as | Exit |
| --- | --- | --- |
| Metric FAIL against the thresholds | evaluation row | 0 |
| Session-grain metric FAIL against the thresholds | `session_evaluation` row | 0 |
| Unreadable or undecodable input | stream row with `error` | 0 |
| Metric ran but was undefined on too little data | measurement row with `is_defined` false, plus a `SKIPPED` evaluation row | 0 |
| Metric never ran, with no usable rate | measurement row with `is_defined` null, plus a `SKIPPED` evaluation row | 0 |
| Session listed cleanly and held no streams | `session.lance` row with `num_streams` 0, no `error` and a null `timestamp_source` — nothing contributed a clock — plus a never-ran session measurement and a `SKIPPED` verdict per session metric; counted in the summary. Rolls up `ERROR`, matching a live report on an empty session | 0 |
| No sensor-timeline sidecar beside the video | stream row measured on `container_pts`, recorded in `timestamp_source` | 0 |
| Session whose streams were not all on sensor times | `session.lance` row with `timestamp_source` `container_pts` or `mixed`, plus a session measurement row with `is_defined` null and every metric column null, and a `SKIPPED` verdict naming the clock. Not counted as a failure, and a re-judge cannot turn it into a pass | 0 |
| Session of one camera, that camera on sensor times | session measurement row with `is_defined` false and `num_sensors` 1, plus a `SKIPPED` verdict — measured, with nothing to compare, which is a different finding from the row above | 0 |
| Sidecar present but unreadable, or of the wrong length | stream row with `error`, as for any unreadable input | 0 |
| Sidecar unreachable: 403, 5xx, or a transport failure | no stream row — the same as an unreachable video, since it is the same denial | nonzero, after the commit |
| Stream unreachable: expired token, 403, 5xx, or an exhausted transport budget | no stream row; counted, and reported by the failure. Its session's row still says how many streams were *found*, so the session rolls up `ERROR` rather than `PASS` | nonzero, after the commit |
| Session that could not be listed | `session.lance` row with `error` set, and nothing else; counted and named by the failure | nonzero, after the commit |
| Invalid config, or an input set expanding to zero sessions | nothing | nonzero |
| Nothing measured anywhere | rows may exist, uncommitted and unread | nonzero |
| Ray execution failure | rows may exist, uncommitted and unread | nonzero |
| Store write or commit failure | nothing visible to a reader | nonzero |
| Interrupted with Ctrl-C | rows may exist, uncommitted and unread | nonzero |

A run that covered its input returns `{"run_id", "sessions", "streams", "unreadable",
"unreachable", "unlisted_sessions", "empty_sessions", "failed_metrics",
"failed_session_metrics", "store_root"}`, which
the kind's `prepare_run` wraps in a `PipelineRunOutput` — the dict as `json_payload`, plus a
one-line `message` naming the run id, the stream count and the findings. Findings are counted
there, never encoded in the exit status. `unreachable` and `unlisted_sessions` are always zero
in that payload, since a run reaching it has none; when they are not, the same counts arrive
in the exception's message instead. Per-session progress is logged at INFO.

---

## Module Inventory

New, under `cosmos_curator/next/recipes/data_integrity/`:

| File | Contents |
| --- | --- |
| `config.py` | Pydantic models (frozen, strict, `extra="forbid"`), `kind: data-integrity`, `load_config` / `resolve_config` with dotted overrides |
| `sessions.py` | `expand_sessions(input_config, *, execution)` for the three input forms, over the shared child-prefix listers |
| `processing.py` | `check_session(record, *, config, run_id, created_at) -> dict`, the Ray UDF, over `discover_session(session_path, *, config) -> list[str]`; nothing raises, and every outcome is a column |
| `pipeline.py` | driver `run_config(config) -> dict` plus an argparse `main()`, and `IncompleteRunError`; the only module that imports Ray |
| `pipeline_kind.py` | `DATA_INTEGRITY_KIND = PipelineKind(name="data-integrity", ...)`, all seven callables, `list_presets` returning `[]`; every import deferred into its callable (`noqa: PLC0415`) and `_prepare_run` resolving the config before returning the closure, mirroring `video_split/pipeline_kind.py` |

Modified: `store.py` (the two renames above), `session_runner.py` (`_run_one_stream` promoted
to `run_one_stream`, plus `max_attempts`, `raise_infrastructure_errors`, `InfrastructureError`
and `is_infrastructure_error`), `store_schema.py` (the `tool` comment at `:155`,
one word), `core/utils/storage/s3_client.py` and `storage_utils.py` (the two hoisted listers),
`next/recipes/multimodal_split/discovery.py` (repointed at them), and
`client/pipeline_cli/builtin_pipeline_kinds.py` (one import, one tuple entry). The two
composition-root tests that pin the registered names —
`tests/cosmos_curator/client/pipeline_cli/test_builtin_pipeline_kinds.py` and
`test_pipeline_app.py`, which spells the same list into an error message — change with it.

## Relationship to `di-check` and `di-session`

| Concern | `di-check` | `di-session` | `data-integrity` |
| --- | --- | --- | --- |
| Unit of input | one video | one session | many sessions |
| Session enumeration | n/a | caller supplies the path | expanded from config |
| Concurrency | none | `ThreadPoolExecutor` | Ray Data |
| Entry point | argparse CLI | argparse CLI | config file + `pipeline_runtime` |
| Metric kernel | shared | shared | shared |
| Store write | `write_run` | `write_run` | `append_rows` then `commit_run` |
| Exit status | `PASS` / `FAIL` / `ERROR` as 0 / 1 / 2 | same | 0 on full coverage, nonzero on operational failure or short coverage |
| Unreachable input | that stream's `ERROR`, in the report | same | counted, not recorded, and the run exits nonzero |

The last two rows are the only ones where this recipe deliberately behaves *differently*
rather than simply doing more, and both follow from scale. A nonzero code on FAIL is useful in
an interactive CLI; a pipeline over thousands of sessions must not report a data-quality
finding as a job failure. Likewise, for one session an unreachable stream is a line in a
report the operator is already reading, while for a fleet run it is a row that would outlive
the run that wrote it.

---

## Alternatives Considered

**One run per session, N runs per invocation.** Call `write_run` once per session and skip
the two renames entirely. Rejected because it writes N rows to `run.lance` and overwrites
`manifest.json` N times for one invocation, so "what did this invocation do" stops being
answerable from the store and no row's `num_streams` describes the invocation as a whole —
which is the run-level provenance this recipe exists to add.

**Aggregate every session into one `write_run(session_path=None)` call.** Also needs no
store change. Rejected because a null `session_path` there propagates into every stream,
measurement and evaluation row, nulling `session_id`, `session_path` and `relative_key`
per row — the provenance that has to survive. Passing null for the run row alone, as
[Persistence](#persistence) does, is a different thing.

**Stream-granularity Ray records, listed in a stage of their own.** Buys one thing today: a
global listing pass can de-duplicate sources, where the single stage relies on the nesting
guard instead. Rejected because a cross-sensor check has no per-stream task to run in, so
this would need the decision reversed later rather than extended — see [Work
Granularity](#work-granularity).

**Session-grain store rows to match.** A `session.lance` table, and verdicts that hang off a
session rather than a stream, is what a cross-sensor check ultimately needs. Deferred at the
time and since built: aligning the unit of work first is what kept it an additive schema
change, and the three tables are specified in [data-integrity-store-schema.md][store-schema]
section 5.

**Reuse `session_runner.run_session` inside a Ray task.** Attractive because it is the
existing single-session orchestrator. Rejected for what surrounds the loop rather than the
loop itself: it re-runs discovery, assembles a `SessionReport` this recipe does not persist,
and owns the progress hooks a CLI needs. `check_session` calls `run_one_stream` directly
instead, which is the part that is shared. Its `max_workers` pool is the shape an
intra-session concurrency knob would take if one is ever wanted — deferred, not rejected,
and until then there is only one knob because the pool's default is a single worker.

**An unreachable stream recorded as a stream row, like an unreadable one.** The cheapest
thing, and what the first implementation did. Rejected because the row is a lie that outlives
the run: `error` reads as a claim about the data, readers resolve to the newest row per
`stream_id`, and so an expired token retroactively marks healthy streams broken. A row that
said "unreachable" in a column of its own would fix the ambiguity but not the supersession,
and it needs a schema change to say it.

**Fail the whole run the moment a stream turns out to be unreachable.** Honest, and it stops
the pollution at the first stream rather than the last. Rejected because a single 503 in a
fleet run would then discard every session already measured, which is the same trade the
listing failure used to make and the reason it is being undone. Committing first and failing
after keeps both halves: the measurements stand, and the run still says it fell short.

**A per-session error row for an unlistable session.** Deferred while there was no
session-grain table to write it to, and now what happens: the row carries `error` and the
run still raises, so a reader can query the stale path instead of reading it out of an
exception message. The count in that message stays, since a run's coverage is a fact about
the run.

**Validate the whole manifest up front, before Ray starts.** Attractive because it makes a
stale path cheap to find. Rejected as the primary answer: proving a path listable is proving
it listable *now*, so it is one more round trip per session against the same failure modes,
and a token that expires mid-run is untouched by it. It is additive later as a fast-fail
convenience, not as the contract.

**A bespoke argparse entry point reusing `di-session`'s exit codes.** Rejected: that returns
1 on a data-quality FAIL. `pipeline.py` keeps an argparse `main()` for local iteration, but
it mirrors the runtime's contract.

**A recipe-local delimiter lister.** Rejected because `multimodal-split` has the same listing
([MR 1104][mr1104]): `curator-next.md`'s rule is that reuse waits until something else needs
it, and something else now does.

---

## Delivery Plan

1. **Prep** — the two store renames with `write_run` following, `_run_one_stream` promoted
   and given `max_attempts`, and the two listers hoisted out of `multimodal-split`. Each is
   small enough to review on sight, and together they unblock everything else.
2. **Config, input contract, processing, CLI registration** — everything except Ray. This
   establishes the exit-status and summary contract early, as `robot_action_split` did.
3. **Ray Data wiring** — the per-session `map` stage, the concurrency knob, and the
   local-Ray tests.
4. **Example config** — plus a run against a multi-session cloud subset, which is the one
   claim on this page that a local run cannot make.

## Testing Strategy

Under `tests/cosmos_curator/next/recipes/data_integrity/`:

| Test | Asserts |
| --- | --- |
| `test_config.py` | resolution, dotted overrides, template completeness, and the rejections: an empty input block, an `az://` or bucketless store root, a deeper `session_depth`, the underscored kind |
| `test_sessions.py` | all three input forms, S3 roots against a fake lister; empty expansion raises; overlapping roots and repeated entries collapse to distinct sessions; a nested session is dropped while a session merely sharing a name prefix is not |
| `test_processing.py` | a session yields a row per stream it holds, plus five measurement and five evaluation rows for each readable one, and its own `session.lance` row with two session measurements and two session verdicts; a failing session metric is counted in `failed_session_metrics`; an unreadable stream yields a stream row carrying `error` without costing its siblings; an empty session yields counters, a session row and `SKIPPED` session rows, but no stream row; an unlistable session is reported as `listing_error` rather than raised, persisting its `session.lance` row with `error` set and nothing beneath it; a mode-`000` video -- a real read failure under a listing that succeeds, not an injected exception -- is counted as *unreachable*, since the open raises `PermissionError`, and so leaves no stream row; every record carries every column, at one type |
| `test_pipeline.py` | local Ray over two sessions with one corrupt file: exit 0, exactly one committed `run_id`, each session's `session_path` present on its own rows, findings persisted, and the coverage counters zero; a third session path nested inside one of them is measured once, under the session enclosing it; a stale session path alongside good ones raises `IncompleteRunError` *after* the good sessions are committed and readable; a run where nothing could be listed commits nothing; an unreachable stream under sessions that all list cleanly raises `IncompleteRunError` while leaving the store's row count and the run row's `num_streams` unchanged; a session holding no streams is counted without failing the run; a failing session metric reaches the store and is counted in the run's `failed_session_metrics`; a run that measured nothing names every cause it had rather than the first, over both a mixed run and the message itself; an unwritable store root fails the run |
| `test_pipeline_kind.py` | the kind's seven callables, and a subprocess guard that registering it imports neither Ray nor Lance |
| `test_store.py` (extend) | a caller can assemble a run from the public builders, `append_rows` and `commit_run` |
| `test_session_runner.py` (extend) | `max_attempts` defaults to one attempt, retries a transport error, and never retries a malformed file or a missing path; an interrupt cuts the backoff short instead of waiting it out; and, under `raise_infrastructure_errors`, an unreachable stream raises with its source and cause while a genuinely missing or corrupt one still gets a verdict |

Plus the registered-name tuples in `test_builtin_pipeline_kinds.py` and `test_pipeline_app.py`,
which spell the same list into an error message. Upstream,
`test_importing_composition_root_defers_config_and_runtime_modules` already pins that
registering a kind does not drag its config and runtime modules into the CLI's import graph,
which is what keeps the Ray-free guard true from the other side.

The boundary is this recipe's own seams. `run_checks`, the instruments and the evaluators
are already covered by the kernel and recipe suites and are not re-tested here.

---

## Open Questions

**How should `session_depth` reach a lister that only returns immediate children?** The
shared lister stops at one level, which is all `multimodal-split` needs, since a session there
is always a direct child of the prefix. Depth 1 is therefore what ships, and a deeper root is
rejected. Supporting one later means either walking level by level from the driver — one
delimited list per prefix, which fans out quickly — or giving the shared lister a depth
argument and letting it walk for both recipes. The same question decides whether a glob is
worth supporting instead of a fixed depth: more general, but it invites recursive listing
costs on a dataset this size. Either way it is additive, and nothing above needs rework for
it.

Three earlier questions are settled, and the sections above describe the outcome rather than
the choice:

- The two store row builders are public, so a driver can compose a run from one place instead
  of per-call `noqa`s or one run per session.
- `check_session` reuses `run_one_stream` rather than restating its wide catch, and the
  bounded retry rides on that helper's new `max_attempts` — default 1, so `di-session` is
  unchanged.
- `session_concurrency` takes a plain default. `next/AGENTS.md` forbids `next/` importing
  the legacy `pipelines/ray_data` tree, which is where the helpers that would size it from
  the live cluster live, so promoting them is that migration's work rather than this
  recipe's.

---

## Acceptance Criteria

The first implementation is complete when:

- one invocation over multiple sessions writes exactly one `run_id` into one store root,
  and every row carries the `session_path` and `session_id` of the session it came from
- a run whose inputs include both data-quality failures and unreadable files exits 0,
  persisting the failures as evaluation rows and each unreadable input as a stream row
  carrying `error`
- invalid config, a Ray execution failure, and a failed store write each exit nonzero, and
  no data-quality finding produces a nonzero exit
- a run whose input includes a session that cannot be listed still commits every session it
  did measure, exits nonzero, and names the unlisted path in what it reports
- a stream that could not be reached leaves no row at all, so an earlier run's measurement of
  that stream is still the newest row a reader resolves to
- an input set that expands to zero sessions is rejected as a configuration error
- a session path nested inside another configured session is measured once, under the session
  enclosing it, so no two rows in a run collide on `stream_id`
- a store written by `di-session` is readable and appendable by the pipeline, and a store
  written by the pipeline is re-judgeable by `reevaluate.reevaluate()` (which stamps
  `tool="di-reevaluate"` and keys off `measurement_run_id`), across both grains and with
  only additive schema change — a version-1 store still reads
- `di-check` and `di-session` keep their current output and exit codes
- `cosmos-curator pipeline template` and `pipeline schema` accept `data-integrity` as their
  kind argument, `pipeline validate` / `pipeline render` accept a `kind: data-integrity`
  config, and the underscored spelling is not accepted anywhere
- `pixi run run-pipeline` executes the recipe from a config file
- `python -m cosmos_curator.next.recipes.data_integrity.cli --help` still does not import
  Ray
- the pipeline runs end to end over a multi-session subset of the target bucket and
  persists a committed run

## Non-Goals

- new metrics or new modalities
- distributed re-evaluation from stored measurements
- restarts after failure, and idempotent re-runs
- any change to the shared storage CLI helpers, or to `di-check` / `di-session` behavior
- any *non-additive* change to the Lance schema: the session-grain tables were added at
  `STORE_SCHEMA_VERSION` 2, and no existing dataset, column or dedup key moved. (Scoped to
  this recipe. Sensor timestamps later took the store to version 3 non-additively, on their
  own terms — see [data-integrity-store-schema.md][store-schema] section 2.)

[mr1104]: https://gitlab-master.nvidia.com/aidot/cosmos-curator-public/cosmos-curator/-/merge_requests/1104
[store-schema]: data-integrity-store-schema.md
[metrics-provenance]: data-integrity-metrics.md#timestamp-provenance
