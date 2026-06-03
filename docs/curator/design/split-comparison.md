# Split Output Comparison — v2 design

This document is the **orientation + rationale** for the split-comparison
package. The code is the ground truth for schemas, signatures, and config
fields; this doc captures the "why" decisions that grep'ing the code won't
surface (Ray Data shape, Arrow placement, caption batching, what's
deliberately absent).

## Goal

Compare two output trees from the split video pipeline and report differences
as a structured report. Two trees are produced by running the same input data
through two pipeline configurations (e.g. two model versions, two encoder
settings); the comparator is the audit step that says whether the
configurations diverged in any material way.

The tool covers:

- A/B compare `summary.json`.
- Discover the set of clips that appear in either output.
- Per clip, run two independent passes:
  - **Stage 1 — metadata comparison.** Compare clip-level metadata JSON:
    structure parity, aesthetic-score parity, motion-filter-score parity, and
    model-based caption similarity (caption embeddings + cosine similarity).
  - **Stage 2 — video index comparison.** Load each side's clip MP4, build a
    `VideoIndex`, and compare in memory.
- Group emitted issues by video key in the final report.

Stage 1 is CPU-bound caption embedding + cheap metadata comparisons; Stage 2
mixes storage IO (MP4 reads) with header-parse + VideoIndex array
construction, which together come out CPU-significant in practice. They
have unrelated resource shapes — Stage 1 wants a few fat actors each
holding a caption-model copy in memory, Stage 2 wants roughly one actor
per core each holding precomputed smart_open params for the two output
roots — so they run as two separate Ray Data pipelines independently.

## Module layout

```text
cosmos_curator/pipelines/video/split_comparison/
  __init__.py              # package marker
  cli.py                   # argparse: --config PATH / --print-default-config
  driver.py                # compare_split_outputs + run_metadata_stage / run_video_index_stage
  clip_discovery.py        # build the pa.Table of clip rows (CLIP_ROW_SCHEMA)
  summary_compare.py       # A/B summary.json comparison (no Ray)
  metadata_stage.py        # Stage 1 actor + per-comparison helpers
  video_index_stage.py     # Stage 2 actor + per-comparison helpers
  result_model.py          # Issue TypedDict, ISSUE_SCHEMA, make_issue, Report -- output contract
  config.py                # SplitComparisonConfig (one frozen pydantic v2 model + nested policies)
  summary_schema.py        # pydantic v2 OutputSummary / Processed / Unprocessed + discriminated union
  summary_loader.py        # load_summary: read summary.json via smart_open + OutputSummary.from_json
  report_io.py             # write_report: dispatches on config.report_format (json / lance); report_path used verbatim
```

Tests mirror the module path under `tests/cosmos_curator/pipelines/video/split_comparison/`.

## Top-level flow

The two clip stages are **independent Ray Data pipelines that both consume the
same clip table**, not a chained `.map(...).map(...)` graph. Each stage
produces a typed issue table; the driver concatenates the three issue tables
(summary + metadata + video index) into the final `Report`.

```text
          summary_a.json        summary_b.json
                │                    │
                └─────────┬──────────┘
                          ▼
                  compare_summaries          (pure Python, no Ray)
                          │
                          ▼
                summary_issues : pa.Table
                          │
                          │      (and, in parallel, the clip table)
                          │
                  discover_clips ──────────►  clips : pa.Table
                                                  │
                                ┌─────────────────┴─────────────────┐
                                ▼                                   ▼
                    ┌───────────────────────┐         ┌───────────────────────┐
                    │ run_metadata_stage    │         │ run_video_index_stage │
                    │   MetadataStage pool  │         │   VideoIndexStage pool│
                    │   → pa.Table          │         │   → pa.Table          │
                    └───────────┬───────────┘         └───────────┬───────────┘
                                │                                 │
                  metadata_issues : pa.Table          video_index_issues : pa.Table
                                │                                 │
                                └─────────────────┬───────────────┘
                                                  ▼
                                       pa.concat_tables
                                                  │
                                                  ▼
                                       Report.issues : pa.Table
```

Each stage reads the clip table from scratch (`ray.data.from_arrow(clips)`); the
Ray Data datasets are not shared because of the resource-shape difference
(CPU-heavy with a held caption model vs. CPU + IO mix on MP4 indexing). Stage 2 is
skippable via `config.compare_video_index`; the metadata stage runs whenever
there are any clips. `Report.stages_run` records which stages actually ran so a
`passed=True` report can't be misread as "everything was checked" when video
index was disabled.

## Stage 1 — metadata

One Ray Data pipeline, one actor pool. Each `MetadataStage` actor loads the
caption embedding model once at construction and reuses it across every clip
routed to it. On each `__call__`, the actor reads the metadata JSON for both
outputs once; structural parity (one-sidedness, corruption) is enforced at
read time, then three comparators (aesthetic, motion, captions) run against
the loaded payloads. See "Caption batching strategy" below for the
encode-amortization strategy.

## Stage 2 — video index

A second Ray Data pipeline with a different resource shape — no caption
model in memory, but the indexer does enough CPU work (header parse,
VideoIndex array build) per row that the default reserves a full core
per actor. Each `VideoIndexStage` actor precomputes the
smart_open params for both ``output_a`` and ``output_b`` once at construction
and holds them as ``self._params_a`` / ``self._params_b``. On each row the
actor loads both outputs' MP4 in parallel, builds the `VideoIndex` + `VideoMetadata`, and
emits one issue per divergent field — `clip_mp4_index_mismatch`,
`clip_mp4_metadata_mismatch`, `clip_mp4_index_dtype_mismatch`. Missing or
unreadable MP4s map to dedicated issue codes; the comparator never propagates
exceptions.

## Stage independence

Stages 1 and 2 are run as **two separate Ray Data pipelines over the same clip
list**, not chained. Both produce flat issue lists. The driver merges them.
This is deliberate:

- The stages have unrelated resource shapes (CPU-heavy with a held model vs
  CPU + IO mix on MP4 indexing). Pipelining one through the other would force
  resources to coexist on the same actor pool or force an explicit handoff.
- They produce independent outputs (issues, no shared state). There's no value
  to pipelining beyond what Ray Data already does inside one map call.
- Re-running just one stage (e.g. only video index) becomes "call the
  function." No plan-variant surgery.

## Where Ray Data earns its keep

- **Stage 1 caption model**: `ActorPoolStrategy` with the model loaded once per
  actor is the textbook fit. Each actor pays the model-load cost once at
  construction and amortizes it across the clips routed to it.
- **Stage 2 per-actor smart_open params**: precomputed once at actor
  construction and held for the life of the actor. Actors are stateful;
  tasks aren't. Threads would have to handle this with thread-local storage.
- **Spilling**: if a future run hits millions of clips, Ray Data spills to
  disk; a thread pool fills RAM with results.

Where it doesn't earn anything: dispatch, plan-variant routing, sharing loaded
data between features. Those are eliminated by design.

## Where Arrow lives (and where it doesn't)

Arrow is the format at the **driver / cross-stage boundary**, not deep inside
the per-row comparators:

- `discover_clips` returns `pa.Table`.
- `ray.data.from_arrow(clips)` keeps blocks in Arrow format in the object
  store.
- Inside a stage actor's `__call__`, Ray Data hands the worker a `pa.Table`
  batch; the actor walks the rows as plain Python dicts and dispatches per-row
  to module-level comparator functions. Compare helpers take plain Python args
  and return `list[Issue]`; they never see `pa.RecordBatch`.
- The actor materializes its emitted rows back into a `pa.Table` via
  `pa.Table.from_pylist(..., schema=ISSUE_SCHEMA)` at the boundary.
- The driver concatenates the three issue tables (summary, metadata, video
  index) via `pa.concat_tables` and stores the result in `Report.issues`.

The Arrow wins (schema enforcement, columnar groupby/filter, Parquet/Lance
persistence) all live where they matter — at the driver, where someone is
going to query or render the report. Inside the actor, dict I/O keeps each
comparator plain Python: readable, unit-testable without Ray, easy to
construct rows via the `make_issue` helper.

## Issue schema design

`result_model.py` defines `ISSUE_SCHEMA` — a wide Arrow schema where the core
columns (`code`, `feature`, `video`, `clip`, `field`, `output`) are queryable
with Arrow compute without any JSON unpacking, plus a `details` column that
holds a JSON-encoded string carrying the per-code variant tail. The rationale
for the hybrid:

- Most analytics questions ("how many issues per video?", "what fraction are
  caption-similarity issues?") only need the core columns. Those stay columnar.
- The variant tail (e.g. `clip_mp4_index_mismatch` carries field-by-field
  mismatch records; `caption_similarity_below_threshold` carries
  `{start_frame, end_frame, similarity, threshold, a, b}`) genuinely doesn't
  fit a fixed schema. A wide
  nullable schema with one column per possible detail key would require a
  schema migration on every new code and leave most rows mostly NULL.

Detail rows are **flattened at the source**: when a clip has 8 mismatched
index fields, the stage emits 8 issue rows (one per field), not one row with
nested mismatches. `group_by("field")` is a one-liner; each row is
self-contained.

`Issue` is a `TypedDict` (not an `attrs` class) — IDE autocomplete only; rows
are constructed via `make_issue(...)` and the Arrow table is the canonical
representation. There is no parallel Python class to keep in sync.

## Videos table and source roots

Alongside `issues`, the `Report` carries a second Arrow table named `videos`
plus two top-level string fields, `source_a` and `source_b`. The split is
deliberate: source roots are per-side, video keys are per-row, and storing
the full `source_video` path on every row would repeat the same prefix
hundreds of times.

```python
VIDEOS_SCHEMA: pa.Schema = pa.schema(
    [
        ("video_key", pa.string()),
        ("in_a", pa.bool_()),
        ("in_b", pa.bool_()),
    ],
)

@attrs.define(frozen=True)
class Report:
    ...
    source_a: str = ""
    source_b: str = ""
```

Consumers reconstruct a full source path as
`f"{report.source_a}{row['video_key']}"` (or `source_b`). Per-side presence
is on the row via `in_a` / `in_b` booleans, so consumers can render
one-sided videos without inspecting the issues table.

### Trust + assert: how `source_a` / `source_b` get derived

The comparator never asks the user for the source root; it derives it from
each summary's data. Algorithm per side:

1. Take the first `(video_key, source_video)` entry from `summary.videos`.
2. Strip `video_key` off the end of `source_video` to recover a candidate
   root.
3. Walk every other entry and assert that `root + video_key` reconstructs
   that row's `source_video`.

When the assertion holds, the root is shipped on the Report. When it fails
(a row's `source_video` doesn't match `root + video_key`), the comparator
emits a structured `summary_source_layout_inconsistent` issue, leaves that
side's source root as `""`, and continues with the comparison — the videos
table itself still ships (it doesn't depend on the roots). Consumers
treating `source_X == ""` as "source path unknown" handle this gracefully
without a separate failure mode.

This is a string-shape check, not an IO existence check: the comparator
never opens the source MP4. File availability is left to whoever ends up
reading the path, and surfaces naturally when `smart_open` fails to open
it.

### Why a second table instead of a column on `issues`

Source roots and per-video identity are per-video, not per-issue. Folding
them into the issues table would duplicate the same root string across
every issue row for a given video — 20k issues / 100 videos = 200x
redundancy on a multi-MB-range path field. The two-table shape keeps
`issues` flat and queryable while consumers join on `video_key` when they
need to reconstruct a source path.

### Persistence

Embedding `source_a` / `source_b` + the `videos` table in the report
keeps it self-describing — consumers don't need to round-trip to
`summary.json` to resolve source-video paths, and the report works as a
standalone audit artifact for anyone passing it around.

`report_io.write_report` writes the videos table alongside `issues` (Lance
gets a third dataset at `<path>.videos.lance`; JSON gets a top-level
`videos` array and `source_a` / `source_b` keys). Older reports without
these fields load with an empty videos table and empty source roots;
consumers that need source paths surface "source path unknown" rather than
crashing.

## Caption batching strategy

The Stage 1 caption model runs on CPU. Two batching levers are in scope; a
third is rejected:

1. **Cross-clip batching inside one batch.** `MetadataStage` is invoked via
   Ray Data `map_batches`; the actor gathers caption-window pairs from every
   clip in the batch and embeds them in one `model.encode(...)` call. Captures
   tokenizer-overhead + framework-dispatch savings across N clips at once.
   Batch size is the `config.metadata_batch_size` knob, applied directly: the
   driver derives block count as `ceil(num_rows / batch_size)`, floored at
   `metadata_workers` so every actor gets at least one block.
2. **Cross-actor parallelism (size the actor pool).** `ActorPoolStrategy(size=N)`
   runs N independent model copies on N CPU workers. Caption embedding
   parallelizes near-linearly across processes; combined with (1) this gets you
   most of the available speedup. `config.metadata_workers` defaults to
   `os.cpu_count() // 2` so other pipelines on the same box aren't starved.
3. **Per-clip intra-window batching only — not used.** Embedding the few
   caption windows of one clip per call gives up the cross-clip batch win for
   no readability benefit; (1) subsumes it.

Stage 2 carries the symmetric `config.video_index_batch_size` knob with a
smaller default — MP4 reads don't batch usefully, so the win is purely
scheduler-tail: more, smaller batches give Ray Data more rebalancing points.
Stage 2 has *no* worker-count knob: actor count is derived at run time as
`floor(cpu_count / video_index_cpus_per_worker)`. Stage 1 exposes
`metadata_workers` because each actor holds a ~200 MB caption model and so
worker count has a memory ceiling unrelated to CPU; Stage 2 actors only hold
smart_open params, so one knob (the CPU reservation) fully determines pool
size.

If caption comparison becomes the bottleneck, the higher-leverage moves are
**model-side** (smaller / quantized / distilled model) and **embedding caches**
(memoize by `(clip_id, output, prompt_hash)` so re-runs only embed what
changed) rather than rebatching the pipeline.

## Caption model

Stage 1's caption comparator uses
[`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) via
`sentence-transformers`. The relevant properties:

- ~30M params, 384-dim embeddings. Small enough to load fast and run on CPU
  at meaningful throughput.
- Pre-normalized embeddings: cosine similarity reduces to a dot product, no
  per-call normalization step needed.
- English-only — matches the project's caption pipeline output.
- bge-v1.5 does **not** require an instruction prefix for symmetric STS
  (which is what we're doing: comparing two caption strings). Embed both
  sides as-is.

Per-actor memory: ~200 MB (model + tokenizer + framework overhead). With
`metadata_workers = os.cpu_count() // 2` on a 16-core box that's ~1.6 GB total
— fine for a CI/audit host. Adjust the default if you need to share the box
with something hungrier.

The model identity is a `CaptionPolicy` knob, not a hard commitment. Swapping
in another sentence-transformers model is a one-line config change. The
`min_similarity = 0.85` default is a placeholder — tune once you have baseline
numbers from a real run.

### Model registration

The model needs an entry in `cosmos_curator/configs/all_models.json` so the
project's standard model-download path resolves it (and the NGC mirror can
publish a pinned copy). Weights are pre-downloaded to the project's local
cache via:

```bash
cosmos-curator local launch --image-name cosmos-curator -- \
  pixi run --as-is python -m cosmos_curator.core.managers.model_cli download \
  --models bge_small_en_v1_5
```

`MetadataStage` resolves the cached path via
`model_utils.get_local_dir_for_weights_name(model_id)` and loads with
`local_files_only=True` so the actor never reaches out to Hugging Face at
runtime — deterministic, network-independent comparison runs.
