# Split Output Comparison Design

## Summary

Split output comparison validates whether two runs of a split video pipeline produced equivalent output. The comparison
starts from each output root's `summary.json`, then loads artifact evidence only for features that cannot be decided from
summary accounting alone.

The target architecture is a staged comparison pipeline with separate units of work for planning, artifact IO, model
compute, and reporting:

- video keys are the planning and final report grouping unit;
- clips are the artifact loading and structure validation unit;
- caption/window pairs are the model batching unit for embedding-based checks.

This keeps report semantics video-oriented while avoiding the long-tail behavior of assigning all artifact work for a
large video to one Ray task.

## Goals

- Keep summary accounting comparison separate from artifact-level feature comparison.
- Avoid video-row long tails when one video has many clips.
- Load per-clip artifacts once per output side, then reuse the normalized views across comparison stages.
- Give IO-heavy stages, cheap CPU stages, and model-backed stages independent Ray resource strategies.
- Load embedding/comparison models once per persistent actor, not once per clip or video.
- Preserve a report shape that can add feature-level results over time.
- Keep feature-specific business rules out of the generic executor.

## Non-Goals

- Treating missing feature output as automatically skipped. If a feature is enabled or present on one side, comparison
  should fail or pass based on the feature's own rules and counts.
- Having the executor understand caption-specific, embedding-specific, or clip-specific business rules.
- Making one video row own all artifact IO for every clip in that video.

## Terminology

### Video Comparison Spec

`VideoComparisonSpec` is the planning and reporting unit. It identifies one summary video key and the clip UUIDs present
under each output root.

Video specs are built on the driver from the two loaded summaries. They are useful for:

- video set comparison;
- clip set expansion;
- report grouping;
- targeted debugging with a single video key.

### Clip Comparison Spec

`ClipComparisonSpec` is the artifact IO unit. It represents one video key plus one clip UUID, with output A and output B
presence and metadata paths.

Clip specs should be produced by expanding video specs. This lets large videos with hundreds of clips spread across the
cluster instead of creating one slow video task.

### Caption View

A caption view is the normalized caption evidence built from per-clip metadata. It should contain the parsed windows,
missing/invalid metadata state, and enough identity fields to tie results back to the video and clip.

Caption structure checks and future embedding-based checks should consume the same caption view. They should not
each parse the raw metadata independently.

### Caption Pair

A caption pair is a comparable caption/window pair derived from normalized caption views. Caption pairs are the natural
batching unit for model-backed embedding comparison.

### Feature Stage

A feature stage owns comparison logic for one output feature or one phase of a feature, such as caption structure or
embedding-based caption comparison. A feature stage knows how to:

- inspect summaries and decide whether work is needed;
- consume prepared rows or views at the right granularity;
- emit compact row-level results;
- reduce row-level results into report-level issues and metrics.

The generic feature comparison pipeline should route rows and resources. It should not own feature business rules.

## Target Flow

```text
compare_split_outputs(...)
|
|-- load summary.json for output A and B
|-- compare summaries
|-- compare features
|   |
|   |-- build video specs from summaries
|   |-- expand video specs into clip specs
|
|-- Ray stage: clip metadata/artifact load
|   |
|   |-- input: ClipComparisonSpec
|   |-- resource shape: IO-bound, high concurrency
|   |-- storage clients: persistent per actor or otherwise reused
|   `-- output: ClipArtifactResult / normalized CaptionView input
|
|-- Ray stage: caption structure preparation and validation
|   |
|   |-- input: loaded clip artifact rows
|   |-- resource shape: cheap CPU
|   |-- build normalized CaptionView once
|   |-- emit structure issues/counts
|   `-- emit comparable CaptionPair rows for model-backed checks
|
|-- Ray stage: embedding-based caption checks
|   |
|   |-- input: batches of CaptionPair rows
|   |-- resource shape: persistent ActorPoolStrategy
|   |-- actor __init__: load model once
|   `-- output: embedding comparison rows/counts/issues
|
`-- reduce phase
    |
    |-- group clip and pair results by video key
    |-- reduce feature rows into feature-level comparisons
    |-- merge with summary-rule issues
    `-- write ComparisonReport
```

## Why Clip Rows Are the Artifact Unit

Videos are uneven. A small video may have a few clips, while a large video can have hundreds. If one Ray row owns an
entire video, a large video can become a long-tail task.

For example, a 480-clip video can require up to 960 metadata JSON reads when comparing output A and output B. If those
reads happen sequentially inside one video row, runtime is dominated by small-object storage latency. Clip-level rows
allow those reads to spread across workers and make the pipeline less sensitive to a single large video.

The first clip-row caption comparison implementation showed a roughly 5-10x manual runtime improvement over the
video-row metadata loading path on a real split-output comparison. This is not a formal benchmark, but it confirms that
the long-tail metadata IO issue was material enough to justify keeping artifact-heavy feature comparisons clip-oriented.

Video-level reporting still matters, so clip rows should carry `video_key` and reducers should assemble final
video-level and feature-level results.

## Why Caption Pairs Are the Model Unit

Embedding-based caption checks add model setup and compute. The model should be loaded by persistent Ray actors, not by
per-row task functions.

The model-backed stage should use `ActorPoolStrategy` with a callable class:

```python
class CaptionEmbeddingWorker:
    def __init__(self, model_config):
        self._model = load_model(model_config)

    def __call__(self, batch):
        return compare_caption_pairs(self._model, batch)
```

This gives the embedding stage:

- one model load per actor;
- explicit CPU/GPU resource ownership;
- batching across clips and videos;
- a clean boundary between structure validation and semantic/model comparison.

## Resolved Work

Some feature results can be decided from summaries alone. A resolved result means artifact IO cannot add information
needed to decide the feature result.

Caption examples:

- Neither output has caption evidence in the summaries. The caption feature passes with zero counts; loading metadata
  would not change the result.
- One output has caption evidence and the other does not. The caption feature fails with `caption_presence_mismatch`;
  loading metadata cannot turn that into a pass.

Resolved work should remain driver-side and should not enter Ray artifact stages.

## Artifact Work

Artifact work means summaries are not enough. The pipeline must load artifact evidence and reduce row-level results.

Caption example:

- Both outputs have caption evidence. The comparison needs `metas/v0/{clip_uuid}.json` for relevant clips so it can
  compare caption windows and later build comparable caption pairs.

Current limitation: caption artifact loading supports only per-clip JSON metadata at `metas/v0/{clip_uuid}.json`.
Outputs written with `--upload-clip-info-in-chunks` (`metas_jsonl/v0`) or `--upload-clip-info-in-lance` are not loaded
for caption window comparison yet. When those outputs have caption counts in `summary.json`, the caption feature can
report `caption_data_missing` because the expected per-clip JSON metadata is absent.

Artifact requirements should remain attached to the specific work item or stage, not to a static comparator class. A
feature may need different artifacts depending on summaries and configuration.

## Driver and Worker Responsibilities

The driver owns:

- loading summaries;
- resolving comparison policy/configuration;
- building video specs;
- expanding video specs into clip specs;
- routing resolved work vs. artifact/model work;
- reducing row-level results into final report data.

Ray workers own:

- loading per-clip artifacts;
- building normalized per-clip views;
- running structure checks;
- running model-backed checks in persistent actors when needed;
- returning compact JSON-serializable rows for reduction.

## Resource Strategy

Different stages should use different execution strategies:

| Stage | Workload | Ray strategy |
| --- | --- | --- |
| Summary comparison | Driver CPU, cheap | Driver-side |
| Clip metadata load | IO-bound small-object reads | `ActorPoolStrategy` workers with persistent storage clients |
| Caption structure validation | Cheap CPU | Task map or fused with load stage |
| Embedding-based caption comparison | Model-backed compute | `ActorPoolStrategy`, batch-oriented |
| Reduce/report | Aggregation | Driver-side by default |

Metadata load actors may also use a small bounded thread pool when processing batches of clip rows. This is an
optimization for small-object storage latency, not an invitation to issue unbounded concurrent reads. The concurrency
limit should be configurable.

The exact split between task and actor stages can evolve, but storage-client and model-backed stages should use
persistent actors so setup is amortized across many rows.

## Diagnostic Selectors

The CLI should support scoped runs for debugging:

- `--limit N`: compare the first N video keys from output A against output B;
- `--video-key KEY`: compare one exact summary video key.

These selectors scope artifact/model work. Summary accounting can still run over the full summaries unless a future
debug mode explicitly asks to suppress it.

When a selector is active, feature reducers must scope expected artifact counts to selected videos/clips so the report
does not claim unselected data is missing.

## Report Shape

Feature results are reported under `feature_comparisons`:

```json
{
  "feature_comparisons": {
    "captions": {
      "status": "passed",
      "metrics": {}
    }
  }
}
```

Issues can include a feature name, video key, clip UUID, and field so report consumers can group failures by feature and
artifact level.

Metrics should use a common envelope with feature-specific contents. `status` is common. Metrics are a dictionary so
each feature can report natural counts without forcing premature schema alignment.

Common count names such as `items_compared`, `items_failed`, or `clips_compared` are useful when they fit the feature,
but features may add their own metrics for caption windows, embeddings, previews, motion filtering, aesthetic filtering,
or output clips.

## Model-Backed Configuration

Model-backed comparison configuration should be feature-owned. The generic Ray pipeline can consume generic resource
fields, such as CPU count, GPU count, actor count, and batch size. It should not understand model names, tokenizer
details, embedding dimensions, or feature-specific thresholds.

Example shape:

```python
CaptionEmbeddingComparisonConfig(
    model_id="...",
    batch_size=128,
    similarity_threshold=0.95,
    resources=StageResources(cpus=2, gpus=1),
)
```

The feature stage interprets this config. The executor only uses the generic resource fields needed to assemble the Ray
stage.

## Stage Fusion Decisions

Artifact loading should be fused with normalized view construction. Raw metadata JSON is not useful after parsing, and
passing it through Ray as an intermediate object would add serialization overhead without making later stages clearer.

Caption structure validation can either be fused with view construction or run as the immediate next cheap CPU stage.
The stronger boundary is between structure/view work and model-backed embedding comparison. Model-backed comparison
should remain a separate actor-backed stage so it can own different resources, batching, and persistent model state.

## Reduction Decisions

Reducers should stay driver-side initially. The expected result rows are compact compared with the input artifact
volume, and driver-side reduction keeps ordering and report generation deterministic.

Ray group/reduce should be introduced only if measured result volume makes driver-side reduction a bottleneck.

## Implementation Direction

The first staged implementation should introduce clip rows directly rather than extracting caption views inside the old
video-row executor as an intermediate step.

The original video-row executor was useful for proving feature comparison semantics, but it preserved the long-tail
problem for large videos. A video with hundreds of clips still assigned all metadata IO to one Ray row. Introducing
caption views inside that shape would have improved code reuse, but it would not have addressed the execution issue that
motivated the pipeline pivot.

Clip rows should be the first implementation target:

- expand `VideoComparisonSpec` into `ClipComparisonSpec` rows on the driver;
- load each clip's output A/B metadata independently;
- build normalized per-clip caption views during artifact load;
- emit compact clip-level structure results and comparable caption pairs;
- reduce clip-level rows back into video-level and feature-level report results.

This is a larger first refactor, but it aligns the implementation with the target architecture and avoids a short-lived
intermediate design.

## Module Direction

The current modules can evolve toward the staged design without changing the public report shape:

- `video_planning.py`: keep shared video/clip spec construction and video comparison result types.
- `compare_features.py`: own Ray-specific feature comparison pipeline assembly.
- caption modules: split reusable caption view construction from structure comparison and future model-backed checks.
- artifact modules: expose clip-level loaders and avoid video-sized artifact bundles as the primary execution unit.

The important boundary is that feature stages consume prepared views and emit row results; they should not each reload
or reparse the same artifact evidence.

The current readability refactor keeps a simpler caption-only `ClipFeaturePlan` shape where each clip feature owns its
load/normalize worker. That is acceptable for caption structure comparison as the only clip feature, but the next
metadata-backed feature should introduce shared clip loading or grouping by artifact/load configuration so caption,
motion, aesthetic, and similar metadata-backed checks do not repeat the same object-store reads.

## Open Questions

- What default metadata read concurrency should each actor use for object storage?
- Which caption embedding/comparison model and thresholds should be used first?
