# Deprecation and Default Changes

## Summary

Cosmos Curator still carries legacy defaults and compatibility paths that are no longer clearly useful. These should be
reconsidered when they make large OSS runs harder to configure, operate, or consume.

The recommended direction is:

- make schema-validated pipeline config files and packaged presets the primary input interface;
- make scalable tabular metadata and embedding output the default for cluster runs;
- prune obsolete built-in model variants;
- keep compatibility artifacts available, but make them explicit;
- investigate whether caption windowing can be removed from user-facing output contracts;
- merge the common GPU-capable Pixi environment into the default environment.

This document is a proposal and migration plan. It should be updated as owners confirm which downstream consumers still
depend on legacy artifacts.

## Decision Rules

In this document, "owners" means the team members accountable for the behavior: product ownership for user-facing
contract decisions, and engineering ownership for implementation, reliability, and maintenance.

Prefer deprecation when a feature meets several of these conditions:

- It is not covered by CI, regular performance benchmarks, or an identified owner.
- It has no identified first-party consumer, documented external consumer, or recent user report after a reasonable
  audit.
- It creates many small files or expensive cleanup behavior on shared filesystems.
- It spends scarce GPU memory or scheduling slots on work that is not the run bottleneck.
- It duplicates another output format or backend with worse operational behavior.
- It exists only for an internal workflow that OSS users cannot exercise.

Do not remove a feature only because it looks old. Keep it when it has a documented quality reason or a clear debugging
role that is hard to replace. When consumer ownership is unclear, treat that as an audit question rather than proof that
the feature is unused.

## Recommended Changes

### 1. Make schema-validated configs the primary pipeline input

**Recommendation:** Treat JSON/YAML config files as the primary way to configure pipelines. Keep a minimal CLI for
config execution, validation, inspection, preset selection, and small overrides. Freeze and deprecate the broad
pipeline-flag surface after schema parity.

**Current behavior:**

- The video `run_pipeline` CLI can already accept a JSON/YAML config file as the sole positional argument.
- The loader accepts either a nested shape with `pipeline` and `args`, or a flat top-level mapping.
- Some integrations already use JSON payloads instead of the broad CLI flag surface.
- The main split pipeline still exposes a very large argparse surface, and new functionality tends to add more flags.
- Config mode is a pass-through mapping into existing arguments; it is not yet the stable, schema-validated contract.

**Motivation:** The CLI has become an unscalable input surface. It is hard to discover, hard to review in MRs, hard to
reproduce, and increasingly difficult for agents or orchestration layers to construct correctly. A schema-validated
pipeline config is the input-side counterpart to Lance on the output side: structured, inspectable, versionable, and
amenable to compatibility checks.

**Compatibility plan:**

1. Define versioned schemas for split, dedup, shard, and future Ray Data pipeline configs.
2. Make config validation produce typed errors for unknown fields, invalid combinations, and deprecated fields.
3. Support layered config resolution: packaged defaults, packaged presets, user config, and CLI overrides.
4. Add commands to render and validate the resolved config, including preset expansion, before execution.
5. Keep a small CLI surface around config files:
   - `validate <config>`;
   - `render <config> [--set path=value]`;
   - `run <config>`;
   - `schema <pipeline>`;
   - `presets list/show`.
6. Stop adding long-tail pipeline knobs as top-level argparse flags.
7. Deprecate the broad per-pipeline flag surface once config mode reaches parity.

This aligns with the Orca design's JSON-in / JSON-out contract. Orca workload definitions can point at pipeline config
files or config fragments instead of rendering hundreds of individual flags.

### 2. Make Lance the scalable metadata default

**Recommendation:** Move large-run metadata output to Lance by default after updating built-in readers. Keep per-clip
JSON as a debug and compatibility mode.

**Current behavior:**

- `--upload-clip-info-in-lance` defaults to `False`.
- When Lance or JSONL chunked metadata is enabled, `ClipWriterStage` disables per-clip metadata JSON by setting
  `_emit_per_clip_metadata = not (upload_clip_info_in_chunks or upload_clip_info_in_lance)`.
- The help text says Lance "also" stages metadata, but the implementation treats Lance as a replacement for
  `metas/v0/{clip_uuid}.json`.
- Some consumers still assume per-clip JSON:
  - `cosmos-curator view` reads `metas/v0/`.
  - split output comparison caption loading supports only `metas/v0/{clip_uuid}.json`.
  - reference docs present `metas/v0/` as the standard metadata layout.

**Motivation:** One JSON file per clip is tolerable for smoke tests but painful for performance runs and cleanup on
metadata-sensitive filesystems such as Lustre. Lance reduces small-file pressure and gives downstream tools a tabular
format that is easier to query, scan, and compact.

**Compatibility plan:**

1. Add a single output-format option, for example `--metadata-output-format {lance,json,jsonl}`, and map existing flags
   to it with deprecation warnings.
2. Update first-party readers to support Lance before flipping the default:
   - `cosmos-curator view`;
   - split output comparison;
   - caption quality and summary inspection tools;
   - any benchmark scripts that inspect clip metadata.
3. Make Lance the default for cluster/performance presets.
4. Keep `json` as an explicit compatibility mode for small tests, manual inspection, and external tools.

**Open question:** Do any downstream training or dataset ingestion jobs require `metas/v0/*.json` directly? If yes, keep
`json` as a supported format until those consumers can read Lance or a generated export.

### 3. Prune obsolete model variants

**Recommendation:** Remove built-in model variants that are obsolete, overlapping, or not part of the recommended
operating modes. Keep only the default inexpensive fallback and the current recommended model family for each backend or
use case. Remove tests and docs for pruned variants instead of preserving coverage for unsupported choices.

**Current behavior:**

- Multiple Qwen generations and sizes are exposed as named captioning variants.
- Cosmos reasoning models are exposed by generation-specific names such as `cosmos_r1` and `cosmos_r2`, with another
  unified Cosmos generation expected.
- Variant keys appear in CLI choices, config files, model ID registries, docs, tests, and downloader paths.
- Some variants exist because they were once useful examples or integration-test targets, not because they are current
  recommended options.

**Motivation:** Every named model variant becomes user-facing API surface. It creates configuration choices users must
understand and maintainers must keep working across downloads, environment constraints, CLI validation, docs, and tests.
Keeping obsolete variants increases maintenance cost without giving users a clearer path.

**Compatibility plan:**

1. Inventory built-in Qwen, Cosmos, Nemotron, embedding, and LM variants exposed through CLI/config/model registries.
2. Choose the small supported set per use case: a default inexpensive fallback, the recommended quality model, and any
   hardware-specific quantized variant that has a current operating reason.
3. Remove obsolete variant keys from CLI choices, model ID registries, docs, and tests.
4. Keep compatibility aliases only for variants with known active usage, and warn for one release before removal.
5. Prefer explicit model-ID or plugin configuration for experimental checkpoints instead of adding named built-in
   variants.

**Open question:** Which model variants have current benchmark ownership or known production usage that justifies keeping
them as built-in choices?

### 4. Investigate whether caption windowing can be removed

**Recommendation:** Treat windowing as a simplification candidate. Audit first-party readers, writers, and
training/export consumers, then remove window-level output contracts where they are not required. Keep internal
caption-batching windows only if model execution still needs them.

**Current behavior:**

- `Clip` has both `windows` and `filter_windows`.
- vLLM prep creates windows and stores per-window model inputs.
- captioning, caption quality flags, token accounting, preview generation, filtering, metadata writing, and
  Cosmos-Predict dataset export all use per-window rows.
- Tests explicitly cover multi-window clips and count caption windows separately from clips.

**Motivation:** Windowing may be carrying more user-facing schema and artifact complexity than the current pipeline
needs. If default TransNetV2 split/caption runs usually produce one caption window per clip, exposing windows as a
general output contract makes metadata, summaries, and downstream readers harder to understand without much benefit.
The remaining question is whether model batching or training/export paths still require multi-window artifacts.

**Investigation plan:**

1. Inventory first-party readers, writers, tests, summaries, and training/export paths that consume `clip.windows`,
   `filter_windows`, window-level videos, captions, or T5 embeddings.
2. Confirm whether caption model execution still needs internal batching windows for long clips.
3. If no downstream contract requires window-level output, expose per-clip metadata as the default view and keep any
   internal windows out of user-facing artifacts.
4. If some consumers still require windows, isolate them behind explicit exports or nested Lance columns rather than
   making every reader reason about window rows.

**Open question:** Which Cosmos training/export paths still consume window-level videos, captions, or T5 embeddings?
Those consumers decide whether windowing can be removed from outputs or only hidden from the default output contract.

### 5. Make "all captions" JSON opt-in

**Recommendation:** Keep `--write-all-caption-json` as an explicit opt-in artifact.

**Current behavior:** The split summary writer builds `v0/all_window_captions.json` from processed-video chunk records
when `write_all_caption_json` is true. The CLI default is false, and `--write-all-caption-json` opts in.

**Motivation:** A single aggregate caption JSON is convenient for manual inspection, but it is another legacy export
that can grow large and duplicates information already present in metadata. In a Lance-default world, this should be an
export command or explicit compatibility artifact, not default output.

**Compatibility plan:** Remove the old negative flag and document how to generate the aggregate from Lance when needed.

### 6. Treat per-clip embedding pickles as compatibility output

**Recommendation:** Prefer grouped Parquet/Lance embedding output for large runs. Keep per-clip embedding pickles only
for explicit compatibility or debugging.

**Current behavior:** With default per-clip metadata mode, the writer can emit one embedding pickle per clip. In chunked
or Lance metadata modes, it buffers embeddings for grouped Parquet output; Lance metadata rows can also carry the
embedding vector when embeddings are generated.

**Motivation:** Per-clip embedding pickles have the same small-file problem as per-clip metadata JSON and are harder to
query than tabular embeddings. They should not be the scalable default.

**Compatibility plan:** Couple this with the metadata output format work: `json` mode can keep old per-clip embedding
artifacts, while `lance` and `jsonl` modes should use grouped embeddings.

### 7. Merge the `unified` Pixi environment into `default`

**Recommendation:** Fold the common GPU-capable `unified` environment into `default`. Keep specialized environments
separate only where they protect known dependency conflicts or heavyweight stacks.

**Current behavior:** `unified` is effectively `default` plus vLLM, CVCUDA, PyNvVideoCodec, PaddleOCR CPU dependencies,
CUDA development libraries, and small support packages. Many advanced stages and GPU workflows select it explicitly with
`conda_env_name == "unified"`.

**Motivation:** The split between `default` and `unified` makes local setup, Ray Data execution, and stage environment
selection harder to explain. A richer `default` environment better matches the common video annotation workflow and
removes many `conda_env_name == "unified"` special cases.

**Compatibility plan:**

1. Move the common dependencies currently supplied by `unified` into `default` and refresh lockfiles/images.
2. Remove or alias `conda_env_name == "unified"` stage declarations that no longer provide isolation.
3. Validate default CPU tests, local smoke tests, and GPU annotation tests against the merged environment.
4. Track dependency churn, solve time, and image size after the merge.
5. Reevaluate the separate `paddle-ocr` environment after the merge; keep `legacy-transformers`, `cuml`, `seedvr`, and
   `sam3` separate where they protect known dependency conflicts or heavyweight stacks.

## Suggested Rollout

1. Publish this document and collect owner feedback for the open questions.
2. Add versioned pipeline config schemas, resolved-config validation, and packaged preset support.
3. Prune obsolete built-in model variants and remove their docs/tests.
4. Merge `unified` into `default` and update stage environment declarations.
5. Add `--metadata-output-format`, map old metadata flags to it, and warn when users rely on old metadata flags.
6. Update first-party readers to support Lance metadata.
7. Flip scalable config presets to Lance and grouped embeddings.
8. Audit window consumers and remove or hide window-level output contracts where they are not required.
