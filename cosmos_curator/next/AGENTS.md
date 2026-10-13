# Curator Next — Agent Guidelines

## Direction and Scope

Curator Next is the starting point for new curation workflows: maintained reference recipes backed by a shared toolkit.
The goal is to replace Xenna and the legacy Ray Data pipelines through explicit workflow migrations.

Implementation and mirrored tests live in `cosmos_curator/next` and `tests/cosmos_curator/next`. During incubation,
schemas, identities, config shapes, and import paths have no backward-compatibility guarantee. Prefer rerunning to
building migrations for superseded development formats; reject or isolate incompatible durable state explicitly.

Change project constraints only with the workflow owner. Current defaults may change with evidence and a documented reason.
Qualification requirements apply when claiming a capability such as managed Ray compatibility or cross-run recovery;
an incremental recipe may land before it meets those requirements if its documentation states the limits.

## Project Constraints

- Use Ray Data as the primary data-parallel execution and composition layer.
- Store canonical derived tabular datasets in Lance; reference media and bulk sensor payloads in user-selected storage.
- Treat execution as at least once. Durable side effects and visible results must tolerate replay.
- Keep pipeline intent separate from deployment topology.
- Do not add dependencies on `cosmos_curator.pipelines.ray_data`; it is deprecated and scheduled for deletion.
- Keep Ray Data visible rather than introducing a Curator runtime, Xenna-style stage hierarchy, or generic DAG language.

## Workstreams and Shared Toolkit

Current workstreams include video, robot action, and AV/multimodal. They developed independently, with intentional
duplication while their contracts took shape. These groupings can change as recipes and requirements evolve.
Common components are now being extracted and accidental implementation differences reconciled. Existing duplication
alone is not a reason to broaden an unrelated change.

Inspect shared components and relevant recipes before implementing common concerns. Reuse components where contracts fit;
keep domain behavior local where reuse is uncertain. Shared APIs should preserve differences justified by data contracts
or measured performance, rather than every historical implementation choice.

Scope extractions and adoptions to the task. Document shared assumptions and guarantees; extend the contract explicitly
when semantics change, with tests for new and existing consumers.

For checkpointing and recovery, `video_transcode`, `video_embed`, and their shared components are the canonical reference.
Migrate other recipes and older recovery helpers toward that pattern, but check the
[extensions it lacks](../../docs/curator/design/curator-next-checkpoint-recovery.md#expansion-across-curator-next)
first: row-level selection (embeddings, robot-action-caption) and several field sets in one commit
(multimodal-embedding) each need an explicit extension, not a force-fit. `lance_fragment_recovery` is deprecated for new
workflows; producers need an explicit append-publication extension to shared recovery before migration.

## Recipe Contracts

Recipes normally live at `cosmos_curator/next/recipes/<recipe_name>/`. Keep recipe-specific configuration, schemas, discovery,
processing, publication, and recovery there, with mirrored tests. Follow the closest useful recipe without copying its
layout mechanically; start with a small working path through a readable Lance result.

Before scaling or declaring a recipe complete, record its contract in module documentation or a README:

- input discovery, snapshot, content identity, and immutability assumptions;
- logical work item, stable identity, and retry boundary;
- canonical Lance outputs, schema versions, and external payload naming/collision behavior;
- publication and reader-visibility boundary;
- recovery boundary and compatibility inputs, when supported;
- failure and quarantine behavior.

Use the recipe's current design notes when changing publication or recovery. When adopting shared recovery, follow the
[shared contract](../../docs/curator/design/curator-next-checkpoint-recovery.md#assumptions-and-guarantees).
Document how the recipe and deployment meet its assumptions, and test its guarantees.
Start from the
[adoption example](../../docs/curator/design/curator-next-checkpoint-recovery.md#adopting-the-reference-pattern): the
recipe's `run_config` calls `run_enrichment`, its tests subclass the shared
[conformance suite](../../tests/cosmos_curator/next/enrichment/conformance.py), and an integration test runs the shared
[fresh-cluster scenario](../../tests/cosmos_curator/next/enrichment/fresh_cluster.py). The
[`enrichment` package docstring](enrichment/__init__.py) lists the rest of the recipe-facing API. Keep processing graphs,
payload durability, and terminal-error policy recipe-owned.

## Configuration and CLI

Runnable reference recipes normally accept versioned YAML or JSON backed by strict Pydantic v2 models.

- Require `schema_version` and an exact recipe `kind` discriminator.
- Organize fields around workflow concepts; reject unknown fields and invalid combinations with useful typed errors.
- Resolve defaults, presets, and narrowly scoped `--set` overrides deterministically. Execute from the canonical resolved
  config and make it renderable as canonical JSON.
- Use the same resolver and resolved models for validation, rendering, schema generation, and execution. Keep templates
  runnable and aligned with that contract.
- Keep secrets out of rendered configs and durable metadata; reference credential profiles or runtime credentials.
- Separate result-defining settings from execution tuning. Resources, batching, retries, timeouts, and progress reporting
  normally should not affect output identities or recovery compatibility.
- Prefer a config path plus small overrides over a second long-form CLI for the same settings.

For the generic pipeline CLI, export a `PipelineKind` from the recipe's `pipeline_kind.py` and register it in
`BUILTIN_PIPELINE_KINDS` in `cosmos_curator/pipelines/builtin_pipeline_kinds.py`. Match the config discriminator exactly.
Defer config, runtime, Ray, and model imports until the selected operation needs them. Update registration, dispatch,
import-laziness, and preset tests as applicable.

Managed Ray composes deployment and recipe configs:

```text
cosmos-curator slurm ray submit CLUSTER_CONFIG -- pixi run --as-is run-pipeline RECIPE_CONFIG
```

Recipe paths must be meaningful inside the runtime environment.

## Ray Data Execution

Use standard reads and transformations (`map`, `flat_map`, `map_batches`), actor pools for reusable model state and GPU
inference, and Arrow-native tabular batches where practical. Let Ray Data handle streaming, backpressure, scheduling,
and worker retries. Driver-side setup and small metadata operations are fine; keep direct-task/actor exceptions for
coordinated global algorithms or managed replicas visible.

Managed Ray recipes must accept late workers, tolerate worker loss, and stall through temporary periods with no workers.
Do not permanently size work from a startup resource snapshot or keep required progress only in worker-local files,
actor memory, or Ray objects. Initialize model and process state for every worker or actor incarnation.
For large downloads, follow the
[`curator_io` concurrency guidance](../../docs/curator/design/ray-data.md#large-source-downloads-and-curator_io).

## Testing and Qualification

Test in proportion to maturity and risk: config resolution, CLI registration/dispatch, identities/schemas, pure processing,
publication, item failures, retry/recovery decisions, and a small Ray Data run that writes and reads Lance output.
Keep a cheap smoke config or equivalent first successful run. Follow the [root testing guidance](../../AGENTS.md#development)
for suite and marker selection.

Before claiming cross-run recovery, exercise:

1. a fresh cluster and driver recovering after durable work has completed;
2. failures around payload, checkpoint, Lance-commit, and final-status boundaries;
3. comparison with a clean reference run by logical identity and schema.

Clean and resumed runs must produce the same visible records and payload references, with no missing or duplicate logical
outputs. Managed Ray qualification additionally requires testing workers joining, leaving, or being requeued, including a
temporary period with no workers. Shared helper adoption alone does not establish qualification; test the recipe's complete
path before claiming it.

## Design References

- [Cosmos Curator Next](../../docs/curator/design/curator-next.md)
- [Managed Ray Clusters for Curator Pipelines on Slurm](../../docs/curator/design/curator-next-slurm-ray.md)
- [Schema-Validated Pipeline Configs](../../docs/curator/design/pipeline-configs.md)
- [Ray Data Design](../../docs/curator/design/ray-data.md)
