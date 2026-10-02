# Curator Next — Agent Guidelines

## Purpose

This file guides Curator Next recipe contributors. The design is incubating, with three levels of guidance:

- **Project constraints** are the current architectural boundaries. Deviate only with the workflow owner.
- **Current defaults** are preferred starting points. Change them when evidence supports a better design, and document
  the reason.
- **Qualification requirements** apply only when claiming a capability such as managed Ray compatibility or recovery
  across Ray runs.

These instructions cover new work under `cosmos_curator.next`. Existing pipelines move here only through explicit
transition work. Do not silently introduce a second execution, storage, or configuration contract.

## Project Constraints

Curator Next is recipe-first during incubation:

- Use Ray Data as the primary data-parallel execution and composition layer.
- Store canonical derived tabular datasets in Lance. Keep media and bulk sensor payloads in user-selected storage and
  reference them from Lance.
- Treat Ray execution as at least once. Durable side effects and visible results must tolerate replay.
- Keep pipeline intent separate from deployment topology.
- Incubate implementation in `cosmos_curator.next` and mirrored tests under `tests/cosmos_curator/next`; do not promise
  stable package paths yet.
- Do not add dependencies from `cosmos_curator.next` to `cosmos_curator.pipelines.ray_data`. That package is deprecated
  and scheduled for deletion.

These constraints do not require a Curator-specific runtime, stage hierarchy, or generic DAG language. A recipe may land
as an incremental local slice before it meets every production qualification requirement, but its documentation must not
claim properties it has not demonstrated.

## Recipe as the Unit of Delivery

A recipe is the unit of delivery, review, testing, benchmarking, and managed Ray qualification. New recipes normally
live at:

```text
cosmos_curator/next/recipes/<recipe_name>/
```

Keep recipe-specific configuration, schemas, processing, publication, recovery, and tests together. A useful starting
layout is:

```text
<recipe_name>/
|-- config.py          # raw and resolved config models
|-- contracts.py       # identities, schemas, and version constants
|-- discovery.py       # source discovery and normalization
|-- processing.py      # testable row or batch operations
|-- lance_sink.py      # durable Lance publication
|-- recovery.py        # optional recipe-specific recovery
|-- pipeline.py        # Ray Data assembly and execution
`-- pipeline_kind.py   # optional adapter for the generic pipeline CLI
```

The filenames are illustrative, not a required module taxonomy. Prefer a coherent vertical slice over empty layers.

Before scaling a recipe or declaring it complete, record its essential contract in module documentation or a README:

- input discovery and any snapshot, content-identity, or immutability assumptions;
- logical work item, stable identity, and retry boundary;
- canonical Lance outputs and schema versions;
- external payload naming and collision behavior;
- publication and reader-visibility boundary;
- recovery boundary and compatibility inputs, when recovery is supported;
- failure and quarantine behavior.

### Keeping Code Local

Keep domain behavior inside a recipe while reuse is uncertain. Small shared infrastructure is appropriate at an existing
project-wide seam, such as recipe registration, config tooling, storage credentials, or managed-cluster integration.
Duplication is acceptable while semantics and performance constraints are still being learned.

Extract a shared implementation when another real recipe needs it and both consumers can share its contract, tests,
dependencies, failure behavior, and performance characteristics. Treat that extraction as explicit refactoring work;
do not make the first recipe design a general toolkit speculatively.

## Configuration and CLI Defaults

Runnable reference recipes normally accept a versioned YAML or JSON config backed by strict Pydantic v2 models.

- Include `schema_version` and an exact recipe `kind` discriminator.
- Organize fields around workflow concepts rather than historical CLI flags.
- Reject unknown fields and invalid combinations with useful typed errors.
- Resolve defaults, presets, and narrowly scoped `--set` overrides deterministically.
- Execute from the canonical resolved config and make that form renderable as canonical JSON.
- Keep secrets out of rendered configs and durable run metadata; reference credential profiles or runtime credentials.
- Distinguish result-defining settings from execution-only tuning. Resource shape, batching, retries, timeouts, and
  progress reporting normally should not affect output identities or recovery compatibility.
- Prefer a config path plus small overrides over a second long-form CLI for the same settings.

For a recipe exposed through the generic pipeline CLI, define and export its `PipelineKind` from the recipe's
`pipeline_kind.py`, then register it in `BUILTIN_PIPELINE_KINDS` in
`cosmos_curator/pipelines/builtin_pipeline_kinds.py`. Use the exact config discriminator as the kind name. Keep
the adapter lightweight by deferring config, runtime, Ray, and model imports until the selected operation needs them.
Update registration, dispatch, import-laziness, and preset tests as applicable.

Keep templates aligned with the user-facing config contract. Use the same resolver and resolved config models for
validation, rendering, schema generation, and execution. Templates should provide a runnable starting point and make
defaults discoverable; their exact presentation may evolve with the CLI.

A managed Ray launcher composes a deployment config with the recipe config:

```text
cosmos-curator slurm ray submit CLUSTER_CONFIG -- pixi run --as-is run-pipeline RECIPE_CONFIG
```

Paths in the recipe config must be meaningful inside the runtime environment, not only on the submitting host.

## Data, Identity, and Publication

Lance is the canonical durable format for derived tabular data. JSON summaries, checkpoint files, and other operational
artifacts may support execution, but they do not replace the documented Lance dataset contract.

Use versioned schemas and stable logical identities for independently retryable work and published results. Payload
locations must be deterministic or collision-safe, and identity normalization must preserve distinctions that matter to
the underlying storage system. Record enough provenance to identify the input assumptions, recipe behavior, schema, and
result-defining configuration.

Assume Ray tasks may execute more than once. External side effects must be idempotent, atomically committed, or
reconciled before publication. Prefer at-least-once computation with exactly-once visible logical results.

A Lance dataset commit is atomic, but a media write plus multiple dataset commits is not. A recipe with multiple durable
outputs must define what readers may consider complete and how partial attempts are repaired. This may use deterministic
payloads, an ordered set of commits, a final marker, or another recipe-appropriate protocol.

Distinguish a logical run from a physical attempt. A fresh Ray cluster after head failure is a new attempt; attempt IDs
may help diagnostics but must not replace stable identities used for deduplication or recovery.

## Ray Data Execution Defaults

The scalable path should remain recognizably Ray Data:

- build work with Ray Data reads or datasets and standard transformations;
- choose `map`, `flat_map`, and `map_batches` according to row shape and batching needs;
- use actor pools for reusable model state and GPU inference;
- keep large tabular batches Arrow-native where practical;
- let Ray Data provide streaming execution, backpressure, scheduling, and worker-level retries.

Driver-side setup and small metadata operations are fine. Direct Ray tasks or actors may handle coordinated global
algorithms or explicitly managed distributed replicas. Keep such exceptions visible instead of hiding them behind a new
pipeline abstraction.

A recipe intended for managed Ray must not assume that the worker set observed at startup is permanent. Late workers
should be able to contribute; worker loss should reduce throughput without corrupting durable output; and a temporary
period with no workers should stall rather than invalidate the logical run. Do not permanently size work from a one-time
resource snapshot or keep required progress only in worker-local files, actor memory, or Ray objects. Initialize model
and process state for every new worker or actor incarnation.

For large-source download stages, follow the
[`curator_io` concurrency guidance](../../docs/curator/design/ray-data.md#large-source-downloads-and-curator_io).

## Recovery Direction

Recovery mechanisms remain provisional and recipe-specific; expect this guidance to evolve as more work shapes are
exercised, while the qualification outcomes below remain required for any recovery claim.

Unexpected Ray head loss ends the current managed-cluster attempt. Recovery across Ray runs means launching a fresh
head, cluster, and driver and using durable recipe-selected state; old Ray lineage and driver memory are unavailable.

A recipe claiming cross-run recovery must be able to:

- reconstruct its work set from durable input and configuration;
- identify durably complete logical work and safely repeat absent or ambiguous work;
- tolerate crashes between payload creation, progress recording, Lance publication, and driver success reporting;
- avoid missing or duplicate visible logical outputs;
- bound expensive recomputation at documented commit or checkpoint boundaries.

Scope recovery compatibility to the reusable work boundary. Include result-defining config, schema and behavior
versions, and immutable model or input revisions where relevant. Avoid invalidating compatible progress merely because
execution tuning changed. Depending on the input contract, compatibility may bind a complete input snapshot, an
individual stable item, or a durable phase output. Reject or isolate incompatible state, but no single fingerprint shape
is required.

Possible recovery mechanisms include committed Lance phase outputs, an idempotent work ledger, atomic completion
records, or deterministic recomputation. Choose the simplest mechanism that meets the recipe's replay cost and
correctness needs. For example, a completion record can be useful when it follows required side effects, contains enough
terminal state to rebuild canonical output, and is validated before reuse. Payload existence alone is usually not proof
of completion. Caching only terminal successes is a useful default when failures may be transient.

For a recipe that publishes complete snapshots, resolve membership from the current input contract rather than stale
checkpoint contents. Use checkpoints to reuse selected work, then rebuild or reconcile the snapshot so removed inputs do
not remain and retried inputs do not duplicate. Other publication semantics may require a different recovery strategy.

Ray Data job-level checkpointing is an optional implementation tool, not the Curator Next recovery contract. Verify the
pinned Ray version and complete read-to-write plan before adopting it. Do not assume it is atomic with Lance or external
payload publication; use recipe-specific durable progress when those boundaries do not compose safely.

## Testing and Qualification

Ordinary recipe development should test, in proportion to the recipe's maturity and risks:

- config validation and canonical resolution, when a config surface exists;
- generic kind registration and dispatch, when the generic CLI applies;
- stable identities and schema conformance;
- pure row or batch processing apart from Ray where practical;
- publication, item failures, retries, and recovery decisions;
- a small local Ray Data run that writes and reads canonical Lance output;
- a cheap smoke config or equivalent first successful run.

Managed Ray qualification is a stronger claim. Before making it, exercise at least:

1. workers joining, leaving, or being requeued during processing, including a temporary period with no workers;
2. a fresh cluster and driver recovering after durable work has completed;
3. failures around the recipe's actual payload, checkpoint, Lance-commit, and final-status boundaries;
4. comparison of the recovered output with a clean reference run by logical identity and schema.

The acceptance criterion is that clean and resumed runs subjected to worker and head loss produce the same visible
records and payload references, with no missing or duplicate logical outputs.

## Recommended Agent Workflow

When creating or substantially changing a recipe:

1. Read this file, the relevant design notes, and the closest existing recipe as evidence rather than a fixed template.
2. State the proposed config, work identity, output schema, publication boundary, and recovery claim.
3. Implement the smallest local vertical slice through a readable Lance result.
4. Scale the data path with idiomatic Ray Data and bounded resource controls.
5. Add failure and recovery tests before claiming qualifications; propose shared extraction separately.

## Non-Goals During Incubation

Do not make a new recipe responsible for designing the final public package layout, replacing Ray Data, recreating the
Xenna stage model, defining an arbitrary user-authored DAG, preserving every historical option, or providing transparent
high availability for the Ray head. Do not extract a general toolkit before recipe experience demonstrates its boundary.

## Design References

- [Cosmos Curator Next](../../docs/curator/design/curator-next.md)
- [Managed Ray Clusters for Curator Pipelines on Slurm](../../docs/curator/design/curator-next-slurm-ray.md)
- [Schema-Validated Pipeline Configs](../../docs/curator/design/pipeline-configs.md)
- [Ray Data Design](../../docs/curator/design/ray-data.md)
