# Ray Data Design

## Summary

This document captures the Ray Data execution architecture used by Cosmos Curator. Curator Next uses Ray Data as its
primary data-parallel execution and composition layer, with new work developed as recipe-owned vertical slices using Ray
Data's native primitives (`with_column`, `map`, `flat_map`, `map_batches`). The deprecated
`cosmos_curator.pipelines.ray_data` package remains available during transition but is not the extension point for new
work. Xenna pipelines continue to run independently until explicitly transitioned.

## Why Ray Data

Cosmos-Xenna implements a custom streaming execution engine — actor pools, queues, auto-scaling, backpressure, work
stealing — on top of Ray Core primitives. This engine works but is proprietary, hard to debug externally, and does not
benefit from upstream Ray improvements.

Ray Data provides the same capabilities (streaming execution, actor pools, backpressure, auto-scaling, fault tolerance)
as a first-party Ray library. Building pipelines on Ray Data offers:

- **Reduced maintenance**: Upstream Ray team maintains the execution engine.
- **Better observability**: Native Ray Dashboard integration for Data pipelines.
- **Arrow-native block format**: Ray Data's internal block format is Arrow. Stages that use
  `map_batches(batch_format="pyarrow")` operate on `pa.Table` directly, enabling columnar ops and zero-copy between
  stages. Intermediate results can be checkpointed to Lance/Parquet with no conversion.
- **Ecosystem compatibility**: Ray Data datasets feed directly into Ray Train, Ray Serve, etc.
- **Operator fusion**: Ray Data automatically fuses adjacent stateless stages into a single task, eliminating plasma
  round-trips where resources align.

---

## Architecture

### Core principle: separate implementations, shared models

Ray Data pipelines are written independently of Xenna pipelines. Each is idiomatic to its own execution engine.
Shared code is limited to model/inference logic, common utilities, and runtime environment support — pipeline
orchestration (data flow, batching, stage wiring) is implemented separately.

```
              ┌───────────────────┐    ┌────────────────────────┐
              │ Xenna pipeline    │    │ Ray Data pipeline      │
              │ CuratorStage +    │    │ map / flat_map /       │
              │ PipelineTask      │    │ map_batches on dicts   │
              └────────┬──────────┘    └───────────┬────────────┘
                       │                           │
                       └──────────┬────────────────┘
                                  │
                       ┌──────────▼─────────┐
                       │  Model / inference │
                       │  code (shared)     │
                       └────────────────────┘
```

- **Lightweight transforms** (text normalization, filters) use column expressions (`ds.with_column`).
- **Per-row transforms** (file I/O, metadata extraction) use `ds.map(fn)` with dict rows.
- **Fan-out / fan-in** (splitting one video into N clips) use `ds.flat_map(fn)`.
- **Stateful stages** (GPU inference) are classes with `__init__` for model loading and `__call__` for inference,
  passed to `ds.map_batches(Class, batch_format="pyarrow")` which manages the actor lifecycle and passes `pa.Table`
  batches.
- **Xenna pipelines** continue to use `CuratorStage`, `PipelineTask`, and `run_pipeline()` unchanged.

### Primitive reference

Ray Data *is* the execution engine. A Ray Data pipeline is a chain of Dataset transforms — use the primitive
that fits each operation:

| Primitive     | Cardinality | Use when                                                             |
|---------------|-------------|----------------------------------------------------------------------|
| `with_column` | 1:1         | Vectorized column transforms (string ops, arithmetic)                |
| `map`         | 1:1         | Per-row transforms with side effects (file I/O, metadata extraction) |
| `flat_map`    | 1:N         | Fan-out (one video row → N clip rows after transcoding)              |
| `map_batches` | N:M         | Batch operations, stateful GPU inference (actor lifecycle)           |

```python
ds = ray.data.from_items([{"prompt": p} for p in prompts])
ds = ds.with_column("prompt", col("prompt").str.lower())
ds = ds.map_batches(GPT2Predictor, batch_size=1, batch_format="pyarrow",
                    num_gpus=0.8, compute=ActorPoolStrategy(size=1),
                    runtime_env=PixiRuntimeEnv("default"))
ds.show()
```

Resource declarations (`num_gpus`, `num_cpus`), batch size, actor pool sizing, and runtime environments are kwargs on
these operations — not stage properties. The processing function doesn't need to know it's running in a pipeline.

### Shape decision: `map` + list columns vs. `flat_map`

When a stage produces N artifacts per input row (e.g. N clips per video), there are two shapes:

1. **`map` + parallel list columns.** Keep one row per source unit (video); emit per-sub-row artifacts as parallel
   list columns (`clip_uuids: list<string>`, `clip_bytes: list<large_binary>`, …). Mirrors how the Xenna pipeline
   stores clips on the `Video` object.
2. **`flat_map`.** Fan out to one row per sub-unit (clip). Downstream stages operate at clip granularity.

**Pick `map` when:**

- Inputs are short and numerous (many YouTube-style videos with moderate clip counts).
- Per-source-unit atomicity matters (commit all clips of a video together).
- Downstream reductions naturally align to the source unit (e.g. per-video summary, per-video manifest).

**Pick `flat_map` when:**

- Inputs are long with many sub-units (multi-camera sessions with thousands of clips per session). A single huge list column
  per row creates memory pressure and blocks Ray Data's streaming block sizing.
- Sub-unit counts vary significantly across rows (heavy skew). Row-level parallelism distributes work more uniformly.
- Downstream stages operate per-sub-unit (per-clip GPU inference). Fan-out once early, reuse the shape throughout.
- Per-source-unit atomicity can be recovered with a terminal `groupby` or side-channel state.

For cosmos-curator, `flat_map` is the safer default because the pipeline must handle multi-camera sessions in addition to
short videos. The current splitting pipeline uses `flat_map` at the transcode stage.

### Arrow as the internal block format

Ray Data's internal block format is Arrow. This means:

- `map_batches(batch_format="pyarrow")` hands the fn a `pa.Table` and expects one back — no Arrow↔dict conversion
  per row. Natural fit for GPU inference (batch a column of prompts through a model) and vectorized compute.
- `ds.map(fn)` / `ds.flat_map(fn)` use dict rows. Ray Data handles the Arrow↔dict materialization internally; when
  two dict-based ops are fused, the dict flows between fns without intermediate Arrow round-trips.
- Any stage boundary can be materialized to Lance or Parquet with zero conversion, enabling checkpointing and
  debugging (inspect the Arrow block between stages).

Stage schema is **documented, not mechanically enforced**. Each stage's input and output columns are described in its
docstring. Construction-time validation of the full pipeline against the initial dataset schema is a latent future
improvement — it would require either a projection-pushdown hook from Ray Data's planner or an application-level
schema-declaration layer. Neither exists today; adding one preemptively would grow API surface without a concrete
optimization to unlock.

### Relationship to existing components

| Component                   | Role                                                               |
|-----------------------------|--------------------------------------------------------------------|
| `CuratorStage` interface    | Used by Xenna pipelines only. Not used by Ray Data pipelines.      |
| `CuratorStageSpec`          | Used by Xenna pipelines only.                                      |
| `PipelineTask` / data model | Used by Xenna pipelines only. Ray Data pipelines use dicts.        |
| `run_pipeline()`            | Xenna entry point. Ray Data pipelines have their own entry points. |
| `PixiRuntimeEnv`            | Shared. Passed as `runtime_env` kwarg to `map_batches`.            |
| Model download              | Shared. `download_models()` is backend-independent.                |
| Model / inference classes   | Shared. The main code reused across both engines.                  |

---

## Building Ray Data Pipelines

New Ray Data pipelines are built one at a time, starting with simple examples and progressing to production workloads.
For pipelines that already have a Xenna implementation, the Ray Data version can be validated against it. Where Ray
Data proves to be the better fit, it may replace the Xenna version; where Xenna works better, it stays.

### S3-compatible object storage (GCS)

An `s3://` location may resolve through a storage profile to Google Cloud
Storage's S3/XML endpoint, `storage.googleapis.com`. PyArrow's embedded AWS
client enables optional response-checksum validation that this endpoint does not
implement compatibly. Curator therefore selects
`AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` for that exact endpoint before
constructing the driver filesystem and propagates the same value through the Ray
job runtime environment before workers reconstruct the filesystem.

The policy is endpoint-scoped: AWS S3 and other compatible endpoints retain
their SDK defaults. A recipe that needs this policy must initialize the Ray job
through `ensure_ray_initialized` before submitting work; an already initialized
job cannot have its runtime environment changed.

### Code sharing

Model/inference code (tokenizers, GPU kernels, model weight loading) and common utilities (runtime environments,
storage I/O) are shared between both engines. Pipeline orchestration (how data is read, batched, passed between stages,
and written) is implemented separately — the orchestration is what differs between Xenna and Ray Data, and forcing it
through a shared interface constrains both sides without meaningful benefit.

### Example: column expression

```python
ds = ds.with_column("prompt", col("prompt").str.lower())
```

### Example: per-row transform (`map`)

For 1:1 transforms that need Python logic (file I/O, metadata extraction):

```python
def read_video(row: dict) -> dict:
    video_bytes = storage_utils.read_bytes(row["video_path"])
    metadata = extract_video_metadata(video_bytes)
    return {**row, "video_bytes": video_bytes, "duration_s": metadata.video_duration}


ds = ds.map(read_video)
```

### Example: fan-out (`flat_map`)

For 1:N transforms where one input row produces multiple output rows:

```python
def transcode(row: dict) -> list[dict]:
    # Write video to temp file, run FFmpeg for each clip, return N clip rows
    ...
    return [
        {"video_path": row["video_path"], "clip_uuid": uid, "clip_bytes": data}
        for uid, data in clips
    ]


ds = ds.flat_map(transcode, num_cpus=4)
```

### Example: stateful stage (`map_batches` — GPU inference)

For stages that need model lifecycle management, a class with `__init__` (model loading) and `__call__` (inference) is
passed to `map_batches`:

```python
class GPT2Predictor:
    def __init__(self):
        self._model = GPT2()
        self._model.setup()

    def __call__(self, batch: pa.Table) -> pa.Table:
        outputs = [self._model.generate(p) for p in batch["prompt"].to_pylist()]
        return with_column(batch, "output", pa.array(outputs))


ds = ds.map_batches(GPT2Predictor, batch_size=1, batch_format="pyarrow", num_gpus=0.8,
                    compute=ActorPoolStrategy(size=1))
```

### vLLM captioning via Ray Data LLM

The first Ray Data captioning path uses `ray.data.llm` to own the vLLM engine actors, GPU scheduling, and continuous
batching. The pipeline still reuses the existing Xenna/Qwen frame preparation and adapts the resulting vLLM inputs into
Ray Data rows.

See [Ray Data Captioning Design](ray-data-captioning.md) for the detailed architecture, tradeoffs, and known technical
debt.

### Curator Next direction

[Curator Next](curator-next.md) uses Ray Data as its primary data-parallel execution layer. New work develops as
recipe-owned vertical slices rather than as a feature-for-feature migration of Xenna pipelines or continued expansion of
`cosmos_curator.pipelines.ray_data`. Existing pipelines move into the Curator Next incubation namespace only through
explicit transition work.

---

## Pixi Multi-Environment Support

Stages that require specific Pixi environments (e.g. `default`, `legacy-transformers`) pass `runtime_env=PixiRuntimeEnv(name)`
as a `map_batches` kwarg. This is the same `PixiRuntimeEnv` already used by the Xenna path — it is a
`ray.runtime_env.RuntimeEnv` subclass that configures `pixi run` as the Python executable.

---

## Open Questions

- **Multi-node model download**: The Xenna path uses `_prepare_to_run_pipeline()` / `download_models()` to distribute
  model weights across nodes before pipeline execution. Ray Data pipelines need an equivalent pre-step. For
  single-node, model download happens in the actor `__init__`. Multi-node download coordination is deferred.
- **Multi-node data locality**: Xenna tracks which node produced each ObjectRef and tries to co-locate downstream
  work. Ray Data has its own locality-aware scheduling but the heuristics differ. Performance comparison for large
  payloads (encoded video bytes) is needed.
- **Projection pushdown / column-dep hints**: Ray Data currently has no planner hook for user operators to declare
  which columns they read. Absent this, stages always see the full block; a stage that ignores a large `video_bytes`
  column still pays for it to cross any non-fused operator boundary in its input. Worth filing an upstream feature
  request once the use case is well-exercised.
