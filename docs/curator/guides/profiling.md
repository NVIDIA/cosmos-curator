# Profiling Guide

Deep-dive into the automatic stage profiling subsystem for
Cosmos Curator pipelines.  This guide covers the architecture,
backend internals, file naming, artifact delivery, and error
handling.

- [Profiling Guide](#profiling-guide)
  - [Overview](#overview)
  - [Architecture](#architecture)
    - [Dynamic Subclass Pattern](#dynamic-subclass-pattern)
    - [Actor Lifecycle Instrumentation](#actor-lifecycle-instrumentation)
  - [ProfilingConfig](#profilingconfig)
  - [Backend Deep-dives](#backend-deep-dives)
    - [CPU Profiling: pyinstrument](#cpu-profiling-pyinstrument)
    - [Memory Profiling: memray](#memory-profiling-memray)
    - [GPU Profiling: torch.profiler](#gpu-profiling-torchprofiler)
  - [ProfilingState Orchestrator](#profilingstate-orchestrator)
    - [LIFO Nesting Order](#lifo-nesting-order)
    - [scope Context Manager](#scope-context-manager)
  - [profiling_scope: Driver-level Setup](#profiling_scope-driver-level-setup)
    - [ArtifactDelivery Integration](#artifactdelivery-integration)
    - [Pre-shutdown Hook Ordering](#pre-shutdown-hook-ordering)
  - [File Naming Convention](#file-naming-convention)
  - [Output Directory Layout](#output-directory-layout)
  - [Multi-Node Storage and S3/Azure Support](#multi-node-storage-and-s3azure-support)
  - [Artifact Delivery Flow](#artifact-delivery-flow)
  - [Combining CPU Profiles](#combining-cpu-profiles)
  - [Summarizing Memory Profiles](#summarizing-memory-profiles)
  - [Viewing Profiling Results](#viewing-profiling-results)
  - [Error Handling](#error-handling)
  - [Best Practice Alignment](#best-practice-alignment)
  - [Source File Reference](#source-file-reference)

## Overview

The profiling subsystem wraps pipeline stages with pluggable
profiling backends so that every `process_data()`,
`stage_setup_on_node()`, and `stage_setup()` call is automatically
instrumented **without requiring any changes to stage code**.

Four backends are supported:

| Backend | CLI Flag | Tool | Output |
|---|---|---|---|
| **CPU** | `--profile-cpu` | pyinstrument | HTML flame-trees + `.pyisession` |
| **Memory** | `--profile-memory` | memray | `.bin` heap captures + HTML flamegraphs |
| **GPU** | `--profile-gpu` | torch.profiler | Chrome Trace JSON (Perfetto) |
| **Tracing** | `--profile-tracing` | OpenTelemetry / Ray | NDJSON span files + OTLP export |

The first three (CPU, Memory, GPU) are **per-stage** backends that
run inside each Ray actor.  Tracing operates at the **cluster level**
via Ray's `_tracing_startup_hook` (see
[Distributed Tracing Guide](../reference/distributed-tracing.md)).

Adding a new per-stage backend requires only local changes inside
`_ProfilingState` -- no stage code or pipeline wiring changes.

The implementation lives in a single module:
`cosmos_curator/core/utils/infra/profiling.py`.

## Architecture

### Dynamic Subclass Pattern

Profiling is injected transparently via a dynamic subclass that
intercepts every stage lifecycle method.  The stage instance's
`__class__` is swapped in place, preserving all existing attributes:

```
+------------------------------+
|      CuratorStage (base)     |
|  process_data(tasks) -> ...  |
+------------------------------+
              ^
              | (dynamic subclass via profiling_wrapper)
+------------------------------+
|    _ProfiledStage(base)      |
|                              |
|  stage_setup_on_node():      |
|    with state.scope():       |
|      super().setup_on_node() |
|                              |
|  stage_setup():              |
|    with state.scope():       |
|      super().stage_setup()   |
|                              |
|  process_data(tasks):        |
|    with state.scope():       |
|      super().process_data()  |
|                              |
|  destroy():                  |
|    state.flush_final()       |
|    super().destroy()         |
|    flush_tracing()           |
+------------------------------+
              |
              | delegates to
              v
+------------------------------+
|     _ProfilingState          |
|  (one per actor, lazy init)  |
|                              |
|  _cpu: _CpuProfilingBackend  |
|  _mem: _MemoryProfilingBknd  |
|  _gpu: _GpuProfilingBackend  |
+------------------------------+
```

Key properties:

- Requires zero changes to 30+ stage constructors.
- Works transparently with all executors (Xenna, StageRunner,
  SequentialRunner) because instrumentation lives inside the
  stage lifecycle methods.
- Keeps `StageTimer` focused on per-task stats.

`profiling_wrapper(stage, config)` is the public API.  It creates
the dynamic subclass via `_make_profiled_stage_class()` and swaps
`stage.__class__` in place.  Called from
`_build_pipeline_stage_specs()` in `pipeline_interface.py`.

### Actor Lifecycle Instrumentation

Every pipeline stage goes through a well-defined lifecycle inside
its Ray actor.  Profiling captures all phases -- not just the hot
`process_data()` loop -- because setup can dominate wall-clock time
(model weight copies, GPU context init) and is invisible to
per-task metrics:

```
Actor lifecycle (inside Ray worker process)
=============================================

stage_setup_on_node()     <-- once per node; weight copies, shared state
      |
      v
stage_setup()             <-- once per actor; model load, GPU init
      |
      v
+----------------------------+
| process_data(batch_1)      |  <-- repeated per batch
| process_data(batch_2)      |
| ...                        |
+----------------------------+
      |
      v
destroy()                 <-- flush final artifacts, flush tracing
```

Each lifecycle method is wrapped in a `_ProfilingState.scope()`
context manager that:

1. Starts all enabled backends.
2. Creates an OTel span with `stage.name`, `stage.lifecycle`, and
   `profiling.artifact_id` attributes.
3. Yields to the real method.
4. Stops all backends and saves artifacts (in `finally`).

## ProfilingConfig

`ProfilingConfig` is a frozen `attrs` class governing which
backends are active.  It is built from CLI arguments by
`_apply_profiling_config()`.

```
_apply_profiling_config(args)
|
+-- Read --profile-cpu, --profile-memory, --profile-gpu,
|   --profile-tracing from args
|
+-- Return None if no flags set (zero overhead)
|
+-- Auto-discover output path from args:
|   probe output_clip_path, output_dataset_path, output_prefix
|
+-- Derive profile_dir = <output-path>/profile
|
+-- Any --profile-* flag implies --perf-profile
|   (enables StageTimer.log_stats())
|
+-- Parse --profile-*-exclude comma-separated lists
|
+-- Return frozen ProfilingConfig
```

| Attribute | Type | Description |
|---|---|---|
| `profile_dir` | `str` | `<output-path>/profile` -- base directory for all backends |
| `s3_profile_name` | `str` | Named credential profile for S3/Azure upload |
| `cpu_enabled` | `bool` | Enable pyinstrument |
| `memory_enabled` | `bool` | Enable memray |
| `gpu_enabled` | `bool` | Enable torch.profiler |
| `tracing_enabled` | `bool` | Enable OpenTelemetry distributed tracing |
| `cpu_exclude` | `frozenset[str]` | Scope names to skip for CPU profiling |
| `memory_exclude` | `frozenset[str]` | Scope names to skip for memory profiling |
| `gpu_exclude` | `frozenset[str]` | Scope names to skip for GPU profiling |

## Backend Deep-dives

### CPU Profiling: pyinstrument

`_CpuProfilingBackend` manages a pyinstrument `Profiler` lifecycle
per stage call.

```
start(call_id)
|
+-- Guard: skip if disabled or excluded
+-- Clean up leftover profiler from failed previous attempt
+-- Profiler().start()
|
v
[stage method runs]
|
v
stop()
|
+-- Profiler.stop()
+-- Dump text summary to stdout (immediate visibility)
+-- Save HTML flame-tree to staging: cpu/<artifact_id>.html
+-- Save .pyisession to staging: cpu/<artifact_id>.pyisession
```

**Output per call:**

- **Text summary** to stdout -- includes wall-clock elapsed time.
- **HTML flame-tree** -- self-contained, viewable in any browser.
- **`.pyisession` file** -- pyinstrument's native format, mergeable
  with `benchmarks/merge_cpu_profiles.py`.

**Error handling:** On startup failure, pyinstrument is disabled
for the actor lifetime (`_cpu_disabled = True`).  The pipeline
continues without CPU profiling for that actor.

### Memory Profiling: memray

`_MemoryProfilingBackend` manages a memray `Tracker` context
manager per stage call.

```
start(call_id)
|
+-- Guard: skip if disabled or excluded
+-- Clean up leftover tracker from failed previous attempt
+-- Resolve .bin path: memory/<artifact_id>.bin
+-- Tracker(native_traces=True, trace_python_allocators=True,
|          follow_fork=True).__enter__()
|
v
[stage method runs]
|
v
stop()
|
+-- Tracker.__exit__()  (may fail with pyinstrument conflict)
+-- .bin is still valid (memray flushes continuously)
+-- Compute stats, dump summary to stdout
+-- Generate HTML flamegraph: memory/<artifact_id>.html
```

**Tracker flags:**

- `native_traces=True` -- captures C-extension stacks (worthwhile
  for GPU/CUDA workloads).
- `trace_python_allocators=True` -- surfaces pymalloc arena
  subdivisions as independent allocations for finer Python-level
  granularity.
- `follow_fork=True` -- tracks allocations in forked child
  processes.

**sys.setprofile conflict:** When both CPU and memory profiling
are active, memray's `Tracker.__exit__()` may fail because it
tries to restore pyinstrument's `sys.setprofile` hook (a C-level
`ProfilerState` object) by calling it, which raises `TypeError`.
The `.bin` capture file is still valid because memray flushes data
continuously.  The error is caught and stats/flamegraph generation
proceeds normally.

**Native symbol lookup:** Native traces may cause memray report readers
to resolve C/C++ symbols.  On machines with `DEBUGINFOD_URLS` set,
missing local debug symbols can trigger network-backed debuginfod
lookups during `memray._memray.compute_statistics()` or flamegraph
generation.  We have seen this primarily in pytest/dev runs, where it
can look like the profiling test suite is hanging.  Pytest clears
`DEBUGINFOD_URLS` globally for this repository so memray tests do not
block on remote debug-symbol downloads.

Normal Curator profiling runs preserve the caller's environment.  If a
direct memray CLI run appears stuck in statistics or flamegraph
generation, run it with debuginfod disabled:

```bash
DEBUGINFOD_URLS= memray stats path/to/profile.bin
DEBUGINFOD_URLS= memray flamegraph path/to/profile.bin
```

**Output per call:**

- **Stats summary** to stdout (total allocations, peak, top sites).
- **`.bin` capture** -- memray's native format, analyzable with
  `memray flamegraph`, `memray stats`, `memray tree`, etc.
- **HTML flamegraph** -- high-watermark allocation flamegraph.

### GPU Profiling: torch.profiler

`_GpuProfilingBackend` manages a `torch.profiler.profile` context
per stage call.

```
start(call_id)
|
+-- Guard: skip if disabled or excluded
+-- Clean up leftover profiler from failed previous attempt
+-- import torch (deferred)
+-- Check torch.cuda.is_available()
|   (deferred to avoid premature CUDA init before Ray
|    assigns GPU resources)
+-- torch.profiler.profile(
|     activities=[CPU, CUDA],
|     record_shapes=True,
|     profile_memory=True,
|     with_stack=True,
|   ).__enter__()
|
v
[stage method runs]
|
v
stop()
|
+-- profiler.__exit__()  (calls torch.cuda.synchronize())
+-- Dump key_averages table to stdout (top 15 by CUDA time)
+-- Export Chrome Trace JSON: gpu/<artifact_id>.json
```

**Deferred CUDA initialization:** `torch.cuda.is_available()` is
checked in `start()`, not `__init__()`.  Calling it too early
(before Ray assigns GPU resources) triggers CUDA lazy-init which
can fail or consume GPU memory before the model loads.

**torch.profiler does NOT use sys.setprofile**, so there is no LIFO
conflict with pyinstrument or memray.

**Output per call:**

- **key_averages table** to stdout -- sorted by `cuda_time_total`,
  top 15 rows.
- **Chrome Trace JSON** -- viewable in
  [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing`.

## ProfilingState Orchestrator

`_ProfilingState` composes all three backends into a single
per-actor orchestrator.  Each instrumented stage actor gets exactly
one `_ProfilingState` instance (created lazily on the first
lifecycle call).

### LIFO Nesting Order

Backend start/stop follows LIFO nesting to unwind
`sys.setprofile` hooks correctly:

```
start():          cpu (outermost) --> mem --> gpu (innermost)
stop_and_save():  gpu --> mem --> cpu (outermost)
```

- **pyinstrument starts first** so it captures the full picture,
  including any overhead from memray.
- **torch.profiler is innermost** because it captures CUDA events
  globally and does NOT use `sys.setprofile`, so it has no LIFO
  conflict.
- Each `stop()` is independently guarded: one backend failure does
  not prevent others from flushing their data.

### scope Context Manager

`_ProfilingState.scope(label)` wraps a lifecycle call:

1. Builds an `artifact_id` matching the file naming convention.
2. Opens an OTel span `"{stage_name}.{label}"` with attributes:
   - `stage.name` -- stage class name.
   - `stage.lifecycle` -- `"setup_on_node"`, `"setup"`,
     `"process_data"`, or `"destroy"`.
   - `profiling.artifact_id` -- correlates the span with its
     profiling files (`cpu/<id>.html`, `memory/<id>.bin`, etc.).
3. Calls `start(label)` to start all enabled backends.
4. Yields to the real method.
5. In `finally`: calls `stop_and_save()` and
   `flush_final_artifacts()`.

On unhandled exceptions, `scope()` also flushes the OTel trace
file (via `flush_tracing()`) because for setup failures,
`destroy()` is never called and Ray kills workers with SIGKILL
(atexit never runs).

## profiling_scope: Driver-level Setup

`profiling_scope(args)` is the top-level context manager used at
the CLI entry point.  It orchestrates the entire profiling and
tracing subsystem.

**Initialization order matters.** The steps below MUST execute in
this exact sequence -- each step depends on the previous one:

```
profiling_scope(args)
|
+-- 1. _apply_profiling_config(args)
|      (returns None if no flags -> yield immediately, zero overhead)
|
+-- 2. ArtifactDelivery.create(kind="profiling")
|      Sets COSMOS_CURATOR_ARTIFACTS_STAGING_DIR env var.
|      Registers pre-shutdown hook for collection.
|      MUST happen first: sets the shared staging directory
|      that all subsequent steps depend on.
|
+-- 3. ArtifactDelivery.create(kind="traces")  [if tracing enabled]
|      MUST happen BEFORE enable_tracing().
|      Reason: enable_tracing() reads STAGING_DIR to determine
|      the trace directory.  If not set yet, it falls back to
|      /tmp/cosmos_curator_traces/ which would not match the
|      staging dir that ArtifactDelivery creates, causing
|      collection to find zero trace files.
|
+-- 4. enable_tracing()  [if tracing enabled]
|      Sets XENNA_RAY_TRACING_HOOK env var (read by ray.init).
|      Sets COSMOS_CURATOR_TRACE_DIR env var (read by workers).
|      Calls setup_tracing() on the driver itself.
|      Configures TracerProvider + span exporters.
|      MUST happen AFTER step 3 (needs staging dir).
|      MUST happen BEFORE step 7 (spans need a provider).
|
+-- 5. _ProfilingState(stage_name="_root", config)
|      Driver-level profiling state (CPU, memory, GPU backends).
|
+-- 6. register_pre_shutdown_hook(_flush_root_profilers)
|      LIFO ordering: runs BEFORE ArtifactDelivery.collect().
|      Ensures root profiling artifacts exist before collection.
|
+-- 7. trace_root_anchor()  [if tracing enabled]
|      Creates the true root span (no parent).
|      Exported immediately so backends receive it first.
|      +-- propagate_trace_context()
|          Writes trace_id:span_id to COSMOS_CURATOR_TRACEPARENT.
|          Workers inherit this via env var at fork time.
|
+-- 8. state.scope("main")
|      Wraps the pipeline in profiling + tracing.
|      +-- yield (pipeline runs)
```

**Tracing setup dependency chain:**

```
ArtifactDelivery("traces")   (step 3)
        |
        | sets STAGING_DIR
        v
enable_tracing()             (step 4)
        |
        | configures TracerProvider
        v
trace_root_anchor()          (step 7)
        |
        | creates root span, propagates context
        v
traced_span / @traced        (application code)
        |
        | creates child spans as no-ops when provider absent
        v
[spans are meaningful only when the full chain above ran]
```

**Key insight for stage authors**: you never call any of these
setup functions.  The `--profile-tracing` CLI flag triggers the
full chain via `profiling_scope()`.  Your stage code just imports
and uses `traced_span`, `@traced`, or `TracedSpan.current()` --
they are always safe to call, even without `--profile-tracing`
(they become zero-cost no-ops).

### ArtifactDelivery Integration

Two `ArtifactDelivery` instances are created, each with a distinct
`kind` (staging subdirectory) and `upload_subdir`:

| Instance | `kind` | Staging path | Upload destination |
|---|---|---|---|
| Profiling | `"profiling"` | `<staging>/profiling/` | `<output-path>/profile/` |
| Traces | `"traces"` | `<staging>/traces/` | `<output-path>/profile/traces/` |

The traces `ArtifactDelivery` **must** be created before
`enable_tracing()`.  `enable_tracing()` reads
`COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` to determine the trace
directory.  If the staging dir env var is not yet set,
`enable_tracing()` falls back to `/tmp/cosmos_curator_traces/`
which would not match the staging directory that
`ArtifactDelivery` creates, causing collection to find zero files.

### Pre-shutdown Hook Ordering

Hooks are registered in order, but executed in LIFO (last-in,
first-out) order during `shutdown_cluster`:

```
Shutdown sequence (LIFO):
  1. _flush_root_profilers  -- writes root files to staging
  2. ArtifactDelivery(traces).collect()
  3. ArtifactDelivery(profiling).collect()  -- root files now exist
  4. Ray shutdown
```

The root profiler flush hook is registered **after** the
ArtifactDelivery hooks, so LIFO ordering ensures it runs **first**.
Without this, root profiles would be written after
ArtifactDelivery has already collected and Ray has shut down.

## File Naming Convention

All profiling artifacts include the method label, call count,
hostname, and PID for unambiguous identification in multi-node
clusters (Ray, Slurm):

```
<StageName>_<label>_<call_count>_<hostname>_<pid>.<ext>
```

The naming is generated by `artifact_id()` (defined in
`tracing.py` so both profiling and tracing share the same
convention):

```
artifact_id(stage_name, call_id) -> str
    Format: "{stage}_{call_id}_{hostname}_{pid}"
```

The `<label>` identifies which method was profiled:

| Label | Method |
|---|---|
| `setup_on_node` | `CuratorStage.stage_setup_on_node()` |
| `setup` | `CuratorStage.stage_setup()` |
| `process_data` | `CuratorStage.process_data()` |
| `main` | Driver process (`_root` scope) |

The `<call_count>` is a monotonically increasing counter shared
across all labels within an actor.  For example, an actor that runs
`setup_on_node` then `setup` then `process_data` will produce
counts 1, 2, 3 respectively.

Examples:

```
cpu/VideoDownloader_setup_on_node_1_node03_5819.html
cpu/VideoDownloader_setup_2_node03_5819.html
cpu/VideoDownloader_process_data_3_node03_5819.html
cpu/VideoDownloader_process_data_3_node03_5819.pyisession
cpu/_root_main_1_node03_5819.html
memory/ClipTranscodingStage_setup_1_node03_10552.bin
memory/ClipTranscodingStage_process_data_2_node03_10552.bin
memory/ClipTranscodingStage_process_data_2_node03_10552.html
gpu/EmbeddingStage_process_data_1_node04_8821.json
traces/node03_5819.jsonl
traces/node04_8821.jsonl
```

Trace files use a simpler naming scheme
(`<hostname>_<pid>.jsonl`) because they contain spans from all
actor methods within a single worker process, not from a specific
stage.

## Output Directory Layout

The final layout after post-pipeline collection and upload:

```
<output-path>/profile/
    cpu/                    pyinstrument HTML + .pyisession
    memory/                 memray .bin + HTML flamegraphs
    gpu/                    torch.profiler Chrome Trace JSON per stage
    traces/                 OpenTelemetry NDJSON span files per worker
```

During the pipeline run, workers write to a **local staging
directory** with the same structure.  `ArtifactDelivery` collects
from all nodes post-pipeline (see
[Artifact Delivery Flow](#artifact-delivery-flow)).

## Multi-Node Storage and S3/Azure Support

Profiling artifacts are always written to `<output-path>/profile`.
The output path inherits the storage backend of the pipeline
output, so S3 URIs (`s3://...`) and Azure URIs (`az://...`) are
supported automatically.

In multi-node Ray or Slurm clusters, each worker writes profiling
artifacts to a **local staging directory** on its node (set via the
`COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` environment variable).  After
the pipeline completes (or fails), `ArtifactDelivery` instances
collect profiling artifacts and trace files from every node,
uploading them to the final `<output-path>/profile` via
`StorageWriter`.

This two-phase approach (local staging + post-pipeline collection)
ensures artifacts survive even when workers are killed via SIGKILL.

```bash
# Remote profiling: artifacts are staged locally, then uploaded to S3
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path s3://bucket/videos/ \
    --output-clip-path s3://bucket/output/ \
    --profile-cpu --profile-memory --verbose
```

When `--output-clip-path` is an S3 path, profiling artifacts will
be uploaded to `s3://bucket/output/profile` automatically.

## Artifact Delivery Flow

Profiling and tracing artifacts follow a two-phase delivery model
to ensure crash-safety in distributed environments:

```
Phase 1: During pipeline (per-worker)
======================================

    Worker A (Node 1)              Worker B (Node 2)
    +-------------------+          +-------------------+
    | staging_dir/      |          | staging_dir/      |
    |   profiling/      |          |   profiling/      |
    |     cpu/*.html    |          |     cpu/*.html    |
    |     memory/*.bin  |          |   traces/         |
    |   traces/         |          |     *.jsonl       |
    |     *.jsonl       |          +-------------------+
    +-------------------+
    (direct pathlib.Path writes -- no remote upload)


Phase 2: Post-pipeline (driver, pre-shutdown hooks)
====================================================

    ArtifactDelivery(kind="profiling").collect()
          |
          +-- RayFileTransport.collect(staging=<base>/profiling)
          |     |
          |     +-- Deploy _NodeCollector on each node
          |     |     (NodeAffinitySchedulingStrategy, num_cpus=0)
          |     |
          |     +-- ray.wait() loop: consume chunks from all nodes
          |     |     in parallel (process whichever is ready first)
          |     |
          |     +-- Kill all actors (finally block)
          |
          +-- StorageWriter(<output-path>/profile)
          |
          +-- success ? cleanup : preserve collect_dir

    ArtifactDelivery(kind="traces").collect()
          |
          +-- RayFileTransport.collect(staging=<base>/traces)
          |     (same parallel ray.wait() pattern)
          |
          +-- StorageWriter(<output-path>/profile/traces)
          |
          +-- success ? cleanup : preserve collect_dir
```

Key components:

- **`RayFileTransport`**
  (`cosmos_curator/core/utils/artifacts/collector.py`): Generic
  cross-node file transport.  Deploys one `_NodeCollector` actor
  per node, which streams files via Ray streaming generators with
  per-file chunking and double-layer backpressure.  The driver
  consumes chunks from all nodes in parallel via adaptive
  `ray.wait(num_returns=len(pending), timeout=0.1)` batching,
  writing each chunk to disk immediately.  Per-node error
  isolation ensures partial collection on failures.

- **`ArtifactDelivery`**
  (`cosmos_curator/core/utils/artifacts/delivery.py`): Generic,
  consumer-agnostic orchestrator.  Sets the
  `COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` environment variable before
  the pipeline, then uses `RayFileTransport` + `StorageWriter` to
  collect and upload artifacts.  Parameterised by `kind` (staging
  subdirectory name) and `upload_subdir` (optional path appended to
  the output directory).

Both delivery instances register as pre-shutdown hooks and include
crash-safe collection (preserve local files on failure) and
idempotency guards (no double-collection).

For a comprehensive deep-dive into the artifact transport
subsystem, see the
[Artifact Transport Guide](../reference/artifact-transport.md).

## Combining CPU Profiles

When profiling with `--profile-cpu`, each stage actor writes its
own `.pyisession` file under `<output-path>/profile/cpu/`.  The
standalone script `benchmarks/merge_cpu_profiles.py` combines
multiple session files into a single report for a unified view
across all stages:

```bash
# Combine all .pyisession files into one HTML report
python benchmarks/merge_cpu_profiles.py /path/to/profiles/cpu/ -o combined.html

# Text output to stdout
python benchmarks/merge_cpu_profiles.py /path/to/profiles/cpu/ --format text

# Timeline mode (preserves chronological call order)
python benchmarks/merge_cpu_profiles.py /path/to/profiles/cpu/ --format text --timeline

# Save the combined session for later inspection
python benchmarks/merge_cpu_profiles.py /path/to/profiles/cpu/ --save-session combined.pyisession
```

## Summarizing Memory Profiles

When profiling with `--profile-memory`, each stage actor writes its
own `.bin` file under `<output-path>/profile/memory/`.  Unlike CPU
profiles, memray captures cannot be merged into a single session.
The standalone script `benchmarks/merge_memory_profiles.py`
computes per-capture statistics and prints a sorted summary table:

```bash
# Summary table sorted by peak memory (default)
python benchmarks/merge_memory_profiles.py /path/to/profiles/memory/

# Sort by total allocated memory
python benchmarks/merge_memory_profiles.py /path/to/profiles/memory/ --sort-by total

# JSON output for programmatic consumption
python benchmarks/merge_memory_profiles.py /path/to/profiles/memory/ --format json -o summary.json

# Also generate HTML flamegraphs for .bin files missing them
python benchmarks/merge_memory_profiles.py /path/to/profiles/memory/ --generate-flamegraphs
```

## Viewing Profiling Results

- **CPU profiles**: Open `.html` files in any browser.  Use
  `.pyisession` files with `pyinstrument --load <file>` for
  interactive exploration.
- **Memory profiles**: Open `.html` flamegraphs in any browser, or
  use `memray flamegraph <file>.bin` / `memray stats <file>.bin`
  for offline analysis.  If the memray command appears stuck while
  resolving native frames, retry with `DEBUGINFOD_URLS=` to disable
  remote debuginfod symbol lookup.  Use `benchmarks/merge_memory_profiles.py`
  to produce a summary table across all captures (see
  [Summarizing Memory Profiles](#summarizing-memory-profiles)).
- **GPU profiles**: Open `.json` files in
  [Perfetto UI](https://ui.perfetto.dev/) or `chrome://tracing`.
- **Combined CPU profiles**: Use
  `benchmarks/merge_cpu_profiles.py` to merge multiple
  `.pyisession` files into one report (see
  [Combining CPU Profiles](#combining-cpu-profiles)).
- **Distributed traces**: Open `.jsonl` files in
  [Perfetto UI](https://ui.perfetto.dev/) for a timeline view of
  cross-actor spans.  Each line is a self-contained JSON span
  object.

## Error Handling

The profiling subsystem follows a strict **"profiling must never
crash the pipeline"** philosophy:

- Every backend catches all exceptions (`except Exception`) and
  uses `# noqa: BLE001` to acknowledge this is intentional.
- On startup failure, a backend disables itself for the rest of the
  actor lifetime (`_cpu_disabled = True`, `_mem_disabled = True`,
  `_gpu_disabled = True`).  The pipeline continues without that
  backend for that actor.
- `stop_on_error()` provides error-path rollback: stops the
  profiler without saving artifacts, leaving any partial files in
  the staging directory for debugging.
- Each backend's `stop()` is independently guarded in
  `_ProfilingState.stop_and_save()` so one backend failure does not
  prevent others from flushing their data.
- Exceptions during `stop()` are recorded on the current OTel span
  via `TracedSpan.current().record_exception(e)` for visibility in
  traces.
- `contextlib.suppress(Exception)` is used for cleanup paths where
  failure is expected (e.g. memray `__exit__` conflicting with
  pyinstrument's `sys.setprofile` hook).

## Best Practice Alignment

The implementation builds on tools recommended by
[Ray's profiling guide](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html),
adapted for **continuous, automatic profiling** in environments
where the Ray Dashboard is not reachable (NVCF, Slurm multi-node,
Docker).

**Why continuous profiling:** Ray's built-in profiling tools
(py-spy live sampling, memray Dashboard UI, task/actor timeline
download) are interactive and require Ray Dashboard access.  In
production deployments the Dashboard is often unreachable, so
Cosmos Curator implements fire-and-forget profiling: enable via CLI
flags, run the pipeline, retrieve all artifacts from the output
directory.  No interactive session required.

| Aspect | Ray Dashboard approach | Cosmos Curator approach |
|---|---|---|
| **CPU** | py-spy (sampling, Dashboard-integrated), cProfile | pyinstrument: automatic per-call HTML flame-trees, `.pyisession` merge -- all written to disk |
| **Memory** | memray (Dashboard UI: flamegraph, table, leaks, native, pymalloc) | Same memray with `native_traces=True`, `trace_python_allocators=True`; automatic `.bin` + HTML flamegraph per call |
| **GPU** | PyTorch Profiler (via Ray Train/Data), Nsight Systems (`runtime_env={"nsight": ...}`) | torch.profiler wrapped per-stage with deferred CUDA init; automatic Chrome Trace JSON per call |
| **Timeline** | Task/Actor timeline (Dashboard download) | OpenTelemetry distributed tracing with domain attributes, local NDJSON + OTLP export |
| **Artifact retrieval** | `/tmp/ray/session_*/logs/` (requires Dashboard or SSH) | Automatic cross-node collection via `ArtifactDelivery` to pipeline output path (local, S3, Azure) |
| **Trigger** | Interactive: click in Dashboard UI | Declarative: `--profile-cpu`, `--profile-memory`, etc. CLI flags |

Ray's Nsight Systems support (`runtime_env={"nsight": ...}`) is
available as a complementary GPU profiling tool for deeper CUDA
kernel analysis beyond what `torch.profiler` captures.

## Source File Reference

| File | Description |
|---|---|
| `cosmos_curator/core/utils/infra/profiling.py` | Profiling backends, `_ProfilingState`, `_ProfiledStage`, `profiling_wrapper`, `profiling_scope` |
| `cosmos_curator/core/utils/infra/tracing.py` | `artifact_id()`, `process_tag()` (shared naming convention) |
| `cosmos_curator/core/utils/artifacts/delivery.py` | `ArtifactDelivery` orchestrator for post-pipeline collection |
| `cosmos_curator/core/utils/artifacts/collector.py` | `RayFileTransport` for cross-node file streaming |
| `benchmarks/merge_cpu_profiles.py` | Merge multiple `.pyisession` files into one report |
| `benchmarks/merge_memory_profiles.py` | Summarize multiple memray `.bin` captures |
