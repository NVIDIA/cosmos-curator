# Distributed Tracing Guide

Deep-dive into the OpenTelemetry-based distributed tracing subsystem
for Cosmos Curator pipelines.  This guide covers the public span API,
the Ray tracing hook, trace context propagation, library
auto-instrumentation, and configuration.

- [Distributed Tracing Guide](#distributed-tracing-guide)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Span Hierarchy](#span-hierarchy)
  - [Public Span API](#public-span-api)
    - [traced_span Context Manager](#traced_span-context-manager)
    - [traced Decorator](#traced-decorator)
    - [TracedSpan Class](#tracedspan-class)
    - [Recording Errors on Spans](#recording-errors-on-spans)
    - [API Summary](#api-summary)
  - [Ray Tracing Hook](#ray-tracing-hook)
    - [How the Hook Works](#how-the-hook-works)
    - [Driver-side Setup: enable_tracing](#driver-side-setup-enable_tracing)
    - [Worker-side Setup: setup_tracing](#worker-side-setup-setup_tracing)
    - [TracingBackend Lifecycle](#tracingbackend-lifecycle)
    - [Dual-exporter Design](#dual-exporter-design)
    - [Trace Context Propagation](#trace-context-propagation)
    - [Root Anchor Span](#root-anchor-span)
    - [Span Flush and Persist](#span-flush-and-persist)
  - [Library Auto-instrumentation](#library-auto-instrumentation)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Resource Attributes](#resource-attributes)
    - [Jaeger Example](#jaeger-example)
  - [Output](#output)
  - [Best Practice Alignment](#best-practice-alignment)
  - [Source File Reference](#source-file-reference)

## Overview

The tracing subsystem captures distributed spans across the entire
pipeline -- from the driver process through every Ray worker actor
-- using [OpenTelemetry](https://opentelemetry.io/docs/concepts/instrumentation/code-based/)
(OTel) as the instrumentation framework and
[Ray's tracing startup hook](https://docs.ray.io/en/latest/ray-observability/user-guides/ray-tracing.html)
for per-worker setup.

The subsystem is split into two modules with a clean separation of
concerns:

- `tracing.py` -- the **public span API** (`TracedSpan`,
  `traced_span`, `@traced`, `StatusCode`).  Independent of Ray,
  importable from any environment, zero-cost when tracing is not
  enabled (OTel's default proxy provider creates lightweight
  `NonRecordingSpan` objects).
- `tracing_hook.py` -- the **Ray-specific hook** (`enable_tracing`,
  `setup_tracing`, `_TracingBackend`, library auto-instrumentors).
  Handles `TracerProvider` configuration, file/OTLP exporters,
  trace context propagation, and span persistence.

Enable tracing with the `--profile-tracing` CLI flag.

## Architecture

```
+---------------------------------------+
| Application / Stage code              |
| (process_data, custom spans, etc.)    |
+---------------------------------------+
             |
             | imports public API
             v
+---------------------------------------+
| tracing.py                            |
| Public span API (no Ray dependency)   |
|                                       |
| TracedSpan          traced_span()     |
| @traced             trace_root_anchor |
| StatusCode          _process_tag()    |
+---------------------------------------+
             |
             | uses OTel trace API
             v
+---------------------------------------+
| opentelemetry.trace (API layer)       |
| get_tracer() -> start_as_current_span |
+---------------------------------------+
             ^
             | configures TracerProvider
             |
+---------------------------------------+
| tracing_hook.py                       |
| Ray-specific hook + OTel SDK setup    |
|                                       |
| enable_tracing()   setup_tracing()    |
| _TracingBackend    TracingConfig      |
| flush_tracing()    propagate_context  |
| _instrument_libraries()              |
+---------------------------------------+
             |
             | exports spans to
             v
+-------------------+  +-------------------+
| ConsoleSpanExporter|  | OTLPSpanExporter  |
| (local .jsonl)    |  | (Jaeger/Tempo/...) |
+-------------------+  +-------------------+
```

The key design decision is the **two-module split**: `tracing.py`
contains only the lightweight OTel trace API (no SDK, no Ray) so it
can be imported freely by any code.  The heavy SDK setup, Ray
integration, and library patching live in `tracing_hook.py` and are
only loaded when tracing is actually enabled.

## Span Hierarchy

When tracing is enabled, three levels of spans are produced:

| Level | Source | What it captures |
|-------|--------|-----------------|
| **Ray spans** (automatic) | Ray OTel hook | Actor creation, method invocations, task scheduling, cross-actor context propagation |
| **Lifecycle spans** | `_ProfiledStage` | `setup_on_node`, `setup`, `process_data`, `destroy` with `stage.name`, `stage.num_input_tasks` |
| **Sample spans** | `StageTimer.time_process()` | Per-sample processing inside `process_data`, with `stage.num_samples`, `stage.source_video_duration_s` |

Additionally, `StageTimer.log_stats()` annotates the current span
with aggregate stats (`stage.process_time_s`, `stage.idle_time_s`,
`stage.rss_before_mb`, `stage.rss_after_mb`, `stage.rss_delta_mb`,
etc.) and `StageTimer.reinit()` emits a `batch_start` event marking
the start of each `process_data` batch.

The resulting span tree:

```
[Ray: actor_method process_data]                         <-- automatic
  +-- [cosmos_curator: VideoDownloader.process_data]      <-- lifecycle span
       | attrs: stage.name, stage.num_input_tasks, stage.process_time_s, ...
       | event: batch_start (stage.input_data_size_b, stage.rss_before_mb)
       |
       +-- [cosmos_curator: VideoDownloader.sample]       <-- per-sample span
       |   attrs: stage.name, stage.num_samples
       +-- [cosmos_curator: VideoDownloader.sample]
       ...
```

This gives three complementary views:

- **Satellite view** (Ray spans): end-to-end distributed timeline
  with scheduling delays.
- **Street view** (application spans): domain-specific attributes
  and per-sample granularity.
- **Microscope view** (pyinstrument): deep CPU call stacks within a
  single process.

## Public Span API

The public API lives in `cosmos_curator/core/utils/infra/tracing.py`
and is the only module that application code needs to import.  All
functions and methods are zero-cost no-ops when no `TracerProvider`
has been configured.

### traced_span Context Manager

Creates a child span under the current active span.  The span is
automatically closed when the context manager exits.  Duration is
recorded by OTel.

```python
from cosmos_curator.core.utils.infra.tracing import TracedSpan, traced_span


class MyStage(CuratorStage):
    def process_data(self, tasks):
        for task in tasks:
            with traced_span("MyStage.download", attributes={"url": task.url}) as span:
                data = download(task.url)
                span.set_attribute("download.bytes", len(data))
            with traced_span("MyStage.inference") as span:
                result = self.model.predict(task.data)
                span.set_attribute("model.batch_size", len(task.data))

        # Annotate the parent span (from _ProfiledStage) without creating a child:
        TracedSpan.current().add_event("all_tasks_done", attributes={"count": len(tasks)})
        return tasks
```

### traced Decorator

For functions where the entire body should be a single span, use
the `@traced` decorator instead of a context manager:

```python
from cosmos_curator.core.utils.infra.tracing import traced


@traced("MyModel.predict", attributes={"model": "resnet50"})
def predict(self, batch: list[Frame]) -> list[float]:
    return self._model(batch)
```

This is equivalent to wrapping the function body in
`with traced_span(...)`.  Uses `wrapt.decorator` for robust
signature preservation across plain functions, bound methods,
class methods, and static methods.

### TracedSpan Class

`TracedSpan` is a lightweight wrapper around an OTel `Span` using
the Null Object pattern -- all methods silently return without error
when the underlying span is `None`.

There are two ways to obtain a `TracedSpan`:

1. **Create a new child span** via `traced_span()` (preferred for
   scoped work).
2. **Wrap the current active span** via `TracedSpan.current()`
   (useful when annotating a span created higher up the call stack,
   e.g. by `_ProfiledStage`).

Methods:

- `set_attribute(key, value)` -- set a single attribute.
- `set_attributes({...})` -- set attributes in bulk.
- `add_event(name, attributes=...)` -- add a timestamped event
  (milestones like `"batch_start"`, `"model_loaded"`).
- `record_exception(exc, attributes=...)` -- record a caught
  exception (see below).
- `set_status(code, description=...)` -- set span status.

### Recording Errors on Spans

Unhandled exceptions that propagate out of a `traced_span` or
`@traced` block are **automatically recorded** by OTel (exception
type, message, and traceback are added as a span event, and the
span status is set to ERROR).

For **caught exceptions** that you handle gracefully but still want
visible in traces, use `record_exception()` and `set_status()`:

```python
from opentelemetry.trace import StatusCode
from cosmos_curator.core.utils.infra.tracing import traced_span

with traced_span("MyStage.download") as span:
    try:
        data = download(url, timeout=30)
    except TimeoutError as e:
        span.record_exception(e)
        span.set_status(StatusCode.ERROR, "Download timed out")
        data = fallback_download(url)  # graceful recovery
```

`StatusCode` is re-exported from `opentelemetry.trace` and has
three values: `UNSET` (default), `OK` (explicit success), and
`ERROR`.

### API Summary

| Function / Class | Purpose |
|---|---|
| `traced_span(name, attributes=...)` | Context manager creating a child span |
| `@traced(name, attributes=...)` | Decorator wrapping a function in a span |
| `TracedSpan.current()` | Wraps the current active span (no new span) |
| `span.set_attribute(key, value)` | Set a single attribute |
| `span.set_attributes({...})` | Set attributes in bulk |
| `span.add_event(name, attributes=...)` | Add a timestamped event |
| `span.record_exception(exc)` | Record a caught exception |
| `span.set_status(StatusCode.ERROR)` | Mark span as failed |

All methods are no-ops when tracing is not enabled (OTel's default
proxy provider creates lightweight `NonRecordingSpan` objects).

## Ray Tracing Hook

The Ray-specific integration lives in
`cosmos_curator/core/utils/infra/tracing_hook.py`.

### How the Hook Works

Ray supports a
[tracing startup hook](https://docs.ray.io/en/latest/ray-observability/user-guides/ray-tracing.html)
that configures OpenTelemetry on every worker process.  The hook is
a Python function referenced as a `"module:attribute"` string:

```python
ray.init(
    _tracing_startup_hook=(
        "cosmos_curator.core.utils.infra.tracing_hook:setup_tracing"
    ),
    ...
)
```

Ray stores this string in its GCS internal KV store and every
worker (including the driver) reads it on startup, so the function
must be importable on all nodes.

Configuration is passed via environment variables because the Ray
hook function signature takes no arguments.

### Driver-side Setup: enable_tracing

`enable_tracing()` is called from `profiling_scope()` on the driver
process.  It performs three actions:

```
enable_tracing()
|
+-- 1. Set XENNA_RAY_TRACING_HOOK env var
|       (read by init_or_connect_to_cluster -> ray.init)
|
+-- 2. Set COSMOS_CURATOR_TRACE_DIR env var
|       (read by setup_tracing on each worker)
|
+-- 3. Clear stale COSMOS_CURATOR_TRACEPARENT
|       (driver is the root -- must not inherit old trace)
|
+-- 4. Call setup_tracing() on the driver itself
|       (workers get their own call via Ray's hook)
|
+-- 5. Ensure PYTHONPATH includes cosmos_curator
        (so bare raylet workers can import the hook)
```

### Worker-side Setup: setup_tracing

`setup_tracing()` is called once per process -- by the Ray hook on
workers, and directly by `enable_tracing()` on the driver.

```
setup_tracing()
|
+-- re-entrancy guard (no-op if already called)
|
+-- TracingConfig.from_env()
|       Parse COSMOS_CURATOR_TRACE_DIR,
|       OTEL_EXPORTER_OTLP_ENDPOINT, etc.
|
+-- _TracingBackend(config)
|       Create span file, open file handle
|
+-- backend.setup_provider()
|       Configure TracerProvider + Resource
|       Attach ConsoleSpanExporter (file)
|       Attach OTLPSpanExporter (remote)
|       Attach remote parent context
|       Activate library auto-instrumentors
|
+-- _current_backend = backend  (module singleton)
|
+-- atexit.register(backend.shutdown)
        Fallback for graceful exits
```

The re-entrancy guard prevents double-setup on the driver (once
from `enable_tracing()`, once if Ray's hook also fires due to
`ignore_reinit_error=True`).

### TracingBackend Lifecycle

`_TracingBackend` manages the per-process `TracerProvider` and local
span file:

```
_TracingBackend lifecycle
=========================

__init__(config)          <-- create trace dir, open .jsonl file
      |
      v
setup_provider()          <-- configure TracerProvider + exporters
      |                       attach remote parent context
      |                       activate library auto-instrumentors
      v
[worker runs stages]      <-- spans accumulate in .jsonl file
      |                       + exported to OTLP collector
      v
flush()                   <-- force_flush + _persist (close file)
      |                       _flushed = True
      v
shutdown()                <-- provider.shutdown + _persist (atexit)
                              no-op persist if _flushed == True
```

Both `flush()` and `shutdown()` are idempotent.  `flush()` uses
`force_flush()` (safe while provider is still active), `shutdown()`
uses `provider.shutdown()` (final exit).

### Dual-exporter Design

Each process attaches **two span processors** to the
`TracerProvider`:

1. **`ConsoleSpanExporter`** -- writes NDJSON (one JSON object per
   line) to a local `.jsonl` file.  Always active.  Produces the
   span files collected post-pipeline by `ArtifactDelivery`.

2. **`OTLPSpanExporter`** (wrapped in `_ResilientOtlpExporter`) --
   sends spans to an OTLP HTTP endpoint (defaults to
   `http://localhost:4318`).  Always active so a local Jaeger or
   Grafana Tempo instance receives spans out of the box.  The
   `_ResilientOtlpExporter` wrapper suppresses connection errors
   (e.g. when no collector is running) so that export failures
   never crash the pipeline or pollute logs with traceback noise.

Both use **`SimpleSpanProcessor`** (not `BatchSpanProcessor`).
This is a deliberate design decision: Ray may SIGKILL worker
processes during `ray.shutdown()` before a batched flush has time
to drain.  `SimpleSpanProcessor` exports each span synchronously
as it completes, ensuring no data loss on abrupt termination.  For
the moderate span count per worker (tens, not thousands), the
per-span overhead is acceptable.

This follows the OTel
[code-based instrumentation](https://opentelemetry.io/docs/concepts/instrumentation/code-based/)
best practice of configuring the TracerProvider with appropriate
exporters and processors for the deployment environment.

### Trace Context Propagation

All processes (driver + workers) share a single `trace_id` so the
entire pipeline appears as one trace in backends like Jaeger.

```
profiling_scope()
      |
      +-- enable_tracing()              <-- sets up driver TracerProvider
      |
      +-- trace_root_anchor() starts    <-- anchor span (root, no parent)
      |     +-- anchor.end()            <-- exported immediately
      |     +-- propagate_context()     <-- reads anchor's trace_id + span_id
      |     |     +-- writes COSMOS_CURATOR_TRACEPARENT env var
      |     |          format: "{trace_id_hex}:{span_id_hex}"
      |     |
      |     +-- state.scope("main")     <-- _root.main (child of anchor)
      |     |     +-- yield (pipeline runs, workers start)
      |     |
      +-- trace_root_anchor exits       <-- detach anchor context
      |
      v
[workers read COSMOS_CURATOR_TRACEPARENT in setup_tracing()]
      |
      +-- _attach_remote_parent()
      |     Construct remote SpanContext(trace_id, span_id)
      |     Attach as current OTel context
      |     All subsequent stage spans become children of the anchor
```

The propagation uses a custom env-var format
(`"{trace_id_hex}:{span_id_hex}"`) rather than the standard W3C
`traceparent` HTTP header because:

- Ray workers inherit environment variables at fork time, not
  HTTP headers.
- The hook function takes no arguments, so the only communication
  channel is the environment.

### Root Anchor Span

`trace_root_anchor()` creates a short-lived root span with **no
parent** (explicit empty OTel context), making it the true root of
the trace.  It is ended **immediately** so backends receive it
before any child spans arrive -- eliminating "invalid parent span
ID" warnings that occur when a long-lived root arrives last.

Even though the anchor span is ended, its context (trace_id +
span_id) stays attached as the current OTel context for the
duration of the `with` block.  All spans created inside inherit
the anchor's trace_id and use its span_id as their parent.

```
trace_anchor  (root, no parent, ends FIRST)
  +-- _root.main              (ends LAST, parent=anchor)
  +-- stage spans             (parent=anchor via env var)
```

### Span Flush and Persist

Workers and the driver have different persist paths:

```
Worker path (primary):
    _ProfiledStage.destroy()
          |
          +-- flush_tracing()
          |     +-- backend.flush()
          |     |     +-- TracerProvider.force_flush()
          |     |     +-- _persist() -> close file, log size
          |     |     +-- _flushed = True
          v
    [worker exits or is killed by SIGKILL]
          |
          v
    atexit -> backend.shutdown()
                +-- (no-op persist if _flushed == True)
                +-- TracerProvider.shutdown()

Driver path (fallback only):
    atexit -> backend.shutdown()
                +-- TracerProvider.shutdown()
                +-- _persist() (if flush() never ran)
```

Ray uses SIGKILL during `ray.shutdown()`, so `atexit` handlers are
never invoked for worker processes.  `flush_tracing()` is the
primary persist path for workers, called explicitly from
`_ProfiledStage.destroy()` before the worker is killed.

## Library Auto-instrumentation

When tracing is enabled, `setup_tracing()` activates OTel
auto-instrumentors for commonly used libraries.  Each instrumentor
monkey-patches its target library so that every call emits an OTel
span automatically.

| Library | Instrumentor Package | What it captures |
|---|---|---|
| **botocore** | `opentelemetry-instrumentation-botocore` | S3 / AWS API calls (GetObject, PutObject, etc.) |
| **requests** | `opentelemetry-instrumentation-requests` | Outbound HTTP requests |
| **urllib3** | `opentelemetry-instrumentation-urllib3` | Low-level HTTP transport (used by boto3, requests) |
| **threading** | `opentelemetry-instrumentation-threading` | Context propagation across threads |
| **logging** | `opentelemetry-instrumentation-logging` | Injects `otelTraceID` / `otelSpanID` into stdlib log records |
| **sqlalchemy** | `opentelemetry-instrumentation-sqlalchemy` | SQL query spans |
| **fastapi** | `opentelemetry-instrumentation-fastapi` | Inbound HTTP endpoint spans |

Instrumentors are gated on `importlib.util.find_spec()`, so only
libraries that are actually installed in the current Pixi
environment are patched.  No errors are raised for missing
libraries.

All instrumentor packages are included in the `profiling` Pixi
feature.

## Configuration

### Environment Variables

**Cosmos Curator specific:**

| Variable | Set by | Read by | Purpose |
|---|---|---|---|
| `COSMOS_CURATOR_TRACE_DIR` | `enable_tracing()` | `setup_tracing()` (workers) | Local directory for span files (defaults to `<staging>/traces/`) |
| `COSMOS_CURATOR_TRACEPARENT` | `propagate_trace_context()` | `setup_tracing()` (workers) | Trace context for correlating worker spans under a single trace |
| `XENNA_RAY_TRACING_HOOK` | `enable_tracing()` | `init_or_connect_to_cluster()` | `"module:attribute"` string passed to `ray.init(_tracing_startup_hook=...)` |

**Standard OTel environment variables:**

| Variable | Purpose |
|---|---|
| `OTEL_SERVICE_NAME` | Override resource `service.name` (default: `cosmos_curator`) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Base collector endpoint (e.g. `http://localhost:4318`) |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Trace-specific override (takes precedence) |
| `OTEL_EXPORTER_OTLP_HEADERS` | Auth headers as `key=value,key2=value2` |
| `OTEL_EXPORTER_OTLP_TIMEOUT` | Export timeout in milliseconds |
| `OTEL_EXPORTER_OTLP_CERTIFICATE` | Path to TLS CA certificate |

All `OTEL_EXPORTER_OTLP_*` variables are standard OTel env vars
read automatically by `OTLPSpanExporter` -- no code changes needed.

### Resource Attributes

Each process (driver and workers) attaches an OTel Resource with
the following attributes.  In multi-node Ray clusters these let you
filter traces by application, node, and worker in Jaeger / Grafana
Tempo:

| Resource Attribute | Value | Purpose |
|---|---|---|
| `service.name` | `cosmos_curator` (or `OTEL_SERVICE_NAME`) | Identifies the application |
| `host.name` | Short hostname (e.g. `node03`) | Identifies the machine |
| `process.pid` | OS PID (e.g. `6135`) | Identifies the worker process |

### Jaeger Example

```bash
# Start Jaeger all-in-one (OTLP HTTP on port 4318)
docker run -d --name jaeger \
  -p 16686:16686 -p 4318:4318 \
  jaegertracing/all-in-one:latest

# Run pipeline with tracing
cosmos-curator local launch --curator-path . -- \
  pixi run --as-is python -m cosmos_curator.pipelines.video.run_pipeline split \
    --input-video-path /config/test_data/raw_videos/ \
    --output-clip-path /config/test_data/output_clips/ \
    --profile-tracing --verbose

# Open Jaeger UI at http://localhost:16686
```

By default the OTLP exporter sends spans to `http://localhost:4318`,
so a Jaeger instance on the host receives spans out of the box.
Because `cosmos-curator local launch` runs inside Docker,
`OTEL_EXPORTER_OTLP_ENDPOINT` must be set **inside** the container
(e.g. baked into the Docker image or passed via `docker run -e`)
to reach a non-localhost collector.

In a multi-node cluster, set `OTEL_EXPORTER_OTLP_ENDPOINT` on
every node (e.g. via the Docker image or Slurm environment) so
that all workers export to the same collector.

> **Note**: The OTLP exporter uses the
> `opentelemetry-exporter-otlp-proto-http` package, which is
> included in the `profiling` Pixi feature (available in all
> runtime environments).

## Output

Tracing produces two types of output:

- **Per-worker NDJSON span files** staged locally under
  `<staging-dir>/traces/` during the run.  Collected to
  `<output-path>/profile/traces/` post-pipeline by
  `ArtifactDelivery` (see
  [Artifact Transport Guide](artifact-transport.md)).
- **Live spans** sent to the OTLP collector (when endpoint is
  configured).

Trace file naming: `trace_spans_<hostname>_<pid>.jsonl`

Each file contains NDJSON -- one self-contained JSON span object
per line.  Files can be viewed in
[Perfetto UI](https://ui.perfetto.dev/) for a timeline view of
cross-actor spans.

> **Note**: Ray's tracing feature is marked as Alpha and the
> `_tracing_startup_hook` is an internal API.  Span files are
> flushed to the local staging directory on a best-effort basis --
> workers killed via SIGKILL may not flush their final spans.
> Post-pipeline, `ArtifactDelivery` collects whatever was staged
> from all nodes.

## Best Practice Alignment

The implementation follows documented patterns from Ray and
OpenTelemetry:

**Ray tracing docs** ([ray.io](https://docs.ray.io/en/latest/ray-observability/user-guides/ray-tracing.html)):

- Uses the `_tracing_startup_hook` mechanism with the
  `"module:attribute"` string format for `ray.init()`.
- Per-worker `TracerProvider` configuration (each process gets its
  own provider, exporters, and span file).
- Hook function is importable on all nodes and takes no arguments.

**OTel code-based instrumentation** ([opentelemetry.io](https://opentelemetry.io/docs/concepts/instrumentation/code-based/)):

- `TracerProvider` + `Resource` configuration with
  `service.name`, `host.name`, and `process.pid` for
  human-readable traces in backends.
- `SimpleSpanProcessor` for crash-safety (vs `BatchSpanProcessor`
  which may lose spans on SIGKILL).
- `ConsoleSpanExporter` for local files + `OTLPSpanExporter` for
  remote delivery -- both active simultaneously.
- Library auto-instrumentors activated via the standard
  `Instrumentor().instrument()` pattern, gated on availability.
- Trace context propagation across process boundaries using
  `SpanContext` with `is_remote=True`.

## Source File Reference

| File | Description |
|---|---|
| `cosmos_curator/core/utils/infra/tracing.py` | Public span API: `TracedSpan`, `traced_span`, `@traced`, `trace_root_anchor`, `StatusCode` |
| `cosmos_curator/core/utils/infra/tracing_hook.py` | Ray hook: `enable_tracing`, `setup_tracing`, `_TracingBackend`, `TracingConfig`, library auto-instrumentors |
