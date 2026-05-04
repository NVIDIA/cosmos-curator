# Artifact Transport Guide

Deep-dive into the `cosmos_curator.core.utils.artifacts` package --
the generic, consumer-agnostic utilities for moving files between
cluster nodes and storage destinations.

- [Artifact Transport Guide](#artifact-transport-guide)
  - [Overview](#overview)
  - [Package Architecture](#package-architecture)
  - [RayFileTransport (Local Collection Transport)](#rayfiletransport-local-collection-transport)
    - [Node Discovery](#node-discovery)
    - [Actor Deployment](#actor-deployment)
    - [Streaming Protocol](#streaming-protocol)
    - [Backpressure Design](#backpressure-design)
    - [Memory Accounting](#memory-accounting)
    - [Concurrency Model](#concurrency-model)
    - [File Reassembly State Machine](#file-reassembly-state-machine)
    - [Multi-file Routing](#multi-file-routing)
    - [Error Handling](#error-handling)
  - [ArtifactDelivery (High-level Orchestrator)](#artifactdelivery-high-level-orchestrator)
    - [Three-phase Lifecycle](#three-phase-lifecycle)
    - [Factory Pattern](#factory-pattern)
    - [Collection Routing](#collection-routing)
    - [Local Collection Path](#local-collection-path)
    - [Remote Collection Path](#remote-collection-path)
    - [Why Collection is Deferred](#why-collection-is-deferred)
    - [Idempotency, Crash-safety, and Partial-failure Tolerance](#idempotency-crash-safety-and-partial-failure-tolerance)
    - [Instrumentation During Shutdown](#instrumentation-during-shutdown)
  - [Environment Variables](#environment-variables)
  - [Source File Reference](#source-file-reference)

## Overview

The artifact transport package gathers files from distributed
Ray worker nodes and delivers them to a destination.  The collection
strategy is chosen **automatically** based on whether the
destination is local or remote:

```
Local destination                    Remote destination (S3 / Azure)
=================                    ===============================

Worker A --+                         Worker A --> StorageWriter --> S3
Worker B --+--> RayFileTransport     Worker B --> StorageWriter --> S3
Worker C --+        |                Worker C --> StorageWriter --> S3
                    v                  (each node uploads directly;
               Driver --> local dir     driver only coordinates)
  (files stream through Ray
   object store to driver)
```

- **Local path**: `RayFileTransport` streams file bytes through the
  Ray object store to the driver, which writes them to disk.
- **Remote path** (S3 / Azure): each worker node uploads directly
  via its own `StorageWriter` instance.  No file data crosses the
  cluster network -- only lightweight metadata returns to the driver.

`ArtifactDelivery` is the high-level orchestrator that manages the
full lifecycle (staging, collection routing, upload) and hides the
strategy selection from consumers.  All classes are fully decoupled
from any specific consumer or subsystem.

Multiple consumers can create separate `ArtifactDelivery` instances
with distinct `kind` parameters.  Each `kind` gets its own staging
subdirectory so different consumers never mix files (e.g.
`<staging>/foo/` vs `<staging>/bar/`).

## Package Architecture

The package is organised into two layers.  Consumer code interacts
only with `ArtifactDelivery` -- never with the low-level components
directly.

```
+-------------------------------+
| Consumer code                 |   Any subsystem that produces
| (any subsystem)               |   cluster-wide files
+-------------------------------+
             |
             v
+-------------------------------+
| ArtifactDelivery              |   delivery.py
| (staging + collect + upload   |   Generic orchestrator
|  lifecycle)                   |
+-------------------------------+
             |
             | StorageWriter.is_remote?
             |
      +------+------+
      |             |
      v (no)        v (yes)
+-----------+  +-------------------+
| RayFile   |  | _NodeUploader     |
| Transport |  | (per-node actor)  |
+-----------+  +-------------------+
(collector.py)         |
      |                v
      v          +-----------+
  driver disk    | Storage   |
                 | Writer    |
                 +-----------+
                 (storage_utils.py)
```

- **Local destinations**: `ArtifactDelivery` uses `RayFileTransport`
  to stream file bytes through the Ray object store to the driver,
  which writes them to the local filesystem.
- **Remote destinations** (S3 / Azure): `ArtifactDelivery` deploys
  a `_NodeUploader` actor on each node.  Each actor reads local
  staged files and uploads directly via `StorageWriter` -- no file
  data crosses the cluster network.

The public API consists of `RayFileTransport` and `CollectResult`
(from `collector.py`), and `ArtifactDelivery` and
`ArtifactDeliveryError` (from `delivery.py`).  Import them directly
from their respective modules.

## RayFileTransport (Local Collection Transport)

`RayFileTransport` streams files from worker node staging directories
to the driver using Ray streaming generators with per-file chunking
and built-in backpressure.  Peak memory is bounded to `chunk_bytes`
(default 128 MB) on both worker and driver.

This transport is used by `ArtifactDelivery` **only for local
destinations**.  For remote destinations (S3 / Azure), each worker
uploads directly via `_NodeUploader` -- see
[Remote Collection Path](#remote-collection-path).

See `cosmos_curator/core/utils/artifacts/collector.py` for
implementation.

### Node Discovery

Before deploying collector actors, the transport queries the Ray
Global Control Store (GCS) for all live nodes via
`get_live_nodes()` (defined in `ray_cluster_utils.py`).  This is a
metadata-only query -- it does not dispatch work.  Each returned
dict contains `NodeID` (unique hex ID) and `NodeName`
(human-readable hostname).

The driver iterates over the returned list to deploy one actor per
node.  Because node discovery is a GCS query (not a Ray task), it
is a single-threaded operation on the driver.

### Actor Deployment

One `_NodeCollector` actor is deployed per live node.  Key design
decisions:

- **`NodeAffinitySchedulingStrategy(node_id, soft=False)`**: hard
  affinity pins the actor to a specific node so it reads files from
  the correct local staging directory.  Soft affinity would allow
  Ray to place the actor on a different node if the target is full,
  which would make it read from the wrong filesystem.
- **`num_cpus=0`**: the actor does file I/O only, not compute.
  Zero CPU reservation avoids starving pipeline actors of compute
  resources during post-pipeline collection.
- **One actor per node**: mirrors the actor-per-node convention
  used throughout the package (`_NodeUploader`).  Each actor handles
  all files on its node to amortize actor creation overhead.

```
collect()
|
+-- get_live_nodes() -> [{NodeID: "abc", NodeName: "host-1"}, ...]
|
+-- for each node:
      _NodeCollector.options(
          scheduling_strategy=NodeAffinity(node_id, soft=False)
      ).remote()
```

### Streaming Protocol

Files are transferred using the `_FileChunk` protocol -- a frozen
attrs dataclass with three fields:

| Field     | Type    | Description                                  |
|-----------|---------|----------------------------------------------|
| `arcname` | `str`   | Relative path within the staging directory   |
| `data`    | `bytes` | Raw bytes for this chunk                     |
| `is_last` | `bool`  | True for the final (or only) chunk           |

Small files (below `chunk_bytes`) produce a single chunk:

```
_FileChunk(arcname="cpu/stage_1.html", data=<100KB>, is_last=True)
```

Large files produce multiple chunks sharing the same `arcname`:

```
_FileChunk(arcname="memory/stage_1.bin", data=<64MB>, is_last=False)
_FileChunk(arcname="memory/stage_1.bin", data=<64MB>, is_last=False)
_FileChunk(arcname="memory/stage_1.bin", data=<64MB>, is_last=False)
_FileChunk(arcname="memory/stage_1.bin", data=<8MB>,  is_last=True)
```

The `stream_files()` method on `_NodeCollector` is a Python
generator.  Ray automatically treats it as a streaming actor method
(`num_returns="streaming"`).  Each `yield` produces an `ObjectRef`
that the driver can consume via the `ray.wait()` loop.

### Backpressure Design

Two independent mechanisms work together to bound memory (the
"double-layer backpressure" design):

**Layer 1 -- Generator-level (worker side):**
`_generator_backpressure_num_objects` limits how many `ObjectRef`s
each worker can have in flight.  With value=2, the worker pauses
its generator after yielding 2 unconsumed refs.

**Layer 2 -- Driver-level (driver side):**
Adaptive `ray.wait(num_returns=len(pending), timeout=0.1)` drains
all refs that become ready within a 100 ms window.  Each ready ref
is processed sequentially -- `ray.get()` materializes one chunk at
a time, writes it to disk, then frees it.  A cumulative stall guard
(`timeout_s`, default 600 s) detects genuine no-progress hangs.

```
Worker (_NodeCollector)                Driver (RayFileTransport)
+-------------------------+            +---------------------------+
| stream_files() generator|            |                           |
|                         |            |  acc.pending dict:        |
|  yield chunk_N  --[ref]-+--slot 1--->|  ray.wait(pending,       |
|                         |            |    num_returns=len(pend), |
|  yield chunk_N+1 -[ref]-+--slot 2    |    timeout=0.1)          |
|                         |   |        +---------------------------+
|  yield chunk_N+2  (*)   |   |                 |
|  BLOCKED until driver   |   |                 v
|  consumes slot 1        |   |         for ref in ready:
+-------------------------+   |           ray.get(ref): ~128 MB
                              |           written to disk, freed
   (*) generator paused       |                 |
   by Ray backpressure        |                 v
                              |         next(gen) pulls new ref
                              +-------> into pending, worker
                                        unblocked to yield
```

The combined per-node in-flight count: 1 ref in the pending dict
(being waited on) + 1 ref buffered in the generator (pre-fetched)
= 2.  The second slot provides pipeline overlap (latency hiding):
while the driver writes chunk N to disk, chunk N+1 is already
serialized/transferred in the object store background.

### Memory Accounting

A common misconception is that 2 backpressure slots means 2x
memory.  In reality:

```
Slot 1 -- ObjectRef in pending dict; bytes live in the Ray
          object store until ray.get() is called.  Cost on
          the driver: pointer only (~0 bytes).
Slot 2 -- ObjectRef pre-yielded by worker, not yet pulled
          by next(gen).  Bytes in the Ray object store.
          Cost on the driver: pointer only (~0 bytes).
ray.get() -- materializes ONE chunk into driver Python heap
             (~chunk_bytes = 128 MB), written to disk, then
             immediately GC-eligible.

Peak driver heap per node = chunk_bytes (128 MB), NOT 2x.
```

With N nodes, peak driver memory is N * 128 MB (one chunk per node
being written concurrently... but the driver is single-threaded, so
only one `ray.get()` materializes at a time).  Effective peak = 128
MB regardless of node count.

### Concurrency Model

Worker-side file reading is **truly parallel** -- each
`_NodeCollector` is a separate Ray actor process on a separate
machine.  All actors execute their `stream_files()` generators
simultaneously.  The driver consumes chunks **single-threaded but
interleaved** across all nodes via `ray.wait()`.

```
Time -->

Node A actor:  [read][yield][read][yield][read][yield][done]
Node B actor:  [read][yield][yield][read][yield][done]
Node C actor:  [read][yield][read][yield][read][yield][yield][done]
                    |    |     |      |     |     |      |     |
                    v    v     v      v     v     v      v     v
Object store:  ... chunks arriving from all nodes concurrently ...
                    |    |     |      |     |     |      |     |
Driver thread:   [wait][get+write][wait][get+write][wait][get+write]...
                 picks   B        picks    A       picks    C
                 first             next             next
                 ready             ready            ready

Total time = max(node_A_time, node_B_time, node_C_time)
NOT           node_A_time + node_B_time + node_C_time
```

**Why `ray.get()` is safe here:**

A naive `ray.get(ref)` blocks until the object is resolved --
dangerous if node A is slow while node B has data ready.  This
module avoids that pitfall by always calling `ray.wait()` first,
then `ray.get()` only on refs confirmed as ready (the "delayed
`ray.get()`" pattern from Ray docs):

```
# BAD -- blocks on A even if B is ready
chunk_a = ray.get(ref_a)  # waits 10s
chunk_b = ray.get(ref_b)  # B was ready instantly, wasted 10s

# GOOD -- only ray.get() refs that are already resolved
ready, _ = ray.wait([ref_a, ref_b, ref_c], num_returns=1)
# ready = [ref_b]  (B resolved first)
chunk = ray.get(ref_b)  # instant -- data already in object store
```

**Concurrency summary:**

| Aspect                          | Parallel? | Mechanism                 |
|---------------------------------|-----------|---------------------------|
| File reading across nodes       | Yes       | Separate actor processes  |
| File reading within one node    | No        | Single actor, sequential  |
| Chunk transfer to object store  | Yes       | Ray object store handles  |
| Driver chunk processing         | No (*)    | ray.wait interleaves      |
| ray.get() blocking risk         | Eliminated| Only after ray.wait ready |
| Total wall time                 | Slowest   | Not sum of all nodes      |

(*) Single-threaded but never the bottleneck: disk writes (~128 MB)
are much faster than network transfers from remote nodes.

### File Reassembly State Machine

On the driver side, `_process_chunk()` implements a state machine
over `_NodeState.current_handle` and `_NodeState.current_arcname`
to reassemble chunked files.  Three transitions are possible per
incoming chunk:

```
incoming chunk
      |
      v
arcname != current_arcname?
+-- YES (new file) ----+
|   close old handle   |
|   mkdir parents      |
|   open(dest, "wb")   |
|   current_arcname =  |
|     chunk.arcname    |
+----------+-----------+
           |
           v
  state.current_handle.write(chunk.data)
           |
           v
  chunk.is_last?
  +-- YES ----------+-- NO ---+
  |  close handle   |  (noop) |
  |  handle = None  |  return |
  |  arcname = ""   +---------+
  |  files_collected += 1
  +------------------+

State transitions for a 400 MB file (chunk_bytes=128 MB):

handle=None  --[chunk 0, new file]-->  handle=open("wb")
handle=open  --[chunk 1, same arc]-->  handle=open (append)
handle=open  --[chunk 2, same arc]-->  handle=open (append)
handle=open  --[chunk 3, is_last ]--> handle=None, count++
```

New files are detected by comparing `chunk.arcname` against
`state.current_arcname` rather than relying on a separate
"start-of-file" flag.  This keeps the `_FileChunk` protocol
minimal (3 fields) and avoids additional coordination state between
worker and driver.

### Multi-file Routing

The driver maintains a `pending` dict mapping each in-flight
`ObjectRef` to the `_NodeState` that produced it.  When
`ray.wait()` returns a ready ref, the driver looks up the
corresponding `_NodeState` to resume writing to the correct file
handle.

```
pending dict (one entry per active node):
+-----------+     +----------------+
| ObjectRef | --> | _NodeState     |
|  (chunk)  |     |   node_name    |
+-----------+     |   gen          |
                  | current_handle |
                  |   files_done=3 |
                  +----------------+
```

Chunks from different nodes arrive in arbitrary order.  A chunk
from node A might arrive, then two from node B, then another from
node A.  The `_NodeState` per-node tracking ensures each chunk is
written to the correct file handle regardless of interleaving.

After processing a ready ref, the driver calls `next(gen)` to
advance the node's generator and obtain the next `ObjectRef`.  This
ref is inserted into `pending` to replace the consumed one.  When
the generator is exhausted (`StopIteration`), the node is moved to
the `ok_nodes` list.

### Error Handling

Error handling provides per-node isolation with configurable
strictness:

- **Per-node isolation**: one node timing out or raising does not
  abort collection from other nodes (when `strict=False`).
- **`strict=True`**: any error raises immediately.
- **`strict=False`** (default): errors are logged as warnings and
  collection continues for remaining nodes.
- **Timeout handling**: if `ray.wait()` returns no ready refs
  within `timeout_s`, all pending nodes are marked as failed and
  the loop breaks.
- **Cleanup guarantees**: the `finally` block kills all actors via
  `ray.kill()` and calls `_finalize_node()` to close any open file
  handles, regardless of success or failure.

```
collect()
|
+-- try:
|     ray.wait() loop
|       |
|       +-- ready? -> process chunk, advance generator
|       +-- timeout? -> mark all pending as failed, break
|       +-- exception?
|             strict=True  -> raise
|             strict=False -> log warning, mark failed
|
+-- finally:
      ray.kill() all actors
      _finalize_node() all states (close open handles)
```

## ArtifactDelivery (High-level Orchestrator)

`ArtifactDelivery` is the generic orchestrator for the three-phase
artifact lifecycle: staging, cross-node collection, and upload.  It
knows nothing about the nature of the files it moves -- all
behaviour is controlled by constructor parameters (`kind`,
`upload_subdir`, `collect_on_shutdown`, `strict`).

See `cosmos_curator/core/utils/artifacts/delivery.py` for
implementation.

### Three-phase Lifecycle

```
Phase 1: Staging (before pipeline)
====================================

  ArtifactDelivery.create(kind="<kind>", output_dir=...)
        |
        +-- Set COSMOS_CURATOR_ARTIFACTS_STAGING_DIR env var
        +-- Derive staging_dir = <base>/<kind>
        +-- Register collect() as pre-shutdown hook (optional)

Phase 2: Pipeline execution
====================================

  Workers write artifacts to <staging>/<kind>/ on local disk.
  No remote uploads during the hot path.

Phase 3: Collection and upload (post-pipeline)
====================================================

  collect()
        |
        +-- _collected guard (idempotent, no-op on 2nd call)
        +-- StorageWriter(upload_dest).is_remote?
        |       |                       |
        |       v (yes)                 v (no)
        |   _collect_remote()      _collect_local(writer)
        |   Deploy _NodeUploader   RayFileTransport.collect()
        |   on each node           directly into dest dir
        |   (direct S3/Azure)      (no temp dir, no rewrite)
        |       |                       |
        +-------+-----------------------+
        |
        +-- return file count
```

### Factory Pattern

Consumers call `ArtifactDelivery.create()` instead of `__init__()`
because the factory handles side effects that a constructor should
not perform:

1. Reads or creates the base staging directory from the
   `COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` env var.  If the env var
   is already set (e.g. by a prior `create()` call from another
   subsystem), it is read idempotently.
2. Derives the staging subdirectory as `<base>/<kind>` so different
   subsystems are isolated.
3. When `collect_on_shutdown=True` (default), registers `collect()`
   as a pre-shutdown hook via `register_pre_shutdown_hook()`.

### Collection Routing

`StorageWriter.is_remote` is the single routing decision point.
`ArtifactDelivery` never inspects path prefixes (`s3://`, `az://`)
directly -- it delegates that knowledge to `StorageWriter`.

```
Remote destination (S3 / Azure)
===============================

Worker A: staging/ --+--> StorageWriter --> S3
Worker B: staging/ --+--> StorageWriter --> S3
Worker C: staging/ --+--> StorageWriter --> S3
    (each node uploads directly; driver only coordinates)

Local destination
=================

Worker A: staging/ --+
Worker B: staging/ --+--> RayFileTransport --> driver --> local dir
Worker C: staging/ --+
    (files gathered to driver first, then written locally)
```

### Local Collection Path

`_collect_local()` is used when the destination is a local path.
Worker nodes may not share a filesystem with the driver, so
`RayFileTransport` streams file data through the Ray object store
to the driver.

Files land at the final destination directly -- there is no
intermediate temp directory.  This avoids doubling disk I/O.  On
partial failure, whatever files were already delivered remain at the
destination for operator inspection.

### Remote Collection Path

`_collect_remote()` deploys a `_NodeUploader` actor on each node
via `NodeAffinitySchedulingStrategy`.  Each actor reads files from
the local staging directory and uploads directly to S3/Azure via
its own `StorageWriter` instance.

No file data crosses the cluster network -- only metadata (upload
counts, error messages) returns to the driver.

**Concurrency comparison with `RayFileTransport`:**

| Aspect                  | RayFileTransport                              | Remote collection                              |
|-------------------------|-----------------------------------------------|------------------------------------------------|
| Pattern                 | Adaptive `ray.wait` (batched, 100 ms timeout) | Adaptive `ray.wait` (batched, 100 ms timeout)  |
| Driver receives         | File bytes (chunks)                            | Metadata only (`_NodeUploadResult`)             |
| Network transfer        | Worker -> Driver -> Disk                       | Worker -> S3 directly                           |
| Backpressure            | Per-chunk via adaptive `ray.wait`              | Per-node via adaptive `ray.wait`                |
| Driver bottleneck       | Yes (writes to local disk)                     | No (only coordinates)                           |

Both paths use the same adaptive `ray.wait(num_returns=len(pending),
timeout=0.1)` batching pattern.  Remote collection is lighter
because all actors upload directly to S3/Azure -- the driver only
receives lightweight `_NodeUploadResult` metadata, not file bytes.

### Why Collection is Deferred

Collection and upload are deliberately **deferred to the end of
execution** rather than performed inline during the pipeline run.
Moving artifacts while stages are actively processing would compete
for network bandwidth, disk I/O, and CPU cycles -- directly
impacting pipeline throughput and stage latency.  By staging files
locally during the run and collecting them in a single batch after
the pipeline completes, the hot path remains uncontested.

By default, delivery instances register as Ray pre-shutdown hooks.
Consumers that need to control collection timing can pass
`collect_on_shutdown=False` to `ArtifactDelivery.create()` and
call `collect()` explicitly.

This design also ensures artifacts survive worker crashes (SIGKILL)
because collection happens from the driver after the pipeline
completes.  File-name collisions across nodes are avoided when
consumers include the hostname and PID in artifact names.

### Idempotency, Crash-safety, and Partial-failure Tolerance

Three guarantees apply to both local and remote collection:

- **Idempotent**: the `_collected` flag prevents double-execution
  if both the pre-shutdown hook and an explicit call fire.
- **Crash-safe**: on failure, the local collection directory is
  preserved on disk and its path is logged so operators can manually
  retry the upload.  On success, it is cleaned up.
- **Partial-failure tolerant** (`strict=False`): nodes that fail
  to deliver files are logged as warnings; collection continues for
  the remaining nodes so no data is unnecessarily lost.
- **Fail-fast** (`strict=True`): the first failure raises
  `ArtifactDeliveryError` so the caller can abort early.  The
  exception chain preserves the original error for inspection.

### Instrumentation During Shutdown

`collect()` is instrumented with `traced_span`.  When it fires as a
pre-shutdown hook, the instrumentation provider on the driver is
still alive, so spans are created normally.  However, if a consumer
also uses `ArtifactDelivery` to collect instrumentation output
files, LIFO hook ordering means the collection span itself may be
written **after** those output files have already been gathered.
As a result, spans from shutdown-triggered collection are
best-effort -- they are fully captured when
`collect_on_shutdown=False` and the consumer calls `collect()`
explicitly during normal execution.

## Environment Variables

The staging directory is communicated from the driver to workers via
environment variables.  An env var is necessary because
`ArtifactDelivery.create()` runs on the driver, while Ray workers
(actors) need to write to the same path.  Workers inherit the
driver's environment when forked, so the env var is the simplest
reliable mechanism to share a temp directory path across all nodes
without adding parameters to stage APIs.

| Variable | Set by | Read by | Purpose |
|---|---|---|---|
| `COSMOS_CURATOR_ARTIFACTS_STAGING_DIR` | `ArtifactDelivery.create()` | Consumer backends on workers | Shared base staging directory for all artifact kinds |

Additional consumer-specific environment variables (e.g. for
tracing or profiling) are documented alongside their respective
subsystems.  See the
[Profiling Guide](../guides/profiling.md#artifact-delivery-flow)
for an example.

## Source File Reference

| File | Description |
|---|---|
| `cosmos_curator/core/utils/artifacts/__init__.py` | Package overview docstring |
| `cosmos_curator/core/utils/artifacts/collector.py` | `RayFileTransport`, `_NodeCollector`, `_FileChunk`, `CollectResult` |
| `cosmos_curator/core/utils/artifacts/delivery.py` | `ArtifactDelivery`, `_NodeUploader`, `ArtifactDeliveryError` |
