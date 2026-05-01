# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-node file transport using Ray streaming generators.

Collects files from a local staging directory on each Ray node and
delivers them to a central output directory on the driver.  Designed
for crash-safe artifact delivery: workers write to a local staging
directory during execution, and this module gathers those files
**post-pipeline** so that artifacts survive even if workers are
killed (SIGKILL) during the run.

This is a **generic** transport -- not coupled to profiling, tracing,
or any specific subsystem.  Any code that needs to move files from
Ray worker nodes to the driver can use :class:`RayFileTransport`.

Design rationale
~~~~~~~~~~~~~~~~

This module follows Ray best practices documented in the Ray core
patterns and tips guides:

* **ray.wait() pipeline processing** (Ray docs "Tip 4: Pipeline data
  processing") -- chunks from all nodes are processed as they arrive
  via ``ray.wait(num_returns=len(pending), timeout=0.1)``, so total
  collection time equals the slowest node, not the sum of all nodes.
  The short timeout drains all refs ready within 100 ms per call,
  reducing gRPC round-trips on large clusters (1000+ nodes) from
  O(total_chunks) to O(total_chunks / batch_size).

* **Limit-pending-tasks pattern** (Ray docs "Pattern: Using ray.wait
  to limit the number of pending tasks") -- the adaptive
  ``ray.wait()`` combined with immediate process-and-discard keeps
  the pending set bounded and prevents unbounded memory growth on the
  driver.

* **Delayed ray.get()** (Ray docs "Tip 1: Delay ray.get()") -- we
  only call ``ray.get(ref)`` after ``ray.wait()`` confirms the ref is
  ready, never eagerly.  This ensures the driver never blocks waiting
  for a single node while another node has data available.

* **Double-layer backpressure** -- two independent mechanisms work
  together to bound memory:

  1. **Generator-level** (worker side):
     ``_generator_backpressure_num_objects`` limits how many
     ``ObjectRef``s each worker can have in flight.  With value=2,
     the worker pauses its generator after yielding 2 unconsumed
     refs.

  2. **Driver-level** (driver side): ``ray.wait()`` with a short
     timeout processes chunks in adaptive batches.  After writing
     each chunk to disk, the ``_FileChunk`` becomes
     garbage-collectable immediately.

  Combined per-node in-flight: 1 ref in the pending dict (being
  waited on) + 1 ref buffered in the generator (pre-fetched) = 2.
  Peak driver memory = ``chunk_bytes`` (one chunk being written).

* **Why NOT ray.util.as_completed()** -- ``as_completed()`` operates
  on a fixed list of ``ObjectRef``s.  Our streaming generators produce
  refs incrementally: after processing one chunk, we call
  ``next(gen)`` to advance the generator and obtain the next ref.
  This dynamic ref-generation pattern is incompatible with
  ``as_completed()``.  The custom ``ray.wait()`` loop handles it
  naturally.

* **Adaptive short-timeout batching** -- on large clusters (1000+
  nodes), ``ray.wait(num_returns=1)`` causes O(total_chunks) gRPC
  round-trips, each passing the full pending-refs list.  Instead,
  ``ray.wait(num_returns=len(pending), timeout=0.1)`` drains all
  refs ready within 100 ms in a single call.  A cumulative stall
  guard (``timeout_s``) replaces the per-call timeout to detect
  genuine no-progress situations.  On small clusters, few refs
  arrive per 100 ms window, so behaviour is equivalent to
  ``num_returns=1`` with negligible latency overhead.

* ``NodeAffinitySchedulingStrategy`` for node pinning (codebase
  convention), not ``STRICT_SPREAD`` placement groups.

* ``num_cpus=0`` on ``_NodeCollector`` -- the actor does file I/O
  only, not compute.  Zero CPU reservation avoids starving pipeline
  actors of compute resources.

* Per-node error isolation -- one node timing out or raising does not
  abort collection from other nodes.

* Explicit actor cleanup via ``ray.kill()`` in a ``finally`` block.

* No cluster lifecycle management -- the caller owns Ray.

Concurrency model
~~~~~~~~~~~~~~~~~

Worker-side file reading is **truly parallel** -- each
``_NodeCollector`` is a separate Ray actor process on a separate
machine.  All actors execute their ``stream_files()`` generators
simultaneously.  The driver consumes chunks **single-threaded but
interleaved** across all nodes via ``ray.wait()``.

Two levels of concurrency work together:

1. **Workers (parallel)**: each actor reads files from its local
   disk and yields ``_FileChunk`` objects into the Ray object store.
   All actors run concurrently -- node A yields chunks while node B
   yields chunks while node C yields chunks.

2. **Driver (single-threaded, interleaved)**: the ``collect()``
   method calls ``ray.wait(list(pending), num_returns=len(pending),
   timeout=0.1)`` which returns all refs ready within 100 ms.
   Whichever node's chunks arrive first get processed first.  The
   driver never waits for node A to finish before touching node B.

::

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

Why ``ray.get()`` is safe here
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A naive ``ray.get(ref)`` blocks until the object is resolved --
dangerous if node A is slow while node B has data ready:

::

    # BAD -- blocks on A even if B is ready
    chunk_a = ray.get(ref_a)  # waits 10s
    chunk_b = ray.get(ref_b)  # B was ready instantly, wasted 10s

This module avoids that pitfall by always calling ``ray.wait()``
first, then ``ray.get()`` only on refs confirmed as ready:

::

    # GOOD -- only ray.get() refs that are already resolved
    ready, _ = ray.wait([ref_a, ref_b, ref_c], num_returns=1)
    # ready = [ref_b]  (B resolved first)
    chunk = ray.get(ref_b)  # instant -- data already in object store

The subsequent ``ray.get()`` is essentially free -- it just
deserialises bytes from the object store into the driver's Python
heap.  No network wait occurs because the object is already local.

Concurrency summary:

+---------------------------------+-----------+---------------------------+
| Aspect                          | Parallel? | Mechanism                 |
+---------------------------------+-----------+---------------------------+
| File reading across nodes       | Yes       | Separate actor processes  |
| File reading within one node    | No        | Single actor, sequential  |
| Chunk transfer to object store  | Yes       | Ray object store handles  |
| Driver chunk processing         | No (*)    | ray.wait interleaves      |
| ray.get() blocking risk         | Eliminated| Only after ray.wait ready |
| Total wall time                 | Slowest   | Not sum of all nodes      |
+---------------------------------+-----------+---------------------------+

(*) Single-threaded but never the bottleneck: disk writes (~128 MB)
are much faster than network transfers from remote nodes.

::

    +-- Node A --+     +-- Node B --+     +-- Node C --+
    | staging/   |     | staging/   |     | staging/   |
    | cpu/...    |     | traces/... |     | memory/... |
    +-----+------+     +-----+------+     +-----+------+
          |                  |                  |
          v                  v                  v
    _NodeCollector     _NodeCollector     _NodeCollector
    stream_files()     stream_files()     stream_files()
    yield chunks       yield chunks       yield chunks
          |                  |                  |
          +--------+---------+--------+---------+
                   |
                   v
          ray.wait(pending, num_returns=len(pending),
                   timeout=_WAIT_BATCH_TIMEOUT_S)
                   |
           ready?--+--empty (stall)?
           |              |
           v              v
      for ref in ready:   stall_s += dt
      ray.get(ref)        stall_s >= timeout_s?
      _process_chunk        YES -> failed, break
      next(gen) -> pending  NO  -> continue
"""

import contextlib
import os
import pathlib
from collections.abc import Generator
from typing import IO

import attrs
import numpy as np
import numpy.typing as npt
import ray
from loguru import logger

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy
from cosmos_curator.core.utils.infra import ray_cluster_utils
from cosmos_curator.core.utils.infra.tracing import TracedSpan, traced_span

# Default chunk size for streaming file transfers.
# Chosen to stay well under the 512 MB gRPC limit
# (ray/_private/ray_constants.py:GRPC_CPP_MAX_MESSAGE_SIZE)
# while being large enough to amortize per-chunk overhead.
# 128 MB reduces per-chunk ray.wait/ray.get round-trip overhead
# on fast networks where transfer time is small relative to
# per-cycle dispatch cost.
_DEFAULT_CHUNK_BYTES: int = 128 * 1024 * 1024  # 128 MB

# Part of the double-layer backpressure design.
#
# Layer 1 (generator, worker side): limits in-flight ObjectRefs per
# worker to this value.  The worker's generator pauses after yielding
# this many unconsumed refs.
#
# Layer 2 (driver side): ray.wait(num_returns=1) ensures we process
# and free one chunk before requesting the next.
#
# With value=2: one ref being transferred/resolved + one ref
# pre-fetched by the generator.  Peak memory per node on the driver
# side = chunk_bytes (one chunk being written to disk).
#
# Per-node queue layout (value=2):
#
#   Worker (_NodeCollector)                Driver (RayFileTransport)
#   +-------------------------+            +------------------------+
#   | stream_files() generator|            |                        |
#   |                         |            |  acc.pending dict:     |
#   |  yield chunk_N  --[ref]-+--slot 1--->|  1 ref in ray.wait()  |
#   |                         |            |  (ray.get -> disk)     |
#   |  yield chunk_N+1 -[ref]-+--slot 2    |                        |
#   |                         |   |        +------------------------+
#   |  yield chunk_N+2  (*)   |   |                 |
#   |  BLOCKED until driver   |   |                 v
#   |  consumes slot 1        |   |         ray.get(ref): ~128 MB
#   +-------------------------+   |         materialized in driver
#                                 |         Python heap, written
#      (*) generator paused       |         to disk, then freed
#      by Ray backpressure        |                 |
#                                 |                 v
#                                 |         next(gen) pulls slot 2
#                                 +-------> into pending (slot 1),
#                                           worker unblocked to
#                                           yield into new slot 2
#
# Memory accounting per node (steady state):
#   slot 1 -- ObjectRef in pending dict; bytes live in the Ray
#             object store until ray.get() is called.  Cost on
#             the driver: pointer only (~0 bytes).
#   slot 2 -- ObjectRef pre-yielded by worker, not yet pulled
#             by next(gen).  Bytes in the Ray object store.
#             Cost on the driver: pointer only (~0 bytes).
#   ray.get() -- materialises ONE chunk into driver Python heap
#                (~chunk_bytes = 128 MB), written to disk, then
#                immediately GC-eligible.
#
# Peak driver heap per node = chunk_bytes (128 MB), NOT 2x.
# The second slot provides pipeline overlap (latency hiding):
# while the driver writes chunk N to disk, chunk N+1 is already
# serialised/transferred in the object store background.
_BACKPRESSURE_OBJECTS: int = 2

# Short timeout for the adaptive ``ray.wait()`` batching strategy.
#
# Instead of ``ray.wait(pending, num_returns=1)`` which makes one gRPC
# round-trip per chunk (O(total_chunks) calls on large clusters), we use
# ``ray.wait(pending, num_returns=len(pending), timeout=0.1)`` to drain
# ALL refs that become ready within 100 ms in a single call.
#
# This is adaptive:
#   - Busy 1000-node cluster: many chunks arrive within 100 ms, all
#     drained in one batch -- dramatically fewer gRPC round-trips.
#   - Quiet cluster with few nodes: 1-2 refs ready in 100 ms, processed
#     immediately -- no unnecessary delay.
#   - No risk of blocking for a slow straggler (unlike a hard
#     ``num_returns=N`` which waits until N refs are ready).
#
# The original ``timeout_s`` (default 600 s) becomes a stall guard:
# if no chunk from ANY node arrives for ``timeout_s`` cumulative
# seconds, the loop triggers ``_handle_timeout``.  The stall counter
# resets to zero on every successful batch.
_WAIT_BATCH_TIMEOUT_S: float = 0.1


@attrs.define(frozen=True)
class _FileChunk:
    """A chunk of a file being streamed from a remote node.

    This is the internal protocol between ``_NodeCollector`` (worker)
    and ``RayFileTransport`` (driver).  It is a typed immutable data
    class rather than a raw tuple -- per the abstraction-boundaries
    rule -- to provide self-documenting fields and type safety across
    the Ray serialization boundary.

    Small files (below ``chunk_bytes``) produce a single chunk with
    ``is_last=True`` containing the entire file.  Large files produce
    multiple chunks sharing the same ``arcname``; the driver appends
    them sequentially until ``is_last=True``.

    ::

        Small file (100 KB, chunk_bytes=128 MB):
            _FileChunk(arcname="cpu/stage_1.html", data=<100KB>, is_last=True)

        Large file (400 MB, chunk_bytes=128 MB):
            _FileChunk(arcname="memory/stage_1.bin", data=<128MB>, is_last=False)
            _FileChunk(arcname="memory/stage_1.bin", data=<128MB>, is_last=False)
            _FileChunk(arcname="memory/stage_1.bin", data=<128MB>, is_last=False)
            _FileChunk(arcname="memory/stage_1.bin", data=<16MB>,  is_last=True)

    Attributes:
        arcname: Relative path within the staging directory.
        data: Raw bytes for this chunk.
        is_last: True for the final (or only) chunk of this file.

    """

    arcname: str
    data: npt.NDArray[np.uint8]
    is_last: bool


@ray.remote(num_cpus=0)
class _NodeCollector:
    """Ephemeral actor that streams files from local staging on its node.

    Pinned to a specific node via ``NodeAffinitySchedulingStrategy``
    at creation time.  Uses ``num_cpus=0`` because the workload is
    purely I/O-bound (reading files from local disk) -- reserving CPU
    would unnecessarily starve pipeline actors of compute resources.

    The ``stream_files()`` method is a Python generator -- Ray
    automatically treats it as a streaming actor method.  Each
    ``yield`` produces an ``ObjectRef`` that the driver can consume
    via the ``ray.wait()`` loop.

    Small files (below ``chunk_bytes``) are yielded in a single
    ``_FileChunk``.  Large files are read in ``chunk_bytes``
    increments and yielded as multiple ``_FileChunk`` objects sharing
    the same ``arcname``, with ``is_last=True`` on the final chunk.

    Backpressure is configured by the caller via
    ``_generator_backpressure_num_objects``.  When the driver falls
    behind (e.g. slow disk writes), the worker's generator pauses
    automatically, preventing unbounded memory growth in the Ray
    object store.

    ::

        _NodeCollector.stream_files(staging_dir, chunk_bytes)
        |
        +-- pathlib.Path(staging_dir).rglob("*")
        |       Find all files recursively
        |
        +-- for each file:
        |     file_size <= chunk_bytes?
        |       YES -> yield _FileChunk(arcname, all_bytes, is_last=True)
        |       NO  -> for each chunk_bytes-sized read:
        |                yield _FileChunk(arcname, chunk, is_last=...)
        |
        |     Backpressure pauses generator if driver is slow
    """

    def stream_files(
        self,
        staging_dir: str,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
        **_ray_kwargs: object,
    ) -> Generator[_FileChunk, None, None]:
        """Yield ``_FileChunk`` objects for each file under *staging_dir*.

        This is a streaming generator method.  Ray detects the
        generator protocol automatically and sets
        ``num_returns="streaming"``.  Each ``yield`` produces an
        ``ObjectRef`` that the driver can ``ray.get()`` immediately.

        Backpressure is configured by the caller via
        ``_generator_backpressure_num_objects`` so the worker blocks
        when the driver falls behind, keeping peak memory bounded to
        ``chunk_bytes``.

        The ``**_ray_kwargs`` parameter absorbs Ray-injected keyword
        arguments (e.g. ``_ray_trace_ctx``) that Ray's tracing
        infrastructure passes to streaming generator methods when
        ``_tracing_startup_hook`` is configured via ``ray.init()``.
        For non-streaming remote methods, Ray strips these arguments
        internally.  For streaming generators, they leak through to
        the user function signature.

        Args:
            staging_dir: Absolute path to the local staging
                directory on this node.
            chunk_bytes: Maximum bytes per yielded chunk.  Files
                smaller than this are yielded in one piece.  Larger
                files are split into multiple sequential chunks.

        Yields:
            ``_FileChunk`` objects -- one per small file, or
            multiple per large file (same ``arcname``, last has
            ``is_last=True``).

        """
        staging = pathlib.Path(staging_dir)
        if not staging.exists():
            return

        for file_path in staging.rglob("*"):
            if not file_path.is_file():
                continue

            arcname = str(file_path.relative_to(staging))
            file_size = file_path.stat().st_size

            if file_size <= chunk_bytes:
                # Small file: single chunk with all bytes.
                yield _FileChunk(
                    arcname=arcname,
                    data=np.fromfile(file_path, dtype=np.uint8),
                    is_last=True,
                )
            else:
                # Large file: stream in chunk_bytes-sized reads.
                with file_path.open("rb") as fh:
                    while True:
                        raw = fh.read(chunk_bytes)
                        if not raw:
                            break
                        is_last = fh.tell() >= file_size
                        yield _FileChunk(
                            arcname=arcname,
                            data=bytes_to_numpy(raw),
                            is_last=is_last,
                        )


@attrs.define(frozen=True)
class _StreamConfig:
    """Per-collection configuration for the ``ray.wait()`` loop.

    Groups the transport parameters into a single typed object so
    that private methods stay under the 5-argument limit (per the
    abstraction-boundaries rule).

    Attributes:
        staging_dir: Staging directory path (same on every node).
        output_dir: Local directory on the driver where files are
            written.
        timeout_s: Maximum seconds to wait per ``ray.wait()`` call.
        chunk_bytes: Maximum bytes per streamed chunk.

    """

    staging_dir: str
    output_dir: pathlib.Path
    timeout_s: float
    chunk_bytes: int


@attrs.define
class _NodeState:
    """Mutable per-node reassembly state for the ``ray.wait()`` loop.

    Each active node in the ``collect()`` loop has one ``_NodeState``
    instance that tracks the streaming generator, the current file
    handle being written, and the number of completed files.

    **Mutable by design**: the ``ray.wait()`` loop processes chunks
    from any node in any order.  When a chunk arrives, its
    ``_NodeState`` is looked up in the ``pending`` dict to resume
    writing to the correct file handle.

    The ``gen`` field holds a Ray ``StreamingObjectRefGenerator``
    (returned by ``actor.stream_files.options(...).remote(...)``).
    It is typed as ``object`` because Ray does not expose a public
    type for streaming generators.  Advancing with ``next(gen)``
    returns the next ``ObjectRef`` quickly (ref creation is
    near-instant on the driver; blocking happens only at
    ``ray.get()``).

    ::

        pending dict (one entry per active node):
        +-----------+     +----------------+
        | ObjectRef | --> | _NodeState     |
        |  (chunk)  |     |   node_name    |
        +-----------+     |   gen          |
                          | current_handle |
                          |   files_done=3 |
                          +----------------+

    Attributes:
        node_name: Human-readable node name for log messages.
        gen: Ray streaming generator handle (``next(gen)`` returns
            the next ``ObjectRef``).
        files_collected: Number of complete files written so far.
        current_handle: Open file handle for the file currently
            being reassembled, or ``None`` between files.
        current_arcname: Relative path of the file currently being
            written, or empty string between files.

    """

    node_name: str
    gen: object
    files_collected: int = 0
    current_handle: IO[bytes] | None = None
    current_arcname: str = ""


@attrs.define
class _CollectAccumulator:
    """Mutable accumulator for the ``ray.wait()`` loop outcomes.

    Groups the per-collection mutable state (pending refs, ok/failed
    node lists) into a single object so helper methods can accept it
    as one argument instead of three, keeping them under the
    5-argument limit.

    Attributes:
        pending: Maps each in-flight ``ObjectRef`` to the
            ``_NodeState`` that produced it.
        ok_nodes: Node names that completed collection.
        failed_nodes: ``(node_name, error_msg)`` pairs for failures.

    """

    pending: dict[ray.ObjectRef, _NodeState] = attrs.Factory(dict)  # type: ignore[type-arg]
    ok_nodes: list[str] = attrs.Factory(list)
    failed_nodes: list[tuple[str, str]] = attrs.Factory(list)


@attrs.define(frozen=True)
class CollectResult:
    """Outcome of a :meth:`RayFileTransport.collect` call.

    Reports per-node success/failure so the consumer can make
    informed decisions about partial collection failures (e.g. log
    warnings and proceed, or abort the pipeline).

    ::

        result = transport.collect(...)

        result.total_files    -> 42
        result.nodes_ok       -> ("node-a", "node-b")
        result.nodes_failed   -> (("node-c", "TimeoutError: ..."),)
        result.all_succeeded  -> False
        result.partial        -> True

    Attributes:
        total_files: Total number of complete files collected
            across all successful nodes.
        nodes_ok: Node names that completed collection without
            error.
        nodes_failed: ``(node_name, error_message)`` pairs for
            each node that failed during collection.

    """

    total_files: int
    nodes_ok: tuple[str, ...]
    nodes_failed: tuple[tuple[str, str], ...]

    @property
    def all_succeeded(self) -> bool:
        """True when every node completed without error."""
        return len(self.nodes_failed) == 0

    @property
    def partial(self) -> bool:
        """True when some nodes succeeded and some failed."""
        return bool(self.nodes_ok) and bool(self.nodes_failed)


class RayFileTransport:
    """Generic cross-node file transport using Ray streaming generators.

    Collects files from a local staging directory on each Ray node
    and delivers them to a central output directory on the driver.
    Fully decoupled from any specific subsystem -- usable for
    profiling artifacts, traces, pipeline outputs, or any other
    files that workers produce during execution.

    This transport does NOT manage the Ray cluster lifecycle -- the
    caller must ensure Ray is initialized before calling ``collect()``
    and keeps it alive until collection completes.

    Files are transferred using Ray's streaming generator protocol
    with per-file chunking and double-layer backpressure (see module
    docstring).  Peak memory on the driver is bounded to
    ``chunk_bytes`` (default 128 MB) per active node, regardless of
    total staging directory size or individual file size.

    Collection uses ``ray.wait()`` to consume chunks from all nodes
    in parallel: whichever node has data ready first gets processed
    first.  Total collection time equals the slowest node, not the
    sum of all nodes (Ray docs "Tip 4: Pipeline data processing").

    ::

        RayFileTransport
        +-- _StreamConfig       (per-collection config)
        +-- _NodeState          (per-node reassembly state)
        +-- _NodeCollector      (per-node streaming actor)
        +-- CollectResult       (per-collection outcome)

        collect(staging_dir, output_dir, ...)
        |
        +-- 1. get_live_nodes()
        |       strict=True: raise RuntimeError if no nodes
        |
        +-- 2. _deploy_collectors(nodes, cfg) -> actors
        |       NodeAffinitySchedulingStrategy(node_id, soft=False)
        |
        +-- 3. _seed_generators(actors, cfg) -> pending, ok_nodes
        |       Start all generators, get first ObjectRef from each
        |       Empty generators -> ok_nodes immediately
        |
        +-- 4. ray.wait() loop (adaptive batching):
        |       stall_s = 0.0
        |       while pending:
        |           ready, _ = ray.wait(list(pending),
        |               num_returns=len(pending),
        |               timeout=_WAIT_BATCH_TIMEOUT_S)
        |           |
        |           +-- empty (stall): stall_s += dt
        |           |     stall_s >= timeout_s -> failed_nodes, break
        |           +-- Ready (batch):
        |               stall_s = 0.0
        |               for ref in ready:
        |                 pop state from pending
        |                 chunk = ray.get(ref)
        |                 _process_chunk(state, chunk, output_dir)
        |                 next(gen) -> pending[next_ref] = state
        |                 StopIteration -> _finalize_node, ok_nodes
        |                 Exception -> strict ? raise : warn, failed
        |
        +-- 5. Return CollectResult
        |
        +-- finally: ray.kill() all actors
        |           _finalize_node() all states

    Usage::

        transport = RayFileTransport()
        result = transport.collect(
            staging_dir="/tmp/cosmos_curator_staging",
            output_dir=pathlib.Path("/tmp/collected"),
        )
        logger.info(f"Collected {result.total_files} artifacts")

    """

    @staticmethod
    def _process_chunk(
        state: _NodeState,
        chunk: _FileChunk,
        output_dir: pathlib.Path,
    ) -> None:
        """Write a single chunk to disk, updating per-node state.

        Implements the process-and-discard pattern from the Ray
        limit-pending-tasks pattern: chunk data is written to disk
        immediately and the ``_FileChunk`` object becomes
        garbage-collectable, keeping driver memory bounded.

        The method manages a small state machine over
        ``state.current_handle`` and ``state.current_arcname``.
        Three transitions are possible per incoming chunk:

        ::

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

        Design decision: we detect new files by comparing
        ``chunk.arcname`` against ``state.current_arcname`` rather
        than relying on a separate "start-of-file" flag.  This
        keeps the ``_FileChunk`` protocol minimal (3 fields) and
        avoids an additional coordination state between worker and
        driver.

        Edge cases:
            - Back-to-back small files: each has ``is_last=True``,
              so the handle opens and closes on every chunk.
            - Worker yields chunks for a new file without sending
              ``is_last=True`` for the previous file (should not
              happen): the old handle is closed when the arcname
              changes, so no file descriptor leak occurs.
            - ``open()`` failure (disk full, permissions): the old
              handle is already closed and cleared to ``None``
              before the ``open()`` attempt.  The exception
              propagates to ``_consume_ready_ref``, which calls
              ``_finalize_node`` -- seeing ``None``, it does
              nothing.  No double-close, no fd leak, no silent
              suppression.
            - ``close()`` failure on ``is_last`` (NFS flush error):
              ``try/finally`` guarantees the state machine resets
              (``handle=None``, ``arcname=""``, count incremented)
              even if ``close()`` raises.  The exception still
              propagates -- it is NOT suppressed.

        Args:
            state: Mutable per-node state to update.
            chunk: The chunk to write.
            output_dir: Root output directory on the driver.

        """
        # New file detected: close old handle, open new one.
        if chunk.arcname != state.current_arcname:
            if state.current_handle is not None:
                logger.warning(
                    f"[RayFileTransport] New file {chunk.arcname!r} arrived before "
                    f"is_last=True for {state.current_arcname!r} on node "
                    f"{state.node_name}; truncating previous file"
                )
                state.current_handle.close()
                # Clear immediately so _finalize_node sees None if
                # the open() below fails.  Without this, the stale
                # (already-closed) handle would cause a double-close
                # attempt in _finalize_node.  The exception from
                # open() still propagates -- we are NOT guarding it.
                state.current_handle = None
            dest = output_dir / chunk.arcname
            dest.parent.mkdir(parents=True, exist_ok=True)
            state.current_handle = dest.open("wb")
            state.current_arcname = chunk.arcname

        state.current_handle.write(chunk.data)  # type: ignore[union-attr]

        # Last chunk of this file: close handle, count it.
        # try/finally ensures the state machine resets even if
        # close() raises (e.g. NFS flush error).  The exception
        # still propagates -- we do NOT suppress it.
        if chunk.is_last:
            try:
                state.current_handle.close()  # type: ignore[union-attr]
            finally:
                state.current_handle = None
                state.current_arcname = ""
                state.files_collected += 1

    @staticmethod
    def _finalize_node(state: _NodeState) -> None:
        """Close any open file handle on a node state.

        Called when a node completes (``StopIteration``), errors out,
        or during the ``finally`` cleanup.  Partial files remain on
        disk for debugging -- they are not deleted.

        Args:
            state: Per-node state whose handle should be closed.

        """
        if state.current_handle is not None:
            with contextlib.suppress(Exception):
                state.current_handle.close()
            state.current_handle = None

    @staticmethod
    def _deploy_collectors(
        nodes: list[dict[str, str]],
    ) -> list[tuple[str, ray.actor.ActorHandle, _NodeState]]:  # type: ignore[type-arg]
        """Deploy one ``_NodeCollector`` actor per node with hard affinity.

        Each actor is pinned to its node via
        ``NodeAffinitySchedulingStrategy(soft=False)`` so it reads
        files from that node's local filesystem.  Hard affinity
        (``soft=False``) guarantees the actor runs on the target
        node -- if Ray cannot place it there, the actor creation
        fails rather than silently landing on the wrong node.

        ::

            get_live_nodes() -> [node_info_A, node_info_B, ...]
                  |
                  v
            for each node_info:
              +------------------------------------------+
              | node_id   = node_info["NodeID"]          |
              | node_name = node_info["NodeName"]        |
              |                                          |
              | actor = _NodeCollector.options(           |
              |   scheduling_strategy=                   |
              |     NodeAffinitySchedulingStrategy(       |
              |       node_id=node_id, soft=False))      |
              | .remote()                                |
              |                                          |
              | state = _NodeState(node_name, gen=None)  |
              +------------------------------------------+
                  |
                  v
            result: [(node_name_A, actor_A, state_A),
                     (node_name_B, actor_B, state_B), ...]

        Design decisions:
            - ``num_cpus=0`` (set on the class decorator): the
              actor performs only file I/O, no computation.  Zero
              CPU reservation avoids starving pipeline actors.
            - ``soft=False``: hard affinity ensures the actor reads
              from the correct node's local staging directory.
              Soft affinity could silently land on the wrong node
              and read the wrong files (or an empty directory).
            - One actor per node (not per file): keeps the actor
              count bounded by the cluster size, and each actor
              streams all files from its node sequentially.

        Args:
            nodes: List of node info dicts from
                ``ray_cluster_utils.get_live_nodes()``.  Each dict
                must contain ``"NodeID"`` and ``"NodeName"`` keys.

        Returns:
            List of ``(node_name, actor_handle, node_state)`` tuples,
            one per node.  The ``_NodeState.gen`` field is ``None``
            at this point -- it is populated later by
            ``_seed_generators()``.

        """
        result: list[tuple[str, ray.actor.ActorHandle, _NodeState]] = []  # type: ignore[type-arg]
        for node_info in nodes:
            node_id = node_info["NodeID"]
            node_name = node_info["NodeName"]
            actor = _NodeCollector.options(  # type: ignore[attr-defined]
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
            ).remote()
            state = _NodeState(node_name=node_name, gen=None)
            result.append((node_name, actor, state))
        return result

    @staticmethod
    def _seed_generators(
        deployed: list[tuple[str, ray.actor.ActorHandle, _NodeState]],  # type: ignore[type-arg]
        cfg: _StreamConfig,
    ) -> tuple[dict[ray.ObjectRef, _NodeState], list[str]]:  # type: ignore[type-arg]
        """Start streaming generators and seed the pending dict.

        Uses a **two-pass approach** to maximise actor placement
        lead time on large clusters:

        1. **Pass 1 -- fire all ``.remote()`` calls**: each call is
           non-blocking and triggers actor placement + generator
           startup concurrently across all nodes.
        2. **Pass 2 -- drain ``next(gen)``**: pull the first
           ``ObjectRef`` from each generator.  By this point, most
           actors have already been placed and started executing,
           so ``next(gen)`` returns quickly.

        On a 1000-node cluster, the single-pass approach (interleaved
        ``.remote()`` + ``next(gen)``) meant that node N's actor
        placement didn't start until node N-1's first chunk was
        ready.  The two-pass approach starts all actor placements
        concurrently, changing seeding time from
        ``sum(node_startup_times)`` to ``max(node_startup_times)``.

        ::

            Pass 1 -- fire .remote() (non-blocking)
            ========================================
            for each (node_name, actor, state) in deployed:
                gen = actor.stream_files.options(...).remote(...)
                state.gen = gen
                    |
                    v
                Ray scheduler receives placement request
                (actors begin placement concurrently)

            Pass 2 -- drain next(gen) (blocking per-node)
            =============================================
            for each (_, _, state) in deployed:
                first_ref = next(state.gen)   <-- actor already placed
                pending[first_ref] = state
                    |
                    v
                StopIteration?
                  YES -> ok_nodes (empty staging dir, skip)
                  NO  -> ref enters ray.wait() loop

        After seeding, each node with files has exactly 1 ref in
        ``pending`` (slot 1) and 1 pre-fetched ref buffered in the
        generator (slot 2).  The ``ray.wait()`` loop in ``collect()``
        takes over from here.

        Args:
            deployed: Output of ``_deploy_collectors()``.
            cfg: Streaming configuration with staging dir and
                chunk size.

        Returns:
            A tuple of:
            - ``pending``: dict mapping ``ObjectRef`` to
              ``_NodeState`` for nodes that have at least one chunk.
            - ``ok_nodes``: list of node names that completed
              immediately (empty staging dirs).

        """
        pending: dict[ray.ObjectRef, _NodeState] = {}  # type: ignore[type-arg]
        ok_nodes: list[str] = []

        # Pass 1: fire all .remote() calls (non-blocking).
        # Each call starts actor placement + generator startup
        # concurrently.  On a 1000-node cluster this gives the Ray
        # scheduler maximum lead time to place all actors before we
        # start blocking on their output.
        for _node_name, actor, state in deployed:
            state.gen = actor.stream_files.options(  # type: ignore[call-overload]
                num_returns="streaming",
                _generator_backpressure_num_objects=_BACKPRESSURE_OBJECTS,
            ).remote(cfg.staging_dir, cfg.chunk_bytes)

        # Pass 2: drain first ref from each generator.
        # By now, most actors have been placed and are yielding their
        # first chunk.  next(gen) returns quickly for placed actors.
        for node_name, _, state in deployed:
            try:
                first_ref = next(state.gen)  # type: ignore[call-overload]
                pending[first_ref] = state
            except StopIteration:
                ok_nodes.append(node_name)
                logger.debug(f"[RayFileTransport] Node {node_name}: empty staging dir, skipping")

        return pending, ok_nodes

    def _handle_timeout(
        self,
        acc: _CollectAccumulator,
        timeout_s: float,
        *,
        strict: bool,
    ) -> None:
        """Handle a ``ray.wait()`` timeout by marking all pending nodes as failed.

        Called when ``ray.wait()`` returns an empty ready list,
        meaning no chunk from any node arrived within ``timeout_s``.
        Iterates over all pending nodes and either raises (strict)
        or logs a warning and records the failure (non-strict).

        ::

            ray.wait() returned empty ready list
                  |
                  v
            for each pending _NodeState:
                  |
                  +-- strict=True?
                  |     YES -> raise RuntimeError (first node only,
                  |            remaining nodes never reached)
                  |     NO  -> logger.warning(...)
                  |            acc.failed_nodes.append(...)
                  |            _finalize_node(state)
                  |            (continue to next pending node)
                  v
            caller clears acc.pending and breaks the loop

        Design decision: in strict mode we raise on the **first**
        pending node rather than collecting all failures.  This is
        intentional -- the caller requested fail-fast semantics, so
        there is no value in continuing to iterate.  In non-strict
        mode, all pending nodes are recorded as failed so the
        ``CollectResult`` gives a complete picture.

        Edge case: if ``acc.pending`` is empty (should not happen
        since the caller checks ``if not ready``), this method is
        a no-op.

        Args:
            acc: Mutable collection accumulator with the pending dict
                and failed_nodes list.
            timeout_s: The timeout that was exceeded.
            strict: If ``True``, raises ``RuntimeError`` for the first
                pending node.

        Raises:
            RuntimeError: If ``strict=True``.

        """
        timeout_msg = f"Timeout ({timeout_s}s) waiting for chunks"
        for timed_out_state in acc.pending.values():
            if strict:
                msg = f"[RayFileTransport] {timeout_msg} from node {timed_out_state.node_name}"
                raise RuntimeError(msg)
            logger.warning(
                f"[RayFileTransport] {timeout_msg} from node {timed_out_state.node_name} (pid={os.getpid()})"
            )
            acc.failed_nodes.append((timed_out_state.node_name, timeout_msg))
            self._finalize_node(timed_out_state)

    def _consume_ready_ref(
        self,
        ref: ray.ObjectRef,  # type: ignore[type-arg]
        state: _NodeState,
        cfg: _StreamConfig,
        acc: _CollectAccumulator,
        *,
        strict: bool,
    ) -> None:
        """Process one ready ``ObjectRef`` from the ``ray.wait()`` loop.

        Fetches the chunk via ``ray.get()``, writes it to disk via
        ``_process_chunk()``, then advances the node's generator.
        On ``StopIteration`` the node is finalized as successful.
        On any other exception, the node is finalized as failed
        (or re-raised if ``strict=True``).

        The consume-advance cycle rotates the two backpressure
        slots, maintaining pipeline overlap between disk I/O on
        the driver and data transfer from the worker:

        ::

            ray.wait() returns ready ref (slot 1)
                  |
                  v
            ray.get(ref)
            +-> ~128 MB materialised in driver Python heap
            |   (slot 1 consumed, ObjectRef freed)
            |
            v
            _process_chunk(state, chunk, output_dir)
            +-> bytes written to disk, chunk becomes GC-eligible
            |   driver heap freed back to ~0 for this node
            |
            v
            next(state.gen)
            +-> pulls slot 2 into pending dict (new slot 1)
            |   worker unblocked: yields next chunk (new slot 2)
            |
            +-- StopIteration -> generator exhausted
            |     _finalize_node(state)
            |     acc.ok_nodes.append(node_name)
            |
            +-- Exception -> node failure
                  strict=True  -> re-raise
                  strict=False -> warn, acc.failed_nodes

        After each cycle, the invariant is restored: 1 ref in
        ``pending`` (slot 1) + 1 pre-fetched ref in the generator
        (slot 2), and peak driver memory = chunk_bytes per node.

        Args:
            ref: The ready ``ObjectRef`` returned by ``ray.wait()``.
            state: Per-node state for this ref.
            cfg: Streaming configuration with the output directory.
            acc: Mutable collection accumulator with pending dict
                and ok/failed node lists.
            strict: If ``True``, re-raises exceptions.

        """
        try:
            chunk: _FileChunk = ray.get(ref)
            self._process_chunk(state, chunk, cfg.output_dir)

            # Advance this node's generator to get the next chunk.
            # next(gen) returns an ObjectRef quickly (ref creation
            # is near-instant; blocking happens at ray.get()).
            try:
                next_ref = next(state.gen)  # type: ignore[call-overload]
                acc.pending[next_ref] = state
            except StopIteration:
                # Generator exhausted: this node is done.
                self._finalize_node(state)
                acc.ok_nodes.append(state.node_name)
                logger.info(
                    f"[RayFileTransport] Collected {state.files_collected} files from node {state.node_name}",
                )
        except Exception as exc:
            self._finalize_node(state)
            if strict:
                raise
            logger.warning(
                f"[RayFileTransport] Failed to collect from node {state.node_name}: {exc}",
                exc_info=True,
            )
            acc.failed_nodes.append((state.node_name, str(exc)))

    def collect(
        self,
        staging_dir: str,
        output_dir: pathlib.Path,
        *,
        timeout_s: float = 600.0,
        strict: bool = False,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
    ) -> CollectResult:
        """Gather staged files from all Ray nodes into *output_dir*.

        Deploys one ``_NodeCollector`` actor per alive node, starts
        all streaming generators simultaneously, then enters a
        ``ray.wait()`` loop that processes chunks from whichever
        node has data ready first.

        The ``ray.wait()`` loop implements the "Pipeline data
        processing" pattern from the Ray documentation: instead of
        waiting for all chunks from node A before starting node B,
        chunks are processed in readiness order.  This means total
        collection time equals the slowest node, not the sum of all
        nodes.

        After processing each chunk, ``next(gen)`` advances that
        node's streaming generator to obtain the next ``ObjectRef``.
        ``next(gen)`` returns quickly because ObjectRef creation is
        near-instant on the driver -- blocking only happens at
        ``ray.get()`` when the value is needed.  When a generator
        is exhausted (``StopIteration``), the node is finalized
        and recorded as successful.

        Timeout semantics: ``timeout_s`` is the cumulative stall
        guard.  The ``ray.wait()`` loop uses a short 100 ms timeout
        per call (adaptive batching).  If no chunk from ANY node
        arrives for ``timeout_s`` cumulative seconds, all remaining
        nodes are marked as failed and the loop exits.

        Error isolation: when ``strict=False``, an exception from
        one node removes only that node from the pending dict and
        records the failure.  Other nodes continue streaming.

        All actors are killed in a ``finally`` block regardless of
        success or failure.

        Concurrency model:
            Worker-side file reading is **truly parallel** -- each
            ``_NodeCollector`` is a separate Ray actor process on a
            separate machine, and all actors execute their
            ``stream_files()`` generators simultaneously.

            The driver (this method) processes chunks
            **single-threaded but interleaved** across all nodes.
            ``ray.wait(list(pending), num_returns=len(pending),
            timeout=0.1)`` returns all refs ready within 100 ms,
            so whichever node's chunks arrive first get processed
            first.  The driver never waits for node A to finish
            before touching node B.

            ``ray.get()`` safety: this method **never** calls
            ``ray.get()`` eagerly.  It always calls ``ray.wait()``
            first, which returns only refs whose data is already in
            the object store.  The subsequent ``ray.get()`` is
            essentially free (deserialise from local object store,
            no network wait).  This is the "Delayed ray.get()"
            pattern from Ray docs (Tip 1).

        ::

            Concurrency timeline (3 nodes):

            Node A actor:  [read][yield][read][yield][done]
            Node B actor:  [read][yield][yield][done]
            Node C actor:  [read][yield][read][yield][yield][done]
                                |    |     |      |     |
                                v    v     v      v     v
            Object store:  chunks arriving concurrently
                                |    |     |      |     |
            Driver thread:   [wait][get+write][wait][get+write]...
                             picks B  picks A  picks C  ...
                             (first   (next    (next
                              ready)   ready)   ready)

            Total wall time = max(node_A, node_B, node_C)
            NOT               node_A + node_B + node_C

        ::

            collect(staging_dir, output_dir, ...)
            |
            v
            Phase 1: Discovery
            +-- get_live_nodes()
            |     no nodes + strict  -> raise RuntimeError
            |     no nodes + lenient -> return empty CollectResult
            |
            v
            Phase 2: Deployment
            +-- _deploy_collectors(nodes)
            |     1 _NodeCollector actor per node (num_cpus=0)
            |     hard affinity via NodeAffinitySchedulingStrategy
            |
            v
            Phase 3: Seeding
            +-- _seed_generators(deployed, cfg)
            |     start generators with backpressure=2
            |     next(gen) -> first ObjectRef into pending dict
            |     empty nodes -> ok_nodes immediately
            |
            v
            Phase 4: Consumption (adaptive ray.wait loop)
            +-- stall_s = 0.0
            +-- while acc.pending:
            |     ray.wait(pending, num_returns=len(pending),
            |              timeout=_WAIT_BATCH_TIMEOUT_S)
            |     |
            |     +-- empty (stall):
            |     |     stall_s += _WAIT_BATCH_TIMEOUT_S
            |     |     stall_s >= timeout_s? -> _handle_timeout, break
            |     |     else -> continue
            |     |
            |     +-- ready refs (batch):
            |           stall_s = 0.0
            |           for ref in ready:
            |             _consume_ready_ref:
            |               ray.get -> _process_chunk -> next(gen)
            |               StopIteration -> ok_nodes
            |               Exception -> failed_nodes (or re-raise)
            |
            v
            Phase 5: Result
            +-- sum files_collected for ok_nodes
            +-- return CollectResult(total_files, ok, failed)
            |
            v
            finally (ALWAYS runs, even on exception):
            +-- 1. _finalize_node() all states
            |      close any open file handles (flush to disk)
            +-- 2. ray.kill() all actors
                   release Ray resources

        Cleanup guarantees:
            The ``finally`` block runs two passes in order:

            1. **Close local file handles first** -- any
               partially-written file is flushed to disk so it
               can be inspected for debugging.  This must happen
               before actor teardown because killing the actor
               could invalidate in-flight data.
            2. **Kill remote actors second** -- releases Ray
               resources (CPU, object store memory) regardless
               of how collection ended (success, timeout,
               exception, or KeyboardInterrupt).

        Preconditions:
            - Ray must be initialized (``ray.init()`` called).
            - The cluster must remain alive until this method
              returns.  Shutting down Ray mid-collection causes
              undefined behaviour.
            - ``staging_dir`` must be the same absolute path on
              every node.

        Args:
            staging_dir: Path to the staging directory on each node.
                Must be the same path on every node (e.g. set via
                an environment variable before the pipeline starts).
            output_dir: Local directory on the driver where collected
                files are written.  Created if it does not exist.
                Files from all nodes are merged into a flat structure
                (artifact names include hostname+PID so collisions
                are not expected).
            timeout_s: Cumulative stall guard timeout.  If no chunk
                from any node arrives for this many cumulative
                seconds (measured in ``_WAIT_BATCH_TIMEOUT_S``
                increments), remaining nodes are marked as failed.
                Defaults to 600 seconds.
            strict: Error handling policy.  When ``False`` (default),
                node failures are logged as warnings and recorded in
                the result.  When ``True``, any error (timeout,
                exception, no alive nodes) is raised to the caller.
            chunk_bytes: Maximum bytes per streamed chunk.  Files
                smaller than this are transferred in one piece.
                Larger files are split into sequential chunks of
                this size.  Defaults to 128 MB.

        Returns:
            A :class:`CollectResult` with per-node outcomes.

        Raises:
            RuntimeError: If ``strict=True`` and no alive nodes are
                found, or if a timeout occurs.
            Exception: If ``strict=True`` and any per-node error
                occurs (re-raises the original exception).

        """
        logger.debug(
            f"[RayFileTransport] collect(staging_dir={staging_dir!r}, "
            f"output_dir={output_dir}, timeout_s={timeout_s}, "
            f"strict={strict}, chunk_bytes={chunk_bytes}) (pid={os.getpid()})"
        )
        nodes = ray_cluster_utils.get_live_nodes(dump_info=False)
        if not nodes:
            msg = "[RayFileTransport] No alive nodes found"
            if strict:
                raise RuntimeError(msg)
            logger.warning(f"{msg}; skipping collection")
            return CollectResult(total_files=0, nodes_ok=(), nodes_failed=())

        with traced_span(
            "RayFileTransport.collect",
            attributes={
                "transport.staging_dir": staging_dir,
                "transport.output_dir": str(output_dir),
                "transport.nodes_total": len(nodes),
                "transport.timeout_s": timeout_s,
                "transport.chunk_bytes": chunk_bytes,
            },
        ) as span:
            return self._collect_inner(
                nodes=nodes,
                staging_dir=staging_dir,
                output_dir=output_dir,
                timeout_s=timeout_s,
                strict=strict,
                chunk_bytes=chunk_bytes,
                span=span,
            )

    def _collect_inner(  # noqa: PLR0913
        self,
        nodes: list[dict[str, str]],
        staging_dir: str,
        output_dir: pathlib.Path,
        timeout_s: float,
        *,
        strict: bool,
        chunk_bytes: int,
        span: TracedSpan,
    ) -> CollectResult:
        """Execute collection inside a traced span.

        Separated from :meth:`collect` to keep the public method
        concise while the span wraps the full operation.  Phase
        events are recorded on ``span`` at each transition so trace
        viewers can see deploy/seed/consume/cleanup timing without
        per-chunk overhead.

        Tracing: not instrumented separately -- called within the
        ``RayFileTransport.collect`` parent span.

        Args:
            nodes: Alive node info dicts from ``get_live_nodes()``.
            staging_dir: Staging directory path (same on every node).
            output_dir: Local destination directory on the driver.
            timeout_s: Cumulative stall guard timeout.  The adaptive
                ``ray.wait()`` loop uses ``_WAIT_BATCH_TIMEOUT_S``
                (100 ms) per call; ``timeout_s`` is the maximum
                cumulative time with zero results before the loop
                triggers ``_handle_timeout``.
            strict: Error handling policy.
            chunk_bytes: Maximum bytes per chunk.
            span: Active ``TracedSpan`` for attribute/event annotation.

        Returns:
            A :class:`CollectResult` with per-node outcomes.

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = _StreamConfig(
            staging_dir=staging_dir,
            output_dir=output_dir,
            timeout_s=timeout_s,
            chunk_bytes=chunk_bytes,
        )

        # Deploy actors and start generators.
        deployed = self._deploy_collectors(nodes)
        all_states = [state for _, _, state in deployed]
        span.add_event("collectors_deployed", attributes={"nodes_deployed": len(deployed)})

        try:
            seed_pending, seed_ok = self._seed_generators(deployed, cfg)
            span.add_event(
                "generators_seeded",
                attributes={
                    "nodes_seeded": len(seed_pending) + len(seed_ok),
                    "nodes_empty": len(seed_ok),
                },
            )
            acc = _CollectAccumulator(
                pending=seed_pending,
                ok_nodes=seed_ok,
            )

            # Main ray.wait() loop: process chunks from whichever
            # node has data ready first (pipeline processing pattern).
            #
            # Adaptive short-timeout batching: instead of
            # ``num_returns=1`` (one gRPC round-trip per chunk), we
            # request ALL pending refs with a short 100 ms timeout.
            # This drains however many refs are ready within the
            # window in a single call -- on a 1000-node cluster,
            # dozens of chunks may arrive within 100 ms.
            #
            # Two-tier timeout design:
            #   Inner: _WAIT_BATCH_TIMEOUT_S (100 ms) -- batching
            #   Outer: timeout_s (600 s) -- stall guard (cumulative
            #          time with zero results)
            stall_s = 0.0
            while acc.pending:
                ready, _ = ray.wait(
                    list(acc.pending),
                    num_returns=len(acc.pending),
                    timeout=_WAIT_BATCH_TIMEOUT_S,
                )

                if not ready:
                    stall_s += _WAIT_BATCH_TIMEOUT_S
                    if stall_s >= timeout_s:
                        self._handle_timeout(acc, timeout_s, strict=strict)
                        acc.pending.clear()
                        break
                    continue

                stall_s = 0.0
                for ref in ready:
                    state = acc.pending.pop(ref)
                    self._consume_ready_ref(ref, state, cfg, acc, strict=strict)

            # Sum files from all successfully completed nodes.
            ok_set = set(acc.ok_nodes)
            total_files = sum(s.files_collected for s in all_states if s.node_name in ok_set)

            span.set_attributes(
                {
                    "transport.total_files": total_files,
                    "transport.nodes_ok": len(acc.ok_nodes),
                    "transport.nodes_failed": len(acc.failed_nodes),
                }
            )
            span.add_event(
                "consume_loop_completed",
                attributes={
                    "total_files": total_files,
                    "nodes_ok": len(acc.ok_nodes),
                    "nodes_failed": len(acc.failed_nodes),
                },
            )

            result = CollectResult(
                total_files=total_files,
                nodes_ok=tuple(acc.ok_nodes),
                nodes_failed=tuple(acc.failed_nodes),
            )
            logger.info(
                f"[RayFileTransport] Collection complete: {result.total_files} files "
                f"from {len(result.nodes_ok)}/{len(nodes)} nodes"
                + (f" ({len(result.nodes_failed)} failed)" if result.nodes_failed else "")
                + f" (pid={os.getpid()})",
            )
            return result

        finally:
            # Graceful first: close any local file handles that are
            # still open (e.g. from an unhandled exception that
            # bypassed the normal flow).  This flushes partial data
            # to disk before we tear down remote actors.
            for state in all_states:
                self._finalize_node(state)

            # Forceful second: kill all remote actors to release
            # Ray resources.  Done after local handles are closed
            # so that actor teardown cannot interfere with in-progress
            # file writes on the driver.
            for _, actor, _ in deployed:
                with contextlib.suppress(Exception):
                    ray.kill(actor)
