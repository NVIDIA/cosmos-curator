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

"""Batch resolution utilities for LazyData collections.

Provides three strategies for materializing collections of LazyData
instances, each optimized for a different access pattern and memory
profile.

::

    Strategy comparison (batch of N items with large payloads):

    +------------------+---------------------------+-----------------------+
    | Strategy         | Peak Python Heap          | When to Use           |
    +------------------+---------------------------+-----------------------+
    | prefetch()       | 0 (non-blocking hint)     | Start of process_data |
    | resolve_as_ready | 1 item at a time          | DEFAULT for consuming |
    | batch_resolve    | N items simultaneously    | Global batch ops ONLY |
    +------------------+---------------------------+-----------------------+

    resolve_as_ready (generator) pipelining:

        Sequential:  [fetch A 5ms]--[proc A]--[fetch B 3ms]--[proc B]
                     total = 5+P+3+P = 8+2P ms

        Pipelined:   [fetch A 5ms]--[proc A]--[proc B]
                     [fetch B 3ms]-+         |
                     total = max(5, 3+P) + P ms  (fetch overlaps processing)

Design decisions:
    - ``resolve_as_ready`` is a generator (not list): yields one result
      at a time so the caller can process + release before the next
      item is materialized.  Peak heap = 1 item instead of N items.
    - ``prefetch`` uses ``fetch_local=True``: tells Ray to start pulling
      data to local Plasma without blocking.  Zero Python heap cost.
    - ``batch_resolve`` uses ``ray.get([list])``: one RPC instead of N,
      but materializes everything at once.  Reserved for stages that
      genuinely need all data simultaneously (rare).

Tracing:
    ``batch_resolve`` is instrumented with ``traced_span`` (batch-level
    I/O).  ``prefetch`` and ``resolve_as_ready`` are NOT traced:

    - ``prefetch``: non-blocking hint, trivial overhead
    - ``resolve_as_ready``: generator lifecycle makes spanning awkward;
      the caller's ``process_data()`` parent span covers timing
"""

from collections.abc import Iterator, Sequence
from typing import Any

import ray

from cosmos_curator.core.utils.data.lazy_data import LazyData


def prefetch(refs: Sequence[LazyData[Any]]) -> None:
    """Non-blocking hint to pull data to local Plasma.

    Call at the start of ``process_data()`` for refs you will need soon.
    Does NOT block, does NOT allocate Python heap memory -- just tells
    Ray to start fetching objects to the local node's Plasma store.

    ::

        Timeline with prefetch:

        prefetch([ref_A, ref_B, ref_C])     <-- non-blocking, returns instantly
        |
        +-- Ray starts background fetches:
        |     [fetch A from remote] (async)
        |     [fetch B from remote] (async)
        |     [fetch C from remote] (async)
        |
        v
        for clip, data in resolve_as_ready(...):
            # data may already be local -- no wait!

    Edge cases:
        - Empty sequence: no-op.
        - Already-resolved items (value is not None): skipped.
        - Items without a ref: skipped.
        - All refs local: ``ray.wait`` returns immediately, no network
          I/O.

    Args:
        refs: LazyData instances whose ObjectRefs should be pre-fetched.

    """
    raw = [r.ref for r in refs if r.ref is not None and r.value is None]
    if raw:
        ray.wait(raw, num_returns=len(raw), timeout=0, fetch_local=True)


def resolve_as_ready[K, T](
    items: Sequence[tuple[K, LazyData[T]]],
) -> Iterator[tuple[K, T | None]]:
    """Yield results in completion order via ``ray.wait`` (generator).

    Memory-optimal resolution strategy: materializes one item at a time,
    yielding control back to the caller between items.  The caller
    should process and release each item before the next is
    materialized.

    ::

        Memory profile (batch of N items with large payloads):

        iteration 1: ray.wait -> item_B ready
                     ray.get(ref_B) -> mmap view
                     yield (B, data_B)
                     caller: process(data_B), release()  <-- freed
        iteration 2: ray.wait -> item_A ready
                     ray.get(ref_A) -> mmap view
                     yield (A, data_A)
                     caller: process(data_A), release()  <-- freed
        ...
        Peak heap: 1 item instead of N items

    Items with already-available values are yielded first (no
    ``ray.wait``).  Empty items (both value and ref are None) yield
    ``(key, None)`` immediately so callers always get a 1:1 mapping
    with the input sequence.  Remaining items are yielded in
    completion order as ``ray.wait`` returns them.

    Duplicate ObjectRef handling:
        Multiple ``(key, LazyData)`` pairs may share the same
        ``ObjectRef`` (e.g., two ``LazyData`` wrappers created via
        ``coerce()`` from the same source).  The ref map groups all
        entries by ``ObjectRef`` so that a single ``ray.wait`` return
        yields all items sharing that ref.  Each ``LazyData`` instance
        resolves independently (``resolve()`` caches per-instance).

    Design decisions:
        - ``ray.wait(num_returns=1)``: small cluster optimization.  For
          100+ node fan-in, use adaptive batching
          (``num_returns=len(pending), timeout=0.1``).  In the consuming
          stage context, we're resolving intra-batch refs (typically
          4-64 items), so ``num_returns=1`` is optimal.
        - Generator (not list): callers control memory by processing
          and releasing each item before requesting the next.  If a
          list were returned, all mmap views would be alive
          simultaneously.

    Args:
        items: Sequence of ``(key, LazyData)`` pairs.  Keys are passed
            through for caller identification (e.g., clip index, task
            reference).

    Yields:
        ``(key, resolved_value)`` tuples in completion order.  Empty
        LazyData items yield ``(key, None)`` so callers can handle
        missing data without positional misalignment.

    """
    ref_map: dict[ray.ObjectRef[T], list[tuple[K, LazyData[T]]]] = {}

    for key, lazy in items:
        if lazy.value is not None:
            yield key, lazy.value
        elif lazy.ref is not None:
            ref_map.setdefault(lazy.ref, []).append((key, lazy))
        else:
            yield key, None

    pending = list(ref_map.keys())
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        for raw_ref in ready:
            for key, lazy in ref_map[raw_ref]:
                resolved = lazy.resolve()
                yield key, resolved


def batch_resolve[T](refs: Sequence[LazyData[T]]) -> list[T | None]:
    """Resolve all refs in one ``ray.get()`` call.

    WARNING: Materializes ALL results into Python heap simultaneously.
    For large payloads this can OOM on GPU nodes where RAM is shared
    with model weights, CUDA context, and Ray overhead.  Use ONLY when
    the algorithm genuinely requires all data at once (e.g., global
    sort, cross-item deduplication) AND the batch is small enough to
    fit in memory.

    For sequential per-item processing, use ``resolve_as_ready()``
    instead.

    Mutates each input ``LazyData`` instance: sets ``.value`` and
    ``.nbytes`` so that subsequent ``.resolve()`` calls return the
    cached value without another ``ray.get()``.  This makes
    ``batch_resolve`` safe to call on the main thread before
    dispatching items to a ``ThreadPoolExecutor`` -- worker threads
    will only read already-materialized ``.value``.

    ::

        batch_resolve([ref_A, ref_B, ref_C]):
            unresolved = [ref_A, ref_C]  (ref_B already resolved)
            ray.get([ref_A, ref_C])      <-- ONE RPC, blocks until ALL ready
            ref_A.value = data_A         <-- mutates LazyData in-place
            ref_C.value = data_C
            return [data_A, data_B, data_C]
            Peak heap: sum(item_sizes) = ALL items alive simultaneously

    Design decisions:
        - Single ``ray.get([list])`` instead of N sequential
          ``ray.get()`` calls: avoids N gRPC round-trips and allows Ray
          to pipeline fetches.
        - Suitable for small bounded batches (< 16 items) or when all
          data is needed simultaneously for a global operation.

    Args:
        refs: LazyData instances to resolve.  Already-resolved items
            are skipped in the ``ray.get`` call but included in the
            result.

    Returns:
        List of resolved values in the same order as input refs.

    """
    to_fetch: list[tuple[int, ray.ObjectRef[T]]] = []
    results: list[T | None] = [r.value for r in refs]
    for i, r in enumerate(refs):
        if r.value is None and r.ref is not None:
            to_fetch.append((i, r.ref))
    if to_fetch:
        indices, raw_refs = zip(*to_fetch, strict=True)
        fetched = ray.get(list(raw_refs))
        for idx, val in zip(indices, fetched, strict=True):
            results[idx] = val
            refs[idx].value = val
            if refs[idx].nbytes == 0 and hasattr(val, "nbytes"):
                refs[idx].nbytes = val.nbytes
    return results
