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

"""Self-managing lazy data wrapper for split-field pipeline transport.

Provides ``LazyData[T]``, a combined type that manages data lifecycle
across pipeline stages.  Data can live inline (in Python heap) or in
Ray Plasma (as an ObjectRef).  Custom pickle ensures that stored data
flows through pass-through stages as a 48-byte ObjectRef instead of
the full payload.

::

    LazyData state machine:

    +-------+  .store()   +--------+  .resolve()  +--------------+
    | inline | ---------> | stored | -----------> | materialized |
    +-------+             +--------+              +--------------+
        |                   |    ^                     |    |
        |  .drop()          |    +---- .release() -----+    |
        v                   v                               v
    +-------+           +-------+                       +-------+
    | empty |           | empty |                       | empty |
    +-------+           +-------+                       +-------+

    Pickle behavior by state:

        inline:       value serialized via PickleBuffer (zero-copy for numpy)
        stored:       only the 48-byte ObjectRef is serialized
        materialized: only the 48-byte ObjectRef (mmap view excluded)
        empty:        both None

    Graceful fallback: if a producer forgets to call ``.store()``,
    ``__getstate__`` serializes the value inline via PickleBuffer.
    Data is never silently lost -- just not split-field optimized.

Current limitations -- split-field pattern disabled:

    The split-field optimization (``.store()`` / ``.release()``) is fully
    implemented but **disabled** across all pipeline stages.  All
    ``.store()`` and ``.release()`` calls are commented out with
    ``TODO(LazyData)`` markers.  Data always flows **inline**.

    Root cause -- ObjectRef ownership in Xenna execution modes:

    ``ray.put()`` inside an actor creates an ObjectRef **owned by that
    actor**.  When the actor process is terminated, the Plasma object
    backing the ObjectRef is garbage-collected, regardless of whether
    other processes still hold the reference.

    **Batch mode** (guaranteed failure):

    Xenna batch mode runs stages sequentially and kills each stage's
    actors (``pool.stop()``) before starting the next stage.  ObjectRefs
    created by Stage N actors are invalidated before Stage N+1 can
    resolve them.

    ::

        Batch mode -- sequential execution, actors killed between stages:

        time --->
        +--[Stage 0 actors]--+
        |  .store() creates  |     +--[Stage 1 actors]--+
        |  ObjectRef owned   |     | .resolve() calls   |
        |  by Stage 0 actor  |     | ray.get(ref)       |
        +--------------------+     +--------------------+
               |         ^              |
               |  pool   |              | OwnerDiedError!
               |  .stop()|              | ObjectRef owner is
               |  kills  |              | already dead.
               +---------+              v

        Timeline:
        |====[S0 run]====|==[S0 kill]==|====[S1 run]====|
                         ^              ^
                    pool.stop()    S1 tries ray.get()
                    kills S0       on S0-owned ref
                    actors         --> CRASH

    **Streaming mode** (race condition at tail):

    In streaming mode all stages run concurrently, so most ObjectRefs
    are resolved while the producer is still alive.  However, when a
    stage finishes its last batch, its actors are stopped while
    downstream stages may still have unresolved ObjectRefs in the
    inter-stage queue.

    ::

        Streaming mode -- concurrent execution, tail race window:

        time --->
        +--[Stage 0 actors]--+
        | .store() ok here   |
        | downstream resolves|     +--[Stage 1 actors]--+
        | while S0 is alive  |     | .resolve() works   |
        +--------------------+     | for most data...   |
              ^         |          |                    |
              |  pool   |          | last few refs in   |
              |  .stop()|          | queue[0] may fail! |
              +---------+          +--------------------+

        Timeline (safe for most data, risky at the tail):
        |=========[S0 run]=========|==[S0 kill]==|
                                     |
        |==[S1 idle]==|====[S1 run]===============|
                                     ^
                                S0 killed. S1 still has
                                items in queue[0] with
                                S0-owned ObjectRefs.

    Current workaround -- inline-only transport:

    With ``.store()`` disabled, data stays inline (value field) at every
    stage boundary.  The ``__getstate__`` fallback serializes via
    PEP 574 PickleBuffer, which is zero-copy for numpy arrays on the
    same node.  This is Phase 1 of the migration.

    ::

        What we have now (inline, no split-field):

        Stage 0          Stage 1          Stage 2
        [produce]   -->  [pass-through]   -->  [consume]
           |                  |                    |
        value=payload      value=payload       value=payload
        ref=None           ref=None            ref=None
           |                  |                    |
        serialize          serialize            serialize
        via PickleBuffer   via PickleBuffer    via PickleBuffer
        (zero-copy numpy)  (zero-copy numpy)   (zero-copy numpy)

        + numpy zero-copy via PickleBuffer (3-14x vs bytes)
        + nbytes metadata without materializing
        + unified API (resolve/store/release/drop)
        - full payload "present" at every stage boundary

        What split-field would give (when ownership is fixed):

        Stage 0          Stage 1          Stage 2
        [.store()]  -->  [pass-through]  -->  [.resolve()]
           |                  |                    |
        value=None         value=None          value=payload
        ref=48 bytes       ref=48 bytes        ref=None
           |                  |                    |
        serialize          serialize            ray.get()
        48-byte ref        48-byte ref         (zero-copy mmap)
        (trivial)          (trivial)

        + intermediate stages serialize only 48 bytes
        + only the consumer touches the actual data
        - requires producer actor to outlive downstream consumers

    Possible fixes (not yet implemented):

    1. **Driver-side ray.put()**: move the ``ray.put()`` call from the
       actor (inside ``.store()``) to the Xenna framework driver, which
       outlives all actors.  Requires framework-level changes.
    2. **Deferred actor teardown**: keep producer actors alive until all
       downstream stages have resolved their ObjectRefs.  Requires
       framework-level changes to the pool lifecycle.
    3. **Detached named objects**: use ``ray.put()`` with a detached
       owner so the object outlives the creating actor.  Adds complexity
       for naming and cleanup.
"""

from typing import Any

import attrs
import ray

from cosmos_curator.core.utils.data.bytes_transport import bytes_to_numpy


@attrs.define(eq=False)
class LazyData[T]:
    """Data that lives inline or in Plasma.  Self-manages transport lifecycle.

    ::

        State table:

        State          | value     | ref       | nbytes  | __getstate__ serializes
        ---------------|-----------|-----------|---------|---------------------------
        empty          | None      | None      | 0       | nbytes + both None
        inline         | <array>   | None      | N       | nbytes + value via PickleBuffer
        stored         | None      | ObjectRef | N       | nbytes + ref only (48 bytes)
        materialized   | <mmap>    | ObjectRef | N       | nbytes + ref only

    Method vocabulary:

        store()    -- push value to Plasma, clear local copy
        resolve()  -- pull from Plasma into local heap (cached)
        release()  -- drop local copy, keep Plasma ref for downstream
        drop()     -- drop everything (local copy + Plasma ref)

    Thread safety:
        NOT thread-safe.  Each actor owns its own LazyData instances.
        Concurrent access from multiple threads on the same instance
        could race on ``value`` / ``ref``.  Use external synchronization
        if shared access is needed (unlikely in the actor-per-stage model).

    Design decisions:
        - Caching after ``.resolve()``: same-node ``ray.get()`` returns
          an mmap view (very lightweight), so caching avoids repeated
          locking and ref-counting overhead without significant memory
          cost.
        - Explicit cleanup (``.release()`` / ``.drop()``): callers know
          when they are done with the data.  Automatic GC timing is
          non-deterministic.
        - Adaptive ``__getstate__``: if a producer forgets ``.store()``,
          the value is serialized inline via PickleBuffer (same as
          Phase 1 behavior).  This graceful fallback prevents data loss.
        - ``attrs.define`` default ``on_setattr`` includes
          ``attrs.setters.convert``, so converters run on every
          assignment automatically when used on data model fields.

    """

    value: T | None = None
    ref: "ray.ObjectRef[T] | None" = attrs.field(default=None, repr=False)
    nbytes: int = 0

    @classmethod
    def coerce(cls, val: "T | LazyData[T] | None") -> "LazyData[T]":
        """Convert a raw value, existing LazyData, or None into a LazyData instance.

        Intended as an ``attrs`` converter on data model fields so that
        assignment works ergonomically::

            video.encoded_data = bytes_to_numpy(raw)   # converter wraps automatically
            video.encoded_data = LazyData(...)    # pass-through (copies ref+value)
            video.encoded_data = None             # produces empty LazyData

        When *val* is already a ``LazyData``, a **new wrapper** is created
        that shares the same ``ObjectRef``.  This allows two consumers to
        resolve independently without aliasing each other's cached value.

        As a runtime convenience, raw ``bytes`` input is auto-converted to
        ``npt.NDArray[np.uint8]`` via ``bytes_to_numpy()`` so that callers do not
        need to remember the conversion step.  The resulting ``LazyData``
        holds a numpy array regardless of whether the input was bytes or
        array.

        Args:
            val: Raw data, an existing LazyData to shallow-copy, or None.

        Returns:
            A new LazyData wrapping the input.

        """
        if isinstance(val, LazyData):
            return cls(ref=val.ref, value=val.value, nbytes=val.nbytes)
        if isinstance(val, bytes):
            arr = bytes_to_numpy(val)
            return cls(value=arr, nbytes=arr.nbytes)  # type: ignore[arg-type]  # bytes -> LazyData[NDArray[uint8]] special case
        size = getattr(val, "nbytes", 0) if val is not None else 0
        return cls(value=val, nbytes=size)

    def resolve(self) -> T | None:
        """Fetch data from Plasma into local heap.  Caches the result.

        If value is already present (inline or previously resolved),
        returns it immediately.  If only a ref is available, calls
        ``ray.get()`` to materialize the data.

        Same-node: returns zero-copy mmap view (~100 bytes Python
        overhead).  Cross-node: copies data to local Plasma first
        (one copy), then mmap.

        Returns:
            The resolved data, or None if neither value nor ref is set.

        """
        if self.value is not None:
            return self.value
        if self.ref is not None:
            self.value = ray.get(self.ref)
            if self.nbytes == 0 and hasattr(self.value, "nbytes"):
                self.nbytes = self.value.nbytes
            return self.value
        return None

    def store(self) -> None:
        """Push value to Plasma.  Clears local copy, keeps ObjectRef.

        Calls ``ray.put()`` synchronously.  After this call, the value
        field is None and the ref field holds the 48-byte ObjectRef.
        At the next stage boundary, only the ref is serialized.

        No-op if value is already None (already stored or empty).
        If called in materialized state (both value and ref set),
        clears the local copy without a duplicate ``ray.put()`` --
        the existing ref already points to the correct Plasma object.

        .. warning:: ObjectRef ownership -- currently disabled

            ``ray.put()`` inside an actor creates an ObjectRef **owned
            by that actor**.  If the actor is killed (e.g. Xenna batch
            mode calls ``pool.stop()`` between stages), the Plasma
            object is garbage-collected and downstream consumers get
            ``ray.exceptions.OwnerDiedError``.

            ::

                Actor A: ray.put(data) -> ObjectRef(owner=A)
                Actor A killed (pool.stop())
                Actor B: ray.get(ObjectRef) -> OwnerDiedError!

            In **batch** mode this is a guaranteed failure.
            In **streaming** mode this is a race condition at the tail
            end of stage processing.

            All ``.store()`` calls are currently commented out across
            the pipeline.  See the module docstring for full details,
            diagrams, and possible fixes.

        ::

            inline:        value = <array>   ref = None      -> ray.put, clear value
            materialized:  value = <mmap>    ref = ObjectRef  -> clear value only
            stored/empty:  value = None                       -> no-op

        """
        if self.value is None:
            return
        if self.nbytes == 0 and hasattr(self.value, "nbytes"):
            self.nbytes = self.value.nbytes
        if self.ref is not None:
            self.value = None
            return
        self.ref = ray.put(self.value)
        self.value = None

    def release(self) -> None:
        """Drop local copy, keep Plasma ref for downstream stages.

        Frees the Python-side value (mmap view or inline array) while
        preserving the ObjectRef.  Downstream stages can still call
        ``.resolve()`` to re-fetch from Plasma.

        ::

            Before:  value = <mmap>    ref = ObjectRef
            After:   value = None      ref = ObjectRef

        """
        self.value = None

    def drop(self) -> None:
        """Drop both local copy and Plasma ref, reset size metadata.

        After this call, ``.resolve()`` returns None and ``nbytes == 0``.
        The Plasma object is freed when all remaining ObjectRef references
        are garbage collected (reference counting).

        ::

            Before:  value = <mmap>    ref = ObjectRef   nbytes = N
            After:   value = None      ref = None        nbytes = 0

        """
        self.value = None
        self.ref = None
        self.nbytes = 0

    def __bool__(self) -> bool:
        """Return True if data is available (inline or via ref)."""
        return self.value is not None or self.ref is not None

    def __getstate__(self) -> dict[str, Any]:
        """Serialize adaptively: ref-only when stored, value when inline.

        When a ref is present (stored or materialized state), only the
        48-byte ObjectRef is serialized -- the mmap view or cached value
        is excluded.  This prevents re-serialization of large payloads
        at every stage boundary.

        When no ref is present (inline state), the value is serialized
        directly.  For numpy arrays this uses PEP 574 PickleBuffer
        (zero-copy).  This is the graceful fallback when ``.store()``
        was not called.
        """
        if self.ref is not None:
            return {"ref": self.ref, "value": None, "nbytes": self.nbytes}
        return {"ref": None, "value": self.value, "nbytes": self.nbytes}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore from pickle."""
        object.__setattr__(self, "ref", state.get("ref"))
        object.__setattr__(self, "value", state.get("value"))
        object.__setattr__(self, "nbytes", state.get("nbytes", 0))
