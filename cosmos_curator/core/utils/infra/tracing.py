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

"""Generic OpenTelemetry span utilities for Cosmos Curator.

This module provides the **public span API** used by application code
to create and annotate distributed traces.  It is independent of Ray
and can be imported safely from any environment.

The Ray-specific tracing hook (worker setup, file export, library
auto-instrumentation) lives in :mod:`tracing_hook`.

Setup vs Usage (two-layer design)
---------------------------------
The tracing system is split into two layers with different roles::

    +-------------------------------------------------+
    |  Stage / application code                       |  <-- YOU ARE HERE
    |  import traced_span, traced, TracedSpan          |
    |  (safe to call always, no-ops when not enabled) |
    +-------------------------------------------------+
                     |
                     | uses
                     v
    +-------------------------------------------------+
    |  tracing.py  (THIS MODULE)                      |
    |  Public span API -- no Ray, no SDK setup        |
    |  traced_span, @traced, TracedSpan.current()     |
    +-------------------------------------------------+
                     |
                     | creates spans via
                     v
    +-------------------------------------------------+
    |  opentelemetry.trace  (OTel API layer)          |
    |  get_tracer() -> start_as_current_span()        |
    |  No-op NonRecordingSpan when no provider set    |
    +-------------------------------------------------+
                     ^
                     | configures TracerProvider
                     |
    +-------------------------------------------------+
    |  tracing_hook.py  (SETUP LAYER)                 |
    |  enable_tracing() -> setup_tracing()            |
    |  Called by profiling_scope() on the driver       |
    |  Triggered by --profile-tracing CLI flag        |
    +-------------------------------------------------+

**You do NOT need to call ``enable_tracing()``.**  Each pipeline's
``main()`` wraps its execution in ``profiling_scope(args)`` --
the top-level context manager that sets up all profiling and
tracing infrastructure::

    # run_pipeline.py (pipeline entry point)
    def main() -> None:
        args = parse_args()
        with profiling_scope(args):   # <-- must be the outermost wrapper
            args.func(args)           #     everything runs inside this

``profiling_scope()`` reads CLI flags (``--profile-tracing``,
``--profile-cpu``, etc.) and configures everything automatically.
Stage code just imports and uses the public API -- no setup calls
needed:

- **``--profile-tracing`` passed**: spans are recorded and exported
  to files and/or an OTLP collector.
- **``--profile-tracing`` NOT passed**: all span API calls become
  zero-cost no-ops (OTel creates ``NonRecordingSpan`` objects).
  No overhead, no errors.

No ``if tracing_enabled`` guards are needed in your code.

Choosing the right API
----------------------
Three forms are available.  Use the decision tree below::

    "I need tracing in my code"
              |
              v
    Is the span a WHOLE function/method body?
              |                    |
              v (yes)              v (no, or partial)
        @traced(name)        traced_span(name)
        (decorator)          (context manager)
              |                    |
              v                    v
    Do I also need to annotate a PARENT span
    created by the framework (e.g. _ProfiledStage)?
              |
              v (yes)
        TracedSpan.current()
        (wraps existing span, NO new span created)

Decision rules:

1. **``@traced(name)``** -- use when the **entire** function body
   is one logical span.  Attributes are fixed at decoration time.
   Cleanest syntax, no indentation.

2. **``traced_span(name)``** -- use when you need a span around a
   **subset** of a function, or when you need to set attributes
   **dynamically** based on values computed during execution.
   Returns a ``TracedSpan`` via ``with ... as span:``.

3. **``TracedSpan.current()``** -- use when you want to annotate a
   span that **already exists** higher up the call stack (e.g. the
   lifecycle span from ``_ProfiledStage``).  Does NOT create a new
   span -- just wraps the active one.

All three are **safe to call always**, even when tracing is not
enabled.  No ``if`` guards are needed.

Public API
----------
:class:`TracedSpan`
    Lightweight wrapper around an OTel ``Span`` with Null Object
    semantics -- all methods are no-ops when the underlying span is
    ``None``.

:func:`traced_span`
    Context manager creating a child span under the current active
    span.  Yields a :class:`TracedSpan`.

:func:`traced`
    Decorator wrapping a function body in an OTel span.

``TracedSpan.current()``
    Class method returning a :class:`TracedSpan` wrapping the
    current active span (for annotating spans created higher up).

``StatusCode``
    Re-exported from ``opentelemetry.trace`` for convenience
    (``UNSET``, ``OK``, ``ERROR``).

Helper functions
----------------
:func:`short_hostname`
    Cached short hostname (everything before the first dot).

:func:`process_tag`
    ``<hostname>_<pid>`` tag for unique artifact file names.

Span hierarchy example
----------------------
When tracing is enabled (``--profile-tracing``), three levels of
spans are produced automatically.  Application code can add custom
child spans at any level::

    [Ray: actor_method process_data]              <-- automatic (Ray hook)
      +-- [cosmos_curator: MyStage.process_data]      <-- _ProfiledStage
           | attrs: stage.name, stage.num_input_tasks,
           |        profiling.artifact_id
           +-- [cosmos_curator: MyStage.download]     <-- your traced_span
           |   attrs: download.url, download.bytes
           +-- [cosmos_curator: MyStage.sample]       <-- StageTimer
           |   attrs: stage.name, stage.num_samples
           +-- [cosmos_curator: MyStage.sample]
           ...

Quick-start example
-------------------
Spans should be created at the **batch** or **phase** level, not
per-item.  At scale (100 M+ items), per-item spans generate
excessive trace data.  Record per-item details as aggregated
attributes on the batch span instead::

    from cosmos_curator.core.utils.infra.tracing import (
        TracedSpan, traced, traced_span, StatusCode,
    )

    class MyStage(CuratorStage):

        @traced("MyStage.setup_model")           # <-- whole method
        def _load_model(self):
            self._model = load_weights()

        def process_data(self, tasks):
            # Batch-level span -- one span per process_data call,
            # NOT one per item:
            with traced_span("MyStage.infer",
                             attributes={"stage.input_count": len(tasks)}) as span:
                results = [self._model(t.data) for t in tasks]
                span.set_attribute("stage.output_count", len(results))

            # Annotate the parent span (no new span created):
            TracedSpan.current().set_attribute("stage.total", len(tasks))
            return tasks
"""

import asyncio
import contextlib
import functools
import os
import platform
import time
from collections.abc import Callable, Generator
from typing import Any

import wrapt  # type: ignore[import-untyped]
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

# Tracer name used for all application-level spans.
_TRACER_NAME = "cosmos_curator"

# Type alias compatible with opentelemetry.util.types.AttributeValue
# (scalar subset -- sequences are rarely needed in our spans).
SpanAttributeValue = str | int | float | bool

# Standard OpenTelemetry environment variables for OTLP export.
# Kept here (the public API module) so both tracing_hook.py and
# application code (e.g. local_vllm_serve_stage.py) share a single
# source of truth for endpoint resolution.
_ENV_OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"
_ENV_OTLP_TRACES_ENDPOINT = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"

# Re-export StatusCode so callers don't need an extra import.
__all__ = [
    "SpanAttributeValue",
    "StatusCode",
    "TracedSpan",
    "artifact_id",
    "get_otlp_endpoint",
    "process_tag",
    "short_hostname",
    "trace_root_anchor",
    "traced",
    "traced_span",
]


def artifact_id(stage_name: str, call_id: str) -> str:
    """Build the artifact base name shared by all profiling backends.

    Format: ``{stage}_{call_id}_{hostname}_{pid}``

    Example: ``MyStage_setup_on_node_1_epaniot-pc_6135``

    This is the single source of truth for artifact file naming.
    Individual backends append their own extension / subdirectory.

    Lives in ``tracing.py`` (lightweight, no heavy deps) so that both
    ``profiling.py`` and ``tracing_hook.py`` can share the same naming
    convention without pulling in ``memray`` / ``pyinstrument`` / ``torch``.

    Args:
        stage_name: Stage class name (e.g. ``"MyStage"``).
        call_id: Label with counter (e.g. ``"process_data_3"``).

    Returns:
        A unique base name for profiling artifacts.

    """
    return f"{stage_name}_{call_id}_{process_tag()}"


@functools.cache
def short_hostname() -> str:
    """Return the short hostname (everything before the first dot).

    Cached because the hostname never changes within a process
    (or across forks).
    """
    return platform.node().split(".")[0]


def process_tag() -> str:
    """Return a ``<hostname>_<pid>`` tag for unique artifact file names.

    In multi-node clusters (Ray, Slurm) several actors of the same
    stage type may run on different nodes.  Including the short
    hostname and the OS PID in every artifact avoids file-name
    collisions and makes it immediately clear which process on which
    node produced the file.

    Note: intentionally **not** cached because ``os.getpid()``
    changes after ``os.fork()`` (Ray actors, multiprocessing workers).
    The hostname portion is cached separately via ``short_hostname()``.
    """
    return f"{short_hostname()}_{os.getpid()}"


def get_otlp_endpoint() -> str:
    """Resolve the OTLP HTTP endpoint from standard OTel environment variables.

    Resolution order (first non-empty value wins):

    1. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` (trace-specific override)
    2. ``OTEL_EXPORTER_OTLP_ENDPOINT`` (general OTLP endpoint)
    3. ``""`` (empty -- OTLP export disabled)

    Returns an empty string when no endpoint is explicitly configured.
    This makes OTLP export **opt-in**: the file-based exporter is always
    active and captures all spans locally; remote OTLP delivery only
    activates when an operator explicitly sets one of the standard OTel
    endpoint env vars or passes ``--profile-tracing-otlp-endpoint``.

    This function is the single source of truth for OTLP endpoint
    resolution.  Used by:

    - ``TracingConfig.from_env()`` in ``tracing_hook.py`` when
      configuring the per-process ``OTLPSpanExporter``.
    - ``build_vllm_serve_config()`` in ``local_vllm_serve_stage.py`` to
      pass ``--otlp-traces-endpoint`` to the ``vllm serve`` subprocess
      so it exports spans to the same collector as the pipeline.

    Returns:
        The resolved OTLP HTTP endpoint URL, or ``""`` if none is set.

    """
    return os.environ.get(_ENV_OTLP_TRACES_ENDPOINT) or os.environ.get(_ENV_OTLP_ENDPOINT) or ""


class TracedSpan:
    """Lightweight wrapper around an OpenTelemetry span.

    Uses the Null Object pattern: all methods silently return
    without error when the underlying ``_span`` is ``None``.

    There are two ways to obtain a ``TracedSpan``:

    1. **Create a new child span** via the :func:`traced_span` context
       manager (preferred for scoped work)::

           with traced_span("MyStage.download", attributes={"url": url}) as span:
               span.set_attribute("bytes", len(data))
               download(url)

    2. **Wrap the current active span** via :meth:`current` (useful
       when you want to annotate a span created higher up the call
       stack, e.g. by ``_ProfiledStage``)::

           TracedSpan.current().set_attributes({
               "stage.rss_after_mb": rss_after,
           })
           TracedSpan.current().add_event("batch_start")

    Error handling example (record caught exceptions)::

        with traced_span("MyStage.download") as span:
            try:
                download(url)
            except TimeoutError as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, "Download timed out")
                # handle gracefully...

    Note: unhandled exceptions that propagate out of a
    :func:`traced_span` block are automatically recorded by OTel
    (``record_exception=True`` is the default for
    ``start_as_current_span``).  The manual methods are for
    *caught* exceptions that should still be visible in traces.

    """

    __slots__ = ("_span",)

    def __init__(self, span: Span | None = None) -> None:
        """Wrap an OTel span (or ``None`` for a no-op instance)."""
        self._span = span

    @classmethod
    def current(cls) -> "TracedSpan":
        """Return a ``TracedSpan`` wrapping the current active OTel span.

        Returns:
            A ``TracedSpan`` wrapping the current span.

        """
        return cls(trace.get_current_span())

    def set_attribute(self, key: str, value: SpanAttributeValue) -> None:
        """Set a single attribute on the span.

        No-op when the underlying span is ``None``.

        Args:
            key: Attribute name (e.g. ``"stage.rss_after_mb"``).
            value: Attribute value (str, int, float, or bool).

        """
        if self._span is not None:
            self._span.set_attribute(key, value)

    def set_attributes(self, attributes: dict[str, SpanAttributeValue]) -> None:
        """Set multiple attributes on the span in bulk.

        No-op when the underlying span is ``None``.

        Usage::

            span.set_attributes({
                "stage.rss_after_mb": rss_after,
                "stage.process_time_s": duration,
            })

        Args:
            attributes: Key-value pairs to attach.

        """
        if self._span is None:
            return
        for key, value in attributes.items():
            self._span.set_attribute(key, value)

    def add_event(
        self,
        name: str,
        *,
        attributes: dict[str, SpanAttributeValue] | None = None,
        timestamp: int | None = None,
    ) -> None:
        """Add a point-in-time event to the span.

        Events are timestamped annotations -- useful for marking
        milestones (e.g. ``"batch_start"``, ``"model_loaded"``)
        without creating a full child span.

        No-op when the underlying span is ``None``.

        Args:
            name: Event name.
            attributes: Optional key-value pairs attached to the event.
            timestamp: Event timestamp in nanoseconds since the Unix
                epoch.  Defaults to ``time.time_ns()`` at call time.

        """
        if self._span is not None:
            self._span.add_event(name, attributes=attributes or {}, timestamp=timestamp or time.time_ns())

    def record_exception(
        self,
        exception: BaseException,
        *,
        attributes: dict[str, SpanAttributeValue] | None = None,
    ) -> None:
        """Record a caught exception on the span.

        Creates an ``exception`` event with the exception type, message,
        and traceback as attributes.  This is for *caught* exceptions
        that you handle gracefully but still want visible in traces.

        Unhandled exceptions that propagate out of a :func:`traced_span`
        block are recorded automatically by OTel -- you do not need to
        call this for those.

        Typically paired with :meth:`set_status`::

            with traced_span("MyStage.download") as span:
                try:
                    download(url)
                except TimeoutError as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR, "Download timed out")

        No-op when the underlying span is ``None``.

        Args:
            exception: The exception instance to record.
            attributes: Optional extra attributes attached to the
                exception event.

        """
        if self._span is not None:
            self._span.record_exception(exception, attributes=attributes or {})

    def set_status(
        self,
        code: StatusCode,
        description: str | None = None,
    ) -> None:
        """Set the status of the span.

        Use ``StatusCode.ERROR`` to mark a span as failed and
        ``StatusCode.OK`` to explicitly mark success.  The default
        status is ``UNSET`` (completed without error).

        No-op when the underlying span is ``None``.

        Args:
            code: ``StatusCode.OK``, ``StatusCode.ERROR``, or
                ``StatusCode.UNSET``.
            description: Optional human-readable status description.
                Only used when *code* is ``ERROR``.

        """
        if self._span is not None:
            self._span.set_status(code, description)


@contextlib.contextmanager
def traced_span(
    name: str,
    *,
    attributes: dict[str, SpanAttributeValue] | None = None,
) -> Generator[TracedSpan, None, None]:
    """Create an OpenTelemetry child span as a context manager.

    **When to use**: for a span around a **subset** of a function, or
    when attributes need to be set **dynamically** based on values
    computed during execution.  For whole-function spans with static
    attributes, prefer :func:`traced` (decorator).

    Creates a child span under the current active span.  When no
    ``TracerProvider`` has been configured (tracing not enabled),
    OTel's default proxy provider creates a lightweight
    ``NonRecordingSpan`` -- effectively a no-op.

    The span is automatically closed when the context manager exits.
    Duration is recorded by OTel (start-to-end of the ``with`` block).
    Unhandled exceptions propagating out of the ``with`` block are
    automatically recorded on the span by OTel.

    In-process 3rd-party OTel spans become children of the active span
    via the standard OTel context (no W3C env injection).  Ray workers
    get the driver's trace via ``propagate_trace_context()`` and
    ``COSMOS_CURATOR_TRACEPARENT`` (see ``tracing_hook``).

    Trace volume is controlled by OTel's ``ParentBasedTraceIdRatio``
    sampler (configured via ``--profile-tracing-sampling``), which
    makes the sampling decision at the root span and propagates it
    to all descendants -- ensuring correct parent-child semantics.

    ::

        traced_span("MyStage.download", attributes={...})
        |
        +-- tracer = get_tracer("cosmos_curator")
        +-- tracer.start_as_current_span(name, attributes)
        |     Creates child span under the current active span
        |     (or a root span if no parent context exists)
        |
        +-- yield TracedSpan(span)
        |     |
        |     v  [caller's code runs inside the with block]
        |        span.set_attribute(...)   <-- dynamic attributes
        |        span.add_event(...)       <-- milestones
        |        span.record_exception()   <-- caught errors
        |
        +-- span.end()  (automatic on with-block exit)
              Duration = start-to-end of with block

    Usage::

        from cosmos_curator.core.utils.infra.tracing import traced_span

        with traced_span("MyStage.download", attributes={"url": url}) as span:
            data = download(url)
            span.set_attribute("bytes", len(data))

    Args:
        name: Span name -- typically ``"StageName.operation"``.
        attributes: Optional key-value pairs attached to the span
            at creation time.  Additional attributes can be set
            dynamically via ``span.set_attribute()`` inside the
            ``with`` block.  Values must be str, int, float, or bool.

    Yields:
        A :class:`TracedSpan` wrapping the OTel span.

    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name, attributes=attributes or {}) as span:
        yield TracedSpan(span)


@contextlib.contextmanager
def trace_root_anchor(
    name: str,
    *,
    attributes: dict[str, SpanAttributeValue] | None = None,
) -> Generator[TracedSpan, None, None]:
    """Create a short-lived root span whose context outlives the span itself.

    **Internal API** -- called by ``profiling_scope()`` in the pipeline
    entry point.  Application / stage code should NOT call this
    directly.  Use :func:`traced_span` or :func:`traced` instead.

    The anchor is created with **no parent** (explicit empty context),
    making it the true root of the trace.  It is ended immediately so
    backends (Jaeger, Tempo, etc.) receive it **before** any child
    spans arrive -- eliminating "invalid parent span ID" warnings
    that occur when a long-lived root arrives last.

    Even though the span is ended, its context (trace_id + span_id)
    stays attached as the current OTel context for the duration of
    the ``with`` block.  All spans created inside this block inherit
    the anchor's trace_id and use its span_id as their parent.

    Preconditions:
        - ``enable_tracing()`` MUST have been called before this
          function, so a real ``TracerProvider`` is active.
        - ``propagate_trace_context()`` should be called inside
          the ``with`` block to share the anchor's trace context
          with worker processes via environment variable.

    ::

        trace_root_anchor("pipeline.trace_anchor")
        |
        +-- tracer.start_span(name, context=EMPTY)
        |     Creates TRUE root span (no parent inherited)
        |
        +-- attach anchor context as current OTel context
        |
        +-- anchor.end()
        |     Exported immediately (SimpleSpanProcessor)
        |     Backends receive root BEFORE children
        |
        +-- yield TracedSpan(anchor)
        |     |
        |     v  [caller's code runs]
        |        propagate_trace_context()  <-- workers inherit IDs
        |        state.scope("main")       <-- child of anchor
        |
        +-- detach anchor context (finally)

    Typical usage in ``profiling_scope()``::

        with trace_root_anchor("pipeline.trace_anchor"):
            propagate_trace_context()   # workers inherit anchor IDs
            with state.scope("main"):   # _root.main is child of anchor
                yield                   # pipeline runs

    Resulting trace hierarchy::

        trace_anchor  (root, no parent, ends FIRST)
          +-- _root.main              (ends LAST, parent=anchor)
          +-- stage spans             (parent=anchor via env var)

    Args:
        name: Span name for the anchor (e.g. ``"pipeline.trace_anchor"``).
        attributes: Optional key-value pairs attached to the span.

    Yields:
        A :class:`TracedSpan` wrapping the (already ended) anchor span.

    """
    tracer = trace.get_tracer(_TRACER_NAME)

    # Create the anchor as a TRUE root span -- pass an empty context
    # so no parent span is inherited from the call site.
    anchor = tracer.start_span(name, context=otel_context.Context(), attributes=attributes or {})

    # Attach the anchor's context so child spans inherit its
    # trace_id and use its span_id as parent_span_id.
    ctx = trace.set_span_in_context(anchor)
    token = otel_context.attach(ctx)

    # End the anchor immediately.  With SimpleSpanProcessor the span
    # is exported to all backends (file + OTLP) synchronously -- so
    # by the time we yield, the root is already in Jaeger.
    anchor.end()

    try:
        yield TracedSpan(anchor)
    finally:
        otel_context.detach(token)


def traced(
    name: str,
    *,
    attributes: dict[str, SpanAttributeValue] | None = None,
) -> Callable[..., Any]:
    """Wrap a function in an OTel span (decorator form).

    **When to use**: when the **entire** function body is one logical
    span and attributes are known at decoration time (static).  For
    partial-function spans or dynamic attributes, prefer
    :func:`traced_span` (context manager).

    Equivalent to wrapping the function body in
    ``with traced_span(name, attributes=...):``.  The span name and
    attributes are fixed at decoration time.

    Uses ``wrapt.decorator`` for robust signature preservation
    across plain functions, bound methods, class methods, and
    static methods.

    Unhandled exceptions are automatically recorded on the span by
    OTel (``record_exception=True`` is the default).

    ::

        @traced("MyModel.predict", attributes={"model": "resnet50"})
        def predict(self, batch):
            ...

        Calling predict(batch) internally does:

        traced_span("MyModel.predict", attributes={"model": "resnet50"})
              |
              +-- span starts (child of current active span)
              +-- predict() body executes
              +-- span ends (duration = function wall-clock time)
              +-- exception? -> auto-recorded on span by OTel

    Works with both sync and async functions.  For ``async def``
    targets, the span stays open for the full coroutine execution
    (not just until the coroutine object is returned).

    Trace volume is controlled by OTel's ``ParentBasedTraceIdRatio``
    sampler (configured via ``--profile-tracing-sampling``), which
    makes the sampling decision at the root span and propagates it
    to all descendants.

    Usage::

        from cosmos_curator.core.utils.infra.tracing import traced

        @traced("MyModel.predict", attributes={"model": "resnet50"})
        def predict(self, batch: list[Frame]) -> list[float]:
            return self._model(batch)

        @traced("MyStage.generate")
        async def generate(self, prompt: str) -> str:
            return await self._engine.generate(prompt)

    This is equivalent to::

        def predict(self, batch: list[Frame]) -> list[float]:
            with traced_span("MyModel.predict", attributes={"model": "resnet50"}):
                return self._model(batch)

    Args:
        name: Span name -- typically ``"ClassName.method"``.
        attributes: Optional static key-value pairs attached to
            every invocation's span.  For dynamic attributes that
            depend on runtime values, use :func:`traced_span`
            instead and call ``span.set_attribute()`` inside the
            ``with`` block.

    Returns:
        A decorator that wraps the target function.

    """

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202, ARG001
        # wrapt.decorator mandates this exact (wrapped, instance, args, kwargs) signature;
        # all four parameters are untyped by design (no stubs for wrapt).
        #
        # For async functions, the sync `with traced_span(...)` block would
        # exit as soon as the coroutine object is returned - before the
        # actual async work executes.
        if asyncio.iscoroutinefunction(wrapped):

            async def _async_traced() -> object:
                with traced_span(name, attributes=attributes):
                    return await wrapped(*args, **kwargs)

            return _async_traced()
        with traced_span(name, attributes=attributes):
            return wrapped(*args, **kwargs)

    return wrapper  # type: ignore[no-any-return]
