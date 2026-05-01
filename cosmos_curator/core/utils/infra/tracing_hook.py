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

"""Ray tracing hook and driver-side tracing setup.

This module contains the **Ray-specific** tracing integration:

Driver side (called from ``profiling_scope()``):
    :func:`enable_tracing` -- sets environment variables and
    pre-initializes the driver with OTel, so that
    ``init_or_connect_to_cluster()`` passes the hook to ``ray.init()``.

Worker side (called automatically by Ray):
    :func:`setup_tracing` -- configures an OpenTelemetry
    ``TracerProvider`` with a ``ConsoleSpanExporter`` that writes
    NDJSON span data (one JSON object per line) to a local file
    under the staging directory.  An ``atexit`` handler flushes
    the provider to ensure all spans reach disk.  Post-pipeline,
    ``ArtifactDelivery`` collects staged trace files from
    all nodes.

The **generic span API** (:class:`TracedSpan`, :func:`traced_span`,
:func:`traced`, ``StatusCode``) lives in :mod:`tracing`.

Configuration is passed via environment variables because the Ray
hook function signature takes no arguments:

    COSMOS_CURATOR_ARTIFACTS_STAGING_DIR
        Local staging directory where all profiling artifacts
        (including traces) are written.  Trace files go to
        ``<staging_dir>/traces/``.  Set by
        ``ArtifactDelivery``
        before the pipeline starts.

    COSMOS_CURATOR_TRACE_DIR
        Local directory where span files are written.  When
        ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` is set, this defaults to
        ``<staging_dir>/traces/``.  Otherwise defaults to
        ``/tmp/cosmos_curator_traces``.

    OTEL_EXPORTER_OTLP_ENDPOINT  *(standard OTel env var)*
        OTLP HTTP endpoint for remote span export.  **Opt-in**:
        when neither this nor ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``
        is set, remote OTLP export is disabled and spans are only
        written to local ``.jsonl`` files.  Set to a collector URL
        (e.g. ``http://localhost:4318``) to enable remote delivery.
        Can also be set via ``--profile-tracing-otlp-endpoint``.

    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT  *(standard OTel env var)*
        Trace-specific override for ``OTEL_EXPORTER_OTLP_ENDPOINT``.
        Takes precedence when both are set.

    OTEL_SERVICE_NAME  *(standard OTel env var)*
        Overrides the ``service.name`` resource attribute attached
        to the ``TracerProvider``.  Default: ``cosmos_curator``.
        Visible as the "service name" column in Jaeger, Grafana
        Tempo, and other tracing backends.

    The ``OTLPSpanExporter`` also honours additional standard OTel
    env vars (``OTEL_EXPORTER_OTLP_HEADERS``,
    ``OTEL_EXPORTER_OTLP_TIMEOUT``, ``OTEL_EXPORTER_OTLP_CERTIFICATE``,
    etc.) -- see the `OTel specification`_ for the full list.

    .. _OTel specification:
       https://opentelemetry.io/docs/specs/otel/protocol/exporter/

The hook is referenced as a ``"module:attribute"`` string::

    ray.init(
        _tracing_startup_hook=(
            "cosmos_curator.core.utils.infra.tracing_hook:setup_tracing"
        ),
        ...
    )

Ray stores this string in its GCS internal KV store and every
worker (including the driver) reads it on startup, so the function
must be importable on all nodes.

Library auto-instrumentation
----------------------------
When tracing is enabled, ``setup_tracing()`` also activates OTel
auto-instrumentors for commonly used libraries.  Each instrumentor
monkey-patches its target library to emit spans automatically:

* **botocore** -- spans for every S3 / AWS API call.
* **requests** -- spans for outbound HTTP requests.
* **urllib3** -- spans for low-level HTTP transport.
* **threading** -- propagates trace context across threads.
* **logging** -- injects ``otelTraceID`` / ``otelSpanID`` into
  stdlib ``logging.LogRecord`` attributes.
* **sqlalchemy** -- spans for SQL queries.
* **fastapi** -- spans for inbound HTTP endpoints (NVCF).

Instrumentors are gated on ``importlib.util.find_spec()`` so they
only activate when the target library is installed in the current
Pixi environment.  No errors are raised for missing libraries.
"""

import atexit
import contextlib
import errno
import importlib.util
import os
import pathlib
import tempfile
from collections.abc import Sequence
from typing import IO

import attrs
from loguru import logger
from opentelemetry import context, trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from cosmos_curator.core.utils.infra.tracing import (
    _ENV_OTLP_ENDPOINT,
    _ENV_OTLP_TRACES_ENDPOINT,
    artifact_id,
    get_otlp_endpoint,
    short_hostname,
)

# Hook module path for ray.init(_tracing_startup_hook=...).
# Must be importable on all Ray worker nodes.
_RAY_TRACING_HOOK = "cosmos_curator.core.utils.infra.tracing_hook:setup_tracing"

# Environment variable set by enable_tracing() and read by
# init_or_connect_to_cluster() to pass the hook to ray.init().
_ENV_RAY_TRACING_HOOK = "XENNA_RAY_TRACING_HOOK"

# Environment variables read by the worker-side hook.
_ENV_TRACE_DIR = "COSMOS_CURATOR_TRACE_DIR"

# Propagated trace context from the driver's root span.
# Format: "{trace_id_hex}:{span_id_hex}" (set by propagate_trace_context()).
# Workers read this to create spans as children of the driver's root span,
# unifying all spans under a single trace_id.
_ENV_TRACEPARENT = "COSMOS_CURATOR_TRACEPARENT"

# Standard OTel env var for overriding the resource service name.
# When set, takes precedence over the default "cosmos_curator".
_ENV_SERVICE_NAME = "OTEL_SERVICE_NAME"


@attrs.define(frozen=True)
class TracingConfig:
    """Frozen configuration for the per-process OTel tracing backend.

    Encapsulates all environment-variable parsing into a single typed
    object with sensible defaults.  Two construction paths:

    * **Driver side**: ``enable_tracing()`` builds a ``TracingConfig``
      from environment variables, sets the trace dir env var, then
      calls ``setup_tracing()`` on the driver.
    * **Worker side**: ``setup_tracing()`` (Ray hook, no arguments)
      calls ``TracingConfig.from_env()`` which reads the env vars.

    Attributes:
        trace_dir: Local directory where span files are written.
            When ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR`` is set, defaults to
            ``<staging_dir>/traces/`` so traces are collected with
            other profiling artifacts post-pipeline.
        otlp_endpoint: OTLP HTTP collector endpoint.  Empty string
            (default) disables remote OTLP export -- only the local
            file-based exporter is active.  Set via
            ``OTEL_EXPORTER_OTLP_ENDPOINT``,
            ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, or
            ``--profile-tracing-otlp-endpoint`` to enable.
        traceparent: W3C-style trace context propagated from the
            driver's root span.  Format:
            ``"{trace_id_hex}:{span_id_hex}"`` or empty string
            when not propagated.  Workers use this to create spans
            as children of the driver's root span, unifying all
            processes under a single ``trace_id``.
        service_name: Value for the OTel ``service.name`` resource
            attribute.  Appears as the "service name" column in
            Jaeger, Grafana Tempo, and other tracing backends.
            Defaults to ``"cosmos_curator"``; overridable via the
            standard ``OTEL_SERVICE_NAME`` environment variable.

    """

    trace_dir: str = "/tmp/cosmos_curator_traces"  # noqa: S108
    otlp_endpoint: str = ""
    traceparent: str = ""
    service_name: str = "cosmos_curator"

    @classmethod
    def from_env(cls) -> "TracingConfig":
        """Build a ``TracingConfig`` from environment variables.

        Called on each Ray worker via ``setup_tracing()``.
        Environment variables are set by ``enable_tracing()`` on the
        driver before ``ray.init()`` propagates them to workers.

        Returns:
            A frozen ``TracingConfig`` populated from the environment.

        """
        # When a delivery class has set a staging directory, default
        # trace_dir to <staging_dir>/traces/ so traces are gathered
        # with other profiling artifacts post-pipeline.
        staging_dir = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")
        default_trace_dir = (
            str(pathlib.Path(staging_dir) / "traces") if staging_dir else "/tmp/cosmos_curator_traces"  # noqa: S108
        )
        return cls(
            trace_dir=os.environ.get(_ENV_TRACE_DIR, default_trace_dir),
            otlp_endpoint=get_otlp_endpoint(),
            traceparent=os.environ.get(_ENV_TRACEPARENT, ""),
            service_name=os.environ.get(_ENV_SERVICE_NAME, "cosmos_curator"),
        )


class _ResilientOtlpExporter(SpanExporter):
    """Wrapper around ``OTLPSpanExporter`` that suppresses connection errors.

    When the OTLP collector (e.g. Jaeger, Grafana Tempo) is not
    reachable, the ``OTLPSpanExporter`` raises ``ConnectionError`` on
    every ``export()`` call.  With ``SimpleSpanProcessor`` this fires
    synchronously on every span end, flooding logs with full tracebacks
    via ``logger.exception()`` inside the processor.

    This wrapper catches the first connection failure, logs a single
    warning, and disables all subsequent export attempts for this
    process.  The file-based exporter (which always works) is
    unaffected, so no span data is lost -- only the real-time OTLP
    delivery is skipped.

    ::

        export(spans)
            |
            v
        _disabled?
        +-- YES --> return SUCCESS (silent no-op)
        |
        +-- NO --> delegate to real exporter
                   |
                   +-- success --> return result
                   |
                   +-- ConnectionError -->
                       log warning ONCE
                       set _disabled = True
                       return SUCCESS (suppress)

    Design decisions:
        - Fail-open: tracing must never crash the pipeline.
          Connection failures are expected when no collector is
          deployed (e.g. local development, CI without Jaeger).
        - Log once: a single warning is sufficient for operators to
          know that OTLP delivery is not working.  Per-span
          tracebacks are noise.
        - No retry: the exporter is disabled permanently for this
          process lifetime.  A transient blip that resolves mid-run
          will not be recovered, but the file exporter captures all
          spans regardless.  Re-enabling would add complexity with
          minimal benefit (typical runs are minutes, not hours).
        - SUCCESS return: returning FAILURE would trigger SDK retry
          logic in BatchSpanProcessor (though we use
          SimpleSpanProcessor which doesn't retry).  Returning
          SUCCESS keeps the contract simple.

    """

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate
        self._disabled = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans, suppressing connection errors after the first failure."""
        if self._disabled:
            return SpanExportResult.SUCCESS

        try:
            return self._delegate.export(spans)
        except Exception as exc:
            # Check the full exception chain for connection-related errors.
            # The OTLPSpanExporter wraps urllib3/requests errors in its own
            # exception hierarchy, so we walk __cause__ / __context__.
            if _is_connection_error(exc):
                self._disabled = True
                logger.debug(
                    f"[otel] OTLP collector unreachable - disabling remote span export "
                    f"for this process. Spans are still written to the local .jsonl file. "
                    f"({type(exc).__name__}: {exc})",
                )
                return SpanExportResult.SUCCESS
            raise

    def shutdown(self) -> None:
        """Shut down the delegate exporter."""
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush the delegate exporter (no-op if disabled)."""
        if self._disabled:
            return True
        return self._delegate.force_flush(timeout_millis)


def _is_connection_error(exc: BaseException) -> bool:
    """Walk the exception chain to detect connection-related failures.

    The OTLP HTTP exporter wraps the underlying ``urllib3`` /
    ``requests`` connection error in its own exception type.  We need
    to check ``__cause__`` and ``__context__`` recursively to find the
    root ``ConnectionError``, ``ConnectionRefusedError``, or
    ``OSError`` with ``errno == ECONNREFUSED``.

    Returns ``True`` if any exception in the chain is a connection
    failure, ``False`` otherwise.
    """
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        # ConnectionRefusedError is a subclass of ConnectionError
        if isinstance(current, ConnectionError):
            return True
        if isinstance(current, OSError) and current.errno == errno.ECONNREFUSED:
            return True
        # Walk both explicit cause and implicit context chains.
        current = current.__cause__ or current.__context__
    return False


class _TracingBackend:
    """Per-process OTel tracing backend.

    Manages the lifecycle of the ``TracerProvider`` and the local span
    file.  Follows the same interface contract as the other profiling
    backends in ``profiling.py`` (``_CpuProfilingBackend``,
    ``_MemoryProfilingBackend``, ``_GpuProfilingBackend``).

    Trace files are written to the local staging directory (set via
    ``COSMOS_CURATOR_ARTIFACTS_STAGING_DIR``).  Post-pipeline,
    ``ArtifactDelivery`` collects them from all nodes.

    ::

        _TracingBackend lifecycle
        =========================

        __init__(config)          <-- create file (no provider yet)
              |
              v
        setup_provider()          <-- configure TracerProvider + exporters
              |                        |
              | (success)              | (raises)
              v                        v
        [worker runs stages]      caller closes _file_handle
              |                   (prevents FD leak; backend
              v                    is never installed as singleton)
        flush()                   <-- force_flush provider + flush file
              |                       file stays OPEN so late spans
              v                       can still be exported
        shutdown()                <-- provider.shutdown (no more exports)
                                      THEN close file + log stats

    ``flush()`` pushes buffered span data to the OS kernel without
    closing the file handle.  Late-arriving spans (e.g. from async
    operations that fail after teardown begins) can still be written
    by the ``ConsoleSpanExporter`` because the file remains open.

    ``shutdown()`` first disables the provider (sets
    ``SimpleSpanProcessor.done = True`` so ``on_end()`` stops
    exporting), **then** closes the file handle.  This ordering
    guarantees the file is never closed while exports can still
    occur.

    Both methods are naturally idempotent: ``force_flush()`` and
    ``file.flush()`` are safe to call repeatedly, and
    ``provider.shutdown()`` / ``close_file_handle()`` have their
    own internal guards.

    If ``setup_provider()`` raises, neither ``flush()`` nor
    ``shutdown()`` will ever be called (the backend is not stored
    as the module singleton and no ``atexit`` handler is registered).
    The caller (``setup_tracing()``) is responsible for closing
    ``_file_handle`` in the failure path to prevent a file descriptor
    leak.

    Attributes:
        _config: Frozen tracing configuration.
        _filepath: Local path to the ``.jsonl`` span file.
        _filename: Base filename for the artifact.
        _file_handle: Open file handle for the span exporter.

    """

    def __init__(self, config: TracingConfig) -> None:
        self._config = config

        # Build a unique artifact ID and filename using the shared
        # artifact_id() convention.  artifact_id lives in tracing.py
        # (lightweight).  The ID is reused as a log tag so operators
        # can correlate log messages with the uploaded trace file.
        self._id = artifact_id("trace", "spans")
        self._filename = f"{self._id}.jsonl"

        # Create the local trace directory and open the output file.
        local_dir = pathlib.Path(config.trace_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        self._filepath = local_dir / self._filename

        # Open in write mode; each worker gets its own file.
        self._file_handle: IO[str] = self._filepath.open("w")
        logger.trace(f"[otel] {self._id}: Trace file opened: {self._filepath}")

    def setup_provider(self) -> None:
        """Configure the global ``TracerProvider`` with file and OTLP exporters.

        Attaches an OTel ``Resource`` with process identity so traces
        are human-readable in backends (Jaeger, Grafana Tempo, etc.)
        instead of showing the SDK default ``"unknown_service"``:

        * ``service.name`` -- application name (default ``"cosmos_curator"``,
          overridable via ``OTEL_SERVICE_NAME``).
        * ``host.name`` -- short hostname for multi-node filtering.
        * ``process.pid`` -- OS PID for multi-worker filtering.

        Sets the OTel global ``TracerProvider`` and attaches:

        1. A ``SimpleSpanProcessor`` + ``ConsoleSpanExporter`` writing
           NDJSON to the local span file (always active).
        2. A ``SimpleSpanProcessor`` + ``_ResilientOtlpExporter``
           (wrapping ``OTLPSpanExporter``) for remote delivery.
           **Only added when** ``otlp_endpoint`` **is non-empty**
           (set via ``OTEL_EXPORTER_OTLP_ENDPOINT``,
           ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT``, or
           ``--profile-tracing-otlp-endpoint``).  Uses
           ``SimpleSpanProcessor`` (not ``BatchSpanProcessor``) so
           each span is sent immediately -- Ray may SIGKILL workers
           before a batched flush completes.  The resilient wrapper
           suppresses connection errors after the first failure so
           unreachable collectors don't flood logs with tracebacks.

        After configuring the provider, activates library
        auto-instrumentors (botocore, requests, etc.).
        """
        # Deferred import: OTel SDK classes are heavy and only needed
        # when actually configuring tracing.  The lightweight
        # opentelemetry.trace API is imported at module level.
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        # Attach a Resource with service identity and process metadata so
        # that traces are human-readable in backends (Jaeger, Tempo, etc.)
        # instead of showing the SDK default "unknown_service".
        #
        # We read OTEL_SERVICE_NAME ourselves (via TracingConfig) because
        # Resource.create() gives programmatic attributes higher priority
        # than env-var-detected ones -- passing the resolved value
        # preserves the standard override semantics.
        #
        # host.name and process.pid enable filtering by node/worker in
        # multi-node Ray clusters where many processes emit spans.
        #
        #   Resource attributes set here
        #   =============================
        #   service.name  = "cosmos_curator" (or OTEL_SERVICE_NAME)
        #   host.name     = short hostname  (e.g. "node03")
        #   process.pid   = OS PID          (e.g. 6135)
        #
        _pid = os.getpid()
        resource = Resource.create(
            {
                "service.name": self._config.service_name,
                "host.name": short_hostname(),
                "process.pid": _pid,
            }
        )
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        logger.trace(
            f"[otel] {self._id}: TracerProvider configured "
            f"(service.name={self._config.service_name}, "
            f"host.name={short_hostname()}, pid={_pid})",
        )

        # File-based exporter (always active).
        #
        # SimpleSpanProcessor exports each span synchronously as it
        # completes -- no batching delay, which is important because
        # workers may be killed abruptly.
        provider.add_span_processor(
            SimpleSpanProcessor(
                ConsoleSpanExporter(
                    out=self._file_handle,
                    formatter=lambda span: span.to_json(indent=None) + os.linesep,
                ),
            ),
        )

        # OTLP remote exporter (opt-in: only active when an endpoint
        # is explicitly configured via env var or CLI flag).
        #
        # Uses SimpleSpanProcessor (not BatchSpanProcessor) so each
        # span is exported synchronously when it completes.  This is
        # critical because Ray may SIGKILL worker processes during
        # ray.shutdown() before a batched flush has time to drain.
        # For localhost the per-span HTTP overhead is < 10ms, and
        # for remote collectors it is still acceptable given the
        # moderate span count per worker (tens, not thousands).
        #
        # IMPORTANT: We set the env var and let OTLPSpanExporter()
        # read it rather than passing endpoint= to the constructor.
        # The SDK appends "/v1/traces" automatically when reading
        # from OTEL_EXPORTER_OTLP_ENDPOINT, but uses the URL as-is
        # when endpoint= is passed directly -- which would miss the
        # required path suffix and silently fail.
        #
        # The OTLPSpanExporter also reads additional standard env
        # vars (OTEL_EXPORTER_OTLP_HEADERS for auth tokens,
        # OTEL_EXPORTER_OTLP_TIMEOUT, OTEL_EXPORTER_OTLP_CERTIFICATE,
        # etc.) so no explicit configuration is needed beyond the
        # base endpoint URL.
        #
        # Wrapped in _ResilientOtlpExporter to suppress connection
        # errors when no collector is deployed.  Without the wrapper,
        # SimpleSpanProcessor logs a full traceback on every span when
        # the collector is unreachable, flooding logs with noise.
        if self._config.otlp_endpoint:
            if not os.environ.get(_ENV_OTLP_ENDPOINT) and not os.environ.get(_ENV_OTLP_TRACES_ENDPOINT):
                os.environ[_ENV_OTLP_ENDPOINT] = self._config.otlp_endpoint

            # Deferred import: heavy OTLP exporter only loaded when
            # an endpoint is configured.
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )

            resilient_exporter = _ResilientOtlpExporter(OTLPSpanExporter())
            provider.add_span_processor(SimpleSpanProcessor(resilient_exporter))

        # Inject remote parent context from the driver's root span.
        #
        # When traceparent is set (format: "trace_id_hex:span_id_hex"),
        # construct a remote SpanContext and attach it as the current
        # context.  All subsequent spans created on this worker become
        # children of the driver's root span, sharing the same trace_id.
        # This unifies the distributed trace across driver + workers.
        if self._config.traceparent:
            self._attach_remote_parent(self._config.traceparent)

        # Library auto-instrumentation (botocore, requests, etc.).
        _instrument_libraries()

    def _attach_remote_parent(self, traceparent: str) -> None:
        """Parse a traceparent string and attach it as the current OTel context.

        Delegates to the module-level :func:`attach_remote_parent`
        function.  Kept as a method for backward compatibility with
        the ``setup_provider()`` call site.

        Args:
            traceparent: ``"{trace_id_hex}:{span_id_hex}"`` string
                set by :func:`propagate_trace_context` on the driver.

        """
        attach_remote_parent(traceparent)

    def propagate_context(self) -> None:
        """Write the current span's trace context to the environment.

        Serializes the active span's ``trace_id`` and ``span_id``
        into ``COSMOS_CURATOR_TRACEPARENT`` so that Ray workers (which
        inherit environment variables at startup) can reconstruct the
        remote parent and create child spans under the same trace.

        Called from ``profiling_scope()`` **inside** the
        ``trace_root_anchor()`` context, after ``enable_tracing()``
        has set up the driver's ``TracerProvider``.  The active span
        at call time is the **anchor** (already ended but still the
        current context), so workers become children of the anchor.

        ::

            profiling_scope()
                  |
                  +-- enable_tracing(config)        <-- sets up TracerProvider
                  |
                  +-- trace_root_anchor() starts    <-- anchor span (root)
                  |     +-- anchor.end()            <-- exported NOW
                  |     +-- propagate_context()     <-- reads anchor's IDs
                  |     |     +-- writes COSMOS_CURATOR_TRACEPARENT env var
                  |     |
                  |     +-- state.scope("main")     <-- _root.main (child of anchor)
                  |     |     +-- yield (pipeline runs, workers start)
                  |     |
                  +-- trace_root_anchor exits       <-- detach anchor context
                  |
                  v
            [workers read COSMOS_CURATOR_TRACEPARENT in setup_tracing()]

        No-op when the current span has no valid trace context (e.g.
        tracing is not actually enabled).
        """
        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx is not None and ctx.trace_id != 0:
            traceparent = f"{ctx.trace_id:032x}:{ctx.span_id:016x}"
            os.environ[_ENV_TRACEPARENT] = traceparent
            logger.debug(
                f"[otel] {self._id}: Root trace context propagated: "
                f"trace_id={ctx.trace_id:032x}, span_id={ctx.span_id:016x}",
            )

    def flush(self) -> None:
        """Force-flush pending spans to disk without closing the file.

        Called explicitly from ``_ProfiledStage.destroy()`` and from
        the exception path in ``_ProfilingState.scope()`` to persist
        trace data **before** Ray kills the worker.

        The file handle stays **open** so that late-arriving spans
        (e.g. from async operations that fail with ``EngineDeadError``
        after teardown begins) can still be exported by the
        ``ConsoleSpanExporter``.  Only ``shutdown()`` closes the file,
        and only after disabling the provider.

        ::

            flush()
              +-- TracerProvider.force_flush()
              |     Drains any buffered spans to the exporter.
              +-- file_handle.flush()
              |     Pushes Python buffer to OS kernel.
              |     Data survives SIGKILL (kernel writes to disk).
              +-- [file remains OPEN]

        Idempotent: calling ``force_flush()`` and ``file.flush()``
        multiple times is safe and cheap.
        """
        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[otel] {self._id}: TracerProvider force_flush failed: {exc}")

        try:
            self._file_handle.flush()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[otel] {self._id}: File buffer flush failed: {exc}")

    def shutdown(self) -> None:
        """Shut down the TracerProvider, then close the trace file.

        Called from the ``atexit`` handler as a fallback for processes
        that exit gracefully (e.g. the driver).

        Order matters: the provider must be shut down **before** the
        file handle is closed.  ``TracerProvider.shutdown()`` sets
        ``SimpleSpanProcessor.done = True``, which prevents further
        ``on_end()`` calls from reaching the ``ConsoleSpanExporter``.
        Only then is it safe to close the underlying file.

        ::

            shutdown()
              +-- TracerProvider.shutdown()
              |     Sets SimpleSpanProcessor.done = True.
              |     No further on_end() -> export() calls.
              +-- close_file_handle()
              |     NOW safe -- no more writes possible.
              +-- log file stats

        Idempotent: ``TracerProvider.shutdown()`` checks its own
        ``_shutdown`` flag internally, and ``close_file_handle()``
        suppresses errors on an already-closed handle.
        """
        try:
            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[otel] {self._id}: TracerProvider shutdown failed: {exc}")

        self.close_file_handle()
        self._log_file_stats()

    def close_file_handle(self) -> None:
        """Close the span file handle, suppressing errors.

        Idempotent: calling on an already-closed handle is a no-op.
        Used by ``shutdown()`` after the provider is disabled, and by
        ``setup_tracing()`` for cleanup when ``setup_provider()``
        fails before the atexit handler is registered.
        """
        with contextlib.suppress(Exception):
            self._file_handle.close()

    def _log_file_stats(self) -> None:
        """Log the final trace file size after closing.

        Called by ``shutdown()`` after ``close_file_handle()`` to
        report the persisted file size.  Separated from
        ``close_file_handle()`` so the close + log sequence is
        explicit in ``shutdown()``'s control flow.
        """
        try:
            file_size = self._filepath.stat().st_size
        except OSError as e:
            logger.trace(f"[otel] {self._id}: Could not stat trace file: {e}")
            file_size = 0

        if file_size == 0:
            logger.trace(f"[otel] {self._id}: Trace file empty (no spans recorded): {self._filepath}")
        else:
            logger.debug(f"[otel] {self._id}: Trace file persisted: {self._filepath} ({file_size} bytes)")


# Per-process singleton.  Set by setup_tracing(), read by
# flush_tracing() and propagate_trace_context().  None when tracing
# is not configured.  The module-level reference is necessary because:
#   1. flush_tracing() is called from _ProfiledStage.destroy() which
#      has no direct reference to the backend.  It flushes buffered
#      spans to the local staging directory.
#   2. propagate_trace_context() is called from profiling_scope()
#      which also has no direct reference.
_current_backend: _TracingBackend | None = None


# Activates OTel auto-instrumentors for commonly used libraries.
# Each instrumentor monkey-patches its target library to emit spans
# for every call (e.g. every S3 PutObject, every HTTP request).
#
# Gated on importlib.util.find_spec() so that an instrumentor is
# only activated when its target library is actually installed in
# the current environment.  This avoids import errors in minimal
# environments (e.g. model-download) while keeping the code
# unconditional -- no try/except needed.
#
#   Target library     Instrumentor package                      What it captures
#   botocore           opentelemetry-instrumentation-botocore     S3/AWS API calls
#   requests           opentelemetry-instrumentation-requests     Outbound HTTP
#   urllib3            opentelemetry-instrumentation-urllib3       Low-level HTTP
#   threading          opentelemetry-instrumentation-threading    Context propagation
#   logging            opentelemetry-instrumentation-logging      trace_id in logs
#   sqlalchemy         opentelemetry-instrumentation-sqlalchemy   SQL queries
#   fastapi            opentelemetry-instrumentation-fastapi      NVCF HTTP endpoints


def _try_instrument(
    target: str, instrumentor_module: str, instrumentor_class: str, label: str, **kwargs: object
) -> None:
    """Try to activate a single OTel auto-instrumentor.

    Gates on both the *target* library and the *instrumentor* package
    via ``importlib.util.find_spec()``.  If either is missing the call
    is a silent no-op.  Errors during activation are logged at DEBUG
    level and swallowed so they never crash a worker.
    """
    if not importlib.util.find_spec(target):
        return
    if not importlib.util.find_spec(instrumentor_module):
        logger.debug(f"[otel] Skipping {label}: {instrumentor_module} not installed")
        return
    try:
        mod = importlib.import_module(instrumentor_module)
        cls = getattr(mod, instrumentor_class)
        cls().instrument(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[otel] Failed to instrument {label}: {exc}")


def _instrument_libraries() -> None:
    """Activate OTel auto-instrumentors for libraries present in this environment.

    Called once from :meth:`_TracingBackend.setup_provider` after the
    ``TracerProvider`` is configured.  Each instrumentor patches its
    target library globally so that all subsequent calls emit OTel
    spans automatically.

    Only instrumentors whose target library is importable are activated;
    the rest are silently skipped.  This makes the function safe to call
    in any Pixi environment (e.g. ``default``, ``unified``,
    ``model-download``) regardless of which libraries are installed.

    The ``tracing`` feature in ``pixi.toml`` installs all instrumentor
    packages in every environment, so the gate is effectively on the
    *target* library alone.

    """
    # botocore (S3 / AWS API calls)
    _try_instrument(
        "botocore", "opentelemetry.instrumentation.botocore", "BotocoreInstrumentor", "botocore (S3/AWS API calls)"
    )

    # requests (outbound HTTP)
    _try_instrument(
        "requests", "opentelemetry.instrumentation.requests", "RequestsInstrumentor", "requests (outbound HTTP)"
    )

    # urllib3 (low-level HTTP transport)
    _try_instrument(
        "urllib3", "opentelemetry.instrumentation.urllib3", "URLLib3Instrumentor", "urllib3 (low-level HTTP)"
    )

    # threading (context propagation across threads)
    _try_instrument(
        "threading",
        "opentelemetry.instrumentation.threading",
        "ThreadingInstrumentor",
        "threading (context propagation)",
    )

    # logging (inject trace_id / span_id into stdlib log records)
    _try_instrument(
        "logging",
        "opentelemetry.instrumentation.logging",
        "LoggingInstrumentor",
        "logging (trace context injection)",
        set_logging_format=False,
    )

    # sqlalchemy (SQL query spans)
    _try_instrument(
        "sqlalchemy", "opentelemetry.instrumentation.sqlalchemy", "SQLAlchemyInstrumentor", "sqlalchemy (SQL queries)"
    )

    # fastapi (NVCF service endpoints)
    _try_instrument(
        "fastapi", "opentelemetry.instrumentation.fastapi", "FastAPIInstrumentor", "fastapi (HTTP endpoints)"
    )


def enable_tracing(*, sampling_rate: float = 1.0, otlp_endpoint: str = "") -> None:
    """Enable distributed OpenTelemetry tracing for the pipeline.

    Performs four actions:

    1. Sets standard OTel sampling env vars so ``TracerProvider``
       auto-configures ``ParentBasedTraceIdRatio`` from the environment.
    2. Sets environment variables read by ``init_or_connect_to_cluster()``
       (in ``ray_cluster_utils.py`` and xenna's ``cluster.py``) to pass
       ``_tracing_startup_hook`` to ``ray.init()``.
    3. Sets environment variables read by :func:`setup_tracing` on each
       worker (trace dir, staging dir, OTLP endpoint).
    4. Calls :func:`setup_tracing` on the **driver** process itself so
       that driver-side spans (``profiling_scope``, ``TracedSpan``,
       ``@traced``) are captured.  Workers get their own
       ``setup_tracing()`` call via Ray's ``_tracing_startup_hook``.

    Each process (driver and workers) writes its own span file to
    the local staging directory.  Post-pipeline,
    ``ArtifactDelivery`` collects and uploads all staged trace
    files.

    Args:
        sampling_rate: Fraction of traces to sample (0.0--1.0).
            Sets ``OTEL_TRACES_SAMPLER`` and ``OTEL_TRACES_SAMPLER_ARG``
            so the OTel SDK auto-configures ``ParentBasedTraceIdRatio``.
            Default: 1.0 (sample all).
        otlp_endpoint: OTLP HTTP collector endpoint (e.g.
            ``http://localhost:4318``).  Empty string (default)
            disables remote OTLP export -- only the local file-based
            exporter is active.  When non-empty, sets
            ``OTEL_EXPORTER_OTLP_ENDPOINT`` so workers inherit it.

    """
    import cosmos_curator  # noqa: PLC0415

    # Ensure Ray workers can import the tracing hook module.
    # The driver can already import cosmos_curator (it's running this
    # code), but bare workers spawned by the raylet may not have the
    # package on their sys.path (e.g. in Docker where the package is
    # installed via editable install or PYTHONPATH).
    pkg_root = str(pathlib.Path(cosmos_curator.__file__).resolve().parent.parent)
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    if pkg_root not in current_pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = f"{pkg_root}{os.pathsep}{current_pythonpath}" if current_pythonpath else pkg_root

    # Trace files go under the staging directory (set by delivery classes)
    # so they are collected with other profiling artifacts post-pipeline.
    # If no staging dir is set, fall back to a temp directory.
    staging_dir = os.environ.get("COSMOS_CURATOR_ARTIFACTS_STAGING_DIR")
    trace_dir = (
        str(pathlib.Path(staging_dir) / "traces")
        if staging_dir
        else str(pathlib.Path(tempfile.gettempdir()) / "cosmos_curator_traces")
    )

    # Set env vars for the worker-side hook to read.
    os.environ[_ENV_RAY_TRACING_HOOK] = _RAY_TRACING_HOOK
    os.environ[_ENV_TRACE_DIR] = trace_dir

    # Clear any stale traceparent from a previous run in the same
    # shell.  The driver is the root -- it must NOT attach to an old
    # trace.  propagate_trace_context() will set this env var later,
    # after the root span is created in profiling_scope().
    os.environ.pop(_ENV_TRACEPARENT, None)

    # Set standard OTel env vars so TracerProvider (constructed without
    # an explicit sampler= arg) auto-configures sampling from env.
    # Workers inherit env vars from driver via Ray.  Any 3rd-party
    # library (vLLM, etc.) that creates its own TracerProvider also
    # picks up these env vars automatically.
    os.environ["OTEL_TRACES_SAMPLER"] = "parentbased_traceidratio"
    os.environ["OTEL_TRACES_SAMPLER_ARG"] = str(sampling_rate)

    # Set the OTLP endpoint env var if explicitly provided via CLI
    # flag.  Workers inherit env vars from the driver, so setting it
    # here propagates to all Ray workers.  When empty (default), the
    # OTLP exporter is not created -- only the local file exporter
    # is active (no ConnectionRefused warnings in NVCF/Slurm).
    otlp_endpoint = otlp_endpoint.strip()
    if otlp_endpoint:
        os.environ[_ENV_OTLP_ENDPOINT] = otlp_endpoint

    # Also configure tracing on the driver process itself.
    # Workers get setup_tracing() via Ray's _tracing_startup_hook,
    # but the driver never goes through that path.  Without this,
    # driver-side spans (profiling_scope, TracedSpan, @traced) would
    # go to OTel's no-op tracer and produce an empty trace file.
    setup_tracing()

    otlp_status = f"OTLP -> {otlp_endpoint}" if otlp_endpoint else "OTLP disabled (file-only)"
    logger.info(
        f"[otel] Distributed tracing enabled; hook={_RAY_TRACING_HOOK}, span files: {trace_dir}/, {otlp_status}",
    )


def setup_tracing() -> None:
    """Configure OpenTelemetry tracing for this process.

    Called once per worker process by Ray's tracing startup hook
    mechanism, and once on the driver by :func:`enable_tracing`.

    Creates a :class:`_TracingBackend`, configures the OTel provider,
    stores the backend in the module-level singleton, and registers
    an ``atexit`` handler as a fallback for graceful exits.

    If ``setup_provider()`` raises (e.g. missing OTel SDK packages,
    resource exhaustion), the file handle opened during backend
    construction is closed before re-raising.  This prevents a file
    descriptor leak because the backend is never installed as the
    module singleton and the ``atexit`` handler is never registered,
    making the normal ``shutdown()`` cleanup path unreachable.

    Environment variables (set by :func:`enable_tracing`):
        COSMOS_CURATOR_ARTIFACTS_STAGING_DIR: Staging directory for artifacts.
        COSMOS_CURATOR_TRACE_DIR: Local span file directory.
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint.
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Trace-specific override.

    """
    global _current_backend  # noqa: PLW0603

    # Re-entrancy guard: setup_tracing() may be called twice on the
    # driver (once from enable_tracing(), once if Ray's
    # _tracing_startup_hook fires on the driver due to
    # ignore_reinit_error=True).  Without this guard the second call
    # would leak a file handle, orphan the first backend's atexit
    # handler, and double-instrument libraries.
    if _current_backend is not None:
        return

    tracing_config = TracingConfig.from_env()
    backend = _TracingBackend(tracing_config)
    try:
        backend.setup_provider()
    except Exception as e:
        # setup_provider() failed before the atexit handler was
        # registered.  Close the file handle opened in __init__()
        # to prevent a file descriptor leak.  The backend is never
        # installed as the singleton, so shutdown() would never run
        # through the normal atexit path.
        logger.warning(f"[otel] setup_provider() failed; closing file handle: {e}", exc_info=True)
        backend.close_file_handle()
        raise
    _current_backend = backend

    # Register atexit handler as a **fallback** for processes that
    # exit normally (e.g. the driver process).  For Ray workers,
    # flush_tracing() is called explicitly from
    # _ProfiledStage.destroy() before the worker is killed.
    #
    # shutdown() disables the provider first (preventing further
    # span exports), then closes the file handle and logs stats.
    atexit.register(backend.shutdown)


def flush_tracing() -> None:
    """Flush pending spans and file buffer to disk without closing the file.

    Called explicitly from ``_ProfiledStage.destroy()`` and the
    exception path in ``_ProfilingState.scope()`` to push span data
    to the OS kernel **before** Ray kills the worker process.

    The file handle stays **open** so that late-arriving spans
    (e.g. from async operations failing during teardown) can still
    be exported by the ``ConsoleSpanExporter``.

    Only ``shutdown()`` (via ``atexit``) closes the file, and only
    after disabling the ``TracerProvider`` so no more exports can
    occur.

    ::

        _ProfiledStage.destroy()
              |
              +-- flush_final_artifacts()   (cpu/mem/gpu backends)
              |
              +-- flush_tracing()           (OTel span flush to disk)
              |     |
              |     +-- backend.flush()
              |     |     +-- TracerProvider.force_flush()
              |     |     +-- file_handle.flush()
              |     |     +-- [file stays OPEN]
              |
              v
        [worker exits or is killed]
              |
              v
        atexit -> backend.shutdown()
                    |
                    +-- TracerProvider.shutdown()
                    |     (prevents further on_end exports)
                    +-- close_file_handle()
                    +-- log file stats

    Idempotent: calling ``force_flush()`` and ``file.flush()``
    multiple times is safe and cheap.
    """
    if _current_backend is not None:
        _current_backend.flush()


def propagate_trace_context() -> None:
    """Delegate to :meth:`_TracingBackend.propagate_context`.

    Module-level entry point for ``profiling_scope()`` which has no
    direct reference to the backend instance.
    """
    if _current_backend is not None:
        _current_backend.propagate_context()


def read_propagated_traceparent() -> str:
    """Return ``COSMOS_CURATOR_TRACEPARENT`` from the environment if set.

    :func:`propagate_trace_context` writes this after the trace anchor
    is created.  ``ProfilingConfig.traceparent`` is often captured
    earlier (e.g. when building stage specs before the anchor exists),
    so worker-side code should prefer
    ``config.traceparent or read_propagated_traceparent()``.
    """
    return os.environ.get(_ENV_TRACEPARENT, "")


def attach_remote_parent(traceparent: str) -> None:
    """Attach a remote parent span context to the **current thread**.

    Parses a ``"{trace_id_hex}:{span_id_hex}"`` string (the same
    format written by :func:`propagate_trace_context`) and attaches
    it as the active OTel context.  All subsequent spans created on
    this thread become children of the remote parent, sharing its
    ``trace_id``.

    No-op when *traceparent* is empty or malformed (logs a warning
    on parse failure).

    Args:
        traceparent: ``"{trace_id_hex}:{span_id_hex}"`` string, or
            empty string to skip.

    """
    if not traceparent:
        return

    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags  # noqa: PLC0415

    try:
        trace_id_hex, span_id_hex = traceparent.split(":")
        remote_ctx = SpanContext(
            trace_id=int(trace_id_hex, 16),
            span_id=int(span_id_hex, 16),
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        parent_ctx = trace.set_span_in_context(NonRecordingSpan(remote_ctx))
        context.attach(parent_ctx)
        logger.trace(
            f"[otel] attach_remote_parent: trace_id={trace_id_hex}, span_id={span_id_hex}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[otel] attach_remote_parent: Failed to parse traceparent: {exc}")
